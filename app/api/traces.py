"""链路追踪(工作台的「链路」标签页)—— 一个**只读**端点,服务端代理 Langfuse。

## 为什么是「代理」而不是「把 Langfuse 的页面嵌进工作台」

**iframe 那条路被 Langfuse 明确堵死了**:`us.cloud.langfuse.com` 的 CSP 里是
**`frame-ancestors 'none'`**(2026-09-27 实测)。⇒ 嵌入不成立,只能由服务端读
出数据、在工作台**自己**画一张表,再给一条**跳过去**的深链。

> ⚠️ 判据是 `frame-ancestors`,**不是** `X-Frame-Options` —— 后者在这些响应里
> **根本没有**。只看那一个头会得出**相反**的结论(「能嵌」),而真去嵌只会拿到
> 一个空白框、不报任何错。

同时:**密钥只留在服务端**。浏览器拿到的是已经整好的行,一个 `LANGFUSE_*`
都不下发。

## 为什么必须走 v2

`GET /api/public/traces` 在本组织上返回 **410
`LEGACY_API_UNAVAILABLE_FOR_NEW_ORGANIZATION`**(实测)⇒ 老端点不可用。
本模块用的是 `GET /api/public/v2/observations`,与仓里另外两处已经验证能通的
v2 调用同源(`scripts/acceptance_ch09.sh` 的 `fetch`、`scripts/intent_cost.py`
的 `_fetch`)。鉴权是 **HTTP Basic**(public : secret)。

## ⚠️ 本模块**唯一**的难点:三种状态必须分得开

| 状态 | 响应 | 页面该显示 |
|---|---|---|
| 观测没开(三个 `LANGFUSE_*` 不全) | `{"available": false, "reason": "观测未启用"}` | 「观测未启用 + 怎么开」 |
| 开着但连不上 / 超时 / 网关 4xx-5xx | `{"available": false, "reason": "连接失败: …"}` | 那一句原因(**红色**) |
| 通了但没有数据 | `{"available": true, "items": []}` | 「还没有链路数据」 |

**前两者渲染成空表就是本仓最忌讳的「故障被读成业务结果」** —— ch09 那次事故
的形状一模一样(Milvus 挂着,而 `POST /api/feedback` 照回 `200 {"pooled": true}`)。
⇒ 结构上把三者分开:**`available` 是唯一的判别键**,而 `items` 键**只在
`available: true` 时才存在** —— 页面若敢无视 `available` 去读 `items`,它会
拿到 `undefined` 而不是一个安静的空表(本仓「静默无效」家族的第五个成员,
不打算加入它)。

## 出站那一跳的两条纪律

1. **有超时**(`langfuse_api_timeout_seconds`,见 `app/config.py` 那一段)——
   「挂起不是异常」的教训见 `app/llm.py`;
2. **出站文本过脱敏**(`app/sanitize.py:redact_api_key`,两把钥匙都抹)。
   ⚠️ **刻意不取 `exc.response.text`** —— 那是网关原文,谁也不知道里面会不会
   带上它自己认识的凭据。诊断要的是状态码,不是响应体。

## 深链的形状是**实测**出来的,不是照记忆写的

| 路径(实测于 2026-09-27,拿真 id 打 `us.cloud.langfuse.com`) | 结果 |
|---|---|
| `/project/{projectId}/traces/{traceId}` | **200** |
| `/project/{projectId}/sessions/{sessionId}` | **200** |
| `/project/{projectId}/nonexistent-xyz`(**对照**) | **404** |
| `/sessions/{sessionId}`(不带项目) | **404** |
| `/trace/{traceId}`(旧式,无项目) | 307 → 上面第一条 |

⇒ 项目 id 是深链的**必需品**(它从 `GET /api/public/projects` 来)。
⚠️ 那个 404 对照是这条结论的**全部**依据 —— 没有它,「200」只说明「这是个存在的
路由前缀」。⚠️ 同时如实记账:**没能在浏览器里点开看一眼**(本机没有可用的浏览器
自动化)。「点开真的看得到东西」这一半是**推断**,依据是「路由存在 + id 是 API
刚给的真 id」,不是实测。
⇒ 因此项目 id 拿不到时**降级成不给链接**(`*_url: null`),页面只显示 id ——
**绝不给一个点开 404 的链接**。

## 这个模块不在实时对话的请求路径上

只有管理台的「链路」标签页切进来时才会打一次。它**不 import langfuse**
(全章唯一的 langfuse 边界仍是 `app/observability.py`,有源码扫描测试守着),
只用了它的 `enabled()` 这一个判据 —— 那是「三个键齐了吗」这条规则的**唯一**
实现,在这里重写一遍就是同一条规则两处实现。
"""

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Query

from app import observability
from app.config import Settings, get_settings
from app.sanitize import redact_api_key

logger = logging.getLogger(__name__)

router = APIRouter()

#: 页面每次切进「链路」都重拉(照「主题分布」那条),所以默认值给**小**。
DEFAULT_LIMIT = 20

#: **硬**上界。页面传 10000 不会真的打到 Langfuse —— 那是参数校验那一层的事
#: (`Query(ge=, le=)`),`tests/test_api_traces.py` 有一条走 HTTP 的用例钉着它。
#: 100 同时也是 Langfuse 自己那一页的上限,再大没有意义。
MAX_LIMIT = 100

_OBSERVATIONS_PATH = "/api/public/v2/observations"
_PROJECTS_PATH = "/api/public/projects"

#: 状态码 → 一句人话。**410 那条不是凑的**:`GET /api/public/traces` 在这个组织上
#: 就返回它(`LEGACY_API_UNAVAILABLE_FOR_NEW_ORGANIZATION`)—— 「HTTP 410」四个字
#: 不足以让人知道该改什么。查不到的状态码就只报码。
_STATUS_HINTS = {
    401: "密钥被拒,检查 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY",
    403: "密钥被拒(权限不足)",
    410: "该接口已下线(老端点不可用,需走 v2)",
    429: "被 Langfuse 限流",
}

#: 原因那一句的长度上界。出站异常文本可能很长,而它要进的是页面上一行。
_REASON_MAX = 200


def _base(settings: Settings) -> str:
    """`rstrip("/")` 是有意义的:`LANGFUSE_BASE_URL` 尾部多敲一个斜杠是常态,
    不去掉会拼出 `…com//api/public/…` —— 而 httpx **照样发得出去**、
    服务端**照样可能应答**。一个谁也不报错的错。"""
    return settings.langfuse_base_url.rstrip("/")


def _auth(settings: Settings) -> tuple[str, str]:
    """HTTP Basic 的两半,**顺序是 (public, secret)**,反了就是 401。"""
    return (settings.langfuse_public_key, settings.langfuse_secret_key)


def request_spec(settings: Settings, limit: int) -> tuple[str, dict[str, Any], tuple[str, str]]:
    """观测那一跳的 `(url, params, auth)` —— **纯函数,单测直接断言它**。

    抽出来的理由与 `observability._outer_cm` 同款:让「拼出来的那一串」可以在
    **不用任何替身**的前提下被验到。
    """
    return (f"{_base(settings)}{_OBSERVATIONS_PATH}", {"limit": limit}, _auth(settings))


def projects_spec(settings: Settings) -> tuple[str, tuple[str, str]]:
    """项目那一跳(深链前缀要用项目 id)。"""
    return (f"{_base(settings)}{_PROJECTS_PATH}", _auth(settings))


def _transport() -> httpx.AsyncBaseTransport | None:
    """**测试缝**。正常返回 `None`(httpx 自己建连接)。

    ⚠️ 注入点刻意留在**传输层**,不是「fetch 函数的替身」:换成
    `httpx.MockTransport` 之后,**真实的 `httpx.AsyncClient` 照常**拼 URL、
    编码 query、生成 `Authorization: Basic …` 头、把响应喂给 `resp.json()` ——
    替身**只**换掉那一根 socket。把 `_get_json` 整个换掉的话,上面四步一步都
    不会被验到,而它们正是这个端点的全部工作(本仓编目过的「替身替被测对象
    完成了语义」)。
    """
    return None


async def _get_json(settings: Settings, url: str, params: dict[str, Any]) -> Any:
    """一次带界的 GET。**不 catch** —— 翻译成人话是调用方的事。"""
    async with httpx.AsyncClient(
        timeout=settings.langfuse_api_timeout_seconds, transport=_transport()
    ) as http:
        resp = await http.get(url, params=params, auth=_auth(settings))
        resp.raise_for_status()
        return resp.json()


def _reason_for(exc: Exception, settings: Settings) -> str:
    """把一次出站故障翻译成**给人看的一句话**。

    只带三样东西:**类型 / 状态码 / 短消息**。刻意**不取** `exc.response.text`
    (网关原文,可能带凭据);两把钥匙都过一遍 `redact_api_key`
    —— 异常文本里会不会出现它们不由我们决定,而这条规矩是「**所有**出站错误
    文本必须过脱敏」。
    """
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        detail = f"HTTP {code}"
        hint = _STATUS_HINTS.get(code)
        if hint:
            detail += f"({hint})"
    elif isinstance(exc, httpx.TimeoutException):
        # 把**那个上界**写出来:看到「超时」而不知道是 5 秒还是 5 分钟,没法查。
        detail = f"请求超时(>{settings.langfuse_api_timeout_seconds:g}s)"
    else:
        detail = f"{type(exc).__name__}: {exc}"
    detail = redact_api_key(detail, settings.langfuse_secret_key)
    detail = redact_api_key(detail, settings.langfuse_public_key)
    return f"连接失败:{detail[:_REASON_MAX]}"


def _to_item(row: dict[str, Any], project_url: str | None) -> dict[str, Any]:
    """一条观测 → 页面那一行。

    ⚠️ 字段名是**真 API 的驼峰**(`sessionId` / `traceId` / `startTime`)——
    这里翻成下划线,`traces` 模块是**唯一**知道那个驼峰的地方(页面只认下面
    这组键,由 `tests/test_api_traces.py` 那条键集断言钉着)。
    """
    trace_id = row.get("traceId")
    session_id = row.get("sessionId")
    return {
        "id": row.get("id"),
        "trace_id": trace_id,
        "session_id": session_id,
        "name": row.get("name"),
        "latency": row.get("latency"),
        "start_time": row.get("startTime"),
        "level": row.get("level"),
        "type": row.get("type"),
        # 真数据里它是 `null`(trace 级标签不在观测级返回)⇒ 给页面一个可迭代的空表。
        "tags": row.get("tags") or [],
        # `project_url` 拿不到时**给 None,不给一个拼错的链接**(见模块 docstring)。
        "trace_url": f"{project_url}/traces/{trace_id}" if project_url and trace_id else None,
        "session_url": (
            f"{project_url}/sessions/{session_id}" if project_url and session_id else None
        ),
    }


async def _project_url(settings: Settings) -> str | None:
    """`{base}/project/{projectId}` —— 深链的前缀。拿不到就 None。

    **刻意不缓存**:缓存要么是进程级全局(它会跨用例、跨「换了一份 settings」
    残留 —— 本仓吃过那类污染),要么得做一把锁。而这一跳的代价是**一次**
    小请求,页面每次切进来也就多它一个。

    ⚠️ 它**只是**深链前缀 ⇒ 它失败**不该**让整页判成「不可用」:
    `recent` 因此把它单独 try 住(数据已经拿到了,只是没法给链接)。
    """
    url, _ = projects_spec(settings)
    try:
        payload = await _get_json(settings, url, {})
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            return None
        pid = rows[0].get("id") if isinstance(rows[0], dict) else None
        return f"{_base(settings)}/project/{pid}" if pid else None
    except Exception:  # noqa: BLE001 —— 见 docstring:它失败不改变「数据可用」
        logger.warning("读 Langfuse 项目 id 失败(深链降级为不给链接)", exc_info=True)
        return None


@router.get("/api/traces/recent")
async def recent(
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """`GET /api/traces/recent` —— 只读,不改任何东西。

    契约:**永远回 200 + 一个能直接渲染的结构**,哪怕 Langfuse 挂了、密钥没了、
    响应不是 JSON。三种状态见模块 docstring 的表。
    """
    if not observability.enabled(settings):
        # ⚠️ 这一句必须在**发请求之前**:观测没开时一个字节都不该出去。
        return {"available": False, "reason": "观测未启用"}

    url, params, _ = request_spec(settings, limit)
    try:
        payload = await _get_json(settings, url, params)
    except Exception as exc:  # noqa: BLE001 —— 契约是「回结构」,不是「把异常穿出去」
        # ⚠️ **宽 catch 是有意的,而 reason 里带异常类型名** —— 那样一个编程错误
        # (比如我自己的映射写错)在页面上是「连接失败:AttributeError: …」,
        # 一眼能看出不是网络问题,而不是被伪装成「Langfuse 连不上」。
        logger.warning("读 Langfuse 观测失败", exc_info=True)
        return {"available": False, "reason": _reason_for(exc, settings)}

    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        # 200 但形状不对(网关换了版本 / 前面站了个代理)—— 同样是「连不上」这一类。
        return {"available": False, "reason": "连接失败:响应里没有 data 数组"}

    project_url = await _project_url(settings)
    return {
        "available": True,
        "reason": None,
        "project_url": project_url,
        "items": [_to_item(r, project_url) for r in rows if isinstance(r, dict)],
    }
