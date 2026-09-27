"""链路端点(`GET /api/traces/recent`)—— 工作台「链路」标签页的那一跳。

## 这个文件守的是什么

**不是**「Langfuse 能不能连上」(那要在验收里真打一次),而是三件在本仓反复
栽过的事:

1. **三种状态必须分得开**(本端点**唯一**的难点)。
   `观测未启用` / `连接失败` / `通了但没数据` —— 前两者渲染成空表,就是本仓
   最忌讳的「**故障被读成业务结果**」(ch09 那次:Milvus 挂着,`POST /api/feedback`
   照回 `200 {"pooled": true}`)。判别力全在 `available` + `reason` + `items`
   **在不在**这三处的组合上。
2. **替身不许替被测对象完成语义**(本仓编目过的头号假绿形态)。
   ⇒ 注入点在**传输层**(`httpx.MockTransport`):真实的 `httpx.AsyncClient`
   照常拼 URL、编码 query、生成 `Authorization: Basic …` 头、把响应喂给
   `resp.json()`。替身**只**换掉那一根 socket。若把 `_get_json` 整个换成假的,
   上面四步一步都不会被验到 —— 而它们正是这个端点的全部工作。
3. **密钥不许进出站文本**(`app/sanitize.py` 那条规矩)。有一条用例专门
   把密钥塞进异常文本,断言它**没**出现在响应里。

## 不联网

单测全程不联网是硬约束 ⇒ 本文件**每一条**都带注入的假 transport。
`_settings()` 里的 `langfuse_base_url` 是 `https://lf.example`(一个绝不存在的
域名)—— 万一哪条用例忘了注入,它会**响亮地**连不上,而不是**悄悄**打到真
Langfuse 上。

## 真实的形状是哪来的

`_PAYLOAD` 是 **2026-09-27 从真 Langfuse(`us.cloud.langfuse.com`)逐字抄回来的**
响应体(`GET /api/public/v2/observations?limit=2`)—— 字段名一个没改。
抄回来的意义:字段名写错的话(比如把 `sessionId` 写成 `session_id`)这条
用例会当场红,而不是等到页面上「会话那一列全是空」时才发现。
⚠️ 真数据里 **`tags` 恒为 `null`**(trace 级标签不在观测级返回),所以那条
「`null` → `[]`」的断言是**照着真实状况**写的,不是凑的。
"""

import json

import httpx
import pytest

from app.api import traces
from app.config import Settings

#: ⚠️ 绝不存在的域名 —— 忘了注入 transport 时「响亮地失败」,而不是打真 Langfuse。
FAKE_BASE = "https://lf.example"

#: 两把**假**钥匙。真钥匙在 `.env` 里,单测永远不读它。
FAKE_PK = "pk-lf-fake-public"
FAKE_SK = "sk-lf-fake-secret-0123456789"

#: 2026-09-27 从真 Langfuse 抄回来的响应体(**逐字**,字段名一个没改)。
_PAYLOAD = {
    "data": [
        {
            "id": "1a643d280fc02415",
            "traceId": "b1b8dd4442adbccdc63cb04ab3080a51",
            "sessionId": "bf6fef2a070f44da971ba32d792eeb25",
            "name": "log_turn",
            "latency": 0.017,
            "startTime": "2026-09-27T09:18:14.303Z",
            "level": "DEFAULT",
            "isRootObservation": False,
            "tags": None,
            "type": "CHAIN",
        },
        {
            "id": "3772a5a815cd14dd",
            "traceId": "b1b8dd4442adbccdc63cb04ab3080a51",
            "sessionId": "bf6fef2a070f44da971ba32d792eeb25",
            "name": "chat",
            "latency": 4.09,
            "startTime": "2026-09-27T09:18:09.892Z",
            "level": "ERROR",
            "isRootObservation": True,
            "tags": None,
            "type": "SPAN",
        },
    ],
    "meta": {"cursor": "eyJ4IjogMX0="},
}

#: 与 `_PAYLOAD` 同一把 traceId / sessionId 的**项目 id**(同一份真捕获里来的)。
PROJECT_ID = "cmudlaooj03zwad0d8aalw2w1"
_PROJECTS_PAYLOAD = {"data": [{"id": PROJECT_ID, "name": "My Project"}]}


def _settings(**over) -> Settings:
    """⚠️ 硬约束:必须传 `_env_file=None`,否则仓库根的 `.env` 会把值补上。"""
    base = {
        "openai_base_url": "http://x",
        "openai_api_key": "k",
        "openai_model": "m",
        "database_url": "mysql://x",
        "langfuse_public_key": FAKE_PK,
        "langfuse_secret_key": FAKE_SK,
        "langfuse_base_url": FAKE_BASE,
    }
    return Settings(**{**base, **over}, _env_file=None)


def _disabled(**over) -> Settings:
    return _settings(langfuse_public_key="", langfuse_secret_key="", **over)


class _Recorder:
    """把**发出去的每一个请求**记下来,再按路径回一份事先定好的东西。

    ⚠️ 它是**传输层**替身(换掉那一根 socket),不是「fetch 函数的替身」:
    URL 拼接、query 编码、Basic 头、`resp.json()` **全都还是产品代码在干**,
    这里只是让那些**可观测**。少了这层观测,「URL 拼错了」与「响应没解析」
    这两类实现错误在**所有**用例上都绿。
    """

    def __init__(self, routes: dict[str, object]) -> None:
        #: 路径后缀 → `(status, payload)` 或 `Exception`(抛出来)。
        self._routes = routes
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        for suffix, outcome in self._routes.items():
            if path.endswith(suffix):
                if isinstance(outcome, Exception):
                    raise outcome
                status, payload = outcome  # type: ignore[misc]
                if isinstance(payload, str):     # 非 JSON 正文(要在产品里炸)
                    return httpx.Response(status, text=payload)
                return httpx.Response(status, json=payload)
        raise AssertionError(f"用例没给这条路径准备响应:{path}(准备了 {list(self._routes)})")

    def auth_header(self, path_suffix: str) -> str:
        for r in self.requests:
            if r.url.path.endswith(path_suffix):
                return r.headers.get("authorization", "")
        raise AssertionError(f"根本没有请求打到 {path_suffix}:{[r.url.path for r in self.requests]}")

    def query_of(self, path_suffix: str) -> dict[str, str]:
        for r in self.requests:
            if r.url.path.endswith(path_suffix):
                return dict(r.url.params)
        raise AssertionError(f"根本没有请求打到 {path_suffix}")


def _use(monkeypatch, routes: dict[str, object]) -> _Recorder:
    """把传输层换成记录器。**返回它**,用例靠它断「发出去的是什么」。"""
    rec = _Recorder(routes)
    monkeypatch.setattr(traces, "_transport", lambda: httpx.MockTransport(rec.handler))
    return rec


_OBS = "/api/public/v2/observations"
_PROJ = "/api/public/projects"

_OK = {_OBS: (200, _PAYLOAD), _PROJ: (200, _PROJECTS_PAYLOAD)}


# ── 1. URL / 鉴权:纯函数,不用替身 ────────────────────────────────────────


def test_request_spec_builds_the_v2_url_and_basic_auth():
    """⚠️ 这条**不用任何替身** —— 它断的就是「拼出来的那一串对不对」。

    三条各有判别力:
    · URL 打的是 **v2**(`GET /api/public/traces` 在本组织上返回
      **410 LEGACY_API_UNAVAILABLE_FOR_NEW_ORGANIZATION**,照记忆写就会踩);
    · `base_url` 尾部的斜杠被吃掉(不去掉会得到 `…com//api/…`,而 httpx
      照样发出去、服务端照样可能应答 —— 静默的错);
    · 鉴权是 **HTTP Basic**,不是 Bearer、也不是塞进 query 的 `?key=`。
    """
    url, params, auth = traces.request_spec(_settings(), 20)
    assert url == f"{FAKE_BASE}/api/public/v2/observations", url
    assert "//api" not in url, f"base_url 尾部的斜杠没被吃掉:{url}"
    assert params == {"limit": 20}
    assert auth == (FAKE_PK, FAKE_SK), "Basic 的两半必须是 (public, secret),顺序反了是 401"

    # 尾部带斜杠 / 带空白也要拼对(部署时 `.env` 里多敲一个斜杠是常态)
    url2, _, _ = traces.request_spec(_settings(langfuse_base_url=FAKE_BASE + "/"), 20)
    assert url2 == f"{FAKE_BASE}/api/public/v2/observations", url2


def test_the_project_lookup_is_also_a_v2_call_to_the_configured_base():
    """深链前缀(`/project/{id}`)要靠**项目 id** 拼出来,而它也得从 API 拿。

    ⚠️ 这一跳**不许**硬编码 Cloud 的地址或项目 id —— 自部署的人换掉
    `LANGFUSE_BASE_URL` 之后,链接必须跟着走(硬编码的话页面上的链接会指向
    **别人的** Langfuse,而它可能正好打得开 —— 一个看不出错的错)。
    """
    url, auth = traces.projects_spec(_settings())
    assert url == f"{FAKE_BASE}/api/public/projects", url
    assert auth == (FAKE_PK, FAKE_SK)
    other, _ = traces.projects_spec(_settings(langfuse_base_url="https://lf.internal/"))
    assert other == "https://lf.internal/api/public/projects", other


# ── 2. 三种状态 ──────────────────────────────────────────────────────────
#
# 这三条是本文件的**核心**:它们断的是「三种状态在**数据形状**上就不一样」,
# 所以页面**不可能**把它们渲染成同一个样子。


@pytest.mark.anyio
async def test_not_enabled_is_a_reason_and_never_touches_the_network(monkeypatch):
    """**状态 ①:观测没开。** ⇒ `available: false` + 人话原因,**零次出站**。

    ⚠️ 「零次出站」是这条用例的一半价值:观测没开时**一个字节都不该出去**
    (这是 ch09 「关掉时整套 no-op 且不 import langfuse」那条纪律的同一件事,
    只不过这里连 HTTP 也不该发)。把 `enabled()` 那道判断删掉的话,
    `_Recorder` 会收到一个打到 `lf.example` 的请求 ⇒ 这条红。
    """
    rec = _use(monkeypatch, _OK)
    body = await traces.recent(settings=_disabled())
    assert body["available"] is False
    assert body["reason"] == "观测未启用"
    assert "items" not in body, (
        "没开的时候**不许**给一个空列表 —— 页面会把 `items: []` 渲染成"
        "「还没有链路数据」,而那正是把「没配」读成「没数据」(本仓最忌讳的那一类)"
    )
    assert rec.requests == [], f"观测没开却发了请求:{[r.url for r in rec.requests]}"


@pytest.mark.anyio
async def test_connect_failure_is_a_reason_not_a_500(monkeypatch):
    """**状态 ②:开着但连不上。** ⇒ 仍然是 `available: false` + 原因。

    判别力:① 不许把异常**穿出**端点(`httpx.ConnectError` 冒出去就是 500,
    而端点的契约是「永远回 200 + 一个能读的结构」);② `reason` 不许是
    `观测未启用`(那样与状态 ① 长得一模一样,页面分不出来);
    ③ 也不许退化成 `items: []`。
    """
    _use(monkeypatch, {_OBS: httpx.ConnectError("[Errno 11001] getaddrinfo failed")})
    body = await traces.recent(settings=_settings())
    assert body["available"] is False
    assert body["reason"] != "观测未启用", "连不上被误报成「没开」—— 两种故障混成一个"
    assert body["reason"].startswith("连接失败"), body["reason"]
    assert "items" not in body, "连不上不许退化成空列表"


@pytest.mark.anyio
async def test_timeout_says_so_and_names_the_budget(monkeypatch):
    """超时是**另一条**路径(`app/llm.py` 那条「挂起不是异常」的教训)。

    原因里要带**那个上界**,否则运维看到「超时」不知道是 5 秒还是 5 分钟。
    """
    _use(monkeypatch, {_OBS: httpx.ReadTimeout("read timed out")})
    body = await traces.recent(settings=_settings(langfuse_api_timeout_seconds=2.5))
    assert body["available"] is False
    assert body["reason"] == "连接失败:请求超时(>2.5s)", body["reason"]


@pytest.mark.anyio
async def test_http_error_status_is_reported_with_its_code(monkeypatch):
    """网关自己回 4xx/5xx(密钥被拒 / 限流 / 端点下线)也算「连不上」这一类。

    ⚠️ 410 要单独给一句人话:**`GET /api/public/traces` 在本组织上就返回
    410 `LEGACY_API_UNAVAILABLE_FOR_NEW_ORGANIZATION`** —— 真踩到时,
    「HTTP 410」这四个字不足以让人知道该改什么。
    """
    _use(monkeypatch, {_OBS: (401, {"message": "unauthorized"})})
    body = await traces.recent(settings=_settings())
    assert body["available"] is False
    assert "401" in body["reason"] and "密钥" in body["reason"], body["reason"]

    _use(monkeypatch, {_OBS: (410, {"message": "legacy"})})
    body = await traces.recent(settings=_settings())
    assert "410" in body["reason"] and "v2" in body["reason"], body["reason"]


@pytest.mark.anyio
async def test_ok_but_empty_is_available_true_with_empty_items(monkeypatch):
    """**状态 ③:通了但没有数据。** ⇒ `available: true` + `items: []`。

    ⚠️ 这条是状态 ③ 的**唯一**形状,与 ①② 的差别就在 `available` 与
    `items` **在不在**上 —— 页面的三分支因此各有各的判据,不会互相吞掉。
    """
    _use(monkeypatch, {_OBS: (200, {"data": [], "meta": {}}), _PROJ: (200, _PROJECTS_PAYLOAD)})
    body = await traces.recent(settings=_settings())
    assert body["available"] is True
    assert body["reason"] is None
    assert body["items"] == []
    assert "items" in body, "通了就必须给 items 键(哪怕是空表)—— 它正是 ③ 的判据"


# ── 3. 通了:出站那一跳 + 解析,全都要真的被验到 ──────────────────────────


@pytest.mark.anyio
async def test_items_come_from_a_real_payload_and_the_request_is_the_v2_one(monkeypatch):
    """⚠️ 这条是「替身不许替被测对象完成语义」的正面写法。

    替身只换 socket;URL / query / Basic 头 / `resp.json()` **全是产品代码做的**
    ⇒ 下面这几条断言才有对象可断:
    · `Authorization: Basic …` 是 **httpx 按 (pk, sk) 现编码的**,不是常量
      (断言解出来的那两半);
    · `limit` 真的进了 query(不传的话网关按它自己的默认值来,页面的「最近 N 条」
      就是假的);
    · 字段名逐条对得上真payload(`sessionId` / `traceId` / `startTime` ——
      写成下划线的话页面那几列会**全是空**,而没有任何东西报错)。
    """
    import base64

    rec = _use(monkeypatch, _OK)
    body = await traces.recent(settings=_settings(), limit=2)

    assert body["available"] is True
    assert len(body["items"]) == 2

    # ── 出站那一跳:路径 / query / Basic 头 ──
    assert rec.query_of(_OBS) == {"limit": "2"}, "limit 没进 query"
    header = rec.auth_header(_OBS)
    assert header.startswith("Basic "), f"鉴权不是 Basic:{header!r}"
    raw = base64.b64decode(header.split(" ", 1)[1]).decode()
    assert raw == f"{FAKE_PK}:{FAKE_SK}", (
        f"Basic 里的两半不对:{raw!r} —— basic 鉴权是 pk:sk,顺序/分隔符错了就是 401"
    )

    # ── 响应解析 ──
    first = body["items"][0]
    assert first["id"] == "1a643d280fc02415"
    assert first["trace_id"] == "b1b8dd4442adbccdc63cb04ab3080a51"
    assert first["session_id"] == "bf6fef2a070f44da971ba32d792eeb25"
    assert first["name"] == "log_turn"
    assert first["latency"] == 0.017
    assert first["start_time"] == "2026-09-27T09:18:14.303Z"
    assert first["level"] == "DEFAULT"
    assert first["type"] == "CHAIN"
    assert first["tags"] == [], "真数据里 tags 是 null ⇒ 给页面必须是可迭代的 []"
    second = body["items"][1]
    assert second["name"] == "chat" and second["level"] == "ERROR"

    # 键集是**与页面的契约**:少一个键页面那格就是 undefined(不报错),多一个键
    # 说明有人悄悄扩了形状。两边都要能被发现。
    assert set(first) == {
        "id", "trace_id", "session_id", "name", "latency", "start_time",
        "level", "type", "tags", "trace_url", "session_url",
    }, sorted(first)


@pytest.mark.anyio
async def test_the_outbound_call_really_carries_the_configured_timeout(monkeypatch):
    """⚠️ 本仓最贵的一条教训(`app/llm.py`):**不传 `timeout` ⇒ SDK 不设超时** ——
    对端一个字节都不回时那次 `await` 谁也等不回来,而**挂起不是异常** ⇒
    `finally` 永不执行、整套收尾逻辑作废。

    判别力在于**这个数真的到了传输层**:httpx 把解析好的四相超时放进
    `request.extensions["timeout"]`,**替身换不掉它** ⇒ 这条不是「我按我以为的
    形状调用了」(假 client 只能验那个),而是「**发出去的那个请求上真的挂着
    这个界**」。把 `timeout=settings.langfuse_api_timeout_seconds` 改成
    `timeout=None`,四相**全 None**,这条当场红。

    ⚠️ 刻意**不给二元组**:本仓记过「二元组会让 write/pool 落回 `None` = 又没上界
    了」—— 标量才是对的,所以这里断四相**逐个**相等。
    """
    rec = _use(monkeypatch, _OK)
    await traces.recent(settings=_settings(langfuse_api_timeout_seconds=2.5))
    ext = rec.requests[0].extensions.get("timeout")
    assert ext == {"connect": 2.5, "read": 2.5, "write": 2.5, "pool": 2.5}, (
        f"出站请求上没有挂着配置那个界(拿到 {ext!r})—— 对端不回话时这次调用永不返回"
    )


@pytest.mark.anyio
async def test_deep_links_point_at_the_project_scoped_ui_routes(monkeypatch):
    """深链的形状 —— **2026-09-27 实测核过**(见报告「深链核实」)。

    核法(不是照记忆):拿真 id 打 `us.cloud.langfuse.com`,三个形状对照:

    | 路径 | 实测 |
    |---|---|
    | `/project/{pid}/traces/{traceId}` | **200** |
    | `/project/{pid}/sessions/{sessionId}` | **200** |
    | `/project/{pid}/nonexistent-xyz`(对照)| **404** |
    | `/sessions/{sessionId}`(不带项目)| **404** |

    ⚠️ 那个 404 对照是这条结论的**全部**依据:没有它,「200」只说明「这是个
    存在的路由前缀」,什么也不能证明。
    ⚠️ **这**里只能断「我们拼的是那两个形状」;「点开真的看得到东西」是验收
    里的事 —— 单测不联网,这条路它够不到。
    """
    _use(monkeypatch, _OK)
    body = await traces.recent(settings=_settings())
    item = body["items"][0]
    root = f"{FAKE_BASE}/project/{PROJECT_ID}"
    assert body["project_url"] == root, body["project_url"]
    assert item["trace_url"] == (
        f"{root}/traces/b1b8dd4442adbccdc63cb04ab3080a51"
    ), item["trace_url"]
    assert item["session_url"] == (
        f"{root}/sessions/bf6fef2a070f44da971ba32d792eeb25"
    ), item["session_url"]


@pytest.mark.anyio
async def test_项目_id_拿不到时降级成不给链接_而不是给一个错链接(monkeypatch):
    """项目 id 那一跳失败 ⇒ `url` 全是 `None`,**不是**一个拼错的链接。

    判别力:降级要是写成「拿不到就拼一个空前缀」,页面会给出一个点开 404 的
    链接 —— 用户看到的是「工作台坏了」,而真实原因只是一个可选字段没拿到。
    页面按 `url 为 None ⇒ 只显示 id` 处理(与 brief 的「核实不了就降级」同款)。
    """
    _use(monkeypatch, {_OBS: (200, _PAYLOAD), _PROJ: httpx.ConnectError("boom")})
    body = await traces.recent(settings=_settings())
    assert body["available"] is True, "项目 id 只是**深链前缀**,拿不到不该把整页判成不可用"
    assert body["project_url"] is None
    assert [i["trace_url"] for i in body["items"]] == [None, None]
    assert [i["session_url"] for i in body["items"]] == [None, None]
    assert body["items"][0]["id"] == "1a643d280fc02415", "id 本身还是要给"


@pytest.mark.anyio
async def test_non_json_body_is_a_reason_not_a_crash(monkeypatch):
    """网关前面站了个代理、回了一页 HTML ⇒ 也是「连不上」这一类。

    ⚠️ 这条走的是 `resp.json()` 抛 `JSONDecodeError` 那条路 —— 它与前面几条
    (`httpx` 自己的异常)是**不同的**代码路径,忘了 catch 就是一个 500。
    """
    _use(monkeypatch, {_OBS: (200, "<html>502 Bad Gateway</html>")})
    body = await traces.recent(settings=_settings())
    assert body["available"] is False
    assert body["reason"].startswith("连接失败"), body["reason"]
    assert "items" not in body


# ── 4. 出站文本不许带密钥 ────────────────────────────────────────────────


@pytest.mark.anyio
async def test_the_secret_key_never_reaches_the_response(monkeypatch):
    """⚠️ 复述类断言要**对着真实来源**验 —— 让替身**真的**把密钥写进异常文本。

    本仓那条规矩(`app/sanitize.py`):所有出站错误文本必须过 `redact_api_key`。
    这里把两把钥匙都塞进异常 => 响应里一个都不许剩。

    反面:不塞的话「响应里没有密钥」是**恒真**的(异常文本里本来就没有),
    这条用例会变成一个漂亮的空壳。
    """
    _use(monkeypatch, {
        _OBS: httpx.ConnectError(f"refused: {FAKE_BASE}/x?key={FAKE_SK}&pk={FAKE_PK}"),
    })
    body = await traces.recent(settings=_settings())
    blob = json.dumps(body, ensure_ascii=False)
    assert FAKE_SK not in blob and FAKE_PK not in blob, f"密钥漏进了响应:{blob}"
    assert "***" in body["reason"], f"没看到脱敏痕迹,可能整段被丢了:{body['reason']}"


# ── 5. `limit` 有上界 ────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_limit_defaults_to_a_small_number_and_reaches_the_api(monkeypatch):
    """默认值要小 —— 页面每次切进来都重拉,拿 100 条是白烧网关。

    ⚠️ **必须走 HTTP**:直调端点函数时 `limit` 拿到的是那个 `Query(...)` 对象
    本身,不是它的 `default` —— 「默认值到底填进了没有」在直调路径上**测不到**
    (那是 FastAPI 解析参数时干的活)。
    """
    rec = _use(monkeypatch, _OK)
    async with _client(_settings()) as client:
        resp = await client.get("/api/traces/recent")
    assert resp.status_code == 200, resp.text[:200]
    assert rec.query_of(_OBS) == {"limit": str(traces.DEFAULT_LIMIT)}
    assert 0 < traces.DEFAULT_LIMIT <= traces.MAX_LIMIT


@pytest.mark.anyio
@pytest.mark.parametrize("bad", ["0", "-1", str(traces.MAX_LIMIT + 1), "10000"])
async def test_out_of_range_limit_is_422_and_never_reaches_langfuse(monkeypatch, bad):
    """⚠️ 上界是**硬**的:页面传 10000 不许真的发出去。

    ⚠️ 这条**必须**走 HTTP 才作数:直调端点函数会**绕过** FastAPI 的参数校验,
    于是「`Query(ge=…, le=…)` 到底在不在」根本测不到(本仓编目过的那类
    「替被测对象跳过了一步」)。走 ASGI 之后,断言的是「**请求在进 handler 之前
    就被拒了**」—— 422 而不是一次打到 Langfuse 的 10000 条查询。
    """
    rec = _use(monkeypatch, _OK)     # 注入记录器:万一它真发出去了,下面会红
    async with _client(_disabled()) as client:
        resp = await client.get("/api/traces/recent", params={"limit": bad})
    assert resp.status_code == 422, (
        f"limit={bad} 该在参数校验那一层被拒(422),实际 {resp.status_code}:{resp.text[:200]!r}"
    )
    assert rec.requests == [], f"越界的 limit 居然真的发出去了:{[r.url for r in rec.requests]}"


# ── 6. 走 HTTP:路由没被静态目录吞掉,且完整happy path真的通 ──────────────


def _client(settings: Settings):
    """ASGI 直连(不跑 lifespan、不建 portal 线程 —— 见 `test_api_topics_db.py`)。

    ⚠️ 必须**覆盖** `get_settings`:不覆盖的话端点到真 `.env` 去拿那三把真钥匙,
    于是这条「单测」会去打真的 Langfuse —— 违反「单测全程不联网」。
    `dependency_overrides` 用完要清掉,否则会漏给同进程后面的用例。
    """
    import contextlib

    from app.config import get_settings
    from app.main import app

    @contextlib.asynccontextmanager
    async def _cm():
        app.dependency_overrides[get_settings] = lambda: settings
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                yield client
        finally:
            app.dependency_overrides.pop(get_settings, None)

    return _cm()


@pytest.mark.anyio
async def test_the_endpoint_answers_over_http_and_is_not_swallowed_by_static():
    """真打一次 HTTP:路由**在** `mount("/")` 之前 ⇒ 拿到 JSON,不是静态 404。

    判别力:`app.main` 里把 `include_router(traces_router)` 挪到 `mount("/")`
    **之后**,静态目录的 catch-all 会先匹配 —— 这条当场变成 404(而服务照常起、
    别的端点全正常,是本仓记过的「新端点 404 但看起来一切正常」)。
    """
    async with _client(_disabled()) as client:
        resp = await client.get("/api/traces/recent")
    assert resp.status_code == 200, f"端点没应答({resp.status_code}):{resp.text[:200]!r}"
    assert resp.headers["content-type"].startswith("application/json"), resp.headers
    body = resp.json()
    assert body["available"] is False and body["reason"] == "观测未启用"


@pytest.mark.anyio
async def test_the_full_happy_path_over_http(monkeypatch):
    """开了观测时的完整一跳(仍然不联网 —— transport 被换掉)。

    这条与直调用例的区别:它走的是 FastAPI 的依赖注入 + 序列化,所以
    「`Depends(get_settings)` 装错了」这类错误在这儿才会露出来。
    """
    _use(monkeypatch, _OK)
    async with _client(_settings()) as client:
        resp = await client.get("/api/traces/recent", params={"limit": 2})
    assert resp.status_code == 200, resp.text[:300]
    body = resp.json()
    assert body["available"] is True
    assert [i["name"] for i in body["items"]] == ["log_turn", "chat"]
    assert body["project_url"] == f"{FAKE_BASE}/project/{PROJECT_ID}"
