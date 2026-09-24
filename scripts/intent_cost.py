"""按意图统计 token 花销 —— 打 Langfuse Metrics API v2。

用法:
    .venv/Scripts/python.exe scripts/intent_cost.py
    .venv/Scripts/python.exe scripts/intent_cost.py --minutes 120
    .venv/Scripts/python.exe scripts/intent_cost.py --intent 物流

⚠️ 四条**实测**出来的坑(spec §3.5),别凭文档改:

1. `query` 必须是**一个 JSON 字符串参数**:`params={"query": json.dumps({...})}`。
   把 view/metrics 平铺成独立 query 参数会 400。
2. **`sum_totalCost` 恒为 0** —— 本项目的模型没在 Langfuse 里配价格。
   所以这张表报 **token**,不报钱;cost 只在非零时才打印。
   **不许把 0 当成本报出去。**
   ⚠️ 但这个 0 必须是**从响应里读出来的**:`metrics` 里不请求 `totalCost` 的话,
   `sum_totalCost` 这个键**根本不在响应里**(`r.get(...)` 恒为 None)⇒ 那句
   「cost 恒为 0」就退化成**断言**、而 `total_cost > 0` 那条分支是**死代码**。
   所以下面 metrics 里显式请求了它(实测键名就是 `sum_totalCost`,值是数字 `0`)。
3. 按 `tags` **过滤**时三个字段都有**实测**约束,缺一个就 400(见 --intent)。
   网关原话逐字抄在这里,别凭文档改:
   - `"type"` 必须是 `"arrayOptions"`(不是 `"string"`),而且**这个键不能省**:
     少给 `type` 时网关回 `{"code":"invalid_union","note":"No matching discriminator",
     "discriminator":"type", …}`;给 `"string"` 时回
     `Filter type 'string' is not supported for dimension type 'string[]'. Expected 'arrayOptions'.`
   - `"operator"` 必须是 **`"any of"` / `"none of"` / `"all of"`** 之一 ——
     **`"contains"` 不是 arrayOptions 的合法算子**(它是 string 那一组的),给了会回
     `Invalid option: expected one of "any of"|"none of"|"all of"`。
   - `"value"` 必须是**数组**:给字符串回 `expected array, received string`。
4. **`--intent` 那一路的响应行里没有 `tags` 键** —— 因为那一版把 `dimensions` 置空,
   回来的是一行**总聚合**(实测:`{"sum_totalTokens": "4850", "count_count": "26"}`)。
   过滤已经在**服务端**做完了,所以这一路**不能**照按 tag 分组那条路去挑 tags。

ingestion 有延迟,所以这里**轮询**到有数据为止,而不是读一次就下结论。
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
# `python scripts/x.py` 时 sys.path[0] 是 **scripts/**、不是仓库根,所以要先补上
# (照 scripts/ 里其余脚本的老样子),否则 `from app.sanitize import …` 直接 ImportError。
sys.path.insert(0, str(REPO))

from app.sanitize import redact_api_key  # noqa: E402  (必须在 sys.path 之后)


def emit(text: str) -> None:
    """cp936 陷阱:非 ASCII 一律走 buffer,别依赖控制台 codec。"""
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()


def _dotenv(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{name} 未在 .env 里配置")


async def _fetch(query: dict) -> dict:
    base = _dotenv("LANGFUSE_BASE_URL").rstrip("/")
    # 脱敏要用**这个脚本自己那把密钥**。本仓**其余 13 处**调用传的都是
    # `openai_api_key`(2026-09-25 最终修复轮重数,口径与做法见下),那是因为
    # 它们处理的是**上游 openai SDK** 的异常文本;这条规矩的**目的**是
    # 「出站文本不许回显凭据」,所以按**碰的是哪把**来传 —— 这里碰的是 Langfuse 那把。
    # 顺带:本脚本**不需要** `OPENAI_API_KEY`,不引入那个无关依赖。
    #
    # ⚠️ 这个数**曾经写作 17,是错的**(spec §15.14 ② 早已撤回那个说法,代码注释
    # 没跟上)。重数的方法与读数:`grep -rn "redact_api_key(" --include=*.py app/`,
    # 去掉 `def`,**按行数调用点** —— `chat.py` 6 / `extract.py` 2 / `refund.py` 1 /
    # `review.py` 1 / `flywheel/tasks.py` 1 / `kb/orchestrate.py` 1 /
    # `memory/tasks.py` 1 = **13**(`app/sanitize.py` 那一处是定义,不算;
    # `memory/tasks.py:248` 那句是注释,也不算)。本文件自己那 2 处
    # (下面两条 `SystemExit`)不在这 13 里 —— 它们传的正是这把 `secret`。
    # 后来人若再看到「N 处」这类计数,先按这个口径重数一遍再引用。
    secret = _dotenv("LANGFUSE_SECRET_KEY")
    auth = (_dotenv("LANGFUSE_PUBLIC_KEY"), secret)
    last = None
    for i in range(4):
        try:
            async with httpx.AsyncClient(timeout=40) as http:
                resp = await http.get(
                    f"{base}/api/public/v2/metrics",
                    params={"query": json.dumps(query)},
                    auth=auth,
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            # 网关的**原话在响应体里**,而 `raise_for_status` 的异常文本只带 URL、**不带 body**
            # ⇒ 查询写错时只能看到一个 30 行的 traceback,真正的原因(哪个字段、合法值是什么)
            # 一个字都看不到。今天定位 arrayOptions 的合法算子正是靠这段 body,所以把它抬上来。
            # 这类 400 是**确定性**的(查询形状不对),不进重试白名单。
            # 出站文本过脱敏:回显的是**响应体原文**,谁也不知道网关哪天会不会在里面
            # 带上它自己认识的凭据 —— 一律过一遍。
            raise SystemExit(
                "Langfuse 拒绝了这个查询:"
                f"{exc.response.status_code} {redact_api_key(exc.response.text[:800], secret)}"
            )
        except (httpx.ConnectError, httpx.ReadError) as exc:
            last = exc
            await asyncio.sleep(1.5 * (i + 1))
    raise SystemExit(f"连不上 Langfuse:{redact_api_key(str(last), secret)}")


async def main() -> None:
    ap = argparse.ArgumentParser(description="按意图统计 token 花销")
    ap.add_argument("--minutes", type=int, default=60, help="回看窗口(分钟)")
    ap.add_argument("--intent", default="", help="只看某一个意图标签(如 物流)")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    query = {
        "view": "observations",                       # v2 只支持 observations / scores-*
        "metrics": [
            {"measure": "totalTokens", "aggregation": "sum"},
            {"measure": "count", "aggregation": "count"},
            # 必须**显式请求** cost,否则 sum_totalCost 键不在响应里 ⇒ 「cost 恒为 0」
            # 那句是凭空的(见 docstring 第 2 条)。
            {"measure": "totalCost", "aggregation": "sum"},
        ],
        "dimensions": [{"field": "tags"}],
        "filters": [],
        "fromTimestamp": (now - timedelta(minutes=args.minutes)).isoformat(),
        "toTimestamp": (now + timedelta(minutes=5)).isoformat(),
        "config": {"row_limit": 100},
    }
    if args.intent:
        query["dimensions"] = []
        query["filters"] = [{
            "column": "tags",
            "operator": "any of",                     # ← arrayOptions 那组算子,不是 "contains"
            "value": [f"intent:{args.intent}"],       # ← 数组,不是字符串
            "type": "arrayOptions",                   # ← 不是 "string",且**这个键不能省**
        }]

    rows = (await _fetch(query)).get("data", [])
    if args.intent:
        # --intent 把 dimensions 置空 ⇒ 回来的是**一行总聚合**,那一行**没有 `tags` 键**
        # (见 docstring 第 4 条)。过滤已在服务端做完,这里只需把请求的意图名贴回去;
        # 若照上面那条路去挑 tags,会恒判「没有找到」而不报任何错。
        # 顺带滤掉 count 为 0 的空桶(不存在的意图会回一行 0/0 —— 那不是「有数据」)。
        usage = [dict(r, tags=[f"intent:{args.intent}"]) for r in rows
                 if int(r.get("count_count") or 0) > 0]
    else:
        usage = [r for r in rows
                 if any(str(t).startswith("intent:") for t in (r.get("tags") or []))]

    emit(f"\n===== 按意图的 token 花销(近 {args.minutes} 分钟)=====\n")
    if not usage:
        if args.intent:
            emit(f"窗口内没有任何带 intent:{args.intent} 标签的观测。")
        else:
            emit("没有找到任何带 intent:* 标签的观测。")
        emit("可能原因:① 窗口内没有新请求 ② Langfuse ingestion 还没到"
             "(v2 端点有延迟,等 1–2 分钟再试) ③ 意图标签没打上(spec §3.4)")
        return

    table = []
    for r in usage:
        labels = [t.split(":", 1)[1] for t in r["tags"] if str(t).startswith("intent:")]
        if not labels:
            continue
        table.append({
            "intent": "/".join(labels),
            "observations": int(r.get("count_count") or 0),
            "tokens": int(r.get("sum_totalTokens") or 0),
            "cost": float(r.get("sum_totalCost") or 0),
        })
    table.sort(key=lambda x: -x["tokens"])

    emit(f"{'意图':<12}{'观测数':>8}{'token':>12}")
    emit("-" * 34)
    for row in table:
        emit(f"{row['intent']:<12}{row['observations']:>8}{row['tokens']:>12}")
    top = table[0]
    emit(f"\n最烧 token 的意图:**{top['intent']}**({top['tokens']} tokens)")

    total_cost = sum(r["cost"] for r in table)
    if total_cost > 0:
        emit(f"\n(成本合计 {total_cost:.6f})")
    else:
        emit("\n(cost 恒为 0:本项目模型未在 Langfuse 里配置价格 —— 只报 token,不报钱)")


if __name__ == "__main__":
    # 两个流都要钉:`emit()` 走 stdout,而 `SystemExit` / traceback 走 **stderr**
    # —— 只钉 stdout 的话,那三条中文报错(如「Langfuse 拒绝了这个查询」)在
    # cp936 管道上会变成乱码(不崩,但一个字都读不出来)。
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    asyncio.run(main())
