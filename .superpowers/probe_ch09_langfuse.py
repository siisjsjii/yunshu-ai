"""ch09 探针:Langfuse 4.15.4 对 Cloud(4.41.0)真的能干什么。

spec §3 的三件事全靠这里的事实,一律**实测**,不照文档猜:

  A. 基础链路:根 span + CallbackHandler,模型调用是否自动成 generation、是否嵌套。
  B. 手动 span:as_type="tool" / "retriever" 能不能开、能不能挂 input/output。
  C. **trace 级属性**:`langfuse.trace.tags` 用 `span.update(**{...})` 事后设,
     能不能落库?能不能被读回?
  D. session_id 走 `propagate_attributes` / `config["metadata"]["langfuse_session_id"]`
     能不能落库?
  E. Metrics API v2 能不能按 tags 维度聚合出成本?

读回用 v2:`GET /api/public/v2/observations`(v1 的 `/traces` 已弃用)。
密钥只从 .env 读,输出一律过 redact_api_key。
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

import httpx

from app.config import get_settings
from app.llm import create_chat_model
from app.sanitize import redact_api_key


def env_from_dotenv(name: str) -> str:
    """直接从 .env 读,不走 Settings(本章尚未给 Settings 加这三个字段)。"""
    if os.environ.get(name):
        return os.environ[name]
    with open(".env", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{name} not found in .env")


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    settings = get_settings()
    key = settings.openai_api_key

    pk = env_from_dotenv("LANGFUSE_PUBLIC_KEY")
    sk = env_from_dotenv("LANGFUSE_SECRET_KEY")
    base = env_from_dotenv("LANGFUSE_BASE_URL")
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", pk)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", sk)
    os.environ.setdefault("LANGFUSE_BASE_URL", base)

    from langfuse import LangfuseOtelSpanAttributes as A
    from langfuse import get_client, propagate_attributes
    from langfuse.langchain import CallbackHandler

    client = get_client()
    print("auth_check:", client.auth_check())

    session_id = f"probe-{uuid.uuid4().hex[:8]}"
    intent_tag = "intent:商品咨询"
    model = create_chat_model(settings)
    started = datetime.now(timezone.utc)

    print("\n=== A/B/C/D: 开根 span,挂 handler,两次模型调用 + 两个手动 span ===")
    with propagate_attributes(trace_name="cs-chat-probe", session_id=session_id):
        with client.start_as_current_observation(
            name="chat", as_type="span", input={"user_input": "退货政策是什么"}
        ) as root:
            handler = CallbackHandler()
            cfg = {"callbacks": [handler]}

            r1 = await model.ainvoke("用一句话说什么是七天无理由退货。", config=cfg)
            print("  第1次模型调用 ok:", bool(r1.content))

            with client.start_as_current_observation(
                name="retrieval", as_type="retriever", input={"query": "退货"}
            ) as rspan:
                rspan.update(output={"chunks": [{"id": 1, "score": 0.7}]})

            with client.start_as_current_observation(
                name="tool:query_order", as_type="tool", input={"order_no": "20240915"}
            ) as tspan:
                tspan.update(output={"ok": True, "summary": "已发货"})

            # ---- C: 意图**事后**补写 trace 级 tags ----
            root.update(**{A.TRACE_TAGS: [intent_tag]})
            print("  root.update(langfuse.trace.tags) 调用完成")

            # ---- D: 也试一次走 config metadata 的写法 ----
            r2 = await model.ainvoke(
                "再说一句话。",
                config={
                    "callbacks": [handler],
                    "metadata": {"langfuse_session_id": session_id + "-META"},
                },
            )
            print("  第2次模型调用 ok:", bool(r2.content))

            trace_id = client.get_current_trace_id()
            root.update(output={"reply": "ok"})

    print("  trace_id:", trace_id)
    client.flush()
    print("  flushed")

    auth = (pk, sk)
    frm = (started - timedelta(minutes=5)).isoformat()
    to = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

    async def get_json(url: str, params: dict, *, tries: int = 4):
        """带退避的连接重试 —— 本机到 Cloud 会偶发 ConnectError。"""
        last = None
        for i in range(tries):
            try:
                async with httpx.AsyncClient(timeout=40) as http:
                    return await http.get(url, params=params, auth=auth)
            except httpx.ConnectError as exc:
                last = exc
                await asyncio.sleep(1.5 * (i + 1))
        raise last

    # ---- 读回:v2/observations -------------------------------------------
    print("\n=== 读回 observations(traceId 过滤)===")
    resp = await get_json(f"{base}/api/public/v2/observations",
                          {"traceId": trace_id, "fromStartTime": frm,
                           "toStartTime": to, "limit": 50})
    print("  status:", resp.status_code)
    if resp.status_code != 200:
        print("  body:", redact_api_key(resp.text, key)[:400])
    else:
        rows = resp.json().get("data", [])
        print(f"  条数={len(rows)}")
        for o in rows:
            print(f"    type={o.get('type'):11} name={o.get('name')!r:26} "
                  f"parent={o.get('parentObservationId')}")
        if rows:
            print("\n  --- 第一条(根)的全部键 ---")
            for k in sorted(rows[0]):
                v = rows[0][k]
                print(f"    {k:22} = {str(v)[:110]}")

    # ---- E: Metrics API v2 按 tags 聚合 ---------------------------------
    print("\n=== E. Metrics API v2(groupBy=tags)===")
    query = {
        "view": "observations",
        "metrics": [{"measure": "totalCost", "aggregation": "sum"},
                    {"measure": "totalTokens", "aggregation": "sum"},
                    {"measure": "count", "aggregation": "count"}],
        "dimensions": [{"field": "tags"}],
        "filters": [],
        "fromTimestamp": frm,
        "toTimestamp": to,
        "config": {"row_limit": 50},
    }
    resp = await get_json(f"{base}/api/public/v2/metrics", {"query": json.dumps(query)})
    print("  status:", resp.status_code)
    body = redact_api_key(resp.text, key)
    try:
        data = json.loads(body).get("data", [])
        print(f"  行数={len(data)}")
        for row in data[:12]:
            print("   ", {k: v for k, v in row.items() if k})
    except Exception:  # noqa: BLE001
        print("  body:", body[:600])

    # 只找我们这条 intent:商品咨询 有没有进 tags 维度
    print("\n  --- 按 intent tag 过滤(验收 5 的形态)---")
    f_query = dict(query)
    f_query["dimensions"] = []
    f_query["filters"] = [
        {"column": "tags", "operator": "contains", "value": intent_tag, "type": "string"}
    ]
    resp = await get_json(f"{base}/api/public/v2/metrics", {"query": json.dumps(f_query)})
    print("  status:", resp.status_code)
    print("  body:", redact_api_key(resp.text, key)[:500])


if __name__ == "__main__":
    asyncio.run(main())
