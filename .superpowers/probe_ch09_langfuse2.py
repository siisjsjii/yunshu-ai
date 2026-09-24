"""ch09 探针 2:**trace 级 tags 到底怎么设才有效**,以及它能被 Metrics v2 按 tag 聚合吗。

探针 1 已证:`LangfuseSpan.update(**kwargs)` 的 kwargs 被**静默忽略**
(源码 docstring 逐字:`**kwargs: Additional keyword arguments (ignored)`)⇒
`root.update(**{"langfuse.trace.tags": [...]})` 什么都没发生。

本探针比三条路:

  A. `propagate_attributes(tags=[...])` **包住整段**(含两次模型调用)
  B. `propagate_attributes(tags=[...])` **中途进入**(手动 __enter__/__exit__),
     进入前有一次模型调用、进入后有一次 —— 模拟「意图在流中途才知道」
  C. `config["metadata"]["langfuse_tags"]` 的写法(文档说支持)

然后**等 ingestion**、轮询读回,再看 Metrics v2 按 tags 分组的成本。

⚠️ ingestion 有延迟(v2 端点尤甚),所以这里**轮询**而不是读一次就下结论。
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
    pk, sk = env_from_dotenv("LANGFUSE_PUBLIC_KEY"), env_from_dotenv("LANGFUSE_SECRET_KEY")
    base = env_from_dotenv("LANGFUSE_BASE_URL")
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", pk)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", sk)
    os.environ.setdefault("LANGFUSE_BASE_URL", base)

    from langfuse import get_client, propagate_attributes
    from langfuse.langchain import CallbackHandler

    client = get_client()
    model = create_chat_model(settings)
    handler = CallbackHandler()
    cfg = {"callbacks": [handler]}
    started = datetime.now(timezone.utc)
    ids: dict[str, str] = {}

    # ---- A: 包住整段 ------------------------------------------------------
    print("=== A. propagate_attributes 包住整段 ===")
    with propagate_attributes(trace_name="probe-A", session_id="sess-A",
                              tags=["intent:AAA", "ch09"]):
        with client.start_as_current_observation(name="A-root", as_type="span") as r:
            await model.ainvoke("回一个字:甲", config=cfg)
            await model.ainvoke("回一个字:乙", config=cfg)
            ids["A"] = client.get_current_trace_id()
            r.update(output={"ok": True})
    print("  A trace:", ids["A"])

    # ---- B: 中途进入 ------------------------------------------------------
    print("=== B. propagate_attributes 中途进入 ===")
    with client.start_as_current_observation(name="B-root", as_type="span") as r:
        await model.ainvoke("回一个字:丙", config=cfg)      # ← 进入**之前**,无 tag
        cm = propagate_attributes(trace_name="probe-B", session_id="sess-B",
                                  tags=["intent:BBB"])
        entered = False
        for how in ("aenter", "enter"):
            try:
                if how == "aenter":
                    await cm.__aenter__()
                else:
                    cm.__enter__()
                entered = True
                print(f"  中途进入成功:{how}")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  {how} 失败:{type(exc).__name__}: {str(exc)[:120]}")
        await model.ainvoke("回一个字:丁", config=cfg)      # ← 进入**之后**
        ids["B"] = client.get_current_trace_id()
        if entered:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                cm.__exit__(None, None, None)
        r.update(output={"ok": True})
    print("  B trace:", ids["B"])

    # ---- C: config metadata 写法 -----------------------------------------
    print("=== C. config['metadata']['langfuse_tags'] ===")
    with client.start_as_current_observation(name="C-root", as_type="span") as r:
        await model.ainvoke(
            "回一个字:戊",
            config={"callbacks": [handler],
                    "metadata": {"langfuse_tags": ["intent:CCC"],
                                 "langfuse_session_id": "sess-C"}},
        )
        ids["C"] = client.get_current_trace_id()
        r.update(output={"ok": True})
    print("  C trace:", ids["C"])

    client.flush()
    print("\nflushed;等 ingestion…")

    auth = (pk, sk)
    frm = (started - timedelta(minutes=5)).isoformat()
    to = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()

    async def get_json(url, params, tries=4):
        last = None
        for i in range(tries):
            try:
                async with httpx.AsyncClient(timeout=40) as http:
                    return await http.get(url, params=params, auth=auth)
            except httpx.ConnectError as exc:
                last = exc
                await asyncio.sleep(1.5 * (i + 1))
        raise last

    async def read_trace(tid, *, deadline=150):
        """轮询到出现为止 —— v2 端点有 ingestion 延迟。"""
        waited = 0
        while waited < deadline:
            resp = await get_json(f"{base}/api/public/v2/observations",
                                  {"traceId": tid, "fromStartTime": frm,
                                   "toStartTime": to, "limit": 50})
            rows = resp.json().get("data", []) if resp.status_code == 200 else []
            if rows:
                return rows, waited
            await asyncio.sleep(10)
            waited += 10
        return [], waited

    await asyncio.sleep(20)
    for label in ("A", "B", "C"):
        rows, waited = await read_trace(ids[label])
        print(f"\n=== 读回 {label}(等了 {waited}s)===")
        if not rows:
            print("  !!! 仍未出现")
            continue
        for o in rows:
            print(f"  type={o.get('type'):11} name={str(o.get('name'))[:22]:24} "
                  f"tags={o.get('tags')}")
        root = [o for o in rows if o.get("name", "").endswith("-root")]
        if root:
            keys = {k: root[0].get(k) for k in
                    ("tags", "sessionId", "traceName", "userId", "metadata")
                    if k in root[0]}
            print("  根观测的 trace 级字段:", keys)

    # ---- Metrics v2 按 tags ----------------------------------------------
    print("\n=== Metrics v2 按 tags 分组 ===")
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
    try:
        for row in json.loads(resp.text).get("data", [])[:15]:
            print("   ", {k: v for k, v in row.items() if k})
    except Exception:  # noqa: BLE001
        print("  body:", redact_api_key(resp.text, key)[:400])


if __name__ == "__main__":
    asyncio.run(main())
