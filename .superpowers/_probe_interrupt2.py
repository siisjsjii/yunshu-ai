"""两个决定端点契约的行为,实测:
  A) 有 pending interrupt 时,用户**不发 resume 而是发一条普通新消息**,会怎样?
  B) 每请求**重新 compile** 图(同 checkpointer),resume 还能接上吗?(我们端点是这么干的)
"""
import asyncio, sys
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

def out(s=""):
    sys.stdout.buffer.write((s+"\n").encode("utf-8")); sys.stdout.buffer.flush()

class S(TypedDict):
    user_input: str
    order_no: str
    trace: list

def resolve(state):
    return {"trace": [f"resolve:{state.get('user_input','')[:12]}"]}

def pick(state):
    if state.get("order_no"):
        return {"trace": ["pick:已有订单号,不打断"]}
    got = interrupt({"frame": "order_choice", "options": ["1001", "1002"]})
    return {"order_no": got, "trace": ["pick:resume 回填"]}

def fetch(state):
    return {"trace": [f"fetch:{state.get('order_no')}"]}

def build(checkpointer):
    g = StateGraph(S)
    g.add_node("resolve", resolve); g.add_node("pick", pick); g.add_node("fetch", fetch)
    g.add_edge(START, "resolve"); g.add_edge("resolve", "pick"); g.add_edge("pick", "fetch"); g.add_edge("fetch", END)
    return g.compile(checkpointer=checkpointer)

CP = InMemorySaver()

async def main():
    cfg = {"configurable": {"thread_id": "x1"}}
    out("=== A) 先跑到 interrupt ===")
    async for m, c in build(CP).astream({"user_input": "这个能退吗", "order_no": "", "trace": []},
                                        config=cfg, stream_mode=["custom","updates"]):
        out(f"    {m}: {str(c)[:90]}")
    st = await CP.aget_tuple(cfg)
    out(f"    待续 next={st.checkpoint['channel_values'].get('trace')} / pending={bool(st.pending_writes)}")

    out("\n=== A2) 此时改发一条普通新消息(不给 resume)===")
    async for m, c in build(CP).astream({"user_input": "那物流呢", "order_no": "", "trace": []},
                                        config=cfg, stream_mode=["custom","updates"]):
        out(f"    {m}: {str(c)[:90]}")

    out("\n=== B) 新 thread:跑到 interrupt,然后用**新 compile 的图** resume ===")
    cfg2 = {"configurable": {"thread_id": "x2"}}
    async for m, c in build(CP).astream({"user_input": "这个能退吗", "order_no": "", "trace": []},
                                        config=cfg2, stream_mode=["custom","updates"]):
        pass
    out("    (已停在 interrupt)")
    async for m, c in build(CP).astream(Command(resume="1002"), config=cfg2,
                                        stream_mode=["custom","updates"]):
        out(f"    {m}: {str(c)[:110]}")

asyncio.run(main())
