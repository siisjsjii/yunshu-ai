"""最小实测:interrupt 在 astream 各 stream_mode 下怎么浮出来,以及 resume 行为。
不联网、不碰业务代码。"""
import asyncio, sys
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

def out(s=""):
    sys.stdout.buffer.write((s + "\n").encode("utf-8")); sys.stdout.buffer.flush()

class S(TypedDict):
    order_no: str
    trace: list

def ask_order(state):
    out("    [节点内] 走到 interrupt 之前")
    picked = interrupt({"frame": "order_choice", "options": ["1001", "1002"]})
    out(f"    [节点内] resume 返回值 = {picked!r}(interrupt 之前的代码**又跑了一遍**)")
    return {"order_no": picked, "trace": ["asked_order"]}

def after(state):
    return {"trace": ["after"], "order_no": state.get("order_no", "")}

g = StateGraph(S)
g.add_node("ask_order", ask_order); g.add_node("after", after)
g.add_edge(START, "ask_order"); g.add_edge("ask_order", "after"); g.add_edge("after", END)
app = g.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "t1"}}

async def main():
    out("=== 1) astream(stream_mode='custom') 首次运行 ===")
    async for x in app.astream({"order_no": "", "trace": []}, config=cfg, stream_mode="custom"):
        out(f"    custom 收到:{x!r}")

    st = await app.aget_state(cfg)
    out(f"    跑完后 state.next = {st.next!r}   values.order_no={st.values.get('order_no')!r}")
    out(f"    state.tasks = {[(t.name, bool(getattr(t,'interrupts',None))) for t in st.tasks]}")

    out("\n=== 2) 同一 thread 用 Command(resume=...) 续跑(custom) ===")
    async for x in app.astream(Command(resume="1002"), config=cfg, stream_mode="custom"):
        out(f"    custom 收到:{x!r}")
    st2 = await app.aget_state(cfg)
    out(f"    续跑后 state.next = {st2.next!r}  order_no={st2.values.get('order_no')!r}")

    out("\n=== 3) 换个新 thread:看 astream 多模式里 __interrupt__ 长什么样 ===")
    cfg2 = {"configurable": {"thread_id": "t2"}}
    async for mode, chunk in app.astream({"order_no": "", "trace": []}, config=cfg2,
                                          stream_mode=["custom", "updates"]):
        out(f"    mode={mode!r} chunk={str(chunk)[:160]}")

asyncio.run(main())
