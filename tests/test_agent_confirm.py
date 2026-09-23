"""确认流的两个节点。

⚠️ `confirm_write` 的三态判定是本文件的重点:`_decision` 必须**失败关闭** ——
认不出来的 resume 载荷一律不放行。默认放行的话,前端一个形状写错就**直接建出工单**,
而写操作是不可逆的。
"""

import pytest

from app.agent import confirm_nodes
from app.tools.executor import APPROVED, DENIED


class _Settings:
    tool_timeout_seconds = 10.0
    tool_retry_attempts = 2
    tool_retry_delay_seconds = 0.0
    # ch09:三个 LANGFUSE_* 任一为空 ⇒ 观测整套 no-op(「单测全程不联网」靠它守)。
    # **替身必须与生产 `Settings` 同形**:`observability.enabled()` 直接读这三个
    # 属性,而那次读发生在 `span()` 的 try **之外** —— 替身漏掉它们不是"降级",
    # 是一个 `AttributeError`,报错指向观测模块而不是"你的替身少字段"。
    # (本仓既定的修法是**扩替身、不给实现加兜底**。)
    langfuse_public_key = ""
    langfuse_secret_key = ""
    langfuse_base_url = ""


async def _no_audit(**_kwargs) -> None:
    """`record_audit` 的哨兵替身:什么都不做,**一行都不落库**。

    ⚠️ **必须是 `async def`,不能是 `lambda **kw: None`。**
    `execute_tool` 里那处是 `await record_audit(...)` —— 一个返回 `None` 的
    普通函数会当场 `TypeError: object NoneType can't be used in 'await'
    expression`,而它落在执行器那个 `except Exception` 里,于是被翻译成
    `ToolInfrastructureError("工具执行失败")`:测试红在一个**与被测行为
    毫无关系**的地方,报错指向「工具执行失败」而不是「你的替身不可 await」。
    (这正是「patch 目标修对了之后才浮出来」的那类故障:目标错着的时候
    替身**根本不被调用**,不可 await 这件事被完全掩盖。)

    **三处写路径共用这一个**(批准 / 取消 / 追加消息各一条):
    只 patch 其中一条的话,另外两条照样往 `tool_audit_logs` 写 ——
    而那张表是**验收 5 读的表**,测试残留会让那条验收从「唯一事实」
    退化成「其中一行是」。
    """


#: `execute_tool` 在**这个模块里**按这个名字查找它(模块顶层
#: `from app.tools.audit import record_audit`),所以 patch 的必须是
#: `app.tools.executor.record_audit` —— 不是 `app.agent.confirm_nodes.record_audit`
#: (那个模块从不 import 它)。三处 `monkeypatch.setattr` **一律不带
#: `raising=False`** —— 那一轮它正是「静默空操作」的成因:目标哪天被改名或挪走,
#: patch 就悄悄退化,而这个文件又开始往**验收 5 要读的那张表**写行、没有任何信号。
#: 不带它,那种漂移会变成 **patch 时刻的 `AttributeError`,响亮**。
_AUDIT_TARGET = "app.tools.executor.record_audit"


def _silent(payload) -> None:
    """不给 collector 的 `emit` —— 这些用例断的不是帧。"""
    return None


def _state(**over):
    base = {
        "conversation_id": "c1",
        "pending_write": {
            "tool_call_id": "call_1",
            "name": "create_ticket",
            "args": {"description": "耳机坏了", "ticket_type": "售后"},
            # ⚠️ **刻意与 `args` 不等**(多一个只有预览才有的键)。
            # 两者相等时 `test_confirm_write_payload_carries_the_preview`
            # 分不出 `pending.get("preview")` 与 `pending.get("args")` ——
            # 把实现改成读错的那一个**照样绿**,而用户看到的卡片就是另一份东西。
            "preview": {"description": "耳机坏了", "ticket_type": "售后",
                        "title": "建工单预览"},
        },
        "write_decision": "",
        "turn_messages": [],
        "messages": [],
    }
    base.update(over)
    return base


# ---- confirm_write:只有 interrupt --------------------------------------


@pytest.mark.anyio
async def test_confirm_write_payload_carries_the_preview(monkeypatch):
    seen: list[dict] = []

    def fake_interrupt(payload):
        seen.append(payload)
        return {"approved": True}

    monkeypatch.setattr(confirm_nodes, "interrupt", fake_interrupt)
    node = confirm_nodes.make_confirm_write_node()
    await node(_state())
    assert seen[0]["frame"] == "ticket_confirm"
    # 断言**整份**预览(含只有 `preview` 才有的那个键):少断言一个键
    # 就分不出实现读的是 `preview` 还是 `args`。
    assert seen[0]["preview"] == {"description": "耳机坏了", "ticket_type": "售后",
                                  "title": "建工单预览"}


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"approved": True}, APPROVED),
        ({"approved": False}, DENIED),
        ({"approved": "true"}, DENIED),      # 字符串不算批准
        ({"approved": 1}, DENIED),           # 真值不算,必须是 True
        ({}, DENIED),
        (None, DENIED),
        ("yes", DENIED),                     # 裸串不是约定形状
    ],
)
def test_decision_fails_closed(payload, expected):
    """**失败关闭**:认不出来的一律不放行。

    写操作不可逆 —— 默认 `APPROVED` 会让一个前端形状写错**直接建出工单**。
    """
    assert confirm_nodes._decision(payload) == expected


@pytest.mark.anyio
async def test_confirm_write_returns_the_decision(monkeypatch):
    monkeypatch.setattr(confirm_nodes, "interrupt", lambda payload: {"approved": True})
    node = confirm_nodes.make_confirm_write_node()
    out = await node(_state())
    assert out["write_decision"] == APPROVED


# ---- apply_write_decision:副作用恰一次 --------------------------------


class _Spec:
    name = "create_ticket"
    kind = "write"
    source = "builtin"
    input_schema = {
        "type": "object",
        "properties": {"description": {"type": "string"}, "ticket_type": {"type": "string"}},
        "required": ["description"],
    }


@pytest.mark.anyio
async def test_approved_writes_once_and_appends_the_tool_message(monkeypatch):
    calls: list = []

    class _Tool:
        async def ainvoke(self, call):
            calls.append(call)

            class _M:
                content = '{"ticket_no": "T-1"}'
            return _M()

    spec = _Spec()
    spec.tool = _Tool()
    # ⚠️ **要 patch 的是 `app.tools.executor.record_audit`,不是
    # `app.agent.confirm_nodes.record_audit`。** 后者在 `confirm_nodes` 里
    # **根本不存在**(那个模块从不 import 它),配上 `raising=False` 就是一条
    # **静默失效的空操作** —— 真实写入照跑,这个文件每轮往
    # **验收 5 要读的那张表**写 3 行。一个看起来在隔离、其实什么都没隔离的装置,
    # 比不写它还糟。(T8 的实现者实测上报;`executor` 才是
    # `execute_tool` 查找那个名字的地方。)
    monkeypatch.setattr(_AUDIT_TARGET, _no_audit)
    node = confirm_nodes.make_apply_write_decision_node(
        registry={"create_ticket": spec}, settings=_Settings(), emit=_silent
    )
    out = await node(_state(write_decision=APPROVED))
    assert len(calls) == 1
    assert [m.tool_call_id for m in out["turn_messages"]] == ["call_1"]
    assert out["pending_write"] == {}


@pytest.mark.anyio
async def test_denied_does_not_write(monkeypatch):
    calls: list = []

    class _Tool:
        async def ainvoke(self, call):
            calls.append(call)
            raise AssertionError("取消的调用**不许被执行**")

    spec = _Spec()
    spec.tool = _Tool()
    # 取消这条**也会**落审计(`permission_denied`)—— 同一条写路径,同样要挡。
    monkeypatch.setattr(_AUDIT_TARGET, _no_audit)
    node = confirm_nodes.make_apply_write_decision_node(
        registry={"create_ticket": spec}, settings=_Settings(), emit=_silent
    )
    out = await node(_state(write_decision=DENIED))
    assert calls == []
    assert out["pending_write"] == {}


@pytest.mark.anyio
async def test_turn_messages_are_appended_not_replaced(monkeypatch):
    """⚠️ **`turn_messages` 是覆写通道,不是追加通道。**

    它承载的是「本轮产生的**全部**消息」,由 `log_turn` 一次性落库。
    这里只返回 `[tool_msg]` 的话,**那条带 `tool_calls` 的 AIMessage 会被丢掉**
    —— 落库的历史里助手消息凭空少一条,而每一轮的回复看起来都正常。
    (这正是 ch07 记过的「累积 vs 覆写」那处坑的同款。)
    """
    from langchain_core.messages import AIMessage

    prior = AIMessage(content="", tool_calls=[
        {"name": "create_ticket", "args": {"description": "x"}, "id": "call_1"}
    ])

    class _Tool:
        async def ainvoke(self, call):
            class _M:
                content = "{}"
            return _M()

    spec = _Spec()
    spec.tool = _Tool()
    monkeypatch.setattr(_AUDIT_TARGET, _no_audit)
    node = confirm_nodes.make_apply_write_decision_node(
        registry={"create_ticket": spec}, settings=_Settings(), emit=_silent
    )
    out = await node(_state(write_decision=APPROVED, turn_messages=[prior]))
    assert len(out["turn_messages"]) == 2
    assert out["turn_messages"][0] is prior


@pytest.mark.anyio
async def test_apply_write_decision_refuses_an_empty_pending_write(monkeypatch):
    """**空 `pending_write` 是接线 bug,不是一次调用** —— 入口必须自己拒(T9)。

    空着往下走会造出一条 **`tool_call_id=""`** 的 ToolMessage:那构成
    「有 tool result、没有对应 tool_call」,上游同样直接 **400** ——
    而且它**看起来像一次正常结果**,没有任何东西会响。

    路由侧(`nodes.route_after_agent`)也守一次,但那是**调用方的自觉**;
    真正的不变量在**这个唯一写口**上(本仓元教训:不变量不要寄存在调用方的记忆里)。
    所以这条用例**直接调节点**,刻意绕过路由。

    ⚠️ 替身**不许自己抛**:替身一抛,`execute_tool` 的 `except Exception`
    会把它翻成 `ToolInfrastructureError` —— 于是「有没有那道守卫」两种实现
    **都**满足 `pytest.raises`,这条用例就成了恒真。这里让替身**正常返回**,
    守卫缺席时节点会**一路跑完并返回** ⇒ RED。
    """
    from app.tools.errors import ToolInfrastructureError

    calls: list = []

    class _Tool:
        async def ainvoke(self, call):
            calls.append(call)

            class _M:
                content = "{}"
            return _M()

    spec = _Spec()
    spec.tool = _Tool()
    monkeypatch.setattr(_AUDIT_TARGET, _no_audit)
    node = confirm_nodes.make_apply_write_decision_node(
        registry={"create_ticket": spec}, settings=_Settings(), emit=_silent
    )
    with pytest.raises(ToolInfrastructureError):
        await node(_state(pending_write={}, write_decision=APPROVED))
    assert calls == [], "空 pending_write 时不该执行任何工具"


# ---- 徽标结算:决议之后必须补一条 `tool_result` 帧(T10 的 bundled 修复)----
#
# ⚠️ **这条帧是前端那个徽标唯一能停下来的机会。**
# `agent` 的循环对每个调用**先**发 `tool_call` 帧、**再**执行;撞到待确认的
# 写调用时它 `continue` 了 ⇒ 那次调用的 `tool_result` **一帧都不发**。而
# `apply_write_decision` 原先根本不收 `emit`,决议之后也不补 ⇒ 前端徽标
# **一直转下去**(转着才是诚实的 —— 那次调用确实还没发生)。
#
# 为什么不「挂起前先把徽标关掉」:那会把徽标的语义改成「这个工具已经跑完了」,
# 而它**还在等用户**。结算发生在决议真正落地的那一处,也就是这里。


class _Emitter:
    """收帧替身。**只收不发** —— 断言的是「节点发了什么」。"""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    def __call__(self, payload) -> None:
        self.frames.append(payload)

    def results_for(self, tool_call_id: str) -> list[dict]:
        return [
            f for f in self.frames
            if f.get("frame") == "tool_result"
            and f.get("tool_call_id") == tool_call_id
        ]


def _deciding_node(*, tool, emit):
    spec = _Spec()
    spec.tool = tool
    return confirm_nodes.make_apply_write_decision_node(
        registry={"create_ticket": spec}, settings=_Settings(), emit=emit
    )


@pytest.mark.anyio
async def test_approved_write_emits_exactly_one_tool_result(monkeypatch):
    """批准 ⇒ **恰好一条** `tool_result` 帧,`tool_call_id` 与那次调用对上。

    **「恰好一条」是重点**:前端 `settleBadge` 结算的是**最后一个**同名且
    未结算的徽章 —— 多补一条会把**下一轮**那张徽章提前结算掉(那一轮甚至
    可能还没开始),于是它明明还在跑、看起来却已经结束。
    """

    class _Tool:
        async def ainvoke(self, call):
            class _M:
                content = '{"ticket_no": "T-1"}'
            return _M()

    monkeypatch.setattr(_AUDIT_TARGET, _no_audit)
    emit = _Emitter()
    await _deciding_node(tool=_Tool(), emit=emit)(_state(write_decision=APPROVED))

    results = emit.results_for("call_1")
    assert len(results) == 1, f"批准路径的 tool_result 帧数不为 1:{emit.frames}"
    # **逐字同款** —— 与 `agent` 循环里那条(`nodes.py`)四个键完全一致。
    # 少一个键都是真的故障:`ok` 前端要读(徽标画成功还是失败),端点也要读
    # (`api/chat.py` 拿它判要不要给 `summary` 过 `redact_api_key`)。
    assert results[0] == {
        "frame": "tool_result",
        "tool_call_id": "call_1",
        "ok": True,
        "summary": '{"ticket_no": "T-1"}',
    }


@pytest.mark.anyio
async def test_denied_write_emits_exactly_one_tool_result(monkeypatch):
    """取消 ⇒ **同样**恰好一条(漏了的话取消路径的徽标**一直转**)。

    `ok=False`:用户取消了,**不是**一次成功的调用 —— 徽标该画成失败态。
    """

    class _Tool:
        async def ainvoke(self, call):
            raise AssertionError("取消的调用**不许被执行**")

    monkeypatch.setattr(_AUDIT_TARGET, _no_audit)
    emit = _Emitter()
    await _deciding_node(tool=_Tool(), emit=emit)(_state(write_decision=DENIED))

    results = emit.results_for("call_1")
    assert len(results) == 1, f"取消路径的 tool_result 帧数不为 1:{emit.frames}"
    assert results[0]["ok"] is False
    # 断**键集**而不是 `summary` 的内容:那句是自由文本(本仓不为它写断言)。
    assert set(results[0]) == {"frame", "tool_call_id", "ok", "summary"}


# ---- 通道本身:声明 + 每轮清零 ------------------------------------------
#
# ⚠️ 上面每一条都是**直接调节点**的 —— 通道没声明、或清零被删掉,它们
# **一条都不会红**。而这两种漏法在生产里都只是「静默」:
# 前者让卡片预览永远是空的、批准永远不生效;后者让**上一轮批准过的写操作
# 这一轮自动放行**。所以下面两条刻意**不看节点返回的 dict**,
# 一条跑真图、一条跑真 `make_resolve_references_node`。


@pytest.mark.anyio
async def test_both_channels_survive_a_real_graph_round_trip():
    """**T4 的 Critical 的同型守卫**(ch06 的 `confidence` 就是这么丢的)。

    通道集合由 `StateGraph(ChatState)` 的**注解决定**;写未声明的通道
    LangGraph **静默丢弃** —— 只 `logger.warning`、**不抛**。
    所以节点级用例证明不了「通道存在」:这里跑一个**真图**,断言两个值
    真的落进了终态。
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    from app.agent.state import ChatState

    payload = {
        "tool_call_id": "call_1",
        "name": "create_ticket",
        "args": {"description": "耳机坏了"},
        "preview": {"description": "耳机坏了", "ticket_type": "售后"},
    }

    async def seed(state) -> dict:
        return {"pending_write": payload, "write_decision": APPROVED}

    graph = StateGraph(ChatState)
    graph.add_node("seed", seed)
    graph.add_edge(START, "seed")
    graph.add_edge("seed", END)
    compiled = graph.compile(checkpointer=InMemorySaver())

    out = await compiled.ainvoke(
        {"conversation_id": "c1"},
        config={"configurable": {"thread_id": "confirm-channels"}},
    )
    # `.get` 而不是下标:通道没声明时这个 key **根本不存在** ——
    # 下标会以 `KeyError` 的形式红掉,连这句「为什么」都印不出来。
    assert out.get("pending_write") == payload, (
        "通道没声明 —— 写入被 LangGraph 静默丢弃"
    )
    assert out.get("write_decision") == APPROVED


@pytest.mark.anyio
async def test_resolve_references_resets_the_two_confirm_channels():
    """每轮开头把这两个槽位清回初值(**通道与它的清零同处一地**)。

    漏了清零的后果是**跨轮串味**:checkpointer 是进程级单例、thread_id =
    session_id,未写的通道保留上一轮的值 ⇒ **上一轮批准过的写操作这一轮自动放行**。
    """
    from app.agent.nodes import make_resolve_references_node

    class _Echo:
        async def ainvoke(self, messages):
            from langchain_core.messages import AIMessage

            return AIMessage(content=messages[-1].content)

    node = make_resolve_references_node(model=_Echo())
    out = await node({
        "user_input": "在吗",
        "pending_write": {"tool_call_id": "call_1", "name": "create_ticket"},
        "write_decision": APPROVED,
    })

    # 同 `.get` 的理由:清零丢了时这个 key **根本不在**返回的 dict 里。
    assert out.get("pending_write") == {}, "清零丢了 —— 上一轮的待确认写操作会串到这一轮"
    assert out.get("write_decision") == "", "清零丢了 —— 上一轮批准过的写操作这一轮自动放行"
