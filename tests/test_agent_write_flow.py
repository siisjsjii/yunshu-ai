"""`agent` 撞到未确认写调用时的「停」与「续」。

⚠️ 每条断言断的都是**只在目标行为发生时才出现的字符串**
(spec §9.7 的 trace 标记)—— **不要断 `agent_steps`**:
ch05 的验收 5 断 `agent_steps >= 2`,而那个名字读作「步数」、实际是
「绑工具轮次的序号且把收敛轮也算进去」,那条断言**零判别力**还漏得掉真回归。
"""

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent import nodes as agent_nodes
from app.tools.executor import APPROVED, ERROR_CONFIRMATION_REQUIRED


class _Chunk:
    """最小 chunk 替身:`.text` + `.tool_calls` + 可相加。

    ⚠️ `tool_call` 条目必须带 `"type": "tool_call"` —— 本仓栽过:
    `BaseTool.ainvoke` 判「这是不是工具调用」**只看**这一个条件。
    """

    def __init__(self, text="", tool_calls=None):
        self.text = text
        self.tool_calls = tool_calls or []

    def __add__(self, other):
        return _Chunk(
            self.text + getattr(other, "text", ""),
            list(self.tool_calls) + list(getattr(other, "tool_calls", []) or []),
        )


class _Bound:
    """`bind_tools` 的产物 —— 真实 LangChain 里它是**另一个** runnable
    (`RunnableBinding`),原 `model` 保持未绑定。

    ⚠️ 替身必须同形。上一版让 `bind_tools` 返回 `self`,于是 `bound is model`
    ——「续跑那一轮走的是**未绑**的 `model`」这句话在替身里**根本不可观测**:
    两条路调的是同一个 `astream`、同一个 `calls` 计数器。而它正是本文件
    要钉的结构保证(那一轮在结构上不可能再发第二次写调用)。
    (同 T3 给替身补 `flush`、ch07 给 `FakeChunk` 换真基类:替身不忠实,
    守护的断言就是恒真的。)
    """

    def __init__(self, inner, tools):
        self._inner = inner
        self.tools = tools
        self.calls = 0

    async def astream(self, msgs):
        self.calls += 1
        async for chunk in self._inner.astream(msgs):
            yield chunk


class _Model:
    def __init__(self, rounds):
        self._rounds = list(rounds)
        self.calls = 0
        self.bound = None

    def bind_tools(self, tools):
        self.bound = _Bound(self, list(tools))
        return self.bound

    async def astream(self, msgs):
        self.calls += 1
        yield self._rounds.pop(0)


class _Settings:
    brand_name = "本店"
    max_agent_steps = 3
    agent_token_budget = 10**9


@pytest.fixture
def patch_ctx(monkeypatch):
    monkeypatch.setattr(agent_nodes, "build_context_messages", lambda **kw: [])
    monkeypatch.setattr(agent_nodes, "count_tokens", lambda s: 1)
    monkeypatch.setattr(agent_nodes, "render_evidence", lambda e: "")
    monkeypatch.setattr(agent_nodes, "journal", type("J", (), {
        "model_ctx": staticmethod(lambda **kw: None)
    }))
    monkeypatch.setattr(
        agent_nodes, "layers", type("L", (), {
            "resplit": staticmethod(lambda h, **kw: None)
        })
    )


def _state(**over):
    base = {
        "conversation_id": "c1",
        "resolved_input": "帮我建个工单",
        "history": [],
        "summary_text": "",
        "evidence": [],
        "summary_upto_msg_id": 0,
        "layer1_from_msg_id": 0,
        "turn_messages": [],
        "pending_write": {},
        "write_decision": "",
    }
    base.update(over)
    return base


@pytest.mark.anyio
async def test_write_call_stops_the_loop_and_records_pending(monkeypatch, patch_ctx):
    """撞到未确认的写调用 ⇒ 停循环 + 记 `pending_write`,**且不发 tool 结果**。"""
    write_call = {
        "name": "create_ticket",
        "args": {"description": "耳机坏了", "ticket_type": "售后"},
        "id": "call_1",
        "type": "tool_call",
    }
    model = _Model([_Chunk("好的。", [write_call])])

    async def fake_execute(**kw):
        from app.tools.executor import ToolOutcome

        return ToolOutcome(
            kw["tool_call"]["id"], kw["tool_call"]["name"], False,
            "需要确认", "需要确认", ERROR_CONFIRMATION_REQUIRED,
            preview=kw["tool_call"]["args"],
        )

    monkeypatch.setattr(agent_nodes, "execute_tool", fake_execute)
    node = agent_nodes.make_agent_node(
        model=model, tools=[], registry={}, settings=_Settings(),
        emit=lambda f: None, context_budget=None,
    )
    out = await node(_state())

    assert out["pending_write"]["tool_call_id"] == "call_1"
    assert out["pending_write"]["preview"]["ticket_type"] == "售后"
    assert "agent:write_pending tool=create_ticket" in out["trace"]
    # **不许**给它回灌 tool 结果 —— 那次调用根本没发生,由
    # `apply_write_decision` 在决议之后补上。
    assert all(not isinstance(m, ToolMessage) for m in out["turn_messages"])
    assert model.calls == 1


@pytest.mark.anyio
async def test_other_calls_in_the_same_round_still_get_tool_messages(
    monkeypatch, patch_ctx
):
    """**同一轮里的只读调用照常执行并回灌。**

    少回灌一个 tool 结果就构成「有 tool_calls 没有对应 tool 消息」,
    上游直接 400 —— 这是 CLAUDE.md 里已有的硬约束。

    ⚠️ **待确认的写调用必须排在前面。** 反过来的话(只读在前),一个
    「撞到写调用就从 `for` 里 `break` 出去」的错误实现**照样绿** ——
    只读那条的回灌在它之前就已经追加完了,这条断言就只剩下「返回时别把
    已经攒下的消息弄丢」那一点判别力。写调用在前,`break` 就会
    **在追加只读结果之前**离开循环 ⇒ RED。
    """
    write_call = {
        "name": "create_ticket", "args": {"description": "x"},
        "id": "call_w", "type": "tool_call",
    }
    read_call = {
        "name": "query_order", "args": {"order_id": "1002"},
        "id": "call_r", "type": "tool_call",
    }
    model = _Model([_Chunk("", [write_call, read_call])])

    async def fake_execute(**kw):
        from app.tools.executor import ToolOutcome

        name = kw["tool_call"]["name"]
        if name == "create_ticket":
            return ToolOutcome(
                "call_w", name, False, "需要确认", "需要确认",
                ERROR_CONFIRMATION_REQUIRED, preview=kw["tool_call"]["args"],
            )
        return ToolOutcome("call_r", name, True, "{}", "{}")

    monkeypatch.setattr(agent_nodes, "execute_tool", fake_execute)
    node = agent_nodes.make_agent_node(
        model=model, tools=[], registry={}, settings=_Settings(),
        emit=lambda f: None, context_budget=None,
    )
    out = await node(_state())
    ids = [m.tool_call_id for m in out["turn_messages"] if isinstance(m, ToolMessage)]
    assert ids == ["call_r"], "只读那条的回灌丢了"


@pytest.mark.anyio
async def test_continuation_round_is_unbound_and_keeps_turn_messages(
    monkeypatch, patch_ctx
):
    """续跑:**一轮不绑 tools**,并把这条新回复**追加**进 `turn_messages`。

    不绑 tools 是**结构保证** —— 那一轮模型在结构上不可能再触发第二次写。
    """
    prior = AIMessage(content="", tool_calls=[
        {"name": "create_ticket", "args": {"description": "x"}, "id": "call_1"}
    ])
    result_msg = ToolMessage(content='{"ticket_no": "T-1"}', tool_call_id="call_1")
    model = _Model([_Chunk("已为您建单:T-1")])

    node = agent_nodes.make_agent_node(
        model=model, tools=[], registry={}, settings=_Settings(),
        emit=lambda f: None, context_budget=None,
    )
    out = await node(
        _state(turn_messages=[prior, result_msg], pending_write={},
               write_decision=APPROVED)
    )
    assert model.calls == 1
    assert "T-1" in out["reply"]
    assert "agent:write_resumed" in out["trace"]
    # 追加而不是替换
    assert len(out["turn_messages"]) == 3
    # ⚠️ 断的是**绑了 tools 的那份没被用过**,不是「`model.bound is None`」:
    # `make_agent_node` 在**工厂**里就 `bind_tools` 了,所以 `bound` 永远是
    # 非 None 的;而只有「bound 是另一个对象」时,这条断言才真的在问
    # 「续跑那一轮走的是哪一份」。
    assert model.bound.calls == 0, "续跑那一轮**不能**走绑了 tools 的那份"


@pytest.mark.anyio
async def test_continuation_does_not_reset_agent_steps(monkeypatch, patch_ctx):
    """⚠️ 续跑时 `steps` 局部变量是 0,直接返回会**把 `agent_steps` 归零**。"""
    prior = AIMessage(content="", tool_calls=[
        {"name": "create_ticket", "args": {"description": "x"}, "id": "call_1"}
    ])
    model = _Model([_Chunk("已建单")])
    node = agent_nodes.make_agent_node(
        model=model, tools=[], registry={}, settings=_Settings(),
        emit=lambda f: None, context_budget=None,
    )
    out = await node(
        _state(turn_messages=[prior], pending_write={}, write_decision=APPROVED,
               agent_steps=2)
    )
    assert out["agent_steps"] == 2


# ---- 路由:空 `pending_write` **不许**进确认流 --------------------------
#
# ⚠️ 这一条**不能**用「跑一遍图看走到哪」代替:图会因为别的理由不走进
# `confirm_write`(比如条件边压根没接),而这里要钉的是**判据本身**。


def test_route_after_agent_sends_pending_writes_to_confirm_write():
    assert agent_nodes.route_after_agent(
        {"pending_write": {"tool_call_id": "call_1", "name": "create_ticket"}}
    ) == "confirm_write"


def test_route_after_agent_sends_everything_else_to_log_turn():
    """**空 `pending_write` 一律汇进 `log_turn`。**

    这条是「空 `pending_write` 不许进 `apply_write_decision`」的**路由那一半**
    (另一半是那个节点入口自己的判断):空着进去会造出 `tool_call_id=""` 的
    ToolMessage ⇒ 上游 400。
    """
    assert agent_nodes.route_after_agent({"pending_write": {}}) == "log_turn"
    # 通道**整个缺席**(直接调路由、或将来多个入口)时同样不许放行 ——
    # 用 `state["pending_write"]` 取值的话这里会 `KeyError`,而路由函数
    # 抛异常时 LangGraph 报的是「路由函数出错」,指向的不是这条不变量。
    assert agent_nodes.route_after_agent({}) == "log_turn"


def test_agent_is_wired_as_a_conditional_outlet_not_a_plain_one():
    """**接线本身**:`agent` 不能再是无条件汇进 `log_turn` 的出口。

    把它也连上 `log_turn` 的后果不是「功能少一个」:挂起那一轮会在
    `confirm_write` 之前**先把这一轮落库**(一半写库、一半没写),
    而单测若不显式 resume 就走不到那儿。
    """
    from app.agent.graph import _OUTLETS

    assert "agent" not in _OUTLETS
    # 其余出口一个都不能顺手删掉 —— `assert "agent" not in _OUTLETS`
    # 对**空元组**同样成立。
    assert set(_OUTLETS) == {
        "complaint_reply", "chitchat_reply", "fallback_reply",
        "refund_offer", "refund_explain",
    }


# ---- 端到端:挂起 → 决议 → 续跑,写调用**恰好执行一次** --------------------
#
# ⚠️ **计数器放在 `ainvoke` 边界上,不放工具体里。** `@tool` 的参数校验发生在
# 函数体**之外**(pydantic 那层),参数不合法时工具体根本不进入 —— 按工具体
# 计数恒为 0,「执行了一次还是两次」就区分不出来了。本仓已记过这条。
#
# 这条用例覆盖的是**本任务的结构保证**:`interrupt()` 不在 `agent` 里(否则
# resume 会把 `agent` 从头重跑,模型被再问一遍、工具调用序列可能变),
# 只有 `confirm_write` 挂起,副作用在它下游的 `apply_write_decision` 里跑一次。


def _ticket_spec(calls: list):
    """`create_ticket` 的登记项,`tool` 槽位套了计次壳。

    名字必须叫 `create_ticket`:`kind` 由 `policy.kind_of(名字)` 给(写操作),
    而**写操作**正是这条链上唯一会触发确认闸的种类。
    """
    import dataclasses

    from langchain.tools import tool

    from app.tools.registry import _spec_from_tool

    @tool
    async def create_ticket(description: str, ticket_type: str = "售后") -> str:
        """为用户创建一张售后工单。"""
        return json.dumps({"ticket_no": "T-1"}, ensure_ascii=False)

    return dataclasses.replace(
        _spec_from_tool(create_ticket, source="builtin"),
        tool=_CountingTool(create_ticket, calls),
    )


class _CountingTool:
    def __init__(self, inner, calls):
        self._inner = inner
        self._calls = calls

    async def ainvoke(self, tool_call):
        self._calls.append(tool_call["args"])
        return await self._inner.ainvoke(tool_call)


class _TurnModel:
    """消解走 `ainvoke`(回显),Agent 的每一轮走 `astream`(按脚本)。"""

    def __init__(self, rounds):
        self._rounds = list(rounds)
        self.streamed = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content=messages[-1].content)

    async def astream(self, msgs):
        self.streamed += 1
        yield self._rounds.pop(0)


class _BusinessIntent:
    """`订单` → BUSINESS ⇒ **直接进 Agent**(不预检索、不过闸)。

    ⚠️ 不能用 `售后`:那张表把「退款退货 / 售后」送进 REFUND 子流程,
    而那个入口自己会 `interrupt()` 弹订单卡片 —— 挂起点就换成了
    `refund_pick_order`,本用例测的「写调用确认」根本走不到。
    """

    def with_structured_output(self, schema, method=None):
        return self

    async def ainvoke(self, messages):
        class _R:
            intent = "订单"
            confidence = 0.9

        return _R()


class _Session:
    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        pass


class _Retriever:
    async def search(self, query):        # `订单` → BUSINESS,不预检索 —— 调到就是走错了路
        raise AssertionError("业务数据类不该走检索")


def _live_graph(*, calls, rounds, session):
    from langgraph.checkpoint.memory import InMemorySaver

    from app.agent.graph import build_graph
    from app.config import Settings

    settings = Settings(
        _env_file=None,
        openai_base_url="https://example.invalid/v1", openai_api_key="sk-test",
        openai_model="m", database_url="mysql+asyncmy://u:p@h:3306/db",
    )
    spec = _ticket_spec(calls)
    graph = build_graph(
        model=_TurnModel(rounds),
        intent_model=_BusinessIntent(),
        tools=[spec.tool],
        registry={"create_ticket": spec},
        settings=settings,
        retriever=_Retriever(),
        session=session,
        conversation_id="conv-write",
        emit=lambda p: None,
        checkpointer=InMemorySaver(),
    )
    return graph


async def _drive(graph, payload, *, thread):
    """跑一次;返回**本次**出现的 interrupt 载荷。"""
    from langgraph.types import Command

    seen = []
    async for mode, chunk in graph.astream(
        payload,
        config={"configurable": {"thread_id": thread}},
        stream_mode=["custom", "updates"],
    ):
        if mode == "updates" and "__interrupt__" in chunk:
            seen.append(chunk["__interrupt__"][0].value)
    return seen


@pytest.mark.anyio
async def test_write_is_executed_exactly_once_across_suspend_and_resume():
    """挂起那一刻**不执行**;决议之后**恰好执行一次**(不是两次)。

    `resume` 会把节点从头重跑(ch06 实测)—— 副作用因此**不能**放在
    `confirm_write` 里,只能放在它下游的 `apply_write_decision`。
    计数在 `ainvoke` 边界:少回灌/多执行一次都在这里看得见。
    """
    from langchain_core.messages import AIMessageChunk, ToolMessage
    from langgraph.types import Command

    calls: list = []
    round1 = AIMessageChunk(content="", tool_calls=[{
        "name": "create_ticket", "args": {"description": "耳机坏了"},
        "id": "call_1", "type": "tool_call",
    }])
    round2 = AIMessageChunk(content="已为您建单:T-1")
    session = _Session()
    graph = _live_graph(calls=calls, rounds=[round1, round2], session=session)

    interrupts = await _drive(
        graph,
        {"conversation_id": "conv-write", "user_input": "帮我建个工单",
         "history": [], "trace": []},
        thread="t-write",
    )

    # 1) 挂起了,载荷是预览卡片(帧名由 `confirm_write` 自己说)
    assert len(interrupts) == 1
    assert interrupts[0]["frame"] == "ticket_confirm"
    assert interrupts[0]["preview"] == {"description": "耳机坏了"}
    # 2) **挂起那一刻写调用一次都没发生**
    assert calls == [], "用户还没确认,写调用就执行了"
    # 2b) **挂起那一轮一行都没落库**(spec §5.1)。
    #     这条同时是 `_OUTLETS` 去掉 `"agent"` 的端到端守卫:把 `agent` 也
    #     无条件连上 `log_turn` 的话,挂起路径会**在 `confirm_write` 之前**
    #     先把这半轮写进库(一半写库、一半没写)。
    assert session.added == [], "挂起的那一轮不该落库"

    # 3) 决议 → 续跑
    await _drive(graph, Command(resume={"approved": True}), thread="t-write")
    assert len(calls) == 1, "`create_ticket` 在挂起/续跑之间**执行次数不为 1**"
    # 3b) resume 走完才落库,且 ReAct 往返完整落进历史:
    #     user + assistant(带 tool_calls) + tool(配对结果)+ assistant(收尾)。
    assert [r.role for r in session.added] == ["user", "assistant", "tool", "assistant"]

    snapshot = await graph.aget_state({"configurable": {"thread_id": "t-write"}})
    state = snapshot.values
    # 4) 两条 trace 标记都在(spec §9.7 的验收口径)
    assert "agent:write_pending tool=create_ticket" in state["trace"]
    assert "agent:write_resumed" in state["trace"]
    # 5) 续跑那一轮的回复是这个**回合**的收尾
    assert "已为您建单" in state["reply"]
    # 6) 上游契约:那条带 `tool_calls` 的 assistant 消息**必须**有配对的
    #    tool 结果(`log_turn` 落的就是这批),否则上游 400。
    turn = state["turn_messages"]
    call_ids = [c["id"] for m in turn if getattr(m, "tool_calls", None)
                for c in m.tool_calls]
    result_ids = [m.tool_call_id for m in turn if isinstance(m, ToolMessage)]
    assert call_ids == ["call_1"]
    assert result_ids == ["call_1"]
