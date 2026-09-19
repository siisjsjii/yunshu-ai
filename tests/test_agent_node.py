"""主力 Agent 的 ReAct 循环:收敛、工具回灌、停止条件、token 预算、流式。"""

import pytest
from langchain_core.messages import ToolMessage

from app.agent.nodes import make_agent_node
from app.config import Settings
from app.schemas import Message
from app.tools.errors import ToolInfrastructureError

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **REQUIRED, **overrides)


def _text(msg) -> str:
    """1.x 里**流式 chunk** 的 content 是 block 列表;这里的 BaseMessage 是
    我们自己用字符串构造的,content 仍是 str —— 两种都兜住,免得断言靠猜。"""
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


class FakeChunk:
    def __init__(self, text="", tool_calls=None, usage=None):
        self.text = text
        # 必须带 "type": "tool_call" —— 见 CLAUDE.md;缺键时 BaseTool.ainvoke
        # 会把整个 dict 当**参数**去校验 schema,每次调用都变成「参数不合法」。
        self.tool_calls = [{"type": "tool_call", **tc} for tc in (tool_calls or [])]
        self.usage_metadata = usage

    def __add__(self, other):
        # 累积 chunk 时**原样**拼接 tool_calls,**必须保留 `"type": "tool_call"`**。
        # 它是 `BaseTool.ainvoke` 判定「这是一个工具调用」的**唯一**依据
        # (CLAUDE.md 的硬约束);`execute_tool` 则原样把 dict 透传给
        # `tool.ainvoke`(`app/tools/executor.py:65`)。丢了它,真实 `@tool` 会把
        # 整个 dict 当**参数**去校验 schema,每次调用都退化成「参数不合法」的
        # 可恢复失败 —— 而本文件的替身工具不查这个键,所以**测试照样全绿**。
        #
        # 说明:本替身**不模拟** LangChain 真实的按 index 合并 + 分片 args 拼接
        # (真实模型会把一次工具调用拆成多个 chunk 流式吐出来),因为那些测试
        # 里每次工具调用都写在**单个** chunk 里,`__add__` 的合并分支用不到。
        # 真实多 chunk 的累积由 T11 的验收 5(真模型、要求 ReAct 不止一步)兜底。
        return FakeChunk(
            text=self.text + other.text,
            tool_calls=self.tool_calls + other.tool_calls,
            usage=other.usage_metadata or self.usage_metadata,
        )


class _BoundModel:
    def __init__(self, owner):
        self._owner = owner

    async def astream(self, messages):
        self._owner.bound_rounds += 1
        self._owner.bound_messages = list(messages)   # 断言入参用
        for chunk in self._owner.rounds.pop(0):
            yield chunk


class ScriptedModel:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.bound_rounds = 0
        self.unbound_rounds = 0

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return _BoundModel(self)

    async def astream(self, messages):
        # 这里**刻意不记**入参:agent 每一轮都绑着工具问,未绑工具的入口只在
        # 步数用尽收尾时走一次 —— 把「回灌的 ToolMessage」记在这儿会永远看不到。
        # 要断回灌就读 `bound_messages`(Task 1 已踩过一次,见 ledger Ruling 6)。
        self.unbound_rounds += 1
        for chunk in self.rounds.pop(0):
            yield chunk


class FakeTool:
    name = "query_order"

    def __init__(self, content='{"status": "已发货"}', error=None):
        self.content = content
        self.error = error
        self.calls = []

    async def ainvoke(self, tool_call):
        self.calls.append(tool_call)
        if self.error is not None:
            raise self.error
        return type("_R", (), {"content": self.content})()


def _node(model, tools=(), registry=None, settings=None, frames=None):
    return make_agent_node(
        model=model, tools=list(tools), registry=registry or {},
        settings=settings or _settings(),
        emit=(frames.append if frames is not None else (lambda p: None)),
    )


def _state(**over):
    base = {"conversation_id": "c1", "user_input": "订单 1001 发货了吗",
            "resolved_input": "订单 1001 发货了吗", "history": []}
    base.update(over)
    return base


@pytest.mark.anyio
async def test_direct_text_answer_converges_without_tools():
    frames = []
    model = ScriptedModel([[FakeChunk("你的"), FakeChunk("订单已发货。")]])
    out = await _node(model, frames=frames).__call__(_state())
    assert out["reply"] == "你的订单已发货。"
    assert out["agent_steps"] == 1
    assert out["tool_calls_made"] == []
    assert frames == [
        {"frame": "token", "text": "你的"},
        {"frame": "token", "text": "订单已发货。"},
    ]


@pytest.mark.anyio
async def test_tool_call_round_emits_frames_and_feeds_result_back():
    frames = []
    tool_obj = FakeTool()
    model = ScriptedModel([
        [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}])],
        [FakeChunk("已发货。")],
    ])
    out = await _node(model, [tool_obj], {"query_order": tool_obj}, frames=frames).__call__(_state())

    assert out["reply"] == "已发货。"
    assert out["agent_steps"] == 2
    assert out["tool_calls_made"] == [{"name": "query_order", "ok": True}]
    assert {"frame": "tool_call", "name": "query_order",
            "args": {"order_id": "1001"}, "tool_call_id": "c1"} in frames
    assert {"frame": "tool_result", "tool_call_id": "c1", "ok": True,
            "summary": '{"status": "已发货"}'} in frames
    # 回灌的 ToolMessage 必须与 tool_call_id 配对。
    # 注意记的是**绑工具**那个入口:agent 每一轮都绑着工具问,未绑工具的入口
    # 只在步数用尽收尾时走一次。
    fed = [m for m in model.bound_messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in fed] == ["c1"]
    # 钉住 `msgs.append(acc)` 那一行:删掉它,上面 fed 的断言**依然全绿**,
    # 而真实链路上会退化成「有 tool 消息、没有前置的 assistant(tool_calls)消息」——
    # OpenAI 兼容 API 直接 400。这正是 CLAUDE.md 点名的「假绿」形态。
    calls_idx = [i for i, m in enumerate(model.bound_messages)
                 if getattr(m, "tool_calls", None)]
    tool_idx = [i for i, m in enumerate(model.bound_messages)
                if isinstance(m, ToolMessage)]
    assert calls_idx and tool_idx, "第二轮入参缺 assistant(tool_calls) 或 ToolMessage"
    assert tool_idx[0] == calls_idx[0] + 1   # 必须**紧邻**
    # trace 让验收 5「ReAct 不止一步」机械可断言
    assert out["trace"] == ["agent:step1 tool=query_order", "agent:converged"]


@pytest.mark.anyio
async def test_step_limit_converges_without_tools_bound():
    """步数用尽 → 最后一次调用**不绑 tools**,结构上保证收敛。"""
    # 脚本按消费顺序对齐:绑工具的 2 轮(== max_agent_steps)各弹一条,
    # 第 3 条留给收尾那一轮(未绑工具)。
    looping = [
        [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": str(i)}, "id": f"c{i}"}])]
        for i in range(2)
    ]
    model = ScriptedModel(looping + [[FakeChunk("收敛了")]])
    tool_obj = FakeTool()
    out = await _node(model, [tool_obj], {"query_order": tool_obj},
                      settings=_settings(max_agent_steps=2)).__call__(_state())
    assert out["agent_steps"] == 2
    assert model.bound_rounds == 2
    assert model.unbound_rounds == 1
    assert out["reply"] == "收敛了"


@pytest.mark.anyio
async def test_token_budget_stops_further_tool_rounds():
    """预算超限后不再进入下一轮工具调用。"""
    big = {"total_tokens": 99999}
    model = ScriptedModel([
        [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}], usage=big)],
        [FakeChunk("收敛了")],
    ])
    tool_obj = FakeTool()
    out = await _node(model, [tool_obj], {"query_order": tool_obj},
                      settings=_settings(agent_token_budget=100)).__call__(_state())
    assert out["usage"]["total_tokens"] >= 99999
    assert model.bound_rounds == 1      # 只走了一轮绑工具的
    assert out["reply"] == "收敛了"


@pytest.mark.anyio
async def test_unknown_tool_error_propagates_as_infrastructure_error():
    """未分类的异常由 executor 判成基础设施故障,**必须向上抛**(→502),
    绝不能被回灌给模型、伪装成「你的订单号查不到」。"""
    tool_obj = FakeTool(error=RuntimeError("boom"))
    model = ScriptedModel([
        [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}])],
    ])
    with pytest.raises(ToolInfrastructureError):
        await _node(model, [tool_obj], {"query_order": tool_obj}).__call__(_state())


@pytest.mark.anyio
async def test_history_and_evidence_reach_the_first_model_call():
    """历史经 `prompts.to_lc_messages` 转、证据并进本轮 human 消息。

    **断言的是发给模型的真实入参**,不是「函数没抛异常」——后者恒真。
    """
    model = ScriptedModel([[FakeChunk("好")]])
    await _node(model).__call__(
        _state(
            history=[Message(role="user", content="在吗")],
            evidence=[{"section_path": "退货政策", "answer": "七天无理由", "category": "退换货"}],
        )
    )
    sent = model.bound_messages
    assert sent[0].type == "system"            # 品牌 system prompt
    assert _text(sent[1]) == "在吗"             # 历史转过来了(只经 to_lc_messages)
    assert "七天无理由" in _text(sent[-1])      # 证据块
    assert "订单 1001 发货了吗" in _text(sent[-1])   # 用户原话
    assert sent[-1].type == "human"


@pytest.mark.anyio
async def test_no_evidence_means_no_evidence_block_in_the_prompt():
    """业务数据类没有证据块 —— 不能留一个空标题在那儿诱导模型。"""
    model = ScriptedModel([[FakeChunk("好")]])
    await _node(model).__call__(_state(evidence=[]))
    assert "以下是知识库中" not in _text(model.bound_messages[-1])
    assert _text(model.bound_messages[-1]) == "订单 1001 发货了吗"
