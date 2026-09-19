"""祛魅热身:手写最裸 Agent 循环的行为。全部用替身,不联网、不碰 DB。"""

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from app.agent.loop import run_agent_loop
from app.config import Settings

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    # _env_file=None:不传的话 pydantic-settings 会读仓库根的 .env。
    return Settings(_env_file=None, **REQUIRED, **overrides)


class FakeAIMessage:
    """替身 AI 消息:text + tool_calls(形状与真 AIMessage 一致)。"""

    def __init__(self, text="", tool_calls=None):
        self.text = text
        # 必须带 "type": "tool_call" —— 见本文件顶部说明与 CLAUDE.md。
        self.tool_calls = [{"type": "tool_call", **tc} for tc in (tool_calls or [])]


class _BoundModel:
    def __init__(self, owner):
        self._owner = owner

    async def ainvoke(self, messages):
        self._owner.bound_rounds += 1
        # 记录入参:循环里**每一轮都是绑着工具**问的,所以「回灌进去的
        # ToolMessage」只能在绑工具的入口上看到(未绑工具的入口只在步数
        # 用尽收尾时走一次)。计划 Task 6 的同一个替身也是这么记的。
        self._owner.bound_messages = list(messages)
        return self._owner.rounds.pop(0)


class ScriptedModel:
    """按脚本回放 AI 消息,并区分调用走的是「绑了工具」还是「未绑工具」的入口。"""

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.bound_rounds = 0
        self.unbound_rounds = 0

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return _BoundModel(self)

    async def ainvoke(self, messages):
        self.unbound_rounds += 1
        self.last_unbound_messages = list(messages)
        return self.rounds.pop(0)


class FakeTool:
    """替身工具:在 ainvoke 边界计数(写在函数体里的话区分不出任何实现)。"""

    name = "query_order"

    def __init__(self, content='{"order_id": "1001", "status": "已发货"}'):
        self.content = content
        self.calls = []

    async def ainvoke(self, tool_call):
        self.calls.append(tool_call)
        return type("_R", (), {"content": self.content})()


@pytest.mark.anyio
async def test_text_only_reply_converges_in_one_step():
    """模型不调工具:一步收敛,且**从未**走未绑工具的入口。"""
    model = ScriptedModel([FakeAIMessage(text="你好呀")])
    reply, steps = await run_agent_loop(
        model=model, messages=[HumanMessage("你好")], tools=[],
        registry={}, settings=_settings(),
    )
    assert reply == "你好呀"
    assert steps == 1
    assert model.bound_rounds == 1
    assert model.unbound_rounds == 0


@pytest.mark.anyio
async def test_tool_call_is_executed_and_result_is_fed_back():
    """模型调工具:执行 → 回灌 ToolMessage → 再问一轮 → 收敛。"""
    model = ScriptedModel([
        FakeAIMessage(tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}]),
        FakeAIMessage(text="你的订单已发货。"),
    ])
    tool_obj = FakeTool()
    reply, steps = await run_agent_loop(
        model=model, messages=[HumanMessage("订单 1001 发货了吗")],
        tools=[tool_obj], registry={"query_order": tool_obj}, settings=_settings(),
    )
    assert reply == "你的订单已发货。"
    assert steps == 2
    assert len(tool_obj.calls) == 1
    assert tool_obj.calls[0]["args"] == {"order_id": "1001"}
    # 回灌:第二轮的消息里必须有一条 tool_call_id 对得上的 ToolMessage
    tool_msgs = [m for m in model.bound_messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_msgs] == ["c1"]
    assert "已发货" in tool_msgs[0].content


@pytest.mark.anyio
async def test_step_limit_forces_convergence_without_tools():
    """步数用尽:最后一轮**不绑 tools**,模型在结构上无法再调。"""
    # 脚本必须与消费顺序对齐:循环里绑工具问 2 轮(== max_steps),各弹走一条;
    # 第 3 条留给收尾那一轮(未绑工具)弹。多写的条目只会被弹到前几条。
    looping = [
        FakeAIMessage(tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": f"c{i}"}])
        for i in range(2)
    ]
    model = ScriptedModel(looping + [FakeAIMessage(text="收敛了")])
    tool_obj = FakeTool()
    reply, steps = await run_agent_loop(
        model=model, messages=[HumanMessage("查")], tools=[tool_obj],
        registry={"query_order": tool_obj}, settings=_settings(), max_steps=2,
    )
    assert steps == 2
    assert model.bound_rounds == 2          # 循环里只绑着工具问了 2 轮
    assert model.unbound_rounds == 1        # 收尾那一轮未绑工具
    assert reply == "收敛了"
