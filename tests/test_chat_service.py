"""对话编排测试。全部用替身,不联网、不碰 DB。

ch05 Task 8:`stream_turn` 连同它的 18 条用例一起删掉了 —— 单轮编排改由图
(`app/agent/nodes.py` 的 agent 节点 + `app/agent/graph.py`)负责,那些守卫
逐条搬进了 `tests/test_agent_node.py` / `tests/test_agent_graph.py` 与
`tests/test_api_chat.py`(逐条搬迁表见 task-8-report.md)。

本文件现在只剩 `prepare_turn`,而且它的**产出语义变了**:从「组装好的消息」
变成「裁剪后的历史」—— 消息组装搬进了 agent 节点(它要往里插证据块)。
"""

import pytest

from app.config import Settings
from app.memory.trim import ContextOverflowError
from app.schemas import Message
from app.services.chat import prepare_turn

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **REQUIRED, **overrides)


@pytest.mark.anyio
async def test_prepare_turn_returns_trimmed_history():
    """预算校验仍在流开始前完成;产出是**裁剪后的历史**,消息组装交给节点。"""
    history = [Message(role="user", content="在吗"),
               Message(role="assistant", content="在的")]
    kept = prepare_turn(settings=_settings(), history=history, user_input="你好")
    assert [m.content for m in kept] == ["在吗", "在的"]


def test_prepare_turn_raises_when_budget_is_exhausted():
    """预算不足仍然抛 ContextOverflowError —— 端点靠它在流开始前返回 400。"""
    huge = "字" * 100_000
    with pytest.raises(ContextOverflowError):
        prepare_turn(settings=_settings(), history=[], user_input=huge)


def test_prepare_turn_applies_the_budget_to_the_history():
    """放不下的历史必须被**真的**裁掉,而不是原样返回。

    与上面两条互补:第一条的历史**放得下**(裁不裁都是那两条),第二条根本不
    返回历史。只有这一条问「`select_history` 到底有没有被调用、算出来的
    available 有没有真的用上」—— 把 `return trim.select_history(history,
    available)` 改成 `return list(history)`,只有它变红。
    """
    history = [
        Message(role="user", content="退" * 2000),
        Message(role="assistant", content="好" * 2000),
    ]
    kept = prepare_turn(
        settings=_settings(
            context_budget_tokens=1000,
            reserved_output_tokens=0,
            safety_margin_tokens=0,
        ),
        history=history,
        user_input="在吗",
    )

    assert kept == []
