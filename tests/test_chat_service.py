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
    """预算不足仍然抛 ContextOverflowError —— 端点靠它在流开始前返回 400。

    ch07 换了判据的**来源**:不再是「system prompt + 本轮输入超过
    `context_budget_tokens`」,而是「从窗口倒推出来的历史预算为负」。
    所以这条用例不再靠一个超长输入去顶穿预算 —— 新口径下输入长度**不参与**
    这个判据(见 `prepare_turn` 的 docstring);改成把窗口压到 `ge=1024` 的下界,
    默认的固定开销与单轮峰值加起来远超它,`history_budget` 必然为负。
    输入刻意用短句:长输入会把这条用例的触发点换成「输入超
    `max_user_input_tokens`」那条(T10 在端点接线,spec §8)。
    """
    with pytest.raises(ContextOverflowError):
        prepare_turn(
            settings=_settings(model_context_window=1024), history=[], user_input="在吗"
        )


def test_prepare_turn_rejects_input_over_the_per_message_cap():
    """**本轮输入本身**超 `max_user_input_tokens` ⇒ 抛(端点据此在流前 400)。

    这条在 T2 到 T10 之间是**空档**:旧口径把 `count_tokens(user_input)` 算进
    「已用」,新口径把它归进单轮峰值的 `max_user_input_tokens`,于是实际输入
    长度**再也没有人比过** —— 50k token 的一句话会一路送到上游。

    判据取「同一段文本在松上限下**不抛**」作对照:少了它,一条「恒定抛异常」
    的实现也能让上面那半条通过。
    """
    text = "这是一句明显超过五个 token 的话"
    with pytest.raises(ContextOverflowError):
        prepare_turn(
            settings=_settings(max_user_input_tokens=5), history=[], user_input=text
        )
    # 对照:同一个输入在足够大的上限下照常返回。
    assert prepare_turn(
        settings=_settings(max_user_input_tokens=2000), history=[], user_input=text
    ) == []


def test_prepare_turn_rejects_a_budget_that_cannot_fit_one_round():
    """**spec §8 的判据是「装不下一轮」,不是「预算为负」**。

    两者在「预算为正、但小于 `per_round_steady`」时结论相反:比如
    `history_budget = 4537` 而 `per_round_steady = 10000`,按 spec 该 400,
    按 `history_budget < 0` 却**静默带着一小段历史往下走** —— 用户拿到一个
    上下文被悄悄截到几乎没有的回答,而没有任何东西报错。
    这条用例就是那个中间区间(实测:`keep_rounds=20 × 10000` 那一支打不过窗口,
    `history_budget` 落在 4537 > 0 上)。
    """
    with pytest.raises(ContextOverflowError):
        prepare_turn(
            settings=_settings(per_round_steady=10000),
            history=[Message(role="user", content="在吗")],
            user_input="在吗",
        )


def test_prepare_turn_applies_the_budget_to_the_history():
    """放不下的历史必须被**真的**裁掉,而不是原样返回。

    与上面两条互补:第一条的历史**放得下**(裁不裁都是那两条),第二条根本不
    返回历史。只有这一条问「`select_history` 到底有没有被调用、算出来的
    历史预算有没有真的用上」—— 把 `return trim.select_history(history,
    history_budget)` 改成 `return list(history)`,只有它变红。

    ch07 的「预算极小」怎么造:旧口径是 `context_budget_tokens=1000` 直接给小
    预算,新口径下历史预算取 `min(keep_rounds × per_round_steady, 窗口匀得出来
    的)`。这里把**按轮数估的那一支**压到 1(`keep_rounds=1` × `per_round_steady=1`
    = 1 token),窗口那一支按默认值算远大于 1,于是 `history_budget == 1`。
    刻意不写「窗口 = 某个刚好勉强够的数」:那要按 system prompt 当前的
    token 数倒推(实测 701),提示词一改这条用例就红在一个与它无关的原因上。
    """
    history = [
        Message(role="user", content="退" * 2000),
        Message(role="assistant", content="好" * 2000),
    ]
    kept = prepare_turn(
        settings=_settings(keep_rounds=1, per_round_steady=1),
        history=history,
        user_input="在吗",
    )

    assert kept == []
