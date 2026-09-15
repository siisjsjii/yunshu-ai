from app.memory.trim import (
    ContextOverflowError,
    _to_rounds,
    compute_available_tokens,
    count_tokens,
    select_history,
)
from app.schemas import Message


def _u(text: str) -> Message:
    return Message(role="user", content=text)


def _a(text: str) -> Message:
    return Message(role="assistant", content=text)


def test_count_tokens_is_positive_for_nonempty_text():
    assert count_tokens("你好") > 0
    assert count_tokens("") == 0


def test_count_tokens_grows_with_length():
    assert count_tokens("退货退款流程是什么" * 5) > count_tokens("退货退款流程是什么")


def test_available_tokens_subtracts_all_three_terms():
    available = compute_available_tokens(
        system_prompt="x" * 10,
        user_input="y" * 10,
        context_budget_tokens=1000,
        reserved_output_tokens=100,
        safety_margin_tokens=50,
    )
    expected = 1000 - 100 - 50 - count_tokens("x" * 10) - count_tokens("y" * 10)
    assert available == expected


def test_available_tokens_can_go_negative():
    """单轮输入超预算时返回负数,由调用方决定抛错。"""
    available = compute_available_tokens(
        system_prompt="",
        user_input="啊" * 5000,
        context_budget_tokens=1000,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
    )
    assert available < 0


def test_select_history_returns_empty_when_budget_is_zero():
    history = [_u("你好"), _a("您好")]
    assert select_history(history, available_tokens=0) == []


def test_select_history_keeps_whole_rounds():
    history = [_u("第一轮问题"), _a("第一轮回答"), _u("第二轮问题"), _a("第二轮回答")]
    one_round = count_tokens("第二轮问题") + count_tokens("第二轮回答")

    kept = select_history(history, available_tokens=one_round)

    assert kept == [_u("第二轮问题"), _a("第二轮回答")]


def test_select_history_drops_oldest_rounds_first():
    history = [_u("老问题"), _a("老回答"), _u("新问题"), _a("新回答")]
    budget = (
        count_tokens("老问题")
        + count_tokens("老回答")
        + count_tokens("新问题")
        + count_tokens("新回答")
    )

    assert select_history(history, available_tokens=budget) == history
    assert select_history(history, available_tokens=budget - 1) == [
        _u("新问题"),
        _a("新回答"),
    ]


def test_select_history_preserves_chronological_order():
    history = [_u("一"), _a("一答"), _u("二"), _a("二答"), _u("三"), _a("三答")]
    kept = select_history(history, available_tokens=10_000)
    assert kept == history


def test_select_history_never_returns_half_a_round():
    """按整轮裁剪 —— 不允许出现有问无答的孤立 user 消息。"""
    history = [_u("很长很长的问题" * 10), _a("很长很长的回答" * 10), _u("新问题"), _a("新回答")]
    kept = select_history(history, available_tokens=1)

    assert kept == []


def test_context_overflow_error_carries_numbers():
    err = ContextOverflowError(used=500, budget=100)
    assert err.used == 500
    assert err.budget == 100
    assert "500" in str(err)
    assert "100" in str(err)


def test_select_history_does_not_skip_oversized_newest_round():
    """最新一轮装不下时直接停止,不回退去保留更老的小轮次。"""
    small = [_u("小问题"), _a("小回答")]
    big = [_u("超长" * 100), _a("超长" * 100)]
    history = small + big

    budget = count_tokens("小问题") + count_tokens("小回答")

    assert select_history(history, available_tokens=budget) == []


def test_select_history_treats_trailing_user_as_own_round():
    """末尾孤立的 user 单独成轮,作为整体被保留或丢弃,不拆开也不丢失。"""
    history = [_u("完整问题"), _a("完整回答"), _u("被打断的问题")]
    lone_round_tokens = count_tokens("被打断的问题")

    kept = select_history(history, available_tokens=lone_round_tokens)

    assert kept == [_u("被打断的问题")]


def _pairing_intact(messages) -> bool:
    """每条 tool 消息前面都必须有带对应 tool_call_id 的 assistant 消息。"""
    pending: set[str] = set()
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            pending |= {tc["id"] for tc in m.tool_calls}
        elif m.role == "tool":
            if m.tool_call_id not in pending:
                return False
    return True


def test_select_history_never_separates_tool_from_its_assistant():
    """裁剪不能把 tool 消息与它的 assistant 父亲切开。

    OpenAI 兼容 API 要求 tool 消息前面必须紧跟着带对应 tool_call_id 的
    assistant 消息,切开就会 400 —— 而且只在历史长到触发裁剪时偶发。

    注意这条断言**区分不出**旧规则:旧规则把一轮工具往返切成
    [user, assistant(tool_calls)] 与 [tool, assistant] 两半,而本用例的预算
    恰好同时够得着这两半(丢弃线落在两者的共同边界上),裁剪结果与 user
    边界规则一致。真正钉住旧规则的,是下面那条
    test_select_history_drops_a_whole_tool_round_instead_of_its_tail ——
    它的预算够得着后半段、够不着前半段,旧规则会留下以 tool 打头的序列。
    """
    old_call = {"id": "call_old", "name": "query_order", "args": {"order_id": "9001"}}
    new_call = {"id": "call_new", "name": "query_logistics", "args": {"order_id": "1001"}}
    history = [
        Message(role="user", content="很早的问题" * 60),
        Message(role="assistant", content="", tool_calls=[old_call]),
        Message(role="tool", content="很老的工具结果" * 60, tool_call_id="call_old"),
        Message(role="assistant", content="很早的回答" * 60),
        Message(role="user", content="订单 1001 的物流到哪了"),
        Message(role="assistant", content="", tool_calls=[new_call]),
        Message(role="tool", content="已揽件", tool_call_id="call_new"),
        Message(role="assistant", content="您的包裹已揽件。"),
    ]
    kept = select_history(history, available_tokens=40)

    assert _pairing_intact(kept)
    assert [m.role for m in kept] == ["user", "assistant", "tool", "assistant"]


def test_select_history_drops_a_whole_tool_round_instead_of_its_tail():
    """裁剪线落在 tool 与它的 assistant 父亲之间时,必须整轮丢掉。

    这是上一条测试没能覆盖到的情形:只有当预算"够得着后半段、够不着前半段"
    时,旧规则(遇 assistant 收轮)才会真的留下以 tool 打头的残缺序列。

    触发条件是"用户消息贵、工具往返便宜" —— 用户提问越大段,越容易命中;
    这也正是它只在长对话里偶发的原因。
    """
    call = {"id": "call_1", "name": "query_logistics", "args": {"order_id": "1001"}}
    history = [
        Message(role="user", content="很早的问题" * 60),
        Message(role="assistant", content="很早的回答" * 60),
        # 这一轮的用户消息很贵,工具往返很便宜
        Message(role="user", content="订单 1001 的物流到哪了" * 30),
        Message(role="assistant", content="", tool_calls=[call]),
        Message(role="tool", content="已揽件", tool_call_id="call_1"),
        Message(role="assistant", content="您的包裹已揽件。"),
        # 最新一轮很便宜,裁剪后应当只剩它
        Message(role="user", content="那什么时候到"),
        Message(role="assistant", content="预计明天送达。"),
    ]

    kept = select_history(history, available_tokens=40)

    assert _pairing_intact(kept)
    assert [m.role for m in kept] == ["user", "assistant"]


def test_round_definition_is_user_delimited():
    """一轮 = 从一条 user 开始,到(不含)下一条 user 为止。"""
    history = [
        Message(role="user", content="q1"),
        Message(role="assistant", content="", tool_calls=[{"id": "c1", "name": "t", "args": {}}]),
        Message(role="tool", content="r1", tool_call_id="c1"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="q2"),
        Message(role="assistant", content="a2"),
    ]
    rounds = _to_rounds(history)
    assert len(rounds) == 2
    assert [m.role for m in rounds[0]] == ["user", "assistant", "tool", "assistant"]
    assert [m.role for m in rounds[1]] == ["user", "assistant"]
