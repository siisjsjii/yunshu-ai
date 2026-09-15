from app.memory.trim import (
    ContextOverflowError,
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
    for msg in kept:
        assert msg.role in ("user", "assistant")


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
