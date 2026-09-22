"""ch07 预算推导。纯函数,不联网、不碰 db。"""

import pytest

from app.config import Settings
from app.memory import budget

#: 构造 Settings 必须传 `_env_file=None` —— 仓库根有真实 .env,
#: pydantic-settings 会自动读它,不传的话「缺字段应报错」那类断言会静默通过。
REQUIRED = dict(
    _env_file=None,
    openai_base_url="https://example.invalid/v1",
    openai_api_key="sk-test",
    openai_model="test-model",
    database_url="mysql+asyncmy://u:p@127.0.0.1:3306/x",
)

SYSTEM_PROMPT = "你是客服。" * 50  # 一段可计数的中文


def _settings(**over):
    return Settings(**{**REQUIRED, **over})


def test_demo_config_arithmetic_is_exact():
    """演示配置下把每个扣减项算清楚 —— 本章所有数值行为的地基。

    这条**不看某个魔数,看等式**:固定开销 + 单轮峰值 + 历史预算 == 窗口。
    等式成立与否,才区分得出「扣全了」和「漏扣了一项」。
    """
    s = _settings(
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        rerank_top_k=5,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
        keep_rounds=20,
        per_round_steady=600,
    )
    b = budget.derive(settings=s, system_prompt=SYSTEM_PROMPT)

    assert b.window == 18000
    assert b.peak == 3 * 1200 + 2000          # 单轮 ReAct 峰值 + 用户输入上限
    assert b.fixed_overhead + b.peak + b.history_budget == b.window
    assert b.layer1_budget + b.layer2_budget == b.history_budget
    # 窗口那一支赢:「想留住的轮数」算出来 20×600=12000 更大
    assert b.history_budget == b.window - b.fixed_overhead - b.peak


def test_rounds_branch_wins_when_window_is_generous():
    """两个数取小的那个 —— 两条分支都真的会赢,否则 min() 是装饰。"""
    s = _settings(model_context_window=200_000, keep_rounds=5, per_round_steady=100)
    b = budget.derive(settings=s, system_prompt=SYSTEM_PROMPT)
    assert b.history_budget == 500            # 5 × 100,而不是窗口那一支


def test_layer_split_uses_the_declared_share():
    s = _settings(model_context_window=18000)
    b = budget.derive(settings=s, system_prompt=SYSTEM_PROMPT)
    assert b.layer1_budget == int(b.history_budget * budget.LAYER1_SHARE)
    assert b.layer2_budget == b.history_budget - b.layer1_budget


def test_fits_one_round_is_false_when_peak_eats_the_window():
    """「连一轮都装不下」必须报出来 —— 自检的判据就是它。"""
    s = _settings(
        model_context_window=1024,
        max_output_tokens=800,
        tool_def_tokens=800,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
    )
    b = budget.derive(settings=s, system_prompt=SYSTEM_PROMPT)
    assert b.fits_one_round is False


def test_tokens_for_chars_uses_the_same_counter_as_everything_else():
    """中文按字数折 token 的口径必须与预算同源。

    另立一个「1 字 = 1.5 token」的系数,就会与 `trim.count_tokens`
    各说各话 —— 而两处都「看起来合理」。这条钉住它们同源。
    """
    from app.memory import trim

    assert budget.tokens_for_chars(200) == trim.count_tokens("中" * 200)
    assert budget.tokens_for_chars(0) == 0
