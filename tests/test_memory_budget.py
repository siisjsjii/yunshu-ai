"""ch07 预算推导。纯函数,不联网、不碰 db。"""

import math

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


#: 逐项敏感性的基准配置。两个数是**故意**不取默认值的:
#: `rerank_top_k=4`(默认 5)⇒ 证据那项的期望增量是 4×100=400,
#: 排除了「乘数写死成 5 也蒙对」;`max_agent_steps=3`(默认 5)同理,
#: 峰值的期望增量是 3×100。基准值也都写在断言里,不用 settings 的默认值反查。
BASE = dict(
    model_context_window=18000,
    rerank_top_k=4,
    max_agent_steps=3,
    tool_result_max_tokens=1200,
    max_user_input_tokens=2000,
    max_output_tokens=2000,
    safety_margin_tokens=512,
    tool_def_tokens=800,
    evidence_block_tokens=250,
    summary_max_chars=200,
)


def test_every_deduction_term_moves_fixed_overhead_by_its_exact_weight():
    """逐个扣减项各加 100,`fixed_overhead` / `peak` 必须**正好**涨对应的量。

    为什么必须有这条:`fixed + peak + history == window` 是**公式的恒等式**
    —— 少扣任何一项时 `by_window` 变大、`history_budget` 跟着变大,等式照样成立。
    删掉 `tool_def_tokens` 或 `summary_max_chars`,上面那几条测试**全绿**。
    只有逐项敏感性区分得开「扣了」与「漏扣了」。

    每一项的**权重不同**,别一律按 +100 断:
    - `evidence_block_tokens` 在公式里乘了 `rerank_top_k`(基准里是 4);
    - `max_agent_steps` 涨一格,峰值涨 `tool_result_max_tokens`;
    - `tool_result_max_tokens` 涨 100,峰值涨 `max_agent_steps × 100`;
    - `max_agent_steps` / `tool_result_max_tokens` / `max_user_input_tokens`
      属于**峰值**那一项,**不进** `fixed_overhead` —— 反过来也要断,
      否则「扣了但扣在错误的项里」同样溜过。
    """
    from app.memory import trim

    def fixed(prompt=SYSTEM_PROMPT, **over):
        s = _settings(**{**BASE, **over})
        return budget.derive(settings=s, system_prompt=prompt).fixed_overhead

    def peak(**over):
        return budget.derive(
            settings=_settings(**{**BASE, **over}), system_prompt=SYSTEM_PROMPT
        ).peak

    base_fixed, base_peak = fixed(), peak()

    # ---- 固定开销:五项各涨 100,各自的权重不同 ----
    assert fixed(tool_def_tokens=BASE["tool_def_tokens"] + 100) - base_fixed == 100
    # 证据项带 rerank_top_k 的乘数(基准 4),不是 +100
    assert (
        fixed(evidence_block_tokens=BASE["evidence_block_tokens"] + 100) - base_fixed
        == BASE["rerank_top_k"] * 100
    )
    assert fixed(max_output_tokens=BASE["max_output_tokens"] + 100) - base_fixed == 100
    assert fixed(safety_margin_tokens=BASE["safety_margin_tokens"] + 100) - base_fixed == 100

    # 梗概长度先经 `tokens_for_chars` 折算,不是字数本身。
    # ⚠️ 如实记账:cl100k 下「中」恰好 1 字 = 1 token(实测
    # `count_tokens("中"*N) == N`),所以**本配置下**经不经折算都是 +100 ——
    # 这一条只钉得住「梗概这一项整个被漏掉」(增量 0)。
    # 「字数→token 必须同源」由 `test_tokens_for_chars_...` 单独钉。
    assert (
        fixed(summary_max_chars=BASE["summary_max_chars"] + 100) - base_fixed
        == budget.tokens_for_chars(BASE["summary_max_chars"] + 100)
        - budget.tokens_for_chars(BASE["summary_max_chars"])
    )

    # 渲染后的 system prompt 本身也是一项:换一段更长的,必须跟着涨
    longer_prompt = SYSTEM_PROMPT + "退换货政策请以商品页公示为准。"
    assert (
        fixed(prompt=longer_prompt) - base_fixed
        == trim.count_tokens(longer_prompt) - trim.count_tokens(SYSTEM_PROMPT)
    )

    # ---- 峰值:三项 ----
    assert peak(max_agent_steps=BASE["max_agent_steps"] + 1) - base_peak == (
        BASE["tool_result_max_tokens"]
    )
    assert (
        peak(tool_result_max_tokens=BASE["tool_result_max_tokens"] + 100) - base_peak
        == BASE["max_agent_steps"] * 100
    )
    assert peak(max_user_input_tokens=BASE["max_user_input_tokens"] + 100) - base_peak == 100

    # ---- 反面:峰值三项**不得**动固定开销(重复扣减 / 扣错项) ----
    assert fixed(max_agent_steps=BASE["max_agent_steps"] + 1) == base_fixed
    assert fixed(tool_result_max_tokens=BASE["tool_result_max_tokens"] + 100) == base_fixed
    assert fixed(max_user_input_tokens=BASE["max_user_input_tokens"] + 100) == base_fixed


def test_rounds_branch_wins_when_window_is_generous():
    """两个数取小的那个 —— 两条分支都真的会赢,否则 min() 是装饰。"""
    s = _settings(model_context_window=200_000, keep_rounds=5, per_round_steady=100)
    b = budget.derive(settings=s, system_prompt=SYSTEM_PROMPT)
    assert b.history_budget == 500            # 5 × 100,而不是窗口那一支


def test_layer_split_uses_the_declared_share():
    s = _settings(model_context_window=18000)
    b = budget.derive(settings=s, system_prompt=SYSTEM_PROMPT)
    assert b.layer1_budget == math.floor(b.history_budget * budget.LAYER1_SHARE)
    assert b.layer2_budget == b.history_budget - b.layer1_budget


def test_layer1_takes_floor_not_truncation_when_budget_goes_negative():
    """spec §7.2 写的是 `floor(0.7 × 历史预算)`;`int()` 对负数是**朝零截断**。

    退化配置(峰值吃穿窗口)下历史预算为负,两者差 1:本用例的配置算出来
    `history_budget == -8388` → `floor(-5871.6) == -5872`,而 `int(-5871.6) == -5871`。

    ⚠️ 这条**只**钉负预算那一支 —— 正预算下 floor 与 int 结果相同,分不开。
    所以它把 `history_budget == -8388` 这个前提也断出来:前提变了(比如别处
    改了默认值)要**当场红**,而不是悄无声息地退化成一条恒真断言。
    """
    s = _settings(
        model_context_window=1024,
        max_output_tokens=800,
        tool_def_tokens=800,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
    )
    b = budget.derive(settings=s, system_prompt=SYSTEM_PROMPT)

    assert b.history_budget == -8388      # 前提:负数(前提守不住就该红)
    assert b.layer1_budget == -5872       # floor,不是 int() 的 -5871
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
