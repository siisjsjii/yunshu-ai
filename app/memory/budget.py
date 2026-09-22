"""ch07 上下文预算:从模型窗口倒推,不写死常量。

**纯计算,不做 IO、不依赖 LangChain** —— 转 BaseMessage 是 `prompts.to_lc_messages`
一处的职责(本仓贯穿性约定)。它因此可以被启动自检、端点、单测各调一次而零成本。

两个口径必须与别处**同源**,否则「只改一个、净效果是反的」:
1. token 一律走 `memory.trim.count_tokens`(tiktoken cl100k_base,对中文偏保守);
2. 中文「按字数折 token」不另立系数,直接用同一个 counter 折 —— 见 `tokens_for_chars`。
"""

import math
from dataclasses import dataclass

from app.config import Settings
from app.memory import trim

#: 层 1 拿历史的七成、层 2 拿三成(spec §7.2)。
LAYER1_SHARE = 0.7


@dataclass(frozen=True)
class ContextBudget:
    """一次推导的全部结果。frozen:推导完就该是只读的。"""

    window: int
    fixed_overhead: int
    peak: int
    history_budget: int
    layer1_budget: int
    layer2_budget: int
    fits_one_round: bool


def tokens_for_chars(chars: int) -> int:
    """把「字数」上限折成 token 数。

    **不写系数**(如「1 字 = 1.5 token」),而是拿一段等长的中文过同一个
    counter —— 另立系数就是两把尺子,而它们会在某个字数区间上给出相反结论,
    且两边看起来都合理。`tests/test_memory_budget.py` 钉住这条同源关系。
    """
    if chars <= 0:
        return 0
    return trim.count_tokens("中" * chars)


def derive(*, settings: Settings, system_prompt: str) -> ContextBudget:
    """窗口 → 历史预算 → 层1/层2。

    `system_prompt` 由调用方渲染后传入(`prompts.render_system_prompt`),
    这样本模块**不必 import prompts**,保住 `memory/` 不依赖 LangChain 的约定。
    """
    fixed_overhead = (
        trim.count_tokens(system_prompt)
        + settings.tool_def_tokens
        + settings.rerank_top_k * settings.evidence_block_tokens
        + tokens_for_chars(settings.summary_max_chars)
        + settings.max_output_tokens
        + settings.safety_margin_tokens
    )
    peak = (
        settings.max_agent_steps * settings.tool_result_max_tokens
        + settings.max_user_input_tokens
    )

    by_window = settings.model_context_window - fixed_overhead - peak
    by_rounds = settings.keep_rounds * settings.per_round_steady
    history_budget = min(by_rounds, by_window)

    # `math.floor`,不是 `int()` —— spec §7.2 写的是 floor。`int()` 对负数是
    # **朝零截断**,退化配置(`history_budget` 为负,`fits_one_round is False`)
    # 下两者差 1。
    layer1_budget = math.floor(history_budget * LAYER1_SHARE)
    return ContextBudget(
        window=settings.model_context_window,
        fixed_overhead=fixed_overhead,
        peak=peak,
        history_budget=history_budget,
        layer1_budget=layer1_budget,
        layer2_budget=history_budget - layer1_budget,
        fits_one_round=history_budget >= settings.per_round_steady,
    )
