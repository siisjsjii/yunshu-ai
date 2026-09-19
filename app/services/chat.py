"""本轮对话的**前置**处理。

ch05 Task 8:单轮编排从这里搬走了 —— `stream_turn` 整个删掉,由
`app/agent/graph.py` 的图负责(指代消解 → 意图识别 → 路由 → 检索/闸/Agent →
落库)。本模块只剩 `prepare_turn`:它必须在图开始跑之前做完预算校验,
因为端点只有"流还没开始"这个窗口能返回 400。
"""

from collections.abc import Sequence

from app.config import Settings
from app.memory import trim
from app.prompts import render_system_prompt
from app.schemas import Message


def prepare_turn(*, settings: Settings, history: Sequence[Message], user_input: str) -> list[Message]:
    """预算校验 + 历史裁剪。返回**裁剪后的历史**(消息组装由 ch05 的节点做)。

    历史由调用方从 MySQL 读出后传入 —— 本函数不做 IO,便于单测。

    预算不足时抛 ContextOverflowError,调用方在响应开始前处理,因此能返回
    400 而不是一个已经开始的 SSE 流。这条约束 ch05 不变:图开始跑之前必须
    已经知道预算够不够。
    """
    system_prompt = render_system_prompt(settings.brand_name)
    available = trim.compute_available_tokens(
        system_prompt=system_prompt,
        user_input=user_input,
        context_budget_tokens=settings.context_budget_tokens,
        reserved_output_tokens=settings.reserved_output_tokens,
        safety_margin_tokens=settings.safety_margin_tokens,
    )
    if available < 0:
        budget = (
            settings.context_budget_tokens
            - settings.reserved_output_tokens
            - settings.safety_margin_tokens
        )
        raise trim.ContextOverflowError(
            used=trim.count_tokens(system_prompt) + trim.count_tokens(user_input),
            budget=budget,
        )
    return trim.select_history(history, available)
