"""本轮对话的**前置**处理。

ch05 Task 8:单轮编排从这里搬走了 —— `stream_turn` 整个删掉,由
`app/agent/graph.py` 的图负责(指代消解 → 意图识别 → 路由 → 检索/闸/Agent →
落库)。本模块只剩 `prepare_turn`:它必须在图开始跑之前做完预算校验,
因为端点只有"流还没开始"这个窗口能返回 400。
"""

from collections.abc import Sequence

from app.config import Settings
from app.memory import budget, trim
from app.prompts import render_system_prompt
from app.schemas import Message


def prepare_turn(*, settings: Settings, history: Sequence[Message], user_input: str) -> list[Message]:
    """预算校验 + 历史裁剪。返回**裁剪后的历史**。

    ch07:预算从「直接给一个数」改成「从模型窗口倒推」(`memory.budget`)。
    本任务只换预算来源,**分层留到 T10** —— 此刻仍是单层
    `trim.select_history`,所以每一步都能单独跑绿。

    预算不足时抛 `ContextOverflowError`,调用方在响应开始前处理,
    因此能返回 400 而不是一个已经开始的 SSE 流。

    `user_input` 此刻**不参与**判据:倒推出来的历史预算只取决于配置与
    system prompt,与本轮输入多长无关(旧口径里它被算进「已用」,新口径把它
    归进单轮峰值的 `max_user_input_tokens` 那一项)。**签名保持不变**,免得
    端点与既有的 `resume` 分支(那里递进来的本来就是 `None`)跟着改一遍;
    而「本轮输入超 `max_user_input_tokens` → 400」那条由端点在 T10 接线
    (spec §8),不在这里。
    """
    b = budget.derive(
        settings=settings, system_prompt=render_system_prompt(settings.brand_name)
    )
    if b.history_budget < 0:
        raise trim.ContextOverflowError(used=b.fixed_overhead, budget=b.window)
    return trim.select_history(history, b.history_budget)
