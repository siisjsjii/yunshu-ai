"""本轮对话的**前置**处理。

ch05 Task 8:单轮编排从这里搬走了 —— `stream_turn` 整个删掉,由
`app/agent/graph.py` 的图负责(指代消解 → 意图识别 → 路由 → 检索/闸/Agent →
落库)。本模块只剩 `prepare_turn`:它必须在图开始跑之前做完预算校验,
因为端点只有"流还没开始"这个窗口能返回 400。

ch07:这里守着 spec §8 那张表里的**两条 400** —— 本轮输入超
`max_user_input_tokens`、以及历史预算连一轮都装不下。两条都必须在流开始前完成。
"""

from collections.abc import Sequence

from app.config import Settings
from app.memory import budget, trim
from app.prompts import render_system_prompt
from app.schemas import Message


def prepare_turn(*, settings: Settings, history: Sequence[Message], user_input: str) -> list[Message]:
    """预算校验 + 历史裁剪。返回**裁剪后的历史**。

    ch07:预算从「直接给一个数」改成「从模型窗口倒推」(`memory.budget`)。
    分层(三层 + 两个锚点)**不在本函数里** —— 它在端点,因为分层要读会话上
    已经落库的两个锚点(`Conversation` 的两列),而本函数只看得到一份历史。

    预算不足时抛 `ContextOverflowError`,调用方在响应开始前处理,
    因此能返回 400 而不是一个已经开始的 SSE 流。两条判据,顺序有意:

    1. **本轮输入**超 `max_user_input_tokens` —— 输入本身超限与历史装不下是
       两回事,但对客户端都是 400,所以都归这一族。放在前面:它连 `derive`
       都不用跑(输入长度与窗口预算无关)。
    2. **历史预算装不下一轮**(`not fits_one_round`)—— 这是 spec §8 的判据。
       T2 实现的是 `history_budget < 0`,两者在「预算为正但小于
       `per_round_steady`」时**结论相反**:那时按 spec 该 400,按 `< 0` 却会
       **静默带着几乎空的历史往下走** —— 用户拿到一个没有任何上下文的回答,
       而没有任何东西报错。`< 0` 被 `not fits_one_round` 完全覆盖
       (`0 < per_round_steady` 恒成立),所以这次改动是收紧、不是放宽。

    `user_input` 允许为 `None`:今天 `resume` 分支**不调**本函数(它拿不到本轮
    输入 —— 挂起那轮的原话在 state 里),但类型上留这个口子,免得将来有人加调用点
    时把 `None` 送进 tiktoken 变成一次 500。
    """
    if user_input is not None:
        used = trim.count_tokens(user_input)
        if used > settings.max_user_input_tokens:
            raise trim.ContextOverflowError(
                used=used, budget=settings.max_user_input_tokens
            )

    b = budget.derive(
        settings=settings, system_prompt=render_system_prompt(settings.brand_name)
    )
    if not b.fits_one_round:            # spec §8:「装不下一轮」就是 400 的判据
        # 这是**配置**故障(窗口 < 固定开销 + 单轮峰值),不是用户输入的问题 ——
        # 用子类的文案,别让运维去猜是不是用户话太多(见 `trim` 的两个类)。
        raise trim.ContextBudgetUnavailable(used=b.fixed_overhead, budget=b.window)
    return trim.select_history(history, b.history_budget)
