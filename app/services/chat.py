"""本轮对话的**前置校验**。

ch05 Task 8:单轮编排从这里搬走了 —— `stream_turn` 整个删掉,由
`app/agent/graph.py` 的图负责(指代消解 → 意图识别 → 路由 → 检索/闸/Agent →
落库)。本模块只剩 `prepare_turn`:它必须在图开始跑之前做完校验,
因为端点只有"流还没开始"这个窗口能返回 400。

ch07:
- 这里守着 spec §8 那张表里的**两条 400** —— 本轮输入超 `max_user_input_tokens`、
  以及历史预算连一轮都装不下。两条都必须在流开始前完成。
- **它不再裁剪历史、也不再返回历史**(见 `prepare_turn` 的 docstring)。
"""

from app.config import Settings
from app.memory import trim
from app.memory.budget import ContextBudget


def prepare_turn(*, settings: Settings, budget: ContextBudget, user_input: str) -> None:
    """**纯校验**:流开始前的两条 400 判据。**不返回历史**。

    ch07 起,历史不再是「裁一条单层的」—— 它由 `layers.split` 分三层
    (层 2 截短、层 1 原文,spec §3),而这一层与旧口径**不能叠加**:
    拿裁剪过的历史去分层 = 层 2 少算 ⇒ `should_summarize` 永不触发,
    而级联的第二环会在日志里**平静地缺席**。
    所以这里只剩校验;`select_history` 连同它的调用一起删除。

    `budget` 由调用方推导后传入 —— 端点本来就要它来算两个层的预算与降级阈值,
    在这里再 `derive` 一次是同一个事实算两遍(也是本仓「一个数只有一个来源」
    那套既有取向;`derive` 是纯函数,但每请求多跑一遍 tiktoken 是白工)。

    两条判据,顺序有意:

    1. **本轮输入**超 `max_user_input_tokens` —— 输入本身超限与历史装不下是
       两回事,但对客户端都是 400,所以都归这一族。放在前面:它连预算都不用看。
    2. **历史预算装不下一轮**(`not budget.fits_one_round`)—— 这是 spec §8 的判据。
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

    if not budget.fits_one_round:       # spec §8:「装不下一轮」就是 400 的判据
        # 这是**配置**故障(窗口 < 固定开销 + 单轮峰值),不是用户输入的问题 ——
        # 用子类的文案,别让运维去猜是不是用户话太多(见 `trim` 的两个类)。
        raise trim.ContextBudgetUnavailable(
            used=budget.fixed_overhead, budget=budget.window
        )
