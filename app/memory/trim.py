from collections.abc import Sequence

import tiktoken

from app.schemas import Message

# cl100k_base 对中文相对 DeepSeek tokenizer 偏保守(高估),
# 因此裁剪会更早触发 —— 估算偏差落在安全的一侧。
_ENCODING = tiktoken.get_encoding("cl100k_base")


class ContextOverflowError(Exception):
    """**本轮输入本身**超出预算,无法通过裁剪历史解决。

    ⚠️ 这句文案只对「用户输入过长」成立:`used` 是本轮输入的 token 数、
    `budget` 是 `max_user_input_tokens`。**配置故障**(窗口比 固定开销+单轮峰值
    还小)走子类 `ContextBudgetUnavailable` —— 那时这两个数分别是**固定开销**
    与**整个窗口**,套用这句话会把运维指向「用户话太多」,而真正要改的是窗口配置
    (T10 记账:这是「报错指向别处」的注释版,只是印刷在异常文本里)。
    """

    def __init__(self, *, used: int, budget: int) -> None:
        self.used = used
        self.budget = budget
        super().__init__(f"本轮输入需要 {used} tokens,超出可用预算 {budget} tokens")


class ContextBudgetUnavailable(ContextOverflowError):
    """**配置**故障:窗口连「固定开销 + 单轮峰值」都装不下(一行历史都放不了)。

    与父类同一族(调用方**只 catch 父类**,端点的 400 路径不用改),
    但文案必须不同 —— 两者说的是两件事,而只有一句真话:
    父类那句说的是「这段话太长」,这里的问题是**窗口配得比开销还小**。

    两个数字仍然都在文本里(`used` = 固定开销与峰值算出来的占地、`budget` = 窗口):
    `tests/test_trim.py` 与 `tests/test_api_chat.py` 都按「数字 + `tokens` 字样」
    断言,换措辞不破坏它们。
    """

    def __init__(self, *, used: int, budget: int) -> None:
        Exception.__init__(
            self,
            f"上下文预算不足:固定开销与单轮峰值已占满窗口"
            f"(需要 {used} tokens,窗口 {budget} tokens)",
        )
        self.used = used
        self.budget = budget


def count_tokens(text: str) -> int:
    """估算文本的 token 数。这是近似值,不是精确计数。"""
    return len(_ENCODING.encode(text))


# ---- ch07 起,本模块只剩三样东西 ----
#
# `count_tokens`(全仓唯一的 token 尺子)、`to_rounds`(轮的边界 = 不变量)、
# 以及两个异常(`ContextOverflowError` 及其子类 —— 400 的词汇表)。
#
# 「历史能用多少 token」不在本模块算:推导在 `app/memory/budget.py`
# (`窗口 - 固定开销 - 单轮峰值`,再与 `keep_rounds × per_round_steady` 取小)。
# 「给定预算裁到能放下」也**不在**本模块:**原来的 `select_history` 已删除** ——
# 它按**原文** token 整轮丢弃,而 ch07 的上下文由 `layers.split` 分层(层 2 按
# **截短后**计数),两者叠加会让层 2 少算 ⇒ 摘要永不触发,级联的第二环静默缺席。
# 今天裁的是两处:`layers`(分层 + 层 2 截短 + 降级挪边界)与
# `prompts.select_layer1`(层 1 按 token 预算从最近一轮往前取)。


def to_rounds(history: Sequence[Message]) -> list[list[Message]]:
    """把消息序列切成整轮。

    一轮 = **从一条 user 消息开始,到(不含)下一条 user 消息为止**。

    为什么不用"遇到 assistant 就收一轮"(ch01 的旧规则):引入 tool 角色后,
    后者会把 tool 消息与它的 assistant 父亲切到不同轮里。OpenAI 兼容 API
    要求 tool 消息前面必须紧跟着带对应 tool_call_id 的 assistant 消息,
    切开就会 400,且只在历史长到触发裁剪时复现,极难定位。

    ch07 起由 `_to_rounds` 改名为 `to_rounds` 公开:分层的降级循环
    (`app/memory/layers.py::degrade`)也要按轮挪边界 —— 挪到轮中间就是上面那个 400。
    """
    rounds: list[list[Message]] = []
    for msg in history:
        if msg.role == "user" or not rounds:
            rounds.append([msg])
        else:
            rounds[-1].append(msg)
    return rounds
