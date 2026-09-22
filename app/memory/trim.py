from collections.abc import Sequence

import tiktoken

from app.schemas import Message

# cl100k_base 对中文相对 DeepSeek tokenizer 偏保守(高估),
# 因此裁剪会更早触发 —— 估算偏差落在安全的一侧。
_ENCODING = tiktoken.get_encoding("cl100k_base")


class ContextOverflowError(Exception):
    """单轮输入本身超出预算,无法通过裁剪历史解决。"""

    def __init__(self, *, used: int, budget: int) -> None:
        self.used = used
        self.budget = budget
        super().__init__(f"本轮输入需要 {used} tokens,超出可用预算 {budget} tokens")


def count_tokens(text: str) -> int:
    """估算文本的 token 数。这是近似值,不是精确计数。"""
    return len(_ENCODING.encode(text))


# 「历史能用多少 token」这件事**不在本模块算**了(ch07):推导在
# `app/memory/budget.py`(`窗口 - 固定开销 - 单轮峰值`,再与
# `keep_rounds × per_round_steady` 取小)。本模块只做后半截 ——
# 给定预算,裁到能放下。


def select_history(
    history: Sequence[Message],
    available_tokens: int,
) -> list[Message]:
    """保留能放下的最近若干整轮历史,按时间正序返回。

    轮的定义见 `to_rounds`(user 边界)。按整轮裁剪保证历史中不出现
    "有问无答"的孤立消息 —— 那会让模型以为上一轮它没回复;
    也保证 tool 消息不会被与它的 assistant 父亲切开。

    计费只算 `content`:`tool_calls` / `tool_call_id` 是结构性元数据,
    不计入预算,否则预算的含义会被悄悄改掉。
    """
    kept: list[list[Message]] = []
    used = 0
    for rnd in reversed(to_rounds(history)):
        cost = sum(count_tokens(msg.content) for msg in rnd)
        if used + cost > available_tokens:
            break
        used += cost
        kept.append(rnd)
    kept.reverse()
    return [msg for rnd in kept for msg in rnd]


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
