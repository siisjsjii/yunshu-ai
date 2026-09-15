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


def compute_available_tokens(
    *,
    system_prompt: str,
    user_input: str,
    context_budget_tokens: int,
    reserved_output_tokens: int,
    safety_margin_tokens: int,
) -> int:
    """算出历史消息可用的 token 预算。可以为负,由调用方决定如何处置。"""
    budget = context_budget_tokens - reserved_output_tokens - safety_margin_tokens
    used = count_tokens(system_prompt) + count_tokens(user_input)
    return budget - used


def select_history(
    history: Sequence[Message],
    available_tokens: int,
) -> list[Message]:
    """保留能放下的最近若干整轮历史,按时间正序返回。

    一轮 = (user, assistant) 两条。按整轮裁剪保证历史中不出现
    "有问无答"的孤立消息 —— 那会让模型以为上一轮它没回复。
    """
    kept: list[list[Message]] = []
    used = 0
    for rnd in reversed(_to_rounds(history)):
        cost = sum(count_tokens(msg.content) for msg in rnd)
        if used + cost > available_tokens:
            break
        used += cost
        kept.append(rnd)
    kept.reverse()
    return [msg for rnd in kept for msg in rnd]


def _to_rounds(history: Sequence[Message]) -> list[list[Message]]:
    """把消息序列切成整轮。末尾孤立的 user(上一轮流被打断)单独成轮。"""
    rounds: list[list[Message]] = []
    pending: list[Message] = []
    for msg in history:
        pending.append(msg)
        if msg.role == "assistant":
            rounds.append(pending)
            pending = []
    if pending:
        rounds.append(pending)
    return rounds
