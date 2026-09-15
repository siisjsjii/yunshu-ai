from collections.abc import AsyncIterator, Sequence

from app.config import Settings
from app.memory import trim
from app.memory.store import SessionStore
from app.prompts import build_messages, render_system_prompt
from app.schemas import Message


def prepare_turn(
    *,
    settings: Settings,
    store: SessionStore,
    session_id: str,
    user_input: str,
) -> list:
    """组装本轮要发给模型的消息。

    预算不足时抛 ContextOverflowError —— 调用方在响应开始前处理,
    因此能返回 400 而不是一个已经开始的 SSE 流。
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

    history = trim.select_history(store.history(session_id), available)
    return build_messages(
        brand_name=settings.brand_name,
        history=history,
        user_input=user_input,
    )


async def stream_turn(
    *,
    settings: Settings,
    store: SessionStore,
    model,
    session_id: str,
    user_input: str,
    messages: Sequence,
) -> AsyncIterator[tuple[str, dict]]:
    """逐 token 产出事件,成功结束后把本轮写入历史。

    产出 (event_name, payload),event_name 取值 "token" / "done"。

    本函数不负责加锁与解锁 —— 锁由 API 层持有,以便在响应开始前
    就能返回 409。
    """
    parts: list[str] = []
    usage = None

    async for chunk in model.astream(messages):
        usage = getattr(chunk, "usage_metadata", None) or usage
        text = chunk.text
        if text:
            parts.append(text)
            yield ("token", {"text": text})

    reply = "".join(parts)

    # 只有流完整走完才会执行到这里。中途抛异常时下面的写入不会发生,
    # 半截回复不会污染历史。
    store.append(
        session_id,
        [
            Message(role="user", content=user_input),
            Message(role="assistant", content=reply),
        ],
    )

    yield ("done", {"finish_reason": "stop", "usage": usage})
