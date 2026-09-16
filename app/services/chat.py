from collections.abc import AsyncIterator, Sequence

from langchain.messages import AIMessage, ToolMessage

from app.config import Settings
from app.memory import trim
from app.prompts import build_messages, render_system_prompt
from app.schemas import Message
from app.services.history import append_turn
from app.tools.executor import execute_tool


def prepare_turn(
    *,
    settings: Settings,
    history: Sequence[Message],
    user_input: str,
) -> list:
    """组装本轮要发给模型的消息。

    历史由调用方从 MySQL 读出后传入 —— 本函数不做 IO,便于单测。

    预算不足时抛 ContextOverflowError,调用方在响应开始前处理,
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

    kept = trim.select_history(history, available)
    return build_messages(
        brand_name=settings.brand_name,
        history=kept,
        user_input=user_input,
    )


async def stream_turn(
    *,
    settings: Settings,
    model,
    session,
    conversation_id: str,
    user_input: str,
    messages: Sequence,
    tools: Sequence,
    registry: dict,
) -> AsyncIterator[tuple[str, dict]]:
    """单轮工具调用编排。

    产出 (event_name, payload),event_name 取值:
    token / tool_call / tool_result / done。

    本函数不负责加锁解锁 —— 锁由 API 层持有。基础设施故障
    (ToolInfrastructureError)向上抛,由 API 层转成 error 帧并终止流。

    **「只做单轮」是结构保证,不是提示词约定**:第二轮用未绑定工具的
    model,模型在结构上无法再调工具。实测确认过"第二轮仍绑 tools 时
    模型这次没再调",但那是模型行为、不是保证,故不采用。
    """
    model_with_tools = model.bind_tools(list(tools))

    # ---- 第一轮:边流边分拣 ----
    # 实测:调工具的提问产出 0 个文本 chunk + 若干 tool_call chunk;
    # 不调工具的提问产出 0 个 tool_call chunk。两者零重叠,
    # 故不需要"先缓冲再判断"的试探逻辑。
    accumulated = None
    first_usage = None
    async for chunk in model_with_tools.astream(messages):
        accumulated = chunk if accumulated is None else accumulated + chunk
        first_usage = getattr(chunk, "usage_metadata", None) or first_usage
        if chunk.text:
            yield ("token", {"text": chunk.text})

    first_text = (getattr(accumulated, "text", "") or "") if accumulated is not None else ""
    tool_calls = list(getattr(accumulated, "tool_calls", None) or [])

    if not tool_calls:
        # 没调工具:第一轮的文本已经流式推完,单次 API 调用即完成。
        await append_turn(
            session=session,
            conversation_id=conversation_id,
            messages=[
                Message(role="user", content=user_input),
                Message(role="assistant", content=first_text),
            ],
        )
        yield ("done", {"finish_reason": "stop", "usage": first_usage})
        return

    # ---- 执行工具 ----
    round_two = list(messages) + [AIMessage(content=first_text, tool_calls=tool_calls)]
    tool_messages: list[Message] = []

    for tool_call in tool_calls:
        yield (
            "tool_call",
            {
                "name": tool_call["name"],
                "args": tool_call["args"],
                "tool_call_id": tool_call["id"],
            },
        )
        outcome = await execute_tool(
            tool_call=tool_call, registry=registry, settings=settings
        )
        yield (
            "tool_result",
            {
                "tool_call_id": outcome.tool_call_id,
                "ok": outcome.ok,
                "summary": outcome.summary,
            },
        )
        round_two.append(
            ToolMessage(content=outcome.content, tool_call_id=tool_call["id"])
        )
        tool_messages.append(
            Message(role="tool", content=outcome.content, tool_call_id=tool_call["id"])
        )

    # ---- 第二轮:不绑 tools,强制收敛为文本 ----
    parts: list[str] = []
    usage = None
    async for chunk in model.astream(round_two):
        usage = getattr(chunk, "usage_metadata", None) or usage
        if chunk.text:
            parts.append(chunk.text)
            yield ("token", {"text": chunk.text})

    reply = "".join(parts)

    # 只有流完整走完才会执行到这里。中途抛异常时下面的写入不会发生,
    # 半截回复不会污染历史。
    await append_turn(
        session=session,
        conversation_id=conversation_id,
        messages=[
            Message(role="user", content=user_input),
            Message(role="assistant", content=first_text, tool_calls=tool_calls),
            *tool_messages,
            Message(role="assistant", content=reply),
        ],
    )
    yield ("done", {"finish_reason": "stop", "usage": usage})
