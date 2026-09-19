"""祛魅热身:最裸的 Agent 循环(**临时产物**,任务 8 落地图后删除)。

没有图、没有 State、没有 checkpointer —— 只有一个 for:
调模型 → 有 tool_call 就执行并回灌 → 没有就收敛。
它存在的意义只是证明「Agent 就是个带工具的循环」,不是魔法。
"""

from langchain_core.messages import ToolMessage

from app.tools.executor import execute_tool


async def run_agent_loop(
    *,
    model,
    messages: list,
    tools,
    registry: dict,
    settings,
    max_steps: int = 5,
) -> tuple[str, int]:
    """跑一轮最裸的 ReAct。返回 (回复文本, 实际步数)。

    每步 `bind_tools` 调模型:无 tool_calls 即收敛;有则逐个执行并回灌。
    步数用尽时**最后一轮不绑 tools** —— 模型在结构上无法再调,必然收敛。
    这是「停止条件是结构保证、不是提示词约定」的第一处体现。
    """
    bound = model.bind_tools(list(tools))
    msgs = list(messages)
    parts: list[str] = []

    for step in range(1, max_steps + 1):
        ai = await bound.ainvoke(msgs)
        parts.append(getattr(ai, "text", "") or "")
        tool_calls = list(getattr(ai, "tool_calls", None) or [])
        if not tool_calls:
            return "".join(parts), step

        msgs.append(ai)
        for call in tool_calls:
            outcome = await execute_tool(
                tool_call=call, registry=registry, settings=settings
            )
            msgs.append(ToolMessage(content=outcome.content, tool_call_id=call["id"]))

    # 步数用尽:不绑 tools 再问一次,收尾。
    final = await model.ainvoke(msgs)
    parts.append(getattr(final, "text", "") or "")
    return "".join(parts), max_steps
