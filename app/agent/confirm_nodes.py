"""建工单确认流的两个节点。

**为什么拆成两个**(spec §9.1):ch06 实测过 —— **`resume` 时节点从头重跑**,
`interrupt()` **之前**的代码会再执行一遍。所以:

| 节点 | 做什么 | 有模型? | 有 `interrupt()`? |
|---|---|---|---|
| `confirm_write` | **只有 `interrupt()`** | 没有 | 有 |
| `apply_write_decision` | 执行或拒绝那次写调用 | 没有 | 没有 |

真正写 `tickets` 表的动作在 `apply_write_decision` 里,它在 resume **之后**
只跑一次 —— 两个节点合起来才保证「**副作用恰好一次**」。

**为什么不把 `interrupt()` 放进 `agent`**(最省事的那种写法):`agent` 里有模型
调用(`chat_temperature=0.7`),续跑会把模型再问一遍 —— 已推给前端的文本
**再推一遍**,而第二次的工具调用序列**可能与第一次不同** ⇒ 卡片上的预览与
真正落库的工单**对不上**。这是「看起来能跑、只在真实点击时错」的一类故障。
"""

from langchain_core.messages import ToolMessage
from langgraph.types import interrupt

from app import observability
from app.tools.errors import ToolInfrastructureError
from app.tools.executor import APPROVED, DENIED, execute_tool


def _decision(value) -> str:
    """resume 载荷 → 三态决议。**失败关闭。**

    只认 `{"approved": True}` 这一种形状是**刻意的**:前端一个形状写错
    (比如传了字符串 `"true"`)就会被判成取消,而不是**直接建出工单**。
    写操作不可逆 —— 认不出来的一律不放行。
    """
    if isinstance(value, dict):
        approved = value.get("approved")
    else:
        approved = getattr(value, "approved", None)
    return APPROVED if approved is True else DENIED


def make_confirm_write_node():
    """工单预览闸。**`interrupt()` 之外不干任何事。**

    载荷里的 `frame` 由**它自己**说,端点只做搬运 —— 所以本章
    **一行端点代码都不用改**(ch06 那处设计的直接回报)。
    """

    async def confirm_write(state) -> dict:
        pending = state.get("pending_write") or {}
        decision = interrupt(
            {
                "frame": "ticket_confirm",
                "preview": pending.get("preview") or {},
            }
        )
        return {
            "write_decision": _decision(decision),
            "trace": ["confirm_write"],
        }

    return confirm_write


def make_apply_write_decision_node(*, registry, settings, emit):
    """决议落地:批准就执行一次,取消就落一条「权限拒绝」审计。

    **两条路都往本轮消息里追加一条 ToolMessage** —— 因为那条带 `tool_calls`
    的 AIMessage 已经在 `turn_messages` 里了,**少回灌一个 tool 结果就构成
    「有 tool_calls 没有对应 tool 消息」,上游直接 400**(CLAUDE.md 的硬约束)。

    **入口自己拒空 `pending_write`**(上抛,见函数体里的说明):路由侧也守一次,
    但唯一写口上的这一道才是「不变量」本身。

    `emit` **是必填的**(没有默认的 no-op),理由与 `emit.py` 那条同族:
    默认值会让「忘了传」退化成一个**静默少一帧**的实现 —— 而那一帧正是
    前端徽标唯一能停下来的机会。
    """

    async def apply_write_decision(state) -> dict:
        pending = state.get("pending_write") or {}
        # ---- 不变量守在这里,不只守在路由上(T9)----------------------
        # `pending_write` 为空 ⇒ **上抛**。空着往下走会造出
        # `tool_call_id=""` 的 ToolMessage —— 那构成「有 tool result、没有对应
        # tool_call」,上游同样直接 **400**;更糟的是它**看起来像一次正常结果**。
        #
        # 为什么**两处都守**(路由侧 `route_after_agent` 也守一次):本仓的元教训是
        # 「**不变量要放在唯一写口上,不要靠每个调用方自觉**」。只守路由 = 把这条
        # 不变量寄存在调用方的记忆里,而 T9 之后入口会变多(续跑、将来的第四条出口,
        # 以及任何直接调这个节点的地方)。这一处是**唯一写口**,它自己拒才算数。
        #
        # 空 `pending_write` 说明**接线坏了**(没有待确认写调用却到达了这个节点),
        # 不是用户动作 —— 与执行器那条「决议取值认不出来就抛」同族:
        # 接线 bug 要**响亮地**暴露,不许伪装成一次调用。抛
        # `ToolInfrastructureError` ⇒ 端点 502。
        if not pending:
            raise ToolInfrastructureError(
                "apply_write_decision 被到达时 pending_write 为空(接线 bug)"
            )
        # ⚠️ **不要写 `or DENIED`。** 空决议说明 `confirm_write` 没跑、或它没写进通道,
        # 那是接线 bug;按 DENIED 处理会在审计表里**谎报一次用户取消** ——
        # 与执行器那条 `!= APPROVED` 闸上抛的理由完全相同,只是层数更高一层。
        # 空串会落进那一支,响亮地抛。
        decision = state.get("write_decision") or ""
        # ⚠️ **`"type": "tool_call"` 这个键必须在。** `BaseTool.ainvoke` 判
        # 「这是不是一次工具调用」**只看它** —— 缺键时它把整个 dict 当成**参数**去
        # 校验工具 schema,于是这次调用退化成一条「参数不合法」的**可恢复**失败:
        # **工单永远不会被建出来**,而调用方看起来一切正常。
        # (T4 的实现者在测试初稿上撞过同一件事,6 条用例红在 pydantic 的
        #  `Field required` 上 —— 那是**测试**;在这里它是**生产**。)
        call = {
            "name": pending.get("name", ""),
            "id": pending.get("tool_call_id", ""),
            "args": pending.get("args") or {},
            "type": "tool_call",
        }
        # ch09:这一处是那次写调用**真正执行**的地方(agent 循环里那次已经
        # `continue` 掉了,它的 `tool:*` span 记的是「待确认」)。不给它开 span
        # 的话,trace 上就**看不到写操作到底执行了没有** —— 而这正是确认流
        # 唯一值得看的一步。名字与 agent 循环里那条同形,便于对照。
        with observability.span(
            f"tool:{call['name']}", as_type="tool",
            input=call["args"], settings=settings,
        ) as sp:
            outcome = await execute_tool(
                tool_call=call,
                registry=registry,
                settings=settings,
                conversation_id=state["conversation_id"],
                write_decision=decision,
            )
            if sp is not None:
                sp.update(output={
                    "ok": outcome.ok,
                    "summary": outcome.summary,
                    "error_kind": outcome.error_kind,
                })
        # ⚠️ **这条帧必须在「决议之后」发,而且只在这里发**(T10 的 bundled 修复)。
        #
        # `agent` 的循环对每个调用**先**发 `tool_call` 帧、**再**执行;撞到待确认
        # 的写调用时它 `continue` 了 ⇒ 那次调用的 `tool_result` **一帧都不发**。
        # 而决议落在**下一次** run 里,所以补发的责任在这个节点上 —— 不补的话
        # 前端那个徽标**一直转**(它只在收到 `tool_result` 时才结算)。
        #
        # **不能改成「挂起前先把徽标关掉」**:那会把徽标的语义变成「这个工具已经
        # 跑完了」,而它**确实还在等用户** —— 让它转着才是诚实的。这条帧发的时刻
        # 就是那次调用唯一一次真正有结果的时刻(执行完 / 明确被拒)。
        #
        # **两条路都要发**:取消同样是「这次调用结束了」,漏掉的话取消路径的
        # 徽标永远转下去(`ok=False` 让前端把它画成失败态)。
        emit({"frame": "tool_result", "tool_call_id": outcome.tool_call_id,
              "ok": outcome.ok, "summary": outcome.summary})
        tool_msg = ToolMessage(content=outcome.content, tool_call_id=call["id"])
        # ⚠️ `turn_messages` 是**覆写**通道,承载「本轮产生的**全部**消息」。
        # 这里只返回 `[tool_msg]` 的话,那条带 `tool_calls` 的 AIMessage
        # 会被丢掉 —— 落库的历史里助手消息凭空少一条,而回复看起来完全正常。
        existing = list(state.get("turn_messages") or [])
        return {
            "messages": [tool_msg],
            "turn_messages": existing + [tool_msg],
            "pending_write": {},
            "trace": [
                "write:approved" if decision == APPROVED else "write:denied"
            ],
        }

    return apply_write_decision
