"""ch05 图组装。

**两个粒度必须分清**:
- **图按请求组装** —— `query_faq` / `create_ticket` 是每请求闭包(见
  `app/tools/business.py` 的说明),图必须绑到本请求的那批工具上。
  `StateGraph` 构造是纯内存操作,不进请求路径。
- **checkpointer 是进程级单例** —— 每请求新建的话,thread 状态下一轮就没了。
"""

from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    make_agent_node,
    make_chitchat_reply_node,
    make_classify_intent_node,
    make_complaint_reply_node,
    make_confidence_gate_node,
    make_fallback_reply_node,
    make_log_turn_node,
    make_resolve_references_node,
    make_retrieve_knowledge_node,
)
from app.agent.refund_nodes import (
    make_refund_expand_retrieve_node,
    make_refund_explain_node,
    make_refund_fetch_order_node,
    make_refund_judge_node,
    make_refund_offer_node,
    make_refund_pick_order_node,
    route_after_fetch,
    route_after_judge,
)
from app.agent.routing import (
    BUSINESS,
    CHITCHAT,
    COMPLAINT,
    FALLBACK,
    KNOWLEDGE,
    REFUND,
    route_by_intent,
)
from app.agent.state import ChatState
from app.memory import budget
from app.prompts import render_system_prompt

#: 出口节点 —— 它们统一汇进 log_turn 再结束。
#:
#: `refund_pick_order` **不在此列**:它可能停在 `interrupt()` 上,那时 run 根本
#: 走不到 `log_turn`("挂起的那一轮不落库",spec §5.1)。它若也连上 log_turn,
#: 挂起路径就成了「一半写库、一半没写」—— 而这在单测里看不出来(单测要显式
#: resume 才走得到那儿)。
_OUTLETS = (
    "agent", "complaint_reply", "chitchat_reply", "fallback_reply",
    "refund_offer", "refund_explain",
)


@lru_cache(maxsize=1)
def get_checkpointer() -> InMemorySaver:
    """进程级单例。

    **无淘汰、无 TTL**:见过的每个 thread 状态常驻进程(含累积的 trace)。
    本章可接受(演示规模);正式版要换成有界的持久化 checkpointer。
    """
    return InMemorySaver()


def _gate_route(state) -> str:
    return "agent" if state.get("gate_passed") else "fallback_reply"


def build_graph(
    *,
    model,
    intent_model,
    tools,
    registry,
    settings,
    retriever,
    session,
    conversation_id,
    emit,
    checkpointer,
    context_budget=None,
):
    """组装本请求的图并编译。

    `context_budget` 由**端点**推导一次后传进来(agent 节点用它给
    `journal.model_ctx` 记用量与预算)。`None` 时这里现算一份 —— 那是给
    **单测与别的调用方**留的口子:`budget.derive` 是纯函数(同一份 settings、
    同一个 system prompt ⇒ 同一个结果),没有 IO,也没有漂移的余地;
    生产路径永远由端点传下来,于是**每请求只推一次**。
    """
    if context_budget is None:
        context_budget = budget.derive(
            settings=settings, system_prompt=render_system_prompt(settings.brand_name)
        )
    graph = StateGraph(ChatState)

    # 消解用**主力模型**(与 Agent 同一个):它要读的是完整对话,不是结构化出参。
    # 代价是这一步走 `chat_temperature`(0.7),改写措辞每次可能不同 ——
    # 评估集(`scripts/run_resolve_eval.py`,extract 温度 0)与它在温度上不完全同源,
    # 已在 T5 报告里记账。
    graph.add_node("resolve_references", make_resolve_references_node(model=model))
    graph.add_node("classify_intent", make_classify_intent_node(model=intent_model))
    graph.add_node(
        "retrieve_knowledge",
        make_retrieve_knowledge_node(retriever=retriever, emit=emit),
    )
    # ⚠️ 这个 `conversation_id` 必须与 state 里的 `conversation_id` 是**同一个值**。
    # 闸用它写 `low_confidence_questions.source_conversation_id`(即「这问题是从哪段
    # 对话里冒出来的」),而 state 里那个是 `thread_id`、是落库 `append_turn` 的依据。
    # 两者不一致时,问题会被记到**别的会话**名下 —— 而**没有任何测试看得见**:
    # 测试全程给闭包和 state 传同一个字面量,读的也是闭包传的那个。
    # 传入方只有一个(T8 的端点,`conversation_id=session_id`,state 里也是
    # `session_id`),所以今天是一致的;这行注释是给**以后**加调用点的人。
    graph.add_node(
        "confidence_gate",
        make_confidence_gate_node(
            settings=settings, session=session, conversation_id=conversation_id
        ),
    )
    graph.add_node(
        "agent",
        make_agent_node(
            model=model, tools=tools, registry=registry, settings=settings, emit=emit,
            context_budget=context_budget,
        ),
    )
    graph.add_node("complaint_reply", make_complaint_reply_node(emit=emit))
    graph.add_node("chitchat_reply", make_chitchat_reply_node(emit=emit))
    graph.add_node("fallback_reply", make_fallback_reply_node(emit=emit))
    # ---- 退款子流程(ch06)----
    # 入口节点的**参数里没有 emit**:它除了 `interrupt()` 什么都不干(见
    # `app/agent/refund_nodes.py` 的 F3 说明)。取数用它下游的节点,这样 resume
    # 重跑入口时不会产生第二次查询。
    graph.add_node("refund_pick_order", make_refund_pick_order_node())
    graph.add_node(
        "refund_fetch_order",
        make_refund_fetch_order_node(registry=registry, settings=settings),
    )
    graph.add_node(
        "refund_expand_retrieve",
        make_refund_expand_retrieve_node(
            model=model, retriever=retriever, emit=emit, settings=settings
        ),
    )
    # 判定**不新建 Agent**:同一个主力模型、一次结构化调用、不绑工具
    # (第二轮的 `bind_tools` 缺失在这里是结构保证,不是提示词约定)。
    graph.add_node("refund_judge", make_refund_judge_node(model=model))
    graph.add_node("refund_offer", make_refund_offer_node(emit=emit))
    graph.add_node("refund_explain", make_refund_explain_node(emit=emit))
    graph.add_node("log_turn", make_log_turn_node(session=session, emit=emit))

    graph.add_edge(START, "resolve_references")
    graph.add_edge("resolve_references", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            KNOWLEDGE: "retrieve_knowledge",
            BUSINESS: "agent",
            COMPLAINT: "complaint_reply",
            CHITCHAT: "chitchat_reply",
            FALLBACK: "fallback_reply",
            REFUND: "refund_pick_order",
        },
    )
    # ---- 退款子流程的走向(三个条件边,判据都是 state 的纯函数)----
    # 入口 → 取数:`refund_pick_order` 无条件连出 —— 有订单号时它直接返回,
    # 缺号时它 interrupt,resume 之后**从它自己**继续,并不会"再走一遍"下游。
    graph.add_edge("refund_pick_order", "refund_fetch_order")
    graph.add_conditional_edges(
        "refund_fetch_order",
        route_after_fetch,
        {"refund_expand_retrieve": "refund_expand_retrieve",
         # 查不到单:报告,不判(refund_decision 在取数节点里已置 False)。
         "refund_explain": "refund_explain"},
    )
    graph.add_edge("refund_expand_retrieve", "refund_judge")
    graph.add_conditional_edges(
        "refund_judge",
        route_after_judge,
        {"refund_offer": "refund_offer", "refund_explain": "refund_explain"},
    )
    graph.add_edge("retrieve_knowledge", "confidence_gate")
    graph.add_conditional_edges(
        "confidence_gate", _gate_route, {"agent": "agent", "fallback_reply": "fallback_reply"}
    )
    for outlet in _OUTLETS:
        graph.add_edge(outlet, "log_turn")
    graph.add_edge("log_turn", END)

    return graph.compile(checkpointer=checkpointer)
