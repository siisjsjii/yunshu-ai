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
from app.agent.routing import (
    BUSINESS,
    CHITCHAT,
    COMPLAINT,
    FALLBACK,
    KNOWLEDGE,
    route_by_intent,
)
from app.agent.state import ChatState

#: 出口节点 —— 它们统一汇进 log_turn 再结束。
_OUTLETS = ("agent", "complaint_reply", "chitchat_reply", "fallback_reply")


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
):
    """组装本请求的图并编译。"""
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
            model=model, tools=tools, registry=registry, settings=settings, emit=emit
        ),
    )
    graph.add_node("complaint_reply", make_complaint_reply_node(emit=emit))
    graph.add_node("chitchat_reply", make_chitchat_reply_node(emit=emit))
    graph.add_node("fallback_reply", make_fallback_reply_node(emit=emit))
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
        },
    )
    graph.add_edge("retrieve_knowledge", "confidence_gate")
    graph.add_conditional_edges(
        "confidence_gate", _gate_route, {"agent": "agent", "fallback_reply": "fallback_reply"}
    )
    for outlet in _OUTLETS:
        graph.add_edge(outlet, "log_turn")
    graph.add_edge("log_turn", END)

    return graph.compile(checkpointer=checkpointer)
