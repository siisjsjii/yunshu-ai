"""ch05 图状态。

用 `TypedDict` + `Annotated` reducer 而非 Pydantic:这是 LangGraph 的惯用法。
节点返回的是**部分**键,由 reducer 合并(`trace` 用 `operator.add` 追加、
其余是**覆盖**语义)—— 搞错会得到「上一轮的值和这一轮混在一起」的诡异状态。
"""

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

from app.schemas import Message


class IntentResult(BaseModel):
    """意图识别的结构化出参。取值越界由 routing 兜底,这里只描述形状。"""

    # 这段枚举是**描述性的第三份标签表**:权威表是 `routing.INTENT_TO_ROUTE`
    # (`INTENT_LABELS` 与它同源,`tests/test_agent_intent.py` 的标签守卫只看那两处,
    # **看不见这里**)。而且 `json_mode` 这条路根本不把 schema 描述发给模型 ——
    # 模型看到的八类在 `prompts.INTENT_SYSTEM_PROMPT` 里。所以这里的措辞改了
    # 不会影响任何行为;留它是为了让读代码的人知道 `intent` 的取值域。
    intent: str = Field(
        description="物流 / 订单 / 商品咨询 / 退款退货 / 售后 / 投诉 / 闲聊 之一;"
        "无法归入任何一类时为「其他」。"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="对本次判断的把握程度。**本章只用于日志**,不改变路由结果。",
    )


class RefundJudgement(BaseModel):
    """退款判定的结构化出参。走 `method="json_mode"`(与抽取/意图/扩写同源)。

    只有两个字段,**没有**「依据」那一项:依据以 `[n]` 编号写进 `reply`
    (编号与 `citations` 帧对齐),多一个字段就多一处模型会写歪的地方。
    `json_mode` 这条路**不把 schema 描述发给模型** —— 模型看到的契约在
    `prompts.REFUND_JUDGE_SYSTEM_PROMPT` 里(同 `IntentResult` 那条注释)。
    """

    can_refund: bool = Field(
        description="这一单能不能退。条款没写明、或条款与这一单对不上时必须为 false。"
    )
    reply: str = Field(default="", description="给用户看的话术,不超过三句话。")


class ChatState(TypedDict):
    """贯穿全图的状态。节点返回的是**部分**键,由 reducer 合并(LangGraph 语义)。"""

    # ---- 输入 ----
    conversation_id: str          # = thread_id = MySQL 会话 id
    user_input: str
    history: list[Message]        # 来自 MySQL(跨轮权威源),转 BaseMessage 只经 prompts.py

    # ---- 指代消解 ----
    resolved_input: str           # 本章 = user_input 原样

    # ---- 意图与检索 ----
    intent: str                   # 八类之一(含「其他」)
    # 必须在这里**声明**:通道集合由 `StateGraph(ChatState)` 的注解决定,而
    # LangGraph 对未声明通道的写入是**静默丢弃**的(`wrote to unknown channel
    # ..., ignoring it`,只 warning 不抛)。T4 初版漏了这一行 —— `classify_intent`
    # 照常写、`log_turn` 照常发,而帧里的 confidence **每帧都是 null**。
    confidence: float             # 分类器自评(0–1);本章只进日志,不参与路由
    evidence: list[dict]          # 知识类:检索到的 chunk(含 score/section_path/chunk_id)
    gate_passed: bool

    # ---- 退款子流程(ch06)----
    # 三个通道**连同它们的每轮清零**一起落地(清零在
    # `nodes.make_resolve_references_node`,计划 PF-2:通道与它的清零必须同处一地)。
    # 漏了清零的后果是**静默串轮** —— checkpointer 是进程级单例、thread_id =
    # session_id,未写的通道保留上一轮的值,于是上一轮填过的订单号会被这一轮
    # 当成本轮槽位:用户明明在问别的,却拿着**上一单**去判能不能退。
    order_no: str                 # 订单号槽位;空串 = 待回填(interrupt 弹卡片)
    order_data: dict              # query_order 的返回;空 dict = 查不到这一单
    refund_decision: bool | None  # None = 还没判 / 判不出来;**不是** False(那是结论)

    # ---- Agent ----
    # 刻意**不**放 ReAct 的消息序列:agent 节点在那一轮内部用局部变量组装
    # (`_stream_round` 的 msgs),落库走 append_turn 的 user+assistant 两条。
    # 放一个没人读写的 state 字段 = 死代码 + 白搭一个 reducer。
    agent_steps: int
    tool_calls_made: list[dict]

    # ---- 输出 ----
    reply: str
    citations: list[dict]
    choices: list[str]            # ["handoff", "ticket"];空则不推帧

    # ---- 日志 ----
    trace: Annotated[list[str], operator.add]
    usage: dict
