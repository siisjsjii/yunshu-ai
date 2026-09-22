"""ch05 图状态。

用 `TypedDict` + `Annotated` reducer 而非 Pydantic:这是 LangGraph 的惯用法。
节点返回的是**部分**键,由 reducer 合并(`trace` 用 `operator.add` 追加、
其余是**覆盖**语义)—— 搞错会得到「上一轮的值和这一轮混在一起」的诡异状态。
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
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

    # ---- 完整历史(跨轮,ch07)----
    # add_messages 是 **append-only** reducer:各节点只管吐**新**消息,框架按序并入。
    # 同一个 id 再次并入是**替换**(LangGraph 的既定语义),这就是「播种幂等」
    # 的另一半 —— `prompts.to_lc_messages` 给每条带上 MySQL 主键做稳定 id。
    #
    # ⚠️ 它**不进** `resolve_references` 的逐轮重置清单 —— 那份清单是给
    # 「没有播种者、只能靠重置」的通道用的,而 `messages` 是**累积**语义。
    # (更正一处早先写错机制的注释:把 `messages` 加进清单**不会**清空历史 ——
    # `add_messages(left, [])` 实测**原样返回 `left`**,空列表更新在 append-only
    # reducer 上是 no-op。决定不改(不放清单)仍然对,但理由是「**不必要**」,
    # 不是「危险」。)
    #
    # 三个写者,加起来才是「完整历史」:
    #   ① 端点(只在快照为空时播种一次,重启自愈);
    #   ② `resolve_references`(**本轮的用户原话** —— 唯一的用户消息来源);
    #   ③ 各出口节点 + `agent`(本轮的客服回复 / ReAct 往返)。
    # 少了 ② 的话它就不是完整历史,而是「客服说过的话的流水」—— 且**完全静默**。
    messages: Annotated[list[AnyMessage], add_messages]

    #: 注入给模型的**梗概全文**(逐轮覆写,由端点播种)。空串 = 还没有梗概。
    #: 与 `history` 同族:覆写语义 + 每轮重新播种,所以**进的是 `stream_input`**,
    #: 不进 `resolve_references` 的重置清单。
    summary_text: str

    #: 这一轮**真的用过**的那一对锚点(逐轮覆写,由端点播种,spec §7.5)。
    #:
    #: 为什么要在 state 里:agent 节点打完 `model_ctx` 那一行时,手里是**扁平**的
    #: 精简版,而 `journal.model_ctx` 要一份 `Layers`(它从中读 `sliding` 与
    #: `bounds`)。没有锚点就没法重新分类,而那行日志会把 `bounds` 报成 `0`
    #: —— `0` 在本章**是个有含义的值**(尚无梗概 / 层 1 起于最早),
    #: 于是「日志在描述一次没发生过的切分」而两边都不报错。
    #:
    #: ⚠️ **必须在 `ChatState` 里声明**:通道集合由 `StateGraph(ChatState)` 的
    #: 注解决定,写没声明的通道 LangGraph **静默丢弃**(只 warning 不抛)——
    #: ch06 的 `confidence` 就是这么丢的(T4 的 Critical)。
    #: 与 `history` / `summary_text` 同族:覆写 + 每轮播种,**不进重置清单**
    #: (续跑那条路读到的是 checkpoint 里上一轮的值,而它只供日志、不参与判断)。
    summary_upto_msg_id: int
    layer1_from_msg_id: int

    #: **本轮**新产生的消息 —— `log_turn` 落库的唯一依据。
    #:
    #: 为什么不直接用 `messages`:那个是**累积**通道(全量),拿它落库 = 每轮把
    #: 整段历史再写一遍 ⇒ **历史翻倍**,而每一轮的回复看起来都正常、每条单测
    #: 只要不数字数就全绿。这个是**覆写**通道,各节点写它时连同 `messages`
    #: 一起写(两个都写)。
    #:
    #: 它是本章新增通道里**唯一一个不会被节点自动覆盖**的(其余每轮都有节点写,
    #: 或由端点每轮播种),所以必须进 `resolve_references` 的逐轮重置 ——
    #: 否则「这轮没产生消息」的路径会原样继承上一轮的值,上一轮的回复被再写一遍。
    turn_messages: list[AnyMessage]

    # ---- 建工单确认流(ch08)----
    # ⚠️ **必须在 `ChatState` 里声明**:通道集合由 `StateGraph(ChatState)` 的注解决定,
    # 写没声明的通道 LangGraph **静默丢弃**(只 warning 不抛)—— ch06 的
    # `confidence` 就是这么丢的(T4 的 Critical)。
    #
    # 两个都**连同它们的每轮清零一起落地**(清零在 `nodes.make_resolve_references_node`,
    # ch05–ch07「通道与它的清零必须同处一地」的第四次应用)。漏了清零的后果是
    # **跨轮串味**:checkpointer 是进程级单例、thread_id = session_id,
    # 未写的通道保留上一轮的值 —— 于是**上一轮批准过的写操作,这一轮自动放行**。
    #
    # 注意:**不需要在端点播种**。它们都在**一轮的中途**被写(`agent` 写
    # `pending_write`、`confirm_write` 写 `write_decision`),而续跑路径不重跑
    # `resolve_references` —— 续跑续的是**同一轮**,清零不该发生。
    pending_write: dict          # 空 dict = 没有待确认的写操作
    write_decision: str          # "" = 未决议;APPROVED / DENIED

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
    # ch07 起 ReAct 的消息序列**进 state**(上头的 `messages` / `turn_messages`)——
    # ch05–ch06 那句「刻意不放」已经作废:那时它只活在 `_stream_round` 的局部
    # 变量里、落库走 append_turn 的 user+assistant 两条,所以本章之前
    # production **零处**写过 `role='tool'` 的行。
    agent_steps: int
    tool_calls_made: list[dict]

    # ---- 输出 ----
    reply: str
    citations: list[dict]
    choices: list[str]            # ["handoff", "ticket"];空则不推帧

    # ---- 日志 ----
    trace: Annotated[list[str], operator.add]
    usage: dict
