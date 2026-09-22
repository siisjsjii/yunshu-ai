"""ch05 各节点的工厂函数。

统一用**闭包工厂**(同 `app/tools/business.py` 的既有做法):模型、工具、
会话、检索器都不是模块级单例,而是每请求经闭包绑定 —— 这样节点既拿到了
依赖,又不会在 import 时做任何 IO。
"""

import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from app.agent.routing import INTENT_TO_ROUTE, OTHER
from app.agent.state import IntentResult
from app.kb.assess import record_low_confidence
from app.prompts import build_intent_messages, build_messages, build_resolve_messages
from app.schemas import Message
from app.services.history import append_turn
from app.tools.executor import execute_tool

logger = logging.getLogger(__name__)


def make_classify_intent_node(*, model):
    """意图识别:一次 LLM(json_mode),输出八类之一。

    解析失败或越界一律降级为「其他」 —— **不抛异常**。理由:意图识别是骨架
    的第一步,它失败时整轮对话不该跟着崩;降级后由 `route_by_intent` 送进
    兜底出口,用户至少能拿到一句「请再说具体些」。

    `confidence` 同样出参,但本章**不参与路由** —— 它只进 state、进 `log_turn`
    的 `trace` 帧(spec §4.2 要求它最终进 done 帧),留给后面的降级路。
    """
    chain = model.with_structured_output(IntentResult, method="json_mode")

    async def classify_intent(state) -> dict:
        try:
            result = await chain.ainvoke(build_intent_messages(state["user_input"]))
            intent = result.intent
            # `getattr` 而不是直接取属性:confidence 只是**日志字段**,出参形状
            # 不合预期时不该把整轮打成 500 —— 本节点的契约就是「降级,不抛异常」。
            # 同文件读模型输出处(`acc.tool_calls`、`chunk.usage_metadata`)是同一种写法。
            # 注意这条兜底**真机不可达**(`with_structured_output` 保证拿到的是校验过的
            # `IntentResult`,其 default 保证字段不缺失),所以它是纯防御、无测试能区分。
            confidence = float(getattr(result, "confidence", 0.0) or 0.0)
        except (OutputParserException, ValidationError) as exc:
            logger.warning("意图识别解析失败,降级为「其他」:%s", exc)
            intent = OTHER
            confidence = 0.0

        if intent not in INTENT_TO_ROUTE:
            intent = OTHER
        return {
            "intent": intent,
            "confidence": confidence,
            "trace": [f"classify_intent:{intent}"],
        }

    return classify_intent


# ---- 固定话术出口 ----
#
# 三个出口**都不调模型**:闲聊与兜底是纯文案(零成本、零延迟),投诉的安抚话术
# 同样固定。这三个工厂的签名里因此根本没有 model 参数 —— 那是「不花模型调用」
# 的**结构保证**,不是「我们记得别调」的行为约定。

COMPLAINT_REPLY = (
    "非常抱歉给您带来不好的体验,您反馈的问题我已经记录下来了。"
    "您可以让我转人工客服,或者为您建一张工单跟进,选哪个都可以。"
)
CHITCHAT_REPLY = "你好呀~我是本店客服小猫,有什么可以帮您的吗?"
FALLBACK_REPLY = "抱歉,我没太理解您的意思,可以再说得具体一些吗?"

CHOICE_HANDOFF = {"key": "handoff", "label": "转人工"}
CHOICE_TICKET = {"key": "ticket", "label": "建工单"}


def make_chitchat_reply_node(*, emit):
    """闲聊:固定话术,不推进任何后续动作。

    **必须自己发 token 帧**:前端是「累积 token 画出气泡」的,只写
    `state["reply"]` 的话后端有话、前端空白 —— 而且所有断言 `reply` 的单测
    照样全绿,验收 4 才会暴露。
    """

    async def chitchat_reply(state) -> dict:
        emit({"frame": "token", "text": CHITCHAT_REPLY})
        # 两个通道**都要写**:`messages` 给 add_messages 累积、`turn_messages` 给
        # `log_turn` 落库(ch07)。只写 `reply` 的节点在 ch07 会**只落用户那一句、
        # 客服回复永远不落库** —— 而每一轮看起来都正常(见 `log_turn` 的说明)。
        return {
            "reply": CHITCHAT_REPLY,
            "messages": [AIMessage(content=CHITCHAT_REPLY)],
            "turn_messages": [AIMessage(content=CHITCHAT_REPLY)],
            "choices": [], "trace": ["chitchat_reply"],
        }

    return chitchat_reply


def make_fallback_reply_node(*, emit):
    """兜底:意图不属于七类、或分类解析失败时走这里(用户明确要求)。

    也是置信度闸不通过时的落点 —— 那时问题已由 confidence_gate 落进低置信度池。
    """

    async def fallback_reply(state) -> dict:
        emit({"frame": "token", "text": FALLBACK_REPLY})
        return {
            "reply": FALLBACK_REPLY,
            "messages": [AIMessage(content=FALLBACK_REPLY)],
            "turn_messages": [AIMessage(content=FALLBACK_REPLY)],
            "choices": [], "trace": ["fallback_reply"],
        }

    return fallback_reply


def make_complaint_reply_node(*, emit):
    """投诉:安抚话术 + 把「转人工」「建工单」两个选项交给用户自己选。

    **两者是两回事,分开给** —— 后端不自动执行任何一个:转人工是前端模拟,
    建工单要用户点了按钮才走 /api/ticket。用户都不点就继续正常对话。
    """

    async def complaint_reply(state) -> dict:
        emit({"frame": "token", "text": COMPLAINT_REPLY})
        emit({"frame": "choices", "options": [CHOICE_HANDOFF, CHOICE_TICKET]})
        return {
            "reply": COMPLAINT_REPLY,
            "messages": [AIMessage(content=COMPLAINT_REPLY)],
            "turn_messages": [AIMessage(content=COMPLAINT_REPLY)],
            "choices": ["handoff", "ticket"],
            "trace": ["complaint_reply:choices"],
        }

    return complaint_reply


# ---- 知识类:强制预检索 + 置信度闸 ----


def make_retrieve_knowledge_node(*, retriever, emit):
    """知识类意图的**强制**预检索(确定性骨架的一步,不走 query_faq 工具)。

    检索器的故障语义原样透传:`KnowledgeRetriever` 把 Milvus/嵌入的故障翻成
    `ToolInfrastructureError`,这里**不接** —— 它必须一路抛到 API 层变 502,
    绝不能被伪装成「没搜到」。

    `citations` **同时**写进 state 并发一帧:state 那份给 Agent 组装引用编号,
    帧那份给前端渲染可点击的来源。少发帧 = ch04 的引用 UI 静默失效。
    """

    async def retrieve_knowledge(state) -> dict:
        chunks = await retriever.search(state["resolved_input"])
        evidence = [
            {
                "chunk_id": c.chunk_id,
                "section_path": c.section_path,
                "question": c.question,
                "answer": c.answer,
                "category": c.category,
                "score": c.score,
            }
            for c in chunks
        ]
        citations = [
            {"n": i + 1, **{k: e[k] for k in
                            ("chunk_id", "section_path", "question", "answer", "category")}}
            for i, e in enumerate(evidence)
        ]
        if citations:
            # 载荷键必须是 **`items`**,不是 `citations`:ch04 的
            # `app/static/index.html` 里是 `ctx.citations = payload.items || []`。
            # 换个键名 = 帧到了、前端仍渲染不出引用(静默失效),而且
            # ch05 的单测只断言「发了一帧」,照样全绿。
            emit({"frame": "citations", "items": citations})
        top = f" top={evidence[0]['score']:.2f}" if evidence else ""
        return {
            "evidence": evidence,
            "citations": citations,
            "trace": [f"retrieve_knowledge:{len(evidence)} hits{top}"],
        }

    return retrieve_knowledge


def make_confidence_gate_node(*, settings, session, conversation_id):
    """置信度闸:卡在检索之后、进 Agent 之前。

    **为什么必须在这儿**:Agent 的答复是流式吐给用户的,等答完再判就晚了
    (ch04 的自评正是那个位置,本章把它撤掉)。证据弱就直接回兜底话术、
    不进 Agent,同时把问题落池留给后面的数据飞轮。

    判据是纯**检索分数阈值**(取最高分),零额外模型调用 —— 最简版;
    正式的置信度检查留给「可观测」那章。
    """

    async def confidence_gate(state) -> dict:
        scores = [e["score"] for e in (state.get("evidence") or [])]
        passed = bool(scores) and max(scores) >= settings.retrieval_score_threshold

        if not passed:
            await record_low_confidence(
                session,
                question=state["user_input"],
                source_conversation_id=conversation_id,
                entry_point="置信度闸",
                reject_reason=(
                    "检索为空"
                    if not scores
                    else f"最高分 {max(scores):.2f} 低于阈值 {settings.retrieval_score_threshold}"
                ),
            )

        return {
            "gate_passed": passed,
            "trace": [f"confidence_gate:{'pass' if passed else 'fail'}"],
        }

    return confidence_gate


# ---- 主力 Agent:手写 ReAct ----


def _total_tokens(chunk) -> int:
    meta = getattr(chunk, "usage_metadata", None) or {}
    return int(meta.get("total_tokens") or 0)


def make_agent_node(*, model, tools, registry, settings, emit):
    """主力 Agent 的 ReAct 循环。

    **不用 ToolNode / create_react_agent**:工具执行必须走 `execute_tool`,
    它承载本项目的错误语义 —— 基础设施故障抛 `ToolInfrastructureError`(→502)、
    重试用白名单(`create_ticket` 永不重试)、`ValidationError`/`ToolNotFound`
    不重试、10s 超时。`ToolNode` 直接 `tool.ainvoke`,这些语义全丢。

    停止条件是**结构保证**:循环里绑着 tools 走 `max_agent_steps` 轮;步数用尽
    或预算超限后,收尾那一轮**不绑 tools**,模型在结构上无法再调。
    """
    bound = model.bind_tools(list(tools))

    async def _stream_round(target, msgs, parts) -> tuple[object, int]:
        acc = None
        used = 0
        async for chunk in target.astream(msgs):
            acc = chunk if acc is None else acc + chunk
            used += _total_tokens(chunk)
            if chunk.text:
                parts.append(chunk.text)
                emit({"frame": "token", "text": chunk.text})
        return acc, used

    async def agent_node(state) -> dict:
        # ⚠️ `state.get("history")` 现在是**精简版**(层 2 截短段 + 层 1 原文段,
        # T10 接线);`state["messages"]` 是累积的**完整**历史 —— 两者的分工见
        # `app/agent/state.py`。收窄成 `history` 不违和,因为精简版正是为这一轮
        # 组装出来的;**别**在这里改成读 `messages`,那会把整段历史原样塞进 prompt。
        msgs = build_messages(
            brand_name=settings.brand_name,
            history=state.get("history") or [],
            evidence=state.get("evidence") or [],
            user_input=state["resolved_input"],
        )
        parts: list[str] = []
        made: list[dict] = []
        trace: list[str] = []
        steps = 0
        usage_total = 0
        needs_final = False
        # 本轮新产生的消息(不含播种进来的历史):tool 往返 + 最终回复。
        # 交给 add_messages 并入 state,再由 log_turn 落库。
        new_messages: list = []

        for step in range(1, settings.max_agent_steps + 1):
            steps = step
            acc, used = await _stream_round(bound, msgs, parts)
            usage_total += used
            tool_calls = list(getattr(acc, "tool_calls", None) or [])

            if not tool_calls:
                needs_final = False
                msgs.append(acc)
                new_messages.append(acc)     # ← 这一轮的输出就是最终回复,收下
                break

            needs_final = True
            msgs.append(acc)
            new_messages.append(acc)         # ← 带 tool_calls 的 assistant
            for call in tool_calls:
                emit({"frame": "tool_call", "name": call["name"],
                      "args": call["args"], "tool_call_id": call["id"]})
                outcome = await execute_tool(
                    tool_call=call, registry=registry, settings=settings
                )
                emit({"frame": "tool_result", "tool_call_id": outcome.tool_call_id,
                      "ok": outcome.ok, "summary": outcome.summary})
                tool_msg = ToolMessage(content=outcome.content, tool_call_id=call["id"])
                msgs.append(tool_msg)
                new_messages.append(tool_msg)      # ← 工具结果,层 2 要截的就是它
                made.append({"name": call["name"], "ok": outcome.ok})
                trace.append(f"agent:step{step} tool={call['name']}")

            if usage_total > settings.agent_token_budget:
                break

        if needs_final:
            # 收尾:不绑 tools。预算已超也照做一次 —— 它是**唯一**能产出
            # 用户可见答复的调用,不做的话这一轮就是「有工具调用、没有回答」。
            #
            # 这一轮的输出**不在上面任何一条消息里**,必须自己收下来,
            # 否则下一轮的完整历史里**没有客服说过的话**。
            final_acc, used = await _stream_round(model, msgs, parts)
            usage_total += used
            new_messages.append(
                final_acc if final_acc is not None else AIMessage(content="".join(parts))
            )

        trace.append("agent:converged")
        # 两个键值**相同、语义不同**,别只写一个:
        #   `messages`      → add_messages 累积进完整历史(跨轮)
        #   `turn_messages` → 逐轮覆写,`log_turn` 落库的唯一依据(只写本轮)
        # 只写前者 ⇒ 落库恒空(或写重);只写后者 ⇒ 完整历史里没有本轮。
        # ⚠️ **不要**写成 `[*new_messages, AIMessage(content=reply)]` —— 无工具
        # 那一轮 `acc` 已经在 `new_messages` 里了,再补一条就是**同一句回复出现
        # 两次**,而它只在「模型一轮直接答完」时发生(带工具的轮次不会)。
        return {
            "reply": "".join(parts),
            "messages": new_messages,
            "turn_messages": new_messages,
            "agent_steps": steps,
            "tool_calls_made": made,
            "usage": {"total_tokens": usage_total},
            "trace": trace,
        }

    return agent_node


# ---- 骨架的首尾两步 ----


def make_resolve_references_node(*, model):
    """指代消解 + Query 改写:把「它能退吗」补全成「猫砂盆能退吗」(ch06)。

    一次纯文本模型调用,出参是**一句改写后的问法**,写进 `resolved_input` ——
    下游(检索、闸、Agent)一律读这个通道,所以「改写」在这里是一次性动作,
    不是给每一处各加一层。

    **失败一律原样透传**:消解是**增强**,不是必需品。ch05 的实现就是原样透传,
    所以「没消解成」的退路与本项目既有行为完全一致 —— 用户最多是少了一点上下文,
    绝不会因为改写失败而答不出话。接的是 `OutputParserException`
    (同文件 `classify_intent` 那一族「模型出参不可用」),**不是裸 `Exception`**:
    后者会把 `AttributeError` 这类**实现缺陷**也伪装成「这轮没改写」,而本仓
    已经吃过太多次「静默降级」。上游故障(超时/401/限流)同样**不接** ——
    与 `classify_intent` 一致,它们该一路抛到端点的 error 帧。
    **空输出也算失败**(模型真会吐空串):不判的话 `resolved_input` 会变成空串,
    整轮对话的输入就没了。

    它同时承担**每轮重置**:图是每请求现编译的,但 checkpointer 是**进程级**
    单例(`get_checkpointer` 的 lru_cache),而 thread_id = session_id ——
    所以同一会话的**第二轮**会带着上一轮的通道值进来。未写的通道**保留旧值**
    (LangGraph 不把未写通道重新写成默认值)。不清零,trace 帧与日志行就会把
    上一轮的 `gate_passed` / `agent_steps` 报成本轮的,而且**一路静默**:
    物流轮的 `gate_passed` 会是上一轮知识检索的结论。

    为什么放在这个节点:它是**每轮第一个**执行节点(START 的唯一出边),
    放这儿等于「每轮开头清一次」,不依赖任何调用方记得播种初值。
    **重置写在 `try` 之外** —— 消解失败时同样要清,不能整段跳过。

    `trace` 通道不在此列 —— 它是 `operator.add` 归约通道,写 `[]` 等于没写,
    清不掉;**它靠 `log_turn` 切片取当轮**(见下)。

    它还负责**把本轮的用户原话送进完整历史**(`messages`,ch07):那是这一轮
    唯一一条不由 agent / 出口节点产出的消息,而 `messages` 的**唯一**另一个写者
    是端点那一次「快照为空时」的播种 —— 不放这儿,state 里的历史就会从第二轮起
    只剩下客服说过的话(静默)。`messages` **不进**上面那份重置清单(累积语义),
    这里写的是**新增的一条**,不是重置。

    **这里返回的每个 key 都必须在 `ChatState` 里声明过**:通道集合由
    `StateGraph(ChatState)` 的注解决定,LangGraph 对未声明通道的写入是
    **静默丢弃**的(T4 的 Critical 就是它)。T7 的 `order_no` / `order_data` /
    `refund_decision` 三个通道**连同它们的清零一起**在 T7 落地(计划 PF-2)
    —— 加通道的人就是加清零的人,别把两者拆到两次改动里。
    """

    async def resolve_references(state) -> dict:
        user_input = state["user_input"]
        resolved = user_input
        try:
            result = await model.ainvoke(
                build_resolve_messages(
                    history=state.get("history") or [], user_input=user_input
                )
            )
            # `.text` 是属性(1.x;`.text()` 已废弃)。不用 `getattr` 兜底:
            # 模型出参没有 `.text` 是**替身/调用形状不对**,不是模型行为 ——
            # 替身不忠实正是 T4 那条 Critical 溜过去的原因,别再纵容。
            rewritten = result.text.strip()
            if not rewritten:
                # 真机上可达的那种失败:模型吐了空串。**不要**把空串写进
                # resolved_input —— 那等于把这一轮的输入清空。
                logger.warning("指代消解输出为空,原样透传:%r", user_input)
        except OutputParserException as exc:
            logger.warning("指代消解失败,原样透传:%s", exc)
            rewritten = ""
        if rewritten:
            resolved = rewritten

        return {
            "resolved_input": resolved,
            # 本轮的用户原话也要进**完整历史**(`messages`,累积通道)。
            #
            # **只在这里加**,不加在端点里:需求 5 的原话是「各节点只管吐**新**
            # 消息、框架自动按顺序并入」,而这正是那个形状;放进端点会让「谁负责
            # 往里加」有两处答案。也正因为它在这里,才**不会**被重复加 —— 本节点
            # 是 START 的唯一出边、每轮只跑一次,而 `resume` 路径**不重跑它**
            # (图从挂起的那个节点继续),续跑续的是**同一轮**,不该凭空多出一条。
            #
            # 少了这一行:`state["messages"]` 与 MySQL 从第二轮起**分叉** ——
            # 库里 user/assistant 成对,而 state 里只剩客服说过的话,每一轮的
            # 提问全丢。它完全静默(回复正常、落库正常),只有「拿快照派生精简版」
            # 的读者会拿到一份**没有用户提问**的上下文(spec §7.4)。
            # 用**原话**而不是 `resolved_input`:完整历史是**用户真说过什么**的
            # 记录,不是喂给模型的加工稿(同一轮 `log_turn` 落库的也是原话)。
            "messages": [HumanMessage(user_input)],
            "trace": ["resolve_references"],
            # 每轮归零的**逐轮**通道:它们描述的是「这一轮」,不是「这段会话」。
            "gate_passed": None,
            "agent_steps": 0,
            "reply": "",
            "choices": [],
            "citations": [],
            "evidence": [],
            "tool_calls_made": [],
            # ch07:`turn_messages` 是**覆写**通道,而 `log_turn` 只拿它落库 ——
            # 不重置的话「这轮没产生消息」的路径会**原样继承上一轮的值**,
            # 上一轮的 assistant 消息被再写一遍(历史里同一个回复出现两次)。
            # 它是这份清单里**唯一一个不会被节点自动覆盖**的通道,所以必须在这儿。
            # 这是 ch05–ch06「**通道与它的清零必须同处一地**」的第三次应用。
            # ⚠️ `messages` **不在此列**(它是累积通道):加进来等于每轮清空
            # 完整历史,而单轮测试完全看不出来。
            "turn_messages": [],
            # 退款子流程的三个槽位(T7 新加)。**通道与它的清零同处一地**:
            # 漏了这三行的后果是**静默串轮** —— 上一轮填过的订单号会被这一轮
            # 当成本轮槽位,用户明明没提订单号却既不弹卡片、又拿着**上一单**
            # 去判能不能退(`tests/test_agent_refund.py::
            # test_second_turn_on_same_thread_clears_refund_slots` 钉着它)。
            "order_no": "",
            "order_data": {},
            "refund_decision": None,
        }

    return resolve_references


def _lc_to_records(messages) -> list[Message]:
    """把本轮 ReAct 的 LangChain 消息转成落库用的纯数据 `Message`。

    - `AIMessage` / `AIMessageChunk` → `role="assistant"`,带 `tool_calls`(可能为空)
    - `ToolMessage` → `role="tool"`,带 `tool_call_id`

    **不复用 `prompts.to_lc_messages`**:那是**反方向**的、且本函数刻意不依赖
    LangChain 之外的东西;而 `memory/` 与 `services/` 不依赖 LangChain 是本仓的
    既有约定 —— 落库的转换因此留在这个本来就 LangChain-facing 的文件里。

    用 `isinstance(m, ToolMessage)` 分流而不是看 `role` 属性:LangChain 消息
    没有 `role` 这个字段(有的是 `type`),按"我以为的形状"取值会在真机上炸。

    `content` 取 `m.content or ""`:`AIMessage` 在只申请调用工具时 content 是
    空串/None;而 `schemas.Message.content` 是 NOT NULL 的列。

    ⚠️ `Message(role="tool")` 必须带**非空** `tool_call_id`,否则
    `schemas.Message` 的 validator 会抛(ch01 加的护栏,本章第一次真的用到)。
    ToolMessage 的这个字段由上游保证非空(`app/agent/nodes.py` 里构造时用的是
    `call["id"]`),所以这里原样透传,不做 `or ""` 那种会把畸形行写进库的兜底。
    """
    out: list[Message] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            out.append(Message(role="tool", content=m.content or "",
                               tool_call_id=m.tool_call_id))
        else:
            out.append(Message(role="assistant", content=m.content or "",
                               tool_calls=m.tool_calls or None))
    return out


def make_log_turn_node(*, session, emit):
    """日志记录:落一行结构化日志、把 trace 发成帧、把这一轮写回 MySQL。

    `trace` 是本轮**唯一**的确定性证据链:验收 1「走了强制检索节点」与
    验收 5「ReAct 不止一步」都靠它断言,而不是靠模型自由文本。

    它同时以 `trace` 帧发给端点(端点折进 `done`、不外推给前端)—— 走的是
    和 token 帧同一条 emit 通道,所以**不需要第二个 stream_mode**。

    ---- ch07:落库的内容从「user + reply」变成「user + 本轮 ReAct 往返」----

    读的是 **`turn_messages`** 而不是 `messages`:

    - `state["messages"]` 是 `add_messages` 通道,**累积的是全量**(播种进来的
      历史 + 本轮新增)。拿它落库 = 每轮把整段历史再写一遍 ⇒ **历史翻倍**,
      而每一轮的回复看起来都正常、每条单测只要不数字数就全绿。
    - `turn_messages` 是**逐轮覆写**的通道,只装本轮新产生的消息。

    **刻意不写「`turn_messages` 为空就退回用 `reply`」的兜底**:那会让「某个节点
    忘了写这个通道」**静默退化成看起来正常的旧行为**(只落一条 assistant),
    而本仓的记性里,这类兜底最后都变成了缺陷的藏身处。宁可让漏写的那轮落出
    一条**空的** assistant 行(既有那几条「落库是 user+assistant 两条」的测试
    会当场变红),也不要它自己悄悄补上。
    """

    async def log_turn(state) -> dict:
        # **只取当轮**。`trace` 是 operator.add 通道,同一 thread 的第二轮
        # 拿到的是「第一轮 + 第二轮」的拼接 —— 直接发出去,验收 1 的
        # 「trace 里有 retrieve_knowledge」会在一条**根本没检索**的物流轮上
        # 通过(上一轮留下的),这是本章最高价值证据链上的假绿通道。
        #
        # `resolve_references` 是 START 的唯一出边、每轮第一个执行,
        # 所以**最后一次**出现它就是当轮起点。找不到时(直接调 log_turn、
        # 或将来拓扑变了)退化为整段,不抛异常。
        accumulated = list(state.get("trace") or [])
        turn_trace = accumulated
        if "resolve_references" in accumulated:
            start = len(accumulated) - 1 - accumulated[::-1].index("resolve_references")
            turn_trace = accumulated[start:]
        full_trace = [*turn_trace, "log_turn"]
        logger.info(
            "chat_turn conv=%s intent=%s gate=%s steps=%s tools=%s trace=%s",
            state.get("conversation_id"), state.get("intent"), state.get("gate_passed"),
            state.get("agent_steps"),
            [t["name"] for t in (state.get("tool_calls_made") or [])],
            " > ".join(full_trace),
        )
        emit({"frame": "trace", "trace": full_trace,
              "intent": state.get("intent"), "gate_passed": state.get("gate_passed"),
              "agent_steps": state.get("agent_steps") or 0,
              # spec §4.2:confidence 要进 done 帧(端点透传,本章只用于日志)。
              # 端点那一半在 T8;**不进这一帧的话,state 里的值没有出口**。
              "confidence": state.get("confidence")})
        await append_turn(
            session=session,
            conversation_id=state["conversation_id"],
            messages=[
                Message(role="user", content=state["user_input"]),
                *_lc_to_records(state.get("turn_messages") or []),
            ],
        )
        return {"trace": ["log_turn"]}

    return log_turn
