"""ch05 各节点的工厂函数。

统一用**闭包工厂**(同 `app/tools/business.py` 的既有做法):模型、工具、
会话、检索器都不是模块级单例,而是每请求经闭包绑定 —— 这样节点既拿到了
依赖,又不会在 import 时做任何 IO。
"""

import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import ToolMessage
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
        return {"reply": CHITCHAT_REPLY, "choices": [], "trace": ["chitchat_reply"]}

    return chitchat_reply


def make_fallback_reply_node(*, emit):
    """兜底:意图不属于七类、或分类解析失败时走这里(用户明确要求)。

    也是置信度闸不通过时的落点 —— 那时问题已由 confidence_gate 落进低置信度池。
    """

    async def fallback_reply(state) -> dict:
        emit({"frame": "token", "text": FALLBACK_REPLY})
        return {"reply": FALLBACK_REPLY, "choices": [], "trace": ["fallback_reply"]}

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

        for step in range(1, settings.max_agent_steps + 1):
            steps = step
            acc, used = await _stream_round(bound, msgs, parts)
            usage_total += used
            tool_calls = list(getattr(acc, "tool_calls", None) or [])

            if not tool_calls:
                needs_final = False
                break

            needs_final = True
            msgs.append(acc)
            for call in tool_calls:
                emit({"frame": "tool_call", "name": call["name"],
                      "args": call["args"], "tool_call_id": call["id"]})
                outcome = await execute_tool(
                    tool_call=call, registry=registry, settings=settings
                )
                emit({"frame": "tool_result", "tool_call_id": outcome.tool_call_id,
                      "ok": outcome.ok, "summary": outcome.summary})
                msgs.append(ToolMessage(content=outcome.content, tool_call_id=call["id"]))
                made.append({"name": call["name"], "ok": outcome.ok})
                trace.append(f"agent:step{step} tool={call['name']}")

            if usage_total > settings.agent_token_budget:
                break

        if needs_final:
            # 收尾:不绑 tools。预算已超也照做一次 —— 它是**唯一**能产出
            # 用户可见答复的调用,不做的话这一轮就是「有工具调用、没有回答」。
            _, used = await _stream_round(model, msgs, parts)
            usage_total += used

        trace.append("agent:converged")
        return {
            "reply": "".join(parts),
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
            "trace": ["resolve_references"],
            # 每轮归零的**逐轮**通道:它们描述的是「这一轮」,不是「这段会话」。
            "gate_passed": None,
            "agent_steps": 0,
            "reply": "",
            "choices": [],
            "citations": [],
            "evidence": [],
            "tool_calls_made": [],
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


def make_log_turn_node(*, session, emit):
    """日志记录:落一行结构化日志、把 trace 发成帧、把这一轮写回 MySQL。

    `trace` 是本轮**唯一**的确定性证据链:验收 1「走了强制检索节点」与
    验收 5「ReAct 不止一步」都靠它断言,而不是靠模型自由文本。

    它同时以 `trace` 帧发给端点(端点折进 `done`、不外推给前端)—— 走的是
    和 token 帧同一条 emit 通道,所以**不需要第二个 stream_mode**。
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
                Message(role="assistant", content=state.get("reply") or ""),
            ],
        )
        return {"trace": ["log_turn"]}

    return log_turn
