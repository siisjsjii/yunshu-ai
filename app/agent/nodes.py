"""ch05 各节点的工厂函数。

统一用**闭包工厂**(同 `app/tools/builtin/` 的既有做法):模型、工具、
会话、检索器都不是模块级单例,而是每请求经闭包绑定 —— 这样节点既拿到了
依赖,又不会在 import 时做任何 IO。
"""

import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from app import observability
from app.agent.routing import INTENT_TO_ROUTE, OTHER
from app.agent.state import IntentResult
from app.kb.assess import record_low_confidence
from app.kb.evidence import evidence_detail
from app.memory import journal, layers
from app.memory.budget import ContextBudget
from app.memory.trim import count_tokens
from app.prompts import (
    build_context_messages,
    build_intent_messages,
    build_resolve_messages,
    render_evidence,
)
from app.retrieval.search import RetrievedChunk
from app.schemas import Message
from app.services.history import append_turn
from app.tools.executor import ERROR_CONFIRMATION_REQUIRED, execute_tool

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


def make_retrieve_knowledge_node(*, retriever, emit, settings):
    """知识类意图的**强制**预检索(确定性骨架的一步,不走 query_faq 工具)。

    检索器的故障语义原样透传:`KnowledgeRetriever` 把 Milvus/嵌入的故障翻成
    `ToolInfrastructureError`,这里**不接** —— 它必须一路抛到 API 层变 502,
    绝不能被伪装成「没搜到」。

    `citations` **同时**写进 state 并发一帧:state 那份给 Agent 组装引用编号,
    帧那份给前端渲染可点击的来源。少发帧 = ch04 的引用 UI 静默失效。

    `settings` 只为观测而收(ch09):`KnowledgeRetriever` **不是** LangChain
    run(Langfuse 的 callback 只挂在 LangChain 的 Runnable 上),所以这一段的
    span 只能手工开 —— 见 spec §3.3。「每个节点的检索结果都能铺开看」那条需求
    的**唯一**落点就是这里。**依赖走显式注入,不留会自己兜底的默认值。**
    """

    async def retrieve_knowledge(state) -> dict:
        # span 只包住**检索本身**:证据清单、citations 帧、trace 都是纯内存加工,
        # 没有可观测的东西;把它们圈进来只会让 span 的耗时读数不再是"检索花了多久"。
        #
        # ⚠️ `span` **不吞业务异常**(它只在 finally 里吞自己的 `__exit__`)——
        # 这是刻意的:`ToolInfrastructureError` 必须原样穿出去变 502。
        with observability.span(
            "retrieval", as_type="retriever",
            input={"query": state["resolved_input"]}, settings=settings,
        ) as sp:
            chunks = await retriever.search(state["resolved_input"])
            if sp is not None:
                # 只记「怎么找到的」这几个字段:id / score / section_path。
                # **不把 chunk 的正文塞进去** —— 那是知识库原文,而这条 trace
                # 会出网(spec §2.1 已记账),记 entry 的规模没有收益。
                sp.update(output={
                    "chunks": [
                        {"id": c.chunk_id, "score": round(c.score, 4),
                         "section_path": c.section_path}
                        for c in chunks
                    ]
                })
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


def _evidence_chunks(evidence: list[dict]) -> list[RetrievedChunk]:
    """通道里的证据(**dict**)→ `evidence_detail` 要的 `RetrievedChunk`。

    ⚠️ **这一步必须显式做,不能"看着像就传过去"** —— 两个形状不一样:
    `state["evidence"]` 装的是 `retrieve_knowledge` 写进去的**键值 dict**
    (`render_evidence` 与 citations 帧共用同一份),而 `app/kb/evidence.py` 读的是
    **属性 `c.score`**。把 dict 直接传过去是 `AttributeError`,而它**只在真机上炸**:
    单测若把 `RetrievedChunk` 塞进 state,那条属性访问照样绿,生产上每一次
    知识问答都 500。这正是本仓记过的「替身的形状必须等于生产的形状」。

    只投影判据真正读的那个字段(分数);其余字段填**通道里那份的同名值**,
    不另行加工 —— `question` / `answer` / `category` 在 dict 里缺键时给空串:
    它们是**另一个读者**(prompt 渲染)的输入,闸这里一个都不用。
    `score` 用 `e["score"]`(**不给默认值**):通道里没有分数说明上游写坏了,
    那时候要响亮地炸,不是当成 0 分静默拦下。
    """
    return [
        RetrievedChunk(
            question=e.get("question", ""),
            answer=e.get("answer", ""),
            category=e.get("category", ""),
            chunk_id=e.get("chunk_id") or 0,
            section_path=e.get("section_path"),
            score=e["score"],
        )
        for e in evidence
    ]


def make_confidence_gate_node(*, settings, session, conversation_id):
    """置信度闸:卡在检索之后、进 Agent 之前。

    **为什么必须在这儿**:Agent 的答复是流式吐给用户的,等答完再判就晚了
    (ch04 的自评正是那个位置,本章把它撤掉)。证据弱就直接回兜底话术、
    不进 Agent,同时把问题落池留给后面的数据飞轮。

    **判据(ch09 起)**:`app/kb/evidence.py:evidence_detail` 的三信号合成分
    (top1 / 条数 / 分差)—— 零额外模型调用,与它比的是
    `settings.evidence_confidence_threshold`(标定出来的,见 spec §4.2)。
    旧判据是「最高分 ≥ `retrieval_score_threshold`」,**单条高分就能过**;
    而"单条高分可能是巧合"正是这次要挡的那一类(spec §4.1)。
    `retrieval_score_threshold` 此后仍归**检索器内部**用(它筛块),
    闸不再读它 —— 两条链路各自一个旋钮,别再让它们互相借。
    """

    async def confidence_gate(state) -> dict:
        evidence = state.get("evidence") or []
        detail = evidence_detail(_evidence_chunks(evidence), settings=settings)
        # `bool(evidence)` **不能删**:空证据时 `detail["confidence"]` 是 0.0,
        # 而阈值被标定成 0 的话 `0.0 >= 0.0` 为真 ⇒ 空着知识进 Agent。
        passed = bool(evidence) and detail["confidence"] >= settings.evidence_confidence_threshold

        if not passed:
            # 三个信号**都写进 reason** —— 审核页旁边就是这段文字,它要回答的是
            # "为什么这条被判成答不了",不是一个分数(spec §4.3)。
            reason = (
                "检索为空"
                if not evidence
                else (
                    f"置信度 {detail['confidence']} 低于阈值 "
                    f"{settings.evidence_confidence_threshold}"
                    f"(打分:top1={detail['top1']} 条数={detail['count']} "
                    f"分差={detail['gap']})"
                )
            )
            await record_low_confidence(
                session,
                question=state["user_input"],
                source_conversation_id=conversation_id,
                entry_point="置信度闸",
                reject_reason=reason,
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


def make_agent_node(*, model, tools, registry, settings, emit, context_budget: ContextBudget):
    """主力 Agent 的 ReAct 循环。

    **不用 ToolNode / create_react_agent**:工具执行必须走 `execute_tool`,
    它承载本项目的错误语义 —— 基础设施故障抛 `ToolInfrastructureError`(→502)、
    重试用白名单(`create_ticket` 永不重试)、`ValidationError`/`ToolNotFound`
    不重试、10s 超时。`ToolNode` 直接 `tool.ainvoke`,这些语义全丢。

    停止条件是**结构保证**:循环里绑着 tools 走 `max_agent_steps` 轮;步数用尽
    或预算超限后,收尾那一轮**不绑 tools**,模型在结构上无法再调。

    `context_budget` 由端点**一次推导**后经 `build_graph` 传进来(它给
    `journal.model_ctx` 记用量与预算)。**刻意没有默认值**:有默认值的话,
    端点忘了传也能跑,而日志里那份预算就是另一个来源算的 —— 本仓的规矩是
    依赖走显式注入(`services/` 收 llm 实例同款),不留会自己兜底的参数。
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
        # ⚠️ `state.get("history")` 是**精简版**(层 2 截短段 + 层 1 原文段,
        # 由端点每轮派生并播种);`state["messages"]` 是累积的**完整**历史 ——
        # 两者的分工见 `app/agent/state.py`。这里读 `history` 不违和,因为精简版
        # 正是为这一轮组装出来的;**别**改成读 `messages`,那会把整段历史原样
        # 塞进 prompt(它是 id 不可比的 LangChain 消息,分层也无从谈起)。
        history = state.get("history") or []
        summary_text = state.get("summary_text") or ""
        evidence = state.get("evidence") or []
        msgs = build_context_messages(
            brand_name=settings.brand_name,
            history=history,
            user_input=state["resolved_input"],
            summary=summary_text,
            evidence=evidence,
        )
        # ---- ch07 §7.6:主力 Agent 每次组装完上下文,一行 `model_ctx` ----
        # **用真的用过的那对锚点重新切分**(端点播进 state),因为 state 里是
        # 扁平的精简版,而 `journal.model_ctx` 要一份 `Layers`(读 `sliding`
        # 与 `bounds`)。用过期值或 `0` 的话,那一行会**平静地描述一次没发生过的
        # 切分** —— 而没有任何断言会因此变红。
        #
        # 切分用 `layers.resplit`(**不再截短一次**):精简版里的层 2 已经是截短过的
        # 形态,再走一遍 `layers.split` 会给工具结果叠上第二个 `[工具结果] ` 前缀,
        # 于是日志与「真正发出去的那批消息」分叉 —— 而那条日志的全部价值就是这个。
        journal.model_ctx(
            conversation_id=state["conversation_id"],
            summary=summary_text,
            layers=layers.resplit(
                history,
                summary_upto_msg_id=state.get("summary_upto_msg_id") or 0,
                layer1_from_msg_id=state.get("layer1_from_msg_id") or 0,
            ),
            # 证据按**渲染后的那一段**数(它就是并进用户消息的东西),
            # 不是按条数、也不是按 `answer` 的裸长度。
            evidence_tokens=count_tokens(render_evidence(evidence)) if evidence else 0,
            budget=context_budget,
        )
        parts: list[str] = []
        made: list[dict] = []
        trace: list[str] = []
        steps = 0
        usage_total = 0
        # 本轮新产生的消息(不含播种进来的历史):tool 往返 + 最终回复。
        # 交给 add_messages 并入 state,再由 log_turn 落库。
        new_messages: list = []

        # ---- ch08:续跑判定(复用**已有的**不变量,不新增通道)----------
        # `turn_messages` 已被 ch07 放进 `resolve_references` 的每轮重置清单,
        # 所以「进场时非空」只可能是**一轮的中途**(`apply_write_decision` 刚
        # 追加了一条 tool 结果)。续跑续的是**同一轮**,不该重跑 ReAct 循环。
        existing_turn = list(state.get("turn_messages") or [])
        if existing_turn:
            msgs = msgs + existing_turn
            trace.append("agent:write_resumed")
            # **一轮不绑 tools**:结构上不可能再触发第二次写调用。
            # ⚠️ 用**未绑**的 `model`,不是 `bound`。
            final_acc, used = await _stream_round(model, msgs, parts)
            usage_total += used
            final_acc = (
                final_acc if final_acc is not None
                else AIMessage(content="".join(parts))
            )
            new_messages = existing_turn + [final_acc]
            return {
                "reply": "".join(parts),
                "messages": [final_acc],
                "turn_messages": new_messages,
                # ⚠️ **不要**写 `"agent_steps": steps` —— 续跑路径上 `steps`
                # 仍是初值 0,会把上一半算出来的步数**归零**。
                "agent_steps": state.get("agent_steps") or 0,
                "tool_calls_made": [],
                "usage": {"total_tokens": usage_total},
                "trace": trace,
            }

        needs_final = False
        pending: dict = {}

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
                # `registry` 是 `name → ToolSpec`(ch08 T4 起)—— 执行器要用
                # 登记项上的 `kind` 推权限与重试,`input_schema` 做校验前置。
                # `write_decision` 走默认的 `pending`:Agent 这条正常路径从不传,
                # 写调用因此**停在这里**(不执行),由确认流决定下一步。
                #
                # ⚠️ **这里刻意不开手工 span**(ch09 T3 订正,spec §15.4)。
                # spec §3.3 原先断言「工具执行一个 span 都不会自动出现」,真机实测
                # **只对了一半**:`execute_tool` 最后落到 `spec.tool.ainvoke`,
                # **那是一个 LangChain run** ⇒ 回调**已经**给了一次
                # `TOOL '<name>'`,而且嵌套正确(在 `agent` 之下)。
                # 再手工开一条 `tool:<name>` 就是**同一个事件表示两遍**
                # (Langfuse 自己的最佳实践原话:Don't emit duplicate
                # dispatch + execution nodes)⇒ 只保留自动那一条。
                outcome = await execute_tool(
                    tool_call=call, registry=registry, settings=settings,
                    conversation_id=state["conversation_id"],
                )
                if outcome.error_kind == ERROR_CONFIRMATION_REQUIRED:
                    # 写操作待确认:那次调用**根本没发生** ⇒ 不回灌 tool 结果,
                    # 由 `apply_write_decision` 在决议之后补上。
                    if not pending:
                        pending = {
                            "tool_call_id": call["id"],
                            "name": call["name"],
                            "args": call["args"],
                            "preview": outcome.preview or dict(call["args"]),
                        }
                        trace.append(f"agent:write_pending tool={call['name']}")
                    else:
                        # 同一轮里的**第二个**待确认写调用:它不会有第二次
                        # confirm 机会,但**必须**补一条 tool 结果 ——
                        # 少回灌一个就构成「有 tool_calls 没有对应 tool 消息」,
                        # 上游直接 400(CLAUDE.md 的硬约束)。
                        stub = ToolMessage(
                            content="本轮已有一个写操作待用户确认,本次未执行。",
                            tool_call_id=call["id"],
                        )
                        msgs.append(stub)
                        new_messages.append(stub)
                    continue
                emit({"frame": "tool_result", "tool_call_id": outcome.tool_call_id,
                      "ok": outcome.ok, "summary": outcome.summary})
                tool_msg = ToolMessage(content=outcome.content, tool_call_id=call["id"])
                msgs.append(tool_msg)
                new_messages.append(tool_msg)      # ← 工具结果,层 2 要截的就是它
                made.append({"name": call["name"], "ok": outcome.ok})
                trace.append(f"agent:step{step} tool={call['name']}")

            if pending:
                # 停循环:交给 `confirm_write` → `apply_write_decision` → 回来续跑。
                # **不发收尾那一轮** —— 用户还没确认,现在就作答等于先把话说死。
                return {
                    "reply": "".join(parts),
                    "messages": new_messages,
                    "turn_messages": new_messages,
                    "agent_steps": steps,
                    "tool_calls_made": made,
                    "pending_write": pending,
                    "usage": {"total_tokens": usage_total},
                    "trace": trace,
                }

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


def route_after_agent(state) -> str:
    """`agent` 之后往哪走。

    **判据是 `pending_write` 非空** —— 那条路径上 `agent` 已经停循环、
    没有发收尾那一轮;其余一律照旧汇进 `log_turn`。
    """
    return "confirm_write" if state.get("pending_write") else "log_turn"


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
            # ⚠️ `messages` **不在此列**(它是累积通道):这份清单是给
            # 「没有播种者、只能靠重置」的通道用的,而 `messages` 每轮都有人吐新
            # 消息进去。**更正一处早先写错机制的注释**:加进来**并不会**清空历史
            # —— `add_messages(left, [])` 实测**原样返回 `left`**(空列表更新在
            # append-only reducer 上是 no-op)。决定不变,但理由是「**不必要**」,
            # 不是「危险」。ch07 新增的 `summary_text` 与两个锚点同样不在此列:
            # 它们每轮由端点播种(见 `app/agent/state.py`)。
            "turn_messages": [],
            # 退款子流程的三个槽位(T7 新加)。**通道与它的清零同处一地**:
            # 漏了这三行的后果是**静默串轮** —— 上一轮填过的订单号会被这一轮
            # 当成本轮槽位,用户明明没提订单号却既不弹卡片、又拿着**上一单**
            # 去判能不能退(`tests/test_agent_refund.py::
            # test_second_turn_on_same_thread_clears_refund_slots` 钉着它)。
            "order_no": "",
            "order_data": {},
            "refund_decision": None,
            # ch08:建工单确认流的两个槽位。**同处一地**(见 state.py 的说明)。
            "pending_write": {},
            "write_decision": "",
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
