import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, format_sse_event
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.emit import make_emitter
from app.agent.graph import build_graph, get_checkpointer
from app.config import Settings, get_settings
from app.db.session import get_session
from app.llm import create_chat_model, create_extract_model
from app.memory import budget, journal, layers, summarize
from app.memory.store import SessionStore
from app.memory.tasks import log_trigger, run_summary_in_background
from app.memory.trim import ContextOverflowError
from app.prompts import render_system_prompt, select_layer1, to_lc_messages
from app.sanitize import redact_api_key
from app.schemas import ChatRequest, TicketRequest
from app.services.chat import prepare_turn
from app.services.history import (
    advance_anchors,
    ensure_conversation,
    load_history,
    load_summaries,
)
from app.tools.errors import ToolInfrastructureError
from app.tools.executor import execute_tool
from app.tools.registry import build_retriever, build_tools, registry_for

router = APIRouter()

logger = logging.getLogger(__name__)

_store: SessionStore | None = None


def get_store(settings: Settings = Depends(get_settings)) -> SessionStore:
    """进程内单例。测试通过 dependency_overrides 替换。"""
    global _store
    if _store is None:
        _store = SessionStore(
            ttl_seconds=settings.session_ttl_seconds,
            max_sessions=settings.max_sessions,
        )
    return _store


def get_chat_model(settings: Settings = Depends(get_settings)):
    return create_chat_model(settings)


def get_intent_model(settings: Settings = Depends(get_settings)):
    """意图识别用的模型。与 `get_chat_model` **并排**做成依赖,不是为了让生产
    代码多一层 —— 是为了给测试留一个 `dependency_overrides` 的注入点。

    意图节点**每一个请求都会跑**,也就每一个请求都会真的 `.ainvoke` 一次。
    若这里就地 `create_extract_model(settings)`(模块级导入的名字),测试想拦
    它就只能去 monkeypatch `chat_api.create_extract_model` —— 而那是**导入进来的
    名字**,patch 它等于 patch 整个模块的绑定,别的端点(抽取)同用一个名时会
    连带被换掉。`Depends` 是本项目既有的注入缝(CLAUDE.md:services 收 llm 实例
    作参数、FastAPI 侧靠 Depends 注入、测试用 dependency_overrides 替换),
    `tests/test_api_chat.py` 的 `client_factory` 里已经有一行
    `app.dependency_overrides[chat_api.get_chat_model] = lambda: model`,
    加一行同形的即可。**不做这层,凡是进入端点/图的用例都会朝
    `https://example.invalid/v1` 发真实请求,而不联网这一条是硬规矩。**
    """
    return create_extract_model(settings)


#: 「resume 打在一个没有挂起流程的会话上」时给客户端的固定文案。
#:
#: **不得出现任何 Python 标识符**:这条路径的默认结局(不校验就放它进流)是
#: 图带着 `Command` 从 START 重开 → `resolve_references` 取不到 `user_input`
#: → `KeyError` → 端点那句 `str(exc)` 把它变成 HTTP 200 的 error 帧,
#: **用户看到的是一句裸的 `'user_input'`**(实测)。既没有可操作性,
#: 也不是服务端故障该有的样子。
RESUME_WITHOUT_PENDING = "该会话没有待处理的流程,请直接发送消息开始新一轮。"


def _frame(event: str, payload: dict) -> bytes:
    return format_sse_event(
        event=event,
        data_str=json.dumps(payload, ensure_ascii=False),
    )


@router.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    store: SessionStore = Depends(get_store),
    model=Depends(get_chat_model),
    intent_model=Depends(get_intent_model),
    session: AsyncSession = Depends(get_session),
) -> EventSourceResponse:
    session_id = request.session_id or uuid.uuid4().hex
    user_id = request.user_id or "demo-user"

    # `lock_for` 与 `acquire` 之间**不得插入 await**:lock_for 返回的锁可能
    # 被紧随其后的容量淘汰摘掉,而这中间一旦让出控制权,那个窗口就可达了
    # (window 眼下靠"这两个调用之间没有 await"关着)。
    lock = store.lock_for(session_id)
    try:
        await asyncio.wait_for(
            lock.acquire(), timeout=settings.session_lock_timeout_seconds
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=409, detail="该会话正在处理另一条消息,请稍后重试"
        ) from exc

    # 预算校验必须在响应开始前完成 —— 一旦开始流式就改不了状态码。
    # 会话读取与历史加载也在锁内:同会话并发时不会读到半轮历史。
    try:
        # ensure_conversation 必须在**持锁之后**。它在锁外有个建会话竞态:
        # 两个并发的首请求都看不到行,其中一个 INSERT 撞主键抛 IntegrityError。
        # 持锁跨过"检查 + 插入"把窗口关掉 —— 这看着像偶然细节,不是。
        # 返回值**必须留着**:降级与分层要读会话上那两个已落库的锚点。
        # (`test_ensure_conversation_runs_under_the_lock` 的替身也因此必须返回
        #  生产形状的对象 —— 见那条用例的说明。)
        conv = await ensure_conversation(
            session=session, session_id=session_id, user_id=user_id
        )
        # `resume` 分支**不推导预算**:它不组装上下文(从 checkpoint 还原)、
        # 也不进 Agent 节点,推导出来没有读者。`build_graph` 拿到 None 时现算
        # 一份(纯函数,同一个值),所以那条路上也不会缺。
        context_budget = None
        if request.resume is None:
            # **全量**历史(带 MySQL 主键)。分层按 `messages.id` 切,所以它必须
            # 是**整段**、**未经裁剪**的 —— 裁剪过的历史会让层 2 少算,
            # 而层 2 的截短后计数正是摘要的触发判据(见下面那段)。
            history = await load_history(session=session, conversation_id=session_id)
            # 预算:本请求**只推导一次**,往下传给 `prepare_turn`(两条 400 判据)
            # 与 `build_graph`(agent 节点打 `model_ctx` 要用同一份)。
            context_budget = budget.derive(
                settings=settings, system_prompt=render_system_prompt(settings.brand_name)
            )
            # 流开始前的两条 400 守卫(本轮输入超限 / 历史预算装不下一轮)在这里
            # 完成 —— 一旦 yield 过首帧,状态码就改不了了。
            #
            # 它是**纯校验、不碰历史**:ch07 起历史由下面的 `layers` 分层派生
            # (spec §3),而旧的单层裁剪(`trim.select_history`,按**原文** token
            # 整轮丢弃)与它**不能叠加** —— 叠在最前面会把层 2 的整轮消息先丢掉,
            # `layer2_tokens` 随之少算,**摘要会在该触发的时候不触发**(而且每轮
            # 丢得越来越顺手),声明的「级联」在日志里永远走不到第二环,
            # 没有任何东西报错。所以那个函数连同这条调用一起删了。
            prepare_turn(
                settings=settings, budget=context_budget, user_input=request.message
            )

            # 请求**开始时**的锚点,先取到局部变量里。
            #
            # ⚠️ 这不是风格:下面那句 `advance_anchors` 是 `session.execute(update(...))`,
            # SQLAlchemy 默认的 `synchronize_session="auto"` 会**把同 session 里
            # 那个 `Conversation` ORM 对象一起改掉**(本项目记账过的身份映射行为)。
            # 所以写库之后再读 `conv.layer1_from_msg_id` 拿到的是**新值** ——
            # 日志里的 `from` 会与 `to` 相等(实测:`from: 23, to: 23`,一个看起来
            # 完全正常的「降级没挪」读法,而它恰恰是验收 2 要 grep 的那一行)。
            # 旧值与新值只有这一个接缝上同时有,取早一步是唯一的取法。
            old_layer1_from = conv.layer1_from_msg_id

            # ---- 降级:层 1 的原文超预算就把边界往后挪(只挪 id,不搬数据)----
            # 挪过的那几轮**自动落进层 2**,下一轮以截短形态出现(spec §3.3)。
            layer1_from = layers.degrade(
                history,
                summary_upto_msg_id=conv.summary_upto_msg_id,
                layer1_from_msg_id=old_layer1_from,
                layer1_budget=context_budget.layer1_budget,
                settings=settings,
            )
            # ---- 分层 + 截短 ----
            got = layers.split(
                history,
                summary_upto_msg_id=conv.summary_upto_msg_id,
                layer1_from_msg_id=layer1_from,
                settings=settings,
            )

            if layer1_from != old_layer1_from:
                # 只在**真的动了**的时候写库:每次请求都写一遍会让
                # 「降级发生了没有」在 DB 层看不出来,也白一次 commit。
                #
                # **级联的第一环就打在**这里**(spec §10.5 验收 2 要 grep 的那行)**:
                # 两个值(旧/新)只有这一个接缝上同时有 —— `layers.degrade` 只返回
                # 新值,`advance_anchors` 只收新值。少这一行,验收 2 的
                # 「层 1 超预算就降级一批」在日志里**根本没有生产者**,
                # 而那条断言要么失败、要么在验收脚本里被写成一条恒真的 grep。
                #
                # 格式与 `journal` / `memory.tasks` 同款(前缀 + JSON、`ensure_ascii=False`)——
                # 另起一种日志形状等于让读日志的人多学一套。
                # `layer1_tokens` / `layer1_budget` 一起报:只报「挪了」而不报
                # 「为什么挪」,读的人还得自己去推(本仓「日志里的数必须是真的」)。
                #
                # ⚠️ **必须打在 `advance_anchors` 之后**(终审 Important 3):这行
                # 是全分支**唯一一条「真假取决于后面那句成不成」的日志**。写库抛了
                # 而日志已经落下 ⇒ 它声称的是一次**没有发生过**的降级,而验收 2
                # 正是 grep 这一行 ⇒ 一个失败请求会被读成「级联的第一环跑过了」。
                # 顺序反过来之后,这行只在**锚点真的推进了**之后才存在。
                #
                # 旧值因此必须来自上面那个局部变量、**不能**现读 `conv` ——
                # 这条 UPDATE 已经把会话对象上的锚点改成新值了(见 `old_layer1_from`
                # 那段的说明;实测现读会打出 `from == to`)。
                await advance_anchors(
                    session=session, conversation_id=session_id, layer1_from=layer1_from
                )
                logger.info(
                    "layer1 降级 %s",
                    json.dumps(
                        {
                            "conversation_id": session_id,
                            "from": old_layer1_from,
                            "to": layer1_from,
                            # 挪**之后**的层 1 用量:它就是「装下了」的证据
                            # (所以必然 <= budget)。
                            "layer1_tokens": got.layer1_tokens,
                            "layer1_budget": context_budget.layer1_budget,
                        },
                        ensure_ascii=False,
                    ),
                )

            # ---- 摘要任务:起在后台,**不 await** ----
            # 它压的是**更早**的一段历史,与这一轮的回复无关,所以可以晚、也可以
            # 失败(失败等于什么都没发生,下一轮再触发)。`trigger` 那行日志只能
            # 在这里打:只有这个接缝同时拿着层 2 的用量与预算(T9 拿不到)。
            if summarize.should_summarize(got, layer2_budget=context_budget.layer2_budget):
                log_trigger(
                    conversation_id=session_id,
                    layer2_tokens=got.layer2_tokens,
                    layer2_budget=context_budget.layer2_budget,
                )
                run_summary_in_background(
                    conversation_id=session_id,
                    settings=settings,
                    model_factory=create_extract_model,
                )   # 不 await
            # ---- 组装精简版(层 2 截短段 + 层 1 原文段)----
            # 层 1 再过一道 `trim_messages`(`prompts.select_layer1`,本仓
            # LangChain 的唯一面):`degrade` 是按**整轮**收敛的,而它有一条
            # 「只剩一轮还超预算就停在原地」的出口 —— 那一条留给这里收口,
            # 宁可少发一段上下文,也不把窗口顶穿。
            trimmed = got.layer2 + select_layer1(
                got.layer1, max_tokens=context_budget.layer1_budget
            )
            # 梗概在**起任务之后**读,与 spec §7.6 的接线顺序一致。理论上这中间
            # 有一个窗口:后台任务(几秒的模型往返)若赶在这次读之前落了库,
            # 这一轮就会同时看到层 2 的原文与它的梗概。**这个窗口无害**:
            # 梗概覆盖的正是本轮展示的那段层 2,而下一轮锚点已经推过去、
            # 层 2 为空,重复自然消失。(真要让本轮完全自洽,把这次读提到
            # 起任务之前即可 —— 代价是与 §7.6 的顺序不一致,收益为零。)
            summaries = await load_summaries(
                session=session, conversation_id=session_id
            )
            # 多段梗概拼成**一段背景**(不是逐段清单);空列表 → 空串。
            summary_text = summarize.join_summaries(summaries)

            # ch07 §7.6:指代消解 / 意图识别共用的那份上下文,**每轮必打**。
            # 位置有两重意义:① 在 `resolve_references` **之前**(它就是这个
            # 上下文最早的两个消费者);② 在**路由之前** —— 闲聊/投诉/兜底/
            # 退款子流程那几轮不进 Agent,而它们恰恰最容易「看起来正常、
            # 其实上下文是错的」。写进 `generate()` 里就漏掉整个这一类。
            #
            # `resume` 分支**不打**:续跑不是新的一轮(spec §5.1),它从
            # checkpoint 还原上下文,`resolve_references` 也不会重跑 ——
            # 这里再打一行会是一条与事实不符的日志。
            #
            # `history=` 用**分层后的精简版**(T10 的收口):4b 要在这行里看见
            # `…` 与 `[工具结果] `,而那两个标记只有 `layers.truncate` 产得出来
            # —— 早先这条线拿的是单层裁剪的输出(`trim.select_history`,T10 已删),
            # 它**只整轮丢弃、从不标注内容**,在结构上承载不了 4b。
            #
            # 这一行**不带锚点**:`history_ctx` 收的是扁平滑窗、不是「用两个锚点
            # 切出来的三层」,给它补一对锚点只能是编的(spec §7.6 的字段表里
            # 也没有 `bounds`;`model_ctx` 才有,它的 `Layers` 自带锚点)。
            journal.history_ctx(
                conversation_id=session_id,
                summaries=summaries,
                history=trimmed,
                budget=context_budget,
            )
        # 工具集与注册表**同源**:绑给模型的与能执行的必须是同一批对象。
        # 两处各取一份时,模型会"看得到却执行不到",退化成一条 ok=false 的
        # 可恢复失败 —— 事件序列长得一模一样,只是永远查不出东西。
        #
        # 这两行也必须在守卫之内。make_query_faq / make_create_ticket 会做
        # 导入、建闭包,ch03 起 build_tools 还在里面组装检索器
        # (build_retriever:读配置 + 取懒加载的 embedder/Milvus 客户端单例),
        # 都是"拿到锁之后"这段里的新代码;它们抛异常时漏放锁的后果不是"慢"
        # —— 持锁的锁既不被 TTL 也不被 LRU 回收,该会话从此永久 409,
        # 症状与"泄漏"毫无相似之处。
        tools = build_tools(session=session, conversation_id=session_id)
        registry = registry_for(tools)

        # emit 必须在图**之外**创建、在节点里才被调用 —— make_emitter 返回的
        # 是可重复调用的函数,每次发帧时现取 writer(见 app/agent/emit.py)。
        emit = make_emitter()
        # 图在这里(而不是在 generate 里)组装:构造 `StateGraph` 与 `compile`
        # 是纯内存操作、不做 IO,而**续跑的待续检查要用它 `aget_state`** ——
        # 那个检查必须发生在响应开始之前(理由见下面那一段)。
        graph = build_graph(
            model=model,
            intent_model=intent_model,
            tools=tools,
            registry=registry,
            settings=settings,
            retriever=build_retriever(session),
            session=session,
            conversation_id=session_id,
            emit=emit,
            checkpointer=get_checkpointer(),
            context_budget=context_budget,
        )

        # 这一线程当前的 state —— **两个分支都要用**,所以在分支之前取一次。
        # (ch06 起它只服务 `resume` 那条待续检查;ch07 起非续跑的那条也要它
        #  决定「要不要播种」。提到分支之前 = 每请求仍然只取**一次**快照,
        #  代价是非续跑的那条路径多了一次内存读。)
        snapshot = await graph.aget_state(
            {"configurable": {"thread_id": session_id}}
        )
        # 播种判据。`messages` 从没被写过时是**空列表**(`add_messages` 通道的
        # 初值就是 `[]`),不是 None —— `or []` 兜的是「这个 thread 还没有
        # checkpoint」那种返回空 dict 的情形。
        #
        # ⚠️ 取的是 `snapshot.values`,**不是** `snapshot.next` —— 后者是
        # 「待续节点名」,与 state 内容无关,取错了**恒得空列表** ⇒ 每轮都播种
        # ⇒ 整段历史被重复追加,而每一轮的回复看起来都正常。
        seeded = list(snapshot.values.get("messages") or [])

        if request.resume is not None:
            # **续跑不是新的一轮**(ch06,spec §5.1),而且**必须先确认真的有待续
            # 任务**。两件事都在这里说清:
            #
            # ① 待续检查。没有待续任务时放它进流,结局实测是:图带着 `Command`
            #    从 START 重开 → `resolve_references` 取不到 `user_input` →
            #    `KeyError` → 端点那句 `str(exc)` 把它变成 **HTTP 200 的 error
            #    帧**,用户看到一句裸的 `'user_input'`。可达场景:服务重启
            #    (InMemorySaver 是进程内的)之后,用户点一张还挂在页面上的旧卡片
            #    —— 页面把 sessionId 留在内存里,这次点击照样发得出来。
            #    检查放在这里,是因为**这里还改得动状态码**(一旦 yield 过首帧
            #    就再也不能):与预算校验同一个窗口。
            #
            # ② 不读历史、不跑 `prepare_turn`。`Command(resume=…)` 只把 resume 值
            #    交回挂起的那个节点,**不会**把输入合并进 state —— 历史、user_input、
            #    槽位全都从 checkpointer 的断点里恢复,读出来没有读者。而
            #    `prepare_turn` 守的是「这**新的一轮**」的两条 400(本轮输入长度 /
            #    历史预算装不下一轮)—— 续跑不是新一轮,它既不新增用户输入、也不重组
            #    上下文,那两条判据在这里**没有对象**。
            #
            # ③ **不播种**(ch07)。续跑续的是**同一轮**,上下文从 checkpoint
            #    还原即可;这里递 `messages` 会被并进 state —— 而续跑的那一轮
            #    本来就会把它自己那批消息再写一遍。
            if not snapshot.next:
                # 409 而不是 422:请求体本身**完全合约定**(spec §5.1 的形状),
                # 冲突的是**这个会话的状态**(没有待续流程)—— 与本文件上面那条
                # 「该会话正在处理另一条消息」同一族。422 在本仓专指"请求本身
                # 不合约定"(类目不在固定集、缺必填字段)。
                raise HTTPException(
                    status_code=409,
                    detail=redact_api_key(
                        RESUME_WITHOUT_PENDING, settings.openai_api_key
                    ),
                )
            stream_input = Command(resume=request.resume)
        else:
            stream_input = {
                "conversation_id": session_id,
                "user_input": request.message,
                # 精简版 = 层 2 截短段 + 层 1 原文段(spec §7.5)。它**不是**
                # 上面读出来的那份全量历史 —— 全量是分层的**输入**,
                # 这一份才是发给模型的东西(见上面那段注释)。
                "history": trimmed,
                # 梗概全文(`join_summaries` 把多段拼成一段);空串 = 还没有梗概。
                "summary_text": summary_text,
                # 这一轮**真的用过**的那对锚点。agent 节点靠它们把扁平的精简版
                # 重新分类成 `Layers`,好让 `model_ctx` 那一行说得准被切的是
                # 哪一段。**两个通道必须在 `ChatState` 里声明过** ——
                # LangGraph 对未声明通道的写入是静默丢弃的(ch06 的教训)。
                "summary_upto_msg_id": conv.summary_upto_msg_id,
                "layer1_from_msg_id": layer1_from,     # 降级**之后**的那个
                "trace": [],
            }
            # **只在 state 没有 messages 时播种**(spec §7.4)。`add_messages` 是
            # append-only:每轮无条件播种会把整段历史重复追加,而每一轮的回复
            # 看起来都**完全正常**(帧、落库、状态码一个都不变)。
            #
            # 两道防线缺一不可:① 这个判据;② `to_lc_messages` 给每条带上的
            # MySQL 主键(稳定 id)—— 同一个 id 再次并入是**替换**而不是追加,
            # 所以「重新播种同一批消息」是幂等的。只有 ① 时,播种在「state 非空
            # 但库里有更多行」时仍会追加(agent 本轮写的消息没有 MySQL id);
            # 只有 ② 时,每轮都要白读一次全量历史。
            #
            # 服务重启自愈:`InMemorySaver` 是**进程内**的,重启后 checkpoint 全空
            # ⇒ 下一次请求 `messages` 为空 ⇒ 自动从 MySQL 重新播种(spec §7.4)。
            # 这就是「MySQL 是权威源」在代码上的落点。
            if not seeded:
                stream_input["messages"] = to_lc_messages(
                    await load_history(session=session, conversation_id=session_id)
                )
    except ContextOverflowError as exc:
        # 两条 400(本轮输入超限 / 历史预算装不下一轮)都从这里出去。文案由
        # `trim` 里那两个类给出(不含任何 Python 标识符,也不含用户文本),
        # 仍然过一遍脱敏 —— 出站文本一律过 `redact_api_key` 是本章不新开例外的规矩。
        lock.release()
        raise HTTPException(
            status_code=400,
            detail=redact_api_key(str(exc), settings.openai_api_key),
        ) from exc
    except BaseException:
        # 拿到锁之后,凡是不返回 EventSourceResponse 的退出都必须释放锁,
        # 否则该会话会永久 409(lock_for 一直返回同一把被持锁)。
        # 用 BaseException 而非 Exception 兜住一切 —— 客户端在响应开始前
        # 断开会让 Starlette 取消端点任务,抛出的 asyncio.CancelledError
        # 是 BaseException 的子类,不是 Exception。
        lock.release()
        raise

    async def generate():
        try:
            yield _frame("meta", {"session_id": session_id, "model": settings.openai_model})

            final = {"trace": [], "intent": None, "gate_passed": None, "agent_steps": 0}
            suspended = False
            # `stream_mode` **必须带上 `updates`**(ch06,spec F1)。实测
            # (langgraph 1.2.11):只给 `custom` 时 `interrupt()` 被**整个吞掉**
            # —— run 照常结束、`state.next` 停在待续节点、一个帧不吐、不报任何错。
            # 用户侧的表现是"问退款之后什么都没发生",连报错都没有。
            #
            # `updates` 只用于**认 interrupt**,其余一律不外推:那是图的原始
            # update 载荷(里面是节点返回值,可能含模型自由文本),前端不认识它。
            async for mode, chunk in graph.astream(
                stream_input,
                config={"configurable": {"thread_id": session_id}},
                stream_mode=["custom", "updates"],
            ):
                if mode == "updates":
                    if "__interrupt__" in chunk:
                        # 挂起:载荷原样是 `{"frame": "order_choice", "options": [...]}`
                        # (spec §5.2),帧名与载荷由它自己说 —— 端点只做搬运,不认
                        # "order_choice" 这个字面量(将来多一种挂起,这里不用改)。
                        value = chunk["__interrupt__"][0].value
                        if not isinstance(value, dict):
                            # 下面两句要 `payload.get(...)` / `payload.items()`,
                            # 非 dict 会变成一句 `AttributeError` 的 error 帧。
                            # **只有我们自己的节点会 `interrupt(...)`,它们一律给
                            # dict**(见 `app/agent/refund_nodes.py`),所以走到这里
                            # 是接线/实现 bug —— 响亮地抛,别让它长成
                            # 「用户看不懂、我们也没法查」的样子。
                            raise TypeError(
                                f"interrupt 载荷必须是 dict,收到 {type(value).__name__}"
                            )
                        suspended = True
                        yield _frame(
                            value.get("frame", "interrupt"),
                            {k: v for k, v in value.items() if k != "frame"},
                        )
                    continue

                payload = chunk
                event = payload.get("frame")
                if event == "trace":
                    # 内部证据链:折进 done 帧,不外推 —— 前端不认识这个帧。
                    final = payload
                    continue
                data = {k: v for k, v in payload.items() if k != "frame"}
                if event == "tool_result" and not data.get("ok", True):
                    data["summary"] = redact_api_key(
                        data.get("summary", ""), settings.openai_api_key
                    )
                yield _frame(event, data)

            if suspended:
                # 挂起的一轮**不发 done**:done 帧自报的是"这一轮跑完了"(它带
                # trace / intent / agent_steps),而挂起时这些全是初值 —— 发出去
                # 是在撒谎,而且 `log_turn` 也没跑(这一轮不落库,spec §5.1)。
                # 前端不读 done 帧,响应结束即恢复输入框。
                return

            yield _frame("done", {
                "finish_reason": "stop",
                # `usage` **刻意写死 None**:`ChatState.usage` 只有 Agent 节点写,
                # 而它**不在** `resolve_references` 的每轮重置清单里 —— 一旦把
                # `state["usage"]` 接到这里,非 Agent 的那几轮(闲聊/投诉/兜底/
                # 退款子流程)就会报**上一轮的 token 数**。今天没有任何读者
                # (前端不读、验收脚本不读),所以先留死值;真要接,必须连
                # 「在每轮重置里把 usage 清掉」一起做。
                "usage": None,
                "trace": final.get("trace") or [],
                "intent": final.get("intent"),
                "confidence": final.get("confidence"),
                "gate_passed": final.get("gate_passed"),
                "agent_steps": final.get("agent_steps") or 0,
            })
        except Exception as exc:
            # 上游异常文本可能带着密钥(见 app/sanitize.py),出站前抹掉。
            yield _frame(
                "error",
                {"message": redact_api_key(str(exc), settings.openai_api_key)},
            )
        finally:
            lock.release()

    # 这一句刻意留在守卫之外:`EventSourceResponse(...)` 只是构造一个对象,
    # 不做 IO,也不启动生成器(generate 的第一个 yield 发生在响应发送时,
    # 那时锁的释放由它自己的 finally 负责)。为它把 `return` 包进 try 换不到
    # 什么,只会让"哪些退出路径必须放锁"这条规则变得含糊。
    return EventSourceResponse(
        generate(),
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/ticket")
async def create_ticket_endpoint(
    request: TicketRequest,
    settings: Settings = Depends(get_settings),
    store: SessionStore = Depends(get_store),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """建工单:「建工单」按钮的后端入口。

    为什么需要它:`create_ticket` 是**模型工具**,只能由模型在对话里调;
    而按钮点击是 HTTP 请求,够不到模型工具。不加这个端点,验收 3 的
    「点建工单写 tickets 表」无法达成(spec §8.2)。

    护栏与对话端点一致:同一把会话锁串行化;`create_ticket` 是非幂等写操作,
    executor 的重试白名单不含它,**永不重试**。
    """
    lock = store.lock_for(request.session_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=settings.session_lock_timeout_seconds)
    except TimeoutError as exc:
        raise HTTPException(status_code=409, detail="该会话正在处理另一条消息,请稍后重试") from exc

    try:
        await ensure_conversation(
            session=session, session_id=request.session_id, user_id="demo-user"
        )
        tools = build_tools(session=session, conversation_id=request.session_id)
        registry = registry_for(tools)
        outcome = await execute_tool(
            tool_call={
                "name": "create_ticket",
                "args": {"description": "用户在会话中主动点击「建工单」", "ticket_type": "其他"},
                "id": "manual-ticket",
                "type": "tool_call",
            },
            registry=registry,
            settings=settings,
        )
        if not outcome.ok:
            raise HTTPException(
                status_code=502,
                detail=redact_api_key(outcome.summary, settings.openai_api_key),
            )
        return json.loads(outcome.content)
    except ToolInfrastructureError as exc:
        # 基础设施故障(DB 不可用等)必须变 502 + 固定文案,**不是** FastAPI 默认的
        # 500 —— 500 会把「服务端出问题」说成「你的请求有问题」,与本仓既定的
        # 错误语义边界不一致(CLAUDE.md:上游/基础设施故障一律 502)。
        # executor 抛出的文本已经是固定文案,这里再过一次 redact_api_key 是
        # 纵深防御:它是出站文本,而出站文本一律要过脱敏。
        raise HTTPException(
            status_code=502,
            detail=redact_api_key(str(exc), settings.openai_api_key),
        ) from exc
    finally:
        lock.release()
