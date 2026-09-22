import asyncio
import json
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
from app.memory import budget, journal
from app.memory.store import SessionStore
from app.memory.trim import ContextOverflowError
from app.prompts import render_system_prompt, to_lc_messages
from app.sanitize import redact_api_key
from app.schemas import ChatRequest, TicketRequest
from app.services.chat import prepare_turn
from app.services.history import ensure_conversation, load_history, load_summaries
from app.tools.errors import ToolInfrastructureError
from app.tools.executor import execute_tool
from app.tools.registry import build_retriever, build_tools, registry_for

router = APIRouter()

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
        conversation = await ensure_conversation(
            session=session, session_id=session_id, user_id=user_id
        )
        if request.resume is None:
            history = await load_history(session=session, conversation_id=session_id)
            # 产出是**裁剪后的历史**(不是组装好的消息):消息组装搬进了
            # agent 节点 —— 它要往本轮 human 消息里插检索证据块。
            history = prepare_turn(
                settings=settings, history=history, user_input=request.message
            )
            # ch07 §7.6:指代消解 / 意图识别共用的那份上下文,**每轮必打**。
            # 位置有两重意义:① 在 `resolve_references` **之前**(它就是这个
            # 上下文最早的两个消费者);② 在**路由之前** —— 闲聊/投诉/兜底/
            # 退款子流程那几轮不进 Agent,而它们恰恰最容易「看起来正常、
            # 其实上下文是错的」。写进 `generate()` 里就漏掉整个这一类。
            #
            # `resume` 分支**不打**:续跑不是新的一轮(spec §5.1),它从
            # checkpoint 还原上下文,`resolve_references` 也不会重跑 ——
            # 这里再打一行会是一条与事实不符的日志。
            journal.history_ctx(
                conversation_id=session_id,
                summaries=await load_summaries(
                    session=session, conversation_id=session_id
                ),
                history=history,
                budget=budget.derive(
                    settings=settings,
                    system_prompt=render_system_prompt(settings.brand_name),
                ),
                # 新建的会话两个锚点取 0,与 `Layers` 的「`0` = 尚无梗概 /
                # 层 1 起于最早」同一套语义。**实测**:真实库上
                # `ensure_conversation` 建完读回来就是 `0 / 0`(标量默认值在
                # INSERT 时落到属性上);`or 0` 只归一**替身**会话那个没跑过
                # flush、属性还是 None 的形状 —— 否则日志里会出现一个
                # 无意义的 `null`,而读日志的人没法把它与「锚点真的没推」分开。
                summary_upto_msg_id=conversation.summary_upto_msg_id or 0,
                layer1_from_msg_id=conversation.layer1_from_msg_id or 0,
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
            #    `prepare_turn` 更不能用:它拿不到本轮输入(挂起那轮的原话在 state
            #    里),真拿 None 递进去会炸在 tiktoken 里 —— 请求语义问题变 500。
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
                "history": history,        # prepare_turn 返回的**裁剪后**历史
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
        lock.release()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
