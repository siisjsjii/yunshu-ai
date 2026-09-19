import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, format_sse_event
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.emit import make_emitter
from app.agent.graph import build_graph, get_checkpointer
from app.config import Settings, get_settings
from app.db.session import get_session
from app.llm import create_chat_model, create_extract_model
from app.memory.store import SessionStore
from app.memory.trim import ContextOverflowError
from app.sanitize import redact_api_key
from app.schemas import ChatRequest
from app.services.chat import prepare_turn
from app.services.history import ensure_conversation, load_history
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
    加一行同形的即可。**不做这层,端点测试里 28 条都会朝
    `https://example.invalid/v1` 发真实请求,而且不联网这一条是硬规矩。**
    """
    return create_extract_model(settings)


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
        await ensure_conversation(
            session=session, session_id=session_id, user_id=user_id
        )
        history = await load_history(session=session, conversation_id=session_id)
        # 产出是**裁剪后的历史**(不是组装好的消息):消息组装搬进了
        # agent 节点 —— 它要往本轮 human 消息里插检索证据块。
        history = prepare_turn(
            settings=settings, history=history, user_input=request.message
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

            # emit 必须在图**之外**创建、在节点里才被调用 —— make_emitter 返回的
            # 是可重复调用的函数,每次发帧时现取 writer(见 app/agent/emit.py)。
            emit = make_emitter()
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

            final = {"trace": [], "intent": None, "gate_passed": None, "agent_steps": 0}
            async for payload in graph.astream(
                {
                    "conversation_id": session_id,
                    "user_input": request.message,
                    "history": history,        # prepare_turn 返回的**裁剪后**历史
                    "trace": [],
                },
                config={"configurable": {"thread_id": session_id}},
                stream_mode="custom",
            ):
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

            yield _frame("done", {
                "finish_reason": "stop",
                "usage": None,
                "trace": final.get("trace") or [],
                "intent": final.get("intent"),
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
