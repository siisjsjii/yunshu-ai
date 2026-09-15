import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, format_sse_event

from app.config import Settings, get_settings
from app.llm import create_chat_model
from app.memory.store import SessionStore
from app.memory.trim import ContextOverflowError
from app.schemas import ChatRequest
from app.services.chat import prepare_turn, stream_turn

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
) -> EventSourceResponse:
    session_id = request.session_id or uuid.uuid4().hex

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
    try:
        messages = prepare_turn(
            settings=settings,
            store=store,
            session_id=session_id,
            user_input=request.message,
        )
    except ContextOverflowError as exc:
        lock.release()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def generate():
        try:
            yield _frame(
                "meta",
                {"session_id": session_id, "model": settings.openai_model},
            )
            async for event, payload in stream_turn(
                settings=settings,
                store=store,
                model=model,
                session_id=session_id,
                user_input=request.message,
                messages=messages,
            ):
                yield _frame(event, payload)
        except Exception as exc:
            # 不泄漏密钥内容 —— 只回传异常本身的文字。
            yield _frame("error", {"message": str(exc)})
        finally:
            lock.release()

    return EventSourceResponse(
        generate(),
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
