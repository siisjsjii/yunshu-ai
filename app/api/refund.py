"""退款单提交端点。

为什么是**独立端点**而不是模型工具:按钮点击是 HTTP 请求,够不到模型工具 ——
与 ch05 的 `/api/ticket` 同一个理由。

守护栏与对话端点一致:同一把会话锁串行化;写库**不重试**(重试会建出两张单)。
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.chat import get_store          # 复用同一把会话锁的单例依赖
from app.config import Settings, get_settings
from app.db.models import RefundRequest
from app.db.session import get_session
from app.memory.store import SessionStore
from app.refund.categories import is_valid_category
from app.sanitize import redact_api_key
from app.schemas import RefundRequestIn
from app.services.history import ensure_conversation
from app.tools.errors import ToolInfrastructureError

router = APIRouter()


async def _persist(session, request: RefundRequestIn) -> RefundRequest:
    """写一行退款单。单独成函数,便于测试注入故障。"""
    row = RefundRequest(
        conversation_id=request.session_id,
        order_no=request.order_no,
        reason_category=request.reason_category,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


@router.post("/api/refund")
async def create_refund(
    request: RefundRequestIn,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
    store: SessionStore = Depends(get_store),
) -> dict:
    """退款表单的提交入口:一行 `refund_requests` + 回填的 id / status。

    **类目校验先于任何 IO**(`pytest` 断的就是这个顺序):不在固定集里就是请求语义
    错 → 422,一行都不写。2024-09 的实现若把校验放在 `session.add` 之后,"先落库
    再校验"照样能返回 422,库里却留了行 —— 调用方看到的是拒绝,数据库里是一次
    成功的退款申请。

    **锁只在一处释放**:`lock.acquire()` 之后的全部退出路径(成功、502、
    以及 `CancelledError` 这类 `BaseException`)都走同一个 `finally`。这与
    `app/api/chat.py` 的多点释放不同,是因为这里没有任何"必须返回非
    EventSourceResponse"的分支 —— 少一个释放点就少一个漏放的机会。持锁的锁既不
    被 TTL 也不被 LRU 回收,漏放 = 该会话**永久** 409,且与对话端点共用同一份
    注册表,会连带毒掉聊天。

    `store` 必须来自 `app.api.chat.get_store`:**另起一份注册表就是另一把锁**,
    同一会话上退款与对话可以同时进临界区,而任何单请求用例都看不出来。
    """
    if not is_valid_category(request.reason_category):
        raise HTTPException(status_code=422, detail="退款原因不在可选范围内")

    # `lock_for` 与 `acquire` 之间不得插入 await(见 app/api/chat.py 同处的注释):
    # 中间让出控制权的话,锁可能被容量淘汰摘掉,那个窗口就可达了。
    lock = store.lock_for(request.session_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=settings.session_lock_timeout_seconds)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=409, detail="该会话正在处理另一条消息,请稍后重试"
        ) from exc

    try:
        await ensure_conversation(
            session=session, session_id=request.session_id, user_id="demo-user"
        )
        row = await _persist(session, request)
        return {
            "id": row.id,
            "conversation_id": row.conversation_id,
            "order_no": row.order_no,
            "reason_category": row.reason_category,
            "status": row.status,
            "created_at": row.created_at.isoformat(),
        }
    except ToolInfrastructureError as exc:
        # 基础设施故障必须变 502 + 固定文案,**不是** FastAPI 默认的 500 ——
        # 500 会把「服务端出问题」说成「你的请求有问题」,与本仓既定的错误语义
        # 边界不一致(CLAUDE.md:上游/基础设施故障一律 502)。这里的文本是
        # 出站文本,一律过脱敏(纵深防御:密钥可能出现在上游异常里)。
        raise HTTPException(
            status_code=502, detail=redact_api_key(str(exc), settings.openai_api_key)
        ) from exc
    finally:
        lock.release()
