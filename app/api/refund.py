"""退款单提交端点。

为什么是**独立端点**而不是模型工具:按钮点击是 HTTP 请求,够不到模型工具 ——
与 ch05 的 `/api/ticket` 同一个理由。

守护栏与对话端点一致:同一把会话锁串行化;写库**不重试**(重试会建出两张单)。
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
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

logger = logging.getLogger(__name__)

#: 基础设施故障对外的**固定文案**。**与 `app/tools/executor.py:91` 同一条**
#: (spec §5.3 / §8:「502 + 固定文案」)—— 不是这里另立的一份。原始异常只进
#: 服务端日志(`logger.exception`),不出站:出站文本里带上游异常原文会把
#: 「服务端出问题」的细节泄漏给调用方。
INFRA_FAILURE_DETAIL = "数据服务暂时不可用"

router = APIRouter()


def _infra_failure(api_key: str) -> HTTPException:
    """基础设施故障 → 502 + 固定文案。

    两条 `except` 共用,是为了让 502 的文案**只有一处** —— 分头写两份的话,
    下一次改文案必然漏掉一条。仍然过一遍 `redact_api_key`(spec §8:出站文本
    一律过它):字面量本身不含密钥,但把"所有出站文本都过同一个出口"这条规则
    留成无例外的,比每次判断"这个字符串要不要脱敏"可靠。
    """
    return HTTPException(
        status_code=502,
        detail=redact_api_key(INFRA_FAILURE_DETAIL, api_key),
    )


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

    **类目校验先于任何 IO**:不在固定集里就是请求语义错 → 422,一行都不写。
    顺序是有载荷的 —— 把校验挪到 `session.add` 之后,"先落库再校验"同样会返回
    422,库里却已经留了行:调用方看到的是**拒绝**,数据库里却是一次**成功的
    退款申请**。`test_rejects_category_outside_the_closed_set` 断的就是这个
    (只看状态码是看不出来的)。

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
        # 纵深防御。**今天这条在本端点不可达**:`ToolInfrastructureError` 全仓只有
        # `app/tools/executor.py:89-91` 一个产地,而本端点手写写库、不经过 executor;
        # `ensure_conversation`(`app/services/history.py`)也只是裸 DB I/O。
        # 留着是为了 `_persist` 将来改成委托给别处时不会静默退回 500。
        logger.exception("退款单提交命中基础设施故障")
        raise _infra_failure(settings.openai_api_key) from exc
    except SQLAlchemyError as exc:
        # **这条才是真会发生的**:`_persist` / `ensure_conversation` 里任何一次
        # 写库失败(MySQL 连不上是最可能的那个)抛的都是裸 `SQLAlchemyError`。
        # 不接它,FastAPI 默认返回 **500**,把「服务端出问题」说成「你的请求有
        # 问题」—— 与 spec §5.3 / §8 的错误语义边界相悖。ch05 spec §8.2 记的是
        # 同一类缺陷(那边靠 executor 的分类表,这边没有可依赖的翻译层)。
        #
        # 分类成「基础设施」是成立的:三个入参的 `max_length` 都对齐了各自列宽
        # (见 `RefundRequestIn`),到不了 DataError;`ensure_conversation` 的建会话
        # 竞态又已被会话锁关掉。剩下的 SQLAlchemyError 就是连接/运维类的。
        logger.exception("退款单提交命中数据库故障")
        raise _infra_failure(settings.openai_api_key) from exc
    finally:
        lock.release()
