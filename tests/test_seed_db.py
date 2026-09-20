"""种子数据测试。需要 MySQL。

**2026-09-20:**原文件的两条用例都打 `faq` 表(计数 + 种子不得含「邮费/运费」)。
`faq` 表已废弃删除(那 12 条在 ch03 就迁进了 `knowledge_chunks`,在线检索查的是
它 + Milvus),`FAQ_ROWS` 随之从 `scripts/seed_db.py` 移除,两条用例失去靶子。
这里保留**幂等性**这一条 —— 它是 `seed()` 契约里与表无关的那部分:
重复跑不堆样例会话/消息/工单。
"""

import pytest
from sqlalchemy import func, select

from app.db.base import get_sessionmaker
from app.db.models import Conversation, MessageRecord, Ticket
from scripts.seed_db import SAMPLE_CONVERSATION, SAMPLE_TICKET_NO, seed

pytestmark = pytest.mark.db


@pytest.mark.anyio
async def test_seed_is_idempotent():
    """连跑两次:样例会话/消息/工单都不应堆积。"""
    await seed()
    await seed()
    async with get_sessionmaker()() as session:
        convs = (
            await session.execute(
                select(func.count())
                .select_from(Conversation)
                .where(Conversation.id == SAMPLE_CONVERSATION)
            )
        ).scalar()
        msgs = (
            await session.execute(
                select(func.count())
                .select_from(MessageRecord)
                .where(MessageRecord.conversation_id == SAMPLE_CONVERSATION)
            )
        ).scalar()
        tickets = (
            await session.execute(
                select(func.count())
                .select_from(Ticket)
                .where(Ticket.ticket_no == SAMPLE_TICKET_NO)
            )
        ).scalar()
    assert convs == 1
    assert msgs == 2
    assert tickets == 1
