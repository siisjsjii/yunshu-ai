"""灌种子数据。用法:.venv/Scripts/python.exe scripts/seed_db.py

幂等:样例会话与工单用固定主键,重复跑不会堆积。

**2026-09-20:`faq` 表已废弃,本脚本不再灌它。**
原来这 12 条 FAQ 种子在 ch03 就被 `build_kb.py` **迁移进 `knowledge_chunks`**
(那里才是权威源,在线检索查的是它 + Milvus),`faq` 表此后没有读写方。
用户删表后本脚本的 faq 段一并删除 —— **FAQ 数据没有丢**,它现在是知识库里的
12 条 `content_type="faq"` 块。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.db.base import get_sessionmaker
from app.db.models import Conversation, MessageRecord, Ticket

SAMPLE_CONVERSATION = "seed0000000000000000000000000000"
SAMPLE_TICKET_NO = "T-SEED-0001"


async def seed() -> None:
    async with get_sessionmaker()() as session:
        if not (
            await session.execute(
                select(Conversation).where(Conversation.id == SAMPLE_CONVERSATION)
            )
        ).scalars().first():
            session.add(
                Conversation(
                    id=SAMPLE_CONVERSATION, user="demo-user", status="active"
                )
            )
            session.add(
                MessageRecord(
                    conversation_id=SAMPLE_CONVERSATION,
                    role="user",
                    content="订单 20240915 的鞋码不对,我想换大一码",
                )
            )
            session.add(
                MessageRecord(
                    conversation_id=SAMPLE_CONVERSATION,
                    role="assistant",
                    content="好的,请提供原规格与目标规格,我为您登记换货。",
                )
            )
            session.add(
                Ticket(
                    ticket_no=SAMPLE_TICKET_NO,
                    conversation_id=SAMPLE_CONVERSATION,
                    description="鞋码偏小,想换成大一码",
                    ticket_type="换货",
                    status="open",
                )
            )

        await session.commit()


if __name__ == "__main__":
    asyncio.run(seed())
    print("种子完成:1 组样例会话/消息/工单(faq 表已废弃,不再灌)")
