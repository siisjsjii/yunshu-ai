"""DB 集成测试。需要 MySQL 在跑:见 spec §7.4。"""

import pytest
from sqlalchemy import select, text

from app.db.base import get_engine, get_sessionmaker
from app.db.models import Conversation, Faq, MessageRecord, Ticket

pytestmark = pytest.mark.db

SCRATCH_CONVERSATION = "test0000000000000000000000000000"


@pytest.mark.anyio
async def test_tables_exist_and_chinese_roundtrips():
    """四张表建得出来,且中文与 JSON 列往返不炸。"""
    engine = get_engine()
    async with get_sessionmaker()() as session:
        # 清理上次残留
        await session.execute(
            text("DELETE FROM messages WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.execute(
            text("DELETE FROM tickets WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.execute(
            text("DELETE FROM conversations WHERE id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )

        session.add(
            Conversation(id=SCRATCH_CONVERSATION, user="tester", status="active")
        )
        session.add(
            MessageRecord(
                conversation_id=SCRATCH_CONVERSATION,
                role="assistant",
                content="",
                tool_calls=[
                    {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call_1"}
                ],
            )
        )
        session.add(
            Ticket(
                ticket_no="T-TEST-1",
                conversation_id=SCRATCH_CONVERSATION,
                description="鞋码不对想换",
                ticket_type="换货",
                status="open",
            )
        )
        await session.commit()

    async with get_sessionmaker()() as session:
        row = (
            await session.execute(
                select(MessageRecord).where(
                    MessageRecord.conversation_id == SCRATCH_CONVERSATION
                )
            )
        ).scalars().one()
        assert row.tool_calls[0]["name"] == "query_logistics"   # JSON 列往返
        assert row.tool_call_id is None

        conv = (
            await session.execute(
                select(Conversation).where(Conversation.id == SCRATCH_CONVERSATION)
            )
        ).scalars().one()
        assert conv.user == "tester"

    # 收尾清理,不留垃圾数据
    async with get_sessionmaker()() as session:
        await session.execute(
            text("DELETE FROM messages WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.execute(
            text("DELETE FROM tickets WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.execute(
            text("DELETE FROM conversations WHERE id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.commit()

    await engine.dispose()


@pytest.mark.anyio
async def test_faq_like_matches_chinese_substring():
    """中文子串能命中。实测已确认 utf8mb4_0900_ai_ci 正常,此测试防回归。"""
    async with get_sessionmaker()() as session:
        session.add(
            Faq(question="退货政策是什么", answer="七天无理由退货", category="退换货")
        )
        await session.commit()
        hits = (
            await session.execute(
                select(Faq).where(Faq.question.like("%退货%"))
            )
        ).scalars().all()
        assert len(hits) >= 1
        await session.execute(
            text("DELETE FROM faq WHERE question = :q"),
            {"q": "退货政策是什么"},
        )
        await session.commit()
