"""两个 DB 工具的测试。需要 MySQL。"""

import asyncio
import json

import pytest
from sqlalchemy import select, text

from app.db.base import get_sessionmaker
from app.db.models import Conversation, Faq, Ticket
from app.tools.business import make_create_ticket, make_query_faq
from app.tools.errors import ToolNotFound

pytestmark = pytest.mark.db

SCRATCH_CONVERSATION = "tooltest000000000000000000000000"


def _call(tool, args: dict) -> str:
    tool_call = {"name": tool.name, "args": args, "id": "call_1", "type": "tool_call"}
    return asyncio.run(tool.ainvoke(tool_call)).content


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    async def _drop():
        async with get_sessionmaker()() as s:
            await s.execute(
                text("DELETE FROM tickets WHERE conversation_id = :c"),
                {"c": SCRATCH_CONVERSATION},
            )
            await s.execute(
                text("DELETE FROM conversations WHERE id = :c"),
                {"c": SCRATCH_CONVERSATION},
            )
            await s.commit()
    asyncio.run(_drop())


def test_query_faq_finds_seeded_row():
    from scripts.seed_db import seed

    asyncio.run(seed())

    async def run():
        async with get_sessionmaker()() as session:
            tool = make_query_faq(session)
            return await tool.ainvoke(
                {"name": "query_faq", "args": {"keyword": "退货"}, "id": "c", "type": "tool_call"}
            )

    hits = json.loads(asyncio.run(run()).content)
    assert hits["count"] >= 1
    assert any("退货" in item["question"] for item in hits["items"])


def test_query_faq_raises_not_found_for_unmatched_keyword():
    """验收 3 的漏召回路径:查不到要走 ToolNotFound,不是返回空列表假装成功。"""
    async def run():
        async with get_sessionmaker()() as session:
            tool = make_query_faq(session)
            return await tool.ainvoke(
                {"name": "query_faq", "args": {"keyword": "邮费"}, "id": "c", "type": "tool_call"}
            )

    with pytest.raises(ToolNotFound):
        asyncio.run(run())


def test_create_ticket_does_not_expose_conversation_id_to_model():
    """conversation_id 必须对模型不可见 —— 让模型填会编造 id。"""
    tool = make_create_ticket(session=None, conversation_id=SCRATCH_CONVERSATION)
    assert set(tool.args_schema.model_json_schema()["properties"]) == {
        "description",
        "ticket_type",
    }


def test_create_ticket_writes_row():
    async def run():
        async with get_sessionmaker()() as session:
            session.add(
                Conversation(id=SCRATCH_CONVERSATION, user="tester", status="active")
            )
            await session.commit()
            tool = make_create_ticket(
                session=session, conversation_id=SCRATCH_CONVERSATION
            )
            return await tool.ainvoke(
                {
                    "name": "create_ticket",
                    "args": {"description": "鞋码不对", "ticket_type": "换货"},
                    "id": "c",
                    "type": "tool_call",
                }
            )

    payload = json.loads(asyncio.run(run()).content)
    assert payload["ticket_no"]
    assert payload["conversation_id"] == SCRATCH_CONVERSATION

    async def check():
        async with get_sessionmaker()() as session:
            row = (
                await session.execute(
                    select(Ticket).where(Ticket.conversation_id == SCRATCH_CONVERSATION)
                )
            ).scalars().one()
            conv = (
                await session.execute(
                    select(Conversation).where(Conversation.id == SCRATCH_CONVERSATION)
                )
            ).scalars().one()
            return row.ticket_type, conv.status

    ticket_type, conv_status = asyncio.run(check())
    assert ticket_type == "换货"
    assert conv_status == "pending_human"   # 建单即转人工
