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

#: 钉字面 % / _ 用的两行临时 FAQ,用例跑完由 _cleanup 抹掉。
#: 第二行的存在是刻意的:它含 "100" 但**不含**字面 %,不转义时 pattern
#: "%100%%" 会把它一并捞出,`count == 1` 随即挂 —— 少了它,下面那条测试
#: 对"到底转义没转义"完全无感(即假绿)。
SCRATCH_FAQ_QUESTIONS = ("100% 纯棉 T 恤怎么洗", "满 100 元有赠品吗")


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
            for question in SCRATCH_FAQ_QUESTIONS:
                await s.execute(
                    text("DELETE FROM faq WHERE question = :q"), {"q": question}
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


def test_query_faq_wildcard_keyword_cannot_match_everything():
    """纯通配符关键词必须落空,而不是把任意三条 FAQ 当结果递回去。

    不转义时 pattern 是 "%%%",对表里任何一行都成立 —— 工具于是返回
    ok=true 加三条与问题无关的答案,模型会照着它们自信作答。这是 Ruling 2
    要堵的洞的另一半:漏召回不是"查不到才发生",通配符能让它**永远查得到**。
    关键词由模型从用户原话里摘("100% 纯棉"这类),% 与 _ 会原样传进来。
    """
    async def run():
        async with get_sessionmaker()() as session:
            tool = make_query_faq(session)
            return await tool.ainvoke(
                {"name": "query_faq", "args": {"keyword": "%"}, "id": "c", "type": "tool_call"}
            )

    with pytest.raises(ToolNotFound):
        asyncio.run(run())


def test_query_faq_matches_literal_percent_not_everything():
    """关键词里的 % 按字面匹配:命中那一行,且只命中那一行。"""
    async def run():
        async with get_sessionmaker()() as session:
            session.add_all(
                [
                    Faq(question=q, answer="临时行", category="测试")
                    for q in SCRATCH_FAQ_QUESTIONS
                ]
            )
            await session.commit()
            tool = make_query_faq(session)
            return await tool.ainvoke(
                {"name": "query_faq", "args": {"keyword": "100%"}, "id": "c", "type": "tool_call"}
            )

    hits = json.loads(asyncio.run(run()).content)
    assert hits["count"] == 1
    assert hits["items"][0]["question"] == SCRATCH_FAQ_QUESTIONS[0]


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
