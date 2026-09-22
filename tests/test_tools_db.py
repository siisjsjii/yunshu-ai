"""两个 DB 工具的测试。需要 MySQL。"""

import asyncio
import json

import pytest
from sqlalchemy import select, text

from app.db.base import get_sessionmaker
from app.db.models import Conversation, Ticket
from app.tools.builtin.tickets import make_create_ticket

pytestmark = pytest.mark.db

SCRATCH_CONVERSATION = "tooltest000000000000000000000000"

# 本文件原先还有 6 条 query_faq 用例(种子命中、漏召回、% / _ 字面匹配、
# 错误文案有界)。ch03 把 query_faq 的内部实现换成向量检索后,它们钉的
# `Faq` 表 LIKE 查询路径已不存在 —— 契约类断言整体搬到
# `tests/test_tools_query_faq.py`(替身检索器,不依赖 MySQL,快路径也能跑),
# 「% / _ 按字面匹配」那三条随 LIKE 一并删除。删除理由与防线去向见该文件头注。


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


def test_create_ticket_clips_labels_instead_of_losing_the_ticket():
    """超长 / 全空白的 ticket_type 都必须被夹住,不能让整张工单丢掉。

    ticket_type 是模型填的**标签**,列宽 64(String(64));description 进的是
    Text 列,不受影响。本机 MySQL 是 STRICT_TRANS_TABLES,超长会抛 DataError
    —— 那是 SQLAlchemyError,T6 的分类表把它归入**不可恢复** → 502 且工单
    全丢:一个被撑爆的标签字段毁掉用户真正的问题描述。故夹标签、保单。
    """
    async def run():
        async with get_sessionmaker()() as session:
            session.add(
                Conversation(id=SCRATCH_CONVERSATION, user="tester", status="active")
            )
            await session.commit()
            tool = make_create_ticket(
                session=session, conversation_id=SCRATCH_CONVERSATION
            )
            ticket_nos = []
            for desc, ttype in (("鞋码不对", "换" * 200), ("尺码咨询", "   ")):
                raw = await tool.ainvoke(
                    {
                        "name": "create_ticket",
                        "args": {"description": desc, "ticket_type": ttype},
                        "id": "c",
                        "type": "tool_call",
                    }
                )
                ticket_nos.append(json.loads(raw.content)["ticket_no"])
            return ticket_nos

    assert all(asyncio.run(run()))   # 两次调用都得回工单号,不是抛出去

    async def check():
        async with get_sessionmaker()() as session:
            rows = (
                await session.execute(
                    select(Ticket).where(Ticket.conversation_id == SCRATCH_CONVERSATION)
                )
            ).scalars().all()
            return {row.description: row.ticket_type for row in rows}

    # 超长 → 截到 64;全空白 → 回落 "其他";两张工单的描述都原样保住。
    assert asyncio.run(check()) == {"鞋码不对": "换" * 64, "尺码咨询": "其他"}
