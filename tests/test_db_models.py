"""DB 集成测试。需要 MySQL 在跑:见 spec §7.4。"""

import pytest
from sqlalchemy import select, text

from app.db.base import get_engine, get_sessionmaker
from app.db.models import Conversation, Faq, MessageRecord, Ticket

pytestmark = pytest.mark.db

SCRATCH_CONVERSATION = "test0000000000000000000000000000"

#: 探针问题必须**不等于** seed 行。`scripts/seed_db.py` 的 `FAQ_ROWS[0]["question"]`
#: 就是「退货政策是什么」—— 探针与它同串时,下面按名清理会连种子行一并删掉
#: (实测日志:faq 表里那一行被本测试删除,直到 test_tools_db 的 seed() 才补回来;
#: 全套件因文件排序侥幸自愈,单跑本文件则把验收 5 依赖的那行留成空档)。
#: 加 probe 尾巴后,按名清理只可能命中本测试自己插入的那一行。
SCRATCH_FAQ_QUESTION = "退货政策是什么probe"


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
                # 中文必须是**真插进去再读回来**才有意义:原先这里是空串,
                # 而 docstring 却声称验了 messages.content 的中文往返 ——
                # 空串在什么编码下都能往返,那条声称是空头支票。
                content="我帮您查一下物流",
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
        assert row.content == "我帮您查一下物流"                  # 中文往返

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
    """中文子串能命中。实测已确认 utf8mb4_0900_ai_ci 正常,此测试防回归。

    pattern 刻意**不是**裸 `%退货%`:种子里就有含「退货」的行,裸 pattern 加
    `len(hits) == 1` 必挂,而原来的 `>= 1` 又可以被种子行单独满足 —— 那样即使
    本测试的 INSERT 整段失效,断言照样全绿(这正是「假绿」的形态)。把 pattern
    锚到探针独有的 probe 尾巴上,命中数就只可能来自本测试自己插的那一行;
    中文部分仍在 pattern 里,所以编码/排序规则一坏,这里同样会红。
    """
    async with get_sessionmaker()() as session:
        session.add(
            Faq(
                question=SCRATCH_FAQ_QUESTION,
                answer="七天无理由退货",
                category="退换货",
            )
        )
        await session.commit()
        hits = (
            await session.execute(
                select(Faq).where(Faq.question.like("%退货%probe%"))
            )
        ).scalars().all()
        assert len(hits) == 1
        assert hits[0].question == SCRATCH_FAQ_QUESTION
        await session.execute(
            text("DELETE FROM faq WHERE question = :q"),
            {"q": SCRATCH_FAQ_QUESTION},
        )
        await session.commit()
