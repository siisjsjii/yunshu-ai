"""历史服务测试。需要 MySQL。"""

import asyncio

import pytest
from sqlalchemy import select, text

from app.db.base import get_sessionmaker
from app.db.models import Conversation, MessageRecord
from app.schemas import Message
from app.services.history import append_turn, ensure_conversation, load_history

pytestmark = pytest.mark.db

#: 草稿会话号。计划与 brief 里写的是 33 字符,但 conversations.id 是
#: String(32)(设计如此:id 复用 ch01 的 uuid4().hex,32 字符),MySQL 8
#: 在 STRICT_TRANS_TABLES 下插 33 字符直接报 DataError 1406,用例根本跑不到
#: 断言。故此处按列宽取 32 字符,断言语义一字未改。
SCRATCH = "histtest000000000000000000000000"


def asyncio_run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _cleanup():
    async def _drop():
        async with get_sessionmaker()() as session:
            await session.execute(
                text("DELETE FROM messages WHERE conversation_id = :c"), {"c": SCRATCH}
            )
            await session.execute(
                text("DELETE FROM conversations WHERE id = :c"), {"c": SCRATCH}
            )
            await session.commit()

    asyncio_run(_drop())
    yield
    asyncio_run(_drop())


def test_ensure_conversation_creates_then_reuses():
    async def run():
        async with get_sessionmaker()() as session:
            first = await ensure_conversation(
                session=session, session_id=SCRATCH, user_id="alice"
            )
            assert first.user == "alice"
            assert first.status == "active"
            # 已存在时忽略新的 user_id,以创建时为准
            again = await ensure_conversation(
                session=session, session_id=SCRATCH, user_id="mallory"
            )
            return again.user

    assert asyncio_run(run()) == "alice"


def test_append_turn_then_load_history_roundtrips_tool_messages():
    tool_calls = [{"id": "c1", "name": "query_logistics", "args": {"order_id": "1001"}}]

    async def run():
        async with get_sessionmaker()() as session:
            await ensure_conversation(session=session, session_id=SCRATCH, user_id="u")
            await append_turn(
                session=session,
                conversation_id=SCRATCH,
                messages=[
                    Message(role="user", content="订单 1001 到哪了"),
                    Message(role="assistant", content="", tool_calls=tool_calls),
                    Message(role="tool", content="已揽件", tool_call_id="c1"),
                    Message(role="assistant", content="您的包裹已揽件。"),
                ],
            )
            return await load_history(session=session, conversation_id=SCRATCH)

    history = asyncio_run(run())
    assert [m.role for m in history] == ["user", "assistant", "tool", "assistant"]
    assert history[1].tool_calls[0]["id"] == "c1"     # JSON 列往返
    # 深层结构整体比对:只断言某字段非空的话,"args 内层字典被拍平成字符串"
    # 这类有损往返照样全绿。
    assert history[1].tool_calls == tool_calls
    assert history[2].tool_call_id == "c1"
    assert history[2].content == "已揽件"              # 中文往返


def test_load_history_returns_chronological_order():
    async def run():
        async with get_sessionmaker()() as session:
            await ensure_conversation(session=session, session_id=SCRATCH, user_id="u")
            await append_turn(
                session=session, conversation_id=SCRATCH,
                messages=[Message(role="user", content="第一句")],
            )
            await append_turn(
                session=session, conversation_id=SCRATCH,
                messages=[Message(role="user", content="第二句")],
            )
            return await load_history(session=session, conversation_id=SCRATCH)

    contents = [m.content for m in asyncio_run(run())]
    assert contents == ["第一句", "第二句"]


def test_load_history_orders_by_id_when_created_at_disagrees():
    """id 序 ≠ created_at 序时,必须按 id(插入序)返回。

    上一条顺序用例区分不出两种实现:两条消息在同一秒内写入,created_at
    打平,MySQL 恰好按插入序返回 —— 把 order_by(MessageRecord.id) 换成
    created_at 它依然全绿(已实测)。这里把 created_at 人为倒挂,让「按 id」
    与「按时间戳」给出相反结果,那条不变式才真正被钉住。
    """

    async def run():
        async with get_sessionmaker()() as session:
            await ensure_conversation(session=session, session_id=SCRATCH, user_id="u")
            # 同一批写入,但先写的那条 created_at 更晚:id 序与时间戳序相反。
            await session.execute(
                text(
                    "INSERT INTO messages (conversation_id, role, content, created_at) "
                    "VALUES (:c, 'user', '先写的', '2999-01-01 00:00:00'),"
                    "       (:c, 'user', '后写的', '2000-01-01 00:00:00')"
                ),
                {"c": SCRATCH},
            )
            await session.commit()
            return await load_history(session=session, conversation_id=SCRATCH)

    assert [m.content for m in asyncio_run(run())] == ["先写的", "后写的"]
