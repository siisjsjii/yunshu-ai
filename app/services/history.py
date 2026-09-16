"""会话历史的读写。只做 DB I/O —— Message -> BaseMessage 的转换在 prompts.py。"""

from collections.abc import Sequence

from sqlalchemy import select

from app.db.models import Conversation, MessageRecord
from app.schemas import Message


async def ensure_conversation(*, session, session_id: str, user_id: str) -> Conversation:
    """取会话,不存在则新建。

    已存在时**忽略传入的 user_id**,以创建时记录的为准 —— 否则任何客户端
    都能改掉一条会话的归属(本章端点没有认证)。
    """
    conversation = (
        await session.execute(
            select(Conversation).where(Conversation.id == session_id)
        )
    ).scalars().one_or_none()

    if conversation is None:
        conversation = Conversation(id=session_id, user=user_id, status="active")
        session.add(conversation)
        await session.commit()
    return conversation


async def load_history(*, session, conversation_id: str) -> list[Message]:
    """按插入序(id)正序读出该会话的全部消息。"""
    rows = (
        await session.execute(
            select(MessageRecord)
            .where(MessageRecord.conversation_id == conversation_id)
            .order_by(MessageRecord.id)
        )
    ).scalars().all()

    return [
        Message(
            role=row.role,
            content=row.content,
            tool_calls=row.tool_calls,
            tool_call_id=row.tool_call_id,
        )
        for row in rows
    ]


async def append_turn(
    *, session, conversation_id: str, messages: Sequence[Message]
) -> None:
    """把一轮的消息一次性写入。

    调用方只在**流完整走完后**才调它 —— 与 ch01「半截回复不污染历史」
    的语义一致,也避免留下"有问无答"的孤儿行。
    """
    for message in messages:
        session.add(
            MessageRecord(
                conversation_id=conversation_id,
                role=message.role,
                content=message.content,
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id,
            )
        )
    await session.commit()
