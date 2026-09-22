"""会话历史的读写。只做 DB I/O —— Message -> BaseMessage 的转换在 prompts.py。"""

from collections.abc import Sequence

from sqlalchemy import func, select, update

from app.db.models import Conversation, ConversationSummary, MessageRecord
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
    """按插入序(id)正序读出该会话的全部消息。

    ch07:**带上主键**(`Message.id`)。分层完全依赖它 —— 两个锚点存的就是
    主键,没有 id 的话 `layers.split` 无从比对,层 2 与层 1 的边界会静默退化
    成「全在层 1」,而每一轮看起来都完全正常。
    """
    rows = (
        await session.execute(
            select(MessageRecord)
            .where(MessageRecord.conversation_id == conversation_id)
            .order_by(MessageRecord.id)
        )
    ).scalars().all()

    return [
        Message(
            id=row.id,
            role=row.role,
            content=row.content,
            tool_calls=row.tool_calls,
            tool_call_id=row.tool_call_id,
        )
        for row in rows
    ]


async def append_turn(
    *, session, conversation_id: str, messages: Sequence[Message]
) -> list[int]:
    """把一轮的消息一次性写入,**返回新行的 id 列表(与入参同序)**。

    调用方只在**流完整走完后**才调它 —— 与 ch01「半截回复不污染历史」
    的语义一致,也避免留下"有问无答"的孤儿行。

    ch07 起返回值是必需的:ReAct 往返(assistant 带 tool_calls / tool / 收尾
    assistant)落库后,要把它们的 `messages.id` 写回 state,下一轮分层才认得出
    哪些消息已经是「库里的历史」。`flush()` 在这里只为一件事 —— 拿自增主键;
    `commit()` 才让整轮同时生效(半轮历史 = 有问无答的孤儿行)。
    """
    records: list[MessageRecord] = []
    for message in messages:
        record = MessageRecord(
            conversation_id=conversation_id,
            role=message.role,
            content=message.content,
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id,
        )
        session.add(record)
        records.append(record)

    await session.flush()          # 拿自增主键;一条 mapper 的插入顺序即 add 顺序
    new_ids = [record.id for record in records]
    await session.commit()
    return new_ids


async def load_summaries(*, session, conversation_id: str) -> list[tuple[int, str]]:
    """按 `seq` 正序读出该会话的全部梗概:`[(seq, content), …]`。

    只读,不做拼接 —— 怎么把多段梗概合成一段注入文本是
    `app.memory.summarize` 的事(本章 T8),这里只管 I/O。
    """
    rows = (
        await session.execute(
            select(ConversationSummary)
            .where(ConversationSummary.conversation_id == conversation_id)
            .order_by(ConversationSummary.seq)
        )
    ).scalars().all()
    return [(row.seq, row.content) for row in rows]


async def advance_anchors(
    *,
    session,
    conversation_id: str,
    summary_upto: int | None = None,
    layer1_from: int | None = None,
) -> None:
    """推进两个锚点中的**任意一个或两个**;没传的那个**原样不动**。

    部分更新是硬要求,不是顺手:降级路径(HTTP 请求里)只推 `layer1_from`,
    而摘要任务只推 `summary_upto`。写成「两个都写」会把没传的那个打回 0
    —— 前者让 `summary_upto_msg_id` 归零 ⇒ 已压过的原文**再压一遍**;
    后者让 `layer1_from_msg_id` 归零 ⇒ 层 2 突然吞掉全部历史。

    两个都没传时**不提交**(一次多余的 commit 也会把调用方同一事务里
    还没写完的东西提前落库)。
    """
    values = {}
    if summary_upto is not None:
        values["summary_upto_msg_id"] = summary_upto
    if layer1_from is not None:
        values["layer1_from_msg_id"] = layer1_from
    if not values:
        return

    await session.execute(
        update(Conversation).where(Conversation.id == conversation_id).values(**values)
    )
    await session.commit()


async def append_summary_and_advance(
    *, session, conversation_id: str, upto_msg_id: int, content: str
) -> int:
    """**原子**落一段梗概并把 `summary_upto_msg_id` 推到 `upto_msg_id`。

    **返回这一段新写入的 `seq`(第几段,从 1 起)。** 调用方(T9 的摘要任务)
    要把它记进 `summary done` 那行日志 —— spec §7.6 的「第 N 段」。

    **为什么是返回而不是让调用方再查一遍**:`seq` 是**本函数自己算出来的**
    (`MAX(seq)+1`)。丢掉它再去查一次,就是把同一个事实算两遍,而两遍**可以
    不一致**(期间并发的另一段提交了,或者读到的不是这一行)。返回出来才是
    单一来源。(这处签名变更(T3 → `-> int`)向后兼容:既有调用方忽略返回值即可。)

    spec §6.1:只成一半的两种后果都很难看,所以这两步必须在同一个事务里提交。
    这就是为什么它们是**一个函数**而不是两个 —— 拆开就会出现「调用方忘了
    两个都调」或「两个都调了但中间抛了」。

    `seq` 取 `MAX(seq)+1`(只增不改),与唯一键 `(conversation_id, seq)`
    互为表里:真并发时后者撞键 ⇒ 本次整体回滚 ⇒ 锚点不推进 ⇒ 下次重来。
    """
    next_seq = (
        await session.execute(
            select(func.coalesce(func.max(ConversationSummary.seq), 0)).where(
                ConversationSummary.conversation_id == conversation_id
            )
        )
    ).scalar_one() + 1

    session.add(
        ConversationSummary(
            conversation_id=conversation_id,
            seq=next_seq,
            upto_msg_id=upto_msg_id,
            content=content,
        )
    )
    await session.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(summary_upto_msg_id=upto_msg_id)
    )
    await session.commit()      # 一次提交,两步同时生效
    return next_seq
