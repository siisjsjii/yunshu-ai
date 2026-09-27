"""会话历史的读写。只做 DB I/O —— Message -> BaseMessage 的转换在 prompts.py。"""

from collections.abc import Sequence

from sqlalchemy import func, select, update

from app.db.models import Conversation, ConversationSummary, MessageRecord
from app.schemas import Message


async def ensure_conversation(*, session, session_id: str, user_id: str) -> Conversation:
    """取会话,不存在则新建。

    已存在时**忽略传入的 user_id**,以创建时记录的为准 —— 否则任何客户端
    都能改掉一条会话的归属。

    ⚠️ **本函数不校验归属**(它只按 id 取 / 建):已存在的行**不管属于谁**都原样
    返回。认证(2026-09-27)之后,三个写端点(`/api/chat/stream` / `/api/ticket` /
    `/api/refund`)在调用本函数之后**各自**比对 `conv.user` 与登录用户,不符就 404;
    两个读端点走下面的 `get_owned_conversation`(那条把「必须是我的」写进 SQL)。
    ⇒ 将来任何新的调用方,**归属得自己查** —— 这里给不出保证,而漏掉的后果是
    「拿别人的会话读 / 写」且没有任何东西报错。
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


async def get_owned_conversation(*, session, conversation_id: str,
                                 user_id: str) -> Conversation | None:
    """取会话,**且必须是这个用户的**;不属于他 ⇒ 回 `None`。

    ⚠️ **「别人的会话」与「不存在的会话」必须是同一个答案**(端点都翻成 404):
    回 403 等于承认「这个 id 存在」,那就给了枚举的口子。它与
    `ensure_conversation` 那条「已存在时忽略传入的 user_id」的分工是:
    那边保的是**归属不被改写**,这边把「必须是我的」写进 SQL。

    (写路径**另有一份**:三个写端点在 `ensure_conversation` 之后各自比对
    `conv.user` —— 它们不能复用本函数,因为「不存在就新建」是它们要的行为。
    两处判据**同一个出口**(404)。)
    """
    return (await session.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.user == user_id)
    )).scalars().one_or_none()


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
    *,
    session,
    conversation_id: str,
    messages: Sequence[Message],
    citations: list[dict] | None = None,
) -> list[int]:
    """把一轮的消息一次性写入,**返回新行的 id 列表(与入参同序)**。

    调用方只在**流完整走完后**才调它 —— 与 ch01「半截回复不污染历史」
    的语义一致,也避免留下"有问无答"的孤儿行。

    `flush()` 在这里只为一件事 —— 拿自增主键;`commit()` 才让整轮同时生效
    (半轮历史 = 有问无答的孤儿行)。

    **返回值的消费者,如实记(终审 Minor 9)**:ch07 曾经把它写成「落库后把
    `messages.id` 写回 state,下一轮分层才认得出哪些消息已经是库里的历史」——
    **那条路不存在**。`app/agent/nodes.py` 的 `log_turn` **丢弃**本函数的返回值,
    而 spec §7.4 已把它订正为:精简版**从 `load_history` 派生**(那一份本来就带
    `Message.id`),不从 `state["messages"]` 派生(后者除播种那一批外拿到的都是
    `add_messages` 现赋的 uuid4,与两个锚点不可比)。所以今天**生产代码里没有
    读者**,只有测试(`tests/test_history.py` 用它逐条比对
    `load_history` 填回的主键与顺序)。
    留着返回值不是为了将来,是因为它**免费**(`flush` 本来就要跑)且是
    「同一事实只有一个来源」的落点:哪个 id 属于刚写进去的行,只有这里知道。
    与它同族的是 `append_summary_and_advance` 的 `-> int`(`seq` 从落库那一步
    返回,而不是让调用方再查一遍)。

    ---- `citations`(历史回载把「文档链接」还回去)----

    当轮回答引用到的知识块,挂到**本轮最后一条 `content` 非空的 assistant 行**
    上。三条都写在断言里(`tests/test_history.py`),理由是各自的一种「静默挂错」:

    · **必须是「最后一条」**:一轮 ReAct 里 assistant 行**有多条**(带
      `tool_calls` 的那条常常 `content` 为空),而带 `[n]` 编号、用户真正看见过
      正文的是最后那条。
    · **必须 `content` 非空**:挂到空气泡上 ⇒ 回载端点那句 `content != ''`
      根本不回它 ⇒ 弹层挂在一个**画不出来的行**上,点了 [n] 什么都不发生。
    · **`[]` 与 `None` 都不挂**:空数组与「没引用」在库里长得一样而含义不同
      (同表的 `tool_calls` 也是 `or None` 的写法)。

    形状与 `app/agent/nodes.py` 造 citations 帧时那份**逐字相同**
    (`{"n": i, "chunk_id", "section_path", "question", "answer", "category"}`)
    —— 前端 `makeCitesClickable` 读的正是这几个键,换个键名 = 帧到了、库也写了,
    **弹层却渲染不出来**,而没有任何东西会红。

    ⚠️ 这一步**只在内存里给某一行挂一个值**,不多一次 IO、也不做任何校验 ——
    它和那几行本身在**同一次 `commit`** 里生效(所以不存在「引用写进去了、
    消息没写进去」这种半截状态)。正因为没有可能失败的分支,这里**刻意不写
    `try`**：加了兜底反而会造出一条「引用静默丢失」的路,而它不报错。
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

    if citations:
        # 两遍走:**先**把全部行建出来(**顺序与入参严格一致**,`flush` 拿到的
        # 自增 id 才与入参同序,返回值那条契约不能破),**再**回头挂引用。
        # 写成一遍(边建边判「这条是不是最后一条」)就得先知道后面还有没有 ——
        # 那要么多一次扫描,要么把「最后一条」的判据散到两个地方。
        attach_at = None
        for i, message in enumerate(messages):
            # `message.content` 为**空串**(空气泡)的不挂 —— 见 docstring。
            if message.role == "assistant" and message.content:
                attach_at = i
        if attach_at is not None:
            # `list(...)` 复制一份:调用方(与 state)手里那个列表随后还可能被
            # 改动,而这一列要的是**落库那一刻**的快照。
            records[attach_at].citations = list(citations)

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
