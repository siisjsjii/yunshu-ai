"""历史服务测试。需要 MySQL。"""

import asyncio

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app.db.base import get_sessionmaker
from app.db.models import Conversation, ConversationSummary, MessageRecord
from app.schemas import Message
from app.services.history import (
    advance_anchors,
    append_summary_and_advance,
    append_turn,
    ensure_conversation,
    load_history,
    load_summaries,
)

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
            # ch07:梗概行必须一起清 —— 留着的话下一次跑 `max(seq)` 会从上一轮的
            # 值接着涨,而「seq 从 1 起」这类断言就再也不成立(不是报错,是漂移)。
            await session.execute(
                text("DELETE FROM conversation_summaries WHERE conversation_id = :c"),
                {"c": SCRATCH},
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
            assert again.user == "alice"

        # 换一个**全新 session** 重查,断的是行里存的值。
        # 同一个 session 里读会被身份映射兜住(expire_on_commit=False,二次
        # select 不会用行覆盖已加载的属性),那样只证明"内存对象没被改过",
        # 证明不了"库里还是 alice" —— 回写若用 UPDATE 实现,照样全绿。
        # Ruling 1 是安全裁决,必须钉在行上。
        async with get_sessionmaker()() as session:
            stored = (
                await session.execute(
                    select(Conversation).where(Conversation.id == SCRATCH)
                )
            ).scalars().one()
            return stored.user

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

        # 另开 session 读回。同一 session 里读到的值**未必**来自 MySQL:身份映射
        # 持弱引用,只要还有东西引用那几个 ORM 实例,select 就命中缓存、直接把
        # 原来那个 Python list 交回来 —— 实测把行改成坏值时断言照样通过。
        # 这条往返是 Ruling 6 明令要钉的,不能建立在"恰好没人引用"上;换成空身份
        # 映射的新 session,值才必然由 JSON 列反序列化而来。
        async with get_sessionmaker()() as session:
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


# ---- ch07:Message.id / append_turn 返回值 / 梗概与两个锚点 ----


def test_load_history_fills_message_ids_from_the_primary_key():
    """`Message.id` 必须由 `load_history` 填上 —— 分层完全依赖它。

    填不上的症状**不是报错**,而是所有消息在分层时被一视同仁:
    两个锚点的比较无从进行,层 2 与层 1 的边界退化成「全在层 1」,
    而每一轮看起来都完全正常、每一条断言都绿。
    """
    async def run():
        async with get_sessionmaker()() as session:
            await ensure_conversation(session=session, session_id=SCRATCH, user_id="u")
            ids = await append_turn(
                session=session, conversation_id=SCRATCH,
                messages=[Message(role="user", content="你好"),
                          Message(role="assistant", content="你好呀")],
            )
            loaded = await load_history(session=session, conversation_id=SCRATCH)
            # 逐条比对**真实主键**,不是 `is not None` ——
            # NOT NULL 列上 `is not None` 是不可能失败的断言,读了会误以为有覆盖。
            assert [m.id for m in loaded] == ids
            return loaded

    loaded = asyncio_run(run())
    assert [m.role for m in loaded] == ["user", "assistant"]
    assert all(m.id is not None and m.id > 0 for m in loaded)


def test_append_turn_returns_new_row_ids_in_order():
    """返回值必须与入参**同序** —— 调用方靠它把 ReAct 往返写回 state。

    「同序」是可区分的:把实现改成 `set(...)` 或倒序返回,下面的列表比较就会红,
    而「返回值非空」这类断言区分不出来。
    """
    async def run():
        async with get_sessionmaker()() as session:
            await ensure_conversation(session=session, session_id=SCRATCH, user_id="u")
            first = await append_turn(
                session=session, conversation_id=SCRATCH,
                messages=[Message(role="user", content="一")],
            )
            second = await append_turn(
                session=session, conversation_id=SCRATCH,
                messages=[Message(role="user", content="二"),
                          Message(role="assistant", content="三")],
            )
            return first, second

    first, second = asyncio_run(run())
    assert len(first) == 1
    assert len(second) == 2
    assert first[0] < second[0] < second[1]      # 自增且同序


#: 一条**真实形状**的引用(键与 `app/agent/nodes.py` 造 citations 帧时那份
#: 完全一致:五个来源键 + `n` 编号)。回载前端拿它做 `[n]` 可点与弹层。
CITATIONS = [
    {"n": 1, "chunk_id": "77", "section_path": "退货退款政策 > 无理由退货",
     "question": "无理由退货的期限是多久?", "answer": "七天。", "category": "退换货"},
    {"n": 2, "chunk_id": "78", "section_path": "售后手册 > 保修说明",
     "question": "保修期多久?", "answer": "一年。", "category": "售后"},
]

#: `_lc_to_records` 落库时那两样形状(同一轮的两个 assistant 行分别长这样)。
TOOL_CALLS = [
    {"id": "call_1", "name": "query_order", "args": {"order_no": "1002"},
     "type": "tool_call"}
]


async def _stored_rows() -> list[MessageRecord]:
    """**新 session** 读该会话的全部行(按 id 升序)。

    为什么不能用 `load_history` 看这件事:`app/schemas.Message` 上没有
    `citations` 字段(它只服务模型上下文),所以那一路看不到这一列的写入。
    也不能在 `append_turn` 那个 session 里读 —— 身份映射持弱引用,读到的
    可能正是内存里那个刚 `add` 进去的对象,断言就变成「靠 refcount 走运」
    (本仓记过的那条)。
    """
    async with get_sessionmaker()() as session:
        return (
            await session.execute(
                select(MessageRecord)
                .where(MessageRecord.conversation_id == SCRATCH)
                .order_by(MessageRecord.id)
            )
        ).scalars().all()


def test_append_turn_attaches_citations_to_the_last_text_assistant_row():
    """引用只挂**本轮最后一条 content 非空的 assistant 行**。

    为什么是「最后一条非空」而不是「最后一条 assistant」:一轮 ReAct 里
    **assistant 行有多条**(带 tool_calls 那条常常 content 为空),而带 `[n]`
    编号、用户真正看见过正文的是**最后那条**. 挂到空气泡上 ⇒ 回载时弹层
    挂在一个**根本不会被画出来的行**上(端点的 `content != ''` 把它滤掉),
    用户点了 [n] 什么也不会发生;挂到更早那条上 ⇒ 弹层挂在**上一句**上,
    而两者都不报错。

    **这条用例的输入必须真的有多条 assistant 行**(含一条空 content 的),
    否则「最后一条」与「第一条」是同一行,断言对错的实现一样绿 —— 本仓
    第 5 类假绿(输入小到触发不了被测行为)。
    """
    async def run():
        async with get_sessionmaker()() as session:
            await ensure_conversation(session=session, session_id=SCRATCH, user_id="u")
            await append_turn(
                session=session,
                conversation_id=SCRATCH,
                messages=[
                    Message(role="user", content="订单 1002 能退吗"),
                    # ① 只申请工具调用、一个字的正文都没有(生产上就是这么落的)
                    Message(role="assistant", content="", tool_calls=TOOL_CALLS),
                    Message(role="tool", content='{"status":"已取消"}',
                            tool_call_id="call_1"),
                    # ② 用户看见过的那条 —— 引用挂它身上
                    Message(role="assistant", content="按政策可以退[1][2]。"),
                ],
                citations=CITATIONS,
            )

    asyncio_run(run())
    rows = asyncio_run(_stored_rows())
    assert [r.role for r in rows] == ["user", "assistant", "tool", "assistant"]
    # 深层结构整体比对(只断「非空」的话,内层字典被拍平成字符串照样绿)
    assert rows[3].citations == CITATIONS, rows[3].citations
    # **空的 assistant 行不许挂**:它是空气泡,回载时根本不存在
    assert rows[1].citations is None, rows[1].citations
    assert rows[0].citations is None and rows[2].citations is None


def test_append_turn_does_not_write_an_empty_citations_array():
    """`citations=[]` 与 `citations=None` **都不写** —— 留在 `None`。

    写一个空数组上去的后果不是「多一列」,是**语义不可分**:回载时
    「这一轮没有引用」与「这一轮有引用区、只是空的」在库里长得一样
    (`JSON_TYPE` 是 'ARRAY' 而不是 'NULL'),而前端只能靠「数组空不空」
    判断 —— 也就是说这个区分**在能被用到的每一个地方都丢了**。
    (同族:`app/db/models.py` 的 `tool_calls` 也走 `or None`。)
    """
    async def run():
        async with get_sessionmaker()() as session:
            await ensure_conversation(session=session, session_id=SCRATCH, user_id="u")
            await append_turn(          # 显式空列表
                session=session, conversation_id=SCRATCH,
                messages=[Message(role="user", content="一"),
                          Message(role="assistant", content="答一")],
                citations=[],
            )
            await append_turn(          # 缺省(等价于 None)
                session=session, conversation_id=SCRATCH,
                messages=[Message(role="user", content="二"),
                          Message(role="assistant", content="答二")],
            )

    asyncio_run(run())
    rows = asyncio_run(_stored_rows())
    assert [r.content for r in rows] == ["一", "答一", "二", "答二"]
    assert all(r.citations is None for r in rows), [r.citations for r in rows]


def test_advance_anchors_moves_only_the_anchor_it_is_given():
    """`advance_anchors` 是**部分更新**:没传的那个锚点必须原样不动。

    这条不是讲究,是防止一次灾难:`layer1_from` 是 T10 降级路径的常用入参,
    而实现若写成 `values(summary_upto_msg_id=summary_upto or 0, …)`,
    「只推层 1 边界」会把 `summary_upto_msg_id` **打回 0** ——
    梗概行还在表里,但锚点说「还没压过」⇒ 那段原文被**再压一遍**,
    每段单看都正常,只有梗概表里悄悄多出一份重复。
    """
    async def run():
        async with get_sessionmaker()() as session:
            await ensure_conversation(session=session, session_id=SCRATCH, user_id="u")
            conv = (
                await session.execute(
                    select(Conversation).where(Conversation.id == SCRATCH)
                )
            ).scalars().one()
            conv.summary_upto_msg_id = 5
            await session.commit()

        # ① 只给 layer1_from:summary_upto 不许动
        async with get_sessionmaker()() as session:
            await advance_anchors(
                session=session, conversation_id=SCRATCH, layer1_from=9
            )
        async with get_sessionmaker()() as session:      # 新 session 读回
            conv = (
                await session.execute(
                    select(Conversation).where(Conversation.id == SCRATCH)
                )
            ).scalars().one()
            assert conv.summary_upto_msg_id == 5
            assert conv.layer1_from_msg_id == 9

        # ② 只给 summary_upto:反向同样不许动
        async with get_sessionmaker()() as session:
            await advance_anchors(
                session=session, conversation_id=SCRATCH, summary_upto=6
            )
        async with get_sessionmaker()() as session:
            conv = (
                await session.execute(
                    select(Conversation).where(Conversation.id == SCRATCH)
                )
            ).scalars().one()
            assert conv.summary_upto_msg_id == 6
            assert conv.layer1_from_msg_id == 9          # ← 没被 or 0 打回

    asyncio_run(run())


def test_append_summary_and_advance_is_atomic():
    """落梗概与推进锚点要么都成、要么都不成。

    只成一半的两种后果都很难看:
    - 梗概落了锚点没推 ⇒ 同一段原文被**再压一遍**(重复梗概,每段单看都正常);
    - 锚点推了梗概没落 ⇒ 那段历史**永久消失**(区间已不在层 2 读取范围,
      而摘要表里没有替换物)。

    **两个方向都断言**:只断「成功时两者都在」区分不出「先落梗概再推锚点、
    中间抛了」的实现。所以再构造一次**提交时被数据库拒绝**的失败,断言
    **锚点没有被推进、失败那条一行都没落**。

    ⚠️ 失败是怎么造出来的(与计划文本不同,理由必须写下来):
    计划里写的是「预置一条 `seq=2`,再让函数去撞它」。**那条撞不上** ——
    实现取的是 `COALESCE(MAX(seq), 0) + 1`,而唯一键是
    `(conversation_id, seq)`;只要 seq 由 max+1 算得,它就**永远是一个空位**,
    预置任何一行都只会把 max 抬高、让函数算出更大的值。实测:预置 seq=2 后
    函数插的是 seq=3,不抛任何错,而计划里那句 `pytest.raises(IntegrityError)`
    会以「DID NOT RAISE」红掉。(这不是缺陷:唯一键本来就只在**真并发**
    下才会响,那正是它存在的意义。)所以这里改用一条**确定会失败**的写入:
    `content` 是 `NOT NULL`,给 None 必然被 MySQL 以 1048 拒绝 ——
    错误仍然发自**这次事务内部的 INSERT**,要验的「要么都成、要么都不成」
    一字未变。(`ConversationSummary.content` 也因此**故意不设** Python 侧
    `default`;谁给它加上默认值,这里的注入会以「DID NOT RAISE」红掉,
    指回 `app/db/models.py` 那段注释。)

    **实测到的机制,以及本用例证不到的那一半**(不写下来就会被读成更强的结论):
    SQLAlchemy 的 autoflush 会在 `session.execute(update(...))` 之前先把待写的
    INSERT 发出去(已用 `before_cursor_execute` 实测:发出的是 INSERT,UPDATE
    **一次都没发**)。所以「锚点没被推进」是**那一步根本没执行**,不是
    「回滚把一条已经发出的 UPDATE 撤了回来」。

    判别力的边界(**用变异测试逐条量过**):
    - 「先提交锚点、再落梗概」两次 commit 的实现 ⇒ **红**,抓得到;
    - 「先提交梗概、再单独提交锚点」两次 commit 的实现 ⇒ **绿**,抓不到 ——
      它的第一步就是失败点,于是「两次提交」与「一次提交」留下的**状态完全
      相同**。数据库层的失败注入区分不了这一个:要区分,失败必须落在第二步
      (UPDATE)上,而两步写的是两个同型 BIGINT 列、值还相同;唯一键则因为
      seq 由 `MAX(seq)+1` 算得而在无并发时**永不响**(它本来就是为并发而生)。
      如实记账,不假装覆盖了。
    """
    async def run():
        async with get_sessionmaker()() as session:
            await ensure_conversation(session=session, session_id=SCRATCH, user_id="u")

        # ① 成功路径:两步都在
        async with get_sessionmaker()() as session:
            await append_summary_and_advance(
                session=session, conversation_id=SCRATCH,
                upto_msg_id=7, content="用户报过订单 1002,尚未解决。",
            )
        async with get_sessionmaker()() as session:
            conv = (await session.execute(
                select(Conversation).where(Conversation.id == SCRATCH)
            )).scalars().one()
            rows = await load_summaries(session=session, conversation_id=SCRATCH)
            assert conv.summary_upto_msg_id == 7
            assert [c for _, c in rows] == ["用户报过订单 1002,尚未解决。"]

        # ② 失败路径:让这次事务里的插入被数据库拒绝 ⇒ 整体回滚,锚点不动
        async with get_sessionmaker()() as session:
            with pytest.raises(IntegrityError):
                await append_summary_and_advance(
                    session=session, conversation_id=SCRATCH,
                    upto_msg_id=42, content=None,     # NOT NULL ⇒ 必然被拒
                )

        async with get_sessionmaker()() as session:
            conv = (await session.execute(
                select(Conversation).where(Conversation.id == SCRATCH)
            )).scalars().one()
            rows = await load_summaries(session=session, conversation_id=SCRATCH)
            assert conv.summary_upto_msg_id == 7          # ← 没有被推到 42
            assert len(rows) == 1                          # ← 失败那条一行都没落
            assert [c for _, c in rows] == ["用户报过订单 1002,尚未解决。"]

    asyncio_run(run())
