"""`/api/conversations` 的 **SQL 语义**在真实库上验(需要 MySQL 在跑)。

**为什么单列一个文件**:`tests/test_api_conversations.py` 的替身**替端点把事做了** ——
它的列表分支自己就按 `created_at` 倒序排,也能自己按 user 筛,于是端点把整个
`.where()` / `.order_by()` 删掉**照样绿**。靠替身验「端点有没有传对 SQL」是验不出来的
(本仓 ch03 的同类教训:`VARCHAR 主键必须显式传 max_length` 那条,假 client 就测不出,
单测绿了还得真机冒烟)。这里三条语义各自造出**能被观测**的输入:

1. **顺序** —— 两个探针会话 `created_at` 显式不同,**且 id 的字典序与时间先后相反**
   (MySQL 无 ORDER BY 时通常按主键/聚簇索引序返回 ⇒ 漏了 `order_by` 也会红,
   而不是「碰巧被插入序蒙对」);
2. **`user` 过滤** —— 一个 `user='someone-else'` 的探针,**同时**要有一个自己的会话
   在场:只断「别人的不在」的话,「端点返回空列表」也满足它;
3. **`summarized` 读锚点** —— 一个**有 `conversation_summaries` 行、但锚点为 0** 的
   会话(「数梗概行数」的实现会说是 True;正确答案是 False),外加一个反向对照
   (锚点 > 0、**没有**梗概行 ⇒ True)。**「两者在正常情况下一致」正是假绿最爱藏身的
   地方,所以这里用真实库造出两者不一致的那个输入。**

⚠️ **探针行用完即删,且删除必须 `commit`** —— `async with session` 退出是 rollback,
只 `execute(delete(...))` 不提交的话,下一轮跑就会看到上一轮的残留。
"""

from datetime import datetime

import pytest
from sqlalchemy import delete, func, select

from app.api.conversations import list_conversations
from app.db.base import get_engine, get_sessionmaker
from app.db.models import Conversation, ConversationSummary, MessageRecord

pytestmark = pytest.mark.db

#: 探针 id 固定且可辨识(<=32 字符)。**`t11probe-` 前缀不会撞真实数据**:
#: 库里 322 个会话的 id 是 `uuid4().hex` 与 `acceptance-*`。
PROBE_PREFIX = "t11probe-"
PROBE_ORDER_OLD = PROBE_PREFIX + "a-old"      # 早,但 id 字典序在前
PROBE_ORDER_NEW = PROBE_PREFIX + "z-new"      # 晚,但 id 字典序在后
PROBE_FOREIGN = PROBE_PREFIX + "foreign"
PROBE_ANCHOR_ZERO = PROBE_PREFIX + "anchor-zero"
PROBE_ANCHOR_SET = PROBE_PREFIX + "anchor-set"
PROBE_IDS = [
    PROBE_ORDER_OLD, PROBE_ORDER_NEW, PROBE_FOREIGN,
    PROBE_ANCHOR_ZERO, PROBE_ANCHOR_SET,
]

#: 三个时刻显式给死:`created_at` 列是 `DateTime`(秒精度),两条 `func.now()`
#: 完全可能落在同一秒上 —— 那样「新在前」的断言就在比较两个相等的值,零判别力。
T_OLD = datetime(2026, 9, 22, 8, 0, 0)
T_NEW = datetime(2026, 9, 22, 8, 0, 1)
T_FOREIGN = datetime(2026, 9, 22, 8, 0, 2)     # 最新:漏过滤时它顶到第一位


def _conv(conv_id, user, created_at, *, summary_upto_msg_id=0, layer1_from_msg_id=0):
    return Conversation(
        id=conv_id, user=user, status="active", created_at=created_at,
        summary_upto_msg_id=summary_upto_msg_id, layer1_from_msg_id=layer1_from_msg_id,
    )


async def _cleanup() -> None:
    """删探针行。**顺序:先消息、再梗概、最后会话**(会话是别人的 FK 目标)。"""
    async with get_sessionmaker()() as session:
        await session.execute(
            delete(MessageRecord).where(MessageRecord.conversation_id.in_(PROBE_IDS))
        )
        await session.execute(
            delete(ConversationSummary).where(
                ConversationSummary.conversation_id.in_(PROBE_IDS)
            )
        )
        await session.execute(delete(Conversation).where(Conversation.id.in_(PROBE_IDS)))
        await session.commit()


async def _insert(rows) -> None:
    async with get_sessionmaker()() as session:
        session.add_all(rows)
        await session.commit()


async def _list_items() -> list[dict]:
    """**新 session** 读(身份映射里的旧对象会让断言变成「靠 refcount 走运」)。"""
    async with get_sessionmaker()() as session:
        return (await list_conversations(session=session))["items"]


@pytest.mark.anyio
async def test_list_orders_newest_first_on_a_real_table():
    """顺序由端点的 `ORDER BY created_at DESC` 决定 —— 替身验不出(它自己排)。

    **id 字典序与时间先后相反**(`a-old` 早、`z-new` 晚):MySQL 在无 ORDER BY 时
    通常按主键序返回,于是「端点忘了排序」得到的是**旧在前**,而不是被插入序蒙对。
    """
    await _cleanup()
    try:
        await _insert([
            _conv(PROBE_ORDER_OLD, "demo-user", T_OLD),
            _conv(PROBE_ORDER_NEW, "demo-user", T_NEW),
        ])
        ids = [i["id"] for i in await _list_items()]
        # 前提:两个探针都真的被读到(否则下面那句在「端点什么都没返回」时也绿)
        assert PROBE_ORDER_NEW in ids and PROBE_ORDER_OLD in ids, (
            f"两个探针会话都该在列表里,实际拿到 {len(ids)} 条、探针缺失"
        )
        assert ids.index(PROBE_ORDER_NEW) < ids.index(PROBE_ORDER_OLD), (
            f"新在前:{PROBE_ORDER_NEW}({T_NEW})应排在 "
            f"{PROBE_ORDER_OLD}({T_OLD})之前,实际 index "
            f"{ids.index(PROBE_ORDER_NEW)} vs {ids.index(PROBE_ORDER_OLD)}"
        )
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.anyio
async def test_list_excludes_other_users_on_a_real_table():
    """`WHERE user = 'demo-user'` 真的落到了 SQL 上(替身那一支是替身自己筛的)。

    **两个断言缺一不可**:只断「别人的不在」的话,一个恒返回空列表的实现照样绿;
    所以同一次请求里自己的会话**必须也在**。别人的那条给**最新**的 `created_at`
    —— 漏过滤时它排第一位,与「多出来一条」在第一个元素上就分开。
    """
    await _cleanup()
    try:
        await _insert([
            _conv(PROBE_ORDER_NEW, "demo-user", T_NEW),
            _conv(PROBE_FOREIGN, "someone-else", T_FOREIGN),
        ])
        ids = [i["id"] for i in await _list_items()]
        assert PROBE_ORDER_NEW in ids, "自己的会话必须在(否则下一条断言恒真)"
        assert PROBE_FOREIGN not in ids, (
            f"别人的会话泄漏进了列表:{PROBE_FOREIGN}(它在结果里的第 "
            f"{ids.index(PROBE_FOREIGN) if PROBE_FOREIGN in ids else -1} 位)"
        )
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.anyio
async def test_summarized_reads_the_anchor_not_the_summary_rows():
    """`summarized` 读 `summary_upto_msg_id > 0`,**不是**「有没有梗概行」。

    这条输入是**真实库里唯一能把两者分开的那种**:`anchor-zero` 有一行
    `conversation_summaries`(所以「数行数」会说 True),而它的锚点是 0(正确答案
    False);`anchor-set` 反过来 —— 锚点 7、**一行梗概都没有**(数行数会说 False)。

    「梗概行真的写进去了」这一步**单独验**(新 session 数一次):不然插入失败时
    这条用例会退化成「两个会话都没有梗概行」,而它对「数行数」的实现**不再有判别力**
    (第 5 类假绿的输入不足)。
    """
    await _cleanup()
    try:
        await _insert([
            _conv(PROBE_ANCHOR_ZERO, "demo-user", T_OLD, summary_upto_msg_id=0),
            _conv(PROBE_ANCHOR_SET, "demo-user", T_NEW,
                  summary_upto_msg_id=7, layer1_from_msg_id=9),
            ConversationSummary(
                conversation_id=PROBE_ANCHOR_ZERO, seq=1, upto_msg_id=5,
                content="这一段是梗概,覆盖到 messages.id=5。",
            ),
        ])
        async with get_sessionmaker()() as session:
            n_rows = (
                await session.execute(
                    select(func.count())
                    .select_from(ConversationSummary)
                    .where(ConversationSummary.conversation_id == PROBE_ANCHOR_ZERO)
                )
            ).scalar()
        assert n_rows == 1, f"探针梗概行应恰好 1 行,实际 {n_rows}"

        items = {i["id"]: i for i in await _list_items()}
        assert items[PROBE_ANCHOR_ZERO]["summarized"] is False, (
            "锚点是 0 ⇒ 没有梗概覆盖到任何一条;有梗概行也不该说 True"
        )
        assert items[PROBE_ANCHOR_SET]["summarized"] is True, (
            "锚点是 7 ⇒ 已有梗概覆盖;没有梗概行也不该说 False"
        )
    finally:
        await _cleanup()
        await get_engine().dispose()
