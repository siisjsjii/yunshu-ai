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
   (锚点 > 0、**没有**梗概行 ⇒ True),再加一个**把两个锚点分开**的探针
   (`summary=0` 而 `layer1=5`,读错列会说 True)。**「两者在正常情况下一致」正是
   假绿最爱藏身的地方,所以这里用真实库造出两者不一致的那个输入。**
4. **回载顺序 + 两类「用户没看见过的行」** —— `list_messages` 的 `.order_by(id)`
   在单测里不可观测(替身自己排);这里逐条插入真消息,并放一行**原始工具载荷**
   (`role='tool'`)与一条**只申请工具调用、content 为空**的 assistant 行
   (回给侧栏就是空气泡),顺序与过滤一起钉。
5. **预览取第一条 user 消息** —— 真实表上放「工具行 + 两条 user 消息」,
   「取最后一条」与「取第一条」给出不同答案。

⚠️ **探针行用完即删,且删除必须 `commit`** —— `async with session` 退出是 rollback,
只 `execute(delete(...))` 不提交的话,下一轮跑就会看到上一轮的残留。
"""

from datetime import datetime

import pytest
from sqlalchemy import delete, func, select

from app.api.conversations import list_conversations, list_messages
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
PROBE_ANCHOR_LAYER1 = PROBE_PREFIX + "anchor-l1"   # summary=0 而 layer1>0
PROBE_MSGS = PROBE_PREFIX + "msgs"                 # 回载顺序 + 工具行
PROBE_PREVIEW = PROBE_PREFIX + "preview"           # 预览取第一条 user
PROBE_IDS = [
    PROBE_ORDER_OLD, PROBE_ORDER_NEW, PROBE_FOREIGN,
    PROBE_ANCHOR_ZERO, PROBE_ANCHOR_SET, PROBE_ANCHOR_LAYER1,
    PROBE_MSGS, PROBE_PREVIEW,
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


def _msg(conversation_id, role, content, *, tool_calls=None):
    """探针消息。`id` 由自增给,**逐条插入**以保证 id 顺序 = 插入顺序。"""
    return MessageRecord(
        conversation_id=conversation_id, role=role, content=content,
        tool_calls=tool_calls, created_at=T_OLD,
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


async def _messages(conversation_id: str) -> list[dict]:
    async with get_sessionmaker()() as session:
        return (await list_messages(conversation_id=conversation_id, session=session))[
            "items"
        ]


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

    `anchor-l1` 是**第三个**探针,用来把两个锚点**分开**:`summary=0` 而 `layer1=5`。
    这个状态在生产上**可达** —— `layers.degrade` 推进 `layer1_from` 时**不需要
    有梗概存在** —— 而它正是「读错锚点」这个变异的唯一观测面
    (`A(0,0)` 与 `B(7,9)` 两个锚点都相关,读错列在它们身上给出同一个答案)。
    读错列的后果是:**在一个没有梗概的会话上报 `summarized: true`**。
    """
    await _cleanup()
    try:
        await _insert([
            _conv(PROBE_ANCHOR_ZERO, "demo-user", T_OLD, summary_upto_msg_id=0),
            _conv(PROBE_ANCHOR_SET, "demo-user", T_NEW,
                  summary_upto_msg_id=7, layer1_from_msg_id=9),
            _conv(PROBE_ANCHOR_LAYER1, "demo-user", T_NEW,
                  summary_upto_msg_id=0, layer1_from_msg_id=5),
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
        assert items[PROBE_ANCHOR_LAYER1]["summarized"] is False, (
            "summary=0 而 layer1=5 ⇒ 没有梗概;读成 layer1_from 会说 True"
        )
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.anyio
async def test_list_messages_is_id_ascending_and_hides_tool_rows():
    """回载**按 id 升序**(spec §5.2),且两类内部行**都不在**里面:
    `role='tool'` 的原始工具载荷、以及 `content=''` 的助手空气泡。

    替身验不出顺序:它的 messages 分支自己按 `m.id` 排 ⇒ 端点写成 `.desc()`
    (「侧栏回载变成新在前」)在单测里全绿。这里**一条一 commit 按内容顺序插**
    (自增 id 因此严格递增、与期望顺序同向),端点一旦反过来,两条断言都红。

    ⚠️ 「端点**完全没写** `order_by`」在本用例里与 `ASC` **不可区分** ——
    InnoDB 全表/索引扫描本来就按主键序返回。本用例杀的是**方向错**(`.desc()`),
    也就是单测完全看不见的那个错法。

    两类被滤掉的行都按**生产形状**造:工具行是 `[工具结果]` 那类原始载荷;
    空 content 那条 assistant **带着 `tool_calls`**(真实库实测 94 行空 content
    全是这一形态)。
    """
    await _cleanup()
    try:
        await _insert([_conv(PROBE_MSGS, "demo-user", T_NEW)])
        # 逐条插(session.add_all 不保证各行的自增 id 与列表顺序一致)
        await _insert([_msg(PROBE_MSGS, "user", "第一句")])
        # 只申请工具调用、还没产出文字的那条 assistant:content 是空串、带 tool_calls
        # (生产形状,见 `app/agent/nodes.py` 的 `content=m.content or ""`)。
        await _insert([_msg(PROBE_MSGS, "assistant", "", tool_calls=[
            {"id": "call_1", "name": "query_order", "args": {"order_no": "1002"},
             "type": "tool_call"}
        ])])
        await _insert([_msg(PROBE_MSGS, "tool", '{"order_no":"1002","status":"已取消"}')])
        await _insert([_msg(PROBE_MSGS, "assistant", "第二答")])

        items = await _messages(PROBE_MSGS)
        assert [i["role"] for i in items] == ["user", "assistant"], (
            f"工具行与空气泡都不该出现,且顺序应是插入序(== id 升序),实际 "
            f"{[(i['role'], i['content']) for i in items]}"
        )
        assert [i["content"] for i in items] == ["第一句", "第二答"]
        assert all("order_no" not in i["content"] for i in items)
        assert all(i["content"] != "" for i in items)
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.anyio
async def test_preview_uses_the_first_user_message_on_a_real_table():
    """预览取**第一条 user 消息**,在真实表上(不是替身替它排好的那种顺序)。

    探针里放:一行工具结果(**id 最小**)、然后两条内容不同的 user 消息。
    - 「取最后一条 user 消息」⇒ 拿到「最后又问的那句」,红;
    - 「取第一条消息、不看 role」⇒ 拿到工具载荷,红。
    """
    await _cleanup()
    try:
        await _insert([_conv(PROBE_PREVIEW, "demo-user", T_NEW)])
        await _insert([_msg(PROBE_PREVIEW, "tool", '{"order_no":"1003","status":"已发货"}')])
        await _insert([_msg(PROBE_PREVIEW, "user", "最先问的那句")])
        await _insert([_msg(PROBE_PREVIEW, "assistant", "先答一句")])
        await _insert([_msg(PROBE_PREVIEW, "user", "最后又问的那句")])

        item = next(i for i in await _list_items() if i["id"] == PROBE_PREVIEW)
        assert item["preview"] == "最先问的那句", (
            f"预览取的是第一条 user 消息,实际拿到 {item['preview']!r}"
        )
    finally:
        await _cleanup()
        await get_engine().dispose()
