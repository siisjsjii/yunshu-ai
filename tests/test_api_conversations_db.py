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
from sqlalchemy import delete, func, select, text

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


def _msg(conversation_id, role, content, *, tool_calls=None, citations=None):
    """探针消息。`id` 由自增给,**逐条插入**以保证 id 顺序 = 插入顺序。"""
    return MessageRecord(
        conversation_id=conversation_id, role=role, content=content,
        tool_calls=tool_calls, citations=citations, created_at=T_OLD,
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
        # 4 行 → 2 项:空气泡那条(第 2 行)**不作为独立 item 出现**,
        # 它的齿轮归并到了本轮的答案上(2026-09-27 修复)。
        assert len(items) == 2, items
        assert [tc["name"] for tc in items[1]["tool_calls"]] == ["query_order"]
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.anyio
async def test_a_react_turn_merges_into_one_item_on_a_real_table():
    """**真实 ReAct 形状的一轮**在真库上:4 行 → **1 条 assistant 项**,
    且那一条**既有答案、又带两个齿轮**。

    这是这次修复的核心读数(实测 336 条带齿轮的 assistant 行里 303 条
    `content` 为空 ⇒ 不归并的话那 303 个齿轮**全部**回不来)。

    **判别力**:
    · 不做归并 ⇒ 出 3 项(空气泡、答案各一条 + user),`len(items) == 2` 红;
    · 归并时**丢掉**被归并行的 `tool_calls` ⇒ `tool_calls == tool_calls_used` 红;
    · 归并时**多算**(把答案行自己那份也算两遍)⇒ 长度 3 的断言红。

    ⚠️ 两个齿轮**分别来自两个不同的空 content 行**(生产上确实会:一轮里可以
    申请调两次工具),这样「只认最后一条」的写法也会红(`query_order` 会丢)。
    """
    tool_calls_used = [
        {"id": "c1", "name": "query_order", "args": {"order_no": "1002"},
         "type": "tool_call"},
        {"id": "c2", "name": "query_product", "args": {"product_id": "p1"},
         "type": "tool_call"},
    ]
    await _cleanup()
    try:
        await _insert([_conv(PROBE_MSGS, "demo-user", T_NEW)])
        await _insert([_msg(PROBE_MSGS, "user", "订单 1002 能退吗")])
        await _insert([_msg(PROBE_MSGS, "assistant", "",
                            tool_calls=tool_calls_used[:1])])
        await _insert([_msg(PROBE_MSGS, "tool", '{"status":"已取消"}')])
        await _insert([_msg(PROBE_MSGS, "assistant", "",
                            tool_calls=tool_calls_used[1:])])
        await _insert([_msg(PROBE_MSGS, "tool", '{"product_id":"p1"}')])
        await _insert([_msg(PROBE_MSGS, "assistant", "按政策可以退[1]。")])

        items = await _messages(PROBE_MSGS)
        assert [i["role"] for i in items] == ["user", "assistant"], (
            f"6 行应当归并成 2 项,实际 {[(i['role'], i['content']) for i in items]}"
        )
        assert items[1]["content"] == "按政策可以退[1]。"
        assert items[1]["tool_calls"] == tool_calls_used, items[1]["tool_calls"]
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.anyio
async def test_two_text_bearing_assistant_rows_stay_two_items_on_a_real_table():
    """**反着断**(真库):归并**不许**把两条**本来就有正文**的 assistant 行并成一条。

    并了 = 凭空抹掉一条用户看见过的回复,而它**不报错** ——
    屏幕上只是少了一句话。归并的判据是「该行有没有正文」,不是「一轮里有几条」。
    """
    await _cleanup()
    try:
        await _insert([_conv(PROBE_MSGS, "demo-user", T_NEW)])
        await _insert([_msg(PROBE_MSGS, "user", "订单 1002 到哪了")])
        await _insert([_msg(PROBE_MSGS, "assistant", "这就为您查询。")])
        await _insert([_msg(PROBE_MSGS, "assistant", "这一单已取消")])

        items = await _messages(PROBE_MSGS)
        assert [i["content"] for i in items] == ["订单 1002 到哪了", "这就为您查询。",
                                                 "这一单已取消"], items
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.anyio
async def test_a_turn_with_only_gears_emits_one_item_on_a_real_table():
    """**边界**(真库):一轮只有齿轮、**没有**带正文的 assistant 行 ⇒
    仍然输出一条 `content: ""` 带 `tool_calls` 的项。

    生产形状 = 「模型申请了工具,那一轮随后报错/没产出正文」
    (`app/agent/nodes.py` 先落带 `tool_calls` 的空行,收尾那条根本没写出来)。
    丢掉它 = 用户回载时连齿轮都看不见,而直播时他明明看见过。

    **判别力**:把「孤立项」那一支删掉 ⇒ `len(items)` 从 2 变 1,当场红。
    """
    await _cleanup()
    try:
        await _insert([_conv(PROBE_MSGS, "demo-user", T_NEW)])
        await _insert([_msg(PROBE_MSGS, "user", "订单 1002 到哪了")])
        await _insert([_msg(PROBE_MSGS, "assistant", "", tool_calls=[
            {"id": "c1", "name": "query_logistics", "args": {"order_no": "1002"},
             "type": "tool_call"}
        ])])

        items = await _messages(PROBE_MSGS)
        assert len(items) == 2, items
        assert items[1]["role"] == "assistant"
        assert items[1]["content"] == "", items[1]
        assert [tc["name"] for tc in items[1]["tool_calls"]] == ["query_logistics"]
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.anyio
async def test_empty_content_without_tool_calls_is_not_fetched_on_a_real_table():
    """`content=''` 且 `tool_calls` 是**字面 JSON `null`** 的行 ⇒ **不产生任何项**。

    ⚠️ **这条不是「另一种写法的杀手」,别把它读成那样**(如实记):端点把
    「这一行有没有工具调用」判成 `JSON_TYPE(tool_calls) = 'ARRAY'`,而换成
    `IS NOT NULL` 或 `JSON_LENGTH(...) > 0` 时,这样一行**会**被 SQL 捞出来 ——
    可是它在归并里是**空操作**(没有齿轮可累积)⇒ **API 输出逐字相同**。
    本机实测(2026-09-27)也证实了这件事:本查询那三种判据取到的行数**都是 2238**,
    因为「`content=''` 且 tool_calls 不是 ARRAY」的行在真实库里**一行都没有**。

    ⇒ 这条用例守的是**归并的边界**(空行不许单独冒出来),**不是**那条判据。
    判据的形状由 `tests/test_api_conversations.py` 的替身钉(它只认 `JSON_TYPE`),
    判据的**真库语义**由下面那条用例钉。
    """
    await _cleanup()
    try:
        await _insert([_conv(PROBE_MSGS, "demo-user", T_NEW)])
        await _insert([_msg(PROBE_MSGS, "user", "在吗")])
        # 生产形状:`_lc_to_records` 对没有工具调用的行写 `tool_calls=None`
        # ⇒ JSON 列里是**字面 JSON `null`**,不是 SQL NULL。
        await _insert([_msg(PROBE_MSGS, "assistant", "", tool_calls=None)])

        items = await _messages(PROBE_MSGS)
        assert [i["content"] for i in items] == ["在吗"], items
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.anyio
async def test_json_type_is_the_only_safe_predicate_on_a_real_table():
    """**真库**上核 `JSON_TYPE` 这条判据为什么唯一安全 —— 三种写法各是多少行。

    这是「库 X 在情况 Y 下表现 Z」那一类断言**必须带可复现证据**的那条规矩:
    证据就是下面这几条**当场跑出来的**读数(本机 2026-09-27 实测:2721 / 336 / 2721)。
    它钉住的是**数据库的语义**:JSON 的 `null` 是一个**标量**,
    `JSON_LENGTH` 对它返回 **1**(所以 `> 0` 恒真),而它在 SQL 上**不是 NULL**
    (所以 `IS NOT NULL` 为真)。同族陷阱见 ch09 的 `evidence_snapshot`
    与 `app/db/models.py` 的 `citations` 列注释。

    ⚠️ **这条红了不代表端点坏了** —— 先看 `db/followup_messages_citations.sql`
    与 `app/api/conversations.py` 的 `JSON_ARRAY` 那段:它断的是**我们对 MySQL 的
    认识**。真变了(比如升到某个版本 `JSON_TYPE` 对 JSON null 返回别的值)就要
    重新裁定那个判据。
    """
    await _cleanup()
    try:
        await _insert([_conv(PROBE_MSGS, "demo-user", T_NEW)])
        await _insert([_msg(PROBE_MSGS, "assistant", "", tool_calls=None)])   # JSON null
        await _insert([_msg(PROBE_MSGS, "assistant", "有齿轮", tool_calls=[
            {"id": "c1", "name": "query_order", "args": {}, "type": "tool_call"}
        ])])

        async with get_sessionmaker()() as session:
            def count(where: str):
                return select(func.count()).select_from(MessageRecord).where(
                    MessageRecord.conversation_id == PROBE_MSGS, text(where)
                )

            got = {}
            for label, where in (
                ("is_not_null", "tool_calls IS NOT NULL"),
                ("json_type", "JSON_TYPE(tool_calls) = 'ARRAY'"),
                ("json_length", "JSON_LENGTH(tool_calls) > 0"),
            ):
                got[label] = (await session.execute(count(where))).scalar()

        assert got["json_type"] == 1, got          # 只有那一条真的有齿轮
        assert got["is_not_null"] == 2, got        # JSON null 也算「非 NULL」
        assert got["json_length"] == 2, got        # JSON 标量的长度是 1 ⇒ > 0 恒真
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.anyio
async def test_tool_calls_and_citations_survive_the_json_columns_on_a_real_table():
    """`tool_calls` / `citations` 两列的**JSON 往返**在真库上验(替身验不出来)。

    替身交回来的是**同一个 Python 对象**(内存里的 list),所以「列类型写错成
    TEXT」「写入时被 `str()` 序列化过一次」这类缺陷在单测里全绿 —— 而它到前端
    的形态是 `ctx.citations` 拿到一个**字符串**,`citations.find` 直接
    `TypeError`(`makeCitesClickable` 的 `if (!ctx.citations.length) return`
    只挡长度 0,不挡字符串),于是 [n] 引用点不开,而**服务端一切正常**。
    这里用**新 session** 读回(身份映射持弱引用,同 session 重读可能拿到内存里
    那个原对象,断言就变成「靠 refcount 走运」)。

    两条都**逐字比深层结构**:只断「非空」的话,`args` 内层字典被拍平成字符串
    照样绿 —— 而那正是 `test_history.py` 里同款断言存在的理由。
    """
    tool_calls = [
        {"id": "call_1", "name": "query_order", "args": {"order_no": "1002"},
         "type": "tool_call"}
    ]
    citations = [
        {"n": 1, "chunk_id": "77", "section_path": "退货退款政策 > 无理由退货",
         "question": "无理由退货的期限是多久?", "answer": "七天。", "category": "退换货"}
    ]
    await _cleanup()
    try:
        await _insert([_conv(PROBE_MSGS, "demo-user", T_NEW)])
        await _insert([_msg(PROBE_MSGS, "user", "订单 1002 能退吗")])
        await _insert([_msg(PROBE_MSGS, "assistant", "", tool_calls=tool_calls)])
        await _insert([_msg(PROBE_MSGS, "tool", '{"status":"已取消"}')])
        await _insert([_msg(PROBE_MSGS, "assistant", "按政策可以退[1]。",
                            citations=citations)])

        items = await _messages(PROBE_MSGS)
        assert [i["role"] for i in items] == ["user", "assistant"]
        # 前提:探针真的写进去了(否则下面那句在「端点恒返回空列表」时也绿)
        assert len(items) == 2, [(i["role"], i["content"]) for i in items]
        # 没有引用的那条:**是 None,不是 []**(空数组与「没引用」不可区分 ——
        # 用户在回载里会看到一个「有引用区、但点不开」的回复)
        assert items[0]["citations"] is None, items[0]["citations"]
        assert items[0]["tool_calls"] is None, items[0]["tool_calls"]
        assert items[1]["citations"] == citations, items[1]["citations"]
        # `tool_calls` 那一列也走**同一条**往返路径(它在这一版之前就存在,
        # 但那时是「原样透传某一行」;现在它经过归并 ⇒ 这条同时钉住了
        # 「归并出来的那个 list 也是从 JSON 列反序列化来的」)。
        assert items[1]["tool_calls"] == tool_calls, items[1]["tool_calls"]
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
