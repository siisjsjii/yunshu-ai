"""GET /api/conversations 与 .../messages。同步用例 + TestClient。

替身与用例的骨架照 task-11-brief.md,**但每一条会话/消息都显式带了
`created_at`(会话另带 `id` 与两个锚点)**—— 这不是风格选择,是不带就**跑不起来**:

`created_at` / `summary_upto_msg_id` / `layer1_from_msg_id` 在模型上都是
`server_default`(**DB 侧**默认值),而 `MessageRecord.id` 是自增主键。手工构造的
ORM 对象**一个都还没 flush**,这些属性全是 `None`;于是
① 替身那句 `sorted(rows, key=lambda c: c.created_at)` 当场
`TypeError: '<' not supported between instances of 'NoneType' and 'NoneType'`;
② 端点里 `conv.created_at.isoformat()` 是 `AttributeError`。
两条都指向替身/端点,而真正缺的是夹具这一笔 —— 生产路径上这些列都是
`NOT NULL`(或自增),永远有值,所以**不给实现加 None 兜底**(那会凭空造出一条
「created_at 可能为 None」的静默分支),补夹具。
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.sql.elements import BinaryExpression

from app.db.models import Conversation, MessageRecord
from app.db.session import get_session
from app.main import app

CONV_A = "a" * 32
CONV_B = "b" * 32

#: 两个时刻。**T0 早、T1 晚**,且「别人的会话」用 T1 —— 列表若漏了 user 过滤,
#: 它不但会出现,还会**排在第一个**(见 test_list_...)。
T0 = datetime(2026, 9, 22, 10, 0, 0)
T1 = datetime(2026, 9, 22, 10, 0, 1)


def _conv(conv_id, user, created_at, *, summary_upto_msg_id=0, layer1_from_msg_id=0):
    """一条会话。锚点显式给 0(DB 侧 `server_default="0"`,未 flush 的对象上仍是 None)。"""
    return Conversation(
        id=conv_id,
        user=user,
        status="active",
        created_at=created_at,
        summary_upto_msg_id=summary_upto_msg_id,
        layer1_from_msg_id=layer1_from_msg_id,
    )


def _msg(msg_id, conversation_id, role, content, *, created_at=T0):
    """一条消息。`id` 显式给 —— 替身按 `m.id` 排序,全 None 会 TypeError。"""
    return MessageRecord(
        id=msg_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        created_at=created_at,
    )


class _StubSession:
    """只支撑这两个端点要用的两种读到的东西:LIST 与 (LIST, 单条)。

    刻意**不做成通用替身** —— 不支持的查询形态直接抛,而不是返回空,
    否则「端点查错了表」会退化成「返回空列表」,而那是这条用例最该发现的事。
    """

    def __init__(self, conversations, messages):
        self.conversations = conversations
        self.messages = messages

    async def execute(self, stmt, *args, **kwargs):
        cols = stmt.column_descriptions
        entity = cols[0]["entity"]
        if entity is Conversation:
            # 两种查询形态,**按比较的是哪一列**区分,不能只看「有没有 where」:
            #   `Conversation.id == <id>`   → 按 id 查单条(→ 404 那条路)
            #   `Conversation.user == <u>`  → 列表查询(还有可能什么都没有)
            # **不要把这两种混成一种**:混了之后「列表接口串了另一个用户的会话」
            # 与「详情接口查不到就 404」两条断言会互相掩盖。
            #
            # 列表这一支**要真按 user 过滤**:替身自己筛,端点漏了 `.where()`
            # 就会把别人的会话带出来(见 `_where_column`)。写成「端点筛不筛都行」
            # 的替身,等于让那条断言恒真。
            clause = _where_clause(stmt)
            if clause is not None and _where_column(clause) == "id":
                row = self.conversations.get(clause.right.value)
                return _Result([row] if row is not None else [])
            rows = list(self.conversations.values())
            if clause is not None:
                column = _where_column(clause)
                if column != "user":
                    raise AssertionError(f"替身不支持的列表过滤列:{column}")
                rows = [c for c in rows if c.user == clause.right.value]
            # ⚠️ 这一句 `sorted` 让「端点有没有写 ORDER BY」在本文件里**不可观测**
            # (替身替它排好了)—— 顺序改由 db 用例钉,见
            # `tests/test_api_conversations_db.py`。
            return _Result(sorted(rows, key=lambda c: c.created_at, reverse=True))
        if entity is MessageRecord:
            cid = _where_clause(stmt).right.value
            if _where_column(_where_clause(stmt)) != "conversation_id":
                raise AssertionError("替身只支持按 conversation_id 查消息")
            rows = [m for m in self.messages if m.conversation_id == cid]
            return _Result(sorted(rows, key=lambda m: m.id))
        raise AssertionError(f"替身不支持的实体:{entity}")


def _where_clause(stmt):
    """取 `select(...).where(col == 值)` 的那个子句;没有 where 就是 None。

    **取不出来时直接抛**,不退化成「返回全部」—— 忽略 where 的替身会让
    「列表读到了别人的会话 / 详情读到了不存在的会话」这类缺陷无从观测。
    """
    clause = stmt.whereclause
    if clause is None:
        return None
    if not isinstance(clause, BinaryExpression):
        raise AssertionError(f"替身不支持的查询形态:{stmt}")
    return clause


def _where_column(clause):
    """这个 where 比的是哪一列(`Conversation.id` / `Conversation.user` / …)。"""
    key = getattr(getattr(clause, "left", None), "key", None)
    if key is None:
        raise AssertionError(f"替身取不出被比较的列:{clause}")
    return key


class _Result:
    """`session.execute(...)` 的返回物:只需 `.scalars()` 后接 `.all()` / `.one_or_none()`。

    **本文件自带一份**,不从 `test_api_chat.py` 导入 —— 跨测试文件互相 import
    会让「这个替身到底支持什么」变得谁也说不清,而它的全部价值恰恰是
    **支持的东西被显式列出来**。
    """

    def __init__(self, rows):
        self._rows = list(rows)

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)

    def one_or_none(self):
        if len(self._rows) > 1:
            raise AssertionError(f"替身期望至多一行,拿到 {len(self._rows)}")
        return self._rows[0] if self._rows else None

    def one(self):
        if len(self._rows) != 1:
            raise AssertionError(f"替身期望恰好一行,拿到 {len(self._rows)}")
        return self._rows[0]


@pytest.fixture
def conv_client():
    """造会话与消息 → 返回 (client, session 替身)。"""
    def make(*, conversations, messages):
        session = _StubSession(conversations, messages)
        async def _override():
            yield session
        app.dependency_overrides[get_session] = _override
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()


def test_list_filters_by_demo_user_and_orders_newest_first(conv_client):
    """固定 `user='demo-user'`(无认证,前端从来不传 user_id),新在前。

    **必须放一个别的 user 的会话进去** —— 不放的话「有没有 WHERE user」
    在输出上完全一样,这条用例就恒真。

    别人的那条**故意给更晚的 `created_at`**:漏过滤时它不只是「多出来一条」,
    而是**顶到第一位**,于是 `== [CONV_A]` 在第一个元素上就红(两种错法
    ——「忘了过滤」与「过滤了但顺序反了」—— 的观测值因此不同)。

    ⚠️ 「新在前」这一半在本文件里**没有判别力**:替身的 LIST 分支自己就按
    `created_at` 倒序排,端点哪怕不写 `ORDER BY` 也照样绿 —— 替身替端点把事做了。
    顺序与 SQL 侧的过滤改由 `tests/test_api_conversations_db.py` 在**真实库**上钉
    (那边两个都会红:把 `order_by` 反过来、把 `.where()` 删掉)。
    """
    convs = {
        CONV_A: _conv(CONV_A, "demo-user", T0),
        CONV_B: _conv(CONV_B, "someone-else", T1),
    }
    client = conv_client(conversations=convs, messages=[])
    with client as c:
        items = c.get("/api/conversations").json()["items"]
    assert [i["id"] for i in items] == [CONV_A]        # ← 别人的那个不在里面


def test_preview_comes_from_the_first_user_message(conv_client):
    """预览取**第一条 user 消息**前 30 字 —— 不是最后一条,也不是条数。

    放**两条** user 消息进去,取值取错(取最后一条)就会红。
    """
    convs = {CONV_A: _conv(CONV_A, "demo-user", T0)}
    msgs = [
        _msg(1, CONV_A, "user", "第一个问题"),
        _msg(2, CONV_A, "assistant", "第一个回答"),
        _msg(3, CONV_A, "user", "后面又问的那个"),
    ]
    client = conv_client(conversations=convs, messages=msgs)
    with client as c:
        items = c.get("/api/conversations").json()["items"]
    assert items[0]["preview"] == "第一个问题"


def test_preview_is_truncated_to_thirty_chars(conv_client):
    """预览是**前 30 字**,不是整条。

    补 brief 之外的这条,是因为上一条**判不了它**:那里第一条 user 消息只有
    5 个字,「截了」与「没截」的输出**一模一样**(本仓第 5 类假绿:输入小到
    触发不了被测行为)。这里的第一条 user 消息 40 个字,`[:30]` 与整条
    必然不同。
    """
    convs = {CONV_A: _conv(CONV_A, "demo-user", T0)}
    first = "退" + "货" * 39            # 40 字
    msgs = [_msg(1, CONV_A, "user", first), _msg(2, CONV_A, "user", "第二条")]
    client = conv_client(conversations=convs, messages=msgs)
    with client as c:
        items = c.get("/api/conversations").json()["items"]
    assert items[0]["preview"] == "退" + "货" * 29      # == first[:30]
    assert len(items[0]["preview"]) == 30
    assert items[0]["preview"] != first                 # ← 整条回出去就会红


def test_summarized_flag_reflects_the_anchor_not_the_row_count(conv_client):
    """`summarized` 读的是 `summary_upto_msg_id > 0`,**不是**「有没有梗概行」。

    两者在正常情况下一致 —— 而「正常情况下一致」正是假绿测试最爱藏身的地方。
    这条构造一个**只有锚点为 0 才是正确答案**的输入:两个会话里,
    A 的锚点是 0、B 的是 12;若实现改去数梗概行数,两个都会是 False,
    于是 B 那条断言变红。

    两个会话都**不带任何消息、也没有梗概表** —— 「数梗概行数」那种实现
    在这里数出 0 行,A 与 B 都会是 False,B 那行因此必红。
    """
    convs = {
        CONV_A: _conv(CONV_A, "demo-user", T0, summary_upto_msg_id=0, layer1_from_msg_id=0),
        CONV_B: _conv(CONV_B, "demo-user", T0, summary_upto_msg_id=12, layer1_from_msg_id=20),
    }
    client = conv_client(conversations=convs, messages=[])
    with client as c:
        items = {i["id"]: i for i in c.get("/api/conversations").json()["items"]}
    assert items[CONV_A]["summarized"] is False
    assert items[CONV_B]["summarized"] is True       # ← 这条把「数行数」判死


def test_messages_endpoint_returns_raw_text_not_truncated(conv_client):
    """回载的是**原文** —— 侧栏切回来要看的就是当初聊了什么。

    放一条长到**必然**会被层 2 截短的消息进去。拿截短版回载的话,
    `…` 会出现在响应里 —— 层 2 的渲染就泄漏到 UI 上了。

    「长到必然被截」是这条用例成立的前提:层 2 对 assistant 的额度是
    `settings.layer2_assistant_chars`(**默认 50** 字,`app/config.py`),
    而下面这条 360 字,截短**一定**会触发(短消息会让原文与截短版
    逐字相同,断言随之恒真 —— 本仓第 5 类假绿)。
    """
    convs = {CONV_A: _conv(CONV_A, "demo-user", T0)}
    long_reply = "这是一条很长的客服答复。" * 30      # 12 × 30 = 360 字 > 50
    assert len(long_reply) > 50
    msgs = [_msg(1, CONV_A, "assistant", long_reply)]
    client = conv_client(conversations=convs, messages=msgs)
    with client as c:
        items = c.get(f"/api/conversations/{CONV_A}/messages").json()["items"]
    assert items[0]["content"] == long_reply          # ← 逐字相等
    assert "…" not in items[0]["content"]


def test_messages_endpoint_404s_for_unknown_conversation(conv_client):
    """不存在的会话 id → 404,文案是 spec §5.2 那一句。

    为什么要连 `detail` 一起断言:`mount("/")` 的静态目录**本来就会**对
    未知路径回 404 —— 只看状态码的话,「端点根本没实现」与「端点实现了并
    正确 404」给出**同一个观测值**(任务未开始时这条用例就是绿的)。
    文案把那两种情况分开:静态目录回的是 `{"detail":"Not Found"}`。
    """
    client = conv_client(conversations={}, messages=[])
    with client as c:
        resp = c.get(f"/api/conversations/{CONV_A}/messages")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "会话不存在"
