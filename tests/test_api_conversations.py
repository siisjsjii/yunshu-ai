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
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList

from app.db.models import Conversation, MessageRecord
from app.db.session import get_session
from app.main import app

CONV_A = "a" * 32
CONV_B = "b" * 32
CONV_C = "c" * 32

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
        terms = _where_terms(stmt)
        if entity is Conversation:
            # 两种查询形态,**按比较的是哪一列**区分,不能只看「有没有 where」:
            #   `Conversation.id == <id>`   → 按 id 查单条(→ 404 那条路)
            #   `Conversation.user == <u>`  → 列表查询(还可能什么都没有)
            # **不要把这两种混成一种**:混了之后「列表接口串了另一个用户的会话」
            # 与「详情接口查不到就 404」两条断言会互相掩盖。
            #
            # 列表这一支**真按 where 里写的条件筛**:端点漏了 `.where()`、或把条件
            # 写错列,都会在断言上现形。写成「端点筛不筛都行」的替身,等于让
            # 那条断言恒真。
            by_id = [t for t in terms if t[0] == "id"]
            if by_id:
                if len(terms) != 1 or by_id[0][1] != "eq":
                    raise AssertionError(f"替身不支持复合的 id 查询:{terms}")
                row = self.conversations.get(by_id[0][2])
                return _Result([row] if row is not None else [])
            rows = [c for c in self.conversations.values() if _matches(c, terms)]
            # ⚠️ 这一句 `sorted` 让「端点有没有写 ORDER BY」在本文件里**不可观测**
            # (替身替它排好了)—— 顺序改由 db 用例钉,见
            # `tests/test_api_conversations_db.py`。
            return _Result(sorted(rows, key=lambda c: c.created_at, reverse=True))
        if entity is MessageRecord:
            # 消息这一支同理:**替身自己按 where 筛**(`conversation_id == …` 与
            # `role != 'tool'` 都在这里生效),但排序仍然是替身做的 ⇒
            # 「按 id 升序」在单测里不可观测,同样由 db 用例钉。
            rows = [m for m in self.messages if _matches(m, terms)]
            return _Result(sorted(rows, key=lambda m: m.id))
        raise AssertionError(f"替身不支持的实体:{entity}")


def _where_terms(stmt) -> list[tuple[str, str, object]]:
    """把 `.where(...)` 拆成 `[(列名, 比较符, 值)]`;没有 where 就是 `[]`。

    支持的**形态**只有「列 `==` 值」与「列 `!=` 值」,多个条件按 AND(这也是本仓
    端点实际用到的全部)。**取不出来时直接抛**,不退化成「返回全部」—— 忽略 where
    的替身会让「列表读到了别人的会话」「详情读到了不存在的会话」「工具行漏进了
    回载」这类缺陷统统无从观测。
    """
    clause = stmt.whereclause
    if clause is None:
        return []
    parts = list(clause.clauses) if isinstance(clause, BooleanClauseList) else [clause]
    terms: list[tuple[str, str, object]] = []
    for part in parts:
        if not isinstance(part, BinaryExpression):
            raise AssertionError(f"替身不支持的查询形态:{stmt}")
        key = getattr(getattr(part, "left", None), "key", None)
        op = getattr(getattr(part, "operator", None), "__name__", None)
        if key is None or op not in ("eq", "ne"):
            raise AssertionError(f"替身不支持的比较:{part}")
        terms.append((key, op, part.right.value))
    return terms


def _matches(obj, terms) -> bool:
    """按 `_where_terms` 的结果判一行是否命中(列名即 ORM 属性名)。"""
    for key, op, value in terms:
        actual = getattr(obj, key)          # 列名写错 ⇒ AttributeError,响亮地炸
        if (actual == value) is (op == "ne"):
            return False
    return True


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


def test_list_filters_by_demo_user(conv_client):
    """固定 `user='demo-user'`(无认证,前端从来不传 user_id)。

    **必须放一个别的 user 的会话进去** —— 不放的话「有没有 WHERE user」
    在输出上完全一样,这条用例就恒真。

    别人的那条**故意给更晚的 `created_at`**:漏过滤时它不只是「多出来一条」,
    而是**顶到第一位**,于是 `== [CONV_A]` 在第一个元素上就红。

    ⚠️ **名字里不再有「orders newest first」**(订正:原先的名字高报了)。
    「新在前」在本文件里**没有判别力** —— 替身的 LIST 分支自己就按 `created_at`
    倒序排,端点哪怕不写 `ORDER BY` 也照样绿(替身替端点把事做了)。
    顺序改由 `tests/test_api_conversations_db.py` 在**真实库**上钉(把 `order_by`
    反过来那边会红);过滤也在那边再钉一次(把 `.where()` 删掉)。
    """
    convs = {
        CONV_A: _conv(CONV_A, "demo-user", T0),
        CONV_B: _conv(CONV_B, "someone-else", T1),
    }
    client = conv_client(conversations=convs, messages=[])
    with client as c:
        items = c.get("/api/conversations").json()["items"]
    assert [i["id"] for i in items] == [CONV_A]        # ← 别人的那个不在里面
    # 一条消息都没有 ⇒ 空串(spec §5.1「没有则空串」)。**必须断**:这个分支
    # 每个探针都会走到,但没人断它 —— 返回 `None` 会违反声明的 str 契约,
    # 而当时九条用例全绿(形态 (a):不设这一句,「没有」与「空串」就没人分开)。
    assert items[0]["preview"] == ""


def test_preview_comes_from_the_first_user_message(conv_client):
    """预览取**第一条 user 消息**前 30 字 —— 不是第一条消息,也不是最后一条。

    放**两条** user 消息 + **一行 `role='tool'` 的工具结果**进去:
    - 「取最后一条 user 消息」⇒ 拿到「后面又问的那个」,红;
    - 「取第一条消息、不看 role」⇒ 拿到工具载荷,红(工具结果 ch07 起真的落表)。

    ⚠️ **本文件判不了「顺序错了」**:替身的 messages 分支自己按 `m.id` 排,
    端点把 `order_by` 反过来照样绿 ⇒ 「取第一条 user 消息」在真实库上由
    `tests/test_api_conversations_db.py` 单独钉(那边插入顺序与 id 序相反)。
    """
    convs = {CONV_A: _conv(CONV_A, "demo-user", T0)}
    msgs = [
        _msg(1, CONV_A, "tool", '{"order_no":"1002","status":"已取消"}'),
        _msg(2, CONV_A, "user", "第一个问题"),
        _msg(3, CONV_A, "assistant", "第一个回答"),
        _msg(4, CONV_A, "user", "后面又问的那个"),
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

    三个会话都**不带任何消息、也没有梗概表** —— 「数梗概行数」那种实现
    在这里数出 0 行,A 与 C 都会是 False,B 那行因此必红。

    **C 是用来分开两个锚点的**(`summary=0` 而 `layer1=5`):这个状态在生产上
    **可达** —— `layers.degrade` 推进 `layer1_from` 时**不需要有梗概存在**。
    缺了它,所有夹具的两个锚点都是相关的(`(0,0)` 与 `(12,20)`),
    于是「把 `summary_upto_msg_id` 写成 `layer1_from_msg_id`」这个变异全绿 ——
    而它的真实后果是:**在一个没有梗概的会话上报 `summarized: true`**。
    """
    convs = {
        CONV_A: _conv(CONV_A, "demo-user", T0, summary_upto_msg_id=0, layer1_from_msg_id=0),
        CONV_B: _conv(CONV_B, "demo-user", T0, summary_upto_msg_id=12, layer1_from_msg_id=20),
        CONV_C: _conv(CONV_C, "demo-user", T0, summary_upto_msg_id=0, layer1_from_msg_id=5),
    }
    client = conv_client(conversations=convs, messages=[])
    with client as c:
        items = {i["id"]: i for i in c.get("/api/conversations").json()["items"]}
    assert items[CONV_A]["summarized"] is False
    assert items[CONV_B]["summarized"] is True       # ← 这条把「数行数」判死
    assert items[CONV_C]["summarized"] is False      # ← 这条把「读错锚点」判死


def test_messages_endpoint_hides_tool_rows_and_empty_assistant_bubbles(conv_client):
    """**用户没看见过的内部机制**都不回载(spec §5.2 裁定):

    ① `role='tool'` 的行 —— 原始工具载荷(`{"order_no": …}`);
    ② **`content=''` 的 assistant 行** —— 「只申请调用工具、还没产出文字」那一形态
       (`app/agent/nodes.py` 写的是 `content=m.content or ""`,它身上只有
       `tool_calls`);回给侧栏就是一个**空气泡**。

    同时断言**同一批里 user 与带文字的 assistant 仍然在**:只断「那两条不在」的话,
    一个恒返回空列表的实现也满足它。

    构造上刻意让 ② 那条**带 `tool_calls`**(生产上就是这个形状):去掉
    `content != ''` 这个条件,它就会作为一条 `content=""` 的气泡出现在结果里。
    """
    convs = {CONV_A: _conv(CONV_A, "demo-user", T0)}
    msgs = [
        _msg(1, CONV_A, "user", "订单 1002 到哪了"),
        _msg(2, CONV_A, "assistant", ""),          # ← 只申请工具调用,没有文字
        _msg(3, CONV_A, "tool", '{"order_no":"1002","status":"已取消"}'),
        _msg(4, CONV_A, "assistant", "这一单已取消"),
    ]
    msgs[1].tool_calls = [
        {"id": "call_1", "name": "query_order", "args": {"order_no": "1002"},
         "type": "tool_call"}
    ]
    client = conv_client(conversations=convs, messages=msgs)
    with client as c:
        items = c.get(f"/api/conversations/{CONV_A}/messages").json()["items"]
    assert [i["role"] for i in items] == ["user", "assistant"]
    assert [i["content"] for i in items] == ["订单 1002 到哪了", "这一单已取消"]
    assert not any("order_no" in i["content"] for i in items)
    assert not any(i["content"] == "" for i in items)


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
