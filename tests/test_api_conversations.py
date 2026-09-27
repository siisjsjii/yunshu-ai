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
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList, Grouping
from sqlalchemy.sql.functions import Function

from app.auth import ADMIN, AuthenticatedUser, require_user
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


def _msg(msg_id, conversation_id, role, content, *, created_at=T0,
         tool_calls=None, citations=None):
    """一条消息。`id` 显式给 —— 替身按 `m.id` 排序,全 None 会 TypeError。

    `tool_calls` / `citations` 默认 None = 真实库里没写过这两样的那些行
    (两列都可空),所以绝大多数用例不需要传。
    """
    return MessageRecord(
        id=msg_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        created_at=created_at,
        tool_calls=tool_calls,
        citations=citations,
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
            # 三种查询形态,**按「有没有 id 条件」分流**(不能只看「有没有 where」):
            #   `Conversation.id == <id>`                 → 按 id 查单条(→ 404)
            #   `Conversation.id == <id> AND user == <u>` → **归属**查询
            #     (`get_owned_conversation`;认证 2026-09-27 起明细端点走它)
            #   `Conversation.user == <u>`                → 列表查询(还可能什么都没有)
            # **不要把前两种混成一种**:混了之后「列表接口串了另一个用户的会话」
            # 与「详情接口查不到就 404」两条断言会互相掩盖。
            #
            # 后两种都**真按 where 里写的条件筛**(归属那一支**必须**按 user 筛):
            # 端点漏了 `.where()`、把条件写错列、或归属查了别人的会话,都会在断言上
            # 现形。写成「端点筛不筛都行」的替身,等于让那条断言恒真 ——
            # 那正是本仓记过的「替身替被测对象完成语义」。
            by_id = [t for t in terms if t[0] == "id"]
            if by_id:
                if by_id[0][1] != "eq":
                    raise AssertionError(f"替身不支持的 id 比较:{terms}")
                row = self.conversations.get(by_id[0][2])
                if row is None:
                    return _Result([])
                if len(terms) == 1:
                    return _Result([row])
                user_terms = [t for t in terms if t[0] == "user"]
                if len(terms) == 2 and len(user_terms) == 1 and user_terms[0][1] == "eq":
                    return _Result([row] if row.user == user_terms[0][2] else [])
                raise AssertionError(f"替身不支持的复合的 id 查询:{terms}")
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


def _where_terms(stmt) -> list[tuple]:
    """把 `.where(...)` 拆成一组**词项**;没有 where 就是 `[]`。词项有三种:

    · `(列名, 比较符, 值)` —— 列 `==` / `!=` 值;
    · `("json_type", 列名, "ARRAY")` —— **只认这一种函数形态**;
    · `("or", [词项, …])` —— 嵌套的 OR 组。

    多个顶层词项按 AND(这就是本仓端点实际用到的全部)。

    **取不出来时直接抛**,不退化成「返回全部」—— 忽略 where 的替身会让
    「列表读到了别人的会话」「详情读到了不存在的会话」「齿轮行漏进了回载」
    这类缺陷统统无从观测。

    ⚠️ **`json_type` 只认 `JSON_TYPE(列) == 'ARRAY'` 这一种写法,别的函数形态
    当场抛**。这是**有意**的:它把「判据用的是哪个函数」也钉住了 ——
    换成 `JSON_LENGTH(tool_calls) > 0` 或 `tool_calls IS NOT NULL` 会在这里
    直接炸(而不是悄悄绿)。⚠️ **必须说明白:这条钉的是判据的「形状」,不是
    行为** —— 在**真实数据**上那三种写法取到的行**逐位相同**(实测本机
    2026-09-27:`role <> 'tool' AND (content <> '' OR <判据>)` 三种写法都是
    **2238** 行),因为「`content=''` 且 tool_calls 不是 ARRAY」的行**一行都没有**。
    行为上的差别只在**将来**某一行写成那个形状时才会出现,而那种行在归并里
    也该被丢掉 —— 两个原因叠加,行为断言**今天写不出来**(如实记在
    `app/api/conversations.py` 的 `JSON_ARRAY` 那段)。

    ⚠️ `json_type` 这一支的语义**由替身自己实现**(`isinstance(v, list)`)——
    因为替身手里拿到的是**已经从 JSON 反序列化过的** Python 对象,不是
    MySQL 的列。⇒ 「字面 JSON `null` 不是 SQL NULL」这条**真库语义替身验不了**,
    它由 `tests/test_api_conversations_db.py` 在真库上钉。
    """
    return _parse_clause(stmt.whereclause)


def _parse_clause(clause) -> list[tuple]:
    if clause is None:
        return []
    parts = list(clause.clauses) if isinstance(clause, BooleanClauseList) else [clause]
    terms: list[tuple] = []
    for part in parts:
        if isinstance(part, Grouping):
            # ⚠️ `or_(...)` 在 SQLAlchemy 2.0 里**不是**直接一个
            # `BooleanClauseList`,而是被包了一层括号组(实测类型
            # `sqlalchemy.sql.elements.Grouping`,它的 `.element` 才是那个
            # `BooleanClauseList`)。少了这一步,端点**正确的** OR 条件会被替身
            # 当成「不支持的查询形态」—— 那是**灯下黑**:它会红在一个正确实现上。
            part = part.element
        if isinstance(part, BooleanClauseList):
            # 嵌套的 OR 组。**只支持这一层** —— 端点用到的就这一层,
            # 再深就该把替身重写成通用求值器了(而那会让「替身支持什么」
            # 变得谁也说不清,正是本文件开头那段要避免的)。
            if getattr(part.operator, "__name__", None) != "or_":
                raise AssertionError(f"替身只支持 or_ 这一种嵌套组:{part}")
            terms.append(("or", _parse_clause(part)))
            continue
        if not isinstance(part, BinaryExpression):
            raise AssertionError(f"替身不支持的查询形态:{part}")
        op = getattr(getattr(part, "operator", None), "__name__", None)
        if op not in ("eq", "ne"):
            raise AssertionError(f"替身不支持的比较:{part}")
        # ⚠️ **按类型分流,不按 `.name`** —— `Column` 也有 `.name`(它就是列名),
        # 按 `.name` 判会把 `messages.tool_calls = :x` 也当成函数调用。
        if isinstance(part.left, Function):
            if part.left.name != "JSON_TYPE":
                raise AssertionError(
                    f"替身只认 JSON_TYPE(...) 这个判据(实测另两种写法不安全):{part}"
                )
            col = list(part.left.clauses)[0]
            terms.append(("json_type", getattr(col, "key", None), part.right.value))
            continue
        key = getattr(getattr(part, "left", None), "key", None)
        if key is None:
            raise AssertionError(f"替身不支持的比较:{part}")
        terms.append((key, op, part.right.value))
    return terms


def _matches(obj, terms) -> bool:
    """按 `_where_terms` 的结果判一行是否命中(列名即 ORM 属性名)。"""
    for term in terms:
        if term[0] == "or":
            if not any(_matches(obj, [sub]) for sub in term[1]):
                return False
            continue
        if not _matches_atom(obj, term):
            return False
    return True


def _matches_atom(obj, term) -> bool:
    """单个词项。列名写错 ⇒ `getattr` 抛 AttributeError,响亮地炸。"""
    if term[0] == "json_type":
        _, key, value = term
        # 替身这边拿到的是**已反序列化**的值:`list` 就是 JSON 的 ARRAY,
        # 别的一律当「不是 ARRAY」(`None` 就是 JSON 的 `null`,见 `_where_terms`)。
        return ("ARRAY" if isinstance(getattr(obj, key), list) else "NULL") == value
    key, op, value = term
    actual = getattr(obj, key)
    # 原来的语义:`op == "ne"` 时「相等」要判 False;`eq` 时相反。
    return (actual == value) is not (op == "ne")


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
        # 认证(2026-09-27):本文件的探针会话都属于 `demo-user`,而端点现在
        # **按当前登录用户**过滤/校验归属 ⇒ 这里显式让「当前登录用户」就是
        # `demo-user`。**刻意不依赖 conftest 那个全局默认装置** ——
        # 默认给的是 `tests-default-user`,与本文件的探针不是一个用户,
        # 而那种不一致会以「列表空 / 明细 404」的形式红,读起来像端点坏了。
        app.dependency_overrides[require_user] = lambda: AuthenticatedUser(
            username="demo-user", role=ADMIN)
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()


def test_list_filters_by_the_logged_in_user(conv_client):
    """列表按**当前登录用户**过滤(认证,2026-09-27)。

    ⚠️ 过滤值**来自 token**,不再是一个写死的 `demo-user`(那个常量连同
    `ChatRequest.user_id` 一起删了)。这里之所以仍然是 `demo-user`,是因为
    **`conv_client` 这个装置显式投了它**(见该装置里那句 override)—— 本文件的
    探针会话本来就都属于它。**别把 `demo-user` 读成产品常量**:
    conftest 的全局默认装置给的是 `tests-default-user`,与这里不是一个人。

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
    """**用户没看见过的内部机制**里,这两样不回**成一条 item**(spec §5.2 裁定):

    ① `role='tool'` 的行 —— 原始工具载荷(`{"order_no": …}`);
    ② `content=''` 的 assistant 行 —— 「只申请调用工具、还没产出文字」那一形态
       (`app/agent/nodes.py` 写的是 `content=m.content or ""`,它身上只有
       `tool_calls`);**它自己不作为一条气泡出现**。

    ⚠️ **② 的语义在 2026-09-27 改过,别照老话读**:②那一行的 `tool_calls`
    **会**被归并到本轮的答案项上(那是修「齿轮回不来」的正解,见
    `test_messages_endpoint_merges_a_turns_gears_into_its_answer`)——
    「不回载」说的只是**它不作为独立的一条 item**。

    同时断言**同一批里 user 与带文字的 assistant 仍然在**:只断「那两条不在」的话,
    一个恒返回空列表的实现也满足它。
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


def test_messages_endpoint_returns_tool_calls_and_citations(conv_client):
    """回载的每一条必须带 `tool_calls` 与 `citations` —— 它们是「工具齿轮」与
    「文档链接」在回载时的**唯一**来源(前端从 assistant 行的 `tool_calls` 画齿轮,
    从 `citations` 做 `[n]` 可点)。

    两条都断**逐字相等**,不是「非空」:值写错了(`args` 被换成 summary、
    `n` 编号错位)在前端表现为「齿轮名字对不上」或「点了弹错来源」,
    而「非空」级别的断言对这两种实现**一样绿**。

    ⚠️ 这条用例只验**端点有没有把这两列原样带出来**;「它们真的被写进库、
    并且 JSON 往返回来仍是同一个结构」是 db 用例的事
    (`tests/test_api_conversations_db.py`)—— 那正是本仓记过的
    「替身会替被测对象完成语义」那一族。
    """
    tool_calls = [
        {"id": "call_1", "name": "query_order", "args": {"order_no": "1002"},
         "type": "tool_call"}
    ]
    citations = [
        {"n": 1, "chunk_id": "77", "section_path": "退货退款政策 > 无理由退货",
         "question": "无理由退货的期限是多久?", "answer": "七天。", "category": "退换货"}
    ]
    convs = {CONV_A: _conv(CONV_A, "demo-user", T0)}
    msgs = [
        _msg(1, CONV_A, "user", "订单 1002 能退吗"),
        _msg(2, CONV_A, "assistant", "按政策可以退[1]。",
             tool_calls=tool_calls, citations=citations),
    ]
    client = conv_client(conversations=convs, messages=msgs)
    with client as c:
        items = c.get(f"/api/conversations/{CONV_A}/messages").json()["items"]

    assert [i["role"] for i in items] == ["user", "assistant"]
    # user 行两列都是 None(真实形状:没有任何写入方往 user 行上写过它们)
    assert items[0]["tool_calls"] is None
    assert items[0]["citations"] is None
    # assistant 行**逐字**相等(深层结构一起比:只断某字段的写法盖不住
    # 「args 内层字典被拍平成字符串」这类有损往返)
    assert items[1]["tool_calls"] == tool_calls
    assert items[1]["citations"] == citations


def test_messages_endpoint_merges_a_turns_gears_into_its_answer(conv_client):
    """一轮里的齿轮(在**空 content** 的 assistant 行上)**归并到本轮那条答案**上。

    这就是 2026-09-27 那次修复的核心:直播时齿轮与正文在**同一个气泡**里
    (`tool_call` 帧与 `token` 帧都往同一个 ctx 上画),而库里它们在**两行**上
    ⇒ 回载必须做这层归并,否则「空气泡那一轮」的齿轮永远回不来
    (实测:336 条带齿轮的 assistant 行里 **303 条** content 为空)。

    **判别力**:不归并 ⇒ 结果里两条 assistant 项(一条空气泡、一条无齿轮的答案)
    ⇒ 下面的 `len(items) == 2` 与 `items[1]["tool_calls"]` 两条都红。
    """
    convs = {CONV_A: _conv(CONV_A, "demo-user", T0)}
    msgs = [
        _msg(1, CONV_A, "user", "订单 1002 到哪了"),
        _msg(2, CONV_A, "assistant", "",
             tool_calls=[{"id": "c1", "name": "query_order", "args": {},
                          "type": "tool_call"}]),
        _msg(3, CONV_A, "tool", '{"order_no":"1002","status":"已取消"}'),
        _msg(4, CONV_A, "assistant", "这一单已取消"),
    ]
    client = conv_client(conversations=convs, messages=msgs)
    with client as c:
        items = c.get(f"/api/conversations/{CONV_A}/messages").json()["items"]

    assert [i["content"] for i in items] == ["订单 1002 到哪了", "这一单已取消"]
    names = [tc["name"] for i in items for tc in (i["tool_calls"] or [])]
    assert names == ["query_order"], names
    # 项数**少于**行数(4 行 → 2 项):这正是「归并」这件事本身
    assert len(items) == 2 < len(msgs)


def test_messages_endpoint_does_not_merge_two_text_bearing_assistant_rows(conv_client):
    """**反着断**:归并**不许**把两条**本来就有正文**的 assistant 行并成一条。

    并了就是**凭空抹掉一条用户看见过的回复**(而它不报错:屏幕上只是少了一句话)。
    归并的判据是「该行有没有正文」,不是「这一轮有几条 assistant 行」——
    这条用例把两者分开。

    (顺带钉住 `tool_calls` 的另一个来源:「先说了一句开场白、再申请调用工具」
    那种行**正文与 tool_calls 在同一条行上**,实测真实库里有 33 条 ——
    它的齿轮必须跟着**自己**那条输出,不能只认累积区。)
    """
    convs = {CONV_A: _conv(CONV_A, "demo-user", T0)}
    msgs = [
        _msg(1, CONV_A, "user", "订单 1002 到哪了"),
        _msg(2, CONV_A, "assistant", "这就为您查询。",
             tool_calls=[{"id": "c1", "name": "query_order", "args": {},
                          "type": "tool_call"}]),
        _msg(3, CONV_A, "assistant", "这一单已取消"),
    ]
    client = conv_client(conversations=convs, messages=msgs)
    with client as c:
        items = c.get(f"/api/conversations/{CONV_A}/messages").json()["items"]

    assert [i["content"] for i in items] == ["订单 1002 到哪了", "这就为您查询。",
                                             "这一单已取消"]
    # 齿轮挂在**有它自己那份** tool_calls 的那条上
    assert [tc["name"] for tc in items[1]["tool_calls"]] == ["query_order"]
    assert items[2]["tool_calls"] is None, items[2]["tool_calls"]


def test_messages_endpoint_emits_the_gears_of_a_turn_with_no_answer(conv_client):
    """**边界**:某一轮**只有齿轮、没有带正文的 assistant 行** ⇒ 仍然输出一条
    `content: ""` 带 `tool_calls` 的项。

    这是刻意的(生产形状是「模型申请了工具、那一轮随后报错」):丢掉它 =
    用户回载时**连齿轮都看不见**,而直播时他明明看见过。

    **判别力**:把这一支删掉(只输出带正文的项)⇒ 下面 `len(items) == 2` 变成 1,当场红。
    """
    convs = {CONV_A: _conv(CONV_A, "demo-user", T0)}
    msgs = [
        _msg(1, CONV_A, "user", "订单 1002 到哪了"),
        _msg(2, CONV_A, "assistant", "",
             tool_calls=[{"id": "c1", "name": "query_logistics", "args": {},
                          "type": "tool_call"}]),
    ]
    client = conv_client(conversations=convs, messages=msgs)
    with client as c:
        items = c.get(f"/api/conversations/{CONV_A}/messages").json()["items"]

    assert len(items) == 2, items
    assert items[1]["role"] == "assistant"
    assert items[1]["content"] == ""
    assert [tc["name"] for tc in items[1]["tool_calls"]] == ["query_logistics"]


def test_the_query_only_accepts_the_json_type_predicate(conv_client):
    """⚠️ **这条钉的是判据的「形状」,不是行为** —— 先把这件事说白。

    端点的 SQL 里那个「这一行有没有工具调用」的判据**只能**是
    `JSON_TYPE(tool_calls) = 'ARRAY'`:另两种写法
    (`IS NOT NULL` / `JSON_LENGTH(...) > 0`)都会把**字面 JSON `null`** 当成
    「有」(实测整表 `IS NOT NULL` 与 `JSON_LENGTH > 0` 都是 **2721** 行,
    而真值是 **336**)。

    **为什么行为断言写不出来**(本机 2026-09-27 实测):本查询的形态是
    `role <> 'tool' AND (content <> '' OR <判据>)`,而那三种判据取到的行数
    **逐位相同(都是 2238)** —— 因为「`content=''` 且 tool_calls 不是 ARRAY」的
    行**一行都没有**(所有空 content 的 assistant 行都带 ARRAY)。
    ⇒ 差别只在**将来**某一行写成那个形状时才出现,而那种行在归并里也该被丢掉。
    所以这条只能在**替身**上钉形状:替身遇到非 `JSON_TYPE` 的函数形态**当场抛**
    (见 `_where_terms`),把判据钉死在源码里;那条 SQL 的**真库语义**由
    `tests/test_api_conversations_db.py` 的用例另行钉住。
    """
    convs = {CONV_A: _conv(CONV_A, "demo-user", T0)}
    client = conv_client(conversations=convs, messages=[])
    with client as c:
        # 正常跑通就说明端点用的判据是替身认的那一种(换写法 ⇒ 替身直接抛 ⇒ 红)
        assert c.get(f"/api/conversations/{CONV_A}/messages").json()["items"] == []


def test_messages_endpoint_404s_for_someone_elses_conversation(conv_client):
    """**存在、但不是你的**会话 ⇒ 与「不存在」同一个出口(认证,2026-09-27)。

    与上一条(不存在的 id)是两回事:那条查得到「查无此会话」,这条**查得到、
    但不属于当前登录用户**。回 403 就等于承认它存在 —— 那是一个可枚举的接口。

    ⚠️ 本用例**同时是替身自检**:`_StubSession` 的复合 id 分支必须**真按 user 筛**
    (见 `_StubSession.execute` 那段),把那一句删掉它当场变红 —— 这与
    `tests/test_api_chat.py::test_fake_session_applies_updates_and_still_rejects_unknown_queries`
    是同一条规矩:替身自己支持了什么,要有东西钉着。

    ⚠️ **真库上的同一性质**由 `tests/test_api_conversations_db.py` 的两个用例钉
    (那边验的是 SQL 真的带上了 user 条件 —— 替身验不出来)。这里留一条**不依赖
    MySQL** 的同款断言:`pytest -m "not db"` 那一档也要罩得住这条性质。
    """
    convs = {CONV_A: _conv(CONV_A, "someone-else", T0)}
    msgs = [_msg(1, CONV_A, "user", "别人的一句话")]
    client = conv_client(conversations=convs, messages=msgs)
    with client as c:
        resp = c.get(f"/api/conversations/{CONV_A}/messages")
    assert resp.status_code == 404, (
        f"别人的会话必须 404(不是 {resp.status_code} —— 403 等于承认它存在)"
    )
    assert resp.json()["detail"] == "会话不存在"


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
