"""`POST /api/feedback`(ch09 飞轮入口 ③)的落池口径 —— 打**真库**。

为什么打真库、而不是像 `tests/test_api_ticket.py` 那样用替身 session:本端点要钉的
三件事里有**两件**替身证不出来 ——

① **「重复 down 只落一行」的幂等**靠服务层**查一次库**(池子上刻意没有唯一键,
   spec §6.1 / §7.1:查重是**语义**判断,唯一键只能管字面全等)。而替身 session
   会**自己把那条 SELECT 答了** —— 那正是本仓编目过的假绿形态(g):被测对象把这件事
   委托给了替身,于是端点把整段查重删掉**照样绿**;
② **「行真的落进了池子」**替身也证明不了(它只能证明「端点是按我以为的形状调用了 add」)。

**为什么用 `httpx.ASGITransport` 而不是 `TestClient`**:`tests/test_api_ticket.py`
文件头记过账 —— `TestClient` 自建 portal 事件循环,会把 `get_engine()` 那个 lru_cache
单例绑到**它的**循环上,污染同进程里排在后面的 db 测试;而 `ASGITransport` 把请求跑在
**测试自己的**循环里,与紧随其后的回查用的是同一个循环。顺带:它也**不跑 lifespan**,
所以 BGE-M3 的预热线程不会被拉起(「单测不加载 2.2GB 权重」是硬规矩)。

**断言一律按探针专用的 `source_conversation_id`(必要时再加探针专用的问题文本)过滤**:
池子是全库共享的,里面躺着前几章演示/验收留下的旧行(实测 2026-09-24:共 30 行、
ids ≤ 305、`entry_point` 是 `置信度闸` / `生成自评`,快照是 **SQL NULL**),
对总量断言会变成「偶尔红、偶尔绿」。加问题文本那一半是为了堵另一种漏:
一个**忽略 `conversation_id`**、把行写到别处的实现,只按会话过滤是照不出来的。
"""

import json
import logging

import httpx
import pytest
from sqlalchemy import text

from app.api import feedback as feedback_api
from app.config import Settings, get_settings
from app.db.base import get_engine, get_sessionmaker
from app.main import app
from app.retrieval.search import RetrievedChunk
from app.tools.errors import ToolInfrastructureError

#: 探针会话 id(≤ 32 字符)。`t12probe-` 前缀**结构上**撞不到真实数据:
#: 库里的真实会话 id 是 `uuid4().hex` 与 `acceptance-*`,测试用的是 `t11probe-`。
PROBE_UP = "t12probe-up"
PROBE_DOWN = "t12probe-down"
PROBE_DUP = "t12probe-dup"
PROBE_FAIL = "t12probe-fail"
PROBE_BAD = "t12probe-bad"

#: 探针问题文本。**每个探针一句独有的** —— 顺带当第二个过滤条件(见文件头)。
Q_UP = "t12 探针:这句话只该在日志里,不该进池子"
Q_DOWN = "t12 探针:这一单我没被答上"
Q_DUP = "t12 探针:同一个 👎 连点两次"
Q_FAIL = "t12 探针:检索挂了也得进池子"
Q_BAD = "t12 探针:非法 value 一个字都不许落"

_REQUIRED_SETTINGS = dict(
    openai_base_url="https://example.invalid/v1",
    openai_api_key="sk-test",
    openai_model="test-model",
    database_url="mysql+aiomysql://u:p@localhost/db",
)

#: 假检索器返回的召回块。用**真的 `RetrievedChunk`**,不用 SimpleNamespace ——
#: `_snapshot` 是按这个形状投影的(`c.chunk_id` / `c.score` / `c.section_path` /
#: `c.answer`),替身换个形状的话被验的就不是生产里的那一支了。
SNAPSHOT_CHUNKS = [
    RetrievedChunk(
        question="怎么退货", answer="七天内可在订单页申请退货。", category="退换货",
        chunk_id=11, section_path="退换货/政策", score=0.5,
    ),
    RetrievedChunk(
        question="运费谁出", answer="质量问题由我们承担运费。", category="退换货",
        chunk_id=12, section_path=None, score=0.25,
    ),
]

#: 期望落进 `evidence_snapshot` 的那份投影(四键、`score` 四位、原文按配置截)。
EXPECTED_SNAPSHOT = [
    {"chunk_id": 11, "score": 0.5, "section_path": "退换货/政策",
     "answer": "七天内可在订单页申请退货。"},
    {"chunk_id": 12, "score": 0.25, "section_path": None,
     "answer": "质量问题由我们承担运费。"},
]


class FakeRetriever:
    """检索器替身:**必须有** —— 端点在 down 那一支会 `build_retriever(...).search()`,
    真实实现要连 Milvus 并现场加载 BGE-M3 权重,而「单测全程不联网」是硬规矩。

    默认返回空 ⇒ 快照留 `null`;要验快照投影的用例显式传 `chunks=`。
    """

    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        #: 重跑用的是**哪一句问题** —— 用错句子(比如拿整段对话去搜)会让
        #: 「回捞出来的片段与这一轮无关」,而那条路看起来一切正常。
        self.calls: list[str] = []

    async def search(self, query):
        self.calls.append(query)
        return list(self.chunks)


class RaisingRetriever:
    """`search` 直接炸的替身:回捞那一半必须**尽力**而**不拦路**。"""

    def __init__(self, exc: Exception):
        self._exc = exc

    async def search(self, query):
        raise self._exc


@pytest.fixture
def client_factory(monkeypatch):
    """造端点级客户端。**换掉检索器与配置,不换 `get_session`** —— 本文件打真库。

    `get_session` 走模块级 `app.db.base.get_sessionmaker`,读仓库根的真实 `.env`,
    这正是本文件要的(真行落进真表)。

    ⚠️ 检索器**默认也要换掉**,不是「传了才换」:任何一个 `value="down"` 的用例
    都会走到回捞那一步,留着真实现就是**一次真实 IO + 2.2GB 权重加载**。
    """

    def make(retriever=None, **settings_overrides):
        app.dependency_overrides[get_settings] = lambda: Settings(
            _env_file=None, **{**_REQUIRED_SETTINGS, **settings_overrides}
        )
        # 形参按生产签名对齐(`(session, settings)`);少收一个抛的是 TypeError,
        # 而用例期待的是 200 —— 红法会指向这行 lambda,不指向被测代码。
        monkeypatch.setattr(
            feedback_api,
            "build_retriever",
            lambda session, settings=None: (
                retriever if retriever is not None else FakeRetriever()
            ),
        )
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )

    yield make
    app.dependency_overrides.clear()


def _body(conversation_id, question, *, value, message_id=4242):
    return {
        "conversation_id": conversation_id,
        "question": question,
        "message_id": message_id,
        "value": value,
    }


async def _delete_probe(conversation_id: str) -> int:
    """删本用例的探针行,返回**删后**剩下的行数(删后检查,不是删前)。"""
    async with get_sessionmaker()() as session:
        await session.execute(
            text("DELETE FROM low_confidence_questions "
                 "WHERE source_conversation_id = :c"),
            {"c": conversation_id},
        )
        await session.commit()
        return (
            await session.execute(
                text("SELECT COUNT(*) FROM low_confidence_questions "
                     "WHERE source_conversation_id = :c"),
                {"c": conversation_id},
            )
        ).scalar_one()


async def _teardown(conversation_id: str) -> None:
    """用例收尾:删探针行 → **删后**复查为 0 → 关连接池。

    为什么要 `dispose()`:anyio 给**每条用例**一个新事件循环,而引擎是跨用例的
    lru_cache 单例 —— 上一条用例留下的 asyncmy 连接会在**下一个**循环里被终结,
    打一句 `ERROR sqlalchemy.pool ... AttributeError: 'NoneType' object has no
    attribute 'send'`(本机 Windows proactor 现象,实测:不关池时跑一遍本文件出
    **5** 句;仓库里 `tests/test_api_refund.py` 同款 5 句,而每条用例都 `dispose()`
    的 `tests/test_kb_assess.py` 是 **0** 句)。它不影响断言,但「测试输出干净」
    在这里是有用的信号 —— 噪声会淹没真错误。
    """
    left = await _delete_probe(conversation_id)
    assert left == 0, f"探针行没删干净,池子里还剩 {left} 行"
    await get_engine().dispose()


async def _rows(conversation_id: str) -> list[dict]:
    """**裸 SQL** 回查(新 session):不经过身份映射,也拿到库里的 JSON 原文 ——
    与 `tests/test_kb_assess.py` 的 T11 用例同一个理由(`text()` 不带类型信息,
    JSON 结果处理器因此不参与,读回来的就是那串文本)。
    """
    async with get_sessionmaker()() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT id, question, entry_point, reject_reason, evidence_snapshot "
                    "FROM low_confidence_questions WHERE source_conversation_id = :c "
                    "ORDER BY id"
                ),
                {"c": conversation_id},
            )
        ).mappings().all()
    return [dict(r) for r in rows]


async def _count_by_question(question: str) -> int:
    """按**问题文本**数一次 —— 堵「端点忽略 `conversation_id`、写到别的会话上」。"""
    async with get_sessionmaker()() as session:
        return (
            await session.execute(
                text("SELECT COUNT(*) FROM low_confidence_questions WHERE question = :q"),
                {"q": question},
            )
        ).scalar_one()


# --------------------------------------------------------------------------
# 一、`_snapshot` 的投影(纯函数,不需要库)
# --------------------------------------------------------------------------


def test_snapshot_projects_top_n_truncates_and_rounds():
    """快照只留审核页要看的四样,且按配置**封顶**与**截短**。

    这条是纯函数用例:`snapshot_top_n=2` 而给 3 条块 ⇒ 第 3 条必须被丢掉
    (不封顶的话池子里那行 JSON 会随召回条数无限长);`snapshot_answer_chars=5`
    ⇒ 原文截到 5 字。两个旋钮在这里**各自可观测**,而端点用例只跑默认值
    (默认 400 字长于任何测试原文 ⇒ 截短那一步在端点用例上恒不触发,
    是本仓记过的「输入小到触发不了被测行为」)。
    """
    settings = Settings(
        _env_file=None, **{**_REQUIRED_SETTINGS,
                           "snapshot_top_n": 2, "snapshot_answer_chars": 5}
    )
    chunks = [
        RetrievedChunk(question="q1", answer="答案很长很长", category="c",
                       chunk_id=1, section_path="s/1", score=0.123456),
        RetrievedChunk(question="q2", answer="短", category="c",
                       chunk_id=2, section_path="s/2", score=0.5),
        RetrievedChunk(question="q3", answer="第三条", category="c",
                       chunk_id=3, section_path="s/3", score=0.9),
    ]
    snap = feedback_api._snapshot(chunks, settings=settings)

    assert [c["chunk_id"] for c in snap] == [1, 2], "条数必须按 snapshot_top_n 封顶"
    assert set(snap[0]) == {"chunk_id", "score", "section_path", "answer"}, (
        "只投影这四个键 —— 审核页是按这个形状读的(ch09 spec §6.2)"
    )
    assert snap[0]["answer"] == "答案很长很长"[:5] and len(snap[0]["answer"]) == 5, (
        f"原文必须按 snapshot_answer_chars 截短,实际 {snap[0]['answer']!r}"
    )
    assert snap[0]["score"] == pytest.approx(0.1235), "score 保留四位"
    assert snap[1]["answer"] == "短", "短于上限的不该被动(否则截短与截爆分不开)"


def test_snapshot_is_none_when_nothing_recalled():
    """零召回 ⇒ **`None`**,不是空列表。

    两者在库里不同:`None` 走 SQLAlchemy 的 JSON 列落成 **JSON `null`**,
    而 `[]` 是一段真的 JSON 数组。审核页读起来是两件事(「检索没召回」
    vs「这里压根没记」),而 `record_low_confidence` 的 docstring 也钉了这一条。
    """
    settings = Settings(_env_file=None, **_REQUIRED_SETTINGS)
    assert feedback_api._snapshot([], settings=settings) is None
    assert feedback_api._snapshot(None, settings=settings) is None


# --------------------------------------------------------------------------
# 二、端点(真库)
# --------------------------------------------------------------------------


@pytest.mark.db
@pytest.mark.anyio
async def test_up_writes_nothing_to_the_pool(client_factory):
    """`up` 不落池,且**不是错误**。

    「不落池」这一半**必须配一个 200 断言** —— 只断「池子里 0 行」的话,
    一个**根本没注册路由**的实现会给出同样的观测,整条用例零判别力
    (TDD 的红期正是那个状态)。所以两个都断。

    ⚠️ 红期的观测是 **405 Method Not Allowed**,不是 404:没注册的 `/api/*` 会被
    `mount("/")` 那个静态 catch-all 接住,而它只放行 GET/HEAD(实测,见 dev-notes
    阶段 7)。查这条的时候别往 404 的方向找。

    问题文本也数一次:只按会话过滤的话,「端点忽略 `conversation_id`、
    把行写到别的会话上」这种漏照不出来(本仓那条「字段的语义要读它的赋值处」)。
    """
    client = client_factory()
    await _delete_probe(PROBE_UP)          # 防上一次崩在断言中间留下的行
    try:
        r = await client.post("/api/feedback",
                              json=_body(PROBE_UP, Q_UP, value="up"))
        assert r.status_code == 200, f"up 不是错误,实际 {r.status_code}:{r.text}"
        assert r.json()["ok"] is True
        assert r.json()["pooled"] is False, "up 的答复自己也该说「没落池」"
        assert await _rows(PROBE_UP) == [], "up 一行都不许落"
        assert await _count_by_question(Q_UP) == 0, "落到了别的会话上(会话 id 没用上)"
    finally:
        await _teardown(PROBE_UP)


@pytest.mark.db
@pytest.mark.anyio
async def test_down_writes_one_row_with_user_feedback_entry_point(client_factory):
    """`down` 落**一行**,`entry_point="用户反馈"`,并带上尽力回捞到的召回片段。

    三样都断,因为每一样都能单独静默失效:
    - `entry_point` 写错(比如沿用 `置信度闸`)⇒ 飞轮的三个入口在池子里**分不开**,
      验收 4 与审核页都会读到一堆分不清来源的行;
    - 快照没落 ⇒ 审核人只有一句问题,「知识库真缺这块」与「有、但没检到」
      看起来**一模一样**(spec §6.2 那正是要回捞的理由);
    - 回捞用的**不是这一轮的问题**(比如拿整段对话去搜)⇒ 快照里是别的知识块,
      而这条路**不报错**。
    """
    retriever = FakeRetriever(SNAPSHOT_CHUNKS)
    client = client_factory(retriever=retriever)
    await _delete_probe(PROBE_DOWN)
    try:
        r = await client.post("/api/feedback",
                              json=_body(PROBE_DOWN, Q_DOWN, value="down"))
        assert r.status_code == 200, f"实际 {r.status_code}:{r.text}"
        assert r.json()["pooled"] is True

        rows = await _rows(PROBE_DOWN)
        assert len(rows) == 1, f"down 必须落且只落一行,实际 {len(rows)} 行:{rows}"
        assert rows[0]["entry_point"] == "用户反馈"
        assert rows[0]["question"] == Q_DOWN, "落池的必须是**用户原话**那一条问题"
        assert rows[0]["reject_reason"], "reject_reason 是非空列,审核页要读它"
        assert json.loads(rows[0]["evidence_snapshot"]) == EXPECTED_SNAPSHOT, (
            "快照是审核页判「真缺 / 没检到」的唯一依据,必须按契约的形状落全"
        )
        assert retriever.calls == [Q_DOWN], (
            f"回捞重跑的是**这一轮的问题**,实际搜了 {retriever.calls}"
        )
    finally:
        await _teardown(PROBE_DOWN)


@pytest.mark.db
@pytest.mark.anyio
async def test_repeated_down_for_the_same_message_writes_only_one_row(client_factory):
    """同一个 👎 连点两次,池子里**仍然只有一行**。

    两次是**两个独立请求**(各有自己的 session)⇒ 查重必须真的落到库上。
    池子上**没有唯一键**(刻意的,spec §7.1),所以这条幂等**只由服务层那次
    查重**保证 —— 把它删掉的话,这里的第二个请求会插出第二行,而那**不报任何错**。

    判据是「同一会话 + 同一问题 + `entry_point=用户反馈`」(spec §6.1):
    池子是一张**问题**池,下游飞轮也是按问题归并的,而 `message_id` 池子里
    **没有对应的列**(它只是前端顺手带上来的定位信息)。
    """
    client = client_factory()
    await _delete_probe(PROBE_DUP)
    try:
        body = _body(PROBE_DUP, Q_DUP, value="down", message_id=9999)
        first = await client.post("/api/feedback", json=body)
        second = await client.post("/api/feedback", json=body)
        assert first.status_code == second.status_code == 200
        # ⚠️ **落库那条断言排在最前**:它才是这条用例的承重墙(0 行与 2 行都红),
        # 而下面两条读的是**响应体**。顺序反过来的话,一个只在响应里说谎、
        # 库里却插了两行的实现在**第一条**响应断言上就红了,库那一半永远走不到
        # —— 证据落在弱断言上,强断言有没有牙就无从知道(变异 M1 实测过)。
        rows = await _rows(PROBE_DUP)
        assert len(rows) == 1, (
            f"重复 down 必须幂等(同一会话 + 同一问题只一行),实际 {len(rows)} 行"
        )
        assert first.json()["pooled"] is True, "第一次必须真落池(否则「只有一行」可能是「一行都没落」)"
        assert second.json()["pooled"] is False, "第二次是去重命中,没落池"
        assert rows[0]["entry_point"] == "用户反馈"
    finally:
        await _teardown(PROBE_DUP)


@pytest.mark.db
@pytest.mark.anyio
@pytest.mark.parametrize("exc_cls", [ToolInfrastructureError, RuntimeError],
                         ids=["infra", "unexpected"])
async def test_retriever_failure_still_pools_the_row_and_logs_loudly(
    client_factory, caplog, exc_cls
):
    """**回捞是尽力而为**:检索挂掉 ⇒ 行**照样落**,快照留空,且**响亮地留痕**。

    三个断言对应三件独立的事:

    ① **200 而不是 502** —— 落池才是这个端点的职责,检索只是附赠。让检索的
       故障把整个请求打掉,用户点了 👎 却什么都没发生,而池子里那**正是**
       该被审的一条问题;
    ② **快照是 `null`** —— 注意是 JSON `null`(`json.loads(...) is None`),
       不是空列表:`[]` 会读成「重跑过、零召回」,与「没能回捞」是两件事;
    ③ **日志带 traceback** —— 光一句"失败了"不够:`record_low_confidence` 的
       docstring 写着「快照为空」与「检索故障」在**数据上长得一样**,审核人
       只能靠日志把它们分开。⚠️ 本仓硬约束复核:`tool_result`/`error` 帧那类
       出站文本要脱敏,日志不是出站文本,所以这里带原始异常是**对的**。

    **两种异常都注入**(不是随便挑一个):生产里 `search()` 抛的是翻译过的
    `ToolInfrastructureError`(Milvus / 嵌入 / 重排 / 回查四条腿,见
    `app/retrieval/search.py` 的翻译边界),而端点是按**尽力而为**的语义写的
    (`except Exception`)。只注入前者的话,一个把 `except` 收窄成
    `except ToolInfrastructureError`(于是编程错误会把请求打成 500)的实现
    **照样绿**;只注入后者的话,验的又不是生产真实的那条路。
    """
    client = client_factory(retriever=RaisingRetriever(exc_cls("检索炸了")))
    await _delete_probe(PROBE_FAIL)
    try:
        with caplog.at_level(logging.WARNING, logger="app.api.feedback"):
            r = await client.post("/api/feedback",
                                  json=_body(PROBE_FAIL, Q_FAIL, value="down"))
        assert r.status_code == 200, (
            f"检索故障不许把落池一起打掉,实际 {r.status_code}:{r.text}"
        )

        rows = await _rows(PROBE_FAIL)
        assert len(rows) == 1, f"行必须照样落,实际 {len(rows)} 行"
        assert rows[0]["entry_point"] == "用户反馈"
        assert json.loads(rows[0]["evidence_snapshot"]) is None, (
            f"回捞不到的快照是 JSON null,实际 {rows[0]['evidence_snapshot']!r}"
        )

        loud = [rec for rec in caplog.records
                if rec.name == "app.api.feedback" and rec.levelno >= logging.WARNING]
        assert loud, "回捞失败必须留痕,否则「真缺这块」与「检索挂了」在数据上分不开"
        assert any(rec.exc_info for rec in loud), (
            "只写一句「失败了」不够 —— 不带 traceback 就查不出挂在哪条腿上"
        )
    finally:
        await _teardown(PROBE_FAIL)


@pytest.mark.db
@pytest.mark.anyio
async def test_unknown_value_is_rejected_and_pools_nothing(client_factory):
    """`value` 是**闭集** `up|down`;别的值 422,且一行都不落。

    这条钉的是校验的**位置**:实现里「不是 up 就当 down」那种写法(先落池、
    由更外层兜住非法值)在真实请求上会把一个前端拼错的值变成一次**池子写入**,
    而那看起来只是「用户点了某个按钮」。
    """
    client = client_factory()
    await _delete_probe(PROBE_BAD)
    try:
        r = await client.post("/api/feedback",
                              json=_body(PROBE_BAD, Q_BAD, value="meh"))
        assert r.status_code == 422, f"非法 value 必须被挡在入口,实际 {r.status_code}"
        assert await _rows(PROBE_BAD) == []
        assert await _count_by_question(Q_BAD) == 0
    finally:
        await _teardown(PROBE_BAD)
