"""审核端点(ch09 §9,`app/api/review.py`)—— **不读库**的那一半。

真库那一半在 `tests/test_api_review_db.py`(H2 的 SQL 过滤、approve 的真
`write_chunks`、H1 的重试)。本文件用**替身 session + 替身 Milvus/嵌入**把
「端点的接线」钉死,其中三件事**只有在这里**才验得动:

① **路由声明顺序**(R1):`/api/review/queue` 必须排在 `/api/review/{review_id}`
   前面。反了的话 `queue` 会被当成 `review_id` 去解析 int ⇒ 422,而**服务照常起**、
   别的断言全绿;
② **过滤在不在 SQL 里**(H2 的第一半):替身 session **不替端点做过滤** ——
   它把语句原样记下来,用例断言 `WHERE` 子句里有那一列。那种「替身自己
   `sorted(...)` / 自己 `if row.status != status: continue`」的假替身是本仓编目过的
   形态 ⑦(替身完成了被测对象的语义),这里刻意避开;
③ **失败路径不改状态**(R5 / H1):向量化炸了之后队列行必须**仍是 pending**,
   且重试**必须真的再向量化一次**。

## 替身的边界

- `session` → 替身,**但只替接口面与身份映射**(`get` 返回**同一个** ORM 对象,
  于是端点改过的 `status` 第二次读得到 —— 与真 session 的 identity map 同款)。
  库上的真行为由 db 文件守。
- `write_chunks` / `vectorize_rows` → 分两类用例:一类**整段打桩**(只验「调了没、
  参数对不对」),一类**只换 Milvus / 嵌入、留真的 `vectorize_rows`**(H1 那条:
  「第二次通过要真的走到向量化」是 `vectorize_rows` 的边界,打桩它就变成空话)。
- `get_vector_store` / `get_embedder` → **模块级 autouse 打桩**。不打的话,
  `store.ensure_collection()` 会在**一条不读库的用例里发起真实连接**,
  而「非 DB 测试绝不碰网络」是硬约束。
"""

import dataclasses
from datetime import datetime

import httpx
import pytest

from app.api import review as review_api
from app.config import Settings, get_settings
from app.db.models import KnowledgeChunk, LowConfidenceQuestion, ReviewQueue
from app.db.session import get_session
from app.main import app
from app.tools.errors import ToolInfrastructureError

_REQUIRED_SETTINGS = dict(
    openai_base_url="https://example.invalid/v1",
    openai_api_key="sk-test",
    openai_model="test-model",
    database_url="mysql+aiomysql://u:p@localhost/db",
)

#: 队列行的 id。**非 1 的整数** —— 写死 1 的话,「端点把 id 常量硬编码进去」这种
#: 实现照样绿(它反正只查 1 号行)。
RQ_ID = 77

#: 那条知识块的 id(`write_chunks` 替身分配给它)。与 `RQ_ID` 取得不同,免得
#: 「端点把两个 id 弄混了」看不出来。
CHUNK_ID = 4242

QUESTION = "换货要自己出运费吗"
EXAMPLE_ANSWER = "换货由我们承担运费,拒收产生的费用另计。"


# --------------------------------------------------------------------------
# 替身
# --------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)


class FakeSession:
    """`AsyncSession` 里本端点用得到的那一块。

    **不替端点做过滤**:它按语句里的表名把预设行交出去,`WHERE` 子句对不对由用例
    看着原始 SQL 自己判(见文件头 ②)。唯一的「语义」是身份映射:`get` 每次返回
    **同一个对象**,所以端点写上去的 `status` 第二次读得到 —— 真 session 就是这样,
    换成「每次现造一行 status=pending」会让 R2 那条幂等用例**恒绿**。

    ⚠️ 它**也不执行 `DELETE`**(`execute` 只记录语句):于是 502 清理(F4)在本文件里
    是 no-op,`kb_rows` 里那条 pending 行会**留下来** —— 这是**刻意**的,好让
    「`write_chunks` 返回 0 时也必须向量化」那条契约(H1 的陷阱,也就是**崩溃遗留
    孤儿**那条真路)在本文件里继续被行使。真库上的清理行为由
    `tests/test_api_review_db.py::test_a_failed_vectorize_discards_its_draft` 守。
    """

    def __init__(self, *, queue_row=None, linked=(), kb_rows=()):
        self.queue_row = queue_row
        self.linked = list(linked)
        self.kb_rows = list(kb_rows)
        #: [(编译出的 SQL 文本, 绑定参数字典)]
        self.statements: list[tuple[str, dict]] = []
        self.gets: list[tuple] = []
        self.commits = 0
        self.rollbacks = 0

    async def get(self, model, pk):
        self.gets.append((model, pk))
        return self.queue_row

    async def execute(self, stmt):
        compiled = stmt.compile()
        self.statements.append((str(stmt), dict(compiled.params)))
        if ReviewQueue.__tablename__ in str(stmt):
            rows = [] if self.queue_row is None else [self.queue_row]
        elif LowConfidenceQuestion.__tablename__ in str(stmt):
            rows = self.linked
        elif KnowledgeChunk.__tablename__ in str(stmt):
            rows = self.kb_rows
        else:  # pragma: no cover - 新语句必须有对应的替身分支
            raise AssertionError(f"替身没准备这条语句:{stmt}")
        return _FakeResult(rows)

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


class FakeEmbedder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t))] * 3 for t in texts]


class FakeStore:
    """Milvus 替身。`fail_first_upserts` 次调用抛 `exc`,之后放行 ——
    「Milvus 抖一下」这件事必须**只发生一次**,否则重试永远走不到成功那一步。
    """

    def __init__(self, *, fail_first_upserts: int = 0,
                 exc: BaseException | None = None):
        self.remaining_failures = fail_first_upserts
        self.exc = exc if exc is not None else ToolInfrastructureError("Milvus 连不上")
        self.ensure_calls = 0
        self.attempts: list[list[str]] = []
        self.upserts: list[list[str]] = []

    def ensure_collection(self):
        self.ensure_calls += 1

    def upsert(self, ids, texts, categories, vectors):
        self.attempts.append(list(ids))
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise self.exc
        self.upserts.append(list(ids))


# --------------------------------------------------------------------------
# 装置
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def vector_deps(monkeypatch):
    """**每条用例**都把 Milvus / 嵌入换成替身(理由见文件头)。

    返回一个可改的小盒子:`vector_deps.store = FakeStore(fail_first_upserts=1)`
    就能换成会抖的那一个 —— 端点拿的是**模块属性**,所以赋值当场生效。
    """
    class Deps:
        def __init__(self):
            self.store = FakeStore()
            self.embedder = FakeEmbedder()

    deps = Deps()
    monkeypatch.setattr(
        review_api, "get_vector_store", lambda uri, collection: deps.store)
    monkeypatch.setattr(
        review_api, "get_embedder", lambda *a, **kw: deps.embedder)
    return deps


@pytest.fixture
def deduping_write_chunks(monkeypatch):
    """只换 `write_chunks` 的替身,**照着真行为**做三元组查重:写进去过就返回 0。

    为什么这个也必须换掉:真 `write_chunks` 的第一条语句是「一次 SELECT 捞出
    全表三元组」(它靠应用层查重,表上没有唯一键),本文件的替身 session 答不了
    那条 SQL。而**它第二次返回 0** 恰恰是 H1 的前提 —— 所以替身必须真的复现
    这个返回值,不能糊一个恒 1 的假货(那样 H1 的用例会**恒绿**)。
    """

    def make(session, *, row_id=CHUNK_ID):
        async def fake_write(sess, chunks):
            fresh = [
                c for c in chunks
                if not any(
                    r.category == c.category and r.questions == c.questions
                    and r.answer == c.answer
                    for r in sess.kb_rows
                )
            ]
            if not fresh:
                return 0
            c = fresh[0]
            sess.kb_rows.append(KnowledgeChunk(
                id=row_id, category=c.category, questions=c.questions,
                answer=c.answer, section_path=c.section_path,
                content_type=c.content_type, is_key_clause=c.is_key_clause,
                vectorize_status="pending",
            ))
            return len(fresh)

        monkeypatch.setattr(review_api, "write_chunks", fake_write)

    return make


@pytest.fixture
def client_factory():
    """造端点级客户端。`get_session` / `get_settings` 都换成替身。"""

    def make(session=None, **overrides):
        app.dependency_overrides[get_session] = lambda: session
        app.dependency_overrides[get_settings] = lambda: Settings(
            _env_file=None, **{**_REQUIRED_SETTINGS, **overrides}
        )
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )

    yield make
    app.dependency_overrides.clear()


def _queue_row(*, status="pending", example_answer=EXAMPLE_ANSWER, row_id=RQ_ID):
    return ReviewQueue(
        id=row_id,
        standard_question=QUESTION,
        example_answer=example_answer,
        occurrences=3,
        status=status,
        first_raw_question="换货运费谁出啊",
        source_conversation_id="t15probe-conv",
        created_at=datetime(2026, 9, 24, 10, 0, 0),
        reviewed_at=None,
        approved_answer=None,
    )


def _pool_row(pool_id, question, snapshot):
    return LowConfidenceQuestion(
        id=pool_id,
        question=question,
        source_conversation_id="t15probe-conv",
        entry_point="用户反馈",
        reject_reason="用户点了 👎(未解决)",
        evidence_snapshot=snapshot,
        matched_review_id=RQ_ID,
        created_at=datetime(2026, 9, 24, 10, 0, pool_id % 60),
    )


def _chunk_row(vectorize_status="pending"):
    return KnowledgeChunk(
        id=CHUNK_ID,
        category="faq",
        questions=QUESTION,
        answer=EXAMPLE_ANSWER,
        section_path=None,
        content_type="faq",
        is_key_clause=False,
        vectorize_status=vectorize_status,
    )


def _where_of(session: FakeSession, *, table: str, require: str | None = None) -> dict:
    """取**唯一**一条打到 `table` 的语句的 `WHERE` 之后那一段 + 绑定参数。

    命中数不为 1 直接红 —— 本仓变异脚本的头号事故是「锚点打到了别处」,
    这里同样不许「多条里随便挑一条」。`require` 是**收窄用的子串**:同一个
    `session` 上打到同一张表的语句可能不止一条(F4 起:写入前后各查一次三元组的
    那两条 `SELECT`,与 `_rows_to_vectorize` 那条),收窄之后仍然必须是**恰好一条**。
    """
    hits = [
        (sql, params) for sql, params in session.statements
        if f"FROM {table}" in sql and (require is None or require in sql)
    ]
    assert len(hits) == 1, f"打到 {table} 的语句应恰好 1 条,实际 {len(hits)}:{hits}"
    sql, params = hits[0]
    where = sql.partition("WHERE")[2] if "WHERE" in sql else ""
    return {"where": where, "params": params, "sql": sql}


# --------------------------------------------------------------------------
# 一、列表(R1 的声明顺序 + H2 的 SQL 过滤)
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_queue_route_is_reachable_and_not_eaten_by_the_id_route(client_factory):
    """R1:`GET /api/review/queue` 必须返回**列表**,不是 422。

    路由是按**声明顺序**匹配的。`/api/review/{review_id}` 排在前面的话,
    `queue` 会去走 `review_id: int` 的解析 ⇒ **422**(报错文案还指向 int 解析,
    看着像前端传错了)。服务照常起,别的端点全部正常 —— 只有这一个 404/422。
    所以这条用例唯一的判别力就是「它没有 422」。
    """
    session = FakeSession(queue_row=_queue_row())
    async with client_factory(session) as client:
        r = await client.get("/api/review/queue")
    assert r.status_code == 200, (
        f"`/api/review/queue` 被 `{{review_id}}` 那条路由吃掉了,"
        f"实际 {r.status_code}:{r.text}(检查两个函数的声明顺序)"
    )
    body = r.json()
    assert isinstance(body, list), f"列表端点必须回数组,实际 {type(body).__name__}"
    assert [row["id"] for row in body] == [RQ_ID]


@pytest.mark.anyio
async def test_queue_filter_is_in_sql_not_in_python(client_factory):
    """H2:`status` 过滤**必须在 SQL 里**。

    替身 session 不替端点过滤(见 `FakeSession`),所以「只回 pending」这件事
    在这里**只能**靠看语句本身 —— 一个「全表捞回来、在 Python 里 if 掉」的实现
    会在这条上红(它的 `WHERE` 里没有那一列)。
    真库上的行为由 `tests/test_api_review_db.py::test_queue_returns_only_pending_rows_from_sql` 守。
    """
    session = FakeSession(queue_row=_queue_row())
    async with client_factory(session) as client:
        r = await client.get("/api/review/queue")
    assert r.status_code == 200, r.text

    seen = _where_of(session, table="review_queue")
    assert "review_queue.status" in seen["where"], (
        f"status 过滤不在 SQL 里(WHERE = {seen['where']!r}) —— 全表捞回来再在 "
        f"Python 里筛,行数一多就是每次请求拖整张表"
    )
    assert list(seen["params"].values()) == ["pending"], (
        f"默认必须只捞 pending,实际绑定 {seen['params']}"
    )


@pytest.mark.anyio
async def test_queue_rejects_an_unknown_status(client_factory):
    """`status` 是**闭集**;别的值 422(而不是「静默当 pending」)。

    静默兜底的后果:前端把 `rejected` 拼错成 `reject` 时,页面显示的是**待审队列**,
    而审核人以为自己在看历史 —— 数据没错,读的人错。
    """
    session = FakeSession(queue_row=_queue_row())
    async with client_factory(session) as client:
        r = await client.get("/api/review/queue", params={"status": "reject"})
    assert r.status_code == 422, f"实际 {r.status_code}:{r.text}"
    assert session.statements == [], "非法取值不该打到库上"


# --------------------------------------------------------------------------
# 二、详情(H3:两条原话 + 非空快照)
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_detail_carries_every_linked_raw_question_with_its_snapshot(client_factory):
    """H3:详情必须带**全部**归并进来的用户原话与各自快照。

    夹具刻意有**两条**关联行、快照**都非空**:只有一条关联行的话,
    「只读了第一条」与「全读了」在断言上**一模一样**;快照为 `None` 的话,
    「原样带出来」与「恒返回 null」也分不开 —— 而审核人正是靠用户原话判
    「这是同一个问题吗」、靠快照判「知识库真缺 / 当时没检到」。
    """
    snap_a = [{"chunk_id": 11, "score": 0.5, "section_path": "退换货/政策",
               "answer": "七天内可申请退货。"}]
    snap_b = [{"chunk_id": 12, "score": 0.25, "section_path": None,
               "answer": "质量问题由我们承担运费。"}]
    session = FakeSession(
        queue_row=_queue_row(),
        linked=[
            _pool_row(901, "换货运费谁出", snap_a),
            _pool_row(902, "换货要自己出邮费吗", snap_b),
        ],
    )
    async with client_factory(session) as client:
        r = await client.get(f"/api/review/{RQ_ID}")
    assert r.status_code == 200, r.text

    body = r.json()
    assert body["id"] == RQ_ID
    assert body["standard_question"] == QUESTION
    assert body["example_answer"] == EXAMPLE_ANSWER
    assert body["occurrences"] == 3

    raws = body["raw_questions"]
    assert [q["question"] for q in raws] == ["换货运费谁出", "换货要自己出邮费吗"], (
        f"归并进来的用户原话必须**一条不落**,实际 {raws}"
    )
    assert [q["evidence_snapshot"] for q in raws] == [snap_a, snap_b], (
        "每条的召回片段快照必须跟着它自己那条原话,不能串、不能丢"
    )

    seen = _where_of(session, table="low_confidence_questions")
    assert "low_confidence_questions.matched_review_id" in seen["where"], (
        f"关联必须在 SQL 里按 matched_review_id 取(WHERE = {seen['where']!r})"
    )
    assert list(seen["params"].values()) == [RQ_ID], (
        f"关联的必须是这一条队列行,实际绑定 {seen['params']}"
    )


@pytest.mark.anyio
async def test_detail_404_for_an_unknown_id(client_factory):
    """不存在的 id ⇒ 404。

    ⚠️ **只断 404 是不够的**:路由没注册时 `mount("/")` 那个静态 catch-all 也回
    404(实测:红期本用例在端点还不存在时就是绿的)。所以再断一条
    「请求真的走到了端点」——它按 id 查过库。
    """
    session = FakeSession(queue_row=None)
    async with client_factory(session) as client:
        r = await client.get("/api/review/12345")
    assert r.status_code == 404, f"实际 {r.status_code}:{r.text}"
    assert session.gets == [(ReviewQueue, 12345)], (
        f"请求没走到端点(或查的不是这个 id):{session.gets}"
    )


# --------------------------------------------------------------------------
# 三、通过(接线 + R2 / R3 / R5 / H1)
# --------------------------------------------------------------------------


@pytest.fixture
def write_recorders(monkeypatch):
    """把 `write_chunks` / `vectorize_rows` 整段换成记录器(只验接线的那几条用)。

    `added` 可配 —— 「知识库里早有这条 ⇒ 返回 0」是 `write_chunks` 的正常返回值
    (它的 docstring 与 `tests/test_kb_writer_db.py` 各钉了一遍)。
    """
    calls = {"chunks": [], "vectorize": [], "chunks_seq": []}

    async def fake_write(session, chunks):
        calls["chunks"].append(list(chunks))
        nxt = calls["chunks_seq"].pop(0) if calls["chunks_seq"] else len(chunks)
        return nxt

    async def fake_vectorize(session, store, embedder, rows):
        calls["vectorize"].append(list(rows))

    monkeypatch.setattr(review_api, "write_chunks", fake_write)
    monkeypatch.setattr(review_api, "vectorize_rows", fake_vectorize)
    return calls


@pytest.mark.anyio
async def test_approve_writes_the_chunk_and_vectorizes_it(
    client_factory, write_recorders,
):
    """brief Step 1 的第 3 条:approve 调 `write_chunks` + `vectorize_rows`。

    三个实参各断一遍(替身计数只是其中最弱的一环):

    - **Chunk 的每个字段逐个对上 dataclass**(brief 给的那行构造是照着
      `app/kb/chunker.py:26` 抄的,但**逐字段核过**才作数):字段名写错在 Python 里
      是 `TypeError`,而**顺序错位**(把 section_path 和 content_type 调个儿)是
      **不报错**的静默错 —— 入库的知识会带着另一个字段的值;
    - `vectorize_rows` 拿到的是**刚写进去的那一行** —— 传空列表的话,
      「通过之后能检索到」这件事永远不成立,而端点会回 200。
    """
    row = _chunk_row()
    session = FakeSession(queue_row=_queue_row(), kb_rows=[row])
    async with client_factory(session) as client:
        r = await client.post(f"/api/review/{RQ_ID}/approve", json={})
    assert r.status_code == 200, f"实际 {r.status_code}:{r.text}"

    assert len(write_recorders["chunks"]) == 1, "write_chunks 必须恰好被调一次"
    chunk = write_recorders["chunks"][0][0]
    assert dataclasses.asdict(chunk) == {
        "category": "faq",
        "questions": QUESTION,
        "answer": EXAMPLE_ANSWER,
        "section_path": None,
        "content_type": "faq",
        "is_key_clause": False,
    }, f"入库的 Chunk 与契约不符:{dataclasses.asdict(chunk)}"

    assert len(write_recorders["vectorize"]) == 1, (
        "刚写进去的行必须**当场**向量化 —— 不然「同一个问题再问就能答对」"
        "要等下一次 scripts/build_kb.py(验收 3 直接落不了地)"
    )
    assert write_recorders["vectorize"][0] == [row], (
        f"向量化的必须是那条新行,实际 {write_recorders['vectorize'][0]}"
    )

    assert session.queue_row.status == "approved"
    assert session.queue_row.approved_answer == EXAMPLE_ANSWER
    assert session.queue_row.reviewed_at is not None, "审核时间必须落库"
    assert r.json() == {"ok": True, "chunks_added": 1, "vectorized": 1}


@pytest.mark.anyio
async def test_approve_uses_the_example_answer_when_the_body_is_empty(
    client_factory, write_recorders,
):
    """不传 `approved_answer`(甚至整个 body 都不传)⇒ 用 `example_answer`。

    审核人点了「通过」但没改答案是最常见的一次操作,而 `body={}` 与**完全不带 body**
    在 FastAPI 上是两条不同的路(后者要求参数可空)。生产上撞到就是 422,
    而前端只会显示一句「请求失败」。
    """
    # ⚠️ 两次调用必须各用**一条自己的队列行**(同一个 session 连续两次的话,
    # 第一次已经把行改成 approved,第二次撞的是 R2 那条 404 —— 于是这条用例
    # 会红在一个与被测行为无关的地方)。
    async with client_factory(
        FakeSession(queue_row=_queue_row(row_id=RQ_ID), kb_rows=[_chunk_row()])
    ) as client:
        empty = await client.post(f"/api/review/{RQ_ID}/approve", json={})
    async with client_factory(
        FakeSession(queue_row=_queue_row(row_id=RQ_ID + 1), kb_rows=[_chunk_row()])
    ) as client:
        bare = await client.post(f"/api/review/{RQ_ID + 1}/approve")

    assert empty.status_code == 200, f"空 body 必须照常通过,实际 {empty.status_code}:{empty.text}"
    assert bare.status_code == 200, f"不带 body 也要能过,实际 {bare.status_code}:{bare.text}"
    assert [c[0].answer for c in write_recorders["chunks"]] == [
        EXAMPLE_ANSWER, EXAMPLE_ANSWER]


@pytest.mark.anyio
async def test_approve_uses_the_edited_answer_and_strips_it(
    client_factory, write_recorders,
):
    """传了 `approved_answer` ⇒ 用**它**(审核人改过的版本),且首尾空白被剥掉。

    没剥的话 `" 答案 "` 会连空白一起进知识库与向量文本,而检索时的 query 是没有
    空白的 —— 这条不报错,只让相似度悄悄差一点。
    """
    session = FakeSession(queue_row=_queue_row(), kb_rows=[_chunk_row()])
    async with client_factory(session) as client:
        r = await client.post(f"/api/review/{RQ_ID}/approve",
                              json={"approved_answer": "  换货运费我们出。  "})
    assert r.status_code == 200, r.text
    assert write_recorders["chunks"][0][0].answer == "换货运费我们出。"
    assert session.queue_row.approved_answer == "换货运费我们出。"


@pytest.mark.anyio
async def test_approve_422_when_the_resolved_answer_is_blank(
    client_factory, write_recorders,
):
    """R3:空答案判在**解析之后**那个值上,而且**一个字都不许写**。

    `approved_answer=""` 与 `example_answer=""` 两条一起才说明问题:实现若只看
    `body.approved_answer`(或在 `or` 上少一层 `strip()`),一个**空串**会被当成
    「传了答案」写进知识库 —— 那条知识块的正文是空的,检索永远召不回它,
    而队列行已经 approved、再也不会有人看它第二眼。
    """
    session = FakeSession(
        queue_row=_queue_row(example_answer="   "), kb_rows=[_chunk_row()])
    async with client_factory(session) as client:
        r = await client.post(f"/api/review/{RQ_ID}/approve",
                              json={"approved_answer": ""})
    assert r.status_code == 422, f"空答案必须 422,实际 {r.status_code}:{r.text}"
    assert write_recorders["chunks"] == [], "校验没过就不许碰知识库"
    assert write_recorders["vectorize"] == []
    assert session.queue_row.status == "pending", "422 不许把队列行改成 approved"
    assert session.statements == [], "422 该在写库之前就返回"


@pytest.mark.anyio
async def test_approving_twice_is_404_and_writes_only_one_chunk(
    client_factory, write_recorders,
):
    """R2:第二次通过 ⇒ **404**,且知识库总共只多了**一条**(判据是 `write_chunks`
    在整个用例里只被调了一次 —— 它自带三元组查重,多调一次不会多写行,
    所以「库里只多一行」这一半**必须**落在 db 文件里,这里守的是「根本没再调」)。
    """
    session = FakeSession(queue_row=_queue_row(), kb_rows=[_chunk_row()])
    async with client_factory(session) as client:
        first = await client.post(f"/api/review/{RQ_ID}/approve", json={})
        second = await client.post(f"/api/review/{RQ_ID}/approve", json={})
    assert first.status_code == 200, first.text
    assert second.status_code == 404, (
        f"已处理过的待审项必须 404(不是静默成功),实际 {second.status_code}:{second.text}"
    )
    assert len(write_recorders["chunks"]) == 1, (
        f"第二次必须**在写库之前**就被挡下,实际写了 {len(write_recorders['chunks'])} 次"
    )
    assert len(write_recorders["vectorize"]) == 1


@pytest.mark.anyio
async def test_reject_touches_nothing_in_the_knowledge_base(
    client_factory, write_recorders, vector_deps,
):
    """R4:驳回**一个字都不碰**知识库,也不向量化。

    「驳回也顺手写一条进去」会很隐蔽:驳回的语义是「这条不该进知识库」,
    而写进去之后**用户问同一个问题就会拿到一条被驳回的答案**,没有任何东西报错。
    """
    session = FakeSession(queue_row=_queue_row(), kb_rows=[_chunk_row()])
    async with client_factory(session) as client:
        r = await client.post(f"/api/review/{RQ_ID}/reject")
    assert r.status_code == 200, r.text

    assert write_recorders["chunks"] == [], "驳回不许写知识库"
    assert write_recorders["vectorize"] == [], "驳回不许向量化"
    # 连替身 Milvus 都不该被碰(打桩 `vectorize_rows` 会掩盖这一点:
    # 一个「先 upsert 再判断要不要驳回」的实现照样能让上面两条通过)。
    assert vector_deps.store.upserts == [], "驳回不许碰 Milvus"
    assert vector_deps.store.ensure_calls == 0, "驳回连集合都不该去建"
    assert session.queue_row.status == "rejected"
    assert session.queue_row.reviewed_at is not None
    assert session.queue_row.approved_answer is None, (
        "驳回不该留下一个「核准答案」(它从没被核准过)"
    )
    # 一条语句都不该打(驳回不需要读知识库/池子)
    assert session.statements == []


@pytest.mark.anyio
async def test_rejecting_twice_is_404(client_factory, write_recorders):
    """R4 的后半:驳回同样幂等 —— 第二次 404,不是静默成功。

    「静默成功」在这里尤其坏:审核台会把一次重复点击读成「又驳回了一条」。
    """
    session = FakeSession(queue_row=_queue_row())
    async with client_factory(session) as client:
        first = await client.post(f"/api/review/{RQ_ID}/reject")
        second = await client.post(f"/api/review/{RQ_ID}/reject")
    assert first.status_code == 200, first.text
    assert second.status_code == 404, (
        f"已处理过的待审项再驳回必须 404,实际 {second.status_code}:{second.text}"
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "exc", [ToolInfrastructureError("Milvus 连不上"), RuntimeError("裸的驱动错误")],
    ids=["translated", "raw"],
)
async def test_a_failed_vectorize_is_a_502_and_leaves_the_row_pending(
    client_factory, vector_deps, deduping_write_chunks, exc,
):
    """R5:向量化失败 ⇒ **502**,队列行**仍是 pending**,而且不许写半个状态。

    两种异常都注入(与 `tests/test_api_feedback.py` 同款理由):生产里
    `vectorize_rows` 会把 pymilvus / torch 的**裸**异常原样抛出来
    (`app/kb/writer.py` 不是翻译边界,只有 `retrieval/search.py` 是),
    只注入 `ToolInfrastructureError` 的话,一个「只认这个类型」的实现会把
    真实故障变成 500 + 一段裸的驱动文本。
    """
    # 用**真的** `vectorize_rows`(不打桩),否则验的就不是「它失败了会怎样」
    vector_deps.store = FakeStore(fail_first_upserts=1, exc=exc)
    session = FakeSession(queue_row=_queue_row())
    deduping_write_chunks(session)          # 行先真的写进去(否则没东西可向量化)
    async with client_factory(session) as client:
        r = await client.post(f"/api/review/{RQ_ID}/approve", json={})
    assert r.status_code == 502, (
        f"向量化失败必须 502(不许降级成「写进 MySQL 了但检索不到」),"
        f"实际 {r.status_code}:{r.text}"
    )
    assert r.json()["detail"] == review_api.VECTORIZE_FAILED_DETAIL, (
        "502 的文案必须是**固定文案** —— 原样回显 pymilvus/torch 的 str(exc) "
        "会把内部实现细节送出站"
    )
    assert session.queue_row.status == "pending", (
        f"502 之后队列行必须仍是 pending(审核人才能重试),"
        f"实际 {session.queue_row.status}"
    )
    assert session.queue_row.approved_answer is None


#: F1 用的密钥:它同时被注入**模块常量**与**这一份 settings**,两者都不含它时
#: 这条用例照不亮任何东西(所以下面断言了两份 key 必然不同)。
CALLER_KEY = "sk-t15-caller-0a1b2c3d4e5f"
_KEYED_DETAIL = f"知识入库失败(向量化未完成),原始错误里的 key={CALLER_KEY}"


@pytest.mark.anyio
async def test_the_502_detail_goes_through_redact_api_key(
    client_factory, vector_deps, deduping_write_chunks, monkeypatch,
):
    """F1:502 的文案**也要过 `redact_api_key`** —— 哪怕它今天是个常量。

    `app/api/refund.py:_infra_failure` 与 `app/api/chat.py` 的那条 502 都是这么
    做的,理由写在那两处:**把「所有出站文本都过同一个出口」这条规则留成无例外的**,
    比每次判断「这个字符串要不要脱敏」可靠。本端点是全章**唯一**绕开它的出口。

    ⚠️ 注入的是**处理之前**的形态:直接把模块常量改成**真的带上密钥**的那一句。
    把已经脱敏的值喂进去的话,这条用例对「端点压根没调 redact」**恒真**
    (本仓那条「在**处理之后**注入」的假绿形态,`tests/test_api_ticket.py:151-212`
    是它的对照写法)。

    判据不是「等于常量」(那是同义反复),而是**出站体里没有那份 key、且有抹掉的
    痕迹** —— 顺带把「用的是**调用方这一份** settings 的 key」也钉住:实现若去读
    `get_settings()` 那份(真 `.env`),注入的这份 key 根本不在它的替换表里,
    下面第一条断言当场红。
    """
    assert CALLER_KEY != get_settings().openai_api_key, (
        "前提:两份 key 必须不同 —— 相同的话「拿进程全局那份脱敏」这个缺陷照不亮"
    )
    monkeypatch.setattr(review_api, "VECTORIZE_FAILED_DETAIL", _KEYED_DETAIL)
    vector_deps.store = FakeStore(fail_first_upserts=1)
    session = FakeSession(queue_row=_queue_row())
    deduping_write_chunks(session)

    async with client_factory(session, openai_api_key=CALLER_KEY) as client:
        r = await client.post(f"/api/review/{RQ_ID}/approve", json={})

    assert r.status_code == 502, f"实际 {r.status_code}:{r.text}"
    detail = r.json()["detail"]
    assert CALLER_KEY not in detail, (
        f"密钥随 502 的文案出站了 —— 这是全章唯一一个没走 redact_api_key 的出口:"
        f"{detail!r}"
    )
    assert "***" in detail, (
        f"抹掉的痕迹该留着(剩下的那句才是排查用的):{detail!r}"
    )


@pytest.mark.anyio
async def test_retry_after_a_failed_vectorize_really_vectorizes(
    client_factory, vector_deps, deduping_write_chunks,
):
    """H1(本章最尖的一处):**第一次向量化失败、重试必须真的再向量化一次**。

    走的是**真的** `vectorize_rows`,只把 Milvus 换成「第一次抛、之后放行」的替身,
    以及一个照着真行为查重的 `write_chunks`(见 `deduping_write_chunks`)。

    为什么非要真不可:brief 里那段 `if added:` 在重试时必然出事 ——
    重试时 `write_chunks` 返回 **0**(三元组已存在),于是 `vectorize_rows`
    **一次都不会被调用**:端点回 200、队列行变 approved、审核人被告知成功,
    而那条知识块**永远停在 pending**(MySQL 有、Milvus 没有)⇒ 那个问题
    **永久答不上**,只能等谁碰巧跑一次 `scripts/build_kb.py`。全程**不报任何错**。

    判别力全在 `upserts` 那条:把 `vectorize_rows` 也打桩的用例在
    「`if added:` 写错」的实现下**照样全绿**(它只看得见「被调了几次」,
    而这里要断的是**第二次真的把行写进了向量库**)。
    """
    vector_deps.store = FakeStore(fail_first_upserts=1)
    session = FakeSession(queue_row=_queue_row())      # kb_rows 从**空**开始:
    deduping_write_chunks(session)                     # 第一次通过才写进去那一行

    async with client_factory(session) as client:
        first = await client.post(f"/api/review/{RQ_ID}/approve", json={})
        assert first.status_code == 502, f"前置:第一次必须失败,实际 {first.status_code}"
        assert vector_deps.store.upserts == [], "前置:第一次的 upsert 不该落地"
        assert [r.vectorize_status for r in session.kb_rows] == ["pending"], (
            "前置:行已经写进 MySQL 了(所以重试会命中查重),只是没向量化"
        )

        second = await client.post(f"/api/review/{RQ_ID}/approve", json={})

    assert second.status_code == 200, (
        f"重试必须成功(Milvus 已经好了),实际 {second.status_code}:{second.text}"
    )
    assert second.json()["chunks_added"] == 0, (
        f"前置:重试时 `write_chunks` 返回的**就是 0**(三元组已存在),"
        f"实际 {second.json()}"
    )
    assert vector_deps.store.upserts == [[str(CHUNK_ID)]], (
        f"重试**必须真的走到向量化** —— 用 `if added:` 把关的话向量化永不发生,"
        f"而端点回 200。实际 upserts={vector_deps.store.upserts},"
        f"attempts={vector_deps.store.attempts}"
    )
    assert [r.vectorize_status for r in session.kb_rows] == ["done"]
    assert session.kb_rows[0].vector_id == str(CHUNK_ID)
    assert session.queue_row.status == "approved"


def _delete_statements(session) -> list[tuple[str, dict]]:
    return [
        (sql, params) for sql, params in session.statements
        if sql.startswith("DELETE")
    ]


def _param_values(params: dict) -> list:
    """绑定参数摊平(那条 `DELETE ... WHERE id IN (...)` 的 id 是个**列表**)。"""
    values: list = []
    for v in params.values():
        values.extend(v if isinstance(v, list) else [v])
    return values


@pytest.mark.anyio
async def test_a_failed_vectorize_discards_the_draft_it_just_wrote(
    client_factory, vector_deps, deduping_write_chunks,
):
    """F4:502 这条路上,**把本次刚写进去的待向量化草稿删掉**。

    为什么非清不可(复审员复现的场景):第一次通过(答案 A)把行写进了 MySQL、
    然后 Milvus 炸 ⇒ 502;审核人把答案改成 B 重试 ⇒ (Q,B) 被向量化、队列行
    approved,而 **(Q,A,pending) 成了永久残留**。下一次 `build_kb.py` 或管理台的
    向量化任务(`app/kb/writer.py` 的 `WHERE vectorize_status == 'pending'`,
    **不筛 category**)会把它送进 Milvus ⇒ **一条没人核准过的草稿静默进了
    可检索知识库**,全程不报错。

    ⚠️ 判据是「**本次新增的 id**」,不是「这个三元组的所有 pending 行」——
    后者会删掉**别人**写进去的行(同一三元组的 pending 行可能是另一个并发请求
    刚写的、或上一次崩溃留下的)。「不许删别人的行」由下一条用例正面钉住。
    """
    vector_deps.store = FakeStore(fail_first_upserts=1)
    session = FakeSession(queue_row=_queue_row())     # kb_rows 从空开始 ⇒ 本次真的写了一条
    deduping_write_chunks(session)
    async with client_factory(session) as client:
        r = await client.post(f"/api/review/{RQ_ID}/approve", json={})
    assert r.status_code == 502, f"前置:这一次必须失败,实际 {r.status_code}"

    deletes = _delete_statements(session)
    assert len(deletes) == 1, (
        f"失败路径必须清掉本次写入的草稿(否则它会静默进知识库),实际 {deletes}"
    )
    _sql, params = deletes[0]
    values = _param_values(params)
    assert CHUNK_ID in values, (
        f"删的必须是**本次**写进去的那一条,实际绑定 {params}"
    )
    assert "pending" in values, (
        f"只许删还没向量化的 —— 已经 done 的行是别人写进 Milvus 的,删了就是丢知识,"
        f"实际绑定 {params}"
    )


@pytest.mark.anyio
async def test_a_failed_vectorize_does_not_delete_a_row_it_did_not_write(
    client_factory, vector_deps, deduping_write_chunks,
):
    """hazard ①:库里那条 pending 行**不是本次写的** ⇒ 一条都不许删。

    场景很常见:`write_chunks` 命中三元组查重、返回 0(上一次请求崩在「写库」与
    「向量化」之间留下的行,或另一个并发请求刚写的),此时失败路径**必须什么都不删**。
    删了就是把别人的行清掉 —— 那一行下一次重试/全表扫描还会用到,而**两边都不报错**。
    """
    vector_deps.store = FakeStore(fail_first_upserts=1)
    existing = _chunk_row()                     # 调用前就在库里的 pending 行
    session = FakeSession(queue_row=_queue_row(), kb_rows=[existing])
    deduping_write_chunks(session)
    async with client_factory(session) as client:
        r = await client.post(f"/api/review/{RQ_ID}/approve", json={})
    assert r.status_code == 502, f"前置:这一次必须失败,实际 {r.status_code}"

    assert _delete_statements(session) == [], (
        "本次一条都没写进去(`write_chunks` 返回 0)⇒ 不许删任何行"
    )
    assert session.kb_rows == [existing], "别人留下的行必须原样在库里"
    assert existing.vectorize_status == "pending", "它还得留在 pending 上等下一次"


@pytest.mark.anyio
async def test_already_vectorized_chunks_are_left_alone(
    client_factory, deduping_write_chunks,
):
    """反向:三元组已在知识库里且**已经向量化过** ⇒ 不该重做。

    这条防的是「H1 的修法走过头」:少了 `vectorize_status != 'done'` 那一步的话,
    每一次通过都要白烧一次嵌入(CPU 上秒级)+ 一次 Milvus 往返,而
    「同一个问题被两个会话都问了、审核人点了两次通过」是很常见的形状。

    ⚠️ 这里**只断 SQL**(`WHERE` 里有没有那一列):本文件的 session 是替身,
    它按表名把行交出去、**不替端点过滤** —— 真正的「done 的行不会被再向量化」
    由 `tests/test_api_review_db.py::test_an_already_done_chunk_is_not_revectorized`
    在真库上守。
    """
    session = FakeSession(
        queue_row=_queue_row(), kb_rows=[_chunk_row(vectorize_status="done")])
    deduping_write_chunks(session)      # 三元组已存在 ⇒ 它照样返回 0
    async with client_factory(session) as client:
        r = await client.post(f"/api/review/{RQ_ID}/approve", json={})
    assert r.status_code == 200, r.text
    assert r.json()["chunks_added"] == 0, "前置:三元组已在库里,没新增"

    seen = _where_of(session, table="knowledge_chunks", require="vectorize_status !=")
    assert "knowledge_chunks.vectorize_status" in seen["where"], (
        f"「已 done 的不重做」必须在 SQL 里(WHERE = {seen['where']!r})"
    )
    assert "done" in seen["params"].values(), (
        f"多出来的那一句该是 vectorize_status != 'done',实际绑定 {seen['params']}"
    )
