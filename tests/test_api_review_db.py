"""审核端点的**真库**那一半(ch09 §9)。

`tests/test_api_review.py` 用替身 session 钉接线与失败路径;这里打真 MySQL,
因为下面四件事**替身证不出来**(本仓编目过的形态 ⑦:替身替被测对象完成了语义):

① **过滤在不在 SQL 里**(H2)。替身 session 只会把预设的行交出去 —— 一个
   「全表捞回来、在 Python 里 `if row.status != status: continue`」的实现
   在替身下**照样绿**,而它的代价是每次开审核页拖整张表。真库上一条 pending
   一条非 pending,谁被漏掉当场看得见;
② **approve 真的写了知识库**(R2)。`write_chunks` 自带三元组查重 ⇒ 「两次通过
   只多一行」这件事**只有真库能断**(替身里那个数字是替身编的);
③ **H1(重试必须真的向量化)**:第一次向量化失败后,库里留下一条
   `vectorize_status='pending'` 的行;重试时 `write_chunks` 返回 **0**。
   这条链路上有两处真行为(查重 + 按三元组回查待向量化的行),缺一不可;
④ **详情真的按 `matched_review_id` 关联**(H3)。两条池子行、快照非空 ——
   只关联一条的实现在这里红,而在只有一条关联行的夹具下**恒绿**。

## 用例的边界

- Milvus / 嵌入**一律替身**(本机 Milvus 没起,且「单测不加载 2.2GB 权重」是硬规矩)。
  所以这里验的是「端点在向量化的**边界**上做对了没有」,不是 Milvus 真的收到了向量 ——
  真 Milvus 由验收脚本覆盖。
- `write_chunks` / `vectorize_rows` **都是真的**(H1 的全部意义就在这两者的组合上)。
- 池子里那 **30 行前几章的旧数据一个字都不碰**:本文件只按**自己造的 id** 增删,
  清理断言也按 id 走。
"""

import httpx
import pytest
from sqlalchemy import delete, text

from app.api import review as review_api
from app.db.base import get_engine, get_sessionmaker
from app.db.models import KnowledgeChunk, LowConfidenceQuestion, ReviewQueue
from app.main import app
from app.tools.errors import ToolInfrastructureError

pytestmark = pytest.mark.db

#: 探针会话 id(≤ 32 字符)。`t15probe-` 前缀**结构上**撞不到真实数据
#: (库里的真实会话 id 是 `uuid4().hex` 与 `acceptance-*`)。
PROBE_CONV = "t15probe-review"

#: 每个用例一句**独有**的问题 —— 它同时是知识块与队列行的过滤条件。
Q_LIST_PENDING = "t15 探针:待审列表只该回我这一条"
Q_LIST_OTHER = "t15 探针:已经处理过的那一条"
Q_DETAIL = "t15 探针:详情要带原话与快照"
Q_APPROVE = "t15 探针:通过之后要能检索到"
Q_RETRY = "t15 探针:第一次入库失败、重试要补上"
Q_REJECT = "t15 探针:驳回一个字都不许写"
Q_DONE = "t15 探针:已经在库里的那条不该重做"
Q_ORPHAN = "t15 探针:崩溃遗留在 pending 上的那一条"

ANSWER = "探针答案:换货运费由我们承担。"


class _FakeEmbedder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t))] * 3 for t in texts]


class _FakeStore:
    """Milvus 替身。`fail_first_upserts` 次抛 `exc`,之后放行(抖一下,不是一直抖)。"""

    def __init__(self, *, fail_first_upserts: int = 0,
                 exc: BaseException | None = None):
        self.remaining_failures = fail_first_upserts
        self.exc = exc if exc is not None else ToolInfrastructureError("Milvus 连不上")
        self.ensure_calls = 0
        self.upserts: list[list[str]] = []

    def ensure_collection(self):
        self.ensure_calls += 1

    def upsert(self, ids, texts, categories, vectors):
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise self.exc
        self.upserts.append(list(ids))


@pytest.fixture(autouse=True)
def fake_vector_deps(monkeypatch):
    """**每条用例**都把 Milvus / 嵌入换掉(本机 Milvus 没起,权重也不该加载)。"""

    class Deps:
        def __init__(self):
            self.store = _FakeStore()
            self.embedder = _FakeEmbedder()

    deps = Deps()
    monkeypatch.setattr(
        review_api, "get_vector_store", lambda uri, collection: deps.store)
    monkeypatch.setattr(review_api, "get_embedder", lambda *a, **kw: deps.embedder)
    return deps


@pytest.fixture
def client():
    """ASGI 客户端。**同步夹具返回未 `__aenter__` 的 AsyncClient** —— 与
    `tests/test_api_feedback.py` 同一个形状:异步夹具在这个仓库的 anyio 装置下
    会在 pytest 的 finalizer 记账上炸(`assert not self._finalizers`),
    而那与用例毫无关系。`ASGITransport` 也不跑 lifespan(不预热权重)。
    """
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


# --------------------------------------------------------------------------
# 造数据 / 收尾(一律按**自己造的 id**)
# --------------------------------------------------------------------------


async def _make_queue_row(*, question, answer=ANSWER, status="pending") -> int:
    async with get_sessionmaker()() as session:
        rq = ReviewQueue(
            standard_question=question,
            example_answer=answer,
            first_raw_question=question,
            status=status,
            source_conversation_id=PROBE_CONV,
        )
        session.add(rq)
        await session.commit()
        return rq.id


async def _make_pool_row(*, review_id: int, question: str, snapshot) -> int:
    async with get_sessionmaker()() as session:
        row = LowConfidenceQuestion(
            question=question,
            source_conversation_id=PROBE_CONV,
            entry_point="用户反馈",
            reject_reason="用户点了 👎(未解决)",
            evidence_snapshot=snapshot,
            matched_review_id=review_id,
        )
        session.add(row)
        await session.commit()
        return row.id


async def _make_chunk_row(*, question, answer=ANSWER, status="done") -> int:
    """直接插一条知识块(**不经 `write_chunks`**)—— 「库里早有这条」的造法。"""
    async with get_sessionmaker()() as session:
        row = KnowledgeChunk(
            category="faq", questions=question, answer=answer,
            section_path=None, content_type="faq", is_key_clause=False,
            vectorize_status=status,
        )
        session.add(row)
        await session.commit()
        return row.id


async def _cleanup(*, queue_ids=(), questions=()) -> None:
    """删掉本用例造的行。**删知识块按 (category, questions) 精确删** —— 不用
    `LIKE`,也不删别人的行(池子里有 30 行前几章的旧数据)。"""
    async with get_sessionmaker()() as session:
        for q in questions:
            await session.execute(
                delete(KnowledgeChunk).where(
                    KnowledgeChunk.category == "faq",
                    KnowledgeChunk.questions == q,
                )
            )
        if queue_ids:
            await session.execute(
                delete(LowConfidenceQuestion).where(
                    LowConfidenceQuestion.matched_review_id.in_(list(queue_ids))
                )
            )
            await session.execute(
                delete(ReviewQueue).where(ReviewQueue.id.in_(list(queue_ids)))
            )
        await session.commit()
    await get_engine().dispose()


async def _queue_row(queue_id: int) -> dict:
    """**新 session** 回查(身份映射持弱引用,同 session 重读会变成「靠 refcount 走运」)。"""
    async with get_sessionmaker()() as session:
        rows = (await session.execute(
            text("SELECT status, approved_answer, reviewed_at, occurrences "
                 "FROM review_queue WHERE id = :i"),
            {"i": queue_id},
        )).mappings().all()
    assert len(rows) == 1, f"队列行 {queue_id} 该恰好一条,实际 {len(rows)}"
    return dict(rows[0])


async def _chunk_rows(question: str) -> list[dict]:
    async with get_sessionmaker()() as session:
        rows = (await session.execute(
            text("SELECT id, answer, vectorize_status, vector_id "
                 "FROM knowledge_chunks WHERE category = 'faq' AND questions = :q "
                 "ORDER BY id"),
            {"q": question},
        )).mappings().all()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# 一、列表(H2)
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_queue_returns_only_pending_rows_from_sql(client):
    """H2:待审列表**只**回 pending —— 而且这条过滤是真的打在 SQL 上的。

    SQL 那一半在这里判不了(端点内部怎么取的看不见),但**行为**判得了:
    真库上并排插一条 pending、一条 approved,回来看谁在谁不在。
    替身 session 版本(它断言的是编译出的 `WHERE`)见
    `tests/test_api_review.py::test_queue_filter_is_in_sql_not_in_python` ——
    两条合起来才是完整的。
    """
    pending_id = await _make_queue_row(question=Q_LIST_PENDING, status="pending")
    other_id = await _make_queue_row(question=Q_LIST_OTHER, status="approved")
    try:
        r = await client.get("/api/review/queue")
        assert r.status_code == 200, f"实际 {r.status_code}:{r.text}"
        ids = [row["id"] for row in r.json()]

        assert pending_id in ids, (
            f"待审的那条必须回,实际回了 {len(ids)} 条(id 里没有 {pending_id})"
        )
        assert other_id not in ids, (
            f"`approved` 的那条**不许**出现在待审列表里(审核人会去处理一条已经处理过的)"
        )
        row = next(r0 for r0 in r.json() if r0["id"] == pending_id)
        assert row["standard_question"] == Q_LIST_PENDING
        assert row["status"] == "pending"
    finally:
        await _cleanup(queue_ids=[pending_id, other_id],
                       questions=[Q_LIST_PENDING, Q_LIST_OTHER])


@pytest.mark.anyio
async def test_queue_status_filter_selects_that_status(client):
    """`?status=` 是**能用**的(不是只有默认值那条路走得通)。"""
    approved_id = await _make_queue_row(question=Q_LIST_OTHER, status="approved")
    try:
        r = await client.get("/api/review/queue", params={"status": "approved"})
        assert r.status_code == 200, r.text
        ids = [row["id"] for row in r.json()]
        assert approved_id in ids, f"按 approved 过滤必须能查到它,实际 {ids}"
    finally:
        await _cleanup(queue_ids=[approved_id], questions=[Q_LIST_OTHER])


# --------------------------------------------------------------------------
# 二、详情(H3)
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_detail_returns_every_linked_raw_question_and_snapshot(client):
    """H3(真库):**两条**归并进来的原话 + 各自**非空**的快照,一条不落。

    夹具刻意两条 + 快照都非空:只关联一条的实现在「夹具只有一条」时**恒绿**,
    而快照为 `None` 时「原样带出来」与「恒返回 null」也分不开。
    """
    snap_a = [{"chunk_id": 11, "score": 0.5, "section_path": "退换货/政策",
               "answer": "七天内可申请退货。"}]
    snap_b = [{"chunk_id": 12, "score": 0.25, "section_path": None,
               "answer": "质量问题由我们承担运费。"}]
    queue_id = await _make_queue_row(question=Q_DETAIL)
    await _make_pool_row(review_id=queue_id, question="换货运费谁出", snapshot=snap_a)
    await _make_pool_row(review_id=queue_id, question="换货要自己出邮费吗", snapshot=snap_b)
    try:
        r = await client.get(f"/api/review/{queue_id}")
        assert r.status_code == 200, f"实际 {r.status_code}:{r.text}"
        body = r.json()
        assert body["standard_question"] == Q_DETAIL
        raws = body["raw_questions"]
        assert [q["question"] for q in raws] == ["换货运费谁出", "换货要自己出邮费吗"], (
            f"两条原话必须都在(按 id 升序),实际 {raws}"
        )
        assert [q["evidence_snapshot"] for q in raws] == [snap_a, snap_b], (
            f"每条的快照必须跟着自己那条原话,实际 {[q['evidence_snapshot'] for q in raws]}"
        )
    finally:
        await _cleanup(queue_ids=[queue_id], questions=[Q_DETAIL])


# --------------------------------------------------------------------------
# 三、通过(R2 / H1)
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_approve_writes_exactly_one_chunk_and_vectorizes_it(client):
    """R2 + brief 第 3 条:通过 ⇒ 库里**恰好多一条**、它是 done、且真的 upsert 了;
    第二次通过 ⇒ 404,库里**仍然只有那一条**。

    「恰好多一条」是**真**查重(`write_chunks` 的三元组)挣来的,替身版本只能断
    「它被调了几次」。而 `vector_id == str(id)` 是「Milvus 主键与 MySQL 行对齐」
    这条约定(`app/kb/writer.py` 的注释)在审核这条路上的实证。
    """
    queue_id = await _make_queue_row(question=Q_APPROVE)
    try:
        assert await _chunk_rows(Q_APPROVE) == [], "前置:知识库里本来没有这一条"

        first = await client.post(f"/api/review/{queue_id}/approve", json={})
        assert first.status_code == 200, f"实际 {first.status_code}:{first.text}"
        assert first.json() == {"ok": True, "chunks_added": 1, "vectorized": 1}

        chunks = await _chunk_rows(Q_APPROVE)
        assert len(chunks) == 1, f"知识库该恰好多一条,实际 {len(chunks)}:{chunks}"
        c = chunks[0]
        assert c["answer"] == ANSWER, "入库的必须是核准答案"
        assert c["vectorize_status"] == "done", (
            f"通过之后它必须**当场**是 done(否则那个问题要等下次 build_kb 才答得上),"
            f"实际 {c['vectorize_status']}"
        )
        assert c["vector_id"] == str(c["id"]), "Milvus 主键必须与 MySQL 行 id 对齐"

        row = await _queue_row(queue_id)
        assert row["status"] == "approved"
        assert row["approved_answer"] == ANSWER
        assert row["reviewed_at"] is not None, "审核时间必须落库"

        second = await client.post(f"/api/review/{queue_id}/approve", json={})
        assert second.status_code == 404, (
            f"重复通过必须 404,实际 {second.status_code}:{second.text}"
        )
        assert len(await _chunk_rows(Q_APPROVE)) == 1, (
            "两次通过总共只许留下一条知识块"
        )
    finally:
        await _cleanup(queue_ids=[queue_id], questions=[Q_APPROVE])


@pytest.mark.anyio
async def test_retry_after_a_failed_vectorize_vectorizes_on_the_second_call(
    client, fake_vector_deps,
):
    """**H1 + F4(真库)**:第一次向量化炸了 ⇒ 502、队列行留 pending、
    **本次写的草稿被清掉**;重试 ⇒ 重新写入并**真的走到向量化**。

    五个观测:
      ① 第一次 502(不降级成「写进 MySQL 了但检索不到」);
      ② 队列行**仍是 pending** —— 它必须**先别动**,审核人才有得重试;
      ③ **库里没有残留的待向量化草稿**(F4):留在那儿的话,下一次全表
         `vectorize_pending`(不筛 category)会把它送进 Milvus ⇒ 一条**没人核准过**
         的草稿静默进了可检索知识库。这条也正是「改答案重试」那个场景的关法 ——
         孤儿是**第一次**的失败产生的,第一次的失败处理器就把它删掉了;
      ④ 重试重新写入(`chunks_added == 1`)并真的 upsert(带那条块的 id);
      ⑤ 重试之后恰好一条、且是 done。

    ⚠️ 判据是 **upserts 边界**(真 `vectorize_rows` + 「第一次抛、之后放行」的
    Milvus 替身),不是「`vectorize_rows` 被调了几次」—— 后者在 `if added:` 那种
    实现下**照样绿**。「重试时 `write_chunks` 返回 0」那条真路(崩溃遗留的孤儿)
    由下一条用例守,因为**本用例的流程在 F4 之后不再经过它**。
    """
    queue_id = await _make_queue_row(question=Q_RETRY)
    fake_vector_deps.store = _FakeStore(fail_first_upserts=1)
    store = fake_vector_deps.store
    try:
        first = await client.post(f"/api/review/{queue_id}/approve", json={})
        assert first.status_code == 502, (
            f"向量化失败必须 502,实际 {first.status_code}:{first.text}"
        )
        assert first.json()["detail"] == review_api.VECTORIZE_FAILED_DETAIL

        assert (await _queue_row(queue_id))["status"] == "pending", (
            "502 之后队列行必须仍是 pending(否则审核人没有重试的入口)"
        )
        assert await _chunk_rows(Q_RETRY) == [], (
            "F4:失败之后**不许留下**这条三元组的待向量化草稿 —— 它会被下一次"
            "全表扫描送进 Milvus(一条没人核准过的答案),而全程不报错"
        )
        assert store.upserts == [], "前置:第一次的 upsert 不该落地"

        second = await client.post(f"/api/review/{queue_id}/approve", json={})
        assert second.status_code == 200, (
            f"重试必须成功,实际 {second.status_code}:{second.text}"
        )
        assert second.json()["chunks_added"] == 1, (
            f"草稿已被清掉 ⇒ 重试是**重新写一条**,实际 {second.json()}"
        )
        assert second.json()["vectorized"] == 1, (
            f"重试**必须真的向量化那一条** —— 这就是 H1,实际 {second.json()}"
        )
        after = await _chunk_rows(Q_RETRY)
        assert [c["vectorize_status"] for c in after] == ["done"], (
            f"重试之后必须补上向量化,实际 {after}"
        )
        assert len(after) == 1, f"重试不许写出第二条,实际 {len(after)}"
        assert store.upserts == [[str(after[0]["id"])]], (
            f"那次 upsert 必须带这条块的 id,实际 {store.upserts}"
        )
        assert (await _queue_row(queue_id))["status"] == "approved"
    finally:
        await _cleanup(queue_ids=[queue_id], questions=[Q_RETRY])


@pytest.mark.anyio
async def test_a_pending_orphan_from_a_crash_is_still_vectorized(
    client, fake_vector_deps,
):
    """H1 的真路:**崩溃遗留的 pending 行**(不是这次写的)⇒ 通过时必须被向量化,
    哪怕 `write_chunks` 返回 **0**。

    为什么这条不能少:F4 的清理只覆盖「本次调用自己写进去的行」—— 进程被杀 /
    请求被取消 / 清理自己失败,都会把一条 pending 行留在库里,而**它分不出是不是
    别人的**。那条路正是 `if added:` 会失效的地方:重试命中三元组查重 ⇒ `added == 0`
    ⇒ 向量化永不发生 ⇒ 端点回 200、队列行 approved,而那块知识**永远答不上**。

    造法:直接插一条 **pending** 的知识块(不经 `write_chunks`),再通过一条
    standard_question/answer 与它逐字相同的待审行 —— 这正是崩溃之后库里的样子。
    """
    orphan_id = await _make_chunk_row(question=Q_ORPHAN, status="pending")
    queue_id = await _make_queue_row(question=Q_ORPHAN)
    try:
        r = await client.post(f"/api/review/{queue_id}/approve", json={})
        assert r.status_code == 200, f"实际 {r.status_code}:{r.text}"
        assert r.json()["chunks_added"] == 0, (
            f"前置:三元组已经在库里(崩溃遗留)⇒ 查重命中、本次没新增,实际 {r.json()}"
        )
        assert r.json()["vectorized"] == 1, (
            f"遗留的 pending 行必须被向量化 —— 挂在 `if added:` 上的话这里是 0,"
            f"而端点照样回 200(那块知识永远答不上),实际 {r.json()}"
        )
        assert fake_vector_deps.store.upserts == [[str(orphan_id)]], (
            f"upsert 必须带那条遗留行的 id,实际 {fake_vector_deps.store.upserts}"
        )
        rows = await _chunk_rows(Q_ORPHAN)
        assert [c["vectorize_status"] for c in rows] == ["done"]
        assert len(rows) == 1, "不许再写一条出来(查重已经命中了)"
        assert (await _queue_row(queue_id))["status"] == "approved"
    finally:
        await _cleanup(queue_ids=[queue_id], questions=[Q_ORPHAN])


@pytest.mark.anyio
async def test_an_already_done_chunk_is_not_revectorized(client, fake_vector_deps):
    """反向:三元组已在库里且**已经 done** ⇒ 一次 upsert 都不发。

    造法:直接插一条 done 的知识块(不经 `write_chunks`),再通过一条
    standard_question/answer 与它逐字相同的待审行。查重命中 ⇒ `added == 0`,
    而「还没 done」那个门把它挡在向量化之外 —— 少了那道门,每一次通过都要白烧
    一次嵌入(CPU 上秒级)+ 一次 Milvus 往返。
    """
    await _make_chunk_row(question=Q_DONE, status="done")
    queue_id = await _make_queue_row(question=Q_DONE)
    try:
        r = await client.post(f"/api/review/{queue_id}/approve", json={})
        assert r.status_code == 200, f"实际 {r.status_code}:{r.text}"
        assert r.json()["chunks_added"] == 0, "前置:三元组已在库里,没新增"
        assert r.json()["vectorized"] == 0, (
            f"已经是 done 的行不该再向量化,实际 {r.json()}"
        )
        assert fake_vector_deps.store.upserts == [], (
            f"不该有任何 upsert,实际 {fake_vector_deps.store.upserts}"
        )
        assert len(await _chunk_rows(Q_DONE)) == 1, "不许写出第二条"
        assert (await _queue_row(queue_id))["status"] == "approved"
    finally:
        await _cleanup(queue_ids=[queue_id], questions=[Q_DONE])


# --------------------------------------------------------------------------
# 四、驳回(R4)
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_reject_changes_only_the_status(client):
    """R4(真库):驳回 ⇒ 知识库行数不变、没有 upsert、状态变 rejected,再驳回 404。

    「驳回也顺手写一条进去」在这里红:`_chunk_rows` 会从 0 变 1。
    """
    queue_id = await _make_queue_row(question=Q_REJECT)
    try:
        assert await _chunk_rows(Q_REJECT) == [], "前置:知识库里本来没有这一条"

        r = await client.post(f"/api/review/{queue_id}/reject")
        assert r.status_code == 200, f"实际 {r.status_code}:{r.text}"
        assert await _chunk_rows(Q_REJECT) == [], (
            "驳回**一个字都不许写进知识库** —— 写进去的话用户问同一个问题会拿到"
            "一条被驳回的答案,而没有东西报错"
        )
        row = await _queue_row(queue_id)
        assert row["status"] == "rejected"
        assert row["reviewed_at"] is not None
        assert row["approved_answer"] is None, "驳回不该留下核准答案"

        again = await client.post(f"/api/review/{queue_id}/reject")
        assert again.status_code == 404, (
            f"重复驳回必须 404(静默成功会让审核台把重复点击读成又处理了一条),"
            f"实际 {again.status_code}"
        )
    finally:
        await _cleanup(queue_ids=[queue_id], questions=[Q_REJECT])


@pytest.mark.anyio
async def test_an_approved_row_can_not_be_rejected_later(client):
    """通过之后**不能再驳回**:那会让「已入库的知识」与「被驳回」同时成立。

    两个端点共用同一个 `status != 'pending'` 守卫,这条钉的就是「共用」。
    """
    queue_id = await _make_queue_row(question=Q_APPROVE)
    try:
        first = await client.post(f"/api/review/{queue_id}/approve", json={})
        assert first.status_code == 200, first.text
        r = await client.post(f"/api/review/{queue_id}/reject")
        assert r.status_code == 404, (
            f"已通过的待审项不许再被驳回,实际 {r.status_code}:{r.text}"
        )
        assert (await _queue_row(queue_id))["status"] == "approved"
    finally:
        await _cleanup(queue_ids=[queue_id], questions=[Q_APPROVE])
