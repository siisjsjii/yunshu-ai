"""KnowledgeRetriever:嵌入 → 混合检索 → 回查 → 重排 → 阈值过滤 → 组装。

过滤与错误翻译用手替身(不碰库);回查、排序、「Milvus 有而 MySQL 没有」打
@pytest.mark.db。
"""

import asyncio

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import OperationalError

from app.db.base import get_engine, get_sessionmaker
from app.db.models import KnowledgeChunk
from app.retrieval.search import KnowledgeRetriever, RetrievedChunk
from app.tools.errors import ToolInfrastructureError

SCRATCH_CATEGORY = "ch03-probe-search"


class _FakeEmbedder:
    def __init__(self, error: Exception | None = None):
        self.calls: list = []
        self.error = error

    def encode(self, texts):
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return [[0.1, 0.1, 0.1] for _ in texts]


class _FakeStore:
    def __init__(self, hits=(), error: Exception | None = None):
        self.calls: list = []
        self.hits = list(hits)
        self.error = error

    def hybrid_search(self, vector, text, top_k, category=None):
        self.calls.append((list(vector), text, top_k))
        if self.error is not None:
            raise self.error
        return list(self.hits)


class _FakeReranker:
    """按候选 answer 文本给分(缺省 0.5),让测试能控制最终顺序。"""

    def __init__(self, scores_by_text=None, error: Exception | None = None):
        self.scores_by_text = scores_by_text or {}
        self.error = error
        self.calls: list = []

    def rerank(self, query, candidates):
        self.calls.append((query, list(candidates)))
        if self.error is not None:
            raise self.error
        return [self.scores_by_text.get(text, 0.5) for _, text in candidates]


def _retriever(store, embedder=None, reranker=None, *, top_k=3, threshold=0.5,
               session=None, hybrid_top_k=50):
    return KnowledgeRetriever(
        session, store, embedder or _FakeEmbedder(), reranker or _FakeReranker(),
        top_k=top_k, score_threshold=threshold, hybrid_top_k=hybrid_top_k,
    )


# ---- 不碰库的部分 ----


@pytest.mark.anyio
async def test_query_is_embedded_and_hybrid_search_is_called():
    """查询向量 = 问题原文,BM25 腿也喂问题原文。"""
    store, embedder = _FakeStore(hits=[]), _FakeEmbedder()
    r = _retriever(store, embedder, top_k=7)
    assert await r.search("邮费是多少") == []
    assert embedder.calls == [["邮费是多少"]]
    assert store.calls == [([0.1, 0.1, 0.1], "邮费是多少", 50)]


@pytest.mark.anyio
async def test_store_failure_becomes_infrastructure_error():
    store = _FakeStore(error=RuntimeError("milvus 连不上"))
    with pytest.raises(ToolInfrastructureError):
        await _retriever(store).search("邮费")


@pytest.mark.anyio
async def test_embedder_failure_becomes_infrastructure_error():
    embedder = _FakeEmbedder(error=RuntimeError("模型加载失败"))
    with pytest.raises(ToolInfrastructureError):
        await _retriever(_FakeStore(), embedder).search("邮费")


class _DeadSession:
    """MySQL 挂了:回查原文时抛**裸** `SQLAlchemyError`(未翻译的原始形态)。

    `rollback_raises=True` 模拟「回滚时发现连接已死,回滚自己也抛」。
    """

    def __init__(self, rollback_raises: bool = False):
        self.rollback_raises = rollback_raises
        self.rollbacks = 0

    async def execute(self, *args, **kwargs):
        raise OperationalError(
            "SELECT knowledge_chunks ...", {}, Exception("Lost connection to MySQL server")
        )

    async def rollback(self):
        self.rollbacks += 1
        if self.rollback_raises:
            raise OperationalError("ROLLBACK", {}, Exception("Lost connection"))


@pytest.mark.anyio
async def test_db_fault_while_loading_rows_becomes_infrastructure_error():
    """回查原文时 MySQL 挂了,必须走 502 那条路 —— **注入的是裸 `OperationalError`**。

    ⚠️ 注入「已经翻译好的」`ToolInfrastructureError` 等于**跳过被验的那一步**:
    在「`_load_rows` 不翻译、裸 SQLAlchemy 错误原样逃出」的实现下照样绿 ——
    这正是本仓定义的假绿(ch05 spec §8.2 记账,`tests/test_api_refund.py:171-186`
    与 `tests/test_api_ticket.py:151-212` 是同一形态的样板)。

    这个洞**只在这一个文件里成立**:`_hybrid_hits` 覆盖嵌入/Milvus、`_rerank`
    覆盖重排器,唯独回查没翻译 —— 漏一处就隐形。后果不是报错而是**降级**:
    多路检索(`multi_search`)会把「库挂了」当成「这一路没搜到」跳过,最终返回空,
    用户看到「知识库里没有这条」;而单路 `query_faq` 经 executor 的分类表是响亮 502。

    断言**类型**而不只是「抛了」:`app/api/chat.py` 只按 `ToolInfrastructureError`
    给 502,原样抛 `OperationalError` 是 500。
    """
    store = _FakeStore(hits=[("1", 0.9)])
    session = _DeadSession()
    with pytest.raises(ToolInfrastructureError):
        await _retriever(store, session=session).search("邮费")
    assert session.rollbacks == 1


@pytest.mark.anyio
async def test_failing_rollback_does_not_replace_the_original_error():
    """回滚**自己也失败**时,逃出去的仍是**原异常**(翻译后的那个),不是回滚的异常。

    回滚失败抛的是裸 `SQLAlchemyError`;让它顶掉原异常逃出本模块,调用方按
    `ToolInfrastructureError` 认故障就认不出 —— 整条链路又变回「没搜到」。
    """
    session = _DeadSession(rollback_raises=True)
    with pytest.raises(ToolInfrastructureError):
        await _retriever(_FakeStore(hits=[("1", 0.9)]), session=session).search("邮费")
    assert session.rollbacks == 1


@pytest.mark.anyio
async def test_cancellation_survives_a_failing_rollback():
    """取消语义**不能让位**:回滚失败时 `CancelledError` 仍是 `CancelledError`。

    超时靠 `asyncio.wait_for` 取消协程;被取消的任务把异常换成
    `ToolInfrastructureError`(或让回滚异常顶出来),`execute_tool` 的超时分类就错了。
    这一条同时钉住「回滚失败不翻译成 ToolInfrastructureError」这个选择。
    """
    session = _DeadSession(rollback_raises=True)
    store = _FakeStore(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await _retriever(store, session=session).search("邮费")
    assert session.rollbacks == 1


@pytest.mark.anyio
async def test_cancellation_rolls_back_the_session():
    """取消后必须回滚,否则 query_faq 重试撞 PendingRollbackError 变 502。"""

    class _RollbackRecorder:
        def __init__(self):
            self.rollbacks = 0

        async def rollback(self):
            self.rollbacks += 1

    session = _RollbackRecorder()
    store = _FakeStore(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await _retriever(store, session=session).search("邮费")
    assert session.rollbacks == 1


# ---- 真 MySQL ----


async def _make_rows(specs) -> dict:
    """[(key, questions, answer)] → 真实行,返回 {key: id}。"""
    async with get_sessionmaker()() as session:
        rows = [
            KnowledgeChunk(
                category=SCRATCH_CATEGORY, questions=q, answer=a, content_type="faq"
            )
            for _, q, a in specs
        ]
        session.add_all(rows)
        await session.commit()
        return {key: row.id for (key, _, _), row in zip(specs, rows)}


async def _cleanup() -> None:
    async with get_sessionmaker()() as session:
        await session.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.category == SCRATCH_CATEGORY)
        )
        await session.commit()


@pytest.mark.db
@pytest.mark.anyio
async def test_returns_chunks_in_rerank_score_order_with_metadata():
    """最终顺序跟**重排分数**,不跟 hybrid 顺序;chunk_id/section_path 回填。"""
    await _cleanup()
    try:
        ids = await _make_rows(
            [("low", "低分问法", "低分答案"), ("high", "高分问法\n第二种问法", "高分答案")]
        )
        # hybrid 返回 [low, high],重排把 high 提上来
        store = _FakeStore(hits=[(str(ids["low"]), 0.9), (str(ids["high"]), 0.6)])
        reranker = _FakeReranker({"高分答案": 0.95, "低分答案": 0.55})
        async with get_sessionmaker()() as session:
            got = await KnowledgeRetriever(
                session, store, _FakeEmbedder(), reranker,
                top_k=3, score_threshold=0.5,
            ).search("怎么退货")
        assert [c.chunk_id for c in got] == [ids["high"], ids["low"]]
        assert got[0].question == "高分问法\n第二种问法"
        assert got[0].score == 0.95
        assert got[0].section_path is None
        assert got[0].category == SCRATCH_CATEGORY
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.db
@pytest.mark.anyio
async def test_threshold_is_inclusive_and_lower_hits_are_dropped():
    await _cleanup()
    try:
        ids = await _make_rows([("edge", "边界问法", "边界答案"), ("low", "低问法", "低答案")])
        store = _FakeStore(hits=[(str(ids["edge"]), 0.9), (str(ids["low"]), 0.8)])
        reranker = _FakeReranker({"边界答案": 0.5, "低答案": 0.4999})
        async with get_sessionmaker()() as session:
            got = await KnowledgeRetriever(
                session, store, _FakeEmbedder(), reranker,
                top_k=3, score_threshold=0.5,
            ).search("q")
        assert [(c.question, c.answer) for c in got] == [("边界问法", "边界答案")]
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.db
@pytest.mark.anyio
async def test_reranker_failure_becomes_infrastructure_error():
    """重排模型故障必须是 502 那条路(需先回查取文本,故打 db)。"""
    await _cleanup()
    try:
        ids = await _make_rows([("real", "真问法", "真答案")])
        store = _FakeStore(hits=[(str(ids["real"]), 0.9)])
        reranker = _FakeReranker(error=RuntimeError("重排模型失败"))
        async with get_sessionmaker()() as session:
            with pytest.raises(ToolInfrastructureError):
                await KnowledgeRetriever(
                    session, store, _FakeEmbedder(), reranker,
                    top_k=3, score_threshold=0.5,
                ).search("q")
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.db
@pytest.mark.anyio
async def test_hit_without_mysql_row_is_skipped_not_fatal():
    await _cleanup()
    try:
        ids = await _make_rows([("real", "真问法", "真答案")])
        store = _FakeStore(hits=[("999999", 0.99), (str(ids["real"]), 0.8)])
        reranker = _FakeReranker({"真答案": 0.9})
        async with get_sessionmaker()() as session:
            got = await KnowledgeRetriever(
                session, store, _FakeEmbedder(), reranker,
                top_k=3, score_threshold=0.5,
            ).search("q")
        assert [(c.question, c.answer) for c in got] == [("真问法", "真答案")]
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.db
@pytest.mark.anyio
async def test_non_numeric_milvus_id_is_skipped():
    await _cleanup()
    try:
        ids = await _make_rows([("real", "真问法", "真答案")])
        store = _FakeStore(hits=[("not-a-number", 0.99), (str(ids["real"]), 0.8)])
        reranker = _FakeReranker({"真答案": 0.9})
        async with get_sessionmaker()() as session:
            got = await KnowledgeRetriever(
                session, store, _FakeEmbedder(), reranker,
                top_k=3, score_threshold=0.5,
            ).search("q")
        assert [c.question for c in got] == ["真问法"]
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.db
@pytest.mark.anyio
async def test_search_does_not_write_anything():
    """检索是只读路径:一次搜索不该改动任何行的状态。"""
    await _cleanup()
    try:
        ids = await _make_rows([("real", "真问法", "真答案")])
        store = _FakeStore(hits=[(str(ids["real"]), 0.9)])
        reranker = _FakeReranker({"真答案": 0.9})
        async with get_sessionmaker()() as session:
            await KnowledgeRetriever(
                session, store, _FakeEmbedder(), reranker,
                top_k=3, score_threshold=0.5,
            ).search("q")
        async with get_sessionmaker()() as session:
            row = (
                await session.execute(
                    select(KnowledgeChunk).where(KnowledgeChunk.id == ids["real"])
                )
            ).scalars().one()
            assert row.vectorize_status == "pending"
            assert row.vector_id is None
    finally:
        await _cleanup()
        await get_engine().dispose()
