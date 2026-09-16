"""KnowledgeRetriever:嵌入 → Milvus Top-K → 阈值过滤 → MySQL 回查 → 字段组装。

过滤与错误翻译用手替身(不碰库,连 MySQL 都不需要);回查、排序、
「Milvus 有而 MySQL 没有」这几件必须真读库,打 @pytest.mark.db。
"""

import pytest
from sqlalchemy import delete, select

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

    def search(self, vector, top_k):
        self.calls.append((list(vector), top_k))
        if self.error is not None:
            raise self.error
        return list(self.hits)


def _retriever(store, embedder=None, *, top_k=3, threshold=0.5, session=None):
    return KnowledgeRetriever(
        session, store, embedder or _FakeEmbedder(), top_k=top_k, score_threshold=threshold
    )


# ---- 不碰库的部分 ----


@pytest.mark.anyio
async def test_query_is_embedded_as_is_and_top_k_goes_to_store():
    """查询向量 = 问题原文,**不**拼 category/questions/answer。

    拼三字段是**入库**时对 chunk 做的事(seed 侧);查询侧再拼一次的话,
    两边就不是同一个空间里的同一件事了。这条钉住这个不对称是刻意的。
    """
    store, embedder = _FakeStore(hits=[]), _FakeEmbedder()
    r = _retriever(store, embedder, top_k=7)
    assert await r.search("邮费是多少") == []
    assert embedder.calls == [["邮费是多少"]]
    assert store.calls == [([0.1, 0.1, 0.1], 7)]


@pytest.mark.anyio
async def test_all_hits_below_threshold_returns_empty_without_touching_db():
    """全被阈值滤掉时**根本不查库**(session=None 也不炸)。

    dense 单路没有重排兜底,阈值是「不相关也硬凑答案」的唯一闸门;
    滤空必须走「未收录」,而不是返回一堆低分块让模型照着编。
    """
    store = _FakeStore(hits=[("43", 0.49), ("44", 0.10)])
    r = _retriever(store, threshold=0.5)
    assert await r.search("邮费") == []


@pytest.mark.anyio
async def test_store_failure_becomes_infrastructure_error():
    """Milvus 故障必须是 502 那条路,不能伪装成「这一条没收录」。"""
    store = _FakeStore(error=RuntimeError("milvus 连不上"))
    with pytest.raises(ToolInfrastructureError):
        await _retriever(store).search("邮费")


@pytest.mark.anyio
async def test_embedder_failure_becomes_infrastructure_error():
    embedder = _FakeEmbedder(error=RuntimeError("模型加载失败"))
    with pytest.raises(ToolInfrastructureError):
        await _retriever(_FakeStore(), embedder).search("邮费")


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
async def test_returns_chunks_in_milvus_score_order_with_full_question_text():
    """顺序跟 Milvus 的相似度,**不**跟 MySQL 的 id;question 是富文本全文。

    id 顺序刻意与分数顺序相反 —— 若实现按查库返回的顺序拼结果,这条会红。
    """
    await _cleanup()
    try:
        ids = await _make_rows(
            [
                ("low", "低分问法", "低分答案"),
                ("high", "高分问法\n第二种问法", "高分答案"),
            ]
        )
        store = _FakeStore(hits=[(str(ids["high"]), 0.93), (str(ids["low"]), 0.61)])
        async with get_sessionmaker()() as session:
            got = await KnowledgeRetriever(
                session, store, _FakeEmbedder(), top_k=3, score_threshold=0.5
            ).search("怎么退货")
        assert got == [
            RetrievedChunk("高分问法\n第二种问法", "高分答案", SCRATCH_CATEGORY),
            RetrievedChunk("低分问法", "低分答案", SCRATCH_CATEGORY),
        ]
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.db
@pytest.mark.anyio
async def test_threshold_is_inclusive_and_lower_hits_are_dropped():
    """恰好等于阈值算命中;低于阈值的那条不出现。"""
    await _cleanup()
    try:
        ids = await _make_rows([("edge", "边界问法", "边界答案"), ("low", "低问法", "低答案")])
        store = _FakeStore(hits=[(str(ids["edge"]), 0.5), (str(ids["low"]), 0.4999)])
        async with get_sessionmaker()() as session:
            got = await KnowledgeRetriever(
                session, store, _FakeEmbedder(), top_k=3, score_threshold=0.5
            ).search("q")
        assert [(c.question, c.answer) for c in got] == [("边界问法", "边界答案")]
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.db
@pytest.mark.anyio
async def test_hit_without_mysql_row_is_skipped_not_fatal():
    """Milvus 里残留的陈旧向量(对应行已从 MySQL 删掉)必须跳过,不能炸。

    这正是「Milvus 只是索引、MySQL 才是权威源」的代价:两边可能短暂不一致。
    """
    await _cleanup()
    try:
        ids = await _make_rows([("real", "真问法", "真答案")])
        store = _FakeStore(hits=[("999999", 0.99), (str(ids["real"]), 0.8)])
        async with get_sessionmaker()() as session:
            got = await KnowledgeRetriever(
                session, store, _FakeEmbedder(), top_k=3, score_threshold=0.5
            ).search("q")
        assert [(c.question, c.answer) for c in got] == [("真问法", "真答案")]
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.db
@pytest.mark.anyio
async def test_non_numeric_milvus_id_is_skipped():
    """pk 是 str(MySQL id),理论上不会是别的东西;真来了也不能变成 500。"""
    await _cleanup()
    try:
        ids = await _make_rows([("real", "真问法", "真答案")])
        store = _FakeStore(hits=[("not-a-number", 0.99), (str(ids["real"]), 0.8)])
        async with get_sessionmaker()() as session:
            got = await KnowledgeRetriever(
                session, store, _FakeEmbedder(), top_k=3, score_threshold=0.5
            ).search("q")
        assert [c.question for c in got] == ["真问法"]
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.db
@pytest.mark.anyio
async def test_search_does_not_write_anything():
    """检索是只读路径:一次搜索不该改动任何行的状态(尤其是 vectorize_status)。"""
    await _cleanup()
    try:
        ids = await _make_rows([("real", "真问法", "真答案")])
        store = _FakeStore(hits=[(str(ids["real"]), 0.9)])
        async with get_sessionmaker()() as session:
            await KnowledgeRetriever(
                session, store, _FakeEmbedder(), top_k=3, score_threshold=0.5
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
