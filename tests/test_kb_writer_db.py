"""writer 的 db 集成测试:真 MySQL + 假 store/embedder(需要 MySQL 在跑)。

覆盖验收 2 的结构基础:重复导入不翻倍、中断后重跑补齐且不重复写。
Milvus 一律用替身 —— 单测不连向量库;真实双写由 T8 首跑与验收 6 覆盖。

**为什么有 _saved_foreign_pending 这套东西**:`vectorize_pending` 按设计全表
重扫 pending(重跑就是靠它补齐),而这里的 store 是替身 —— 库里若存在真实
pending 行(建库半途中断留下的),它们会被替身「写」成 done 却没有真向量,
等于用跑测试污染了真实建库状态。用例前后把它们原样还原。
"""

import pytest
from sqlalchemy import delete, select, update

from app.db.base import get_engine, get_sessionmaker
from app.db.models import KnowledgeChunk
from app.kb.chunker import Chunk
from app.kb.writer import vectorize_pending, write_chunks

pytestmark = pytest.mark.db

SCRATCH_CATEGORY = "ch03-probe-writer"
SECTION = "退货政策 > 运费说明"


class _FakeEmbedder:
    def __init__(self):
        self.calls: list = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t))] * 3 for t in texts]


class _FakeStore:
    """记录 upsert 边界的调用;`fail_on_id` 模拟「写这一行时进程被打断」。

    按**行 id** 而不是调用次数触发:库里可能还有别的 pending 行排在前面,
    按次数计会打偏(计数器放在 upsert 边界 —— ch02 教训,不放函数体里)。
    """

    def __init__(self, fail_on_id: str | None = None):
        self.calls: list = []
        self.fail_on_id = fail_on_id

    def ensure_collection(self):
        pass

    def upsert(self, ids, vectors):
        if self.fail_on_id is not None and self.fail_on_id in ids:
            # 抛在记录之前:这次写没落地,重跑必须重写这一批。
            # 只炸一次 —— 中断是**一次性事故**,重跑时那条 Milvus 已经好了,
            # 否则第二次跑还会在同一行炸,「重跑补齐」永远走不到。
            self.fail_on_id = None
            raise RuntimeError("模拟中断")
        self.calls.append(tuple(ids))

    def upserted_ids(self) -> list[str]:
        return [i for call in self.calls for i in call]


def _chunk(n: int, *, section_path: str = SECTION) -> Chunk:
    return Chunk(
        category=SCRATCH_CATEGORY,
        questions=f"运费怎么算{n}",
        answer=f"第 {n} 条运费说明正文。",
        section_path=section_path,
        content_type="policy",
        is_key_clause=False,
    )


async def _scratch_rows(session) -> list[KnowledgeChunk]:
    return list(
        (
            await session.execute(
                select(KnowledgeChunk)
                .where(KnowledgeChunk.category == SCRATCH_CATEGORY)
                .order_by(KnowledgeChunk.id)
            )
        )
        .scalars()
        .all()
    )


async def _cleanup() -> None:
    async with get_sessionmaker()() as session:
        await session.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.category == SCRATCH_CATEGORY)
        )
        await session.commit()


async def _saved_foreign_pending() -> list[tuple[int, str | None]]:
    """记下非本测试的 pending 行,供用例结束后还原。"""
    async with get_sessionmaker()() as session:
        rows = (
            await session.execute(
                select(KnowledgeChunk.id, KnowledgeChunk.vector_id).where(
                    KnowledgeChunk.vectorize_status == "pending",
                    KnowledgeChunk.category != SCRATCH_CATEGORY,
                )
            )
        ).all()
    return [(r[0], r[1]) for r in rows]


async def _restore_foreign_pending(saved: list[tuple[int, str | None]]) -> None:
    if not saved:
        return
    async with get_sessionmaker()() as session:
        for row_id, vector_id in saved:
            await session.execute(
                update(KnowledgeChunk)
                .where(KnowledgeChunk.id == row_id)
                .values(vectorize_status="pending", vector_id=vector_id)
            )
        await session.commit()


@pytest.mark.anyio
async def test_write_chunks_twice_does_not_duplicate():
    """同一批语料导入两遍,行数不变 —— 三元组查重的幂等。"""
    await _cleanup()
    try:
        async with get_sessionmaker()() as session:
            assert await write_chunks(session, [_chunk(1), _chunk(2), _chunk(3)]) == 3
        async with get_sessionmaker()() as session:
            assert await write_chunks(session, [_chunk(1), _chunk(2), _chunk(3)]) == 0
        async with get_sessionmaker()() as session:
            rows = await _scratch_rows(session)
            assert [r.questions for r in rows] == ["运费怎么算1", "运费怎么算2", "运费怎么算3"]
            assert {r.vectorize_status for r in rows} == {"pending"}
            assert {r.content_type for r in rows} == {"policy"}
            assert {r.section_path for r in rows} == {SECTION}
            assert {r.is_key_clause for r in rows} == {False}
            assert {r.vector_id for r in rows} == {None}
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.anyio
async def test_write_chunks_links_neighbours_only_within_same_section():
    """同 section 的相邻新块双向串联;跨 section 不串(否则溯源会串到别的章节)。"""
    await _cleanup()
    other = "退货政策 > 退款时限"
    try:
        async with get_sessionmaker()() as session:
            await write_chunks(
                session, [_chunk(1), _chunk(2), _chunk(3, section_path=other)]
            )
        async with get_sessionmaker()() as session:
            first, second, third = await _scratch_rows(session)
            assert first.next_chunk_id == second.id
            assert second.prev_chunk_id == first.id
            # 第三条换了 section,不与第二条相连
            assert second.next_chunk_id is None
            assert third.prev_chunk_id is None and third.next_chunk_id is None
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.anyio
async def test_vectorize_pending_marks_done_and_backfills_vector_id():
    await _cleanup()
    saved = await _saved_foreign_pending()
    store, embedder = _FakeStore(), _FakeEmbedder()
    try:
        async with get_sessionmaker()() as session:
            await write_chunks(session, [_chunk(1), _chunk(2)])
        async with get_sessionmaker()() as session:
            await vectorize_pending(session, store, embedder)
        async with get_sessionmaker()() as session:
            rows = await _scratch_rows(session)
            assert {r.vectorize_status for r in rows} == {"done"}
            assert {r.vector_id for r in rows} == {str(r.id) for r in rows}
            # 喂给嵌入的是拼好的三字段文本,不是零散字段
            fed = [t for call in embedder.calls for t in call]
            assert any("运费怎么算1" in t and SCRATCH_CATEGORY in t for t in fed)
    finally:
        await _restore_foreign_pending(saved)
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.anyio
async def test_interrupted_then_rerun_completes_without_duplicate_upserts():
    """验收 2 的结构基础:写第 3 行时被打断 → 重跑 → pending 清零,且每个 pk 恰写一次。

    重跑用**新 session**:同 session 重读会命中身份映射里的旧对象,
    断言就变成「靠 refcount 走运」而不是真读库。
    """
    await _cleanup()
    saved = await _saved_foreign_pending()
    embedder = _FakeEmbedder()
    try:
        async with get_sessionmaker()() as session:
            await write_chunks(session, [_chunk(i) for i in (1, 2, 3, 4)])
        async with get_sessionmaker()() as session:
            ids = [r.id for r in await _scratch_rows(session)]
        store = _FakeStore(fail_on_id=str(ids[2]))

        # 第一跑:处理到第 3 行时中断
        with pytest.raises(RuntimeError):
            async with get_sessionmaker()() as session:
                await vectorize_pending(session, store, embedder, batch_size=1)

        async with get_sessionmaker()() as session:
            rows = await _scratch_rows(session)
            done = [r.id for r in rows if r.vectorize_status == "done"]
            pending = [r.id for r in rows if r.vectorize_status == "pending"]
            assert done == ids[:2], f"中断前应完成前 2 行,实际 {done}"
            assert pending == ids[2:], f"中断后应留后 2 行 pending,实际 {pending}"

        # 第二跑:补齐剩下两行
        async with get_sessionmaker()() as session:
            await vectorize_pending(session, store, embedder, batch_size=1)

        async with get_sessionmaker()() as session:
            rows = await _scratch_rows(session)
            assert {r.vectorize_status for r in rows} == {"done"}
            assert {r.vector_id for r in rows} == {str(i) for i in ids}

        # 计数器在 upsert 边界:两次运行合计,本测试的每个 pk 恰好写一次
        written = [i for i in store.upserted_ids() if int(i) in set(ids)]
        assert sorted(written) == sorted(str(i) for i in ids), (
            f"每个 pk 应恰好写一次,实际写入序列 {written}"
        )
    finally:
        await _restore_foreign_pending(saved)
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.anyio
async def test_vectorize_pending_is_noop_when_nothing_pending():
    """没有 pending 行时:一次查询就收敛,不空转、不打扰 store/embedder。"""
    await _cleanup()
    saved = await _saved_foreign_pending()
    if saved:
        pytest.skip(f"库里有 {len(saved)} 行真实 pending(建库未完成),本用例的"
                    "「全库无 pending」前提不成立;跑完 build_kb 再跑")
    store, embedder = _FakeStore(), _FakeEmbedder()
    try:
        async with get_sessionmaker()() as session:
            assert await vectorize_pending(session, store, embedder) == 0
        assert store.calls == [] and embedder.calls == []
    finally:
        await _cleanup()
        await get_engine().dispose()
