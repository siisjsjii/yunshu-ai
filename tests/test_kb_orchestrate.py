"""后台任务编排(db):专用线程 + 独立 engine 的胶水,真 MySQL + fake 向量库/嵌入。

不加载真模型 —— store/embedder 全注入。验证:任务在后台线程跑完、
JobStore 状态流转、行被向量化、忙时拒绝。
"""

import asyncio

import pytest
from sqlalchemy import delete, select

from app.config import get_settings
from app.db.base import get_engine, get_sessionmaker
from app.db.models import KnowledgeChunk
from app.kb.chunker import Chunk
from app.kb.jobs import JobStore
from app.kb.orchestrate import start_job
from app.kb.writer import write_chunks

pytestmark = pytest.mark.db

SCRATCH_CATEGORY = "ch04-probe-orch"


class _FakeEmbedder:
    def encode(self, texts):
        return [[float(len(t))] * 3 for t in texts]


class _FakeStore:
    def __init__(self):
        self.upserted: list = []

    def ensure_collection(self):
        pass

    def upsert(self, ids, vectors):
        self.upserted.extend(ids)

    def flush(self):
        pass

    def count(self):
        return len(set(self.upserted))


async def _cleanup():
    async with get_sessionmaker()() as s:
        await s.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.category == SCRATCH_CATEGORY)
        )
        await s.commit()


async def _wait(job_store, job_id, timeout=10.0):
    for _ in range(int(timeout / 0.05)):
        job = job_store.get(job_id)
        if job.status != "running":
            return job
        await asyncio.sleep(0.05)
    return job_store.get(job_id)


@pytest.mark.anyio
async def test_vectorize_job_runs_in_background_and_completes():
    await _cleanup()
    try:
        async with get_sessionmaker()() as s:
            await write_chunks(s, [Chunk(SCRATCH_CATEGORY, "问", "答", "路径", "policy")])

        store = _FakeStore()
        js = JobStore()
        job = start_job(js, "vectorize", get_settings(), store=store, embedder=_FakeEmbedder())
        assert job is not None and job.status == "running"

        done = await _wait(js, job.id)
        assert done.status == "done", done.message
        # `processed` 是**全表** pending 数:库里若残留其它 pending 行(上一轮
        # 验收挖矿留下的),会 > 1。只断言「至少处理了本测试插入的那行」——
        # 精确计数会把这个测试绑死在全局状态上(实测 flaky)。
        assert done.result["processed"] >= 1

        async with get_sessionmaker()() as s:
            row = (await s.execute(
                select(KnowledgeChunk).where(KnowledgeChunk.category == SCRATCH_CATEGORY)
            )).scalars().one()
            assert row.vectorize_status == "done"
            assert row.vector_id == str(row.id)
        assert js.is_busy() is False
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.anyio
async def test_busy_store_returns_none():
    """忙时 start_job 返回 None(与 JobStore.start 语义一致)。"""
    await _cleanup()
    try:
        async with get_sessionmaker()() as s:
            await write_chunks(s, [Chunk(SCRATCH_CATEGORY, "问", "答", "路径", "policy")])
        js = JobStore()
        # 直接占住运行槽
        assert js.start("vectorize") is not None
        assert start_job(js, "vectorize", get_settings(), store=_FakeStore(),
                         embedder=_FakeEmbedder()) is None
    finally:
        await _cleanup()
        await get_engine().dispose()
