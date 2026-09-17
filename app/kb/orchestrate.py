"""后台任务编排(ch04):把向量化 / 挖知识跑在专用线程 + 独立 engine 上。

不阻塞事件循环:向量化/挖知识会**同步**加载/使用 2.2GB BGE-M3 与 Milvus,
若跑在事件循环里会冻结聊天接口。做法:

- 每个任务一个 `threading.Thread(daemon=True)`,线程内 `asyncio.run`。
- 线程**自建** `create_async_engine`,不用 `get_engine()` 的 lru_cache 单例
  (那个绑在首次使用它的主事件循环上,跨线程复用会出异步连接问题)。
- 任务进度通过 `progress` 回调写进 `JobStore`,前端轮询读回。

`start_job` 的 `store`/`embedder`/`model` 缺省时用真实单例,测试注入 fake。
"""

import asyncio
import threading

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.models import KnowledgeChunk
from app.kb.jobs import Job, JobStore
from app.kb.mining import mine_knowledge
from app.kb.writer import vectorize_pending
from app.llm import create_extract_model
from app.retrieval.embedder import get_embedder
from app.retrieval.milvus import get_vector_store
from app.sanitize import redact_api_key


def _spawn(job_store: JobStore, job: Job, coro_fn) -> None:
    def target() -> None:
        try:
            asyncio.run(coro_fn())
        except Exception as exc:
            # 正常路径里 coro 自己会置 done/failed;这里兜住线程级意外,
            # 避免 job 永远停在 running。
            job_store.update(job.id, status="failed",
                             message=redact_api_key(str(exc), get_settings().openai_api_key))
    threading.Thread(target=target, name=f"kb-job-{job.id}", daemon=True).start()


def _fresh_factory(settings):
    """任务专属 engine + sessionmaker(独立于主循环的 lru_cache 单例)。"""
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _count(session, status: str | None) -> int:
    stmt = select(func.count(KnowledgeChunk.id))
    if status is not None:
        stmt = stmt.where(KnowledgeChunk.vectorize_status == status)
    return (await session.execute(stmt)).scalar_one()


async def _run_vectorize(job_store, job_id, settings, store, embedder) -> None:
    engine, factory = _fresh_factory(settings)
    try:
        async def progress(msg): job_store.update(job_id, message=msg)

        async with factory() as session:
            pending = await _count(session, "pending")
        store.ensure_collection()
        if pending:
            await progress(f"待向量化 {pending} 行")
            async with factory() as session:
                processed = await vectorize_pending(
                    session, store, embedder,
                    batch_size=settings.embedding_batch_size, progress=progress)
        else:
            processed = 0
        async with factory() as session:
            done = await _count(session, "done")
        job_store.update(
            job_id, status="done", message=f"完成,处理 {processed} 行",
            result={"processed": processed, "total_pending_before": pending,
                    "milvus_count": store.count(), "done_rows": done})
    finally:
        await engine.dispose()


async def _run_mine(job_store, job_id, settings, store, embedder, model) -> None:
    engine, factory = _fresh_factory(settings)
    try:
        async def progress(msg): job_store.update(job_id, message=msg)

        result = await mine_knowledge(
            session_factory=factory, batch_size=settings.mine_batch_conversations,
            dedupe_threshold=settings.dedupe_threshold, model=model,
            store=store, embedder=embedder, dry_run=False, progress=progress)
        job_store.update(
            job_id, status="done",
            message=f"抽出 {result['extracted']} 条,保留 {result['kept']} 条",
            result=result)
    finally:
        await engine.dispose()


def start_job(job_store: JobStore, job_type: str, settings, *,
              store=None, embedder=None, model=None) -> Job | None:
    """起一个后台任务。忙(已有 running)返回 None;否则返回 running 的 Job。"""
    job = job_store.start(job_type)
    if job is None:
        return None

    store = store or get_vector_store(settings.milvus_uri, settings.milvus_collection)
    embedder = embedder or get_embedder(
        settings.embedding_model_path, settings.embedding_max_length,
        settings.embedding_batch_size)

    if job_type == "vectorize":
        _spawn(job_store, job, lambda: _run_vectorize(job_store, job.id, settings, store, embedder))
    elif job_type == "mine":
        model = model or create_extract_model(settings)
        _spawn(job_store, job, lambda: _run_mine(job_store, job.id, settings, store, embedder, model))
    else:
        job_store.update(job.id, status="failed", message=f"未知任务类型 {job_type}")
    return job
