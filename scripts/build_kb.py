"""离线建库:语料导入 + faq 迁移 + 向量化补齐。

用法:
    .venv/Scripts/python.exe scripts/build_kb.py
    .venv/Scripts/python.exe scripts/build_kb.py --source 退货政策.md

**重跑 = 幂等补齐**:已入库的行按三元组跳过,未向量化的行(pending)重扫
重写。中断了直接再跑一次即可,这正是验收 2。

离线任务,**故障直接崩、不兜底**(spec §6.7):崩了人工重跑,幂等保证安全。
把失败悄悄吞掉只会留下一个「看起来建好了、其实缺一块」的库。
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select, update

from app.config import get_settings
from app.db.base import get_engine, get_sessionmaker
from app.db.models import Faq, KnowledgeChunk
from app.kb.ingest import faq_migration, parse_corpus_dir, parse_corpus_file
from app.kb.writer import vectorize_pending, write_chunks
from app.retrieval.embedder import get_embedder
from app.retrieval.milvus import get_vector_store

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"


def _out(message: str) -> None:
    """钉死输出编码(cp936 控制台):中文与 ⚠ 都不在 GBK 里,print 会直接崩。"""
    sys.stdout.buffer.write((message + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def _corpus_chunks(args, settings) -> tuple[list, bool]:
    """返回 (语料 chunk, 是否同时迁移 faq)。

    `--source` 限定单个文件时**不**迁移 faq:那是一次定点补录,不该顺带
    把 faq 表再走一遍(虽然幂等,但输出会误导人以为动了别的东西)。
    """
    opts = {
        "max_chars": settings.chunk_max_chars,
        "overlap_chars": settings.chunk_overlap_chars,
    }
    if args.source:
        path = Path(args.source)
        if not path.is_absolute():
            path = KNOWLEDGE_DIR / path
        return parse_corpus_file(path, **opts), False
    return parse_corpus_dir(KNOWLEDGE_DIR, **opts), True


async def _load_faq_chunks() -> tuple[list, int]:
    async with get_sessionmaker()() as session:
        rows = list((await session.execute(select(Faq))).scalars().all())
    return faq_migration(rows), len(rows)


async def _status_counts() -> tuple[int, int, int]:
    async with get_sessionmaker()() as session:
        total = (await session.execute(select(func.count(KnowledgeChunk.id)))).scalar_one()
        done = (
            await session.execute(
                select(func.count(KnowledgeChunk.id)).where(
                    KnowledgeChunk.vectorize_status == "done"
                )
            )
        ).scalar_one()
    return total, done, total - done


async def _reindex(session) -> None:
    """全表打回待向量化。

    配合下面 drop 集合同用 —— 两者缺一不可:只 drop 集合的话,MySQL 里
    全是 done 的行,重跑时一条都不会被捡起,结果是**空集合配全 done**。
    """
    result = await session.execute(
        update(KnowledgeChunk).values(vectorize_status="pending", vector_id=None)
    )
    await session.commit()
    return result.rowcount


async def _run(args) -> None:
    settings = get_settings()
    started = time.perf_counter()

    store = get_vector_store(settings.milvus_uri, settings.milvus_collection)
    if args.reindex:
        async with get_sessionmaker()() as session:
            reset = await _reindex(session)
        store.drop_collection()
        _out(f"--reindex:{reset} 行打回 pending,Milvus 集合已删除,开始重建")

    corpus, with_faq = _corpus_chunks(args, settings)
    _out(f"语料切块:{len(corpus)} 块" + (f"(仅 {args.source})" if args.source else ""))

    chunks = list(corpus)
    if with_faq:
        faq_chunks, faq_rows = await _load_faq_chunks()
        chunks += faq_chunks
        _out(f"faq 迁移:{faq_rows} 条 → {len(faq_chunks)} 块")
    _out(f"待写入:{len(chunks)} 块")

    async with get_sessionmaker()() as session:
        new_rows = await write_chunks(session, chunks)
    _out(f"新增入库:{new_rows} 行(其余 {len(chunks) - new_rows} 块已存在,跳过)")

    embedder = get_embedder(
        settings.embedding_model_path,
        settings.embedding_max_length,
        settings.embedding_batch_size,
    )
    embed_started = time.perf_counter()
    async with get_sessionmaker()() as session:
        processed = await vectorize_pending(
            session, store, embedder, batch_size=settings.embedding_batch_size
        )
    _out(f"向量化补齐:{processed} 行(耗时 {time.perf_counter() - embed_started:.1f}s)")

    total, done, pending = await _status_counts()
    milvus_total = store.count()
    _out(f"knowledge_chunks:总 {total} / done {done} / pending {pending}")
    _out(f"Milvus 集合 {settings.milvus_collection}:{milvus_total} 条")
    if pending:
        _out(f"⚠ 仍有 {pending} 行未向量化 —— 再跑一次本脚本补齐")
    elif milvus_total != done:
        _out(
            f"⚠ Milvus 条数({milvus_total})与 done 行数({done})不一致:集合里可能"
            f"残留已删 MySQL 行的陈旧向量,drop 集合后重跑可对齐"
        )
    _out(f"总耗时 {time.perf_counter() - started:.1f}s")

    await get_engine().dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="ch03 离线建库(可重复跑)")
    parser.add_argument(
        "--source",
        help="只导入 knowledge/ 下的某一个 .md(可给相对名或绝对路径);不给则全量",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="重建向量索引:全表打回 pending + 删掉 Milvus 集合,再全量重算",
    )
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
