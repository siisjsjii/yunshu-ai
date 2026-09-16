"""从历史客服对话里挖知识:分批抽 → 暂存 → 整体去重 → 入库。

用法:
    .venv/Scripts/python.exe scripts/mine_qa.py
    .venv/Scripts/python.exe scripts/mine_qa.py --batch-size 10 --dry-run

独立脚本 + 外部调度(用户裁决):可手动跑,也可挂计划任务,本章演示手动。

**只负责挖掘与入库,不负责向量化** —— 新增的 kept 行是 pending 状态,
跑完提示执行 `scripts/build_kb.py` 补齐向量(复用同一条补齐路径,
不为挖矿另开一条写入通道)。
"""

import argparse
import asyncio
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.db.base import get_engine, get_sessionmaker
from app.kb.mining import (
    MineParseError,
    batch_conversations,
    dedupe_pairs,
    existing_fingerprints,
    finalize_staging,
    find_near_duplicate,
    keep_pairs,
    load_turns,
    mine_batch,
    question_fingerprint,
    render_conversations,
    staging_fingerprints,
    write_staging,
)
from app.llm import create_extract_model
from app.retrieval.embedder import get_embedder
from app.retrieval.milvus import get_vector_store


def _out(message: str) -> None:
    sys.stdout.buffer.write((message + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


async def _run(args) -> None:
    settings = get_settings()
    started = time.perf_counter()
    batch_no = args.batch_no or f"mine-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    _out(f"批次号:{batch_no}")

    async with get_sessionmaker()() as session:
        turns = await load_turns(session)
    if not turns:
        _out("没有可用于挖掘的对话(需要至少一问一答)。")
        return
    conversations = len({t.conversation_id for t in turns})
    _out(f"读到 {len(turns)} 轮问答,来自 {conversations} 个会话")

    # ---- 去重基准必须在抽取**之前**取快照 ----
    #
    # 本轮抽出的条目马上就会被写进 staging(为了按 batch_no 留痕),而
    # staging 又是去重的参照之一 —— 在循环之后再快照,本轮的产物会把自己
    # 全部判成"已存在",结果是**一条都不保留**,而且看起来像是"这轮确实没
    # 抽出新东西"(实测踩过:44 条全部丢弃)。
    async with get_sessionmaker()() as session:
        seen = await existing_fingerprints(session)
        seen |= await staging_fingerprints(session)
    _out(f"去重基准:已入库 {len(seen)} 条问法指纹")

    # 抽取用 temperature=0 的模型:结构化输出要稳,与 services/extract.py 同路。
    model = create_extract_model(settings)
    all_pairs = []
    failed_batches = 0
    batch_index = 0
    async with get_sessionmaker()() as session:
        for group in batch_conversations(turns, args.batch_size):
            batch_index += 1
            text = render_conversations(group)
            try:
                pairs = await mine_batch(model=model, conversation_text=text)
            except MineParseError as exc:
                # 一批没抽好不该拖垮整跑 —— 记数继续。上游故障(401/超时)
                # 不是 MineParseError,会照常把脚本崩掉,那才是要人介入的。
                failed_batches += 1
                _out(f"  第 {batch_index} 批解析失败,跳过:{exc}")
                continue
            all_pairs.extend(pairs)
            if not args.dry_run:
                await write_staging(
                    session,
                    batch_no=batch_no,
                    source_ref=group[0].conversation_id if group else None,
                    pairs=pairs,
                )
            _out(f"  第 {batch_index} 批:{len(group)} 轮 → {len(pairs)} 条问答对")

    _out(f"抽取完成:共 {len(all_pairs)} 条,失败 {failed_batches} 批")
    if failed_batches:
        _out("⚠ 有批次解析失败 —— 这些会话本轮没有产出知识,可重跑(已入库的会去重跳过)")

    # ---- 整体去重:先字面指纹(staging 内 + 已入库),再向量近重复 ----
    kept, dropped = dedupe_pairs(all_pairs, seen)
    _out(f"字面去重后:{len(kept)} 条保留,{len(dropped)} 条丢弃")

    store = get_vector_store(settings.milvus_uri, settings.milvus_collection)
    embedder = get_embedder(
        settings.embedding_model_path,
        settings.embedding_max_length,
        settings.embedding_batch_size,
    )
    near_duplicates = []
    for pair in kept:
        if await find_near_duplicate(
            store=store,
            embedder=embedder,
            question=pair.question,
            threshold=settings.dedupe_threshold,
        ):
            near_duplicates.append(pair)
    if near_duplicates:
        # 按指纹剔除,不用 id()/身份比较 —— 那依赖对象身份,同一批里
        # 两个内容相同的 QaPair 会被当成不同对象漏掉。
        near_prints = {question_fingerprint(p.question) for p in near_duplicates}
        kept = [p for p in kept if question_fingerprint(p.question) not in near_prints]
        dropped = dropped + near_duplicates
        _out(f"向量近重复再丢 {len(near_duplicates)} 条(阈值 {settings.dedupe_threshold})")

    _out(f"最终保留 {len(kept)} 条,丢弃 {len(dropped)} 条")
    for pair in kept[:10]:
        _out(f"  ✓ {pair.question[:40]} [{pair.category[:12]}]")
    if len(kept) > 10:
        _out(f"  ...(其余 {len(kept) - 10} 条略)")

    if args.dry_run:
        _out("--dry-run:不写库(既不入 knowledge_chunks,也不写 staging)。")
        await get_engine().dispose()
        return

    async with get_sessionmaker()() as session:
        inserted = await keep_pairs(session, kept)
        marked_kept, marked_discarded = await finalize_staging(
            session, batch_no=batch_no, kept=kept
        )
    _out(f"入库 {inserted} 条(重复的已跳过)")
    _out(f"staging 落痕:kept {marked_kept} / discarded {marked_discarded}")
    if inserted:
        _out("⚠ 新增行是待向量化状态,请跑 scripts/build_kb.py 补齐向量后再验收")
    _out(f"staging 批次号 {batch_no}(保留备查,不自动清空)")
    _out(f"总耗时 {time.perf_counter() - started:.1f}s")

    await get_engine().dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="ch03 对话挖知识(可重复跑)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="每批喂几个会话(默认取配置 MINE_BATCH_CONVERSATIONS)",
    )
    parser.add_argument("--batch-no", default=None, help="指定批次号(默认按时间生成)")
    parser.add_argument(
        "--dry-run", action="store_true", help="只抽与去重,不写库"
    )
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = get_settings().mine_batch_conversations
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
