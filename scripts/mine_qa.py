"""从历史客服对话里挖知识:分批抽 → 暂存 → 整体去重 → 入库。

用法:
    .venv/Scripts/python.exe scripts/mine_qa.py
    .venv/Scripts/python.exe scripts/mine_qa.py --batch-size 10 --dry-run

独立脚本 + 外部调度(用户裁决):可手动跑,也可挂计划任务。

编排逻辑在 `app/kb/mining.py:mine_knowledge` —— 本脚本与 web 后台任务
(ch04 管理台)共用同一份,这里只是 CLI 壳:读配置、造模型/向量库/嵌入、
接一个写 stdout 的 progress、打印结果。
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.db.base import get_sessionmaker
from app.kb.mining import mine_knowledge
from app.llm import create_extract_model
from app.retrieval.embedder import get_embedder
from app.retrieval.milvus import get_vector_store


def _out(message: str) -> None:
    sys.stdout.buffer.write((message + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


async def _run(args) -> None:
    settings = get_settings()

    async def progress(message: str) -> None:
        _out(message)

    result = await mine_knowledge(
        session_factory=get_sessionmaker(),
        batch_size=args.batch_size,
        dedupe_threshold=settings.dedupe_threshold,
        model=create_extract_model(settings),
        store=get_vector_store(settings.milvus_uri, settings.milvus_collection),
        embedder=get_embedder(
            settings.embedding_model_path,
            settings.embedding_max_length,
            settings.embedding_batch_size,
        ),
        batch_no=args.batch_no,
        dry_run=args.dry_run,
        progress=progress,
    )

    _out(f"抽取 {result['extracted']} 条,字面去重后保留 {result['kept']} 条,"
         f"丢弃 {result['discarded']} 条(向量近重复 {result['near_dup_dropped']} 条)")
    for item in result["kept_qa"][:10]:
        _out(f"  ✓ {item['question'][:40]} [{item['category'][:12]}]")
    if len(result["kept_qa"]) > 10:
        _out(f"  ...(其余 {len(result['kept_qa']) - 10} 条略)")
    if args.dry_run:
        _out("--dry-run:不写库(既不入 knowledge_chunks,也不写 staging)。")
    else:
        _out(f"入库 {result['inserted']} 条(重复的已跳过)")
        if result["inserted"]:
            _out("⚠ 新增行是待向量化状态,请跑 scripts/build_kb.py 补齐向量后再验收")


def main() -> None:
    parser = argparse.ArgumentParser(description="ch03 对话挖知识(可重复跑)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="每批喂几个会话(默认取配置 MINE_BATCH_CONVERSATIONS)",
    )
    parser.add_argument("--batch-no", default=None, help="指定批次号(默认按时间生成)")
    parser.add_argument("--dry-run", action="store_true", help="只抽与去重,不写库")
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = get_settings().mine_batch_conversations
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
