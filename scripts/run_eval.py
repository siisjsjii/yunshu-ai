"""四策略检索评估:纯 dense / 纯 BM25 / 混合(RRF) / 混合+Rerank。

用法:
    .venv/Scripts/python.exe scripts/run_eval.py
    .venv/Scripts/python.exe scripts/run_eval.py --top-k 10

读 `evals/测试集.md`(CSV:300 用例,5 桶),对每个 query 用四种策略各检索一次,
回查 MySQL 算 Recall@K / MRR / 平均置信度,按桶分桶,写 `evals/results/latest.json`
并打印对照表。

口径(闭式):
- 命中:检索结果的 `section_path`/`category`/`questions` 里出现 `expect_section`
  的任一分量(按 + | / 拆分);`应拒答=是` 的用例反向计(召到相关章节反而算错)。
- Recall@K = 应答用例中 top-K 命中期望章节的比例;MRR = 首个命中的倒数排名均值;
  置信度 = top-1 分数均值(各策略分数尺度不同,仅供定阈值,不横向比)。

输出编码经 emit() 以 UTF-8 字节写 stdout(cp936 陷阱)。
"""

import argparse
import asyncio
import csv
import io
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.db.base import get_engine, get_sessionmaker
from app.db.models import KnowledgeChunk
from app.retrieval.embedder import get_embedder
from app.retrieval.milvus import get_vector_store
from app.retrieval.reranker import get_reranker
from sqlalchemy import select

CASES = Path(__file__).resolve().parents[1] / "evals" / "测试集.md"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "evals" / "results"

_K = 10


def emit(line: str = "") -> None:
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        print(line)
        return
    stream.write(line.encode("utf-8") + b"\n")
    stream.flush()


def load_cases() -> list[dict]:
    text = CASES.read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    out = []
    for r in rows[1:]:
        if len(r) < 6:
            continue
        out.append({
            "id": r[0], "bucket": r[1], "query": r[2],
            "expect_section": r[3], "expect_points": r[4],
            "should_refuse": r[5] == "是",
        })
    return out


def _match_sections(haystack: str, expect_section: str) -> bool:
    parts = [p.strip() for p in re.split(r"[+|/]", expect_section) if p.strip()]
    return any(p in haystack for p in parts)


async def load_chunks() -> dict[str, dict]:
    async with get_sessionmaker()() as session:
        rows = (await session.execute(select(KnowledgeChunk))).scalars().all()
    return {
        str(r.id): {
            "section_path": r.section_path or "",
            "category": r.category or "",
            "questions": r.questions or "",
            "answer": r.answer or "",
        }
        for r in rows
    }


def hit(chunks: dict, hit_id: str, expect_section: str) -> bool:
    c = chunks.get(str(hit_id))
    if c is None:
        return False
    haystack = f"{c['section_path']} {c['category']} {c['questions']}"
    return _match_sections(haystack, expect_section)


async def main() -> None:
    parser = argparse.ArgumentParser(description="ch04 四策略检索评估")
    parser.add_argument("--top-k", type=int, default=_K)
    args = parser.parse_args()
    top_k = args.top_k

    settings = get_settings()
    cases = load_cases()
    chunks = await load_chunks()
    store = get_vector_store(settings.milvus_uri, settings.milvus_collection)
    embedder = get_embedder(
        settings.embedding_model_path, settings.embedding_max_length,
        settings.embedding_batch_size)
    reranker_available = Path(settings.reranker_model_path).exists()
    reranker = get_reranker(settings.reranker_model_path, settings.reranker_use_fp16)
    if not reranker_available:
        emit(f"⚠ 重排权重未就绪({settings.reranker_model_path} 不存在),混合+Rerank 策略将跳过")

    # 预嵌入所有 query(省得每策略重复加载)
    queries = [c["query"] for c in cases]
    emit(f"嵌入 {len(queries)} 条 query…")
    qvecs = embedder.encode(queries)

    def strategies():
        yield "纯dense", lambda q, qv: store.search(qv, top_k)
        yield "纯BM25", lambda q, qv: store.bm25_search(q, top_k)
        yield "混合(RRF)", lambda q, qv: store.hybrid_search(qv, q, top_k)

        def hybrid_rerank(q, qv):
            hits = store.hybrid_search(qv, q, max(50, top_k))
            if not hits:
                return []
            texts = [(i, chunks.get(str(i), {}).get("answer", "")) for i, _ in hits]
            scores = reranker.rerank(q, texts)
            ranked = sorted(zip(hits, scores), key=lambda x: -x[1])
            return [(i, s) for (i, _), s in ranked[:top_k]]

        yield "混合+Rerank", hybrid_rerank

    results: dict = {"top_k": top_k, "strategies": {}}
    for name, fn in strategies():
        if name == "混合+Rerank" and not reranker_available:
            emit(f"⚠ {name}:权重未就绪,跳过")
            continue
        started = time.perf_counter()
        # 每桶累计
        bucket_stats: dict[str, dict] = defaultdict(
            lambda: {"n": 0, "hits": 0, "rr": 0.0, "conf_sum": 0.0,
                     "hits5": 0, "hits10": 0})
        for case, qv in zip(cases, qvecs):
            hits = fn(case["query"], qv)
            b = bucket_stats[case["bucket"]]
            b["n"] += 1
            if hits:
                b["conf_sum"] += hits[0][1]
            # 命中排名
            rank = None
            for rank_i, (hid, _) in enumerate(hits, start=1):
                if hit(chunks, hid, case["expect_section"]):
                    rank = rank_i
                    break
            if case["should_refuse"]:
                # 应拒答:召到相关章节反而算错(正确=rank 为 None)
                if rank is None:
                    b["hits"] += 1
                    b["rr"] += 0.0  # 拒答正确不计 MRR
            else:
                if rank is not None:
                    b["hits"] += 1
                    b["rr"] += 1.0 / rank
                    if rank <= 5:
                        b["hits5"] += 1
                    if rank <= 10:
                        b["hits10"] += 1
        # 汇总
        strategy_result = {}
        for bucket, st in sorted(bucket_stats.items()):
            n = st["n"]
            strategy_result[bucket] = {
                "n": n,
                "recall@5": round(st["hits5"] / n, 4) if n else 0,
                "recall@10": round(st["hits10"] / n, 4) if n else 0,
                "mrr": round(st["rr"] / n, 4) if n else 0,
                "conf": round(st["conf_sum"] / n, 4) if n else 0,
                "answer_ok": round(st["hits"] / n, 4) if n else 0,
            }
        results["strategies"][name] = strategy_result
        emit(f"{name} 完成(耗时 {time.perf_counter() - started:.1f}s)")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "latest.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(f"\n结果写入 {out}")

    # 打印对照表
    buckets = sorted({c["bucket"] for c in cases})
    strategies_names = list(results["strategies"])
    emit("\n===== Recall@10 / MRR / 置信度 对照(按桶)=====")
    for bucket in buckets:
        emit(f"\n【{bucket}】")
        for s in strategies_names:
            r = results["strategies"][s].get(bucket, {})
            if not r:
                continue
            emit(f"  {s:12} recall@10={r['recall@10']:.3f} mrr={r['mrr']:.3f} "
                 f"conf={r['conf']:.3f} 应答正确={r['answer_ok']:.3f}")

    await get_engine().dispose()


if __name__ == "__main__":
    asyncio.run(main())
