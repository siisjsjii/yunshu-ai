"""检索评估集。用法:

    .venv/Scripts/python.exe evals/run_retrieval_eval.py
    .venv/Scripts/python.exe evals/run_retrieval_eval.py --threshold 0.55 --repeat 3

需要真实 Milvus + BGE-M3 + MySQL(会加载 2.2GB 权重,首跑约 1 分钟)。

**评分口径是闭式的**:每个用例给出期望在**同一块**里出现的若干**逐字**片段
(取自语料原文),命中判 true。没有"模型答得算不算对"的主观判断 —— 这与
ch01 那个被样本拟合的 expected_solution 口径不同,分数可以直接引用。

干扰项(expect_empty)反过来评:**必须一条都召不回**。dense 单路没有重排
兜底,「不相关也硬凑答案」是本章最大的质量风险,只统计命中率会把它盖住。

输出编码经 emit() 以 UTF-8 字节写 stdout(cp936 陷阱,同 run_tool_selection_eval)。
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.db.base import get_engine, get_sessionmaker
from app.retrieval.embedder import get_embedder
from app.retrieval.milvus import get_vector_store
from app.retrieval.search import KnowledgeRetriever

CASES = Path(__file__).with_name("retrieval_cases.jsonl")


def emit(line: str = "") -> None:
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        print(line)
        return
    stream.write(line.encode("utf-8") + b"\n")
    stream.flush()


def load_cases() -> list[dict]:
    lines = CASES.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _blob(chunk) -> str:
    return f"{chunk.category}\n{chunk.question}\n{chunk.answer}"


def judge(case: dict, chunks) -> tuple[bool, str]:
    """返回 (是否命中, 说明)。闭式判定,不做主观打分。"""
    if case.get("expect_empty"):
        if not chunks:
            return True, "正确落空"
        return False, f"不该召回却回了 {len(chunks)} 条:{chunks[0].answer[:30]}"
    if not chunks:
        return False, "一条都没召回(全被阈值滤掉?)"
    needed = case["expect_contains"]
    for chunk in chunks:
        text = _blob(chunk)
        if all(needle in text for needle in needed):
            return True, "命中"
    # 落空也要说清楚:是没召回、还是召回了但不对题
    return False, f"召回了 {len(chunks)} 条但没有一块同时含 {needed}"


async def _print_distribution(store, embedder, cases, top_k) -> None:
    """打印每条用例的原始相似度(不过滤),用于定阈值。

    关注两个极值:**命中用例的最低 top-1 分**(阈值不能高过它)与
    **干扰项的最高 top-1 分**(阈值必须高过它)。两者之间就是可用区间。
    """
    import statistics as _stats

    vectors = embedder.encode([c["query"] for c in cases])
    positive_tops, negative_tops = [], []
    emit("\n===== 原始相似度分布(未过滤,Top-K 取第 1 名)=====")
    for case, vector in zip(cases, vectors):
        hits = store.search(vector, top_k)
        top = hits[0][1] if hits else float("nan")
        if case.get("expect_empty"):
            negative_tops.append(top)
            mark = "干扰"
        else:
            positive_tops.append(top)
            mark = "正例"
        emit(f"  {mark} {top:6.3f}  {case['query']}")
    emit("")
    emit(f"正例 top-1:最低 {min(positive_tops):.3f} / 中位 {_stats.median(positive_tops):.3f}"
         f" / 最高 {max(positive_tops):.3f}")
    emit(f"干扰 top-1:最高 {max(negative_tops):.3f} / 最低 {min(negative_tops):.3f}")
    emit(f"→ 可用阈值区间:({max(negative_tops):.3f}, {min(positive_tops):.3f}]"
         if max(negative_tops) < min(positive_tops)
         else f"→ 无干净区间:干扰最高 {max(negative_tops):.3f} ≥ 正例最低 {min(positive_tops):.3f},"
              "两者重叠 —— 需要靠 Top-K 或语料层面解决,不是调阈值能解决的")


async def run_once(retriever, cases) -> tuple[int, list[dict]]:
    results = []
    for case in cases:
        chunks = await retriever.search(case["query"])
        ok, why = judge(case, chunks)
        results.append({"case": case, "ok": ok, "why": why, "chunks": chunks})
    return sum(1 for r in results if r["ok"]), results


async def main() -> None:
    parser = argparse.ArgumentParser(description="ch03 检索评估(真实链路)")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖相似度阈值")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖 Top-K")
    parser.add_argument("--repeat", type=int, default=1, help="重复跑几遍看稳定性")
    parser.add_argument(
        "--dist", action="store_true", help="只打印未命中用例的分数分布,便于定阈值"
    )
    args = parser.parse_args()

    settings = get_settings()
    threshold = args.threshold if args.threshold is not None else settings.retrieval_score_threshold
    top_k = args.top_k if args.top_k is not None else settings.retrieval_top_k
    cases = load_cases()
    emit(f"用例 {len(cases)} 条 | 阈值 {threshold} | Top-K {top_k} | 重复 {args.repeat} 遍")

    store = get_vector_store(settings.milvus_uri, settings.milvus_collection)
    embedder = get_embedder(
        settings.embedding_model_path,
        settings.embedding_max_length,
        settings.embedding_batch_size,
    )
    if args.dist:
        await _print_distribution(store, embedder, cases, top_k)
        await get_engine().dispose()
        return

    scores = []
    for run in range(args.repeat):
        async with get_sessionmaker()() as session:
            retriever = KnowledgeRetriever(
                session, store, embedder, top_k=top_k, score_threshold=threshold
            )
            started = time.perf_counter()
            hits, results = await run_once(retriever, cases)
        scores.append(hits)
        emit(f"\n===== 第 {run + 1} 遍:{hits}/{len(cases)} = {hits / len(cases):.1%}"
             f"(耗时 {time.perf_counter() - started:.1f}s)=====")
        if args.dist:
            for r in results:
                case, chunks = r["case"], r["chunks"]
                top = chunks[0].answer[:34].replace("\n", " ") if chunks else "—"
                flag = "✓" if r["ok"] else "✗"
                emit(f"  {flag} {case['query'][:18]:20} {r['why'][:28]:30} top={top}")

    if args.repeat > 1:
        emit(f"\n稳定性:{scores}(每遍一致 = 检索链路确定性 OK)")

    # 失败用例逐条列出 —— 失败信息才是这次评估的产出
    if not args.dist:
        failed = [r for r in results if not r["ok"]]
        if failed:
            emit(f"\n未命中 {len(failed)} 条:")
        for r in failed:
            emit(f"  ✗ {r['case']['query']}({r['case'].get('note', '')})")
            emit(f"      {r['why']}")
            for chunk in r["chunks"]:
                emit(f"      · [{chunk.category[:12]}] {chunk.answer[:44].replace(chr(10), ' ')}")

    await get_engine().dispose()


if __name__ == "__main__":
    asyncio.run(main())
