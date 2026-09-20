"""检索评估集。用法:

    .venv/Scripts/python.exe evals/run_retrieval_eval.py
    .venv/Scripts/python.exe evals/run_retrieval_eval.py --threshold 0.55 --repeat 3

需要真实 Milvus + BGE-M3 + MySQL(会加载 2.2GB 权重,首跑约 1 分钟)。

**评分口径是闭式的**:每个用例给出期望在**同一块**里出现的若干**逐字**片段
(取自语料原文),命中判 true。没有"模型答得算不算对"的主观判断 —— 这与
ch01 那个被样本拟合的 expected_solution 口径不同,分数可以直接引用。

干扰项(expect_empty)反过来评:**必须一条都召不回**。「不相关也硬凑答案」
是检索最大的质量风险,只统计命中率会把它盖住,所以干扰项必须单独评。

`--dist` 走的是**在线同一条链路**(混合检索 + 重排),打印的是重排器输出的
原始 sigmoid 分数 —— 阈值卡的就是这个尺度。别再用 dense 单路的余弦分定阈值:
两者分布不可通约,0.58 就是这么被沿用两章的(见 `app/config.py` 该字段旁注释)。

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
from app.retrieval.reranker import get_reranker
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


async def _print_distribution(retriever, cases) -> None:
    """打印每条用例的**原始重排分数**(阈值 0,不过滤),用于定阈值。

    走**当前链路**(混合检索 + 重排),与在线检索同一条路。早期版本这里直接
    调 `store.search(vector, top_k)`(**dense 单路**),ch04 换了链路之后它打的
    是 dense 余弦分、而阈值卡的是重排 sigmoid 分 —— 两者不可通约,照着它定阈值
    必然定错。**这正是 0.58 从未被复核的机制原因**,别再改回去。

    关注两个极值:**能命中用例的最低 top-1 分**(阈值不能高过它)与
    **干扰项的最高 top-1 分**(阈值必须高过它)。两者之间就是可用区间。

    「能命中」沿用 `judge` 的同一套闭式口径(某一召回块里含齐期望片段),
    与阈值无关:阈值 0 时仍召不回期望块的用例是**检索质量问题**,调阈值救不回来,
    不能把它们的最低分当成区间上界(那样会得出"无干净区间"的错误结论)。
    """
    positive_tops: list[float] = []
    negative_tops: list[float] = []
    hitting_tops: list[float] = []  # 能命中的正例的 top-1 分 → 阈值**上界**
    positive_total = missed = empty_cases = 0

    emit("\n===== 原始重排分数分布(阈值 0,当前混合+重排链路,取第 1 名)=====")
    for case in cases:
        chunks = await retriever.search(case["query"])
        top = chunks[0].score if chunks else None
        negative = bool(case.get("expect_empty"))
        hit, _ = judge(case, chunks)
        if negative:
            if top is not None:
                negative_tops.append(top)
            mark = "干扰"
        else:
            positive_total += 1
            if top is None:
                # 连候选都没有(混合检索返回空)—— 它既不在上界里也不在分母里,
                # 单独计数,免得下面的「另有 N 条」少数一条。
                empty_cases += 1
            else:
                positive_tops.append(top)
            if hit:
                hitting_tops.append(top)
            else:
                missed += 1
            mark = "正例" if hit else "漏召"
        shown = "   nan" if top is None else f"{top:6.3f}"
        emit(f"  {mark} {shown}  {case['query']}")

    emit("")
    if positive_tops:
        emit(f"正例 top-1(共 {positive_total} 条,含漏召的):最低 {min(positive_tops):.3f}"
             f" / 中位 {statistics.median(positive_tops):.3f} / 最高 {max(positive_tops):.3f}")
    if negative_tops:
        emit(f"干扰 top-1(共 {len(negative_tops)} 条):最高 {max(negative_tops):.3f}"
             f" / 最低 {min(negative_tops):.3f}")
    if hitting_tops:
        emit(f"其中**能命中**的正例 {len(hitting_tops)} 条:top-1 最低 {min(hitting_tops):.3f}"
             f" / 最高 {max(hitting_tops):.3f}   ← 阈值上界取这个最低值")
    if negative_tops and hitting_tops and max(negative_tops) < min(hitting_tops):
        emit(f"→ 可用阈值区间:({max(negative_tops):.3f}, {min(hitting_tops):.3f}]"
             f"  宽 {min(hitting_tops) - max(negative_tops):.3f}")
    else:
        emit("→ 无干净区间:干扰项 top-1 与能命中正例的 top-1 重叠 ——"
             " 这不是调阈值能解决的,要靠 Top-K 或语料层面解决")
    if missed:
        emit(f"注:另有 {missed} 条正例在阈值 0 下也召不回期望块,**与阈值无关**"
             "(检索质量问题,单独记账)。")
    if empty_cases:
        emit(f"注:{empty_cases} 条正例连候选都没召回到(混合检索返回空),不在上面的"
             "分数统计里 —— 那是索引/语料层面的问题。")


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
        "--dist", action="store_true",
        help="只打印原始重排分数分布(阈值 0,当前链路)与可用阈值区间",
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
    reranker = get_reranker(settings.reranker_model_path)

    if args.dist:
        # 阈值**固定 0**:要的是重排器的原始 sigmoid 分,过滤掉就看不见被滤掉的
        # 那一截,区间的下界(干扰项最高分)会整段消失。
        async with get_sessionmaker()() as session:
            retriever = KnowledgeRetriever(
                session, store, embedder, reranker, top_k=top_k, score_threshold=0.0
            )
            await _print_distribution(retriever, cases)
        await get_engine().dispose()
        return

    scores = []
    for run in range(args.repeat):
        async with get_sessionmaker()() as session:
            retriever = KnowledgeRetriever(
                session, store, embedder, reranker,
                top_k=top_k, score_threshold=threshold,
            )
            started = time.perf_counter()
            hits, results = await run_once(retriever, cases)
        scores.append(hits)
        emit(f"\n===== 第 {run + 1} 遍:{hits}/{len(cases)} = {hits / len(cases):.1%}"
             f"(耗时 {time.perf_counter() - started:.1f}s)=====")

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
