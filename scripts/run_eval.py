"""四策略检索评估:纯 dense / 纯 BM25 / 混合(RRF) / 混合+Rerank。

用法:
    .venv/Scripts/python.exe scripts/run_eval.py
    .venv/Scripts/python.exe scripts/run_eval.py --top-k 10
    .venv/Scripts/python.exe scripts/run_eval.py --limit 5 --trigger manual

ch09 起:跑完**追加一行** `eval_runs`(趋势表 `scripts/eval_trend.py` 的数据源)。
`--limit 0`(默认)= 全跑,行为/输出与 ch04 那版**一字不变**;`--limit N` 只跑前 N 条,
且那一轮 `case_count` 记的是**实际条数**(不同规模的轮次不许看起来可比 —— spec §10.2)。

⚠️ **`latest.json` 是「最近跑的那一轮」,`--limit N` 的轮次同样会写它** ——
验收脚本拿 `--limit 5` 跑两轮之后,`admin.html` 的评测页看到的就是那 5 条的结果
(不是全量)。要它回到全量,再跑一次不带 `--limit` 的即可。

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
from typing import Any, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.db.base import get_engine, get_sessionmaker
from app.db.models import EvalRun, KnowledgeChunk
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


def select_cases(cases: list[dict], limit: int) -> list[dict]:
    """`--limit`:只取前 N 条;**0(或负数)= 全跑,原样返回**。

    单独拎出来是为了让「实际跑了几条」这件事**只有一个来源** ——
    落库的 `case_count` 与真正喂给策略的用例必须是同一批(spec §10.2)。
    """
    if limit <= 0:
        return cases
    return cases[:limit]


class EvalCtx(NamedTuple):
    """四种策略要用的三件套 + 语料。

    构造**不加载权重**(embedder / reranker 都是懒加载),所以单测可以拿假件组装它。
    """

    chunks: dict[str, dict]
    store: Any
    embedder: Any
    reranker: Any
    reranker_available: bool


async def build_ctx(settings) -> EvalCtx:
    """建 Milvus / 嵌入 / 重排三件套 + 取语料。"""
    chunks = await load_chunks()
    store = get_vector_store(settings.milvus_uri, settings.milvus_collection)
    embedder = get_embedder(
        settings.embedding_model_path, settings.embedding_max_length,
        settings.embedding_batch_size)
    reranker_available = Path(settings.reranker_model_path).exists()
    reranker = get_reranker(settings.reranker_model_path)
    if not reranker_available:
        emit(f"⚠ 重排权重未就绪({settings.reranker_model_path} 不存在),混合+Rerank 策略将跳过")
    return EvalCtx(chunks=chunks, store=store, embedder=embedder,
                   reranker=reranker, reranker_available=reranker_available)


async def evaluate_cases(cases: list[dict], top_k: int, ctx: EvalCtx) -> dict:
    """跑四种策略,返回 `latest.json` 的内容(**不落盘、不落库**)。

    任何一步抛异常都直接向上冒 —— 调用方(`run_round`)据此保证
    「半途而废的一轮不会被记成一条完整的评估」。
    """
    chunks, store, embedder = ctx.chunks, ctx.store, ctx.embedder

    # 预嵌入所有 query(省得每策略重复加载)
    queries = [c["query"] for c in cases]
    emit(f"嵌入 {len(queries)} 条 query…")
    qvecs = embedder.encode(queries)

    def strategies():
        yield "纯dense", lambda q, qv: store.search(qv, top_k)
        yield "纯BM25", lambda q, qv: store.bm25_search(q, top_k)
        yield "混合(RRF)", lambda q, qv: store.hybrid_search(qv, q, top_k)

        def hybrid_rerank(q, qv):
            hits = store.hybrid_search(qv, q, max(10, top_k))
            if not hits:
                return []
            texts = [(i, chunks.get(str(i), {}).get("answer", "")) for i, _ in hits]
            scores = ctx.reranker.rerank(q, texts)
            ranked = sorted(zip(hits, scores), key=lambda x: -x[1])
            return [(i, s) for (i, _), s in ranked[:top_k]]

        yield "混合+Rerank", hybrid_rerank

    results: dict = {"top_k": top_k, "strategies": {}}
    for name, fn in strategies():
        if name == "混合+Rerank" and not ctx.reranker_available:
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

    return results


def write_latest(results: dict) -> Path:
    """写 `evals/results/latest.json`。

    **路径与结构一个字都不许动** —— `admin.html` 的评测页读它(spec §10.1)。
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "latest.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(f"\n结果写入 {out}")
    return out


def print_report(results: dict, cases: list[dict]) -> None:
    """打印按桶的对照表(口径与 ch04 那版逐字相同)。"""
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


async def _record_eval_run(*, trigger_by: str, case_count: int, metrics: dict) -> int:
    """往 `eval_runs` 追加一行,返回它的 id。

    **engine 用的是 `app/db/base.py` 的 lru_cache 单例,不另起一个。**
    理由:`app/kb/orchestrate.py` 的**后台任务**必须自建 engine,因为那条单例绑在
    「首次使用它的那个事件循环」上,而后台线程里是 `asyncio.run` 的**另一个**循环;
    本脚本从头到尾只有**一条**事件循环(命令行的 `asyncio.run`),不存在跨循环复用,
    所以单例是正确且更省的选择 —— 而且 `main` 原本就在用它的 `get_sessionmaker()`
    读 `knowledge_chunks`。dispose 由 `main` 的 `finally` 负责(**不会再留一个没人关的
    engine**:今天若为落库新起一个,`main` 尾部那句 dispose 管的是另一个对象)。
    """
    async with get_sessionmaker()() as session:
        row = EvalRun(trigger_by=trigger_by, case_count=case_count, metrics=metrics)
        session.add(row)
        await session.commit()
        return row.id


async def run_round(cases: list[dict], top_k: int, ctx: EvalCtx, *, trigger_by: str,
                    on_results=None) -> dict:
    """一轮 = **评估 + 落一行 `eval_runs`**,返回 `latest.json` 的内容。

    顺序是承重的:评估**完整**跑完才有落库那一步 —— `evaluate_cases` 抛出的任何
    异常都在写行之前冒出去,所以「Milvus 挂了 / key 用尽」这类半途而废的一轮
    **结构上不可能**被记成一条完整的评估(趋势表拿不到误导性的行)。

    `on_results` 是「结果算完了、但还没落库」那个**唯一**时机的钩子:`main` 用它把
    `latest.json` 写出去 —— 顺序与 ch04 那版一致(**先落结果文件,再落库**;
    落库若失败,人手里至少还有那份刚算出来的结果)。用例不传它,
    于是碰不到 `evals/results/latest.json`。
    """
    results = await evaluate_cases(cases, top_k, ctx)
    if on_results is not None:
        on_results(results)
    await _record_eval_run(
        trigger_by=trigger_by,
        case_count=len(cases),
        # **现有 latest.json 的内容原样进 metrics**,外面包一层 ——
        # 这样「评估集的形状」与「这一轮跑了多少条」是两个独立的键,
        # 不会因为 --limit 而在同一个键上出现两种含义(spec §10.3)。
        metrics={"top_k": top_k, "case_count": len(cases),
                 "strategies": results["strategies"]},
    )
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ch04 四策略检索评估")
    parser.add_argument("--top-k", type=int, default=_K)
    parser.add_argument("--limit", type=int, default=0,
                        help="只跑前 N 条(0 = 全跑)。给验收脚本用;"
                             "**case_count 记的是实际条数**,别让不同规模的轮次看起来可比")
    parser.add_argument("--trigger", default="manual",
                        help="触发方式,写进 eval_runs.trigger_by")
    return parser.parse_args(argv)


async def main() -> None:
    args = _parse_args()
    top_k = args.top_k
    # 先取全量、再按 --limit 截 —— `--limit 0` 时原样返回,与 ch04 那版一字不变。
    cases = select_cases(load_cases(), args.limit)
    ctx = await build_ctx(get_settings())
    try:
        results = await run_round(cases, top_k, ctx, trigger_by=args.trigger,
                                  on_results=write_latest)
        print_report(results, cases)
    finally:
        await get_engine().dispose()


if __name__ == "__main__":
    asyncio.run(main())
