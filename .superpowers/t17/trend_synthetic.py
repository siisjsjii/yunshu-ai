"""T17 的**运行-读**验证装置(一次性,不进仓库):直接调 `render()` 喂造好的数,
把真实库里那两轮跑不出来的形状打出来 —— ↓ / ↑ / top_k 变 / 缺一个策略。

**不写库**:`render` 是纯函数,这里只喂内存里的 view。
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_trend import _overall, emit, render  # noqa: E402

BUCKETS = ("A_policy", "B_model", "C_colloquial", "D_absent", "E_multi")


def view(rid, minute, case_count, top_k, *, dense, bm25, rrf, rerank=None):
    """把一个策略的总读数摊回五个桶(每桶 60 条)—— `_overall` 再按 n 加权合回原值。"""
    def spread(v):
        if v is None:
            return None
        return {b: {"n": case_count // 5, "recall@10": v[0], "mrr": v[1]} for b in BUCKETS}

    strategies = {"纯dense": spread(dense), "纯BM25": spread(bm25), "混合(RRF)": spread(rrf)}
    if rerank is not None:
        strategies["混合+Rerank"] = spread(rerank)
    return {
        "id": rid,
        "at": datetime(2026, 9, 24, 21, minute, 0),
        "case_count": case_count,
        "inner_case_count": case_count,
        "top_k": top_k,
        # 与 `_row_view` 一样:分桶的原始分数 → 按条数加权的总读数
        "strategies": {k: _overall(v) for k, v in strategies.items() if v is not None},
    }


views = [
    # 第 1 轮:基准。**不打箭头**(没有可比对象)
    view(1, 3, 300, 10, dense=(0.747, 0.652), bm25=(0.710, 0.587), rrf=(0.747, 0.634),
         rerank=(0.747, 0.671)),
    # 第 2 轮:同一个 case_count ⇒ 可比 ⇒ 该出 ↓ / ↑
    view(2, 15, 300, 10, dense=(0.740, 0.650), bm25=(0.710, 0.587), rrf=(0.700, 0.655),
         rerank=(0.747, 0.671)),
    # 第 3 轮:top_k 变了 ⇒ 不可比(即使条数一样)
    view(3, 22, 300, 5, dense=(0.730, 0.640), bm25=(0.700, 0.580), rrf=(0.690, 0.640),
         rerank=(0.740, 0.660)),
    # 第 4 轮:条数变了 + 重排权重不在(少一列)
    view(4, 31, 5, 5, dense=(1.000, 0.900), bm25=(1.000, 0.850), rrf=(1.000, 0.767)),
    # 第 5 轮:与第 4 轮同规模 ⇒ 可比 ⇒ 箭头回来
    view(5, 40, 5, 5, dense=(0.800, 0.700), bm25=(1.000, 0.850), rrf=(1.000, 0.700)),
]

for line in render(views):
    emit(line)
