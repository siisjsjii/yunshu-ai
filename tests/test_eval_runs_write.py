"""`scripts/run_eval.py` 落一行 `eval_runs` 那一段 —— 真库往返(ch09 T17)。

db 标记:读**真实** .env(不加 `_env_file=None`),需要 MySQL 已起 + ch09 的 DDL 已应用。

## 为什么这一段值得一个真库用例(H4)

`eval_runs` 是趋势表(`scripts/eval_trend.py`)的**唯一数据源**。这一行写错了,
趋势表会把「5 条的一轮」与「300 条的一轮」画在同一把尺子上 —— 而**没有任何东西会报错**。
所以这里写的是**真的 `EvalRun` 行、真的 commit**,再用**新 session** 读回来
(不 stub 写口:替身替被测对象完成语义是本仓编目过的形态 ⑦)。

## 判别力(H3)

`test_eval_run_row_carries_case_count_separately` **不是**「自己算好 5 再喂给写口」——
那种写法下,一个 `case_count=len(load_cases())` 的实现照样绿(它压根没机会算错)。
用例走的是**脚本自己那两步**:`select_cases(全量, 5)` → `run_round(...)`,
`case_count` 由 `run_round` 从**它拿到的那批 cases** 现算。

用例开头的**自检**是承重的:全量条数必须 ≠ 5 ——否则「记了实际条数」与
「记了全量」在读数上**一模一样**,用例退化成恒真(本仓形态 ⑤)。

## 半途而废的一轮(H2)

`test_failed_round_writes_no_row` 注入的是**处理之前**的形态:一个**真会抛**
`ToolInfrastructureError` 的 store(Milvus 挂掉那条路径的原型),
让**真的** `evaluate_cases` 在第一个策略的第一个 query 上炸。
然后断「库里没有这一轮的痕」—— 「先写行、再评估」的实现必红。
"""

import pytest
from sqlalchemy import delete, select

from app.db.base import get_sessionmaker
from app.db.models import EvalRun
from app.tools.errors import ToolInfrastructureError
from scripts import run_eval

pytestmark = pytest.mark.db

#: `--limit 5` 那一轮的条数。**与全量条数不同**是这条用例的前提(见自检)。
LIMIT = 5
TOP_K = 10
#: 探针触发值:真实轮次用的是 `manual` / `scheduled`,不会撞上。
PROBE_TRIGGER = "t17probe"


class _FakeEmbedder:
    """假嵌入:每条 query 一个定长向量。**不加载 BGE-M3、不联网**。"""

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _EmptyStore:
    """三种检索都返回空 —— 真的 `evaluate_cases` 会跑完三种策略,指标全 0。"""

    def search(self, qvec, k):
        return []

    def bm25_search(self, q, k):
        return []

    def hybrid_search(self, qvec, q, k):
        return []


class _DeadStore(_EmptyStore):
    """Milvus 不可达那种故障:第一个策略的第一次检索就抛。

    **这正是本仓那条「注入处理之前的形态」**:抛的是 `ToolInfrastructureError`
    本身 —— `evaluate_cases` 里没有任何 `except` 会再翻译一次,所以这条异常
    一路上抛到 `run_round` 的调用方(与真机上 Milvus 挂掉时同一条路)。
    """

    def search(self, qvec, k):
        raise ToolInfrastructureError("Milvus 不可达(T17 探针)")


def _ctx(store=None) -> "run_eval.EvalCtx":
    return run_eval.EvalCtx(
        chunks={},
        store=store or _EmptyStore(),
        embedder=_FakeEmbedder(),
        reranker=None,
        reranker_available=False,
    )


async def _cleanup() -> None:
    """删掉本文件的探针行并**提交**(`async with session` 退出是 rollback)。"""
    async with get_sessionmaker()() as session:
        await session.execute(
            delete(EvalRun).where(EvalRun.trigger_by == PROBE_TRIGGER)
        )
        await session.commit()


async def _probe_rows() -> list[EvalRun]:
    """**新 session** 读回:同 session 重读是否打到库取决于身份映射里还有谁在引用。"""
    async with get_sessionmaker()() as session:
        return list(
            (
                await session.execute(
                    select(EvalRun)
                    .where(EvalRun.trigger_by == PROBE_TRIGGER)
                    .order_by(EvalRun.id)
                )
            )
            .scalars()
            .all()
        )


@pytest.fixture(autouse=True)
async def _no_probe_leftovers():
    await _cleanup()
    yield
    await _cleanup()


@pytest.mark.anyio
async def test_eval_run_row_carries_case_count_separately():
    """`--limit 5` 那一轮:两处 `case_count` 都是**实际**条数,不是全量条数。"""
    full = run_eval.load_cases()
    # 自检:全量条数必须 ≠ LIMIT。相等的话「记实际条数」与「记全量」读数相同,
    # 这条断言就变成恒真 —— 它守护的性质当场消失(本仓形态 ⑤)。
    assert len(full) > LIMIT, (
        f"评估集只有 {len(full)} 条,与 LIMIT={LIMIT} 区分不开 —— "
        "请把 LIMIT 调小,否则这条用例没有判别力"
    )
    cases = run_eval.select_cases(full, LIMIT)
    assert len(cases) == LIMIT

    await run_eval.run_round(cases, TOP_K, _ctx(), trigger_by=PROBE_TRIGGER)

    rows = await _probe_rows()
    assert len(rows) == 1, "一轮评估应该正好落一行"
    row = rows[0]
    # 顶层那一列(H3:记 5 而不是 300)
    assert row.case_count == LIMIT
    assert row.case_count != len(full)
    # R4:`metrics` 是「现有 latest.json 的内容原样进、外面包一层」——
    # 所以 `case_count` 在这里**也有**一份,而且与顶层同值。
    assert row.metrics["case_count"] == LIMIT
    assert row.metrics["top_k"] == TOP_K
    # 三种策略都在(重排权重不在时跳过 —— 与脚本自己的行为一致)
    assert set(row.metrics["strategies"]) == {"纯dense", "纯BM25", "混合(RRF)"}


@pytest.mark.anyio
async def test_failed_round_writes_no_row():
    """评估半途炸掉 ⇒ **一行都不写**:写库在评估**之后**,顺序是承重的。"""
    cases = run_eval.select_cases(run_eval.load_cases(), LIMIT)

    with pytest.raises(ToolInfrastructureError):
        await run_eval.run_round(cases, TOP_K, _ctx(store=_DeadStore()),
                                 trigger_by=PROBE_TRIGGER)

    assert await _probe_rows() == [], (
        "评估在第一个策略就抛了,却仍然落下了一行 —— "
        "趋势表会把这一轮当成一次**完整**的评估"
    )
