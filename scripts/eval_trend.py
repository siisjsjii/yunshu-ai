"""评估趋势表:把 `eval_runs` 里的每一轮摆在一起,**标出每个指标相对上一轮的增减**。

用法:
    .venv/Scripts/python.exe scripts/run_eval.py --limit 5 --trigger manual
    .venv/Scripts/python.exe scripts/run_eval.py --limit 5 --trigger manual   # 再来一轮
    .venv/Scripts/python.exe scripts/eval_trend.py

需求原话是「哪个指标在下滑要**一眼看出来**」,所以每个指标都带一个相对上一轮的
箭头与差值 —— 而不是让人自己去比两行的数字。

## 三条口径(都是刻意的)

1. **每策略一行两个指标 = 跨桶按条数加权的总读数。** `eval_runs.metrics` 里存的是
   **分桶**的分数(`{"混合+Rerank": {"A_policy": {"n": 4, "recall@10": …}}}`),
   趋势表要的是「这一轮整体怎么样」。按 `n` 加权平均与「整体 Recall@10」同义;
   不加权的话,一个只有 1 条的桶与 60 条的桶等重。
2. **差值由**打印出来的**那两个三位小数算**。即先各自 `round(v, 3)` 再相减、再
   `round(…, 3)`。这样读者拿表上两个数字手算一遍,得到的**逐位相同** ——
   本仓对「验不了的数字」有过教训,趋势表不许再产一个。
3. **比较的是同一指标的前后两轮**:值各自打在自己的行上,差值打在下一行的
   **同一列**(列宽按指标名对齐,所以「哪个指标的差」一望即知)。
   第一轮没有可比对象,**不打箭头**。

## 「不可比」怎么呈现

`--limit 5` 的那一轮与全量那一轮,**分母不同**(一条分式与一条百分比)。
把它们的差值打出来是**误导** —— 那个差里混着「样本换了」,不是「模型变了」。
同理 `--top-k` 变了,`recall@10` 的**定义**就变了(检索只返回 top-k 条时,
`recall@10` 天生封顶在 `recall@k`)。

⇒ 只要 `case_count` 或 `top_k` 与上一轮不同,这一轮**打一行 `≠ 与上一轮不可比(…)`**,
**一个箭头都不打** —— 不打,而不是打了再加个脚注。脚注会被略过,箭头不会。
(第一轮是唯一的例外:它同样不打箭头,但理由是「没有上一轮」。)
"""

import asyncio
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.base import get_engine, get_sessionmaker
from app.db.models import EvalRun
from sqlalchemy import select

#: 表里展示的指标。分数口径见 `app/db/models.EvalRun` / spec §10.3。
METRICS = ("recall@10", "mrr")

#: 策略的**列序**。⚠️ 不能靠 JSON 里的键序 —— `metrics` 是 MySQL 的 JSON 列,
#: 而 MySQL 存对象时**按 (键长, 字典序) 重排**(实测:落库后是
#: `纯BM25, 纯dense, 混合(RRF), 混合+Rerank`,与 `run_eval.py` 的产出顺序不同)。
#: 这里钉死成 `run_eval.py` 打印对照表的那个顺序(读者的心智模型),
#: 表里没见过的策略名按名字排在最后。
STRATEGY_ORDER = ("纯dense", "纯BM25", "混合(RRF)", "混合+Rerank")

#: 列宽(按**显示宽度**算:CJK 字符占 2 格,`str.ljust` 会算成 1)。
W_ROUND, W_TIME, W_COUNT = 4, 19, 6
#: 一个策略占的宽度 = `recall@10=0.000`(15) + 空格 + `mrr=0.000`(9)。
W_FIELD1, W_FIELD2 = 15, 9
W_CELL = W_FIELD1 + 1 + W_FIELD2
GAP = "  "
W_PREFIX = W_ROUND + len(GAP) + W_TIME + len(GAP) + W_COUNT + len(GAP)

ARROW_DOWN, ARROW_UP, ARROW_SAME = "↓", "↑", "="
#: 上一轮没有这个策略(或这一轮没有)时的占位 —— 不是 0。
NO_VALUE = "-"


def emit(line: str = "") -> None:
    """UTF-8 字节写 stdout(本机 locale 是 cp936,直接 print 中文会炸)。"""
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        print(line)
        return
    stream.write(line.encode("utf-8") + b"\n")
    stream.flush()


def _width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _width(s))


def _overall(buckets: dict) -> dict[str, float]:
    """一个策略的各桶分数 → **按条数加权**的总读数(与「整体 Recall@10」同义)。"""
    n = sum(int(b.get("n") or 0) for b in buckets.values())
    if n <= 0:
        return {}
    out: dict[str, float] = {}
    for m in METRICS:
        total = sum(float(b.get(m) or 0.0) * int(b.get("n") or 0)
                    for b in buckets.values())
        out[m] = total / n
    return out


def _metric_cell(vals: dict[str, float]) -> str:
    fields = []
    for m in METRICS:
        v = vals.get(m)
        fields.append(f"{m}={v:.3f}" if v is not None else NO_VALUE)
    return pad(fields[0], W_FIELD1) + " " + pad(fields[1], W_FIELD2)


def _delta_field(cur: float | None, prev: float | None) -> str:
    """一个指标相对上一轮的箭头 + 差值。

    差值由**打印出来的那两个三位小数**算(`round` 后再相减),所以表上两个数字
    手算一遍就能逐位复现 —— 见模块 docstring 的口径 2。
    """
    if cur is None or prev is None:
        return NO_VALUE
    a, b = round(prev, 3), round(cur, 3)
    d = round(b - a, 3)
    if d == 0:
        # `abs` 是必须的:`b - a` 可能是个极小的负数,`-0.0` 会被格式化成 `-0.000`。
        return f"{ARROW_SAME} {abs(d):.3f}"
    return f"{ARROW_DOWN if d < 0 else ARROW_UP} {d:+.3f}"


def _delta_cell(cur: dict[str, float], prev: dict[str, float]) -> str:
    fields = [_delta_field(cur.get(m), prev.get(m)) for m in METRICS]
    return pad(fields[0], W_FIELD1) + " " + pad(fields[1], W_FIELD2)


def _row_view(row: EvalRun) -> dict:
    """一行 `eval_runs` → 趋势表要用的形状。"""
    metrics = row.metrics or {}
    strategies = metrics.get("strategies") or {}
    return {
        "id": row.id,
        "at": row.created_at,
        "case_count": row.case_count,
        # `metrics.case_count` 与顶层那一列**应当同值**(spec §10.3 刻意留两份)。
        # 不同值时表尾会喊一声 —— 那说明有人把其中一个改成了别的含义。
        "inner_case_count": metrics.get("case_count"),
        "top_k": metrics.get("top_k"),
        "strategies": {name: _overall(buckets) for name, buckets in strategies.items()},
    }


def _incomparable_reason(prev: dict, cur: dict) -> str | None:
    """与上一轮不可比的理由;可比时返回 None。"""
    parts = []
    if prev["case_count"] != cur["case_count"]:
        parts.append(f"条数 {prev['case_count']}→{cur['case_count']}")
    if prev["top_k"] != cur["top_k"]:
        parts.append(f"top_k {prev['top_k']}→{cur['top_k']}")
    return "、".join(parts) if parts else None


def render(views: list[dict]) -> list[str]:
    """把若干轮的读数排成表(纯函数,便于对着输出核)。"""
    seen: list[str] = []
    for v in views:
        for name in v["strategies"]:
            if name not in seen:
                seen.append(name)
    names = sorted(seen, key=lambda n: (
        STRATEGY_ORDER.index(n) if n in STRATEGY_ORDER else len(STRATEGY_ORDER), n))

    lines = ["===== 评估趋势 ====="]
    lines.append(
        "（↑/↓ = 相对上一轮的增减;第 1 轮没有可比对象、不打箭头。"
        "**条数或 top_k 与上一轮不同的两轮标「不可比」并整行不打箭头** —— "
        "分母不同的两个分数之间的差不是「变化」。）"
    )
    header = (pad("轮次", W_ROUND) + GAP + pad("时间", W_TIME) + GAP
              + pad("条数", W_COUNT) + GAP
              + GAP.join(pad(n, W_CELL) for n in names))
    lines.append(header)

    for i, v in enumerate(views):
        at = v["at"].strftime("%Y-%m-%d %H:%M:%S") if v["at"] else "?"
        cells = [(_metric_cell(v["strategies"].get(n, {})) if n in v["strategies"]
                  else pad(NO_VALUE, W_CELL)) for n in names]
        lines.append(pad(f"{i + 1:>3}", W_ROUND) + GAP + pad(at, W_TIME) + GAP
                     + pad(str(v["case_count"]), W_COUNT) + GAP
                     + GAP.join(cells))

        if i == 0:
            continue  # 第一轮:没有上一轮可比,不打箭头(brief 明写)
        prev = views[i - 1]
        reason = _incomparable_reason(prev, v)
        if reason:
            lines.append(" " * W_PREFIX
                         + f"≠ 与上一轮不可比（{reason}）:本条不打印增减")
            continue
        deltas = [(_delta_cell(v["strategies"].get(n, {}), prev["strategies"].get(n, {}))
                   if n in v["strategies"] and n in prev["strategies"]
                   else pad(NO_VALUE, W_CELL)) for n in names]
        lines.append(" " * W_PREFIX + GAP.join(deltas))

    mismatched = [v for v in views
                  if v["inner_case_count"] is not None
                  and v["inner_case_count"] != v["case_count"]]
    if mismatched:
        ids = ", ".join(str(v["id"]) for v in mismatched)
        lines.append("")
        lines.append(f"⚠ 有 {len(mismatched)} 轮的 `metrics.case_count` 与顶层 "
                     f"`case_count` 不一致(eval_runs.id: {ids})—— 两张尺子,别信任何一个")
    return lines


async def main() -> None:
    try:
        async with get_sessionmaker()() as session:
            # 按 (created_at, id) 升序:**秒精度**下两轮可能落在同一秒上,
            # 那时只有自增主键能定先后(`created_at` 单独排会是随机的)。
            rows = list(
                (
                    await session.execute(
                        select(EvalRun).order_by(EvalRun.created_at, EvalRun.id)
                    )
                )
                .scalars()
                .all()
            )
    finally:
        await get_engine().dispose()

    if not rows:
        emit("===== 评估趋势 =====")
        emit("（eval_runs 里还没有任何一轮 —— 先跑 "
             "`scripts/run_eval.py --limit 5 --trigger manual`)")
        return

    for line in render([_row_view(r) for r in rows]):
        emit(line)


if __name__ == "__main__":
    asyncio.run(main())
