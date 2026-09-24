"""评估趋势表:把 `eval_runs` 里的每一轮摆在一起,**标出每个指标相对上一轮的增减**。

用法:
    .venv/Scripts/python.exe scripts/run_eval.py --limit 5 --trigger manual
    .venv/Scripts/python.exe scripts/run_eval.py --limit 5 --trigger manual   # 再来一轮
    .venv/Scripts/python.exe scripts/eval_trend.py
    .venv/Scripts/python.exe scripts/eval_trend.py --last 2      # 只看最近两轮

需求原话是「哪个指标在下滑要**一眼看出来**」,所以每个指标都带一个相对上一轮的
箭头与差值 —— 而不是让人自己去比两行的数字。

## 两张表:头条(每策略)+ **按桶**(spec §10.2 点名要的那张)

```
【头条】轮次 / 时间 / 条数 / 每策略跨桶加权总分(recall@10、mrr)   ← brief 草图的那张
【按桶】轮次 / 时间 / 条数 / 桶 / 每策略 × recall@10、mrr、answer_ok
```

**为什么必须有「按桶」那张**(这不是格式偏好,是功能问题):头条值是把五个桶按条数
加权合出来的,而 `D_absent`(应拒答桶,n=60,占 1/5)的 `recall@10`/`mrr`
**结构性为 0** —— 它没有「正确章节」可召回。于是:

- 纯dense 的 `recall@10` 分桶值实测 **A 0.817 / B 1.000 / C 0.933 / D 0.000 / E 0.983**,
  头条值 **0.747**,而四个**可答**桶的平均是 **0.933**;
- **一个只在 A_policy 上发生的 0.067 真实回退,在头条值上只动 0.013(≈1/5)** ——
  「一眼看出下滑」这件事被稀释掉了;
- `answer_ok`(应拒答桶**唯一**有意义的指标:检索为空/不命中相关章节才算对)
  在头条里根本不显示 ⇒ **拒答行为的回退完全看不见**。

所以分桶表**默认就打**(不是藏在开关后面),并且它比 spec §10.2 多带一列
`answer_ok` —— §10.2 列的是「各桶 Recall@10 / MRR」,而对 `D_absent` 来说
那两个指标恒为 0、只有一个恒 1 的 `answer_ok` 能反映它。这一处偏离已记账。

## 四条口径(都是刻意的)

1. **头条值是跨桶按条数加权的总读数**(与「整体 Recall@10」同义);分桶表原样给
   每个桶自己的值。不加权的话,一个只有 1 条的桶与 60 条的桶等重。
2. **差值由**打印出来的**那些三位小数算**。即先各自 `round(v, 3)` 再相减、再
   `round(…, 3)`。这样读者拿表上两个数字手算一遍,得到的**逐位相同** ——
   本仓对「验不了的数字」有过教训,趋势表不许再产一个。
3. **比较的是同一指标的前后两轮**:头条表把差值打在下一行的**同一列**;
   分桶表把它**内联在同一格里**(那格里只有一行可放,内联能省一半行数)。
   两种都保证「差值紧挨着它比较的那两个数」。第一轮没有可比对象,**不打箭头**。
4. `--last N` 是**截断历史**:截后的第一轮按「没有上一轮」处理(不打箭头)——
   这样**表上出现的每一个箭头都能用表上的数字手算复核**。想看被截掉那一轮的
   增减就把 N 调大。

## 「不可比」怎么呈现

`--limit 5` 的那一轮与全量那一轮,**分母不同**(一条分式与一条百分比)。
把它们的差值打出来是**误导** —— 那个差里混着「样本换了」,不是「模型变了」。
同理 `--top-k` 变了,`recall@10` 的**定义**就变了(检索只返回 top-k 条时,
`recall@10` 天生封顶在 `recall@k`)。

⇒ 只要 `case_count` 或 `top_k` 与上一轮不同,这一轮**打一行 `≠ 与上一轮不可比(…)`**,
**一个箭头都不打** —— 不打,而不是打了再加个脚注。脚注会被略过,箭头不会。
(第一轮是唯一的例外:它同样不打箭头,但理由是「没有上一轮」。)

## 已知限制(记账,不打算改)

- **(M-2)`_width` 只把 `W`/`F` 算 2 格**:`↓`/`↑`/`⚠` 属 East-Asian **Ambiguous**,
  这里按 1 格算。把 Ambiguous 渲染成宽字符的终端上,箭头后面会错一列。
  ⚠️ **「列对齐是核过的」这个说法要打折** —— 核验用的就是本模块这份 `_width`,
  是自指的,查不出这一类。属外观级。
- **(M-4)`_incomparable_reason` 只看 `case_count` / `top_k`**:同一个规模、同一个
  top_k,但**标尺换了**(用例文件被就地改过、知识库变了)会被判成可比并打箭头。
  要真正闭掉这条,`metrics` 里得再存一份「用例集摘要」(那是 spec §10.3 之外的
  一个新键,没做)。今天的暴露面有限:`--limit` 总是取**前 N**条,不会换样本。
"""

import argparse
import asyncio
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.base import get_engine, get_sessionmaker
from app.db.models import EvalRun
from sqlalchemy import select

#: 头条表(每策略跨桶加权)展示的指标。口径见 `app/db/models.EvalRun` / spec §10.3。
METRICS = ("recall@10", "mrr")
#: 分桶表展示的指标。比 spec §10.2 多一个 `answer_ok` —— 对 `D_absent`(应拒答桶)
#: 来说 recall@10/mrr 结构性为 0,**只有它**能反映拒答行为。
BUCKET_METRICS = ("recall@10", "mrr", "answer_ok")

#: 策略的**列序**。⚠️ 不能靠 JSON 里的键序 —— `metrics` 是 MySQL 的 JSON 列,
#: 而 MySQL 存对象时**按 (键长, 字典序) 重排**(实测:落库后是
#: `纯BM25, 纯dense, 混合(RRF), 混合+Rerank`,与 `run_eval.py` 的产出顺序不同)。
#: 这里钉死成 `run_eval.py` 打印对照表的那个顺序(读者的心智模型),
#: 表里没见过的策略名按名字排在最后。
STRATEGY_ORDER = ("纯dense", "纯BM25", "混合(RRF)", "混合+Rerank")

#: 列宽(按**显示宽度**算:CJK 字符占 2 格,`str.ljust` 会算成 1;见「已知限制 M-2」)。
W_ROUND, W_TIME, W_COUNT = 4, 19, 6
#: 头条表一个策略占的宽度 = `recall@10=0.000`(15) + 空格 + `mrr=0.000`(9)。
W_FIELD1, W_FIELD2 = 15, 9
W_CELL = W_FIELD1 + 1 + W_FIELD2
#: 分桶表一个格 = `0.000`(5) + 空格 + `↓ -0.000`(8)。
W_BFIELD = 14
GAP = "  "
W_PREFIX = W_ROUND + len(GAP) + W_TIME + len(GAP) + W_COUNT + len(GAP)

ARROW_DOWN, ARROW_UP, ARROW_SAME = "↓", "↑", "="
#: 该轮 / 该策略没有这个数(例如重排权重不在、「混合+Rerank」被跳过)时的占位。
#: **不是 0** —— 0 会被读成「没变化」。
NO_VALUE = "-"

LEGEND = (
    "（↑/↓ = 相对上一轮的增减;`=` = 没变;第 1 轮没有可比对象、不打箭头。"
    "**条数或 top_k 与上一轮不同的两轮标「不可比」并整个不打箭头** —— "
    "分母不同的两个分数之间的差不是「变化」。"
    "`-` = 该轮没有这个数(例如权重不在时跳过了「混合+Rerank」),**不是 0**。）"
)


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
    """一个策略的各桶分数 → **按条数加权**的总读数(与「整体 Recall@10」同义)。

    ⚠️ 它是**头条值**:`D_absent` 那种结构性为 0 的桶会把每个头条值稀释掉
    (占比多少就稀释多少)—— 这正是分桶表必须默认打出来的原因(见模块 docstring)。
    """
    n = sum(int(b.get("n") or 0) for b in buckets.values())
    if n <= 0:
        return {}
    out: dict[str, float] = {}
    for m in METRICS:
        total = sum(float(b.get(m) or 0.0) * int(b.get("n") or 0)
                    for b in buckets.values())
        out[m] = total / n
    return out


def _by_bucket(buckets: dict) -> dict[str, dict[str, float]]:
    """`{桶: {指标: 值}}` —— 分桶表要的形状(原样,不做跨桶加权)。"""
    out: dict[str, dict[str, float]] = {}
    for bucket, b in buckets.items():
        vals = {m: float(b[m]) for m in BUCKET_METRICS if b.get(m) is not None}
        if vals:
            out[bucket] = vals
    return out


def _metric_cell(vals: dict[str, float]) -> str:
    fields = []
    for m in METRICS:
        v = vals.get(m)
        fields.append(f"{m}={v:.3f}" if v is not None else NO_VALUE)
    return pad(fields[0], W_FIELD1) + " " + pad(fields[1], W_FIELD2)


def _delta_field(cur: float | None, prev: float | None) -> str:
    """一个指标相对上一轮的箭头 + 差值。

    差值由**打印出来的那些三位小数**算(`round` 后再相减),所以表上两个数字
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
        "by_bucket": {name: _by_bucket(buckets) for name, buckets in strategies.items()},
    }


def _incomparable_reason(prev: dict, cur: dict) -> str | None:
    """与上一轮不可比的理由;可比时返回 None。见「已知限制 M-4」。"""
    parts = []
    if prev["case_count"] != cur["case_count"]:
        parts.append(f"条数 {prev['case_count']}→{cur['case_count']}")
    if prev["top_k"] != cur["top_k"]:
        parts.append(f"top_k {prev['top_k']}→{cur['top_k']}")
    return "、".join(parts) if parts else None


def _names(views: list[dict]) -> tuple[list[str], list[str]]:
    seen: list[str] = []
    buckets: list[str] = []
    for v in views:
        for name in v["strategies"]:
            if name not in seen:
                seen.append(name)
        for name, bs in v["by_bucket"].items():
            for b in bs:
                if b not in buckets:
                    buckets.append(b)
    names = sorted(seen, key=lambda n: (
        STRATEGY_ORDER.index(n) if n in STRATEGY_ORDER else len(STRATEGY_ORDER), n))
    return names, sorted(buckets)


def _at(v: dict) -> str:
    return v["at"].strftime("%Y-%m-%d %H:%M:%S") if v["at"] else "?"


def _head_prefix(names: list[str]) -> str:
    return (pad("轮次", W_ROUND) + GAP + pad("时间", W_TIME) + GAP
            + pad("条数", W_COUNT) + GAP
            + GAP.join(pad(n, W_CELL) for n in names))


def _strategy_lines(views: list[dict], names: list[str]) -> list[str]:
    """头条表:每策略跨桶加权总分 + 相对上一轮的箭头(差值另起一行,同一列)。"""
    lines = ["", "---- 每策略(跨桶加权;分桶见下)----", _head_prefix(names)]
    for i, v in enumerate(views):
        cells = [(_metric_cell(v["strategies"].get(n, {})) if n in v["strategies"]
                  else pad(NO_VALUE, W_CELL)) for n in names]
        lines.append(pad(f"{i + 1:>3}", W_ROUND) + GAP + pad(_at(v), W_TIME) + GAP
                     + pad(str(v["case_count"]), W_COUNT) + GAP + GAP.join(cells))

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
    return lines


def _bucket_lines(views: list[dict], names: list[str], buckets: list[str]) -> list[str]:
    """分桶表:一行 = (轮次 × 桶),列 = 各策略,每格内联这一格的增减。"""
    w_bucket = max([_width("桶")] + [_width(b) for b in buckets])
    prefix = (pad("轮次", W_ROUND) + GAP + pad("时间", W_TIME) + GAP
              + pad("条数", W_COUNT) + GAP + pad("桶", w_bucket) + GAP)
    lines: list[str] = []
    for metric in BUCKET_METRICS:
        lines.append("")
        lines.append(f"---- 按桶 · {metric} ----")
        lines.append(prefix + GAP.join(pad(n, W_BFIELD) for n in names))
        for i, v in enumerate(views):
            # 第一轮没有上一轮 ⇒ 不打箭头;不可比的那一轮整轮不打箭头(理由一行说明)。
            prev = views[i - 1] if i > 0 else None
            reason = _incomparable_reason(prev, v) if prev is not None else None
            if reason:
                lines.append(" " * _width(prefix)
                             + f"≠ 与上一轮不可比（{reason}）:本轮不打印增减")
            for j, b in enumerate(buckets):
                cells = []
                for n in names:
                    val = v["by_bucket"].get(n, {}).get(b, {}).get(metric)
                    if val is None:
                        cells.append(pad(NO_VALUE, W_BFIELD))
                        continue
                    text = f"{round(val, 3):.3f}"
                    if prev is not None and reason is None:
                        pval = prev["by_bucket"].get(n, {}).get(b, {}).get(metric)
                        text += f" {_delta_field(val, pval)}"
                    cells.append(pad(text, W_BFIELD))
                # 一轮的五行里只有第一行印 轮次/时间/条数 —— 后面四行留白:
                # 每行都印一遍的话,一屏里同一个时间戳会出现五次,表就不好读了。
                row = (pad(f"{i + 1:>3}", W_ROUND) + GAP + pad(_at(v), W_TIME) + GAP
                       + pad(str(v["case_count"]), W_COUNT) + GAP) if j == 0 \
                    else " " * _width(prefix)
                lines.append(row + pad(b, w_bucket) + GAP + GAP.join(cells))
    return lines


def render(views: list[dict]) -> list[str]:
    """把若干轮的读数排成表(纯函数:不连库、不读文件,可以直接喂造好的 views)。"""
    names, buckets = _names(views)
    lines = ["===== 评估趋势 =====", LEGEND]
    lines += _strategy_lines(views, names)
    lines += _bucket_lines(views, names, buckets)

    mismatched = [v for v in views
                  if v["inner_case_count"] is not None
                  and v["inner_case_count"] != v["case_count"]]
    if mismatched:
        ids = ", ".join(str(v["id"]) for v in mismatched)
        lines.append("")
        lines.append(f"⚠ 有 {len(mismatched)} 轮的 `metrics.case_count` 与顶层 "
                     f"`case_count` 不一致(eval_runs.id: {ids})—— 两张尺子,别信任何一个")
    return lines


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估趋势表(ch09 §10.2)")
    parser.add_argument("--last", type=int, default=0,
                        help="只看最近 N 轮(0 = 全部)。**注意是截断历史**:截后的第一轮"
                             "按「没有上一轮」处理、不打箭头 —— 这样表上每个箭头都能"
                             "用表上的数字手算复核;想看那一轮的增减就把 N 调大")
    return parser.parse_args(argv)


async def main() -> None:
    args = _parse_args()
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

    views = [_row_view(r) for r in rows]
    if args.last > 0:
        views = views[-args.last:]
    for line in render(views):
        emit(line)


if __name__ == "__main__":
    asyncio.run(main())
