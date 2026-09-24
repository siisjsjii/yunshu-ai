"""`scripts/eval_trend.py` 的趋势表 —— **纯函数,不需要 db 标记**(它不碰库)。

## 为什么这份文件存在(上一轮我说错了,这里订正)

T17 的初版报告里写过「趋势表**不可单测** ⇒ 用实跑验证」。**那个判据是错的**:
`render()` 是**纯函数**(`scripts.eval_trend` 连 import 都不连库 —— `get_engine()`
只在 `main()` 里被调),`_row_view` 只读 ORM 对象的四个属性 ⇒ 拿
`types.SimpleNamespace` 造的假行就能把「行 → 视图 → 文本」整条链路跑起来。
真正的判据是「**它是不是纯函数**」,不是「它是不是格式化报告」。

错的代价当场就显出来了:初版的变异 M4(差值符号取反)只有「输出变了」、
**没有红** —— 因为没有断言存在。

## 造数的口径(便于手算复核)

假行里的桶都是 `n=60`、五个桶各一个,所以**头条值 = 五个桶的算术平均**
(那正是 `_overall` 的按条数加权在等权下的样子)、分桶值就是格子里的那个数。
每条用例的注释里都写了算式 —— 断言写的是**具体的数字**,不是「有箭头」。

⚠️ **别对整份输出做 `grep ↓`**:表头那行图例自己就含 `↑`/`↓`/`=`/`-`
(那是给读表的人看的),于是那种 grep **即使一条增减行都没打出来也匹配** ——
本仓编目过的「恒真的假断言」。下面的 `_body()` 先把图例滤掉再断言,
`test_legend_itself_would_fool_a_naive_grep` 把这件事钉住。
"""

import re
from datetime import datetime
from types import SimpleNamespace

from scripts.eval_trend import _delta_field, _row_view, render

BUCKETS = ("A_policy", "B_model", "C_colloquial", "D_absent", "E_multi")


def _strategies(**per_strategy) -> dict:
    """`{策略: {桶: (recall@10, mrr, answer_ok)}}` → `metrics["strategies"]` 的形状。"""
    return {
        name: {
            b: {"n": 60, "recall@5": v[0], "recall@10": v[0], "mrr": v[1],
                "conf": 0.5, "answer_ok": v[2]}
            for b, v in buckets.items()
        }
        for name, buckets in per_strategy.items()
    }


def _view(rid: int, strategies: dict, *, case_count: int = 300, top_k: int = 10,
          minute: int = 0) -> dict:
    """一行 `eval_runs` → `_row_view` 的产物(走**真**的 `_row_view`,不另写一份)。"""
    row = SimpleNamespace(
        id=rid,
        trigger_by="manual",
        case_count=case_count,
        created_at=datetime(2026, 9, 24, 21, minute, 0),
        metrics={"top_k": top_k, "case_count": case_count, "strategies": strategies},
    )
    return _row_view(row)


def _body(lines: list[str]) -> list[str]:
    """滤掉表头那行图例(它以 `（` 开头,自己就含 ↑/↓/=/-,见模块 docstring)。"""
    return [ln for ln in lines if not ln.startswith("（")]


def _cells(line: str) -> list[str]:
    """一行里的各格:按 2 个以上空格切开(格内只有一个空格)。"""
    return [t for t in re.split(r"\s{2,}", line.strip()) if t]


#: 第一轮的四个策略(桶 A 是 0.800/0.700,其余见算式)。
#: 头条(等权平均):recall@10 = (0.8+1.0+0.9+0.0+1.0)/5 = **0.740**;mrr = 0.660
BASE = dict(
    **{
        "纯dense": {
            "A_policy": (0.800, 0.700, 0.800), "B_model": (1.000, 0.900, 1.000),
            "C_colloquial": (0.900, 0.800, 0.900), "D_absent": (0.000, 0.000, 1.000),
            "E_multi": (1.000, 0.900, 1.000),
        },
        "纯BM25": {
            "A_policy": (0.800, 0.700, 0.800), "B_model": (0.900, 0.800, 0.900),
            "C_colloquial": (0.700, 0.500, 0.700), "D_absent": (0.000, 0.000, 1.000),
            "E_multi": (0.900, 0.800, 0.900),
        },
    }
)
#: 第二轮:A_policy 的 dense 掉 0.100(BM25 的 B_model 涨 0.050,用来验 ↑)。
NEXT = {
    "纯dense": dict(BASE["纯dense"], A_policy=(0.700, 0.600, 0.700)),
    "纯BM25": dict(BASE["纯BM25"], B_model=(0.950, 0.800, 0.950)),
}
#: 第二轮头条:纯dense recall@10 = (0.7+1.0+0.9+0.0+1.0)/5 = **0.720** ⇒ ↓ -0.020
#:           纯BM25 recall@10 = (0.8+0.95+0.7+0.0+0.9)/5 = **0.670** ⇒ ↑ +0.010


def test_first_round_has_no_arrow():
    """第一轮没有可比对象 ⇒ **一个箭头都不打**(body 里也不许有)。"""
    lines = render([_view(1, _strategies(纯dense=BASE["纯dense"]))])
    body = _body(lines)
    assert not any(("↓" in ln or "↑" in ln) for ln in body), \
        "第一轮不该有箭头"
    assert not any(ln.strip().endswith("0.000") and "=" in ln.split()[-2:] for ln in body)
    # 但那几个数都在(不然「没箭头」可以是「整张表没打」)
    assert any("recall@10=0.740 mrr=0.660" in ln for ln in body)


def test_legend_itself_would_fool_a_naive_grep():
    """⚠️ 把这条钉住:图例那行**自己**就含 ↑/↓/=/-,所以「grep 整份输出」是恒真的。"""
    lines = render([_view(1, _strategies(纯dense=BASE["纯dense"]))])
    legend = [ln for ln in lines if ln.startswith("（")][0]
    for glyph in ("↑", "↓", "=", "-"):
        assert glyph in legend
    # 而 body 里一个都没有 ⇒ 反面:断言必须只看 body
    assert not any("↑" in ln for ln in _body(lines))


def test_comparable_rounds_print_signed_deltas():
    """同规模、同 `top_k` ⇒ 该出正确的符号与**具体的数值**(手算可复核)。"""
    lines = render([
        _view(1, _strategies(**BASE), minute=3),
        _view(2, _strategies(**NEXT), minute=15),
    ])
    body = _body(lines)

    # 头条的差值行:紧跟在第 2 轮那一行的后面
    row2 = next(i for i, ln in enumerate(body) if ln.strip().startswith("2   2026"))
    delta = body[row2 + 1]
    assert "↓ -0.020" in delta, f"纯dense 的头条 recall@10 应掉 0.020:{delta!r}"
    assert "↑ +0.010" in delta, f"纯BM25 的头条 recall@10 应涨 0.010:{delta!r}"

    # 分桶表:差值内联在同一格里(0.700 相对 0.800 ⇒ ↓ -0.100)
    assert any(re.search(r"0\.700 ↓ -0\.100", ln) for ln in body), \
        "分桶表里 A_policy 的 dense recall@10 该内联 ↓ -0.100"
    assert any(re.search(r"0\.950 ↑ \+0\.050", ln) for ln in body), \
        "分桶表里 B_model 的 BM25 recall@10 该内联 ↑ +0.050"


def test_incomparable_case_count_prints_marker_and_no_arrows():
    """条数不同 ⇒ 打 `≠` 且**整个 body 里一个箭头都没有**。"""
    lines = render([
        _view(1, _strategies(**BASE), case_count=300, minute=3),
        _view(2, _strategies(**NEXT), case_count=60, minute=15),
    ])
    body = _body(lines)
    assert any("≠ 与上一轮不可比" in ln and "条数 300→60" in ln for ln in body)
    assert not any(("↓" in ln or "↑" in ln) for ln in body), \
        "不可比的两轮之间不许出现任何箭头"


def test_incomparable_top_k_prints_marker_and_no_arrows():
    """条数一样但 `top_k` 变了 ⇒ 一样判不可比(recall@10 的定义变了)。"""
    lines = render([
        _view(1, _strategies(**BASE), top_k=10, minute=3),
        _view(2, _strategies(**NEXT), top_k=5, minute=15),
    ])
    body = _body(lines)
    assert any("≠ 与上一轮不可比" in ln and "top_k 10→5" in ln for ln in body)
    assert not any(("↓" in ln or "↑" in ln) for ln in body)


def test_missing_strategy_renders_dash_not_zero():
    """某一轮没有的策略渲染成 `-`,**不是 `0.000`**(0 会被读成「没变化」)。"""
    assert _delta_field(None, 0.5) == "-"
    assert _delta_field(0.5, None) == "-"

    only_round1 = _strategies(**BASE)
    with_extra = _strategies(**BASE, **{"混合(RRF)": BASE["纯dense"]})
    lines = render([
        _view(1, only_round1, minute=3),
        _view(2, with_extra, minute=15),
    ])
    body = _body(lines)
    row2 = next(i for i, ln in enumerate(body) if ln.strip().startswith("2   2026"))
    tokens = _cells(body[row2 + 1])
    # 一个格的**内部**是「两位数 + 一个空格」,而格与格之间是两个空格 ⇒ 切开之后
    # 两个可比的策略各出 2 个 token,缺席的那个策略出 1 个 `-`。
    assert tokens[-1] == "-", f"第 1 轮没有「混合(RRF)」,那一格该是 -:{tokens!r}"
    assert tokens.count("-") == 1, f"只有缺席的那一格该是 -,其余都有数:{tokens!r}"
    assert tokens.count("= 0.000") == 4, f"两个可比策略 × 两个指标:{tokens!r}"


def test_bucket_view_is_on_by_default_and_shows_the_dilution():
    """**分桶表默认就打**(不是藏在开关后面)—— 且它把头条值被稀释那件事摆出来。

    纯dense 的头条 recall@10 = 0.740,而 A_policy 那一格是 0.800;
    `D_absent`(n=60,占 1/5)的分桶值是 0.000 ⇒ 它把头条值往下拉。
    """
    lines = render([_view(1, _strategies(纯dense=BASE["纯dense"]))])
    body = _body(lines)
    assert any(ln.startswith("---- 按桶 · recall@10 ----") for ln in lines)
    assert any(ln.startswith("---- 按桶 · answer_ok ----") for ln in lines)

    a_row = next(ln for ln in body if "A_policy" in ln)
    d_row = next(ln for ln in body if "D_absent" in ln)
    assert _cells(d_row)[-1] == "0.000", "D_absent 的 recall@10 结构性为 0"
    assert _cells(a_row)[-1] == "0.800", "A_policy 的 recall@10 是 0.800,不是头条的 0.740"
    # `answer_ok` 那一张:D_absent 是 1.000(拒答桶唯一有意义的指标)
    idx = [i for i, ln in enumerate(body) if ln.startswith("---- 按桶 · answer_ok")][0]
    d_ans = next(ln for ln in body[idx:] if "D_absent" in ln)
    assert _cells(d_ans)[-1] == "1.000"
