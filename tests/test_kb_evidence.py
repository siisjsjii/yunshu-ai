"""置信度合成的三条信号(spec §4.1)。

⚠️ 用例的输入要**真的走到那条分支**:条数信号封顶用 evidence_max_count=3,
所以"封顶"的用例至少要有 4 条;**分差**的用例要真的建出 top1/top2 两个不同分。

⚠️ 三条权重用例里**两条是精确值断言**(不是两个置信度比大小)。原本那条
「比大小」的写法(R3)必须换掉:排出来的序在「权重被硬编码成默认值」的实现下
**一模一样**,它对着一个坏实现也是绿的。
"""

import pytest

from app.config import Settings
from app.kb.evidence import evidence_confidence, evidence_detail
from app.retrieval.search import RetrievedChunk


def _settings(**over):
    base = {
        "openai_base_url": "http://x", "openai_api_key": "k",
        "openai_model": "m", "database_url": "mysql://x",
        "evidence_min_score": 0.15, "evidence_max_count": 3,
        "w_evidence_top1": 0.6, "w_evidence_count": 0.2, "w_evidence_gap": 0.2,
    }
    return Settings(**{**base, **over}, _env_file=None)


def _c(score, cid=1):
    return RetrievedChunk(question="q", answer="a", category="c",
                          chunk_id=cid, score=score)


def test_empty_evidence_is_zero():
    s = _settings()
    assert evidence_confidence([], settings=s) == 0.0
    assert evidence_detail([], settings=s)["confidence"] == 0.0


def test_single_chunk_has_no_gap_signal():
    """只有一条时 top2 记 0 ⇒ gap 就是 top1。这条**不是**"没有信息",是设计。"""
    s = _settings()
    d = evidence_detail([_c(0.8)], settings=s)
    assert d["top1"] == 0.8 and d["top2"] == 0.0 and d["gap"] == 0.8
    assert d["count"] == 1


def test_ties_make_gap_zero_and_drag_confidence_down():
    """两条同分 ⇒ gap=0,置信度**明显低于**只有一条同分的情况。"""
    s = _settings()
    tie = evidence_confidence([_c(0.8, 1), _c(0.8, 2)], settings=s)
    lone = evidence_confidence([_c(0.8, 1)], settings=s)
    assert tie < lone


def test_count_signal_saturates_at_max_count():
    """4 条 ≥ max_count=3 ⇒ 条数信号打满,再加条目**不再变**。"""
    s = _settings()
    three = evidence_confidence([_c(0.5, i) for i in range(3)], settings=s)
    four = evidence_confidence([_c(0.5, i) for i in range(4)], settings=s)
    assert three == four


def test_low_scores_do_not_count_toward_the_count_signal():
    """低于 evidence_min_score 的块**不计入**条数 —— 否则一堆噪声会把置信度抬起来。"""
    s = _settings()
    noisy = evidence_confidence([_c(0.5, 1)] + [_c(0.01, i) for i in range(2, 6)], settings=s)
    clean = evidence_confidence([_c(0.5, 1)], settings=s)
    assert noisy == clean


def test_confidence_is_always_in_unit_interval():
    s = _settings()
    assert 0.0 <= evidence_confidence([_c(1.0, i) for i in range(10)], settings=s) <= 1.0


# ---- R3:把 brief 那条「比大小」的权重用例换成两条**精确值**断言 ----
# 原用例的两条断言都是 `confidence(a) > confidence(b)`,而那个序在「权重读配置」
# 与「权重硬编码成默认值」两种实现下**完全一样** ⇒ 它抓不到任何东西。精确值断言
# 才区分得开:下面两条在硬编码实现下必红。


def test_top1_weight_only_returns_top1_exactly():
    """另两个权重压成 0 ⇒ 输出必须**恰好等于** top1(不是"比谁大")。"""
    s = _settings(w_evidence_top1=1.0, w_evidence_count=0.0, w_evidence_gap=0.0)
    # top1=0.8、top2=0.3:若实现把权重写死成 0.6/0.2/0.2,这里会得到 0.7133 而不是 0.8。
    assert evidence_confidence([_c(0.8, 1), _c(0.3, 2)], settings=s) == pytest.approx(0.8)


def test_count_weight_only_returns_saturated_count_exactly():
    """只留条数权重 ⇒ 输出**恰好等于** min(有效条数 / max_count, 1.0)。

    max_count 取 4、有效条数取 2 ⇒ 期望值 0.5,**整除**、不引入 1/3 那种无限
    循环小数的舍入误差(实现会把结果 round 到 4 位,用 1/3 会让 approx 的默认
    相对容差 1e-6 判红 —— 那是"测舍入"不是"测权重")。
    """
    s = _settings(w_evidence_top1=0.0, w_evidence_count=1.0, w_evidence_gap=0.0,
                  evidence_max_count=4)
    # 硬编码权重下这里是 0.6*0.9 + 0.2*0.5 + 0.2*0.1 = 0.66,不是 0.5。
    assert evidence_confidence([_c(0.9, 1), _c(0.8, 2)], settings=s) == pytest.approx(2 / 4)
