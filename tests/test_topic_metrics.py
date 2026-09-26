"""评测指标 —— 纯函数,所以每一处口径都可以钉死。

⚠️ **本文件由 ch10-B Task 9 建**,不是 Task 10。

理由:训练脚本的 `compute_metrics` 与 Task 10 的评测报告**必须同源**
(Task 9 计划的那句「同一组函数,不许各写一份」)。Task 9 先跑,所以
`app/topic/metrics.py` 的基础四个函数由 Task 9 落地。

⚠️ **Task 10 请往本文件里追加 `mislabelled_flow` 的用例,不要重建本文件**:
重建会把这里 micro/macro-F1 的用例一起抹掉 —— Task 10 的计划里**没有**它们。
(Task 10 的 Step 2「跑测试,确认失败」仍然成立:它 import 的
`mislabelled_flow` 今天还不存在 ⇒ ImportError ⇒ 红。)
"""

import pytest

from app.topic.metrics import (
    label_count_match,
    macro_f1,
    micro_f1,
    per_class_prf,
    subset_accuracy,
)

Y_TRUE = [["退换货"], ["退换货", "尺码"], ["物流"]]
Y_PRED = [["退换货"], ["退换货"], ["物流"]]


def test_per_class_prf_reports_support():
    rows = {r["label"]: r for r in per_class_prf(Y_TRUE, Y_PRED, ["退换货", "尺码", "物流"])}
    assert rows["退换货"]["support"] == 2  # 两条真实里有它
    assert rows["退换货"]["tp"] == 2
    assert rows["尺码"]["support"] == 1
    assert rows["尺码"]["tp"] == 0  # 模型漏了
    assert rows["尺码"]["fn"] == 1
    assert rows["尺码"]["precision"] == 0.0  # 分母 0 时按 0 处理,不是 NaN


def test_precision_denominator_zero_is_zero_not_nan():
    """没预测过任何一条的类,精确率是 0 而不是 NaN。

    NaN 会**污染整张表**(求和/均值全变 NaN),而报告会看起来「有数」。
    """
    rows = {r["label"]: r for r in per_class_prf([["评价"]], [["评价"]], ["评价", "账号"])}
    assert rows["账号"]["precision"] == 0.0
    assert rows["账号"]["precision"] == rows["账号"]["precision"]  # 不是 NaN


def test_recall_denominator_zero_is_zero_not_nan():
    """`support == 0`(该类的真值一条都没有)时召回率也是 0。

    这条与上面那条是**两个不同的分母**;只写一条的话,另一条可以
    返回 NaN 而没有任何东西会红。
    """
    rows = {r["label"]: r for r in per_class_prf([["评价"]], [["评价"]], ["评价", "账号"])}
    assert rows["账号"]["recall"] == 0.0
    assert rows["账号"]["f1"] == 0.0


def test_subset_accuracy_is_exact_set_match():
    """整条完全一致 —— 多标签最严的指标,也是「一个不多一个不少」的最严读法。"""
    assert subset_accuracy(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)
    # 顺序不同但集合相同 ⇒ 算对
    assert subset_accuracy([["尺码", "退换货"]], [["退换货", "尺码"]]) == 1.0


def test_label_count_match_is_about_the_count_only():
    """**个数**对就算对(标签内容可以错)—— 这是需求原话「一个不多一个不少」
    的字面读法,与 `subset_accuracy` 是两个不同的问题,不要合并。

    ⚠️ **计划原文这条写错了**:它拿上面那对 `Y_TRUE` / `Y_PRED` 断言 `== 1.0`,
    而那对里第 2 条的个数是 **2 vs 1**(真值 `[退换货, 尺码]`、预测 `[退换货]`),
    正确读数是 **2/3**。它把「多数行对」当成了「全对」——
    如果照抄,这条断言会在正确实现上红、而在把上限写成 `1.0` 的实现上绿。
    这里改成两组**各自能分辨**的输入。
    """
    # 内容全错、个数相同 ⇒ 算对
    assert label_count_match([["尺码", "退换货"]], [["运费", "物流"]]) == 1.0
    # 个数不同 ⇒ 算错(哪怕有一个标签是对的、哪怕集合被包含)
    assert label_count_match([["尺码", "退换货"]], [["退换货"]]) == 0.0
    # 那对共享输入上的真实读数:两对三
    assert label_count_match(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)


def test_label_count_match_is_not_subset_accuracy():
    """反向断言:上面那条 `== 1.0` 必须**不能**由 subset_accuracy 冒充。

    没有这一条的话,一个把 `label_count_match` 写成 `return subset_accuracy(...)`
    的实现,在上面那两条用例上**同样成立**。
    """
    assert label_count_match([["尺码", "退换货"]], [["运费", "物流"]]) == 1.0
    assert subset_accuracy([["尺码", "退换货"]], [["运费", "物流"]]) == 0.0


#: 一组 micro 与 macro **不相等**的输入 —— 这是让两个函数互相有判别力的前提。
#: 退换货:tp=2/fp=0/fn=0;尺码:tp=1/fp=1/fn=0;物流:tp=0/fp=0/fn=1。
#: micro 走全局 TP=3/FP=1/FN=1 ⇒ P=R=0.75 ⇒ F1=0.75;
#: macro 走逐类平均 ⇒ (1.0 + 0.6667 + 0.0)/3 = 0.5556 ⇒ **两者必须不同**。
_IMBALANCED_TRUE = [["退换货"], ["退换货"], ["尺码"], ["物流"]]
_IMBALANCED_PRED = [["退换货"], ["退换货"], ["尺码"], ["尺码"]]


def test_micro_f1_is_global_not_per_class_average():
    got = micro_f1(_IMBALANCED_TRUE, _IMBALANCED_PRED, ["退换货", "尺码", "物流"])
    assert got == pytest.approx(0.75)


def test_macro_f1_is_the_unweighted_per_class_average():
    got = macro_f1(_IMBALANCED_TRUE, _IMBALANCED_PRED, ["退换货", "尺码", "物流"])
    assert got == pytest.approx((1.0 + 2 * 0.5 * 1.0 / 1.5 + 0.0) / 3)


def test_micro_and_macro_are_not_the_same_number_here():
    """判别力守卫:在这个输入上两者**必须不同**。

    否则「把 macro_f1 实现成 micro_f1」在全部用例上都不红,而报告里
    「micro 被大类主导、macro 被小类主导」那句就成了一句空话。
    """
    labels = ["退换货", "尺码", "物流"]
    m = micro_f1(_IMBALANCED_TRUE, _IMBALANCED_PRED, labels)
    M = macro_f1(_IMBALANCED_TRUE, _IMBALANCED_PRED, labels)
    assert m != pytest.approx(M)


def test_macro_f1_uses_the_given_labels_as_the_denominator():
    """macro 的分母是**传进来的那张 `labels` 表**,不是「出现过的类」。

    这是 sklearn `f1_score(labels=[...], average="macro")` 的口径,刻意跟它
    对齐:两边都没有的类贡献 0 → macro 被拉低。**这不是缺陷,是 macro 的
    定义**,而 §8.4 那三列并排时它有意义:`只看合成 40` 那一列里必然有几类
    support=0,它们的 0 会把那一列的 macro 压低。

    ⚠️ 所以报告在打出 macro 时必须说明这一条,否则那三个数会被读成
    「模型在合成子集上更差」。(Task 10 的活,这里只把口径钉死。)
    """
    got = macro_f1([["评价"]], [["评价"]], ["评价", "账号"])
    assert got == pytest.approx(0.5)  # 账号 那一类:tp=fp=fn=0 ⇒ F1=0.0


def test_per_class_prf_covers_every_label_in_LABELS_order():
    """行的顺序 = 传进来的 `labels` 的顺序 —— 报告直接按它排版。

    ⚠️ 挑 `["退换货", "发票"]` 是刻意的:按码点 `发票`(U+53D1)< `退换货`(U+9000),
    **与给定顺序不同** ⇒ 一个 `for label in sorted(labels)` 的实现会红。
    随手挑一对同序的(比如 `["物流", "退换货"]`)则两个实现都不红 —— 这条就白写了。
    """
    labels = ["退换货", "发票"]
    assert [r["label"] for r in per_class_prf(Y_TRUE, Y_PRED, labels)] == labels
