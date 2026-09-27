"""评测指标 —— 纯函数,所以每一处口径都可以钉死。

⚠️ **本文件由 ch10-B Task 9 建**,Task 10 **在上面追加**了一节(没有重建)。

理由:训练脚本的 `compute_metrics` 与 Task 10 的评测报告**必须同源**
(Task 9 计划的那句「同一组函数,不许各写一份」)。Task 9 先跑,所以
`app/topic/metrics.py` 的基础四个函数由 Task 9 落地。

⚠️ **Task 10 已按当时的约定「追加、不重建」做完**(`mislabelled_flow` 的 9 条用例 +
两条**装配处**用例,见文件尾那一节)。重建本文件会把这里 micro/macro-F1 的用例一起抹掉
—— Task 10 的计划里**没有**它们。当年那句「`mislabelled_flow` 今天还不存在 ⇒ ImportError
⇒ 红」已经用掉了,现在这一句只作历史记录读。
"""

import pytest

from app.topic.metrics import (
    label_count_match,
    macro_f1,
    micro_f1,
    mislabelled_flow,
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


def test_unequal_row_counts_raise_instead_of_silently_truncating():
    """⚠️ **订正轮 1 · M4:`zip` 会静默截断。**

    `zip(y_true, y_pred)` 只在较短的那一边上跑,**而分母也没变** ⇒
    指标会算在一个**没人指定的子集**上,数字看起来完全正常。
    三个入口(`per_class_prf` / `subset_accuracy` / `label_count_match`)都要拦。
    """
    with pytest.raises(ValueError, match="行数"):
        per_class_prf([["评价"]], [["评价"], ["评价"]], ["评价"])
    with pytest.raises(ValueError, match="行数"):
        subset_accuracy([["评价"]], [["评价"], ["评价"]])
    with pytest.raises(ValueError, match="行数"):
        label_count_match([["评价"]], [["评价"], ["评价"]])


def test_micro_and_macro_also_raise_on_unequal_row_counts():
    """它们走 `per_class_prf`,所以那条守卫**必须**跟着生效 ——
    只守直调的那三个函数、把 micro/macro 漏掉,是一条看起来严、实际松的写法。"""
    labels = ["评价", "账号"]
    with pytest.raises(ValueError, match="行数"):
        micro_f1([["评价"]], [["评价"], ["评价"]], labels)
    with pytest.raises(ValueError, match="行数"):
        macro_f1([["评价"]], [["评价"], ["评价"]], labels)


def test_per_class_prf_covers_every_label_in_LABELS_order():
    """行的顺序 = 传进来的 `labels` 的顺序 —— 报告直接按它排版。

    ⚠️ 挑 `["退换货", "发票"]` 是刻意的:按码点 `发票`(U+53D1)< `退换货`(U+9000),
    **与给定顺序不同** ⇒ 一个 `for label in sorted(labels)` 的实现会红。
    随手挑一对同序的(比如 `["物流", "退换货"]`)则两个实现都不红 —— 这条就白写了。
    """
    labels = ["退换货", "发票"]
    assert [r["label"] for r in per_class_prf(Y_TRUE, Y_PRED, labels)] == labels


# ------------------------------------------------------- 误判流向矩阵(ch10-B Task 10)


def test_mislabelled_flow_counts_true_to_pred_pairs():
    """**误判流向矩阵**:行=真实标签,列=预测标签,格=「本该是 i 却被判成 j」。

    ⚠️ 它**不是**经典混淆矩阵 —— 多标签没有唯一的预测类。名字刻意不叫混淆矩阵:
    本仓吃过名字与语义不符的亏(`agent_steps` 读作「步数」,实际是轮次序号)。
    """
    flow = mislabelled_flow([["退换货"]], [["运费"]], ["退换货", "运费"])
    assert flow[("退换货", "运费")] == 1
    assert flow[("退换货", "退换货")] == 0  # 判对的**不**进这张矩阵


def test_flow_ignores_pairs_where_the_prediction_is_also_true():
    """真实 `[退换货, 运费]` 预测 `[运费]`:这**不是**误判流向,是漏召回。

    算进去的话矩阵会把「漏了一个」记成「把退换货认成了运费」——
    那会把人引向错误的修法(改边界 vs 提召回)。
    """
    flow = mislabelled_flow([["退换货", "运费"]], [["运费"]], ["退换货", "运费"])
    assert flow.get(("退换货", "运费"), 0) == 0


def test_flow_is_a_dense_labels_by_labels_grid():
    """⚠️ **网格是稠密的**:`labels × labels` 的每一个格子都有键(值可以是 0)。

    上面第一条用例用 `flow[("退换货", "退换货")]` **直接取键**(不是 `.get`)——
    所以「只把命中过的配对放进 dict」的实现会在那里 KeyError,而错误信息读起来
    像「矩阵少了对角的格子」,不像「稠密性没实现」。这条把口径**写明**:
    `matrix_flow.csv` 是一张 17×17 的表,空格子必须印 0。

    顺带钉住「格子只由 `labels` 决定」:真值/预测里出现过的类目**不会**变出额外格子。
    """
    labels = ["退换货", "物流", "尺码"]
    flow = mislabelled_flow([["退换货"]], [["物流"]], labels)
    assert set(flow) == {(i, j) for i in labels for j in labels}
    assert flow[("退换货", "物流")] == 1
    assert sum(flow.values()) == 1  # 其余 8 格都是 0


def test_flow_counts_every_missed_true_label_against_every_wrong_prediction():
    """一条真值 `[退换货, 尺码]` 预测 `[运费]`:**两行都记一次**。

    只记第一个真值(或只记第一个预测)的实现会让这两格静默变成 0 或 1 个 ——
    而矩阵「看起来还是有数」,只是把「多诉求句子被判错时错了几处」少算了一半。
    """
    flow = mislabelled_flow([["退换货", "尺码"]], [["运费"]], ["退换货", "尺码", "运费"])
    assert flow[("退换货", "运费")] == 1
    assert flow[("尺码", "运费")] == 1


def test_flow_does_not_count_a_missed_label_as_a_confusion():
    """真实 `[退换货, 尺码]` 预测 `[尺码]`:`退换货` 是**漏召回**,不是「被认成了尺码」。

    `尺码` 出现在真值里 ⇒ 它不满足 `j ∉ true` 那个条件。少这个条件的实现会把
    这一格记成 1,而「漏了一个」与「认错了」是**两种不同的病**。
    """
    flow = mislabelled_flow([["退换货", "尺码"]], [["尺码"]], ["退换货", "尺码"])
    assert flow[("退换货", "尺码")] == 0
    assert all(v == 0 for v in flow.values()), "漏召回不该在任何格子上留下数"


def test_flow_never_takes_a_correctly_predicted_label_as_a_confusion_source():
    """⚠️ **这条是变异探针 M3 逼出来的,上面那条抓不到那个变异。**

    上面那条(以及 `test_flow_is_all_zero_for_a_perfect_prediction`)的预测**没有多出来的
    标签** ⇒ 配对的内层循环一次都不跑 ⇒ 把 `i ∉ pred` 那个条件去掉**照样全绿**。
    换句话说:那两条断言看着像在守「漏召回不是认错」,其实**只守住了 `j ∉ true` 那一半**。

    这里的输入同时给两边都留了活口:

    * `退换货` 被漏了、`运费` 是凭空多出来的 ⇒ `(退换货, 运费)` 是**真的**误判流向;
    * `尺码` **判对了**(真值预测都有它)⇒ 它**不能**当源头,`(尺码, 运费)` 必须是 0 ——
      否则矩阵会写「本该是尺码、却被判成运费」,而尺码明明判出来了;
    * `尺码` 在真值里 ⇒ 它**也不能**当靶子,`(退换货, 尺码)` 必须是 0。

    ⇒ 整张表只有一个非零格。去掉任一条件,这个 `sum == 1` 都会变。
    """
    flow = mislabelled_flow([["退换货", "尺码"]], [["尺码", "运费"]], ["退换货", "尺码", "运费"])
    assert flow[("退换货", "运费")] == 1  # 唯一真的误判流向
    assert flow[("尺码", "运费")] == 0    # 判对的标签不能当**源头**
    assert flow[("退换货", "尺码")] == 0  # 真值里的标签不能当**靶子**
    assert sum(flow.values()) == 1


def test_flow_accumulates_across_rows():
    """跨行**相加**,不是「出现过就记 1」。"""
    flow = mislabelled_flow(
        [["退换货"], ["退换货"], ["退换货"]],
        [["运费"], ["运费"], ["退换货"]],
        ["退换货", "运费"],
    )
    assert flow[("退换货", "运费")] == 2


def test_flow_is_all_zero_for_a_perfect_prediction():
    """全对 ⇒ 矩阵全 0(判对的行一个格子都不进)。标签顺序不影响结果(集合语义)。"""
    flow = mislabelled_flow([["退换货", "尺码"], ["物流"]], [["尺码", "退换货"], ["物流"]],
                            ["退换货", "尺码", "物流"])
    assert all(v == 0 for v in flow.values())
    assert sum(flow.values()) == 0


def test_flow_raises_on_unequal_row_counts():
    """与其余几个函数同一个守卫(`_aligned`)—— `zip` 静默截断在这里同样致命:
    会在一段**没人指定的子集**上数流向,而格子里的数看起来完全正常。"""
    with pytest.raises(ValueError, match="行数"):
        mislabelled_flow([["评价"]], [["评价"], ["评价"]], ["评价"])


# ---------------------------------── 评测脚本与训练脚本的**装配处**(ch10-B Task 10)


def test_eval_script_and_training_script_decode_logits_identically():
    """⚠️ **钉的是「装配处」,不是两个函数各自的性质。**

    本仓编目过的形态:两个函数各有测试,**它们的装配处没人断言** ⇒ 把两边对调照样绿。
    这里的两边是:

    * `scripts/train_topic_clf.py::metrics_from_logits`(早停 + `train_meta.json` 的
      四指标 + 阈值扫描) —— **它不返回 `y_pred`**,矩阵拿不到;
    * `scripts/eval_topic_clf.py::predict_from_logits`(评测报告要 `y_pred` 画两张矩阵)。

    ⇒ 阈值解码(`sigmoid(logits) >= t`)在仓里**有两处**。判别力全部压在一个格子上:
    真值里有 `退换货`、而它的 logit 恰好是 **0.0**(`sigmoid(0.0) == 0.5` 精确相等),
    于是 `>=` 判它入选、`>` 判它出局 ⇒ 两边的 micro-F1 分别是 0.8 与 0.5。

    ⚠️ 「两处实现」本身**不是**本仓推荐的形状(「不变量要放在唯一写口上」);
    这里没有把解码抽到唯一写口,是因为 `app/topic/metrics.py` 刻意保持**零第三方依赖**
    (B9 的模块 docstring 第一条)。所以退而求其次:**让两处不可能安静地漂开**。
    """
    import numpy as np

    from app.topic.taxonomy import LABELS
    from scripts.eval_topic_clf import predict_from_logits
    from scripts.train_topic_clf import metrics_from_logits

    logits = np.full((2, len(LABELS)), -5.0, dtype="float32")
    logits[0, LABELS.index("退换货")] = 0.0  # sigmoid == 0.5 **恰好** —— 判别力在这一格
    logits[0, LABELS.index("物流")] = 5.0
    logits[1, LABELS.index("尺码")] = 3.0

    y_true = [["退换货"], ["尺码"]]
    label_matrix = np.zeros((2, len(LABELS)), dtype="float32")
    for i, labs in enumerate(y_true):
        for label in labs:
            label_matrix[i, LABELS.index(label)] = 1.0

    for t in (0.5, 0.3):
        y_pred = predict_from_logits(logits, t, list(LABELS))
        got = metrics_from_logits(logits, label_matrix, t)
        for key, want in (
            ("micro_f1", micro_f1(y_true, y_pred, LABELS)),
            ("macro_f1", macro_f1(y_true, y_pred, LABELS)),
            ("subset_accuracy", subset_accuracy(y_true, y_pred)),
            ("label_count_match", label_count_match(y_true, y_pred)),
        ):
            assert got[key] == pytest.approx(want), f"t={t} 的 {key} 两处解码不一致"

    # 顺序也钉住:返回的是 `LABELS` 顺序,不是 `pred` 集合的迭代序。
    assert predict_from_logits(logits, 0.5, list(LABELS))[0] == ["退换货", "物流"]


def test_eval_script_strata_split_the_rows_and_their_label_carries_its_own_count():
    """⚠️ **装配处**:表头「全体 120」与谓词 `lambda r: True` 是**两个分开的东西**。

    表头写死之后,一条「谓词改错、表头照旧」的改动能产出一张**每个数都真、
    只有列名是假的**表 —— 而 §8.4 那张表正是让读的人外推「真机预期表现」的那一列
    (spec 自己写着「只看真实 80 那一列才是」)。

    这里钉两件事:① 三个口径**正好切开**行集(真实 + 合成 == 全体);
    ② 表头里那个数**就是该口径的行数**(现算的,不是串常量)。

    判别力:把 `real` 的谓词写成 `provenance == "synthetic"`,两个口径的行数与
    名字**同时**从 2/1 变成 1/2 ⇒ 两处断言一起红;把表头写死成 `"全体 120"`,
    第二条断言红。
    """
    import numpy as np

    from app.topic.taxonomy import LABELS
    from scripts.eval_topic_clf import STRATA, slice_rows, stratum_label

    rows = [{"provenance": "real"}, {"provenance": "real"}, {"provenance": "synthetic"}]
    y_true = [["退换货"], ["物流"], ["尺码"]]
    logits = np.zeros((3, len(LABELS)), dtype="float32")

    counts, headers = {}, {}
    for key, base, keep in STRATA:
        sub_true, sub_logits = slice_rows(rows, y_true, logits, keep)
        assert len(sub_logits) == len(sub_true), "两个子集必须同源,否则是拿 A 的真值配 B 的预测"
        counts[key] = len(sub_true)
        headers[key] = stratum_label(base, len(sub_true))

    assert counts == {"all": 3, "real": 2, "synthetic": 1}
    assert headers == {"all": "全体 3", "real": "只看真实 2", "synthetic": "只看合成 1"}
    # 切开 —— 与 `main()` 里那条 `SystemExit` 守卫是同一条不变量(守卫在真数据上跑)。
    assert counts["real"] + counts["synthetic"] == counts["all"]
