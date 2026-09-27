"""主题分类的指标 —— **纯函数**。

**谁在用**:`scripts/train_topic_clf.py` 的 `compute_metrics`(早停盯 micro-F1)
与 `scripts/eval_topic_clf.py` 的评测报告(Task 10)。**一处实现,两处 import** ——
本仓记过的那类账(`retrieval_score_threshold` 改一处漏一处)在这里的形态是
「训练早停用一个口径、报告用另一个口径」,而两份数看起来都正常。

**不用 sklearn 的 `confusion_matrix` / `f1_score`**:那些是**单标签**语义,
喂多标签列表会广播成一个巨大的错误结果。`macro_f1` 的平均口径刻意与
sklearn 的 `average="macro"` 对齐(见它的 docstring),但实现是自己的。

## 两条口径纪律

1. **分母为 0 一律返回 `0.0`,不许 `NaN`。** `NaN` 会顺着求和/均值**污染整张表**,
   而报告看起来「有数」。
2. **标签集合由调用方给** —— 顺序就是 `LABELS` 的顺序(报告直接照它排版),
   而**所有类都进表**,包括一条都没出现过的。缺行会让「17 行的表」变成 15 行,
   而读的人只会以为这类不存在。
"""

from __future__ import annotations


def _as_sets(rows) -> list[set]:
    """把「每条的标签列表」变成集合列表。

    集合语义是刻意的:多标签的预测顺序没有意义,`["尺码","退换货"]` 与
    `["退换货","尺码"]` 是同一件事(`subset_accuracy` 有一条用例钉着它)。
    """
    return [set(r) for r in rows]


def _aligned(y_true, y_pred) -> tuple[list[set], list[set]]:
    """真值与预测**对齐**后再变成集合 —— 行数不同就**响亮地抛**。

    ⚠️ **订正轮 1 · M4**:不查行数的话,`zip(y_true, y_pred)` 会**静默截断**到较短的
    那一边,而分母(`len(y_true_sets)`)没变 ⇒ 指标算在一个**没人指定的子集**上,
    数字落在 0–1 之间、看起来完全正常。这是本仓「静默无效」家族的又一名成员。

    ⚠️ 守卫放在这里而不是各函数里,是为了让 `micro_f1` / `macro_f1`(它们走
    `per_class_prf`)**也**跟着生效 —— 本仓那条「不变量要放在唯一写口上」。
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"真值与预测的行数必须相同(现在 {len(y_true)} vs {len(y_pred)})—— "
            "`zip` 会静默截断到较短的那一边,而分母不变"
        )
    return _as_sets(y_true), _as_sets(y_pred)


def per_class_prf(y_true, y_pred, labels) -> list[dict]:
    """每类的支持数、TP/FP/FN 与 P/R/F1 —— §8.1 那张 17 行表的**唯一来源**。

    每行:`{"label", "support", "tp", "fp", "fn", "precision", "recall", "f1"}`。

    - `support` = 真值里有这一类的条数(不是「这一类的样本总数」);
    - `precision` 的分母是「预测过这一类的条数」,`recall` 的分母是 `support`;
      任一为 0 ⇒ 该指标返回 `0.0`(不是 `NaN`,见模块 docstring 第 1 条);
    - `f1` 由 P/R 算,所以 P=R=0 时是 `0.0` —— **不能**写成
      `2*tp/(2*tp+fp+fn)` 之后再单独兜底,那样两种写法在 P=R=0 而 tp>0 时
      给出不同的数(tp>0 时 P、R 不可能都是 0,所以今天不会踩;但口径只能有一个)。
    """
    y_true_sets, y_pred_sets = _aligned(y_true, y_pred)
    rows = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true_sets, y_pred_sets) if label in t and label in p)
        fp = sum(1 for t, p in zip(y_true_sets, y_pred_sets) if label not in t and label in p)
        fn = sum(1 for t, p in zip(y_true_sets, y_pred_sets) if label in t and label not in p)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "label": label,
            "support": tp + fn,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        })
    return rows


def mislabelled_flow(y_true, y_pred, labels) -> dict[tuple[str, str], int]:
    """**误判流向矩阵** —— 行 = 真实标签,列 = 预测标签,格 = 「本该是 i、却被判成 j」的次数。

    spec §8.2 的第 2 张矩阵(`evals/topic/matrix_flow.csv`,17×17)。它**不是**
    经典混淆矩阵 —— 多标签没有唯一的预测类,所以这里**只记「认错」**,不记「漏了」:

    一条样本**只在**同时满足下面两条时贡献格子 `(i, j) *`:

    1. `i ∈ 真值` 且 `i ∉ 预测`  —— i 被漏了;
    2. `j ∈ 预测` 且 `j ∉ 真值`  —— j 是凭空多出来的;
    3. 再多一条:样本里**至少要有**一个 j(否则 `i` 就只是单纯的漏召回)。

    第 2 个条件(即 `j ∉ 真值`)是这张表**唯一容易写错的地方**:去掉它,
    真实 `[退换货, 尺码]` / 预测 `[尺码]` 会被记成「把退换货认成了尺码」——
    而那是**漏召回**,与「认错」是两种不同的病、要两种不同的修法。
    (`tests/test_topic_metrics.py` 里有两条用例从正反两面钉着它。)

    ⚠️ **网格是稠密的**:返回的 dict 恰好有 `len(labels) ** 2` 个键,没命中的格子是 `0`。
    `matrix_flow.csv` 是一张 17×17 的表 —— 空格子必须印 0,而不是缺列。
    附带一个好处:判对的格子(对角线)也**存在且为 0**,读的人不会把「缺格」
    误读成「没统计」。

    ⚠️ **只认 `labels` 里的类目**(与 `per_class_prf` 同款):真值/预测里出现
    `labels` 之外的字符串时**不进任何格子**,也不会变出额外的键。今天不可达
    (预测由 `labels` 解码而来),写在这里是为了不让它将来变成一条静默丢数据的路。

    ⚠️ 一个格子里的数**不是「有几条样本」**:一条样本可以同时给多个格子各加一次
    (真值 `[退换货, 尺码]` 预测 `[运费]` ⇒ `(退换货,运费)` 与 `(尺码,运费)` 各 1)。
    所以行/列之和都**不等于**任何一列的 support —— 别拿它去对 `per_class_prf` 的数。
    """
    y_true_sets, y_pred_sets = _aligned(y_true, y_pred)
    flow = {(i, j): 0 for i in labels for j in labels}
    for true_set, pred_set in zip(y_true_sets, y_pred_sets):
        missed = [i for i in labels if i in true_set and i not in pred_set]
        wrong = [j for j in labels if j in pred_set and j not in true_set]
        for i in missed:
            for j in wrong:
                flow[(i, j)] += 1
    return flow


def subset_accuracy(y_true, y_pred) -> float:
    """**整条完全一致率** —— 标签集合逐条完全相同才算对(最严、最诚实)。

    这是需求原话「一个不多一个不少」的最严读法;字面读法是 `label_count_match`。
    **两个都要报,不要合并**(§6.5)。
    """
    y_true_sets, y_pred_sets = _aligned(y_true, y_pred)
    if not y_true_sets:
        return 0.0
    hit = sum(1 for t, p in zip(y_true_sets, y_pred_sets) if t == p)
    return hit / len(y_true_sets)


def label_count_match(y_true, y_pred) -> float:
    """**标签个数完全一致率** —— 个数对就算对,**内容可以全错**。

    ⚠️ 与 `subset_accuracy` 是两个不同的问题,别合并:那条是集合相等,
    这条只看 `len`。测试里有一条反向断言钉着这一点
    (`label_count_match` 写成 `subset_accuracy` 会红)。
    """
    y_true_sets, y_pred_sets = _aligned(y_true, y_pred)
    if not y_true_sets:
        return 0.0
    hit = sum(1 for t, p in zip(y_true_sets, y_pred_sets) if len(t) == len(p))
    return hit / len(y_true_sets)


def micro_f1(y_true, y_pred, labels) -> float:
    """**micro-F1** —— 全局 TP/FP/FN 汇总后算一次。

    被**大类主导**(17 类里「退换货」那种大类的每一条都进同一个池子)。
    早停盯它(`TrainingArguments.metric_for_best_model="micro_f1"`,spec §7.4):
    验证集每类仅约 7 条,拿抖动极大的 macro 早停等于**让噪声决定什么时候停**。
    """
    rows = per_class_prf(y_true, y_pred, labels)
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def macro_f1(y_true, y_pred, labels) -> float:
    """**macro-F1** —— 每类 F1 的**未加权**平均,被**小类主导**。

    ⚠️ 分母是**传进来的 `labels` 这张表**,不是「出现过的类」—— 这与 sklearn 的
    `f1_score(labels=[...], average="macro")` 一致。于是 §8.4 那三列并排时,
    「只看合成 40」那一列里 support=0 的类会贡献 `0.0`、把那一列的 macro 压低。
    **这不是缺陷,是 macro 的定义** —— 报告里必须写一句,否则那三个数会被读成
    「模型在合成子集上更差」。

    它**不参与任何决策**,只在报告里与 micro 并排(§8.1)。
    """
    rows = per_class_prf(y_true, y_pred, labels)
    if not rows:
        return 0.0
    return sum(r["f1"] for r in rows) / len(rows)
