"""合成数据的**形态与禁区**检查 —— 全是纯函数,不打网络。"""

import pytest

from app.topic.synth import (
    FORMS,
    QUOTA,
    forbidden_words,
    shape_ok,
    violates_forbidden,
)
from app.topic.taxonomy import HEAD_LABELS, LABELS, OTHER


def test_quota_covers_every_label():
    assert set(QUOTA) == set(LABELS)


def test_head_labels_get_a_bigger_quota():
    """「四大类领头」在**配额上**也要是真的,不能只是术语表里的说法。"""
    others = [q for lb, q in QUOTA.items() if lb not in HEAD_LABELS]
    assert all(QUOTA[lb] >= max(others) for lb in HEAD_LABELS), (
        "头四类的配额必须不低于其余类的最大配额"
    )


def test_other_class_has_a_quota():
    """「其他」**要留配额** —— 语料里它不会少,而合成时最容易忘。"""
    assert QUOTA["其他"] > 0


def test_multi_label_share_is_at_least_thirty_percent():
    """多标签占比 ≥30% 是**方案 A 的全部价值所在**。

    少了它,模型退化成「只报最显眼的那个类」,验收 ③
    (「买大了想退」同时命中多类)从原理上过不了。
    """
    assert FORMS["multi"] >= 0.30


def test_boundary_share_is_at_least_fifteen_percent():
    """近邻边界对 ≥15% —— 少了它,`运费↔物流`、`退换货↔保修维修` 学不开。"""
    assert FORMS["boundary"] >= 0.15


def test_forms_sum_to_one():
    assert abs(sum(FORMS.values()) - 1.0) < 1e-9


@pytest.mark.parametrize(
    "question",
    [
        "保修维修怎么办",  # 类目名
        "我想问退换货的事",  # 类目名
        "关于质量问题",  # 类目名
        "运费管钱物流管货",  # 边界说明的原词
        "价保是补差价",  # 边界说明的原词
    ],
)
def test_forbidden_words_catch_label_names_and_boundary_phrases(question):
    """**禁词**:合成的问句里不许出现类目名或边界说明的原词。

    不设这条,模型学到的就是**关键词匹配** —— F1 会很漂亮,
    而真机上一句不含类目名的真话就废了。这是本章最贵的一条数据纪律。
    """
    assert violates_forbidden(question), f"「{question}」应当命中禁词"


@pytest.mark.parametrize(
    "question",
    ["买大了想退", "快递到哪了", "能开专票吗", "满多少包邮", "用一年了能修吗"],
)
def test_real_questions_pass_the_filter(question):
    """**反向**:真实用户话本来就不含类目名 —— 这条防止禁词表宽到把真话也拦了。

    禁词表写宽了的后果是**合成数据全被丢弃**,而执行者只看到「生成了 0 条」,
    容易误以为是模型没返回,而不是自己的过滤器写坏了。
    """
    assert violates_forbidden(question) == []


def test_forbidden_words_are_derived_from_taxonomy_not_hand_written():
    """禁词表必须**从 `taxonomy` 派生**,不能手写一份。

    手写的会与类目表漂移:加了第 18 类而忘了加进禁词表,
    合成数据里就会出现类目名的原词 —— 静默的,只有真机才暴露。
    """
    words = forbidden_words()
    assert "退换货" in words and "保修维修" in words
    # ⚠️ 上面那两条**手写一张两个词的表就能通过**(它们只能证明「有两个词」)。
    #    这条测试的名字承诺的是「派生」,所以要真去比一遍**覆盖率**:
    #    除「其他」(它的说明是「以上都不是」,不是问句里会出现的东西)之外,
    #    **每一个类目名都必须在禁词表里** —— 少一个就红,这正是「加了第 18 类
    #    而忘了加进禁词表」那条漂移的判别式。
    missing = set(LABELS) - {OTHER} - words
    assert not missing, f"这些类目名没有进禁词表(手写表与 taxonomy 漂移了):{sorted(missing)}"


def test_other_is_not_a_forbidden_word():
    """「其他」**不是禁词** —— 它的边界说明是「以上都不是」,不是问句里会出现的东西。

    `forbidden_words()` 从 `set(LABELS)` 起手,所以「其他」默认**在表里**,
    靠一行 `words.discard(OTHER)` 排掉。少了那一行不会有任何东西红:
    合成出来的问句本来就极少含「其他」两字 —— 也就是说这条是**零可观测后果**的
    静默差别,只能靠这条测试钉住。
    """
    assert OTHER not in forbidden_words()


def test_forbidden_words_do_not_swallow_ordinary_questions():
    """禁词表不许宽到把**真实问句**也拦了。

    这条是上一条的反面。禁词表写宽了的表现是「合成数据全被丢弃」,
    而执行者只会看到「生成了 0 条」,很容易误判成模型没返回。

    ⚠️ 诚实记一笔:这一条**在 `forbidden_words()` 返回空集时恒真** ——
    单看它没有判别力。它守的是「别写宽」这一个方向,另一个方向由上面
    `catch_label_names_and_boundary_phrases` 那条守;两条**合起来**才有判别力。
    """
    for q in (
        "这个多少钱",
        "什么时候能到",
        "能不能开票",
        "我要改成另一个地址",
        "这个颜色还有别的吗",
    ):
        assert violates_forbidden(q) == [], f"「{q}」是正常问句,不该被禁词表拦住"


def test_shape_ok_accepts_single_and_multi():
    assert shape_ok(["尺码"])
    assert shape_ok(["尺码", "退换货"])
    assert shape_ok(["尺码", "退换货", "运费"])


def test_prompt_has_a_literal_JSON_and_no_bare_braces():
    """prompt 的两条本仓硬约束 —— 由**整段核**守住,不是靠看干跑输出。

    硬约束:prompt 里必须出现字面 `JSON`(`json_mode` 要求),
    且**不得有裸花括号**(`ChatPromptTemplate` 按 f-string 解析,一个 `{` 就炸)。

    ⚠️ 为什么不能用 `--dry-run` 代替这条:干跑把每批 prompt 截到 400 字
    (`prompt[:400]`),而完整的类目表有 17 行、远超 400 字 ⇒ **后十几个类目的文本
    在干跑输出里压根看不到**。只靠肉眼的话,「后面的类目里有没有花括号」是
    **没被看过**的 —— 这正是本仓「证据看起来齐了」那一类假绿。
    """
    from scripts.gen_topic_data import batch_size, build_prompt

    for label in QUOTA:
        for form in FORMS:
            prompt = build_prompt(label, form, batch_size(label, form))
            assert "JSON" in prompt, f"{label}/{form} 的 prompt 里没有字面 JSON(json_mode 硬约束)"
            assert "{" not in prompt and "}" not in prompt, (
                f"{label}/{form} 的 prompt 里有裸花括号 —— `ChatPromptTemplate` 那条路会炸"
            )


def test_requested_batch_sizes_realise_the_quota_and_the_form_shares():
    """**要出去的条数**必须兑现 `QUOTA` 与 `FORMS` —— 不然那两组只是没人读的常量。

    ⚠️ 为什么非有这条不可:上面所有形态测试读的都是**常量**
    (`FORMS["multi"] >= 0.30`),而一个**无视 `FORMS`**、每批固定要 10 条的生成器
    照样能通过它们全部 —— 常量对、生成器不读,一样能「全绿而数据没有形态」。
    这条把算式(`batch_size`)与比例绑在一起:逐 (类, 形态) 加总后,
    **要出去的那批**必须同时满足「够配额」与「multi / boundary 的占比不塌」。
    """
    from scripts.gen_topic_data import batch_size

    for label in QUOTA:
        counts = {form: batch_size(label, form) for form in FORMS}
        total = sum(counts.values())
        assert total >= QUOTA[label], f"{label} 一批只要 {total} 条,少于配额 {QUOTA[label]}"
        # 取整会带来 ±1 的偏移(头四类 single 那批是 50 而不是 49.5)⇒ 留 2 个百分点的容差。
        # 但「multi 掉到 10%」「boundary 没了」这种偷换一定会红。
        assert counts["multi"] / total >= 0.30 - 0.02, f"{label}: multi 只占 {counts['multi'] / total:.1%}"
        assert counts["boundary"] / total >= 0.15 - 0.02, (
            f"{label}: boundary 只占 {counts['boundary'] / total:.1%}"
        )


def test_shape_ok_rejects_empty_duplicates_and_overlong():
    assert not shape_ok([]), "没有标签的样本没有训练价值"
    assert not shape_ok(["尺码", "尺码"]), "重复标签是标注错误"
    assert not shape_ok(["尺码", "退换货", "运费", "价保"]), "4 个诉求超出本章的形态假设"
    # ⚠️ 上面三条各只钉住一个拒绝理由(空 / 重复 / 超长),而 `shape_ok` 的**第四个**
    #    理由是「标签必须是真类目」—— 不补这一条,把那个 `all(lb in LABELS …)`
    #    整句删掉,上面三条**全都照样绿**(模型跑偏吐出个自造标签时会静默入库)。
    assert not shape_ok(["尺码", "不存在的类目"]), "表外的标签必须拒掉"
