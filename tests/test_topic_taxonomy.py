"""17 类权威表:它是**全章的唯一类目来源**,所以这里断的是它的形状与自洽。"""

import re

import pytest

from app.agent.routing import INTENT_LABELS
from app.topic.taxonomy import (
    BOUNDARY,
    COUNTER,
    HEAD_LABELS,
    INTENT_TO_TOPICS,
    LABELS,
    OTHER,
    POSITIVE,
    render_taxonomy_for_prompt,
)

EXPECTED = (
    "退换货", "物流", "尺码", "发票", "质量问题", "运费", "优惠活动", "价保",
    "支付", "订单修改", "库存补货", "商品信息", "保修维修", "账号", "会员积分",
    "评价", "其他",
)


def test_labels_are_exactly_the_seventeen():
    """顺序也是契约 —— 训练时它是标签 id,推理时它必须逐位相同。"""
    assert LABELS == EXPECTED
    assert len(set(LABELS)) == 17


def test_head_labels_lead_the_table():
    """「退换货、物流、尺码、发票四大类领头」—— 是**位置**,不只是说法。

    顺序错了不会报错,但 `LABELS[:4]` 会被别处当成头四类用(测试集配额、
    报告里的分区),于是「头四类样本更足」这条保证静默失效。
    """
    assert HEAD_LABELS == ("退换货", "物流", "尺码", "发票")
    assert LABELS[:4] == HEAD_LABELS


@pytest.mark.parametrize("label", EXPECTED)
def test_every_label_has_a_boundary_sentence(label):
    """每类都要有边界说明 —— 少一句,近邻类目就靠模型自由发挥。"""
    assert BOUNDARY.get(label, "").strip(), f"{label} 缺边界说明"


@pytest.mark.parametrize("label", EXPECTED)
def test_every_label_has_positive_examples(label):
    if label == OTHER:
        pytest.skip("「其他」的边界是「以上都不是」,不要求正例")
    assert len(POSITIVE.get(label, ())) >= 1, f"{label} 缺正例"


def test_counter_examples_point_at_real_labels():
    """反例必须指向一个**真实存在的**类目 —— 指错了就没人能照着裁决。"""
    for label, pairs in COUNTER.items():
        assert label in LABELS, f"反例表的键 {label} 不是合法类目"
        for text, target in pairs:
            assert target in LABELS, f"{label} 的反例「{text}」指向了不存在的类目 {target}"
            assert target != label, f"{label} 的反例指向了自己"


def test_near_neighbour_boundaries_are_pinned():
    """用户给的三条近邻边界必须**逐条**在表里,且有反例钉着。

    这三条是「近邻类目靠边界说明划开」的全部内容。少了任何一条,
    模型只能靠字面词猜,而这三对恰恰是字面无差别、只有语义差别的。
    """
    assert "保修维修" in COUNTER and any(
        t == "保修维修" for _, t in COUNTER["退换货"]
    ), "「修归保修维修」这条边界没钉住"
    assert "运费" in COUNTER and any(t == "运费" for _, t in COUNTER["物流"]), (
        "「运费管钱、物流管货」这条边界没钉住"
    )
    assert "价保" in COUNTER and any(t == "价保" for _, t in COUNTER["优惠活动"]), (
        "「价保是补差价、优惠活动是券和满减」这条边界没钉住"
    )


def test_intent_labels_are_all_mapped():
    """⚠️ **加意图就必须加映射** —— 这条是任务 A1 加第九类时那条「响亮地失败」的兑现。

    没有它,`INTENT_TO_TOPICS` 会静默少一行,而下游(分布页的意图口径统计)
    只会少一个键,不报错。
    """
    missing = [label for label in INTENT_LABELS if label not in INTENT_TO_TOPICS]
    assert not missing, f"这些意图没有映射到主题:{missing}"


def test_mapping_targets_are_all_real_labels():
    for intent, topics in INTENT_TO_TOPICS.items():
        assert topics, f"{intent} 映射到了空集"
        for t in topics:
            assert t in LABELS, f"{intent} 映射到了不存在的主题 {t}"


def test_non_topical_intents_land_on_other():
    """投诉 / 闲聊 / 转人工**不是主题** —— 它们只能落「其他」。

    这是刻意的(见 spec §4.2),不是漏写。**两个词表的定位差异是设计的一部分**:
    意图答的是「用户想干什么」,主题答的是「问题关于什么」。
    """
    for intent in ("投诉", "闲聊", OTHER, "转人工"):
        assert INTENT_TO_TOPICS[intent] == (OTHER,), (
            f"{intent} 应当映射到「{OTHER}」—— 若这是有意改动,请同时改这条测试与 spec §4.2"
        )


def test_rendered_block_carries_every_label_and_boundary():
    """渲染块是 prompt 的**唯一**类目来源;漏一类,那个类就永远训不出来。"""
    block = render_taxonomy_for_prompt()
    for label in LABELS:
        assert label in block, f"渲染块里没有 {label}"
        if label != OTHER:
            assert BOUNDARY[label] in block, f"渲染块里没有 {label} 的边界说明"
    assert block.count("\n") >= len(LABELS)


def test_rendered_block_has_no_curly_braces():
    """`ChatPromptTemplate` 按 f-string 解析,裸花括号会炸(本仓硬约束)。

    实测报错长这样:`Single '}' is not allowed for a for loop` —— 报错位置指向
    prompt 组装那一行,与「类目表里有个花括号」毫无关系。
    """
    block = render_taxonomy_for_prompt()
    assert "{" not in block and "}" not in block
