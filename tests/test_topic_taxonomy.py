"""17 类权威表:它是**全章的唯一类目来源**,所以这里断的是它的形状与自洽。"""

import csv
import io
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
from scripts.export_taxonomy_review import (
    FOOTNOTE,
    HEADER,
    OUT as CSV_OUT,
    render_csv_text,
)

EXPECTED = (
    "退换货", "物流", "尺码", "发票", "质量问题", "运费", "优惠活动", "价保",
    "支付", "订单修改", "库存补货", "商品信息", "保修维修", "账号", "会员积分",
    "评价", "其他",
)

#: spec §4.1 那 10 行**有**反例的类目 —— 是**故意双写**的清单(与 `EXPECTED` 同理):
#: 正本在 spec §4.1,这里是它的代码化看守。
COUNTER_KEYS = (
    "退换货", "物流", "尺码", "质量问题", "运费",
    "优惠活动", "价保", "订单修改", "商品信息", "保修维修",
)

#: spec §4.1 反例列写 `—` 的那 7 行 —— 它们**不许**有 `COUNTER` 条目。
COUNTERLESS = ("发票", "支付", "库存补货", "账号", "会员积分", "评价", "其他")


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


def test_counter_key_set_is_exactly_the_rows_spec_gives_a_counter():
    """**反例表的键集本身是契约** —— 删掉一条(如 `COUNTER["价保"]`)不会红任何别的断言,
    而它删的是**用户刚签字的那份内容**。这里把两个方向都钉住:

    - 10 行有反例的必须有条目(不许少);
    - 7 行反例列写 `—` 的必须没有条目(不许多 —— 凭空加一条反例,
      等于在没有设计依据的情况下给某个类目划边界)。
    """
    assert set(COUNTER) == set(COUNTER_KEYS), (
        f"COUNTER 的键集与 spec §4.1 不符:多了 {set(COUNTER) - set(COUNTER_KEYS)}、"
        f"少了 {set(COUNTER_KEYS) - set(COUNTER)}"
    )
    assert not (set(COUNTER) & set(COUNTERLESS)), (
        f"spec §4.1 的反例列写的是 `—`:{set(COUNTER) & set(COUNTERLESS)}"
    )
    assert len(COUNTER_KEYS) == 10 and len(COUNTERLESS) == 7
    assert set(COUNTER_KEYS) | set(COUNTERLESS) == set(LABELS), (
        "17 类必须被「有反例 / 无反例」完整划分 —— 新增类目时要一并决定它属于哪一侧"
    )


def test_rendered_block_carries_every_label_and_boundary():
    """渲染块是 prompt 的**唯一**类目来源;漏一类,那个类就永远训不出来。"""
    block = render_taxonomy_for_prompt()
    for label in LABELS:
        assert label in block, f"渲染块里没有 {label}"
        if label != OTHER:
            assert BOUNDARY[label] in block, f"渲染块里没有 {label} 的边界说明"
    assert block.count("\n") >= len(LABELS)


def test_rendered_block_carries_every_counter_pair():
    """反例必须**逐条进渲染块** —— 只断类目名与边界说明的话,一个「把反例行整段丢掉」
    的渲染会**通过全部断言**,而 `render_taxonomy_for_prompt` 的 docstring 明说
    它渲染「边界说明、正例、反例」三样,近邻类目的裁决内容**就是那些反例**。

    这里断的是整行文本(而不是「两个词都出现过」)—— 指向与归属任一写错都会红。
    """
    block = render_taxonomy_for_prompt()
    for label, pairs in COUNTER.items():
        for text, target in pairs:
            line = f"「{text}」归 {target},不归 {label}"
            assert line in block, f"渲染块里没有反例行:{line}"


def test_rendered_block_carries_every_positive_example():
    """正例必须**逐条进渲染块** —— 与上面那条反例守卫同款(计划订正 D,2026-09-26)。

    `POSITIVE` 按 spec §4.1 的「正例」列逐类收齐(上面 `test_every_label_has_positive_examples`
    守住了「每类 ≥1 条」),**但它此前一个 prompt 都没进过** —— 而 `POSITIVE` 自己的
    docstring 明说「边界说明**不足以**让模型学会认它(其他),给一句原话比给一句否定式更有效」。
    ⇒ 缺这条断言的话,「把正例行整段丢掉」的渲染会**通过其余全部断言**:
    类目名在、边界说明在、反例在,而正例静默消失。

    这里断的是**整行文本**(含与反例行对齐的四空格缩进),而不是「那个词出现过」——
    少写 `例:` 前缀、或缩进写歪都会红。
    """
    block = render_taxonomy_for_prompt()
    for label, examples in POSITIVE.items():
        assert examples, f"{label} 在 POSITIVE 里是空的,这条断言对它零判别力"
        for text in examples:
            line = f"    · 例:「{text}」"
            assert line in block, f"渲染块里没有正例行:{line}"


def test_rendered_block_has_no_curly_braces():
    """`ChatPromptTemplate` 按 f-string 解析,裸花括号会炸(本仓硬约束)。

    实测报错长这样:`Single '}' is not allowed for a for loop` —— 报错位置指向
    prompt 组装那一行,与「类目表里有个花括号」毫无关系。
    """
    block = render_taxonomy_for_prompt()
    assert "{" not in block and "}" not in block


def test_committed_csv_is_in_sync_with_the_constants():
    """盘上那份 CSV 必须**逐字节**等于用常量现算出来的那份(含编码与 BOM)。

    它是本章**唯一一份由人签字**的产物(CP-1),而且**与源码同处一地入库** ——
    改了 `BOUNDARY` / `COUNTER` 而忘了重跑导出脚本的话,用户签过的那张表
    会**静默地**与代码不一致:没有任何东西会红,而后面所有预标与合成
    都照那张表走。

    比较走的是**同一条行产出路径**(`render_csv_text`),所以这条断言
    断的不是「文件存在」,而是「文件 == f(常量)」。
    """
    on_disk = CSV_OUT.read_bytes()
    assert on_disk == render_csv_text().encode("utf-8-sig"), (
        f"{CSV_OUT} 与 `taxonomy.py` 的常量不一致 —— "
        "重跑一遍 `.venv/Scripts/python.exe scripts/export_taxonomy_review.py` 再提交"
    )
    # BOM 单列一条:这是给中文 Windows 上的 Excel 看的文件,少了它中文全乱码,
    # 而「乱码」是用户侧的现象,不是任何别的断言能照到的。
    assert on_disk[:3] == b"\xef\xbb\xbf", "CSV 少了 UTF-8 BOM"

    rows = list(csv.reader(io.StringIO(on_disk.decode("utf-8-sig"))))
    assert rows[0] == HEADER, "表头被动了"
    data = [r for r in rows[1:] if r and r[0].isdigit()]
    assert [r[1] for r in data] == list(LABELS), "17 行数据被动了(条数或顺序)"
    assert rows[-1] == [FOOTNOTE], "尾注不在最后一行 —— 它不能被读成第 18 类"


def test_csv_footnote_answers_cp1_item_three():
    """spec §4.2 说「必须写进表里」——「表」是**用户真正打开的那份文件**。

    这条单独钉尾注的**内容**(而不只是「有一行尾注」):CP-1 的第 ③ 问
    就是「17 类里没有投诉/闲聊/转人工你是否认可」,尾注必须自己答得了它。
    """
    for token in ("投诉", "闲聊", "转人工", "意图", "主题", "§4.2"):
        assert token in FOOTNOTE, f"尾注里没有 {token},回答不了 CP-1 的第 ③ 问"
