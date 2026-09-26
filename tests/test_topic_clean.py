"""清洗:脱敏 + 格式。**不做错别字修正**(ch10 spec §5.1,用户批准的设计变更)。"""

import pytest

from app.topic.clean import clean, normalize, redact


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("我的手机号是13800138000,发货了吗", "我的手机号是<手机号>,发货了吗"),
        ("订单20240915001什么时候到", "订单<订单号>什么时候到"),
        ("邮箱 a.b@example.com 能改吗", "邮箱 <邮箱> 能改吗"),
    ],
)
def test_redact_replaces_identifiers(raw, expected):
    assert redact(raw) == expected


def test_redact_keeps_short_numbers():
    """短数字**不是**标识符:「满99元包邮」「175穿什么码」里的数是语义的一部分。

    把 99 也脱敏掉,「满多少钱包邮」这类问题的判别特征就没了 ——
    而这正是头四类之一的运费类。
    """
    assert redact("满99元包邮") == "满99元包邮"
    assert redact("175穿什么码") == "175穿什么码"


def test_normalize_folds_fullwidth_and_whitespace():
    assert normalize("退货  怎么   走?") == "退货 怎么 走?"
    assert normalize("退货!!!怎么走") == "退货!怎么走"
    assert normalize("　退货　怎么走　") == "退货 怎么走"


def test_normalize_fullwidth_to_halfwidth():
    """全角标点与字母对模型是**不同的 token**,归一化能省下不少表示成本。"""
    assert normalize("ＡＢＣ１２３") == "ABC123"
    assert normalize("退货,怎么走") == "退货,怎么走"


def test_clean_does_not_fix_typos():
    """**这条断的是一个「没做」的决定,不是疏忽。**

    改错别字会造成训练/部署不一致,而且**测试集也会被修过 ⇒ F1 虚高且测不出**。
    错别字保留原样,由增强阶段主动注入(任务 8)。
    `next_clean` 若哪天开始「顺手修一下」,这条会红 —— **那是设计变更,要改的是 spec**。
    """
    assert clean("我要退或,买大 了") == "我要退或,买大 了"


def test_clean_is_idempotent():
    """清洗两次 == 清洗一次。

    不幂等的话,「训练侧洗过一遍、推理侧再洗一遍」会得到不同的文本 ——
    train/serve skew 的一个隐蔽来源。
    """
    for raw in ("我的手机号是13800138000,发货了吗", "退货  怎么   走?", "ＡＢＣ"):
        once = clean(raw)
        assert clean(once) == once


def test_clean_handles_empty_and_whitespace_only():
    assert clean("") == ""
    assert clean("   ") == ""


def test_both_sides_use_the_same_clean():
    """⚠️ **训练侧与推理侧必须 import 同一个 `clean`。**

    这条用**源码扫描**而不是运行时断言 —— 要抓的是「有没有人写下去」,
    不是「跑到了没有」(照 ch09 `test_no_module_outside_observability_imports_langfuse`
    的先例)。

    漏掉任何一侧的后果:模型在真机上看到的是没洗过的(或洗过两遍的)文本,
    而**没有任何东西会报错** —— 它只让线上准确率悄悄低于测试集读数。
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"^\s*from\s+app\.topic\.clean\s+import\s+.*\bclean\b", re.M)

    for rel in ("scripts/prepare_topic_data.py", "scripts/classify_topics.py"):
        path = root / rel
        assert path.exists(), f"{rel} 不存在 —— 这条守卫失去了目标,请更新它而不是删掉"
        assert pattern.search(path.read_text(encoding="utf-8")), (
            f"{rel} 没有 import app.topic.clean.clean —— train/serve skew 的开始"
        )
