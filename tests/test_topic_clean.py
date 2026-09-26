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
    # 修复轮 1(Minor 3):上面两例只有 2~3 位,**只把下限钉到 ≤3**——
    # 把 `\d{8,32}` 改成 `\d{4,32}` / `\d{5,32}` 照样全绿,而那句「下限取小了会误伤语义数字」
    # 的注释因此完全没有东西守。这一例是 7 位(4~7 位里最长的那种)。
    assert redact("订单 1234567 到哪了") == "订单 1234567 到哪了"


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
    `clean` 若哪天开始「顺手修一下」,这条会红 —— **那是设计变更,要改的是 spec**。
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


def test_clean_runs_the_whole_pipeline():
    """修复轮 1(I1):上面几条把 `redact` 与 `normalize` **各自**钉住了,

    却没有一条说 `clean` 真的会调它们 —— 实测:把 `clean` 换成「只 normalize、不脱敏」,
    或者漏掉开头那遍 `normalize`,**9 条照样全绿**(变异 M1 / M2)。
    这条钉的是**组装**:一个输入同时吃到三个阶段 ——
    折全角数字(不折的话 `1[3-9]\\d{9}` 根本匹配不上「１３８００１３８０００」)、
    去重复标点、合并空白、再把手机号换成占位符。
    """
    assert clean("手机１３８００１３８０００,退货!!!  怎么走?") == "手机<手机号>,退货! 怎么走?"


def test_clean_output_is_nfkc_normalized_across_placeholder_boundaries():
    """修复轮 2:`clean` **第二遍** `normalize` 的用途,钉在这里。

    `redact` 插进去的尖括号会与**紧随其后的组合标记**规范组合:
    `<` + U+0338 → U+226E,`>` + U+0338 → U+226F
    (0x110000 全码点穷举,`<` / `>` 各**恰好一个**这样的码点)。
    所以「手机号 + U+0338」洗出来必须是**合成后**的 U+226F 一个码点,
    不是裸的 `>` 加 U+0338 两个码点。

    ⚠️ **这条就该对着「删掉第二遍 `normalize`」判红**(实测过:删掉 ⇒ 红)。
    修复轮 1 曾把那一步当「可证的 no-op」删掉 —— 它的理由来自一次**字母表里
    没有组合记号**的模糊测试,而采样空间(字母表 × 形状 × 长度)决定了结论的适用范围:
    `len <= 6` 的随机串既造不出 11 位手机号,也就撞不上这个边界。

    ⚠️ 组合记号写在源码里是**看不见**的 —— 所以它们一律用 `chr(0x0338)` 这种码点写法,不写裸字符。
    """
    import unicodedata

    mark = chr(0x0338)  # COMBINING LONG SOLIDUS OVERLAY —— 看不见,故用码点写
    gt = chr(0x226F)  # 上面那个记号紧跟 `>` 时合成出来的字

    cases = (
        ("我的手机号是13800138000" + mark + ",发货了吗", "我的手机号是<手机号" + gt + ",发货了吗"),
        ("订单20240915001" + mark + "什么时候到", "订单<订单号" + gt + "什么时候到"),
        ("邮箱 a.b@example.com" + mark + " 能改吗", "邮箱 <邮箱" + gt + " 能改吗"),
    )
    for raw, expected in cases:
        got = clean(raw)
        assert got == expected
        # 真正的性质:输出本身落在 NFKC 规范形里(上面那条钉的是具体文本)
        assert unicodedata.normalize("NFKC", got) == got


def test_both_sides_use_the_same_clean():
    """⚠️ **训练侧与推理侧必须 import 同一个 `clean`。**

    这条用**源码扫描**而不是运行时断言 —— 要抓的是「有没有人写下去」,
    不是「跑到了没有」(照 ch09 `test_no_module_outside_observability_imports_langfuse`
    的先例)。

    漏掉任何一侧的后果:模型在真机上看到的是没洗过的(或洗过两遍的)文本,
    而**没有任何东西会报错** —— 它只让线上准确率悄悄低于测试集读数。

    修复轮 1(Minor 4):原来只认 `from app.topic.clean import … clean` **一种写法**,
    一个**正确**的任务 3 / 任务 13 若写成 `from app.topic import clean` 或
    `import app.topic.clean` 就会被判红 —— 那会逼它们为了变绿而**改自己的 import**,
    正是这条守卫最不该干的事。现在三种惯用写法都放行。
    要守的性质只有一条:**源码里没写这个 import ⇒ 红**(哪怕文件里压根没有这一行)。
    (不管别名:`from app.topic.clean import normalize as clean` 依然会绿 ——
    那是刁难,不是会犯的错;漏掉整个 import 才是。)
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    patterns = (
        re.compile(r"^\s*from\s+app\.topic\.clean\s+import\s+.*\bclean\b", re.M),
        re.compile(r"^\s*from\s+app\.topic\s+import\s+.*\bclean\b", re.M),
        re.compile(r"^\s*import\s+app\.topic\.clean\b", re.M),
    )

    for rel in ("scripts/prepare_topic_data.py", "scripts/classify_topics.py"):
        path = root / rel
        assert path.exists(), f"{rel} 不存在 —— 这条守卫失去了目标,请更新它而不是删掉"
        source = path.read_text(encoding="utf-8")
        assert any(p.search(source) for p in patterns), (
            f"{rel} 没有 import app.topic.clean 的 clean —— train/serve skew 的开始"
            "(接受的写法:`from app.topic.clean import clean` /"
            " `from app.topic import clean` / `import app.topic.clean`)"
        )
