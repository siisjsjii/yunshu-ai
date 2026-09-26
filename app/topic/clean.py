"""主题分类语料的清洗 —— **训练侧与推理侧的唯一实现**。

⚠️ **这个模块必须被两侧同时 import**:
`scripts/prepare_topic_data.py`(训练语料)与 `scripts/classify_topics.py`(推理)。
只在一侧清洗 = 经典的 train/serve skew:模型在真机上看到的是另一种文本。
`tests/test_topic_clean.py::test_both_sides_use_the_same_clean` 守着这条。

**刻意不做错别字修正**(用户 2026-09-25 批准)。理由:
① 改干净会让训练语料比真实用户输入更规整,而**测试集也会被同样修过**
   ⇒ 指标虚高,且这个下降在指标上不可见;
② 修正本身会引入幻觉(「起球」→「气球」),静默污染全部数据;
③ 这件事应该反过来做 —— 增强阶段**主动注入**错别字(见
   `scripts/prepare_topic_data.py` 的 augment 子命令)。
"""

import re
import unicodedata

#: 手机号。**必须带 `1[3-9]` 前缀**,不能写成 `\d{11}` ——
#: 后者会把「20240915001」这类订单号也吃掉一半。
_PHONE = re.compile(r"1[3-9]\d{9}")

#: 订单号 / 长数字串。**下限取 8**:实测订单号形如 `20240915001`(11 位),
#: 而「满99元」「175码」这类语义数字都短于 8 位。下限取小了会误伤语义数字。
_LONG_NUMBER = re.compile(r"\d{8,32}")

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_PUNCT_RUN = re.compile(r"([!?,.;:])\1+")
_WS_RUN = re.compile(r"\s+")


def redact(text: str) -> str:
    """脱敏:手机号 / 订单号 / 邮箱 → 占位符。

    顺序有意义:**先手机号,再长数字**。反过来的话手机号会先被
    `<订单号>` 吃掉(它也是 8 位以上的数字串)。
    """
    text = _PHONE.sub("<手机号>", text)
    text = _EMAIL.sub("<邮箱>", text)
    return _LONG_NUMBER.sub("<订单号>", text)


def normalize(text: str) -> str:
    """格式归一:全角→半角、合并空白、去重复标点、去首尾空白。"""
    # NFKC 把全角字母数字标点折成半角(ＡＢＣ１２３ → ABC123)。
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT_RUN.sub(r"\1", text)
    text = _WS_RUN.sub(" ", text)
    return text.strip()


def clean(text: str) -> str:
    """`normalize` → `redact` → `normalize`。三步各自为什么:

    - **第一个 `normalize`**:先做格式归一(折全角、并空白、去重复标点、去首尾空白),
      后面的正则才谈得上匹配。注意 **Python 的 `\\d` 是 Unicode 感知的** ——
      「１２３４５６７８」这种全角数字串**也**满足 `\\d{8,32}`;而 `1[3-9]\\d{9}` 的
      前导 `1` 是**半角字形**,全角的「１３８００１３８０００」匹配不上。
      所以少了这一遍,全角手机号**不是**「脱不了敏」,而是脱成**另一个占位符** `<订单号>`。
    - **`redact`**:把手机号 / 订单号 / 邮箱换成占位符。
    - **第二个 `normalize`(⚠️ 不是装饰)**:`redact` **插进去的占位符在上下文里不是
      NFKC 稳定的** —— 尖括号会与**紧随其后的组合标记**规范组合:
      `<` + U+0338 → U+226E ≮,`>` + U+0338 → U+226F ≯。
      所以这一遍保证的是「**输出一定是 NFKC 规范形式**」,而不只是「清理插入产生的空白」。

    ⚠️ **这一步曾被当成「可证的 no-op」删掉过(修复轮 1),是实测把它纠正回来的**:
    0x110000 全码点穷举,`<` / `>` 与后随码点**合成成一个字**的情况各**恰好一个** ——
    U+0338;而它**可达**(手机号 / 订单号 / 邮箱后面紧跟 U+0338 时,两步版与三步版输出不同)。
    修复轮 1 之所以误判,是因为那次的模糊测试字母表里根本没有组合记号,
    且采样形状(长度 ≤ 6 的短串)也造不出「11 位手机号 + 紧邻的组合记号」。
    **别再把这一步当冗余删掉。**
    """
    return normalize(redact(normalize(text)))
