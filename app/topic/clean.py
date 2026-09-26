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
    """`normalize` → `redact`,**两步**,顺序不能反。

    ① `normalize` 先做格式归一 —— 这也是**全角数字能被脱敏的前提**:
       「１３８００１３８０００」不先折成半角,`1[3-9]\\d{9}` / `\\d{8,32}` 都匹配不上。
    ② `redact` 再把手机号 / 订单号 / 邮箱换成占位符。

    ⚠️ **这里原先还有第三遍 `normalize`**(`normalize(redact(normalize(text)))`),
    **修复轮 1 删掉了 —— 它是可证的 no-op**。理由:`redact` 的三个占位符非空、NFKC 稳定、
    不含空白、也不含 `_PUNCT_RUN` 盯的 `!?,.;:` 里任何字符 ⇒ 它既造不出新的空白连写、
    也造不出新的标点连写,末尾那遍没有东西可做(实测:12 万条随机串 + 2 万条真实形态拼接
    + 16,105 条小字母表穷举,两种实现的输出**差异 0 条**)。
    原先挂在它上面的那条理由(「少了第二遍,全角数字永远脱不了敏」)也是**错的**:
    折全角的是**第一遍** `normalize`;少了第一遍的后果不是不脱敏,而是脱成**另一个占位符**
    (`<订单号>` —— Python 的 `\\d` 认全角数字,而末尾那遍会把全角折成半角后才轮到它)。
    """
    return redact(normalize(text))
