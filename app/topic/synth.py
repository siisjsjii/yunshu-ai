"""合成数据的**配额与禁区** —— 纯数据 + 纯函数,因为它们是可单测的那一半。

方案 A(用户 2026-09-25 批准)的全部价值在这三件事上:
① **配额**:17 类每类都够;头四类加倍;「其他」不能忘。
② **形态强制**:多标签 ≥30%、近邻边界对 ≥15%。
③ **禁词**:问句里不许出现类目名与边界说明的原词。

三条都不会在训练时暴露问题 —— 它们是**数据纪律**,只能在造数据时守住。

**零 IO、零 LLM、零 DB**:打网络的那一半在 `scripts/gen_topic_data.py` 里。
"""

from app.topic.taxonomy import COUNTER, HEAD_LABELS, LABELS, OTHER

#: 每类配额。头四类加倍。
#: 逐项加起来是 **945**(4×90 + 13×45);而脚本逐 (类, 形态) 拆成
#: `round(QUOTA[label] * share)` 去要,**实际要 962 条**(对数取整的缘故:
#: 头四类 91/类、其余 46/类)。⚠️ 这不是「约 800」—— 本章实测(
#: 2026-09-26)合成产物就是 **962 行**,加上真实语料 462 行 ⇒ **1424 行**。
_BASE = 45
QUOTA: dict[str, int] = {lb: (_BASE * 2 if lb in HEAD_LABELS else _BASE) for lb in LABELS}

FORMS: dict[str, float] = {"single": 0.55, "multi": 0.30, "boundary": 0.15}

#: 三条近邻边界的核心词(spec §4.1,用户给定)。
#:
#: ⚠️ **这四个词是刻意手挑的,不是从 `BOUNDARY` 机械拆出来的** —— 而那是一个
#: 有理由的取舍,别把它读成偷懒。`BOUNDARY` 里被 `**…**` 强调的那些核心词按规则
#: 拆出来会得到「修」「钱」「货」「到手就坏」之类,而**「修」会拦掉「用一年了能修吗」、
#: 「钱」会拦掉「这个多少钱」** —— 那两句都是真实问句(后者就在
#: `tests/test_gen_topic_data.py` 的反向用例里,前者在语料里)。拆词会误伤真话,
#: 而误伤的表现是「合成数据全被丢弃」,看起来像模型没返回。
#:
#: 真正必须与 `taxonomy` 同源的是**类目名**(下面 `set(LABELS)`)—— 加第 18 类时
#: 手写的表会漏,且漏了不报错。那一条由
#: `test_forbidden_words_are_derived_from_taxonomy_not_hand_written` 逐类核对覆盖。
_BOUNDARY_PHRASES = frozenset({"管钱", "管货", "补差价", "券和满减"})


def confusable(label: str) -> str | None:
    """这个类目**最容易混的那个类目** —— 有声明就用声明,没有返回 `None`。

    ⚠️ 订正轮 I1 加的。原来自合成 prompt 里只写「与它最容易混的那个类目同时出现」,
    **让模型自由联想**,而自由联想出来的不是易混而是**共现**:据订正前那一轮的抽查,
    `物流` 的边界批次撞的是 账号 / 评价 / 会员积分。
    ⚠️ 那份证据(订正前的产物)**已在同一轮里被删掉**(append-only,重跑要先删文件),
    所以这条引用**今天复核不了** —— 那次抽查的原文留在
    `.superpowers/sdd/2026-09-25-ecommerce-cs-ch10b-topic-classifier/task-4-report.md` §9.2
    (本机 workspace,与 `sdd/**` 同款不入版本控制)。

    `COUNTER` 是**全章唯一**的「谁跟谁易混」权威源(它逐条写着「这句话归 X,不归 Y」),
    所以这里只用它。它只声明了**一部分**类目(就是它的那些键,别在这里写死个数 ——
    那是又一处会与 taxonomy 漂移的手写清单);没声明的返回 `None`,那一支回退成
    「让模型自己挑」的说法。⚠️ **回退支挑出来的邻居是共现、不是易混**(controller
    2026-09-26 已裁定接受并记账,根因是 spec §4.1 自己留了 7 行 `—`)——
    所以回退支与声明支**不是一个强度**,别把两批混着引用。
    """
    pairs = COUNTER.get(label, ())
    if not pairs:
        return None
    # ⚠️ 只认**第一条**,而这里**响亮地**挡住「有第二条」的情况:今天每个声明的类目
    #    恰好一条反例 ⇒ 无影响;但哪天某个类写下两条且目标不同,第二条会被**静默忽略**
    #    —— 那正是本仓「看起来会生效、什么都没做、还不报错」那一族。
    #    配套 `test_counter_declares_at_most_one_counterexample_per_class`。
    assert len(pairs) == 1, (
        f"{label} 的 COUNTER 里有 {len(pairs)} 条反例,而 `confusable` 只认第一条 —— "
        f"要么收窄成一条,要么把这里改成能表达多邻居的形态"
    )
    return pairs[0][1]


def forbidden_words() -> set[str]:
    """**从 taxonomy 派生**类目名,外加近邻边界的核心词(`_BOUNDARY_PHRASES`)。

    刻意手写一份类目名的话,加了类目就会漏 —— 而漏了不会报错,
    只会让某几类悄悄带上「术语表腔」,在真机上失效。
    """
    words = set(LABELS)
    words |= _BOUNDARY_PHRASES
    # 「以上都不是」是「其他」的说明,不是问句里会出现的东西,排掉。
    words.discard(OTHER)
    return words


def violates_forbidden(question: str) -> list[str]:
    """返回命中的禁词(空列表 = 合规)。**返回明细而不是布尔**,便于报表统计。"""
    return sorted(w for w in forbidden_words() if w in question)


def shape_ok(labels: list[str]) -> bool:
    """多标签形态检查:非空、不重复、≤3 个、都是合法类目。"""
    if not labels or len(labels) > 3:
        return False
    if len(set(labels)) != len(labels):
        return False
    return all(lb in LABELS for lb in labels)
