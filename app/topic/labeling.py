"""标注链路上的纯函数 —— 全部零 IO、零依赖,故可密集单测。

放在 `app/topic/` 而不是 `scripts/` 里,是因为**它们要被两侧用**:
去重与分层抽样既在离线造数据时跑,也在评测脚本里跑(测试集冻结后的再切分)。
"""

import random

from app.topic.clean import clean


def dedupe_questions(rows: list[dict]) -> list[dict]:
    """按**清洗后的文本**去重,保留每个文本**首次出现**的那一行。

    调用方负责先按来源优先级排序(池子 → 对话 → 测试集.md),
    本函数不做优先级判断 —— 那是策略,这是机制。
    """
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        text = clean(row.get("question", ""))
        if not text or text in seen:
            continue
        seen.add(text)
        out.append({**row, "question": text})
    return out


def validate_evidence(
    question: str, labels: list[str], evidence: dict[str, str]
) -> tuple[list[str], list[str]]:
    """校验每个标签的「证据串」是不是原文的子串。

    返回 `(通过的标签, 被拒的标签)` —— **被拒的也要返回**,不能悄悄吞掉:
    从 3 个标签缩成 1 个会改变训练分布,而那是静默的。

    ⚠️ **空证据串必须显式拒绝**:`"" in "任意文本"` 恒为 True,
    不特判的话 `{"尺码": ""}` 会被判成「证据合法」—— 一个一眼看不见的假绿。

    两侧都过 `clean()`:问句侧与预标喂进 prompt 的文本同口径,
    证据侧则防止全角/空白差异把一条**真的**证据误拒。
    """
    normalized = clean(question)
    accepted, rejected = [], []
    for label in labels:
        ev = (evidence.get(label) or "").strip()
        # 证据本身也要过一遍清洗,否则全角/空白差异会造成假拒。
        if ev and clean(ev) and clean(ev) in normalized:
            accepted.append(label)
        else:
            rejected.append(label)
    return accepted, rejected


def pick_review_sample(
    rows: list[dict], *, per_label: int, seed: int = 20260925
) -> list[dict]:
    """按类分层抽 `per_label` 条,用于人工复核(spec §6.3)。

    **同一条多标签行可能被多个类抽中** —— 用 id 去重后返回,
    所以返回条数**可能少于** `per_label × 类目数`。这是对的:
    它的目的是「每一类都有人看过」,不是「恰好 N 行」。

    种子可注入 ⇒ 同一种子两次结果相同,「我审的是哪几条」可复现。

    ⚠️ **「可复现」的确切含义(实现者据实测订正的措辞,2026-09-26)**:
    它复现的是「**同一份 `rows`(逐行同序) + 同一个 seed**」,不是「同一个语料」。
    两层原因,缺一不成:

    1. **桶的遍历顺序**是各标签**首次出现**的先后(`dict` 保序)⇒ 语料重排会
       把 rng 的同一段随机数分配给**不同**的桶;
    2. **三个桶共用同一个 `rng`** ⇒ 前一个桶的内容/条数一变,
       后面所有桶抽到的东西跟着变。

    ⇒ **「语料重跑过」不等于「抽出的还是同一批」**(别拿这条去断言跨语料的稳定性)。
    真正承重的是「同一份输入可复现」—— 那正是复核 CSV 需要的那一档。

    ⚠️ 实测(2026-09-26,真实 1424 行 `prelabeled.jsonl`,见
    `.superpowers/ch10b_t6_perm_probe.py`):逐行同序 **84** 条;同一份语料**整体倒序**
    后是 **85** 条、与前者只重叠 13 条;随机重排后 83 条、只重叠 4 条。

    ⚠️ **「哪 84 条」这件事本身只记在产物里**:`evals/topic/labels/trainval.csv` **入库**,
    而它的输入 `evals/topic/prelabeled.jsonl` **不入库**(与 `corpus.jsonl` /
    `synthetic.jsonl` 同例,见 `dev-notes/ch10.md`)⇒ **clone 出来是重跑不出那 84 条的**
    (即使重新预标出一模一样的 1424 行,只要行序不同就换一批)。CSV 自己就是那份记录。
    """
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        for label in row.get("labels") or []:
            by_label.setdefault(label, []).append(row)

    picked: dict[str, dict] = {}
    for label, bucket in by_label.items():
        # 桶内先按 id 排序再打乱 ⇒ 抽中谁与**桶内的行序**无关
        # (桶的遍历顺序与共用 rng 的代价见 docstring ⚠️)。
        shuffled = sorted(bucket, key=lambda r: r["id"])
        rng.shuffle(shuffled)
        for row in shuffled[:per_label]:
            picked[row["id"]] = row
    return [picked[k] for k in sorted(picked)]
