"""标注链路上的纯函数 —— 全部零 IO、零依赖,故可密集单测。

放在 `app/topic/` 而不是 `scripts/` 里,是因为**它们要被两侧用**:
去重与分层抽样既在离线造数据时跑,也在评测脚本里跑(测试集冻结后的再切分)。
"""

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
