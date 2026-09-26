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
