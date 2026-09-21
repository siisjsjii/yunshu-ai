"""意图识别标注样例评估。需真实 key(打网络),不属于单测。

用法:
    .venv/Scripts/python.exe scripts/run_intent_eval.py
报告每条的期望/实际 + 总准确率 + **按标签分组**的准确率 + **confidence 分布**;
**不做字符串断言**,只出数字(deepseek 在 temperature=0 下依然非确定)。

「其他」是本章的验收点之一,**单独统计**(spec §10 验收 2:怪问题落「其他」)。
confidence 分布用来定 `intent_confidence_threshold`(spec §9 的待实测项)。
"""

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.nodes import make_classify_intent_node
from app.config import get_settings
from app.llm import create_extract_model

CASES = Path(__file__).resolve().parents[1] / "evals" / "intent_cases.jsonl"

OTHER = "其他"


def emit(line: str = "") -> None:
    # 控制台是 cp936;`✓`/`✗` 不在 GBK 里,直接 print 会崩。
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        print(line)
        return
    stream.write(line.encode("utf-8") + b"\n")
    stream.flush()


async def main() -> int:
    settings = get_settings()
    node = make_classify_intent_node(model=create_extract_model(settings))
    rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]

    hit = 0
    missing_confidence = 0
    per_label: dict[str, list[int]] = {}      # 期望标签 -> [命中, 总数]
    conf_of: list[float] = []
    conf_hit: list[float] = []
    conf_miss: list[float] = []

    for row in rows:
        got = await node({"user_input": row["text"]})
        ok = got["intent"] == row["expected"]
        hit += ok
        # 缺键**单独计数**:`.get("confidence", 0)` 会把「节点不再出参」伪装成
        # 一片 conf=0.00,而这正是本节要看的分布。
        if "confidence" not in got:
            missing_confidence += 1
        conf = float(got.get("confidence") or 0.0)
        conf_of.append(conf)
        (conf_hit if ok else conf_miss).append(conf)
        bucket = per_label.setdefault(row["expected"], [0, 0])
        bucket[0] += ok
        bucket[1] += 1
        emit(f"{'OK ' if ok else 'MISS'} 期望={row['expected']:<6} "
             f"实际={got['intent']:<6} conf={conf:.2f}  {row['text']}")

    emit()
    emit(f"准确率 {hit}/{len(rows)} = {hit / len(rows):.1%}")
    emit(f"缺 confidence 的行数 {missing_confidence}/{len(rows)}")

    emit()
    emit("按期望标签:")
    for label in sorted(per_label, key=lambda k: (-per_label[k][1], k)):
        got_hit, total = per_label[label]
        emit(f"  {label:<6} {got_hit}/{total}")

    other_hit, other_total = per_label.get(OTHER, [0, 0])
    emit(f"「{OTHER}」专项 {other_hit}/{other_total}")

    emit()
    emit("confidence 分布:")
    emit(f"  全部   min={min(conf_of):.2f} 均值={sum(conf_of) / len(conf_of):.2f} max={max(conf_of):.2f}")
    if conf_hit:
        emit(f"  判对   min={min(conf_hit):.2f} 均值={sum(conf_hit) / len(conf_hit):.2f} max={max(conf_hit):.2f}")
    if conf_miss:
        emit(f"  判错   min={min(conf_miss):.2f} 均值={sum(conf_miss) / len(conf_miss):.2f} max={max(conf_miss):.2f}")
    emit(f"  直方图 {dict(sorted(Counter(round(c, 1) for c in conf_of).items()))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
