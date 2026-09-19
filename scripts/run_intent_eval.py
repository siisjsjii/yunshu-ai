"""意图识别标注样例评估。需真实 key(打网络),不属于单测。

用法:
    .venv/Scripts/python.exe scripts/run_intent_eval.py
报告每条的期望/实际与总准确率;**不做字符串断言**,只出数字
(deepseek 在 temperature=0 下依然非确定)。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.nodes import make_classify_intent_node
from app.config import get_settings
from app.llm import create_extract_model

CASES = Path(__file__).resolve().parents[1] / "evals" / "intent_cases.jsonl"


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
    for row in rows:
        got = (await node({"user_input": row["text"]}))["intent"]
        ok = got == row["expected"]
        hit += ok
        emit(f"{'OK ' if ok else 'MISS'} 期望={row['expected']:<6} 实际={got:<6} {row['text']}")

    emit()
    emit(f"准确率 {hit}/{len(rows)} = {hit / len(rows):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
