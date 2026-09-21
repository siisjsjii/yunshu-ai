"""指代消解 / Query 改写标注样例评估。需真实 key(打网络),不属于单测。

用法:
    .venv/Scripts/python.exe scripts/run_resolve_eval.py
    .venv/Scripts/python.exe scripts/run_resolve_eval.py --model chat

口径:`must_contain` 是**闭式**的 —— 只要求改写结果里出现该关键词,**不做字符串全等**
(模型在 temperature=0 下依然非确定,本仓已记账)。走的是**生产节点**
`make_resolve_references_node`,不是另抄一份 prompt 调用。

**每条都打两个标签**(比准确率本身重要):
- `透传` / `改写`:输出与用户原话是否一字未动 —— 反面提醒是「全都透传」也能拿 100%;
- `[透传即可满足]`:该条的 `must_contain` **在本轮原话里就有**(如本来就完整的那条),
  所以它对「有没有真的改写」**零判别力** —— 它守的是另一件事(别把完整的问题改写坏)。

`--model extract`(默认)用 `create_extract_model`(温度 0),与
`run_intent_eval.py` 同源、可复跑;`--model chat` 用 `create_chat_model`(温度 0.7),
**即生产图给消解节点的那一个**(`app/agent/graph.py` 传的是主力模型)。
两者温度不同,数字不可混用。
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.nodes import make_resolve_references_node
from app.config import get_settings
from app.llm import create_chat_model, create_extract_model
from app.schemas import Message

CASES = Path(__file__).resolve().parents[1] / "evals" / "resolve_cases.jsonl"


def emit(line: str = "") -> None:
    # 控制台是 cp936;改写结果是中文,一律走字节出口(本仓平台陷阱)。
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        print(line)
        return
    stream.write(line.encode("utf-8") + b"\n")
    stream.flush()


def _history(rows: list[dict]) -> list[Message]:
    return [Message(role=r["role"], content=r["content"]) for r in rows]


async def main() -> int:
    parser = argparse.ArgumentParser(description="指代消解评估集")
    parser.add_argument(
        "--model",
        choices=("extract", "chat"),
        default="extract",
        help="extract=温度 0(默认,可复跑);chat=生产的那个模型(温度 0.7)",
    )
    args = parser.parse_args()

    settings = get_settings()
    model = (
        create_chat_model(settings) if args.model == "chat"
        else create_extract_model(settings)
    )
    node = make_resolve_references_node(model=model)

    rows = [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:                      # 空文件不许 ZeroDivisionError
        emit(f"用例文件是空的:{CASES}")
        return 1

    emit(f"模型 {settings.openai_model} / {args.model} / 用例 {len(rows)} 条")
    emit()

    hit = 0
    passed_through = 0
    satisfiable_by_passthrough = 0

    for i, row in enumerate(rows, 1):
        out = await node(
            {"user_input": row["input"], "history": _history(row.get("history") or [])}
        )
        resolved = out["resolved_input"]
        ok = row["must_contain"] in resolved
        hit += ok
        passthrough = resolved.strip() == row["input"].strip()
        passed_through += passthrough
        # 「关键词本来就在原话里」= 这条用例即使实现**从不改写**也会 OK。
        # 自动算,不靠人工标注 —— 免得像 ch04 那样把弱口径当强证据引用。
        weak = row["must_contain"] in row["input"]
        satisfiable_by_passthrough += weak

        emit(
            f"{'OK ' if ok else 'MISS'} [{i}] {'透传' if passthrough else '改写'} "
            f"原话={row['input']!r} → {resolved!r}"
            f"(要含 {row['must_contain']!r}){' [透传即可满足]' if weak else ''}"
        )

    emit()
    emit(f"准确率 {hit}/{len(rows)} = {hit / len(rows):.1%}")
    emit(
        f"原样透传 {passed_through}/{len(rows)} 条;"
        f"其中 {satisfiable_by_passthrough}/{len(rows)} 条的关键词**本来就在原话里** "
        f"(这些条测不出「有没有改写」)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
