"""工具选择评估集。

用法:.venv/Scripts/python.exe evals/run_tool_selection_eval.py(需真实 key 与 MySQL)

评分口径:闭式精确匹配 —— 工具名是枚举,不存在"措辞不同"的模糊地带。
与 ch01 的 expected_solution(自由文本、关键词口径最终被证明是样本拟合)
形成对比:本口径的分数可以直接引用。

只跑「工具选择」这一段,不执行工具 —— 故不产生任何 DB 写入。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.db.base import get_sessionmaker
from app.llm import create_chat_model
from app.tools.registry import build_tools

CASES = Path(__file__).with_name("tool_selection_cases.jsonl")


def load_cases() -> list[dict]:
    lines = CASES.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def select_tool(model, tools, text: str) -> str | None:
    """跑一轮工具选择,返回模型选中的工具名;没调工具则返回 None。"""
    bound = model.bind_tools(tools)
    accumulated = None
    async for chunk in bound.astream([{"role": "user", "content": text}]):
        accumulated = chunk if accumulated is None else accumulated + chunk
    tool_calls = list(getattr(accumulated, "tool_calls", None) or [])
    return tool_calls[0]["name"] if tool_calls else None


async def main() -> int:
    settings = get_settings()
    model = create_chat_model(settings)
    cases = load_cases()

    async with get_sessionmaker()() as session:
        tools = build_tools(session=session, conversation_id="_eval")

        print(f"模型:{settings.openai_model}  用例数:{len(cases)}\n")
        hits = 0
        misses: list[tuple[dict, str | None]] = []

        for case in cases:
            actual = await select_tool(model, tools, case["text"])
            ok = actual == case["expected"]
            hits += ok
            print(
                f"  [{'✓' if ok else '✗'}] {case['text'][:30]:<32}"
                f" 期望={case['expected']}  实际={actual}"
            )
            if not ok:
                misses.append((case, actual))

    total = len(cases)
    print(f"\n工具选择准确率:{hits}/{total} = {hits / total:.1%}")

    if misses:
        print(f"\n未命中 {len(misses)} 条(附备注,便于判断是标注问题还是模型问题):")
        for case, actual in misses:
            print(f"  - {case['text']}")
            print(f"      期望={case['expected']}  实际={actual}")
            if case.get("note"):
                print(f"      备注:{case['note']}")

    return 1 if misses else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
