"""抽取评估集。按字段分别算准确率 —— 混在一起算会掩盖问题。

需要真实 API key(读 .env)。用法:
    .venv/Scripts/python.exe evals/run_extract_eval.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.llm import create_extract_model  # noqa: E402
from app.memory.trim import count_tokens  # noqa: E402
from app.prompts import build_extract_messages  # noqa: E402
from app.services.extract import ExtractionError, extract_structured  # noqa: E402

CASES = Path(__file__).with_name("extract_cases.jsonl")
FIELDS = ("order_id", "request_type", "expected_solution")

# 若遇到限流(429)报错,可调大该值在两次调用之间稍作等待。
SLEEP_BETWEEN_CALLS = 0.0


def load_cases() -> list[dict]:
    lines = CASES.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def run_case(model, case: dict) -> dict:
    try:
        result = await extract_structured(model=model, text=case["text"])
    except ExtractionError as exc:
        return {"text": case["text"], "error": str(exc), "hits": {}}

    actual = result.model_dump()
    hits = {
        field: actual[field] == case["expected"][field] for field in FIELDS
    }
    return {"text": case["text"], "actual": actual, "hits": hits}


async def token_deviation_check(model) -> None:
    """独立核对 tiktoken 估算偏差。

    结构化抽取路径直接返回 ExtractResult,usage_metadata 到不了这里,
    所以另起一段:用同样的消息,先算 count_tokens,再裸调模型读
    response.usage_metadata["input_tokens"],两者并排比较。
    """
    cases = load_cases()
    by_len = sorted(range(len(cases)), key=lambda i: len(cases[i]["text"]))
    picks = [
        ("最长", cases[by_len[-1]]),
        ("最短", cases[by_len[0]]),
        ("中等", cases[by_len[len(by_len) // 2]]),
    ]

    print("token 估算偏差核对(独立于结构化抽取):")
    for label, case in picks:
        messages = build_extract_messages(case["text"])
        estimated = count_tokens("".join(m.content for m in messages))
        try:
            resp = await model.ainvoke(messages)
            actual = (resp.usage_metadata or {}).get("input_tokens")
        except Exception as exc:  # 裸调用失败不改变退出码,只报告
            print(f"  [{label}] {case['text']!r} → 裸调用失败:{exc}")
            continue
        ratio = (estimated / actual) if actual else 0.0
        print(f"  [{label}] text={case['text']!r}")
        print(f"    count_tokens(估) = {estimated}")
        print(f"    usage.input_tokens(实) = {actual}")
        print(f"    估算/实际 = {ratio:.3f}")


async def main() -> int:
    settings = get_settings()
    model = create_extract_model(settings)
    cases = load_cases()

    print(f"模型:{settings.openai_model}  用例数:{len(cases)}\n")

    results = []
    for case in cases:
        results.append(await run_case(model, case))
        if SLEEP_BETWEEN_CALLS:
            await asyncio.sleep(SLEEP_BETWEEN_CALLS)

    failures = [r for r in results if "error" in r]
    scored = [r for r in results if "error" not in r]

    # 分字段准确率
    print(f"{'字段':<20}{'命中':>6}{'总数':>6}{'准确率':>10}")
    print("-" * 42)
    for field in FIELDS:
        hit = sum(1 for r in scored if r["hits"].get(field))
        total = len(scored)
        rate = hit / total if total else 0.0
        print(f"{field:<20}{hit:>6}{total:>6}{rate:>9.1%}")

    if failures:
        print(f"\n抽取失败 {len(failures)} 条:")
        for r in failures:
            print(f"  - {r['text'][:30]}… → {r['error']}")

    print("\n逐条结果:")
    for r in results:
        if "error" in r:
            print(f"  [ERROR] {r['text'][:34]}")
            continue
        marks = "".join("✓" if r["hits"][f] else "✗" for f in FIELDS)
        print(f"  [{marks}] {r['text'][:34]}")

    print()
    await token_deviation_check(model)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
