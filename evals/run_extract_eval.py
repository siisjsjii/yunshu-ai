"""抽取评估集。按字段分别算准确率 —— 混在一起算会掩盖问题。

需要真实 API key(读 .env)。用法:
    .venv/Scripts/python.exe evals/run_extract_eval.py

评分口径:
- `order_id` / `request_type` 为闭式取值,用精确匹配。
- `expected_solution` 是自由文本,精确匹配必然为 0(表述不同即算错),
  因此**主口径为关键词命中**:去掉空白与常见标点后,该用例的全部关键词
  都出现在模型输出中。精确匹配结果作为已知无效的基线一并打印,不隐藏。
"""

import asyncio
import json
import sys
import unicodedata
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

# 归一化时剔除的字符:空白 + 中英文常见标点。
_PUNCT = set(
    " \t\n\r　"
    "，。、；：！？…·"
    ",.;:!?\"'`"
    "“”‘’"
    "()（）[]【】{}<>《》"
    "-—_/\\|~@#$%^&*+="
)


def normalize(text: str) -> str:
    """去掉空白与常见标点,并统一大小写/全半角。用于关键词匹配。"""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if ch not in _PUNCT)


def keywords_hit(actual: str, keywords: list[str]) -> bool:
    """全部关键词都出现才算命中。空列表表示该用例无此口径(见报告)。"""
    norm_actual = normalize(actual)
    return all(normalize(kw) in norm_actual for kw in keywords)


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
    keywords = case.get("solution_keywords", [])
    return {
        "text": case["text"],
        "actual": actual,
        "hits": hits,
        "solution_keywords": keywords,
        "keyword_hit": keywords_hit(actual["expected_solution"], keywords),
    }


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

    # 分字段准确率。分母是**实际解析成功**的用例数,解析失败不计入。
    # expected_solution 用关键词口径;其精确匹配作为基线单列。
    kw_scored = [r for r in scored if r["solution_keywords"]]
    exact_solution = sum(1 for r in scored if r["hits"].get("expected_solution"))

    print(f"{'字段':<28}{'命中':>6}{'总数':>6}{'准确率':>10}")
    print("-" * 50)
    for field in ("order_id", "request_type"):
        hit = sum(1 for r in scored if r["hits"].get(field))
        total = len(scored)
        rate = hit / total if total else 0.0
        print(f"{field:<28}{hit:>6}{total:>6}{rate:>9.1%}")

    kw_hit = sum(1 for r in kw_scored if r["keyword_hit"])
    kw_total = len(kw_scored)
    kw_rate = kw_hit / kw_total if kw_total else 0.0
    print(f"{'expected_solution (关键词命中)':<24}{kw_hit:>6}{kw_total:>6}{kw_rate:>9.1%}")

    # 已知无效的基线,单独列出以示对比 —— 不隐藏。
    print(
        f"{'expected_solution (精确匹配基线)':<22}"
        f"{exact_solution:>6}{len(scored):>6}"
        f"{(exact_solution / len(scored) if scored else 0.0):>9.1%}"
    )

    # 关键词未命中时把实际输出和缺失的关键词一并打出 —— 否则无法区分
    # "抽取错了" 与 "措辞不同"。这是关键词口径能自查的前提。
    kw_missed = [r for r in kw_scored if not r["keyword_hit"]]
    if kw_missed:
        print(f"\n关键词未命中 {len(kw_missed)} 条(附实际输出,供人工判断是抽取错误还是措辞差异):")
        for r in kw_missed:
            actual_solution = r["actual"]["expected_solution"]
            missing = [
                kw
                for kw in r["solution_keywords"]
                if normalize(kw) not in normalize(actual_solution)
            ]
            print(f"  - {r['text'][:30]}…")
            print(f"      实际输出:{actual_solution}")
            print(f"      缺失关键词:{missing}")

    excluded = [r for r in scored if not r["solution_keywords"]]
    if excluded:
        print(
            f"\n注:关键词口径的分母为 {kw_total} —— 有 {len(excluded)} 条用例"
            f"未设关键词(金标与正确输出之间不存在有区分度的共同词),"
            f"不计入该口径:"
        )
        for r in excluded:
            print(f"  - {r['text'][:30]}… → 实际输出:{r['actual']['expected_solution']}")

    if failures:
        print(f"\n抽取失败 {len(failures)} 条:")
        for r in failures:
            print(f"  - {r['text'][:30]}… → {r['error']}")

    print("\n逐条结果(order / type / solution关键词 / solution精确):")
    for r in results:
        if "error" in r:
            print(f"  [ERROR] {r['text'][:34]}")
            continue
        marks = "".join("✓" if r["hits"][f] else "✗" for f in FIELDS)
        kw = "✓" if r["keyword_hit"] else "✗"
        print(f"  [{marks}|kw {kw}] {r['text'][:34]}")

    print()
    await token_deviation_check(model)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
