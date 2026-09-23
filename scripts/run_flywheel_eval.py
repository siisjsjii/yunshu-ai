"""飞轮第一步(标准化)与第二步(查重)的标注样例评估。需真实 key(打网络),不属于单测。

用法:
    .venv/Scripts/python.exe scripts/run_flywheel_eval.py
    .venv/Scripts/python.exe scripts/run_flywheel_eval.py --model chat

**一份评估集,两种用例**(`evals/flywheel_cases.jsonl`,靠 `kind` 字段区分;
**为什么不是两个文件**:spec §8.1 说的就是「**那份**评估集要含一组同义/不同义的问题对
供 dedupe 判」,而一个入口跑两种用例读者少记一件事。没有 `kind` 键的行按 `normalize`
处理 —— 早先写的 10 条因此**一字未动**,前后两次读数可比):

- `normalize`(默认):`raw` + `expect_contains` / `expect_not_contains` / `max_chars`
  (+ 可选 `expect_answer_empty`)。走**生产函数**
  `app.flywheel.normalize.normalize_question`,不是另抄一份 prompt 调用;
- `dedupe`:`candidate` + `pending`(已有问题清单)+ `expect_index`
  (期望命中的**编号,从 1 开始**;`-1` = 都不匹配)+ 可选 `hard`(近域硬负例)。
  走**生产函数** `app.flywheel.dedupe.find_duplicate`。

口径全是**闭式**的(子串是否出现、字数上限、编号是不是那一个),不做字符串全等 ——
模型在 temperature=0 下依然非确定,本仓已记账。

**每类都打标签**(比准确率本身重要):

- normalize:`透传`(输出与用户原话是否一字未动)、`[透传即可满足]`(**自动算**的 ——
  原话本身就能满足该条全部判据 ⇒ 这条对「模型有没有做事」零判别力。
  与 `run_resolve_eval.py` 那版相比多算了 `expect_not_contains` 与 `max_chars`:
  只查「关键词在不在原话里」会把「原话里带着必须被剔除的噪声」的用例**标反**);
- dedupe:`[硬负例]` —— 近域硬负例单独计数。**这是这份用例集里唯一有压力的那一类**:
  本仓记过「干扰项里 3 条离阈值很远、不构成压力」的教训,只放明显不同的对子,
  这个评估集没有信息量。

`--model extract`(默认)用 `create_extract_model`(温度 0);`--model chat` 用
`create_chat_model`(温度 0.7)。两者温度不同,数字不可混用。
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.db.models import ReviewQueue
from app.flywheel.dedupe import find_duplicate
from app.flywheel.normalize import normalize_question
from app.llm import create_chat_model, create_extract_model

CASES = Path(__file__).resolve().parents[1] / "evals" / "flywheel_cases.jsonl"


def emit(line: str = "") -> None:
    # 控制台是 cp936;用例与输出都是中文,一律走字节出口(本仓平台陷阱)。
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        print(line)
        return
    stream.write(line.encode("utf-8") + b"\n")
    stream.flush()


def judge_normalize(row: dict, standard: str, answer: str) -> list[str]:
    """返回**没通过的原因**列表(空 = OK)。

    逐条列出原因而不是只回一个布尔:一条用例挂在哪一项上,决定了它是「模型没做到」
    还是「标注写错了」—— 后者在这个仓里出过(`\\d{4,32}` 匹配「99」那种同义反复判据)。
    """
    bad: list[str] = []
    for want in row.get("expect_contains") or []:
        if want not in standard:
            bad.append(f"缺关键词 {want!r}")
    for unwanted in row.get("expect_not_contains") or []:
        if unwanted in standard:
            bad.append(f"混进了 {unwanted!r}")
    if len(standard) > row["max_chars"]:
        bad.append(f"超长 {len(standard)}>{row['max_chars']}")
    if row.get("expect_answer_empty") and answer:
        bad.append("业务性问题不该有示例答案,却写了")
    return bad


def judge_dedupe(actual: int, expect: int) -> list[str]:
    return [] if actual == expect else [f"期望编号 {expect},实际 {actual}"]


def satisfiable_by_passthrough(row: dict) -> bool:
    """「原话原样透传」能不能满足这条的全部判据。能 ⇒ 这条用例是弱用例。"""
    raw = row["raw"]
    if any(want not in raw for want in row.get("expect_contains") or []):
        return False
    if any(unwanted in raw for unwanted in row.get("expect_not_contains") or []):
        return False
    return len(raw) <= row["max_chars"]


def _pending(texts: list[str]) -> list[ReviewQueue]:
    """把清单里的句子包成**未落库**的 `ReviewQueue`(查重只看 `standard_question`)。

    用真 ORM 对象而不是 SimpleNamespace:生产把它当 `ReviewQueue` 用,替身换个形状的话
    被验的就不是生产里那一支了(本仓记过的替身形状教训)。
    """
    return [
        ReviewQueue(standard_question=t, example_answer="", first_raw_question=t)
        for t in texts
    ]


async def run_normalize(rows: list[dict], model) -> tuple[int, int, int, int]:
    hit = passed_through = weak = 0
    for i, row in enumerate(rows, 1):
        out = await normalize_question(row["raw"], model=model)
        standard, answer = out["standard_question"], out["example_answer"]
        bad = judge_normalize(row, standard, answer)
        ok = not bad
        hit += ok
        passthrough = standard.strip() == row["raw"].strip()
        passed_through += passthrough
        is_weak = satisfiable_by_passthrough(row)
        weak += is_weak

        emit(
            f"{'OK ' if ok else 'MISS'} [N{i}] {'透传' if passthrough else '改写'} "
            f"{row['raw']!r}\n"
            f"        → 标准问题={standard!r}({len(standard)} 字,上限 {row['max_chars']})\n"
            f"        → 示例答案={answer!r}"
            f"{'  [透传即可满足]' if is_weak else ''}"
            + (f"\n        ✗ {';'.join(bad)}" if bad else "")
        )
    return hit, len(rows), passed_through, weak


async def run_dedupe(rows: list[dict], model) -> tuple[int, int, int, int]:
    hit = hard_total = hard_hit = 0
    for i, row in enumerate(rows, 1):
        pending = _pending(row["pending"])
        found = await find_duplicate(row["candidate"], pending, model=model)
        # 编号从 1 开始(-1 = 都没匹配上),与提示词里给模型的编号同一套口径。
        actual = pending.index(found) + 1 if found is not None else -1
        bad = judge_dedupe(actual, row["expect_index"])
        ok = not bad
        hit += ok
        is_hard = bool(row.get("hard"))
        hard_total += is_hard
        hard_hit += is_hard and ok

        emit(
            f"{'OK ' if ok else 'MISS'} [D{i}] 候选={row['candidate']!r} 期望={row['expect_index']}"
            f"{'  [硬负例]' if is_hard else ''}\n"
            f"        清单={row['pending']}\n"
            f"        → 实际={actual}"
            + (f"(命中 {row['pending'][actual - 1]!r})" if actual > 0 else "(没有匹配)")
            + (f"\n        ✗ {';'.join(bad)}" if bad else "")
        )
    return hit, len(rows), hard_hit, hard_total


async def main() -> int:
    parser = argparse.ArgumentParser(description="飞轮标注样例评估集(标准化 + 查重)")
    parser.add_argument(
        "--model",
        choices=("extract", "chat"),
        default="extract",
        help="extract=温度 0(默认,可复跑);chat=温度 0.7",
    )
    args = parser.parse_args()

    settings = get_settings()
    model = (
        create_chat_model(settings) if args.model == "chat"
        else create_extract_model(settings)
    )

    rows = [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:                     # 空文件不许 ZeroDivisionError
        emit(f"用例文件是空的:{CASES}")
        return 1

    normalize_rows = [r for r in rows if r.get("kind", "normalize") == "normalize"]
    dedupe_rows = [r for r in rows if r.get("kind") == "dedupe"]
    emit(
        f"模型 {settings.openai_model} / {args.model} / 用例 {len(rows)} 条"
        f"(标准化 {len(normalize_rows)} + 查重 {len(dedupe_rows)})"
    )

    emit("\n---- 标准化 ----")
    n_hit, n_total, passed_through, weak = await run_normalize(normalize_rows, model)
    emit(
        f"\n标准化 {n_hit}/{n_total} = {n_hit / n_total:.1%};原样透传 {passed_through}/{n_total} 条;"
        f"其中 {weak}/{n_total} 条 [透传即可满足]"
        f"(这些条测不出「模型有没有做事」,只测「别把完整的问题改坏」)"
    )

    emit("\n---- 查重 ----")
    d_hit, d_total, hard_hit, hard_total = await run_dedupe(dedupe_rows, model)
    emit(
        f"\n查重 {d_hit}/{d_total} = {d_hit / d_total:.1%};"
        f"其中**近域硬负例** {hard_hit}/{hard_total}(这 {hard_total} 条是唯一有压力的一类)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
