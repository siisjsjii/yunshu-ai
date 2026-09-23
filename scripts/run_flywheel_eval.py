"""飞轮第一步(标准化)的标注样例评估。需真实 key(打网络),不属于单测。

用法:
    .venv/Scripts/python.exe scripts/run_flywheel_eval.py
    .venv/Scripts/python.exe scripts/run_flywheel_eval.py --model chat

口径:`expect_contains` / `expect_not_contains` / `max_chars` 全是**闭式**的
(子串是否出现、字数上限),不做字符串全等 —— 模型在 temperature=0 下依然非确定,
本仓已记账。走的是**生产函数** `app.flywheel.normalize.normalize_question`,
不是另抄一份 prompt 调用。

**每条都打两个标签**(比准确率本身重要):

- `透传`:标准问题与用户原话是否一字未动。反面提醒是「全部透传」也能拿满分;
- `[透传即可满足]`:**自动算**的 —— 原话本身就能同时满足该条的全部判据
  (含 `expect_not_contains` 与 `max_chars`)⇒ 这条用例对「模型有没有做事」
  **零判别力**,它守的是另一件事(别把完整的问题改坏)。
  与 `run_resolve_eval.py` 那版相比这里多算了后两项:只查「关键词在不在原话里」
  会把「原话里带着必须被剔除的噪声」的用例也标成弱,那是**标反**。

`--model extract`(默认)用 `create_extract_model`(温度 0);`--model chat` 用
`create_chat_model`(温度 0.7),与 `run_intent_eval.py` / `run_resolve_eval.py` 同源。
两者温度不同,数字不可混用。
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.flywheel.normalize import normalize_question
from app.llm import create_chat_model, create_extract_model

CASES = Path(__file__).resolve().parents[1] / "evals" / "flywheel_cases.jsonl"


def emit(line: str = "") -> None:
    # 控制台是 cp936;标准化结果是中文,一律走字节出口(本仓平台陷阱)。
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        print(line)
        return
    stream.write(line.encode("utf-8") + b"\n")
    stream.flush()


def judge(row: dict, standard: str, answer: str) -> list[str]:
    """闭式判判据。返回**没通过的原因**列表(空 = OK)。

    逐条列出原因而不是只回一个布尔:一条用例挂在哪一项上,决定了它是
    「模型没做到」还是「标注写错了」—— 后者在这个仓里出过(`\\d{4,32}` 匹配
    「99」那种同义反复判据)。
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


def satisfiable_by_passthrough(row: dict) -> bool:
    """「原话原样透传」能不能满足这条的全部判据。能 ⇒ 这条用例是弱用例。"""
    raw = row["raw"]
    if any(want not in raw for want in row.get("expect_contains") or []):
        return False
    if any(unwanted in raw for unwanted in row.get("expect_not_contains") or []):
        return False
    return len(raw) <= row["max_chars"]


async def main() -> int:
    parser = argparse.ArgumentParser(description="飞轮标准化标注样例评估集")
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

    emit(f"模型 {settings.openai_model} / {args.model} / 用例 {len(rows)} 条")
    emit()

    hit = 0
    passed_through = 0
    weak = 0
    for i, row in enumerate(rows, 1):
        out = await normalize_question(row["raw"], model=model)
        standard, answer = out["standard_question"], out["example_answer"]
        bad = judge(row, standard, answer)
        ok = not bad
        hit += ok
        passthrough = standard.strip() == row["raw"].strip()
        passed_through += passthrough
        is_weak = satisfiable_by_passthrough(row)
        weak += is_weak

        emit(
            f"{'OK ' if ok else 'MISS'} [{i}] {'透传' if passthrough else '改写'} "
            f"{row['raw']!r}\n"
            f"        → 标准问题={standard!r}({len(standard)} 字,上限 {row['max_chars']})\n"
            f"        → 示例答案={answer!r}"
            f"{'  [透传即可满足]' if is_weak else ''}"
            + (f"\n        ✗ {';'.join(bad)}" if bad else "")
        )

    emit()
    emit(f"准确率 {hit}/{len(rows)} = {hit / len(rows):.1%}")
    emit(
        f"原样透传 {passed_through}/{len(rows)} 条;"
        f"其中 {weak}/{len(rows)} 条 [透传即可满足]"
        f"(这些条测不出「模型有没有做事」,只测「别把完整的问题改坏」)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
