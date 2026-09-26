"""主题分类的语料加工(四步)。子命令:
    collect   三源合流 → evals/topic/corpus.jsonl
    split     分层抽样 80/10/10 + 冻结测试集
    augment   数据增强(**只扩训练集**)

用法:
    .venv/Scripts/python.exe scripts/prepare_topic_data.py collect
"""

import argparse
import asyncio
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.db.base import get_engine
# ⚠️ 这一行是**同源守卫**要求的(tests/test_topic_clean.py::test_both_sides_use_the_same_clean):
#    训练侧必须与推理侧用**同一份** `clean`。它在本文件里也被真的用到(数「洗后为空」那几行),
#    不是一条只为过守卫而存在的 import。
from app.topic.clean import clean
from app.topic.labeling import dedupe_questions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "topic" / "corpus.jsonl"
TESTING_MD = ROOT / "evals" / "测试集.md"


def _out(message: str) -> None:
    """钉死输出编码:本机 `sys.stdout.encoding` 是 **gbk**,而 `⚠` / `†` 这类字符
    不在 GBK 里 —— 输出里哪天多一个就会 `UnicodeEncodeError`(ch09 那条「脚本自己
    喷错误行」的先例)。照 `scripts/build_kb.py` 的房规把边界钉成 UTF-8,不依赖控制台 codec。

    ⚠️ **实测(2026-09-26)**:本文件当前这几行(中文、`→`)**在 GBK 下恰好都编得出来**
    ⇒ 这条钉的是「以后加个 `⚠` 不会突然崩」,不是「今天就崩」。别把它读成已经发生过。"""
    sys.stdout.buffer.write((message + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


async def _from_db() -> list[dict]:
    """池子 + 对话里的用户话。

    两处都取**去重前**的全量:去重交给 `dedupe_questions` 一处做 ——
    两个地方各去一遍的话,「按什么去重」这件事就有两份实现。

    ⚠️ 两条查询的 `source` **必须不同**(`pool` / `chat`)。写成一个值的话,
    「池子 > 对话 > 测试集.md」这条优先级就只剩一条来源,分布也读不出真数。
    """
    rows: list[dict] = []
    eng = get_engine()
    async with eng.connect() as conn:
        for source, q in (
            ("pool", text("SELECT question FROM low_confidence_questions ORDER BY id")),
            ("chat", text("SELECT content FROM messages WHERE role='user' ORDER BY id")),
        ):
            for (value,) in (await conn.execute(q)).all():
                rows.append({"question": value, "source": source})
    await eng.dispose()
    return rows


def _from_testing_md() -> list[dict]:
    """`evals/测试集.md` 的 300 条**人工写的**政策问句 —— 借用作分类语料。

    它是**检索**评估集,借用不污染检索评估:两个任务不同,
    同一批问句在两处各算各的分母。
    """
    rows: list[dict] = []
    with TESTING_MD.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("问题(query)") or "").strip()
            if q:
                rows.append({"question": q, "source": "evalmd"})
    return rows


async def collect() -> None:
    # 来源**优先级顺序**就是这里的顺序:池子 > 对话 > 测试集.md。
    # 去重保留首次出现的那一条,所以顺序决定了重复问题归谁名下。
    rows = (await _from_db()) + _from_testing_md()
    for r in rows:
        r["provenance"] = "real"
    blank = sum(1 for r in rows if not clean(r["question"]))
    kept = dedupe_questions(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, r in enumerate(kept, 1):
            f.write(json.dumps({"id": f"r-{i:04d}", **r}, ensure_ascii=False) + "\n")

    _out(f"写出 {len(kept)} 条 → {OUT}")
    _out(f"  来源分布: {dict(Counter(r['source'] for r in kept))}")
    # 两个数**分开报**:它们的成因不同(空串 vs 与前面某条重复),
    # 合起来报会把「洗后为空只有几条」读成几十条,而那正是选题材时要看的读数。
    _out(f"  读入 {len(rows)} 条:洗后为空 {blank} 条、与前面重复 {len(rows) - blank - len(kept)} 条,都没写出")
    # ⚠️ 这几个数会**随运行次数增长**(验收脚本每跑一次就往池子与对话里写),
    #    所以引用它时必须带日期,别当常量。


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["collect", "split", "augment"])
    args = ap.parse_args()
    if args.step == "collect":
        asyncio.run(collect())
    else:
        raise SystemExit(f"{args.step} 还没实现(由后续任务补上)")


if __name__ == "__main__":
    main()
