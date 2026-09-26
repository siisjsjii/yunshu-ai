"""把待审样本导成 CSV(给人改),再把改完的收回来。

**为什么要走 CSV**:120 条 100% 过 + 85 条抽审,在编辑器里排着改最快;
而且改了什么、改了多少,**进 git,可追溯** —— 页面里点一遍是留不下痕迹的。

用法:
    .venv/Scripts/python.exe scripts/export_label_review.py export
    # ← 用户改 evals/topic/labels/trainval.csv 之后
    .venv/Scripts/python.exe scripts/export_label_review.py import

⚠️ **`export` 出来的那份 CSV 是**空着后两列**入库的**(B6-A,2026-09-26):
两列留空**不是漏填**,而是刻意的 —— 用户改完之后 `git diff` 就是**他改了什么的逐字记录**
(计划那句「进 git,可追溯」的落地)。所以**不要在 commit 前手填任何一列**。

⚠️ **`import` 判定「改」用的是**逐字**的 `改`**:这一列的值只认 `""` / `ok` / `改`
(认不出的值**响亮地报错**,不静默当成「没改」—— 见 `do_import` 的注释)。

⚠️ **用户改完之后,不许再跑 `export`** —— 它会**照当前语料重新抽一遍并覆盖**那份 CSV,
用户填的东西一个字都不剩。要回收就跑 `import`。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.topic.labeling import pick_review_sample
from app.topic.taxonomy import LABELS

ROOT = Path(__file__).resolve().parents[1]
PRELABELED = ROOT / "evals" / "topic" / "prelabeled.jsonl"
LABELS_DIR = ROOT / "evals" / "topic" / "labels"
REVIEWED = ROOT / "evals" / "topic" / "reviewed.jsonl"

#: 抽审量:每类 5 条 × 17 类 ≈ 85 条(spec §6.3)。
#: ⚠️ **实际导出条数会**少于** 85** —— 同一条多标签行会被多个类抽中,再按 id 去重
#: (`pick_review_sample` 的 docstring)。实测(2026-09-26,`prelabeled.jsonl` 逐行同序):**84 条**。
PER_LABEL = 5

HEADER = ["id", "问题", "预标标签", "判定(ok/改)", "最终标签", "备注"]

#: `do_import` 真的要读的列。少任何一列都必须**响亮地报** ——
#: 列名对不上时,`row.get("最终标签")` 会**恒为 None** ⇒ 每一行都静默回落到预标标签,
#: 而打印出来的是「回收 84 条;判『改』的 0 条」—— 一个看着像「用户什么都没改」的错答案。
#: (`备注` 不在其中:它只给人看,回收时一个字都不读。)
REQUIRED_COLUMNS = ("id", "问题", "预标标签", "判定(ok/改)", "最终标签")

#: 「判定」一列只认这三个值(空 = 没填,按 ok 算)。
VERDICTS = ("", "ok", "改")


def _load() -> list[dict]:
    return [
        json.loads(l)
        for l in PRELABELED.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def export() -> None:
    rows = _load()
    sample = pick_review_sample(rows, per_label=PER_LABEL)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out = LABELS_DIR / "trainval.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in sample:
            w.writerow([r["id"], r["question"], "|".join(r["labels"]), "", "", ""])
    print(f"导出 {len(sample)} 条 → {out}")
    # ⚠️ 打印**按类的覆盖**,让人一眼看出哪类抽得少(某类样本不足时会少拿)。
    from collections import Counter
    c = Counter(lb for r in sample for lb in r["labels"])
    missing = [lb for lb in LABELS if c[lb] == 0]
    print(f"  每类条数:{dict(c)}")
    if missing:
        print(f"  ⚠️ 这些类**一条都没抽到**(样本不足):{missing} —— 它们没有人工复核覆盖")


def do_import() -> None:
    """回收:用户填了「最终标签」的按用户的,没填的按预标的。

    「判定」列只影响报表(改了多少条),不影响入库的标签 —— 标签一律以
    「最终标签」列为准,空了才回落到预标。
    """
    src = LABELS_DIR / "trainval.csv"
    changed = 0
    out_rows = []
    bad_verdicts = []
    known_labels = set(LABELS)
    # ⚠️ `newline=""` 是 `csv` 文档的硬要求(「If csvfile is a file object,
    #    it should be opened with newline=''」)—— 不加的话**带引号字段里的换行**
    #    读不对,而本机(Windows)还会多吞一个 `\r`。
    with src.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            # 不静默:表头对不上时下面每一行都会回落成预标,而读数看着完全正常。
            raise SystemExit(
                f"{src} 的表头缺列:{missing}\n"
                f"  实际表头:{reader.fieldnames}\n"
                f"  (列名对不上 ⇒ 每一行都会**静默回落**成预标标签)"
            )
        for row in reader:
            final = (row.get("最终标签") or "").strip()
            # ⚠️ 逐段 `strip()`(实现者 2026-09-26 的订正):`replace(",", "|")` 已经
            # 表明「用户会拿逗号当分隔符」,而人打逗号时**几乎总会跟一个空格**
            # (`尺码, 退换货`)。不 strip 的话那一段是 `" 退换货"` ⇒ 撞下面的合法性检查、
            # 整个 import 停在 SystemExit —— 那时为了往下走就得**去改用户填的那份 CSV**,
            # 而它进 git 的意义正是「用户改了什么」的逐字记录 ⇒ 不该由我们代笔规范化。
            # 合法性检查要拦的是**错字**(`尺碼`),不是空白。
            labels = [x.strip() for x in final.replace(",", "|").split("|") if x.strip()] or \
                     [x.strip() for x in (row["预标标签"] or "").split("|") if x.strip()]
            verdict = (row.get("判定(ok/改)") or "").strip()
            if verdict not in VERDICTS:
                # 不静默:认不出的值若当成「没改」,错误率会被**少算**(看起来比实际好)。
                bad_verdicts.append((row.get("id"), verdict))
            if verdict == "改":
                changed += 1
            invalid = [lb for lb in labels if lb not in known_labels]
            if invalid:
                # ⚠️ 响亮地报,不静默丢弃 —— 手打的标签名容易有错字。
                raise SystemExit(
                    f"{row['id']} 的标签里有不合法类目:{invalid}\n"
                    f"  (类目名要**逐字**取自 17 类表;分隔符用 `|` 或 `,`,两边的空白无所谓)"
                )
            out_rows.append({"id": row["id"], "question": row["问题"],
                             "labels": labels, "reviewed": True})
    if bad_verdicts:
        raise SystemExit(
            f"「判定(ok/改)」里有认不出的值(只认 {VERDICTS}):{bad_verdicts}\n"
            f"  (若把它当成「没改」,错误率会被**少算** ⇒ 先改 CSV,再重跑 import)"
        )
    REVIEWED.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n",
        encoding="utf-8",
    )
    print(f"回收 {len(out_rows)} 条 → {REVIEWED};其中用户判「改」的 {changed} 条")
    # ⚠️ 「judged 改」的比例就是**预标错误率的观测值**(只覆盖被抽到的那些),
    #    它要进报告,与 F1 并排(spec §6.3)。


def _pin_stdout_encoding() -> None:
    """把 stdout 钉成 UTF-8。**这是本机的硬约束,不是美化。**

    本机 locale 是 **cp936**:`⚠️`(U+26A0 + VS16)与 `⇒` **编不进 GBK**
    (实测 2026-09-26:`→` 能编,`⚠️` / `⇒` / `✓` 都不能),而 `⚠️` 只出现在
    「某类一条都没抽到」那条 print 里 ⇒ 不钉的话,**恰恰在最需要它输出的那条路径上**
    抛 `UnicodeEncodeError`:CSV 已经写好(所以产物是好的)、命令却以退出码 1 结束、
    那行警告消失 —— 「产物是好的」与「日志里没有警告」叠在一起最难分辨的一种假绿。

    与 `scripts/prelabel_topics.py` 同款做法;那边有一条子进程测试钉住它
    (`tests/test_topic_labeling.py::test_main_pins_stdout_encoding`)。
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    _pin_stdout_encoding()
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "import"])
    args = ap.parse_args()
    export() if args.action == "export" else do_import()


if __name__ == "__main__":
    main()
