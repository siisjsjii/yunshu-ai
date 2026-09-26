"""把待审样本导成 CSV(给人改),再把改完的收回来。

**为什么要走 CSV**:120 条 100% 过 + 85 条抽审,在编辑器里排着改最快;
而且改了什么、改了多少,**进 git,可追溯** —— 页面里点一遍是留不下痕迹的。

用法:
    .venv/Scripts/python.exe scripts/export_label_review.py export
    # ← 用户改 evals/topic/labels/trainval.csv 之后
    .venv/Scripts/python.exe scripts/export_label_review.py import

⚠️ **`export` 出来的那份 CSV 是**空着后三列**入库的**(B6-A,2026-09-26):
`判定(ok/改)` / `最终标签` / `备注` 留空**不是漏填**,而是刻意的 —— 用户改完之后
`git diff` 就是**他改了什么的逐字记录**(计划那句「进 git,可追溯」的落地)
⇒ **不要在 commit 前手填任何一列**(用户只需要填前两列,`备注` 只给人看)。

⚠️ **错误率的读数取自「标签变了没有」,不取自「判定」列**(订正轮 1,2026-09-26):
`判定` 只认 `""` / `ok` / `改`(认不出的值**响亮地报错**),它只当**交叉校验**用 ——
「这条被改了」的机械定义是**生效标签 ≠ 预标标签**(见 `do_import` 的 docstring)。

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

#: 「判定」一列只认这三个值(空 = 没填)。它**只当交叉校验**:改了没有由标签说了算。
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

    ⚠️ **错误率的读数取自「标签变了没有」,不是取自「判定」列**(订正轮 1,2026-09-26)。
    旧实现只数 `判定 == "改"`,而用户**填了「最终标签」却忘了填「判定」**时(完全可能发生)
    打印的是「判『改』的 0 条」—— 一个**看着像「用户什么都没改」的错答案**,
    而 `git diff` 里明明躺着两条更正(spec §6.3 的读数会因此判「≤10% ⇒ 接受、开训」)。
    **机械定义**(不需要用户再报一次):**生效标签 ≠ 预标标签 ⇒ 这一条被改了**。

    两列**互相矛盾**时的处置是**不对称**的,理由各自不同:
    - `判定=改` 而标签**没**变 ⇒ **响亮报错**。最常见的成因是把更正写进了「备注」列
      (三列名长得像)⇒ 那一行会以**旧标签**入库、而记录说「改过」,两边互相拆台;
    - 标签变了而 `判定` 没写「改」 ⇒ **照算**,并把 id 列出来**警告**。少填一个格子
      不该拦下整跑(尤其「没填的按预标算」这句话是我们对用户说过的),
      而「少算错误率」这个后果由「照算」直接解决。
    """
    src = LABELS_DIR / "trainval.csv"
    prelabeled = {r["id"]: r for r in _load()}
    changed = 0
    out_rows: list[dict] = []
    bad_verdicts: list[tuple] = []
    contradictions: list[str] = []
    unverified: list[str] = []          # 标签变了、却没写「改」
    requestioned: list[str] = []        # 「问题」列与预标不一致
    dup_labels: list[str] = []
    seen: set[str] = set()
    known_labels = set(LABELS)
    # ⚠️ `newline=""` 是 `csv` 文档的硬要求(「If csvfile is a file object,
    #    it should be opened with newline=''」)—— 不加的话**带引号字段里的换行**
    #    读不对,而本机(Windows)还会多吞一个 `\r`。
    try:
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
                # ⚠️ **多出字段**:`DictReader` 把它们塞进 `row[None]`,而按列名取值**看不见它**。
                # 实测(2026-09-26):在**文本编辑器**里把「最终标签」填成 `尺码,退换货,物流`
                # (没加引号)⇒ 该行字段数比表头多 ⇒ 列错位,`最终标签` 只剩 `尺码`,
                # **两个更正被无声吞掉**;Excel 存盘会自动加引号,所以这条只在文本编辑器
                # (而计划推荐的就是「在编辑器里排着改」)那条路上炸。
                if None in row:
                    raise SystemExit(
                        f"这一行的字段比表头多(列错位了):{row.get('id')!r} 那一行\n"
                        f"  多出来的字段:{row[None]!r};按列名读到的整行 = {row}\n"
                        f"  最可能的成因:在文本编辑器里往某个单元格打了半角逗号却**没加引号**\n"
                        f"  ⇒ 要么把 `,` 换成 `|`,要么给整格加一对半角引号"
                    )
                rid = (row.get("id") or "").strip()
                if rid not in prelabeled:
                    # 空白行 / 纯逗号行 / 纯空格行都落在这里(id 会洗成空串),
                    # 免得它们变成 `{"id": "", "labels": []}` 那种幽灵记录 ——
                    # 那要到任务 7 才炸成 `KeyError: ''`,报错指向一个不相干的地方。
                    raise SystemExit(
                        f"这一行的 id 不在预标语料里:{rid!r};整行 = {row}\n"
                        f"  (空白行 / 纯逗号行会在这里被拦下;自编号的行也一样 —— "
                        f"id 必须逐字取自 trainval.csv 原本那一列)"
                    )
                if rid in seen:
                    raise SystemExit(
                        f"id 重复:{rid!r}\n"
                        f"  (两行同 id 会让产物里出现两条标签互相矛盾的记录,"
                        f"而任务 7 的按 id 合并是**静默**只留后者 —— 谁赢只取决于行序)"
                    )
                seen.add(rid)
                pre = prelabeled[rid]
                # 「预标标签」列是「变了没有」的比较基准 ⇒ 它必须与语料同源。
                raw_pre = [x.strip() for x in (row.get("预标标签") or "").split("|") if x.strip()]
                if set(raw_pre) != set(pre["labels"]):
                    raise SystemExit(
                        f"{rid} 的「预标标签」列与 prelabeled.jsonl 不一致:"
                        f"{raw_pre} vs {pre['labels']}\n"
                        f"  ⇒ 这份 CSV 与跟它配套的那份语料**不同源**,"
                        f"「改了没有」就没有基准了"
                    )
                final = (row.get("最终标签") or "").strip()
                # ⚠️ 逐段 `strip()`(实现者 2026-09-26 的订正):`replace(",", "|")` 已经
                # 表明「用户会拿逗号当分隔符」,而人打逗号时**几乎总会跟一个空格**
                # (`尺码, 退换货`)。不 strip 的话那一段是 `" 退换货"` ⇒ 撞下面的合法性检查、
                # 整个 import 停在 SystemExit —— 那时为了往下走就得**去改用户填的那份 CSV**,
                # 而它进 git 的意义正是「用户改了什么」的逐字记录 ⇒ 不该由我们代笔规范化。
                # 合法性检查要拦的是**错字**(`尺碼`),不是空白。
                labels = [x.strip() for x in final.replace(",", "|").split("|") if x.strip()] \
                    or raw_pre
                # 重复类目去重(**保留首次出现的顺序**)。不去的话 `退换货|退换货` 会过校验、
                # 原样入库 —— 脏数据,而没有任何东西会报错。
                # (它**不会**影响下面那句「改了没有」:那里按**集合**比,
                #  `{"退换货"} == {"退换货"}` 恒真 ⇒ 去重与否对读数没有影响。别把这条当理由。)
                deduped = list(dict.fromkeys(labels))
                if deduped != labels:
                    dup_labels.append(rid)
                labels = deduped
                invalid = [lb for lb in labels if lb not in known_labels]
                if invalid:
                    # ⚠️ 响亮地报,不静默丢弃 —— 手打的标签名容易有错字。
                    raise SystemExit(
                        f"{rid} 的标签里有不合法类目:{invalid}\n"
                        f"  (类目名要**逐字**取自 17 类表;分隔符用 `|` 或 `,`,两边的空白无所谓)"
                    )
                verdict = (row.get("判定(ok/改)") or "").strip()
                if verdict not in VERDICTS:
                    # 不静默:认不出的值若当成「没改」,错误率会被**少算**(看起来比实际好)。
                    bad_verdicts.append((rid, verdict))
                if set(labels) != set(pre["labels"]):
                    changed += 1
                    if verdict != "改":
                        unverified.append(rid)
                elif verdict == "改":
                    contradictions.append(rid)
                # 题面**以预标为准**(Controller 2026-09-26 裁定):任务 7 的合并是
                # `{**pre[id], "labels": ...}` ⇒ 训练语料本来就取预标那份;
                # 照抄 CSV 那份只会让 `reviewed.jsonl` 这个「人工劳动的记录」里
                # 题面与**证据串**不同源。代价(已知情接受):用户有意订正题面时被忽略 ——
                # 所以下面把那几行**打出来**,不是无声丢弃。
                if (row.get("问题") or "").strip() != pre["question"]:
                    requestioned.append(rid)
                out_rows.append({"id": rid, "question": pre["question"],
                                 "labels": labels, "reviewed": True})
    except UnicodeDecodeError as exc:
        # 不含糊、也不写产物,但默认报的是 codec 措辞(「'utf-8' codec can't decode byte …」),
        # 读起来像代码坏了。用户侧最常见的原因是**用 Excel 存成了「CSV(逗号分隔)」**。
        raise SystemExit(
            f"{src} 不是 UTF-8:{exc!r}\n"
            f"  ⇒ 大概是 Excel 存盘时选了「CSV(逗号分隔)」;请另存为「CSV UTF-8」再重跑"
        ) from None
    if not out_rows:
        raise SystemExit(
            f"{src} 一行数据都没有(只剩表头?)—— 拒绝写出空产物\n"
            f"  (export 出来的是 1 行表头 + 84 行数据;这份文件被清空过?)"
        )
    if bad_verdicts:
        raise SystemExit(
            f"「判定(ok/改)」里有认不出的值(只认 {VERDICTS}):{bad_verdicts}\n"
            f"  (若把它当成「没改」,错误率会被**少算** ⇒ 先改 CSV,再重跑 import)"
        )
    if contradictions:
        raise SystemExit(
            f"这些行的「判定」说改过、但标签与预标**一模一样**:{contradictions}\n"
            f"  ⇒ 更正大概写进了别的列(「备注」不参与回收,它只给人看)\n"
            f"  (这一行若照原样入库,产物里就是**旧标签**,而记录说改过 —— 两边互相拆台)"
        )
    REVIEWED.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n",
        encoding="utf-8",
    )
    print(f"回收 {len(out_rows)} 条 → {REVIEWED};其中用户判「改」的 {changed} 条")
    # ⚠️ 「判『改』」这条读数**取自「标签与预标不同」**(见 docstring 的机械定义),
    #    「判定」列只当交叉校验 —— 它就是**预标错误率的观测值**(只覆盖被抽到的那些),
    #    要进报告,与 F1 并排(spec §6.3)。
    if unverified:
        print(f"  ⚠️ 这 {len(unverified)} 条的标签与预标不同、而「判定」列没写「改」:{unverified}")
        print("     ⇒ 已**照标签**计入上面的读数;「判定」列只当交叉校验用,不用重填")
    if requestioned:
        print(f"  ⚠️ 这 {len(requestioned)} 行的「问题」列与预标不一致,回收时**以预标为准**:"
              f"{requestioned}")
    if dup_labels:
        print(f"  ⚠️ 这 {len(dup_labels)} 行的标签里有重复类目,已去重(保留首次出现):{dup_labels}")


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
