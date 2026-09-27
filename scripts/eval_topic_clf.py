"""ch10-B Task 10:在**冻结的测试集**上评测 17 类多标签主题分类器 —— 验收 ① 的落点。

```
.venv/Scripts/python.exe scripts/eval_topic_clf.py --report evals/topic/report.md
```

一条命令产出四份产物(`spec §8.6`):

| 产物 | 给谁 |
|---|---|
| `report.md` | 给人读 |
| `report.json` | 机器可读、可 diff |
| `matrix_confusion.csv` | 每类 2×2(TP/FP/FN/TN),§8.2 的第 1 张 |
| `matrix_flow.csv` | 17×17 **误判流向矩阵**,§8.2 的第 2 张 |
| `misjudged.csv` | 判错样本(≤60 条),**给人判「谁错了」**(§8.5) |

## 三条不许动的东西

1. **只读冻结的测试集。** 它是**验收 ① 的唯一依据**,而且测试集没参与过任何选择
   (切分 / 早停 / 阈值扫描全在 train / val 上)。所以这份报告是**无偏读数**;
   反过来,谁要是拿它去调阈值或选 checkpoint,这一句话就当场作废。
2. **不参与任何随机**(spec §8.6):权重只读、`model.eval()`、`torch.no_grad()`、
   固定 batch、单一设备。**同一份权重跑两次,四份产物逐字节相同** ——
   `scripts/acceptance_ch10.sh` 会用两次运行的 sha256 核对这一点。
3. **不许重训、不许改 `THRESHOLD`。** 阈值是**纯后处理**;重训会换掉权重,并让
   `train_meta.json` 那份已被复核过的四指标**全部作废**,而**没有任何东西会报错**。

## 为什么默认跑 CPU(`--device`)

不是性能考虑(120 条 × 64 token,CPU 几秒),是**逐字节可复现**:cuBLAS / cuDNN 的
算子选择带启发式,同一份权重两次前向的最后几个 bit 不保证相同,而阈值比较正好落在
那几位上时预测会翻。§8.6 那句「跑两次逐字节相同」要么靠 CPU 成立,要么就得放弃。

## 这份报告里最容易被读错的四件事(都已印进 `report.md` 的正文)

1. **`†` 会铺满几乎整张表** —— `17 × 15 = 255 > 169`(测试集的标签槽总数),
   120 行**算术上装不下**「17 类各 ≥15」。那是数据规模的实况,不是模型不行。
2. **测试集的标签是预标产物**,只有 12/120 带人工复核痕迹 ⇒ 那个 F1 **不是**
   在人工标注的黄金集上测的。
3. **CP-2 的 `0/84 = 0.0%` 撑不起「已知上界」那句话**,必须连同它的三条限定一起读。
4. **三列的 macro-F1 不可互比** —— support = 0 的类贡献 0(`macro_f1` 的定义)。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

# `python scripts/x.py` 时 sys.path[0] 是 **scripts/**、不是仓库根,所以要先补上
# (与 build_kb.py / train_topic_clf.py / calibrate_evidence.py 同款)。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.topic.metrics import (
    label_count_match,
    macro_f1,
    micro_f1,
    mislabelled_flow,
    per_class_prf,
    subset_accuracy,
)
from app.topic.model import load_artifacts, load_rows
from app.topic.taxonomy import LABELS

#: spec §8.3:**`support < 15` 的类那一行标 `†`**。⚠️ **不许为了迁就数据调低它** ——
#: 调低等于把「这一行的 F1 不成结论」这条判据换成一个让表格好看的数字。
SUPPORT_MIN = 15

#: 冻结的测试集。**只读**;报告里会印它的行数与字节 sha256,让「这些数是哪份数据上测的」有据可查。
TEST_PATH = Path("evals/topic/topic_test.jsonl")
MODEL_DIR = Path("models/topic-clf")
REPORT_PATH = Path("evals/topic/report.md")

#: 订正 13-H:冻结测试集上**同时**算 `t = 0.5`(产物基线)与 `t = 0.3`(val 扫描的赢家)。
#: 测试集没参与任何选择 ⇒ 后一行不是数据窥探,而是「0.3 到底真不真好」的无偏读数。
ALT_THRESHOLD = 0.3

#: 判错样本的导出条数上界(spec §8.5)。**本轮不产出那份 CSV**(归后续任务),
#: 但把「有多少条被判错」这个数先算出来入账,免得后面再算一遍。
ERROR_SAMPLE_LIMIT = 60

#: 三个口径(spec §8.4)。**表头里的那个数由这里现算**,不是写死的字符串 ——
#: 「`全体 120`」与「`keep=lambda r: True`」是两个分开的东西,写死之后一条
#: 「谓词改错、表头照旧」的改动能产出一张**每个数都真、只有列名是假的**表。
#: 本仓记过名字与语义不符的账(`agent_steps` 读作「步数」,实际是轮次序号)。
STRATA: tuple[tuple[str, str, object], ...] = (
    ("all", "全体", lambda r: True),
    ("real", "只看真实", lambda r: r.get("provenance") == "real"),
    ("synthetic", "只看合成", lambda r: r.get("provenance") == "synthetic"),
)


def predict_from_logits(logits, threshold: float, labels) -> list[list[str]]:
    """`logits` + 阈值 → 每条的预测标签列表(**按 `labels` 的顺序**)。

    ⚠️ **这是阈值解码的第二处实现。** 第一处在
    `scripts/train_topic_clf.py::metrics_from_logits` 里(`sigmoid(logits) >= t`),
    而那个函数**不返回 `y_pred`**,两张矩阵拿不到它。`app/topic/metrics.py` 刻意
    保持**零第三方依赖**(它的模块 docstring 第一条),所以解码抽不到那个唯一写口里。

    ⇒ 退而求其次:**不让两处安静地漂开**。`tests/test_topic_metrics.py` 的
    `test_eval_script_and_training_script_decode_logits_identically` 拿一组
    logit **恰好等于 0.0**(`sigmoid(0.0) == 0.5` 精确相等)的输入把两边钉在一起 ——
    那一格上 `>=` 与 `>` 给出不同的预测,进而给出不同的 micro-F1(0.8 vs 0.5)。

    比较用 `>=`(与训练侧逐字一致):`sigmoid(logit) >= t`。
    """
    probs = torch.sigmoid(torch.as_tensor(np.asarray(logits), dtype=torch.float32)).numpy()
    hits = probs >= threshold
    return [[labels[j] for j in range(len(labels)) if hits[i][j]] for i in range(hits.shape[0])]


def four_metrics(y_true, y_pred, labels) -> dict:
    """`spec §8.1` 的四个指标。**全部来自 `app.topic.metrics`,这里没有第二份实现。**"""
    return {
        "micro_f1": micro_f1(y_true, y_pred, labels),
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "subset_accuracy": subset_accuracy(y_true, y_pred),
        "label_count_match": label_count_match(y_true, y_pred),
    }


def confusion_2x2(row: dict, n_rows: int) -> dict:
    """一类的 2×2(§8.2 的第 1 张表)。`tn` 是**推**出来的:`n − tp − fp − fn`。"""
    tp, fp, fn = row["tp"], row["fp"], row["fn"]
    return {"tp": tp, "fp": fp, "fn": fn, "tn": n_rows - tp - fp - fn}


def slice_rows(rows: list[dict], y_true, logits, keep) -> tuple[list, np.ndarray]:
    """按行号取子集 —— 三个口径(全体 / 真实 / 合成)走的是**同一段代码**。

    ⚠️ 返回的是 `(y_true 子集, logits 子集)` 这个**顺序**;`logits` 是按行号的数组、
    `y_true` 是列表,两边的行号都由 `keep` 现算 ⇒ 永远不会错位。
    (这正是不用「两个分别过滤的列表」的理由:那样两个过滤器一旦不一致就是
    「拿 A 的真值配 B 的预测」,而所有指标照样落在 0–1 之间。)
    """
    idx = [i for i, r in enumerate(rows) if keep(r)]
    return [y_true[i] for i in idx], logits[idx]


def stratum_label(base: str, n: int) -> str:
    """口径的表头 —— **那个数是从行数现算的,不是写死的**。"""
    return f"{base} {n}"


def evaluate_cell(rows, y_true, logits, labels, threshold, keep, *, expect_n=None) -> dict:
    """一个 (口径 × 阈值) 格的四个指标 + 条数。

    `expect_n` 传进来时**当场核对**「表头里的那个数」与「格子里实际用了几行」同源 ——
    两者一旦分家,报出来的就是一张**每个数都真、只有列名是假的**表。
    """
    sub_true, sub_logits = slice_rows(rows, y_true, logits, keep)
    if expect_n is not None and len(sub_true) != expect_n:
        raise SystemExit(f"!!! 口径行数与表头不符:表头写 {expect_n},实际 {len(sub_true)} 行")
    y_pred = predict_from_logits(sub_logits, threshold, labels)
    return {"n": len(sub_true), **four_metrics(sub_true, y_pred, labels)}


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def f4(value) -> str:
    """四位小数 —— **固定格式**是「两次运行逐字节相同」的一部分(别让它随 repr 变)。"""
    return f"{value:.4f}"


def render_evidence(evidence) -> str:
    """把一条样本的 `evidence` 渲染成给人读的一行。**取不到就返回空串 —— 不许编。**

    ⚠️ 这里是「**不许编**」那条规矩的落点。缺证据串的行(字段不存在 / 不是 dict /
    空 dict)一律留空,而不是写 `(无)`、更不是拿题面回填 —— 那份 CSV 是给**人**判
    「模型错 还是 标签错」用的,一个编出来的证据串会让人**判反**。

    ⚠️ 证据串**只用于人判**,**不许**拿它去反推标签对不对:它是**预标那一步的输入**,
    不是裁决(`spec §6.2` 那根「字面提到」的结构性保证说的是另一回事)。
    """
    if not isinstance(evidence, dict) or not evidence:
        return ""
    return " ; ".join(f"{label}:{text}" for label, text in evidence.items())


def write_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    """`newline=""` 是 `csv` 模块的硬要求;显式 `encoding="utf-8"` 是本仓的平台纪律。"""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="ch10-B:冻结测试集上的主题分类器评测")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--test", default=str(TEST_PATH))
    ap.add_argument("--report", default=str(REPORT_PATH))
    ap.add_argument("--json", default=None, help="默认与 --report 同目录")
    ap.add_argument("--matrix-confusion", default=None, help="默认与 --report 同目录")
    ap.add_argument("--matrix-flow", default=None, help="默认与 --report 同目录")
    ap.add_argument("--misjudged", default=None, help="默认与 --report 同目录")
    ap.add_argument("--alt-threshold", type=float, default=ALT_THRESHOLD,
                    help="订正 13-H 要求的第二行;默认 0.3")
    ap.add_argument("--batch-size", type=int, default=64)
    #: ⚠️ 默认 **cpu**:见模块 docstring「为什么默认跑 CPU」。
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    cli = ap.parse_args()

    report_path = Path(cli.report)
    out_dir = report_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(cli.json) if cli.json else out_dir / "report.json"
    conf_path = Path(cli.matrix_confusion) if cli.matrix_confusion else out_dir / "matrix_confusion.csv"
    flow_path = Path(cli.matrix_flow) if cli.matrix_flow else out_dir / "matrix_flow.csv"
    misjudged_path = Path(cli.misjudged) if cli.misjudged else out_dir / "misjudged.csv"

    art = load_artifacts(cli.model_dir)
    labels = list(art["labels"])
    # ⚠️ **标签顺序是契约**(spec §2.7):产物那份与代码里那份不一致 ⇒ 停,别报数。
    #    两侧不一致会产出一张**完全错的分布图,而每个组件都工作正常** ——
    #    而报告里那些数**照样落在 0–1 之间**。
    if labels != list(LABELS):
        raise SystemExit(f"!!! {cli.model_dir}/labels.json 与 app.topic.taxonomy.LABELS 不一致 —— "
                         f"先修标签顺序再看指标\n  产物:{labels}\n  代码:{list(LABELS)}")
    threshold = float(art["threshold"])
    max_length = int(art["max_length"])

    rows = load_rows(cli.test)
    y_true = [list(r["labels"]) for r in rows]
    test_bytes = Path(cli.test).read_bytes()
    test_sha = hashlib.sha256(test_bytes).hexdigest()

    device = torch.device(cli.device)
    tokenizer = AutoTokenizer.from_pretrained(cli.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(cli.model_dir).to(device)
    # `eval()` 关掉 dropout ⇒ 这才有资格说「不参与任何随机」。
    model.eval()

    enc = tokenizer([r["question"] for r in rows], truncation=True,
                    max_length=max_length, padding=True, return_tensors="pt")
    chunks = []
    with torch.no_grad():
        for start in range(0, len(rows), cli.batch_size):
            batch = {k: v[start:start + cli.batch_size].to(device) for k, v in enc.items()}
            chunks.append(model(**batch).logits.to("cpu").numpy())
    logits = np.concatenate(chunks, axis=0)

    # ------------------------------------------------ 口径与阈值的格
    defs = tuple((key, stratum_label(base, sum(1 for r in rows if keep(r))), keep)
                 for key, base, keep in STRATA)
    # ⚠️ 三个口径**必须正好切开这张测试集**:真实 + 合成 == 全体。对不上就停 ——
    #    对不上意味着有行既不是 real 也不是 synthetic,而那张「三列并排」表会
    #    把那些行**静默丢掉**,列名却照旧写着「全体 N」。
    n_all = sum(1 for r in rows if STRATA[0][2](r))
    n_real = sum(1 for r in rows if STRATA[1][2](r))
    n_syn = sum(1 for r in rows if STRATA[2][2](r))
    if n_real + n_syn != n_all:
        raise SystemExit(f"!!! 三个口径切不开这张测试集:real {n_real} + synth {n_syn} != "
                         f"全体 {n_all} ⇒ 有行既不是 real 也不是 synthetic,"
                         "而「三列并排」那张表会把它静默丢掉")
    grid = {
        str(t): {key: evaluate_cell(rows, y_true, logits, labels, t, keep,
                                    expect_n=sum(1 for r in rows if keep(r)))
                 for key, _, keep in defs}
        for t in (threshold, cli.alt_threshold)
    }

    # 主口径(t = 产物里的阈值)的每类表与两张矩阵
    y_pred = predict_from_logits(logits, threshold, labels)
    per_class = per_class_prf(y_true, y_pred, labels)
    flow = mislabelled_flow(y_true, y_pred, labels)
    flagged = [r["label"] for r in per_class if r["support"] < SUPPORT_MIN]
    unflagged = [r["label"] for r in per_class if r["support"] >= SUPPORT_MIN]

    # spec §8.5:Misjudged 的行 —— 判错的**条数**与那份给人判的 CSV 的**行**。
    # ⚠️ **不许在这里自动判「谁错了」** —— 「模型错 还是 标签错」正是**要人来判**的那件事。
    #    那个区分是本次复核的全部价值:如果判错里多数是**标签错**,那 F1 低不是模型的问题,
    #    要修的是数据不是训练 —— 而只看一个 F1 数字永远分不出来。
    #    ⇒ 本脚本只负责**把现场摆好**(原话 / 真实标签 / 预测标签 / 预标的证据串)。
    wrong = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if set(t) != set(p)]
    misjudged = [[rows[i]["question"],
                  "、".join(y_true[i]) or "(无)",
                  "、".join(y_pred[i]) or "(无)",
                  # 证据串取自**这一行自己**的 `evidence`。冻结测试集 120 行**全部**带非空
                  # evidence(实测),所以不必去 join `prelabeled.jsonl` —— 少一个输入就少一处
                  # 按 id 索引的机会(本仓记过「按 id 建索引会把行静默塌掉」的账)。
                  render_evidence(rows[i].get("evidence"))]
                 for i in wrong[:ERROR_SAMPLE_LIMIT]]

    # 三个口径上的 support 实况(订正 10 那条「算术上装不下」的依据)
    support_by_def = {}
    for key, _, keep in defs:
        sub_true, _ = slice_rows(rows, y_true, logits, keep)
        support_by_def[key] = {
            "rows": len(sub_true),
            "slots": sum(len(t) for t in sub_true),
            "enough": sum(1 for r in per_class_prf(sub_true, sub_true, labels)
                          if r["support"] >= SUPPORT_MIN),
        }

    human_reviewed = sum(1 for r in rows if r.get("human_reviewed"))
    provenance = {p: sum(1 for r in rows if r.get("provenance") == p) for p in ("real", "synthetic")}

    # ------------------------------------------------ report.json
    payload = {
        "model_dir": Path(cli.model_dir).as_posix(),
        "threshold": threshold,
        "alt_threshold": cli.alt_threshold,
        "max_length": max_length,
        "device": cli.device,
        "test_file": Path(cli.test).as_posix(),
        "test_rows": len(rows),
        "test_file_sha256_bytes": test_sha,
        "test_provenance": provenance,
        "test_human_reviewed_rows": human_reviewed,
        "test_label_slots": sum(len(t) for t in y_true),
        "labels": labels,
        "grid": grid,
        "per_class": per_class,
        "confusion_2x2": {r["label"]: confusion_2x2(r, len(rows)) for r in per_class},
        "flow": {f"{i}\t{j}": v for (i, j), v in flow.items()},
        "flagged_support_under": SUPPORT_MIN,
        "flagged_labels": flagged,
        "unflagged_labels": unflagged,
        "support_by_stratum": support_by_def,
        "subset_accuracy": grid[str(threshold)]["all"]["subset_accuracy"],
        "label_count_match": grid[str(threshold)]["all"]["label_count_match"],
        "wrong_rows": len(wrong),
        "wrong_rows_limit": ERROR_SAMPLE_LIMIT,
        "misjudged_rows": len(misjudged),
        "misjudged_file": misjudged_path.as_posix(),
        "misjudged_evidence_source": "topic_test.jsonl 每一行自己的 evidence 字段",
        "misjudged_note": ("spec §8.5 —— 给人判「模型错 还是 标签错」用,"
                           "本脚本不预判、也不产出自动化结论"),
        "train_meta": art["meta"],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------ 两张矩阵 CSV
    write_csv(
        conf_path,
        ["label", "support", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "flagged"],
        [[r["label"], r["support"], r["tp"], r["fp"], r["fn"],
          confusion_2x2(r, len(rows))["tn"], f4(r["precision"]), f4(r["recall"]), f4(r["f1"]),
          "†" if r["support"] < SUPPORT_MIN else ""] for r in per_class],
    )
    write_csv(
        flow_path,
        ["真实\\预测", *labels],
        [[i, *[flow[(i, j)] for j in labels]] for i in labels],
    )

    write_csv(misjudged_path, ["原话", "真实标签", "预测标签", "预标的证据串"], misjudged)

    # ------------------------------------------------ report.md
    tm = art["meta"]
    #: CP-2 的读数(`dev-notes/ch10.md` 阶段 7)。**硬编码是刻意的**:它是**一次已发生的人工
    #: 复核**的读数,不是这个脚本能复算的东西 —— 复核记录在 `evals/topic/reviewed.jsonl`,
    #: 而「用户当时改没改」那件事没有任何机器可读的痕迹(见下面三条限定第 1 条)。
    cp2 = {
        "reviewed": 84, "changed": 0,
        "note": "抽审 = 训练/验证集按类分层各抽 5 条(17×5 = 85,实收 84;产物 `evals/topic/reviewed.jsonl`)",
    }
    sweep = tm.get("threshold_sweep", {})
    md: list[str] = []
    add = md.append

    add("# ch10-B 主题分类器 · 冻结测试集评测报告")
    add("")
    add("> 由 `scripts/eval_topic_clf.py` 一条命令产出(验收 ①)。"
        "**只读冻结测试集 + 一个 model checkpoint 路径,不参与任何随机** ——"
        "同一份权重跑两次,本报告与三份附件的字节完全相同。")
    add("")
    add("## 0. 读这张表之前必须先知道的三件事")
    add("")
    add("**(a) 测试集的标签是「预标产物」,不是人工标注的黄金集(订正 13-I)。**")
    add("")
    add(f"`evals/topic/topic_test.jsonl` 的 {len(rows)} 行里,只有 **{human_reviewed} 行**带着"
        "人工复核痕迹(`human_reviewed`,来自 CP-2 那 84 条抽审流过来的);"
        f"其余 **{len(rows) - human_reviewed} 行**只有模型预标 + 证据串。"
        f"(构成:{provenance.get('real', 0)} 真实 + {provenance.get('synthetic', 0)} 合成。)")
    add("")
    add("⇒ 下面这个 F1 **不是**「在人工标注的黄金集上测出来的」,它是"
        "**「模型输出 vs 预标标签」的一致度** —— 预标自己错了多少,这里看不出来。")
    add("")
    add("**(b) `†` 会铺满几乎整张表 —— 那是算术,不是模型的病(订正 10)。**")
    add("")
    add(f"`†` 的判据是 spec §8.3 的 `support < {SUPPORT_MIN}`。17 类 × {SUPPORT_MIN} ="
        f" **{len(labels) * SUPPORT_MIN}** 个标签槽,而整份测试集一共只有"
        f" **{sum(len(t) for t in y_true)}** 个 ⇒ **算术上装不下**「17 类各 ≥{SUPPORT_MIN}」。"
        "实测支撑数:")
    add("")
    add(markdown_table(
        ["口径", "行数", "标签槽", f"够 {SUPPORT_MIN} 的类"],
        [[name, str(support_by_def[key]["rows"]), str(support_by_def[key]["slots"]),
          f"**{support_by_def[key]['enough']} / {len(labels)}**"]
         for key, name, _ in defs],
    ))
    add("")
    add(f"⇒ **不许把阈值从 {SUPPORT_MIN} 调低来迁就数据**(那等于把判据换成一个让表格好看的数字);"
        f"也**不许**把带 `†` 的那 {len(flagged)} 行读成「模型在这 {len(flagged)} 类上不行」——"
        f"它们说的是**每一类的 F1 都没有统计基础**(support 不够,置信区间比差值还宽),"
        "所以本报告里**没有一条**可以按类下结论;能引用的只有全体/分层那几个汇总数。")
    add("")
    add("**(c) 「只看真实 80」那一列才是这个模型在真机上的预期表现(spec §8.4)。**")
    add("")
    add("合成语料即便人工核过,它**与训练集同源**,同源样本上的表现天然偏高。"
        "三个口径并排看,只有 `real` 那一列能外推。")
    add("")

    add("## 1. 每类 P / R / F1 + support(17 行)")
    add("")
    add(f"> `†` 样本数不足(support < {SUPPORT_MIN}),该行 F1 不成结论(spec §8.3)。"
        f"**本表 {len(flagged)} / {len(labels)} 行带 `†`。**")
    add("")
    add(markdown_table(
        ["类目", "support", "TP", "FP", "FN", "精确率", "召回率", "F1", ""],
        [[r["label"], str(r["support"]), str(r["tp"]), str(r["fp"]), str(r["fn"]),
          f4(r["precision"]), f4(r["recall"]), f4(r["f1"]),
          "†" if r["support"] < SUPPORT_MIN else ""] for r in per_class],
    ))
    add("")
    add(f"**`†` 的完整类目清单({len(flagged)} 个)**:{'、'.join(flagged) if flagged else '(无)'}")
    add("")
    add(f"**够 {SUPPORT_MIN} 的类目({len(unflagged)} 个)**:"
        f"{'、'.join(unflagged) if unflagged else '(无)'}")
    add("")

    add("## 2. micro-F1 / macro-F1")
    add("")
    cell = grid[str(threshold)]["all"]
    add(markdown_table(
        ["指标", "值", "被谁主导"],
        [["**micro-F1**", f4(cell["micro_f1"]), "**大类**:17 类的 TP/FP/FN 汇成一个池子算一次"],
         ["macro-F1", f4(cell["macro_f1"]), "**小类**:每类 F1 的**未加权**平均"]],
    ))
    add("")
    smallest = min(per_class, key=lambda r: (r["support"], r["label"]))
    biggest = max(per_class, key=lambda r: (r["support"], r["label"]))
    add("两者的差别是**权重口径**不同,不是「两个不同的准确率」:micro 里大类(本报告里是"
        f"`{biggest['label']}`,support {biggest['support']}`)的每一条都进同一个池子,"
        "所以它压得住小类的抖动;macro 里每类**等权**,"
        f"于是 `{smallest['label']}`(support {smallest['support']})与"
        f"`{biggest['label']}`(support {biggest['support']})说话一样响。"
        "spec §7.4 因此拿 micro 早停 —— 验证集每类仅约 7 条,拿 macro 早停等于让噪声决定什么时候停。")
    add("")

    add("## 3. 两条直译需求原话的指标(§6.5)")
    add("")
    add(markdown_table(
        ["指标", "值", "读法"],
        [["**整条完全一致率**(subset accuracy)", f4(cell["subset_accuracy"]),
          "标签**集合**完全相同 —— 「一个不多一个不少」的**最严**读法"],
         ["**标签个数完全一致率**", f4(cell["label_count_match"]),
          "**个数**对就算对,**内容可以全错** —— 需求原话的**字面**读法"]],
    ))
    add("")
    add("**两个都要报,不要合并**(§6.5):个数一致率天然高于(或等于)整条一致率,"
        "只报前者会把「认错了类但个数没错」读成「判对了」。")
    add("")
    add(f"本轮判错的样本共 **{len(wrong)} 条**,最多 {ERROR_SAMPLE_LIMIT} 条导进 "
        f"[`misjudged.csv`](misjudged.csv)(实际导出 **{len(misjudged)} 条**,"
        f"列:原话 / 真实标签 / 预测标签 / 预标的证据串)。")
    add("")
    add("**那份 CSV 是给「人」判 `谁错了` 用的(spec §8.5)** —— 逐条判:"
        "**模型错**(标签对、模型没学到位)归模型的账;**标签错**(模型对、预标标错)归数据的账。"
        "**这个区分是本次复核的全部价值**:如果判错里**多数是标签错**,那 F1 低就不是模型的问题,"
        "要修的是数据不是训练 —— 而只看一个 F1 数字**永远分不出来**。")
    add("")
    add("⚠️ 本脚本**不预判**「谁错」:它只把现场摆好(原话 + 两边标签 + 预标当时的证据串)。"
        "证据串取自**这一行自己**的 `evidence` 字段;取不到的条目**留空**,不编。")
    add("")

    add("## 4. 两张矩阵(§8.2)")
    add("")
    add("**(1) 每类一张 2×2**(`matrix_confusion.csv`) —— TP / FP / FN / TN。"
        "它是 §1 那张表 P/R/F1 的**来源**,读数对得上。`tn` 是推出来的:`n − tp − fp − fn`。")
    add("")
    add("**(2) 一张 17×17 的「误判流向矩阵」**(`matrix_flow.csv`) —— 行 = 真实标签,"
        "列 = 预测标签,格 = 「本该是 i、却被判成 j」的次数。")
    add("")
    add("> ⚠️ **它刻意不叫「混淆矩阵」。** 单标签的混淆矩阵是「真实类 × 预测类」;"
        "多标签**没有唯一的预测类**,直接画 17×17 会得到一张人人能画、谁也读不懂的图。"
        "所以这里只记**认错**(真值里有 `i`、预测里没有 `i`、却多了个真值里没有的 `j`),"
        "**不记漏召回**(真值 `[退换货, 尺码]` 预测 `[尺码]` 时,`(退换货,尺码)` 这一格是 0)。"
        "—— 名字与语义不符是本仓吃过亏的地方(`agent_steps` 读作「步数」,实际是轮次序号)。")
    add("")
    add("这一眼能看出哪两类最容易混(§4.1 三条边界的验收集:「运费↔物流」、"
        "「退换货↔保修维修」)。本表在前 12 个非零格:")
    add("")
    top = sorted(((v, i, j) for (i, j), v in flow.items() if v), key=lambda x: (-x[0], x[1], x[2]))
    if top:
        add(markdown_table(["次数", "真实", "被判成"],
                           [[str(v), i, j] for v, i, j in top[:12]]))
    else:
        add("**(一个非零格都没有 —— 120 条全部判对。)**")
    add("")
    add(f"非零格共 **{len(top)}** 个;整张表 17×17 = **{len(labels) ** 2}** 格。")
    add("")

    add("## 5. 三个口径并排(spec §8.4)")
    add("")
    add(f"阈值 `t = {threshold}`(训练产物里的基线)。")
    add("")
    add(markdown_table(
        ["指标", "全体 " + str(grid[str(threshold)]["all"]["n"]),
         "**只看真实** " + str(grid[str(threshold)]["real"]["n"]),
         "只看合成 " + str(grid[str(threshold)]["synthetic"]["n"])],
        [["micro-F1"] + [f4(grid[str(threshold)][k]["micro_f1"]) for k in ("all", "real", "synthetic")],
         ["macro-F1"] + [f4(grid[str(threshold)][k]["macro_f1"]) for k in ("all", "real", "synthetic")],
         ["整条完全一致率"] + [f4(grid[str(threshold)][k]["subset_accuracy"])
                          for k in ("all", "real", "synthetic")],
         ["标签个数完全一致率"] + [f4(grid[str(threshold)][k]["label_count_match"])
                            for k in ("all", "real", "synthetic")]],
    ))
    add("")
    add("**「只看真实 80」那一列才是这个模型在真机上的预期表现。**")
    add("")
    add("> ⚠️ **这一行的 macro-F1 三列不可互比(订正 13-J)。** `macro_f1` 的分母是"
        "**传进来的那张 17 类表**,不是「出现过的类」⇒ **support = 0 的类贡献 `0.0`**,"
        "而合成那 40 行里一类都够不上 15、真实那 80 行里也只有 1 类够。"
        "⇒ 三个 macro 之间的差**同时**包含「模型变差」与「分母里多了几个零」两件事,"
        "而这张表分不出来 —— 别把它读成「模型在合成子集上更差」。")
    add("")

    add("## 6. 阈值:冻结测试集上的两行(订正 13-H)")
    add("")
    add(f"`t = {threshold}` 是训练产物里的基线(§7.6,`inference_config.json`);"
        f"`t = {cli.alt_threshold}` 是 `train_meta.json` 里 val 扫描的赢家。")
    add("")
    add(markdown_table(
        ["阈值", "口径", "n", "micro-F1", "macro-F1", "整条一致率", "个数一致率"],
        [[f"t={t}", name, str(grid[str(t)][key]["n"]), f4(grid[str(t)][key]["micro_f1"]),
          f4(grid[str(t)][key]["macro_f1"]), f4(grid[str(t)][key]["subset_accuracy"]),
          f4(grid[str(t)][key]["label_count_match"])]
         for t in (threshold, cli.alt_threshold) for key, name, _ in defs],
    ))
    add("")
    d_all = grid[str(cli.alt_threshold)]["all"]["micro_f1"] - grid[str(threshold)]["all"]["micro_f1"]
    add(f"**在冻结测试集上,`{cli.alt_threshold}` 相对 `{threshold}` 的 micro-F1 差是 "
        f"`{d_all:+.4f}`。**")
    add("")
    v_half = sweep.get(f"{threshold:.1f}", {}).get("micro_f1")
    v_alt = sweep.get(f"{cli.alt_threshold:.1f}", {}).get("micro_f1")
    if v_half is not None and v_alt is not None:
        add(f"作为对照,同一对数在 **val** 上是 `{v_alt:.4f} − {v_half:.4f} = "
            f"{(v_alt - v_half) * 100:+.2f}` 点(`train_meta.json` 的 `threshold_sweep`)。"
            f"⚠️ **`{cli.alt_threshold}` 在 val 上高 {(v_alt - v_half) * 100:.2f} 点,"
            "但 val 同时是「用来挑这个阈值」的那个集合** —— 在同一批行上挑出来的增益"
            "**没有判别力**(挑最大的总会挑到一个非负的差)。")
        add("")
    add("测试集**没有参与过任何选择**(切分 / 早停 / 阈值扫描全在 train / val 上),"
        "所以上面那一行**不是数据窥探**,是「`0.3` 到底真不真好」的无偏读数;"
        "而它**不被采用** —— 本轮不改 `THRESHOLD`(仍是产物里的 "
        f"`{threshold}`),**更不重训**:阈值是**纯后处理**,重训会换掉权重、让 "
        "`train_meta.json` 那份已被复核过的四指标**全部作废**,而**没有任何东西会报错**。")
    add("")

    add("## 7. 训练集标签错误率(CP-2,§6.3)")
    add("")
    add("spec §6.3 的判据是「抽审错误率 > 10% ⇒ 不训练」。CP-2 的读数是"
        f" **{cp2['changed']} / {cp2['reviewed']} = 0.0%**({cp2['note']}),"
        "远在门槛之内 ⇒ 已据此开训。")
    add("")
    add("⚠️ **这个 `0.0%` 撑不起「训练标签的错误率是本章 F1 的已知上界」那句话** —— "
        "把它印在上面的 F1 旁边会让人读成「F1 被一个零错误的标签集兜着」。"
        "它实际能代表的只有下面这三条限定之内的事:")
    add("")
    add("1. 它是一次**「看图通过」,没有逐条的核对痕迹** —— 产物里没有「哪一条对过、"
        "哪一条没对过」的记录。`0.0%` 是「没标错」,不是「逐条验过」,也不可复现、不可分解。")
    add("2. 样本构成**偏向合成**:84 条里 `pool 2 + chat 4 + evalmd 19 + gen 59`"
        " ⇒ 按「人写 / 模型生成」分是 **25 : 59**。")
    add("3. 按**更严的口径**(只算真实使用产生的 `pool + chat`,6 条),17 类里有 "
        "**13 类一条都没有**。")
    add("")
    add("⇒ 这个 `0.0%` 主要在说「**模型对合成句的预标,人看着没毛病**」;"
        "它**不是**这条流水线的性质,更**不是**本章 F1 的上界。"
        "(spec §6.3 抽审只覆盖 9%,剩余 875 条无人复核。)")
    add("")

    add("## 8. 出处与复现")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 权重目录 | `{Path(cli.model_dir).as_posix()}` |")
    add(f"| 训练时间 | `{tm.get('trained_at_utc')}` |")
    add(f"| 基座 / 种子 | `{tm.get('base')}` / `{tm.get('seed')}` |")
    add(f"| 最优 epoch / step | `{tm.get('best_epoch')}` / `{tm.get('best_step')}` |")
    add(f"| 数据指纹(训练集) | `{tm.get('data_fingerprint')}`({tm.get('data_fingerprint_algorithm')}) |")
    add(f"| 指纹的范围与强弱 | `{tm.get('data_fingerprint_scope')}` —— {tm.get('data_fingerprint_note')} |")
    add(f"| 训练 / 验证行数 | `{tm.get('data_rows')}` / `{tm.get('data_val_rows')}` |")
    add(f"| 测试集 | `{Path(cli.test).as_posix()}`({len(rows)} 行) |")
    add(f"| 测试集 sha256(盘上字节) | `{test_sha}` |")
    add(f"| 设备 | `{cli.device}`(见模块 docstring:CPU 是为了逐字节可复现) |")
    add("")
    add("附件:`report.json`(机器可读)、`matrix_confusion.csv`、`matrix_flow.csv`、"
        f"`misjudged.csv`(§8.5 的人工复核现场,{len(misjudged)} 条)。")
    add("")

    report_path.write_text("\n".join(md), encoding="utf-8")

    print(f"rows={len(rows)} device={cli.device} threshold={threshold} alt={cli.alt_threshold}")
    for t in (threshold, cli.alt_threshold):
        for key, name, _ in defs:
            c = grid[str(t)][key]
            print(f"  t={t} {name:<12} n={c['n']:<4} micro={c['micro_f1']:.4f} "
                  f"macro={c['macro_f1']:.4f} subset={c['subset_accuracy']:.4f} "
                  f"count={c['label_count_match']:.4f}")
    print(f"flagged(support<{SUPPORT_MIN}) {len(flagged)}/{len(labels)}: {'、'.join(flagged)}")
    print(f"wrong_rows={len(wrong)} flow_nonzero_cells={len(top)} "
          f"misjudged_rows={len(misjudged)}")
    print(f"wrote {report_path.as_posix()} {json_path.as_posix()} "
          f"{conf_path.as_posix()} {flow_path.as_posix()} {misjudged_path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
