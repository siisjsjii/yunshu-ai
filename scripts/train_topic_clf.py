"""ch10-B Task 9:全参微调 RoBERTa-wwm-ext,训一个 17 类**多标签**主题分类器。

```
.venv/Scripts/python.exe scripts/train_topic_clf.py
```

产物 `models/topic-clf/`(**gitignore**):权重 + `config.json` + tokenizer +
`labels.json` / `inference_config.json` / `train_meta.json`。

## 三条不许动的东西

1. **不许读冻结的测试集。** 那是验收 ① 的唯一依据,而且它的标签是预标 + 人工裁决的
   成品 —— 训练读它一眼都算污染。本脚本只读训练集与验证集两个路径(见模块底的常量),
   `tests/test_topic_model.py` 有一条源码扫描钉着它。
2. **`val.jsonl` 只读。** 增强只扩训练集(spec §5.4);扩了验证集,§8 那套指标就没意义了。
3. **不许改语料来「修」指标。** 语料里有一行零标签(`r-0049`「你是」,`labels=[]`)——
   它如实进了训练,`encode_rows` 把它编成全零向量。删掉它会让 `train_meta.json`
   里的行数与产物对不上,而那正是「这份权重是哪份数据训的」那条凭据要防的事。

## 三条防线(§7.1),都在下面看得见

① `from_pretrained(..., problem_type="multi_label_classification")` —— **显式传,不靠猜**;
② `encode_rows` 编出的是 **float32 多热**(见 `app/topic/model.py`);
③ `tests/test_topic_model.py` 里有一条**真的跑 forward** 的用例:
   loss 逐位等于手算的 `binary_cross_entropy_with_logits`。

⚠️ ②才是**第一跳**就拦得住的那道:那个「从 `labels.dtype` 猜任务」的猜测一旦写进
config 就锁死,而它写进去的时间点是**第一个 batch**。详见 spec §2.3 与本仓
`tests/test_topic_model.py` 末尾**四条**用例(其中一条**订正了 spec 的那句话**:
17 列 long 的多热标签**不是静默的,是当场抛**;真正静默的是 1-D long)。

## 12-C:产物里那三列是**原句的快照**

`train_augmented.jsonl` 的增强行是从原件 `{**r}` 继承来的,所以它们的
`evidence` / `rejected_labels` / `parse_failed` 指的是**原件那句题面**,而增强行
的题面**已经换过**。⇒ 本脚本**不读那三列**(`encode_rows` 只用 `question` 与
`labels`)。危险动作是「拿 `evidence` 去核对增强行的题面」——那会得到一个**自信的错答案**。

⚠️ 同理(12-D):产物里 `parse_failed` **全 `False`**,那**不是**这一轮的读数
(增强行继承原值,包括这一轮真解析失败的那条)。今天无害,因为没人读它。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# `python scripts/x.py` 时 sys.path[0] 是 **scripts/**、不是仓库根,所以要先补上
# (与 build_kb.py / gen_topic_data.py / calibrate_evidence.py 同款)。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from app.topic.metrics import label_count_match, macro_f1, micro_f1, subset_accuracy
from app.topic.model import (
    FINGERPRINT_ALGORITHM,
    TopicDataset,
    data_fingerprint,
    encode_rows,
    ensure_label_metadata,
    label_metadata,
    load_rows,
    save_artifacts,
)
from app.topic.taxonomy import LABELS

BASE = "hfl/chinese-roberta-wwm-ext"
#: spec §7.3:实测问题中位 16 字 / 最长 31 字 / ≥48 字 0 条 ⇒ 64 绰绰有余。
MAX_LEN = 64
SEED = 42
#: spec §7.6:全局单阈值基线。**不做每类阈值** —— 每类 ~7 条上调的是噪声。
THRESHOLD = 0.5
#: spec §7.6:验证集上做一次 0.3–0.7 扫描,表进 `train_meta.json`(Task 10 抄进报告)。
SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7)

#: ⚠️ **本模块只读这两个路径。** 冻结的测试集**一个字都不读** —— 见模块 docstring 第 1 条。
TRAIN_PATH = Path("evals/topic/train_augmented.jsonl")
VAL_PATH = Path("evals/topic/val.jsonl")
OUT_DIR = Path("models/topic-clf")


def metrics_from_logits(logits, label_matrix, threshold: float = THRESHOLD) -> dict:
    """`logits` + 真值多热 → 四个指标。**训练早停与评测报告的唯一口径。**

    ⚠️ 这里**只做一次** sigmoid + 比较(`micro_f1` / `macro_f1` / `subset_accuracy`
    / `label_count_match` 都来自 `app.topic.metrics`,没有第二份实现)。
    阈值扫描也走这个函数 —— 否则「0.5 那一行」与「训练报的 val_metrics」
    会是两次独立计算,对不上时没人知道信哪个。
    """
    probs = torch.sigmoid(torch.as_tensor(np.asarray(logits), dtype=torch.float32)).numpy()
    hits = probs >= threshold
    y_pred = [[LABELS[j] for j in range(len(LABELS)) if hits[i][j]] for i in range(hits.shape[0])]
    y_true = [
        [LABELS[j] for j in range(len(LABELS)) if label_matrix[i][j] >= 0.5]
        for i in range(len(label_matrix))
    ]
    return {
        "micro_f1": micro_f1(y_true, y_pred, LABELS),
        "macro_f1": macro_f1(y_true, y_pred, LABELS),
        "subset_accuracy": subset_accuracy(y_true, y_pred),
        "label_count_match": label_count_match(y_true, y_pred),
    }


def annotate_metrics(eval_result: dict, *, epoch, checkpoint) -> dict:
    """给 `Trainer.evaluate()` 的返回值**补出处**,并**删掉那个会撒谎的 `epoch`**。

    ⚠️ **订正轮 1 · M1**:`evaluate()` 返回的 `epoch` 是 `trainer.state.epoch`,
    对 trainer 而言那是**最后一轮**(实测 15);而这批数来自
    `load_best_model_at_end` 载回的那个 checkpoint(**实测第 13 轮**,
    用 `eval_loss 0.07422567` 与 `log_history` 对上了)。
    ⇒ 原样落进 `train_meta.json` 会让 `best_epoch: 13.0` 与
    `val_metrics.epoch: 15.0` **并排却矛盾** —— 而两个数都「看起来有出处」。
    """
    out = {key: value for key, value in eval_result.items() if key != "epoch"}
    out["_metrics_epoch"] = epoch
    out["_metrics_checkpoint"] = checkpoint
    return out


def repair_train_meta(out_dir) -> bool:
    """把 `train_meta.json` 的 `val_metrics` 修成 `annotate_metrics` 的形状。**幂等。**

    与 `ensure_label_metadata` 同一类活:**纯元数据缺陷,权重一个字节都不动**,
    所以对一份已经训好的产物补它**不需要重训**(重训会让已复核的读数全部作废)。

    ⚠️ 它**只动 `val_metrics` 里的那三个键**(去掉会撒谎的 `epoch`,补上
    `_metrics_epoch` / `_metrics_checkpoint`),**指纹 / 行数 / 四个指标 /
    超参一个都不碰**。
    """
    path = Path(out_dir) / "train_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在")
    meta = json.loads(path.read_text(encoding="utf-8"))
    fixed = annotate_metrics(meta["val_metrics"], epoch=meta.get("best_epoch"),
                             checkpoint=meta.get("best_model_checkpoint"))
    if fixed == meta["val_metrics"]:
        return False
    meta["val_metrics"] = fixed
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def make_compute_metrics(label_matrix):
    """`Trainer` 要的 `compute_metrics` —— 只是把真值多热绑上去。"""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return metrics_from_logits(logits, np.asarray(labels))

    return compute_metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="ch10-B 主题分类器全参微调")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--epochs", type=int, default=15, help="配合早停,见 spec §7.3")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="⚠️ CUDA OOM 时降这个(先试 16);**不要**动 max_length —— 它由实测句长定的")
    ap.add_argument("--train", default=str(TRAIN_PATH))
    ap.add_argument("--val", default=str(VAL_PATH))
    #: 纯元数据修复口:只读 `out_dir/config.json`,**不加载模型、不碰权重**。
    #: 给「已经训好的产物补 `id2label`」用(订正轮 1 · I2)——
    #: 那类缺陷不需要重训,而重训会让已复核的读数全部作废。
    ap.add_argument("--fix-metadata-only", action="store_true",
                    help="只补 out_dir 的元数据(config.json 的标签映射 + train_meta.json 的"
                         " val_metrics 出处),不加载模型、不碰权重,然后退出")
    cli = ap.parse_args()

    out_dir = Path(cli.out_dir)
    if cli.fix_metadata_only:
        print(f"ensure_label_metadata changed={ensure_label_metadata(out_dir, labels=LABELS)}")
        print(f"repair_train_meta     changed={repair_train_meta(out_dir)}")
        return 0

    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
          f"cuda={torch.version.cuda} n_gpu={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"gpu0={torch.cuda.get_device_name(0)}")

    set_seed(SEED)

    train_rows = load_rows(cli.train)
    val_rows = load_rows(cli.val)
    # ⚠️ 这两行是「这份权重是哪份数据训的」的全部依据。指纹**可能代表不了**语料本身
    #    (T8 复审实测:增强产物不可复现)⇒ 行数必须跟着一起记。
    fp = data_fingerprint(cli.train)
    n_augmented = sum(1 for r in train_rows if r.get("augmented"))
    n_unique_ids = len({r["id"] for r in train_rows})
    print(f"train_rows={len(train_rows)} train_unique_ids={n_unique_ids} "
          f"augmented_rows={n_augmented} val_rows={len(val_rows)}")
    print(f"data_fingerprint={fp} ({FINGERPRINT_ALGORITHM})")
    if n_unique_ids and len(train_rows) - n_augmented != n_unique_ids:
        # 12-B 的那条不变量:行数 = 原件数 + 增强数,而**原件数 == 唯一 id 数**。
        # 对不上 ⇒ 语料被按 id 塌过、或增强行丢了 id 前缀,两种都要人看一眼。
        print(f"!!! WARNING: len(rows)-augmented({len(train_rows) - n_augmented}) "
              f"!= unique_ids({n_unique_ids}) —— 语料形状与 12-B 的记录不符,先查再训")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    train_encoded = encode_rows(train_rows, tokenizer, max_length=MAX_LEN)
    val_encoded = encode_rows(val_rows, tokenizer, max_length=MAX_LEN)
    val_labels = np.stack([r["labels"].numpy() for r in val_encoded])

    # spec §7.3 的 warmup_ratio=0.1 在 v5 里只能写成绝对步数(见下面 TrainingArguments 的注释)。
    steps_per_epoch = math.ceil(len(train_encoded) / cli.batch_size)
    warmup_steps = int(0.1 * steps_per_epoch * cli.epochs)
    print(f"steps_per_epoch={steps_per_epoch} total_steps={steps_per_epoch * cli.epochs} "
          f"warmup_steps={warmup_steps} (v5 无 warmup_ratio,由 0.1 换算)")

    # ⚠️ `problem_type` **显式传**,不靠 transformers 从 labels.dtype 猜 ——
    #    那个猜测在第一个 batch 上写进 config 就锁死(§2.3)。
    # ⚠️⚠️ **订正轮 1 · I2:`id2label` / `label2id` 也必须显式传。** 不传的话
    #    `save_model` 落下来的 config.json 里是 `LABEL_0…LABEL_16`(实测),
    #    而**加载不报错** ⇒ 任何走 `config.id2label` 的代码会把 17 个类目全叫
    #    `LABEL_5`,scores 正常、写库成功、页面画得出来、日志里没有一行不对。
    _label_meta = label_metadata(LABELS)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(LABELS), problem_type="multi_label_classification",
        id2label=_label_meta["id2label"], label2id=_label_meta["label2id"],
    )

    args = TrainingArguments(
        output_dir=str(out_dir / "_ckpt"),
        learning_rate=2e-5,
        weight_decay=0.01,
        # ⚠️ **v5 把 `warmup_ratio` 整个删掉了**(实测:全仓 `grep -r warmup_ratio
        #    transformers/` 一个命中都没有,连弃用别名都没有)⇒ 只能给绝对步数。
        #    spec §7.3 的「warmup_ratio = 0.1」在这里换算成 0.1 × 总训练步数,
        #    换算过程与结果写进 `train_meta.json`(见 `warmup_steps`)。
        warmup_steps=warmup_steps,
        per_device_train_batch_size=cli.batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=cli.epochs,
        # ⚠️ v5 叫 `eval_strategy`,**不是** v4 的 `evaluation_strategy`。
        eval_strategy="epoch",
        # ⚠️ 已核源码 training_args.py:1718 —— `load_best_model_at_end=True` 时
        #    它必须**等于** eval_strategy,否则直接抛错。
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # ⚠️ 盯 **micro-F1**,不盯 macro:验证集每类仅 ~7 条,
        #    拿 macro 早停等于让噪声决定什么时候停(spec §7.4)。
        #    已核 trainer.py:3293-3301 —— 它会去找 `eval_micro_f1`,正是
        #    `compute_metrics` 返回的那个键名。
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_steps=20,
        seed=SEED,
        report_to=[],
        # ⚠️ 关掉它(实测订正:「不关就 AttributeError」在本机 5.17.0 上**不成立**,
        #    见 `app/topic/model.py` 的 `TopicDataset` docstring)。仍然关掉是因为:
        #    它的作用是「按模型 forward 的签名删列」,而我们的三个列名是我们自己定的,
        #    没有任何理由让这一层去动它们 —— 少一处可能被误删的地方。
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=TopicDataset(train_encoded),
        eval_dataset=TopicDataset(val_encoded),
        # ⚠️ v5 叫 `processing_class`,**不是** v4 的 `tokenizer`。
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(val_labels),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3,
                                         early_stopping_threshold=0.001)],
    )

    started = time.time()
    trainer.train()
    train_seconds = time.time() - started

    # 最优 epoch / 步数:从 log_history 里取 micro-F1 最高的那一行。
    # ⚠️ **不拿 `state.epoch`** —— 那是「最后一轮」,而早停之下最后一轮往往不是最好的。
    evals = [h for h in trainer.state.log_history if "eval_micro_f1" in h]
    best = max(evals, key=lambda h: h["eval_micro_f1"]) if evals else {}

    val_metrics = annotate_metrics(trainer.evaluate(), epoch=best.get("epoch"),
                                   checkpoint=trainer.state.best_model_checkpoint)
    # spec §7.6:阈值扫描。走**同一个** `metrics_from_logits`,所以「0.5 那一行」
    # 与上面的 `val_metrics` 必然一致(不一致就是这里错了,不是巧合)。
    sweep = {f"{t:.1f}": metrics_from_logits(
        trainer.predict(TopicDataset(val_encoded)).predictions, val_labels, t) for t in SWEEP}

    # 权重 + config.json。`save_artifacts` 只写那三个 json 与 tokenizer。
    trainer.save_model(str(out_dir))
    # 兜底:即使 `from_pretrained` 那两行被人挪掉,标签映射也必须与 `labels.json` 逐位一致。
    # 幂等 ⇒ 正常情况下这里是 no-op(返回 False)。
    print(f"ensure_label_metadata changed_config={ensure_label_metadata(out_dir, labels=LABELS)}")
    save_artifacts(
        out_dir, tokenizer=tokenizer, max_length=MAX_LEN, threshold=THRESHOLD, labels=LABELS,
        meta={
            "base": BASE,
            "seed": SEED,
            "max_length": MAX_LEN,
            "threshold": THRESHOLD,
            "hyperparams": {
                "learning_rate": 2e-5, "weight_decay": 0.01,
                # ⚠️ spec §7.3 写的是 `warmup_ratio=0.1`,而 v5 把这个参数**删了**
                #    (全仓 grep 零命中,连弃用别名都没有)⇒ 这里落的是换算结果,
                #    并把换算依据一起记下来,免得后人以为我们改了超参。
                "warmup_steps": warmup_steps,
                "warmup_ratio_equivalent": 0.1,
                "warmup_note": "transformers 5.17.0 无 warmup_ratio;warmup_steps = int(0.1*steps_per_epoch*epochs)",
                "steps_per_epoch": steps_per_epoch,
                "per_device_train_batch_size": cli.batch_size,
                "per_device_eval_batch_size": 64, "num_train_epochs": cli.epochs,
                "early_stopping_patience": 3, "early_stopping_threshold": 0.001,
                "metric_for_best_model": "micro_f1",
            },
            # --- 数据凭据:指纹只能代表**一次运行**,所以行数与生成日期一起记(12-A)---
            "data_fingerprint": fp,
            "data_fingerprint_algorithm": FINGERPRINT_ALGORITHM,
            "data_fingerprint_scope": Path(cli.train).as_posix(),
            "data_fingerprint_note": (
                "T8 复审实测增强产物不可复现(改写那一半在 temperature=0 下仍不确定)"
                "⇒ 这个 sha256 是「一次运行」的指纹,不是「这份语料」的"
            ),
            "data_rows": len(train_rows),
            "data_augmented_rows": n_augmented,
            "data_unique_ids": n_unique_ids,
            "data_val_rows": len(val_rows),
            "data_file_mtime_utc": datetime.fromtimestamp(
                Path(cli.train).stat().st_mtime, tz=timezone.utc).isoformat(),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            # --- 读数 ---
            "device": str(trainer.args.device),
            "train_seconds": round(train_seconds, 1),
            "best_epoch": best.get("epoch"),
            "best_step": best.get("step"),
            "global_step": trainer.state.global_step,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "val_metrics": val_metrics,
            "threshold_sweep": sweep,
        },
    )

    print("=== val_metrics (threshold=%.1f) ===" % THRESHOLD)
    for k in ("micro_f1", "macro_f1", "subset_accuracy", "label_count_match"):
        print(f"  {k} = {val_metrics.get('eval_' + k)}")
    print("=== threshold sweep (micro_f1 / macro_f1 / subset_acc / count_match) ===")
    for t, m in sweep.items():
        print(f"  t={t}  {m['micro_f1']:.4f}  {m['macro_f1']:.4f}  "
              f"{m['subset_accuracy']:.4f}  {m['label_count_match']:.4f}")
    print(f"data_fingerprint={fp} device={trainer.args.device} "
          f"best_epoch={best.get('epoch')} best_step={best.get('step')} "
          f"train_seconds={train_seconds:.1f} out_dir={out_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
