"""模型侧可单测的那一半 —— **不发网络请求、不加载预训练权重**。

`scripts/train_topic_clf.py` 负责循环与早停;这里负责「把数据编成张量」、
「把产物写对」、「给语料算指纹」。那三件事各自有一个**静默失效**要防:

| 函数 | 要防的静默失效 |
|---|---|
| `encode_rows` | 标签 dtype 不对 ⇒ 被 `transformers` 猜成单标签交叉熵(§2.3) |
| `save_artifacts` / `load_artifacts` | 标签顺序两侧不一致 ⇒ 一张完全错的分布图,每个组件都正常 |
| `data_fingerprint` | 行尾参与哈希 ⇒ 指纹随 checkout 策略变,溯源凭据**安静地给错答案** |

⚠️ 本模块**不做清洗**。语料在 Task 3/7/8 已经洗过(`app/topic/clean.py`),
训练侧与推理侧同源那条不变量由 `tests/test_topic_clean.py` 的源码扫描守着,
在这里再洗一遍才是 train/serve skew。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from app.topic.taxonomy import LABELS

#: 指纹算法的自我描述 —— 写进 `train_meta.json`,让读的人不必猜是哪种哈希。
FINGERPRINT_ALGORITHM = "sha256(lf-normalized utf-8 bytes)[:16]"


def load_rows(path) -> list[dict]:
    """读一份 JSONL 语料,**逐行**,空行跳过。

    ⚠️ **返回的是 list,不是 dict —— 不许按 `id` 建索引。**
    见 `encode_rows` 的 docstring(12-B)。
    """
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def data_fingerprint(path) -> str:
    """对语料**内容**算 sha256 前 16 位 —— 「这份权重是哪份数据训的」的凭据。

    ⚠️ **指纹是对 LF 归一化后的文本算的**,不是对磁盘字节算的。
    实测:`evals/topic/train_augmented.jsonl` **盘上是 CRLF**(2315 个 CRLF、
    0 个裸 LF),而两种算法给出**不同**的值(`read_bytes()` → `2ca5aba59a0afc2f`,
    LF 归一化 → `74bd7d056f68b68c`)。照 `read_bytes()` 写,指纹就**随 checkout
    的行尾策略变化** —— 换台 `autocrlf=false` / Linux 的机器算出来是另一个值,
    后人要么以为语料被换了、要么放弃这条凭据。**它一坏就是「静默给一个错的
    溯源答案」**,正是本仓「静默无效」家族的长相。

    `Path.read_text` 走通用换行(CRLF/CR → LF),所以**这一行就是归一化**;
    不要去改成 `read_bytes()`。

    ⚠️ **这条凭据能代表的比看起来弱**:T8 复审实测,增强产物**不可复现**
    (改写那一半在 `temperature=0` 下仍不确定,连跑 3 次得 3 个样)⇒ 这个
    sha256 是「**一次运行**」的指纹,不是「这份语料」的。所以 `train_meta.json`
    里除了指纹还必须记**行数与生成日期**(见训练脚本的 `meta`)。
    """
    text = Path(path).read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def encode_rows(rows: list[dict], tokenizer, *, max_length: int) -> list[dict]:
    """把语料编成 `input_ids` / `attention_mask` / `labels`。

    ⚠️ **`labels` 必须是 float32 的多热向量。** `transformers` 在
    `problem_type` 为 None 时**看 dtype 猜任务**(本机 5.17.0,
    `models/bert/modeling_bert.py:1122-1141`):整数会被判成单标签分类,
    而那个猜测**在第一个 batch 上写进 config 就锁死**。显式传 `problem_type`
    是第二道防线(训练脚本里),这里是第一道 —— 也是唯一在**第一跳**就
    拦得住的那一道(见 `tests/test_topic_model.py` 末尾**四条**真跑 forward 的用例)。

    ⚠️ **逐行消费,不许按 `id` 去重 / join / 建索引。**
    `train_augmented.jsonl` 的 1157 条**增强行与它们的原件共用 `id`**
    (实测 2315 行 / 1158 个 id)。危险动作只有一句 `{r["id"]: r for r in rows}` ——
    它让 2315 行**静默塌成 1158**,增强全没了,而**指标照常打**(只是那些指标
    描述的是另一个训练集)。`tests/test_topic_model.py` 里有一条能红的断言钉着它。

    ⚠️ 也不许跳过 `labels` 为空的行:训练集里**真的**有这样一行
    (`r-0049`「你是」),它编成全零向量即可。丢行会让 `len(train)` 与产物对不上,
    而那是没人会看的数。

    ⚠️ 顺带一条**不许做的**:不许拿 `evidence` / `rejected_labels` 去核对增强行的
    题面 —— 那三列在增强行上是**原句的快照**(题面已换),会给出一个自信的错答案。
    """
    texts = [r["question"] for r in rows]
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True,
                    return_tensors="pt")
    out = []
    for i, row in enumerate(rows):
        vec = torch.zeros(len(LABELS), dtype=torch.float32)
        for label in row["labels"]:
            try:
                vec[LABELS.index(label)] = 1.0
            except ValueError as exc:
                raise KeyError(f"不存在的类目:{label}") from exc
        out.append({
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels": vec,
        })
    return out


class TopicDataset(torch.utils.data.Dataset):
    """把 `encode_rows` 的产物包成 `Trainer` 能吃的数据集。

    ⚠️ **实测订正**:计划的原文说「不包一层,`Trainer` 会读 `dataset.column_names`
    而普通 list 没有它 ⇒ 直接 `AttributeError`」。**在本机 5.17.0 上不成立**:
    `trainer.py:988` 的 `if is_datasets_available() and isinstance(dataset,
    datasets.Dataset)` 让 `column_names` 那条路**只对 `datasets.Dataset` 生效**,
    而裸 list / torch Dataset 走 `else`(`_get_collator_with_removed_columns`)。
    实测传 `list[dict]` 给 `Trainer` 能正常训完。

    这一层仍然保留,但理由换成两条**真的**成立的:

    ① 它是 `torch.utils.data.Dataset` 的**正经实现**(`DataLoader` 的语义就是这样),
       不靠鸭子类型;② 字段就是我们自己编的那三个,写在一处比散在脚本里好读。

    **它不是一条必需的错误规避** —— 别把它当防线,防线是 `labels` 的 dtype。
    """

    #: `Trainer` 会读它 —— 有了它,将来真换成 `datasets.Dataset` 时行为不变。
    column_names = ["input_ids", "attention_mask", "labels"]

    def __init__(self, encoded: list[dict]):
        self.rows = encoded

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def save_artifacts(out_dir, *, tokenizer, max_length: int, threshold: float,
                   labels, meta: dict) -> None:
    """写产物。**`labels.json` 是推理侧标签顺序的唯一来源。**

    服务读它、**不自己写一份** —— 两侧顺序不一致会产出一张完全错的分布图,
    而每个组件都工作正常(模型有输出、scores 在 0–1、写库成功、页面画得出来)。

    `tokenizer=None` 时不写 tokenizer(单测走这条路)。⚠️ `save_pretrained`
    会往 `out_dir` 写好几个文件,所以别指望 `out_dir` 里只有那三个 json。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.json").write_text(
        json.dumps(list(labels), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "inference_config.json").write_text(
        json.dumps({"max_length": max_length, "threshold": threshold},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if tokenizer is not None:
        tokenizer.save_pretrained(out)


def load_artifacts(model_dir) -> dict:
    """读产物。服务与评测脚本都走这里 —— **不要各自 `open()` 一遍**。"""
    d = Path(model_dir)
    return {
        "labels": json.loads((d / "labels.json").read_text(encoding="utf-8")),
        **json.loads((d / "inference_config.json").read_text(encoding="utf-8")),
        "meta": json.loads((d / "train_meta.json").read_text(encoding="utf-8")),
    }
