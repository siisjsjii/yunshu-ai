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


def label_metadata(labels) -> dict:
    """由 `labels` 生成 HF config 的三个标签键 —— **一处生成,两处用**。

    返回 `num_labels` / `id2label` / `label2id`。⚠️ `id2label` 的键是**字符串**
    (JSON 的键只能是字符串),所以它可以直接与从磁盘读回来的那份 `dict` 比相等。

    为什么要它:HF 保存的 `config.json` 里 `id2label` 默认是
    `{"0": "LABEL_0", ...}` —— 见 `ensure_label_metadata`。
    """
    return {
        "num_labels": len(labels),
        "id2label": {str(i): label for i, label in enumerate(labels)},
        "label2id": {label: i for i, label in enumerate(labels)},
    }


def ensure_label_metadata(out_dir, *, labels) -> bool:
    """把 `config.json` 的标签映射**补齐 / 纠正**成 `labels`。**幂等**,可对已有目录重跑。

    返回「这次有没有真的动过文件」(第二次调用必须是 `False`,而且字节不变)。

    ⚠️⚠️ **为什么需要它(订正轮 1 · I2)** —— 本仓「静默无效」家族的又一名成员:

    `AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=17)` 存下来的
    `config.json`,`id2label` 实测是 `{"0": "LABEL_0", …, "16": "LABEL_16"}`、
    `num_labels` 是 `null`。**加载不报错**(HF 用 `len(id2label)` 补出 17)。
    于是任何一段走 `model.config.id2label[j]` 或
    `pipeline("text-classification", model=…)` 的代码会拿到
    **17 个类目全叫 `LABEL_5`** —— 而 scores 正常、写库成功、页面画得出来、
    **日志里没有一行不对**。那正是 spec §2.7 逐字点名的陷阱。

    **纠正的是值,不只是缺键**:`data.get(k) == v` 比的是值 ⇒ 顺序错、对调过
    也会被改写回 `labels` 的顺序(`tests/test_topic_model.py` 有一条反向断言钉着)。

    ⚠️ 它是**幂等**的,所以「对一份已经训好的产物补元数据」不需要重训
    —— 纯元数据缺陷,权重一个字节都不动。
    """
    path = Path(out_dir) / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在 —— 先 save_model / save_pretrained 再来补标签映射")
    data = json.loads(path.read_text(encoding="utf-8"))
    want = label_metadata(labels)
    if all(data.get(key) == value for key, value in want.items()):
        return False
    data.update(want)
    # `sort_keys=True` + 末尾换行是与 HF 自己的 `to_json_string` 同款,
    # 免得下一次 `save_pretrained` 把整个文件重排一遍(那会让「改动前后 sha256」
    # 变得没法解释)。
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return True


def save_artifacts(out_dir, *, tokenizer, max_length: int, threshold: float,
                   labels, meta: dict) -> None:
    """写产物。**`labels.json` 是推理侧标签顺序的唯一来源。**

    服务读它、**不自己写一份** —— 两侧顺序不一致会产出一张完全错的分布图,
    而每个组件都工作正常(模型有输出、scores 在 0–1、写库成功、页面画得出来)。

    ⚠️ **不要用 `config.json` 的 `id2label` 拿类目名 —— 以 `labels.json` 为准。**
    那个 `id2label` 是给 HF 自己看的元数据,而它**可以是错的**(出厂默认
    `LABEL_0…`);`labels.json` 才是本仓的权威来源,`ensure_label_metadata`
    的职责只是把 config 对齐到它。

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
    """读产物。服务与评测脚本都走这里 —— **不要各自 `open()` 一遍**。

    ⚠️ **不要用 `config.json` 的 `id2label` 拿类目名 —— 以 `labels.json` 为准**
    (见 `save_artifacts` 与 `ensure_label_metadata` 的 docstring)。
    """
    d = Path(model_dir)
    return {
        "labels": json.loads((d / "labels.json").read_text(encoding="utf-8")),
        **json.loads((d / "inference_config.json").read_text(encoding="utf-8")),
        "meta": json.loads((d / "train_meta.json").read_text(encoding="utf-8")),
    }
