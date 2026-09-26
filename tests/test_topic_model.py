"""模型侧可单测的那一半:编码、产物读写、数据指纹。

⚠️ **单测全程不联网** —— 所以这里用一个**假的 tokenizer**,不加载 RoBERTa;
下面那几条**真的跑模型 forward** 的用例也用**本地现搭的小 `BertConfig`**
(`BertForSequenceClassification(BertConfig(...))` 是随机初始化,不碰 HF Hub)。
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

from app.topic.model import (
    TopicDataset,
    data_fingerprint,
    encode_rows,
    ensure_label_metadata,
    label_metadata,
    load_artifacts,
    load_rows,
    save_artifacts,
)
from app.topic.taxonomy import LABELS


class FakeTokenizer:
    """只做「按字符切」的假分词器 —— 测的是我们的编码逻辑,不是 HF 的。"""

    def __call__(self, texts, *, truncation, max_length, padding, return_tensors):
        import torch

        ids = [[min(ord(c), 100) for c in t[:max_length]] for t in texts]
        width = max(len(x) for x in ids)
        return {
            "input_ids": torch.tensor([x + [0] * (width - len(x)) for x in ids]),
            "attention_mask": torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids]),
        }


# ---------------------------------------------------------------- 编码


def test_labels_are_float_not_int():
    """⚠️ **这是本章最贵的一条断言。**

    `transformers/models/bert/modeling_bert.py:1122-1141`(本机 5.17.0 逐字)
    的原文:`problem_type` 为 None 时,它看 `labels.dtype` —— 整数(dtype 是
    long/int)会被判成 **`single_label_classification`**(softmax + 交叉熵),
    **float 才是多标签**(BCE)。而且这个猜测**在第一个 batch 上写进 config 就锁死**。

    后果(实测的那半,见本文件末尾那四条「真的跑一遍」的用例):17 列 long 的多热
    标签会让交叉熵**形状对不上而抛**;1-D long 则**真的会一路训下去**,
    模型被 softmax 推着**每句只报一个类**,而 loss 照常下降、训练看起来完全正常。
    不管哪一种,验收 ③ 都过不了。
    """
    rows = [{"question": "买大了想退", "labels": ["尺码", "退换货"]}]
    out = encode_rows(rows, FakeTokenizer(), max_length=16)
    assert out[0]["labels"].dtype.is_floating_point, "标签必须是 float —— 否则会被判成单标签交叉熵"


def test_encoding_is_a_multi_hot_over_LABELS_order():
    """多热向量的**位置必须由 `LABELS` 决定** —— 它是标签 id 的权威顺序。"""
    rows = [{"question": "买大了想退", "labels": ["尺码", "退换货"]}]
    vec = encode_rows(rows, FakeTokenizer(), max_length=16)[0]["labels"]
    assert vec[LABELS.index("尺码")] == 1.0
    assert vec[LABELS.index("退换货")] == 1.0
    assert float(vec.sum()) == 2.0


def test_the_vector_width_is_the_label_count_not_the_sentence_length():
    """反向断言:上面那条 `sum() == 2.0` 单独拿不出来 —— 一个**长度等于类目数**的
    向量才可能同时满足它;写成长度等于句长(或写死 2)的实现在这条上就红。"""
    vec = encode_rows([{"question": "买大了想退", "labels": ["尺码", "退换货"]}],
                      FakeTokenizer(), max_length=16)[0]["labels"]
    assert vec.shape == (len(LABELS),)


def test_unknown_label_raises_loudly():
    """不认识的类目**必须抛**,不许静默丢弃。

    静默丢弃的表现是「这条样本少了一个标签」,而它会让多标签样本
    系统性变少 —— 正是方案 A 花大力气造出来的那 30%。
    """
    with pytest.raises(KeyError, match="不存在的类目"):
        encode_rows([{"question": "x", "labels": ["尺码", "不存在的类目"]}],
                    FakeTokenizer(), max_length=16)


def test_encode_rows_never_dedupes_by_id():
    """⚠️ **12-B:不许按 id 去重 / join。**

    `train_augmented.jsonl` 的 **1157 条增强行与它们的原件共用 `id`**
    (实测:2315 行 / 1158 个 id)。今天安全,只是因为 `encode_rows` 逐行消费 ——
    **没有任何东西守着它**。

    危险动作只有一句:`{r["id"]: r for r in rows}`。它让 2315 行**静默塌成 1158**,
    增强全没了,而**指标照常打**(只是那些指标描述的是另一个训练集)。
    """
    rows = [
        {"id": "r-0001", "question": "买大了", "labels": ["尺码"]},
        {"id": "r-0001", "question": "买多了", "labels": ["尺码"]},  # 增强行:id 与上一行相同
        {"id": "r-0002", "question": "快递到哪了", "labels": ["物流"]},
    ]
    assert len(encode_rows(rows, FakeTokenizer(), max_length=16)) == 3, \
        "按 id 去重/join 会把增强行静默吃掉(2315 → 1158)"


def test_encode_rows_keeps_rows_with_no_labels():
    """训练集里**真的**有这样一行(`r-0049`「你是」,`labels=[]`)。

    它必须被编成一个全零向量(而不是抛、也不是被悄悄丢掉)——
    丢行会让 `len(train)` 与产物对不上,而那是没人会看的数。
    """
    out = encode_rows([{"id": "r-0049", "question": "你是", "labels": []}],
                      FakeTokenizer(), max_length=16)
    assert len(out) == 1
    assert float(out[0]["labels"].sum()) == 0.0
    assert out[0]["labels"].dtype.is_floating_point


def test_real_augmented_corpus_still_carries_duplicate_ids():
    """**从磁盘现算**,不是硬编码:产物里必须**仍有**共用 id 的增强行。

    这条守的是「产物被按 id 塌过之后再拿去训练」—— 那时上面那条合成用例
    照样绿(它测的是 `encode_rows`,不是语料)。
    """
    rows = load_rows("evals/topic/train_augmented.jsonl")
    assert len(rows) > len({r["id"] for r in rows}), "增强行丢了 —— 语料被按 id 塌过"


def test_train_script_never_reads_the_frozen_test_set():
    """⚠️ 冻结测试集是**验收 ① 的唯一依据**,训练**一个字都不许读**。

    源码扫描(与 ch09 那条「全章唯一的 langfuse 边界」同款):抓的是
    「有没有人写下这个路径」。它不证明训练没读,它证明**没有人写出那个路径** ——
    而这是这条红线今天唯一可自动化的形态。

    ⚠️ 因此这两个文件的注释里只说「冻结的测试集」,**不写文件名** ——
    写了就会把这条扫描变成一条自咬的断言(扫描命中自己那句「我们不读它」)。

    ⚠️⚠️ **订正轮 1 · M2:扫描面从 1 个文件扩到 2 个。** 第一版只扫
    `scripts/train_topic_clf.py`,而**真正的读盘口是 `app/topic/model.py::load_rows`**
    (训练脚本只是调它)—— 往 `load_rows` 里塞一个默认路径,这条扫描原来看不见。
    复审实测那条扫描**确实有牙**(把路径写进脚本就红,R1),只是覆盖比它读起来窄。
    ⚠️ **残余(如实记账,别假装已覆盖)**:`--train <path>` 是**命令行参数**,
    运行期传进来的路径扫不到;这条扫描守的是「**源码里有没有写死那个路径**」。
    """
    for rel in ("scripts/train_topic_clf.py", "app/topic/model.py"):
        assert "topic_test" not in Path(rel).read_text(encoding="utf-8"), rel


def test_train_script_passes_problem_type_explicitly():
    """spec §7.1①:基座加载时**显式**传 `problem_type`,不靠 transformers 猜。

    ⚠️ 脚本本身跑不进单测(要 400MB 权重 + 网络 + 好几分钟),所以这条是
    **源码扫描** —— 它抓的是「那一行还在不在」。这比没有强,但比行为断言弱:
    它拦不住「参数传了却被下一行覆盖」。(§7.1 的另外两道防线由本文件里
    `encode_rows` 的 dtype 断言与那四条真跑 forward 的用例守着。)

    ⚠️⚠️ **锚点必须锚在代码上,不能锚在注释上 —— 这条吃过一次。**
    第一版写的是 `assert 'problem_type="multi_label_classification"' in src`,
    而脚本的**模块 docstring 里就逐字引用了这半句**(「① `from_pretrained(...,
    problem_type="multi_label_classification")`」)⇒ 把代码那一行删掉,
    **扫描照样命中注释、测试照样绿**。变异探针 M14 实测抓到了这一次假绿。
    所以这里用正则**连着调用形状一起**要:`from_pretrained(\n BASE,\n
    num_labels=len(LABELS),\n problem_type=...` —— 注释里那半句匹配不上。
    """
    src = Path("scripts/train_topic_clf.py").read_text(encoding="utf-8")
    assert re.search(
        r"AutoModelForSequenceClassification\.from_pretrained\(\s*"
        r'BASE,\s*num_labels=len\(LABELS\),\s*problem_type="multi_label_classification"',
        src,
    ), "基座加载那一行必须显式传 problem_type(靠猜会锁死单标签交叉熵)"


def test_train_script_early_stops_on_micro_f1_not_macro():
    """spec §7.4:早停盯 **micro-F1**。验证集每类 ~7 条,macro 抖动极大,
    拿它早停等于**让噪声决定什么时候停**。

    同样只能源码扫描(跑一遍训练要几分钟 + 网络)。macro-F1 照常算、照常报告,
    但**不参与决策** —— 所以这条扫描只看 `metric_for_best_model` 那一行。
    """
    src = Path("scripts/train_topic_clf.py").read_text(encoding="utf-8")
    assert 'metric_for_best_model="micro_f1"' in src


def test_val_metrics_carry_the_checkpoint_they_came_from():
    """⚠️ **订正轮 1 · M1**:`Trainer.evaluate()` 返回的 `epoch` 是
    `trainer.state.epoch` = **最后一轮**,而这批数来自 `load_best_model_at_end`
    载回的那个 checkpoint ⇒ 落进 `train_meta.json` 后,
    `best_epoch: 13.0` 与 `val_metrics.epoch: 15.0` **并排却矛盾**
    (复审用 `eval_loss` 对上了:0.07422567 正是第 13 轮那一行)。
    ⇒ 那个键**必须**被换掉,并标出处。
    """
    from scripts.train_topic_clf import annotate_metrics

    fake = {"eval_loss": 0.07422567, "eval_micro_f1": 0.8804, "epoch": 15.0}
    out = annotate_metrics(fake, epoch=13.0, checkpoint="models/topic-clf/_ckpt/checkpoint-949")

    assert "epoch" not in out, "那个 epoch 是最后一轮、不是这批数的来源"
    assert out["_metrics_epoch"] == 13.0
    assert out["_metrics_checkpoint"].endswith("checkpoint-949")
    assert out["eval_micro_f1"] == 0.8804, "别的键一个都不许丢"
    assert out["eval_loss"] == 0.07422567


def test_repair_train_meta_drops_the_lying_epoch_and_is_idempotent(tmp_path):
    """与 `ensure_label_metadata` 同一类活:**纯元数据缺陷,不重训**。

    这条同时钉住「只动那三个键」—— 指纹 / 行数 / 四个指标 / 超参必须**原样**。
    """
    from scripts.train_topic_clf import repair_train_meta

    meta = {
        "base": "hfl/chinese-roberta-wwm-ext",
        "data_fingerprint": "74bd7d056f68b68c",
        "data_rows": 2315,
        "best_epoch": 13.0,
        "best_model_checkpoint": "models/topic-clf/_ckpt/checkpoint-949",
        "val_metrics": {"eval_loss": 0.0742, "eval_micro_f1": 0.8804, "epoch": 15.0},
    }
    (tmp_path / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    assert repair_train_meta(tmp_path) is True
    fixed = json.loads((tmp_path / "train_meta.json").read_text(encoding="utf-8"))

    assert "epoch" not in fixed["val_metrics"]
    assert fixed["val_metrics"]["_metrics_epoch"] == 13.0
    assert fixed["val_metrics"]["eval_micro_f1"] == 0.8804
    # 「只动那三个键」:别的一律原样
    assert fixed["data_fingerprint"] == "74bd7d056f68b68c"
    assert fixed["data_rows"] == 2315
    assert fixed["val_metrics"]["eval_loss"] == 0.0742

    assert repair_train_meta(tmp_path) is False, "第二次必须无事可做"
    assert json.loads((tmp_path / "train_meta.json").read_text(encoding="utf-8")) == fixed


def test_metrics_from_logits_honours_the_threshold_it_is_given():
    """阈值是训练早停(`compute_metrics`)与阈值扫描**共用的那一个参数**。

    这条钉的是它**真的被用上了**:一个把 `hits = probs >= 0.5` 写死、忽略
    `threshold` 的实现,会让 `train_meta.json` 里 0.5 那一行与训练报的
    `val_metrics` 悄悄脱钩 —— 而那两个数**本来就该逐位相同**,脱钩了没人看得出来。
    """
    from scripts.train_topic_clf import metrics_from_logits

    # ⚠️ 其余 15 类必须**明显低于**阈值:`sigmoid(0.0) == 0.5` **正好落在 0.5 那条线上**
    #    (第一次就是这么写错的 —— 实测拿到 1/9 而不是 2/3)。用 -5.0(sigmoid ≈ 0.0067)。
    logits = [[-5.0] * len(LABELS)]
    logits[0][LABELS.index("退换货")] = 0.2  # sigmoid ≈ 0.5498:过 0.5、不过 0.6
    logits[0][LABELS.index("物流")] = 1.0  # sigmoid ≈ 0.7311:两条都过
    truth = [[0.0] * len(LABELS)]
    truth[0][LABELS.index("退换货")] = 1.0  # 真值只要「退换货」

    # 阈值 0.6:只有「物流」过线 ⇒ tp=0 / fp=1 / fn=1 ⇒ F1 = 0
    assert metrics_from_logits(logits, truth, 0.6)["micro_f1"] == 0.0
    # 阈值 0.5:「退换货」也过线 ⇒ 多报了一个「物流」⇒ P=0.5 / R=1.0 ⇒ F1 = 2/3
    assert metrics_from_logits(logits, truth, 0.5)["micro_f1"] == pytest.approx(2 / 3)

    # ⚠️⚠️ **订正轮 1 · I1:两个键必须分别断言,而且要用「两个值不相等」的输入。**
    #    上面那条只断 `micro_f1`,而 `subset_accuracy` 与 `label_count_match` 的
    #    **装配处**(这个 dict 字面量)没有任何东西走到 —— 复审变异 R6 把这两个键
    #    **对调**,33 条测试全绿、exit 0。而它们在 `train_meta.json` 里互换之后
    #    (0.7655 ↔ 0.8345,都在合理区间),Task 10 会把「整条完全一致率」贴到
    #    「标签个数一致率」下面 —— **没有任何东西会红**,而这两个数正是 §8.1
    #    点名的那两句需求原话。
    #    这组输入:真值 {退换货, 尺码}、预测 {退换货, 物流} ⇒ **个数相同、集合不同**
    #    ⇒ subset=0.0 而 count=1.0,两个键**必然可分辨**。
    logits2 = [[-5.0] * len(LABELS)]
    logits2[0][LABELS.index("退换货")] = 3.0
    logits2[0][LABELS.index("物流")] = 3.0
    truth2 = [[0.0] * len(LABELS)]
    truth2[0][LABELS.index("退换货")] = 1.0
    truth2[0][LABELS.index("尺码")] = 1.0
    got = metrics_from_logits(logits2, truth2, 0.5)
    assert got["subset_accuracy"] == 0.0, "集合不同(退换货+尺码 vs 退换货+物流)"
    assert got["label_count_match"] == 1.0, "个数相同(都是 2)"


# ---------------------------------------------------------------- 数据集包装


def test_topic_dataset_is_indexable_and_declares_columns():
    """`TopicDataset` 必须可下标、有长度、并声明 `column_names`。

    ⚠️ **实测订正(计划原文这条的理由写错了)**:计划说「不包一层,
    `Trainer` 会读 `dataset.column_names` 而普通 list 没有它 ⇒ AttributeError」。
    本机 transformers 5.17.0 上**不成立** —— `trainer.py:988` 是

        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, ...)
        else:
            data_collator = self._get_collator_with_removed_columns(...)

    走 `column_names` 的那条路**只对 `datasets.Dataset` 生效**;裸 list /
    torch Dataset 走 else 分支(实测:传 `list[dict]` 给 `Trainer` 能正常训)。

    这一层仍然保留,理由换成两条**真的**成立的:① 它让我们有一处能写
    「字段就是模型 forward 要的那三个」的地方;② `column_names` 让将来真换成
    `datasets.Dataset` 时行为不变。**它不是一条必需的错误规避。**
    """
    encoded = encode_rows([{"question": "买大了想退", "labels": ["尺码"]}],
                          FakeTokenizer(), max_length=16)
    ds = TopicDataset(encoded)
    assert len(ds) == 1
    assert ds[0]["labels"].dtype.is_floating_point
    assert ds.column_names == ["input_ids", "attention_mask", "labels"]


def test_topic_dataset_is_a_torch_dataset():
    """`Trainer` 用 `DataLoader` 装它 —— 老老实实继承 `torch.utils.data.Dataset`,
    不要靠鸭子类型。"""
    import torch.utils.data

    assert isinstance(TopicDataset([]), torch.utils.data.Dataset)


# ---------------------------------------------------------------- 产物读写


def test_save_and_load_artifacts_roundtrip(tmp_path):
    """`labels.json` 是**推理侧标签顺序的唯一来源**,必须能原样读回。"""
    save_artifacts(tmp_path, tokenizer=None, max_length=64, threshold=0.5, labels=LABELS,
                   meta={"seed": 1})
    arts = load_artifacts(tmp_path)
    assert tuple(arts["labels"]) == LABELS
    assert arts["max_length"] == 64
    assert arts["threshold"] == 0.5
    assert arts["meta"]["seed"] == 1


def test_saved_labels_match_taxonomy(tmp_path):
    """⚠️ 产物里的顺序必须与 `taxonomy.LABELS` **逐位相同**。

    两侧不一致会产出一张**完全错的分布图,而每个组件都工作正常** ——
    模型有输出、scores 在 0–1、写库成功、页面画得出来。没有任何东西会报错。
    """
    save_artifacts(tmp_path, tokenizer=None, max_length=64, threshold=0.5, labels=LABELS, meta={})
    on_disk = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    assert on_disk == list(LABELS)


def test_save_artifacts_does_not_require_a_tokenizer(tmp_path):
    """`tokenizer=None` 时不许碰 `save_pretrained`(单测走的就是这条路,不联网)。"""
    save_artifacts(tmp_path, tokenizer=None, max_length=8, threshold=0.4, labels=LABELS, meta={})
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "inference_config.json", "labels.json", "train_meta.json",
    ]


# ------------------------------------------- 标签映射(订正轮 1 · I2)


def _decoy_config(num_labels: int) -> dict:
    """HF `save_pretrained` 落下来的那份**诱饵** config.json(实测形状)。"""
    return {
        "architectures": ["BertForSequenceClassification"],
        "model_type": "bert",
        "num_labels": None,  # 实测:这个键真的存在,值是 null
        "problem_type": "multi_label_classification",
        "id2label": {str(i): f"LABEL_{i}" for i in range(num_labels)},
        "label2id": {f"LABEL_{i}": i for i in range(num_labels)},
    }


def _write_decoy(tmp_path, num_labels: int) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(_decoy_config(num_labels), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def test_label_metadata_is_the_single_source_of_the_label_map():
    """三个键由 `labels` **一处**生成。`id2label` 的键是**字符串**(JSON 的键只能是字符串)。"""
    meta = label_metadata(LABELS)
    assert meta["num_labels"] == len(LABELS)
    assert [meta["id2label"][str(i)] for i in range(len(LABELS))] == list(LABELS)
    assert meta["label2id"]["退换货"] == 0
    assert meta["label2id"]["其他"] == len(LABELS) - 1


def test_ensure_label_metadata_repairs_the_hf_default_decoy(tmp_path):
    """⚠️ **订正轮 1 · I2:HF 落的 `config.json` 里 `id2label` 默认是 `LABEL_0…`。**

    **加载不报错**(`num_labels` 由 `len(id2label)` 推出来,实测仍得 17),所以
    这条路走到 Task 11 / 任何一段顺手代码里就是:

        model.config.id2label[j]  或  pipeline("text-classification", model=...)

    ⇒ **17 个类目全叫 `LABEL_5`**,而 scores 正常、写库成功、页面画得出来、
    **日志里没有一行不对** —— 正是 spec §2.7 逐字点名的那条静默陷阱。

    类目名的权威来源只有一个:`labels.json`。这条用例断的就是**两者逐位一致**。
    """
    save_artifacts(tmp_path, tokenizer=None, max_length=64, threshold=0.5, labels=LABELS, meta={})
    _write_decoy(tmp_path, len(LABELS))

    assert ensure_label_metadata(tmp_path, labels=LABELS) is True

    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    on_disk = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    assert cfg["num_labels"] == len(on_disk)
    assert [cfg["id2label"][str(i)] for i in range(len(on_disk))] == on_disk
    assert cfg["label2id"] == {label: i for i, label in enumerate(on_disk)}
    assert cfg["model_type"] == "bert", "别的键一个都不许丢"
    assert cfg["problem_type"] == "multi_label_classification"


def test_ensure_label_metadata_is_idempotent(tmp_path):
    """**幂等**:第二次必须返回 `False` 且**字节不变** —— 它是给已有目录重跑用的。"""
    save_artifacts(tmp_path, tokenizer=None, max_length=64, threshold=0.5, labels=LABELS, meta={})
    _write_decoy(tmp_path, len(LABELS))

    assert ensure_label_metadata(tmp_path, labels=LABELS) is True
    after_first = (tmp_path / "config.json").read_bytes()
    assert ensure_label_metadata(tmp_path, labels=LABELS) is False
    assert (tmp_path / "config.json").read_bytes() == after_first


def test_ensure_label_metadata_fixes_a_wrong_mapping_not_just_a_missing_one(tmp_path):
    """⚠️ **反向断言**:映射**顺序错了**(两个类目对调)也必须被纠正。

    一个「只在缺键时补」的实现(`if "id2label" not in data`)会在这条上绿 ——
    而「键在、值错」正是最危险的那种(它看起来已经设好了)。这个变异由
    `--only R2b` 的探针实测钉着。
    """
    save_artifacts(tmp_path, tokenizer=None, max_length=64, threshold=0.5, labels=LABELS, meta={})
    _write_decoy(tmp_path, len(LABELS))

    wrong = _decoy_config(len(LABELS))
    wrong["id2label"] = {str(i): LABELS[i] for i in range(len(LABELS))}
    wrong["id2label"]["0"], wrong["id2label"]["1"] = wrong["id2label"]["1"], wrong["id2label"]["0"]
    (tmp_path / "config.json").write_text(
        json.dumps(wrong, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    assert ensure_label_metadata(tmp_path, labels=LABELS) is True
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert [cfg["id2label"][str(i)] for i in range(len(LABELS))] == list(LABELS)


# ---------------------------------------------------------------- 数据指纹(12-A)


def test_fingerprint_is_the_same_for_crlf_and_lf(tmp_path):
    """⚠️ **12-A:`data_fingerprint()` 必须对 LF 归一化后的文本算。**

    实测:`evals/topic/train_augmented.jsonl` **盘上是 CRLF**(2315 个 CRLF、
    0 个裸 LF)。照 `read_bytes()` 算,指纹**随 checkout 的行尾策略变化** ——
    换台 `autocrlf=false` / Linux 的机器算出来是另一个值,后人要么以为语料
    被换了、要么放弃这条凭据。**它一坏就是「静默给一个错的溯源答案」。**
    """
    body = '{"a": 1}\n{"b": 2}\n'
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    lf.write_text(body, encoding="utf-8", newline="\n")
    crlf.write_text(body, encoding="utf-8", newline="\r\n")

    # 反向断言:两条**真的**不同行尾 —— 否则这条用例在两个实现下都恒真。
    assert b"\r\n" in crlf.read_bytes()
    assert b"\r\n" not in lf.read_bytes()
    assert crlf.read_bytes() != lf.read_bytes()

    assert data_fingerprint(crlf) == data_fingerprint(lf)


def test_fingerprint_is_sha256_of_the_lf_text_first_16(tmp_path):
    """把算法本身钉死:sha256(**LF 归一化后**的 utf-8 字节)[:16]。

    这条对 `read_bytes()` 的实现**直接红** —— 文件在盘上是 CRLF,
    两种算法在这里给出不同的值(这正是 12-A 表格里那两行)。
    """
    p = tmp_path / "x.jsonl"
    p.write_text("a\nb", encoding="utf-8", newline="\r\n")  # 盘上是 CRLF

    assert b"\r\n" in p.read_bytes()  # 前提自检:别让它悄悄退化成 LF 而变成恒真
    assert data_fingerprint(p) == hashlib.sha256(b"a\nb").hexdigest()[:16]
    assert data_fingerprint(p) != hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# ------------------------------------------------------- 真的跑一遍 forward(spec §7.1③)


def _tiny_bert(problem_type):
    """本地现搭的小 `BertConfig` —— **不下载、不联网**。

    基座 `hfl/chinese-roberta-wwm-ext` 的 `config.json` 实测 `model_type: "bert"`
    (名字里的 roberta 指的是 wwm 训练配方,不是 RoBERTa 架构),所以这里用
    `BertConfig` / `BertForSequenceClassification` 与真机同架构。
    """
    import torch
    from transformers import BertConfig, BertForSequenceClassification

    cfg = BertConfig(
        vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=64, num_labels=len(LABELS),
    )
    cfg.problem_type = problem_type
    torch.manual_seed(0)
    return BertForSequenceClassification(cfg)


_IDS = [[1, 2, 3, 4], [5, 6, 7, 8]]
_AM = [[1, 1, 1, 1], [1, 1, 1, 1]]


def test_explicit_problem_type_gives_bce_not_cross_entropy():
    """spec §7.1③:显式 `problem_type="multi_label_classification"` 时,
    loss **逐位等于**手算的 `binary_cross_entropy_with_logits`。

    这是「模型真的在做多标签」的唯一直接证据 —— 前面几条只证明
    **我们编出来的张量**是 float 多热,证明不了模型拿它算了什么。
    """
    import torch
    import torch.nn.functional as F

    model = _tiny_bert("multi_label_classification")
    labels = torch.zeros(2, len(LABELS))
    labels[0, LABELS.index("尺码")] = 1.0
    labels[0, LABELS.index("退换货")] = 1.0
    labels[1, LABELS.index("物流")] = 1.0

    out = model(input_ids=torch.tensor(_IDS), attention_mask=torch.tensor(_AM), labels=labels)

    bce = F.binary_cross_entropy_with_logits(out.logits, labels)
    # ⚠️ v5 的 `SequenceClassifierOutput` **没有** `.config` —— 读模型上的那个。
    assert model.config.problem_type == "multi_label_classification"
    assert out.loss.item() == pytest.approx(bce.item(), abs=1e-6)

    # 反向断言:它**不是**交叉熵 —— 否则「loss 等于某个数」这句没判别力。
    ce = F.cross_entropy(out.logits, labels.argmax(-1))
    assert out.loss.item() != pytest.approx(ce.item(), abs=1e-6)


def test_float_labels_are_what_keeps_the_guess_off_single_label():
    """反向断言:`problem_type=None` 时,float 多热**仍然**落进 multi_label
    (走的是 `else` 那一支)⇒ 「标签是 float」这道防线**单独就够**。

    这条与下一条合起来说明:§2.3 那个猜测**真的会跑**,而它的走向由
    **标签的 dtype 与形状**决定,不由我们想要什么决定。
    """
    import torch
    import torch.nn.functional as F

    model = _tiny_bert(None)
    labels = torch.zeros(2, len(LABELS))
    labels[0, 0] = 1.0
    labels[1, 5] = 1.0

    out = model(input_ids=torch.tensor(_IDS), attention_mask=torch.tensor(_AM), labels=labels)

    assert model.config.problem_type == "multi_label_classification"  # 被写进 config
    assert out.loss.item() == pytest.approx(
        F.binary_cross_entropy_with_logits(out.logits, labels).item(), abs=1e-6
    )


def test_one_dimensional_long_labels_silently_become_single_label():
    """⚠️ **这就是 spec §2.3 说的那个静默后果的确切实例。**

    同一条数据、同一个模型,只把标签换成 **1-D 的 long 类索引**(单标签写法):
    `problem_type=None` ⇒ 猜成 `single_label_classification` ⇒ softmax + 交叉熵,
    **loss 逐位等于手算的 `cross_entropy`**、训练一路跑下去、什么也不报。

    ⇒ 「标签是 float」那条断言是**唯一**能在第一跳拦住它的东西
    (`BertForSequenceClassification` 的 config 一旦被写上,后面就锁死了)。
    """
    import torch
    import torch.nn.functional as F

    model = _tiny_bert(None)
    idx = torch.tensor([0, 5])  # 1-D long:单标签写法的形状

    out = model(input_ids=torch.tensor(_IDS), attention_mask=torch.tensor(_AM), labels=idx)

    assert model.config.problem_type == "single_label_classification"
    assert out.loss.item() == pytest.approx(F.cross_entropy(out.logits, idx).item(), abs=1e-6)


def test_seventeen_column_long_labels_explode_loudly_instead():
    """⚠️ **实测订正 spec §2.3 的一句话**(本机 5.17.0,BERT 头):它说「整数张量
    (哪怕有 17 列)会静默选单标签」并把后果描述成「loss 照常下降、训练看起来
    完全正常」。**17 列 long 的那条路不是静默的 —— 它当场抛**:

        ValueError: Expected input batch_size (2) to match target batch_size (34)

    因为 `CrossEntropyLoss(logits.view(-1, 17), labels.view(-1))` 的形状对不上。
    **真正静默的是 1-D long**(见上一条)。

    这条用例存在的意义:**别把「会抛」当成「有防线」** —— 抛发生在**第一个
    batch**,而 `problem_type` 那时已经写进 config;把标签改回 float、
    或改用 1-D 的写法,就再也没有东西拦了。
    """
    import torch

    model = _tiny_bert(None)
    labels = torch.zeros(2, len(LABELS), dtype=torch.long)
    labels[0, 0] = 1

    with pytest.raises(ValueError, match="batch_size"):
        model(input_ids=torch.tensor(_IDS), attention_mask=torch.tensor(_AM), labels=labels)
