"""旁路推理服务的内核:产物 → logits → sigmoid → 阈值 → 标签。

## 那一件必须防住的事(spec §2.7 / §9.1)

模型输出的第 5 个 logit 对应哪个类目,**靠的是训练时标签列表的下标**。
若服务自己硬编码一份类目顺序、而它与训练用的顺序不一致 —— 得到的是
**一张完全错的分布图,而每一个组件都工作正常**:模型有输出、scores 在 0–1 之间、
写库成功、页面画得出来。**没有任何东西会报错。**

⇒ 三道防线,**全部落在产物上**:

| 读什么 | 从哪读 | 写死会怎样 |
|---|---|---|
| 类目顺序 | `labels.json` | 分布图整张错(错位),每个组件都正常 |
| 阈值 | `inference_config.json` | 选中的类目数量整体偏多/偏少,而 scores 仍全在 0–1 |
| `max_length` | `inference_config.json` | 与训练侧截断长度不一致 ⇒ 服务看到的是另一种文本 |

⚠️ **不要用 `config.json` 的 `id2label` 拿类目名** —— 它可以是错的(出厂默认
`LABEL_0…`,见 `app/topic/model.py::ensure_label_metadata` 的 docstring)。
**`labels.json` 才是权威**。

## 唯一读侧

产物一律走 `app.topic.model.load_artifacts`(它的 docstring 逐字写着
「服务与评测脚本都走这里 —— **不要各自 `open()` 一遍**」),
它同时读 `labels.json` / `inference_config.json` / `train_meta.json`。

## 阈值解码在仓里是**第三处**

`scripts/train_topic_clf.py::metrics_from_logits` 与
`scripts/eval_topic_clf.py::predict_from_logits` 是前两处(后者多返回 `y_pred` 画矩阵,
两者由 `tests/test_topic_metrics.py` 钉在一起)。本模块**不能**直接复用任何一个:
服务还要逐标签的 `scores`(分布页与 `topic_classifications.scores` 都读它),
而那两个都只返回类别名。⇒ 「三处不安静地漂开」由
`tests/test_topic_service.py::test_service_decodes_exactly_like_the_eval_script`
钉在同一格上(`sigmoid(logit) == 0.5` 精确相等那一格)。
"""

from __future__ import annotations

import torch

from app.topic.model import load_artifacts


class TopicClassifier:
    """旁路推理服务的内核。**标签顺序与阈值一律来自产物目录**,不在这里写死。

    见 `app/topic/model.py` 的 `save_artifacts`:那是唯一的写侧,
    这里是唯一的读侧 —— 两侧不一致会产出一张完全错的分布图而无人报错。
    """

    def __init__(self, model_dir, *, model=None, tokenizer=None, device: str = "cpu"):
        """`model` / `tokenizer` 非 None 时用注入的那一份(**单测走这条路,不加载真权重**)。

        `device` 默认 **CPU** —— 不是性能考虑,与 `scripts/eval_topic_clf.py` 同一条理由:
        cuBLAS / cuDNN 的算子选择带启发式,同一份权重两次前向的最后几个 bit 不保证相同,
        而阈值比较正好落在那几位上时**预测会翻**。要看「跑两次逐字节相同」就得钉 CPU。
        """
        arts = load_artifacts(model_dir)
        self.model_dir = str(model_dir)
        self.labels = list(arts["labels"])          # ← 来自产物
        self.threshold = float(arts["threshold"])   # ← 来自产物
        self.max_length = int(arts["max_length"])   # ← 来自产物
        # `train_meta.json` 原样带着 —— `/healthz` 把它印出来(「这份权重是谁训的」),
        # 服务自己**一个字段都不解释**,免得又变成一处对产物的第二读法。
        self.meta = dict(arts["meta"])
        self.device = torch.device(device)

        if model is None:
            # 延迟 import:单测注入模型时**一行 transformers 都不加载**。
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model.to(self.device)
            # `eval()` 关掉 dropout —— 与评测脚本同款,服务也不该有随机性。
            model.eval()

        self.model = model
        # ⚠️ `tokenizer=None` **只在注入模型时有意义**(单测的替身不看输入)。
        #    真实构造路径上它一定非 None —— 没有任何一条生产分支能走到 None。
        self.tokenizer = tokenizer

    def _forward(self, texts: list[str]) -> torch.Tensor:
        """一批文本 → logits(**始终在 CPU 上**)。

        `tokenizer is None` 那条分支是**注入替身专用**的测试接缝(见 `__init__`)。
        """
        with torch.no_grad():
            if self.tokenizer is None:
                return self.model().logits
            enc = self.tokenizer(texts, truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt")
            enc = {key: value.to(self.device) for key, value in enc.items()}
            return self.model(**enc).logits.to("cpu")

    def predict(self, texts: list[str]) -> list[dict]:
        """逐条返回 `{"labels": [...], "scores": {类目: 概率}}`。

        **每个 logit 独立过阈值**(多标签),**不取 argmax** —— 取 argmax 就退化成
        单标签了,而验收 ③ 要的正是「同一句同时命中多个类目」。

        全低于阈值 ⇒ **空列表**,而不是硬塞一个最大的:硬塞会让每个类目都凭空涨一批,
        而分布页读的就是它。
        """
        texts = list(texts)
        if not texts:
            # ⚠️ **空输入连模型都不碰** —— 不是优化:空张量进模型既浪费一次前向,
            #    也可能在真实权重上报错。`_ExplodingModel` 把这句话变成可断言的事实。
            return []

        probs = torch.sigmoid(self._forward(texts))
        out = []
        for row in probs:
            scores = {label: float(row[i]) for i, label in enumerate(self.labels)}
            out.append({
                # 顺序是 `labels` 的顺序(产物里那个),不是阈值排序。
                "labels": [label for label, p in scores.items() if p >= self.threshold],
                "scores": scores,
            })
        return out
