"""旁路推理服务:标签顺序、阈值、批量形状。

⚠️ 单测**不加载真权重**(不联网、不慢)。`TopicClassifier` 的模型是注入的。

⚠️ **本文件里每一条「来自产物」的断言都用了非默认值** —— 这是 spec §9.1 那条防线的
全部判别力所在:写 `LABELS` 断 `LABELS`、写 `0.5` 断 `0.5` 是**同义反复**,
一个把类目顺序与阈值写死在服务里的实现照样绿,而它守的偏偏是「服务不许自己写一份」。
(计划订正 14-A / 14-B / 14-D 就是这三条;14-C 是「没调用模型」那条。)
"""

import pytest

from app.topic.taxonomy import LABELS
from topic_service.model import TopicClassifier


class _FakeModel:
    """返回固定的 logits —— 测的是**我们**的解码逻辑,不是模型。"""

    def __init__(self, logits):
        self.logits = logits

    def __call__(self, **kwargs):
        import torch

        return type("Out", (), {"logits": torch.tensor(self.logits)})()


def _classifier(tmp_path, logits, *, labels=None, threshold=0.5, max_length=64):
    import json

    # ⚠️⚠️ **计划订正 14(controller,2026-09-27)—— 这里的三个参数是刻意的** ⚠️⚠️
    # 原稿把 `labels` / `threshold` 写死成 `LABELS` / `0.5`(**也就是实现的默认期望值**),
    # 于是那三条「来自产物」的断言变成了**同义反复**:
    # **一个把 `LABELS`、`0.5` 写死在服务里的实现照样绿** ——
    # 而它们的 docstring 说的正是「服务不许自己写一份」。
    # ⇒ 现在**可以传进被打乱 / 非默认的值**,断言读出来的必须是**那个值**。
    #
    # ⚠️ **实现者补记(T11)**:原稿只写 `labels.json` + `inference_config.json`
    # 两个文件,而 brief 的 Interfaces 指名让服务消费
    # `app.topic.model.load_artifacts` —— 那个函数**还读第三个文件**
    # `train_meta.json`(真实产物里一定有:`save_artifacts` 无条件写它)。
    # 只写两个文件的话,一个**正确**的实现会在这里抛 `FileNotFoundError`。
    # ⇒ 补上第三个文件,而不是让实现绕过 `load_artifacts`(那是唯一读侧)。
    (tmp_path / "labels.json").write_text(
        json.dumps(list(labels if labels is not None else LABELS)), encoding="utf-8")
    (tmp_path / "inference_config.json").write_text(
        json.dumps({"max_length": max_length, "threshold": threshold}), encoding="utf-8")
    (tmp_path / "train_meta.json").write_text(
        json.dumps({"data_fingerprint": "deadbeefdeadbeef",
                    "data_fingerprint_scope": "tests/test_topic_service.py"},
                   ensure_ascii=False),
        encoding="utf-8")
    return TopicClassifier(tmp_path, model=_FakeModel(logits), tokenizer=None)


def test_label_order_comes_from_the_artifact(tmp_path):
    """⚠️ 标签顺序**必须来自产物**,服务不许自己写一份。

    两侧顺序不一致会产出一张**完全错的分布图,而每个组件都工作正常**:
    模型有输出、scores 在 0–1、写库成功、页面画得出来。

    ⚠️ **订正 14-A**:原稿写进去的就是 `LABELS`、再断言等于 `LABELS` ⇒ **同义反复**,
    写死 `LABELS` 的实现照样绿。⇒ 现在写一份**打乱的**,断言读出的是**打乱的那个顺序**。
    """
    scrambled = list(reversed(LABELS))          # 顺序**被打乱**(内容仍是同样 17 个)
    c = _classifier(tmp_path, [[0.0] * len(LABELS)], labels=scrambled)
    assert c.labels == scrambled                # ← 必须是**产物里那个顺序**
    assert c.labels != list(LABELS)             # ← 且**不是**代码里那份(判别力在这一句)


def test_the_label_set_is_the_same_either_way(tmp_path):
    """顺序来自产物,**类目集合**仍必须等于 `taxonomy.LABELS`(spec §9.1 那条断言)。

    ⚠️ 这两件事要**分开**断:顺序错了是「分布图整张错」,集合错了是「类目表两处漂移」。
    合在一句里时,打乱顺序的正确实现会**误红**,而漏一个类目的实现可能**误绿**。
    """
    scrambled = list(reversed(LABELS))
    c = _classifier(tmp_path, [[0.0] * len(LABELS)], labels=scrambled)
    assert sorted(c.labels) == sorted(LABELS)


def test_threshold_comes_from_the_artifact(tmp_path):
    """⚠️ **订正 14-B**:原稿写 `0.5`、断 `0.5` ⇒ 写死 0.5 的实现照样绿。用**非默认值**。"""
    c = _classifier(tmp_path, [[0.0] * len(LABELS)], threshold=0.7)
    assert c.threshold == 0.7


def test_max_length_comes_from_the_artifact(tmp_path):
    """⚠️ **订正 14-D**(spec §9.1):`max_length` 也在产物里,**不许在服务里写死** ——
    「两侧截断长度不一致」与标签顺序是**同一族的静默失效**。同样用**非默认值**。"""
    c = _classifier(tmp_path, [[0.0] * len(LABELS)], max_length=48)
    assert c.max_length == 48


def test_sigmoid_decoding_picks_every_label_above_threshold(tmp_path):
    """多标签:**每个** logit 独立过阈值,不是取 argmax。

    取 argmax 就退化成单标签了 —— 而验收 ③ 正是要「同时命中多个类目」。
    """
    logits = [[-5.0] * len(LABELS)]
    logits[0][LABELS.index("尺码")] = 5.0
    logits[0][LABELS.index("退换货")] = 5.0
    out = _classifier(tmp_path, logits).predict(["买大了想退"])
    assert set(out[0]["labels"]) == {"尺码", "退换货"}


def test_all_below_threshold_yields_empty_labels(tmp_path):
    """全都不够阈值 ⇒ 空标签,而**不是**硬塞一个最大的。

    硬塞会让每个类目都凭空涨一批 —— 而分布页读的就是它。
    """
    out = _classifier(tmp_path, [[-9.0] * len(LABELS)]).predict(["???"])
    assert out[0]["labels"] == []


def test_scores_are_probabilities(tmp_path):
    out = _classifier(tmp_path, [[0.0] * len(LABELS)]).predict(["x"])
    assert all(0.0 <= v <= 1.0 for v in out[0]["scores"].values())
    assert set(out[0]["scores"]) == set(LABELS)


def test_predict_handles_a_batch(tmp_path):
    out = _classifier(tmp_path, [[0.0] * len(LABELS)] * 3).predict(["a", "b", "c"])
    assert len(out) == 3


class _ExplodingModel:
    """**被调用就抛** —— 用来把「没调用模型」从一句声称变成一个可断言的事实。

    ⚠️ **订正 14-C**:原稿那条测试叫 `..._without_calling_the_model`,
    但**只断言了返回值是 `[]`** —— 名字声称的比断的多。
    一个**真的**把空列表喂进模型的实现(浪费一次前向、还可能因空张量报错)照样绿。
    """

    def __call__(self, **kwargs):
        raise AssertionError("空输入不该调用模型")


def test_empty_input_returns_empty_without_calling_the_model(tmp_path):
    import json

    (tmp_path / "labels.json").write_text(json.dumps(list(LABELS)), encoding="utf-8")
    (tmp_path / "inference_config.json").write_text(
        json.dumps({"max_length": 64, "threshold": 0.5}), encoding="utf-8")
    (tmp_path / "train_meta.json").write_text("{}", encoding="utf-8")
    c = TopicClassifier(tmp_path, model=_ExplodingModel(), tokenizer=None)
    assert c.predict([]) == []          # ← 若它调了模型,这里会抛


# ---------------------------------------------------------------- 实现者补的部分(T11)
#
# ⚠️ 下面四条**不在 brief 的稿子里**。加它们的理由写在每条自己的 docstring 里 ——
# 两条是「本仓已有的第二处解码」那条账(见 `tests/test_topic_metrics.py` 的同名用例),
# 两条是 `create_app` 的装配处(单测里从没被行使过,而 Step 5 才第一次真跑它)。


def test_service_decodes_exactly_like_the_eval_script(tmp_path):
    """⚠️ **本仓的阈值解码将因此变成三处,这条用例是它们的装配处。**

    `tests/test_topic_metrics.py::test_eval_script_and_training_script_decode_logits_identically`
    已经把 `scripts/train_topic_clf.py` 与 `scripts/eval_topic_clf.py` 钉在一起;
    本任务的服务是**第三处**(它多返回一个 `scores`,所以两处都不能直接复用)。
    ⇒ 不钉的话,第三处与那两处**可以安静地漂开** —— 而漂开的样子是
    「服务预测的类目与评测报告的类目对不上」,两边都不会报错。

    判别力压在与那条同名用例**同一格**上:`退换货` 的 logit 恰好 **0.0**
    (`sigmoid(0.0) == 0.5` 精确相等)⇒ `>=` 判它入选、`>` 判它出局。
    """
    from scripts.eval_topic_clf import predict_from_logits

    logits = [[-5.0] * len(LABELS)]
    logits[0][LABELS.index("退换货")] = 0.0     # sigmoid == 0.5 **恰好**
    logits[0][LABELS.index("物流")] = 5.0
    out = _classifier(tmp_path, logits, threshold=0.5).predict(["x"])
    assert out[0]["labels"] == predict_from_logits(logits, 0.5, list(LABELS))[0]
    assert "退换货" in out[0]["labels"]          # ← 这一句在 `>` 的实现下会红


def _client(tmp_path, logits, **kwargs):
    from fastapi.testclient import TestClient

    from topic_service.server import create_app

    return TestClient(create_app(_classifier(tmp_path, logits, **kwargs)))


def test_healthz_reports_ok(tmp_path):
    """`GET /healthz` 是 Task 13 / 验收脚本「服务起没起」的唯一探针。"""
    r = _client(tmp_path, [[0.0] * len(LABELS)]).get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_endpoint_returns_one_result_per_text(tmp_path):
    """端点把 `{"texts": [...]}` 逐条映射成 `{"labels", "scores"}` —— **顺序与条数**是契约。

    Task 13 的批处理靠「返回条数 == 输入条数」决定整批原子性(`scripts/classify_topics.py`),
    条数对不上时它抛;所以这里的形状错了会让那边**响亮地**红 —— 但只有这里能说清
    是**谁**把顺序弄丢的。

    ⚠️ 断言用的是**标签内容**,不是 `len()` —— 一个把结果整体反序的实现条数照样对。
    """
    logits = [[-5.0] * len(LABELS) for _ in range(2)]
    logits[0][LABELS.index("尺码")] = 5.0
    logits[1][LABELS.index("运费")] = 5.0
    r = _client(tmp_path, logits).post("/predict", json={"texts": ["买大了", "运费多少"]})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["labels"] for x in results] == [["尺码"], ["运费"]]


def test_predict_endpoint_on_empty_texts_never_calls_the_model(tmp_path):
    """空批量在**端点**这一层也不喂模型 —— `_ExplodingModel` 把这句话变成事实。"""
    from fastapi.testclient import TestClient

    from topic_service.server import create_app

    import json

    (tmp_path / "labels.json").write_text(json.dumps(list(LABELS)), encoding="utf-8")
    (tmp_path / "inference_config.json").write_text(
        json.dumps({"max_length": 64, "threshold": 0.5}), encoding="utf-8")
    (tmp_path / "train_meta.json").write_text("{}", encoding="utf-8")
    app = create_app(TopicClassifier(tmp_path, model=_ExplodingModel(), tokenizer=None))
    r = TestClient(app).post("/predict", json={"texts": []})
    assert r.status_code == 200
    assert r.json()["results"] == []
