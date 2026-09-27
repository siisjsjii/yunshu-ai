"""`topic_service` 的 HTTP 面 —— 两个端点,一个内核。

| 端点 | 给谁 |
|---|---|
| `POST /predict` | `scripts/classify_topics.py` 跑批(spec §9.2,**Task 13**)与验收脚本 |
| `GET /healthz` | 「服务起没起」的探针 —— 批处理**连不上就响亮失败**(§9.2),这是它先问的那一句 |

**两个端点都是同步 `def`,不是 `async def`。** 分类器里那一段是**同步**的 torch 前向
(CPU 上几十毫秒到数秒),写成 `async def` 会**占住整个事件循环** ——
本仓 ch09 已经用血记过这条账(`POST /api/feedback` 的尽力回捞)⇒ 交给
Starlette 的线程池跑才是对的形状。

⚠️ **这个端点里没有一次对产物的第二读法**:`labels` / `threshold` / `max_length`
全在 `TopicClassifier.__init__` 从产物读定,这里只做搬运(spec §9.1 那条防线)。
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """批量请求。**空列表是合法的** —— 返回空 `results`,不喂模型、不报错。"""

    texts: list[str] = Field(default_factory=list)


def create_app(classifier) -> FastAPI:
    """`classifier` 由调用方注入 —— 单测给替身,**不加载真权重**。"""
    app = FastAPI(title="主题分类旁路服务", version="ch10-B")

    @app.get("/healthz")
    def healthz() -> dict:
        """服务活着 + **它到底读到了什么**。

        这三个数(`threshold` / `max_length` / `labels`)与它们**唯一合法的来源是产物**
        ⇒ 「服务手里那份与 `models/topic-clf` 里那份不一样」这件事在这里**当场看得见**,
        而不是等到分布图整张错之后没人说得出为什么。

        ⚠️⚠️ **`labels` 必须是那张表本身,不能只印 `num_labels`**(订正轮 1 · I-1)。
        订正前这里只有 `num_labels` —— 那对**顺序**这一维**是瞎的**:
        把 `labels.json` **整体反序**(17 个类目一条不差)⇒ 服务照常起、
        `num_labels` 还是 17、阈值与长度全对、scores 全在 0–1、写库成功、
        分布页画得出来 —— **每一类都错位,没有任何东西报错**,而这是本服务
        **唯一的可核对出口**。计数只能抓「加/减了类目」,抓不到重排;
        spec §2.7 / §9.1 点名的恰恰是**重排**那一种。
        """
        return {
            "status": "ok",
            "model_dir": classifier.model_dir,
            # ⚠️ 顺序也要印 —— `num_labels` 对顺序零覆盖(见 docstring)。
            "labels": list(classifier.labels),
            "num_labels": len(classifier.labels),
            "threshold": classifier.threshold,
            "max_length": classifier.max_length,
            # 「这份权重是谁训的」——批处理写进 `topic_classifications.model_version`
            # 要靠它(`train_meta.json` 原样带出,服务不解释任何字段)。
            "train_meta": classifier.meta,
        }

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict:
        """`{"texts": [...]}` → `{"results": [{"labels", "scores"}, ...]}`。

        **返回条数恒等于输入条数、顺序一一对应** —— 那是 Task 13 批处理的契约
        (它靠「条数对不上就抛」保整批原子;那份脚本是 **Task 13** 的活,今天还没有)。

        ⚠️ **同一个 `self.model` 会被多线程并发前向**(两个端点都是同步 `def`,
        Starlette 把它们丢进线程池 ⇒ 并发请求 = 并发前向)。**今天刻意不加锁**
        (订正轮 1 · M-3),理由是:

        ① 按 spec §9.2 的设计,唯一的调用方 `scripts/classify_topics.py`(**Task 13**)
           **串行**跑批 —— 一批一次 `POST`,批与批之间不并发;
        ② `model.eval()` 下只有**只读**前向,没有参数更新(`__init__` 里已经关掉 dropout)。

        ⚠️ **这处取舍有先例,而且是「有先例的反面」**:本仓给**共享 torch 模型**加过锁 ——
        `app/retrieval/embedder.py` 的 `_encode_lock`(后台任务与聊天检索会同时打进来,
        并发 encode 不保证安全)。**那个先例的前提这里没有:没有第二个调用方、没有并发成分。**
        ⇒ 若将来出现**第二个**调用方且它会并发打进来,加锁的位置在**这一层**
        (不是 `TopicClassifier` 里):要包住的是「一次前向」,不是「一个实例」。
        """
        return {"results": classifier.predict(req.texts)}

    return app
