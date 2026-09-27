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

        把 `labels` 的条数与 `threshold` / `max_length` 印出来,是刻意的:
        那两个数**唯一合法的来源是产物** ⇒ 「服务手里那份与 `models/topic-clf`
        里那份不一样」这件事在这里**当场看得见**,而不是等到分布图整张错之后
        没人说得出为什么。
        """
        return {
            "status": "ok",
            "model_dir": classifier.model_dir,
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
        """
        return {"results": classifier.predict(req.texts)}

    return app
