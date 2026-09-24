"""待审队列的人工审核端点(ch09 §9)。

四个端点:

| 方法 | 路径 | 作用 |
|---|---|---|
| `GET`  | `/api/review/queue?status=pending` | 列表:标准化问题、出现次数、示例答案、状态 |
| `GET`  | `/api/review/{id}` | 详情:上面那些 + **归并进来的用户原话清单**(每条带它自己的召回片段快照) |
| `POST` | `/api/review/{id}/approve` | body `{"approved_answer": str?}`(可省 / 可为空 body);不传就用 `example_answer` |
| `POST` | `/api/review/{id}/reject` | 置 `status='rejected'` |

## 通过 ⇒ **立刻**写知识库并向量化(验收 3 的全部)

不这么做的话,「同一个问题再问就能答对」要等下一次 `scripts/build_kb.py` ——
验收 3 直接落不了地(ch09 spec §9.2)。代价是**同步**烧一次 BGE-M3 嵌入 +
一次 Milvus upsert(审核人等着看结果,后台化只会让「点通过之后到底成没成」
变成一个说不清的状态);这个耗时**未在真实链路上量过**(本机 Milvus 没起,
见 T15 报告 §H5),别拿它当已知数引用。

## ⚠️ 一个组合陷阱:查重命中 ⇒ `write_chunks` 返回 0,**不是**「不用向量化了**

`write_chunks` 自带 `(category, questions, answer)` 三元组查重,返回的是**新增**
行数 ⇒ 它**可能返回 0**。把向量化挂在 `if added:` 下面是一条**静默的永久缺失**:

    第一次通过 → 写进 MySQL(pending)→ Milvus 炸 → 502,队列行留 pending(对)
    审核人重试 → 三元组已存在 ⇒ `added == 0` ⇒ **从不向量化** ⇒ 回 200、
    队列行变 approved,而那条知识块**永远停在 pending**,那个问题**永久答不上**。

所以向量化的准入条件不是 `added`,而是**「这个三元组在库里是不是还没 done」** ——
`_rows_to_vectorize` 是它的唯一判据(重试与新建两条路走同一个出口)。

## 状态只有一个写口

`review_queue.status` 的**唯一**写入方就是本模块的 `approve` / `reject`
(飞轮流水线只建 `pending` 行)。两个端点都按 `status != 'pending'` 挡第二次调用
⇒ **第二次是 404,不是静默成功**(静默成功会让审核台把一次重复点击读成
「又处理了一条」)。

## 顺序:路由按**声明顺序**匹配

`/api/review/queue` 必须声明在 `/api/review/{review_id}` **之前**。反了的话
`queue` 会去走 `review_id: int` 的解析 ⇒ **422**,而服务照常起、别的端点全正常。
`tests/test_api_review.py` 有一条专钉它的用例。

## store / embedder 从哪来

走 `get_vector_store` / `get_embedder` 两个**进程内单例**(`@lru_cache`),
与 `app/tools/registry.py:build_retriever` 用的是**同一批对象**(同参 ⇒ 同实例,
否则这里白建一份连接与一份 2.2GB 权重)。刻意**不从 `build_retriever` 的返回值里
掏 `_store` / `_embedder`**:那是检索器的私有属性,今天长得一样不代表明天一样。
（`app/kb/orchestrate.py:106` 那两条也是这么取的,形状一致。）
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select

from app.config import Settings, get_settings
from app.db.models import KnowledgeChunk, LowConfidenceQuestion, ReviewQueue
from app.db.session import get_session
from app.kb.chunker import Chunk
from app.kb.writer import vectorize_rows, write_chunks
from app.retrieval.embedder import get_embedder
from app.retrieval.milvus import get_vector_store

logger = logging.getLogger(__name__)

#: 核准入库的知识块用的 category / content_type。**与 spec §9.2 逐字一致** ——
#: 检索侧不做 category 过滤,这里只是溯源用的元数据。
KB_CATEGORY = "faq"
KB_CONTENT_TYPE = "faq"

#: 向量化失败时的出站文案。**固定文案,绝不回显原异常** ——
#: `vectorize_rows` 抛出来的可能是 pymilvus / torch / SQLAlchemy 的原文,
#: 而这是**出站**文本(前端会原样显示)。本仓那条「基础设施故障一律固定文案」
#: 正是为此。
VECTORIZE_FAILED_DETAIL = "知识入库失败(向量化未完成),该待审项仍未处理,请稍后重试"

#: `review_queue.status` 的三个取值(`db/ch09.sql` 的列注释同)。
#: 列表端点按闭集校验:非法取值**响亮地 422**,不做「静默当 pending」——
#: 前端把 `rejected` 拼错成 `reject` 时,静默兜底会让审核人看着**待审队列**
#: 却以为自己在看历史。
STATUSES = ("pending", "approved", "rejected")

router = APIRouter()


class ApproveRequest(BaseModel):
    """核准请求体。`approved_answer` 省略/为空 ⇒ 用队列行的 `example_answer`。

    ⚠️ 端点把它声明成**可空参数**(`body: ApproveRequest | None = None`):
    审核人点了「通过」而不改答案是最常见的一次操作,前端很可能**不带 body**
    或带一个空 JSON。只收必填 body 的话那两种都会 422,而前端只会显示
    「请求失败」—— 明明是一次合法的通过。
    """

    approved_answer: str | None = None


def _row_dict(rq: ReviewQueue) -> dict:
    """队列行 → 出站字典(列表与详情共用同一个投影,审核页只有一套读法)。"""
    return {
        "id": rq.id,
        "standard_question": rq.standard_question,
        "example_answer": rq.example_answer,
        "occurrences": rq.occurrences,
        "status": rq.status,
        "approved_answer": rq.approved_answer,
        "first_raw_question": rq.first_raw_question,
        "source_conversation_id": rq.source_conversation_id,
        "created_at": rq.created_at,
        "reviewed_at": rq.reviewed_at,
    }


async def _linked_pool_rows(session, review_id: int) -> list[LowConfidenceQuestion]:
    """归并到这条队列行的**池子行**(按 `matched_review_id`)。

    这些行就是审核页上那两块数据:`question`(用户原话,一条队列行**可能有多条**)
    与 `evidence_snapshot`(落池当轮/回捞时的召回片段)。
    """
    return list(
        (
            await session.execute(
                select(LowConfidenceQuestion)
                .where(LowConfidenceQuestion.matched_review_id == review_id)
                .order_by(LowConfidenceQuestion.id)
            )
        )
        .scalars()
        .all()
    )


async def _rows_to_vectorize(session, chunk: Chunk) -> list[KnowledgeChunk]:
    """三元组命中的、**还没 done** 的知识块行。

    这是「通过之后必须真的可检索」这条性质的**唯一判据**(见模块 docstring 那个
    陷阱):新建时它返回刚写进去的那一行,重试时它返回上一次留下的 pending 行 ——
    **同一个出口**,两条路都不会漏。已 `done` 的行不重复处理(否则每次通过都要
    白烧一次嵌入 + 一次 Milvus 往返)。
    """
    return list(
        (
            await session.execute(
                select(KnowledgeChunk)
                .where(
                    KnowledgeChunk.category == chunk.category,
                    KnowledgeChunk.questions == chunk.questions,
                    KnowledgeChunk.answer == chunk.answer,
                    KnowledgeChunk.vectorize_status != "done",
                )
                .order_by(KnowledgeChunk.id)
            )
        )
        .scalars()
        .all()
    )


# ⚠️ **本函数必须声明在 `get_review` 之前** —— 见模块 docstring「顺序」那段。
@router.get("/api/review/queue")
async def list_queue(
    status: str = "pending",
    session=Depends(get_session),
) -> list[dict]:
    """待审列表。默认只回 `pending`;过滤在** SQL** 里(不是捞全表再在 Python 里筛)。"""
    if status not in STATUSES:
        raise HTTPException(
            status_code=422, detail=f"status 必须是 {'/'.join(STATUSES)} 之一")
    rows = (
        (
            await session.execute(
                select(ReviewQueue)
                .where(ReviewQueue.status == status)
                .order_by(ReviewQueue.id.desc())
            )
        )
        .scalars()
        .all()
    )
    return [_row_dict(rq) for rq in rows]


@router.get("/api/review/{review_id}")
async def get_review(review_id: int, session=Depends(get_session)) -> dict:
    """详情:队列行 + **归并进来的全部用户原话**(每条各带自己的召回片段快照)。"""
    rq = await session.get(ReviewQueue, review_id)
    if rq is None:
        raise HTTPException(status_code=404, detail="待审问题不存在")
    out = _row_dict(rq)
    out["raw_questions"] = [
        {
            "id": row.id,
            "question": row.question,
            "entry_point": row.entry_point,
            "reject_reason": row.reject_reason,
            "evidence_snapshot": row.evidence_snapshot,
            "created_at": row.created_at,
        }
        for row in await _linked_pool_rows(session, review_id)
    ]
    return out


@router.post("/api/review/{review_id}/approve")
async def approve_review(
    review_id: int,
    body: ApproveRequest | None = None,
    session=Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict:
    """通过 ⇒ 写知识库 + **立刻**向量化;失败一律 502 且队列行**留待重试**。

    返回 `{"ok", "chunks_added", "vectorized"}`。`chunks_added == 0` **不是失败**
    (知识库里早就有那条三元组),响应里那个 0 本身就是给验收脚本/审核台读的说明
    (spec §13-9):那时「答不对」的原因是别的,不是这一条没入库。
    """
    rq = await session.get(ReviewQueue, review_id)
    # 已处理过(approve 或 reject)⇒ 404。**不许用静默成功**:
    # 第二次通过会让下面整段再跑一遍(多烧一次嵌入),而审核台读到的是一次成功。
    if rq is None or rq.status != "pending":
        raise HTTPException(status_code=404, detail="待审问题不存在或已处理")

    # 校验判在**解析之后**那个值上:只看 `body.approved_answer` 的话,一个空串会被
    # 当成「审核人传了答案」写进知识库 —— 那条块的正文是空的,检索永远召不回它,
    # 而队列行已经 approved、再也不会有人看它第二眼。
    answer = ((body.approved_answer if body is not None else None)
              or rq.example_answer).strip()
    if not answer:
        raise HTTPException(status_code=422, detail="核准答案不能为空")

    chunk = Chunk(
        category=KB_CATEGORY,
        questions=rq.standard_question,
        answer=answer,
        section_path=None,
        content_type=KB_CONTENT_TYPE,
        is_key_clause=False,
    )
    added = await write_chunks(session, [chunk])
    rows = await _rows_to_vectorize(session, chunk)
    if rows:
        store = get_vector_store(settings.milvus_uri, settings.milvus_collection)
        embedder = get_embedder(
            settings.embedding_model_path,
            settings.embedding_max_length,
            settings.embedding_batch_size,
        )
        try:
            # `ensure_collection` 是幂等的:集合在就跳过。放在这里是为了让
            # 「通过」这件事**自己站得住** —— 一台刚起来的 Milvus(集合还没建)
            # 不该让审核台上每一次通过都 502 到有人想起来去点管理台那个按钮。
            store.ensure_collection()
            await vectorize_rows(session, store, embedder, rows)
        except Exception as exc:  # noqa: BLE001
            # **不许静默降级成「写进 MySQL 了但检索不到」** —— 那是本仓那条
            # 「基础设施故障绝不伪装成成功」的反面。502 + 固定文案,并把原始异常
            # 记进服务日志(它只进日志、不出站)。
            #
            # 抓 `Exception` 而不是某个窄类型:`vectorize_rows` **不是**翻译边界
            # (`app/retrieval/search.py` 才是),pymilvus / torch / SQLAlchemy 的
            # 裸异常都会从它那儿原样出来。收窄成 `ToolInfrastructureError` 的话,
            # 真实故障会变成 500 + 一段裸的驱动文本。
            try:
                # 回滚再抛:失败若来自 `vectorize_rows` 里那次 `commit`(库层错误),
                # 会话会停在**待回滚**态 —— 不回滚的话连收尾都可能抛
                # PendingRollbackError,把 502 变成 500。
                await session.rollback()
            except Exception:  # noqa: BLE001
                logger.warning("向量化失败后回滚也失败,保留原异常", exc_info=True)
            logger.error("审核通过后向量化失败 review_id=%s", review_id, exc_info=True)
            raise HTTPException(
                status_code=502, detail=VECTORIZE_FAILED_DETAIL) from exc

    # ⚠️ 状态**在向量化成功之后**才改:上面任何一条失败路径(502)都要让这一行
    # 留在 pending,审核人才能重试。反过来先改后向量化的话,失败时那一行已经
    # approved,而内容永远进不了检索 —— 一次点击就把这条缺口「处理」掉了。
    rq.status = "approved"
    rq.approved_answer = answer
    rq.reviewed_at = func.now()
    await session.commit()
    logger.info(
        "审核通过 review_id=%s chunks_added=%s vectorized=%s",
        review_id, added, len(rows),
    )
    return {"ok": True, "chunks_added": added, "vectorized": len(rows)}


@router.post("/api/review/{review_id}/reject")
async def reject_review(review_id: int, session=Depends(get_session)) -> dict:
    """驳回:只改状态,**一个字都不碰知识库**(也不向量化)。

    「驳回也顺手写一条进去」会很隐蔽:驳回的语义是「这条不该进知识库」,
    而写进去之后**用户问同一个问题就会拿到一条被驳回的答案**,没有任何东西报错。
    """
    rq = await session.get(ReviewQueue, review_id)
    if rq is None or rq.status != "pending":
        raise HTTPException(status_code=404, detail="待审问题不存在或已处理")
    rq.status = "rejected"
    rq.reviewed_at = func.now()
    await session.commit()
    return {"ok": True, "status": "rejected"}
