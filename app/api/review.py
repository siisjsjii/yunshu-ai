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

## 502 的收尾:**把本次刚写进去的草稿删掉**(F4)

`write_chunks` **提交在向量化之前**(那是 ch03 的双写顺序:MySQL 是权威源、
Milvus 崩了可以按 pending 重扫补齐)。于是失败路径会留下一条**没人核准过**的行:

    approve#1(答案 A)→ 写入 (Q,A,pending)→ Milvus 炸 ⇒ 502
    → 审核人把答案改成 B 重试 ⇒ (Q,B) 被向量化、队列行 approved,
      而 (Q,A,pending) **永久残留**
    → 下一次 `build_kb.py` / 管理台的向量化任务(`vectorize_pending` 全表扫
      pending,**不筛 category**)把它送进 Milvus ⇒ **被否掉的草稿静默进了
      可检索知识库**。

所以 502 与取消两条路径都调 `_discard_drafts`:删掉**本次新增的 id 里、还没
向量化的**那些行。判据是「**本次新增的 id**」(调用前后各查一次三元组、取差集),
**不是**「这个三元组的所有 pending 行」—— 后者会删掉**别人**写进去的行(同一
三元组的 pending 行可能是另一个并发请求刚写的、也可能是崩溃遗留的,而本仓在这条
路上**刻意不加锁**)。代价与残留见 `_discard_drafts` 的 docstring。

## 状态只有一个写口

`review_queue.status` 的**唯一**写入方就是本模块的 `approve` / `reject`
(飞轮流水线只建 `pending` 行)。两个端点都按 `status != 'pending'` 挡第二次调用
⇒ **第二次是 404,不是静默成功**(静默成功会让审核台把一次重复点击读成
「又处理了一条」)。

## 三处「T15 复审已裁定接受」的取舍,写在这里免得后人当成 bug

- **`?status=` 是闭集**(`pending|approved|rejected`),**没有「全部」这个取值** ——
  要看全部得逐个指名(审核台今天只要 `pending`);未知取值 422,不静默当 pending。
- **`approve` 接受空 body**(`{}`)或**不带 body**:审核人点了「通过」而没改答案
  是最常见的一次操作,只收必填 body 会让它 422。
- **`ensure_collection()`** 幂等,每次通过多一次 `has_collection` RPC —— 换来的
  是「一台刚起来的 Milvus 不会让审核台从此每次都 502」。
- **并发双 `approve` 不加锁**(复审已裁定:重复向量化同一行、同主键幂等无害;
  只有两个请求落在最初几毫秒内才会多出一行,有界、可恢复)。前端由 T16 用
  「请求在飞时禁用按钮」关掉现实触发路径。

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

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete, func, select

from app.config import Settings, get_settings
from app.db.models import KnowledgeChunk, LowConfidenceQuestion, ReviewQueue
from app.db.session import get_session
from app.kb.chunker import Chunk
from app.kb.writer import vectorize_rows, write_chunks
from app.retrieval.embedder import get_embedder
from app.retrieval.milvus import get_vector_store
from app.sanitize import redact_api_key

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


async def _rows_with_triple(session, chunk: Chunk) -> list[KnowledgeChunk]:
    """这个三元组在库里的**全部**行(含 `done`)。

    只给 502 的清理用:写入前后各取一次,差集就是**本次这次调用写进去的**行。
    这也是「不许删别人的行」那条判据的**唯一**来源。
    """
    return list(
        (
            await session.execute(
                select(KnowledgeChunk).where(
                    KnowledgeChunk.category == chunk.category,
                    KnowledgeChunk.questions == chunk.questions,
                    KnowledgeChunk.answer == chunk.answer,
                )
            )
        )
        .scalars()
        .all()
    )


async def _rollback_quietly(session) -> None:
    """回滚,**自己炸了也不上抛**。

    失败若来自 `vectorize_rows` 里那次 `commit`(库层错误),会话会停在**待回滚**态
    —— 不回滚的话,后面的清理与 FastAPI 的收尾都可能抛 PendingRollbackError,
    把 502 变成 500(而 500 的文案指向别处)。
    """
    try:
        await session.rollback()
    except Exception:  # noqa: BLE001
        logger.warning("回滚失败,保留原异常", exc_info=True)


async def _discard_drafts(session, chunk: Chunk, before_ids: set[int]) -> int:
    """把**本次刚写进去、还没向量化**的行删掉。**保证不抛**,返回删了几条。

    ⚠️ 调用它之前先 `_rollback_quietly`:会话若是待回滚态,这条 `DELETE` 自己会
    撞 `PendingRollbackError`,那时草稿就清不掉了(而它**看起来**像"已经尽力")。

    为什么必须清:见模块 docstring「502 的收尾」—— 不清的话,一条**没人核准过**
    的草稿会被下一次全表 `vectorize_pending` 扫进 Milvus。

    判据写死的两条:
    - **只删「本次新增的 id」**(`before_ids` 的差集):同一三元组的 pending 行可能
      是别人写的,删它就是丢别人的数据,而两边都不报错;
    - **只删 `vectorize_status == 'pending'`**:已经 `done` 的行意味着 Milvus 里
      已经有它的向量,删了就是**知识消失**。

    已知残留(如实记,别读成"清干净了"):本次**没有**新增行(查重命中,比如上一次
    请求崩在写入与向量化之间留下的行)时,这条路径什么都不删 —— 那种行下一次通过
    仍会被 `_rows_to_vectorize` 捞起来向量化,而它**分不出**是不是别人的。
    另:若 Milvus 已经写成功、只是紧接着那次 `commit` 失败,这里会把 MySQL 行删掉
    (Milvus 里留下一个孤儿向量,检索时按"MySQL 里没有这一行"跳过)—— 队列行仍是
    pending,审核人重试即恢复。
    """
    try:
        after = {r.id for r in await _rows_with_triple(session, chunk)}
        doomed = sorted(after - set(before_ids))
        if not doomed:
            return 0
        await session.execute(
            delete(KnowledgeChunk).where(
                KnowledgeChunk.id.in_(doomed),
                KnowledgeChunk.vectorize_status == "pending",
            )
        )
        await session.commit()
        logger.warning(
            "审核通过失败,已清掉本次写入的 %d 条待向量化草稿:%s", len(doomed), doomed)
        return len(doomed)
    except BaseException:  # noqa: BLE001
        # 清理是**尽力而为**:它自己炸了也不许把 502 变成 500,更不许盖掉原异常
        # (取消那一支尤其:换上来的异常会让「客户端断开」变成别的语义)。
        logger.warning("清理本次写入的草稿失败(502 的语义不受影响)", exc_info=True)
        return 0


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
    # 写入**之前**记下这个三元组已有的 id —— 502 的清理只许删「本次新增的」那些,
    # 而写入一提交,这个信息就没有别的来源了(ch03 的双写顺序:MySQL 先落)。
    before_ids = {r.id for r in await _rows_with_triple(session, chunk)}

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
        except asyncio.CancelledError:
            # **取消不是故障**:客户端断开时那条请求就该停,这里不许把它翻成 502。
            # 但草稿照样要清 —— 不然一次"点了通过又立刻关页面"就会留下一条
            # 没人核准的行,等着被扫描进知识库。
            await _rollback_quietly(session)
            await _discard_drafts(session, chunk, before_ids)
            raise
        except Exception as exc:  # noqa: BLE001
            # **不许静默降级成「写进 MySQL 了但检索不到」** —— 那是本仓那条
            # 「基础设施故障绝不伪装成成功」的反面。502 + 固定文案,并把原始异常
            # 记进服务日志(它只进日志、不出站)。
            #
            # 抓 `Exception` 而不是某个窄类型:`vectorize_rows` **不是**翻译边界
            # (`app/retrieval/search.py` 才是),pymilvus / torch / SQLAlchemy 的
            # 裸异常都会从它那儿原样出来。收窄成 `ToolInfrastructureError` 的话,
            # 真实故障会变成 500 + 一段裸的驱动文本。
            await _rollback_quietly(session)
            await _discard_drafts(session, chunk, before_ids)
            logger.error("审核通过后向量化失败 review_id=%s", review_id, exc_info=True)
            raise HTTPException(
                status_code=502,
                # 固定文案**也要过一遍 `redact_api_key`**:它今天是常量、抹不到
                # 任何东西,但「所有出站文本都过同一个出口」这条规则要留成
                # **无例外**的(`app/api/refund.py:_infra_failure` 与
                # `app/api/chat.py` 那条 502 都是这么写的,理由在那边)。
                # 每次判断"这个字符串要不要脱敏"迟早会判错一次。
                detail=redact_api_key(VECTORIZE_FAILED_DETAIL, settings.openai_api_key),
            ) from exc

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
