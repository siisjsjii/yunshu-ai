"""用户满意度反馈(ch09 spec §6)。

`up` 只记日志不落池;`down` 把**该轮的用户问题**送进低置信度池,并
**按该轮问题重跑一次检索尽力回捞**召回片段。

⚠️ **回捞是「重跑」不是「当轮」** —— 本仓**没有任何地方持久化每轮的检索结果**
(`evidence` 只在图 state 里,intent 也没落库,spec §2.7)。所以 `evidence_snapshot`
是**事后重跑**的结果。为什么这样反而更好用:重跑召得到 ⇒「知识库有、当时没检到」;
召不到 ⇒「真缺这块」—— 正是审核人要判的那件事。
语义偏差已记在 spec §6.2(重跑用的是**当前**的知识库,而那一轮用的是**当时**的)。

这是本章飞轮的**第三个入口**(另两个是 `置信度闸` 与 `生成自评`),落的是**同一张**
`low_confidence_questions` 表 —— 三个入口靠 `entry_point` 在池子里分开。

本模块与 `app/agent/nodes.py:_snapshot` 是**同名但不同形状的两个函数**:
那边吃的是塞进图 state 的**键值 dict**(`c.get(...)`),这里吃的是
`retriever.search()` 刚返回的 **`RetrievedChunk` 对象**(`c.chunk_id`),
**刻意不抽公共函数**(抽的话要么给 `RetrievedChunk` 加适配、要么让节点侧多一层
转换,都不划算)—— 两处的交叉说明写在 `nodes.py` 那一侧。
"""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.models import LowConfidenceQuestion
from app.db.session import get_session
from app.flywheel.tasks import start_flywheel_job_safely
from app.kb.assess import record_low_confidence
from app.retrieval.search import KnowledgeRetriever
from app.tools.registry import build_retriever

logger = logging.getLogger(__name__)

#: 池子里「用户点了 👎」这个入口的取值(spec §7.2 的三个生产取值之一)。
#: 幂等查重与落池**共用这一个字面量** —— 两处写岔的话,查重会永远查不到,
#: 于是每次重复点击都**再落一行**,而没有任何东西报错。
ENTRY_USER_FEEDBACK = "用户反馈"

router = APIRouter()


class FeedbackRequest(BaseModel):
    """请求体(spec §6.1)。

    `question` 是**前端手上就有的那条用户消息原文** —— 不让后端从 `messages`
    反查,因为「一条问题在一次会话里被问了两次」时反查有歧义(spec §6.1 的表下)。

    `message_id` 是**请求契约的一部分但服务端不用它**:池子那张表上
    **没有存消息 id 的列**,而本章的 DDL 已经落地(`init_db.py` 永不加列),
    所以它既不能当幂等键、也不进库。留着是因为前端按 spec 会带上它,
    而幂等判据落在「会话 + 问题文本 + 入口」上(见 `post_feedback`)。
    """

    conversation_id: str = Field(min_length=1, max_length=32)
    question: str = Field(min_length=1, max_length=2000)
    message_id: int | None = None
    value: str = Field(pattern="^(up|down)$")


def _snapshot(chunks, *, settings: Settings) -> list[dict] | None:
    """召回块 → 落池用的快照(Top-N)。**空召回返回 `None`,不是 `[]`。**

    `None` 与 `[]` 在库里是两样东西:前者走 SQLAlchemy 的 JSON 列落成 **JSON
    `null`**,后者是一段真的 JSON 数组 —— 审核页把它们读成
    「没能回捞」vs「重跑过、零召回」。四键与 `nodes.py:_snapshot` 逐字对齐
    (审核页只有一套读法)。
    """
    if not chunks:
        return None
    return [
        {
            "chunk_id": c.chunk_id,
            "score": round(c.score, 4),
            "section_path": c.section_path,
            # 池子是给审核人看的窄表:整块原文塞进去只会让那一行读不动。
            "answer": (c.answer or "")[: settings.snapshot_answer_chars],
        }
        for c in chunks[: settings.snapshot_top_n]
    ]


@router.post("/api/feedback")
async def post_feedback(
    body: FeedbackRequest,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """👎 落池 + 尽力回捞;👍 只留一行日志。

    **`up` 走日志而不是池子**:池子是「答不上的问题」的待办队列,👍 进去会让
    飞轮去修一个**已经答对**的问题。留日志是为了让「用户满意」与「用户根本没点」
    在排查时能分开 —— 这与池子无关,所以不落库。
    """
    if body.value == "up":
        logger.info("feedback up:conv=%s(不落池)", body.conversation_id)
        return {"ok": True, "pooled": False}

    # 幂等:同一条消息的 down 重复提交只落一行。**池子上刻意没有唯一键**
    # (spec §7.1:查重是**语义**判断,唯一键只能管字面全等),所以查重在服务层
    # 做一次「该会话 + 该入口 + 该问题」的 SELECT。判据里**没有 `message_id`**:
    # 池子没有存它的列(见 `FeedbackRequest` 的说明),而问题文本正是下游飞轮
    # 归并的单位 —— 同一个问题被问了两次、被点了两次 👎,在池子里就是同一个缺口。
    existing = (
        await session.execute(
            select(LowConfidenceQuestion.id).where(
                LowConfidenceQuestion.source_conversation_id == body.conversation_id,
                LowConfidenceQuestion.entry_point == ENTRY_USER_FEEDBACK,
                LowConfidenceQuestion.question == body.question,
            )
        )
    ).first()
    if existing is not None:
        logger.info("feedback down 去重命中:conv=%s 该问题已在池中", body.conversation_id)
        return {"ok": True, "pooled": False, "reason": "already_pooled"}

    # 尽力回捞:重跑一次真实检索。**召不到就留空**(那正是「真缺这块」的信号)。
    snapshot = None
    try:
        retriever: KnowledgeRetriever = build_retriever(session, settings)
        chunks = await retriever.search(body.question)
        snapshot = _snapshot(chunks, settings=settings)
    except Exception:  # noqa: BLE001
        # 检索挂掉不许拦住落池 —— 落池才是这个端点的职责,快照只是附赠。
        # ⚠️ 但**必须响亮地记**(WARNING + traceback):快照为空与「检索故障」
        # 在**数据上长得一模一样**,审核人只能靠这条日志把它们分开。
        # ⚠️ 抓 `Exception` 而不是某个窄类型:回捞的四条腿(Milvus / 嵌入 /
        # 重排 / 回查)都已被 `retrieval/search.py` 翻成 `ToolInfrastructureError`,
        # 但这里的语义是「附赠品失败不拦路」,编程错误也不该把用户的 👎 打掉。
        # 取消(`CancelledError`)是 `BaseException`,**不在此列** —— 那要照常上抛。
        logger.warning("feedback 回捞召回片段失败,快照留空", exc_info=True)

    await record_low_confidence(
        session,
        question=body.question,
        source_conversation_id=body.conversation_id,
        entry_point=ENTRY_USER_FEEDBACK,
        reject_reason="用户点了 👎(未解决)",
        evidence_snapshot=snapshot,
    )
    # **落池之后** fire-and-forget 起一轮飞轮(ch09 §8.3;
    # `docs/superpowers/specs/2026-09-23-ecommerce-cs-ch09-observe-flywheel-design.md:805`
    # 写的是「**落池后** fire-and-forget 起一个后台任务」,**没有**限定入口)——
    # 与 `app/agent/nodes.py` 那两处(置信度闸 / 生成自评)同一个形状:不 await、
    # 也不许抛。这是飞轮的**第三个入口**(requirement ③ 把 👎 算作三个入口之一),
    # 少了这一行,用户点的那个 👎 只能等**人**去点管理台按钮才进飞轮。
    #
    # ⚠️ 放在 `record_low_confidence` **之后**:它是「刚落的那行顺手喂过去」,
    # 排在前面的话,飞轮会在这一行**还没提交**时就去 `WHERE matched_review_id
    # IS NULL` 里捞它 —— 捞不到(另一个 session、另一个提交),于是这一轮白跑,
    # 而返回体里的 `pooled: true` 让人以为喂过了。
    start_flywheel_job_safely(settings)

    return {"ok": True, "pooled": True, "snapshot_chunks": len(snapshot or [])}
