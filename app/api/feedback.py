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

`evidence_snapshot` 这一列在本端点有**三个**取值(不是两个,详见
`RECALL_FAILED_SNAPSHOT`):`None`(零召回)/ 真快照(列表)/ **哨兵(回捞失败)**。

本模块与 `app/agent/nodes.py:_snapshot` 是**同名但不同形状的两个函数**:
那边吃的是塞进图 state 的**键值 dict**(`c.get(...)`),这里吃的是
`retriever.search()` 刚返回的 **`RetrievedChunk` 对象**(`c.chunk_id`),
**刻意不抽公共函数**(抽的话要么给 `RetrievedChunk` 加适配、要么让节点侧多一层
转换,都不划算)—— 两处的交叉说明写在 `nodes.py` 那一侧。
"""

import asyncio
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.models import LowConfidenceQuestion
from app.db.session import get_session
from app.flywheel.tasks import start_flywheel_job_safely
from app.kb.assess import ENTRY_USER_FEEDBACK, record_low_confidence
from app.retrieval.search import KnowledgeRetriever
from app.tools.errors import ToolInfrastructureError
from app.tools.registry import build_retriever

logger = logging.getLogger(__name__)

#: 池子里「用户点了 👎」这个入口的取值(spec §7.2 的三个生产取值之一)。
#: 幂等查重与落池共用它 —— 两处写岔的话,查重会永远查不到,于是每次重复点击
#: 都**再落一行**,而没有任何东西报错。
#: ⚠️ **字面量本身住在 `app/kb/assess.py`**(落池那个写口),与另两个入口
#: (`ENTRY_GATE` / `ENTRY_SELF_ASSESS`)放在一起 —— 三个取值只该有一个定义处
#: (最终修复轮之前它们散在三个文件里,那正是「同一个值两处实现」的形状)。
#: 这里只 import,不再转手重定义。

#: 回捞**失败**时落进 `evidence_snapshot` 的哨兵(ch09 最终修复轮,复审 D5)。
#:
#: **为什么不复用 `None`**:本仓约定 `None` = 「当轮确实零召回」,而审核页正是
#: 靠这一列判**「知识库真缺这块」还是「有、但没检到」**。回捞失败与零召回在
#: **数据上一模一样**的话,事故现场就成了:Milvus 挂着,端点照样回
#: `200 {"pooled": true}`,人在 `admin.html` 上读到「召回片段快照:无」,
#: 于是得出一个**诊断**——「知识库没有这块知识」——而真相是**服务不可用**。
#: 本仓那条「基础设施故障绝不伪装成结果」在这里被违反得更隐蔽:
#: 它伪装成了**结论**。⇒ 现在它伪装不成,审核页会把哨兵单画一句
#: 「召回失败(检索服务不可用)」。
#:
#: **形状上不可能与真快照相撞**:真快照是**列表**(每项四键 `chunk_id` /
#: `score` / `section_path` / `answer`,见 `_snapshot`),哨兵是**单个 JSON
#: 对象**、只有一个键 —— 顶层类型不同(数组 vs 对象),读侧判 `Array.isArray`
#: 就能分开。**不要**把它写成 `[]`:那是第三个取值(空数组),而且
#: `scripts/acceptance_ch09.sh` 的 ② 专门断言 `SNAP=0` 要红。
#: **只读**:写进 JSON 列之后不再改动,别就地增删这个对象。
RECALL_FAILED_SNAPSHOT: dict = {"error": "recall_failed"}


async def _search_bounded(retriever, query: str, *, settings: Settings):
    """带墙钟上界的 `retriever.search`(最终修复轮,复审 D5)。

    **它只圈得住 `await` 的那一半 —— 这句话是结论,不是免责声明**:
    `KnowledgeRetriever.search` 内部的大头是**同步调用**(BGE-M3 的 torch 前向、
    `hybrid_search` 与重排的 pymilvus 往返),事件循环在它们里面**根本跑不到
    定时器**(ch07 实测过的那条:`wait_for` 的定时器在循环被阻塞时不触发,
    而且它**连「迟到触发」都不算准**——定时器回调要等控制权回到循环才处理,
    所以同步段真卡了 30s、定时器到 10s 也不会先炸)。⇒ 本函数买到的是
    「**收尾那几个 `await`**(`_load_rows` 的 MySQL 回查、取消路径上的 `rollback`)
    不许无限等」,不是「整条检索不许超过 N 秒」。

    **要连同步那半也圈住**,得把它挪进线程(`asyncio.to_thread`),但
    `_load_rows` 用的是调用方的 `AsyncSession`(不可跨线程)⇒ 那是重写检索器,
    不在一次修复轮的射程里。如实记账,别把本函数读成万能的。

    超时**翻成 `ToolInfrastructureError`**(本仓的基础设施故障词汇),
    而不是让调用方看见一个裸的 `TimeoutError`:与 `app/retrieval/search.py`
    那四条腿同一个出口;调用方(feedback)按「尽力而为」处理它,落哨兵。
    """
    try:
        return await asyncio.wait_for(
            retriever.search(query), timeout=settings.retrieval_timeout_seconds
        )
    except TimeoutError as exc:
        raise ToolInfrastructureError(
            f"知识检索超时(超过 {settings.retrieval_timeout_seconds} 秒)"
        ) from exc

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

    # 尽力回捞:重跑一次真实检索。三种结局**在数据上必须分得开**:
    #   召回到块 ⇒ 真快照(列表);重跑过、零召回 ⇒ `None`;回捞**失败** ⇒ 哨兵。
    # 最后那一支原先与「零召回」落成同一个值(都是 JSON `null`),审核页会把
    # **服务不可用**读成**知识库缺这块** —— 见 `RECALL_FAILED_SNAPSHOT`。
    snapshot = None
    recalled = 0          # 真快照有几条(`snapshot_chunks` 的**唯一**来源)
    try:
        retriever: KnowledgeRetriever = build_retriever(session, settings)
        chunks = await _search_bounded(retriever, body.question, settings=settings)
        recalled = len(chunks)
        snapshot = _snapshot(chunks, settings=settings)
    except Exception:  # noqa: BLE001
        # 检索挂掉不许拦住落池 —— 落池才是这个端点的职责,快照只是附赠。
        # ⚠️ 但**落哨兵 + 响亮地记**(WARNING + traceback):哨兵保证审核页上
        # 「回捞失败」不会被读成「零召回」,日志说清是哪一条腿挂的。
        # ⚠️ 抓 `Exception` 而不是某个窄类型:回捞的四条腿(Milvus / 嵌入 /
        # 重排 / 回查)都已被 `retrieval/search.py` 翻成 `ToolInfrastructureError`,
        # 而 `_search_bounded` 又给超时补了同一种;但这里的语义是「附赠品失败
        # 不拦路」,编程错误也不该把用户的 👎 打掉。
        # 取消(`CancelledError`)是 `BaseException`,**不在此列** —— 那要照常上抛。
        snapshot = RECALL_FAILED_SNAPSHOT
        logger.warning(
            "feedback 回捞召回片段失败,快照落哨兵 %s", RECALL_FAILED_SNAPSHOT,
            exc_info=True,
        )

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

    # `snapshot_chunks` 是**真快照的条数**(回捞失败那一支是 0 —— 哨兵不是片段)。
    # ⚠️ 别把它写成 `len(snapshot or [])`:哨兵是个 JSON 对象,`len()` 会给出 1,
    # 于是一次**回捞失败**会自报「快照里有一条」,而这个数正是前端/验收脚本
    # 用来判回捞成没成的(它也会把 `SNAP=1` 喂给验收 ② 那句「快照居然有内容」)。
    # 审核人看的是**列里的哨兵**,不是这个数 —— 两者说的是两件事,别互相推。
    return {"ok": True, "pooled": True, "snapshot_chunks": recalled}
