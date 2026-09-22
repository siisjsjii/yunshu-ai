"""会话侧栏的两个**只读**端点(spec §5.1 / §5.2)。

前端要用它们画出「历史会话」列表与「切回某个会话」的原文。两个端点都
**不碰模型、不碰图、不碰工具** —— 纯 DB 读,所以也**不加会话锁**:锁保护的是
「同一会话上两条消息的临界区」,而这里没有任何写、也没有跨行的一致性要求。

写接口(`POST /api/chat/stream`)是 MySQL 的**唯一**权威写入方,这里只读它写下的
东西 —— 尤其**不重新渲染**任何内容(见 `list_messages` 的说明)。
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, MessageRecord
from app.db.session import get_session

logger = logging.getLogger(__name__)

#: 无认证(spec §5.1 的产品口径),所以列表**固定**按这个 user 过滤 ——
#: 与会话端点 `request.user_id or "demo-user"` 的默认值**同一个字面量**:
#: 两边不一致的话,前端建出来的会话一个都不会出现在列表里,而两边都不报错。
DEMO_USER = "demo-user"

#: 预览取前多少字(spec §5.1)。
PREVIEW_CHARS = 30

router = APIRouter()


async def _preview(session: AsyncSession, conversation_id: str) -> str:
    """该会话**第一条 `role='user'` 消息**的前 30 字;没有则空串(spec §5.1)。

    为什么读**整段**再用 Python 取第一条 user 消息,而不是
    `WHERE role='user' LIMIT 1`:替身(见 `tests/test_api_conversations.py`)把
    「`MessageRecord` 上有 where 子句」定义为**按会话查全段**,不支持第二条
    `where`;`LIMIT` 在替身里也不生效(它按 id 排完就全给回来)。写成
    「按会话查 + Python 里取」是同一个语义,且在真实库上结果完全一致 ——
    会话的消息量是演示规模,这一点点多读不构成理由去为它另立一条替身分支。

    **必须是第一条 user 消息,不是最后一条、也不是「第一条消息」**:
    侧栏要的是「这次聊的是什么事」的入口印象,而最后一条 user 消息是
    「后来又问了什么」(spec §5.1 的原文:`第一条 role='user' 消息`)。
    """
    rows = (
        await session.execute(
            select(MessageRecord)
            .where(MessageRecord.conversation_id == conversation_id)
            .order_by(MessageRecord.id)
        )
    ).scalars().all()
    first_user = next((m for m in rows if m.role == "user"), None)
    return first_user.content[:PREVIEW_CHARS] if first_user is not None else ""


@router.get("/api/conversations")
async def list_conversations(session: AsyncSession = Depends(get_session)) -> dict:
    """侧栏列表:`{"items": [{id, created_at, preview, summarized}]}`,新在前。

    过滤与排序都在 **SQL** 里(`WHERE user = 'demo-user' ORDER BY created_at DESC`,
    spec §5.1 的字面)。`DEMO_USER` 与会话端点 `request.user_id or "demo-user"`
    的默认值**同一个字面量**:两边不一致的话,前端建出来的会话一个都不会出现在
    列表里,而两边都不报错。

    ⚠️ **「过滤对不对」与「顺序对不对」这两件事,替身验不出来** ——
    `tests/test_api_conversations.py` 的替身自己就会按 `created_at` 倒序排、
    也可以选择自己把 user 筛掉,于是端点把整个 `.where()` / `.order_by()`
    删掉照样绿(替身替它把事做了)。那两条语义因此改由
    `tests/test_api_conversations_db.py` 在**真实库**上钉(造出顺序与过滤
    都能被观测的输入)。

    **不分页**(spec §5.1:演示规模,与会话数的量级匹配;不是分页接口,别按分页
    写前端)。真实库上这个列表实测有 322 条(历次验收累积)—— 仍然全量返回。

    **`preview` 是 N+1 次查询**(每条会话一次,见 `_preview`)。演示规模下这个
    代价可接受(百条量级、一次请求一串主键索引点查);真要收成一条 SQL,得按
    会话分组取每组第一条 user 消息(窗口函数),而那时「不分页」这条决定也要
    一起重估 —— 两件事的前提是同一个(会话数还小)。
    """
    rows = (
        await session.execute(
            select(Conversation)
            .where(Conversation.user == DEMO_USER)
            .order_by(Conversation.created_at.desc())
        )
    ).scalars().all()

    items = []
    for conv in rows:
        items.append(
            {
                "id": conv.id,
                "created_at": conv.created_at.isoformat(),
                # N+1 次查询,每条会话一次。同样是演示规模的取舍:真要在一条
                # SQL 里做,得按会话分组取每组的第一条 user 消息(窗口函数),
                # 而那条 SQL 在替身里同样无法表达(见 `_preview`)。
                "preview": await _preview(session, conv.id),
                # 读者是那两个锚点里的 `summary_upto_msg_id`,`> 0` = 已有梗概覆盖
                # (spec §3.1:`0` = 尚无任何梗概)。**不是**「有没有
                # `conversation_summaries` 行」—— 正常情况下两者一致,而锚点是
                # **权威**(梗概覆盖到哪条只有它说了算);数行数还要求把梗概表
                # 读进来,凭空多一次查询。
                "summarized": conv.summary_upto_msg_id > 0,
            }
        )
    return {"items": items}


@router.get("/api/conversations/{conversation_id}/messages")
async def list_messages(
    conversation_id: str, session: AsyncSession = Depends(get_session)
) -> dict:
    """某会话的全部消息,**按 id 升序**,回的是**原文**(spec §5.2)。

    为什么是原文:侧栏切回来要看的就是「当初聊了什么」。**不能**拿发给模型的
    那份精简版回载 —— 层 2 的截短(`app/memory/layers.py:truncate`,
    assistant 50 字 + `…`、tool 结果 60 字)是**为模型省的**,回给它等于把
    「这条回复本来只有 50 字」写进 UI,而**没有任何东西报错**;梗概更不能回:
    它是**替换物**,原文还在库里,回梗概等于让用户看不见自己说过的话。
    本端点因此**不导入 `app.memory.layers`** —— 让它连误用的机会都没有。

    404 只表示「这个 id 在库里不存在」:前端点的那条会话可能已被删/清库,
    这时该给一个明确的「会话不存在」,而不是 200 + 空列表(空列表是
    「这个会话真的没有消息」的语义,两者混在一起,前端分不出要画哪个)。
    """
    conv = (
        await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
    ).scalars().one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    rows = (
        await session.execute(
            select(MessageRecord)
            .where(MessageRecord.conversation_id == conversation_id)
            .order_by(MessageRecord.id)
        )
    ).scalars().all()
    return {
        "items": [
            {
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
            }
            for m in rows
        ]
    }
