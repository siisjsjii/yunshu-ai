"""会话侧栏的两个**只读**端点(spec §5.1 / §5.2)。

前端要用它们画出「历史会话」列表与「切回某个会话」的原文。两个端点都
**不碰模型、不碰图、不碰工具** —— 纯 DB 读,所以也**不加会话锁**:锁保护的是
「同一会话上两条消息的临界区」,而这里没有任何写、也没有跨行的一致性要求。

写接口(`POST /api/chat/stream`)是 MySQL 的**唯一**权威写入方,这里只读它写下的
东西 —— 尤其**不重新渲染**任何内容(见 `list_messages` 的说明)。
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, MessageRecord
from app.db.session import get_session

#: 无认证(spec §5.1 的产品口径),所以列表**固定**按这个 user 过滤 ——
#: 与会话端点 `request.user_id or "demo-user"` 的默认值**同一个字面量**:
#: 两边不一致的话,前端建出来的会话一个都不会出现在列表里,而两边都不报错。
DEMO_USER = "demo-user"

#: `messages` 表里**不进对话回载**的那一类行(spec §5.2)。
#:
#: ch07 起工具结果也落这张表 —— 写入方是 `app/agent/nodes.py`(把 ReAct 往返的
#: LangChain 消息翻成 `app.schemas.Message`,content 取 `m.content or ""`)
#: → `app/services/history.py:append_turn`(真正 `session.add` 那几个 `MessageRecord`
#: 的地方)。**不是** `app/memory/journal.py`:那个模块是 `model_ctx` / `history_ctx`
#: 那几行 JSON 上下文日志的组装,一条表都不写。
#:
#: 这些行是**模型与工具之间**的往返,不是用户看见过的对话。回载给侧栏的话,
#: `{"order_no":"1002","status":"已取消"}` 这种**原始工具载荷**会被当成一条消息气泡画出来。
TOOL_ROLE = "tool"

#: 预览取前多少字(spec §5.1)。
PREVIEW_CHARS = 30

router = APIRouter()


async def _preview(session: AsyncSession, conversation_id: str) -> str:
    """该会话**第一条 `role='user'` 消息**的前 30 字;没有则空串(spec §5.1)。

    为什么读**整段**再用 Python 取第一条 user 消息,而不是
    `WHERE role='user' LIMIT 1`:**`LIMIT` 在替身里不生效** ——
    `tests/test_api_conversations.py` 的 messages 分支按 where 筛完就**整段**
    返回(它按 id 排了序,但不认 `LIMIT`),于是「取第一条」这一步在**那个替身上
    仍然由 Python 完成**;真写 `LIMIT` 的话,单测绿得毫无意义(它根本没有 LIMIT 语义)。
    (替身**能**解析复合 where —— `BooleanClauseList` 那一支,`list_messages` 的
    `role != 'tool'` + `content != ''` 就靠它;这里没写 SQL 过滤**不是**因为替身不支持。)
    写成「按会话查 + Python 里取」是同一个语义,真实库上结果也完全一致 ——
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
    """某会话**用户看见过的**对话,**按 id 升序**,回的是**原文**(spec §5.2)。

    为什么是原文:侧栏切回来要看的就是「当初聊了什么」。**不能**拿发给模型的
    那份精简版回载 —— 层 2 的截短(`app/memory/layers.py:truncate`,
    assistant 50 字 + `…`、tool 结果 60 字)是**为模型省的**,回给它等于把
    「这条回复本来只有 50 字」写进 UI,而**没有任何东西报错**;梗概更不能回:
    它是**替换物**,原文还在库里,回梗概等于让用户看不见自己说过的话。
    本端点因此**不导入 `app.memory.layers`** —— 让它连误用的机会都没有。

    **两条「用户没看见过的内部机制」都不回载**(spec §5.2 裁定):

    ① `role='tool'` 的行 —— ch07 起工具结果也落这张表(见 `TOOL_ROLE` 那段),
       回给侧栏的话 `{"order_no":"1002","status":"已取消"}` 这类**原始工具载荷**
       会被当成一条消息气泡画出来;
    ② **`content` 为空的行** —— 那条 assistant 消息是「**只申请调用工具、还没产出
       文字**」的形态(`app/agent/nodes.py` 把它写成 `content=m.content or ""`,
       它身上只有 `tool_calls`),回给侧栏就是**一个空气泡**。

    **为什么取 `content != ''` 这个更窄的条件,而不是「assistant 且不带 tool_calls」**:

    两者**不等价** —— 后者会**误伤用户看见过的东西**。`app/agent/nodes.py` 的
    `_stream_round` **边累积 chunk 边把文字发 token 帧**(`if chunk.text: emit(...)`),
    而 `_lc_to_records` 把 `content=m.content or ""` 与 `tool_calls=m.tool_calls or None`
    **一起**写库 ⇒ **「先说了一句开场白、再申请调用工具」那种 assistant 行是结构上
    可达的**。对那种行:
    · `content != ''` ⇒ **返回它**(那段开场白**以 token 帧流出去过,是用户看见过的**);
    · 「不带 `tool_calls`」⇒ **丢掉它**(把用户见过的一句话抹掉)。

    所以本条件是**更窄、更保守**的那一个:它只滤掉**用户没见过的空气泡**,
    不误伤任何见过的东西。为什么它不会顺手滤掉别的:
    · `user` 行不可能是空串(`ChatRequest.message` 是 `min_length=1`,续跑那条路
      根本不写新行;真实库实测空 content 的行**全部**是 `role='assistant'`,别的
      角色 0 条);
    · `tool` 行已被①挡掉。
    (更宽的那个写法要排除的正是「只有 tool_calls、一个字都没有」那一小类,
    而那与「content 为空」在真实数据上重合 —— 但**不要**据此把它当成等价物:
    它多滤掉的那部分恰恰是用户见过的。)

    **两条过滤都在 SQL 里**,不是读回来再筛 —— 与列表的 user 过滤同一条理由:
    替身验不出「端点有没有传对 SQL」。

    ⚠️ **「按 id 升序」由 db 用例钉,单测钉不住** —— 替身的 messages 分支自己就按
    `m.id` 排(端点把 `order_by` 反过来它照样绿)。db 用例里探针消息**一条一 commit
    按内容顺序插入**(自增 id 因此严格递增、与期望顺序同向),端点写成 `.desc()`
    就与断言反向、当场红。

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
            .where(
                MessageRecord.conversation_id == conversation_id,
                MessageRecord.role != TOOL_ROLE,
                # 空 content = 「只申请了工具调用、一个字都没产出」那条 assistant
                # ⇒ 空气泡。**不能**改成「assistant 且不带 tool_calls」:
                # 那会连「先说了开场白、再申请调用工具」的行一起滤掉,而那句
                # 开场白是**发过 token 帧、用户看见过**的(理由见 docstring)。
                MessageRecord.content != "",
            )
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
