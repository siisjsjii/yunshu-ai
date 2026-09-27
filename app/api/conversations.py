"""会话侧栏的两个**只读**端点(spec §5.1 / §5.2)。

前端要用它们画出「历史会话」列表与「切回某个会话」的原文。两个端点都
**不碰模型、不碰图、不碰工具** —— 纯 DB 读,所以也**不加会话锁**:锁保护的是
「同一会话上两条消息的临界区」,而这里没有任何写、也没有跨行的一致性要求。

写接口(`POST /api/chat/stream`)是 MySQL 的**唯一**权威写入方,这里只读它写下的
东西 —— 尤其**不重新渲染**任何内容(见 `list_messages` 的说明)。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import AuthenticatedUser, require_user
from app.db.models import Conversation, MessageRecord
from app.db.session import get_session
from app.services.history import get_owned_conversation

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

#: 「这一行的 `tool_calls` 里真的有工具调用」**唯一**正确的 SQL 判据。
#:
#: ⚠️ **必须是 `JSON_TYPE(citations 那一列同理) = 'ARRAY'`,不是另两种写法**
#: —— 本机 2026-09-27 实测三种写法在整张 `messages` 上的读数:
#:
#: | 写法 | 行数 |
#: |---|---|
#: | `tool_calls IS NOT NULL` | **2721** |
#: | **`JSON_TYPE(tool_calls) = 'ARRAY'`** | **336** ✓ |
#: | `JSON_LENGTH(tool_calls) > 0` | **2721** ← 不安全 |
#:
#: 前两种都**不是**「有工具调用」:这一列是 `JSON`,而 Python 的 `None` 落库是
#: **字面 JSON `null`**(`none_as_null=False`)—— 它 **SQL 上不是 NULL**(所以
#: `IS NOT NULL` 为真),而 `JSON_LENGTH` 对 JSON **标量**返回 **1**(所以 `> 0`
#: 为真,**而且它对任何一行都恒真**)。⇒ 这两条判据都会把「没有工具调用」的行
#: 当成有。同族陷阱见 `app/db/models.py` 的 `citations` 列注释与 ch09 的
#: `evidence_snapshot` 那一条。
#:
#: ⚠️ **但第三种写法今天在行为上区分不了**(本机实测,别把它读成「这条判据
#: 被测试守住了」):本查询的形态是 `role != 'tool' AND (content != '' OR <判据>)`,
#: 而三种判据取到的行数**逐位相同**(都是 2238)—— 因为 `role != 'tool'` 且
#: `content = ''` 且「有非 NULL 的 tool_calls 但不是 ARRAY」的行**一行都没有**
#: (实测 0 行;所有 `content=''` 的 assistant 行**都**带 ARRAY)。
#: 所以这条判据今天守的是**将来**:哪一天某一行写成了「content 为空、又没有工具
#: 调用」,前两种写法就会把它当成「有工具调用」捞出来。判据形状由
#: `tests/test_api_conversations.py` 的替身钉(它只认这一种,别的**当场抛**)。
JSON_ARRAY = "ARRAY"

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
async def list_conversations(
    user: Annotated[AuthenticatedUser, Depends(require_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    """侧栏列表:`{"items": [{id, created_at, preview, summarized}]}`,新在前。

    过滤与排序都在 **SQL** 里(`WHERE user = <当前登录用户> ORDER BY created_at
    DESC`,spec §5.1)。

    ⚠️ **过滤值来自 token,不再是一个常量**(认证,2026-09-27)。此前这里写死
    `DEMO_USER = "demo-user"`,与会话端点 `request.user_id or "demo-user"` 的
    默认值**同一个字面量**;那个字面量连同 `ChatRequest.user_id` 一起删了 ——
    留下它的话,「谁建的会话」与「列表查谁」会**各自漂移**,而两边都不报错
    (表现是所有会话从侧栏消失)。

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
            .where(Conversation.user == user.username)
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
    conversation_id: str,
    user: Annotated[AuthenticatedUser, Depends(require_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    """某会话**用户看见过的**对话,**按 id 升序**,回的是**原文**(spec §5.2)。

    为什么是原文:侧栏切回来要看的就是「当初聊了什么」。**不能**拿发给模型的
    那份精简版回载 —— 层 2 的截短(`app/memory/layers.py:truncate`,
    assistant 50 字 + `…`、tool 结果 60 字)是**为模型省的**,回给它等于把
    「这条回复本来只有 50 字」写进 UI,而**没有任何东西报错**;梗概更不能回:
    它是**替换物**,原文还在库里,回梗概等于让用户看不见自己说过的话。
    本端点因此**不导入 `app.memory.layers`** —— 让它连误用的机会都没有。

    ---- 每一条里多带的两样(2026-09-27 修复:回载丢掉了「齿轮」与「文档链接」)----

    ---- 每一条里多带的两样,以及**按轮归并**(2026-09-27 修复)----

    ⚠️ **返回的 item 不再一一对应表里的行**(项数**少于**行数)。先看两样东西:

    · `tool_calls` → 工具齿轮(名字取每项的 `name`,与 `tool_call` 帧的
      `payload.name` 同一个来源 —— 实测真实库里这一列的形状是
      `[{"id":…, "name":…, "args":…, "type":"tool_call"}]`,**不是** OpenAI 生
      `function.name` 那一层);
    · `citations` → `[n]` 可点(`app/static/index.html:makeCitesClickable`),
      **原样透传**该行的列值。

    **为什么要按轮归并**(而不是「一条行 = 一条 item」):直播时齿轮与正文
    **本来就在同一个气泡里** —— `tool_call` 帧与 `token` 帧都往**同一个 ctx** 上画
    (`app/static/index.html` 的 `handleBlock`),用户看见的是「一个气泡,上面挂着
    这一轮所有的齿轮」。而库里长这样(`app/agent/nodes.py` 的 `_stream_round`
    + `_lc_to_records` 逐行落库):

    ```
    user("订单 1002 能退吗")
    assistant(content="", tool_calls=[query_order])   ← 齿轮在这条上(**没有正文**)
    tool('{"status":"已取消"}')                        ← 不回载(见下①)
    assistant("按政策可以退[1]。")                     ← 正文在这条上(没有 tool_calls)
    ```

    一条行一条 item 的话,回载会得到「一个空气泡 + 一句没有齿轮的回复」,
    与直播时那个气泡**长得完全不一样**。所以:以 `user` 行为轮边界,
    把**同一轮**里那些 `content=''` 的行的 `tool_calls` 累积起来,挂到**本轮
    那条带正文的 assistant item** 上。

    归并的三条细则(每条都由 `tests/test_api_conversations_db.py` 钉):

    · 带正文的 assistant 行的 `tool_calls` = 「本轮累积到的」**+「它自己的」**
      —— 后者不是可省的:实测有 **33** 条「先说了开场白、再申请调用工具」的行,
      **正文与 `tool_calls` 在同一条行上**(拿掉自己那份 = 这些齿轮静默消失);
    · `content=''` 的行**自己不输出**(它就是上一条的齿轮来源);
    · 一轮里**有多条**带正文的 assistant 行时**不合并** —— 它们各输出一条。
      合并会凭空抹掉一条用户看见过的回复。

    **边界(刻意的)**:某一轮**只有齿轮、没有带正文的 assistant 行**(例如
    模型申请了工具、那一轮随后报错)⇒ **仍然输出一条 `content: ""`、带
    `tool_calls` 的 assistant 项**。它看起来像个空气泡,但**有齿轮**;
    丢掉它 = 用户回载时连齿轮都看不见,而直播时他明明看见过。

    **两条「用户没看见过的内部机制」**:

    ① `role='tool'` 的行 —— ch07 起工具结果也落这张表(见 `TOOL_ROLE` 那段),
       回给侧栏的话 `{"order_no":"1002","status":"已取消"}` 这类**原始工具载荷**
       会被当成一条消息气泡画出来。**它今天也不进归并**:齿轮在 assistant 行上
       已经有了,工具结果行在这条链路上**没有任何消费者**。
    ② **`content=''` 且没有工具调用**的行 —— 回给侧栏就是**一个空气泡**
       (既没有正文、也没有齿轮可挂)。⚠️ **它不再是「一律滤掉」**:判据是
       **`content != '' OR JSON_TYPE(tool_calls) = 'ARRAY'`** —— 空 content 但
       **带工具调用**的行**要取出来**(它们正是齿轮的载体,实测 **303** 条)。
       那个判据为什么必须这么写、另两种写法的读数,见 `JSON_ARRAY` 那段。

    ⚠️ **恢复率前后各是多少**(本机 2026-09-27 实测,同一份数据):
    只按 `content != ''` 取(修复前)**33/336** 的带齿轮行能恢复(约 10%);
    归并之后 **336/336**(每一个齿轮都有归属)。判据是「有齿轮的行有没有
    被取出来、并挂到一条会输出的 item 上」。

    ⚠️ **`tool_result` 帧里的「成功/失败」没有落库**(没有那一列),所以回载的
    齿轮**画不出失败态** —— 这是已知局限,不是省略(前端也没有去猜,见
    `renderHistory` 那段)。

    **三条过滤都在 SQL 里**(归属 / `role` / `content` 与齿轮),不是读回来再筛
    —— 与列表的 user 过滤同一条理由:替身验不出「端点有没有传对 SQL」。

    ⚠️ **「按 id 升序」由 db 用例钉,单测钉不住** —— 替身的 messages 分支自己就按
    `m.id` 排(端点把 `order_by` 反过来它照样绿)。db 用例里探针消息**一条一 commit
    按内容顺序插入**(自增 id 因此严格递增、与期望顺序同向),端点写成 `.desc()`
    就与断言反向、当场红。

    404 表示**两件事之一**:「这个 id 在库里不存在」,或「它**不是你的**」
    (认证,2026-09-27)。两者**必须共用一个出口** —— 分开回(比如别人的回 403)
    等于承认「这个 id 存在」,那就是一个可枚举的接口。前端点的那条会话也可能
    已被删/清库,这时该给一个明确的「会话不存在」,而不是 200 + 空列表
    (空列表是「这个会话真的没有消息」的语义,两者混在一起,前端分不出要画哪个)。
    """
    # 归属检查在**读消息之前**:别人的会话连一行消息都不该被查出来
    # (顺序反了的话,`rows` 那一次查询已经把别人的消息读进了内存,虽然最终
    #  会被 404 挡住 —— 「读到了再丢掉」与「根本不读」在响应上一样,区别只在
    #  有没有那次越权读)。
    conv = await get_owned_conversation(
        session=session, conversation_id=conversation_id, user_id=user.username
    )
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    rows = (
        await session.execute(
            select(MessageRecord)
            .where(
                MessageRecord.conversation_id == conversation_id,
                MessageRecord.role != TOOL_ROLE,
                # 空 content 的行**不再一律滤掉** —— 条件是「有正文 **或**
                # 有工具调用」,理由见 docstring 的「按轮归并」那一段:
                # 带 tool_calls 的空 content 行**正是**齿轮的载体(实测 303 条),
                # 滤掉它们 = 齿轮回不来。
                #
                # 判据用 `JSON_TYPE(...) = 'ARRAY'`(理由与另两种写法的读数见
                # `JSON_ARRAY` 那段),**不要**换成 `IS NOT NULL` / `JSON_LENGTH > 0`。
                or_(
                    MessageRecord.content != "",
                    func.JSON_TYPE(MessageRecord.tool_calls) == JSON_ARRAY,
                ),
            )
            .order_by(MessageRecord.id)
        )
    ).scalars().all()

    # ── 按轮归并(行 → 项)──────────────────────────────────
    #
    # 一轮 = 一条 `user` 行到**下一条** `user` 行之间。
    #
    # 为什么按轮而不是按行:**直播时齿轮与正文本来就在同一个气泡里** ——
    # `tool_call` 帧与 `token` 帧都往**同一个 ctx** 上画(见
    # `app/static/index.html` 的 `handleBlock`),所以用户看见的是「一个气泡,
    # 上面挂着这一轮所有的齿轮」。而回载拿到的是一**行**一行:齿轮在
    # `content=''` 的那条 assistant 行上,正文在**最后**那条。要让回载长得和
    # 直播一样,就必须做这层归并。
    #
    # 归并规则:
    #   · 本轮**带正文**的 assistant 行 → 输出一条 item,它的 `tool_calls` 是
    #     「本轮此前累积到的 + 它**自己**的」(后者不是可省的:实测有 33 条
    #     「先说了开场白、再申请调用工具」的行,**正文与 tool_calls 在同一条行上**);
    #   · `content=''` 的行 → 只把 `tool_calls` 收进累积区,**自己不输出**
    #     (它就是上面那条的齿轮来源);
    #   · 一轮里**有多条**带正文的 assistant 行时**不合并**(它们各输出一条)。
    #     这是刻意的:合并会凭空抹掉一条用户看见过的回复。
    #
    # ⚠️ **输出的项数因此会少于行数** —— 被归并掉的正是那些空 content 的行。
    # (前端不需要知道这件事:它按 item 画,拿到什么画什么。)
    items: list[dict] = []
    pending_calls: list = []
    pending_from = None          # 本轮**第一个**贡献齿轮的行的 created_at(给孤立项用)

    def _emit_user(m) -> None:
        items.append({
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat(),
            "tool_calls": None,
            "citations": None,
        })

    def _emit_assistant(m, calls) -> None:
        items.append({
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat(),
            # 没有齿轮时给 **None**,不是 `[]`(前端写的是 `m.tool_calls || []`;
            # 而空数组与「这一行没有工具调用」在库里本来就不可区分)。
            "tool_calls": list(calls) or None,
            "citations": m.citations,
        })

    def _flush_turn() -> None:
        """一轮结束:只有齿轮、**没有**带正文的 assistant 行 ⇒ 补一条孤立项。

        ⚠️ **这是刻意的,不是漏网**:一轮在「模型申请了工具、但那一轮报错/没
        产出正文」时就是这个形状(`app/agent/nodes.py` 先落带 `tool_calls` 的
        空行,收尾那条根本没写出来)。丢掉它 = 用户回载时**连齿轮都看不见**,
        而直播时他明明看见过。它看起来像个**空气泡**(content 是空串),但**带齿轮**
        —— 前端照常画(徽章区有内容,正文区是空的)。
        """
        nonlocal pending_calls, pending_from
        if pending_calls:
            # `pending_from` **不可能是 None**:它就是在累积第一条时被赋值的
            # (`tool_calls` 非空 ⇒ 一定走过那一支)⇒ 这里**不写 `or 兜底`**,
            # 那种兜底只会把「累积区与来源行脱节」这个真实的错法盖住。
            items.append({
                "role": "assistant",
                "content": "",
                "created_at": pending_from.isoformat(),
                "tool_calls": list(pending_calls),
                # 空 content 的行上**永远**是 None:`append_turn` 只把引用挂到
                # 带正文的那条 assistant 行上(见 `app/services/history.py`)。
                "citations": None,
            })
        pending_calls = []
        pending_from = None

    for m in rows:
        if m.role == "user":
            _flush_turn()        # 上一条 user 到这一条 user 之间就是上一轮
            _emit_user(m)
            continue
        if m.content:
            _emit_assistant(m, [*pending_calls, *(m.tool_calls or [])])
            pending_calls, pending_from = [], None
        else:
            # 空 content 的行:只可能是「有工具调用」(SQL 的条件②把其余的挡掉了)。
            if not pending_calls:
                pending_from = m.created_at
            pending_calls.extend(m.tool_calls or [])
    _flush_turn()                # 最后那一轮

    return {"items": items}
