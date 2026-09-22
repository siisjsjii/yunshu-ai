"""ch07 T8:把层 2 的**原文**压成一段事实备忘 —— prompt + 触发判定 + 原子落库。

**这是本章唯一不可逆的动作。** 层 2 的截短只是把文本变短(原文还在库里、下一轮
还能重新截),梗概一旦落成,那段原文就**从此不再进上下文**。所以本模块每一步
判断都偏保守:

- **触发用严格的 `>`**:装得下就不压。压缩是手段、不是成绩(验收 3 的反向断言
  就是「默认窗口下纯聊天二十轮一次都不触发」);
- **触发读 `Layers.layer2_tokens`** —— 那是层 2 **截短后**的数。数条数看不见工具
  结果涨,而工具结果是本章最大的一块(单条封顶 1200 token,截完 70);
- **失败等于什么都没发生**:推进边界那一步只在落库成功**之后**。反过来写会
  让一段历史**永久消失**(区间已不在层 2 的读取范围里,而摘要表里没有替换物),
  且没有任何东西会报错。两步的原子性本身在
  `app/services/history.append_summary_and_advance`(T3);本模块只负责
  「**只在成功之后**调它,而且只调一次」。

**依赖方向的一处反向边(记账):** 本模块 `from app.services.history import …`,
而本仓的方向是 `services → memory`。这条与 `app/retrieval/search.py` 反向引用
`app/tools/errors.py` 同款,是 T8 brief 明确指定的消费关系(T3 把原子落库放在
`services/history.py`,因为那一步是 DB I/O)。只差一层,不构成环:
`services/history.py` 只依赖 `app.db.models` 与 `app.schemas`。

**本模块不依赖 LangChain**,也不 import `app.prompts`:摘要的 system prompt 与
消息组装都是自己的(`build_summary_messages` 给的是**纯 dict** —— ChatOpenAI
的 `ainvoke` 收得下 `{"role", "content"}` 这种形态,这在本仓是唯一能同时满足
「memory 不碰 LangChain」与「消息能真的发出去」的形状)。模型作为**参数**传进来,
与 `services/` 里那几个函数的既有约定一致(不在模块层建全局单例)。
"""

import logging
from collections.abc import Sequence

from app.memory.layers import Layers
from app.schemas import Message
from app.services.history import append_summary_and_advance

logger = logging.getLogger(__name__)

#: 四样提炼物 + 三条硬约束(spec §4.1,逐条写进来)。
#:
#: **为什么必须写「什么不算」**:ch03 的挖知识 prompt 首版漏了这条,把客服
#: 「抱歉查不到运费」这种**非答案**挖成了知识,直接把验收的正确答案挤下 top-1
#: —— **模型的失败被挖进知识库,再教它下次继续失败。** 梗概有同一个失效模式,
#: 而且更狠:梗概**每轮都注入一遍**,且原文不再喂,编出来的事实**无法追溯**。
SUMMARY_SYSTEM_PROMPT = """你要把一段电商客服对话压缩成一段**事实备忘**。

提炼四样,其他的都不要:
1. 用户问过的商品 / 款式;
2. 用户报过的订单号、手机号等标识(**原样抄写,不要改写**);
3. 用户明确的诉求;
4. **还没解决的问题**。

硬约束:
- **对话里没有出现的内容,一个字都不许编。** 没有订单号就不要写订单号。
- 寒暄、客套、致谢、对话状态(「用户发了一条消息」)一律不留。
- 长度控制在几十到一两百字。

输出这一段备忘本身,不要解释、不要加标题、不要输出 JSON。"""

#: 多段梗概之间的连接符。**必须与 `journal.history_ctx` 里算 `tokens.summary`
#: 用的拼法一致** —— 不一致的话,日志报的 token 数不是模型实际收到的那段文本
#: 的数,而那是一个「看起来正常、其实什么也没说」的观测面。
SUMMARY_SEPARATOR = "\n\n"


def render_turns(turns: Sequence[Message]) -> str:
    """把一段历史渲染成给摘要模型看的文本。**喂的是原文,不是截短版。**

    拿截短版去提炼 = 把截断损失**焊进梗概**,而梗概是**永久背景**:截掉的那部分
    原文从此不在任何地方出现过(层 2 里也没有了)。所以本函数**不调** `layers.truncate`,
    也不接 `Settings`。

    **`role="tool"` 的行必须在内**:用户报过的订单号与订单真实状态就住在工具结果里,
    而验收 2 的落点正是它(「最开始那个订单后来怎么说」要能靠梗概答对)。

    `assistant` 的 `tool_calls` **不单独渲染**:它带的是调用参数,而那份数据在紧随
    其后的工具结果里已经以完整形态出现过一次;再渲染一遍只是让同一串订单号在
    提炼素材里出现两次(同一事实重复出现会诱导模型把它写成一个"重要事实")。
    `content` 为空的行照渲染(模型只申请调工具的那条 assistant 就是空的),
    过滤掉它会让「谁调的工具」这条线索断掉。
    """
    return "\n".join(f"{turn.role}: {turn.content}" for turn in turns)


def build_summary_messages(*, turns: Sequence[Message]) -> list:
    """组装摘要请求:**system(那段 prompt)+ user(渲染出来的原文)**。

    返回**纯 dict**,不是 LangChain 消息对象 —— `app/memory/` 不依赖 LangChain
    是本仓的硬约定(见模块 docstring)。真机那条路上,`ainvoke` 拿到的就是它;
    形状能不能被吃下由 `convert_to_messages` 那条测试钉住。

    这里**只有两条消息**:摘要的输入是「原文区间」这一样东西,不喂上一段梗概
    (spec §3.3:同一事实被反复有损压缩会逐次失真,而「压完不回头重写」保证每段
    梗概只被压一次)。
    """
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": render_turns(turns)},
    ]


def should_summarize(layers: Layers, *, layer2_budget: int) -> bool:
    """层 2 的**截短后** token 数**严格超过**预算 ⇒ 触发摘要。

    纯函数、**不读 `Settings`**:读的话「边界」这件事就得靠构造一堆无关字段
    (或靠 `.env` 里的默认值)来测,而边界正是这一处唯一要钉的东西。

    `>` 而不是 `>=`:**装得下就不压**。差一个等号,就是「刚好装满时每轮压一次」
    与「一次都不压」的区别 —— 而摘要不可逆(压缩是成本,不是成绩)。
    """
    return layers.layer2_tokens > layer2_budget


def join_summaries(summaries: Sequence[tuple[int, str]]) -> str:
    """把 `load_summaries` 的 `[(seq, content), …]` 拼成**一段**背景文本。

    按传入顺序(即 `seq` 序)拼接,段间是 `SUMMARY_SEPARATOR`(`"\\n\\n"`)。
    空列表 → **空串**(不是 `None`):`summary_text` 通道与 `build_context_messages`
    都按空串判「还没有梗概」。

    只收两元组 —— `journal._summary_rows` 那种三元组形态不在本函数的契约里。
    收错了要**当场炸**(值解包失败),而不是悄悄取到 `seq` 去当正文。
    """
    return SUMMARY_SEPARATOR.join(content for _, content in summaries)


async def summarize_range(
    *,
    model,
    session,
    conversation_id: str,
    turns: Sequence[Message],
    upto_msg_id: int,
) -> str | None:
    """把 `turns` 压成一段梗概,**成功才**落库并把边界推到 `upto_msg_id`。

    返回梗概文本;`None` 表示**什么都没做**(区间为空,或模型没吐出任何文本)。
    **异常原样抛出**,由调用方(T9 的后台任务)决定怎么记日志 —— 在这里吞掉的话,
    「上游 401」与「压完了」在调用方看来一模一样。

    **顺序即不变量**:`ainvoke` 在前,`append_summary_and_advance` 在后,而且
    只调一次。反过来(先推边界再落库,或者失败后仍落一次空梗概)会让这段历史
    **永久消失**:区间已经不在层 2 的读取范围里,而摘要表里没有替换物 ——
    没有报错、没有日志,只是「最开始那个订单后来怎么说」再也答不出来。
    失败时本函数不留任何痕迹,层 2 原封不动,下一次触发再压一遍即可(所以不重试)。

    `upto_msg_id` **由调用方给**,本函数不自己算:区间应当以重读到的锚点为
    (T9 的 `_reload_state`),这里再算一次就是同一条规则两处实现 ——
    漂移的表现是「压的区间与层 2 对不上」,一段历史既不在层 2、也不在梗概里。

    `session` 只做转发(落库要它),本函数自己不碰 DB。
    """
    if not turns:
        return None

    result = await model.ainvoke(build_summary_messages(turns=turns))
    text = result.text.strip()
    if not text:
        # 模型吐了空/空白:落一段空梗概再推边界,等于**静默删掉**那段历史
        # (库里留下的替换物是空文本,而验收 2 要的订单号就在被删的那段里)。
        # 当作「这次没压成」处理:不落、不推,下次再来。
        logger.warning(
            "摘要模型没吐任何文本,本次不落库也不推边界:conversation_id=%s", conversation_id
        )
        return None

    await append_summary_and_advance(
        session=session,
        conversation_id=conversation_id,
        upto_msg_id=upto_msg_id,
        content=text,
    )
    return text
