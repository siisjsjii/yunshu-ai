"""ch07 三层切分与层 2 截短。**纯函数,不依赖 LangChain。**

三层与两个锚点见 spec §3.1:锚点是 `messages.id`,不变量是
`0 <= summary_upto_msg_id <= layer1_from_msg_id`,两个动作都**只挪 id、不搬数据**。

两个锚点的语义(spec §3.1 的表,逐字):
`summary_upto_msg_id` = 梗概已覆盖到哪条(**含**),`0` = 尚无任何梗概;
`layer1_from_msg_id` = 层 1 从哪条(**含**)起,`0` = 层 1 起于最早、**层 2 为空**。
三段因此是 `[.., summary_upto]` / `(summary_upto, layer1_from)` / `[layer1_from, ..]`,
**不重不漏** —— off-by-one 会静默丢消息或重复注入。
"""

from dataclasses import dataclass, field

from app.config import Settings
from app.memory import trim
from app.schemas import Message

#: 截短后追加的省略号。用 `…` 而不是 `...`:它是验收 4b 做**形态匹配**的锚,
#: 而三个点会与正文里本来就有的句号混淆。
ELLIPSIS = "…"


@dataclass(frozen=True)
class Layers:
    layer2: list[Message] = field(default_factory=list)
    layer1: list[Message] = field(default_factory=list)
    #: **截短后**的 token 数 —— 触发摘要看的就是它(spec §3.2)。
    layer2_tokens: int = 0
    #: 原文 token 数 —— 触发降级看的是它。
    layer1_tokens: int = 0
    #: 切分时用的两个锚点。**跟着 `Layers` 一起走**,而不是让每个调用方各记一份:
    #: 调用方各记一份的后果是「日志里的 bounds 与实际切分用的锚点可以不一致,
    #: 而两边都不报错」。`0` 是**有含义的值**(尚无梗概 / 层 1 起于最早,
    #: 见模块 docstring),所以它不能被当成「未知」的哨兵。
    #:
    #: ch07 T7 补:观测量(`journal.model_ctx` 的 `bounds`)要能说出「被截的是
    #: **哪一段**」。锚点若由调用方另行传入、且带一个 `0` 的默认值,那行日志在
    #: 拿不到锚点时会**报出 `{0, 0}` 而切分用的是真锚点** —— 一个说不出任何东西
    #: 却又长得像真值的观测面(本仓「期望值等于默认值」那类假绿的生产版)。
    summary_upto_msg_id: int = 0
    layer1_from_msg_id: int = 0


def _tail(history, *, from_id: int) -> list[Message]:
    """层 1:`id >= from_id`(**含**),按原序。`from_id == 0` 表示起于最早。

    `id is None` 的消息(手工构造、尚未落库的)**一律算层 1**:它们是最新的,
    而层 1 正是"最近原文"那一层。`_middle` 的 docstring 说明为什么它们**只能**
    在这一层。
    """
    return [m for m in history if m.id is None or m.id >= from_id]


def _middle(history, *, after_id: int, before_id: int) -> list[Message]:
    """层 2:`after_id < id < before_id`,两端都**不含**,按原序。

    `before_id == 0` ⇒ **层 2 为空**(spec §3.1 表里 `layer1_from_msg_id` 的 `0` 语义,
    也是新会话的形态:两个锚点都是 0,全量历史都在层 1)。
    这里**不能**把 `0` 当成"到末尾":那样新会话的每一条消息会同时出现在层 1 与层 2,
    而调用方拼上下文用的正是 `layer2 + layer1`(spec §7)—— 等于每条注入两遍,
    **不报错、不丢消息,只是上下文凭空翻倍**。

    `id is None` 的消息**进不了层 2**(它们在 `_tail` 里),否则同一条会被注入两遍。
    """
    if before_id == 0:
        return []
    return [m for m in history if m.id is not None and after_id < m.id < before_id]


def _first_id(round_: list[Message]) -> int | None:
    """一轮里第一条**有 id** 的消息的 id(轮的起点,正常就是那条 user)。"""
    return next((m.id for m in round_ if m.id is not None), None)


def truncate(message: Message, *, settings: Settings) -> Message:
    """按角色截短一条消息。

    **只截 `content`,不动结构字段**(`tool_calls` / `tool_call_id`)——
    截断 `tool_calls` 就是把 assistant 与它的 tool 消息拆开 ⇒ 上游 400,
    且只在历史长到触发分层时才复现。
    """
    if message.role == "user":
        return message                      # 原话一个字不动
    if message.role == "tool":
        limit = settings.layer2_tool_chars
        prefix = "[工具结果] "
    else:
        limit = settings.layer2_assistant_chars
        prefix = ""
    if len(message.content) <= limit:
        return message
    return message.model_copy(
        update={"content": f"{prefix}{message.content[:limit]}{ELLIPSIS}"}
    )


def split(
    history, *, summary_upto_msg_id: int, layer1_from_msg_id: int, settings: Settings
) -> Layers:
    """切三层。**层 2 的 token 按截短后的版本数** —— 见模块 docstring 与 spec §3.2。

    「按截短后数」是层 2 存在的全部意义:按原文数的话 §3.4 的截短就退化成
    纯渲染装饰,级联(何时摘要)与截短有没有做**完全无关**。
    """
    layer2_raw = _middle(
        history, after_id=summary_upto_msg_id, before_id=layer1_from_msg_id
    )
    layer1 = _tail(history, from_id=layer1_from_msg_id)
    layer2 = [truncate(m, settings=settings) for m in layer2_raw]
    return Layers(
        layer2=layer2,
        layer1=layer1,
        layer2_tokens=sum(trim.count_tokens(m.content) for m in layer2),
        layer1_tokens=sum(trim.count_tokens(m.content) for m in layer1),
        # 锚点原样带上:它们是这次切分的**输入**,`Layers` 与它的观测面
        # (`journal.model_ctx` 的 `bounds`)必须是同一份,否则日志说的与
        # 实际切的那一刀可以不一致,而两边都不报错。
        summary_upto_msg_id=summary_upto_msg_id,
        layer1_from_msg_id=layer1_from_msg_id,
    )


def degrade(
    history, *, summary_upto_msg_id: int, layer1_from_msg_id: int,
    layer1_budget: int, settings: Settings,
) -> int:
    """层 1 超预算就把边界往后挪,**循环到收敛**,返回新的 `layer1_from_msg_id`。

    为什么是循环不是一次判断:挪一次会同时改变两层的大小(层 1 变小、层 2 变大),
    而层 2 变大**不会**反过来影响层 1 —— 但它决定了摘要的触发,所以收敛后才算数。

    边界**只落在轮的起点**(user 消息)上:否则会把一轮切开,连带把 tool 与它的
    assistant 拆到两层里(`trim.to_rounds` 的既有理由,ch01 的 400 就是它)。

    只剩一轮还超预算时**停在原地**(层 1 仍超预算,由调用方决定怎么办)。
    再往后挪一次不会有任何好处:那一轮已经是层 1 的全部,把它也让出去只会让
    层 1 更空,而"装得下"这个信号会与事实相反。

    **两处保证 `cur` 严格增大**(挪到下一轮的起点;挪不动就返回),
    循环因此一定终止 —— 这条必须由代码保证:边界挪一次若没有前进,
    `while True` 会**原地死循环**,而它跑在请求路径上。
    """
    cur = layer1_from_msg_id
    while True:
        got = split(
            history, summary_upto_msg_id=summary_upto_msg_id,
            layer1_from_msg_id=cur, settings=settings,
        )
        if got.layer1_tokens <= layer1_budget:
            return cur
        rounds = trim.to_rounds(got.layer1)
        if len(rounds) <= 1:
            return cur      # 只剩一轮还超:再挪就空了,停在原地由调用方决定
        # 边界挪到**下一轮的起点** —— 这才叫"只落在轮的起点"。
        # (不能挪到"被让掉那一轮的最后一条":层 1 从一条 assistant 起头时,
        #  `to_rounds` 把这条 assistant 单独算一轮,而那个位置正好等于当前边界
        #  —— `cur` 不前进。**实测**:那种写法(且没有下面这条前进性检查)
        #  跑 `pytest tests/test_memory_layers.py` **挂死不返回**(30s 超时被 kill);
        #  有检查时它退化成"返回一个落在轮中间的 id",`test_degrade_moves_...` 判红。)
        nxt = _first_id(rounds[1])
        if nxt is None or nxt <= cur:
            return cur
        if nxt <= summary_upto_msg_id:
            return summary_upto_msg_id      # 守住不变量
        cur = nxt
