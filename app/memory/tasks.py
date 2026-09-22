"""ch07 T9:后台摘要执行体 —— 专用线程 + 线程内 `asyncio.run` + 自建 engine。

**为什么必须在另一个线程里**:摘要是本章唯一**不可逆**的动作(spec §3.3),而且
要一次模型往返。跑在请求路径上就是「用户这一轮的回复等它」—— 但它压的是**更早**
的一段历史,与这一轮的回复毫无关系。所以它可以晚、也可以失败。

**为什么自建 engine**:`app/db/base.py:get_engine()` 是 `lru_cache` 单例,绑在
**首次使用它的那个事件循环**上。后台线程里 `asyncio.run` 开的是**新循环**,复用
那个单例会拿到属于别的循环的异步连接 —— 这是 ch04 实测踩出来的
(`app/kb/orchestrate.py` 的模块 docstring 记着同一条)。本模块照抄它的三条:
**专用线程** / **线程内 `asyncio.run`** / **自建 engine 且任务结束 `dispose()`**
(最后一条在 `finally` —— 异常路径同样要释放,见下)。

**单测的两个接缝**:`_thread_target`(同步的线程体,`finally` 里摘在跑标记)与
`_run_body`(同步的干活入口,内部一个 `asyncio.run`)。有了它们,「标记有没有被
摘掉」「异常有没有被接住」都能**确定性**地验 —— 起真线程再等它跑完会变成时序
断言,而不稳定的断言最后会被人删掉,不是被修好。

**失败等于什么都没发生**:边界**只在成功后**推进,而推进它的唯一途径是 T8 的
`summarize_range`(里面「落库 + 推锚点」是一个原子动作,`app/services/history.py`
的 `append_summary_and_advance`)。所以本模块把异常**接住、记一行 `summary fail`、
返回** —— 不接住的话线程只会打一条 `Exception in thread` 噪音,而那是「没人处理」
的样子,`summary fail` 这条观测面也就永远不存在。

**依赖方向的一条反向边(与 T8 同款,一并记账)**:本模块 `from app.services.history
import load_history`,而本仓的方向是 `services → memory`。理由与 T8 的
`summarize.py → services.history` 一致(那段读历史的 I/O 只有一份实现,在这边
重写一遍 `MessageRecord` 的查询就是同一条规则两处实现)。**不构成环**:
`app/services/history.py` 只依赖 `app.db.models` 与 `app.schemas`,不回头引
`memory`。

**在跑标记是进程内的**(`_INFLIGHT`),多进程挡不住 —— 第二道防线是
`conversation_summaries` 的 `(conversation_id, seq)` 唯一键:真撞上时后者令本次
整体回滚 ⇒ 锚点不推进 ⇒ 下次重来(`ConversationSummary` 的 docstring 记着这条)。

**五个生命周期节点**(spec §7.6)在本模块说同一套话:`trigger` / `start` /
`done` / `skip` / `fail`,每行都带 `conversation_id`。其中 `trigger` 由**触发方**
打(见 `log_trigger`)—— 它是唯一知道层 2 用量与预算的地方,本模块不知道。
`skip` 的四种原因**必须分开**(见四个 `SKIP_*` 常量),它们的运维含义是相反的。
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.models import Conversation
from app.memory import layers
from app.memory.summarize import summarize_range
from app.sanitize import redact_api_key
from app.schemas import Message
from app.services.history import load_history

logger = logging.getLogger(__name__)

#: 进程内在跑的会话。**测试可以读它,生产不该读**(brief 的原话)。
#:
#: 「同一会话同一时刻只有一个摘要任务」是本模块存在的**首要理由**:两个任务并发
#: 会把同一段原文压两遍,产出两段**各自都完全正常**的梗概 —— 与「压对了」在数据上
#: 只差一段,没有任何东西会报错,而注入的梗概从此带着一段重复。
_INFLIGHT: set[str] = set()

#: 保护 `_INFLIGHT` 的 check-then-add。**不能省**:没有它,两个线程可以同时看到
#: 「不在跑」然后各自加进去 —— 恰好就是上面那条要防的形态。
_INFLIGHT_LOCK = threading.Lock()

#: `summary skip` 的四种原因 —— **必须是四个不同的值**。
#:
#: 「区间为空」是**正常**(这一轮确实没有可压的东西);「模型吐了空」是**故障**
#: (spec §12.2 的连带):`summarize_range` 返回 `None` 时这两种成因长得一模一样,
#: 而后者**没有退避** —— 下一轮触发会再打一次模型,一直打。混成一句 skip,
#: 运维看到的是「一直在跳过」,真实原因却是模型坏了。
SKIP_ALREADY_RUNNING = "already_running"
SKIP_EMPTY_RANGE = "empty_range"
SKIP_BLANK_MODEL_OUTPUT = "blank_model_output"
#: 线程起不来(资源耗尽那类)。另立一个值的理由与上一条同款:它**不是**
#: 「这次没东西可压」(那是正常),也不是「已有任务在跑」(那说明上一轮还在跑),
#: 而是**进程本身出问题了** —— 三者混成一个 skip,读日志的人分不出该不该动手。
SKIP_THREAD_NOT_STARTED = "thread_not_started"


@dataclass
class _RunState:
    """跨 `_run` / `_run_body` 的**事实**:这一段到底写进去了没有。

    **不能靠「异常抛在写之前」来推断。** `_run` 里在写**之后**才抛的异常同样会走到
    `summary fail` 那一行,而且有**两条**这样的路径:

    1. `async with factory() as session:` 的 **`__aexit__`**(关 session)——
       它跑在 `await summarize_range(...)` 返回**之后、with 体之外**;
    2. `finally` 里的 `engine.dispose()`(连接已断 / 网络抽风)。

    这两条路上**那一段已经提交、`summary_upto_msg_id` 已经推过去了**,若日志照旧
    声称「锚点未推进」,运维会以为这段历史还没被覆盖(去重压一遍,或者以为丢了)。

    这个字段存在的**全部意义**就是回答「这段历史被覆盖了没有」—— 答反了比不写更糟。
    所以它由**真的执行到哪一步**决定:置位点在 `_run` 的 **with 体内**、`await`
    返回的那一刻(`if written is not None:`)—— 放在 with 外面就会漏掉上面第 1 条。
    """

    anchors_advanced: bool = False


def _emit(event: str, *, conversation_id: str, level: int = logging.INFO, **fields) -> None:
    """一行生命周期日志:`summary <event> <json>`。

    JSON 而不是自由文本,理由与 `app/memory/journal.py` 相同:断言要断**契约**
    (哪个事件、哪些字段、什么值),而人读的是文案。`conversation_id` 恒在 ——
    spec §7.6 要求五个节点**都**带它,而「都带」只能靠一个统一出口保证。
    """
    payload = {"conversation_id": conversation_id, **fields}
    logger.log(level, "summary %s %s", event, json.dumps(payload, ensure_ascii=False))


def log_trigger(*, conversation_id: str, layer2_tokens: int, layer2_budget: int) -> None:
    """`summary trigger`:触发那把闸落下时由**触发方**打(T10 的端点调用它)。

    为什么不在本模块里打:触发判据是「层 2 的**截短后** token 数 > 预算」
    (T8 的 `should_summarize`),而这两个数**只有算分层的那一边有**。本模块
    拿不到它们 —— 硬凑一行只有会话 id 的「触发了」,就是本仓最忌讳的那种
    「看起来正常、其实什么也没说」的观测面(验收 2 的触发断言正是靠这两个数)。
    """
    _emit(
        "trigger",
        conversation_id=conversation_id,
        layer2_tokens=layer2_tokens,
        layer2_budget=layer2_budget,
    )


def run_summary_in_background(*, conversation_id: str, settings: Settings, model_factory) -> bool:
    """起一个后台摘要任务,**返回是否真的起了**。

    `True` = 起了;`False` = **没起**(并留一行 `summary skip`,原因在 `reason` 里)。
    返回值是 T10 与验收 4 的观测面 —— **不是**便利:调用方(请求路径)靠它知道
    「这一轮的压缩已经开始」,而两种取值对应两种完全不同的后续(等它 / 下一轮再说)。

    没起的两种成因(brief 里只写了第一种,第二种是本任务自审时补的):
    `SKIP_ALREADY_RUNNING`(这个会话已有一个在跑)与 `SKIP_THREAD_NOT_STARTED`
    (线程压根起不来)。**后者必须也把登记撤掉** —— 否则那个会话会被**永久**当成
    「已有任务在跑」,而那是一条没有任何痕迹的锁死(与 `_thread_target` 的
    `finally` 要防的是同一件事,只是发生在更早一步)。

    **本函数只做两件事**:在锁里查+登记+起线程,**不碰 DB、不建模型** ——
    那两样都要在新线程的新事件循环里做(见模块 docstring)。因此它是同步且
    **立即返回**的,请求路径上不会有任何等待。

    `settings` 与 `model_factory` 由调用方注入(与 `services/` 里那几个函数的
    既有约定一致:不在模块层建全局单例)。`model_factory(settings)` 只在
    **真的有东西可压**时才被调用。
    """
    with _INFLIGHT_LOCK:
        if conversation_id in _INFLIGHT:
            _emit("skip", conversation_id=conversation_id, reason=SKIP_ALREADY_RUNNING)
            return False
        # 登记必须在**起线程之前、且在锁里**:反过来的话,两个几乎同时到达的调用
        # 都能通过上面那个检查,于是两个线程压同一段原文。
        _INFLIGHT.add(conversation_id)
        try:
            threading.Thread(
                target=_thread_target,
                name=f"summary-{conversation_id}",
                # daemon:摘要是**可以丢的**活。进程退出时不需要等它,而它自己失败
                # 也无妨(边界不动,下一轮再来)—— 与 ch04 的后台任务同款。
                daemon=True,
                kwargs={
                    "conversation_id": conversation_id,
                    "settings": settings,
                    "model_factory": model_factory,
                },
            ).start()
        except Exception:
            # 起不来 ⇒ 撤销登记(见 docstring),并且**不往上抛**:给一个后台任务
            # 起的线程失败,不该让用户这一轮的回复变成 500 —— 「摘要失败不影响
            # 回复」是本模块从头到尾的那条线。
            #
            # 只接 `Exception` 不接 `BaseException`:`_INFLIGHT` 是**进程内**的
            # 内存状态,KeyboardInterrupt 那条路上进程本来就要走,登记跟着一起
            # 消失 —— 而把 Ctrl-C 吞掉才是真的错(ch04 的 `_spawn` 同款)。
            _INFLIGHT.discard(conversation_id)
            _emit("skip", conversation_id=conversation_id, reason=SKIP_THREAD_NOT_STARTED)
            return False
    return True


def _thread_target(*, conversation_id: str, settings: Settings, model_factory) -> None:
    """线程体。**`finally` 里摘在跑标记 —— 失败路径也必须摘。**

    漏掉 `finally`(比如只在成功分支里摘)的后果不是「多跑一次」:那个会话
    **再也压不了** —— 之后每一次 `run_summary_in_background` 都被当成「已有任务
    在跑」拒掉。用户侧每一轮看起来都完全正常,只是梗概永远不再更新、层 2 无限涨。
    这是个**静默**的永久锁死,所以标记的摘除写在这里,而不是写在 `_run_body` 里
    (那里会漏掉异常路径)。
    """
    try:
        _run_body(
            conversation_id=conversation_id, settings=settings, model_factory=model_factory
        )
    finally:
        with _INFLIGHT_LOCK:
            _INFLIGHT.discard(conversation_id)


def _run_body(*, conversation_id: str, settings: Settings, model_factory) -> None:
    """真正干活的部分:**同步入口,内部一个 `asyncio.run`**。**绝不抛。**

    为什么是同步的:它是线程体(`_thread_target`)调用的那个,而线程里没有事件
    循环 —— `asyncio.run` 在这里开一个属于**本线程**的新循环,engine 也在里面
    建(见模块 docstring)。

    为什么把异常在这里接住:`_thread_target` 里没有第二道网,抛出去只会变成
    「线程打了条噪音然后死掉」—— 边界固然没动(它只在成功后动),但
    `summary fail` 那条观测面就没了,而它是排查“为什么梗概一直不更新”的唯一入口。
    接住之后**什么都不做**:不重试(spec §8「留日志,不重试,边界不动」)。
    """
    state = _RunState()
    try:
        asyncio.run(
            _run(
                conversation_id=conversation_id,
                settings=settings,
                model_factory=model_factory,
                state=state,
            )
        )
    except Exception as exc:                      # noqa: BLE001 —— 见 docstring
        _emit(
            "fail",
            conversation_id=conversation_id,
            level=logging.WARNING,
            # 异常文本常常就是**上游响应体原文**(openai 的 401 把 key 原样写在
            # 里面),所以出站前一律过 redact_api_key(本仓硬规矩)。
            error=redact_api_key(str(exc), settings.openai_api_key),
            # 走到这一步时边界动了没有 —— **问 `_RunState`,不问代码位置**。
            # (落库成功过 ⇒ 那一段已经永久取代了原文,即使失败发生在它之后。)
            anchors_advanced=state.anchors_advanced,
        )


async def _run(
    *, conversation_id: str, settings: Settings, model_factory, state: _RunState
) -> None:
    """一次摘要任务的完整流程。engine 在这里建、在 `finally` 里释放。

    `state` 是**回填事实**用的(见 `_RunState`):走到哪一步了,不是靠位置猜。
    """
    started = time.monotonic()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    try:
        # ---- 第一步永远是**重读**,不是「信触发时的快照」----
        # 触发方那一侧的快照发出任务时就过时了:边界可能已经被降级挪过。
        # 用旧的 (a, b) 去压,压出来的区间与当前层 2 对不上 —— 结果是**一段历史
        # 既不在层 2、也不在梗概里**(静默跳过)。
        summary_upto, layer1_from, history = await _reload_state(engine, conversation_id)

        # 区间就是**层 2**:`(summary_upto, layer1_from)` 两端都不含。
        # 这里直接借用 `layers._middle` 而不是自己写一遍 —— 那是**同一条区间规则**
        # 的第二处实现,而它的漂移形态(ch07 spec §12.1 ①)正是「同一条消息同时
        # 落在两层里 ⇒ 上下文凭空翻倍」,两边都不报错。`layer1_from == 0`
        # (层 1 起于最早 ⇒ 层 2 为空)的语义也一并由它保证。
        turns = layers._middle(history, after_id=summary_upto, before_id=layer1_from)

        if not turns:
            _emit(
                "skip",
                conversation_id=conversation_id,
                reason=SKIP_EMPTY_RANGE,
                summary_upto=summary_upto,
                layer1_from=layer1_from,
            )
            return

        _emit(
            "start",
            conversation_id=conversation_id,
            summary_upto=summary_upto,
            layer1_from=layer1_from,
            turns=len(turns),
        )

        # 模型**迟到这里才建**:区间为空时不碰 `model_factory`(那一步在真机上是
        # 建一个指向上游的客户端 —— 没有活干就不该有它)。
        model = model_factory(settings)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            written = await summarize_range(
                model=model,
                session=session,
                conversation_id=conversation_id,
                turns=turns,
                # 上界取**重读到的** `layer1_from`:新层 2 就是 `(layer1_from, …)`,
                # 于是「已覆盖到哪条」与「层 2 从哪条起」严丝合缝。这个值**不能**
                # 取区间内最后一条的 id —— 那条规则与分层那条是两处实现,而
                # off-by-one 在这里的表现是「一条消息既没进梗概、又留在层 2」。
                upto_msg_id=layer1_from,
            )
            # ⚠️ **置位必须在 with 体内、`await` 返回的那一刻**:`__aexit__`(关
            # session)跑在这行的**后面**,它抛同样会走到 `summary fail` —— 而那时
            # 这次原子落库**已经提交**。挪到 with 外面(曾经的写法)就漏掉那条路径,
            # 报出一个说反了的 `anchors_advanced: false`。
            if written is not None:
                state.anchors_advanced = True

        if written is None:
            # 区间**非空**却什么都没压出来 ⇒ `summarize_range` 的另一半含义:
            # **模型吐了空**(空区间在上面就 return 了)。这与「区间为空」分开记。
            #
            # 注意 `no_backoff=True` 是**如实的**:这里没有任何退避机制,下一轮
            # 触发会立刻再打一次模型 —— 模型坏掉时的症状是「每个轮次一次无效
            # 调用」,而日志里那一串 skip 看着像「一直没东西可压」。
            _emit(
                "skip",
                conversation_id=conversation_id,
                reason=SKIP_BLANK_MODEL_OUTPUT,
                summary_upto=summary_upto,
                layer1_from=layer1_from,
                turns=len(turns),
                no_backoff=True,
            )
            return

        # 段号是**落库那一步自己算出来的**,一路原样带到这里(spec §7.6 的
        # 「第 N 段」)—— 在这里重算或写死都不是同一个事实。
        seq, _text = written
        _emit(
            "done",
            conversation_id=conversation_id,
            seq=seq,
            upto_msg_id=layer1_from,
            covered_from=summary_upto,
            turns=len(turns),
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        # **不记梗概正文、不记 prompt**:日志既是密钥泄漏面也是日志膨胀源
        # (spec §8)。要正文的地方是 `model_ctx`(它记的是组装后的上下文窗口)。
    finally:
        # **每条退出路径都要释放**(含异常与提前 return):漏掉的话每压一次
        # 泄漏一个连接池,而摘要任务每轮都可能起 —— 症状是 MySQL 侧连接数缓慢
        # 爬到上限,很久以后才表现为「随机连不上」。
        await engine.dispose()


async def _reload_state(engine, conversation_id: str) -> tuple[int, int, list[Message]]:
    """重读 `(summary_upto, layer1_from, 该会话的全部历史)`。**单测的接缝。**

    返回的是**整段历史**而不是切好的区间:区间由 `_run` 用 `layers._middle` 从
    **重读到的**两个锚点现切。这样「重读」与「切区间」用的是同一份锚点,不可能
    出现「用新锚点判定、用旧锚点切」那种两处答案。

    ⚠️ 与 brief 的 Interfaces 那行有一处**故意的偏离**:那里写的是「重读
    `(summary_upto, layer1_from, 区间内的原文)`」,而这里第三个元素给的是全部
    历史。原因是 brief 自己的测试:它把本函数替换成一个返回 `_HISTORY()`(id 1..9)
    的替身,却断言 `seen["turns"] == [4, 5, 6]` —— 也就是**切区间这一步发生在
    `_run` 里**。按 brief 的字面去切会让那条断言恒真(替身给什么就传什么),
    而它本来要钉的正是「只压区间内那几条」。见 T9 报告。

    `engine` 是本任务自建的那个(不是 `get_engine()` 的单例)—— 见模块 docstring。
    `conversation` 为 `None` 时**抛**:调用方(端点)在起任务前已经
    `ensure_conversation` 过,读不到就是真故障,吞掉只会变成一条没有锚点的
    「跳过」日志。
    """
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        conversation = (
            await session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
        ).scalars().one_or_none()
        if conversation is None:
            raise LookupError(f"会话不存在,无法重读锚点:{conversation_id}")
        return (
            conversation.summary_upto_msg_id,
            conversation.layer1_from_msg_id,
            await load_history(session=session, conversation_id=conversation_id),
        )
