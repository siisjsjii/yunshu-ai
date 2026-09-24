"""飞轮后台任务(ch09 §8.3):把一批「低置信度问题 → 待审队列」跑在专用线程上。

形状**逐条对齐 ch04 的 `app/kb/orchestrate.py`**(那是验过的模板),三处一个字都不能省:

1. **自建 engine**(`create_async_engine` + 任务结束 `dispose()`)—— `get_engine()`
   的 lru_cache 单例绑在**首次使用它的事件循环**上,后台线程里 `asyncio.run`
   复用会出跨循环的异步连接问题。
2. **专用线程 + 线程内 `asyncio.run`** —— 一批飞轮是若干次模型往返(秒级到分钟级),
   跑在事件循环里会把聊天接口冻住。
3. **终态与 dispose 都在 `finally` 里** —— 运行槽**唯一的释放口**是
   `JobStore.update(status="done"|"failed")`,漏掉不是「多跑一次」,是那个会话
   **再也跑不了**(之后每一次都被当成「已有任务在跑」),而用户侧一切正常
   (ch07 记过同款)。这里还多一层:`except BaseException`(见 `_spawn`)——
   `CancelledError` 是 `BaseException`,只抓 `Exception` 会让它漏过去。

## 「终态」有**三条**出口,不是一个 finally

`job_store.start()` 一返回就**已经占住了槽**,而后面还有两步可能失败,每一步
都得自己把终态写下去,否则槽**永久**占着(端点此后永远 409、三处钩子静默拿到
`None`,唯一的痕迹是 `GET /api/kb/jobs` 里一条卡在 running 的 job):

  1. 协程内部抛 ⇒ `_run_flywheel` 的 `finally`(它自己也可能只是被 `_spawn` 兜住);
  2. **线程起不来**(`Thread.start()` 抛)⇒ **没有任何协程会跑**,只能由 `_spawn`
     的 `except` 兜 —— 这一条是 T14 复审逮到的,它的前提「终态由协程的 finally 兜」
     **根本不成立**;
  3. 兜底那句写终态的语句自己炸(脱敏失败等)⇒ `_write_failed` **保证不抛**。

## 脱敏用的是**调用方**那份 settings

`_write_failed` 脱的是调用方传进来的 `settings` 的 key,**不是** `get_settings()`
那份进程全局配置 —— 后者在参数配错时是**静默失效**:`str(exc)` 里带的若是调用方
那份 key,拿全局那份去 `replace` 等于什么都没抹,而 `JobStore.message` 是
**出站**文本(`GET /api/kb/jobs` 原样回给前端)。本模块因此**不 import
`get_settings`**(T14 复审 F1;删掉 import 顺带让那行代码没法被悄悄加回来)。

## 两种入口

- `start_flywheel_job`:**手动**触发(端点 / 脚本),忙时返回 `None`,由调用方决定
  要不要 409 —— 与 `orchestrate.start_job` 同一套约定。
- `start_flywheel_job_safely`:**落池之后**那三处 fire-and-forget 用的形状
  (`置信度闸` / `商品咨询`自评轮 / `POST /api/feedback`)。区别只有一条:
  它**绝不抛**。

## 「挂起不是异常」⇒ 终态还有**第四条**出口:寿命上界

上面那三条出口**全部**以「有异常」为前提(协程抛 / 线程起不来 / 兜底自己炸)。
2026-09-24 的 T16 走查实测到第四条:**任务只是卡住**,谁都没抛 ——
三条任务停在 `running` 666s / 245s / 382s,**零日志**、`message` 空串、库里一条
开着却空转的事务,而槽是**单槽**的 ⇒ 端点此后永远 409、**连手动那个
「跑一轮飞轮」也拿不到槽**,唯一的出路是**重启客服服务**。
⇒ `finally` 根本没被执行,而「每条出口都进终态」那句保证在有界等待之外是假的。

**两半各治一半,缺一不可:**

1. **让等待有界**(`app/llm.py:_build` 的 `timeout=`)—— 治因:对端静默时
   每次往返在 `settings.llm_timeout_seconds` 内抛 `APITimeoutError`,
   于是「卡住」重新变回一种**异常**,上面那三条出口立刻全部生效
   (它同时救了对话/抽取/摘要 —— 那几个入口此前同样会被一次静默拖死)。
2. **让任务有寿命上界**(下面 `_run_flywheel` 里的 `asyncio.timeout`)—— 兜住
   **其它**任何一种无界等待(将来某个新依赖、某个锁、某个没配 timeout 的客户端)。
   它不替代第 1 条:没有第 1 条时它每次都要等到 300s 才收场,而且那 300s 里
   用户看到的是「任务在跑」。

**为什么不用「JobStore 里的存活检查 / 强制清槽」**:JobStore **停不下**那条线程
—— 把槽提前放掉只等于允许**第二批**与那个僵尸并发跑,而单槽的全部意义正是
「同一张池子两条流水线并发只会互相抢行」(`occurrences` 会跟着涨错);
僵尸稍后写自己的终态时状态也会打架。上界必须长在**干活的那一侧**。

**被上界杀掉的那一批一行都不会落地**(`run_flywheel` 整批只提交一次)。
这是**刻意**的:池子行靠 `matched_review_id IS NULL` 幂等,下一轮会重新吃到它,
而半批落地的中间态会让「哪些算处理过」变得没有判据。所以那条 message 里
必须把这件事写出来 —— 「消失了」与「被上界杀掉的」在 `GET /api/kb/jobs` 上
要分得开。

### 死线那句话必须活到最后一手(T16b 复审 F1)

`JobStore.message` 是操作员**唯一**的信号(卡住时它只有一个空串),而
`async with factory() as session` **退出时自己也会抛** —— 失步连接上的 rollback /
连接归还会抛 `PendingRollbackError` / `InterfaceError`,它的 `str()` 正是本仓
ch08 记过的那段**裸 SQLAlchemy 内部文本**。那时到达 `_spawn` 兜底的就不是
`FlywheelDeadlineExceeded` 了,**出站文案跟着换成那段内部文本,而死线这句话没了**。
⇒ `_run_flywheel` 里那层 `except BaseException` 会在死线确实放过枪时**把原因抢回来**
(真正的收尾故障进 `__cause__`,服务日志里照样看得到)。

⚠️ **实测边界(别把这条读成「确认会在真库上发生」)**:真 MySQL 上「死线取消一个慢查询」
那条路**没有**抛(`SELECT SLEEP(8)` + 0.4s 死线);这条守卫挡的是**收尾抛**那一类。
确定性复现是「`__aexit__` 会抛的 session」,见
`.superpowers/probe_t16b_f1_clobber.py`(两种形态都在那份输出里)。
"""

import asyncio
import logging
import threading

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.flywheel.pipeline import run_flywheel
from app.kb.jobs import Job, JobStore, get_job_store
from app.llm import create_extract_model
from app.sanitize import redact_api_key

logger = logging.getLogger(__name__)

#: 进 `JobStore` 的任务类型。`GET /api/kb/jobs` 把它**原样**回给前端
#: (`_job_dict` 读的是 `Job.type`)—— 所以新类型不需要改那两个只读端点。
JOB_KIND = "flywheel"


#: 脱敏自己失败时用的固定文案。**绝不回显未脱敏原文** —— 宁可丢原因,
#: 也不许把可能带密钥的文本写进 `JobStore.message`(它经 `GET /api/kb/jobs` 出站)。
UNREDACTED_REASON_MESSAGE = "任务异常结束(原因未能脱敏,详见服务日志)"


class FlywheelDeadlineExceeded(RuntimeError):
    """寿命上界到了(见模块 docstring「挂起不是异常」那一节)。

    它是**普通异常**是刻意的:终态的写入走的是那三条**已经验过**的出口
    (协程的 `finally` → `_spawn` 的兜底,后者把原因写进 `JobStore.message`),
    不新增第四个写口。⇒ `str(它)` 就是那条**出站**文案,所以里面
    **不许出现任何可能带密钥的东西**(它要过 `redact_api_key`,那只是第二道)。
    """

    def __init__(self, *, limit: float) -> None:
        # `:g` 而不是 `:.0f` —— 亚秒配置(测试里就是 0.3/0.5)在 `:.0f` 下会渲染成
        # 「上限 0s」,一条**看起来像配置坏了**的文案(实测见过)。
        super().__init__(
            f"任务超时(上限 {limit:g}s):本轮已放弃。整批一次提交 ⇒ "
            f"这一批一行都没落地,池子里的行仍在(下一轮会重新吃到)"
        )


def _write_failed(job_store: JobStore, job: Job, settings, exc: BaseException) -> None:
    """把「线程级意外」写成终态。**本函数保证不抛**(它已经是最后一道防线)。

    两半各自可能失败,所以各自兜住:

    ① **脱敏用调用方那份 settings 的 key**,不是进程全局那份。`_spawn` 兜的是
       **调用方传进来的** settings(端点那条路来自 `Depends(get_settings)`,三处钩子
       来自节点手上的那份),而 `str(exc)` 里带的正是**那一份** key —— 拿全局那份去
       `replace` 等于什么都没抹。`JobStore.message` 是**出站**文本
       (`GET /api/kb/jobs` 原样回给前端),所以那等于把密钥送出去。
       脱敏自己炸了(设置对象缺字段等)就退成**固定文案**:宁可丢原因,
       既不丢终态、也绝不回显原文。

    ② **写终态**:先看它是不是已经 `done`(跑成了、只有 `dispose()` 炸了 ——
       不许改口),再写 `failed`。`JobStore.update` 不是「对终态 job 只覆盖 message」,
       它连 `status` 一起改。
    """
    try:
        message = redact_api_key(str(exc), settings.openai_api_key)
    except BaseException:  # noqa: BLE001
        logger.error("飞轮失败原因脱敏失败 job=%s", job.id, exc_info=True)
        message = UNREDACTED_REASON_MESSAGE
    try:
        # ⚠️ 变量名**不能叫 `job`** —— 那是形参,在这里再绑一次就把它变成局部名,
        # 于是**上面那行 `logger.error(..., job.id)` 会 `UnboundLocalError`**
        # (赋值在后面),而它看起来像「日志模块坏了」。踩过一次(见 T14 报告)。
        latest = job_store.get(job.id)
        if latest is not None and latest.status == "done":
            return
        job_store.update(job.id, status="failed", message=message)
    except BaseException:  # noqa: BLE001
        logger.error("飞轮写终态失败 job=%s(运行槽可能被占住)", job.id, exc_info=True)


def _spawn(job_store: JobStore, job: Job, settings, coro_fn) -> None:
    """专用线程 + 线程内 `asyncio.run`(ch04 模板)。**两条出口都要落终态。**

    `settings` 是**调用方那一份**(不是 `get_settings()`):兜底要用**它的** key
    脱敏(见 `_write_failed`),拿进程全局那份去脱敏在参数配错时是静默失效的。

    ⚠️ **`except BaseException`,不是 `except Exception`** —— 与 ch04 刻意不同。
    `CancelledError` 与 `KeyboardInterrupt` 都是 `BaseException`:只抓 `Exception`
    时它们会穿过去,`target` 静默结束,而 job 可能**还停在 running**
    (协程自己那条 `finally` 一般能兜住,但那一条正是本文件要守的性质,
    不该由它单独承担最后一层)。本仓的房规就是 `except BaseException`
    (ch03 的取消路径同款)。
    """
    def target() -> None:
        try:
            asyncio.run(coro_fn())
        except BaseException as exc:  # noqa: BLE001
            logger.error("飞轮后台任务异常结束 job=%s", job.id, exc_info=True)
            # 终态在这里再写一次**是刻意的**:它把 `_run_flywheel` 那条通用文案
            # (「任务异常结束(未进终态)」)换成**真正的原因** —— 池子里那批行
            # 只会「一直不消失」,没有原因的 failed 等于没写。
            _write_failed(job_store, job, settings, exc)

    try:
        threading.Thread(
            target=target, name=f"flywheel-job-{job.id}", daemon=True).start()
    except BaseException as exc:  # noqa: BLE001
        # ⚠️ **这一层不能省**:`job_store.start()` 已经把 job 置成 running(**槽已经
        # 占上了**),而线程起不来时**没有任何协程会跑** —— 「终态由协程的 finally 兜」
        # 这个前提在此**根本不成立**。漏掉的代价不是「少跑一轮」,是**单槽被永久占死**:
        # 端点此后永远 409、三处钩子静默拿到 `None`,唯一的痕迹是
        # `GET /api/kb/jobs` 里一条卡在 running 的 job。
        logger.error("飞轮后台线程起不来 job=%s", job.id, exc_info=True)
        _write_failed(job_store, job, settings, exc)


def _fresh_factory(settings):
    """任务专属 engine + sessionmaker(独立于主循环的 lru_cache 单例)。

    ⚠️ 这一行也是**测试的注入接缝** —— 打桩它就不再有「自建 engine」这条性质,
    所以另有一条 `@pytest.mark.db` 的用例**不打桩**,在真 engine 上验
    「照 settings 建 + 真的被 dispose 了」。
    """
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _run_flywheel(job_store: JobStore, job_id: str, settings, model_factory) -> None:
    engine, factory = _fresh_factory(settings)
    #: 死线是不是**本函数自己**放的那一枪。见下面那层 `except BaseException`。
    deadline_hit = False
    try:
        # 模型在**后台线程里**才建(`start_flywheel_job` 那三处是 fire-and-forget
        # 调它,调用方是用户请求的那条线程 —— 那边一毫秒都不该花在建模型上)。
        model = model_factory()
        try:
            async with factory() as session:
                # 寿命上界(模块 docstring 第四条出口)。**只包住流水线**:
                # 终态的写入与 `engine.dispose()` 必须在它**外面** —— 被上界杀掉时
                # 那两步更要做完(否则槽照样占死,那就等于没修)。
                #
                # ⚠️ `except` 写在 `async with timeout_cm` 的**外面**(用 try 包住
                # 整个 with 块):`TimeoutError` 是那个 CM 在 `__aexit__` 里抛的,
                # 写在里面会连它自己的取消转换一起吞掉(`uncancel` 也对不上)。
                timeout_cm = asyncio.timeout(settings.flywheel_job_timeout_seconds)
                try:
                    async with timeout_cm:
                        result = await run_flywheel(
                            session=session, model=model,
                            batch_size=settings.flywheel_batch_size)
                except TimeoutError as exc:
                    # ⚠️ **必须用 `expired()` 收窄**:`except TimeoutError` 比这个
                    # 取消作用域**本身**宽 —— 流水线内部任何一个 `TimeoutError`
                    # (别人的 `wait_for`、某个库自己的超时)原本会被贴成
                    # 「任务超时(上限 300s)」,归因按构造就是错的。今天它潜伏
                    # (openai 会把读超时包成 `APITimeoutError`,不是内置 `TimeoutError`),
                    # 但错的那一天没人看得出来。
                    if not timeout_cm.expired():
                        raise
                    deadline_hit = True
                    raise FlywheelDeadlineExceeded(
                        limit=settings.flywheel_job_timeout_seconds) from exc
            job_store.update(
                job_id, status="done",
                message=(f"处理 {result['processed']} 行:"
                         f"归并 {result['merged']}、新建 {result['created']}、"
                         f"失败 {result['failed']}"),
                result=result)
        except BaseException as exc:  # noqa: BLE001 —— CancelledError 是 BaseException
            # ⚠️ **死线那句话不许被冲掉**(T16b 复审 F1)。`JobStore.message` 是
            # 操作员唯一的信号(§T16 报告:卡住时它只有一个空串),而
            # `async with factory() as session` **退出时自己也会抛** ——
            # rollback / 连接归还在失步的连接上会抛 `PendingRollbackError` /
            # `InterfaceError`,它的 `str()` 是本仓 ch08 记过的那段**裸 SQLAlchemy
            # 内部文本**。那时到达 `_spawn` 兜底的就**不是**死线异常,出站文案跟着
            # 变成那段内部文本,而死线这句话**没了**。
            # ⇒ 只要死线确实放过枪,就以它为准重新抛(原文进 `__cause__`,
            # 服务日志里照样看得到真正的收尾故障)。
            #
            # **两处构造同一个异常是刻意的**:这里这一处是「收尾把原因换掉」的
            # 补救口,上面那一处是「正常路径」的出口 —— 文案本身在异常类里,
            # 只有一份。
            if deadline_hit and not isinstance(exc, FlywheelDeadlineExceeded):
                raise FlywheelDeadlineExceeded(
                    limit=settings.flywheel_job_timeout_seconds) from exc
            raise
    finally:
        # 终态与 dispose **同一个 finally**,理由见模块 docstring 第 3 条。
        #
        # `status == "running"` 的守卫:正常跑完时**不许**把 done 覆写成 failed
        # (那会让「成功」丢掉,而 result 还在 —— 前端读到的是一对自相矛盾的值)。
        # 这一层只保「协程**没抛异常**却没进终态」那条路;协程抛出去的那条路由
        # `_spawn` 的兜底接住,并且**它也有同一个方向的守卫**(见上面 `latest`
        # 那几行)—— 两层守卫方向一致,合起来才是 R6 要的「每条出口都进终态、
        # 但已成的终态不被覆写」。
        latest = job_store.get(job_id)
        if latest is not None and latest.status == "running":
            job_store.update(job_id, status="failed", message="任务异常结束(未进终态)")
        await engine.dispose()


def start_flywheel_job(*, settings, model_factory=None) -> str | None:
    """起一个后台飞轮任务,返回 job id。**忙(已有任务在跑)时返回 `None`。**

    返回 `None` 而不是抛:与 `orchestrate.start_job` 同一套约定,由调用方决定
    「忙」是不是错误(端点是 409,落池那三处**不是错误**)。

    `model_factory` 是**零参可调用**,在后台线程里被调**恰好一次**;缺省是
    `create_extract_model(settings)`(温度 0 —— 标准化与查重都是结构化判断,
    与 ch04 的挖知识同一个模型口径)。它同时也是测试的注入点。
    """
    job_store = get_job_store()
    job = job_store.start(JOB_KIND)
    if job is None:
        return None
    if model_factory is None:
        model_factory = lambda: create_extract_model(settings)  # noqa: E731
    _spawn(job_store, job, settings,
           lambda: _run_flywheel(job_store, job.id, settings, model_factory))
    return job.id


def start_flywheel_job_safely(settings) -> str | None:
    """**落池之后**那三处 fire-and-forget 的入口:起不动也绝不影响调用方。

    为什么独独要有这一层:那三处分别长在用户的**聊天轮**(`confidence_gate` 与
    `agent` 的知识轮)与**反馈 POST** 里,而它们对飞轮做的事只是「顺手把刚落的
    那一行喂过去」。后台任务的装配故障(线程起不来、JobStore 出问题)不该把用户
    一次正常的请求变成 500 —— 飞轮是**旁观者**,不是那条请求的一部分。

    吞掉的是 `Exception`,**不是 `BaseException`**:取消(`CancelledError`)必须
    照常上抛(客户端断开时那条请求就该停),这与本仓其它「尽力而为」的分支
    同款(`app/api/feedback.py` 的回捞、`app/kb/orchestrate.py` 的兜底)。

    ⚠️ **忙(`None`)不是故障**:池子里那行下一轮还会被吃到,所以这里**不打日志**
    —— 打了的话,「飞轮一直在跑」会伪装成「飞轮起不来」。
    """
    try:
        return start_flywheel_job(settings=settings)
    except Exception:  # noqa: BLE001
        logger.warning("飞轮后台任务没起来(不影响本次请求)", exc_info=True)
        return None
