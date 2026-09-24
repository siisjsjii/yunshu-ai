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

## 两种入口

- `start_flywheel_job`:**手动**触发(端点 / 脚本),忙时返回 `None`,由调用方决定
  要不要 409 —— 与 `orchestrate.start_job` 同一套约定。
- `start_flywheel_job_safely`:**落池之后**那三处 fire-and-forget 用的形状
  (`置信度闸` / `商品咨询`自评轮 / `POST /api/feedback`)。区别只有一条:
  它**绝不抛**。
"""

import asyncio
import logging
import threading

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.flywheel.pipeline import run_flywheel
from app.kb.jobs import Job, JobStore, get_job_store
from app.llm import create_extract_model
from app.sanitize import redact_api_key

logger = logging.getLogger(__name__)

#: 进 `JobStore` 的任务类型。`GET /api/kb/jobs` 把它**原样**回给前端
#: (`_job_dict` 读的是 `Job.type`)—— 所以新类型不需要改那两个只读端点。
JOB_KIND = "flywheel"


def _spawn(job_store: JobStore, job: Job, coro_fn) -> None:
    """专用线程 + 线程内 `asyncio.run`(ch04 模板)。

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
            #
            # ⚠️ 但**不许把 `done` 覆写成 `failed`**:`JobStore.update` 不是
            # 「只覆盖 message」,它会连 `status` 一起改(本条注释的前一版就是这么
            # 写错的)。会走到这里而 job 已经是 `done` 的只有一种情形 ——
            # **一批跑成了、`dispose()` 却炸了**;那时 `result` 已经落地,再把状态
            # 覆成 `failed` 只会让前端读到一对自相矛盾的值(与 `_run_flywheel`
            # 的 finally 里那条守卫同一个理由)。dispose 的失败留在**日志**里。
            # ⚠️ 变量名**不能叫 `job`** —— 那是 `_spawn` 的形参,`target` 里再绑一次
            # 就把它变成局部名,于是**上面那行 `logger.error(..., job.id)` 会
            # `UnboundLocalError`**(赋值在后面),而它看起来像「日志模块坏了」。
            latest = job_store.get(job.id)
            if latest is not None and latest.status == "done":
                return
            job_store.update(
                job.id, status="failed",
                message=redact_api_key(str(exc), get_settings().openai_api_key))

    threading.Thread(target=target, name=f"flywheel-job-{job.id}", daemon=True).start()


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
    try:
        # 模型在**后台线程里**才建(`start_flywheel_job` 那三处是 fire-and-forget
        # 调它,调用方是用户请求的那条线程 —— 那边一毫秒都不该花在建模型上)。
        model = model_factory()
        async with factory() as session:
            result = await run_flywheel(
                session=session, model=model,
                batch_size=settings.flywheel_batch_size)
        job_store.update(
            job_id, status="done",
            message=(f"处理 {result['processed']} 行:"
                     f"归并 {result['merged']}、新建 {result['created']}、"
                     f"失败 {result['failed']}"),
            result=result)
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
    _spawn(job_store, job,
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
