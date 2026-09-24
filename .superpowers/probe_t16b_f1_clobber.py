"""T16b 复审 F1 的**确定性复现**:死线取消 ⇒ session 退出抛 ⇒ 超时文案被冲掉。

复审用「`__aexit__` 会抛的 session 替身」演示了传播性质,但它**明确说自己没有用
真·失步连接复现**。本探针按两种形态各跑一次,把「值不值得修」变成读数:

  A. **真形态**:真 MySQL + 真 `_fresh_factory` + 真 SQL,让死线把**一个 DB await**
     (不是模型调用)取消掉 —— 看 session 退出到底会不会抛。
  B. **性质形态**:session 替身的 `__aexit__` 抛(SQLAlchemy 的
     `PendingRollbackError`/`InterfaceError` 就是这一类)—— 复审那个演示,我复跑一遍。

两种形态看的是**同一件事**:到达 `_spawn` 兜底的那个异常是 `FlywheelDeadlineExceeded`
还是别的。`JobStore.message` 是操作员唯一的信号(§T16 报告),所以判据就是它的原文。

用法:
    .venv/Scripts/python.exe .superpowers/probe_t16b_f1_clobber.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402


def out(text_: str = "") -> None:
    sys.stdout.buffer.write((text_ + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


async def _wait(job_store, job_id, *, timeout=30.0):
    for _ in range(int(timeout / 0.05)):
        job = job_store.get(job_id)
        if job is not None and job.status != "running":
            return job
        await asyncio.sleep(0.05)
    return job_store.get(job_id)


async def case_a_real_db_cancel() -> None:
    """A:死线取消一个**真的** DB await(慢查询),看 session 退出抛不抛。"""
    from app.config import get_settings
    from app.flywheel import tasks
    from app.kb.jobs import get_job_store

    out("\n===== A. 真 MySQL:死线取消一个慢查询,然后看 session 退出 =====")
    settings = get_settings().model_copy(
        update={"flywheel_job_timeout_seconds": 0.4})
    get_job_store.cache_clear()
    store = get_job_store()

    real_factory = tasks._fresh_factory          # 真 engine / 真 sessionmaker
    seen: dict = {}

    async def slow_query(*, session, model, batch_size):
        # 这一行是**真的** SQL:取消落在等 MySQL 回包的地方(asyncmy 在等 socket)
        await session.execute(text("SELECT SLEEP(8)"))
        return {"processed": 0, "merged": 0, "created": 0, "failed": 0}

    orig_run = tasks.run_flywheel
    tasks.run_flywheel = slow_query
    try:
        job_id = tasks.start_flywheel_job(
            settings=settings, model_factory=lambda: object())
        job = await _wait(store, job_id)
        seen["job"] = job
    finally:
        tasks.run_flywheel = orig_run

    out(f"  status  = {job.status!r}")
    out(f"  message = {job.message!r}")
    out(f"  ↑ 是不是死线那句话: {'是' if '超时' in (job.message or '') else '★ 不是 ★'}")
    # 真 engine 已在任务里 dispose 过;这里再兜一次防跨循环噪声
    engine, _ = real_factory(settings)
    await engine.dispose()


async def case_b_exit_raises() -> None:
    """B:session 替身的 `__aexit__` 抛(复审那个演示的复跑)。"""
    from app.config import get_settings
    from app.flywheel import tasks
    from app.kb.jobs import get_job_store

    out("\n===== B. session 退出抛(替身):死线取消 + __aexit__ 抛 =====")
    settings = get_settings().model_copy(
        update={"flywheel_job_timeout_seconds": 0.3})
    get_job_store.cache_clear()
    store = get_job_store()

    class _BoomOnExit:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            # 与本仓 ch08 记过的那条同族:`pending rollback` 的 session 再被用时,
            # `str()` 就是这段裸的 SQLAlchemy 内部文本
            raise RuntimeError(
                "This Session's transaction has been rolled back due to a previous "
                "exception during flush. To begin a new transaction with this Session, "
                "first issue Session.rollback(). Original exception was: simulated")

    class _BoomMaker:
        def __call__(self):
            return _BoomOnExit()

    class _NoopEngine:
        async def dispose(self):
            return None

    async def slow(*, session, model, batch_size):
        # 卡在一个 DB await 上:死线会把它取消
        await asyncio.sleep(30)
        return {"processed": 0, "merged": 0, "created": 0, "failed": 0}

    orig_factory, orig_run = tasks._fresh_factory, tasks.run_flywheel
    tasks._fresh_factory = lambda s: (_NoopEngine(), _BoomMaker())
    tasks.run_flywheel = slow
    try:
        job_id = tasks.start_flywheel_job(settings=settings, model_factory=lambda: object())
        job = await _wait(store, job_id)
    finally:
        tasks._fresh_factory, tasks.run_flywheel = orig_factory, orig_run

    out(f"  status  = {job.status!r}")
    out(f"  message = {(job.message or '')[:160]!r}")
    out(f"  ↑ 是不是死线那句话: {'是' if '超时' in (job.message or '') else '★ 不是 ★'}")


async def main() -> None:
    await case_a_real_db_cancel()
    await case_b_exit_raises()


if __name__ == "__main__":
    asyncio.run(main())
