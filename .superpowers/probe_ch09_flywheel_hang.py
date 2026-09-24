"""T16b 复现探针:飞轮后台任务在「对端不回包」时**永久**卡住、槽被占死。

复现的是**机制**(不是触发条件):

    真实触发(网络那一段)   → 首次 TCP+TLS 连接既不失败也不成功(静默丢弃)
    本探针把这一段换成一个  → 本地「黑洞」监听端口(listen 之后**永不 accept**)

两者对客户端是同一件事:**连接建立成功了,但一个字节的响应都不回来**。
差别只在触发源可控(见 task-16-report §6-C1 的现场读数:666s 仍是 running、
零日志、innodb_trx 空转、443 CLOSE_WAIT)。

探针做四件事:
① 真的起一个飞轮任务(**从运行中的事件循环里起** —— 与三处落池钩子/端点同形状),
   模型用**真的** `create_extract_model`(不打桩),库用**真的** `.env` 那一个;
② 卡住期间 dump **全部线程的栈**(faulthandler + sys._current_frames);
③ 查库:`information_schema.innodb_trx` 里那条「开着却空转」的事务;
④ 再起一个任务 ⇒ 拿到 None(端点那条路就是 409)。

用法:
    .venv/Scripts/python.exe .superpowers/probe_ch09_flywheel_hang.py
    .venv/Scripts/python.exe .superpowers/probe_ch09_flywheel_hang.py --outside-loop
    .venv/Scripts/python.exe .superpowers/probe_ch09_flywheel_hang.py --real-gateway --seconds 30
"""

import argparse
import asyncio
import faulthandler
import gc
import logging
import os
import socket
import sys
import threading
import time
import traceback

# 脚本在 `.superpowers/` 下 ⇒ sys.path[0] 是它,`import app` 会找不到
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def out(text: str = "") -> None:
    """输出边界钉死 UTF-8(本机 locale 是 cp936:`print('⇒')` 会当场 UnicodeEncodeError)。"""
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def loud_logging() -> None:
    """只把这次要看的三家日志打到 stdout(不碰 log/app.log —— 那是真服务日志)。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("    [log] %(name)s %(levelname)s %(message)s"))
    for name in ("openai._base_client", "httpx2", "httpx", "app.flywheel"):
        lg = logging.getLogger(name)
        lg.disabled = False
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)
        lg.propagate = False


def blackhole() -> tuple[socket.socket, int]:
    """一个**永不 accept** 的监听端口:三次握手在内核里完成(连接成功),之后静默。

    在监听 ⇒ 客户端 connect() 立刻成功(不像关闭端口那样 2s 才拒绝);
    不 accept ⇒ 请求发得出去、响应永远不来。这就是「对端不回包」的最小复刻。
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)                      # 内核替我们完成握手,而我们从不 accept
    return srv, srv.getsockname()[1]


def dump_all_thread_stacks(tag: str) -> None:
    """全部线程的栈 —— 「卡在哪一行」的证据之一。

    ⚠️ 只有这一份是**不够**的:协程一旦挂起,它的 Python 帧**不在线程栈上**
    (所以这里只看得到 `select()` / `GetQueuedCompletionStatus`,看不到
    「正在等哪个 HTTP 响应」)。真正指名道姓的是下面 `dump_suspended_tasks`。
    """
    out(f"\n===== 全部线程的栈({tag})=====")
    frames = sys._current_frames()
    for th in threading.enumerate():
        frame = frames.get(th.ident)
        out(f"--- 线程 {th.name!r} daemon={th.daemon} alive={th.is_alive()}")
        if frame is None:
            out("    (拿不到帧)")
            continue
        for line in traceback.format_stack(frame)[-9:]:
            out("    " + line.rstrip())
    out("===== 栈结束 =====\n")


def _frames_of(coro) -> list:
    """某个**挂起中**的协程/生成器自己的帧(顺着 `f_back`,含被 inline 的 await 链)。"""
    frame = getattr(coro, "cr_frame", None) or getattr(coro, "ag_frame", None)
    frames = []
    while frame is not None:
        frames.append(frame)
        frame = frame.f_back
    frames.reverse()
    return frames


def _await_chain(obj, *, depth: int = 0) -> list[str]:
    """顺着 `cr_await` 一层层往下走 —— 直到看见「在等谁」。

    `Task.get_stack()` 只走 `f_back`,一旦挂起的是**另一个 Task/Future**
    (langchain 到处这么干),链就断在那里;所以要显式跟 `cr_await`。
    """
    lines: list[str] = []
    while obj is not None and depth < 80:
        depth += 1
        if isinstance(obj, asyncio.Task):
            lines.append(f"→ Task {obj.get_name()!r} {obj.get_coro().__qualname__}")
            obj = obj.get_coro()
            continue
        if isinstance(obj, asyncio.Future):          # 裸 Future(= 在等 I/O 回调)
            lines.append(f"→ Future {type(obj).__name__} done={obj.done()} {obj!r:.60}")
            break
        frames = _frames_of(obj)
        for f in frames:
            if "site-packages" in (f.f_code.co_filename or "") or "app" in (
                    f.f_code.co_filename or ""):
                lines.append(f"    {os.path.basename(f.f_code.co_filename)}:"
                             f"{f.f_lineno} in {f.f_code.co_name}")
        nxt = getattr(obj, "cr_await", None) or getattr(obj, "ag_await", None)
        if nxt is obj:
            break
        obj = nxt
    if lines and not lines[-1].startswith(("→ Future", "→ Task")):
        lines.append(f"→ (链在此断:等的是 {type(obj).__name__},没有更深的 Python 帧)")
    return lines


def dump_suspended_tasks(tag: str) -> None:
    """**挂起中**的协程的 await 链 —— 「等在哪一个 await 上」的直接证据。"""
    out(f"\n===== 挂起中的协程 await 链({tag})=====")
    try:
        seen = set()
        for obj in gc.get_objects():
            if not isinstance(obj, asyncio.Task) or id(obj) in seen:
                continue
            seen.add(id(obj))
            chain = _await_chain(obj)
            if not any("flywheel" in ln or "httpx" in ln or "openai" in ln or "langchain" in ln or "base.py" in ln for ln in chain):
                continue
            out("--- " + "\n--- ".join(chain))
    except Exception as exc:  # noqa: BLE001 —— 探针不该因为这一层挂了就没输出
        out(f"    (dump 失败:{exc!r})")
    out("===== await 链结束 =====\n")


async def db_state() -> None:
    """那条「开着却空转」的事务 —— 与 T16 报告的现场读数同款。"""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.config import get_settings

    engine = create_async_engine(get_settings().database_url)
    try:
        async with engine.connect() as conn:
            rows = (await conn.execute(text(
                "SELECT trx_id, trx_state, trx_started, trx_rows_locked,"
                " trx_rows_modified, TIMESTAMPDIFF(SECOND, trx_started, NOW()) AS age_s"
                " FROM information_schema.innodb_trx ORDER BY trx_started"))).all()
            out(f"innodb_trx 共 {len(rows)} 条:")
            for r in rows:
                out(f"    {tuple(r)}")
            procs = (await conn.execute(text(
                "SELECT id, command, time, state, LEFT(info, 80)"
                " FROM information_schema.processlist ORDER BY id"))).all()
            out("processlist:")
            for p in procs:
                out(f"    {tuple(p)}")
    finally:
        await engine.dispose()


PROBE_ROW_REASON = "T16b 探针合成的池子行(跑完即删,不是真数据)"


async def _pool_counts() -> tuple[int, int]:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.config import get_settings

    engine = create_async_engine(get_settings().database_url)
    try:
        async with engine.connect() as conn:
            total = (await conn.execute(text(
                "SELECT COUNT(*) FROM low_confidence_questions"))).scalar_one()
            unn = (await conn.execute(text(
                "SELECT COUNT(*) FROM low_confidence_questions"
                " WHERE matched_review_id IS NULL"))).scalar_one()
            return int(total), int(unn)
    finally:
        await engine.dispose()


async def _insert_probe_row() -> int:
    """插一条**探针自己的**池子行(`matched_review_id IS NULL` ⇒ 会被吃到)。

    收工时**按 id 删掉它自己** —— 那是我加的合成行,留着会污染审核队列;
    库里的**既有**数据一行都不动(见报告里的前后计数)。
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.config import get_settings

    engine = create_async_engine(get_settings().database_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(
                "INSERT INTO low_confidence_questions"
                " (question, source_conversation_id, entry_point, reject_reason)"
                " VALUES (:q, NULL, '用户反馈', :r)"),
                {"q": "T16b 探针:这条是合成行", "r": PROBE_ROW_REASON})
            new_id = (await conn.execute(text("SELECT LAST_INSERT_ID()"))).scalar_one()
            return int(new_id)
    finally:
        await engine.dispose()


async def _delete_probe_row(row_id: int) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.config import get_settings

    engine = create_async_engine(get_settings().database_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(
                "DELETE FROM low_confidence_questions"
                " WHERE id = :i AND reject_reason = :r"), {"i": row_id, "r": PROBE_ROW_REASON})
    finally:
        await engine.dispose()


def _metrics(store, job_id: str, t0: float) -> None:
    from app.flywheel.tasks import start_flywheel_job

    job = store.get(job_id)
    out(f"t={time.time() - t0:.1f}s  status={job.status!r} message={job.message!r}")
    if job.finished_at is not None:
        out(f"job 寿命(created_at → finished_at)= {job.finished_at - job.created_at:.2f}s")
    out(f"is_busy() = {store.is_busy()}")
    second = start_flywheel_job(settings=SETTINGS)
    out(f"再起一个 -> {second!r}" + ("   (None = 端点此刻就是 409)" if second is None else ""))
    dump_all_thread_stacks(f"t={time.time() - t0:.1f}s")
    dump_suspended_tasks(f"t={time.time() - t0:.1f}s")


SETTINGS = None


def main() -> None:
    global SETTINGS
    from app.config import get_settings
    from app.flywheel.tasks import start_flywheel_job
    from app.kb.jobs import get_job_store

    ap = argparse.ArgumentParser()
    ap.add_argument("--outside-loop", action="store_true",
                    help="对照组:**没有**运行中的事件循环(模块线程直接起)")
    ap.add_argument("--real-gateway", action="store_true",
                    help="对照组:不打黑洞,真的打 .env 里那个网关(应当正常跑完)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--llm-timeout", type=float, default=None,
                    help="覆盖 settings.llm_timeout_seconds(修好后用它把演示压到秒级)")
    ap.add_argument("--job-timeout", type=float, default=None,
                    help="覆盖 settings.flywheel_job_timeout_seconds(寿命上界)")
    ap.add_argument("--synthetic-row", action="store_true",
                    help="插一条探针自己的池子行(池子空时才有活干),收工删掉它")
    args = ap.parse_args()

    srv, port = blackhole()
    out(f"黑洞监听端口 = {port}(listen 了但永不 accept)")
    settings = get_settings()
    over = {}
    if not args.real_gateway:
        over["openai_base_url"] = f"http://127.0.0.1:{port}/v1"
    else:
        srv.close()
    if args.llm_timeout is not None:
        over["llm_timeout_seconds"] = args.llm_timeout
    if args.job_timeout is not None:
        over["flywheel_job_timeout_seconds"] = args.job_timeout
    settings = settings.model_copy(update=over) if over else settings
    SETTINGS = settings
    out(f"base_url = {settings.openai_base_url}   batch_size = {settings.flywheel_batch_size}")
    out(f"起点:inside_running_loop={not args.outside_loop}  "
        f"gateway={'real' if args.real_gateway else 'blackhole'}")

    get_job_store.cache_clear()
    store = get_job_store()

    if args.outside_loop:
        # 对照:模块线程、**没有任何事件循环**在跑
        row_id = asyncio.run(_insert_probe_row()) if args.synthetic_row else None
        try:
            out(f"池子(前):total/unmatched = {asyncio.run(_pool_counts())}"
                + (f"  探针行 id={row_id}" if row_id else ""))
            t0 = time.time()
            job_id = start_flywheel_job(settings=settings)
            out(f"start_flywheel_job -> {job_id!r}(返回即已占槽)")
            time.sleep(args.seconds)
            _metrics(store, job_id, t0)
            if not args.real_gateway:
                asyncio.run(db_state())
        finally:
            if row_id is not None:
                asyncio.run(_delete_probe_row(row_id))
                out(f"已删掉探针行 {row_id};池子(后):total/unmatched = "
                    f"{asyncio.run(_pool_counts())}")
    else:
        async def go() -> None:
            row_id = await _insert_probe_row() if args.synthetic_row else None
            try:
                out(f"池子(前):total/unmatched = {await _pool_counts()}"
                    + (f"  探针行 id={row_id}" if row_id else ""))
                t0 = time.time()
                job_id = start_flywheel_job(settings=settings)   # ★ 真 model_factory
                out(f"start_flywheel_job -> {job_id!r}(返回即已占槽)")
                if job_id is None:
                    out("!!! 没起来 —— 槽被占着")
                    return
                got = asyncio.Event()      # 事件循环在卡住期间**仍然可用**
                asyncio.get_running_loop().call_later(2.0, got.set)
                await got.wait()
                out("(事件循环在卡住期间照常跑:2s 定时器已触发)")
                await asyncio.sleep(max(0.0, args.seconds - 2.0))
                _metrics(store, job_id, t0)
                if not args.real_gateway:
                    await db_state()
            finally:
                if row_id is not None:
                    await _delete_probe_row(row_id)
                    out(f"已删掉探针行 {row_id};池子(后):total/unmatched = "
                        f"{await _pool_counts()}")

        asyncio.run(go())

    out(f"\n探针结束;进程退出会带走那条守护线程与它的事务")


if __name__ == "__main__":
    loud_logging()
    main()
