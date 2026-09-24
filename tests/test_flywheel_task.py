"""飞轮后台任务(ch09 §8.3):专用线程 + 自建 engine + 终态一定落地。

本文件**刻意不打模块级 `pytestmark = pytest.mark.db`**:除了最后一条,这里的用例
全是**纯胶水**(线程、JobStore 状态流转、409、终态、结果透传),它们必须被
`-m "not db"` 跑到 —— 把整份文件标成 db,这条链路就在「不需要 MySQL」的那一轮里
**完全不跑**,而它恰恰是「后台任务」这件事的全部。

## 替身的边界(哪一条性质由哪条用例守)

- `_fresh_factory` → **除最后一条 db 用例以外全部打桩**。所以「自建 engine +
  dispose」这条性质在本文件里**只由那一条守** —— 打桩版本验的是**胶水**
  (线程真的起了、状态真的流转、槽真的摘了)。假 engine 让那三件事变成空话的
  风险是明摆着的:替身**替被测对象完成了语义**(本仓编目过的形态 ⑦)。
- `run_flywheel` → **全部打桩,一个例外都没有**。真流水线会吃
  `low_confidence_questions` 里 `matched_review_id IS NULL` 的行 —— 库里躺着
  **30 行前几章演示/验收留下的旧数据**(T14 实测:`COUNT(*)` = 30、`max(id)` = 305、
  `matched_review_id IS NOT NULL` = **0**),它们是**别人的数据**;而且真流水线会打
  真实模型。打桩之后本文件对那张表**只有读**(最后一条 db 用例里的一次 `SELECT 1`),
  一个字都不写。

## 「跑完了」这件事怎么断才算数(H3)

`calls` 记的是**流水线替身拿到的三个实参**:批大小取自 `settings`(用例显式传
**非默认值**,这样「读的是配置」与「写死 10」当场分得开)、模型就是 `model_factory`
交出来的那个对象、session 就是 sessionmaker 交出来的那一个。只断
「`session is not None`」的话,一个**压根没传 session**(或传了个新造的)的实现
照样绿。

## 那三处 fire-and-forget 钩子

`app/agent/nodes.py` 两处 + `app/api/feedback.py` 一处,全部由 `tests/conftest.py`
的 autouse 装置换成**记录器**(不打桩的话,任何走到落池的用例都会真的起线程去吃
上面那 30 行旧数据)。本文件只覆盖**闸**那一处(它最便宜、且 `pass`/`fail` 两支
都能现造);另两处的「落池 ⇒ 起了飞轮」断在**它们各自的落池用例**里 ——
`tests/test_agent_protocol.py::test_useful_false_falls_back_and_records_one_pool_row`
与 `tests/test_api_feedback.py::test_down_writes_one_row_with_user_feedback_entry_point`
—— 断言跟着「落池」这件事走,而不是在这里重搭一套协议/端点脚手架。
"""

import asyncio
import logging
import threading

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

from app.agent.nodes import make_confidence_gate_node
from app.config import Settings, get_settings
from app.db.base import get_engine
from app.flywheel import tasks
from app.kb.jobs import get_job_store
from app.main import app

#: 流水线替身的返回值。四个键**都带上非零值** —— 全 0 的话「有没有原样透传」
#: 与「照原样编了一个」在断言上分不开。
PIPELINE_RESULT = {"processed": 7, "merged": 3, "created": 2, "failed": 2}

#: `model_factory` 的返回值。只要是个**身份可辨的对象**即可 —— 用例断的是
#: 「流水线拿到的是它」,而不是它像不像一个 ChatOpenAI。
MODEL = object()

_REQUIRED = dict(
    openai_base_url="https://example.invalid/v1",
    openai_api_key="sk-test",
    openai_model="test-model",
    database_url="mysql+asyncmy://u:p@h:3306/db",
)


def _settings(**over) -> Settings:
    # `_env_file=None`:仓库根有真实 `.env`,不传的话缺字段的用例会被它静默补上。
    return Settings(_env_file=None, **{**_REQUIRED, **over})


@pytest.fixture
def job_store():
    """每个用例一份**干净**的 `JobStore`。

    `get_job_store()` 是**进程级单例**(`@lru_cache(maxsize=1)`,生产就靠它串行),
    所以测试必须显式清掉它:上一条用例留下的 running 任务会让这一条**永远是 409**
    (第二轮 `start` 直接返回 None),而红起来指向的是「实现坏了」。
    """
    get_job_store.cache_clear()
    yield get_job_store()
    get_job_store.cache_clear()


# --------------------------------------------------------------------------
# 替身
# --------------------------------------------------------------------------


class _FakeSession:
    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _FakeSessionMaker:
    """`async_sessionmaker(engine)` 的替身:记下**它交出去的每一个 session**。"""

    def __init__(self) -> None:
        self.sessions: list = []

    def __call__(self) -> _FakeSession:
        session = _FakeSession()
        self.sessions.append(session)
        return session


class _FakeEngine:
    def __init__(self) -> None:
        self.disposes = 0

    async def dispose(self) -> None:
        self.disposes += 1


def _patch_fresh_factory(monkeypatch):
    """把「自建 engine」换成假替身。**除最后那条 db 用例以外,全部用例都走这里。**"""
    engine, maker = _FakeEngine(), _FakeSessionMaker()
    monkeypatch.setattr(tasks, "_fresh_factory", lambda settings: (engine, maker))
    return engine, maker


def _patch_pipeline(monkeypatch, *, result=None, exc=None, on_call=None):
    """打桩**流水线**并记下它拿到的实参(见文件头「怎么断才算数」)。

    `on_call` 在**流水线被调用的那一刻**跑(用来把那个线程卡住 —— 409 用例靠它)。
    """
    calls: list = []

    async def fake_run_flywheel(*, session, model, batch_size):
        calls.append({"session": session, "model": model, "batch_size": batch_size})
        if on_call is not None:
            on_call()
        if exc is not None:
            raise exc
        return dict(PIPELINE_RESULT if result is None else result)

    monkeypatch.setattr(tasks, "run_flywheel", fake_run_flywheel)
    return calls


async def _wait(job_store, job_id, *, until=None, timeout=10.0):
    """轮询到终态(`until` 是额外条件)。

    ⚠️ **只等 `status != running` 是不够的**:同一件事有两次写 —— `_run_flywheel`
    的 finally 先落 `failed`(通用文案),`_spawn` 的兜底随后把**失败原因**补上去。
    只等状态会在原因还没落上时就返回,于是「原因写在 message 里」那条断言**偶发红**。
    """
    for _ in range(int(timeout / 0.02)):
        job = job_store.get(job_id)
        if job is not None and job.status != "running" and (until is None or until(job)):
            return job
        await asyncio.sleep(0.02)
    return job_store.get(job_id)


# --------------------------------------------------------------------------
# 一、胶水(不读库)
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_job_completes_and_clears_the_running_flag(monkeypatch, job_store):
    """跑一次 → `done`;**且跑完标记被摘掉**(不摘的话第二次永远 409)。

    ⚠️ 本用例**先断「跑完那四条」再断「槽被摘掉」**:只断后者的话,一个
    **压根没跑流水线**的实现(起了线程就立刻置 done)同样绿 —— 那是本仓
    「假绿测试」里最常见的一种,而这里正是它的高发区(线程里的事看不见)。
    """
    engine, maker = _patch_fresh_factory(monkeypatch)
    calls = _patch_pipeline(monkeypatch)
    settings = _settings(flywheel_batch_size=3)      # 非默认值:见文件头

    job_id = tasks.start_flywheel_job(settings=settings, model_factory=lambda: MODEL)
    assert job_id is not None, "没有任务在跑时必须起得来"

    job = await _wait(job_store, job_id)
    assert job.status == "done", f"实际 {job.status}:{job.message}"

    # ① 流水线**真的被调过一次**,而且三个实参都对(见文件头 H3 那一段)
    assert len(calls) == 1, f"流水线该被调一次,实际 {calls}"
    assert calls[0]["batch_size"] == 3, (
        f"批大小必须取自 settings(不是写死的默认值),实际 {calls[0]['batch_size']}"
    )
    assert calls[0]["model"] is MODEL, "模型必须来自 model_factory(注入点没接上)"
    assert calls[0]["session"] is maker.sessions[0], (
        "session 必须是 sessionmaker 交出来的那一个(自建 engine 那条路上的)"
    )
    # ② 自建 engine 被收掉(这里只能断「那个假替身被 dispose 了一次」;
    #    真 engine 那条由文件末尾的 db 用例守)
    assert engine.disposes == 1, f"任务结束必须 dispose,实际 {engine.disposes} 次"

    # ③ 槽被摘掉 —— **这条才是本用例真正守的回归**:槽一漏,第二次永远 409
    again = tasks.start_flywheel_job(settings=settings, model_factory=lambda: MODEL)
    assert again is not None, "跑完必须摘掉运行槽,否则那个会话再也跑不了"
    assert await _wait(job_store, again) is not None


@pytest.mark.anyio
async def test_job_result_carries_the_pipelines_four_keys(monkeypatch, job_store):
    """流水线返回的四个键**原样**进 job(审核页/前端读的就是 `result`)。

    原样:包一层、漏一个键、把 `failed` 吞掉都会让这条红 —— 而 `failed`
    正是「池子里有一批行一直在失败」的唯一信号(`processed` 与
    `merged + created` 的差额也是靠它读的)。
    """
    _patch_fresh_factory(monkeypatch)
    _patch_pipeline(monkeypatch)

    job_id = tasks.start_flywheel_job(
        settings=_settings(), model_factory=lambda: MODEL)
    job = await _wait(job_store, job_id)

    assert job.result == PIPELINE_RESULT, f"result 必须原样透传,实际 {job.result}"
    assert set(job.result) == {"processed", "merged", "created", "failed"}


@pytest.mark.anyio
async def test_second_job_while_running_gets_409(monkeypatch, job_store):
    """**经端点**第二次触发 → 409「已有任务在跑」。

    走真的 `POST /api/kb/jobs/flywheel`(不是直接调 `start_flywheel_job`):
    端点才是验收脚本与前端用的那个入口,而它自己的那一行判断(忙 → 409)
    只有在这里才被行使。顺带钉住 `GET /api/kb/jobs/{job_id}` 对**新任务类型**
    原样可用(`type` 取自 Job,不需要新增端点)。
    """
    _patch_fresh_factory(monkeypatch)
    release = threading.Event()
    _patch_pipeline(monkeypatch, on_call=release.wait)

    app.dependency_overrides[get_settings] = lambda: _settings(
        flywheel_batch_size=3)
    job_id = None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = await client.post("/api/kb/jobs/flywheel")
            assert first.status_code == 201, f"实际 {first.status_code}:{first.text}"
            job_id = first.json()["job_id"]

            second = await client.post("/api/kb/jobs/flywheel")
            assert second.status_code == 409, (
                f"任务在跑时第二次必须 409,实际 {second.status_code}:{second.text}"
            )
            assert second.json()["detail"] == "已有任务在跑"

            got = await client.get(f"/api/kb/jobs/{job_id}")
            assert got.status_code == 200
            assert got.json()["type"] == "flywheel"
            assert got.json()["status"] == "running"
    finally:
        app.dependency_overrides.clear()
        release.set()                      # 别把那条线程留在卡住的状态里
        if job_id is not None:
            done = await _wait(job_store, job_id)
            assert done.status == "done", done.message


@pytest.mark.anyio
async def test_a_raising_pipeline_lands_in_failed_and_releases_the_slot(
    monkeypatch, job_store
):
    """流水线抛异常 ⇒ **终态 `failed`** 且槽被摘掉(「跑标记的摘除」那一条)。

    这里走的是 **job 自己那条 `finally`**(不是 `_spawn` 的兜底 —— 那条只保
    「线程级的意外」)。停在 running 的后果不是「多跑一次」,是那个会话
    **再也跑不了**,而用户侧每一轮看起来都完全正常(ch07 记过同款)。
    """
    engine, _ = _patch_fresh_factory(monkeypatch)
    _patch_pipeline(monkeypatch, exc=RuntimeError("流水线炸了(替身注入)"))
    settings = _settings()

    job_id = tasks.start_flywheel_job(settings=settings, model_factory=lambda: MODEL)
    job = await _wait(
        job_store, job_id, until=lambda j: "流水线炸了" in (j.message or "")
    )

    assert job.status == "failed", f"异常路径必须落终态,实际 {job.status}"
    assert "流水线炸了" in job.message, (
        f"失败原因必须留在 message 里(池子里那批行只会「一直不消失」,"
        f"没人知道为什么),实际 {job.message!r}"
    )
    assert engine.disposes == 1, "异常路径同样要把自建 engine 收掉"
    assert job_store.is_busy() is False, "槽没摘 ⇒ 下一个任务永远 409"
    assert tasks.start_flywheel_job(
        settings=settings, model_factory=lambda: MODEL) is not None


@pytest.mark.anyio
async def test_base_exception_path_also_lands_in_a_terminal_status(
    monkeypatch, job_store
):
    """`CancelledError` 是 **`BaseException`** —— 只抓 `Exception` 的兜底会漏掉它。

    `except Exception` 那条兜底对取消**看不见**:协程被取消时它自己不会再落终态,
    于是任务**永远停在 running**,运行槽再也不释放。本仓的房规是
    `except BaseException`(ch03 的取消路径就是这么写的)。

    直接 `await _run_flywheel(...)` 而不经线程:那条路上的 CancelledError 是
    **确定性**的(线程里要靠超时才能造出来,而超时本身就是不确定的)。
    """
    engine, _ = _patch_fresh_factory(monkeypatch)
    _patch_pipeline(monkeypatch, exc=asyncio.CancelledError())
    job = job_store.start(tasks.JOB_KIND)
    assert job is not None

    with pytest.raises(asyncio.CancelledError):
        await tasks._run_flywheel(job_store, job.id, _settings(), lambda: MODEL)

    assert job_store.get(job.id).status == "failed"
    assert engine.disposes == 1
    assert job_store.start(tasks.JOB_KIND) is not None, "取消路径也必须摘掉运行槽"


@pytest.mark.anyio
async def test_a_failing_engine_factory_does_not_leak_the_slot(monkeypatch, job_store):
    """**engine 都造不出来**(URL 写错)⇒ 照样不许留下一个 running 的僵尸任务。

    这一条走的是 `_spawn` 的兜底(自建 engine 在 `try` 之外,与 ch04 模板一致),
    断的是「线程级意外」那一层 —— 与上面两条合起来,三条出口都盖住了:

      - 流水线抛异常 → job 自己的 finally
      - 协程被取消   → job 自己的 finally
      - 装配期抛异常 → `_spawn` 的 `except BaseException`
    """
    def boom(settings):
        raise RuntimeError("database_url 写错了(替身注入)")

    monkeypatch.setattr(tasks, "_fresh_factory", boom)
    settings = _settings()

    job_id = tasks.start_flywheel_job(settings=settings, model_factory=lambda: MODEL)
    job = await _wait(job_store, job_id)

    assert job.status == "failed", f"实际 {job.status}:{job.message}"
    assert job_store.is_busy() is False
    assert tasks.start_flywheel_job(
        settings=settings, model_factory=lambda: MODEL) is not None


@pytest.mark.anyio
async def test_spawn_fallback_never_downgrades_a_done_job(job_store):
    """**跑成了就别改口**:`_spawn` 那条兜底会覆写 `failed` 的**原因**,但不许把
    `done` 覆成 `failed`。

    会走到这条路而 job 已经是 `done` 的只有一种情形 —— **一批跑成了、
    `dispose()` 却炸了**(协程正常返回前抛)。那时 `result` 已经落地(前端读的
    就是它),再把状态改成 `failed` 会让「处理 7 行」与「任务失败」**同时**出现在
    同一个 job 上;与 `_run_flywheel` 那条 finally 守卫是同一个理由。

    ⚠️ 它钉的是一件**很容易被写错**的事实:`JobStore.update` **不是**
    「对已是终态的 job 只覆盖 message」—— 它连 `status` 一起改(本文件这一段
    的第一版就是照那句话写的,实测被这条用例逮住)。
    """
    async def coro():
        job_store.update(job.id, status="done", message="跑完了",
                         result=dict(PIPELINE_RESULT))
        raise RuntimeError("dispose 炸了(替身注入)")

    job = job_store.start(tasks.JOB_KIND)
    assert job is not None
    tasks._spawn(job_store, job, coro)

    # 先等 `result` 落地(它证明协程真的跑过),再等那条线程收尾 —— 兜底若要
    # 覆写,就在这两步之间发生(同一线程、纯 CPU、微秒级)。
    await _wait(job_store, job.id, until=lambda j: j.result is not None)
    name = f"flywheel-job-{job.id}"
    for _ in range(500):
        if not any(t.name == name for t in threading.enumerate()):
            break
        await asyncio.sleep(0.01)

    settled = job_store.get(job.id)
    assert settled.status == "done", (
        f"兜底把一次成功的任务改口成 {settled.status}:{settled.message}"
        f"(result 还在:{settled.result})"
    )
    assert settled.result == PIPELINE_RESULT


# --------------------------------------------------------------------------
# 二、fire-and-forget 的那一层守卫(不读库)
# --------------------------------------------------------------------------


def test_safely_swallows_the_launch_failure_and_logs_it(caplog):
    """`start_flywheel_job_safely`:**装配故障只留日志,绝不抛给调用方**。

    它是三处落池点唯一的调用形状,而那三处分别长在用户的**聊天轮**与**反馈 POST**
    里 —— 后台任务起不来的唯一可接受后果是「这一轮没顺手喂飞轮」,不是「用户
    看到 500」。吞掉必须**留痕**(带 traceback):否则池子里的行只会「一直不消失」,
    没人知道是飞轮压根没跑起来。
    """
    def boom(**kwargs):
        raise RuntimeError("线程起不来(替身注入)")

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(tasks, "start_flywheel_job", boom)
        with caplog.at_level(logging.WARNING, logger="app.flywheel.tasks"):
            assert tasks.start_flywheel_job_safely(_settings()) is None
    finally:
        monkey.undo()

    loud = [rec for rec in caplog.records
            if rec.name == "app.flywheel.tasks" and rec.levelno >= logging.WARNING]
    assert loud, "吞掉必须留在日志里,否则「没起任务」与「没有待处理的行」分不开"
    assert any(rec.exc_info for rec in loud), "不带 traceback 就查不出挂在哪一步"


def test_safely_returns_none_when_a_job_is_already_running(monkeypatch):
    """忙 ⇒ `None`(**不是故障**):池子里那行下一轮还会被吃到,连日志都不该打。"""
    monkeypatch.setattr(
        tasks, "start_flywheel_job", lambda **kwargs: None)
    assert tasks.start_flywheel_job_safely(_settings()) is None


# --------------------------------------------------------------------------
# 三、那三处落池钩子里的**闸**那一处(不读库)
# --------------------------------------------------------------------------


class _GateSession:
    """闸只对 session 要 `add` + `commit`(`record_low_confidence` 的接口面)。"""

    def __init__(self) -> None:
        self.added: list = []

    def add(self, row) -> None:
        self.added.append(row)

    async def commit(self) -> None:
        pass


def _gate_settings(**over) -> Settings:
    """闸那两条用例的配置。阈值显式给 0.5(**不用生产默认值**):

    「弱证据算出来 0.2267、强证据 0.74」这两条数据要**落在它两侧**,用例才谈得上
    分「拦下」与「放行」两支;拿生产默认值(将来被标定挪动)会让某一天两支同时
    倒向一边,而那红起来指向的是「实现坏了」。后半句与 `tests/test_agent_gate_ch09.py`
    同款 —— 那一份的理由是「两个结论必须来自数据,不能来自配置」。
    """
    return _settings(evidence_confidence_threshold=0.5, **over)


def _evidence(score: float, chunk_id: int = 1) -> dict:
    """一条证据 —— 六个键与 `retrieve_knowledge` 写进通道的**逐字相同**(dict 形状)。"""
    return {
        "chunk_id": chunk_id, "section_path": "退换货 > 退货政策",
        "question": "怎么退货", "answer": "七天无理由退货",
        "category": "退换货", "score": score,
    }


@pytest.mark.anyio
async def test_gate_pooling_fires_the_flywheel_once_with_this_settings(flywheel_hooks):
    """闸拦下 ⇒ **落池之后**真的起了一轮飞轮,且传的是**这一份** settings。

    ⚠️ 断的是**同一性**(`is`),不是「被调过一下」:传 `get_settings()`(真 `.env`)
    而不是端点/节点手上那份配置的实现,照样能让 `len(calls) == 1` 通过 ——
    而它的后果是飞轮**跑到另一个库上**去,那条路不报任何错。

    ⚠️ 前提断言(`added` 非空)不能省:少了它,一个「压根没落池」的实现会让
    下面那条同样绿,而这一条用例的全部意义正是「**落池之后**才起」。
    """
    session = _GateSession()
    settings = _gate_settings()
    node = make_confidence_gate_node(
        settings=settings, session=session, conversation_id="c1")

    out = await node({"user_input": "猫砂盆多少钱", "evidence": [_evidence(0.2)]})

    assert out["gate_passed"] is False, "前置:这条证据必须被拦下"
    assert len(session.added) == 1, "前置:拦下时必须先落池"
    assert len(flywheel_hooks.calls) == 1, (
        f"落池之后必须起一轮飞轮,实际起了 {len(flywheel_hooks.calls)} 次"
    )
    assert flywheel_hooks.calls[0] is settings, (
        "起飞轮必须用这一份 settings(传 get_settings() 会跑到另一个库上)"
    )


@pytest.mark.anyio
async def test_gate_pass_does_not_fire_the_flywheel(flywheel_hooks):
    """过了闸 ⇒ **一次都不许起**(没有新行落池,起了就是白跑一批)。

    「只在没通过的那一支」是一条**位置**约束:把那一行挪到 `if not passed`
    外面,这个文件里其余**全部**用例照样绿(它们只断 `fail` 那一支),
    而每一轮正常的知识问答都会白起一次飞轮 —— 一次真模型往返加上一次库扫描。
    """
    session = _GateSession()
    settings = _gate_settings()
    node = make_confidence_gate_node(
        settings=settings, session=session, conversation_id="c1")

    out = await node({"user_input": "退货政策",
                      "evidence": [_evidence(0.9, i) for i in (1, 2, 3)]})

    assert out["gate_passed"] is True, "前置:这一组强证据必须过闸"
    assert session.added == [], "前置:过闸不该落池"
    assert flywheel_hooks.calls == [], (
        f"过闸就没落池,不该起飞轮,实际起了 {len(flywheel_hooks.calls)} 次"
    )


# --------------------------------------------------------------------------
# 四、真 engine(**这条不打桩 `_fresh_factory`**)
# --------------------------------------------------------------------------


@pytest.mark.db
@pytest.mark.anyio
async def test_real_engine_is_built_from_settings_and_disposed(monkeypatch, job_store):
    """**不打桩 `_fresh_factory`** —— 验的就是「自建 engine + dispose」本身。

    为什么非真不可:打桩版让替身**替被测对象完成了语义**(形态 ⑦)—— 十遍都证明
    不了后台线程里那个 engine 真的连得上 MySQL、也证明不了它真的被收掉。而这正是
    brief 说「一个字都不能省」的第一条。

    「被 dispose 了」有一个**真发生才有**的痕迹(SQLAlchemy 2.0.53 的
    `Engine.dispose` 源码逐字:先 `self.pool.dispose()`、再
    `self.pool = self.pool.recreate()`;而 `QueuePool.dispose` 把池里的连接逐个
    close 掉):

      ① engine 的**池对象换了一个** —— 只有 dispose 会换(`AsyncEngine.dispose`
         是协程,**漏掉 `await` 的话两条都不成立**);
      ② 老池子里那条**真连上过**的连接**不见了**(checkedin 1 → 0)。

    ⚠️ 流水线**仍然打桩**:真流水线会吃池子里那 30 行前几章的旧数据(别人的数据),
    还会打真实模型。替身只做一件**真事** —— 用这个 engine 真跑一次 `SELECT 1`
    (`session.bind` 就是 `_fresh_factory` 造的那个 engine;它是真连上库的证据),
    然后立刻返回。
    """
    settings = get_settings()          # ★ 真 `.env` —— db 用例的分界就在这一行
    seen: dict = {}

    async def fake_run_flywheel(*, session, model, batch_size):
        engine = session.bind
        assert engine is not None
        # 真连一次(连接用完即回池 ⇒ checkedin 变成 1)
        async with engine.connect() as conn:
            assert (await conn.execute(text("SELECT 1"))).scalar_one() == 1
        seen["engine"] = engine
        seen["pool_before"] = engine.pool
        seen["pooled_connection"] = engine.pool.checkedin()
        return dict(PIPELINE_RESULT)

    monkeypatch.setattr(tasks, "run_flywheel", fake_run_flywheel)
    try:
        job_id = tasks.start_flywheel_job(settings=settings, model_factory=lambda: MODEL)
        job = await _wait(job_store, job_id)
        assert job.status == "done", f"实际 {job.status}:{job.message}"
        assert job.result == PIPELINE_RESULT

        engine = seen["engine"]
        assert engine is not get_engine(), (
            "自建 engine:**不许**复用 `get_engine()` 的 lru_cache 单例"
            "(它绑在首次使用它的事件循环上,跨线程复用会出问题)"
        )
        assert engine.url == make_url(settings.database_url), "engine 必须照 settings 建"
        assert engine.sync_engine.pool._pre_ping is True, "pool_pre_ping 与模板一致"

        assert seen["pooled_connection"] == 1, (
            "前置:那次 SELECT 1 真的从池里取了一条连接 —— 不然下面那条「连接不见了」"
            "是恒真的(池子本来就是空的)"
        )
        assert engine.pool is not seen["pool_before"], (
            "真 engine 没被 dispose(池对象没被换过)。`await` 漏了的话就是这个观测"
        )
        assert seen["pool_before"].checkedin() == 0, (
            "老池子里那条连接必须被 dispose 关掉(它是一次真实的 MySQL 连接)"
        )
    finally:
        # 与 `tests/test_api_feedback.py` 同款:跨事件循环的 asyncmy 连接会在
        # 下一个循环里打出一串无害但很吵的 `ERROR sqlalchemy.pool`。
        await get_engine().dispose()
