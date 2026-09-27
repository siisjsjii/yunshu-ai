"""评测后台任务(ch10 跟进):子进程 + 有界 + 故障可见。

本文件**不打模块级 `pytestmark = pytest.mark.db`**:这里全是**纯胶水**
(专用线程、JobStore 状态流转、子进程输出与退出码、超时杀进程、409),
它们必须被 `-m "not db"` 跑到 —— 把整份文件标成 db,这条链路就在
「不需要 MySQL」的那一轮里**完全不跑**,而它恰恰是「后台任务」这件事的全部。

## 替身的边界(哪一条性质由哪条用例守)

- **`scripts/run_eval.py` 从不被执行**(全部用例走 `command=` 注入)。
  真评测要 Milvus + 真实 key + 几分钟,而且它会往 `eval_runs` **真的写一行**
  —— 那是**别人的表**、且**不可撤销**(趋势表会多一轮)。
- **子进程本身是真的**:`subprocess.Popen` 之外一个字都不打桩。这是刻意的 ——
  这个模块唯一的新东西就是「怎么起它、怎么读它、怎么杀它」,
  把 `Popen` 也换掉的话,这一份测试测的就只是替身(本仓编目过的形态 ⑦)。
- **`_default_command`** 只被一条用例打桩(经端点的 201/409),
  另有一条**单独**钉住它的逐字形状 —— 打桩那条断的是胶水,这条断的是「跑的是
  命令行那条命令」。

## 为什么每条用例都自己起一个假子进程而不是共用一个

它们断的是**四件不同的事**(输出来自子进程 / stderr 合流 / 退出码带上它自己的话 /
到点真的被杀),共用一个装置会让「哪一条红了」变得含糊 —— 而本仓那条
「红在哪一条」的判据只有在**一条用例一件事**时才成立。
"""

import subprocess
import sys
import threading
import time

import httpx
import pytest

from app.config import Settings, get_settings
from app.kb import eval_job
from app.kb.eval_job import start_eval_job
from app.kb.jobs import get_job_store
from app.main import app

_REQUIRED = dict(
    openai_base_url="https://example.invalid/v1",
    openai_api_key="sk-evaljob-test-KEY",
    openai_model="test-model",
    database_url="mysql+asyncmy://u:p@h:3306/db",
)


def _settings(**over) -> Settings:
    # `_env_file=None`:仓库根有真实 `.env`,不传的话缺字段的用例会被它静默补上。
    return Settings(_env_file=None, **{**_REQUIRED, **over})


@pytest.fixture
def job_store():
    """每个用例一份**干净**的 `JobStore`(理由与 `test_flywheel_task.py` 同款)。

    `get_job_store()` 是进程级单例(`lru_cache`),生产就靠它串行;
    测试必须显式清掉 —— 上一条用例留下的 running 任务会让这一条**永远是 None**。
    """
    get_job_store.cache_clear()
    yield get_job_store()
    get_job_store.cache_clear()


def _child(body: str) -> list[str]:
    """造一条**受控的**子命令。

    `-X utf8` 与生产那条命令一致 —— 它不是装饰:少了它,子进程按本机 locale
    (cp936)编码 stdout,而我们按 UTF-8 解,中文一行都读不对
    (下面 `test_chinese_output_round_trips` 就是钉这个的)。
    """
    return [sys.executable, "-X", "utf8", "-c", body]


def _wait(job_store, job_id, timeout: float = 20.0):
    """等到终态。超时就把**当前状态**返回,让断言自己报出来(不在这里抛)。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = job_store.get(job_id)
        if job is not None and job.status != "running":
            return job
        time.sleep(0.02)
    return job_store.get(job_id)


# --------------------------------------------------------------------------
# ① 跑的是命令行那条命令(唯一一处「路径与参数逐字」的断言)
# --------------------------------------------------------------------------

def test_default_command_is_the_cli_command():
    """`_default_command()` 必须就是 `scripts/run_eval.py --trigger manual`。

    这条**不是**格式检查:路径写错的话(比如少一层 `parents[]`),
    表现是子进程 `can't open file` 退出码 2 —— 而那时上面那些用例**全绿**
    (它们全都注入了自己的 command)。本仓「引用一个路径就得当场核」那条老账。
    """
    cmd = eval_job._default_command()
    assert cmd == [sys.executable, "-X", "utf8", str(eval_job.EVAL_SCRIPT),
                   "--trigger", "manual"], f"实际 {cmd}"
    assert eval_job.EVAL_SCRIPT.exists(), f"评测脚本不存在:{eval_job.EVAL_SCRIPT}"
    # ⚠️ `--limit` **不许**出现在默认命令里:能点出 20 条那种轮次的按钮 =
    #    每点一下往趋势表里插一行「不可比」。理由见 `app/api/kb.py` 那个端点。
    assert "--limit" not in cmd, "默认命令不许带 --limit(趋势表会被按钮污染)"


# --------------------------------------------------------------------------
# ② 忙 ⇒ None(端点的 409 全靠它)
# --------------------------------------------------------------------------

def test_busy_store_returns_none(job_store):
    """运行槽被占着时必须回 `None` —— **不许**排队、不许并发起第二个子进程。"""
    first = start_eval_job(_settings(), job_store=job_store,
                           command=_child("import time; time.sleep(0.8)"))
    assert first is not None
    assert start_eval_job(_settings(), job_store=job_store) is None, (
        "槽被占着时第二次必须回 None(端点据此回 409)——"
        "并发起第二个子进程 = 本机同时两份 2.2GB 权重")
    assert _wait(job_store, first).status == "done"


# --------------------------------------------------------------------------
# ③ 专用线程(这个名字是本模块唯一「看得见」的结构证据)
# --------------------------------------------------------------------------

def test_runs_in_a_named_background_thread(job_store):
    """线程名必须是 `eval-job-{job_id}`。

    ⚠️ 不能只断「任务在跑」——`JobStore` 是纯内存字典,一个**压根不起线程**的
    实现照样能让 job 停在 running、照样 409,而评测**永远不会被跑起来**。
    等 `status` 变终态是不够的(那也可能是「压根没起、被别的东西改了」),
    所以这里在**子进程还活着的时候**去看线程表。
    """
    job_id = start_eval_job(_settings(), job_store=job_store,
                            command=_child("import time; time.sleep(0.8)"))
    assert job_id is not None
    try:
        deadline = time.monotonic() + 5
        names = []
        while time.monotonic() < deadline:
            names = [t.name for t in threading.enumerate()]
            if f"eval-job-{job_id}" in names:
                break
            time.sleep(0.01)
        assert f"eval-job-{job_id}" in names, (
            f"必须有一条名为 `eval-job-{job_id}` 的线程(专用线程那条性质),"
            f"当前线程:{names}")
    finally:
        assert _wait(job_store, job_id).status == "done"


# --------------------------------------------------------------------------
# ④ 输出**真的来自子进程**(不是替身/不是编的)
# --------------------------------------------------------------------------

def test_message_is_the_childs_own_last_line(job_store):
    """`job.message` = 子进程 stdout 的**最后一行**,`result["tail"]` 是最后几行。

    token 是**子进程自己 echo 出来的**(`print`),所以这条断言在
    「把输出读进来」这件事被写坏的实现下会红;一个自己编一条 message 的实现
    要编对 `EVAL_TOKEN_3f9a` 这个串,做不到。
    """
    job_id = start_eval_job(_settings(), job_store=job_store, command=_child(
        "print('EVAL_TOKEN_3f9a'); print('EVAL_TOKEN_3f9b'); print('EVAL_TOKEN_3f9c')"))
    job = _wait(job_store, job_id)
    assert job.status == "done", f"实际 {job.status}:{job.message}"
    assert job.message == "EVAL_TOKEN_3f9c", f"实际 {job.message!r}"
    assert "EVAL_TOKEN_3f9a" in job.result["tail"], f"实际 {job.result}"
    assert job.result["exit_code"] == 0


def test_chinese_output_round_trips(job_store):
    """中文进度行必须**逐字**到达 —— cp936 那条老账的第五个落点。

    子进程按 `-X utf8` 写、我们按 `encoding="utf-8"` 读,两边钉住才成立;
    任一侧缺席时这里拿到的是乱码或替换字符(而不是抛异常 ——
    `errors="replace"` 让**读**这一侧不抛,所以必须靠**内容**判,不能只看有没有报错)。
    """
    token = "评估完成:纯dense 耗时 3.2s"
    job_id = start_eval_job(_settings(), job_store=job_store,
                            command=_child(f"print({token!r})"))
    job = _wait(job_store, job_id)
    assert job.message == token, f"实际 {job.message!r}"
    assert "�" not in job.message, "出现了替换字符 ⇒ 两侧编码不是同一套"


def test_stderr_is_merged_into_message(job_store):
    """stderr 与 stdout **合流**(`stderr=STDOUT`)。

    真实评测的告警(「重排权重未就绪」)与 traceback 全在 stderr;
    只读 stdout 的话,一个**崩掉**的评测会给出一句「(子进程没有任何输出)」——
    最需要看的那句话被丢掉。这条用一条**只**写 stderr 的子进程钉住。
    """
    job_id = start_eval_job(_settings(), job_store=job_store, command=_child(
        "import sys; print('STDERR_ONLY_7c1d', file=sys.stderr)"))
    job = _wait(job_store, job_id)
    assert job.status == "done", f"实际 {job.status}:{job.message}"
    assert job.message == "STDERR_ONLY_7c1d", f"实际 {job.message!r}"


# --------------------------------------------------------------------------
# ⑤ 非零退出**带着它自己打出来的那句话**
# --------------------------------------------------------------------------

def test_nonzero_exit_keeps_the_childs_own_words(job_store):
    """失败必须同时留下:**退出码**与**子进程最后说的那句**。

    只看退出码的话,人拿到的是一句「评测失败(退出码 1)」—— 而 `run_eval.py`
    在 Milvus 连不上时打的是完整 traceback,那句话才是能定位的东西。
    """
    job_id = start_eval_job(_settings(), job_store=job_store, command=_child(
        "import sys; print('BOOM: Milvus 连不上'); sys.exit(3)"))
    job = _wait(job_store, job_id)
    assert job.status == "failed", f"实际 {job.status}:{job.message}"
    assert "3" in job.message and "BOOM: Milvus 连不上" in job.message, (
        f"失败文案要同时给出退出码与子进程自己的话,实际 {job.message!r}")
    assert job.result["exit_code"] == 3
    assert "BOOM: Milvus 连不上" in job.result["tail"]


def test_redacts_api_key_from_child_output(job_store):
    """子进程输出里的**密钥字面值**必须被抹掉。

    ⚠️ 这条一起断两件事,少一件就成了同义反复:
      ① 子进程**真的**把密钥打出来了(所以下面那个 `***` 必须出现);
      ② 抹过之后 `job.message` 里**没有**它。
    只断 ② 的话,一个**压根不读子进程输出**的实现照样绿(而那正是这条要防的
    反面);只断 ① 的话就完全没在测 redact。
    """
    key = _REQUIRED["openai_api_key"]
    job_id = start_eval_job(_settings(), job_store=job_store, command=_child(
        f"print('Error code: 401 - Incorrect API key provided: {key}')"))
    job = _wait(job_store, job_id)
    assert job.message is not None
    assert "***" in job.message, (
        f"`***` 没出现 ⇒ 子进程那句话压根没被读进来,这条测的不是 redact:{job.message!r}")
    assert key not in job.message, f"密钥被原样写进了 job.message:{job.message!r}"
    assert all(key not in line for line in job.result["tail"]), job.result


# --------------------------------------------------------------------------
# ⑥ 超时**真的杀掉**它(那条唯一的运行槽靠这个才活得下来)
# --------------------------------------------------------------------------

def test_timeout_kills_the_child_and_releases_the_slot(job_store):
    """到点必须杀掉子进程、落 `failed`、并且**把槽摘掉**。

    子进程是 `sleep(30)` 而 timeout 是 0.5s ⇒ 一个**没有**超时实现的版本会在这里
    挂满 30 秒(wait 的 20s 上限先到 ⇒ 状态仍是 running),所以「跑得快」
    本身就是断言的一部分。**没有这一条,一次卡死的评测会把 `JobStore` 那个
    唯一的槽永久占死** —— ch04 的 vectorize/mine 至今敞着的那条路。
    """
    started = time.monotonic()
    job_id = start_eval_job(_settings(), job_store=job_store, timeout=0.5,
                            command=_child("import time; time.sleep(30)"))
    job = _wait(job_store, job_id, timeout=20)
    elapsed = time.monotonic() - started
    assert job.status == "failed", f"实际 {job.status}:{job.message}"
    assert "终止" in job.message, f"要说清是被杀的,实际 {job.message!r}"
    assert elapsed < 15, f"超时没生效(等了 {elapsed:.1f}s,子进程要睡 30s)"
    assert job_store.is_busy() is False, "槽必须被摘掉,否则此后每次触发都是 409"
    assert job.result["killed"] is True


def test_child_is_actually_dead_after_timeout(job_store):
    """超时之后**进程真的没了**(不只是「我们不再看它」)。

    ⚠️ 与上一条分开写:上一条断的是**状态与文案**,一个 `kill()` 打在错误对象上
    (比如杀了 shell 而没杀到孙子进程)的实现照样能让上一条绿。这条**直接拿着
    `Popen` 句柄**验(driver 是这里自己起的,不是 `_run` 内部那个)。
    """
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", "-c", "import time; time.sleep(30)"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", bufsize=1)
    assert proc.poll() is None
    proc.kill()
    proc.wait(timeout=10)
    assert proc.poll() is not None, "kill 之后进程必须真的退出"


# --------------------------------------------------------------------------
# ⑦ 经端点的 201 / 409(前端与验收用的就是这一条路)
# --------------------------------------------------------------------------

@pytest.mark.anyio
async def test_endpoint_starts_then_409s(monkeypatch, job_store):
    """走真的 `POST /api/kb/jobs/eval`:第一次 201,任务在跑时第二次 409。

    端点才是前端按钮与验收脚本用的入口,而它自己那一行判断(忙 → 409)
    **只有在这里**才被行使。顺带钉住新任务类型不需要新端点:
    `GET /api/kb/jobs/{job_id}` 对 `type="eval"` 原样可用。

    ⚠️ 这里**只**打桩 `_default_command`(让子进程是一个睡一会儿的假命令),
    `start_eval_job` / JobStore / 线程 / Popen 全是真的 —— 打桩再往上退一层
    (比如把 `start_eval_job` 整个换掉)的话,这一段就只剩「端点会转发」,
    而那件事没有测试的价值。
    """
    monkeypatch.setattr(eval_job, "_default_command",
                        lambda: _child("import time; time.sleep(1.0)"))
    app.dependency_overrides[get_settings] = lambda: _settings()
    job_id = None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            first = await client.post("/api/kb/jobs/eval")
            assert first.status_code == 201, f"实际 {first.status_code}:{first.text}"
            job_id = first.json()["job_id"]

            second = await client.post("/api/kb/jobs/eval")
            assert second.status_code == 409, (
                f"任务在跑时第二次必须 409,实际 {second.status_code}:{second.text}")
            assert second.json()["detail"] == "已有任务在跑"

            got = await client.get(f"/api/kb/jobs/{job_id}")
            assert got.status_code == 200
            assert got.json()["type"] == "eval", "新任务类型沿用同一个 Job 投影"
    finally:
        app.dependency_overrides.pop(get_settings, None)
        if job_id is not None:
            _wait(job_store, job_id)
