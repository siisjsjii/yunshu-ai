"""评测后台任务(ch10 跟进):把 `scripts/run_eval.py` 跑成**子进程**。

工作台「评测」页那个「运行评测」按钮打的就是这里(`POST /api/kb/jobs/eval`)。
它存在的理由只有一句:让「跑一轮评估」这件事**不必离开浏览器**去开一个终端 ——
而**评测逻辑一行都不在这里**,全部在 `scripts/run_eval.py` 里。

## 为什么是子进程,而不是照 `app/kb/orchestrate.py` 那样「线程 + 自建 engine」

`scripts/run_eval.py` 的两处落库(`load_chunks` / `_record_eval_run`)与它的读库
**写死用 `app/db/base.py` 的 `get_sessionmaker()` 单例**。那条单例绑在**首次使用它的
事件循环**上(客服服务的主循环),而后台线程里是 `asyncio.run` 的**另一个**循环 ——
跨循环复用同一条连接池正是 ch04(`orchestrate.py`)与 ch07(`memory/tasks.py`)
各记过一次的故障。要在进程内跑,得给那个脚本开一个 engine 注入口(改三处签名,
并牵动它自己的单测);子进程**根本不存在这条缝**:它有自己的进程、自己的循环、
自己的 engine、末尾自己 `dispose`。

第二条理由是**代价**,方向相反但同样重要:进程内跑会与聊天检索**共用**同一个
BGE-M3 / 重排器实例(两者都是 `lru_cache` 单例),看着更省内存;但那意味着评测的每一批
前向都要与真实用户请求抢那把 encode 锁(ch04 的 `_encode_lock`)—— 于是「在工作台上
点一下评测」的表现是「聊天同时变慢」。子进程各跑各的,代价是同一时刻多驻留一份权重
(**它必须与向量化/挖知识/飞轮共用那一个 `JobStore` 运行槽**,这也是原因之一:
那四条路任意两条同时起,本机就会同时扛两份 2.2GB)。

## 它与命令行是**同一条命令**

命令行跑的是 `scripts/run_eval.py --trigger manual`;这里跑的**逐字同一条**
(`sys.executable -X utf8`,cwd = 仓库根)。所以这个文件里没有任何评测口径 ——
`evaluate_cases` / `run_round` / `latest.json` 的形状全在那一份里。

`--trigger` 取 `manual`(与命令行一致)。**刻意不新增第三种取值**:
`app/db/models.py` 与 DDL 的注释写着 `manual|scheduled`,新增会让那两处变假,
而今天**没有任何读方**需要区分「命令行点的」与「按钮点的」。`--limit` 同理**不接**
(理由写在 `app/api/kb.py` 那个端点的 docstring 里)。

## 三道收口

1. **有界**:`eval_job_timeout_seconds` 到点 `kill()`,job 落 `failed` 并**说清**
   是被杀的。没有这一条,一次卡死的评测会把那个唯一的槽**永久**占死。
2. **故障可见**:子进程的 stdout 与 stderr **合流**(`stderr=STDOUT`)逐行读进
   `job.message`(只留最后一行),失败时把**最后几行**留在 `result["tail"]` ——
   「跑挂了」必须带着**它自己打出来的那句话**,不能只留一个退出码。
3. **出站文本过 sanitize**:进 `job.message` 的每一行都过 `redact_api_key`
   —— 子进程的 traceback 里可能带上游响应体原文(本仓那条「`str(exc)` 可能是
   上游响应体」的老账)。

⚠️ **已经跑进 `eval_runs` 的那些行业务上不可撤销** —— 与 `vectorize` / `mine` 不同,
这个任务**没有幂等重跑**这一说:每跑成一轮就多一行。`eval_trend.py` 会把它摆进趋势表,
所以「点一下按钮」的代价是**趋势表里多一轮**,这不是缺陷、是它的语义。
"""

import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from app.config import Settings
from app.kb.jobs import JobStore, get_job_store
from app.sanitize import redact_api_key

#: 仓库根 = `app/kb/eval_job.py` 往上数三层。cwd 钉在它上面,免得子进程的相对路径
#: 取决于**服务是从哪个目录起的**(uvicorn 从仓库根起是惯例,不是保证)。
REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_eval.py"

#: 失败时留在 `result["tail"]` 里的行数。只留**最后几行** —— 一份评测日志可以很长,
#: 而人要看的是「它挂在哪一句」。
TAIL_LINES = 5

#: `job.message` 里只放**最后一行**的截断版本(那个字段会被前端整行渲染)。
MAX_MESSAGE_CHARS = 300


def _default_command() -> list[str]:
    """命令行等价物。

    `-X utf8` 不是装饰:本机 locale 是 **cp936**,子进程若按它编码 stdout,
    那些中文进度行会以 GBK 字节出来,而我们按 UTF-8 解 —— 一屏乱码(本仓
    cp936 家族的第五次)。子进程自己打印非 ASCII 时的输出边界由它钉住。
    """
    return [sys.executable, "-X", "utf8", str(EVAL_SCRIPT), "--trigger", "manual"]


def _run(job_store: JobStore, job_id: str, settings: Settings,
         command: list[str], timeout: float) -> None:
    """**在工作线程里**跑评测子进程并把状态写回 `job_store`。永不抛(异常自己收)。

    `command` / `timeout` 是给测试留的缝:子进程那条路只有拿一个**受控的**子命令
    才测得了(真评测要 Milvus + 真实 key + 几分钟),而这里要验的四件事
    —— 线程真的起了、输出**真的来自子进程**、非零退出**带着它自己的话**、
    到点**真的被杀掉** —— 每一件都必须在这个函数的真身上跑。
    """
    api_key = settings.openai_api_key
    tail: deque[str] = deque(maxlen=TAIL_LINES)

    proc = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,       # 合流:`evaluate_cases` 的告警与 traceback 都在 stderr
        text=True,
        encoding="utf-8",               # 与子进程的 `-X utf8` 成对
        errors="replace",               # 认不出的字节也别让**读**这一侧抛
        bufsize=1,                      # 行缓冲:进度行要**边跑边**进 job.message
    )

    def reader() -> None:
        # 逐行读的另一半理由:输出量可能远超 `job.message` 能承载的长度,
        # 攒到最后一次性读(= `communicate`)既看不到进度、也可能吃到一大块内存。
        assert proc.stdout is not None
        for line in proc.stdout:
            text = redact_api_key(line.strip(), api_key)
            if not text:
                continue
            tail.append(text)
            job_store.update(job_id, message=text[:MAX_MESSAGE_CHARS])

    reader_thread = threading.Thread(
        target=reader, name=f"eval-job-{job_id}-out", daemon=True)
    reader_thread.start()

    killed = False
    deadline = time.monotonic() + timeout
    while proc.poll() is None:
        if time.monotonic() >= deadline:
            killed = True
            proc.kill()
            break
        time.sleep(0.2)

    try:
        proc.wait(timeout=30)           # kill 之后收尸;正常路径上它早就退了
    except subprocess.TimeoutExpired:    # pragma: no cover - 收尸都收不掉,只能放弃
        job_store.update(
            job_id, status="failed", result={"tail": list(tail)},
            message="⚠ 评测子进程在多次终止之后仍未退出,已放弃跟踪(槽已释放)")
        return

    reader_thread.join(timeout=10)       # 让最后几行进 tail/结果
    lines = list(tail)

    if killed:
        job_store.update(
            job_id, status="failed", result={"killed": True, "tail": lines},
            message=f"⚠ 超过 {int(timeout)} 秒仍未结束,已终止该子进程。"
                    f"这一轮**不会**写 latest.json、也**没有**往 eval_runs 落行"
                    f"(评估算完才落库)。")
        return

    code = proc.returncode
    if code == 0:
        job_store.update(
            job_id, status="done", result={"exit_code": 0, "tail": lines},
            message=lines[-1] if lines else "✓ 评测完成(子进程没有输出)")
        return

    job_store.update(
        job_id, status="failed", result={"exit_code": code, "tail": lines},
        message=f"⚠ 评测失败(退出码 {code}):"
                + (lines[-1] if lines else "(子进程没有任何输出)"))


def start_eval_job(settings: Settings, *, job_store: JobStore | None = None,
                   command: list[str] | None = None,
                   timeout: float | None = None) -> str | None:
    """起一轮评测;**忙**(已有任务在跑)时返回 `None`。

    形状与 `app/kb/orchestrate.py:start_job` 逐字一致:同一个 `JobStore`、
    同一条「忙 ⇒ None」(端点把它翻成 409)、同一种「专用线程 + 名字里带 job id」。
    与 `_spawn` 的差别只有一处:这里**没有** `async` 可跑 —— 活是子进程干的,
    线程只负责看着它。

    `command` / `timeout` 缺省时取真命令与 `settings.eval_job_timeout_seconds`。
    """
    store = job_store if job_store is not None else get_job_store()
    job = store.start("eval")
    if job is None:
        return None

    cmd = list(command) if command is not None else _default_command()
    limit = settings.eval_job_timeout_seconds if timeout is None else timeout

    def target() -> None:
        try:
            _run(store, job.id, settings, cmd, limit)
        except Exception as exc:
            # 线程级意外(起进程本身就失败 —— 比如 `sys.executable` 不可用)。
            # 必须落终态:漏掉的话 job 永远停在 running,而那个槽是本进程**唯一**的
            # 一个(`JobStore` 的 `_running_id`)。文案过 sanitize,与 `_spawn` 同款。
            store.update(job.id, status="failed",
                         message=redact_api_key(str(exc), settings.openai_api_key))

    threading.Thread(target=target, name=f"eval-job-{job.id}", daemon=True).start()
    return job.id
