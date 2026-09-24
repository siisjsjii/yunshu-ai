"""T14 探针(二):在**有** conftest 守卫的条件下跑那三处落池用例,看有没有东西漏出来。

与 `.superpowers/t14_hook_probe.py`(不带守卫 ⇒ 起线程 + JobStore 里多出一条
`flywheel` 任务)配成一对:同一件事,唯一变量是那份守卫。

在**进程内**跑 pytest,跑完直接数两样东西:
  ① `threading.enumerate()` 里有没有 `flywheel-job-*`;
  ② 进程级 `JobStore` 里有没有 `flywheel` 类型的任务。
"""

import sys
import threading

import pytest

sys.path.insert(0, ".")

TARGETS = [
    "-m", "not db",
    "tests/test_agent_gate_ch09.py",
    "tests/test_agent_gate.py",
    "tests/test_agent_protocol.py",
    # ⚠️ `tests/test_flywheel_task.py` **刻意不在**这张表里:它直接调
    # `start_flywheel_job`,本来就会起真线程(那正是它要测的东西)。混进来
    # 会把「守卫漏了」与「T14 自己在测线程」两件事搅成一个观测。
    "-p", "no:cacheprovider",
]


def emit(line: str) -> None:
    sys.stdout.buffer.write((line + "\n").encode("utf-8"))


rc = pytest.main(TARGETS)

from app.kb.jobs import get_job_store  # noqa: E402  (pytest 跑完再 import 也一样)

threads = [t.name for t in threading.enumerate() if t.name.startswith("flywheel-job-")]
jobs = [(j.type, j.status) for j in get_job_store().list()]

emit("pytest 退出码: %s" % rc)
emit("跑完后 flywheel-job-* 线程: %s" % threads)
emit("跑完后 JobStore 里的任务: %s" % jobs)
emit("结论(两样都空 = 守卫生效): %s" % (not threads and not jobs))
