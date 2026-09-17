"""内存任务注册表(ch04):后台任务的状态与串行控制。

向量化 / 挖知识这类耗时操作跑在后台线程,前端轮询这里的 Job 状态。
同时只允许一个 running 任务 —— 忙时新任务被拒,天然避免两份 2.2GB
权重常驻与 Milvus 并发写。进程重启即失(与 SessionStore 同模式,
管理/演示工具,不持久化)。
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class Job:
    id: str
    type: str                    # "vectorize" | "mine"
    status: str                  # "running" | "done" | "failed"
    message: str = ""
    result: dict | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._running_id: str | None = None

    def start(self, job_type: str) -> Job | None:
        """起一个任务;忙(已有 running)时返回 None。"""
        with self._lock:
            if self._running_id is not None:
                return None
            job = Job(id=uuid.uuid4().hex[:12], type=job_type, status="running")
            self._jobs[job.id] = job
            self._running_id = job.id
            return job

    def update(self, job_id: str, *, status=None, message=None, result=None) -> None:
        """更新状态/消息/结果。进入终态(done/failed)即释放运行槽。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if status is not None:
                job.status = status
            if message is not None:
                job.message = message
            if result is not None:
                job.result = result
            if job.status in ("done", "failed"):
                job.finished_at = time.time()
                if self._running_id == job_id:
                    self._running_id = None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def is_busy(self) -> bool:
        with self._lock:
            return self._running_id is not None


@lru_cache(maxsize=1)
def get_job_store() -> JobStore:
    return JobStore()
