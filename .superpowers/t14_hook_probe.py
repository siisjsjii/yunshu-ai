"""T14 探针:证明 `tests/conftest.py` 那个 autouse 装置**是承重的**。

问的问题:`app/agent/nodes.py` 那两处落池钩子在不打桩时**真的会起线程**吗?
(如果不会,conftest 那段守卫就是多余的,而它会掩盖真问题。)

做法:直接调一次闸节点 —— **不经过 pytest、因而没有 conftest 的守卫**。
`_run_flywheel` 被换成**纯 no-op 协程**(不碰库、不建模型、不打网络),
所以本探针是安全的;要观测的只是「线程起没起」与「JobStore 的运行槽有没有被占」。

输出边界钉 UTF-8(`sys.stdout.buffer.write`),本机 locale 是 cp936。
"""

import asyncio
import sys
import threading

sys.path.insert(0, ".")

from app.agent.nodes import make_confidence_gate_node  # noqa: E402
from app.config import Settings  # noqa: E402
from app.flywheel import tasks  # noqa: E402
from app.kb.jobs import get_job_store  # noqa: E402


def emit(line: str) -> None:
    sys.stdout.buffer.write((line + "\n").encode("utf-8"))


class _Session:
    def __init__(self):
        self.added = []

    def add(self, row):
        self.added.append(row)

    async def commit(self):
        pass


def _settings(**over) -> Settings:
    return Settings(
        _env_file=None,
        openai_base_url="http://x", openai_api_key="k", openai_model="m",
        # 假 URL:即便真起线程也不会连到任何真库(探针本身不打网络)
        database_url="mysql://x",
        evidence_confidence_threshold=0.5, **over)


def _ev(score):
    return {"chunk_id": 1, "section_path": "s", "question": "q",
            "answer": "a", "category": "c", "score": score}


async def main() -> None:
    get_job_store.cache_clear()

    # 流水线换成 no-op:探针不碰库、不建模型。**`_spawn` 一个字都不动** ——
    # 要验的正是它。
    async def noop(**kwargs):
        return {"processed": 0, "merged": 0, "created": 0, "failed": 0}

    tasks.run_flywheel = noop

    before = [t.name for t in threading.enumerate() if t.name.startswith("flywheel-job-")]
    session = _Session()
    node = make_confidence_gate_node(
        settings=_settings(), session=session, conversation_id="probe")
    out = await node({"user_input": "猫砂盆多少钱", "evidence": [_ev(0.2)]})

    await asyncio.sleep(0.3)      # 给线程一点时间起来
    after = [t.name for t in threading.enumerate() if t.name.startswith("flywheel-job-")]

    emit("落池了(闸拦下): %s" % (len(session.added) == 1))
    emit("闸的结论: %s" % out["gate_passed"])
    emit("起线程前 flywheel-job-* 线程: %s" % before)
    emit("起线程后 flywheel-job-* 线程: %s" % after)
    emit("JobStore 运行槽被占: %s" % get_job_store().is_busy())
    emit("JobStore 里的任务: %s" % [(j.type, j.status) for j in get_job_store().list()])


asyncio.run(main())
