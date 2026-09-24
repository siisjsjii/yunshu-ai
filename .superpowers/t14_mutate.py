"""T14 变异验证:每条关键断言**真的会红**吗?

本仓的元教训(ch07):「变异后没红」有两个互斥的解释 —— 断言无判别力,或
**变异压根没生效**。所以每一步都:
  ① 锚点**锚在代码上、不锚在注释上**;
  ② 替换前断言**命中数恰好为 1**(不是 1 就打印 `!!!` 并跳过,不猜);
  ③ 输出里看不到 `N passed|failed` 就打印 `!!!`(那是「不要再加 -q」那条的自动化);
  ④ **不把输出接进任何截断/过滤管道**;
  ⑤ 每步跑完**立刻还原**,并断言还原成功(文件字节数回到原值)。

用法:`.venv/Scripts/python.exe .superpowers/t14_mutate.py`
"""

import re
import subprocess
import sys
from pathlib import Path

# ⚠️ 本机 locale 是 cp936:脚本自己打印非 ASCII(`⇒`、中文)会 UnicodeEncodeError。
# 输出边界钉死成 UTF-8(CLAUDE.md 的平台陷阱那条,已在本仓复发过三次)。
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

SUMMARY = re.compile(r"(\d+) (passed|failed)")


def nl_of(path: Path) -> str:
    data = path.read_bytes()
    return "\r\n" if b"\r\n" in data else "\n"


def read(path: Path) -> str:
    return path.open("r", encoding="utf-8", newline="").read()


def write(path: Path, text: str) -> None:
    path.open("w", encoding="utf-8", newline="").write(text)


MUTATIONS = [
    dict(
        name="M1 掐掉 dispose(三处「一个字都不能省」的第 3 条)",
        path="app/flywheel/tasks.py",
        old=["        await engine.dispose()"],
        new=["        pass  # MUTANT: dispose 被掐掉"],
        tests=["tests/test_flywheel_task.py::test_real_engine_is_built_from_settings_and_disposed",
               "tests/test_flywheel_task.py::test_job_completes_and_clears_the_running_flag"],
        expect="RED",
    ),
    dict(
        name="M2 dispose 不 await(协程造出来就丢)",
        path="app/flywheel/tasks.py",
        old=["        await engine.dispose()"],
        new=["        engine.dispose()  # MUTANT: 漏 await"],
        tests=["tests/test_flywheel_task.py::test_real_engine_is_built_from_settings_and_disposed"],
        expect="RED",
    ),
    dict(
        name="M3 复用 get_engine() 的 lru_cache 单例(ch04 记过的跨循环问题)",
        path="app/flywheel/tasks.py",
        old=["    engine = create_async_engine(settings.database_url, pool_pre_ping=True)"],
        new=["    from app.db.base import get_engine",
             "    engine = get_engine()  # MUTANT: 复用单例"],
        tests=["tests/test_flywheel_task.py::test_real_engine_is_built_from_settings_and_disposed"],
        expect="RED",
    ),
    dict(
        name="M4 拿掉 finally 里的终态兜底(协程静默返回 ⇒ 槽永不释放)",
        path="app/flywheel/tasks.py",
        old=["        latest = job_store.get(job_id)",
             "        if latest is not None and latest.status == \"running\":",
             "            job_store.update(job_id, status=\"failed\", message=\"任务异常结束(未进终态)\")",
             "        await engine.dispose()"],
        new=["        # MUTANT: 终态兜底被拿掉",
             "        await engine.dispose()"],
        tests=["tests/test_flywheel_task.py::test_base_exception_path_also_lands_in_a_terminal_status"],
        expect="RED",
    ),
    dict(
        name="M5 批大小写死 10(不读 settings)",
        path="app/flywheel/tasks.py",
        old=["                batch_size=settings.flywheel_batch_size)"],
        new=["                batch_size=10)  # MUTANT: 写死"],
        tests=["tests/test_flywheel_task.py::test_job_completes_and_clears_the_running_flag"],
        expect="RED",
    ),
    dict(
        name="M5b 拿掉 `_spawn` 兜底里「不许把 done 改成 failed」的守卫",
        path="app/flywheel/tasks.py",
        old=["            latest = job_store.get(job.id)",
             "            if latest is not None and latest.status == \"done\":",
             "                return",
             "            job_store.update("],
        new=["            job_store.update("],
        tests=["tests/test_flywheel_task.py::test_spawn_fallback_never_downgrades_a_done_job"],
        expect="RED",
    ),
    dict(
        name="M6 闸那处钩子挪到 `if not passed` 之外(过闸也起飞轮)",
        path="app/agent/nodes.py",
        old=["        if not passed:"],
        new=["        start_flywheel_job_safely(settings)  # MUTANT: 挪到判断之外",
             "        if not passed:"],
        tests=["tests/test_flywheel_task.py::test_gate_pass_does_not_fire_the_flywheel"],
        expect="RED",
    ),
    dict(
        name="M7 闸那处传一个**别的** settings 对象(model_copy)",
        path="app/agent/nodes.py",
        # ⚠️ 锚点带上**行边界**:20 空格那一条(自评)里**包含**12 空格那一条作为
        # 子串,不夹住整行的话会命中 2 次(本仓记过的「锚点打在另一处」)。
        old=["", "            start_flywheel_job_safely(settings)", ""],
        new=["", "            start_flywheel_job_safely(settings.model_copy())  # MUTANT", ""],
        tests=["tests/test_flywheel_task.py::test_gate_pooling_fires_the_flywheel_once_with_this_settings"],
        expect="RED",
    ),
    dict(
        name="M8 拿掉 👎 那处钩子(R4 的入口 ③)",
        path="app/api/feedback.py",
        old=["    start_flywheel_job_safely(settings)",
             "",
             "    return {\"ok\": True, \"pooled\": True, \"snapshot_chunks\": len(snapshot or [])}"],
        new=["    # MUTANT: 👎 的钩子被拿掉",
             "    return {\"ok\": True, \"pooled\": True, \"snapshot_chunks\": len(snapshot or [])}"],
        tests=["tests/test_api_feedback.py::test_down_writes_one_row_with_user_feedback_entry_point"],
        expect="RED",
    ),
    dict(
        name="M9 拿掉自评那处钩子(入口 ②)",
        path="app/agent/nodes.py",
        old=["                    start_flywheel_job_safely(settings)"],
        new=["                    pass  # MUTANT: 自评的钩子被拿掉"],
        tests=["tests/test_agent_protocol.py::test_useful_false_falls_back_and_records_one_pool_row"],
        expect="RED",
    ),
    dict(
        name="M10 让端点不再 409(忙时照样返回一个 job_id)",
        path="app/api/kb.py",
        old=["    job_id = start_flywheel_job(settings=settings)",
             "    if job_id is None:",
             "        raise HTTPException(status_code=409, detail=\"已有任务在跑\")"],
        new=["    job_id = start_flywheel_job(settings=settings)  # MUTANT: 409 被拿掉",
             "    job_id = job_id or \"fake\""],
        tests=["tests/test_flywheel_task.py::test_second_job_while_running_gets_409"],
        expect="RED",
    ),
    dict(
        name="M11 守卫失效:autouse 装置不再生效(conftest)",
        path="tests/conftest.py",
        old=["@pytest.fixture(autouse=True)",
             "def _no_flywheel_threads(flywheel_hooks):"],
        new=["@pytest.fixture(autouse=False)  # MUTANT: 不再自动生效",
             "def _no_flywheel_threads(flywheel_hooks):"],
        # ⚠️ 这一条的**判据不是 pytest 红不红** —— 守卫失效时那些用例照样全绿
        # (落池本身没坏),坏的是「单测白白起真线程 + 占进程级 JobStore 运行槽」。
        # 所以期望写成 GREEN,真正的证据是下面那条探针。
        tests=["tests/test_agent_gate_ch09.py"],
        expect="GREEN",
        probe=".superpowers/t14_guard_probe.py",
        probe_must_contain="结论(两样都空 = 守卫生效): False",
    ),
]


def run(cmd: list[str]) -> tuple[int, str]:
    # ⚠️ `-X utf8`:子进程的 stdout 是管道 ⇒ 它会用 locale(cp936)编码,而 pytest
    # 的输出里有中文 —— 不钉编码的话子进程自己就 UnicodeEncodeError(本仓记过:
    # 「跨进程测试给子进程加 `-X utf8`」)。
    proc = subprocess.run([PY, "-X", "utf8", *cmd], cwd=ROOT, capture_output=True)
    out = proc.stdout.decode("utf-8", errors="replace") + proc.stderr.decode("utf-8", errors="replace")
    return proc.returncode, out


def verdict(rc: int, out: str, expect: str) -> str:
    m = SUMMARY.search(out)
    if m is None:
        return "!!! 看不到 N passed|failed(输出可能被截断/被过滤)"
    red = rc != 0
    ok = (red if expect == "RED" else not red)
    return ("OK  " if ok else "!!! ") + out.strip().splitlines()[-1]


def main() -> None:
    failures = 0
    for mut in MUTATIONS:
        path = ROOT / mut["path"]
        nl = nl_of(path)
        original = read(path)
        old = nl.join(mut["old"])
        new = nl.join(mut["new"])
        hits = original.count(old)
        if hits != 1:
            print(f"!!! {mut['name']}: 锚点命中 {hits} 次(必须恰好 1),**跳过**")
            failures += 1
            continue
        write(path, original.replace(old, new, 1))
        try:
            rc, out = run(["-m", "pytest", *mut["tests"], "-p", "no:cacheprovider"])
            line = verdict(rc, out, mut["expect"])
            print(f"{mut['name']}\n    {line}")
            if line.startswith("!!!"):
                failures += 1
            if mut.get("probe"):
                rc2, out2 = run([mut["probe"]])
                print("    守卫探针:" + out2.strip().replace("\n", " | "))
                want = mut.get("probe_must_contain")
                if want and want not in out2:
                    print(f"!!! 探针没说出预期的那句:{want}")
                    failures += 1
        finally:
            write(path, original)
            assert read(path) == original, "还原失败!"
    print(f"\n变异步数={len(MUTATIONS)} 有问题={failures}")


main()
