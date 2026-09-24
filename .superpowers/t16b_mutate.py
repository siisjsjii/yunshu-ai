"""T16b 变异检测:两条新断言各自有没有判别力。

规矩照本仓那套(踩过五次的那几条):
① 锚点**锚在代码上**,不锚在注释上;② 每次变异后**断言命中数恰好为 1**;
③ 看不到 `N passed|failed` 就打印 `!!!`(那是「不要再加 -q」的自动化版本);
④ **绝不把证据输出接进任何截断/过滤管道** —— 全量写文件,再读文件。
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

MUTANTS = [
    {
        "name": "M1 拿掉 llm 的 timeout ⇒ 回到「无上界等待」",
        "file": ROOT / "app" / "llm.py",
        "old": "        timeout=settings.llm_timeout_seconds,\n",
        "new": "",
        "args": ["tests/test_llm.py", "-m", "not db", "-k", "timeout or silent"],
    },
    {
        # ⚠️ 这条锚点**改过一次**:fix round 1 之前它锚的是一整块
        # `try/except TimeoutError` 旧代码,那次改动之后命中数掉成 0 ——
        # 脚本当场报 `!!! 锚点没打上/打重了` 并**拒绝计数**(本仓记过的
        # 「变异后没红有两个互斥解释」那条规矩,靠的就是这个断言)。
        # 换成单行锚点:`asyncio.timeout(None)` 就是「不设上界」。
        "name": "M2 拿掉飞轮的寿命上界 ⇒ 卡住的任务不再有出口",
        "file": ROOT / "app" / "flywheel" / "tasks.py",
        "old": "                timeout_cm = asyncio.timeout(settings.flywheel_job_timeout_seconds)\n",
        "new": "                timeout_cm = asyncio.timeout(None)  # MUTANT: 寿命上界被拿掉\n",
        "args": ["tests/test_flywheel_task.py", "-m", "not db",
                 "-k", "is_killed_by_the_deadline"],
    },
    {
        "name": "M3(fix round 1 / F1)拿掉「把死线原因抢回来」那层守卫",
        "file": ROOT / "app" / "flywheel" / "tasks.py",
        "old": (
            "            if deadline_hit and not isinstance(exc, FlywheelDeadlineExceeded):\n"
            "                raise FlywheelDeadlineExceeded(\n"
            "                    limit=settings.flywheel_job_timeout_seconds) from exc\n"
            "            raise\n"
        ),
        "new": "            raise  # MUTANT: 死线那句话不再被抢回来\n",
        "args": ["tests/test_flywheel_task.py", "-m", "not db", "-k", "survives"],
    },
    {
        "name": "M4(fix round 1)拿掉 `expired()` 收窄",
        "file": ROOT / "app" / "flywheel" / "tasks.py",
        "old": (
            "                    if not timeout_cm.expired():\n"
            "                        raise\n"
            "                    deadline_hit = True\n"
        ),
        "new": "                    deadline_hit = True  # MUTANT: 归因不再收窄\n",
        "args": ["tests/test_flywheel_task.py", "-m", "not db", "-k", "inner_timeout"],
    },
    {
        "name": "M5(fix round 1)文案回到 `:.0f`",
        "file": ROOT / "app" / "flywheel" / "tasks.py",
        "old": 'f"任务超时(上限 {limit:g}s):本轮已放弃。整批一次提交 ⇒ "',
        "new": 'f"任务超时(上限 {limit:.0f}s):本轮已放弃。整批一次提交 ⇒ "',
        "args": ["tests/test_flywheel_task.py", "-m", "not db", "-k", "sub_second"],
    },
]


def main() -> int:
    bad = 0
    for m in MUTANTS:
        path = m["file"]
        original = path.read_text(encoding="utf-8")
        hits = original.count(m["old"])
        print(f"\n=== {m['name']}")
        print(f"    锚点命中数 = {hits}(必须恰好为 1)")
        if hits != 1:
            print("    !!! 锚点没打上/打重了 —— 这次变异不算数,红绿都不可信")
            bad += 1
            continue
        try:
            path.write_text(original.replace(m["old"], m["new"]), encoding="utf-8")
            proc = subprocess.run(
                [PY, "-m", "pytest", *m["args"]],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            out = proc.stdout + proc.stderr
            log = ROOT / ".superpowers" / f"t16b_mutate_{m['file'].stem}.txt"
            log.write_text(out, encoding="utf-8")
            summary = [ln for ln in out.splitlines()
                       if "passed" in ln or "failed" in ln or "error" in ln]
            if not any(("passed" in ln or "failed" in ln) for ln in summary):
                print(f"    !!! 看不到 N passed|failed —— 全量输出见 {log}")
                bad += 1
            else:
                print("    " + " | ".join(summary[-3:]))
                print(f"    (全量输出:{log})")
        finally:
            path.write_text(original, encoding="utf-8")
    print(f"\n变异总数 {len(MUTANTS)},锚点问题 {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
