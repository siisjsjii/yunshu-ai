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
        "name": "M2 拿掉飞轮的寿命上界 ⇒ 卡住的任务不再有出口",
        "file": ROOT / "app" / "flywheel" / "tasks.py",
        "old": (
            "            try:\n"
            "                async with asyncio.timeout(settings.flywheel_job_timeout_seconds):\n"
            "                    result = await run_flywheel(\n"
            "                        session=session, model=model,\n"
            "                        batch_size=settings.flywheel_batch_size)\n"
            "            except TimeoutError as exc:\n"
            "                raise FlywheelDeadlineExceeded(\n"
            "                    limit=settings.flywheel_job_timeout_seconds) from exc\n"
        ),
        "new": (
            "            result = await run_flywheel(  # MUTANT: 拿掉寿命上界\n"
            "                session=session, model=model,\n"
            "                batch_size=settings.flywheel_batch_size)\n"
        ),
        "args": ["tests/test_flywheel_task.py", "-m", "not db", "-k", "deadline"],
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
