"""T17 变异验证(证据文件,随任务提交):把每条变异**逐条**打进源码、跑用例、看红不红。

规则照本仓的教训来:
- 锚点**锚在代码上,不锚在注释上**;
- 每次变异后**断言命中数恰好为 1**(打不上锚点 = 「变异压根没生效」,与「用例没判别力」是两回事);
- **不把输出接进任何截断/过滤管道** —— 整个 pytest 输出原样打出来;
- 看不到 `passed|failed` 就打 `!!!`。

**fix round 1 的变化**:新增 `tests/test_eval_trend.py` 之后,M4(差值符号取反)
从「只有输出变了、没有红」变成**有测试的红**。M5/M6 是这一轮新增的渲染变异。
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

# 本机 locale 是 cp936:本脚本自己打中文也要钉输出边界,否则重定向到文件后是乱码。
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv/Scripts/python.exe")

WRITE_TESTS = "tests/test_eval_runs_write.py"
TREND_TESTS = "tests/test_eval_trend.py"

MUTATIONS = [
    (
        "M1 顶层 case_count 取全量而不是实际跑的那批",
        "scripts/run_eval.py",
        "        case_count=len(cases),",
        "        case_count=len(load_cases()),",
        f"{WRITE_TESTS}::test_eval_run_row_carries_case_count_separately",
    ),
    (
        "M2 metrics 里的 case_count 取全量",
        "scripts/run_eval.py",
        '        metrics={"top_k": top_k, "case_count": len(cases),',
        '        metrics={"top_k": top_k, "case_count": len(load_cases()),',
        f"{WRITE_TESTS}::test_eval_run_row_carries_case_count_separately",
    ),
    (
        "M3 先落库、再评估(半途而废的一轮被记成完整的一轮)",
        "scripts/run_eval.py",
        "    results = await evaluate_cases(cases, top_k, ctx)\n    if on_results is not None:",
        "    await _record_eval_run(trigger_by=trigger_by, case_count=len(cases),\n"
        '                           metrics={"top_k": top_k})\n'
        "    results = await evaluate_cases(cases, top_k, ctx)\n    if on_results is not None:",
        f"{WRITE_TESTS}::test_failed_round_writes_no_row",
    ),
    (
        "M4 差值符号取反(表上两个数字手算与箭头对不上)",
        "scripts/eval_trend.py",
        '    return f"{ARROW_DOWN if d < 0 else ARROW_UP} {d:+.3f}"',
        '    return f"{ARROW_DOWN if d < 0 else ARROW_UP} {-d:+.3f}"',
        TREND_TESTS,
    ),
    (
        "M5 删掉「不可比」分支(不同规模的轮次之间照样打箭头)",
        "scripts/eval_trend.py",
        '    return "、".join(parts) if parts else None',
        "    return None  # MUTANT",
        TREND_TESTS,
    ),
    (
        "M6 分桶表不再默认渲染(只剩一个可选开关背后的东西)",
        "scripts/eval_trend.py",
        "    lines += _bucket_lines(views, names, buckets)",
        "    pass  # MUTANT",
        TREND_TESTS,
    ),
]


def _run(cmd):
    # 子进程的 stdout 是管道,Python 会按 locale(cp936)编码 —— 不钉的话
    # 父进程按 utf-8 解码会 UnicodeDecodeError(本仓记过的跨进程编码陷阱)。
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)


def main():
    print("=== 变异前的基线(两条用例文件各自先跑一遍,确认全绿)===")
    for nodeid in (WRITE_TESTS, TREND_TESTS):
        out = _run([PY, "-m", "pytest", nodeid])
        last = [ln for ln in (out.stdout + out.stderr).splitlines() if ln.strip()][-1:]
        print(f"{nodeid}: {last}  returncode={out.returncode}")
        if out.returncode != 0:
            print("!!! 基线就红了,先修再谈变异")
            return 1

    for label, path, old, new, nodeid in MUTATIONS:
        p = ROOT / path
        src = p.read_text(encoding="utf-8")
        hits = src.count(old)
        print("=" * 78)
        print(f"### {label}")
        print(f"锚点命中数 = {hits}(必须恰好 1)")
        if hits != 1:
            print("!!! 锚点没打上(或打多了)—— 这次变异**没有生效**,"
                  "不能当成「用例没判别力」")
            continue
        p.write_text(src.replace(old, new), encoding="utf-8")
        try:
            out = _run([PY, "-m", "pytest", nodeid])
            print(out.stdout)
            print(out.stderr)
            merged = out.stdout + out.stderr
            if "passed" not in merged and "failed" not in merged:
                print("!!! 输出里没有 passed|failed —— 这次运行不算数")
                continue
            failed = [ln for ln in merged.splitlines()
                      if ln.startswith("FAILED") or ln.startswith("failed")]
            print("### 判定:", "RED(用例红了)" if out.returncode != 0
                  else "!!! GREEN(用例没红 —— 断言没有判别力)")
            for ln in failed:
                print("    ", ln)
        finally:
            p.write_text(src, encoding="utf-8")
            assert (ROOT / path).read_text(encoding="utf-8") == src, "还原失败!"
    print("=" * 78)
    print("sha256(scripts/eval_trend.py) =",
          hashlib.sha256((ROOT / "scripts/eval_trend.py").read_bytes()).hexdigest()[:12])


if __name__ == "__main__":
    sys.exit(main())
