"""变异探针:`app/kb/eval_job.py` 的四处「收口」是不是**真的被断言守着**。

四条变异 **全是产品级**(改的是被测对象,不是测试),每条对应一道收口:

  M1 `encoding="utf-8"` → `cp936`   ⇒ `test_chinese_output_round_trips` 必须红
  M2 `stderr=STDOUT`   → `DEVNULL`  ⇒ `test_stderr_is_merged_into_message` 必须红
  M3 `proc.kill()`     → `pass`     ⇒ `test_timeout_kills_the_child_and_releases_the_slot` 必须红
  M4 去掉 `redact_api_key(...)`      ⇒ `test_redacts_api_key_from_child_output` 必须红

规矩(本仓用事故换来的那几条,一条都不省):
  · **字节级**读写(绝不用 `write_text`:Windows 会把 `\\n` 翻成 `\\r\\n`);
  · 每次变异**断言锚点命中数恰好为 1**(0 ⇒ 变异压根没生效,2 ⇒ 改错了地方);
  · **报告红在哪一条** —— 红在无关的用例上等于零判别力;
  · `finally` 里**字节还原**,收工再核 sha256 与开工相同;
  · 每条子进程带 **timeout**(被 SIGKILL 过三次,留下过没还原的变异);
  · 看不到 `N passed|failed` 就打印 `!!!`(那是「不要再加 -q」的自动化版本)。
"""

import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "app" / "kb" / "eval_job.py"
PY = REPO / ".venv" / "Scripts" / "python.exe"

MUTATIONS = [
    ("M1 编码", b'encoding="utf-8",',
     b'encoding="cp936",',
     "tests/test_eval_job.py::test_chinese_output_round_trips"),
    ("M2 stderr", b"stderr=subprocess.STDOUT,",
     b"stderr=subprocess.DEVNULL,",
     "tests/test_eval_job.py::test_stderr_is_merged_into_message"),
    ("M3 杀进程", b"            proc.kill()",
     b"            pass  # KILLED",
     "tests/test_eval_job.py::test_timeout_kills_the_child_and_releases_the_slot"),
    ("M4 脱敏", b"text = redact_api_key(line.strip(), api_key)",
     b"text = line.strip()",
     "tests/test_eval_job.py::test_redacts_api_key_from_child_output"),
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def main() -> int:
    original = TARGET.read_bytes()
    print(f"开工 sha256[:16] = {sha(original)}  ({TARGET})")
    failures = []
    try:
        for name, old, new, node in MUTATIONS:
            hits = original.count(old)
            print(f"\n=== {name} ===  锚点命中 {hits} 次")
            if hits != 1:
                failures.append(f"{name}: 锚点命中 {hits} 次(要求恰好 1)—— 变异没生效")
                print("   !!! 变异没生效,这一条不作数")
                continue
            TARGET.write_bytes(original.replace(old, new, 1))
            try:
                proc = subprocess.run(
                    [str(PY), "-m", "pytest", node, "-p", "no:cacheprovider"],
                    cwd=str(REPO), capture_output=True, timeout=300,
                    encoding="utf-8", errors="replace")
            except subprocess.TimeoutExpired:
                failures.append(f"{name}: 子进程超时 —— 不作数")
                print("   !!! 超时")
                continue
            finally:
                TARGET.write_bytes(original)          # ← 先还原,再看输出
            out = (proc.stdout or "") + (proc.stderr or "")
            tail = out.strip().splitlines()[-3:]
            print("   " + "\n   ".join(tail))
            if not any(("passed" in ln or "failed" in ln or "error" in ln.lower())
                       for ln in out.splitlines()):
                failures.append(f"{name}: 看不到 N passed|failed —— 不作数")
                print("   !!! 看不到计数行,这一条不作数")
                continue
            if proc.returncode == 0:
                failures.append(f"{name}: **没红** —— 那条断言对它零判别力")
                print("   XX 没红(**这条收口今天没人守**)")
            else:
                print(f"   OK 按预期变红(退出码 {proc.returncode})")
    finally:
        TARGET.write_bytes(original)
    after = TARGET.read_bytes()
    print(f"\n收工 sha256[:16] = {sha(after)}")
    if after != original:
        print("!!! 源码没有字节还原 —— 立即停下")
        return 2
    print("源码字节还原:一致")
    if failures:
        print("\n有问题的变异:")
        for f in failures:
            print("  -", f)
        return 1
    print("四条变异全部按预期变红。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
