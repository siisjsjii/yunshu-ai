"""变异探针:`tests/test_static_wiring.py` 的四条是不是**真的会红**。

这一份测试全是**源码级**断言,而源码级断言最容易写成「恒真」——
把 HTML/JS 改坏之后它必须红,否则它对「名字对不上」那一族零判别力。

四条变异(改的都是**被测的静态页**,不是测试):
  W1 JS 里一个字面量 id 写错        ⇒ `test_every_referenced_id_exists` 必须红
  W2 `TAB_NAMES` 里删掉「首页」      ⇒ `test_tabs_and_panels_match` 必须红
  W3 一个 `data-goto` 指向不存在的页  ⇒ `test_home_goto_buttons…` 必须红
  W4 首页某张卡的 HTML id 拼错        ⇒ `test_home_cards_wire_up…` 必须红

规矩与 `ch10f_evaljob_mutation_probe.py` 同款(字节读写 / 锚点命中数 == 1 /
报告红在哪一条 / finally 字节还原 + 核 sha256 / 每条带 timeout / 没有计数行就 `!!!`)。
"""

import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ADMIN = REPO / "app" / "static" / "admin.html"
PY = REPO / ".venv" / "Scripts" / "python.exe"

#: (名字, 目标文件, 锚点字节, 替换字节, 该红的 node id, 命中数要求)
MUTATIONS = [
    ("W1 字面量 id", ADMIN,
     'const box = $("topic-bars");', 'const box = $("topic-barss");',
     "tests/test_static_wiring.py::test_every_referenced_id_exists", 1),
    ("W2 TAB_NAMES 少一个", ADMIN,
     'const TAB_NAMES = ["首页", "入库", "待审", "评测", "主题分布", "链路"];',
     'const TAB_NAMES = ["入库", "待审", "评测", "主题分布", "链路"];',
     "tests/test_static_wiring.py::test_tabs_and_panels_match", 1),
    ("W3 data-goto 打错", ADMIN,
     '<button class="home-go" data-goto="入库" type="button">进入入库</button>',
     '<button class="home-go" data-goto="入库页" type="button">进入入库</button>',
     "tests/test_static_wiring.py::test_home_goto_buttons_point_at_real_tabs", 1),
    ("W4 首页卡的 id 拼错", ADMIN,
     'id="home-tp-line"', 'id="home-tpp-line"',
     "tests/test_static_wiring.py::test_home_cards_wire_up_to_existing_elements", 1),
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def main() -> int:
    originals = {p: p.read_bytes() for p in {m[1] for m in MUTATIONS}}
    for p, d in originals.items():
        print(f"开工 sha256[:16] = {sha(d)}  ({p.name})")
    failures = []
    try:
        for name, target, old, new, node, want in MUTATIONS:
            src = originals[target]
            hits = src.count(old.encode("utf-8"))
            print(f"\n=== {name} ===  锚点命中 {hits} 次")
            if hits != want:
                failures.append(f"{name}: 锚点命中 {hits} 次(要求 {want})—— 变异没生效")
                print("   !!! 变异没生效,这一条不作数")
                continue
            target.write_bytes(src.replace(old.encode("utf-8"), new.encode("utf-8"), 1))
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
                target.write_bytes(originals[target])
            out = (proc.stdout or "") + (proc.stderr or "")
            print("   " + "\n   ".join(out.strip().splitlines()[-3:]))
            if not any(("passed" in ln or "failed" in ln or "error" in ln.lower())
                       for ln in out.splitlines()):
                failures.append(f"{name}: 看不到 N passed|failed —— 不作数")
                print("   !!! 看不到计数行,这一条不作数")
                continue
            if proc.returncode == 0:
                failures.append(f"{name}: **没红** —— 那条断言对它零判别力")
                print("   XX 没红(**这条接线今天没人守**)")
            else:
                print(f"   OK 按预期变红(退出码 {proc.returncode})")
    finally:
        for p, d in originals.items():
            p.write_bytes(d)
    ok = True
    for p, d in originals.items():
        after = p.read_bytes()
        print(f"\n收工 sha256[:16] = {sha(after)}  ({p.name})")
        if after != d:
            print(f"!!! {p.name} 没有字节还原 —— 立即停下")
            ok = False
    if not ok:
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
