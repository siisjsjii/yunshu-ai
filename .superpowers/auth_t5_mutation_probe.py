"""T5 的变异探针:**证明两条新装置真的有牙**,而且恢复是**逐字节**的。

每个变异分三步:备份字节 → 打锚点 → 跑 pytest(打印退出码)→ 还原 → 复查哈希。

⚠️ 全程 `Path.read_bytes` / `write_bytes` —— 本仓 `core.autocrlf=true`,
`write_text` 会把还原写成一次 CRLF 重写,于是一个逐字节相同的文件在
`git status` 里显示 ` M`(假红)。**还原绝不用 `git checkout --`**(同上)。

⚠️ 锚点**打在代码上、不锚在注释上**,且每次必须**命中恰好 1 次** ——
命中 0 次或多次时本项**作废并打印 `!!!`**(本仓记过的变异事故:锚点失配被
当成「变异后没红 = 断言有牙」)。

⚠️ `expect` 是**事先写下的**期望,不是「跑出来什么就记什么」——
`green` 那一条(M5)拆掉的是一道**冗余**的层,预期就是绿的,理由见那条的注释。

用法:`python .superpowers/auth_t5_mutation_probe.py`
"""

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"

CASES = [
    {
        "name": "M1 拿掉 /api/kb/* 的 router 级 require_admin(等于「忘了挂守卫」)",
        "file": ROOT / "app" / "api" / "kb.py",
        "edits": [(b"router = APIRouter(dependencies=[Depends(require_admin)])",
                   b"router = APIRouter()")],
        "tests": ["tests/test_auth_wiring.py"],
        "expect": "red",
    },
    {
        "name": "M2 把 /api/auth/me 的 require_user 换成 require_admin",
        "file": ROOT / "app" / "api" / "auth.py",
        "edits": [(b"from app.auth import AuthenticatedUser, create_token, require_user, verify_password",
                   b"from app.auth import (AuthenticatedUser, create_token, require_admin,\n                        require_user, verify_password)"),
                  (b"Depends(require_user)]):", b"Depends(require_admin)]):")],
        "tests": ["tests/test_auth_wiring.py"],
        "expect": "red",
    },
    {
        # ⚠️ 这条**只**跑 `tests/test_api_auth.py`(不带 test_auth_wiring.py),
        # 为的是单独证明**运行时**那条 401 用例有牙 —— 否则它的红会被静态那条
        # 盖住,谁有牙就分不出来了。
        "name": "M4 拿掉 /api/traces 的守卫(只跑运行时那条 401 用例)",
        "file": ROOT / "app" / "api" / "traces.py",
        "edits": [(b"router = APIRouter(dependencies=[Depends(require_admin)])",
                   b"router = APIRouter()")],
        "tests": ["tests/test_api_auth.py"],
        "expect": "red",
    },
    {
        # ⚠️ **预期就是绿的**,这不是「装置没牙」,而是「拆掉的是一道冗余的层」:
        # `/api/conversations` 的**端点签名里**本来就写着 `Depends(require_user)`
        # (T4 落的),router 那一行是**第二道**(纵深)。⇒ 拿掉它端点的保护一点没少,
        # 两条用例当然该绿。
        # **反面才是要点**:`/api/kb|review|topics|traces/*` 那 17 个端点**只有**
        # router 一行(签名里没有),所以 M1 必须红。两层混在一起看会得出错的结论
        # ——「删掉守卫测试照样绿」对 M5 成立、对 M1 不成立。
        "name": "M5 拿掉 /api/conversations 的 router 守卫(签名里还有一道 ⇒ 预期仍绿)",
        "file": ROOT / "app" / "api" / "conversations.py",
        "edits": [(b"router = APIRouter(dependencies=[Depends(require_user)])",
                   b"router = APIRouter()")],
        "tests": ["tests/test_auth_wiring.py", "tests/test_api_auth.py"],
        "expect": "green",
    },
    {
        # 复审给的那条反例(修复轮 1 的 Important):把守卫**换型**,不是删掉。
        # 修前它**整套测试全绿** —— `require_admin` 在守卫词表里、
        # `test_workbench_routes_require_admin` 按前缀跳过 `/api/extract`、
        # 运行时无 token 照样 401,而 conftest 把两个守卫都替成同一个 admin 假用户。
        # 修后**只有** `test_the_guard_matrix_matches_spec_64_exactly` 该红
        # —— 那正是「粗断言对它瞎、细的那条才管得住」的直接证据。
        "name": "M6(复审反例)把 /api/extract 的 require_user 换成 require_admin",
        "file": ROOT / "app" / "api" / "extract.py",
        "edits": [(b"router = APIRouter(dependencies=[Depends(require_user)])",
                   b"router = APIRouter(dependencies=[Depends(require_admin)])"),
                  (b"from app.auth import require_user",
                   b"from app.auth import require_admin")],
        "tests": ["tests/test_auth_wiring.py"],
        "expect": "red",
        "expect_only": ["test_the_guard_matrix_matches_spec_64_exactly"],
    },
    {
        # 锚点用 `\r\n`:这两个文件在工作区里是 CRLF(实测),而 `write_text` 会把
        # 还原写成 LF —— 那正是本探针头一条警告说的那件事。
        "name": "M3 拿掉 /api/feedback 的归属检查(越权写缺口回来)",
        "file": ROOT / "app" / "api" / "feedback.py",
        "edits": [(b'    if conv is None:\r\n        raise HTTPException(status_code=404, detail="\xe4\xbc\x9a\xe8\xaf\x9d\xe4\xb8\x8d\xe5\xad\x98\xe5\x9c\xa8")',
                   b"    if conv is None:\r\n        pass  # MUTATED: conv is None")],
        "tests": ["tests/test_api_feedback.py"],
        "expect": "red",
    },
]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def run(tests):
    proc = subprocess.run(
        [str(PY), "-m", "pytest", *tests, "-p", "no:cacheprovider"],
        cwd=str(ROOT), capture_output=True,
    )
    out = (proc.stdout + proc.stderr).decode("utf-8", "replace")
    return proc.returncode, out


def main() -> int:
    bad = 0
    for case in CASES:
        path = case["file"]
        expected = case["expect"]
        original = path.read_bytes()
        before = sha(original)
        print(f"\n{'=' * 74}\n{case['name']}\n文件={path.name}  原始 sha={before}"
              f"  预期={expected}")

        mutated, ok = original, True
        for old, new in case["edits"]:
            hits = mutated.count(old)
            if hits != 1:
                print(f"!!! 锚点命中 {hits} 次(必须是 1):{old[:60]!r} —— 本项作废")
                ok = False
                break
            mutated = mutated.replace(old, new, 1)
        if not ok:
            bad += 1
            continue

        try:
            path.write_bytes(mutated)
            code, out = run(case["tests"])
            tail = [ln for ln in out.splitlines()
                    if "passed" in ln or "failed" in ln or "error" in ln]
            got = "red" if code != 0 else "green"
            flag = "OK" if got == expected else "!!! 与预期不符"
            print(f"变异后:exit={code} -> {got} :: "
                  f"{tail[-1] if tail else '(没有 N passed/failed 行!)'}  [{flag}]")
            if got != expected:
                bad += 1
            failed = sorted({
                ln.split("::")[1].split()[0]
                for ln in out.splitlines() if ln.startswith("FAILED")
            })
            for ln in out.splitlines():
                if ln.startswith("FAILED") or ln.startswith("ERROR"):
                    print("   ", ln)
            # ⚠️ 「哪几条红了」也要对:一条粗断言跟着红了,就分不出新那条细的
            # 有没有牙(本仓记过的「证据落在弱断言上,强断言有没有牙无从知道」)。
            if case.get("expect_only") is not None:
                same = failed == sorted(case["expect_only"])
                print(f"    红了这几条={failed}  期望={sorted(case['expect_only'])}  "
                      f"[{'OK' if same else '!!! 与预期不符'}]")
                if not same:
                    bad += 1
        finally:
            path.write_bytes(original)

        after = sha(path.read_bytes())
        same = after == before
        print(f"还原后 sha={after}  "
              f"{'== 原始(逐字节相同)' if same else '!!! 与原始不同 —— 还原坏了'}")
        if not same:
            bad += 1

    print(f"\n{'=' * 74}\n复原状态下重跑(证明「还原之后真的能过」,不是「本来就红」):")
    for tests in (["tests/test_auth_wiring.py"], ["tests/test_api_auth.py"],
                  ["tests/test_api_feedback.py"]):
        code, out = run(tests)
        tail = [ln for ln in out.splitlines() if "passed" in ln or "failed" in ln]
        flag = "OK" if code == 0 else "!!! 复原后居然红"
        if code != 0:
            bad += 1
        print(f"  {tests[0]} -> exit={code} :: {tail[-1] if tail else '(??)'}  [{flag}]")

    print(f"\n结论:{'全部探针行为符合预期' if bad == 0 else f'{bad} 项异常'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
