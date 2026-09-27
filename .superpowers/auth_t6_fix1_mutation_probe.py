"""认证 T6 修复轮 1 · 变异探针:⑧ 那条断言**真的有判别力**吗。

本仓的头号风险是「假绿测试」—— 所以一条新断言写完之后要问的不是
「它绿了吗」,而是「**把它守护的那个实现改回去,它会不会红**」。

这里改回去的正是修复前那段:唤醒**全部**等待者 → 只唤醒**最后**一个。
预期:探针 ⑧ 从 3/3 掉到 1/3、并且**判红**(exit != 0)。

⚠️ 字节级读/写(`read_bytes` / `write_bytes`)—— 本机 `core.autocrlf=true`,
用 `write_text` 恢复会把整个文件重写成 CRLF,于是 `git status` 上出现一个
**字节其实没变**的 ` M`。恢复之后再逐字节比一次,比不出来就响亮地报。

用法:`.venv/Scripts/python.exe .superpowers/auth_t6_fix1_mutation_probe.py`
前置:客服服务起在 8000(与探针同一条前置)。
"""

import pathlib
import subprocess
import sys

#: ⚠️ 本机 locale 是 cp936,`✓` / `✅` / `⇒` 都不在 GBK 里 —— 不钉住输出边界,
#: 这个脚本会在**判词那一行**自己崩掉(`UnicodeEncodeError`),而崩溃发生在
#: `finally` 恢复文件**之后**、判词**之前** ⇒ 「看起来像变异探针坏了」。
#: 这正是本仓平台陷阱那一段点名过的形态,钉死 utf-8。
sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUTH = ROOT / "app" / "static" / "auth.js"
PROBE = ROOT / ".superpowers" / "auth_t6_login_probe.mjs"

#: 锚点锚在**代码**上,不锚在注释上(注释正是 diff 最容易改到的东西)。
ANCHOR = """        const pending = waiters;
        waiters = [];
        for (const w of pending) {
          try { await w(true); } catch (e) { /* 一个等待者出错不许拖住别人 */ }
        }"""

#: 修复前的形态:只唤醒最后一个(前面那些 promise 永远不落定)。
MUTANT = """        const last = waiters[waiters.length - 1];
        waiters = [];
        if (last) await last(true);"""

orig = AUTH.read_bytes()
text = orig.decode("utf-8")

hit = text.count(ANCHOR)
print(f"锚点命中数 = {hit}(必须是 1 —— 0 表示变异压根没打上,>1 表示打错了地方)")
if hit != 1:
    sys.exit(2)

try:
    AUTH.write_bytes(text.replace(ANCHOR, MUTANT, 1).encode("utf-8"))
    print("已注入变异:等待者队列 → 只唤醒最后一个\n")
    r = subprocess.run(
        ["node", str(PROBE)],
        cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace",
    )
    out = r.stdout + r.stderr
    with (ROOT / ".superpowers" / "auth_t6_fix1_mutation_probe.txt").open(
            "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    for line in out.splitlines():
        if "⑧" in line or "第 " in line or "!!! " in line or "逐条通过" in line \
                or "条失败" in line:
            print("  " + line)
    print(f"\n变异后 node 退出码 = {r.returncode}")
    print("判据:退出码**非 0** 且 ⑧ 那几轮读到 1/3 ⇒ 这条断言真的有判别力")
    print("结果:", "有判别力 ✅" if r.returncode != 0 else "**没有判别力** ❌(变异后照样绿)")
    sys.exit(0 if r.returncode != 0 else 1)
finally:
    AUTH.write_bytes(orig)
    back = AUTH.read_bytes()
    same = back == orig
    print(f"\n已恢复 auth.js:与原始字节{'逐字节相同 ✅' if same else '**对不上** ❌'}")
    if not same:
        sys.exit(3)
