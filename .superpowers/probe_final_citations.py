"""最终修复轮:把被跟踪文档里引用的 `.superpowers/**` 路径**逐条解析一遍**。

为什么需要它:本章好几条承重数字的**唯一凭据**是 `.superpowers/` 下的一个探针
或一份转录,而 `.superpowers/` **不是 gitignore 的**(只有 `.superpowers/sdd/`
自带 `*`)⇒ 「哪些该入库」以前是**随机的**,新人 clone 下来引用解析不了也看不出
少了什么。修复轮把规则写进 `CLAUDE.md` 的 ch09 一节,这个脚本就是规则的判据。

用法:`python .superpowers/probe_final_citations.py`
退出码:0 = 每条引用都解析得开(或在**例外表**里);1 = 有引用落空。
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: 被检查的文档(全是被 git 跟踪的;`.superpowers/sdd/**` 里的报告不在其列,
#: 因为它们自己就不进版本控制「见下面的例外表」)。
SOURCES = [
    "CLAUDE.md",
    "AGENTS.md",
    "dev-notes/ch09.md",
    "scripts/acceptance_ch09.sh",
    "docs/superpowers/specs/2026-09-23-ecommerce-cs-ch09-observe-flywheel-design.md",
]

#: 解析不开是**允许**的那些 —— 每一条都要有理由,否则规则就成了「看心情」。
ALLOWED = {
    ".superpowers/sdd/.gitignore": "这是 `.superpowers/sdd/` 的忽略规则文件本身,不该入库",
    ".superpowers/sdd/": "整个目录自带 .gitignore(`*`),sdd 账本/报告一律**本机 workspace,不入版本控制**",
}

#: 前缀匹配的例外(`.superpowers/sdd/…` 下的任意文件)。
ALLOWED_PREFIXES = (".superpowers/sdd/",)

#: 直接写死在文档里的「本机临时文件」/「已删」引用,解析不开是**已知且记账**的。
#: 键是文档里写的那个串,值是理由与它现在在哪。
KNOWN_LOCAL = {
    ".superpowers/probe_t4_filter.py":
        "spec §15 明写「两个都**已删**」(探针跑完即废,结论已抄进 spec)",
    ".superpowers/t17/latest.json.bak":
        "**运行产物**(一份 300 条的一次性结果),无跟踪文档引用它 ⇒ 最终修复轮 "
        "`git rm --cached` 退出版本控制,文件仍在磁盘上;"
        "`evals/results/` 被 gitignore 的理由与它相同",
}


def tracked(path: str) -> bool:
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=REPO, capture_output=True, text=True,
    )
    return proc.returncode == 0


def main() -> int:
    pattern = re.compile(r"\.superpowers/[A-Za-z0-9_./-]+")
    cited: dict[str, list[str]] = {}
    for src in SOURCES:
        text = (REPO / src).read_text(encoding="utf-8")
        for raw in pattern.findall(text):
            path = raw.rstrip(".,:;)'\"`。、")
            if "..." in path or path.endswith("/"):
                continue          # 文档里的省略写法(`.superpowers/.../x.md`),不算一条
            cited.setdefault(path, []).append(src)

    ok, missing, allowed = [], [], []
    for path, where in sorted(cited.items()):
        if tracked(path):
            ok.append(path)
        elif path in ALLOWED or path.startswith(ALLOWED_PREFIXES):
            allowed.append(f"{path}(例外:{ALLOWED.get(path, 'sdd 目录,本机 workspace')})")
        elif path in KNOWN_LOCAL:
            allowed.append(f"{path}(例外:{KNOWN_LOCAL[path]})")
        else:
            missing.append(f"{path}  ← 被 {', '.join(sorted(set(where)))} 引用")

    print(f"引用到的 `.superpowers/**` 路径:{len(cited)} 条")
    print(f"  解析得开(已被 git 跟踪):{len(ok)}")
    for line in ok:
        print(f"    + {line}  ← {', '.join(sorted(set(cited[line])))}")
    print(f"  例外(已记账):{len(allowed)}")
    for line in allowed:
        print(f"    - {line}")
    if missing:
        print(f"  !!! 解析不开且没有记账:{len(missing)}")
        for line in missing:
            print(f"    - {line}")
        return 1
    print("  落空且未记账:0 —— 全部引用在**全新 clone** 里都解析得开")
    return 0


if __name__ == "__main__":
    sys.exit(main())
