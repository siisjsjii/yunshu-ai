#!/usr/bin/env python
"""订正轮 1 · F6 的定向探针:④ 第 ③ 层的 artefact 集是否真的收得住那句回复。

复审实测的那句(5 次运行里红了 1 次,**红在承重句上**):

    已为您转接人工客服,**当前排在您前面还有 1 位,预计 7 分钟左右接入**

它给了位次与时长,却不含「排队」「等待」两词 —— 而那一层原本只认这两个词。

⚠️ 本探针**从验收脚本里原样抽出** `has_needle` / `any_needle` 两个函数再跑
(不复制粘贴 —— 抄一份就是漂移的开始);中文只出现在**文件内容**里,
argv 上只有路径与 hex 码点。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "acceptance_ch10.sh"
TMP = Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "ch10_t16_needle"
TMP.mkdir(parents=True, exist_ok=True)
BASH = os.environ.get("CH10_BASH") or r"D:\kit\Git\usr\bin\bash.exe"

CASES = [
    # (标签, 回复文本, 期望:wait 类在不在 / join 类在不在)
    ("复审那句(只有位次+时长,没有「排队」「等待」)", "已为您转接人工客服,当前排在您前面还有 1 位,预计 7 分钟左右接入", True, True),
    # ⚠️ JOIN 这一列**期望「不在」**:这句里没有「接入」两字(`接入` 是**另一类** artefact)
    # —— 探针第一版把它写成「在」,红的是**探针自己的期望**,不是被测的判据。
    ("原文那种(含「排队」「等待」)", "已为您转接人工客服,当前排队第 2 位,预计等待约 7 分钟", True, False),
    ("只说接入(最弱的一类)", "好的,马上为您接入人工客服。", False, True),
    ("什么都没转述(该判红)", "您的问题我已经记录下来了,请您稍后再问一次。", False, False),
]


def extract_function(text: str, name: str) -> str:
    start = text.index("%s() {" % name)
    end = text.index("\n}", start) + 2
    return text[start:end]


def main() -> int:
    text = SCRIPT.read_text(encoding="utf-8")
    helpers = extract_function(text, "has_needle") + "\n\n" + extract_function(text, "any_needle")
    needles = dict(re.findall(r"^(H_[A-Z]+)=(\"[0-9a-f ]+\")", text, re.M))
    for key in ("H_WAIT", "H_QUEUE", "H_LINEUP", "H_ETA", "H_JOIN"):
        if key not in needles:
            raise SystemExit("!!! 抽不到 %s 的码点定义" % key)
    sh = TMP / "needles.sh"
    sh.write_text(
        # `has_needle` 用的是验收脚本里的 `$PYTHON`(那里早就定义好了)—— 这里是另一个
        # shell,得自己给(`set -u` 下没给会**当场报 unbound variable**,而探针会把
        # 那当成「needle 不在」⇒ 一条自己造的假红)。
        'PYTHON="%s"\n' % (REPO / ".venv" / "Scripts" / "python.exe").as_posix()
        + "set -u\n" + helpers + "\n"
        + "".join("%s=%s\n" % (k, v) for k, v in needles.items())
        + """
f="$1"
any_needle "$f" "$H_WAIT" "$H_QUEUE" "$H_LINEUP" "$H_ETA"; echo "WAIT=$?"
any_needle "$f" "$H_JOIN"; echo "JOIN=$?"
""", encoding="utf-8")
    fails = 0
    for label, reply, want_wait, want_join in CASES:
        path = TMP / "reply.txt"
        path.write_bytes(reply.encode("utf-8"))
        out = subprocess.run([BASH, str(sh), str(path)], cwd=REPO, capture_output=True,
                             text=True, timeout=120).stdout
        got = {k: v.strip() for k, v in
               (line.split("=") for line in out.splitlines() if "=" in line)}
        w, j = int(got.get("WAIT", "9")), int(got.get("JOIN", "9"))
        hit = ((w == 0) == want_wait) and ((j == 0) == want_join)
        fails += 0 if hit else 1
        sys.stdout.buffer.write(
            ("%s %s(WAIT=%d 期望%s / JOIN=%d 期望%s)\n"
             % ("PASS" if hit else "FAIL", label, w,
                "在" if want_wait else "不在", j, "在" if want_join else "不在")).encode("utf-8"))
    sys.stdout.buffer.write(
        ("\n%s:抽取的函数=%d 字节,码点=%s\n"
         % ("全部符合预期" if not fails else "**有 %d 条不符**" % fails,
            len(helpers), " ".join(sorted(needles)))).encode("utf-8"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
