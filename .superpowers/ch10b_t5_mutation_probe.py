"""T5 变异探针:改坏实现,确认指定测试**变红**。

规矩(本仓 ch07 复盘那五条事故换来的,逐条照做):
- 锚点**锚在代码上**,不锚在注释上;
- 每次变异后**断言命中数恰好为 1**(打不上锚 = 那次「没红」不算数);
- 逐字节读、逐字节写(`Path.read_bytes` / `write_bytes`),**绝不用 `write_text`**
  —— 上一轮有人因此在 Windows 上把整份文件改成 CRLF,探针自己报了假警;
- `finally` 里还原;
- 输出**不接任何截断/过滤管道**;看不到 `N passed|failed` 就打印 `!!!`
  (node id 过期会让 pytest 退出码 4,而 `tail` 会把错误切掉)。
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

T = "tests/test_topic_taxonomy.py"
L = "tests/test_topic_labeling.py"

#: (标签, 文件, 原文锚点, 改成, 目标 node id)
CASES = [
    (
        "A 渲染函数整段丢掉正例",
        "app/topic/taxonomy.py",
        '        for text in POSITIVE.get(label, ()):\n'
        '            lines.append(f"    · 例:「{text}」")\n',
        "",
        f"{T}::test_rendered_block_carries_every_positive_example",
    ),
    (
        "B 正例行的缩进与反例行错开(两空格)",
        "app/topic/taxonomy.py",
        '            lines.append(f"    · 例:「{text}」")',
        '            lines.append(f"  · 例:「{text}」")',
        f"{T}::test_rendered_block_carries_every_positive_example",
    ),
    (
        "C 空证据串不特判(去掉 `ev and`)",
        "app/topic/labeling.py",
        "        if ev and clean(ev) and clean(ev) in normalized:",
        "        if clean(ev) in normalized:",
        f"{L}::test_empty_evidence_string_is_rejected_not_accepted",
    ),
    (
        "D 证据侧不清洗",
        "app/topic/labeling.py",
        "        if ev and clean(ev) and clean(ev) in normalized:",
        "        if ev and ev in normalized:",
        f"{L}::test_both_sides_of_the_evidence_check_are_cleaned",
    ),
    (
        "E 问句侧不清洗",
        "app/topic/labeling.py",
        "    normalized = clean(question)",
        "    normalized = question",
        f"{L}::test_both_sides_of_the_evidence_check_are_cleaned",
    ),
    (
        "F 被拒标签不返回(吞掉)",
        "app/topic/labeling.py",
        "    return accepted, rejected",
        "    return accepted, []",
        f"{L}::test_fabricated_evidence_is_rejected",
    ),
    (
        "G 解析失败当成 False(订正 C 的旧形态)",
        "scripts/prelabel_topics.py",
        "    except json.JSONDecodeError:\n        return [], {}, True",
        "    except json.JSONDecodeError:\n        return [], {}, False",
        f"{L}::test_prelabel_wiring_writes_the_three_readings_and_resumes",
    ),
    (
        "H 形状闸被拿掉",
        "scripts/prelabel_topics.py",
        "    if not isinstance(raw_labels, list) or not isinstance(raw_ev, dict):\n"
        "        return [], {}, True\n",
        "",
        f"{L}::test_prelabel_wiring_writes_the_three_readings_and_resumes",
    ),
    (
        "I run 不调 validate_evidence",
        "scripts/prelabel_topics.py",
        "            accepted, rejected = validate_evidence("
        'row["question"], labels, evidence)',
        "            accepted, rejected = labels, []",
        f"{L}::test_prelabel_wiring_writes_the_three_readings_and_resumes",
    ),
    (
        "J `_already_done` 恒空(断点续跑失效)",
        "scripts/prelabel_topics.py",
        'def _already_done() -> set[str]:\n    if not OUT.exists():\n        return set()',
        'def _already_done() -> set[str]:\n    return set()\n    if not OUT.exists():\n        return set()',
        f"{L}::test_prelabel_wiring_writes_the_three_readings_and_resumes",
    ),
    (
        "K flagged 只在解析失败时 +1",
        "scripts/prelabel_topics.py",
        "            if rejected:\n                flagged += 1",
        "            if rejected and bad:\n                flagged += 1",
        f"{L}::test_prelabel_wiring_writes_the_three_readings_and_resumes",
    ),
    # ---- 订正轮 1:评审抓出的三条假绿(M1 / M8 / M15)+ 两条 Minor(Mn1 / Mn2)----
    (
        "M1(F1) zero_label 只在解析失败时 +1",
        "scripts/prelabel_topics.py",
        "            if not accepted:\n                zero_label += 1",
        "            if bad:\n                zero_label += 1",
        f"{L}::test_prelabel_wiring_writes_the_three_readings_and_resumes",
    ),
    (
        "M8(F2) 删掉 main() 里那一行 _pin_stdout_encoding()(函数留着)",
        "scripts/prelabel_topics.py",
        "    _pin_stdout_encoding()\n    ap = argparse.ArgumentParser()",
        "    ap = argparse.ArgumentParser()",
        f"{L}::test_main_pins_stdout_encoding",
    ),
    (
        "M15(F3) 调了 bind 但丢掉返回值",
        "scripts/prelabel_topics.py",
        '    return model.bind(response_format={"type": "json_object"})',
        '    model.bind(response_format={"type": "json_object"})\n    return model',
        f"{L}::test_prelabel_wiring_writes_the_three_readings_and_resumes",
    ),
    (
        "Mn1 非字符串标签静默过滤",
        "scripts/prelabel_topics.py",
        "        [_as_label(x) for x in raw_labels],",
        "        [x for x in raw_labels if isinstance(x, str)],",
        f"{L}::test_prelabel_wiring_writes_the_three_readings_and_resumes",
    ),
    (
        "Mn2 空批仍打 0.0%(把 `if todo:` 写死成 True)",
        "scripts/prelabel_topics.py",
        "    if todo:\n",
        "    if True:  # noqa\n",
        f"{L}::test_prelabel_wiring_writes_the_three_readings_and_resumes",
    ),
    (
        "M 不钉 stdout 编码(cp936 下故障路径的 print 会炸)",
        "scripts/prelabel_topics.py",
        '    if hasattr(sys.stdout, "reconfigure"):\n'
        '        sys.stdout.reconfigure(encoding="utf-8")',
        "    return None",
        f"{L}::test_main_pins_stdout_encoding",
    ),
    (
        "L 不绑 response_format",
        "scripts/prelabel_topics.py",
        '    return model.bind(response_format={"type": "json_object"})',
        "    return model",
        f"{L}::test_prelabel_wiring_writes_the_three_readings_and_resumes",
    ),
]


def emit(line: str) -> None:
    """逐条打、逐条 flush —— 中途崩了也不丢已经跑出来的读数。"""
    sys.stdout.buffer.write((line + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def main() -> None:
    for name, rel, old, new, node in CASES:
        path = ROOT / rel
        original = path.read_bytes()
        text = original.decode("utf-8")
        n = text.count(old)
        if n != 1:
            emit(f"!!! {name}: 锚点命中 {n} 次(必须恰好 1 次)—— 这次没红不算数")
            continue
        # 守卫只在**真替换**上做:空串替换(A / H)天然「已在文件里」;
        # **纯删除**(new 是 old 的一部分,M8 就是)也天然在文件里。
        # 其余按**整行**查 —— 裸子串会假警(实测:L 的 `    return model` 是
        # `    return model.bind(…)` 的前缀)。
        if new and new not in old:
            assert f"{new}\n" not in text, f"{name}: 替换后的文本已经在文件里了"
        try:
            path.write_bytes(text.replace(old, new, 1).encode("utf-8"))
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", node],
                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            m = re.search(r"(\d+) (passed|failed|error)", out)
            if not m:
                emit(f"!!! {name}: 输出里没有 `N passed|failed`(exit={proc.returncode})"
                     f" —— 盲区,别读成绿。\n{out[-800:]}")
                continue
            verdict = "RED ✔" if proc.returncode != 0 else "GREEN ✘(无判别力!)"
            if proc.returncode == 4:
                verdict = "!!! exit=4(node id 过期 / 用法错)"
            emit(f"{verdict}  {name}  →  {node}  (exit={proc.returncode}, {m.group(0)})")
        finally:
            path.write_bytes(original)
            assert path.read_bytes() == original, f"{name}: 还原不完整!"


if __name__ == "__main__":
    main()
