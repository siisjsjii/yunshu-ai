"""T6 变异探针:把实现改坏,确认**这条测试真的会红**。

规矩(本仓 ch07 全章复盘的元教训,逐条照做):
- 锚点**锚在代码上,不锚在注释上**;
- 每次变异后**断言锚点命中数恰好为 1**(打不上锚点的变异 = 假红/假绿,必须 `!!!`);
- **绝不用 `Path.write_text` 还原**(上一轮有人在 Windows 上把整份文件改成 CRLF、
  探针自己报了假警)⇒ 全程**字节**读写,并在 `finally` 里用 `read_bytes() == 原始字节` 复核;
- **绝不把证据输出接进任何截断/过滤管道**;看不到 `N passed|failed` 就打 `!!!`。

用法:
    .venv/Scripts/python.exe -X utf8 .superpowers/ch10b_t6_mutation_probe.py
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = "tests/test_topic_labeling.py"

LAB = ROOT / "app" / "topic" / "labeling.py"
EXP = ROOT / "scripts" / "export_label_review.py"

#: (名字, 文件, 要替换的原文, 替换成什么, [(测试 id, 期望的 pytest 结果), …])
MUTATIONS = [
    (
        "M1 ★必查:删掉 `rng.shuffle(shuffled)`",
        LAB,
        "        shuffled = sorted(bucket, key=lambda r: r[\"id\"])\n        rng.shuffle(shuffled)\n",
        "        shuffled = sorted(bucket, key=lambda r: r[\"id\"])\n",
        [
            (f"{T}::test_different_seeds_pick_different_samples", "failed"),
            # ⚠️ 订正 6-C 的证据:**这行删掉之后,「同一种子两次相同」照样绿**
            (f"{T}::test_pick_review_sample_is_reproducible_with_a_seed", "passed"),
        ],
    ),
    (
        "M2 只记第一个标签(`row['labels'][:1]`)",
        LAB,
        "        for label in row.get(\"labels\") or []:",
        "        for label in (row.get(\"labels\") or [])[:1]:",
        [
            (f"{T}::test_multi_label_rows_are_reachable_via_a_label_that_is_not_the_first", "failed"),
            # ⚠️ 证据:brief 那条 `..._count_toward_every_label` 对这个变异**零判别力**
            (f"{T}::test_multi_label_rows_count_toward_every_label", "passed"),
        ],
    ),
    (
        "M3 `random.Random(seed)` → `random.Random()`(种子被丢掉)",
        LAB,
        "    rng = random.Random(seed)",
        "    rng = random.Random()",
        [(f"{T}::test_pick_review_sample_is_reproducible_with_a_seed", "failed")],
    ),
    (
        "M4 每类多抽一条(`[:per_label + 1]`)",
        LAB,
        "        for row in shuffled[:per_label]:",
        "        for row in shuffled[:per_label + 1]:",
        [(f"{T}::test_pick_review_sample_takes_from_every_label", "failed")],
    ),
    (
        "M5 导出时把预标标签写进「最终标签」列",
        EXP,
        "            w.writerow([r[\"id\"], r[\"question\"], \"|\".join(r[\"labels\"]), \"\", \"\", \"\"])\n",
        "            w.writerow([r[\"id\"], r[\"question\"], \"|\".join(r[\"labels\"]), \"\",\n"
        "                        \"|\".join(r[\"labels\"]), \"\"])\n",
        [(f"{T}::test_export_writes_the_stratified_sample_with_the_judgement_columns_blank", "failed")],
    ),
    (
        "M6 去掉表头缺列的响亮报错",
        EXP,
        "        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]",
        "        missing = []",
        [(f"{T}::test_import_refuses_input_that_would_silently_change_the_reading", "failed")],
    ),
    (
        "M7 最终标签**不**逐段 strip(逗号后带空格就炸)",
        EXP,
        "            labels = [x.strip() for x in final.replace(\",\", \"|\").split(\"|\") if x.strip()] or \\\n"
        "                     [x.strip() for x in (row[\"预标标签\"] or \"\").split(\"|\") if x.strip()]\n",
        "            labels = [x for x in final.replace(\",\", \"|\").split(\"|\") if x.strip()] or \\\n"
        "                     [x.strip() for x in (row[\"预标标签\"] or \"\").split(\"|\") if x.strip()]\n",
        [(f"{T}::test_import_takes_the_final_labels_and_falls_back_to_the_prelabels", "failed")],
    ),
    (
        "M8 「判定」认不出时不当成「没改」",
        EXP,
        "            verdict = (row.get(\"判定(ok/改)\") or \"\").strip()\n"
        "            if verdict not in VERDICTS:",
        "            verdict = (row.get(\"判定(ok/改)\") or \"\").strip()\n"
        "            if False:",
        [
            (f"{T}::test_import_refuses_input_that_would_silently_change_the_reading", "failed"),
            # ⚠️ 反面:把「认不出的值」当「没改」→ 读数悄悄变小,而这条**照样绿**
            (f"{T}::test_import_takes_the_final_labels_and_falls_back_to_the_prelabels", "passed"),
        ],
    ),
    (
        "M9 `verdict == \"改\"` → `if verdict:`(空判定也算改)",
        EXP,
        "            if verdict == \"改\":",
        "            if verdict or verdict == \"改\":",
        [(f"{T}::test_import_takes_the_final_labels_and_falls_back_to_the_prelabels", "failed")],
    ),
    (
        "M10 删掉 `main()` 里的钉编码",
        EXP,
        "    _pin_stdout_encoding()\n    ap = argparse.ArgumentParser()",
        "    ap = argparse.ArgumentParser()",
        [(f"{T}::test_export_script_pins_stdout_encoding", "failed")],
    ),
    (
        "M11 合法类目检查改成静默丢弃",
        EXP,
        "            invalid = [lb for lb in labels if lb not in known_labels]",
        "            invalid = []",
        [
            (f"{T}::test_import_refuses_input_that_would_silently_change_the_reading", "failed"),
            (f"{T}::test_import_takes_the_final_labels_and_falls_back_to_the_prelabels", "passed"),
        ],
    ),
]

SUMMARY_RE = re.compile(r"(\d+) (passed|failed)")


def _run(tests):
    return subprocess.run(
        [sys.executable, "-m", "pytest", *tests, "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def main() -> None:
    problems = 0
    for name, path, old, new, expects in MUTATIONS:
        original = path.read_bytes()
        old_b, new_b = old.encode("utf-8"), new.encode("utf-8")
        print(f"\n=== {name}  [{path.relative_to(ROOT)}]")
        hits = original.count(old_b)
        # ⚠️ 锚点必须**恰好命中一次** —— 0 次是「变异压根没生效」,>1 次是「改的不止一处」。
        if hits != 1:
            print(f"  !!! 锚点命中 {hits} 次(应为 1)⇒ 这次变异**不作数**")
            problems += 1
            continue
        try:
            path.write_bytes(original.replace(old_b, new_b, 1))
            for test_id, want in expects:
                # ⚠️ **一条测试一次 pytest 调用** —— 汇总行里 `N failed` 的多寡
                # 说不清「是哪一条红」,而这次探针的全部价值就在于「哪一条红」。
                proc = _run([test_id])
                counts = dict((w, int(n)) for n, w in SUMMARY_RE.findall(proc.stdout))
                if not counts:
                    print(f"  !!! 输出里没有 `N passed|failed`(exit={proc.returncode})"
                          f" —— 这次读数不作数")
                    print(proc.stdout)
                    print(proc.stderr)
                    problems += 1
                    continue
                got = "failed" if counts.get("failed") else "passed"
                mark = "✅ 红" if got == "failed" else "绿灯"
                flag = "" if got == want else "   !!! 与期望不符"
                print(f"  {mark}  {test_id.split('::')[-1]}  (期望 {want}){flag}"
                      f"   [{counts}]")
                if got != want:
                    problems += 1
        finally:
            path.write_bytes(original)
            if path.read_bytes() != original:
                print(f"  !!! 还原失败 —— {path} 已被改动,立刻停下")
                problems += 1
            else:
                print(f"  · 已逐字节还原 {path.name}")
    print(f"\n{'=' * 60}\n问题数:{problems}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
