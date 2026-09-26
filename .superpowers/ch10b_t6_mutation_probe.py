"""T6 变异探针:把实现改坏,确认**这条测试真的会红**。

规矩(本仓 ch07 全章复盘的元教训,逐条照做):
- 锚点**锚在代码上,不锚在注释上**,且每次变异后**断言锚点命中数恰好为 1**;
- **绝不用 `Path.write_text` 还原**(上一轮有人在 Windows 上把整份文件改成 CRLF、
  探针自己报了假警)⇒ 全程**字节**读写,并在 `finally` 里用 `read_bytes() == 原始字节` 复核;
- **绝不把证据输出接进任何截断/过滤管道**;看不到 `N passed|failed` 就打 `!!!`;
- **一条测试一次 pytest 调用** —— 汇总行的 `N failed` 说不清「是哪一条红」,
  而这次探针的全部价值就在于「**红在哪条**」。

订正轮 1 新增:C1(读数来源)/ I1(列错位)/ I2(未知 id、重复 id)/ I3(题面来源)/
S18(判定与标签矛盾)/ Mn2·Mn3·Mn4·Mn5。

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

#: 复用同一条测试名,免得每处都抄一长串
T_EVERY_LABEL = f"{T}::test_pick_review_sample_takes_from_every_label"
T_REPRODUCIBLE = f"{T}::test_pick_review_sample_is_reproducible_with_a_seed"
T_DIFF_SEEDS = f"{T}::test_different_seeds_pick_different_samples"
T_MULTI_REACH = f"{T}::test_multi_label_rows_are_reachable_via_a_label_that_is_not_the_first"
T_MULTI_BRIEF = f"{T}::test_multi_label_rows_count_toward_every_label"
T_EXPORT_BLANK = f"{T}::test_export_writes_the_stratified_sample_with_the_judgement_columns_blank"
T_ROUNDTRIP = f"{T}::test_import_takes_the_final_labels_and_falls_back_to_the_prelabels"
T_C1 = f"{T}::test_import_counts_the_error_rate_from_the_labels_not_from_the_verdict"
T_I3 = f"{T}::test_import_takes_the_question_from_the_prelabels_not_from_the_csv"
T_DEDUPE = f"{T}::test_import_dedupes_repeated_labels"
T_REFUSE = f"{T}::test_import_refuses_input_that_would_silently_change_the_reading"
T_MALFORMED = f"{T}::test_import_refuses_malformed_rows"
T_NOT_UTF8 = f"{T}::test_import_explains_a_file_that_is_not_utf8"
T_PIN = f"{T}::test_export_script_pins_stdout_encoding"

#: (名字, 文件, 要替换的原文, 替换成什么, [(测试 id, 期望的 pytest 结果), …])
MUTATIONS = [
    # ---- 分层抽样(labeling.py;锚点自 B6-A 起没动过)----
    (
        "M1 ★B6-A 必查:删掉 `rng.shuffle(shuffled)`",
        LAB,
        "        shuffled = sorted(bucket, key=lambda r: r[\"id\"])\n        rng.shuffle(shuffled)\n",
        "        shuffled = sorted(bucket, key=lambda r: r[\"id\"])\n",
        [
            (T_DIFF_SEEDS, "failed"),
            # ⚠️ 订正 6-C 的证据:**这行删掉之后,「同一种子两次相同」照样绿**
            (T_REPRODUCIBLE, "passed"),
        ],
    ),
    (
        "M2 只记第一个标签(`row['labels'][:1]`)",
        LAB,
        "        for label in row.get(\"labels\") or []:",
        "        for label in (row.get(\"labels\") or [])[:1]:",
        [
            (T_MULTI_REACH, "failed"),
            # ⚠️ 证据:brief 那条 `..._count_toward_every_label` 对这个变异**零判别力**
            (T_MULTI_BRIEF, "passed"),
        ],
    ),
    (
        "M3 `random.Random(seed)` → `random.Random()`(种子被丢掉)",
        LAB,
        "    rng = random.Random(seed)",
        "    rng = random.Random()",
        [(T_REPRODUCIBLE, "failed")],
    ),
    (
        "M4 每类多抽一条(`[:per_label + 1]`)",
        LAB,
        "        for row in shuffled[:per_label]:",
        "        for row in shuffled[:per_label + 1]:",
        [(T_EVERY_LABEL, "failed")],
    ),
    # ---- 导出端 ----
    (
        "M5 导出时把预标标签写进「最终标签」列",
        EXP,
        "            w.writerow([r[\"id\"], r[\"question\"], \"|\".join(r[\"labels\"]), \"\", \"\", \"\"])\n",
        "            w.writerow([r[\"id\"], r[\"question\"], \"|\".join(r[\"labels\"]), \"\",\n"
        "                        \"|\".join(r[\"labels\"]), \"\"])\n",
        [(T_EXPORT_BLANK, "failed")],
    ),
    (
        "M10 删掉 `main()` 里的钉编码",
        EXP,
        "    _pin_stdout_encoding()\n    ap = argparse.ArgumentParser()",
        "    ap = argparse.ArgumentParser()",
        [(T_PIN, "failed")],
    ),
    # ---- ★订正轮 1:回收端 ----
    (
        "M-C1 ★必查:读数改回「只数判定列」(订正前的行为)",
        EXP,
        "                if set(labels) != set(pre[\"labels\"]):\n"
        "                    changed += 1\n"
        "                    if verdict != \"改\":\n"
        "                        unverified.append(rid)\n"
        "                elif verdict == \"改\":\n"
        "                    contradictions.append(rid)\n",
        "                if verdict == \"改\":\n"
        "                    changed += 1\n"
        "                if set(labels) != set(pre[\"labels\"]) and verdict != \"改\":\n"
        "                    unverified.append(rid)\n"
        "                elif set(labels) == set(pre[\"labels\"]) and verdict == \"改\":\n"
        "                    contradictions.append(rid)\n",
        [(T_C1, "failed"), (T_ROUNDTRIP, "passed")],
    ),
    (
        "M-C1b 少填「改」时不再警告(读数仍对,但看不见了)",
        EXP,
        "                        unverified.append(rid)",
        "                        pass",
        [(T_C1, "failed")],
    ),
    (
        "M-I1 去掉「字段比表头多」那条(列错位被吞)",
        EXP,
        "                if None in row:",
        "                if False:",
        [(T_MALFORMED, "failed")],
    ),
    (
        "M-I2a 去掉「id 不在预标语料里」那条",
        EXP,
        "                if rid not in prelabeled:",
        "                if False:",
        [(T_MALFORMED, "failed")],
    ),
    (
        "M-I2b 去掉「id 重复」那条",
        EXP,
        "                if rid in seen:",
        "                if False:",
        [(T_MALFORMED, "failed")],
    ),
    (
        "M-I2c 去掉「预标标签列与语料同源」那条",
        EXP,
        "                if set(raw_pre) != set(pre[\"labels\"]):",
        "                if False:",
        [(T_MALFORMED, "failed")],
    ),
    (
        "M-I3 题面照抄 CSV 那份(不取预标)",
        EXP,
        "                out_rows.append({\"id\": rid, \"question\": pre[\"question\"],\n"
        "                                 \"labels\": labels, \"reviewed\": True})\n",
        "                out_rows.append({\"id\": rid, \"question\": (row.get(\"问题\") or \"\").strip(),\n"
        "                                 \"labels\": labels, \"reviewed\": True})\n",
        [(T_I3, "failed")],
    ),
    (
        "M-S18 去掉「判定=改 但标签没变」那条(两边互相拆台)",
        EXP,
        "                    contradictions.append(rid)",
        "                    pass",
        [(T_REFUSE, "failed")],
    ),
    (
        "M-Mn3 空产物不再拒绝(只有表头也照写)",
        EXP,
        "    if not out_rows:",
        "    if False:",
        [(T_MALFORMED, "failed")],
    ),
    (
        "M-Mn4 重复类目不去重",
        EXP,
        "                deduped = list(dict.fromkeys(labels))",
        "                deduped = labels",
        [(T_DEDUPE, "failed")],
    ),
    (
        "M-Mn5 不再把「不是 UTF-8」翻成人话",
        EXP,
        "    except UnicodeDecodeError as exc:",
        "    except ZeroDivisionError as exc:   # 变异:不再拦解码错",
        [(T_NOT_UTF8, "failed")],
    ),
    # ---- B6-A 那几条护栏(锚点跟着重写挪过,这里重验一遍)----
    (
        "M7 最终标签不逐段 strip(逗号后带空格就炸)",
        EXP,
        "                labels = [x.strip() for x in final.replace(\",\", \"|\").split(\"|\") if x.strip()] \\\n"
        "                    or raw_pre\n",
        "                labels = [x for x in final.replace(\",\", \"|\").split(\"|\") if x.strip()] \\\n"
        "                    or raw_pre\n",
        [(T_ROUNDTRIP, "failed")],
    ),
    (
        "M12 去掉表头缺列那条",
        EXP,
        "            missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]",
        "            missing = []",
        [(T_REFUSE, "failed")],
    ),
    (
        "M13 「判定」认不出也不再报(静默当成没改)",
        EXP,
        "                if verdict not in VERDICTS:",
        "                if False:",
        [(T_REFUSE, "failed")],
    ),
    (
        "M14 不合法类目改成静默丢弃",
        EXP,
        "                invalid = [lb for lb in labels if lb not in known_labels]",
        "                invalid = []",
        [(T_REFUSE, "failed")],
    ),
]

SUMMARY_RE = re.compile(r"(\d+) (passed|failed)")


def _run(test_id):
    return subprocess.run(
        [sys.executable, "-m", "pytest", test_id, "-p", "no:cacheprovider"],
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
                proc = _run(test_id)
                counts = {w: int(n) for n, w in SUMMARY_RE.findall(proc.stdout)}
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
                print(f"  {mark}  {test_id.split('::')[-1]}  (期望 {want}){flag}   [{counts}]")
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
