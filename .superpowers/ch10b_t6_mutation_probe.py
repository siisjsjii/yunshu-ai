"""T6 变异探针:把实现改坏,确认**这条测试真的会红**。

⚠️ **同时只许一个人跑这个探针。** 复审自己踩过:两个运行器跑在同一棵树上,
一个把另一个的变异当成了「原始」⇒ **同一天先报 `问题数:0`、后报 `问题数:2`**,
而「逐字节还原」的日志**照样打印**(那是还原到了被污染的那一份)。
⇒ 本探针因此有两道自己的检查:开跑时把三份源码的 sha256 打出来(并对照 `HEAD`),
**每次变异前**再断言三份源码与开跑那一刻**逐字节相同**(`树洁癖复查 ✅`)。

规矩(本仓 ch07 全章复盘的元教训,逐条照做):
- 锚点**锚在代码上,不锚在注释上**,且每次变异后**断言锚点命中数恰好为 1**;
- **绝不用 `Path.write_text` 还原**(上一轮有人在 Windows 上把整份文件改成 CRLF、
  探针自己报了假警)⇒ 全程**字节**读写,并在 `finally` 里用 `read_bytes() == 原始字节` 复核;
- **绝不把证据输出接进任何截断/过滤管道**;看不到 `N passed|failed` 就打 `!!!`;
- **一条测试一次 pytest 调用** —— 汇总行的 `N failed` 说不清「是哪一条红」,
  而这次探针的全部价值就在于「**红在哪条**」;
- **真产物不许被动**:开跑与收工时都量一次 `evals/topic/labels/trainval.csv`
  (它此刻**正被用户改**,是 CP-2 的输入)。

订正轮 1:C1 / I1 / I2a-c / I3 / S18 / Mn3-5。
订正轮 2 新增:F4 的 N1(顺序敏感)/ N4(id 不 strip)/ N5(同源护栏放松成子集)/
F1(桩失效 ⇒ 子进程跑真 export)。

用法:
    .venv/Scripts/python.exe -X utf8 .superpowers/ch10b_t6_mutation_probe.py
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = "tests/test_topic_labeling.py"

LAB = ROOT / "app" / "topic" / "labeling.py"
EXP = ROOT / "scripts" / "export_label_review.py"
TEST = ROOT / "tests" / "test_topic_labeling.py"
ARTIFACT = ROOT / "evals" / "topic" / "labels" / "trainval.csv"

#: 树洁癖盯的三份源码。
SRC = (LAB, EXP, TEST)

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
T_ORDER_PAD = f"{T}::test_import_is_not_fooled_by_ordering_or_padding"
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
    (
        "M-F1 ★订正轮 2:桩失效(main() 走的是**导入时绑死**的别名)",
        EXP,
        # ⚠️ **两处修改,缺一不成**:光是 `main()` 改个写法没用 —— `export` 是**全局查找**,
        #    调用那一刻读的仍是测试替换过的那个属性(第一版这么写,实测**全绿**)。
        #    真正让桩失效的是「**导入时**就把函数对象绑到一个别名上」——
        #    这正是「改个名 / 换个封装」在真实重构里的样子。
        [
            ("    export() if args.action == \"export\" else do_import()\n",
             "    _REAL_EXPORT() if args.action == \"export\" else do_import()\n"),
            ("if __name__ == \"__main__\":\n    main()\n",
             "_REAL_EXPORT = export      # 变异:导入时绑死(桩从此拦不住它)\n\n\n"
             "if __name__ == \"__main__\":\n    main()\n"),
        ],
        None,       # ← 多处修改时用上面那个列表,这一格不用
        # 红在哪句:桩失效 ⇒ 子进程里真跑了 export ⇒ 它写进 **tmp**(路径已被测试改过)
        # ⇒ `assert not (tmp / "trainval.csv").exists()` 失配。**真产物一行不动。**
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
    # ---- ★订正轮 2 的 F4:复审实测「全绿」的那三个 ----
    (
        "M-N1 ★订正轮 2:集合比较 → 列表比较(顺序敏感)",
        EXP,
        "                if set(labels) != set(pre[\"labels\"]):",
        "                if list(labels) != list(pre[\"labels\"]):",
        [(T_ORDER_PAD, "failed")],
    ),
    (
        "M-N4 ★订正轮 2:id 不 strip()",
        EXP,
        "                rid = (row.get(\"id\") or \"\").strip()",
        "                rid = row.get(\"id\") or \"\"",
        [(T_ORDER_PAD, "failed")],
    ),
    (
        "M-N5 ★订正轮 2:同源护栏「集合相等」→「子集」",
        EXP,
        "                if set(raw_pre) != set(pre[\"labels\"]):",
        "                if not set(raw_pre) <= set(pre[\"labels\"]):",
        [(T_MALFORMED, "failed")],
    ),
    # ---- B6-A / 订正轮 1 的其余护栏(锚点跟着重写挪过,每次重验)----
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


def _git_blob(rel):
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision_report(snapshot) -> None:
    """开跑前把「我到底在测哪一版」记下来,并对照 `HEAD`。"""
    print("树洁癖复查(开跑那一刻的版本;要跟报告的其余读数对上):")
    for p in SRC:
        rel = p.relative_to(ROOT).as_posix()
        cur = snapshot[p]
        blob = _git_blob(rel)
        if blob is None:
            state = "未入库"
        elif cur == blob:
            state = "== HEAD(逐字节)"
        elif cur.replace(b"\r\n", b"\n") == blob:
            state = "== HEAD(仅行尾差异 / autocrlf)"
        else:
            state = "与 HEAD **不同** —— 本轮有未提交改动(报告里要写明)"
        print(f"  · {rel}  {len(cur)} B  sha256={hashlib.sha256(cur).hexdigest()[:16]}  {state}")


def _hygiene_ok(snapshot) -> bool:
    """每次变异**前**:三份源码必须与开跑那一刻逐字节相同。"""
    for p, b in snapshot.items():
        if p.read_bytes() != b:
            print(f"  !!! 树被污染:{p.relative_to(ROOT)} 与开跑时不同"
                  f" —— 是不是**有别人在同一棵树上跑**?这次变异不作数")
            return False
    return True


def _run(test_id):
    return subprocess.run(
        [sys.executable, "-m", "pytest", test_id, "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def main() -> None:
    snapshot = {p: p.read_bytes() for p in SRC}
    art_before = ARTIFACT.read_bytes() if ARTIFACT.exists() else None
    _revision_report(snapshot)
    print(f"  · 真产物 {ARTIFACT.name}  sha256={_sha256(ARTIFACT)[:16]}"
          f"(它**正被用户改**,本轮不许动它)")

    problems = 0
    for name, path, old, new, expects in MUTATIONS:
        original = path.read_bytes()
        print(f"\n=== {name}  [{path.relative_to(ROOT)}]")
        if not _hygiene_ok(snapshot):
            problems += 1
            continue
        print("  · 树洁癖复查 ✅")
        # 一个变异可以是**多处**修改(`old` 给成 [(原文, 替换), …]);单处时就是 (old, new)。
        edits = old if isinstance(old, list) else [(old, new)]
        try:
            ok = True
            for o, n in edits:
                ob, nb = o.encode("utf-8"), n.encode("utf-8")
                cur = path.read_bytes()
                # ⚠️ 锚点必须**恰好命中一次** —— 0 次是「变异压根没生效」,>1 次是「改的不止一处」。
                hits = cur.count(ob)
                if hits != 1:
                    print(f"  !!! 锚点命中 {hits} 次(应为 1):{o.splitlines()[0][:50]!r}"
                          f" ⇒ 这次变异**不作数**")
                    problems += 1
                    ok = False
                    break
                path.write_bytes(cur.replace(ob, nb, 1))
            if not ok:
                continue
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

    art_after = ARTIFACT.read_bytes() if ARTIFACT.exists() else None
    same = art_after == art_before
    print(f"\n真产物 {ARTIFACT.name}:"
          f" {'未被动过 ✅' if same else '!!! 被改动了 —— 立刻停下'}"
          f"(sha256={_sha256(ARTIFACT)[:16]})")
    if not same:
        problems += 1
    print(f"{'=' * 60}\n问题数:{problems}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
