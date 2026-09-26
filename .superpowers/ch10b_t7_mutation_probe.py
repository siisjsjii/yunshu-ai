"""T7(B7-A)变异探针:把实现改坏,确认**这条测试真的会红**。

⚠️ **同时只许一个人跑这个探针。** 复审自己踩过:两个运行器跑在同一棵树上,
一个把另一个的变异当成了「原始」⇒ **同一天先报 `问题数:0`、后报 `问题数:2`**,
而「逐字节还原」的日志**照样打印**(那是还原到了被污染的那一份)。
⇒ 本探针因此有两道自己的检查:开跑时把四份源码的 sha256 打出来(并对照 `HEAD`),
**每次变异前**再断言四份源码与开跑那一刻**逐字节相同**(`树洁癖复查 ✅`)。

规矩(本仓 ch07 全章复盘的元教训,逐条照做):
- 锚点**锚在代码上,不锚在注释上**,且每次变异后**断言锚点命中数恰好为 1**;
- **绝不用 `Path.write_text` 还原**(上一轮有人在 Windows 上把整份文件改成 CRLF、
  探针自己报了假警)⇒ 全程**字节**读写,并在 `finally` 里用 `read_bytes() == 原始字节` 复核;
- **绝不把证据输出接进任何截断/过滤管道**;看不到 `N passed|failed` 就打 `!!!`;
- **一条测试一次 pytest 调用** —— 汇总行的 `N failed` 说不清「是哪一条红」,
  而这次探针的全部价值就在于「**红在哪条**」;
- **真产物不许被动**:开跑与收工时都量一次 `evals/topic/labels/trainval.csv`
  (**用户此刻正在改它**,是 CP-2 的输入)与 `evals/topic/labels/test.csv`(本轮新产物)。

本轮的五个覆盖点(与 brief 的订正 9-B/9-C/9-D/9-E 一一对应):
文件名接不上 / train-vs-test 重复没接、近重复被删 / 零标签行没处置 / 全量导出写成抽样 /
命令行分派够不着。另有两条**同义反复**的证据(M1 的 `test_split_is_reproducible` 与
M4 的过度处置),它们**期望绿** —— 那是「这条断言零判别力」的凭据,不是失败。

用法:
    .venv/Scripts/python.exe -X utf8 .superpowers/ch10b_t7_mutation_probe.py
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = "tests/test_topic_labeling.py"

LAB = ROOT / "app" / "topic" / "labeling.py"
PREP = ROOT / "scripts" / "prepare_topic_data.py"
EXP = ROOT / "scripts" / "export_label_review.py"
TEST = ROOT / "tests" / "test_topic_labeling.py"

TRAINVAL = ROOT / "evals" / "topic" / "labels" / "trainval.csv"
TESTCSV = ROOT / "evals" / "topic" / "labels" / "test.csv"

#: 树洁癖盯的四份源码。
SRC = (LAB, PREP, EXP, TEST)

#: 复用同一条测试名,免得每处都抄一长串
T_COMPOSITION = f"{T}::test_test_set_has_the_prescribed_composition"
T_NO_OVERLAP = f"{T}::test_splits_do_not_overlap"
T_ALL_USED = f"{T}::test_all_rows_are_used"
T_REPRODUCIBLE = f"{T}::test_split_is_reproducible"
T_DIFF_SEEDS = f"{T}::test_different_seeds_give_different_test_sets"
T_HEAD = f"{T}::test_head_labels_are_oversampled_into_the_test_set"
T_RARE = f"{T}::test_rare_labels_are_not_squeezed_out_of_the_test_set"
T_PREDICATE = f"{T}::test_unusable_target_predicate"
T_UNTRUSTED = f"{T}::test_rows_with_an_untrusted_target_are_dropped_out_of_the_split"
T_OVERLAP = f"{T}::test_train_test_overlap_reports_exact_duplicates_and_near_pairs"
T_OVERLAP_EMPTY = f"{T}::test_train_test_overlap_is_empty_when_nothing_matches"
T_NEAR = f"{T}::test_train_test_overlap_honours_the_near_threshold"
T_SPLIT = f"{T}::test_split_wires_the_overlay_the_zero_label_rows_and_the_overlap_removal"
T_DISPATCH_SPLIT = f"{T}::test_main_dispatches_split"
T_CONSTS = f"{T}::test_the_frozen_test_set_path_points_at_topic_test_jsonl"
T_DISPATCH_EXPORT = f"{T}::test_main_dispatches_export_test"
T_EXPORT_BLANK = f"{T}::test_export_writes_the_stratified_sample_with_the_judgement_columns_blank"
T_EXPORT_TEST = f"{T}::test_export_test_writes_every_row_with_the_judgement_columns_blank"

#: (名字, 文件, 要替换的原文, 替换成什么, [(测试 id, 期望的 pytest 结果), …])
MUTATIONS = [
    # ---- 抽样机制(app/topic/labeling.py)----
    (
        "M1 ★必查:删掉 take() 里的 `rng.shuffle(b)`",
        LAB,
        '            b.sort(key=lambda r: r["id"])\n            rng.shuffle(b)\n',
        '            b.sort(key=lambda r: r["id"])\n',
        [
            (T_DIFF_SEEDS, "failed"),
            # ⚠️ **证据**:这行删掉后「同一种子两次相同」照样绿 —— 那条是**同义反复**
            #    (与 pick_review_sample 上记的同款,订正 6-C)。
            (T_REPRODUCIBLE, "passed"),
        ],
    ),
    (
        # ⚠️ 锚点**必须带下一行**:`    rng = random.Random(seed)` 在 `labeling.py` 里
        #    **出现两次**(`pick_review_sample` 与 `stratified_split`)—— 第一版只锚那一行,
        #    探针自己报「锚点命中 2 次 ⇒ 不作数」(那正是这条纪律存在的理由)。
        "M2 `random.Random(seed)` → `random.Random()`(种子被丢掉)",
        LAB,
        "    rng = random.Random(seed)\n    by_prov",
        "    rng = random.Random()\n    by_prov",
        [(T_REPRODUCIBLE, "failed")],
    ),
    (
        "M3 去掉「靶子不可信先摘掉」(`usable = list(rows)`)",
        LAB,
        "    usable = [r for r in rows if not is_unusable_target(r)]\n",
        "    usable = list(rows)\n",
        [(T_UNTRUSTED, "failed")],
    ),
    (
        "M4 谓词过度处置:零标签就排除(不看 rejected_labels)",
        LAB,
        '    return not (row.get("labels") or []) and bool(row.get("rejected_labels"))\n',
        '    return not (row.get("labels") or [])\n',
        [
            (T_PREDICATE, "failed"),
            # ⚠️ 这一条**期望红**:过度处置会把「模型真判零诉求」的那行也丢掉
            #    (`r-0049` 的形状)—— 反面对照那一半就在这条测试里。
            (T_UNTRUSTED, "failed"),
        ],
    ),
    (
        "M5 轮转取 → 每次从**最大的桶**取(小类被挤没)",
        LAB,
        "            k = keys[i % len(keys)]\n",
        "            k = max(keys, key=lambda x: len(buckets[x]))\n",
        [
            (T_RARE, "failed"),
            # ⚠️ **证据**:brief 那条 `>= 8` 在这里**照样绿** —— 等量语料下随机/按比例抽
            #    也够 8 ⇒ 它对「有没有分层」零判别力(我在报告里记了这条)。
            (T_HEAD, "passed"),
        ],
    ),
    (
        "M6 train_test_overlap 用**原文**比较(不过 clean)",
        LAB,
        '    for row in train:\n        text = clean(row.get("question") or "")\n',
        '    for row in train:\n        text = row.get("question") or ""\n',
        [(T_OVERLAP, "failed")],
    ),
    (
        "M7 `near` 默认值 0.8 → 1.0(近重复只剩「完全相同」)",
        LAB,
        "                       near: float = 0.8) -> dict[str, object]:\n",
        "                       near: float = 1.0) -> dict[str, object]:\n",
        [
            (T_OVERLAP, "failed"),
            # ⚠️ **实测订正了探针的预期**:`T_NEAR` 在这里**绿**。
            #    它钉的是「`near` **是个参数**」(低门槛那句),不是**默认值** ——
            #    默认值由 `T_OVERLAP` 钉(上面那条红了)。两个方向由下面的 M7b 补齐。
            (T_NEAR, "passed"),
        ],
    ),
    (
        "M7b 阈值写死 `0.8`(参数被忽略)",
        LAB,
        "            if score >= near:\n",
        "            if score >= 0.8:\n",
        [
            (T_NEAR, "failed"),
            (T_OVERLAP, "passed"),      # 默认值本来就是 0.8 ⇒ 这条不动
        ],
    ),
    (
        "M8 完全相同的那对**也**计入 near_pairs(读数虚高)",
        LAB,
        '            exact.append(row["id"])\n            continue\n',
        '            exact.append(row["id"])\n',
        [(T_OVERLAP, "failed")],
    ),
    # ---- `split` 子命令(scripts/prepare_topic_data.py)----
    (
        "M9 ★订正 9-B:产物名写成 `test.jsonl`",
        PREP,
        'TEST_NAME = "topic_test.jsonl"\n',
        'TEST_NAME = "test.jsonl"\n',
        [(T_SPLIT, "failed")],
    ),
    (
        "M10 ★订正 9-C:完全相同的不摘除(数据泄漏)",
        PREP,
        "    if exact_ids:\n",
        "    if False:\n",
        [(T_SPLIT, "failed")],
    ),
    (
        "M11 `reviewed.jsonl` 的覆盖不生效(人工复核成果进不了语料)",
        PREP,
        "    if REVIEWED.exists():\n        for line in REVIEWED.read_text(encoding=\"utf-8\").splitlines():\n",
        "    if False:\n        for line in REVIEWED.read_text(encoding=\"utf-8\").splitlines():\n",
        [(T_SPLIT, "failed")],
    ),
    (
        "M12 ★订正 9-E:零标签行的去向不打印",
        PREP,
        '    zero = [r for r in rows if not (r.get("labels") or [])]\n    if zero:\n',
        "    zero = []\n    if zero:\n",
        [(T_SPLIT, "failed")],
    ),
    (
        "M13 测试集构成那条 assert 不再拦(synth 41 也照写)",
        PREP,
        "    parts = stratified_split(rows, test_real=80, test_synth=40)\n",
        "    parts = stratified_split(rows, test_real=80, test_synth=41)\n",
        [(T_SPLIT, "failed")],
    ),
    (
        "M14 ★命令行分派:删掉 `elif args.step == \"split\"`",
        PREP,
        '    elif args.step == "split":\n        split()\n',
        "",
        [(T_DISPATCH_SPLIT, "failed")],
    ),
    # ---- 导出端(scripts/export_label_review.py)----
    (
        # ⚠️ **这条的第一版预期红错了两处,都是实测订正的**(探针纪律的价值就在这):
        #    ① `T_EXPORT_TEST` 在这里**绿** —— 它 `monkeypatch` 掉了 `TOPIC_TEST`,
        #       所以**常量本身指错它也照样绿**(那是「替身替被测对象完成了语义」的近亲:
        #       被测的那件事被测试自己换掉了);
        #    ② 于是补了 `T_CONSTS`(直接钉字面 + 钉两侧一致),这条变异改由它来红。
        "M15 ★订正 9-B:export-test 读 `test.jsonl`",
        EXP,
        'TOPIC_TEST = ROOT / "evals" / "topic" / "topic_test.jsonl"\n',
        'TOPIC_TEST = ROOT / "evals" / "topic" / "test.jsonl"\n',
        [(T_CONSTS, "failed"), (T_EXPORT_TEST, "passed")],
    ),
    (
        "M16 export-test 写成**抽样**(照抄 export 的 per_label=5)",
        EXP,
        '    _write_review_csv(rows, LABELS_DIR / "test.csv")\n',
        '    _write_review_csv(pick_review_sample(rows, per_label=PER_LABEL), LABELS_DIR / "test.csv")\n',
        [(T_EXPORT_TEST, "failed")],
    ),
    (
        "M17 导出时把预标标签写进「最终标签」列",
        EXP,
        '            w.writerow([r["id"], r["question"], "|".join(r["labels"]), "", "", ""])\n',
        '            w.writerow([r["id"], r["question"], "|".join(r["labels"]), "",\n'
        '                        "|".join(r["labels"]), ""])\n',
        [
            (T_EXPORT_TEST, "failed"),
            # ⚠️ 共用写口(`_write_review_csv`)的证据:trainval 那条**一起红**。
            (T_EXPORT_BLANK, "failed"),
        ],
    ),
    (
        "M18 ★命令行分派:删掉 `elif args.action == \"export-test\"`",
        EXP,
        '    elif args.action == "export-test":\n        export_test()\n    else:\n        do_import()\n',
        "    else:\n        do_import()\n",
        [(T_DISPATCH_EXPORT, "failed")],
    ),
]

SUMMARY_RE = re.compile(r"(\d+) (passed|failed)")


def _git_blob(rel):
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "(不存在)"


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
    """每次变异**前**:四份源码必须与开跑那一刻逐字节相同。"""
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
    art_before = {p: (p.read_bytes() if p.exists() else None) for p in (TRAINVAL, TESTCSV)}
    _revision_report(snapshot)
    for p in (TRAINVAL, TESTCSV):
        print(f"  · 真产物 {p.name}  sha256={_sha256(p)[:16]}(开工前)")

    problems = 0
    for name, path, old, new, expects in MUTATIONS:
        original = path.read_bytes()
        print(f"\n=== {name}  [{path.relative_to(ROOT)}]")
        if not _hygiene_ok(snapshot):
            problems += 1
            continue
        print("  · 树洁癖复查 ✅")
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

    for p in (TRAINVAL, TESTCSV):
        same = (p.read_bytes() if p.exists() else None) == art_before[p]
        print(f"\n真产物 {p.name}:"
              f" {'未被动过 ✅' if same else '!!! 被改动了 —— 立刻停下'}"
              f"(sha256={_sha256(p)[:16]})")
        if not same:
            problems += 1
    print(f"{'=' * 60}\n问题数:{problems}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
