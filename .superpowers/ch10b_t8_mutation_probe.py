"""T8 变异探针:把实现改坏,确认**这条测试真的会红**、以及**红在哪一条**。

⚠️ **同时只许一个人跑这个探针。** 复审在 T6 踩过:两个运行器跑在同一棵树上,
一个把另一个的变异当成了「原始」⇒ 同一天先报 `问题数:0`、后报 `问题数:2`,
而「逐字节还原」的日志**照样打印**(那是还原到了被污染的那一份)。

规矩(本仓 ch07 全章复盘的元教训,逐条照做):
- 锚点**锚在代码上,不锚在注释上**,且每次变异后**断言锚点命中数恰好为 1**;
- **绝不用 `Path.write_text` 还原**(Windows 上会把 LF 翻成 CRLF,本仓栽过两次)
  ⇒ 全程**字节**读写,并在 `finally` 里用 `read_bytes() == 原始字节` 复核;
- **绝不把证据输出接进任何截断/过滤管道**;看不到 `N passed|failed` 就打 `!!!`;
- **一条测试一次 pytest 调用** —— 汇总行的 `N failed` 说不清「是哪一条红」,
  而这次探针的全部价值就在于「**红在哪条**」;
- **真产物一个字节都不许动**:开跑与收工时都量 `evals/topic/` 那三份冻结产物的 sha256
  (`train.jsonl` 是 Task 9 的训练输入;`topic_test.jsonl` 是验收 ① 报数的唯一依据)。

覆盖本任务的三处订正:
- **11-B**(冻结产物没有装置守着)⇒ M1 / M2 / M3 三个变异;
- **11-C**(循环下标当种子)⇒ M4 / M5 / **M16**;
- **11-D**(那条断言零判别力)⇒ M6 + 一段**直接读数**(见文件末尾的 `_note_11d`);
- **11-A**(`--limit` 那把脚枪)⇒ M13。

**订正轮 1**(复审的 F1 / F2 / F3)⇒ **M17 / M18 / M19**,并加了「**红在那一句**」的锚点:
每个变异可以带第三格 `anchor` —— 红出来的输出里**必须**含这段字(断言消息),否则报 `!!!`。
复审指出的正是这件事:**「收集成功、但测试因无关原因失败」也会被算成红**,判别力是零。
(锚点比对要 `-X utf8`:`_run` 里钉了,否则中文断言消息在 cp936 管道上是乱码、永远对不上。)

用法:
    .venv/Scripts/python.exe -X utf8 .superpowers/ch10b_t8_mutation_probe.py
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = "tests/test_topic_labeling.py"

LAB = ROOT / "app" / "topic" / "labeling.py"
SCR = ROOT / "scripts" / "prepare_topic_data.py"
TEST = ROOT / "tests" / "test_topic_labeling.py"

#: 三份**冻结产物** —— 本轮一个字节都不许动。
FROZEN = [
    ROOT / "evals" / "topic" / "train.jsonl",
    ROOT / "evals" / "topic" / "val.jsonl",
    ROOT / "evals" / "topic" / "topic_test.jsonl",
]

SRC = (LAB, SCR, TEST)

T_DRIFT = f"{T}::test_label_drift_detects_changed_count"
T_DRIFT_SET = f"{T}::test_label_drift_is_set_semantics_on_the_other_two_faces"
T_REPRO = f"{T}::test_inject_typo_is_deterministic"
T_SEED_USED = f"{T}::test_inject_typo_uses_the_seed_to_choose_among_the_candidates"
T_CHANGES = f"{T}::test_inject_typo_actually_changes_something"
T_NEVER_EMPTY = f"{T}::test_inject_typo_never_empties_the_text"
T_SEED_ID = f"{T}::test_typo_seed_comes_from_the_row_id"
T_CROSSPROC = f"{T}::test_typo_seed_is_stable_across_processes"
T_FROZEN = f"{T}::test_augment_appends_augmented_rows_and_leaves_the_frozen_artifacts_byte_identical"
T_READ_GUARD = f"{T}::test_augment_refuses_to_read_a_frozen_artifact"
T_WRITE_GUARD = f"{T}::test_augment_refuses_to_write_a_frozen_artifact"
T_ORDER = f"{T}::test_augment_gives_the_same_typo_no_matter_the_row_order"
T_DRIFTED = f"{T}::test_augment_drops_the_rows_whose_labels_drifted"
T_LIMIT = f"{T}::test_augment_limit_only_samples_the_rewriting"
T_MALFORMED = f"{T}::test_augment_reports_a_malformed_answer_without_counting_it_as_drift"
T_NOT_JSON = f"{T}::test_augment_keeps_a_json_syntax_failure_out_of_the_drift_count"
T_PATHS = f"{T}::test_the_augment_paths_point_at_the_right_files"
T_DISPATCH = f"{T}::test_main_dispatches_augment_and_forwards_the_limit"
T_LIMIT_OTHER = f"{T}::test_main_refuses_limit_on_the_other_steps"

#: (名字, 文件, 要替换的原文, 替换成什么, [(测试 id, 期望的 pytest 结果), …])
MUTATIONS = [
    # ---- ★ 11-B:那句「验证集与测试集一条都不许动」此前只是散文 ----
    (
        "M1 ★11-B 输入守卫拆掉(有人改成「train + val 一起增强」)",
        SCR,
        "    assert train_path.name == TRAIN_NAME, (\n",
        "    assert True or train_path.name == TRAIN_NAME, (   # 变异:守卫拆掉\n",
        # 红在哪句:`pytest.raises(AssertionError)` **没有**收到异常
        [(T_READ_GUARD, "failed", "DID NOT RAISE"),
         # 证据:另外两条**照样绿** ⇒ 少了输入守卫就没别的东西守得住它
         (T_FROZEN, "passed"), (T_WRITE_GUARD, "passed")],
    ),
    (
        "M2 ★11-B 输出守卫拆掉(产物可以写成 val/topic_test)",
        SCR,
        "    assert out_path.name == AUGMENTED_NAME, (\n",
        "    assert True or out_path.name == AUGMENTED_NAME, (   # 变异:守卫拆掉\n",
        # 红在哪句:`topic_test.jsonl` 的字节前后不同(真的被覆盖了)
        [(T_WRITE_GUARD, "failed", "DID NOT RAISE"),
         (T_FROZEN, "passed"), (T_READ_GUARD, "passed")],
    ),
    (
        "M3 ★11-B **绕过两条名字守卫**偷写 val.jsonl(spec 里那句「静默的破坏」)",
        SCR,
        "            f.flush()\n",
        "            f.flush()\n"
        "            (TOPIC_DIR / \"val.jsonl\").open(\"a\", encoding=\"utf-8\").write(\"x\\n\")\n",
        # ⚠️ 两条名字守卫都对 `train_path`/`out_path` 生效、这里一个都不碰
        #    ⇒ **只有** sha256 那条断言会红。这正是「逐字节相同」那句的存在理由。
        [(T_FROZEN, "failed", "被改动了"),
         # 证据:别的用例不读冻结产物 ⇒ 它们对这一条**零判别力**
         (T_LIMIT, "passed"), (T_DRIFTED, "passed")],
    ),
    # ---- ★ 11-C:种子要表示「这一行是谁」,不是「它排第几」----
    (
        "M4 ★11-C 种子换成**循环下标**(本任务原稿的写法)",
        SCR,
        'inject_typo(new_text, typo_seed(r["id"]))',
        # ⚠️ **不许在这行加行尾注释**:那个逗号在**下一行**,注释会把逗号一起吃掉
        #    ⇒ 源码变成语法错 ⇒ 「红」是 **SyntaxError 的红**,判别力一点也没验到。
        #    (第一版就是这么写的,`_run` 的 SyntaxError 检查把它拦下来了 ——
        #     ch07 那条「正则吃掉闭括号让 JS 语法崩而被当成 RED」的同款。)
        "inject_typo(new_text, i)",
        # 红在哪句:`seen["正序"] == seen["倒序"]`(同一 id 在两份行序里注出不同的错别字)
        # ⚠️ **第一次跑时这一条是绿灯** —— 当时夹具只有四行,下标种子恰好全程与
        #    id 种子撞出同一个候选 ⇒ 那条断言对 11-C **零判别力**。
        #    夹具加了两行(**M14** 反过来钉住那段自检)之后才有判别力。
        #    这是本次任务最值钱的一处发现:**「跑过变异」不等于「变异能被测出来」**。
        [(T_ORDER, "failed", "同一行在两份行序里"),
         # 证据:其余各条**全绿** ⇒ 「行序无关」只有那一条守得住
         (T_FROZEN, "passed"), (T_SEED_ID, "passed")],
    ),
    (
        "M5 ★11-C 换回内置 `hash()`(本仓硬约束)",
        LAB,
        '    return int.from_bytes(hashlib.sha256(row_id.encode("utf-8")).digest(), "big")\n',
        "    return hash(row_id)   # 变异:对 str 每进程随机化\n",
        # 红在哪句:两个子进程算出的种子不同
        [(T_CROSSPROC, "failed", "两个进程算出的种子不同"),
         # ⚠️ **订正轮 1 之后这一格也红了**(F3 加的 ⑧ 断言比的是 `typo_seed(id)`,
         #    而 `hash()` 给的是另一个数)⇒ 判别力比上一轮更强。
         #    但「`hash()` 换进程就不稳」这件事**只有跨进程那条测得出来** ——
         #    ⑧ 抓的是「不是 `typo_seed`」,另一件事。
         (T_ORDER, "failed", "种子没从 id 派生"),
         # 证据:**同进程内的 `typo_seed` 专属用例全绿** ——
         # 这正是本仓那句「同进程的测试完全测不出来」
         (T_SEED_ID, "passed")],
    ),
    # ---- ★ 11-D:那条「不许注成空串」的断言 ----
    (
        "M6 ★11-D `inject_typo` 改成恒等函数(一条错别字都不注入)",
        LAB,
        "    src, dst = rng.choice(candidates)\n    return text.replace(src, dst, 1)\n",
        "    return text   # 变异:恒等\n",
        # 红在哪句:`out != "退货"`(加固后的那半);顺带 `any(c != …)`
        [(T_NEVER_EMPTY, "failed", "那等于一条错别字都没注入"), (T_CHANGES, "failed"),
         # 证据:**原稿那条断言在恒等实现下照样绿** —— 直接读数见 `_note_11d()`
         (T_REPRO, "passed")],
    ),
    # ---- `label_drift`:集合语义的三个面 ----
    (
        "M7 `label_drift` 恒判漂移(把合法增强全丢光)",
        LAB,
        "    return set(before) != set(after)\n",
        "    return True   # 变异:恒判漂移\n",
        [(T_DRIFT, "failed", "顺序不算变化"), (T_DRIFT_SET, "failed"),
         (T_DRIFTED, "failed")],
    ),
    (
        "M8 `label_drift` 改成**比长度**(重复被算成漂移)",
        LAB,
        "    return set(before) != set(after)\n",
        "    return len(before) != len(after)   # 变异:比长度\n",
        # ⚠️ 证据:**brief 给的那条 `test_label_drift_detects_changed_count` 照样绿**
        #    ⇒ 少了 `test_label_drift_is_set_semantics_…`,那个错法没有任何东西守着。
        [(T_DRIFT_SET, "failed", "重复不该被算成漂移"), (T_DRIFT, "passed")],
    ),
    # ---- `_rewrite` 的两道闸 ----
    (
        "M9 改写后的句子**不过清洗**(训练语料两种口径并存)",
        SCR,
        "    cleaned = clean(new_q)\n",
        "    cleaned = new_q   # 变异:不洗\n",
        [(T_FROZEN, "failed", "增强行的题面没过清洗"), (T_MALFORMED, "passed")],
    ),
    (
        "M10 形状闸失效后**返回空标签**(原稿那种错法:把故障记成漂移)",
        SCR,
        "    if (not isinstance(new_q, str) or not isinstance(new_labels, list)\n"
        "            or not all(isinstance(x, str) for x in new_labels)):\n"
        "        return row[\"question\"], row[\"labels\"], True\n",
        "    if (not isinstance(new_q, str) or not isinstance(new_labels, list)\n"
        "            or not all(isinstance(x, str) for x in new_labels)):\n"
        "        return \"\", [], True   # 变异:返回空标签 ⇒ 被算成「漂移」\n",
        # 红在哪句:那两行按原句写回的断言(它们被 `label_drift` 判成漂移丢掉了)
        [(T_MALFORMED, "failed", "解析失败的行被丢了"), (T_FROZEN, "passed")],
    ),
    # ---- ★ 11-A:`--limit` 与命令行 ----
    (
        "M11 `main()` 不转发 `--limit`(小样变成 75 分钟全量)",
        SCR,
        "        asyncio.run(augment(args.limit or None))\n",
        "        asyncio.run(augment())   # 变异:limit 丢掉\n",
        [(T_DISPATCH, "failed", "没被转发到"),
         # 证据:直接调 `p.augment(limit=…)` 的那条**照样绿**
         (T_LIMIT, "passed")],
    ),
    (
        "M12 `--limit` 给了别的步骤时**静默忽略**",
        SCR,
        '    if args.limit and args.step != "augment":\n',
        '    if False and args.limit and args.step != "augment":   # 变异:不拦\n',
        [(T_LIMIT_OTHER, "failed", "DID NOT RAISE")],
    ),
    (
        "M13 ★11-A 那把脚枪:**就地裁掉 train.jsonl**",
        SCR,
        "    todo = rows[:limit] if limit else rows\n",
        "    todo = rows[:limit] if limit else rows\n"
        "    if limit:   # 变异:裁的是语料本身(冻结的训练集)——Task 9 的输入\n"
        "        train_path.write_text(\n"
        "            \"\\n\".join(json.dumps(r, ensure_ascii=False) for r in todo) + \"\\n\",\n"
        "            encoding=\"utf-8\")\n",
        # 红在哪句:`train.jsonl` 跑前跑后的**字节相同**那句
        [(T_LIMIT, "failed", "被就地改了"), (T_DISPATCH, "passed")],
    ),
    # ---- ★ 用例自检本身:夹具失去判别力时,**它**要先红 ----
    (
        "M14 ★夹具截回四行(判别力归零)—— 用例自检必须自己红",
        TEST,
        # ⚠️ 锚点是**夹具的那两行**(后加的、为了判别力的那两行);删掉它们。
        '    {"id": "r-0005", "question": "颜色发错了,能换货吗", "source": "chat",\n'
        '     "provenance": "real", "labels": ["质量问题", "退换货"], "split": "train"},\n'
        '    {"id": "s-0006", "question": "保修期内坏了,运费谁出", "source": "gen",\n'
        '     "provenance": "synthetic", "labels": ["保修维修", "运费"], "split": "train"},\n',
        "",
        # 红在哪句(⚠️ **是自检那一句,不是主断言**):
        #   `assert by_index["正序"] != by_index["倒序"]` ——
        #   「夹具没有判别力……」那段。这正是它在 11-C 上的用途:
        #   夹具一旦悄悄失去判别力,自检先报出来,而不是让主断言变成一条**恒真**。
        #   (M4 第一次跑时这里报的是绿灯 —— 那时自检用的是 0 起的下标,
        #    与实现的 `enumerate(todo, 1)` 差一,于是它**替假绿背了书**。)
        [(T_ORDER, "failed", "夹具没有判别力")],
    ),
    # ================= 订正轮 1(F1 / F2 / F3)=================
    (
        "M17 ★F1 JSON **语法**那一支返回空标签(复审实测:67 条全绿)",
        SCR,
        "    except json.JSONDecodeError:\n"
        "        return row[\"question\"], row[\"labels\"], True\n",
        "    except json.JSONDecodeError:\n"
        "        return \"\", [], True   # 变异:语法错也返回空标签 ⇒ 被算成漂移\n",
        # 红在哪句:那条新用例里 `set(aug) == {…}` 的**断言消息**
        #   (「语法没解出来的那行被丢了(该按原句写回)」)。
        # ⚠️ 证据:**形状那一支的用例照样绿** —— 两支各要一条用例,这就是 F1 的全部理由。
        [(T_NOT_JSON, "failed", "语法没解出来的那行被丢了"),
         (T_MALFORMED, "passed")],
    ),
    (
        "M18 ★F2 `TOPIC_DIR` 少一层(`ROOT / \"evals\"`)",
        SCR,
        'TOPIC_DIR = ROOT / "evals" / "topic"\n',
        'TOPIC_DIR = ROOT / "evals"   # 变异:少一层\n',
        # 红在哪句:新加的那句**字面量**断言(消息里有「TOPIC_DIR 指错目录了」)。
        # ⚠️ 复原先验过:少了那句字面量断言,这一格**全绿**(同义反复)。
        [(T_PATHS, "failed", "TOPIC_DIR 指错目录了")],
    ),
    (
        "M19 ★F3 种子换成**常数 0**(复审实测:67 条全绿)",
        SCR,
        'inject_typo(new_text, typo_seed(r["id"]))',
        "inject_typo(new_text, 0)",
        # 红在哪句:行序无关那条里 ⑧ 的 `expected` 断言(消息里有「种子没从 id 派生」)。
        # ⚠️ **行序那条断言自己是绿的**(常数种子当然与行序无关)—— 所以 ⑧ 必须存在。
        [(T_ORDER, "failed", "种子没从 id 派生")],
    ),
    # ---- 两条「替身/断言本身有没有判别力」的钉子 ----
    (
        "M15 `_rewrite` 不把这一行的**原标签**喂进 prompt",
        SCR,
        '        labels=" / ".join(row["labels"]),\n',
        '        labels="",   # 变异:原标签不进 prompt\n',
        # 红在哪句:替身里那句 `assert joined in prompt`(spec §5.4 的第二条硬约束)
        # —— 少了它,「逐条带原标签」这件事**没有任何东西守着**。
        [(T_FROZEN, "failed", "prompt 里没有这一行的原标签"), (T_LIMIT, "failed")],
    ),
    (
        "M16 `typo_seed` 返回常数(id 被丢掉)",
        LAB,
        '    return int.from_bytes(hashlib.sha256(row_id.encode("utf-8")).digest(), "big")\n',
        "    return 0   # 变异:与 id 无关\n",
        # 红在哪句:`len(set(seeds)) == len(ids)`(400 个 id 撞成一个种子)。
        # ⚠️ **订正轮 1 之前这一格的 `T_ORDER` 是绿的**(常数种子当然与行序无关)
        #    ⇒ 「可复现」与「与 id 有关」是两件事,少了 F3 加的那条 ⑧ 断言就分不开 ——
        #    它现在红在「种子没从 id 派生」。
        [(T_SEED_ID, "failed", "撞出了重复种子"),
         (T_ORDER, "failed", "种子没从 id 派生")],
    ),
]

SUMMARY_RE = re.compile(r"(\d+) (passed|failed)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(rel: str):
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def _revision_report(snapshot) -> None:
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
    print("三份**冻结产物**(本轮一个字节都不许动):")
    for p in FROZEN:
        print(f"  · {p.name}  {_sha256(p)[:16]}")


def _hygiene_ok(snapshot) -> bool:
    for p, b in snapshot.items():
        if p.read_bytes() != b:
            print(f"  !!! 树被污染:{p.relative_to(ROOT)} 与开跑时不同"
                  f" —— 是不是**有别人在同一棵树上跑**?这次变异不作数")
            return False
    return True


def _run(test_id: str):
    # ⚠️ `-X utf8` 是给**断言文本的锚点比对**用的:本机 locale 是 cp936,
    # 不钉的话 pytest 把中文断言消息按 GBK 写进管道、这边按 UTF-8 解 ⇒ 全是乱码,
    # 于是「红在哪句」这一步永远对不上。它只影响本进程的编码,不改任何测试语义
    # (子进程的编码各自由它们自己的 `-X utf8` 决定,见那两条用例的注释)。
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", test_id, "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def _note_11d() -> None:
    """11-D 的**直接读数**(不经过 pytest)。

    原稿那条断言是 `assert inject_typo("退货", s).strip()`,而它的实现是
    `text.replace(src, dst, 1)`(dst 非空、src ≠ dst)⇒ 输入非空就**结构上不可能**
    返回空。所以「恒等函数」这种一眼就该被抓的错实现,在原稿那条断言下**照样绿**。
    这里把两版断言各跑一遍,把读数打出来。
    """
    print("\n=== 11-D 的直接读数(原稿那条断言 vs 加固后那条)")

    def identity(text: str, seed: int) -> str:
        return text          # ← 「恒等函数」那种错实现(一条错别字都不注入)

    brief_ok = all(identity("退货", s).strip() for s in range(50))
    changed = [identity("退货", s) != "退货" for s in range(50)]
    print(f"  · 原稿断言 `assert inject_typo('退货', s).strip()` 在恒等实现下:"
          f"{'**仍然通过**(判别力 0)' if brief_ok else '红'}")
    print(f"  · 加固后那句 `out != '退货'`:恒等实现下 {sum(changed)}/50 个种子"
          f"「变了」⇒ 断言**红**")
    if not brief_ok or any(changed):
        print("  !!! 与预期不符(原稿那条断言居然红了 / 恒等实现居然变了?)—— 先复核 `_TYPO_MAP`")


def main() -> None:
    snapshot = {p: p.read_bytes() for p in SRC}
    frozen_before = {p: p.read_bytes() for p in FROZEN}
    _revision_report(snapshot)
    _note_11d()

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
            for exp in expects:
                test_id, want = exp[0], exp[1]
                #: 订正轮 1 的复审指出:**「收集成功、但测试因无关原因失败」也会被算成红**。
                #  第三格是**期望红在哪句**的锚点(断言消息里必有的一段字);给了它就要求
                #  红出来的那份输出**真的含这段字**,否则报 `!!!` 并计问题数。
                #  (期望绿的那几项是「零判别力」的证据,锚点无意义 ⇒ 一律留空。)
                anchor = exp[2] if len(exp) > 2 else None
                proc = _run(test_id)
                # ⚠️ **语法错的红不算红**(ch07 全章复盘那条):变异把源码改成语法错时,
                #    pytest 照样打 `1 failed` —— 而它一个断言都没跑到。
                if "SyntaxError" in (proc.stdout + proc.stderr):
                    print(f"  !!! {test_id.split('::')[-1]}:变异把源码改成了**语法错**"
                          f" ⇒ 这次读数不作数(「红」不是任何断言的功劳)")
                    problems += 1
                    continue
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
                where = ""
                if got == "failed" and anchor is not None:
                    # ⚠️ 这一步问的是**「红在那一句上吗」**,不是「红没红」。
                    #    (复审自己在 T8 的探针上指出:有 `1 failed` 不够 ——
                    #     无关理由失败的红,判别力是零。)
                    if anchor in proc.stdout:
                        where = f"   红在含「{anchor}」那句 ✅"
                    else:
                        where = f"   !!! 红了但**不是**期望那句(找「{anchor}」没找到)"
                        problems += 1
                print(f"  {mark}  {test_id.split('::')[-1]}  (期望 {want}){flag}{where}"
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

    print("\n三份冻结产物:")
    for p in FROZEN:
        same = p.read_bytes() == frozen_before[p]
        print(f"  · {p.name}  {'未被动过 ✅' if same else '!!! 被改动了 —— 立刻停下'}"
              f"  sha256={_sha256(p)[:16]}")
        if not same:
            problems += 1
    print(f"{'=' * 60}\n问题数:{problems}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
