"""主题分类的语料加工(四步)。子命令:
    collect   三源合流 → evals/topic/corpus.jsonl
    split     分层抽样 80/10/10 + 冻结测试集
    augment   数据增强(**只扩训练集**)

用法:
    .venv/Scripts/python.exe scripts/prepare_topic_data.py collect
"""

import argparse
import asyncio
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.db.base import get_engine
# ⚠️ 这一行是**同源守卫**要求的(tests/test_topic_clean.py::test_both_sides_use_the_same_clean):
#    训练侧必须与推理侧用**同一份** `clean`。它在本文件里也被真的用到(数「洗后为空」那几行),
#    不是一条只为过守卫而存在的 import。
from app.topic.clean import clean
from app.topic.labeling import dedupe_questions, is_unusable_target, stratified_split, train_test_overlap

ROOT = Path(__file__).resolve().parents[1]
TOPIC_DIR = ROOT / "evals" / "topic"
OUT = TOPIC_DIR / "corpus.jsonl"
TESTING_MD = ROOT / "evals" / "测试集.md"

#: 切分端的输入。两个都**不入库**(与 `corpus.jsonl` / `synthetic.jsonl` 同例,
#: 见 `dev-notes/ch10.md`):能进 git 的是切分的**产物**。
#: `reviewed.jsonl` 是**人工劳动的唯一记录**,它入库(与 `labels/trainval.csv` 同理)。
PRELABELED = TOPIC_DIR / "prelabeled.jsonl"
REVIEWED = TOPIC_DIR / "reviewed.jsonl"

#: ⚠️ **订正 9-B**:测试集产物叫 `topic_test.jsonl` —— 不是 `test.jsonl`。
#: 本任务其余几处(`Interfaces` / `export-test` / 评测脚本 / `git add`)全按它读写;
#: 名字写错**不报错**,只是评测读到一个空的/不存在的文件 ⇒
#: **用户那 120 条的复核对指标零影响,而报告照常打印**。做成常量,别在循环里拼字符串。
TEST_NAME = "topic_test.jsonl"


def _out(message: str) -> None:
    """钉死输出编码:本机 `sys.stdout.encoding` 是 **gbk**,而 `⚠` / `†` 这类字符
    不在 GBK 里 —— 输出里哪天多一个就会 `UnicodeEncodeError`(ch09 那条「脚本自己
    喷错误行」的先例)。照 `scripts/build_kb.py` 的房规把边界钉成 UTF-8,不依赖控制台 codec。

    ⚠️ **实测(2026-09-26)**:本文件当前这几行(中文、`→`)**在 GBK 下恰好都编得出来**
    ⇒ 这条钉的是「以后加个 `⚠` 不会突然崩」,不是「今天就崩」。别把它读成已经发生过。"""
    sys.stdout.buffer.write((message + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


async def _from_db() -> list[dict]:
    """池子 + 对话里的用户话。

    两处都取**去重前**的全量:去重交给 `dedupe_questions` 一处做 ——
    两个地方各去一遍的话,「按什么去重」这件事就有两份实现。

    ⚠️ 两条查询的 `source` **必须不同**(`pool` / `chat`)。写成一个值的话,
    「池子 > 对话 > 测试集.md」这条优先级就只剩一条来源,分布也读不出真数。
    """
    rows: list[dict] = []
    eng = get_engine()
    async with eng.connect() as conn:
        for source, q in (
            ("pool", text("SELECT question FROM low_confidence_questions ORDER BY id")),
            ("chat", text("SELECT content FROM messages WHERE role='user' ORDER BY id")),
        ):
            for (value,) in (await conn.execute(q)).all():
                rows.append({"question": value, "source": source})
    await eng.dispose()
    return rows


def _from_testing_md() -> list[dict]:
    """`evals/测试集.md` 的 300 条**人工写的**政策问句 —— 借用作分类语料。

    它是**检索**评估集,借用不污染检索评估:两个任务不同,
    同一批问句在两处各算各的分母。
    """
    rows: list[dict] = []
    with TESTING_MD.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = (row.get("问题(query)") or "").strip()
            if q:
                rows.append({"question": q, "source": "evalmd"})
    return rows


async def collect() -> None:
    # 来源**优先级顺序**就是这里的顺序:池子 > 对话 > 测试集.md。
    # 去重保留首次出现的那一条,所以顺序决定了重复问题归谁名下。
    rows = (await _from_db()) + _from_testing_md()
    for r in rows:
        r["provenance"] = "real"
    blank = sum(1 for r in rows if not clean(r["question"]))
    kept = dedupe_questions(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, r in enumerate(kept, 1):
            f.write(json.dumps({"id": f"r-{i:04d}", **r}, ensure_ascii=False) + "\n")

    _out(f"写出 {len(kept)} 条 → {OUT}")
    _out(f"  来源分布: {dict(Counter(r['source'] for r in kept))}")
    # 两个数**分开报**:它们的成因不同(空串 vs 与前面某条重复),
    # 合起来报会把「洗后为空只有几条」读成几十条,而那正是选题材时要看的读数。
    _out(f"  读入 {len(rows)} 条:洗后为空 {blank} 条、与前面重复 {len(rows) - blank - len(kept)} 条,都没写出")
    # ⚠️ 这几个数会**随运行次数增长**(验收脚本每跑一次就往池子与对话里写),
    #    所以引用它时必须带日期,别当常量。


def _load_prelabeled() -> dict[str, dict]:
    """`prelabeled.jsonl` → `{id: row}`,并把 `reviewed.jsonl` 里的**覆盖**上去。

    ⚠️ **覆盖那一步是承重的**:不这么做的话,用户改过的标签不进语料 ——
    而它**只影响报表、不影响模型**(训练用的是这份语料)。那正是本仓编目过的
    「看起来做完了、其实没接上」。
    """
    pre: dict[str, dict] = {}
    for line in PRELABELED.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            pre[row["id"]] = row
    if REVIEWED.exists():
        for line in REVIEWED.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                pre[r["id"]] = {**pre[r["id"]], "labels": r["labels"], "human_reviewed": True}
    return pre


def split() -> None:
    """切分并**冻结测试集**。

    ⚠️ 标签以**人改过的**为准(见 `_load_prelabeled`)。

    这个函数里有四处「错了也不报错」的东西,各自对应下面一段:

    1. **两行零标签**(订正 9-E)—— 它们 `labels` 都是 `[]`,而来源不同
       (`r-0049` 模型真判零诉求 / `s-0423` 证据校验机械拒到空)。
       `stratified_split` 把**靶子不可信**的那种先摘掉;这里负责把**去向**打出来 ——
       不打印的话,「那行到底进没进训练」没有任何地方看得见。
    2. **train-vs-test 的完全相同**(订正 9-C)—— 必须**移出训练侧**,否则是数据泄漏。
    3. **头四类在测试集里的条数** —— < 15 的类那条 F1 不成结论,报告里要标 `†`。
    4. **产物名**(订正 9-B)—— `topic_test.jsonl`,不是 `test.jsonl`。
    """
    pre = _load_prelabeled()
    rows = list(pre.values())
    n_human = sum(1 for r in rows if r.get("human_reviewed"))
    _out(f"读入 {len(rows)} 条;其中 {n_human} 条用的是**人工复核过的**标签"
         f"({REVIEWED.name} 的覆盖{'生效' if n_human else '不存在/为空'})")

    parts = stratified_split(rows, test_real=80, test_synth=40)

    # ---- 【第 2 条】train-vs-test 的重复检查(9-C)----
    # 验证集也算**训练侧**:它参与早停与阈值选择,泄漏的后果与训练集同级。
    over = train_test_overlap(parts["train"] + parts["val"], parts["test"])
    exact_ids = set(over["exact_train_ids"])
    near_pairs = over["near_pairs"]
    removed = 0
    if exact_ids:
        for name in ("train", "val"):
            before = len(parts[name])
            parts[name] = [r for r in parts[name] if r["id"] not in exact_ids]
            removed += before - len(parts[name])
    _out(f"  train-vs-test:完全相同 {len(exact_ids)} 条(其中 {removed} 条在训练侧,已移出)、"
         f"近重复 {len(near_pairs)} 对(只报不删 —— 那是启发式判据,删了就是拿判据改数据)")
    # ⚠️ **不许** `assert` 近重复为 0:它是**预期的**(T4 的 prompt 里就有那些例句),
    #    而这条判据是启发式 ⇒ 「报出来给人看」才是它的全部用途。
    for tid, xid, score in near_pairs[:10]:
        _out(f"    {tid} ~ {xid}  相似度 {score:.3f}")
    if len(near_pairs) > 10:
        _out(f"    …另有 {len(near_pairs) - 10} 对")

    # ---- 【第 1 条】零标签行的去向(9-E)----
    # ⚠️ 放在**摘除之后**:要报的是「落在哪一侧」,**摘除也会改这个答案**。
    # ⚠️ 处置靠**与 `stratified_split` 里那次摘除同一个谓词**(`is_unusable_target`),
    #    不在这里再写一遍 `labels == [] and rejected_labels` ——
    #    两处各自维护的判据就是本仓记过的漂移形状。
    zero = [r for r in rows if not (r.get("labels") or [])]
    if zero:
        side = {r["id"]: name for name, items in parts.items() for r in items}
        _out(f"零标签行 {len(zero)} 条 —— **来源不同 ⇒ 处置不同**,逐条列去向:")
        for r in zero:
            if is_unusable_target(r):
                where = "**已排除**(靶子不可信:零标签 + rejected_labels 非空)"
            else:
                where = side.get(r["id"], "**不在任何一份里**(不该发生,去查 split)")
            _out(f"  {r['id']} 「{r['question']}」 rejected_labels={r.get('rejected_labels')}"
                 f" → {where}")

    # ---- 【第 4 条】写出三份(测试集的名字见订正 9-B)----
    for name, items in parts.items():
        path = TOPIC_DIR / (TEST_NAME if name == "test" else f"{name}.jsonl")
        path.write_text(
            "\n".join(json.dumps({**r, "split": name}, ensure_ascii=False) for r in items) + "\n",
            encoding="utf-8",
        )
        _out(f"{name}: {len(items)} 条 → {path}")

    # 三层报告的构成核对(spec §8.4)—— 数不对说明硬条件没满足,响亮地报。
    c = Counter(r["provenance"] for r in parts["test"])
    assert c["real"] == 80 and c["synthetic"] == 40, f"测试集构成不对:{dict(c)}"
    from app.topic.taxonomy import HEAD_LABELS
    per = Counter(lb for r in parts["test"] for lb in r["labels"])
    thin = [lb for lb in HEAD_LABELS if per[lb] < 15]
    if thin:
        _out(f"  ⚠️ 头四类里这些在测试集不到 15 条:{thin} —— 它们的 F1 **不成结论**,"
             f"报告里要标 †(spec §8.3)")
    # ⚠️ 三个数**分开打**:`human_reviewed` 的条数(人工劳动的覆盖面)与
    #    测试集构成(报数依据)是两件事,合起来报会让人以为「测试集也审过了」。
    _out(f"  ⚠️ 以上是**训练侧**的读数;测试集 {len(parts['test'])} 条要 100% 人工过"
         f"(`export-test` → 用户改 → `import-test`),那之前它**还不能用来报数**。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["collect", "split", "augment"])
    args = ap.parse_args()
    if args.step == "collect":
        asyncio.run(collect())
    elif args.step == "split":
        split()
    else:
        raise SystemExit(f"{args.step} 还没实现(由后续任务补上)")


if __name__ == "__main__":
    main()
