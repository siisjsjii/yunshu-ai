"""主题分类的语料加工(四步)。子命令:
    collect   三源合流 → evals/topic/corpus.jsonl
    split     分层抽样 80/10/10 + 冻结测试集
    augment   数据增强(**只扩训练集**)

用法:
    .venv/Scripts/python.exe scripts/prepare_topic_data.py collect
    .venv/Scripts/python.exe scripts/prepare_topic_data.py augment --limit 20   # 小样:看漂移率
    .venv/Scripts/python.exe scripts/prepare_topic_data.py augment               # 全量(约 75 分钟)
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

from app.config import get_settings
from app.db.base import get_engine
# ⚠️ 与 `scripts/prelabel_topics.py` **同款**:`_rewrite` 要用同一个抽取模型工厂
#    (temperature=0),不许在这里另建一个 `ChatOpenAI`。
from app.llm import create_extract_model
# ⚠️ 这一行是**同源守卫**要求的(tests/test_topic_clean.py::test_both_sides_use_the_same_clean):
#    训练侧必须与推理侧用**同一份** `clean`。它在本文件里也被真的用到(数「洗后为空」那几行),
#    不是一条只为过守卫而存在的 import。
from app.topic.clean import clean
from app.topic.labeling import (
    dedupe_questions,
    inject_typo,
    is_unusable_target,
    label_drift,
    stratified_split,
    train_test_overlap,
    typo_seed,
)
from app.topic.taxonomy import render_taxonomy_for_prompt

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

#: 增强的**唯一**输入(spec §5.4)。⚠️ 它同时是**冻结产物** —— Task 9 拿它训练,
#: 所以 `augment` **只读**它,而且**就地改它**是本任务明令避开的那把脚枪(订正 11-A)。
TRAIN_NAME = "train.jsonl"
TRAIN = TOPIC_DIR / TRAIN_NAME
#: 增强的产物 —— **新文件,不是就地改 `TRAIN`**。
#: 名字与输入逐字不同,也**不是**那两份冻结产物(`val.jsonl` / `topic_test.jsonl`)。
AUGMENTED_NAME = "train_augmented.jsonl"
TRAIN_AUGMENTED = TOPIC_DIR / AUGMENTED_NAME


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
    # ⚠️ **逐份计数**(不是只记总数):「val 那一侧到底摘了几条」是**验收集自检**要读的
    #    读数 —— 它若恒为 0,就说明这份语料再也分不出「摘除只走 train」那种错法
    #    (订正轮 1 的 I2;tests 里 ③a 读它)。
    dropped = {name: 0 for name in ("train", "val")}
    if exact_ids:
        for name in ("train", "val"):
            before = len(parts[name])
            parts[name] = [r for r in parts[name] if r["id"] not in exact_ids]
            dropped[name] = before - len(parts[name])
    removed = sum(dropped.values())
    # ⚠️ **订正轮 1 的 M2**:原来写「完全相同 N 条(**其中** M 条在训练侧,已移出)」——
    #    「其中」是**空的**:`train_test_overlap` 收到的就是训练侧,所以被检出的每一条
    #    都在 train/val 里,M 恒等于 N。两个数并排只会让人以为「有些不在训练侧」。
    #    现在改成 `assert`:它若**不等**,说明摘除只走过一侧(见上面那个 for),那才是真问题。
    assert removed == len(exact_ids), (
        f"摘除数 {removed} 与检出数 {len(exact_ids)} 对不上 —— 训练侧有一半没被摘"
        f"(`for name in (…)` 少了哪一侧?)"
    )
    _out(f"  train-vs-test:完全相同 {len(exact_ids)} 条(**全部在训练侧**,已移出)、"
         f"近重复 {len(near_pairs)} 对(只报不删 —— 那是启发式判据,删了就是拿判据改数据)")
    _out(f"    train 摘 {dropped['train']} 条、val 摘 {dropped['val']} 条"
         f"(⚠️ **val 那一侧不许恒为 0** —— 恒 0 说明语料分不出「摘除只走 train」那种错法)")
    # ⚠️ **不许** `assert` 近重复为 0:它是**预期的**(T4 的 prompt 里就有那些例句),
    #    而这条判据是启发式 ⇒ 「报出来给人看」才是它的全部用途。
    for tid, xid, score in near_pairs[:10]:
        _out(f"    {tid} ~ {xid}  相似度 {score:.3f}")
    if len(near_pairs) > 10:
        _out(f"    …另有 {len(near_pairs) - 10} 对")

    # ---- 【第 1 条】零标签行的去向(9-E)----
    # ⚠️ 放在**摘除之后**:要报的是「落在哪一侧」,**摘除也会改这个答案**。
    # ⚠️ **「落到哪儿」是读数,「为什么」才是谓词** —— 两者**不许共用一个变量**
    #    (订正轮 1 的 **I1**,复审实测):原来 `is_unusable_target(r)` 为真时,
    #    `where` 被赋成**硬编码**的「已排除」、**从不查 `side`** ⇒ 摘除失效时
    #    (`usable = list(rows)`)打印**照旧**说「已排除」,而那行实际上在 `train`。
    #    这一行打印是 dev-notes / 报告里「零标签两行去向」那个读数的**唯一来源**
    #    ⇒ 它属于本仓「读数看着完全正常」那一族。
    # ⚠️ 判据仍**只有一处实现**(`is_unusable_target`),但这里只用它给**理由**;
    #    落位一律从 `parts` 里查 —— 两边分开,谁也盖不住谁。
    zero = [r for r in rows if not (r.get("labels") or [])]
    if zero:
        side = {r["id"]: name for name, items in parts.items() for r in items}
        _out(f"零标签行 {len(zero)} 条 —— **来源不同 ⇒ 处置不同**,逐条列去向:")
        for r in zero:
            # ⚠️ 这一句**只读落位**,一个字的谓词成分都没有 —— 它才是读数。
            actual = side.get(r["id"], "**不在任何一份里**")
            # ⚠️ 谓词只给**理由**,不给落位 —— 两者分开,谁也盖不住谁(I1)。
            reason = ("靶子不可信:零标签 + rejected_labels 非空,被 stratified_split 排除"
                      if is_unusable_target(r) else "模型真判了零诉求(不是处理产物)")
            _out(f"  {r['id']} 「{r['question']}」 rejected_labels={r.get('rejected_labels')}"
                 f" → {actual}({reason})")

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
    # ⚠️ **`†` 是按层、对全部 17 类算的**(订正轮 1 的 **M3**,controller 裁定):
    #    spec §8.3 的判据是「**每类** support < 15」,不是「头四类 < 15」;
    #    而 §8.4 的三列各有**自己的一套** `†`(只看真实那 80 条与只看合成那 40 条
    #    的分母差一倍)。这里三个层都算,Task 8/9 生成报告时**照这个来**。
    from app.topic.taxonomy import LABELS
    for layer, subset in (
        ("全体 120", parts["test"]),
        ("只看真实 80", [r for r in parts["test"] if r["provenance"] == "real"]),
        ("只看合成 40", [r for r in parts["test"] if r["provenance"] == "synthetic"]),
    ):
        per = Counter(lb for r in subset for lb in r["labels"])
        thin = [lb for lb in LABELS if per[lb] < 15]
        ok = [f"{lb} {per[lb]}" for lb in LABELS if per[lb] >= 15]
        _out(f"  † {layer}:17 类里 **{len(thin)} 类 support < 15**(那一行 F1 不成结论);"
             f"够 15 的只有 {ok or '—'}")
        if thin:
            _out(f"      标 † 的:{thin}")
    # ⚠️ 三个数**分开打**:`human_reviewed` 的条数(人工劳动的覆盖面)与
    #    测试集构成(报数依据)是两件事,合起来报会让人以为「测试集也审过了」。
    _out(f"  ⚠️ 以上是**训练侧**的读数;测试集 {len(parts['test'])} 条要 100% 人工过"
         f"(`export-test` → 用户改 → `import-test`),那之前它**还不能用来报数**。")


REWRITE_PROMPT = """下面是一句电商客服场景里的买家提问,以及它**已经标注好**的标签。

原句:{question}
标签:{labels}
这些标签的含义(权威类目表):

{taxonomy}

请把这句话**换一种说法**,要求:
1. **诉求的个数与类别一个都不许变** —— 原来是 2 个诉求,改写后还是那 2 个;
   原来有「尺码」,改写后这句话里「买大了」这层意思必须还在。
2. 同义词替换、句式调整(比如把陈述句改成疑问句),不要只改了标点。
3. 不要引入新的诉求,也不要删掉任何一个。
4. 像真人在客服窗口打出来的。

输出一个 JSON 对象,两个字段:
- question:改写后的句子
- labels:改写后这句话应当打上的标签数组(**必须与原标签集合完全相同**)

只输出 JSON,不要别的内容。"""


async def _rewrite(model, row: dict) -> tuple[str, list[str], bool]:
    """返回 `(改写后的句子, 它自己报的标签, 这一条是不是**没解析出来**)`。

    ⚠️ 调用方会用 `label_drift` 比对原标签与它报的标签 —— **不一致就整条丢弃**。
    不在这里悄悄「修正」成原标签:那样会把「模型觉得该改标签」这个信号抹掉,
    而那个信号正是我们要观测的东西(它的比例就是 spec §5.4 的语料质量读数)。

    ⚠️ **「没解析出来」一律原样返回**,并把第三个值置位(照
    `scripts/prelabel_topics.py::_label` 的先例:`parse_failed` 是一列**读数**,
    不是异常)。若在这里返回**空标签**,`label_drift` 会把一条本来没问题的样本
    判成漂移而丢掉 ⇒ 「网络/解析抖动」与「标签真的漂了」两个读数**混在一起分不开**。

    ⚠️ **形状闸**(原稿只挡了 JSON 语法):`labels` 吐成字符串时
    `list("退换货")` 会按**字符**迭代,每个字符都查无此类目 ⇒ 同样被算成漂移;
    吐成 int 时 `list(5)` 直接 `TypeError` ⇒ **打断整跑**(那是 75 分钟)。
    两种都不是「JSON 没解出来」,但都不是正常结果 ⇒ 归同一档。

    ⚠️ 改写后的句子要过一遍 `clean()`(spec §5.1:训练侧与推理侧**同源**)——
    训练语料里其余每一行都是清洗过的,而 `encode_rows` **不再洗**
    (`app/topic/model.py` 直接读 `row["question"]`)⇒ 不洗这里,增强行进训练时
    就是另一种口径。`clean()` **不修错别字**(§5.1),所以它不抵消增强。
    """
    from langchain_core.messages import HumanMessage

    prompt = REWRITE_PROMPT.format(
        question=row["question"],
        labels=" / ".join(row["labels"]),
        taxonomy=render_taxonomy_for_prompt(),
    )
    resp = await model.ainvoke([HumanMessage(content=prompt)])
    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return row["question"], row["labels"], True
    if not isinstance(data, dict):
        return row["question"], row["labels"], True
    new_q, new_labels = data.get("question"), data.get("labels")
    if (not isinstance(new_q, str) or not isinstance(new_labels, list)
            or not all(isinstance(x, str) for x in new_labels)):
        return row["question"], row["labels"], True
    cleaned = clean(new_q)
    if not cleaned:                      # 模型吐了一句纯空白 ⇒ 回落原句,不算漂移
        return row["question"], row["labels"], True
    return cleaned, (new_labels or row["labels"]), False


async def augment(limit: int | None = None) -> None:
    """增强**只扩训练集**。验证集与测试集一条都不许动。

    扩了测试集,§8 那套指标就不再有意义 —— 而那是个**静默**的破坏:
    指标会变好看,没人会去查测试集是不是被动过。

    ⚠️ **订正 11-B:上面那句话在订正之前只是散文**(本仓已编目:一句看起来成立的
    注释不是守卫)。现在它由三样东西守着 —— 开头那两条 `assert`(输入只许叫
    `train.jsonl`、产物只许叫 `train_augmented.jsonl`)、那条断言**至少一条**的
    用例(把 `TRAIN` 指到 `val.jsonl` ⇒ 必须红)、以及跑前跑后
    `val.jsonl` / `topic_test.jsonl` **逐字节相同**的用例。

    ⚠️ `limit` **只**决定「这次改写几行」(订正 11-A)。**不裁 `train.jsonl`** ——
    那是冻结的训练集(Task 9 的输入),裁它就是就地改掉它,而 `git status` 里
    只会安静地多一行 ` M train.jsonl`。原件**照旧全部写出**,小样只让产物
    少掉那几条增强行 —— 这件事会**响亮地打出来**(别让它冒充全量产物)。
    """
    train_path, out_path = TRAIN, TRAIN_AUGMENTED
    # ---- 11-B 的两条守卫。⚠️ 位置承重:写在**打开任何文件之前** ----
    # 断言若落在 `open(..., "w")` 之后,一次失败的运行会先留下一个**空的**产物文件,
    # 而「拦住了」与「拦住了但先写了一份错的」是两回事。
    assert train_path.name == TRAIN_NAME, (
        f"增强的输入只许是 {TRAIN_NAME},而现在是 {train_path.name} —— "
        f"验证集参与早停与阈值选择,把它加强进产物与测试集被动过同级"
    )
    assert out_path.name == AUGMENTED_NAME, (
        f"增强产物只许叫 {AUGMENTED_NAME},而现在是 {out_path.name} —— "
        f"写进冻结产物是个**静默**的破坏:指标会变好看,没人会去查"
    )

    rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    todo = rows[:limit] if limit else rows
    _out(f"训练集 {len(rows)} 行({train_path.name},只读);本次改写 {len(todo)} 行"
         + (f" —— ⚠️ **小样**(--limit {limit}),这个产物**不是**全量!"
            if limit else ""))

    settings = get_settings()
    model = create_extract_model(settings)
    kept, drifted, parse_failed = 0, 0, 0
    with out_path.open("w", encoding="utf-8") as f:
        # 原件照写 —— 增强是**追加**,不是替换。
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for i, r in enumerate(todo, 1):
            new_text, new_labels, bad = await _rewrite(model, r)
            if bad:
                parse_failed += 1
            if label_drift(r["labels"], new_labels):
                # ⚠️ **静默的标签漂移** —— 整条丢弃,不修。
                #    修的话要判断「是它漂了还是原标错了」,而那需要人。
                drifted += 1
                continue
            f.write(json.dumps(
                # ⚠️ 种子从**行 id** 派生,不用循环下标(订正 11-C):
                #    下标会让可复现性依赖行序,而 `train.jsonl` 是会被重排的。
                {**r, "question": inject_typo(new_text, typo_seed(r["id"])),
                 "labels": new_labels, "augmented": True}, ensure_ascii=False) + "\n")
            kept += 1
            # 逐行 flush:整跑约 75 分钟(1158 × 逐条打网络),崩在中途时
            # 盘上那份至少是**可读的**半成品(断点续跑本章没做,如实记在报告里)。
            f.flush()
            if i % 25 == 0:
                _out(f"  … {i}/{len(todo)}(追加 {kept}、漂移 {drifted}、解析失败 {parse_failed})")

    # ⚠️ **三个读数都要打,别只打一个**:
    #    `kept`         = 追加了几条;
    #    `drifted`      = **语料质量的读数** —— 它高说明预标不稳(或 prompt 里
    #                     「保持诉求个数与类别不变」没被遵守);
    #    `parse_failed` = **故障读数,它是 0 才正常**(那些行按**原句**写回)。
    #    三者混起来报会把「网络抖动」记成「标签漂移」,而两个读数都救不回来。
    _out(f"增强追加 {kept} 条;因标签漂移丢弃 {drifted} 条;"
         f"解析失败 {parse_failed} 条(这些行按**原句**写回)")
    _out(f"产物 {out_path.name}:{len(rows) + kept} 行(原件 {len(rows)} + 增强 {kept})")
    if parse_failed:
        _out("⚠️ 解析失败不为 0 ⇒ 先停下看产物:那些行的题面是**原句**"
             "(不是改写的),标签是原标签 —— 它们不是「漂移」,别混着读。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["collect", "split", "augment"])
    # ⚠️ **订正 11-A**:小样跑必须走这个参数,不许「先把 `train.jsonl` 截成 20 条」。
    ap.add_argument("--limit", type=int, default=0,
                    help="只对 augment 有意义:只改写前 N 行(0/缺省 = 全量)")
    args = ap.parse_args()
    # 「参数被静默忽略」是本仓最不喜欢的一族:有人打 `split --limit 20` 想看小样,
    # 脚本照跑全量(而 `split` 会**覆盖那三份冻结产物**)—— 必须响亮地停。
    if args.limit < 0:
        raise SystemExit("--limit 不许为负(0 表示全量)")
    if args.limit and args.step != "augment":
        raise SystemExit(f"--limit 只对 augment 有意义(`{args.step}` 不支持它)")
    if args.step == "collect":
        asyncio.run(collect())
    elif args.step == "split":
        split()
    else:
        asyncio.run(augment(args.limit or None))


if __name__ == "__main__":
    main()
