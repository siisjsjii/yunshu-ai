"""合成语料:配额 + 形态强制 + 禁词自检。

**打网络** —— 不是单测,不进 `pytest` 默认集。

用法:
    .venv/Scripts/python.exe scripts/gen_topic_data.py --dry-run   # 只打印 prompt
    .venv/Scripts/python.exe scripts/gen_topic_data.py             # 生成 + 结尾自检
    .venv/Scripts/python.exe scripts/gen_topic_data.py --check     # 只核已有产物,不生成

⚠️ **输出是 append**(`"a"`),为的是断点续跑;所以**重跑之前先删掉
`evals/topic/synthetic.jsonl`**,否则同一批问题会进两遍(而 `id` 会从文件行数接着编,
不会撞号,但问题会重复)。脚本不替调用方做这件事 —— 删文件是不可逆的,交给人来决定。

**四个纯函数把「值得测的那一半」从 IO 里摘出来**(各自的理由不同,不是一个理由):
- `plan()` —— 批次清单,`run` 的唯一循环来源;它是「`FORMS` 有没有被真读」的**接线点**。
- `accept()` —— 三道门 + 形态基数;真跑 `dropped = 0` ⇒ **这三道门在生产上一次都没开过火**,
  不抽出来就无从测量(禁词纪律的执行点原本只是一个没人测过的 `if`)。
- `make_row()` —— 行的形状(`id` 与七个键),下游按 `id` 索引。
- `check_rows()` —— 对**产物**的复核(`--check`);测试断的是常量,交付物长什么样得另有人看。
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.llm import create_extract_model
from app.topic.synth import (
    FORMS,
    QUOTA,
    confusable,
    shape_ok,
    violates_forbidden,
)
from app.topic.taxonomy import (
    HEAD_LABELS,
    LABELS,
    OTHER,
    render_taxonomy_for_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "topic" / "synthetic.jsonl"

#: 一次要几条。**不要一次要 200 条** —— 长输出里模型会开始重复自己,
#: 而重复的样本对训练没有增量,却会占掉配额。
#:
#: ⚠️ **今天它没有被接线**:`run` 里每次要的条数是 `batch_size(label, form)`
#: (即 `max(2, round(QUOTA[label] * FORMS[form]))`),头四类的 single 那一批
#: 因此要 **50 条**,而不是 12 条。留在这里是把它要表达的那条经验留下 ——
#: 但**别读成「已经按 12 条一批发了」**(本仓「引用了教训 ≠ 免疫于它」那一类)。
#: 实测(2026-09-26,962/962 全收、零重复问句)显示 50 条一批**这批数据上没出问题**,
#: 所以没有为了对齐这行注释去改行为 —— 那会让已量到的读数失效。
BATCH = 12

#: 多标签 / 边界的门槛(`check_rows` 用)。与 `FORMS` 是**两件事**:
#: `FORMS` 是**要出去**的比例,这两个是**收回来**的纪律 —— 按 `FORMS` 拆分出来的
#: 比例按代数就是 0.30,所以只断言 `FORMS` 等于什么都没说(订正轮 I2)。
MIN_MULTI_SHARE = 0.30
MIN_BOUNDARY_SHARE = 0.15


def batch_size(label: str, form: str) -> int:
    """这一批要几条 —— 从 `QUOTA` 与 `FORMS` 推出来的那个算式,**只此一处**。

    单独成函数是为了可单测:「要出去的条数有没有兑现 `FORMS` 的比例」是一件**纯算术**,
    与打不打网络无关。埋在 `run` 里就只能在联网时看结果 —— 而 `FORMS` 本身只是
    一组常量,一个**无视 FORMS**、每批固定要 10 条的生成器能通过全部形态测试。
    """
    return max(2, round(QUOTA[label] * FORMS[form]))


def plan() -> list[tuple[str, str, int]]:
    """要造的批次清单 —— **(类目, 形态, 条数)**,`run` 的唯一循环来源。

    抽出来同样是**为了可单测**(订正轮 I3):`batch_size` 抽出来之后,`run` 里那句
    `n = batch_size(...)` **仍然是没被测过的一行**,而它正是「`FORMS` 有没有被真读」
    的接线点。计划本身是纯算术,所以它自己能被测,`run` 就只剩 IO。
    """
    return [(label, form, batch_size(label, form)) for label in QUOTA for form in FORMS]


def reject_reason(item: object, seed_label: str, form: str) -> str | None:
    """**唯一的那张判据表**:`None` = 收下;否则返回一句「为什么不要它」。

    ① 形态合法(非空、不重复、≤3、都是真类目);
    ② **形态自身的基数**:`multi` / `boundary` 的定义就是「带 2–3 个诉求」⇒ 必须 ≥2 个标签;
    ③ 问句存在且非空;
    ④ 不含禁词(否则模型学成关键词匹配);
    ⑤ 主诉求真的是这一类(模型经常跑偏;「其他」是共用的落点,不做这一判)。

    ⚠️ **一处实现、两处调用**(本仓那条「不变量要放在唯一写口上,不要靠每个调用方自觉」):
    生成时由 `accept` 读它拦输入,核产物时由 `check_rows` 读它复核交付物。
    两处**不是两套各自维护的谓词** —— 否则「形态基数」这种规则一旦只在一边改,
    两道防线就会互相对不上,而那正是本仓记过的漂移形状。
    (实测:把 ② 那一行摘掉,生成侧与产物侧的用例**同时**变红 —— 因为它们读的是同一处。)

    ⚠️ **为什么值得把它从 `run` 里抠出来**:真跑 962 条 `dropped = 0` ⇒ 这几道门
    在生产上**一次都没开过火**。禁词纪律是这一章最贵的一条,而它的执行点原本只是一个
    没人测过的 `if` —— 「禁词表写坏了」与「模型没写禁词」在读数上长得一模一样。
    ⚠️ **空问句也算不过**:`item.get("question", "")` 的默认值让「缺 question 键」与
    「question 是空串」都能走到写库那一步 —— 前者写出一行没有问句的训练数据,后者写出一行
    **空问句**。本仓 ch07 那条道理(「有个占位的坏值」比「缺值」更坏)在这里同样成立。
    ⚠️ **非 dict 条目**也挡在门外:模型偶尔直接返回字符串数组,原实现会在 `item.get` 上
    `AttributeError` 把整轮生成打断。

    ⚠️ **② 是订正轮 2 补的(F2),而它补的正是 I1 那个缺陷的第二条来路**:上一版的签名里
    **没有 `form`**,所以「boundary 批应当是 2 个标签」这条要求**只活在提示词里** ——
    模型下次把边界批退回单标签 ⇒ 147 行**全被接受**、`dropped = 0`、脚本照打
    「保留 962 条,丢弃 0 条」,而那时 `check_rows` 的两道占比门也判通过(F1)。
    """
    if not isinstance(item, dict):
        return "条目整个不是对象(模型偶尔直接吐字符串数组)"
    labels = item.get("labels")
    if not isinstance(labels, list) or not shape_ok(labels):
        return "形态不合法(shape_ok:空 / 重复 / 超 3 个 / 表外标签)"
    if form in ("multi", "boundary") and len(labels) < 2:
        return f"{form} 形态只有 1 个标签(这两种形态的定义就是 2–3 个诉求)"
    question = item.get("question", "")
    if not isinstance(question, str) or not question.strip():
        return "问句缺失或为空"
    hits = violates_forbidden(question)
    if hits:
        return f"含禁词 {hits}"
    if seed_label not in labels and seed_label != OTHER:
        return "seed_label 不在自己的 labels 里"
    return None


def accept(item: object, seed_label: str, form: str) -> bool:
    """**三道门 + 形态基数** —— 生成时的过滤点。纯函数(订正轮 I3 抽出,订正轮 2 补 `form`)。

    判据全在 `reject_reason` 那一处;核产物时读的是**同一张表**(不是另写一份)。
    """
    return reject_reason(item, seed_label, form) is None


def _boundary_desc(label: str) -> str:
    """`boundary` 形态那一句 —— **点名**易混类目,并说明这一批要**两个标签**。

    ⚠️ 订正轮 I1:上一版只说「与它最容易混的那个类目同时出现,让人必须读完才分得清」,
    读起来是一道**单选辨析题**(而且要求 1 又说「主诉求是「{label}」」,单数),
    实测边界批 **147/147 全是单标签** —— 而它撞的那个类目**明明在句子里出现了**。
    两句都补上:① 两个诉求**都要打标**;② 撞哪个类目**由 `COUNTER` 指定**,不自由联想。
    """
    other = confusable(label)
    tail = "**两个诉求都要打标** —— 这是本轮**唯一**要求多标签的形式(别的形式只要最显眼那个)。"
    if other is None:
        # `COUNTER` 没声明易混类目的那些类:退回「让模型自己挑」的说法,但**打标要求照旧**。
        return (f"涉及「{label}」,而且**与它最容易混的那个类目**同时出现,{tail}"
                f"要让人**读完才分得清**,而不是一眼就能归类")
    return (f"涉及「{label}」**和「{other}」两个**诉求 —— 这正是这两个类目最容易混的地方,"
            f"{tail}要让人**读完才分得清**,而不是一眼就能归类")


def build_prompt(label: str, form: str, n: int) -> str:
    """组装一批合成用的提示词。

    ⚠️ 三条硬约束写进 prompt,它们对应 `test_gen_topic_data.py` 的三条形态测试:
    ① **不许出现类目名与边界说明的原词**;
    ② `multi` / `boundary` 形态必须**真的带 2–3 个诉求**,且每个诉求在句子里**字面可指**;
    ③ 说人话 —— 像真实买家在客服窗口里打出来的,带口语与省略。

    ⚠️ 本仓硬约束:提示词里必须出现字面 `JSON` 字样,且**不得有裸花括号**。
    """
    form_desc = {
        "single": f"只涉及「{label}」**一个**诉求",
        "multi": f"涉及「{label}」**以及另外 1–2 个**不同的诉求(共 2–3 个)",
        "boundary": _boundary_desc(label),
    }[form]
    labels_line = "- labels:该问句应当打上的标签数组(**标签名必须逐字来自上面的类目表**)"
    # 「连点名的那两个也不许写出来」只在**真的点了名**的那一支说 ——
    # 回退支(`COUNTER` 没声明的那些类)一个字都没点名,写上去就是一句对不上的话。
    banned_extra = ",**也包括上面点名的那两个**" if confusable(label) and form == "boundary" else ""
    if form == "boundary":
        # 把「两个标签」写进**输出结构**那一句,而不只写在形态描述里 ——
        # 上一版就是只有形态描述(且不含基数),模型只给了一个。
        labels_line += ",**本批应当是 2 个**"
    return f"""你在为一个电商客服系统造**训练语料**。

下面是本系统权威的类目表(含每类的边界说明、正例与容易混的反例):

{render_taxonomy_for_prompt()}

请造 {n} 条**真实的买家问句**,要求:
1. 这些问句的主诉求是「{label}」,且{form_desc}。
2. **绝对不许出现任何类目名**(如类目表里那些词{banned_extra}),
   也**不许出现边界说明里的原词**。
   买家不会说「我要咨询退换货类问题」,他只会说「买大了想退」。
3. 像真人在客服窗口打出来的:口语、有省略、可以带错别字。
4. 每条 5–30 个字。

输出一个 JSON 数组,每个元素是对象,含两个字段:
- question:问句本身
{labels_line}

只输出 JSON,不要别的内容。"""


def make_row(item: dict, label: str, form: str, index: int) -> dict:
    """一条产出记录的**形状** —— 七个键与 `id` 都在这里,`run` 只负责写。

    ⚠️ 订正轮 I4:`id` 是下游的索引键 —— T5 的预标做 `r["id"] not in done`、
    T7 按 `r["id"]` 排序、已审合并也按它。真实语料用的是 **`r-####`**,
    合成这边用 **`s-####`**:两个来源会合流进同一个列表,**前缀不同**才分得清。
    没有它的话 T5 在第一行合成数据上就 `KeyError: 'id'`。
    """
    return {"id": f"s-{index:04d}", "question": item["question"], "labels": item["labels"],
            "provenance": "synthetic", "source": "gen", "form": form, "seed_label": label}


def load_rows(path: Path) -> list[dict]:
    """读产物。坏行**响亮地抛** —— 静默跳过会让自检在一个「少了几行」的文件上判通过。"""
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno} 不是合法 JSON —— 自检不在读不全的文件上判通过") from exc
    return rows


def check_rows(rows: list[dict]) -> tuple[bool, str]:
    """核**产物**的纪律:`(是否通过, 给人看的报表)`。纯函数(订正轮 I2,I2-2 扩过)。

    ⚠️ **为什么非要这一层**:测试里 `FORMS["multi"] >= 0.30` 断的是**常量**,而
    `plan` 按 `FORMS` 拆分出来的比例**按代数就等于 0.30** —— 「多标签 ≥30%」这条纪律
    在测试里是**按构造成立**的,交付物长什么样,它一个字都没说。

    判据(controller 2026-09-26 裁定):**按全局判**,逐类只打印 ——
    头四类**永远**到不了 30%(见下面 `_plan_note`),那是取整效应,
    把 `FORMS["multi"]` 抬上去凑一个逐类读数才是真的错。

    **核哪些**(订正轮 2 补了后四条):
    ① `id` 齐全且唯一;② 逐类行数 ≥ `QUOTA`;③ 全局多标签 ≥ 30%;④ 全局边界 ≥ 15%;
    ⑤ 形态基数、⑥ 形态合法、⑦ 禁词、⑧ `seed_label ∈ labels` —— **这四条不在本函数里另写**,
    逐行读的是 `reject_reason` 那张**唯一**的判据表(与生成侧的 `accept` 同一处)。

    ⚠️ **⑤ 与 ③④ 是两件事,别互相代偿**(F1):③④ 判的是**占比**,而 `form` 列由
    `plan()` 写死、模型摸不到 ⇒ ④ 实际是个**掉行探测器**(掉够 4 条才开火);
    ③ 今天有 15pp 余量**只是因为 I1 修好了**。把 147 条边界行各自退化成单标签
    (I1 那个缺陷原样重放)⇒ 多标签 290/962 = 30.15%、边界 147/962 = 15.28%,
    **③④ 两道门都判通过**。所以基数必须按**形态**单独判。
    """
    problems: list[str] = []

    ids = [r.get("id") for r in rows]
    if any(not i for i in ids):
        problems.append(f"有 {sum(1 for i in ids if not i)} 行没有 id(T5 按 id 索引)")
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        problems.append(f"id 有重复:{dup[:5]}")

    total = len(rows)
    if not total:
        return False, "产物是空的 —— 没有任何东西可核\n"

    # ── F1 + F3:读**同一张判据表**(`reject_reason`),在产物上把生成时那几道门再核一遍 ──
    # ⚠️ 为什么不在这里另写一遍谓词:那样「形态基数」这类规则一旦只在一边改,两道防线
    #    就会互相对不上 —— 本仓记过的漂移形状。读同一处 ⇒ 摘掉那一行,两边**同时**红。
    # ⚠️ 其中 **F1 那条**(形态基数)是复审实测出来补的:订正轮 1 的两道**占比**门对
    #    I1 那个缺陷**零判别力** —— 把 147 条 boundary 行各自退化成只留主标签,
    #    多标签落到 290/962 = 30.15%、边界 147/962 = 15.28%,**两道门都判通过**。
    #    为什么:boundary 那道判的是**边界行的占比**,而 `form` 列完全由 `plan()` 写死
    #    (模型摸不到)⇒ 它其实是个**掉行探测器**(掉够 4 条才开火);而多标签那道今天
    #    有 15pp 余量,**纯粹是因为 I1 修好了** —— I1 一退,余量就没了。
    #    ⇒ 形态自身的基数必须**按形态单独判**,不能靠占比代偿。
    rejected: dict[str, list[str]] = {}
    for r in rows:
        why = reject_reason(r, r.get("seed_label"), r.get("form"))
        if why:
            rejected.setdefault(why, []).append(str(r.get("id")))
    for why, bad_ids in rejected.items():
        problems.append(f"{len(bad_ids)} 行没过「{why}」,例如 {bad_ids[0]}")

    lines = [f"总行数 {total}", "逐类(每类都打印:行数 / 配额 / 这一类的多标签占比 / 边界形态占比)"]
    for lb in LABELS:
        sub = [r for r in rows if r.get("seed_label") == lb]
        n_multi = sum(1 for r in sub if len(r.get("labels") or []) >= 2)
        n_bound = sum(1 for r in sub if r.get("form") == "boundary")
        share_m, share_b = n_multi / max(1, len(sub)), n_bound / max(1, len(sub))
        mark = "✓" if len(sub) >= QUOTA[lb] else "✗"
        if len(sub) < QUOTA[lb]:
            problems.append(f"{lb} 只有 {len(sub)} 行,少于配额 {QUOTA[lb]}")
        lines.append(f"  {lb:4s} 行 {len(sub):4d} {mark} 配额 {QUOTA[lb]:4d} │ "
                     f"多标签 {n_multi:4d} ({share_m:5.1%}) │ 边界 {n_bound:3d} ({share_b:5.1%})")

    share_multi = sum(1 for r in rows if len(r.get("labels") or []) >= 2) / total
    share_bound = sum(1 for r in rows if r.get("form") == "boundary") / total
    lines.append(f"全局:多标签 {share_multi:.1%}(门槛 {MIN_MULTI_SHARE:.0%})/ "
                 f"边界形态 {share_bound:.1%}(门槛 {MIN_BOUNDARY_SHARE:.0%})")
    if share_multi < MIN_MULTI_SHARE:
        problems.append(f"全局多标签占比 {share_multi:.1%} < {MIN_MULTI_SHARE:.0%}")
    if share_bound < MIN_BOUNDARY_SHARE:
        problems.append(f"全局边界形态占比 {share_bound:.1%} < {MIN_BOUNDARY_SHARE:.0%}")
    lines.extend(_boundary_note(rows))
    lines.append(_plan_note())
    lines.append("结论:**通过**" if not problems
                 else "结论:**不通过**\n" + "\n".join(f"  - {p}" for p in problems))
    return not problems, "\n".join(lines) + "\n"


def _boundary_note(rows: list[dict]) -> list[str]:
    """把边界那 15.3% **拆开**打出来 —— 一个合计数会把两件事混成一件(订正轮 2,I-3)。

    - **声明支**:`COUNTER` 点过名的真近邻对(91 条);
    - **回退支**:`COUNTER` 没声明的 7 类,模型**自己挑**的邻居 —— 复审逐行核过全部 147 条,
      判定那是**共现**而不是易混(发票 / 支付 / 其他 三类甚至逐行换邻居)。根因是
      spec §4.1 自己在那 7 行写了 `—`。**controller 2026-09-26 裁定:接受并记账,
      不在本任务里补 `COUNTER`**(改 taxonomy 超出本任务);⇒ 两批**不是一个强度**,
      合计数不能拿来当「147 条边界对都学得开」的证据。
    - 「其他」+ 真类目那一列**单独计数**:它在 `INTENT_TO_TOPICS` 下是**合法标注**
      (`其他` 就是转人工/投诉/闲聊的落点),而在 `BOUNDARY` 的释义(「以上都不是」)下读起来像矛盾。
      决策点在 T5 的预标口径(T5 会整列覆盖 `labels`),不在这里。
    """
    b = [r for r in rows if r.get("form") == "boundary"]
    declared = [r for r in b if confusable(r.get("seed_label"))]
    fallback = [r for r in b if not confusable(r.get("seed_label"))]
    lines = [
        f"边界形态拆分:{len(b)} 条 = **{len(declared)} 条声明过的真近邻对**"
        f"(`COUNTER` 点名的那些)+ **{len(fallback)} 条回退支的共现对**"
        f"(邻居由模型自选,共现≠易混;类目 = {sorted({r.get('seed_label') for r in fallback})})",
    ]
    other_rows = [r for r in rows
                  if OTHER in (r.get("labels") or []) and len(r.get("labels") or []) > 1]
    by_form: dict[str, int] = {}
    for r in other_rows:
        by_form[str(r.get("form"))] = by_form.get(str(r.get("form")), 0) + 1
    lines.append(f"「其他」+ 真类目的行:{len(other_rows)} 条 {by_form}"
                 f"(「合法标注 / 矛盾释义」两种读法见 T4 报告 §6.2 与 §9.7,决策点在 T5 口径)")
    return lines


def _plan_note() -> str:
    """把「计划侧」的取整效应写出来 —— 它是纯算术,与模型怎么答无关。"""
    batches = plan()
    total = sum(n for _, _, n in batches)
    multi = sum(n for _, form, n in batches if form == "multi")
    head_total = sum(n for lb, _, n in batches if lb in HEAD_LABELS)
    head_multi = sum(n for lb, form, n in batches if lb in HEAD_LABELS and form == "multi")
    other_total = total - head_total
    other_multi = multi - head_multi
    return (f"⚠️ 计划侧(纯算术,与模型无关):multi 批全局 {multi}/{total} = {multi / total:.1%};"
            f"头四类 {head_multi}/{head_total} = {head_multi / head_total:.1%}、"
            f"其余 {other_multi}/{other_total} = {other_multi / other_total:.1%} —— "
            f"头四类那 29.7% 是 `round(90×0.30)=27` 与 `round(90×0.55)=50` 的**取整**效应,"
            f"不是生成器少要了(所以门槛按**全局**判,逐类只打印)。")


async def run(dry_run: bool, check_only: bool = False) -> int:
    if check_only:
        return _report(OUT)
    settings = get_settings()
    model = create_extract_model(settings)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # `id` 从**已有行数**接着编(`s-0001`…)。每个存活行恰好一行 ⇒ 行数就是这个编号。
    # 不这么做的话,断点续跑(append)会从 1 重新编 ⇒ id 撞号,而 T5 的
    # `r["id"] not in done` 与 T7 的按 id 排序都会**静默**出错。
    next_id = _existing_rows(OUT) + 1
    kept, dropped = 0, 0
    with OUT.open("a", encoding="utf-8") as f:
        for label, form, n in plan():
            prompt = build_prompt(label, form, n)
            if dry_run:
                print(f"--- {label}/{form} n={n} ---\n{prompt[:400]}…\n")
                continue
            raw = await _call(model, prompt)
            for item in raw:
                # ⚠️ 三道门 + 形态基数全在 `accept` 里(纯函数、有测试)。真跑里这三道没开过火。
                if not accept(item, label, form):
                    dropped += 1
                    continue
                f.write(json.dumps(make_row(item, label, form, next_id),
                                   ensure_ascii=False) + "\n")
                next_id += 1
                kept += 1
            f.flush()
    if dry_run:
        return 0
    # ⚠️ 这两个数**必须打出来**。丢弃率高得离谱通常意味着**禁词表写宽了**
    #    或模型没守格式 —— 而不是「模型不行」。别跳过这个读数。
    print(f"保留 {kept} 条,丢弃 {dropped} 条(丢弃率 {dropped / max(1, kept + dropped):.1%})")
    return _report(OUT)


def _existing_rows(path: Path) -> int:
    """产物里**已有的行数**(不解析 JSON —— 每存活行恰好一行,行数就是编号上限)。"""
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _report(path: Path) -> int:
    """跑产物自检并**据此决定退出码** —— 不通过就响亮地红。"""
    if not path.exists():
        print(f"[check] {path} 不存在 —— 没有可核的产物")
        return 1
    ok, report = check_rows(load_rows(path))
    print(f"\n=== 产物自检 {path.relative_to(ROOT)} ===\n{report}")
    return 0 if ok else 1


async def _call(model, prompt: str) -> list[dict]:
    """一次调用 + 解析。解析失败返回空列表(**不抛**)—— 合成可以少几条,不能整轮中断。"""
    from langchain_core.messages import HumanMessage

    try:
        resp = await model.ainvoke([HumanMessage(content=prompt)])
    except Exception as exc:                      # noqa: BLE001 —— 网络抖动重来即可
        print(f"  [warn] 调用失败,跳过这批:{type(exc).__name__}")
        return []
    text = (resp.text or "").strip()
    # 模型常把 JSON 包在 ```json 里 —— 剥掉再解析。
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def main() -> None:
    # ⚠️ 钉死输出编码:本机 `sys.stdout.encoding` 是 **gbk**,而本脚本的输出里
    #    有中文(保留/丢弃那行的读数)。照 `scripts/prepare_topic_data.py` 的房规
    #    把边界钉成 UTF-8,不依赖控制台 codec —— 否则输出被重定向到文件时,
    #    「保留 N 条」那行会按 GBK 落盘,而读的人按 UTF-8 解。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印 prompt,不打网络")
    ap.add_argument("--check", action="store_true", help="只核已有产物,不生成")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args.dry_run, args.check)))


if __name__ == "__main__":
    main()
