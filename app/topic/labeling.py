"""标注链路上的纯函数 —— 全部零 IO、零依赖,故可密集单测。

放在 `app/topic/` 而不是 `scripts/` 里,是因为**它们要被两侧用**:
去重与分层抽样既在离线造数据时跑,也在评测脚本里跑(测试集冻结后的再切分)。
"""

import random
from collections import defaultdict

from app.topic.clean import clean


def dedupe_questions(rows: list[dict]) -> list[dict]:
    """按**清洗后的文本**去重,保留每个文本**首次出现**的那一行。

    调用方负责先按来源优先级排序(池子 → 对话 → 测试集.md),
    本函数不做优先级判断 —— 那是策略,这是机制。
    """
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        text = clean(row.get("question", ""))
        if not text or text in seen:
            continue
        seen.add(text)
        out.append({**row, "question": text})
    return out


def validate_evidence(
    question: str, labels: list[str], evidence: dict[str, str]
) -> tuple[list[str], list[str]]:
    """校验每个标签的「证据串」是不是原文的子串。

    返回 `(通过的标签, 被拒的标签)` —— **被拒的也要返回**,不能悄悄吞掉:
    从 3 个标签缩成 1 个会改变训练分布,而那是静默的。

    ⚠️ **空证据串必须显式拒绝**:`"" in "任意文本"` 恒为 True,
    不特判的话 `{"尺码": ""}` 会被判成「证据合法」—— 一个一眼看不见的假绿。

    两侧都过 `clean()`:问句侧与预标喂进 prompt 的文本同口径,
    证据侧则防止全角/空白差异把一条**真的**证据误拒。
    """
    normalized = clean(question)
    accepted, rejected = [], []
    for label in labels:
        ev = (evidence.get(label) or "").strip()
        # 证据本身也要过一遍清洗,否则全角/空白差异会造成假拒。
        if ev and clean(ev) and clean(ev) in normalized:
            accepted.append(label)
        else:
            rejected.append(label)
    return accepted, rejected


def pick_review_sample(
    rows: list[dict], *, per_label: int, seed: int = 20260925
) -> list[dict]:
    """按类分层抽 `per_label` 条,用于人工复核(spec §6.3)。

    **同一条多标签行可能被多个类抽中** —— 用 id 去重后返回,
    所以返回条数**可能少于** `per_label × 类目数`。这是对的:
    它的目的是「每一类都有人看过」,不是「恰好 N 行」。

    种子可注入 ⇒ 同一种子两次结果相同,「我审的是哪几条」可复现。

    ⚠️ **「可复现」的确切含义(实现者据实测订正的措辞,2026-09-26)**:
    它复现的是「**同一份 `rows`(逐行同序) + 同一个 seed**」,不是「同一个语料」。
    两层原因,缺一不成:

    1. **桶的遍历顺序**是各标签**首次出现**的先后(`dict` 保序)⇒ 语料重排会
       把 rng 的同一段随机数分配给**不同**的桶;
    2. **三个桶共用同一个 `rng`** ⇒ 前一个桶的内容/条数一变,
       后面所有桶抽到的东西跟着变。

    ⇒ **「语料重跑过」不等于「抽出的还是同一批」**(别拿这条去断言跨语料的稳定性)。
    真正承重的是「同一份输入可复现」—— 那正是复核 CSV 需要的那一档。

    ⚠️ 实测(2026-09-26,真实 1424 行 `prelabeled.jsonl`,见
    `.superpowers/ch10b_t6_perm_probe.py`):逐行同序 **84** 条;同一份语料**整体倒序**
    后是 **85** 条、与前者只重叠 13 条;随机重排后 83 条、只重叠 4 条。

    ⚠️ **「哪 84 条」这件事本身只记在产物里**:`evals/topic/labels/trainval.csv` **入库**,
    而它的输入 `evals/topic/prelabeled.jsonl` **不入库**(与 `corpus.jsonl` /
    `synthetic.jsonl` 同例,见 `dev-notes/ch10.md`)⇒ **clone 出来是重跑不出那 84 条的**
    (即使重新预标出一模一样的 1424 行,只要行序不同就换一批)。CSV 自己就是那份记录。
    """
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        for label in row.get("labels") or []:
            by_label.setdefault(label, []).append(row)

    picked: dict[str, dict] = {}
    for label, bucket in by_label.items():
        # 桶内先按 id 排序再打乱 ⇒ 抽中谁与**桶内的行序**无关
        # (桶的遍历顺序与共用 rng 的代价见 docstring ⚠️)。
        shuffled = sorted(bucket, key=lambda r: r["id"])
        rng.shuffle(shuffled)
        for row in shuffled[:per_label]:
            picked[row["id"]] = row
    return [picked[k] for k in sorted(picked)]


def is_unusable_target(row: dict) -> bool:
    """这行的**靶子不可信**吗 —— 零标签**且****有**被证据校验拒掉的标签。

    ⚠️ **为什么要把这件事单列成一个谓词**(计划订正 9-E,controller 2026-09-26):
    `prelabeled.jsonl` 里**两行**是零标签,而来源**不同**:

    - `r-0049`「你是」—— 模型**真判了零诉求**(`rejected_labels == []`);
    - `s-0423`「首重多少,超了咋算?」—— **证据校验机械拒到空**
      (`rejected_labels == ['运费']`),而它的真实主题**几乎肯定是运费**。

    后者是**处理产物,不是判断**。把 `labels == []` 喂进训练,是在教模型
    「这句话没有主题」(一条**负样本**),而那句话不成立。而这两行在产物里
    **逐字节相同** ⇒ 只看 `labels` 分不开它们。

    ⚠️ 判据是「**有**被拒的标签」,不是「有 `rejected_labels` 这个键」:
    键在本章产物里**每行都有**(值可能是 `[]`),缺键只可能是别的调用方。

    ⚠️ **谓词只在这一处实现**:`stratified_split` 用它**排除**、`split()` 用它
    **打印去向**。两处各写一遍 `labels == [] and rejected_labels` 就是本仓记过的
    漂移形状(「不变量要放在唯一写口上,不要靠每个调用方自觉」)。
    """
    return not (row.get("labels") or []) and bool(row.get("rejected_labels"))


def stratified_split(
    rows: list[dict], *, test_real: int, test_synth: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 20260925,
) -> dict[str, list[dict]]:
    """按主标签分层切 训练/验证/测试。

    **测试集构成是硬条件**(spec §5.3):先从 real 与 synthetic 里各取
    `test_real` / `test_synth` 条进测试集,**剩下的**再按 8:1:1 切训练与验证。
    反过来做(先整体 8:1:1 再调整)会得到「看起来对、构成不对」的测试集,
    而 §8.4 那三层报告会因此打不出来。

    「主标签」= `labels[0]`。用它分层而不是用全部标签,是因为一个样本
    只能进一份;用全部标签分层会产生归属冲突,而解决冲突的规则又是一处
    需要被解释的实现细节。

    ⚠️ **靶子不可信的行先被摘掉**(`is_unusable_target`,订正 9-E)——
    它们**哪一份都不进**,包括测试集:一条靶子错的样本在测试集里只会
    把「模型对了」记成「模型错了」。排除掉的行不在返回值里 —— 调用方要报它们的
    去向,得自己按 `labels == []` 把零标签行找出来,再用**同一个谓词**分辨
    「靶子不可信」与「模型真判零诉求」(`split()` 就是这么做的)。

    ⚠️⚠️ **`ratios[2]` 不参与任何计算**(订正轮 1 的 I3,controller 2026-09-26 裁定):
    测试集是**先按 `test_real` / `test_synth` 固定条数取走**的,剩下的只按
    `ratios[0] : ratios[1]` 分训练与验证。三元组留着是为了让签名与 spec 的「8:1:1」对得上,
    **但它既不读第三个元素、也不该读** —— 用它定测试集条数的话,§8.4 那三层报告的
    「真实 80 / 合成 40」就再也凑不出来(那正是这个方法存在的前提)。
    ⚠️ 这句以前写的是「按 8:1:1 切」—— 一句**看起来会生效、其实什么都没做**的话,
    本仓那条「静默无效」家族的注释版。
    """
    usable = [r for r in rows if not is_unusable_target(r)]
    rng = random.Random(seed)
    by_prov: dict[str, list[dict]] = defaultdict(list)
    for r in usable:
        by_prov[r["provenance"]].append(r)

    def take(pool: list[dict], n: int) -> list[dict]:
        """按主标签分层取 n 条。

        ⚠️ 它**只返回取出的那些**(订正轮 1 的 M1):原版还返回一个「剩下」的列表,
        而两处调用都写 `take(...)[0]`、「剩下」谁也没用 —— 而且那个局部变量**也叫
        `rest`**,与外层真正被用的 `rest` 同名,读起来像「剩余」却被丢弃。
        """
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            buckets[(r.get("labels") or ["其他"])[0]].append(r)
        for b in buckets.values():
            b.sort(key=lambda r: r["id"])
            rng.shuffle(b)
        # 轮转取,保证每个类都有代表(而不是某个大类被抽干)。
        picked: list[dict] = []
        keys = sorted(buckets)
        i = 0
        while len(picked) < n and any(buckets[k] for k in keys):
            k = keys[i % len(keys)]
            if buckets[k]:
                picked.append(buckets[k].pop())
            i += 1
        return picked

    test = take(by_prov.get("real", []), test_real) + \
        take(by_prov.get("synthetic", []), test_synth)
    test_ids = {r["id"] for r in test}
    rest = [r for r in usable if r["id"] not in test_ids]

    rest_sorted = sorted(rest, key=lambda r: r["id"])
    rng.shuffle(rest_sorted)
    n_train = round(len(rest_sorted) * ratios[0] / (ratios[0] + ratios[1]))
    return {"train": rest_sorted[:n_train], "val": rest_sorted[n_train:], "test": test}


def _char_bigrams(text: str) -> set[str]:
    """字符二元组(短于 2 字时退化成一个「整串」元素)。

    用**字符**而不是词:`jieba` 那类分词会引入一份需要被解释的词典,
    而这里的用途只是「这两句像不像」。
    """
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def train_test_overlap(train: list[dict], test: list[dict],
                       near: float = 0.8) -> dict[str, object]:
    """报出训练侧与测试侧的重复/近重复。**纯函数、零依赖。**

    ⚠️ **为什么要这一步**(T5 复审发现,见 Task 7 节首那段):
    `POSITIVE` 的 34 句正例进了 **T4 生成器**的 prompt,而 T4 的禁词表
    **不覆盖这些例句** ⇒ 重跑 T4 可能把「买大了」「175 穿什么码」整句抄进问句。
    已发生的一半:**冻结测试集里 8 行与渲染块例句字面重叠**(6 行「运费怎么算」自 T1 起、
    2 行「什么材质」由 T5 的 Step 0 新带入)。

    **判据分两档,处置不同**:
    - **完全相同**(清洗后文本相等)⇒ 报进 `exact_train_ids`(`split()` 负责移出训练侧),
      它在测试侧留着;
    - **近重复**(字符二元组 Jaccard ≥ `near`)⇒ **只报不删** —— 删了就是拿判据改数据,
      而这条判据本身是启发式。

    返回 `{"exact_train_ids": [...], "near_pairs": [(train_id, test_id, 相似度), ...]}`。

    ⚠️ **三处口径,少一处读数就会虚高或虚低**:
    - 比较的是**清洗后**的文本(`clean()`,与训练侧同口径)—— 用原文比会把
      「全角空格 / 重复标点」这种差异漏成两个不同的句子;
    - 与测试侧某行**完全相同**的训练行**不再进 `near_pairs`**:同一个事实报两遍
      会让「近重复 N 对」这个读数虚高,而 `exact` 那一档谁也不会漏看;
    - `train` 是**训练侧**(调用方把 `train + val` 一起传进来)—— 验证集参与早停与
      阈值选择 ⇒ 它泄漏的后果与训练集同级。

    ⚠️ **`near=0.8` 是默认值,不是标定值**(订正轮 1 如实记账)。已知的偏差方向是
    **偏松**(会把不该算的一对算进来),成因有一处是本章自己的清洗:
    `clean()` 把订单号/手机号**脱敏**成同一个占位符 ⇒
    「查一下订单号为100086的订单」与「查一下订单号为10086的订单」在**清洗后**只差一个字符
    (真实语料上实测 **0.867**,就是这么来的 —— 见 `dev-notes/ch10.md` 阶段 8)。
    两句话在**语义上**都只是「查订单」,所以报出来不算错;但**换一批语料会有别的形态**。
    ⇒ 要动这个阈值,**先标定再改**(把 `near` 从 0.5 扫到 0.95 看对数怎么变 ——
    本函数是纯函数、零 IO,一行就能跑),不要凭感觉调。
    """
    by_text: dict[str, list[str]] = defaultdict(list)
    for row in test:
        text = clean(row.get("question") or "")
        if text:
            by_text[text].append(row["id"])
    test_items = [(text, _char_bigrams(text), ids) for text, ids in by_text.items()]

    exact: list[str] = []
    near_pairs: list[tuple[str, str, float]] = []
    for row in train:
        text = clean(row.get("question") or "")
        if not text:
            continue
        if text in by_text:
            exact.append(row["id"])
            continue
        grams = _char_bigrams(text)
        for test_text, test_grams, test_ids in test_items:
            union = grams | test_grams
            if not union:
                continue
            score = len(grams & test_grams) / len(union)
            if score >= near:
                near_pairs.extend((row["id"], tid, score) for tid in test_ids)
    near_pairs.sort(key=lambda p: (-p[2], p[0], p[1]))
    return {"exact_train_ids": exact, "near_pairs": near_pairs}
