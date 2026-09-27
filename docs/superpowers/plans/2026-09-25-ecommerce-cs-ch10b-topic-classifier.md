# ch10-B:多标签主题分类器 —— 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 微调一个 17 类的多标签主题分类器,把飞轮攒下的低置信度问题批量归类写进 `topic_classifications`,并在管理台看到各主题分布。

**Architecture:** 四条互不耦合的边界把它与主链路隔开 —— ① `app/topic/` 是**纯数据与纯函数**(零依赖、零 IO),② `topic_service/` 是**独立进程**的旁路推理服务(不进 `app/`),③ 批处理是**仓库里的独立脚本**(读池 → HTTP 调服务 → 写表),④ 主链路**零调用**(源码扫描测试守着)。训练语料三源合流 + 大模型配额合成,测试集 100% 人工复核;评测输出每类 P/R/F1 + 两张矩阵 + **三个口径**的报告。

**Tech Stack:** Python 3.13 / **transformers 5.17.0**(注意 v5 与 v4 的 API 差异)/ torch 2.11.0+cu128 / datasets 5.0.1 / scikit-learn 1.9.1 / FastAPI / MySQL 8.0.46

**Spec:** `docs/superpowers/specs/2026-09-25-ecommerce-cs-ch10-topic-classifier-design.md`(§4–§10 是任务来源)

## Global Constraints

- **17 类表的唯一来源是 `app/topic/taxonomy.py`。** 预标 prompt、合成配额、训练标签顺序、评测类目、分布页图例**全部 import 它**。任何一处硬编码类目名就是漂移的开始。
- **推理侧的标签顺序必须来自训练产物**(`models/topic-clf/labels.json`),**服务不许自己写一份**。两侧顺序不一致会产出一张**完全错的分布图而每个组件都正常**。
- **清洗函数训练侧与推理侧同源**(`app/topic/clean.py` 一处实现),写成测试守着。
- **清洗不做错别字修正**(用户 2026-09-25 批准的设计变更)。错别字保留,并在增强阶段**主动注入**。
- **`problem_type` 必须显式传 `"multi_label_classification"`**,并断言标签是 float。整数标签会被 transformers 静默判成单标签交叉熵,且在第一个 batch 上锁死。
- **transformers v5 名字**:`Trainer(processing_class=...)`(非 `tokenizer=`)、`TrainingArguments(eval_strategy=...)`(非 `evaluation_strategy=`);`use_fast` 在 v5 被静默丢弃。
- **`load_best_model_at_end=True` 时 `save_strategy` 必须等于 `eval_strategy`**,否则直接抛错。
- **早停盯 micro-F1,不盯 macro-F1**(验证集每类仅 ~7 条,macro 是噪声)。
- **不用 `pos_weight`**(它改概率尺度,阈值语义与页面上的置信度就不再是概率)。
- **阈值只扫全局单阈值(0.3–0.7),不做每类阈值**。
- **JSON 列一律用 `JSON_TYPE()` 读**,不用 `IS NULL` / `IS NOT NULL`。
- **基础设施故障绝不伪装成业务结果**:服务连不上 ⇒ 响亮失败、退出码非 0,**绝不写空标签**。
- **`init_db.py` 永不加列**;每章带 `db/chNN.sql`,本章是一张**全新的表**。
- **单测全程不联网**;`Settings(...)` 在测试里必须传 `_env_file=None`;db 测试打 `@pytest.mark.db`。
- **不要再往命令行加 `-q`**。**绝不把证据输出接进任何截断/过滤管道。**
- **含中文的请求体不能走 `curl` 的 argv**(MSYS2 按 CP936 重编码)—— 走 stdin heredoc 或 httpx。
- 前端(任务 15)按用户给的例外走 **Vibe Coding**。

### ⚠️ 两个人工检查点(不可跳过、不可由 agent 代劳)

| 检查点 | 在哪 | 用户要做什么 |
|---|---|---|
| **CP-1:术语表过目** | 任务 1 Step 6 | 打开 `evals/topic/taxonomy_review.csv`,看 17 类的边界说明与正反例 —— **后面所有预标、合成、裁决都照它走** |
| **CP-2:标签复核** | 任务 6 **Step 6**(⚠️ **订正 7**:原先写「Step 3」是笔误;且本任务**跨**检查点,已拆成 B6-A / B6-B,见 Task 6 节头) | 改 `evals/topic/labels/trainval.csv`(**实测 84 行**;原先写的「85」是 `17×5` 的**上界** —— 有一条多标签行被两个桶同时抽中、按 id 去重后少 1)与**任务 7 切分之后才导出**的 `test.csv`(120 行,100% 过) |

**CP-1 不通过不许开工任务 3 及以后。CP-2 不回收不许开工任务 7 及以后。**

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `app/topic/taxonomy.py` | **新建**。17 类权威表 + 边界说明 + 正反例 + 9→17 映射 + 给 prompt 用的渲染函数。纯数据、零依赖、零 IO |
| `app/topic/clean.py` | **新建**。清洗(脱敏 + 格式),**训练与推理唯一实现** |
| `app/topic/labeling.py` | **新建**。证据串校验、标签漂移检查、分层抽样 —— 全是纯函数,可单测 |
| `app/db/models.py` | 改。加 `TopicClassification` |
| `db/ch10.sql` | **新建**。`topic_classifications` 建表(新表,只 CREATE) |
| `app/api/topics.py` | **新建**。`GET /api/topics/distribution` |
| `app/main.py` | 改。挂 `topics_router`(必须在 `mount("/")` **之前**) |
| `topic_service/` | **新建**。`model.py` / `server.py` / `__main__.py` |
| `scripts/prepare_topic_data.py` | **新建**。语料合流 + 分层抽样 + 增强(三个子命令) |
| `scripts/gen_topic_data.py` | **新建**。配额合成 + 形态强制 + 禁词自检 |
| `scripts/prelabel_topics.py` | **新建**。大模型预标(证据串)+ 断点续跑 |
| `scripts/export_label_review.py` | **新建**。导出/回收人工复核 CSV |
| `scripts/train_topic_clf.py` | **新建**。全参微调 + 早停 + 产物落盘 |
| `scripts/eval_topic_clf.py` | **新建**。指标 + 两张矩阵 + 三层报告 |
| `scripts/classify_topics.py` | **新建**。跑批:读池 → 调服务 → 写表 |
| `scripts/acceptance_ch10.sh` | **新建**。验收 ①–③ |
| `app/static/admin.html` | 改。「主题分布」标签页(Vibe Coding) |
| `tests/test_topic_taxonomy.py` 等 | **新建**。见各任务 |
| `models/topic-clf/` | 训练产物,**gitignore** |

**新增依赖**:无。全部用既有依赖(torch / transformers / datasets / sklearn / FastAPI / SQLAlchemy)。**不许装 `optimum`**(理由见 spec §2.4 —— 它会把 transformers 降到 4.57,毁掉 ch03/ch04 的检索链路)。

---

## Task 1: 17 类权威表 `app/topic/taxonomy.py` + 给用户过目的 CSV

**Files:**
- Create: `app/topic/__init__.py`(空)
- Create: `app/topic/taxonomy.py`
- Create: `tests/test_topic_taxonomy.py`
- Create: `scripts/export_taxonomy_review.py`(把表导成 CSV 供 CP-1 过目)

**Interfaces:**
- Consumes: `app.agent.routing.INTENT_LABELS`(任务 A1 之后的 9 元组)
- Produces:
  - `LABELS: tuple[str, ...]` —— 17 个类目,**顺序即标签 id 顺序**(训练与推理都以此为准)
  - `HEAD_LABELS: tuple[str, ...]` —— 头四类,必须等于 `LABELS[:4]`
  - `BOUNDARY: dict[str, str]` —— 类目 → 一句边界说明
  - `POSITIVE: dict[str, tuple[str, ...]]` —— 类目 → 正例原话
  - `COUNTER: dict[str, tuple[tuple[str, str], ...]]` —— 类目 → ((反例原话, 应归类目), …)
  - `INTENT_TO_TOPICS: dict[str, tuple[str, ...]]` —— 9→17 **有损投影**
  - `render_taxonomy_for_prompt() -> str` —— 渲染成给预标/合成 prompt 用的文本块
  - `OTHER = "其他"`

- [ ] **Step 1: 写失败的测试**

Create `tests/test_topic_taxonomy.py`:

```python
"""17 类权威表:它是**全章的唯一类目来源**,所以这里断的是它的形状与自洽。"""

import re

import pytest

from app.agent.routing import INTENT_LABELS
from app.topic.taxonomy import (
    BOUNDARY,
    COUNTER,
    HEAD_LABELS,
    INTENT_TO_TOPICS,
    LABELS,
    OTHER,
    POSITIVE,
    render_taxonomy_for_prompt,
)

EXPECTED = (
    "退换货", "物流", "尺码", "发票", "质量问题", "运费", "优惠活动", "价保",
    "支付", "订单修改", "库存补货", "商品信息", "保修维修", "账号", "会员积分",
    "评价", "其他",
)


def test_labels_are_exactly_the_seventeen():
    """顺序也是契约 —— 训练时它是标签 id,推理时它必须逐位相同。"""
    assert LABELS == EXPECTED
    assert len(set(LABELS)) == 17


def test_head_labels_lead_the_table():
    """「退换货、物流、尺码、发票四大类领头」—— 是**位置**,不只是说法。

    顺序错了不会报错,但 `LABELS[:4]` 会被别处当成头四类用(测试集配额、
    报告里的分区),于是「头四类样本更足」这条保证静默失效。
    """
    assert HEAD_LABELS == ("退换货", "物流", "尺码", "发票")
    assert LABELS[:4] == HEAD_LABELS


@pytest.mark.parametrize("label", EXPECTED)
def test_every_label_has_a_boundary_sentence(label):
    """每类都要有边界说明 —— 少一句,近邻类目就靠模型自由发挥。"""
    assert BOUNDARY.get(label, "").strip(), f"{label} 缺边界说明"


@pytest.mark.parametrize("label", EXPECTED)
def test_every_label_has_positive_examples(label):
    if label == OTHER:
        pytest.skip("「其他」的边界是「以上都不是」,不要求正例")
    assert len(POSITIVE.get(label, ())) >= 1, f"{label} 缺正例"


def test_counter_examples_point_at_real_labels():
    """反例必须指向一个**真实存在的**类目 —— 指错了就没人能照着裁决。"""
    for label, pairs in COUNTER.items():
        assert label in LABELS, f"反例表的键 {label} 不是合法类目"
        for text, target in pairs:
            assert target in LABELS, f"{label} 的反例「{text}」指向了不存在的类目 {target}"
            assert target != label, f"{label} 的反例指向了自己"


def test_near_neighbour_boundaries_are_pinned():
    """用户给的三条近邻边界必须**逐条**在表里,且有反例钉着。

    这三条是「近邻类目靠边界说明划开」的全部内容。少了任何一条,
    模型只能靠字面词猜,而这三对恰恰是字面无差别、只有语义差别的。
    """
    assert "保修维修" in COUNTER and any(
        t == "保修维修" for _, t in COUNTER["退换货"]
    ), "「修归保修维修」这条边界没钉住"
    assert "运费" in COUNTER and any(t == "运费" for _, t in COUNTER["物流"]), (
        "「运费管钱、物流管货」这条边界没钉住"
    )
    assert "价保" in COUNTER and any(t == "价保" for _, t in COUNTER["优惠活动"]), (
        "「价保是补差价、优惠活动是券和满减」这条边界没钉住"
    )


def test_intent_labels_are_all_mapped():
    """⚠️ **加意图就必须加映射** —— 这条是任务 A1 加第九类时那条「响亮地失败」的兑现。

    没有它,`INTENT_TO_TOPICS` 会静默少一行,而下游(分布页的意图口径统计)
    只会少一个键,不报错。
    """
    missing = [label for label in INTENT_LABELS if label not in INTENT_TO_TOPICS]
    assert not missing, f"这些意图没有映射到主题:{missing}"


def test_mapping_targets_are_all_real_labels():
    for intent, topics in INTENT_TO_TOPICS.items():
        assert topics, f"{intent} 映射到了空集"
        for t in topics:
            assert t in LABELS, f"{intent} 映射到了不存在的主题 {t}"


def test_non_topical_intents_land_on_other():
    """投诉 / 闲聊 / 转人工**不是主题** —— 它们只能落「其他」。

    这是刻意的(见 spec §4.2),不是漏写。**两个词表的定位差异是设计的一部分**:
    意图答的是「用户想干什么」,主题答的是「问题关于什么」。
    """
    for intent in ("投诉", "闲聊", OTHER, "转人工"):
        assert INTENT_TO_TOPICS[intent] == (OTHER,), (
            f"{intent} 应当映射到「{OTHER}」—— 若这是有意改动,请同时改这条测试与 spec §4.2"
        )


def test_rendered_block_carries_every_label_and_boundary():
    """渲染块是 prompt 的**唯一**类目来源;漏一类,那个类就永远训不出来。"""
    block = render_taxonomy_for_prompt()
    for label in LABELS:
        assert label in block, f"渲染块里没有 {label}"
        if label != OTHER:
            assert BOUNDARY[label] in block, f"渲染块里没有 {label} 的边界说明"
    assert block.count("\n") >= len(LABELS)


def test_rendered_block_has_no_curly_braces():
    """`ChatPromptTemplate` 按 f-string 解析,裸花括号会炸(本仓硬约束)。

    实测报错长这样:`Single '}' is not allowed for a for loop` —— 报错位置指向
    prompt 组装那一行,与「类目表里有个花括号」毫无关系。
    """
    block = render_taxonomy_for_prompt()
    assert "{" not in block and "}" not in block
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_taxonomy.py`
Expected: **FAIL** —— `ModuleNotFoundError: No module named 'app.topic'`

- [ ] **Step 3: 建 `app/topic/__init__.py`(空)与 `app/topic/taxonomy.py`**

`app/topic/taxonomy.py` 的模块 docstring 必须写清三件事(它们各自对应一条测试):

```python
"""17 类权威主题表 —— **全章唯一的类目来源**。

**谁在用**:预标 prompt、合成配额、训练标签顺序、评测报告的分区、
分布页的图例 —— 全部 import 这个模块。任何一处再手写一份类目清单,
就是漂移的开始(本仓已有先例:`retrieval_score_threshold` 改一处漏一处)。

**「主题」不是「意图」。** 意图(ch06 的九类)答的是「用户想干什么」,
主题答的是「问题关于什么」。所以「投诉 / 闲聊 / 转人工」这三个意图
**没有对应的主题**,只能落「其他」(见 `INTENT_TO_TOPICS`)。
这不是漏写,是两套词表的定位差异。

**`LABELS` 的顺序是契约**:训练时它是标签 id 的顺序,推理时必须逐位相同。
两侧不一致会产出一张**完全错的分布图,而每个组件都工作正常** ——
所以顺序写进训练产物(`labels.json`),服务读产物、不自己写。
"""

LABELS: tuple[str, ...] = (
    "退换货", "物流", "尺码", "发票", "质量问题", "运费", "优惠活动", "价保",
    "支付", "订单修改", "库存补货", "商品信息", "保修维修", "账号", "会员积分",
    "评价", "其他",
)
HEAD_LABELS: tuple[str, ...] = LABELS[:4]

BOUNDARY: dict[str, str] = {
    # 边界说明**必须能裁决**,不能是类名的同义反复。
    # 反面教材:「退换货 = 关于退换货的问题」—— 那等于没说。
    "退换货": "要退货、换货、退款,或询问退换货的规则与流程",
    "物流": "包裹的**位置与运输**:发货没、到哪了、何时到、配送范围(管**货**)",
    "尺码": "尺寸、大小、肥瘦、身高体重对应的码数",
    "发票": "开票、发票内容、抬头、税率、寄送",
    "质量问题": "**到手就坏**:破损、色差、起球、漏水、少发漏发",
    "运费": "**钱**:运费多少、包邮门槛、偏远加价、退货运费谁承担(管**钱**)",
    "优惠活动": "券、满减、折扣、秒杀、活动规则",
    "价保": "**补差价**:买后降价退差价、价保期限与申请",
    "支付": "付款方式、支付失败、分期、账单、扣款异常",
    "订单修改": "改地址、改商品、取消、合并(**未发货前的改单**)",
    "库存补货": "有没有货、何时补货、到货通知、预售",
    "商品信息": "商品的**属性与参数**:材质、成分、功能、规格、型号、适用场景",
    "保修维修": "**修**:保修期、保修范围、维修流程、上门/寄修",
    "账号": "登录、注册、密码、绑定手机/邮箱、被盗、实名",
    "会员积分": "会员等级、积分获取与抵扣、会员权益",
    "评价": "评价、晒单、追评、差评删除、评价奖励",
    OTHER: "以上都不是",
}

POSITIVE: dict[str, tuple[str, ...]] = { ... }   # 见 spec §4.1 的正例列,逐类照抄
COUNTER: dict[str, tuple[tuple[str, str], ...]] = { ... }  # 见 spec §4.1 的反例列

INTENT_TO_TOPICS: dict[str, tuple[str, ...]] = {
    "物流": ("物流",),
    "订单": ("订单修改",),
    "商品咨询": ("商品信息",),
    "退款退货": ("退换货",),
    # ch06 的「售后」把「修」与「坏」揉成一个,这里拆开 —— **这是一处有损投影**,
    # 别期待 逆向(正向(x)) == x。
    "售后": ("保修维修", "质量问题"),
    "投诉": (OTHER,),
    "闲聊": (OTHER,),
    OTHER: (OTHER,),
    "转人工": (OTHER,),
}


# ⚠️⚠️ **计划订正 5-A(controller,2026-09-26)—— 这是 Task 1 的原稿,已被 Task 5 Step 0 取代** ⚠️⚠️
# 照抄下面这一版会**静默回退「计划订正 D」**(把 `POSITIVE` 重新挤出 prompt)。
# 现在多加一行:每类的正例也要渲染(形如 `    · 例:「{text}」`),`COUNTER` 没声明的 7 类照样有。
# 改的是**渲染函数**,`POSITIVE` / `BOUNDARY` / `COUNTER` 常量表与用户签过字的
# `evals/topic/taxonomy.csv` **一字未动**。真在 `app/topic/taxonomy.py`。
def render_taxonomy_for_prompt() -> str:
    """渲染成类目表文本块,供预标与合成的 prompt 使用。

    **不含花括号**(`ChatPromptTemplate` 按 f-string 解析)。
    含边界说明、正例与反例 —— 反例是「近邻类目怎么划开」的实际内容,
    只给边界说明不给反例的话,模型对「刚收到就坏了」会犹豫。
    """
    lines = []
    for label in LABELS:
        lines.append(f"- {label}:{BOUNDARY[label]}")
        for text, target in COUNTER.get(label, ()):
            lines.append(f"    · 「{text}」归 {target},不归 {label}")
    return "\n".join(lines)
```

> ⚠️ `BOUNDARY` / `POSITIVE` / `COUNTER` 的**内容逐条照抄 spec §4.1 那张表**。spec 是设计源,这里是它的代码化。

- [ ] **Step 4: 跑测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_taxonomy.py`
Expected: **PASS**(且打印通过数)

- [ ] **Step 5: 写导出脚本**

Create `scripts/export_taxonomy_review.py`:

```python
"""把 17 类表导成 CSV,给人过目(CP-1)。

**为什么要有这一步**:后面所有预标、合成、裁决都照这张表走 ——
它歪了整章数据都歪。而 CSV 能在 Excel 里排着看,比读 Python 源码快得多。

用法:
    .venv/Scripts/python.exe scripts/export_taxonomy_review.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.topic.taxonomy import BOUNDARY, COUNTER, HEAD_LABELS, LABELS, POSITIVE

OUT = Path(__file__).resolve().parents[1] / "evals" / "topic" / "taxonomy_review.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        # utf-8-sig:Excel 在中文 Windows 上按本地编码打开无 BOM 的 CSV 会乱码。
        w = csv.writer(f)
        w.writerow(["#", "类目", "头四类", "边界说明", "正例", "反例(应归哪类)"])
        for i, label in enumerate(LABELS, 1):
            pos = " / ".join(POSITIVE.get(label, ()))
            neg = " / ".join(f"{t}→{tgt}" for t, tgt in COUNTER.get(label, ()))
            w.writerow([i, label, "★" if label in HEAD_LABELS else "", BOUNDARY[label], pos, neg])
    print(f"已写出 {OUT}({len(LABELS)} 行)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 跑导出脚本,交给用户过目(🚧 CP-1,阻塞后续所有任务)**

Run: `.venv/Scripts/python.exe scripts/export_taxonomy_review.py`

**停下来,把 `evals/topic/taxonomy_review.csv` 交给用户。** 明确请他看:
① 每类的边界说明**能不能裁决**(而不是类名的同义反复);② 三条近邻边界有没有问题;
③ 17 类里**没有**「投诉/闲聊/转人工」这一条他是否认可(spec §4.2)。

**用户点头之前不要开工任务 2 及以后。**

- [ ] **Step 7: Commit**

```bash
git add app/topic/__init__.py app/topic/taxonomy.py tests/test_topic_taxonomy.py \
        scripts/export_taxonomy_review.py evals/topic/taxonomy_review.csv
git commit -m "ch10-B T1: 17 类权威主题表 + 9→17 映射 + 给用户过目的 CSV"
```

---

## Task 2: 清洗 `app/topic/clean.py`(训练/推理唯一实现)

**Files:**
- Create: `app/topic/clean.py`
- Create: `tests/test_topic_clean.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `clean(text: str) -> str` —— **训练与推理都必须调它**
  - `redact(text: str) -> str` / `normalize(text: str) -> str`(分开导出,便于单测各自的口径)

- [ ] **Step 1: 写失败的测试**

Create `tests/test_topic_clean.py`:

```python
"""清洗:脱敏 + 格式。**不做错别字修正**(ch10 spec §5.1,用户批准的设计变更)。"""

import pytest

from app.topic.clean import clean, normalize, redact


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("我的手机号是13800138000,发货了吗", "我的手机号是<手机号>,发货了吗"),
        ("订单20240915001什么时候到", "订单<订单号>什么时候到"),
        ("邮箱 a.b@example.com 能改吗", "邮箱 <邮箱> 能改吗"),
    ],
)
def test_redact_replaces_identifiers(raw, expected):
    assert redact(raw) == expected


def test_redact_keeps_short_numbers():
    """短数字**不是**标识符:「满99元包邮」「175穿什么码」里的数是语义的一部分。

    把 99 也脱敏掉,「满多少钱包邮」这类问题的判别特征就没了 ——
    而这正是头四类之一的运费类。
    """
    assert redact("满99元包邮") == "满99元包邮"
    assert redact("175穿什么码") == "175穿什么码"


def test_normalize_folds_fullwidth_and_whitespace():
    assert normalize("退货  怎么   走?") == "退货 怎么 走?"
    assert normalize("退货!!!怎么走") == "退货!怎么走"
    assert normalize("　退货　怎么走　") == "退货 怎么走"


def test_normalize_fullwidth_to_halfwidth():
    """全角标点与字母对模型是**不同的 token**,归一化能省下不少表示成本。"""
    assert normalize("ＡＢＣ１２３") == "ABC123"
    assert normalize("退货,怎么走") == "退货,怎么走"


def test_clean_does_not_fix_typos():
    """**这条断的是一个「没做」的决定,不是疏忽。**

    改错别字会造成训练/部署不一致,而且**测试集也会被修过 ⇒ F1 虚高且测不出**。
    错别字保留原样,由增强阶段主动注入(任务 8)。
    `clean` 若哪天开始「顺手修一下」,这条会红 —— **那是设计变更,要改的是 spec**。
    """
    assert clean("我要退或,买大 了") == "我要退或,买大 了"


def test_clean_is_idempotent():
    """清洗两次 == 清洗一次。

    不幂等的话,「训练侧洗过一遍、推理侧再洗一遍」会得到不同的文本 ——
    train/serve skew 的一个隐蔽来源。
    """
    for raw in ("我的手机号是13800138000,发货了吗", "退货  怎么   走?", "ＡＢＣ"):
        once = clean(raw)
        assert clean(once) == once


def test_clean_handles_empty_and_whitespace_only():
    assert clean("") == ""
    assert clean("   ") == ""
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_clean.py`
Expected: **FAIL** —— `ModuleNotFoundError: No module named 'app.topic.clean'`

- [ ] **Step 3: 写实现**

Create `app/topic/clean.py`:

```python
"""主题分类语料的清洗 —— **训练侧与推理侧的唯一实现**。

⚠️ **这个模块必须被两侧同时 import**:
`scripts/prepare_topic_data.py`(训练语料)与 `scripts/classify_topics.py`(推理)。
只在一侧清洗 = 经典的 train/serve skew:模型在真机上看到的是另一种文本。
`tests/test_topic_clean.py::test_both_sides_use_the_same_clean` 守着这条。

**刻意不做错别字修正**(用户 2026-09-25 批准)。理由:
① 改干净会让训练语料比真实用户输入更规整,而**测试集也会被同样修过**
   ⇒ 指标虚高,且这个下降在指标上不可见;
② 修正本身会引入幻觉(「起球」→「气球」),静默污染全部数据;
③ 这件事应该反过来做 —— 增强阶段**主动注入**错别字(见
   `scripts/prepare_topic_data.py` 的 augment 子命令)。
"""

import re
import unicodedata

#: 手机号。**必须带 `1[3-9]` 前缀**,不能写成 `\d{11}` ——
#: 后者会把「20240915001」这类订单号也吃掉一半。
_PHONE = re.compile(r"1[3-9]\d{9}")

#: 订单号 / 长数字串。**下限取 8**:实测订单号形如 `20240915001`(11 位),
#: 而「满99元」「175码」这类语义数字都短于 8 位。下限取小了会误伤语义数字。
_LONG_NUMBER = re.compile(r"\d{8,32}")

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_PUNCT_RUN = re.compile(r"([!?,.;:])\1+")
_WS_RUN = re.compile(r"\s+")


def redact(text: str) -> str:
    """脱敏:手机号 / 订单号 / 邮箱 → 占位符。

    顺序有意义:**先手机号,再长数字**。反过来的话手机号会先被
    `<订单号>` 吃掉(它也是 8 位以上的数字串)。
    """
    text = _PHONE.sub("<手机号>", text)
    text = _EMAIL.sub("<邮箱>", text)
    return _LONG_NUMBER.sub("<订单号>", text)


def normalize(text: str) -> str:
    """格式归一:全角→半角、合并空白、去重复标点、去首尾空白。"""
    # NFKC 把全角字母数字标点折成半角(ＡＢＣ１２３ → ABC123)。
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT_RUN.sub(r"\1", text)
    text = _WS_RUN.sub(" ", text)
    return text.strip()


def clean(text: str) -> str:
    """`normalize` → `redact` → `normalize`。

    两步各自为什么:

    - **第一个 `normalize`**:把全角字母数字标点折成半角,后面的正则才谈得上匹配
      (「１２３４５６７８」这种全角数字串也满足 `\\d{8,32}` —— Python 的 `\\d`
      是 Unicode 感知的)。
    - **`redact`**:把手机号 / 邮箱 / 长数字串换成占位符。
    - **第二个 `normalize`(⚠️ 不是装饰)**:`redact` **插进去的占位符本身
      不是 NFKC 稳定的** —— 尖括号 `<` / `>` 会与**紧随其后的组合标记**规范组合
      (`<` + U+0338 → U+226E ≮,`>` + U+0338 → U+226F ≯)。所以这一遍保证的是
      「**输出一定是 NFKC 规范形式**」,而不只是「清理插入产生的空白」。

    ⚠️ **这一条是被实测纠正回来的**(ch10-B T2 复审):原先这里写的是「第三个
    normalize 是冗余、可证的 no-op」,而那是**错的** —— NFKC 作用于**相邻对**,
    占位符边界会与后随的组合标记组合。0x110000 全码点穷举找到**恰好一个**反例
    码点 U+0338。**别再把这一步当冗余删掉。**
    """
    return normalize(redact(normalize(text)))
```

- [ ] **Step 4: 加那条「两侧同源」的源码扫描测试**

追加到 `tests/test_topic_clean.py` 末尾:

```python
def test_both_sides_use_the_same_clean():
    """⚠️ **训练侧与推理侧必须 import 同一个 `clean`。**

    这条用**源码扫描**而不是运行时断言 —— 要抓的是「有没有人写下去」,
    不是「跑到了没有」(照 ch09 `test_no_module_outside_observability_imports_langfuse`
    的先例)。

    漏掉任何一侧的后果:模型在真机上看到的是没洗过的(或洗过两遍的)文本,
    而**没有任何东西会报错** —— 它只让线上准确率悄悄低于测试集读数。
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"^\s*from\s+app\.topic\.clean\s+import\s+.*\bclean\b", re.M)

    for rel in ("scripts/prepare_topic_data.py", "scripts/classify_topics.py"):
        path = root / rel
        assert path.exists(), f"{rel} 不存在 —— 这条守卫失去了目标,请更新它而不是删掉"
        assert pattern.search(path.read_text(encoding="utf-8")), (
            f"{rel} 没有 import app.topic.clean.clean —— train/serve skew 的开始"
        )
```

> ⚠️ **这条测试在任务 3/13 之前会红**(那两个脚本还不存在)。这是**刻意的**:把它写在清洗任务里,等于给后面两个脚本**预先钉了一个必须满足的条件**。执行者看到红是正常的,任务 3 与任务 13 各自让它变绿一半 —— **不许为了让测试变绿而放宽这条断言**。

- [ ] **Step 5: 跑清洗自身的测试(排除上一步那条)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_clean.py -k "not both_sides"`
Expected: **PASS**

- [ ] **Step 6: Commit**

```bash
git add app/topic/clean.py tests/test_topic_clean.py
git commit -m "ch10-B T2: 清洗(脱敏+格式,不做错别字修正)+ 训练/推理同源守卫"
```

---

## Task 3: 语料合流(三源 → 一份去重语料)

**Files:**
- Create: `scripts/prepare_topic_data.py`(本任务只做 `collect` 子命令)
- Create: `app/topic/labeling.py`(本任务只放 `dedupe_questions`)
- Create: `tests/test_topic_labeling.py`

**Interfaces:**
- Consumes: `app.topic.clean.clean`
- Produces:
  - `app.topic.labeling.dedupe_questions(rows: list[dict]) -> list[dict]` —— 按清洗后的文本去重,保留**首个**来源;每行有 `provenance`(`"real"`)与 `source`(`"pool"|"chat"|"evalmd"`)
  - `scripts/prepare_topic_data.py collect` → 写 `evals/topic/corpus.jsonl`

- [ ] **Step 1: 写失败的测试**

Create `tests/test_topic_labeling.py`:

```python
"""标注相关的纯函数:去重、证据串校验、标签漂移、分层抽样。"""

import pytest

from app.topic.labeling import dedupe_questions


def test_dedupe_keeps_first_occurrence_and_source():
    """重复问题只留一条 —— 且**保留先出现的那个来源**(来源优先级由调用方排序决定)。"""
    rows = [
        {"question": "退货怎么走", "source": "pool"},
        {"question": "退货怎么走", "source": "chat"},
    ]
    out = dedupe_questions(rows)
    assert len(out) == 1
    assert out[0]["source"] == "pool"


def test_dedupe_is_on_the_cleaned_text():
    """「退货  怎么走」与「退货怎么走」是同一句 —— 按原文去重会漏。

    ⚠️ **这条测试自证的是「机制对」,不是「今天差多少」。** 实测(2026-09-26):
    按清洗后去重与按原文去重,在当天的 1259 行真实语料上**只差 1 条**,
    而且那 1 条的差异来自**脱敏** —— 具体是**订单号**:唯一的多变体清洗键是
    `订单 <订单号> 的物流到哪了`,由 `订单 20240808 的物流到哪了` 与
    `订单 20240901 的物流到哪了` 两条脱敏后撞成一条。
    **今天池子+对话里零个「只差空白或全角」的变体对。**
    ⇒ 设计仍然对(清洗后才是真正要分类的文本),但**别把「空白/全角很常见」
    当成实测事实引用** —— 它是「应该如此」的稳健性论证,不是「测得如此」。
    """
    rows = [
        {"question": "退货 怎么走", "source": "pool"},
        {"question": "退货　　怎么走", "source": "chat"},   # 全角空格
    ]
    assert len(dedupe_questions(rows)) == 1


def test_dedupe_drops_blank_after_cleaning():
    assert dedupe_questions([{"question": "   ", "source": "pool"}]) == []


def test_dedupe_of_empty_list_is_empty():
    assert dedupe_questions([]) == []
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_labeling.py`
Expected: **FAIL** —— `ModuleNotFoundError: No module named 'app.topic.labeling'`

- [ ] **Step 3: 写 `app/topic/labeling.py` 的第一个函数**

```python
"""标注链路上的纯函数 —— 全部零 IO、零依赖,故可密集单测。

放在 `app/topic/` 而不是 `scripts/` 里,是因为**它们要被两侧用**:
去重与分层抽样既在离线造数据时跑,也在评测脚本里跑(测试集冻结后的再切分)。
"""

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
```

- [ ] **Step 4: 写 `scripts/prepare_topic_data.py` 的 `collect` 子命令**

```python
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from app.db.base import get_engine
from app.topic.clean import clean           # noqa: F401  ← 同源守卫要求这一行
from app.topic.labeling import dedupe_questions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "topic" / "corpus.jsonl"
TESTING_MD = ROOT / "evals" / "测试集.md"


async def _from_db() -> list[dict]:
    """池子 + 对话里的用户话。

    两处都取**去重前**的全量:去重交给 `dedupe_questions` 一处做 ——
    两个地方各去一遍的话,「按什么去重」这件事就有两份实现。
    """
    rows: list[dict] = []
    eng = get_engine()
    async with eng.connect() as conn:
        # ⚠️ **两条查询的 source 必须不同**(`pool` / `chat`)。这里一度两处都写成 `"pool"`,
        #    与下面 `collect()` 里那段优先级注释自相矛盾。
        #    它是**承重的**,不是标签洁癖 —— 实测(2026-09-26,三种配置各跑一遍真实语料):
        #      · 两处都写 `"pool"` ⇒ `{'pool': 163, 'evalmd': 299}` —— **chat 整个消失**,
        #        两个 DB 来源**再也分不开**;
        #      · **顺序反过来**(chat 在前、pool 在后,标签仍不同)⇒ `{'chat': 163, ...}`
        #        —— 池子一条不剩;
        #      · 正确配置 ⇒ `{'pool': 33, 'chat': 130, 'evalmd': 299}`。
        #    ⇒ **两种写错法给出的是两个不同的错读数**,别把它们当成一回事。
        #    无论哪一种,**都没有任何东西会报错** —— 只有来源分布悄悄不对。
        for q, src in (
            (text("SELECT question FROM low_confidence_questions ORDER BY id"), "pool"),
            (text("SELECT content FROM messages WHERE role='user' ORDER BY id"), "chat"),
        ):
            for (value,) in (await conn.execute(q)).all():
                rows.append({"question": value, "source": src})
    await eng.dispose()
    return rows


def _from_testing_md() -> list[dict]:
    """`evals/測試集.md` 的 300 条**人工写的**政策问句 —— 借用作分类语料。

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
    kept = dedupe_questions(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, r in enumerate(kept, 1):
            f.write(json.dumps({"id": f"r-{i:04d}", **r}, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"写出 {len(kept)} 条 → {OUT}")
    print("  来源分布:", dict(Counter(r["source"] for r in kept)))
    print("  清洗后为空被丢弃:", len(rows) - len(kept))
    # ⚠️ 这个数会**随运行次数增长**(验收脚本每跑一次就往池子写),
    #    所以引用它时必须带日期,别当常量。


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["collect", "split", "augment"])
    args = ap.parse_args()
    if args.step == "collect":
        asyncio.run(collect())
    else:
        raise SystemExit(f"{args.step} 还没实现(由后续任务补上)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 跑收集,拿真实读数**

Run: `.venv/Scripts/python.exe scripts/prepare_topic_data.py collect`(需 MySQL 在 3307 运行)

Expected: 打印去重后的条数与来源分布。**把这三个数抄进 `dev-notes/ch10.md`。**

> ⚠️ 若池子里有 `<订单号>` 之类占位符比例异常高,先检查 `redact` 的长数字下限(8)是不是吃掉了语义数字 —— 用 `grep -c "<订单号>" evals/topic/corpus.jsonl` 看一眼实际分布,别凭感觉。

- [ ] **Step 6: 跑测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_labeling.py`
Expected: **PASS**

- [ ] **Step 7: Commit**

```bash
git add app/topic/labeling.py scripts/prepare_topic_data.py tests/test_topic_labeling.py
git commit -m "ch10-B T3: 清洗模块 + 三源合流语料(去重按清洗后文本)"
```

---

## Task 4: 配额合成 + 形态强制 + 禁词自检

**Files:**
- Create: `scripts/gen_topic_data.py`
- Create: `tests/test_gen_topic_data.py`(测**纯函数**,不打网络)

**Interfaces:**
- Consumes: `app.topic.taxonomy`(LABELS / HEAD_LABELS / `render_taxonomy_for_prompt`)
- Produces:
  - `app.topic.synth.QUOTA: dict[str, int]` —— 每类配额(头四类加倍)
  - `app.topic.synth.FORMS: dict[str, float]` —— `{"single": 0.55, "multi": 0.30, "boundary": 0.15}`
  - `app.topic.synth.violates_forbidden(question: str) -> list[str]` —— 返回命中的禁词列表(空 = 合规)
  - `app.topic.synth.shape_ok(labels: list[str]) -> bool` —— 多标签形态检查
  - `scripts/gen_topic_data.py` → `evals/topic/synthetic.jsonl`

- [ ] **Step 1: 写失败的测试**

Create `tests/test_gen_topic_data.py`:

```python
"""合成数据的**形态与禁区**检查 —— 全是纯函数,不打网络。"""

import pytest

from app.topic.synth import (
    FORMS,
    QUOTA,
    forbidden_words,
    shape_ok,
    violates_forbidden,
)
from app.topic.taxonomy import HEAD_LABELS, LABELS


def test_quota_covers_every_label():
    assert set(QUOTA) == set(LABELS)


def test_head_labels_get_a_bigger_quota():
    """「四大类领头」在**配额上**也要是真的,不能只是术语表里的说法。"""
    others = [q for lb, q in QUOTA.items() if lb not in HEAD_LABELS]
    assert all(QUOTA[lb] >= max(others) for lb in HEAD_LABELS), (
        "头四类的配额必须不低于其余类的最大配额"
    )


def test_other_class_has_a_quota():
    """「其他」**要留配额** —— 语料里它不会少,而合成时最容易忘。"""
    assert QUOTA["其他"] > 0


def test_multi_label_share_is_at_least_thirty_percent():
    """多标签占比 ≥30% 是**方案 A 的全部价值所在**。

    少了它,模型退化成「只报最显眼的那个类」,验收 ③
    (「买大了想退」同时命中多类)从原理上过不了。
    """
    assert FORMS["multi"] >= 0.30


def test_boundary_share_is_at_least_fifteen_percent():
    """近邻边界对 ≥15% —— 少了它,`运费↔物流`、`退换货↔保修维修` 学不开。"""
    assert FORMS["boundary"] >= 0.15


def test_forms_sum_to_one():
    assert abs(sum(FORMS.values()) - 1.0) < 1e-9


@pytest.mark.parametrize(
    "question",
    [
        "保修维修怎么办",          # 类目名
        "我想问退换货的事",        # 类目名
        "关于质量问题",            # 类目名
        "运费管钱物流管货",        # 边界说明的原词
        "价保是补差价",            # 边界说明的原词
    ],
)
def test_forbidden_words_catch_label_names_and_boundary_phrases(question):
    """**禁词**:合成的问句里不许出现类目名或边界说明的原词。

    不设这条,模型学到的就是**关键词匹配** —— F1 会很漂亮,
    而真机上一句不含类目名的真话就废了。这是本章最贵的一条数据纪律。
    """
    assert violates_forbidden(question), f"「{question}」应当命中禁词"


@pytest.mark.parametrize(
    "question",
    ["买大了想退", "快递到哪了", "能开专票吗", "满多少包邮", "用一年了能修吗"],
)
def test_real_questions_pass_the_filter(question):
    """**反向**:真实用户话本来就不含类目名 —— 这条防止禁词表宽到把真话也拦了。

    禁词表写宽了的后果是**合成数据全被丢弃**,而执行者只看到「生成了 0 条」,
    容易误以为是模型没返回,而不是自己的过滤器写坏了。
    """
    assert violates_forbidden(question) == []


def test_forbidden_words_are_derived_from_taxonomy_not_hand_written():
    """禁词表必须**从 `taxonomy` 派生**,不能手写一份。

    手写的会与类目表漂移:加了第 18 类而忘了加进禁词表,
    合成数据里就会出现类目名的原词 —— 静默的,只有真机才暴露。
    """
    words = forbidden_words()
    assert "退换货" in words and "保修维修" in words


def test_shape_ok_accepts_single_and_multi():
    assert shape_ok(["尺码"])
    assert shape_ok(["尺码", "退换货"])
    assert shape_ok(["尺码", "退换货", "运费"])


def test_shape_ok_rejects_empty_duplicates_and_overlong():
    assert not shape_ok([]), "没有标签的样本没有训练价值"
    assert not shape_ok(["尺码", "尺码"]), "重复标签是标注错误"
    assert not shape_ok(["尺码", "退换货", "运费", "价保"]), "4 个诉求超出本章的形态假设"
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_gen_topic_data.py`
Expected: **FAIL** —— `ModuleNotFoundError: No module named 'app.topic.synth'`

- [ ] **Step 3: 写 `app/topic/synth.py`**

```python
"""合成数据的**配额与禁区** —— 纯数据 + 纯函数,因为它们是可单测的那一半。

方案 A(用户 2026-09-25 批准)的全部价值在这三件事上:
① **配额**:17 类每类都够;头四类加倍;「其他」不能忘。
② **形态强制**:多标签 ≥30%、近邻边界对 ≥15%。
③ **禁词**:问句里不许出现类目名与边界说明的原词。

三条都不会在训练时暴露问题 —— 它们是**数据纪律**,只能在造数据时守住。
"""

from app.topic.taxonomy import BOUNDARY, HEAD_LABELS, LABELS, OTHER

#: 每类配额。头四类加倍。总量约 800(真实语料约 400,合起来 ~1200)。
_BASE = 45
QUOTA: dict[str, int] = {lb: (_BASE * 2 if lb in HEAD_LABELS else _BASE) for lb in LABELS}

FORMS: dict[str, float] = {"single": 0.55, "multi": 0.30, "boundary": 0.15}


def forbidden_words() -> set[str]:
    """**从 taxonomy 派生**类目名与边界说明里的实词短语。

    刻意手写一份的话,加了类目就会漏 —— 而漏了不会报错,
    只会让某几类悄悄带上「术语表腔」,在真机上失效。

    边界说明的取词规则:去掉解释性的括号与标点,取其中的**实词短语**。
    这里保守地取「说明里出现的类目名」+「三条近邻边界的核心词」,
    而不是把整句说明拆成词 —— 拆词会误伤正常问句(见下面那条反向测试)。
    """
    words = set(LABELS)
    # 三条近邻边界的核心词(spec §4.1,用户给定)。
    words |= {"管钱", "管货", "补差价", "券和满减"}
    # 「以上都不是」是「其他」的说明,不是问句里会出现的东西,排掉。
    words.discard(OTHER)
    return words


def violates_forbidden(question: str) -> list[str]:
    """返回命中的禁词(空列表 = 合规)。**返回明细而不是布尔**,便于报表统计。"""
    return sorted(w for w in forbidden_words() if w in question)


def shape_ok(labels: list[str]) -> bool:
    """多标签形态检查:非空、不重复、≤3 个、都是合法类目。"""
    if not labels or len(labels) > 3:
        return False
    if len(set(labels)) != len(labels):
        return False
    return all(lb in LABELS for lb in labels)
```

- [ ] **Step 4: 在 `tests/test_gen_topic_data.py` 补一条反向约束**

```python
def test_forbidden_words_do_not_swallow_ordinary_questions():
    """禁词表不许宽到把**真实问句**也拦了。

    这条是上一条的反面。禁词表写宽了的表现是「合成数据全被丢弃」,
    而执行者只会看到「生成了 0 条」,很容易误判成模型没返回。
    """
    from app.topic.synth import violates_forbidden

    for q in (
        "这个多少钱",
        "什么时候能到",
        "能不能开票",
        "我要改成另一个地址",
        "这个颜色还有别的吗",
    ):
        assert violates_forbidden(q) == [], f"「{q}」是正常问句,不该被禁词表拦住"
```

- [ ] **Step 5: 写 `scripts/gen_topic_data.py`(调模型,打网络)**

```python
"""合成语料:配额 + 形态强制 + 禁词自检。

**打网络** —— 不是单测,不进 `pytest` 默认集。

用法:
    .venv/Scripts/python.exe scripts/gen_topic_data.py --dry-run   # 只打印 prompt
    .venv/Scripts/python.exe scripts/gen_topic_data.py
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.llm import create_extract_model
from app.topic.synth import FORMS, QUOTA, shape_ok, violates_forbidden
from app.topic.taxonomy import HEAD_LABELS, OTHER, render_taxonomy_for_prompt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "topic" / "synthetic.jsonl"

#: 一次要几条。**不要一次要 200 条** —— 长输出里模型会开始重复自己,
#: 而重复的样本对训练没有增量,却会占掉配额。
BATCH = 12


def build_prompt(label: str, form: str, n: int) -> str:
    """组装一批合成用的提示词。

    ⚠️ 三条硬约束写进 prompt,它们对应 `test_gen_topic_data.py` 的三条形态测试:
    ① **不许出现类目名与边界说明的原词**;
    ② `multi` 形态必须**真的带 2–3 个诉求**,且每个诉求在句子里**字面可指**;
    ③ 说人话 —— 像真实买家在客服窗口里打出来的,带口语与省略。

    ⚠️ 本仓硬约束:提示词里必须出现字面 `JSON` 字样,且**不得有裸花括号**。
    """
    # ⚠️⚠️ **计划订正 A(controller,2026-09-26)—— 下面这段是计划原稿,已实现取代** ⚠️⚠️
    # 真在 `scripts/gen_topic_data.py`。**照抄下面这一版会原样复发 I1**:
    #
    # 旧 `boundary` 只写「与它最容易混的那个类目同时出现,让人必须读完才分得清」——
    # **一个字都没提标签基数**,读起来像一道**单选辨析题**。实测后果:边界形式产出
    # **147/147 全单标签**(而 `multi` 对解剖学上一样的双诉求句打 2–3 个),于是
    # 「≥15% 近邻边界对」这项覆盖**名不副实**。
    #
    # 现在的实现(订正轮 1 + 2):
    #   ① **明写基数** —— 边界批两个诉求**都要打标**;
    #   ② 把 `taxonomy.COUNTER[label]` **声明的那个易混类目**直接交给模型,**不让它自己挑**;
    #   ③ `COUNTER` **没声明**的 7 类(§4.1 反例列写 `—` 的那 7 行)退回共现措辞 ——
    #      ⚠️ **这 56 条是「共现对」,不是「真近邻对」**:复审逐行核过 147 条,
    #      其中真近邻对只有 **91 条**;`发票`/`支付`/`其他` 三类甚至**逐行换邻居**。
    #      裁定是**接受并记账**(理由见 `check_rows` 报表里的拆分与 SDD 账本),
    #      **不在这里补 `COUNTER`** —— 那是改 taxonomy,是全章唯一的类目来源。
    form_desc = {
        "single": f"只涉及「{label}」**一个**诉求",
        "multi": f"涉及「{label}」**以及另外 1–2 个**不同的诉求(共 2–3 个)",
        "boundary": f"涉及「{label}」,而且**与它最容易混的那个类目**同时出现,"
                    f"让人必须读完才分得清",
    }[form]
    return f"""你在为一个电商客服系统造**训练语料**。

下面是本系统权威的类目表(含每类的边界说明与容易混的反例):  # ⚠️ 订正 5-C:真实现已改成「边界说明、**正例**与容易混的反例」

{render_taxonomy_for_prompt()}

请造 {n} 条**真实的买家问句**,要求:
1. 这些问句的主诉求是「{label}」,且{form_desc}。
2. **绝对不许出现任何类目名**(如类目表里那些词),也**不许出现边界说明里的原词**。
   买家不会说「我要咨询退换货类问题」,他只会说「买大了想退」。
3. 像真人在客服窗口打出来的:口语、有省略、可以带错别字。
4. 每条 5–30 个字。

输出一个 JSON 数组,每个元素是对象,含两个字段:
- question:问句本身
- labels:该问句应当打上的标签数组(**标签名必须逐字来自上面的类目表**)

只输出 JSON,不要别的内容。"""


async def run(dry_run: bool) -> None:
    settings = get_settings()
    model = create_extract_model(settings)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    kept, dropped = 0, 0
    with OUT.open("a", encoding="utf-8") as f:      # append:断点续跑
        for label in QUOTA:
            for form, share in FORMS.items():
                n = max(2, round(QUOTA[label] * share))
                prompt = build_prompt(label, form, n)
                if dry_run:
                    print(f"--- {label}/{form} n={n} ---\n{prompt[:400]}…\n")
                    continue
                raw = await _call(model, prompt)
                for item in raw:
                    # ⚠️⚠️ **计划订正 B(controller,2026-09-26)—— 这段也是计划原稿,已实现取代** ⚠️⚠️
                    # 两处会直接坑到下一个执行者,**别照抄**:
                    #
                    # ① **行形状少了 `"id"`。** 下面写的是六个键,而**同一份计划的 Task 5**
                    #    (本文档 `todo = [r for r in rows if r["id"] not in done]`)按 `r["id"]` 索引
                    #    ⇒ 照这段抄,Task 5 读到第一条合成数据就 `KeyError: 'id'`。
                    #    实现里发的是 `"id": f"s-{n:04d}"`(`s-0001`…`s-0962`)。
                    #    ⚠️ **`s-` 前缀是必要的**:真实语料那边是 `r-0001`…`r-0462`,
                    #    两个文件被 Task 5 合流进**同一个列表**,前缀撞了就会互相顶掉。
                    #
                    # ② **三道门是内联的 `if`,不是那张唯一判据表。** 现在是
                    #    `reject_reason(item, seed_label, form) -> str | None`(**一处实现、两处调用**:
                    #    生成侧 `accept` 读它拦输入,核产物时 `check_rows` 读**同一张表**复核)。
                    #    为什么非要抽出来:真跑 962 条 `dropped = 0` ⇒ 这几道门在生产上
                    #    **一次都没开过火** —— 「禁词表写坏了」与「模型没写禁词」在读数上长得一样。
                    #    ⚠️ 抽出来之后又补了**形态基数**(`multi`/`boundary` 必须 ≥2 个标签):
                    #    「boundary 应当是 2 个」这条要求原先**只活在提示词里**(见订正 A)。
                    q, labels = item.get("question", ""), item.get("labels", [])
                    if not shape_ok(labels):
                        dropped += 1
                        continue
                    if violates_forbidden(q):
                        dropped += 1
                        continue
                    if label not in labels and label != OTHER:
                        dropped += 1
                        continue
                    f.write(json.dumps(
                        {"question": q, "labels": labels, "provenance": "synthetic",
                         "source": "gen", "form": form, "seed_label": label},
                        ensure_ascii=False) + "\n")
                    kept += 1
                f.flush()
    if not dry_run:
        # ⚠️ 这两个数**必须打出来**。丢弃率高得离谱通常意味着**禁词表写宽了**
        #    或模型没守格式 —— 而不是「模型不行」。别跳过这个读数。
        print(f"保留 {kept} 条,丢弃 {dropped} 条(丢弃率 {dropped / max(1, kept + dropped):.1%})")


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印 prompt,不打网络")
    args = ap.parse_args()
    asyncio.run(run(args.dry_run))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 先干跑,确认 prompt 里没有裸花括号**

Run: `.venv/Scripts/python.exe scripts/gen_topic_data.py --dry-run | head -40`
Expected: 打印两批 prompt,**肉眼确认没有 `{` `}`** —— 有的话 `ChatPromptTemplate` 那条路会炸(本仓硬约束)。

- [ ] **Step 7: 跑纯函数测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_gen_topic_data.py`
Expected: **PASS**

- [ ] **Step 8: Commit**

```bash
git add app/topic/synth.py scripts/gen_topic_data.py tests/test_gen_topic_data.py
git commit -m "ch10-B T4: 配额合成 + 形态强制 + 禁词自检(三条数据纪律)"
```

---

## Task 5: 大模型预标(证据串 + 断点续跑)

**Files:**
- Modify: `app/topic/labeling.py`(加 `validate_evidence`)
- Modify: `tests/test_topic_labeling.py`
- Create: `scripts/prelabel_topics.py`

**Interfaces:**
- Consumes: `app.topic.taxonomy.render_taxonomy_for_prompt`,`app.llm.create_extract_model`
- Produces:
  - `app.topic.labeling.validate_evidence(question: str, labels: list[str], evidence: dict[str,str]) -> tuple[list[str], list[str]]` —— 返回 `(accepted_labels, rejected_labels)`
  - `scripts/prelabel_topics.py` → `evals/topic/prelabeled.jsonl`(append-only,可断点续跑)

- [ ] **Step 0: 把 `POSITIVE` 接进渲染块(计划订正 D,controller 2026-09-26)**

**先读背景再动手。** `app/topic/taxonomy.py` 的 `POSITIVE` 是 B1 按 spec §4.1 的「正例」列
**逐类**收的(17 类每类 ≥1 条,`tests/test_topic_taxonomy.py:70` 有断言守着),
**但它一个 prompt 都没进过** —— `render_taxonomy_for_prompt()` 只渲染 `BOUNDARY` + `COUNTER`。
(`scripts/export_taxonomy_review.py` 把它写进了给用户过目的 CSV,所以它不是死数据;
**死的是「从没进过任何 prompt」这一半**。)

而 `POSITIVE` 自己的 docstring 逐字写着:

> 「其他」也有正例(§4.1 第 17 行):它的边界是「以上都不是」,
> 但**边界说明不足以让模型学会认它**,给一句原话比给一句否定式更有效。

⇒ **作者论证过它该进 prompt,而渲染函数从来没输出它。** 这是本仓那条
「代码里一句**看起来会生效**的话,其实什么都没做」的形状(家族里已有四个成员)。

**改法**:让 `render_taxonomy_for_prompt()` 在每类的边界说明**下面**渲染它的正例
(形如 `    · 例:「{text}」`,与现行反例行的缩进对齐),**`COUNTER` 没声明的 7 类照样有正例**。
`tests/test_topic_taxonomy.py` 里照
`test_rendered_block_carries_every_counter_pair` 的体例**补一条同款**(断言**整行文本**在块里,
而不是「两个词都出现过」),并确认 `test_rendered_block_has_no_curly_braces` 仍然绿。

⚠️ **必须一并记账的两条**(写进报告与 commit message):
1. **这是一处对已复审代码(`app/topic/taxonomy.py`,Task 1 的交付物)的改动** ——
   改的是渲染**函数**,不是 `POSITIVE` 常量本身(那张常量表与 CP-1 用户签过字的 CSV 一字未动)。
2. **连带影响**:`render_taxonomy_for_prompt()` **也被 Task 4 的生成器用** ⇒
   **下一次重跑 Task 4 会得到不同的问题**。现存那 962 条是在**没有正例**的渲染块下生成的。
   对训练**无影响**(Task 5 用 `{**row, "labels": accepted}` **整列覆盖** `labels`),
   但这条漂移要如实写下来,不要装作没有。

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_topic_labeling.py`:

```python
# ---- 证据串校验(ch10 spec §6.2)----

from app.topic.labeling import validate_evidence


def test_evidence_must_be_a_substring_of_the_question():
    """**这是「字面提到」这个口径的结构性保证。**

    没有它,「字面提到」只是 prompt 里的一句话,模型可以凭语义联想打标签,
    而你从输出上看不出来。有了证据串校验,它变成一个**可自动检验**的条件。
    """
    ok, bad = validate_evidence(
        "买大了想退", ["尺码", "退换货"], {"尺码": "买大了", "退换货": "想退"}
    )
    assert ok == ["尺码", "退换货"]
    assert bad == []


def test_fabricated_evidence_is_rejected():
    """模型编了一个原文里没有的片段 —— 这一条必须被挑出来。"""
    ok, bad = validate_evidence(
        "买大了想退", ["尺码", "退换货"],
        {"尺码": "买大了", "退换货": "退款政策"},   # 原文里没有「退款政策」
    )
    assert ok == ["尺码"]
    assert bad == ["退换货"]


def test_label_without_evidence_is_rejected():
    ok, bad = validate_evidence("买大了想退", ["尺码"], {})
    assert ok == []
    assert bad == ["尺码"]


def test_empty_evidence_string_is_rejected_not_accepted():
    """空串是任何字符串的子串 —— **不特判的话这条会假绿**。

    `"" in "任意文本"` 为 True,所以只写 `if ev in question` 的话,
    模型返回 `"尺码": ""` 会被判为「证据合法」。
    """
    ok, bad = validate_evidence("买大了想退", ["尺码"], {"尺码": "   "})
    assert ok == []
    assert bad == ["尺码"]


def test_evidence_is_matched_after_cleaning():
    """证据比对在**清洗后**的文本上做 —— 与预标时喂进去的文本口径一致。"""
    ok, _ = validate_evidence("买大了 想退", ["尺码"], {"尺码": "买大了"})
    assert ok == ["尺码"]


def test_rejected_labels_are_returned_not_silently_dropped():
    """被拒的标签要**返回出来**,不能悄悄吞掉。

    吞掉的后果:一条问题从「3 个标签」变成「1 个标签」而无人知晓,
    而这会**改变训练分布** —— 多标签样本会系统性变少,正是方案 A 要防的。
    """
    _, bad = validate_evidence("买大了想退", ["尺码", "退换货"], {"尺码": "买大了"})
    assert bad == ["退换货"]
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_labeling.py -k evidence`
Expected: **FAIL** —— `ImportError: cannot import name 'validate_evidence'`

- [ ] **Step 3: 实现 `validate_evidence`**

追加到 `app/topic/labeling.py`:

```python
def validate_evidence(
    question: str, labels: list[str], evidence: dict[str, str]
) -> tuple[list[str], list[str]]:
    """校验每个标签的「证据串」是不是原文的子串。

    返回 `(通过的标签, 被拒的标签)` —— **被拒的也要返回**,不能悄悄吞掉:
    从 3 个标签缩成 1 个会改变训练分布,而那是静默的。

    ⚠️ **空证据串必须显式拒绝**:`"" in "任意文本"` 恒为 True,
    不特判的话 `{"尺码": ""}` 会被判成「证据合法」—— 一个一眼看不见的假绿。
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
```

- [ ] **Step 4: 跑测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_labeling.py`
Expected: **PASS**

- [ ] **Step 5: 写 `scripts/prelabel_topics.py`**

```python
"""大模型预标:照术语表打标签,**每个标签必须附一句原文证据**。

**打网络** —— 不是单测。

断点续跑:`evals/topic/prelabeled.jsonl` 是 append-only,
重跑时跳过已经有结果的 id(照 `build_kb.py`「重跑 = 幂等补齐」的先例)。

用法:
    .venv/Scripts/python.exe scripts/prelabel_topics.py --limit 5
    .venv/Scripts/python.exe scripts/prelabel_topics.py
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.llm import create_extract_model
from app.topic.labeling import validate_evidence
from app.topic.taxonomy import render_taxonomy_for_prompt

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "evals" / "topic" / "corpus.jsonl"
SYNTH = ROOT / "evals" / "topic" / "synthetic.jsonl"
OUT = ROOT / "evals" / "topic" / "prelabeled.jsonl"

# ⚠️ **计划订正 5-B(controller,2026-09-26)**:下面这段与真实现有两处差 ——
#   ① `render_taxonomy_for_prompt()` 现在**也渲染正例**(Step 0 / 订正 D),
#      所以描述它的那句从「含每类的**边界说明与**容易混的反例」
#      **必须改成「含每类的边界说明、正例与容易混的反例」**
#      —— 不改的话,这句话本身就是**假的**(本仓那条「代码里一句看起来成立、
#      其实描述的不是代码」的形状)。
#   ② `run()` 里取模型的那行是 `_bind_json(create_extract_model(settings))`,
#      不是裸的 `create_extract_model(settings)`(见 Step 5 的 `_bind_json`)。
PROMPT = """你在为一款电商客服系统做**多标签主题标注**。

权威类目表(含每类的边界说明与容易混的反例):

{taxonomy}

请判断下面这句话**字面提到了几个诉求**,每个诉求打一个标签。

判据是「**字面提到**」,不是「用户真正想要什么」:
- 「买大了想退」→ 尺码、退换货(字面两个诉求)
- 「退货运费谁承担」→ 退换货、运费
- 「保修期内坏了能退吗」→ 保修维修、退换货
- 「满多少钱包邮」→ 运费(**不是**优惠活动)
- 「我要转人工」→ 其他(这不是主题)

**一个不多,一个不少。**

输出一个 JSON 对象,两个字段:
- labels:标签数组,每个标签名**逐字来自上面的类目表**
- evidence:对象,键是标签名,值是**这句话里支持该标签的那几个字**(必须是原句的**连续片段**,照抄,不要改写)

用户这句话是:
{question}

只输出 JSON,不要别的内容。"""


async def run(limit: int | None) -> None:
    done = _already_done()
    rows = []
    for path in (CORPUS, SYNTH):
        if path.exists():
            rows += [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    todo = [r for r in rows if r["id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"总 {len(rows)} 条,已完成 {len(done)} 条,本次处理 {len(todo)} 条")

    settings = get_settings()
    model = create_extract_model(settings)
    flagged = 0        # 有标签被证据校验拒掉
    parse_failed = 0   # JSON 根本没解出来 —— **与上面那个是两回事,必须分开数**
    zero_label = 0     # 解出来了,但一个标签都没落下(含上面那种,也含模型真判了零诉求)
    with OUT.open("a", encoding="utf-8") as f:
        for row in todo:
            labels, evidence, bad = await _label(model, row["question"])
            if bad:
                parse_failed += 1
            accepted, rejected = validate_evidence(row["question"], labels, evidence)
            if rejected:
                flagged += 1
            if not accepted:
                zero_label += 1
            f.write(json.dumps(
                {**row, "labels": accepted, "rejected_labels": rejected,
                 # ⚠️ 「分得开」的**唯一**保证就是这一列(见 `_label` 的 docstring)
                 "parse_failed": bad,
                 "evidence": {k: v for k, v in evidence.items() if k in accepted}},
                ensure_ascii=False) + "\n")
            f.flush()
    # ⚠️ **三个读数都要打,别只打一个**:
    #    `flagged`      = 有标签被证据校验拒掉 —— **质量读数**,直接进 spec §6.3
    #    `parse_failed` = JSON 没解出来 —— **故障读数,它是 0 才正常**
    #    `zero_label`   = 一个标签都没落下(含上面那种,也含「模型真的判了零诉求」)
    print(f"完成。被证据校验拒掉标签的 {flagged} 条({flagged / max(1, len(todo)):.1%});"
          f"**JSON 解析失败 {parse_failed} 条**;空标签 {zero_label} 条")
    if parse_failed:
        print("⚠️ 解析失败不为 0 ⇒ **先停下看产物,别直接进 Task 6** ——"
              "这些行的标签是空的,而不是「判定了没有主题」。")


async def _label(model, question: str) -> tuple[list[str], dict[str, str], bool]:
    """返回 `(labels, evidence, parse_failed)`。

    ⚠️ **`parse_failed` 必须一路传出去**(计划订正 C,controller 2026-09-26)。
    本节原稿在 `JSONDecodeError` 时 `return [], {}`,调用方写出的行是
    `{"labels": [], "rejected_labels": [], "evidence": {}}` —— 与「模型**真的**返回零标签」
    (合法,例如闲聊)写出来的行**逐字节相同**。而原稿那行注释写的是
    「它与「有标签但被拒」是两回事,**分开记**才能在人审时看出是哪一种」—— **那句是假的**。

    后果不是「少个字段」:① 空标签行进 Task 6 人审、进 Task 7 分层抽样时**看起来是正常行**,
    而它可能是「模型吐了散文、JSON 没解出来」;② `flagged` 那个「质量读数」
    (要进 spec §6.3 的训练标签错误率)会**系统性偏小**。

    ⚠️ **可选但优先**:若 `with_structured_output({...}, method="json_mode")` 在本网关可用,
    优先用它 —— 它能**结构性**消掉这条失败路径(本仓 `app/services/extract.py` 走的就是
    `json_mode`;`function_calling` / `json_schema` 在本网关返回 400)。
    **先查 Context7 或实测再决定**,并在报告里说明用了哪个、为什么。
    无论用哪个,`parse_failed` 这个读数与那一列**都要在**。
    """
    from langchain_core.messages import HumanMessage

    prompt = PROMPT.format(taxonomy=render_taxonomy_for_prompt(), question=question)
    resp = await model.ainvoke([HumanMessage(content=prompt)])
    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], {}, True
    if not isinstance(data, dict):
        return [], {}, True
    return list(data.get("labels") or []), dict(data.get("evidence") or {}), False


def _already_done() -> set[str]:
    if not OUT.exists():
        return set()
    return {
        json.loads(l)["id"]
        for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(run(args.limit or None))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 小样跑一遍,看证据串质量**

Run: `.venv/Scripts/python.exe scripts/prelabel_topics.py --limit 10`

**打开 `evals/topic/prelabeled.jsonl` 肉眼读这 10 条。** 检查:
① `evidence` 里的每个值**真的是原句的连续片段**(不是改写的);② 被拒的标签是**真的错**还是清洗口径问题;
③ **`parse_failed` 列必须全是 `false`** —— 不是 0 就先别往下跑,那是**故障**不是质量(计划订正 C)。
另外**抽 2 条核 `evidence` 是不是逐字子串**:拿产物里的 `evidence` 值去 `clean()` 后的问句里 `in` 一遍,
**自己算一次**,别只看脚本说「被拒 0 条」——「被拒 0」在**标签本来就都对**与
**校验器根本没跑**两种实现下**读数相同**。

> ⚠️ 如果「被拒标签」的比例很高(比如 >30%),**先别往下跑** —— 那通常说明 prompt 里的「照抄、不要改写」没被遵守,改 prompt 重来比标完全部再返工便宜得多。已写出的 10 条可以用 `rm evals/topic/prelabeled.jsonl` 清掉重来(append-only 的代价)。

- [ ] **Step 7: 全量预标**

Run: `.venv/Scripts/python.exe scripts/prelabel_topics.py`

- [ ] **Step 8: Commit**

```bash
git add app/topic/taxonomy.py app/topic/labeling.py scripts/prelabel_topics.py \
        tests/test_topic_labeling.py tests/test_topic_taxonomy.py
git commit -m "ch10-B T5: 大模型预标(证据串校验 + 解析失败可见 + 断点续跑;正例接进渲染块)"
```

---

## Task 6: 人工复核 CSV(导出 → 用户改 → 回收)

> ⚠️ **计划订正 6-B(controller,2026-09-26):本任务**跨**了人工检查点 CP-2,必须拆成两段执行。**
>
> 原稿把「写代码 + 导出 + **等用户改** + 回收」串成一条任务,而 **Step 6 之后是人的时间**,
> 不是 agent 的时间。照原样派,实现者会在 Step 7 无输入可回收(CSV 还是空的)。
>
> - **B6-A(agent 做,做完就停)**:Step 1–6。写 `pick_review_sample` + `export_label_review.py`,
>   跑 `export`,**把「代码 + 那份空着两列的 CSV」一起 commit**(CSV 里 `判定` / `最终标签`
>   两列为空 —— 这正是刻意的:用户改完之后 `git diff` 就是**他改了什么的逐字记录**),
>   然后把 CSV 交给用户,**停**。
> - **B6-B(用户改完之后才做)**:Step 7–8。跑 `import` 回收 → `reviewed.jsonl`,
>   用「总条数 / 判『改』的条数」两个读数补 `dev-notes/ch10.md`,再 commit。
>
> ⚠️ **B6-B 之前不许开工 Task 7** —— 切分要用**用户改过的标签**(spec §6.3)。

**Files:**
- Modify: `app/topic/labeling.py`(加 `pick_review_sample`)
- Modify: `tests/test_topic_labeling.py`
- Create: `scripts/export_label_review.py`

**Interfaces:**
- Consumes: `evals/topic/prelabeled.jsonl`
- Produces:
  - `app.topic.labeling.pick_review_sample(rows, *, per_label: int, seed: int = 20260925) -> list[dict]`
    —— 按类分层各抽 N 条,纯函数、**可注入随机种子**
    ⚠️ **计划订正 6-A(controller,2026-09-26)**:这行原先写的是 `seeds=None`(**拼错**,复数),
    而本任务正文的测试、实现、以及 `export_label_review.py` 的调用点**全都用 `seed`(单数)**。
    以**单数 `seed`** 为准 —— 真在 `app/topic/labeling.py`。
  - `scripts/export_label_review.py export` → `evals/topic/labels/trainval.csv`
  - `scripts/export_label_review.py import` → 把用户改完的 CSV 写回 `evals/topic/reviewed.jsonl`

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_topic_labeling.py`:

```python
from app.topic.labeling import pick_review_sample


def _rows():
    rows = []
    for label in ("尺码", "退换货", "运费"):
        for i in range(20):
            rows.append({"id": f"{label}-{i}", "labels": [label], "question": f"{label}问题{i}"})
    return rows


def test_pick_review_sample_takes_from_every_label():
    """**按类分层**抽 —— 不是随机抽。

    随机抽 15 条很可能一条「评价」都抽不到,而人审看不出那个类的系统性错误。
    """
    sample = pick_review_sample(_rows(), per_label=5, seed=0)
    assert len(sample) == 15
    from collections import Counter
    assert Counter(next(iter(r["labels"])) for r in sample) == {
        "尺码": 5, "退换货": 5, "运费": 5
    }


def test_pick_review_sample_is_reproducible_with_a_seed():
    """同一种子两次结果相同 —— 否则「我审的是哪 85 条」说不清。

    ⚠️ **计划订正 6-C(controller,2026-09-26):这条**自己**是**同义反复**,
    光有它**不足以**说明 seed 被用上了** —— 一个**完全忽略 seed** 的实现
    (根本不 shuffle、或每次都按 id 排序返回)同样满足「同一种子两次相同」。
    ⇒ **必须配下面那条 `test_different_seeds_pick_different_samples`**,
    两条合起来才把「随机且可复现」这件事钉住。
    """
    a = pick_review_sample(_rows(), per_label=5, seed=42)
    b = pick_review_sample(_rows(), per_label=5, seed=42)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_different_seeds_pick_different_samples():
    """**不同种子给出不同样本** —— 这一条才是「seed 真的被用上了」的判据。

    只有上面那条时,把 `rng.shuffle(shuffled)` 整行删掉,**上面那条照样绿**
    (输入 20 条里取 5 条,不洗牌就是固定取前 5 条 —— 确定,但**不是抽样**)。
    本仓把这种叫「断言在它本该禁止的实现下依然通过」。

    ⚠️ 用 `per_label=5` / 每类 20 条:`C(20,5)` 很大,两个种子撞出**同一集合**
    的概率可忽略;万一将来有人把样本调小,这条会**偶发红**,
    那时该改的是**这条测试的规模**,不是把它删掉。
    """
    a = pick_review_sample(_rows(), per_label=5, seed=0)
    b = pick_review_sample(_rows(), per_label=5, seed=1)
    assert [r["id"] for r in a] != [r["id"] for r in b]


def test_pick_review_sample_takes_what_is_available():
    """某类只有 2 条时,拿 2 条而不是报错 —— 但要**如实少拿**。"""
    rows = [{"id": "a", "labels": ["尺码"], "question": "x"}] * 1 + \
           [{"id": f"b{i}", "labels": ["运费"], "question": f"y{i}"} for i in range(10)]
    sample = pick_review_sample(rows, per_label=5, seed=0)
    from collections import Counter
    got = Counter(next(iter(r["labels"])) for r in sample)
    assert got["尺码"] == 1 and got["运费"] == 5


def test_multi_label_rows_count_toward_every_label():
    """多标签行对**每个**它带的标签都算一个样本。

    只按第一个标签计数的话,多标签样本会集中在某一个类的抽样里,
    而另一个类的「5 条」里一条多标签都没有 —— 人审就看不到那类的多标签错误。
    """
    rows = [{"id": "m1", "labels": ["尺码", "退换货"], "question": "买大了想退"}]
    sample = pick_review_sample(rows, per_label=5, seed=0)
    assert [r["id"] for r in sample] == ["m1"]
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_labeling.py -k pick_review`
Expected: **FAIL** —— `ImportError: cannot import name 'pick_review_sample'`

- [ ] **Step 3: 实现 `pick_review_sample`**

追加到 `app/topic/labeling.py`:

```python
import random


def pick_review_sample(
    rows: list[dict], *, per_label: int, seed: int = 20260925
) -> list[dict]:
    """按类分层抽 `per_label` 条,用于人工复核(spec §6.3)。

    **同一条多标签行可能被多个类抽中** —— 用 id 去重后返回,
    所以返回条数**可能少于** `per_label × 类目数`。这是对的:
    它的目的是「每一类都有人看过」,不是「恰好 N 行」。

    种子可注入 ⇒ 同一种子两次结果相同,「我审的是哪几条」可复现。
    """
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        for label in row.get("labels") or []:
            by_label.setdefault(label, []).append(row)

    picked: dict[str, dict] = {}
    for label, bucket in by_label.items():
        # 排序后再打乱:输入顺序不同(比如语料重跑过)不会改变抽出的集合,
        # 只要同一个 label 的桶内容相同。
        shuffled = sorted(bucket, key=lambda r: r["id"])
        rng.shuffle(shuffled)
        for row in shuffled[:per_label]:
            picked[row["id"]] = row
    return [picked[k] for k in sorted(picked)]
```

- [ ] **Step 4: 跑测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_labeling.py`
Expected: **PASS**

- [ ] **Step 5: 写导出/回收脚本**

> ⚠️⚠️ **计划订正 7(controller,2026-09-26)—— 下面这个代码块与盘上实现对不上,四处,照抄会静默劣化** ⚠️⚠️
>
> 四处都是 B6-A 的评审**逐处核实过、且各有测试与变异钉住**的修复。真在 `scripts/export_label_review.py`。
>
> | # | 计划原稿 | 真实现 | 照抄的后果 |
> |---|---|---|---|
> | **A** | `def main() -> None:` 直接 `argparse` | 第一行加 `_pin_stdout_encoding()` | ⚠️ **与 T5 在 `prelabel_topics.py` 上修的是同一个 cp936 坑**:`⚠️`(U+26A0)编不进 GBK,而它**只**出现在「某类一条都没抽到」那条 print 里 ⇒ **产物写好了、命令却以退出码 1 结束**,那条唯一的警告被吞掉 |
> | **B** | `with src.open(encoding="utf-8-sig") as f:` | 加 `newline=""` | 与 `csv` 官方文档不一致;表内的换行/引号解析在特定输入上出错 |
> | **C** | 直接 `csv.DictReader(f)` 开读 | 先核 `REQUIRED_COLUMNS`;**表头缺列或「判定」值认不出 ⇒ 响亮 `SystemExit`,且在写产物之前** | 计划原稿里**表头少了「判定」列时 `.get()` 回落到 `""` ⇒ 恒不判「改」⇒ 错误率静默为 0**,而 §6.3 拿它做「训 / 不训」的决策 |
> | **D** | `[x for x in final.replace(",", "\|").split("\|") if x.strip()]`(**保留了未 strip 的 `x`**) | 逐段 `strip()` | 用户填 `尺码, 退换货` ⇒ `" 退换货"` 不是合法类目 ⇒ **响亮报错,但报的是「不合法类目」而不是「你多打了个空格」**,逼人去改他那份 CSV |
>
> ⚠️ 另外**一条口径**已由 controller 裁定、plan 里没有:错误率读数
> (`changed`)必须数「**生效标签 ≠ 预标标签**」的行,**不是**数「判定」列 ——
> 否则「用户改了标签但忘了填判定」会读成 0%(**静默**,而 §6.3 拿它决定训不训)。
> 详见 `task-6-report.md` 的 §订正轮 1 与账本里那条设计裁定。

Create `scripts/export_label_review.py`:

```python
"""把待审样本导成 CSV(给人改),再把改完的收回来。

**为什么要走 CSV**:120 条 100% 过 + 85 条抽审,在编辑器里排着改最快;
而且改了什么、改了多少,**进 git,可追溯** —— 页面里点一遍是留不下痕迹的。

用法:
    .venv/Scripts/python.exe scripts/export_label_review.py export
    # ← 用户改 evals/topic/labels/trainval.csv 之后
    .venv/Scripts/python.exe scripts/export_label_review.py import
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.topic.labeling import pick_review_sample
from app.topic.taxonomy import LABELS

ROOT = Path(__file__).resolve().parents[1]
PRELABELED = ROOT / "evals" / "topic" / "prelabeled.jsonl"
LABELS_DIR = ROOT / "evals" / "topic" / "labels"
REVIEWED = ROOT / "evals" / "topic" / "reviewed.jsonl"

#: 抽审量:每类 5 条 × 17 类 ≈ 85 条(spec §6.3)。
PER_LABEL = 5

HEADER = ["id", "问题", "预标标签", "判定(ok/改)", "最终标签", "备注"]


def _load() -> list[dict]:
    return [
        json.loads(l)
        for l in PRELABELED.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def export() -> None:
    rows = _load()
    sample = pick_review_sample(rows, per_label=PER_LABEL)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out = LABELS_DIR / "trainval.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for r in sample:
            w.writerow([r["id"], r["question"], "|".join(r["labels"]), "", "", ""])
    print(f"导出 {len(sample)} 条 → {out}")
    # ⚠️ 打印**按类的覆盖**,让人一眼看出哪类抽得少(某类样本不足时会少拿)。
    from collections import Counter
    c = Counter(lb for r in sample for lb in r["labels"])
    missing = [lb for lb in LABELS if c[lb] == 0]
    print(f"  每类条数:{dict(c)}")
    if missing:
        print(f"  ⚠️ 这些类**一条都没抽到**(样本不足):{missing} —— 它们没有人工复核覆盖")


def do_import() -> None:
    """回收:用户填了「最终标签」的按用户的,没填的按预标的。

    「判定」列只影响报表(改了多少条),不影响入库的标签 —— 标签一律以
    「最终标签」列为准,空了才回落到预标。
    """
    src = LABELS_DIR / "trainval.csv"
    changed = 0
    out_rows = []
    with src.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            final = (row.get("最终标签") or "").strip()
            labels = [x for x in final.replace(",", "|").split("|") if x.strip()] or \
                     [x for x in (row["预标标签"] or "").split("|") if x.strip()]
            if row.get("判定(ok/改)", "").strip() == "改":
                changed += 1
            invalid = [lb for lb in labels if lb not in LABELS]
            if invalid:
                # ⚠️ 响亮地报,不静默丢弃 —— 手打的标签名容易有空格/错字。
                raise SystemExit(f"{row['id']} 的标签里有不合法类目:{invalid}")
            out_rows.append({"id": row["id"], "question": row["问题"],
                             "labels": labels, "reviewed": True})
    REVIEWED.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n",
        encoding="utf-8",
    )
    # ⚠️ 订正 8(controller,2026-09-26):**这一句在盘上已经不这么写了。**
    # 真实现是「其中**标签与预标不同**的 {changed} 条」 —— 因为 `changed` 的**语义**
    # 已由订正 7 改成「生效标签 ≠ 预标标签」(那才是 spec §6.3 的错误率分子),
    # 而「用户判『改』的」是**判定列**的口径,**两者会在合法输入上分歧**
    # (用户改了标签但没写判定 ⇒ 按标签数是 N、按判定数是 0)。
    # 措辞不改的话,谁把这一行抄进 spec §6.3 就会写错名 —— 本仓「名字与语义不符」那条形状。
    print(f"回收 {len(out_rows)} 条 → {REVIEWED};其中用户判「改」的 {changed} 条")
    # ⚠️ 「judged 改」的比例就是**预标错误率的观测值**(只覆盖被抽到的那些),
    #    它要进报告,与 F1 并排(spec §6.3)。


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "import"])
    args = ap.parse_args()
    export() if args.action == "export" else do_import()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 导出,交给用户(🚧 CP-2,阻塞任务 7 及以后)**

Run: `.venv/Scripts/python.exe scripts/export_label_review.py export`

**停下来,把 `evals/topic/labels/trainval.csv` 交给用户。** 明确说清:
① 他只需填 **「判定(ok/改)」** 与 **「最终标签」** 两列;② 没填的按预标算;
③ 这一份是**测错误率**的(不是逐条改),所以**「标签与预标不同」的条数**本身就是产物
(⚠️ 订正 8:**不是**「用户判『改』的条数」—— 后者是判定列口径,见 Step 5 块首的注记)。

> ⚠️ 测试集那份 CSV(120 条,100% 过)由任务 7 在切分之后导出 —— **切分要用到用户改过的标签,所以本任务必须先回收**。

- [ ] **Step 7: 回收用户的修改**

Run: `.venv/Scripts/python.exe scripts/export_label_review.py import`

**把打印的两个数(总条数、**「标签与预标不同」的条数**)抄进 `dev-notes/ch10.md`。**
⚠️ 订正 8:第二个数的**名字**必须照实现打印的那个用 —— 它**不是**「用户判『改』的条数」。
若打印里同时给了「警告:有 N 行标签变了但没写判定」,**那个数也要抄**(它说明用户漏填了判定列)。

- [ ] **Step 8: Commit**

```bash
# ⚠️ 订正 7:本任务的 commit **分两次**(Task 6 跨 CP-2,见节头)。
# B6-A(紧接 Step 6 之后):
git add app/topic/labeling.py scripts/export_label_review.py \
        tests/test_topic_labeling.py evals/topic/labels/trainval.csv
git commit -m "ch10-B T6: 人工抽审 CSV(按类分层各 5 条,测错误率而非逐条改)"
# ⚠️ **原稿在这里还列了 `evals/topic/reviewed.jsonl`** —— 那个文件在 B6-A 阶段
#    **还不该存在**(用户还没改完 CSV)。照抄会 `git add` 一个不存在的路径而失败。
# B6-B(用户改完 CSV、跑完 Step 7 的 `import` 之后):
git add evals/topic/reviewed.jsonl dev-notes/ch10.md
git commit -m "ch10-B T6-B: 回收人工复核(错误率读数 + reviewed.jsonl)"
```

---

*(计划续:任务 7–16 见下)*

## Task 7: 分层抽样 + 冻结测试集

> ⚠️ **本任务多承接一条(T5 复审发现,controller 2026-09-26 裁定交给这里)**:
> **必须加一条 train-vs-test 的重复/近重复检查。**
>
> 起因:T5 的 Step 0 把 `POSITIVE` 的 34 句正例接进了渲染块,而该渲染块**同时被 T4 的生成器用**
> ⇒ **(a) 已发生**:冻结测试集(`source=evalmd`,299 行)里 **8 行含渲染块里的例句原句** ——
> 6 行含「运费怎么算」(`COUNTER` 那句,T1 起就在 prompt 里)+ **2 行含「什么材质」**
> (`POSITIVE['商品信息']`,**本次新带入**,占 2/299 = 0.7%);
> **(b) 未来风险**:T4 的禁词表只覆盖**类目名 + 边界说明原词**、**不覆盖这些例句**
> ⇒ **下次重跑 T4 可能把「买大了」「175 穿什么码」整句抄进问句** ⇒ 训练侧抄了测试侧的句子。
> ⇒ 本任务的切分**不能只按标签分层**,还要**按清洗后的文本做一次 train-vs-test 重复检查**
> (完全相同 ⇒ 必须移出训练侧;近重复 ⇒ 报出条数并入账)。
> **并把这条风险的量写进 spec §8「对外报数的唯一依据」那一段的账里** ——
> 报测试集分数时要知道其中 8 行与 prompt 里的例句字面重叠。
>
> ⚠️ **9-C:上面这条要求目前只是一段话,没有对应的 Step 与测试。** ⇒ 补 **Step 4b**(见下),
> 它必须是**能红的**那种实现(拿一份人为造出「训练侧含测试侧原句」的输入喂进去,必须报出来)。

> ⚠️⚠️ **计划订正 9(controller,2026-09-26)—— 本任务有四处必须先订正,两处会让它跑不起来或静默不接上** ⚠️⚠️
>
> **9-A(结构,与 Task 6 同款):本任务**也**跨了人工检查点,必须拆成两段。**
> **Step 6 之后是人的时间** —— 用户要把 120 条 `test.csv` **100% 过一遍**。
> 照原样派,实现者会在 `import-test` 那一步无输入可回收(`test.csv` 还是空的)。
> - **B7-A(agent 做,做完就停)**:Step 1–5 + **`export-test`**;
>   commit「代码 + 切分产物(`train.jsonl` / `val.jsonl` / `topic_test.jsonl`)+ **空着两列的 `test.csv`**」,
>   把 CSV 交给用户,**停**。
> - **B7-B(用户改完之后才做)**:`import-test` → 回写 `topic_test.jsonl` → 补 dev-notes → commit。
> - ⚠️ CSV 先以**空着 `判定` / `最终标签` 两列**的状态入库是**刻意的**(与 Task 6 同款):
>   用户改完后的 `git diff` 就是**他改了什么的逐字记录**。
>
> **9-B(会让它静默不接上)**:Step 4 的代码块写 `f"{name}.jsonl"` ⇒ 产出 `test.jsonl`,
> 而本任务其余**四处**(Interfaces / Step 6 两处 / Step 7 的 `git add`)全按 **`topic_test.jsonl`** 读写。
> 照原稿抄 ⇒ **用户那 120 条的复核对指标零影响,而报告照常打印**。
> ⇒ **已在 Step 4 的代码块里就地订正,以 `topic_test.jsonl` 为准。**
>
> **9-D(Step 6 欠规定 = 让实现者自己发明)**:原稿只说「加两个动作」,**没给代码、没定产物名**,
> 而 Step 7 的 `git add` 里又冒出一个**第三个**名字 `reviewed_test.jsonl`。
> ⇒ **规定如下**:
> - `import-test` **必须复用** trainval 那条路**同一套校验**(列错位 / id 白名单 / 重复 id /
>   表头缺列 / 判定认不出 / 空产物 / 重复类目 / 非 UTF-8 的**人话**报错)。
>   **做法:把 `do_import` 里那段校验抽成一个共用函数,两处调用** ——
>   本仓那条「不变量要放在唯一写口上,不要靠每个调用方自觉」。
>   **不许复制一份**:两份各自维护的校验就是本仓记过的漂移形状。
> - **产物名定死**:`import-test` **写回 `evals/topic/topic_test.jsonl`**(标签列换成复核后的),
>   **不产出 `reviewed_test.jsonl`** ⇒ Step 7 的 `git add` 里那个名字**去掉**。
>   `topic_test.jsonl` 是验收 ① 对外报数的**唯一**依据。
> - ⚠️ **回写这一步不能省**(原稿自己也强调了):用户改完的标签若只落在 `test.csv`,
>   而评测读的是回写前的 `topic_test.jsonl`,**用户那 120 条的工作对指标零影响**。
>
> **9-E(数据,原稿没预见 —— 已实测)**:`prelabeled.jsonl` 里有 **2 行是零标签**
> (`labels == []`,实测:有 `labels[0]` 的行 **1422** / 1424)。它的**来源不同**:
> - `r-0049`「你是」是**模型真判了零诉求**(复审指出:≤4 字的 31 行里 30 行判 `其他`,只有它判 `[]`);
> - `s-0423`「首重多少,超了咋算?」是**证据校验机械拒到空**(`rejected_labels == ['运费']`),
>   而它的真实主题**几乎肯定是运费**。
> ⇒ **把这行以空标签喂进训练,是在教模型「这句话没有主题」** —— 而那是个**处理产物,不是判断**。
> **规定**:`split()` 必须**显式处理并打印**这两行(列 id / 问句 / `rejected_labels` / 落在哪一侧),
> 且**把「`labels == []` 且 `rejected_labels` 非空」的行排除出 train/val**
> (排除了就**不许**再进测试集 —— 它靶子不可信)。**这一条要有测试,且变异能红。**

**Files:**
- Modify: `app/topic/labeling.py`(加 `stratified_split`)
- Modify: `tests/test_topic_labeling.py`
- Modify: `scripts/prepare_topic_data.py`(`split` 子命令)

**Interfaces:**
- Consumes: `evals/topic/prelabeled.jsonl` + `evals/topic/reviewed.jsonl`(人改过的标签)
- Produces:
  - `app.topic.labeling.stratified_split(rows, *, test_real: int, test_synth: int, ratios=(0.8,0.1,0.1), seed) -> dict[str, list[dict]]`,键为 `"train"|"val"|"test"`
  - `evals/topic/topic_test.jsonl`(**冻结进 git**)+ `evals/topic/train.jsonl` + `val.jsonl`

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_topic_labeling.py`:

```python
from app.topic.labeling import stratified_split


def _mixed(n_real=100, n_synth=200):
    rows = []
    for i in range(n_real):
        rows.append({"id": f"r{i}", "provenance": "real", "labels": ["尺码"],
                     "question": f"真问题{i}"})
    for i in range(n_synth):
        rows.append({"id": f"s{i}", "provenance": "synthetic", "labels": ["运费"],
                     "question": f"合成问题{i}"})
    return rows


def test_test_set_has_the_prescribed_composition():
    """⚠️ **测试集构成是抽样的硬条件,不是抽完再看结果**(spec §5.3)。

    「真实 80 + 合成 40」是 §8.4 那三层报告能打出来的前提:
    少了它,「只看真实」那一列就没有足够的样本。
    """
    parts = stratified_split(_mixed(), test_real=80, test_synth=40, seed=0)
    test = parts["test"]
    from collections import Counter
    c = Counter(r["provenance"] for r in test)
    assert len(test) == 120
    assert c["real"] == 80 and c["synthetic"] == 40


def test_splits_do_not_overlap():
    """三份**互不相交** —— 有交集就是数据泄漏,而 F1 会因此虚高。"""
    parts = stratified_split(_mixed(), test_real=80, test_synth=40, seed=0)
    ids = [r["id"] for part in parts.values() for r in part]
    assert len(ids) == len(set(ids))


def test_all_rows_are_used():
    parts = stratified_split(_mixed(), test_real=80, test_synth=40, seed=0)
    assert sum(len(p) for p in parts.values()) == 300


def test_split_is_reproducible():
    a = stratified_split(_mixed(), test_real=80, test_synth=40, seed=7)
    b = stratified_split(_mixed(), test_real=80, test_synth=40, seed=7)
    assert [r["id"] for r in a["test"]] == [r["id"] for r in b["test"]]


def test_head_labels_are_oversampled_into_the_test_set():
    """头四类在测试集里每类 ≥15 条(spec §5.3)—— 否则那条 F1 是噪声。

    这条测试用的语料**故意让头四类样本充足**,好让「配额」这件事可观测;
    真实语料不够时,`stratified_split` 应当**如实少给**并在返回里标注,
    而不是硬凑(硬凑会重复使用同一样本 ⇒ 数据泄漏)。
    """
    rows = []
    for label in ("退换货", "物流", "尺码", "发票"):
        for i in range(60):
            rows.append({"id": f"{label}{i}", "provenance": "real",
                         "labels": [label], "question": f"{label}问题{i}"})
    for i in range(60):
        rows.append({"id": f"x{i}", "provenance": "synthetic",
                     "labels": ["评价"], "question": f"评价问题{i}"})
    parts = stratified_split(rows, test_real=80, test_synth=40, seed=0)
    from collections import Counter
    c = Counter(lb for r in parts["test"] for lb in r["labels"])
    for label in ("退换货", "物流", "尺码", "发票"):
        assert c[label] >= 8, f"头四类里的 {label} 在测试集只有 {c[label]} 条"
```

> ⚠️ 最后一条断言用 **`>= 8` 而不是 `>= 15`** —— 因为 80 条真实样本分给 4 个类,理论上限是每类 20,但分层抽样的实现细节会让实际值浮动。**写死在 15 会让测试对实现细节过敏**。
>
> ⚠️⚠️ **计划订正 10(controller,2026-09-26):原稿这一句的后半是假的,已删** ⚠️⚠️
> 原稿写的是「**真正的 ≥15 由验收脚本在真实数据上核对,不在单测里假装**」—— **那句必然核对不过**,
> 照它写验收 ① 就是一条恒红的断言。**实测(用冻结的 `topic_test.jsonl` 复算)**:
>
> | 层 | 标签槽 | 够 15 的类 |
> |---|---|---|
> | 全体 120 | 169 | **3 / 17**(退换货 18、运费 15、保修维修 15) |
> | **只看真实 80** | 112 | **1 / 17**(只有退换货) |
> | 只看合成 40 | 57 | **0 / 17** |
>
> **算术上就不可能**:`17 类 × 15 = 255` 个标签槽,而 120 行的测试集只有 **169** 个槽
> ⇒ **哪怕切分做到完全均匀,120 行也装不下「17 类各 ≥15」**。
>
> ⇒ **验收 ① 不许出现任何形式的 ≥15 断言,也不许退成 `≥ 8`**
> (复审实测:`≥ 8` 对「有没有分层」**零判别力** —— 把分层整个删掉、或改成「一条都不分层」,
> `test_head_labels_...` 与 `test_test_set_has_the_preset_composition` **都照样绿**)。
> 验收该断的是:**构成**(120 = real 80 + synth 40)/ **三份不相交** / **合计 = 输入 − 不可信靶子**;
> support 与 `†` 清单**只打出来入账**。
>
> ⚠️ **而 `†` 会铺满几乎整张表**(全体 14/17、只看真实 16/17)—— 这是**规格与语料现实对不上**,
> 不是实现缺陷。**必须在章级文档里如实写**(见 Task 10 / Task 16 的注记),不许在报告里
> 悄悄把阈值从 15 调低 —— 那是**拿判据去迁就数据**,与 T4 里拒绝过的
> 「抬高 `FORMS["multi"]` 去凑一个四舍五入造出来的 30%」是同一族。
>
> ⚠️ **头四类的两半都没实现**:`take()` 是**按主标签近均匀轮转**(每类 ~5 条,实测主标签计数 5/5/4/5),
> **既不超额给头四类、也不按比例为其余类分配**。所以 **`物流` 那 12 条不是池子不够**
> (真实池 55 条),**是机制没加权**;而 `尺码`(真实池 **4**)与 `发票`(真实池 **14**)
> 那两处才是**真的池子封顶**。**别把两者混成一句「结构性」**。

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_labeling.py -k split`
Expected: **FAIL** —— `ImportError: cannot import name 'stratified_split'`

- [ ] **Step 3: 实现 `stratified_split`**

追加到 `app/topic/labeling.py`:

```python
from collections import defaultdict


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
    """
    rng = random.Random(seed)
    by_prov: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_prov[r["provenance"]].append(r)

    def take(pool: list[dict], n: int) -> tuple[list[dict], list[dict]]:
        """按主标签分层取 n 条,返回 (取出, 剩下)。"""
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
        picked_ids = {r["id"] for r in picked}
        rest = [r for r in pool if r["id"] not in picked_ids]
        return picked, rest

    test = take(by_prov.get("real", []), test_real)[0] + \
           take(by_prov.get("synthetic", []), test_synth)[0]
    test_ids = {r["id"] for r in test}
    rest = [r for r in rows if r["id"] not in test_ids]

    rest_sorted = sorted(rest, key=lambda r: r["id"])
    rng.shuffle(rest_sorted)
    n_train = round(len(rest_sorted) * ratios[0] / (ratios[0] + ratios[1]))
    return {"train": rest_sorted[:n_train], "val": rest_sorted[n_train:], "test": test}
```

- [ ] **Step 4: `split` 子命令 + 冻结测试集**

在 `scripts/prepare_topic_data.py` 的 `main` 里接上 `split`,实现:

```python
def split() -> None:
    """切分并**冻结测试集**。

    ⚠️ 标签以**人改过的**为准:`reviewed.jsonl` 里的覆盖 `prelabeled.jsonl`。
    不这么做的话,用户那 85 条的复核成果不会进训练集 —— 而它会**静默地**
    只影响报表,不影响模型。
    """
    pre = {json.loads(l)["id"]: json.loads(l)
           for l in (ROOT / "evals/topic/prelabeled.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    rev_path = ROOT / "evals/topic/reviewed.jsonl"
    if rev_path.exists():
        for l in rev_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                pre[r["id"]] = {**pre[r["id"]], "labels": r["labels"], "human_reviewed": True}
    rows = list(pre.values())

    parts = stratified_split(rows, test_real=80, test_synth=40)
    # ⚠️⚠️ 计划订正 9-B(controller,2026-09-26)—— **原稿这里是 `f"{name}.jsonl"`,会写出
    # `test.jsonl`,而本任务其余四处(Interfaces / Step 6 的两句 / Step 7 的 git add)
    # 全都按 `topic_test.jsonl` 读写。**照原稿抄的后果不是报错,是**静默不接上**:
    # `split` 写出 `test.jsonl`,而 `import-test` 与评测脚本读 `topic_test.jsonl`
    # ⇒ 用户那 120 条的复核**对指标零影响**,而报告照常打印。**以 `topic_test.jsonl` 为准。**
    TEST_NAME = "topic_test.jsonl"
    for name, items in parts.items():
        path = ROOT / "evals" / "topic" / (TEST_NAME if name == "test" else f"{name}.jsonl")
        path.write_text(
            "\n".join(json.dumps({**r, "split": name}, ensure_ascii=False) for r in items) + "\n",
            encoding="utf-8",
        )
        print(f"{name}: {len(items)} 条 → {path}")

    # 三层报告的构成核对(spec §8.4)—— 数不对说明硬条件没满足,响亮地报。
    from collections import Counter
    c = Counter(r["provenance"] for r in parts["test"])
    assert c["real"] == 80 and c["synthetic"] == 40, f"测试集构成不对:{dict(c)}"
    from app.topic.taxonomy import HEAD_LABELS
    per = Counter(lb for r in parts["test"] for lb in r["labels"])
    thin = [lb for lb in HEAD_LABELS if per[lb] < 15]
    if thin:
        print(f"  ⚠️ 头四类里这些在测试集不到 15 条:{thin} —— 它们的 F1 **不成结论**,"
              f"报告里要标 †(spec §8.3)")
```

- [ ] **Step 4b: train-vs-test 重复/近重复检查(计划订正 9-C —— 原稿只有一段话,没有 Step)**

加一个**纯函数**到 `app/topic/labeling.py`:

```python
def train_test_overlap(train: list[dict], test: list[dict],
                       near: float = 0.8) -> dict[str, object]:
    """报出训练侧与测试侧的重复/近重复。**纯函数、零依赖。**

    ⚠️ **为什么要这一步**(T5 复审发现,见 Task 7 节首那段):
    `POSITIVE` 的 34 句正例进了 **T4 生成器**的 prompt,而 T4 的禁词表
    **不覆盖这些例句** ⇒ 重跑 T4 可能把「买大了」「175 穿什么码」整句抄进问句。
    已发生的一半:**冻结测试集里 8 行与渲染块例句字面重叠**(6 行「运费怎么算」自 T1 起、
    2 行「什么材质」由 T5 的 Step 0 新带入)。

    **判据分两档,处置不同**:
    - **完全相同**(清洗后文本相等)⇒ **必须移出训练侧**(`split()` 负责移),它在测试侧留着;
    - **近重复**(字符二元组 Jaccard ≥ `near`)⇒ **只报不删** —— 删了就是拿判据改数据,
      而这条判据本身是启发式。报出条数与最像的几对,人来看。

    返回 `{"exact_train_ids": [...], "near_pairs": [(train_id, test_id, 相似度), ...]}`。
    """
```

`split()` 里接上:① 先按 `exact_train_ids` 把那些行**从 train/val 里摘掉**;
② 把两个数(`exact` 条数、`near` 对数)与最像的几对**打印出来**;
③ **不许** `assert` 近重复为 0(它是预期的,不是错误)。

**测试(必须能红)**:造一份「训练侧含一条与测试侧逐字相同的问句 + 一条相似度高的」输入,
断 `exact_train_ids` 抓到了那一条、`near_pairs` 非空。
**变异**:把 `clean(q)` 换成原文比较 / 把 `near` 阈值抬到 1.0 ⇒ 对应的断言必须红。

- [ ] **Step 5: 跑切分,拿真实读数**

Run: `.venv/Scripts/python.exe scripts/prepare_topic_data.py split`

**把三份的条数、测试集的真实/合成构成、以及那个 `†` 清单抄进 `dev-notes/ch10.md`。**

- [ ] **Step 6: 导出测试集复核 CSV 并交给用户(🚧 与 CP-2 同一批)**

⚠️ **订正 9-D:产物名已定死,别自己发明**(原稿只说「加两个动作」,而 Step 7 的 `git add`
里还冒出一个**第三个**名字 `reviewed_test.jsonl` —— 那是原稿的内部矛盾)。

在 `scripts/export_label_review.py` 里加两个动作:

- `export-test`:读 **`evals/topic/topic_test.jsonl`**(⚠️ **不是 `test.jsonl`**,见订正 9-B),
  把**全部 120 条**导成 `evals/topic/labels/test.csv`(表头与 `trainval.csv` **同一个**);
- `import-test`:回收用户改完的 CSV,**把复核后的标签写回 `evals/topic/topic_test.jsonl`**。
  ⚠️ **不产出 `reviewed_test.jsonl`** —— Step 7 的 `git add` 里那个名字**去掉**。

⚠️ **`import-test` 必须复用 trainval 那条路的同一套校验,不许复制一份**:
列错位(`row[None]`)/ id 白名单 / 重复 id / 表头缺列 / 判定值认不出 /
空产物 / 单元格内重复类目 / 非 UTF-8 的**人话**报错 / 「题面以语料为准」。
**做法**:把 `do_import` 里那段校验抽成一个**共用函数**,两处调用 ——
本仓那条「不变量要放在唯一写口上,不要靠每个调用方自觉」。
两份各自维护的校验就是本仓记过的漂移形状(T4 的 `reject_reason` 是同一条纪律的正例)。
⚠️ **`test.csv` 与 `trainval.csv` 的「预标」基准不同**:前者的基准是 `topic_test.jsonl` 的
`labels`,后者是 `prelabeled.jsonl` 的 —— 抽出共用函数时**基准要作参数**,别写死。

⚠️ **回写这一步不能省。** 用户改完的标签如果只落在 `test.csv` 或一个旁路文件里,而评测脚本读的仍是回写前的 `topic_test.jsonl`,那么**用户那 120 条的工作对指标零影响** —— 而报告会照常打印、看起来完全正常。这是一处「看起来做完了、其实没接上」。

**冻结的语义**:`topic_test.jsonl` 一旦被 `import-test` 写过,此后**任何任务都不许再改它**(任务 8 的增强、任务 9 的切分都不碰测试集)。它是验收 ① 对外报数的唯一依据。

Run:
```bash
.venv/Scripts/python.exe scripts/export_label_review.py export-test
# ← 停下来,把 evals/topic/labels/test.csv 交给用户,100% 过一遍
.venv/Scripts/python.exe scripts/export_label_review.py import-test
```

> ⚠️ **V11(controller,2026-09-26)—— 用户拍板「跳过人工复核」,所以上面这三步实际只跑了第一步。**
> **`import-test` 没有被执行,`test.csv` 也没有被人改过。**
>
> **这不是「B7-B 忘了做」**,是一条**明确选定的路**:
> - 跑了 `import-test` 会给 120 行**全部**打上 `human_reviewed: True`,而其中 **108 行没有人看过**
>   (只有 12 行是从 CP-2 那 84 条流过来的)—— 那正是本仓最忌讳的「看起来做完了、其实没做」。
> - **冻结语义不受影响**:`topic_test.jsonl` 由 `split()` 写出后**没有任何任务再改它**,
>   它仍是验收 ① 对外报数的唯一依据。
> - **代价必须记账**(见 Task 16 的注记 ④):报告里那个 F1 **是在预标标签上测的**,
>   不许被读成「在人工标注的黄金集上测出来的」。
> - **回退路径**:若日后用户想补做,`test.csv` 与 `import-test` 都还在,
>   补回来只改 `topic_test.jsonl` 一处。

**把「用户改了几条 / 共几条」抄进 `dev-notes/ch10.md`。** 这个比例与任务 6 的抽审比例合起来,构成 spec §6.3 那条「训练集标签错误率」的读数。

- [ ] **Step 7: Commit**

```bash
# ⚠️ 订正 9-A / 9-D:本任务的 commit **分两次**(它跨检查点)。
# B7-A(Step 1–5 + export-test 之后):
git add app/topic/labeling.py scripts/prepare_topic_data.py \
        scripts/export_label_review.py tests/test_topic_labeling.py \
        evals/topic/train.jsonl evals/topic/val.jsonl evals/topic/topic_test.jsonl \
        evals/topic/labels/test.csv
git commit -m "ch10-B T7-A: 分层抽样 80/10/10 + 测试集冻结(等人工 100% 复核)"
# ⚠️ **原稿在这里多列了 `evals/topic/reviewed_test.jsonl`** —— 那个文件**不该存在**
#    (订正 9-D:`import-test` 只回写 `topic_test.jsonl`),照抄会 add 一个不存在的路径。
# B7-B(用户改完 test.csv、跑完 import-test 之后):
git add evals/topic/topic_test.jsonl dev-notes/ch10.md
git commit -m "ch10-B T7-B: 回写人工复核后的测试集标签(冻结)"
```

---

## Task 8: 数据增强(**只扩训练集**)

> ⚠️⚠️ **计划订正 11(controller,2026-09-26)—— 五处,一处是脚枪(11-A,已就地改)、
> 一处是「文档里的承诺没有装置守」(11-B)** ⚠️⚠️
>
> **11-B(最要紧的一条)**:`augment()` 的 docstring 写着「验证集与测试集一条都不许动」,
> 以及「扩了测试集…是个**静默**的破坏」—— **但这些只是散文,没有任何东西守着它**。
> ⇒ **补一个 Step 与一条测试**:跑 `augment` 的**前后**,`val.jsonl` 与 `topic_test.jsonl`
> 必须**逐字节相同**(`sha256` 对比),并且在 `augment()` 里**显式 `assert`** 它只读 `train.jsonl`。
> **要能红**:把 `train_path` 指向 `val.jsonl` ⇒ 那条断言必须红。
> (本仓已编目过:一句**看起来成立**的注释不是守卫。)
>
> **11-C**:`inject_typo(new_text, i)` 用**循环下标**当种子 ⇒ 可复现性依赖**输入行序**,
> `train.jsonl` 一旦重排,错别字就变。⇒ **改成从行 id 派生一个稳定种子**
> (例如 `int(hashlib.sha256(r["id"].encode()).hexdigest()[:8], 16)`)。
> ⚠️ 仍要 `random.Random(seed)`(本仓硬约束:**不许用内置 `hash()`**,它对 str 每进程随机化)。
>
> **11-D**:`test_inject_typo_never_empties_the_text` **零判别力** ——
> `inject_typo` 的实现是 `text.replace(src, dst, 1)`,只要输入非空就**结构上不可能**返回空,
> 而「原样返回 `text`」的错实现**照样绿**。
> ⇒ 要么**加强**它(例如断言「有可替换词时**必定**变了」),要么在注释里**标成不承重**。
> 别留着一条看起来在守什么、其实什么都没守的断言。
>
> **11-E**:全量约 **75 分钟**(1158 条 × T5 实测 3.9 秒/条)。
>
> ⚠️ 另**不要**让实现者写 `dev-notes`(见 11-F 的流程约定,写 `task-8-report.md`)。

**Files:**
- Modify: `app/topic/labeling.py`(加 `label_drift`、`inject_typo`)
- Modify: `tests/test_topic_labeling.py`
- Modify: `scripts/prepare_topic_data.py`(`augment` 子命令)

**Interfaces:**
- Consumes: `evals/topic/train.jsonl`
- Produces:
  - `app.topic.labeling.label_drift(before: list[str], after: list[str]) -> bool` —— 标签集合变了就 `True`
  - `app.topic.labeling.inject_typo(text: str, seed: int) -> str` —— 确定性注入一个错别字
  - `evals/topic/train_augmented.jsonl`

- [ ] **Step 1: 写失败的测试**

```python
from app.topic.labeling import inject_typo, label_drift


def test_label_drift_detects_changed_count():
    assert label_drift(["尺码", "退换货"], ["退换货"])          # 少了一个
    assert label_drift(["退换货"], ["尺码", "退换货"])          # 多了一个
    assert not label_drift(["尺码", "退换货"], ["退换货", "尺码"])  # **顺序不算变化**


def test_inject_typo_is_deterministic():
    """同种子同输入 → 同输出。增强产物要**可复现**,否则「这份语料是哪来的」说不清。"""
    assert inject_typo("买大了想退", 1) == inject_typo("买大了想退", 1)


def test_inject_typo_actually_changes_something():
    """不是恒等函数 —— 否则「注入了错别字」这句话是空的。"""
    changed = [inject_typo("我要退货", s) for s in range(20)]
    assert any(c != "我要退货" for c in changed)


def test_inject_typo_never_empties_the_text():
    """不许注成一个空串 —— 空样本会进训练集而看不出来。"""
    for s in range(50):
        assert inject_typo("退货", s).strip()
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_labeling.py -k "drift or typo"`
Expected: **FAIL**

- [ ] **Step 3: 实现**

```python
def label_drift(before: list[str], after: list[str]) -> bool:
    """增强后标签集合变了没有。**顺序不算变化**(标签是集合语义)。

    这是增强阶段最要紧的一条自检:把「买大了想退」改成「不喜欢这个想退」,
    标签就从 `尺码+退换货` 变成 `退换货` —— 而它看起来只是一次正常增强,
    **没有任何东西会报错**,训练分布却悄悄偏了。
    """
    return set(before) != set(after)


#: 同音/形近替换对。**保守**:只收在电商语境下几乎不会改变语义的字。
_TYPO_MAP = (
    ("退货", "退或"), ("尺码", "尺马"), ("运费", "云费"), ("发票", "发飘"),
    ("快递", "快第"), ("订单", "定单"), ("颜色", "艳色"), ("保修", "报修"),
)


def inject_typo(text: str, seed: int) -> str:
    """确定性地注入一个错别字(同音/形近)。**只用于训练集**。

    为什么这么做:部署时用户就是会打错字,而清洗阶段**刻意不修错别字**
    (spec §5.1)。如果不注入,训练语料比真实输入干净 ⇒ 真机掉分,
    而**测试集也是干净的,测不出来**。

    找不到可替换的字时返回原文(不算失败)—— 一句话里没有那些词很正常。
    """
    rng = random.Random(seed)
    candidates = [(a, b) for a, b in _TYPO_MAP if a in text]
    if not candidates:
        return text
    src, dst = rng.choice(candidates)
    return text.replace(src, dst, 1)
```

- [ ] **Step 4: 写 `augment` 子命令**

在 `prepare_topic_data.py` 里实现 `augment`:对 `train.jsonl` 的每条,**同义词替换 + 句式微调**(大模型)+ **注入错别字**(确定性),然后**逐条重验标签**:

```python
async def augment() -> None:
    """增强**只扩训练集**。验证集与测试集一条都不许动。

    扩了测试集,§8 那套指标就不再有意义 —— 而那是个**静默**的破坏:
    指标会变好看,没人会去查测试集是不是被动过。
    """
    train_path = ROOT / "evals" / "topic" / "train.jsonl"
    rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    settings = get_settings()
    model = create_extract_model(settings)
    kept, drifted = 0, 0
    with (ROOT / "evals" / "topic" / "train_augmented.jsonl").open("w", encoding="utf-8") as f:
        # 原件照写 —— 增强是**追加**,不是替换。
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for i, r in enumerate(rows):
            new_text, new_labels = await _rewrite(model, r)   # 大模型改写 + 回标
            if label_drift(r["labels"], new_labels):
                # ⚠️ **静默的标签漂移** —— 整条丢弃,不修。
                #    修的话要判断「是它漂了还是原标错了」,而那需要人。
                drifted += 1
                continue
            f.write(json.dumps(
                {**r, "question": inject_typo(new_text, i), "labels": new_labels,
                 "augmented": True}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"增强追加 {kept} 条;因标签漂移丢弃 {drifted} 条")
    # ⚠️ `drifted` 是**语料质量的读数**:它高说明预标不稳(或 prompt 里
    #    「保持诉求个数与类别不变」没被遵守)。打出来,别吞掉。
```

`_rewrite` **逐条**调模型,prompt 必须带原标签 + 边界说明,并明确要求「保持诉求个数与类别不变」:

```python
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


async def _rewrite(model, row: dict) -> tuple[str, list[str]]:
    """返回 (改写后的句子, 它自己报的标签)。

    ⚠️ 调用方会用 `label_drift` 比对原标签与它报的标签 —— **不一致就整条丢弃**。
    不在这里悄悄「修正」成原标签:那样会把「模型觉得该改标签」这个信号抹掉,
    而那个信号正是我们要观测的东西(它的比例就是 §8 报告的语料质量读数)。
    """
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
        return row["question"], row["labels"]      # 解析失败:原样返回 ⇒ 不构成漂移
    return data.get("question") or row["question"], list(data.get("labels") or row["labels"])
```

> ⚠️ 解析失败时**原样返回**而不是返回空标签 —— 返回空标签会被 `label_drift` 判成漂移而丢弃一条本来没问题的样本,把「网络抖动」记成「标签漂移」。两个读数混在一起之后就分不清了。

- [ ] **Step 5: 小样试跑,看漂移率**

> ⚠️⚠️ **计划订正 11-A(controller,2026-09-26)—— 原稿这里是个脚枪** ⚠️⚠️
> 原稿写「**先把 `train.jsonl` 截成 20 条试跑**」—— **`train.jsonl` 是已入库的冻结产物**,
> 是 Task 9 的训练输入。照它做就是**就地改掉训练集**,而 `git status` 会显示 ` M`,
> 谁在收尾时一 `git add -A` 就把一份 20 行的训练集提交了。
> ⇒ **给 `augment` 加一个 `--limit N` 参数**,小样跑 `--limit 20`。
> **任何情况下都不许改 `train.jsonl` / `val.jsonl` / `topic_test.jsonl` 的**内容。

Run:
```bash
.venv/Scripts/python.exe scripts/prepare_topic_data.py augment --limit 20
```
看 `drifted` 比例。若 >20%,**先改 `_rewrite` 的 prompt 再全量跑** —— 全量跑完再发现就得重来。
(⚠️ **20% 这个阈值是拍的,没有标定** —— 它只是个「先停下来看看」的提示,不是判据。)

> ⚠️ **订正 11-E(耗时)**:全量是 **1158 条 × 逐条调模型**。按 T5 实测的 **3.9 秒/条**,
> 大约是 **75 分钟**量级。**派活时按这个估。**

- [ ] **Step 6: 跑纯函数测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_labeling.py`
Expected: **PASS**

- [ ] **Step 7: Commit**

```bash
git add app/topic/labeling.py scripts/prepare_topic_data.py \
        tests/test_topic_labeling.py evals/topic/train_augmented.jsonl
git commit -m "ch10-B T8: 数据增强(只扩训练集 + 标签漂移自检 + 注入错别字)"
```

---

## Task 9: 全参微调 `scripts/train_topic_clf.py`

> ⚠️⚠️ **计划订正 12(controller,2026-09-26)—— 四条,全是 T8 交接过来的;
> 12-A 会让「这份权重是哪份数据训的」这条凭据**静默给一个错的答案**** ⚠️⚠️
>
> **12-A(`data_fingerprint()` 的行尾歧义 —— 我实测)**:计划只写「对 `train_augmented.jsonl` 的
> 内容算 sha256 前 16 位」,而**两种实现给出不同的值**:
>
> | 算法 | sha256 前 16 位 |
> |---|---|
> | `read_bytes()`(该文件**盘上是 CRLF**:2315 个 CRLF、0 个裸 LF) | `2ca5aba59a0afc2f` |
> | `read_text(encoding="utf-8")`(通用换行 ⇒ LF) | `74bd7d056f68b68c` |
>
> ⇒ 照 `read_bytes()` 写,指纹**随 checkout 的行尾策略变化** —— 换台 `autocrlf=false`/Linux 的机器
> 算出来是另一个值,后人要么以为语料被换了、要么放弃这条凭据。**它一坏就是「静默给一个错的溯源答案」。**
> ⇒ **改法**:对 **LF 归一化后**的文本算(`p.read_text(encoding="utf-8").encode("utf-8")`),
> 并在 docstring 里写明「指纹是对 LF 归一化的内容算的」。
> **配一条测试**:同一份内容分别以 CRLF 与 LF 写进 `tmp_path`,两条指纹**必须相同**。
> ⚠️ **不要顺手加全仓 `.gitattributes`** —— 那是全仓范围的决定(ch10 已裁定过),这里不需要它。
> ⚠️ **并且**:T8 复审实测,**增强产物不可复现**(改写那一半在 `temperature=0` 下仍不确定,
> 连跑 3 次得 3 个样)⇒ **`train_augmented.jsonl` 的 sha256 是「一次运行」的指纹,不是「这份语料」的**。
> ⇒ `train_meta.json` 里**除了指纹还要记行数与生成日期**,否则那条凭据会被读得比它能代表的强。
>
> **12-B(增强行与原件共用 `id`)**:`train_augmented.jsonl` 的 1157 条增强行**与它们的原件 id 相同**。
> 计划里 `encode_rows` 是**逐行消费**(已核),所以**今天安全** —— 但**没有任何东西守着它**。
> 危险动作只有一句:`{r["id"]: r for r in rows}`(去重 / join 预测 / 按 id 做错误分析)
> ⇒ 2315 行**静默塌成 1158**,增强全没了而指标照常打。
> ⇒ **改法**:`encode_rows` 的 docstring 里写死一句「**不许按 id 去重/join**」+ 一条断言能红。
> ⚠️ 另提醒:`train_test_overlap` 的 `exact_train_ids` / `near_pairs` 是**按 id** 报的,
> 把它跑在 `train_augmented.jsonl` 上时,报出来的 id **指哪一行不明确**。
>
> **12-C(陈旧的列)**:增强行的 `evidence` / `rejected_labels` / `parse_failed` 是**原句的快照**
> (复审复算 **1157/1157** 逐字节等于其原件那一行),而题面已经换过。
> ⇒ **在 Task 9 计划里写一行说明**。⚠️ **不许就地清空或改名** —— 那要重跑 89 分钟,收益只是好看。
> 危险动作是「拿 `evidence` 去核对增强行的题面」,那会得到一个**自信的错答案**。
>
> **12-D**:产物里的 `parse_failed` 列**不是这一轮的读数** —— 增强行经 `{**r}` 继承原值,
> 故 1157 行**全是 `False`**,**包括这一轮真解析失败的那条**(`r-0088`)。
> ⇒ 「产物里 `parse_failed` 全 False」**不能**读成「这一轮零解析失败」。今天无害(Task 9/10 不读它)。
>
> ⚠️ **另两条留给 Task 16 的章级文档同步**(不在本任务):
> **M3** —— `_TYPO_MAP` 的 8 对**全是同音替换**,而 spec 两处写「(同音/形近/**多字漏字**)」;
> **裁定改 spec 措辞、不改表**(改表会让已提交的产物与代码不再对应,而重跑 89 分钟且改写那半本就不确定)。
> **M6** —— dev-notes 里两条 `.superpowers` 引用解析不开(T5/T18 时期就在),按 ch09 判据落进那张例外表。
>
> ⚠️⚠️ **计划订正 13(controller,2026-09-27)—— T9 实现者与复审**各自实测**推翻了四处;
> 其中两处推翻了 spec 正文** ⚠️⚠️
>
> **13-A `spec §2.3` 那句「静默」对 17 列 `long` 不成立 —— 它是当场抛。**
> 实测(实现者与复审**各自**在本地小 `BertConfig` 上跑过,报文**一字不差**):
> ```python
> # 小 BertConfig + problem_type=None
> #   labels 是 (2, 17) 的 long  ⇒ ValueError: Expected input batch_size (2) to match target batch_size (34)
> #   而且抛之后 model.config.problem_type 已经是 single_label_classification
> #   ⇒ 那个「猜」确实在第一个 batch 之前/当刻就锁死
> #   labels 是 1-D long        ⇒ 【静默】loss 逐位等于手算 cross_entropy,一路训下去
> #   labels 是 float 多热      ⇒ multi_label_classification,loss 逐位等于 BCEWithLogitsLoss
> ```
> ⇒ **结论不变但更锋利**:**「会抛」不等于「有防线」** —— 第一跳唯一拦得住的**只有防线②(labels 必须 float)**。
> 已写成 4 条**真跑 forward** 的离线用例(本地小 `BertConfig`,不联网)。
>
> **13-B `warmup_ratio` 在 transformers 5.17.0 被整个删掉**(不是改名,**是删参数** ——
> `grep -rl warmup_ratio .venv/.../transformers/` **0 个文件**;
> `inspect.signature(TrainingArguments.__init__)` 113 个参数里**有** `warmup_steps`、**没有** `warmup_ratio`)。
> ⇒ **spec §2.2 那张「v5 三处差异」表漏了最贵的一条**(另两条只是改名),§7.3 还写着 `warmup_ratio = 0.1`。
> 已换算 `warmup_steps = int(0.1 × 73 × 15) = 109`,并把**换算依据落进 `train_meta.json`**。
>
> **13-C 本节的 `TopicDataset` 理由错了。** `transformers/trainer.py:987-991` 是
> `if is_datasets_available() and isinstance(dataset, datasets.Dataset): … else: _get_collator_with_removed_columns`
> ⇒ **只有 `datasets.Dataset` 才走 `column_names`**。实测(`list[dict]`、`TopicDataset`、**裸对象** × 开/关
> `remove_unused_columns`)**5 种组合全部正常训完**。⇒ 这一层**保留但不再写成防线**(docstring 已如实改)。
>
> **13-D `spec §7.4` 说验证集 120 条,实为 145。**(`wc -l val.jsonl = 145`,205 个标签槽。)
>
> ⚠️ 另:**本节 Step 1 的 `label_count_match` 用例期望值写错了**(应 **2/3** 不是 `1.0`),
> 实现者已在自己的测试文件里改对 —— **订正 13 把它同步进计划**,免得后来人照抄得到一条必红的测试。

**Files:**
- Create: `scripts/train_topic_clf.py`
- Create: `app/topic/model.py`(可单测的那一半:数据集编码 + 产物读写)
- Create: `tests/test_topic_model.py`

**Interfaces:**
- Consumes: `evals/topic/train_augmented.jsonl`、`val.jsonl`、`app.topic.taxonomy.LABELS`
- Produces:
  - `app.topic.model.encode_rows(rows, tokenizer, *, max_length) -> list[dict]` —— 输出 `input_ids` / `attention_mask` / **`labels`(float32 多热向量)**
  - `app.topic.model.save_artifacts(out_dir, *, tokenizer, max_length, threshold, labels)` —— 写 `labels.json` / `inference_config.json` / `train_meta.json`
  - `models/topic-clf/`(gitignore)

- [ ] **Step 1: 写失败的测试**

Create `tests/test_topic_model.py`:

```python
"""模型侧可单测的那一半:编码、产物读写。

⚠️ **单测全程不联网** —— 所以这里用一个**假的 tokenizer**,不加载 RoBERTa。
"""

import json

import pytest

from app.topic.model import encode_rows, load_artifacts, save_artifacts
from app.topic.taxonomy import LABELS


class FakeTokenizer:
    """只做「按字符切」的假分词器 —— 测的是我们的编码逻辑,不是 HF 的。"""

    def __call__(self, texts, *, truncation, max_length, padding, return_tensors):
        import torch

        ids = [[min(ord(c), 100) for c in t[:max_length]] for t in texts]
        width = max(len(x) for x in ids)
        return {
            "input_ids": torch.tensor([x + [0] * (width - len(x)) for x in ids]),
            "attention_mask": torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids]),
        }


def test_labels_are_float_not_int():
    """⚠️ **这是本章最贵的一条断言。**

    `transformers/loss/loss_utils.py:94-118` 的原文:`problem_type` 为 None 时,
    它看 `labels.dtype` —— 整数(dtype 是 long/int)会被判成
    **`single_label_classification`**(softmax + 交叉熵),**float 才是多标签**
    (BCE)。而且这个猜测**在第一个 batch 上定下来就锁死**。

    后果:模型被 softmax 推着**每句只报一个类**,而 **loss 照常下降、
    训练看起来完全正常** —— 但验收 ③ 从原理上过不了。
    """
    rows = [{"question": "买大了想退", "labels": ["尺码", "退换货"]}]
    out = encode_rows(rows, FakeTokenizer(), max_length=16)
    assert out[0]["labels"].dtype.is_floating_point, "标签必须是 float —— 否则会被判成单标签交叉熵"


def test_encoding_is_a_multi_hot_over_LABELS_order():
    """多热向量的**位置必须由 `LABELS` 决定** —— 它是标签 id 的权威顺序。"""
    rows = [{"question": "买大了想退", "labels": ["尺码", "退换货"]}]
    vec = encode_rows(rows, FakeTokenizer(), max_length=16)[0]["labels"]
    assert vec[LABELS.index("尺码")] == 1.0
    assert vec[LABELS.index("退换货")] == 1.0
    assert float(vec.sum()) == 2.0


def test_unknown_label_raises_loudly():
    """不认识的类目**必须抛**,不许静默丢弃。

    静默丢弃的表现是「这条样本少了一个标签」,而它会让多标签样本
    系统性变少 —— 正是方案 A 花大力气造出来的那 30%。
    """
    with pytest.raises(KeyError, match="不存在的类目"):
        encode_rows([{"question": "x", "labels": ["尺码", "不存在的类目"]}],
                    FakeTokenizer(), max_length=16)


def test_topic_dataset_is_indexable_and_declares_columns():
    """`Trainer` 会读 `column_names`(`trainer.py:1174`),而普通的 list 没有它。

    这条测试钉的是「包了一层」这件事本身:去掉 `TopicDataset` 直接传 list,
    训练会在 `Trainer` 内部 `AttributeError` —— 报错指向 transformers,
    读起来像它的 bug。
    """
    from app.topic.model import TopicDataset

    encoded = encode_rows([{"question": "买大了想退", "labels": ["尺码"]}],
                          FakeTokenizer(), max_length=16)
    ds = TopicDataset(encoded)
    assert len(ds) == 1
    assert ds[0]["labels"].dtype.is_floating_point
    assert ds.column_names == ["input_ids", "attention_mask", "labels"]


def test_save_and_load_artifacts_roundtrip(tmp_path):
    """`labels.json` 是**推理侧标签顺序的唯一来源**,必须能原样读回。"""
    save_artifacts(tmp_path, tokenizer=None, max_length=64, threshold=0.5, labels=LABELS,
                   meta={"seed": 1})
    arts = load_artifacts(tmp_path)
    assert tuple(arts["labels"]) == LABELS
    assert arts["max_length"] == 64
    assert arts["threshold"] == 0.5


def test_saved_labels_match_taxonomy(tmp_path):
    """⚠️ 产物里的顺序必须与 `taxonomy.LABELS` **逐位相同**。

    两侧不一致会产出一张**完全错的分布图,而每个组件都工作正常** ——
    模型有输出、scores 在 0–1、写库成功、页面画得出来。没有任何东西会报错。
    """
    save_artifacts(tmp_path, tokenizer=None, max_length=64, threshold=0.5, labels=LABELS, meta={})
    on_disk = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    assert on_disk == list(LABELS)
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_model.py`
Expected: **FAIL** —— `ModuleNotFoundError: No module named 'app.topic.model'`

- [ ] **Step 3: 实现 `app/topic/model.py`**

```python
"""模型侧可单测的那一半 —— **不 import torch 以外的重东西,不联网**。

训练脚本(`scripts/train_topic_clf.py`)负责循环与早停;这里负责
「把数据编成张量」与「把产物写对」,因为那两件事各自有一个**静默失效**
要防(标签 dtype、标签顺序)。
"""

import json
from pathlib import Path

from app.topic.taxonomy import LABELS


def encode_rows(rows: list[dict], tokenizer, *, max_length: int) -> list[dict]:
    """把语料编成 `input_ids` / `attention_mask` / `labels`。

    ⚠️ **`labels` 必须是 float32 的多热向量。** `transformers` 在
    `problem_type` 为 None 时**看 dtype 猜任务**:整数会被判成单标签分类,
    而那个猜测在第一个 batch 上锁死。显式传 `problem_type` 是第二道防线
    (训练脚本里),这里是第一道。
    """
    import torch

    texts = [r["question"] for r in rows]
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True,
                    return_tensors="pt")
    out = []
    for i, row in enumerate(rows):
        vec = torch.zeros(len(LABELS), dtype=torch.float32)
        for label in row["labels"]:
            try:
                vec[LABELS.index(label)] = 1.0
            except ValueError as exc:
                raise KeyError(f"不存在的类目:{label}") from exc
        out.append({
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels": vec,
        })
    return out


class TopicDataset:
    """把 `encode_rows` 的产物包成 `Trainer` 能吃的数据集。

    ⚠️ **不能直接把 `list[dict]` 交给 `Trainer`。** 已核安装版源码
    (`transformers/trainer.py:1165-1174`):默认 `remove_unused_columns=True` 时,
    `_remove_unused_columns` 会读 `dataset.column_names` —— 普通 list 或裸
    `torch.utils.data.Dataset` 都没有这个属性,直接 `AttributeError`,而报错
    指向 `Trainer` 内部,读起来像 transformers 的 bug。

    两条路:包成 `datasets.Dataset`(要多一层格式转换),或者**包成最小的
    torch Dataset + 关掉 `remove_unused_columns`**(见训练脚本)。这里选后者:
    样本量小、字段就是我们自己编的那三个,少一层格式转换就少一处出错的地方。
    """

    #: `Trainer` 会读它 —— 有了它 `remove_unused_columns` 那条路才走得到,
    #: 但因为我们在 `TrainingArguments` 里关掉了那个开关,它只是让报错更友好。
    column_names = ["input_ids", "attention_mask", "labels"]

    def __init__(self, encoded: list[dict]):
        self.rows = encoded

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def save_artifacts(out_dir, *, tokenizer, max_length: int, threshold: float,
                   labels, meta: dict) -> None:
    """写产物。**`labels.json` 是推理侧标签顺序的唯一来源。**

    服务读它、**不自己写一份** —— 两侧顺序不一致会产出一张完全错的分布图,
    而每个组件都工作正常。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.json").write_text(
        json.dumps(list(labels), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "inference_config.json").write_text(
        json.dumps({"max_length": max_length, "threshold": threshold},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if tokenizer is not None:
        tokenizer.save_pretrained(out)


def load_artifacts(model_dir) -> dict:
    """读产物。服务与评测脚本都走这里 —— **不要各自 open() 一遍**。"""
    d = Path(model_dir)
    return {
        "labels": json.loads((d / "labels.json").read_text(encoding="utf-8")),
        **json.loads((d / "inference_config.json").read_text(encoding="utf-8")),
        "meta": json.loads((d / "train_meta.json").read_text(encoding="utf-8")),
    }
```

- [ ] **Step 4: 跑测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_model.py`
Expected: **PASS**

- [ ] **Step 5: 写训练脚本**

Create `scripts/train_topic_clf.py`。**关键部分照抄下面的形状**(其余是标准 Trainer 代码):

```python
BASE = "hfl/chinese-roberta-wwm-ext"

def main() -> None:
    ...
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    # ⚠️ `problem_type` **显式传**,不靠 transformers 从 labels.dtype 猜 ——
    #    见 app/topic/model.py 与 spec §2.3。
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=len(LABELS), problem_type="multi_label_classification",
    )

    args = TrainingArguments(
        output_dir=str(out_dir / "_ckpt"),
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=15,
        # ⚠️ v5 叫 `eval_strategy`,**不是** v4 的 `evaluation_strategy`。
        eval_strategy="epoch",
        # ⚠️ `load_best_model_at_end=True` 时它必须**等于** eval_strategy,否则直接抛错。
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # ⚠️ 盯 **micro-F1**,不盯 macro:验证集每类仅 ~7 条,
        #    拿 macro 早停等于让噪声决定什么时候停(spec §7.4)。
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_steps=20,
        seed=SEED,
        report_to=[],
        # ⚠️ **必须关掉**:默认 True 时 `Trainer._remove_unused_columns` 会读
        #    `dataset.column_names` 并按 signature 删列(`trainer.py:1165-1174`)。
        #    我们的数据集是自己包的最小 torch Dataset(字段就是模型 forward 要的
        #    那三个),关掉它既避开那条 AttributeError 路径,也避免它按名字误删
        #    `labels`(那会让训练**静默**变成无监督)。
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        # ⚠️ `Trainer` **不能**直接收 `list[dict]` —— 见 `app/topic/model.py`
        #    的 `TopicDataset` docstring。
        train_dataset=TopicDataset(train_encoded),
        eval_dataset=TopicDataset(val_encoded),
        # ⚠️ v5 叫 `processing_class`,**不是** v4 的 `tokenizer`。
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3,
                                         early_stopping_threshold=0.001)],
    )
    trainer.train()

    save_artifacts(out_dir, tokenizer=tokenizer, max_length=MAX_LEN,
                   threshold=0.5, labels=LABELS,
                   meta={
                       "base": BASE, "seed": SEED,
                       # ⚠️ **数据指纹**:没有它,「这个权重是哪份数据训的」永远说不清,
                       #    而语料是会被重造的。
                       "data_fingerprint": data_fingerprint(),
                       "val_metrics": trainer.evaluate(),
                   })
```

`compute_metrics` 返回 `{"micro_f1", "macro_f1", "subset_accuracy", "label_count_match"}`,**与任务 10 的评测指标同源**(同一组函数,不许各写一份)。

`data_fingerprint()` 对 `train_augmented.jsonl` 的内容算 sha256 前 16 位。

- [ ] **Step 6: 跑训练**

Run: `.venv/Scripts/python.exe scripts/train_topic_clf.py`

Expected: 训练跑完,`models/topic-clf/` 里有 `labels.json` / `inference_config.json` / `train_meta.json` / 权重。

**把 `train_meta.json` 里的 `val_metrics` 与 `data_fingerprint` 抄进 `dev-notes/ch10.md`。**

> ⚠️ 若 CUDA OOM,先把 `per_device_train_batch_size` 降到 16 —— **不要**去动 `max_length`(它由实测的句长定的)。

- [ ] **Step 7: 确认产物目录已被忽略(已核:`models/` 早就在 `.gitignore` 里,不用新增规则)**

Run: `grep -n "^models/" .gitignore`
Expected: 打印出 `models/`(它在 ch04 起就在了)。

⚠️ **不要为此新增一条 `models/topic-clf/`** —— `models/` 已经覆盖它,再加一条是噪音,
而下一个人会以为「这条路径特殊、需要单独排除」。**真正要做的只有 Step 8 的验证。**

- [ ] **Step 8: 确认真的没被 git 跟踪**

Run: `git status --short | grep -c "models/topic-clf" ; git check-ignore -v models/topic-clf/model.safetensors`
Expected: 计数为 `0`,且 `check-ignore` 打印出刚加的那条规则

- [ ] **Step 9: Commit**

```bash
git add scripts/train_topic_clf.py app/topic/model.py tests/test_topic_model.py .gitignore
git commit -m "ch10-B T9: RoBERTa-wwm-ext 全参微调(显式 problem_type + micro-F1 早停)"
```

---

## Task 10: 评测 `scripts/eval_topic_clf.py`(每类指标 + 两张矩阵 + 三层报告)

> ⚠️ **计划订正 10 带下来的一条(controller,2026-09-26)**:**`†` 必须按层算,而且会对全部 17 类算,不是只对头四类。**
>
> spec §8.3 的判据是「**每类** support < 15 ⇒ 标 `†`」,而 §8.4 要求分三层报。
> **实测的 support(从冻结的 `topic_test.jsonl` 复算)**:
>
> | 层 | 标签槽 | 够 15 的类 |
> |---|---|---|
> | 全体 120 | 169 | **3 / 17** |
> | **只看真实 80** | 112 | **1 / 17** |
> | 只看合成 40 | 57 | **0 / 17** |
>
> ⇒ **这份报告的表会几乎铺满 `†`**,而那**不是缺陷**:`17 × 15 = 255 > 169`,
> 120 行的测试集**算术上装不下**「17 类各 ≥15」。**如实标,不许把阈值从 15 调低** ——
> 那是拿判据迁就数据。
> ⚠️ 报告里**必须**出现一句人话解释这件事,否则读的人会把它读成「模型不行」。
>
> ⚠️⚠️ **计划订正 13(controller,2026-09-27)—— 本任务再有六处;前两处照抄就出错** ⚠️⚠️
>
> **13-E(照抄会覆盖已有文件)**:上面 Files 写的是 **Create `app/topic/metrics.py` 与
> `tests/test_topic_metrics.py`** —— ⚠️ **这两个文件 B9 已经建出来了**
> (`per_class_prf` / `subset_accuracy` / `label_count_match` / `metrics_from_logits` 都在里面)。
> ⇒ **改成 Modify**。本任务只**追加** `mislabelled_flow`(B9 刻意留给这里的那一个)。
> ⇒ 连带 Step 2「Expected: **FAIL**」的**红因只有一条**:
> `ImportError: cannot import name 'mislabelled_flow'`(复审实测过这个报文)。
> **不是**「整个文件不存在」—— 别把它读成全红。
>
> **13-F(一条必红的测试)**:Step 1 里
> `assert label_count_match(Y_TRUE, Y_PRED) == 1.0` —— ⚠️ **按给定数据答案是 `2/3`**
> (三行:1==1 ✓、**1 != 2** ✗、1==1 ✓)⇒ 照抄必红。
> B9 的实现者**已经独立发现并改对了自己的那份**,只是**计划文本没跟上**。⇒ **同步改成 `pytest.approx(2/3)`。**
>
> **13-G(Step 5.6 那条会把 CP-2 读数读大)**:原稿要报告印「训练集标签错误率(来自任务 6 的
> `judged 改` 比例)与 F1 并排,并写一句『训练标签的错误率是本章 F1 的已知上界』」。
> ⚠️ **CP-2 的实际读数是 `0 / 84 = 0.0%`,而它是一次「看图通过」、没有逐条核对痕迹**
> (样本 70% 是合成;按更严的口径 **13/17 类零真实覆盖**)。
> ⇒ 把 `0%` 印在 F1 旁边会让人读成「F1 被一个零错误的标签集兜着」——**那是这个数撑不起的结论**。
> **改法**:两个数**都给**,但**必须带上那三条限定**(见 `dev-notes/ch10.md` 阶段 7);
> 或者只印读数不印那句「已知上界」。**二选一,不许只印数。**
>
> **13-H(B9 复审留给本任务的一条,原稿没有)**:报告要**在冻结测试集上把 `t=0.5` 与 `t=0.3`
> 两行都算出来**。理由:测试集**没有参与任何选择**(切分、早停、阈值扫描都在 val / train 上),
> 所以这一行**不是数据窥探**,是「`0.3` 到底真不真好」的**无偏读数**。
> ⚠️ 同时**如实写一句**「`0.3` 在 val 上高 **2.47** 点,但 **val 同时是选择集**,
> 在同一批行上挑出来的增益没有判别力」。
> ⚠️ **不要改 `THRESHOLD`**(保持 0.5),**更不要重训** —— 阈值是**纯后处理**,
> 重训会换掉权重并让 `train_meta.json` 那份被复核过的四指标**全部作废**,而**没有任何东西会报错**。
>
> **13-I(测试集标签的来源必须印在报告里)**:`topic_test.jsonl` 的标签**是预标产物**,
> **120 行里只有 12 行带 `human_reviewed`**(从 CP-2 那 84 条流过来的)。
> ⇒ 报告里那个 F1 **必须带这个限定**,不许被读成「在人工标注的黄金集上测出来的」。
>
> **13-J(`macro-F1` 进三层表时的口径)**:若把 `macro_f1` 打进「全体 / 只看真实 / 只看合成」三列,
> **必须写明「support = 0 的类贡献 0 ⇒ 三列的 macro 不可互比」** ——
> 否则那三个数会被读成「模型在合成子集上更差」。B9 复审已把这条写进 `metrics.py` 的 docstring,
> 报告照抄一句即可。



**Files:**
- Modify: `app/topic/metrics.py`(⚠️ **订正 13-E:不是 Create** —— B9 已建,
  本任务只追加 `mislabelled_flow`)
- Modify: `tests/test_topic_metrics.py`(⚠️ **同样不是 Create**)
- Create: `scripts/eval_topic_clf.py`

**Interfaces:**
- Consumes: `evals/topic/topic_test.jsonl`(冻结)、`models/topic-clf/`
- Produces:
  - `app.topic.metrics.per_class_prf(y_true, y_pred, labels) -> list[dict]`(每行 `label/支持数/精确率/召回率/F1`)
  - `app.topic.metrics.mislabelled_flow(y_true, y_pred, labels) -> dict[tuple[str,str], int]` —— **误判流向矩阵**
  - `app.topic.metrics.subset_accuracy(y_true, y_pred) -> float`
  - `app.topic.metrics.label_count_match(y_true, y_pred) -> float`
  - `evals/topic/report.md` / `report.json` / `matrix_confusion.csv` / `matrix_flow.csv`

- [ ] **Step 1: 写失败的测试**

Create `tests/test_topic_metrics.py`:

```python
"""评测指标 —— 纯函数,所以每一处口径都可以钉死。"""

import pytest

from app.topic.metrics import (
    label_count_match,
    mislabelled_flow,
    per_class_prf,
    subset_accuracy,
)

Y_TRUE = [["退换货"], ["退换货", "尺码"], ["物流"]]
Y_PRED = [["退换货"], ["退换货"], ["物流"]]


def test_per_class_prf_reports_support():
    rows = {r["label"]: r for r in per_class_prf(Y_TRUE, Y_PRED, ["退换货", "尺码", "物流"])}
    assert rows["退换货"]["support"] == 2      # 两条真实里有它
    assert rows["退换货"]["tp"] == 2
    assert rows["尺码"]["support"] == 1
    assert rows["尺码"]["tp"] == 0             # 模型漏了
    assert rows["尺码"]["fn"] == 1
    assert rows["尺码"]["precision"] == 0.0    # 分母 0 时按 0 处理,不是 NaN


def test_precision_denominator_zero_is_zero_not_nan():
    """没预测过任何一条的类,精确率是 0 而不是 NaN。

    NaN 会**污染整张表**(求和/均值全变 NaN),而报告会看起来「有数」。
    """
    rows = {r["label"]: r for r in per_class_prf([["评价"]], [["评价"]], ["评价", "账号"])}
    assert rows["账号"]["precision"] == 0.0
    assert rows["账号"]["precision"] == rows["账号"]["precision"]  # 不是 NaN


def test_subset_accuracy_is_exact_set_match():
    """整条完全一致 —— 多标签最严的指标,也是「一个不多一个不少」的最严读法。"""
    assert subset_accuracy(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)
    # 顺序不同但集合相同 ⇒ 算对
    assert subset_accuracy([["尺码", "退换货"]], [["退换货", "尺码"]]) == 1.0


def test_label_count_match_is_about_the_count_only():
    """**个数**对就算对(标签内容可以错)—— 这是需求原话「一个不多一个不少」
    的字面读法,与 `subset_accuracy` 是两个不同的问题,不要合并。"""
    # ⚠️ 订正 13-F:原稿这里写 `== 1.0`,**按给定数据答案是 2/3**
    #    (三行:1==1 ✓、1 != 2 ✗、1==1 ✓)⇒ 照抄必红。B9 的实现者已独立发现并改对了自己的那份。
    assert label_count_match(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)
    assert label_count_match([["尺码", "退换货"]], [["运费", "物流"]]) == 1.0


def test_mislabelled_flow_counts_true_to_pred_pairs():
    """**误判流向矩阵**:行=真实标签,列=预测标签,格=「本该是 i 却被判成 j」。

    ⚠️ 它**不是**经典混淆矩阵 —— 多标签没有唯一的预测类。名字刻意不叫混淆矩阵:
    本仓吃过名字与语义不符的亏(`agent_steps` 读作「步数」,实际是轮次序号)。
    """
    flow = mislabelled_flow([["退换货"]], [["运费"]], ["退换货", "运费"])
    assert flow[("退换货", "运费")] == 1
    assert flow[("退换货", "退换货")] == 0     # 判对的**不**进这张矩阵


def test_flow_ignores_pairs_where_the_prediction_is_also_true():
    """真实 `[退换货, 运费]` 预测 `[运费]`:这**不是**误判流向,是漏召回。

    算进去的话矩阵会把「漏了一个」记成「把退换货认成了运费」——
    那会把人引向错误的修法(改边界 vs 提召回)。
    """
    flow = mislabelled_flow([["退换货", "运费"]], [["运费"]], ["退换货", "运费"])
    assert flow.get(("退换货", "运费"), 0) == 0
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_metrics.py`
Expected: **FAIL**

- [ ] **Step 3: 实现 `app/topic/metrics.py`**

按上面的接口实现。要点:
- 所有除法都**显式处理分母为 0**,返回 `0.0` 而不是 `NaN`(见测试的注释);
- `mislabelled_flow` 只记 `true_i` 且 `i ∉ pred` 且 `j ∈ pred` **且 `j ∉ true`** 的配对 —— 最后那个条件就是上面第二条测试守的性质;
- **不许用 sklearn 的 `confusion_matrix`**:它是单标签语义,喂多标签会广播成一个巨大的错误矩阵。

- [ ] **Step 4: 跑测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_metrics.py`
Expected: **PASS**

- [ ] **Step 5: 写评测脚本,产出三层报告**

`scripts/eval_topic_clf.py` 的 `--report` 落 `evals/topic/report.md`,必须包含:

1. **每类 P/R/F1 + support 表** —— **`support < 15` 的行标 `†`**,表头写一行「† 样本数不足,该行 F1 不成结论」(spec §8.3)。
2. **micro-F1 / macro-F1 两个都给**,并写明差别(micro 被大类主导、macro 被小类主导)。
3. **整条完全一致率** + **标签个数完全一致率**。
4. **两张矩阵**:`matrix_confusion.csv`(每类 2×2)与 `matrix_flow.csv`(17×17 误判流向)。
5. **三个口径并排**(spec §8.4):

   | | 全体 120 | 只看真实 80 | 只看合成 40 |

   **「只看真实 80」那一列才是这个模型在真机上的预期表现** —— 这句话要**印在报告里**。
6. **训练集标签错误率**(来自任务 6 的 `judged 改` 比例)与 **F1 并排**印出,并写一句「训练标签的错误率是本章 F1 的已知上界」(spec §6.3)。
7. 评测脚本**只读冻结的 `topic_test.jsonl` + checkpoint 路径,不参与任何随机** —— 同一份权重跑两次,报告逐字节相同。

- [ ] **Step 6: 跑评测**

Run: `.venv/Scripts/python.exe scripts/eval_topic_clf.py --report evals/topic/report.md`

**把 micro-F1(三个口径各一个)、以及 `†` 的类目清单抄进 `dev-notes/ch10.md`。**

- [ ] **Step 7: 用真实数据核对头四类的测试集样本数**

Run: `.venv/Scripts/python.exe -c "import json,collections; rows=[json.loads(l) for l in open('evals/topic/topic_test.jsonl',encoding='utf-8') if l.strip()]; c=collections.Counter(lb for r in rows for lb in r['labels']); print({k:c[k] for k in ['退换货','物流','尺码','发票']})"`
Expected: 四个数。**若有一个 <15,如实记进报告与 dev-notes** —— 那是数据实况,不是要掩盖的失败。

- [ ] **Step 8: Commit**

```bash
git add app/topic/metrics.py scripts/eval_topic_clf.py tests/test_topic_metrics.py \
        evals/topic/report.md evals/topic/report.json \
        evals/topic/matrix_confusion.csv evals/topic/matrix_flow.csv
git commit -m "ch10-B T10: 评测(每类 P/R/F1 + 两张矩阵 + 三口径报告 + † 标注)"
```

---

## Task 11: 旁路推理服务 `topic_service/`

**Files:**
- Create: `topic_service/__init__.py`(空)、`topic_service/model.py`、`topic_service/server.py`、`topic_service/__main__.py`
- Create: `tests/test_topic_service.py`

**Interfaces:**
- Consumes: `app.topic.model.load_artifacts`(读 `labels.json` / `inference_config.json`)
- Produces:
  - `topic_service.model.TopicClassifier(model_dir)` —— `.labels`(**必须来自产物**)、`.predict(texts) -> list[dict]`
  - `topic_service.server.create_app(classifier) -> FastAPI`(`POST /predict`、`GET /healthz`)
  - `python -m topic_service --model models/topic-clf --port 8103`

- [ ] **Step 1: 写失败的测试**

Create `tests/test_topic_service.py`:

```python
"""旁路推理服务:标签顺序、阈值、批量形状。

⚠️ 单测**不加载真权重**(不联网、不慢)。`TopicClassifier` 的模型是注入的。
"""

import pytest

from app.topic.taxonomy import LABELS
from topic_service.model import TopicClassifier


class _FakeModel:
    """返回固定的 logits —— 测的是**我们**的解码逻辑,不是模型。"""

    def __init__(self, logits):
        self.logits = logits

    def __call__(self, **kwargs):
        import torch

        return type("Out", (), {"logits": torch.tensor(self.logits)})()


def _classifier(tmp_path, logits, *, labels=None, threshold=0.5, max_length=64):
    import json

    # ⚠️⚠️ **计划订正 14(controller,2026-09-27)—— 这里的三个参数是刻意的** ⚠️⚠️
    # 原稿把 `labels` / `threshold` 写死成 `LABELS` / `0.5`(**也就是实现的默认期望值**),
    # 于是那三条「来自产物」的断言变成了**同义反复**:
    # **一个把 `LABELS`、`0.5` 写死在服务里的实现照样绿** ——
    # 而它们的 docstring 说的正是「服务不许自己写一份」。
    # ⇒ 现在**可以传进被打乱 / 非默认的值**,断言读出来的必须是**那个值**。
    (tmp_path / "labels.json").write_text(
        json.dumps(list(labels if labels is not None else LABELS)), encoding="utf-8")
    (tmp_path / "inference_config.json").write_text(
        json.dumps({"max_length": max_length, "threshold": threshold}), encoding="utf-8")
    return TopicClassifier(tmp_path, model=_FakeModel(logits), tokenizer=None)


def test_label_order_comes_from_the_artifact(tmp_path):
    """⚠️ 标签顺序**必须来自产物**,服务不许自己写一份。

    两侧顺序不一致会产出一张**完全错的分布图,而每个组件都工作正常**:
    模型有输出、scores 在 0–1、写库成功、页面画得出来。

    ⚠️ **订正 14-A**:原稿写进去的就是 `LABELS`、再断言等于 `LABELS` ⇒ **同义反复**,
    写死 `LABELS` 的实现照样绿。⇒ 现在写一份**打乱的**,断言读出的是**打乱的那个顺序**。
    """
    scrambled = list(reversed(LABELS))          # 顺序**被打乱**(内容仍是同样 17 个)
    c = _classifier(tmp_path, [[0.0] * len(LABELS)], labels=scrambled)
    assert c.labels == scrambled                # ← 必须是**产物里那个顺序**
    assert c.labels != list(LABELS)             # ← 且**不是**代码里那份(判别力在这一句)


def test_the_label_set_is_the_same_either_way(tmp_path):
    """顺序来自产物,**类目集合**仍必须等于 `taxonomy.LABELS`(spec §9.1 那条断言)。

    ⚠️ 这两件事要**分开**断:顺序错了是「分布图整张错」,集合错了是「类目表两处漂移」。
    合在一句里时,打乱顺序的正确实现会**误红**,而漏一个类目的实现可能**误绿**。
    """
    scrambled = list(reversed(LABELS))
    c = _classifier(tmp_path, [[0.0] * len(LABELS)], labels=scrambled)
    assert sorted(c.labels) == sorted(LABELS)


def test_threshold_comes_from_the_artifact(tmp_path):
    """⚠️ **订正 14-B**:原稿写 `0.5`、断 `0.5` ⇒ 写死 0.5 的实现照样绿。用**非默认值**。"""
    c = _classifier(tmp_path, [[0.0] * len(LABELS)], threshold=0.7)
    assert c.threshold == 0.7


def test_max_length_comes_from_the_artifact(tmp_path):
    """⚠️ **订正 14-D**(spec §9.1):`max_length` 也在产物里,**不许在服务里写死** ——
    「两侧截断长度不一致」与标签顺序是**同一族的静默失效**。同样用**非默认值**。"""
    c = _classifier(tmp_path, [[0.0] * len(LABELS)], max_length=48)
    assert c.max_length == 48


def test_sigmoid_decoding_picks_every_label_above_threshold(tmp_path):
    """多标签:**每个** logit 独立过阈值,不是取 argmax。

    取 argmax 就退化成单标签了 —— 而验收 ③ 正是要「同时命中多个类目」。
    """
    logits = [[-5.0] * len(LABELS)]
    logits[0][LABELS.index("尺码")] = 5.0
    logits[0][LABELS.index("退换货")] = 5.0
    out = _classifier(tmp_path, logits).predict(["买大了想退"])
    assert set(out[0]["labels"]) == {"尺码", "退换货"}


def test_all_below_threshold_yields_empty_labels(tmp_path):
    """全都不够阈值 ⇒ 空标签,而**不是**硬塞一个最大的。

    硬塞会让每个类目都凭空涨一批 —— 而分布页读的就是它。
    """
    out = _classifier(tmp_path, [[-9.0] * len(LABELS)]).predict(["???"])
    assert out[0]["labels"] == []


def test_scores_are_probabilities(tmp_path):
    out = _classifier(tmp_path, [[0.0] * len(LABELS)]).predict(["x"])
    assert all(0.0 <= v <= 1.0 for v in out[0]["scores"].values())
    assert set(out[0]["scores"]) == set(LABELS)


def test_predict_handles_a_batch(tmp_path):
    out = _classifier(tmp_path, [[0.0] * len(LABELS)] * 3).predict(["a", "b", "c"])
    assert len(out) == 3


class _ExplodingModel:
    """**被调用就抛** —— 用来把「没调用模型」从一句声称变成一个可断言的事实。

    ⚠️ **订正 14-C**:原稿那条测试叫 `..._without_calling_the_model`,
    但**只断言了返回值是 `[]`** —— 名字声称的比断的多。
    一个**真的**把空列表喂进模型的实现(浪费一次前向、还可能因空张量报错)照样绿。
    """

    def __call__(self, **kwargs):
        raise AssertionError("空输入不该调用模型")


def test_empty_input_returns_empty_without_calling_the_model(tmp_path):
    import json

    (tmp_path / "labels.json").write_text(json.dumps(list(LABELS)), encoding="utf-8")
    (tmp_path / "inference_config.json").write_text(
        json.dumps({"max_length": 64, "threshold": 0.5}), encoding="utf-8")
    c = TopicClassifier(tmp_path, model=_ExplodingModel(), tokenizer=None)
    assert c.predict([]) == []          # ← 若它调了模型,这里会抛
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_service.py`
Expected: **FAIL** —— `ModuleNotFoundError: No module named 'topic_service'`

- [ ] **Step 3: 实现**

`topic_service/model.py` 的要点:

```python
class TopicClassifier:
    """旁路推理服务的内核。**标签顺序与阈值一律来自产物目录**,不在这里写死。

    见 `app/topic/model.py` 的 `save_artifacts`:那是唯一的写侧,
    这里是唯一的读侧 —— 两侧不一致会产出一张完全错的分布图而无人报错。
    """

    def __init__(self, model_dir, *, model=None, tokenizer=None):
        arts = load_artifacts(model_dir)
        self.labels = list(arts["labels"])          # ← 来自产物
        self.threshold = float(arts["threshold"])   # ← 来自产物
        self.max_length = int(arts["max_length"])   # ← 来自产物
        ...
```

`predict(texts)` 用 sigmoid 逐位过阈值(`torch.sigmoid(logits) >= threshold`),返回 `[{"labels": [...], "scores": {label: p}}]`。**全低于阈值就返回空列表**,不取 argmax。

`topic_service/server.py` 暴露 `POST /predict`(`{"texts": [...]}`)与 `GET /healthz`,与 `app/api/kb.py` 的 router 同款写法。

- [ ] **Step 4: 跑测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_service.py`
Expected: **PASS**

- [ ] **Step 5: 真机冒烟(单测绿了还不算 —— 本仓规矩)**

Run:
```bash
.venv/Scripts/python.exe -m topic_service --model models/topic-clf --port 8103
# 另一个终端:
curl -s -X POST http://127.0.0.1:8103/predict -H 'Content-Type: application/json' \
  -d '{"texts":["买大了想退"]}'
```
Expected: 返回 JSON,`labels` 里**同时有**两个类目左右。**把返回值原样抄进 `dev-notes/ch10.md`。**

> ⚠️ **中文请求体不能走 `curl` 的 argv**(MSYS2 按 CP936 重编码)。上面这条会失败 —— 换成 stdin heredoc:
> ```bash
> curl -s -X POST http://127.0.0.1:8103/predict -H 'Content-Type: application/json' --data-binary @- <<'JSON'
> {"texts":["买大了想退"]}
> JSON
> ```

- [ ] **Step 6: Commit**

```bash
git add topic_service/ tests/test_topic_service.py
git commit -m "ch10-B T11: 旁路推理服务(标签顺序与阈值一律读产物)"
```

---

## Task 12: `topic_classifications` 表

**Files:**
- Modify: `app/db/models.py`(加 `TopicClassification`)
- Create: `db/ch10.sql`
- Create: `tests/test_topic_table_db.py`

> ⚠️ **`tests/conftest.py` 里没有 `session` fixture**(已核,2026-09-25)。本仓 db 测试的既有写法是直接 `get_sessionmaker()` 自建会话,并在**用完即删 + `commit`**
> (见 `tests/test_api_conversations_db.py` 的模块 docstring:「探针行用完即删,且删除必须 `commit` —— `async with session` 退出是 rollback」)。
> **照那个形状写**,不要新发明一个 fixture。

**Interfaces:**
- Produces: `app.db.models.TopicClassification`(`id` / `low_confidence_question_id` / `labels` / `scores` / `model_version` / `classified_at`)

- [ ] **Step 1: 写失败的测试**

Create `tests/test_topic_table_db.py`:

```python
"""表形状 —— 走真库(db 标记),因为要验的是 SQL 层的唯一键与 JSON 类型。

⚠️ **没有 `session` fixture**:本仓 conftest 不提供它。照
`tests/test_api_conversations_db.py` 的既有写法,自建 engine+sessionmaker,
**探针行用完即删且必须 commit**(`async with session` 退出是 rollback)。
"""

import pytest
from sqlalchemy import delete, text

from app.db.base import get_engine, get_sessionmaker
from app.db.models import TopicClassification

pytestmark = pytest.mark.db

#: 探针 id 用高位段,不撞真实数据。**用完即删。**
PROBE_ID = 999001


@pytest.mark.anyio
async def test_unique_key_makes_reclassification_idempotent():
    """⚠️ 唯一键是**幂等的保证**:同一条池子行重算 = **覆盖**,不是追加。

    与 ch09 的 `review_queue` 刻意不加唯一键**规矩相反,而这是对的**:
    那边是**语义归并**(字面唯一键会在一次合理的归并上响亮地 1062),
    这边是**确定性重算**(同一输入就该覆盖旧值)。两件事性质相反。
    """
    engine = get_engine()
    sm = get_sessionmaker()
    try:
        async with sm() as session:
            await session.execute(
                delete(TopicClassification).where(
                    TopicClassification.low_confidence_question_id == PROBE_ID
                )
            )
            session.add(TopicClassification(
                low_confidence_question_id=PROBE_ID, labels=["尺码", "退换货"],
                scores={"尺码": 0.9}, model_version="test",
            ))
            await session.commit()

            session.add(TopicClassification(
                low_confidence_question_id=PROBE_ID, labels=["运费"],
                scores={}, model_version="test",
            ))
            with pytest.raises(Exception):
                await session.commit()
            await session.rollback()

            await session.execute(
                delete(TopicClassification).where(
                    TopicClassification.low_confidence_question_id == PROBE_ID
                )
            )
            await session.commit()          # ⚠️ 探针行必须 commit 掉,否则下一轮看到残留
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_labels_are_readable_with_json_type():
    """⚠️ JSON 列一律用 `JSON_TYPE()` 读,**不用 `IS NULL` / `IS NOT NULL`**。

    `JSON` 列的 `none_as_null=False` ⇒ Python 的 `None` 落库是**字面 JSON `null`**,
    SQL 上**不是 NULL**。ch09 已经用血换过这一条(T19 拿 `IS NOT NULL`
    去数「有快照的行」,把 JSON `null` 数成了非空)。这里把读法钉死。
    """
    engine = get_engine()
    sm = get_sessionmaker()
    try:
        async with sm() as session:
            await session.execute(
                delete(TopicClassification).where(
                    TopicClassification.low_confidence_question_id == PROBE_ID
                )
            )
            session.add(TopicClassification(
                low_confidence_question_id=PROBE_ID, labels=[], scores={},
                model_version="test",
            ))
            await session.commit()

            got = (await session.execute(text(
                "SELECT JSON_TYPE(labels) FROM topic_classifications "
                "WHERE low_confidence_question_id = :qid"
            ), {"qid": PROBE_ID})).scalar()
            assert got == "ARRAY", f"空标签数组落库应当是 JSON 数组,实际是 {got}"

            await session.execute(
                delete(TopicClassification).where(
                    TopicClassification.low_confidence_question_id == PROBE_ID
                )
            )
            await session.commit()
    finally:
        await engine.dispose()
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_table_db.py`
Expected: **FAIL** —— `ImportError: cannot import name 'TopicClassification'`

- [ ] **Step 3: 加 ORM 模型**

在 `app/db/models.py` 末尾加 `TopicClassification`,docstring 必须写清三件事:

1. **唯一键的取舍**(与 `review_queue` 相反,以及为什么);
2. **JSON 列的读法**(`JSON_TYPE()`);
3. **与 `db/ch10.sql` 的形状差异清单**(照既有各章的写法,逐条对过 `SHOW CREATE TABLE`)。

- [ ] **Step 4: 写 `db/ch10.sql`**

照 `db/ch09.sql` 的写法:头部注释块(`SET NAMES utf8mb4`、设计源路径、**不幂等**声明、两条建库路径的走法),然后 `CREATE TABLE`。**本表是全新的,所以只有 CREATE,没有 ALTER。**

- [ ] **Step 5: 建表**

```bash
.venv/Scripts/python.exe scripts/init_db.py      # create_all,只建不存在的表
```

⚠️ **新表只跑 `init_db.py` 即可**(`create_all` 会建它)。`db/ch10.sql` 是为了**让 DDL 成为权威形状**与**给别的库升级用** —— 两条路径的形状差异要在 ORM docstring 里逐条记(照 ch08/ch09 的写法,别写「只剩两处」这种源不支持的绝对断言,数一遍再写)。

- [ ] **Step 6: 跑测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_table_db.py`
Expected: **PASS**

- [ ] **Step 7: 核对两路建表的形状**

Run: `.venv/Scripts/python.exe -c "import asyncio,sys; from sqlalchemy import text; from app.db.base import get_engine;  ..."` —— 或直接连库跑 `SHOW CREATE TABLE topic_classifications\G`。

**把两路的差异逐条记进 ORM docstring 与 `dev-notes/ch10.md`。** 差异本身不是缺陷(本仓既有各章都有),**没记下来**才是。

- [ ] **Step 8: Commit**

```bash
git add app/db/models.py db/ch10.sql tests/test_topic_table_db.py
git commit -m "ch10-B T12: topic_classifications 表(唯一键保证幂等 + JSON 读法记账)"
```

---

## Task 13: 批处理 `scripts/classify_topics.py`

> ⚠️⚠️ **计划订正 15(controller,2026-09-27)—— 五处;前两处是**计划里根本没写**的东西** ⚠️⚠️
>
> **15-A(`model_version` 计划里一个字都没有,而那一列是 `NOT NULL`)。**
> T12 建的表有 `model_version varchar NOT NULL`;T11 复审裁定(**Q4**):
> `train_meta.json` 里**确实没有权重指纹**(只有语料 `data_fingerprint` 及 `trained_at_utc` 等)。
> ⇒ **`model_version = f"{meta['base']}@{meta['data_fingerprint']}::{meta['trained_at_utc']}"`**
> —— **不能单用 `data_fingerprint`**(它的 docstring 逐字写着它是「**语料**的、**一次运行**的」指纹
> ⇒ 两份**权重不同**的训练可以有同一个值,而那正是这一列要分的东西)。
> **< 128 字符**(列宽),**不许把整个 dict 塞进去**(分布页要 `COUNT(DISTINCT model_version)`)。
> **并在代码注释里写明这是个诚实的近似** —— 严格的权重指纹要改 `save_artifacts`(`sha256(model.safetensors)`)
> 并**重跑训练**,那是 T9/T10 的账,不该在这里补。
>
> **15-B(`classify_batch` 只吃替身,而 `main()` 需要的**真** HTTP 客户端计划里没有)。**
> T11 复审裁定(**Q3**),**照这个写**:
> - 替身 `predict()` **返回裸 list**(逐字照本节 Step 1 的 `_FakeClient`,不许改);
> - **真**客户端的 `predict()`:`payload = json.loads(resp.text)` ⇒
>   **显式校验它是 `dict` 且含 `"results"`**(形状漂移要**响亮**失败)⇒ `return payload["results"]`;
> - **补一条客户端侧单测**(假 transport 回 `{"results": [...]}`)钉住那个键 ——
>   它今天**只在服务侧被钉**,客户端侧一个字都没有;
> - ⛔ **不许**让 `classify_batch` 自己去拆 `["results"]`(那会把 HTTP 形状漏进纯逻辑层,替身也就与本节 Step 1 不符)。
>
> **15-C(空标签的处置 —— T12 交过来的账)**:**`labels: []` 是合法 JSON,`NOT NULL` 拦不住**。
> ⇒ **允许写**(服务返回空是**合法的模型输出**:全部低于阈值)。
> **但必须数出来并打印**「本批有空标签 N 条」——
> 否则「服务每次都返回空」会被分布页读成「这些问题没有主题」,而那是**故障被读成业务结果**
> (本节 Step 1 第一条测试防的就是这个,但那条防的是**抛异常**,防不了**返回空**)。
>
> **15-D(多一条形状校验,很便宜)**:`classify_batch` 要校验每条结果的
> **`labels` 里每个名字都在 `taxonomy.LABELS` 里** —— 那是「标签顺序/类目表两处漂移」这条链上
> **最靠近数据的那一道**;代价一行。
>
> **15-E(Step 6 会真的往共享表里写行)**:`low_confidence_questions` 是**共享且只追加**的,
> `topic_classifications` 也是。⇒ 报告里**必须报准确条数**,并且**任何「查最近这几条」式的断言必须按
> `low_confidence_question_id` 过滤**(本仓记过「被上次运行的数据污染 ⇒ 偶尔红偶尔绿」那一类)。
>
> ---
>
> ⚠️⚠️ **计划订正 17(controller,2026-09-27)—— T13 交付后回填的三条** ⚠️⚠️
>
> **17-B(`main()` 的写侧,计划里一个字都没有 —— 这是本节最大的缺口)。**
> 15-B 补了 **HTTP 侧**、15-A 补了 `model_version`,**唯独落下「写进 MySQL 的那一侧」**
> ⇒ 实现者只能自己设计,于是产生了下面这处**偏离**:
> **`write` 是同步回调 ⇒ 落库是「整次运行一个事务」,不是 §9.2 字面写的「一批一事务」。**
> **裁定:接受这个偏离,并把它写进计划。** 理由(实现者给的,我认同):
> ① 方向**更严** —— **不留半份结果**;而「半份结果」在分布页上与完整结果**长得一模一样**,
>    没有任何东西能分辨;② 代价是异常时前面成功的批次也不落库,但**重跑幂等**(唯一键保证覆盖),
>    **不丢**;③ §9.2 那句「一批一事务」的原意是「别写半份」,这个实现**更彻底地**满足了它。
> ⚠️ **要一并写清**:`write` 回调的**契约**(它收到什么、负责什么、失败怎么办),
> 否则下一个读的人会以为它是「一批一次」的。
>
> **17-C(两处措辞,都是「名字/注释与实际不符」那一族)**:
> ① Step 1 的 `test_a_batch_is_all_or_nothing` **docstring 写「第 7 条」,而装置里坏的那条在第 6 个**;
> ② 四条用例一律 `pytest.raises(Exception)` —— **分不出错因**(服务连不上 / 形状不对 / 条数不符
>    会得到同一个异常类型)。实现者**逐字保留了原稿**、另加了三条改用窄类型的。
> ⇒ **订正:把这四条收窄到各自的异常类型**,并把「第 7 条」改对。
> (**名字与语义不符**本仓已编目;`raises(Exception)` 是它的同族 —— 一条**永远通过**的断言。)
>
> **17-D(流程)**:`.superpowers/ch10b_t13_*`(探针 / 证据 / pytest 转录)留在本机 **未 `git add`**。
> **判据一致**:`CLAUDE.md` 那条「**被跟踪文档引用为凭据 ⇒ 入库**」——
> 今天**没有**任何被跟踪的文档引用它们,**不必入库**;
> ⚠️ **但 dev-notes 一旦引用某个读数,那个读数的凭据就要先 `git add`**(本仓 T8 那次就是这么办的)。
> ⇒ **controller 写 dev-notes 时逐条核**。

**Files:**
- Create: `scripts/classify_topics.py`
- Create: `tests/test_classify_topics.py`

**Interfaces:**
- Consumes: `app.topic.clean.clean`(**必须 import —— 任务 2 的守卫守着**)、`topic_service` 的 HTTP 接口、`app.db.models.TopicClassification`
- Produces: `topic_classifications` 的行

- [ ] **Step 1: 写失败的测试**

Create `tests/test_classify_topics.py`:

```python
"""批处理:整批原子、服务故障响亮失败、幂等覆盖。用**假服务**(不联网)。"""

import pytest

from scripts.classify_topics import classify_batch


class _FakeClient:
    def __init__(self, results=None, error=None):
        self.results = results
        self.error = error
        self.calls = []

    async def predict(self, texts):
        self.calls.append(list(texts))
        if self.error:
            raise self.error
        return self.results


@pytest.mark.anyio
async def test_service_failure_does_not_write_anything():
    """⚠️ **基础设施故障绝不伪装成业务结果。**

    连不上服务时写一行空标签的后果:分布页会把「服务挂了」读成
    「这些问题没有主题」—— 正是 ch09 那条 👎 回捞失败被读成
    「知识库缺这块」的同款事故(spec §9.2)。
    """
    written = []
    client = _FakeClient(error=ConnectionError("connect refused"))

    with pytest.raises(Exception):
        await classify_batch([{"id": 1, "question": "x"}], client=client,
                             write=written.append, batch_size=8)
    assert written == [], "服务故障时不许写任何行"


@pytest.mark.anyio
async def test_a_batch_is_all_or_nothing():
    """一批里第 7 条失败 ⇒ **整批不写**。

    部分写入会留下「一半归过类」的池子,而重跑时分不清哪些是这次的、
    哪些是上次的。
    """
    written = []
    client = _FakeClient(results=[{"labels": ["尺码"], "scores": {}}] * 5 +
                                 [_ for _ in [None]] +   # 第 6 条形状不对
                                 [{"labels": ["运费"], "scores": {}}] * 2)
    with pytest.raises(Exception):
        await classify_batch([{"id": i, "question": f"q{i}"} for i in range(8)],
                             client=client, write=written.append, batch_size=8)
    assert written == []


@pytest.mark.anyio
async def test_clean_is_applied_before_sending():
    """⚠️ 发出去的文本必须是**清洗过**的 —— 训练侧也洗过,两侧同源。

    不洗的话模型在真机上看到的是另一种文本(spec §5.1)。
    """
    client = _FakeClient(results=[{"labels": [], "scores": {}}])
    await classify_batch([{"id": 1, "question": " 退货   怎么走  13800138000 "}],
                         client=client, write=lambda r: None, batch_size=8)
    sent = client.calls[0][0]
    assert sent == "退货 怎么走 <手机号>"


@pytest.mark.anyio
async def test_results_carry_the_question_id():
    """结果必须能对回池子行 —— 对不回去的批处理没有意义。"""
    written = []
    client = _FakeClient(results=[{"labels": ["尺码"], "scores": {}}])
    await classify_batch([{"id": 42, "question": "买大了"}],
                         client=client, write=written.append, batch_size=8)
    assert written[0]["low_confidence_question_id"] == 42
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_classify_topics.py`
Expected: **FAIL** —— `ModuleNotFoundError`

- [ ] **Step 3: 实现**

`classify_batch(rows, *, client, write, batch_size)` 的语义:
- 对每一批:清洗 → `client.predict(texts)` → 校验返回条数**与输入一致**(不一致 ⇒ 抛) → 全部成功才 `write(...)`;
- 任何异常**向上抛**,已写的批次保留(那是**已经成功**的批次,不丢);
- 幂等由表的唯一键保证(重跑覆盖)。

`main()`:
- `--limit` / `--batch-size` / `--service-url`(默认 `http://127.0.0.1:8103`)/ `--dry-run`;
- 读池:`SELECT id, question FROM low_confidence_questions ORDER BY id`;
- `--dry-run` 只打印「会写哪些行」,不落库。

- [ ] **Step 4: 跑测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_classify_topics.py`
Expected: **PASS**

- [ ] **Step 5: 跑清洗同源守卫(它现在应该全绿了)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topic_clean.py`
Expected: **PASS** —— `test_both_sides_use_the_same_clean` 这条**现在应该绿**(任务 3 让 `prepare_topic_data.py` 绿了一半,本任务让 `classify_topics.py` 绿另一半)

- [ ] **Step 6: 干跑 + 真跑**

Run:
```bash
.venv/Scripts/python.exe scripts/classify_topics.py --dry-run --limit 20
.venv/Scripts/python.exe scripts/classify_topics.py
```

**把写入行数、以及分布页会看到的两个数(总行数 / 不同问题数)抄进 `dev-notes/ch10.md`。**

- [ ] **Step 7: Commit**

```bash
git add scripts/classify_topics.py tests/test_classify_topics.py
git commit -m "ch10-B T13: 批处理(整批原子 + 服务故障响亮失败 + 清洗同源)"
```

---

## Task 14: `GET /api/topics/distribution`

> ⚠️⚠️ **计划订正 17-A(controller,2026-09-27)—— 分布页的「不同问题数」今天是个没意义的数** ⚠️⚠️
>
> **T13 报上来的,我在库里独立复核过**:
> ```
> SELECT COUNT(*), COUNT(DISTINCT low_confidence_question_id) FROM topic_classifications  ->  65, 65
> SELECT COUNT(*), COUNT(DISTINCT question)                  FROM low_confidence_questions ->  65, 33
> ```
> ⇒ **`COUNT(DISTINCT low_confidence_question_id)` 与 `COUNT(*)` 恒等**(那一列有唯一键 `uk_pool_question`),
> 而本任务原稿那句注释说它是为了处理「池子有大量重复题面」—— **它处理不了**。
> 页面会显示「总行数 65 / 不同问题数 **65**」,而池子里**只有 33 条不同的题面**。
> ⇒ **改法**:要数「不同问题」就得**回池子按题面数** ——
> `LEFT JOIN low_confidence_questions q ON q.id = t.low_confidence_question_id`
> 再 `COUNT(DISTINCT q.question)`(**summary 与每个 bucket 的 `distinct` 都要改**)。
> ⚠️ 注意那 33 是**题面**的不同数,**不是清洗后的**;若想按清洗后文本数,就在 SQL 之外用 `clean()` 数一层
> —— **别在 SQL 里假装洗过**(`clean()` 是 Python 侧的唯一实现,这一章已经为「清洗两处实现」付过一次账)。
> ⚠️ **顺手核对语义**:`total` 应当 = **分类结果的条数**(65),
> 而「不同问题数」是**池子的题面数**(33) —— 两个数**口径不同**,报告的措辞要说清,别让人以为对不上是 bug。

**Files:**
- Create: `app/api/topics.py`
- Modify: `app/main.py`(挂 router,**必须在 `mount("/")` 之前**)
- Create: `tests/test_api_topics_db.py`(db)

> ⚠️ **`tests/conftest.py` 里没有 `client` fixture**(已核,2026-09-25)。而且本仓的
> `TestClient` 路线**刻意不接真库** —— `tests/test_api_ticket.py` 的 docstring 写着原因:
> `TestClient` 在它自己的 portal 事件循环里跑请求,会把 `get_engine()` 那个
> **lru_cache 单例绑到那个循环上**,退出 `with` 后同进程里后面所有 db 测试都会拿到
> 跨循环的连接。
> **所以本任务走既有 db 测试的写法:直调端点函数 + 真 `get_sessionmaker()`**
> (照 `tests/test_api_conversations_db.py` 直调 `list_conversations` 的先例)。
> 这个任务的全部价值就是验 `JSON_TABLE` 那段 SQL,**必须**打到真库。

**Interfaces:**
- Produces:
  - `GET /api/topics/distribution` → `{"total": int, "distinct_questions": int, "last_classified_at": str|null, "model_versions": [str], "buckets": [{"label": str, "count": int, "distinct": int}]}`

- [ ] **Step 1: 写失败的测试**

Create `tests/test_api_topics_db.py`:

```python
"""分布端点 —— **走真库、直调端点函数**。

⚠️ 两个刻意的选择:
① **不用 `TestClient`**:它会把 `get_engine()` 的 lru_cache 单例绑到 portal 循环上,
   同进程后面的 db 测试会拿到跨循环连接(原因写在 `tests/test_api_ticket.py` 的 docstring 里);
② **不用替身**:本任务的全部价值就是验 `JSON_TABLE` 那段 SQL,替身把它替掉就什么都没测了。
   照 `tests/test_api_conversations_db.py` 直调 `list_conversations` 的先例。

探针行**用完即删且 commit** —— `async with session` 退出是 rollback。
"""

import pytest
from sqlalchemy import delete

from app.api.topics import distribution
from app.db.base import get_engine, get_sessionmaker
from app.db.models import TopicClassification
from app.topic.taxonomy import LABELS

pytestmark = pytest.mark.db

#: 探针用的池子行 id。**高位段,不撞真实数据。**
P1, P2 = 999101, 999102


async def _seed(sm):
    """三条探针:P1 两个标签、P2 一个标签、P2 **与** P1 是同一个问题文本
    (为了「不同问题数 < 行数」可观测)。"""
    async with sm() as s:
        await s.execute(delete(TopicClassification).where(
            TopicClassification.low_confidence_question_id.in_([P1, P2])))
        s.add(TopicClassification(low_confidence_question_id=P1,
                                  labels=["尺码", "退换货"], scores={}, model_version="probe"))
        s.add(TopicClassification(low_confidence_question_id=P2,
                                  labels=["退换货"], scores={}, model_version="probe"))
        await s.commit()


async def _cleanup(sm):
    async with sm() as s:
        await s.execute(delete(TopicClassification).where(
            TopicClassification.low_confidence_question_id.in_([P1, P2])))
        await s.commit()


@pytest.mark.anyio
async def test_aggregation_uses_json_table():
    """⚠️ 标签是 JSON 数组,要按标签计数就必须展开它。

    实测(MySQL 8.0.46,`.superpowers/probe_ch10_jsontable.py`):
    `JSON_TABLE(labels, '$[*]' COLUMNS (label VARCHAR(64) PATH '$'))` 可用,
    返回 `[('尺码', 1), ('退换货', 1)]`。**已验证,不是照文档推的。**

    反面做法:把标签拉回 Python 再数 —— 那在数据量上去之后会变成
    「一次请求拉全表」,而它在演示规模下完全看不出来。
    """
    engine = get_engine()
    sm = get_sessionmaker()
    try:
        await _seed(sm)
        async with sm() as session:
            body = await distribution(session=session)
        counts = {b["label"]: b["count"] for b in body["buckets"]}
        assert counts["尺码"] >= 1 and counts["退换货"] >= 2
    finally:
        await _cleanup(sm)
        await engine.dispose()


@pytest.mark.anyio
async def test_multi_label_row_counts_toward_every_label():
    """一行带两个标签,在**两个**桶里各算一次 —— 这是多标签的正确读法。"""
    engine = get_engine()
    sm = get_sessionmaker()
    try:
        await _seed(sm)
        async with sm() as session:
            body = await distribution(session=session)
        assert sum(b["count"] for b in body["buckets"]) > body["total"]
    finally:
        await _cleanup(sm)
        await engine.dispose()


@pytest.mark.anyio
async def test_zero_count_labels_are_still_returned():
    """⚠️ **一个标签都没有的类目也要出现在结果里(count=0)。**

    不补零的话,页面上的条形图会**少几类**,而少的那几类看起来像
    「这一类没问题」—— 恰恰相反,它们是一条样本都没有。
    """
    engine = get_engine()
    sm = get_sessionmaker()
    try:
        async with sm() as session:
            body = await distribution(session=session)
        labels = {b["label"] for b in body["buckets"]}
        assert labels == set(LABELS), f"缺这些类目:{set(LABELS) - labels}"
    finally:
        await engine.dispose()


@pytest.mark.anyio
async def test_empty_table_returns_zeroes_not_an_error():
    """没跑过批处理时返回零值结构,而不是 500 —— 页面第一次打开就是这个状态。"""
    engine = get_engine()
    sm = get_sessionmaker()
    try:
        async with sm() as session:
            body = await distribution(session=session)
        assert body["total"] >= 0
        assert all(b["count"] == 0 for b in body["buckets"] if b["label"] not in
                   ("尺码", "退换货"))  # 真实库里可能有别的行,别把「表非空」当前提
    finally:
        await engine.dispose()
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_topics_db.py`
Expected: **FAIL** —— `404`(路由不存在)

- [ ] **Step 3: 实现 `app/api/topics.py`**

```python
"""主题分布(ch10)—— **只读**,给管理台的「主题分布」标签页用。

服务的是「先补哪块知识」这个决策,所以页面上的数必须能被追溯:
除了每类的条数,还要给出**不同问题数**(池子有大量重复题面,
只报行数会把某个类虚高)与**最后归类时间**(图是旧的还是新的)。
"""

from fastapi import APIRouter, Depends
from sqlalchemy import text

from app.db.session import get_session

router = APIRouter()

#: ⚠️ 标签是 JSON 数组,按标签计数必须展开它。`JSON_TABLE` 在 MySQL 8.0.46
#: 上实测可用(`.superpowers/probe_ch10_jsontable.py`)。**不许**改成
#: 「把 labels 拉回 Python 再数」—— 那在数据量上去之后会变成一次请求拉全表,
#: 而在当前 63 行的规模下完全看不出来。
_DISTRIBUTION = text("""
    SELECT jt.label AS label,
           COUNT(*) AS n,
           COUNT(DISTINCT t.low_confidence_question_id) AS distinct_n
    FROM topic_classifications t,
         JSON_TABLE(t.labels, '$[*]' COLUMNS (label VARCHAR(64) PATH '$')) jt
    GROUP BY jt.label
    ORDER BY n DESC
""")

_SUMMARY = text("""
    SELECT COUNT(*) AS total,
           COUNT(DISTINCT low_confidence_question_id) AS distinct_questions,
           MAX(classified_at) AS last_classified_at,
           COUNT(DISTINCT model_version) AS versions
    FROM topic_classifications
""")


@router.get("/api/topics/distribution")
async def distribution(session=Depends(get_session)):
    summary = (await session.execute(_SUMMARY)).one()
    rows = (await session.execute(_DISTRIBUTION)).all()
    total = int(summary.total or 0)

    buckets = [
        {"label": r.label, "count": int(r.n), "distinct": int(r.distinct_n)}
        for r in rows
    ]
    # ⚠️ 把**一个标签都没有的类目**也补上(count=0)。
    #    只返回有数据的类:页面上的条形图会「少几类」,而少的那几类看起来
    #    像「这一类没问题」—— 恰恰相反,它们是**一条样本都没有**。
    seen = {b["label"] for b in buckets}
    buckets += [{"label": lb, "count": 0, "distinct": 0} for lb in LABELS if lb not in seen]
    buckets.sort(key=lambda b: (-b["count"], LABELS.index(b["label"])))

    return {
        "total": total,
        "distinct_questions": int(summary.distinct_questions or 0),
        "last_classified_at": (
            summary.last_classified_at.isoformat(sep=" ", timespec="minutes")
            if summary.last_classified_at else None
        ),
        "model_versions": [],   # 由下面的查询填;为空表示还没跑过批处理
        "buckets": buckets,
    }
```

`model_versions` 用第二条查询填(页面顶部要显示「这个图是谁算的」):

```python
_VERSIONS = text(
    "SELECT DISTINCT model_version FROM topic_classifications ORDER BY model_version"
)
```

> ⚠️ `LABELS` 必须 `from app.topic.taxonomy import LABELS` —— **补零这件事正是「17 类表是唯一来源」的一个用法**:类目清单从权威表来,不从已有数据来。

- [ ] **Step 4: 挂 router(⚠️ **两行**,不是一行)**

本仓的 router 是**成套**的:`app/main.py:10-16` 的 import 块 + `:116-129` 的 include 块。
**只加 include 会 `NameError`。**

① 在 import 块里按字母序加(`topics` 排在 `review` 之后,即**最后一行**):

```python
from app.api.topics import router as topics_router
```

② 在 `app.include_router(review_router)` 之后加:

```python
app.include_router(topics_router)
```

⚠️ **必须在 `app.mount("/", ...)` 之前** —— 挂反了静态目录会抢走 `/api/*`(本仓硬约束)。

- [ ] **Step 5: 跑测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_topics_db.py`
Expected: **PASS**

- [ ] **Step 6: ⚠️ 补上「主链路零调用分类器」的源码扫描测试(spec §3.3 ① / §9.4)**

**这一步是计划自查时补上的 —— 原稿漏了它。** 它是本章**两条不变量**里的第一条,没有它,「实时对话主链路不调它」就只是一句写在 spec 里的话。加进 `tests/test_topics_boundary.py`:

```python
"""⚠️ 纪律:实时对话主链路**零处**调用主题分类器。

这是本章两条不变量里的第一条。用**源码扫描**而不是运行时断言 ——
要抓的是「有没有人写下去」,不是「跑到了没有」
(照 ch09 `test_no_module_outside_observability_imports_langfuse` 的先例)。

**为什么它值得一条测试**:分类器接进主链路的最可能形态是「顺手在
`nodes.py` 里加一句兜底」—— 而那样做的后果是主链路多一次
几百毫秒的旁路调用,且**没有任何断言会红**;等发现时,「实时对话不调它」
这个约定已经消失了,而当初为什么定这条约定也没人记得。
"""

import re
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"

#: 主链路上**不许**出现的名字。加一个就加一条 —— 别写成通配,
#: 通配会把「日志里提了一句 topic」也判成违规,那会诱人去放宽它。
#
# ⚠️⚠️ **计划订正 16(controller,2026-09-27)—— 这一条漏了一种写法** ⚠️⚠️
# 下面那个正则只认 `from app.topic.model import …` 与 `import app.topic.model`,
# **漏掉 `from app.topic import model`**(以及 `from app.topic import model, metrics`)。
# 而**本仓另一条同款守卫明确把三种惯用写法都放行过** ——
# `tests/test_topic_clean.py::test_both_sides_use_the_same_clean` 的 docstring 逐字写着:
# 「原来只认 `from app.topic.clean import … clean` **一种写法**,一个**正确**的
# 任务 3 / 任务 13 若写成 `from app.topic import clean` 或 `import app.topic.clean`
# 就会被判红 —— 那会逼它们为了变绿而**改自己的 import**」。
# ⇒ **这里是它的镜像**:那条守卫怕**误红**,这条怕**误绿**。
# **三条惯用写法一个都不许漏**,并**各配一条元测试**(把写法喂进去 ⇒ 必须命中)。
FORBIDDEN = re.compile(
    r"^\s*(?:import\s+topic_service\b"
    r"|from\s+topic_service\b"
    r"|import\s+app\.topic\.(?:model|metrics)\b"
    r"|from\s+app\.topic\.(?:model|metrics)\b"
    r"|from\s+app\.topic\s+import\s+.*\b(?:model|metrics)\b)",
    re.M,
)

#: ⚠️ **已知不可覆盖**:动态导入(`importlib.import_module("app.topic.model")`)、
#: `__import__`、以及把模块名拼成字符串 —— **正则看不见**。
#: ⇒ **如实写进测试的 docstring**,别让这条守卫读起来比它实际覆盖的宽
#: (本仓已编目:一句「看起来成立」的注释不是守卫)。
#: 真正兜住动态导入的是**别的东西**(比如服务是**独立进程**、主链路里根本没有它的地址)。

#: 扫描范围:请求路径那几处。
SCOPE = ("agent", "api/chat.py")


def test_main_chain_never_imports_the_topic_classifier():
    offenders = []
    for rel in SCOPE:
        target = APP / rel
        paths = target.rglob("*.py") if target.is_dir() else [target]
        for p in paths:
            if FORBIDDEN.search(p.read_text(encoding="utf-8")):
                offenders.append(p.relative_to(APP).as_posix())
    assert offenders == [], (
        f"实时对话主链路里出现了主题分类器 —— 本章约定它只在旁路跑:{offenders}"
    )


def test_the_scan_actually_scans_something():
    """⚠️ **元测试**:上一条若因为 SCOPE 写错而扫了个空目录,它会**恒绿**。

    对着一个不存在的路径 `rglob` 返回空迭代器,`offenders` 永远是 `[]` ——
    一条永远通过的守卫,比没有守卫更糟。
    """
    found = 0
    for rel in SCOPE:
        target = APP / rel
        found += len(list(target.rglob("*.py"))) if target.is_dir() else int(target.exists())
    assert found >= 5, f"扫描范围只命中了 {found} 个文件 —— SCOPE 大概写错了"
```

- [ ] **Step 7: 跑边界测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_topics_boundary.py`
Expected: **PASS**(`test_the_scan_actually_scans_something` 保证它扫到了真东西)

- [ ] **Step 8: 真机看一眼**

Run: `.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000`(另开终端)
```bash
curl -s http://127.0.0.1:8000/api/topics/distribution | head -c 400
```
Expected: 返回 JSON。**把输出抄进 `dev-notes/ch10.md`。**

- [ ] **Step 9: Commit**

```bash
git add app/api/topics.py app/main.py tests/test_api_topics_db.py tests/test_topics_boundary.py
git commit -m "ch10-B T14: GET /api/topics/distribution(JSON_TABLE 聚合)+ 主链路零调用守卫"
```

---

## Task 15: 管理台「主题分布」标签页(**Vibe Coding**)

**Files:**
- Modify: `app/static/admin.html`

> 按用户给的例外,**不套 brainstorm / TDD / code review** —— 起一版,用户看效果说改什么。

- [ ] **Step 1: 加标签按钮**

在 `admin.html:172-176` 那组 `<nav class="tabs">` 里加一个(`switchTab` 是现成的,不用改):

```html
    <button class="tab" data-tab="主题分布" type="button">主题分布</button>
```

- [ ] **Step 2: 加标签页容器**

在 `#tab-评测` 那个 `div` 之后加 `<div id="tab-主题分布" style="display:none">`,里面放:
- 顶部一行**常驻**的汇总:总行数 · 不同问题数 · **最后归类时间** · `model_version`;
- 一个条形区(每行 = 一个类目,条宽按条数占比);
- **一句常驻的诚实提示**(见 Step 4)。

- [ ] **Step 3: 加渲染函数**

新增 `renderTopics()`,在 `switchTab` 切到这个标签时调用一次。**不引图表库** —— 用 div 宽度画条,与 `admin.html` 零依赖的现状一致:

```javascript
  async function renderTopics() {
    const box = $("topic-bars");
    box.textContent = "加载中…";
    const r = await fetch("/api/topics/distribution").then((x) => x.json());
    const max = Math.max(1, ...r.buckets.map((b) => b.count));
    box.innerHTML = "";
    for (const b of r.buckets) {
      const row = document.createElement("div");
      row.className = "topic-row";
      row.innerHTML = `<span class="topic-name">${b.label}</span>
        <span class="topic-bar"><i style="width:${(b.count / max) * 100}%"></i></span>
        <span class="topic-num">${b.count} 条 / ${b.distinct} 个问题</span>`;
      box.appendChild(row);
    }
    // ⚠️ 空态要说人话 —— 池子还没攒起来时这张图是空的,
    //    而「空白」读起来像「页面坏了」。
    if (!r.buckets.length) box.textContent = "还没有归类结果。先跑 scripts/classify_topics.py。";
  }
```

> ⚠️ `switchTab` 是**同步**函数、且现有的几个标签页是直接显示的。切到「主题分布」时要**触发一次加载**(每次切都刷,数据会变)。动手前先读 `switchTab` 的现有实现(`admin.html:1051`)再改。

- [ ] **Step 4: 加那句「诚实要求」(spec §10)**

在条形图**上方常驻**一行小字,内容由接口返回的数据填:

```javascript
    $("topic-asof").textContent =
      `数据截至 ${r.last_classified_at || "—"} ·  共 ${r.total} 行 / ${r.distinct_questions} 个不同问题`;
```

并在 `total < 100` 时**额外**追加一句:

```javascript
    if (r.total < 100) {
      $("topic-caveat").textContent =
        "样本量还很少,这张图**不足以支撑「先补哪块知识」的决策**。";
    }
```

> 理由(spec §9.5):池子今天只有 63 行。**页面不写这句,它就会被当成分布结论读。**

- [ ] **Step 5: 手工验(前端没有自动化测试,靠人看)**

1. 起服务:`.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000`
2. 浏览器开 `http://localhost:8000/admin.html`,点「主题分布」
3. 期望:看到条形图、每行两个数、顶部有「数据截至」与那句小字

**把截图或描述记进 `dev-notes/ch10.md`。**

- [ ] **Step 6: Commit**

```bash
git add app/static/admin.html
git commit -m "ch10-B T15: 管理台主题分布页(Vibe Coding)"
```

---

## Task 16: 验收脚本 + 章级文档同步

> ⚠️ **计划订正 10 带下来的三条(controller,2026-09-26)**:
>
> **① 验收 ① 不许断言头四类 ≥15,也不许退成 ≥8。**
> 那句「真正的 ≥15 由验收脚本在真实数据上核对」是**必然落空的假话**(已从 Task 7 删掉)。
> 而 `≥ 8` **同样是对真实列撒谎、且零判别力**(把分层整个删掉它照样绿)。
> 验收该断的是:**构成**(120 = real 80 + synth 40)/ **三份不相交** / **合计 = 输入 − 不可信靶子**;
> support 与 `†` 清单**只打出来入账**。
>
> **② 章级文档要写进 spec 的实现订正(三处)**:
> - **§5.3**:「头四类每类 ≥15、其余类按比例」**两半都没实现** —— `take()` 是按主标签近均匀轮转。
>   且**算术上不可达**:`17 × 15 = 255 > 169` 个标签槽。
>   **如实写「`物流` 那 12 条不是池子不够(真实池 55 条),是机制没加权」**,
>   与 `尺码`(真实池 4)/`发票`(真实池 14)的**真池子封顶**分开写。
> - **§8.3 / §8.4**:写明 `†` 判据是「**每一层各自**」的 support
>   ⇒ 「只看真实 80」那一列会挂**比全体多得多的** `†`(16/17 vs 14/17),而**那正是 §8.4 指定的权威列**。
>   不写明的话,验收的人会把它读成缺陷。
> - **§6.3**:CP-2 的 0.0% 是「**看图通过**」而**无逐条核对痕迹**;样本 70% 是合成、
>   严口径下 13/17 类零真实覆盖 ⇒ **不许把「预标错误率 0%」当成流水线的性质引用**。
>
> **③ 本任务要把「本章测不出结论性 per-class 指标」写进「已知问题与未达成项」**:
> 根因是**真实语料按类稀薄**(尺码 4 / 发票 14),而 §8.4 指定的权威列是「只看真实 80」;
> **解救办法是补真实语料,不是调阈值、也不是把测试集做大**
> (做大只改善合并列;真实列上 `尺码` 的上限永远是 4)。
>
> **④ 本任务必须写明测试集标签的来源(用户 2026-09-26 拍板「跳过人工复核」)**:
> `evals/topic/topic_test.jsonl` 的标签**是预标产物**;实测 **120 行里只有 12 行带 `human_reviewed`**
> (那 12 行是从 CP-2 复核过的那 84 条里流过去的:`train 60 + val 12 + test 12 = 84`)。
> ⇒ **报告里那句 F1 必须带这个限定**,不许被读成「在人工标注的黄金集上测出来的」。
> ⚠️ **`import-test` 因此没有被执行** —— 跑了它会给 120 行**全部**打上 `human_reviewed: True`,
> 而其中 108 行**没有人看过**,那正是本仓最忌讳的「看起来做完了、其实没做」。


**Files:**
- Modify: `scripts/acceptance_ch10.sh`(任务 A3 建的那个,补 ①–③ 三节)
- Modify: `CLAUDE.md`
- Modify: `dev-notes/ch10.md`

- [ ] **Step 1: 补验收 ①(测试集 F1 报告能跑出来)**

```bash
# ---- 验收 ①:测试集各类目 F1 与矩阵报告 ----
.venv/Scripts/python.exe scripts/eval_topic_clf.py --report /tmp/ch10_report.md > /tmp/ch10_eval.log 2>&1
if [ -s /tmp/ch10_report.md ] && grep -q "micro-F1" /tmp/ch10_report.md; then
  ok "评测报告已生成:$(grep -c '' /tmp/ch10_report.md) 行"
else
  bad "评测报告没生成或内容不对,见 /tmp/ch10_eval.log"
fi
```

⚠️ 判据**不能只是「文件存在」** —— 一个空文件也「存在」。上面同时断了非空与关键字段。

- [ ] **Step 2: 补验收 ②(跑批 + 分布页有统计)**

```bash
# ---- 验收 ②:归类写库 + 分布接口能读到 ----
.venv/Scripts/python.exe scripts/classify_topics.py --limit 20 > /tmp/ch10_classify.log 2>&1
DIST=$(curl -s http://127.0.0.1:8000/api/topics/distribution)
if python -c "import json,sys; d=json.loads('''$DIST'''); sys.exit(0 if d['total'] > 0 and d['buckets'] else 1)"; then
  ok "分布接口有数据:$(echo "$DIST" | head -c 120)"
else
  bad "分布接口没有数据:$DIST"
fi
```

> ⚠️ 把 JSON 用 `'''$DIST'''` 内插进 `python -c` 是脆的(引号/转义)。**照 `acceptance_ch09.sh` 里读 JSON 的既有做法写** —— 它有现成的样板,别自己发明。

- [ ] **Step 3: 补验收 ③(多诉求句同时命中多类)**

```bash
# ---- 验收 ③:「买大了想退」必须**同时**命中多个类目 ----
# ⚠️ 这一条**不走模型** —— 直接打旁路服务的 /predict。
#    走对话链路的话测的就是「模型会不会调工具」,与分类器无关。
RESP=$(curl -s -X POST http://127.0.0.1:8103/predict -H 'Content-Type: application/json' --data-binary @- <<'JSON'
{"texts":["买大了想退"]}
JSON
)
N=$(python -c "import json,sys; print(len(json.loads(sys.stdin.read())['results'][0]['labels']))" <<< "$RESP")
if [ "$N" -ge 2 ]; then
  ok "多诉求句命中 $N 个类目:$RESP"
else
  bad "「买大了想退」只命中 $N 个类目(期望 ≥2):$RESP"
fi
```

- [ ] **Step 4: 加「输出干净」自检**

照 `acceptance_ch09.sh` 的做法:把本轮输出转录一份,在判词**之前**扫它 —— 命中 `command not found` / `syntax error` / `unexpected EOF` 就判红。

> ⚠️ 踩过的坑:文案里的反引号被 bash 当命令替换执行,喷了一屏错误而脚本照样报 6/6。**所有含反引号的文案改用单引号包裹**,并靠这条自检兜住。

- [ ] **Step 5: 跑验收**

Run: `bash scripts/acceptance_ch10.sh`

⚠️ **跑之前先清掉 8000 / 8103 的残留进程**(否则 curl 到旧代码 —— 本仓记过的那类假红)。**把全文转录抄进 `dev-notes/ch10.md`。**

- [ ] **Step 6: 同步 `CLAUDE.md`**

加 ch10 小节:章号、分支、交付了什么、**五条命门**(problem_type 从 dtype 猜 / 标签顺序 / train-serve 同源 / 整批原子 / JSON 读法)、以及**指向 `dev-notes/ch10.md` 与 spec**。同时改「高频命令」段,加:

```bash
# ch10(前置:MySQL + 真实 key;起服务前需要先起旁路推理服务)
.venv/Scripts/python.exe -m topic_service --model models/topic-clf --port 8103
.venv/Scripts/python.exe scripts/prepare_topic_data.py collect   # 语料合流
.venv/Scripts/python.exe scripts/train_topic_clf.py              # 训练(约十几分钟)
.venv/Scripts/python.exe scripts/eval_topic_clf.py --report evals/topic/report.md
.venv/Scripts/python.exe scripts/classify_topics.py --dry-run
bash scripts/acceptance_ch10.sh
```

- [ ] **Step 7: 把 §13「已知局限」逐条落地到 `dev-notes/ch10.md`**

**八条一条都不许省**,尤其第 5、6 两条(转人工依赖模型调工具 / 错别字鲁棒性**没有任何测试证明**)。**收尾时不许把它们写成「全绿」。**

- [ ] **Step 8: 全量回归**

Run: `.venv/Scripts/python.exe -m pytest -m "not db"`
Expected: **PASS,且打印出通过数**(不信「没报错就是过了」)

- [ ] **Step 9: Commit**

```bash
git add scripts/acceptance_ch10.sh CLAUDE.md dev-notes/ch10.md
git commit -m "ch10-B T16: 验收 ①–③ + 章级文档同步 + 已知局限落地"
```

---

## 完成清单(收尾时逐条核对,不许跳)

- [ ] **CP-1**:`taxonomy_review.csv` 用户过目过(任务 1 Step 6)
- [ ] **CP-2**:`trainval.csv` 与 `test.csv` 用户改过并回收(任务 6 Step 6、任务 7 Step 6)
- [ ] 测试集 **120 条、真实 80 + 合成 40**,且 100% 人工过
- [ ] 评测报告有**三个口径**,且印了「只看真实 80 才是真机预期」
- [ ] 报告里 **`†` 标注**了 support <15 的类,并说清「该行 F1 不成结论」
- [ ] 报告里**与 F1 并排**印了训练集标签错误率(来自人工抽审的「改」比例)
- [ ] `models/topic-clf/` 真的**没进 git**(`git check-ignore` 验过)
- [ ] `transfer_to_human`(A 支)与 `topic_service` 都**不在** `app/agent/` 或 `app/api/chat.py` 的 import 里(源码扫描测试绿)
- [ ] `pytest -m "not db"` 全绿且**打印了通过数**
- [ ] `bash scripts/acceptance_ch10.sh` 的转录抄进了 `dev-notes/ch10.md`
- [ ] `dev-notes/ch10.md` **每个阶段都有记录**(不是收尾一次性补的)
- [ ] **spec §13 那八条已知局限**逐条落到了 dev-notes,**没有被写成「全绿」**
