# ch07 上下文管理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch01 那条单层裁剪换成三层上下文管理 —— 最近原文 / 中间截短 / 最远梗概,两个锚点划边界,中间层超预算时后台异步摘要,调模型的上下文按固定顺序拼装保住前缀缓存,上下文每轮可观测,前端加多会话侧栏。

**Architecture:** 历史按 `messages.id` 切成三段,边界只由 `conversations` 表上两个 `BIGINT` 锚点(`summary_upto_msg_id` / `layer1_from_msg_id`)决定,**降级只挪 id、不搬数据**。层 2 的 token **按截短后的版本计数** —— 这是让截短真正生效的开关。层 2 超预算时起一个**专用线程 + 自建 engine** 的后台任务(复刻 ch04 `app/kb/orchestrate.py`),把原文区间压成一段梗概追加进 `conversation_summaries`,成功后才推进锚点。LangChain 面**全部收在 `prompts.py`**,`memory/` 保持纯 `schemas.Message`。

**Tech Stack:** Python 3.13 / FastAPI 0.141 / LangGraph 1.2.11 / LangChain 1.4(+langchain-core 1.6.3)/ SQLAlchemy 2.0.53(async, asyncmy)/ MySQL 8.0 / tiktoken 0.14 / 原生 HTML+CSS+JS(无构建工具链)

**Spec:** `docs/superpowers/specs/2026-09-22-ecommerce-cs-ch07-context-design.md`
**读计划的人请同时读 spec** —— 计划写「怎么做」,spec 写「为什么这么定」,而本章几乎每一处「为什么」都对应一次真实故障。

## Global Constraints

以下每条都是**硬约束**,每个任务的要求都隐含包含它:

- **依赖方向单向**:`api → services → {tools, db, memory, prompts, llm}`。本章新增 `memory/budget.py` 收 `Settings` 与**已渲染的** `system_prompt` 作参数(不 import `prompts`,保住 `memory/` 的纯度)。
- **`app/prompts.py` 是消息组装的唯一出口,也是本章 LangChain 的唯一面**。`trim_messages` 只在这里调。
- **`memory/` 与 `services/history.py` 不依赖 LangChain**;`Message → BaseMessage` 只经 `prompts.py:to_lc_messages`。
- **`services/` 的函数接收 llm 实例作参数**;测试靠 `dependency_overrides` 替换。
- **错误语义边界**:`422` 只表示「请求/模型输出不合约定」;上游与基础设施故障一律 `502` + 固定文案。**所有出站错误文本**(SSE `error` 帧、`tool_result` 失败 `summary`、422/502 detail、日志里的异常摘要)必须过 `app/sanitize.py:redact_api_key`。
- **`ToolInfrastructureError` 必须一路上抛**,绝不回灌给模型 —— 数据库故障不能伪装成「你的订单号查不到」。
- **单测全程不联网**。`Settings(...)` 构造必须传 `_env_file=None`。db 测试读真实 `.env` 并打 `@pytest.mark.db`。异步测试用 `@pytest.mark.anyio`,不用 pytest-asyncio。
- **不要再往命令行加 `-q`**:`pytest.ini` 的 `addopts` 已有一个,叠加成 `-qq` 会**整行不打印 `N passed`**。
- **含中文的请求体不走 `curl` 的 argv**(MSYS2 按 CP936 重编码成 `error parsing the body`):一律 stdin heredoc 或 httpx。
- **脚本打印非 ASCII 要钉输出边界**:`sys.stdout.buffer.write(...encode("utf-8"))`,不要依赖控制台 codec。
- **本仓头号风险是「假绿测试」**。三条具体形态,每一条都在前六章真实发生过:
  1. **期望值恰好等于 fallback/default 会产生的结果**;
  2. **在处理之后注入** —— 把**已经被处理好的值**喂给被测对象,于是「处理」那一步永远不被验(ch06 一章中了三次);
  3. **变量名不等于语义** —— 断言一个字段前,先读它是怎么被赋值的(ch05 的 `agent_steps` 断错)。
  写断言前先问:**「如果实现改错了,这条的输出会不会不同?」**
- **`astream(stream_mode="custom")` 会吞掉 `interrupt()`**;**resume 时节点从头重跑**;**未在 `ChatState` 声明的通道写入被静默丢弃**(只 `logger.warning`)。三条都是本机实测事实。

---

## 文件结构

| 文件 | 新/改 | 职责 |
|---|---|---|
| `app/config.py` | 改 | ch07 的 12 个配置项;退役 3 个旧项 |
| `app/memory/budget.py` | 新 | 窗口 → 历史预算 → 层1/层2 推导;**启动自检** |
| `app/memory/layers.py` | 新 | 按锚点切三层 + 层 2 截短 + **降级循环**(纯) |
| `app/memory/summarize.py` | 新 | 摘要 prompt 组装 + 触发判定(纯)+ 原子落库 |
| `app/memory/tasks.py` | 新 | 后台摘要执行体(专用线程 + 自建 engine) |
| `app/memory/journal.py` | 新 | `model_ctx` / `history_ctx` 组装与原样落盘 |
| `app/memory/trim.py` | 改 | 删 `compute_available_tokens`;`_to_rounds` → 公开 `to_rounds` |
| `app/logging_setup.py` | 新 | `log/app.log` 落盘配置 |
| `app/schemas.py` | 改 | `Message.id` |
| `app/db/models.py` | 改 | `ConversationSummary` + `Conversation` 两列 + `MessageRecord` 不动 |
| `db/ch07.sql` | 新 | 新表 + 两个锚点列 |
| `app/services/history.py` | 改 | 写工具行、读梗概、推进锚点 |
| `app/services/chat.py` | 改 | `prepare_turn` 换血 |
| `app/prompts.py` | 改 | `build_context_messages` + token_counter 适配器 |
| `app/agent/state.py` | 改 | `messages`(add_messages)+ 三个通道 |
| `app/agent/nodes.py` | 改 | agent 读 `history`;`log_turn` 写 ReAct 往返 |
| `app/api/chat.py` | 改 | 播种 + 降级 + 起后台任务 + 400 |
| `app/api/conversations.py` | 新 | 两个只读端点 |
| `app/main.py` | 改 | `setup_logging` + 启动自检 |
| `app/static/index.html` | 改 | 会话侧栏(Vibe Coding) |
| `evals/summary_cases.jsonl` | 新 | 摘要标注样例 |
| `scripts/run_summary_eval.py` | 新 | 标注样例跑一遍 |
| `scripts/acceptance_ch07.sh` | 新 | 章级端到端验收 |

**任务依赖**:
```
T1 ─┬→ T2
    └→ T8 ─→ T9 ─┐
T3 ─→ T4 ─┐       │
          ├→ T6 ──┼→ T10 ─→ T11 ─→ T12
T5 ───────┘       │
T7 ───────────────┘
T8 ─→ T13
T10 → T13
```
(T3 = 数据层,给 T4 的 `Message.id` 铺路;T7 的 journal 给 T10 用)

---

### Task 1: 配置项 + 预算推导 + 启动自检

**Files:**
- Modify: `app/config.py`
- Modify: `app/tools/registry.py:40`
- Create: `app/memory/budget.py`
- Test: `tests/test_memory_budget.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: 无(本章第一个任务)
- Produces:
  - `app.config.Settings` 新增:`model_context_window` / `max_output_tokens` / `max_user_input_tokens` / `tool_result_max_tokens` / `rerank_top_k` / `keep_rounds` / `per_round_steady` / `layer2_assistant_chars` / `layer2_tool_chars` / `summary_max_chars` / `evidence_block_tokens` / `tool_def_tokens`
  - `app.memory.budget.ContextBudget`(frozen dataclass):`window` / `fixed_overhead` / `peak` / `history_budget` / `layer1_budget` / `layer2_budget` / `fits_one_round`
  - `app.memory.budget.derive(*, settings, system_prompt: str) -> ContextBudget`
  - `app.memory.budget.tokens_for_chars(chars: int) -> int`
  - `app.memory.budget.LAYER1_SHARE: float = 0.7`

- [ ] **Step 1: 写失败测试**

`tests/test_memory_budget.py`:

```python
"""ch07 预算推导。纯函数,不联网、不碰 db。"""

import pytest

from app.config import Settings
from app.memory import budget

#: 构造 Settings 必须传 `_env_file=None` —— 仓库根有真实 .env,
#: pydantic-settings 会自动读它,不传的话「缺字段应报错」那类断言会静默通过。
REQUIRED = dict(
    _env_file=None,
    openai_base_url="https://example.invalid/v1",
    openai_api_key="sk-test",
    openai_model="test-model",
    database_url="mysql+asyncmy://u:p@127.0.0.1:3306/x",
)

SYSTEM_PROMPT = "你是客服。" * 50  # 一段可计数的中文


def _settings(**over):
    return Settings(**{**REQUIRED, **over})


def test_demo_config_arithmetic_is_exact():
    """演示配置下把每个扣减项算清楚 —— 本章所有数值行为的地基。

    这条**不看某个魔数,看等式**:固定开销 + 单轮峰值 + 历史预算 == 窗口。
    等式成立与否,才区分得出「扣全了」和「漏扣了一项」。
    """
    s = _settings(
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        rerank_top_k=5,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
        keep_rounds=20,
        per_round_steady=600,
    )
    b = budget.derive(settings=s, system_prompt=SYSTEM_PROMPT)

    assert b.window == 18000
    assert b.peak == 3 * 1200 + 2000          # 单轮 ReAct 峰值 + 用户输入上限
    assert b.fixed_overhead + b.peak + b.history_budget == b.window
    assert b.layer1_budget + b.layer2_budget == b.history_budget
    # 窗口那一支赢:「想留住的轮数」算出来 20×600=12000 更大
    assert b.history_budget == b.window - b.fixed_overhead - b.peak


def test_rounds_branch_wins_when_window_is_generous():
    """两个数取小的那个 —— 两条分支都真的会赢,否则 min() 是装饰。"""
    s = _settings(model_context_window=200_000, keep_rounds=5, per_round_steady=100)
    b = budget.derive(settings=s, system_prompt=SYSTEM_PROMPT)
    assert b.history_budget == 500            # 5 × 100,而不是窗口那一支


def test_layer_split_uses_the_declared_share():
    s = _settings(model_context_window=18000)
    b = budget.derive(settings=s, system_prompt=SYSTEM_PROMPT)
    assert b.layer1_budget == int(b.history_budget * budget.LAYER1_SHARE)
    assert b.layer2_budget == b.history_budget - b.layer1_budget


def test_fits_one_round_is_false_when_peak_eats_the_window():
    """「连一轮都装不下」必须报出来 —— 自检的判据就是它。"""
    s = _settings(
        model_context_window=1024,
        max_output_tokens=800,
        tool_def_tokens=800,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
    )
    b = budget.derive(settings=s, system_prompt=SYSTEM_PROMPT)
    assert b.fits_one_round is False


def test_tokens_for_chars_uses_the_same_counter_as_everything_else():
    """中文按字数折 token 的口径必须与预算同源。

    另立一个「1 字 = 1.5 token」的系数,就会与 `trim.count_tokens`
    各说各话 —— 而两处都「看起来合理」。这条钉住它们同源。
    """
    from app.memory import trim

    assert budget.tokens_for_chars(200) == trim.count_tokens("中" * 200)
    assert budget.tokens_for_chars(0) == 0
```

`tests/test_config.py` 的两处改动:

```python
# 原本(约 164 行)
assert s.retrieval_top_k == 3
# 改为
assert s.rerank_top_k == 5

# 原本(约 190-191 行)
Settings(_env_file=None, **REQUIRED, retrieval_top_k=bad)
assert "retrieval_top_k" in str(exc.value)
# 改为
Settings(_env_file=None, **REQUIRED, rerank_top_k=bad)
assert "rerank_top_k" in str(exc.value)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_memory_budget.py tests/test_config.py`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.memory.budget'`,以及 `rerank_top_k` 不存在

- [ ] **Step 3: 改 `app/config.py`**

删掉 `retrieval_top_k` 那一行(72 行附近,连同它的注释块),`context_budget_tokens` / `reserved_output_tokens` **本任务先留着**(T2 退役它们,因为 `prepare_turn` 还在读)。
新增:

```python
    # ---- ch07:上下文管理。全部带界 —— 写错要在启动时炸,不能等运行时 ----
    #
    # `model_context_window` / `max_output_tokens` **取代**了本章之前的
    # `context_budget_tokens` / `reserved_output_tokens`:旧的两个数是「直接给一个
    # 预算」,新的口径是「从模型窗口倒推」,而两者同时存在就是两个含义重叠的旋钮
    # —— 本项目已经吃过这个亏(`reranker_use_fp16` 在 GPU 分支落地后无人读,已删)。
    model_context_window: int = Field(default=18000, ge=1024)
    max_output_tokens: int = Field(default=2000, ge=1)
    max_user_input_tokens: int = Field(default=2000, ge=1)
    # 单个工具结果的上限,同时也是「单轮 ReAct 峰值」的一项。
    # `max_agent_steps`(已有)与它相乘就是峰值。
    tool_result_max_tokens: int = Field(default=1200, ge=1)
    # 取代 `retrieval_top_k`(同一把旋钮)。**读点有两个,不是一个**:
    # `app/tools/registry.py:40` 与 `evals/run_retrieval_eval.py:193` ——
    # 只改前者会在跑评估脚本时 AttributeError,而 `pytest.ini` 的
    # `testpaths = tests` 不覆盖 `evals/`,所以**单测全绿也发现不了**。
    rerank_top_k: int = Field(default=5, ge=1)
    # 历史预算 = min(keep_rounds × per_round_steady, 窗口匀得出来的)。
    keep_rounds: int = Field(default=20, ge=1)
    #
    # ⚠️ `per_round_steady` 是**估算,不是实测** —— 与 `dedupe_threshold=0.95`
    # 同族:写下来但还没验证过,**不要当成已验证的**。
    # 它只要 < 演示配置下的 窗口/keep_rounds,「想留住的轮数」那一支就会胜出,
    # 历史预算会远小于窗口能匀出来的量,层 1 会**每轮都降级**。
    # 校准方法:跑一轮真实对话,从 `model_ctx` 日志读每轮实际占用,取中位数回填。
    per_round_steady: int = Field(default=600, ge=1)
    # 层 2 的截短:客服答复留几个字、工具结果留几个字。
    layer2_assistant_chars: int = Field(default=50, ge=1)
    layer2_tool_chars: int = Field(default=60, ge=1)
    # 梗概长度上限,**同时也是「注入梗概」这项固定开销的来源**。
    summary_max_chars: int = Field(default=200, ge=1)
    # 单块检索证据 / 五个工具定义渲染后的估算开销。同样是估算值。
    evidence_block_tokens: int = Field(default=250, ge=1)
    tool_def_tokens: int = Field(default=800, ge=1)
```

- [ ] **Step 4: 改 `app/tools/registry.py:40`**

```python
        top_k=settings.rerank_top_k,
```

- [ ] **Step 5: 写 `app/memory/budget.py`**

```python
"""ch07 上下文预算:从模型窗口倒推,不写死常量。

**纯计算,不做 IO、不依赖 LangChain** —— 转 BaseMessage 是 `prompts.to_lc_messages`
一处的职责(本仓贯穿性约定)。它因此可以被启动自检、端点、单测各调一次而零成本。

两个口径必须与别处**同源**,否则「只改一个、净效果是反的」:
1. token 一律走 `memory.trim.count_tokens`(tiktoken cl100k_base,对中文偏保守);
2. 中文「按字数折 token」不另立系数,直接用同一个 counter 折 —— 见 `tokens_for_chars`。
"""

from dataclasses import dataclass

from app.config import Settings
from app.memory import trim

#: 层 1 拿历史的七成、层 2 拿三成(spec §7.2)。
LAYER1_SHARE = 0.7


@dataclass(frozen=True)
class ContextBudget:
    """一次推导的全部结果。frozen:推导完就该是只读的。"""

    window: int
    fixed_overhead: int
    peak: int
    history_budget: int
    layer1_budget: int
    layer2_budget: int
    fits_one_round: bool


def tokens_for_chars(chars: int) -> int:
    """把「字数」上限折成 token 数。

    **不写系数**(如「1 字 = 1.5 token」),而是拿一段等长的中文过同一个
    counter —— 另立系数就是两把尺子,而它们会在某个字数区间上给出相反结论,
    且两边看起来都合理。`tests/test_memory_budget.py` 钉住这条同源关系。
    """
    if chars <= 0:
        return 0
    return trim.count_tokens("中" * chars)


def derive(*, settings: Settings, system_prompt: str) -> ContextBudget:
    """窗口 → 历史预算 → 层1/层2。

    `system_prompt` 由调用方渲染后传入(`prompts.render_system_prompt`),
    这样本模块**不必 import prompts**,保住 `memory/` 不依赖 LangChain 的约定。
    """
    fixed_overhead = (
        trim.count_tokens(system_prompt)
        + settings.tool_def_tokens
        + settings.rerank_top_k * settings.evidence_block_tokens
        + tokens_for_chars(settings.summary_max_chars)
        + settings.max_output_tokens
        + settings.safety_margin_tokens
    )
    peak = (
        settings.max_agent_steps * settings.tool_result_max_tokens
        + settings.max_user_input_tokens
    )

    by_window = settings.model_context_window - fixed_overhead - peak
    by_rounds = settings.keep_rounds * settings.per_round_steady
    history_budget = min(by_rounds, by_window)

    layer1_budget = int(history_budget * LAYER1_SHARE)
    return ContextBudget(
        window=settings.model_context_window,
        fixed_overhead=fixed_overhead,
        peak=peak,
        history_budget=history_budget,
        layer1_budget=layer1_budget,
        layer2_budget=history_budget - layer1_budget,
        fits_one_round=history_budget >= settings.per_round_steady,
    )
```

- [ ] **Step 6: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_memory_budget.py tests/test_config.py`
Expected: PASS

- [ ] **Step 7: 全量回归(确认没碰坏别处)**

Run: `.venv/Scripts/python.exe -m pytest -m "not db"`
Expected: PASS(此时 `context_budget_tokens` 仍在,`prepare_turn` 照常)

- [ ] **Step 8: 提交**

```bash
git add app/config.py app/tools/registry.py app/memory/budget.py tests/test_memory_budget.py tests/test_config.py
git commit -m "feat(ch07): 配置项 + 预算推导(窗口倒推)+ 同源的中文字数折 token"
```

---

### Task 2: `prepare_turn` 换血 —— 退役旧的三个预算配置

**Files:**
- Modify: `app/config.py`(删两项)、`app/memory/trim.py`(删 `compute_available_tokens`)、`app/services/chat.py`
- Test: `tests/test_trim.py`、`tests/test_chat_service.py`、`tests/test_api_chat.py`、`tests/test_config.py`

**Interfaces:**
- Consumes: `app.memory.budget.derive`(T1)
- Produces: `app.services.chat.prepare_turn(*, settings, history, user_input) -> list[Message]` **签名不变**,内部换实现

- [ ] **Step 1: 先看现状 —— 把要删的东西和它的读点数清楚**

Run:
```bash
grep -rn "context_budget_tokens\|reserved_output_tokens\|compute_available_tokens" app/ tests/
```
Expected(2026-09-22 实测):`app/config.py` 2 处、`app/memory/trim.py` 4 处、`app/services/chat.py` 4 处、`tests/test_trim.py` 4 处、`tests/test_chat_service.py` 3 处、`tests/test_api_chat.py` 6 处、`tests/test_config.py` 2 处。
**这条 Step 不是仪式**:漏掉任何一处的症状是 `AttributeError`,而它在 pytest 里看起来就像「配置项没定义」,不指向真正的原因。

- [ ] **Step 2: 改 `app/services/chat.py`**

整个 `prepare_turn` 换成:

```python
def prepare_turn(*, settings: Settings, history: Sequence[Message], user_input: str) -> list[Message]:
    """预算校验 + 历史裁剪。返回**裁剪后的历史**。

    ch07:预算从「直接给一个数」改成「从模型窗口倒推」(`memory.budget`)。
    本任务只换预算来源,**分层留到 T10** —— 此刻仍是单层
    `trim.select_history`,所以每一步都能单独跑绿。

    预算不足时抛 `ContextOverflowError`,调用方在响应开始前处理,
    因此能返回 400 而不是一个已经开始的 SSE 流。
    """
    b = budget.derive(
        settings=settings, system_prompt=render_system_prompt(settings.brand_name)
    )
    if b.history_budget < 0:
        raise trim.ContextOverflowError(used=b.fixed_overhead, budget=b.window)
    return trim.select_history(history, b.history_budget)
```

import 段相应改成 `from app.memory import budget, trim`。

- [ ] **Step 3: 删 `app/memory/trim.py::compute_available_tokens`**

连同它的 docstring 一起删。`count_tokens` / `select_history` / `_to_rounds` **保留**
(前者 T6 还要用,后两者 T4/T10 还要用)。`ContextOverflowError` 也保留在 `trim.py`。

- [ ] **Step 4: 删 `app/config.py` 的两个旧项**

```python
    context_budget_tokens: int = 8192
    reserved_output_tokens: int = 1024
```

`safety_margin_tokens` **保留**(它仍参与固定开销)。

- [ ] **Step 5: 逐个修测试文件**

`tests/test_trim.py`:删掉两条调用 `compute_available_tokens` 的用例(它们测的函数已不存在),
**但把它们覆盖的语义补成一条新的**:

```python
def test_budget_leaves_no_room_for_history_when_overhead_eats_the_window():
    """原来由 compute_available_tokens 直接覆盖的语义:预算可以是负的。

    删掉旧函数不等于删掉这条不变量 —— 它现在由 budget.derive 承担,
    而「负预算」正是端点返回 400 的判据,不能没有覆盖。
    """
    from app.config import Settings
    from app.memory import budget

    s = Settings(
        _env_file=None,
        openai_base_url="https://example.invalid/v1",
        openai_api_key="sk-test",
        openai_model="m",
        database_url="mysql+asyncmy://u:p@127.0.0.1:3306/x",
        model_context_window=1024,
        max_output_tokens=2000,
        tool_def_tokens=5000,
    )
    b = budget.derive(settings=s, system_prompt="你是客服。")
    assert b.history_budget < 0
```

`tests/test_chat_service.py` / `tests/test_api_chat.py`:`context_budget_tokens=…` /
`reserved_output_tokens=…` / `safety_margin_tokens=…` 三行换成对应的新项。
**注意语义要对等,不要机械替换**:原来 `context_budget_tokens=200` 是在造一个
「预算极小」的场景,换成 `model_context_window=` 时要倒推着给一个小窗口,让
`history_budget` 同样为负或极小。

`tests/test_config.py`:删掉两条对旧项的断言(约 24-26 行)。

- [ ] **Step 6: 跑测试**

Run: `.venv/Scripts/python.exe -m pytest -m "not db"`
Expected: PASS。若出现 `AttributeError: 'Settings' object has no attribute 'context_budget_tokens'`,回到 Step 1 的 grep 找漏改的调用点。

- [ ] **Step 7: 提交**

```bash
git add -u
git commit -m "refactor(ch07): prepare_turn 改用窗口倒推的预算,退役三个旧配置项"
```

---

### Task 3: 数据层 —— `Message.id` + 新表 + 两个锚点列

**Files:**
- Create: `db/ch07.sql`
- Modify: `app/db/models.py`、`app/schemas.py`、`app/services/history.py`
- Test: `tests/test_db_models.py`、`tests/test_history.py`、`tests/test_schemas.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `app.schemas.Message.id: int | None`(默认 `None`)
  - `app.db.models.Conversation.summary_upto_msg_id: int` / `layer1_from_msg_id: int`
  - `app.db.models.ConversationSummary`
  - `app.services.history.load_history` 现在**填充 `Message.id`**
  - `app.services.history.append_turn(*, session, conversation_id, messages) -> list[int]` 改为**返回新写入行的 id 列表**(顺序与入参一致)
  - `app.services.history.load_summaries(*, session, conversation_id) -> list[tuple[int, str]]`(seq, content)
  - `app.services.history.advance_anchors(*, session, conversation_id, summary_upto=None, layer1_from=None) -> None`
  - `app.services.history.append_summary_and_advance(*, session, conversation_id, upto_msg_id, content) -> None`(**原子**)

- [ ] **Step 1: 写建表 SQL**

`db/ch07.sql`:

```sql
-- ch07:上下文管理
--
-- 两个锚点必须**手写 ALTER**:scripts/init_db.py 跑的是
-- `Base.metadata.create_all`,它只建不存在的**表**,不改已有表 ——
-- 靠它加列会静默什么也不做,而代码里已经在读那两列。
--
-- ⚠️ **文件内的顺序是刻意的:ALTER 在前、CREATE 在后。**
-- mysql 客户端**遇到第一个错误就中止整个脚本**。反过来排的话,
-- 在「梗概表已存在、但两个锚点列还没加」的库上(先跑过 init_db.py 的
-- create_all 就是这情形),CREATE 会报 1050 中止 ⇒ **ALTER 永远不执行** ⇒
-- 库缺两列,而报错读起来像「表已存在 ⇒ 已经装好了」。
-- ALTER 放前面则三条路径(全新 / 老库升级 / 重跑)里前两条都能跑完。

-- 0 = 尚无梗概(不含任何消息)
-- 0 = 层 1 起于最早,层 2 为空
-- 不变量:0 <= summary_upto_msg_id <= layer1_from_msg_id
ALTER TABLE conversations
  ADD COLUMN summary_upto_msg_id BIGINT NOT NULL DEFAULT 0,
  ADD COLUMN layer1_from_msg_id  BIGINT NOT NULL DEFAULT 0;

-- **裸 CREATE TABLE,不加 IF NOT EXISTS** —— 与 ch03/04/06 一致。
-- 加了它的后果很具体:`scripts/init_db.py` 的 `create_all` 会先按 ORM 建表,
-- 那时这句**静默跳过**,而 ORM 若没声明 `uk_conv_seq`,唯一键就**永远不存在**
-- 且不报错(ORM 侧已同时声明,见 `app/db/models.py` 的 `__table_args__`)。
-- 裸语句再跑一次会响亮地报错,那是更诚实的失败 —— 前提是**它排在 ALTER 之后**。
CREATE TABLE conversation_summaries (
  id              BIGINT       NOT NULL AUTO_INCREMENT,
  conversation_id VARCHAR(32)  NOT NULL,
  seq             INT          NOT NULL COMMENT '第 N 段,从 1 起,只增不改',
  upto_msg_id     BIGINT       NOT NULL COMMENT '这一段覆盖到哪条 messages.id(含)',
  content         TEXT         NOT NULL COMMENT '梗概正文,几十到一两百字',
  created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  -- 并发保护的第二道:同一会话两个摘要任务同时提交时,后者撞唯一键
  -- ⇒ 失败 ⇒ 锚点不推进 ⇒ 下次重来。内存锁挡不住多进程,这个能。
  UNIQUE KEY uk_conv_seq (conversation_id, seq),
  KEY idx_conv (conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

- [ ] **Step 2: 写失败测试**

`tests/test_db_models.py` 追加(注意文件顶部已有 `pytestmark = pytest.mark.db`):

```python
# ---- ch07:conversation_summaries + conversations 两个锚点 ----

SCRATCH_SUMMARY_CONV = "ch07probe0000000000000000000000"


async def _cleanup_summary_rows() -> None:
    """**必须 commit** —— `async with session` 退出是 rollback,
    不 commit 的 DELETE 原地作废,探针行留库,下一轮的 `.one()` 撞
    MultipleResultsFound。(计划文本里的 DELETE 就没有 commit。)"""
    async with get_sessionmaker()() as session:
        await session.execute(
            text("DELETE FROM conversation_summaries WHERE conversation_id = :c"),
            {"c": SCRATCH_SUMMARY_CONV},
        )
        await session.execute(
            text("DELETE FROM conversations WHERE id = :c"), {"c": SCRATCH_SUMMARY_CONV}
        )
        await session.commit()


@pytest.mark.anyio
async def test_conversation_anchor_columns_default_to_zero():
    """两个锚点列存在、且有默认值 0 —— 这是「还没压过」的哨兵。"""
    await _cleanup_summary_rows()
    async with get_sessionmaker()() as session:
        session.add(Conversation(id=SCRATCH_SUMMARY_CONV, user="tester", status="active"))
        await session.commit()

    async with get_sessionmaker()() as session:      # 新 session 读回
        row = (
            await session.execute(
                select(Conversation).where(Conversation.id == SCRATCH_SUMMARY_CONV)
            )
        ).scalars().one()
        assert row.summary_upto_msg_id == 0
        assert row.layer1_from_msg_id == 0

    await _cleanup_summary_rows()


@pytest.mark.anyio
async def test_summary_rows_roundtrip_and_seq_is_unique_per_conversation():
    """中文往返 + `(conversation_id, seq)` 唯一键真的在拦人。

    唯一键这条**必须实测**:它是并发保护的第二道,而「我以为建了唯一键」
    与「真建了」在并发出问题之前完全没有区别。
    """
    from sqlalchemy.exc import IntegrityError

    await _cleanup_summary_rows()
    async with get_sessionmaker()() as session:
        session.add(Conversation(id=SCRATCH_SUMMARY_CONV, user="tester", status="active"))
        session.add(
            ConversationSummary(
                conversation_id=SCRATCH_SUMMARY_CONV, seq=1,
                upto_msg_id=12, content="用户问过订单 1002 能不能退,尚未解决。",
            )
        )
        await session.commit()

    async with get_sessionmaker()() as session:
        row = (
            await session.execute(
                select(ConversationSummary).where(
                    ConversationSummary.conversation_id == SCRATCH_SUMMARY_CONV
                )
            )
        ).scalars().one()
        assert row.content.startswith("用户问过订单 1002")   # 中文往返
        assert row.upto_msg_id == 12
        assert row.seq == 1

    async with get_sessionmaker()() as session:
        session.add(
            ConversationSummary(
                conversation_id=SCRATCH_SUMMARY_CONV, seq=1,
                upto_msg_id=99, content="重复的 seq",
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()

    await _cleanup_summary_rows()
```

`tests/test_history.py` 追加:

**⚠️ 本文件的用例是同步的**(`def test_x()` + 内部的 `asyncio_run(run())`),
不是 `@pytest.mark.anyio` —— 与本文件既有 8 条保持一致。新加的三条照这个形状写。
同时把文件顶部 import 补上 `ConversationSummary`、`append_summary_and_advance`、
`advance_anchors`、`load_summaries`;`_cleanup` fixture 里补一句删
`conversation_summaries` 的 DELETE(否则上一次的梗概行会污染下一次的 `max(seq)`)。

```python
def test_load_history_fills_message_ids_from_the_primary_key():
    """`Message.id` 必须由 `load_history` 填上 —— 分层完全依赖它。

    填不上的症状**不是报错**,而是所有消息在分层时被一视同仁:
    两个锚点的比较无从进行,层 2 与层 1 的边界退化成「全在层 1」,
    而每一轮看起来都完全正常、每一条断言都绿。
    """
    async def run():
        async with get_sessionmaker()() as session:
            await ensure_conversation(session=session, session_id=SCRATCH, user_id="u")
            ids = await append_turn(
                session=session, conversation_id=SCRATCH,
                messages=[Message(role="user", content="你好"),
                          Message(role="assistant", content="你好呀")],
            )
            loaded = await load_history(session=session, conversation_id=SCRATCH)
            # 逐条比对**真实主键**,不是 `is not None` ——
            # NOT NULL 列上 `is not None` 是不可能失败的断言,读了会误以为有覆盖。
            assert [m.id for m in loaded] == ids
            return loaded

    loaded = asyncio_run(run())
    assert [m.role for m in loaded] == ["user", "assistant"]
    assert all(m.id is not None and m.id > 0 for m in loaded)


def test_append_turn_returns_new_row_ids_in_order():
    """返回值必须与入参**同序** —— 调用方靠它把 ReAct 往返写回 state。

    「同序」是可区分的:把实现改成 `set(...)` 或倒序返回,下面的列表比较就会红,
    而「返回值非空」这类断言区分不出来。
    """
    async def run():
        async with get_sessionmaker()() as session:
            await ensure_conversation(session=session, session_id=SCRATCH, user_id="u")
            first = await append_turn(
                session=session, conversation_id=SCRATCH,
                messages=[Message(role="user", content="一")],
            )
            second = await append_turn(
                session=session, conversation_id=SCRATCH,
                messages=[Message(role="user", content="二"),
                          Message(role="assistant", content="三")],
            )
            return first, second

    first, second = asyncio_run(run())
    assert len(first) == 1
    assert len(second) == 2
    assert first[0] < second[0] < second[1]      # 自增且同序


def test_append_summary_and_advance_is_atomic():
    """落梗概与推进锚点要么都成、要么都不成。

    只成一半的两种后果都很难看:
    - 梗概落了锚点没推 ⇒ 同一段原文被**再压一遍**(重复梗概,每段单看都正常);
    - 锚点推了梗概没落 ⇒ 那段历史**永久消失**(区间已不在层 2 读取范围,
      而摘要表里没有替换物)。

    **两个方向都断言**:只断「成功时两者都在」区分不出「先落梗概再推锚点、
    中间抛了」的实现。所以再构造一次**撞唯一键**的失败提交,断言
    **锚点没有被推进**。
    """
    async def run():
        async with get_sessionmaker()() as session:
            await ensure_conversation(session=session, session_id=SCRATCH, user_id="u")

        # ① 成功路径:两步都在
        async with get_sessionmaker()() as session:
            await append_summary_and_advance(
                session=session, conversation_id=SCRATCH,
                upto_msg_id=7, content="用户报过订单 1002,尚未解决。",
            )
        async with get_sessionmaker()() as session:
            conv = (await session.execute(
                select(Conversation).where(Conversation.id == SCRATCH)
            )).scalars().one()
            rows = await load_summaries(session=session, conversation_id=SCRATCH)
            assert conv.summary_upto_msg_id == 7
            assert [c for _, c in rows] == ["用户报过订单 1002,尚未解决。"]

        # ② 失败路径:预置一条 seq=2,再让函数去撞它 ⇒ 整体回滚,锚点不动
        async with get_sessionmaker()() as session:
            session.add(ConversationSummary(
                conversation_id=SCRATCH, seq=2, upto_msg_id=99, content="占位",
            ))
            await session.commit()
        async with get_sessionmaker()() as session:
            with pytest.raises(IntegrityError):
                await append_summary_and_advance(
                    session=session, conversation_id=SCRATCH,
                    upto_msg_id=42, content="这条必须整条不生效",
                )

        async with get_sessionmaker()() as session:
            conv = (await session.execute(
                select(Conversation).where(Conversation.id == SCRATCH)
            )).scalars().one()
            rows = await load_summaries(session=session, conversation_id=SCRATCH)
            assert conv.summary_upto_msg_id == 7          # ← 没有被推到 42
            assert len(rows) == 1                          # ← 失败那条一行都没落
```

- [ ] **Step 3: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_db_models.py tests/test_history.py -m db`
Expected: FAIL —— `ImportError: cannot import name 'ConversationSummary'`,以及 `AttributeError: summary_upto_msg_id`

- [ ] **Step 4: 改 `app/schemas.py`**

给 `Message` 加一个字段(放在 `role` 之后):

```python
    #: MySQL `messages.id`。**本章起由 `load_history` 填充**。
    #:
    #: 为什么必须是它:两个锚点(`summary_upto_msg_id` / `layer1_from_msg_id`)
    #: 存的就是 MySQL 的主键,而分层是拿消息**逐条比对这两个 id** 做的 ——
    #: `Message` 没有 id 的话,分层根本无从下手。
    #:
    #: 顺带解掉另一个坑:`prompts.to_lc_messages` 用它做 LangChain 消息的 `id`。
    #: `add_messages` 是 **append-only**,无 id 的消息会被当场赋一个全新 uuid
    #: ⇒ 重新播种同一批消息会被**再追加一遍**,而每一轮的回复看起来都正常。
    #: 稳定 id 让重播种变成幂等。
    #:
    #: 默认 None:手工构造的消息(如 `log_turn` 里那两条)在落库前没有 id。
    id: int | None = None
```

- [ ] **Step 5: 改 `app/db/models.py`**

`Conversation` 加两列:

```python
    # ch07 两个锚点。**两侧默认值都要**:`default` 让 ORM 插入时补值,
    # `server_default` 让表本身有 DEFAULT(裸 SQL 省略也不至于 1364)——
    # 只留前者会让 create_all 建的表与 db/ch07.sql 建的表**形状不同**。
    summary_upto_msg_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default="0"
    )
    layer1_from_msg_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default="0"
    )
```

新增 ORM(`RefundRequest` 之后):

```python
class ConversationSummary(Base):
    """会话梗概(ch07,DDL: db/ch07.sql)。**只追加,不删除、不重写。**

    `seq` 从 1 起;`upto_msg_id` 是这一段覆盖到哪条 `messages.id`(含)。
    唯一键 `(conversation_id, seq)` 是并发保护的第二道。
    """

    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    upto_msg_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
```

(`Integer` 需要在 sqlalchemy import 里补上。)

- [ ] **Step 6: 改 `app/services/history.py`**

- `load_history`:构造 `Message` 时带上 `id=row.id`
- `append_turn`:返回新行 id 列表 —— 逐条 `session.add` 后 `await session.flush()`,
  收集 `[r.id for r in records]`,再 `commit()`
- 新增 `load_summaries` / `advance_anchors` / `append_summary_and_advance`

```python
async def append_summary_and_advance(
    *, session, conversation_id: str, upto_msg_id: int, content: str
) -> None:
    """**原子**落一段梗概并把 summary_upto_msg_id 推到 upto_msg_id。

    spec §6.1:只成一半的两种后果都很难看,所以这两步必须在同一个事务里提交。
    这就是为什么它们是**一个函数**而不是两个 —— 拆开就会出现「调用方忘了
    两个都调」或「两个都调了但中间抛了」。
    """
    next_seq = (
        await session.execute(
            select(func.coalesce(func.max(ConversationSummary.seq), 0)).where(
                ConversationSummary.conversation_id == conversation_id
            )
        )
    ).scalar_one() + 1

    session.add(
        ConversationSummary(
            conversation_id=conversation_id, seq=next_seq,
            upto_msg_id=upto_msg_id, content=content,
        )
    )
    await session.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(summary_upto_msg_id=upto_msg_id)
    )
    await session.commit()      # 一次提交,两步同时生效
```

- [ ] **Step 7: 建表并跑测试**

Run:
```bash
.venv/Scripts/python.exe scripts/init_db.py
.venv/Scripts/python.exe -m pytest tests/test_db_models.py tests/test_history.py tests/test_schemas.py -m db
```
Hmm —— `init_db.py` 只建**表**,不建列。两个锚点列要靠 `db/ch07.sql`:

```bash
docker exec -i mysql mysql -uroot -p"$MYSQL_ROOT_PASSWORD" mewhelp < db/ch07.sql
```
(实际口令以 `.env` 的 `DATABASE_URL` 为准;端口是 **3307**,不是 3306。)

Expected: PASS

- [ ] **Step 8: 提交**

```bash
git add db/ch07.sql app/db/models.py app/schemas.py app/services/history.py tests/test_db_models.py tests/test_history.py
git commit -m "feat(ch07): Message.id + conversation_summaries 表 + conversations 两个锚点列"
```

---

### Task 4: 分层与截短(纯函数)

> **⚠️ 本任务下面的示例代码有三处缺陷,已在实现时订正。**
> 发现并验证它们的是实现者(三处都做了变异运行)。**以代码为准,不要照抄下面的片段**:
> ① `_after` 的区间语义写反 ⇒ 两个锚点都是 0 时(每个新会话)同一条消息**同时落在两层**,
> 上下文静默翻倍;② `degrade` 把边界设成「被丢弃那一轮的最后一个 id」⇒
> 某些轮次形状下**死循环**(实测 `timeout 30` → exit=124);
> ③ 核心测试的第三条断言是 `x < x`,**不可能满足** —— 而它守护的正是
> 「层 2 按截短后计数」这条全章最容易静默失效的性质。
> 详见 spec §12.1。

**Files:**
- Modify: `app/memory/trim.py`(`_to_rounds` → 公开 `to_rounds`)
- Create: `app/memory/layers.py`
- Test: `tests/test_memory_layers.py`

**Interfaces:**
- Consumes: `app.schemas.Message.id`(T3)、`app.memory.budget`(T1)
- Produces:
  - `app.memory.trim.to_rounds(history) -> list[list[Message]]`(原 `_to_rounds` 改名公开)
  - `app.memory.layers.Layers`(frozen dataclass):`layer2: list[Message]` / `layer1: list[Message]` / `layer2_tokens: int` / `layer1_tokens: int`
  - `app.memory.layers.split(history, *, summary_upto_msg_id, layer1_from_msg_id, settings) -> Layers`
  - `app.memory.layers.degrade(history, *, summary_upto_msg_id, layer1_from_msg_id, layer1_budget, settings) -> int`(返回**新**的 `layer1_from_msg_id`)
  - `app.memory.layers.truncate(message, *, settings) -> Message`

- [ ] **Step 1: 写失败测试**

`tests/test_memory_layers.py`:

```python
"""ch07 分层与截短。纯函数 —— 不联网、不碰 db、不碰 LangChain。"""

import pytest

from app.config import Settings
from app.memory import layers
from app.schemas import Message

REQUIRED = dict(
    _env_file=None,
    openai_base_url="https://example.invalid/v1",
    openai_api_key="sk-test",
    openai_model="m",
    database_url="mysql+asyncmy://u:p@127.0.0.1:3306/x",
)


def _settings(**over):
    return Settings(**{**REQUIRED, **over})


def _history():
    """三轮,id 1..9。第 2 轮带一次工具调用(assistant + tool 两条)。"""
    return [
        Message(id=1, role="user", content="你好"),
        Message(id=2, role="assistant", content="你好呀,有什么可以帮您的吗?"),
        Message(id=3, role="user", content="订单 1002 能退吗"),
        Message(id=4, role="assistant", content="", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1002"},
             "id": "call_1", "type": "tool_call"}
        ]),
        Message(id=5, role="tool", content='{"order_no":"1002","status":"已取消"}',
                tool_call_id="call_1"),
        Message(id=6, role="assistant", content="您的订单 1002 当前状态为已取消,可以申请退款。"),
        Message(id=7, role="user", content="那就退吧"),
        Message(id=8, role="assistant", content="好的,已经为您登记。"),
        Message(id=9, role="user", content="多久到账"),
    ]


def test_split_puts_three_segments_end_to_end_without_gap_or_overlap():
    """三段不重不漏 —— off-by-one 会**静默丢消息**或**重复注入**。"""
    s = _settings()
    got = layers.split(
        _history(), summary_upto_msg_id=2, layer1_from_msg_id=7, settings=s
    )
    assert [m.id for m in got.layer2] == [3, 4, 5, 6]
    assert [m.id for m in got.layer1] == [7, 8, 9]
    # 1、2 已被梗概覆盖,两层都不含它们
    assert all(m.id not in (1, 2) for m in got.layer2 + got.layer1)


def test_split_with_zero_anchors_puts_everything_in_layer1():
    s = _settings()
    got = layers.split(
        _history(), summary_upto_msg_id=0, layer1_from_msg_id=0, settings=s
    )
    assert [m.id for m in got.layer1] == list(range(1, 10))
    assert got.layer2 == []


def test_layer2_tokens_are_counted_on_the_truncated_form():
    """**本任务的核心断言**:层 2 的计数必须按截短后。

    按原文数的话,截短就退化成纯渲染装饰 —— 层 2 该何时触发摘要还是何时触发,
    而**所有输出看起来都正常**。这条用「截短前后计数必须不同」把它区分开。
    """
    s = _settings(layer2_assistant_chars=5, layer2_tool_chars=5)
    raw = layers.split(_history(), summary_upto_msg_id=0, layer1_from_msg_id=7, settings=s)
    assert raw.layer2_tokens < sum(
        len(m.content) for m in raw.layer2
    ) + 1  # 便宜的存在性护栏,真正的判别在下一句
    # 截短确实发生了:助手回复被截到 5 字
    assert any(m.content.endswith("…") for m in raw.layer2 if m.role == "assistant")
    # 而同样一段历史,若按原文数会显著更大
    from app.memory import trim
    raw_tokens = sum(trim.count_tokens(m.content) for m in raw.layer2)
    assert raw.layer2_tokens < raw_tokens


def test_truncate_leaves_user_content_untouched():
    s = _settings(layer2_assistant_chars=3)
    msg = Message(id=1, role="user", content="这是一句很长的用户原话,一个字都不该动")
    assert layers.truncate(msg, settings=s).content == msg.content


def test_truncate_keeps_tool_calls_intact_for_pairing():
    """只截 `tool` 的 content,**不动 `tool_calls`**。

    截断 `tool_calls` 就是把 assistant 与它的 tool 消息拆开 ⇒ 上游 400,
    而且**只在历史长到触发分层时才复现**。
    """
    s = _settings(layer2_assistant_chars=3, layer2_tool_chars=3)
    a = Message(id=4, role="assistant", content="很长的一句解释" * 10, tool_calls=[
        {"name": "query_order", "args": {"order_id": "1002"}, "id": "call_1",
         "type": "tool_call"}
    ])
    out = layers.truncate(a, settings=s)
    assert out.tool_calls == a.tool_calls
    t = Message(id=5, role="tool", content="x" * 500, tool_call_id="call_1")
    assert layers.truncate(t, settings=s).tool_call_id == "call_1"
    assert len(layers.truncate(t, settings=s).content) < 500


def test_degrade_moves_the_boundary_forward_only_and_lands_on_a_round_start():
    """降级只把 layer1_from 往后挪,且必须落在**轮的起点**(user 消息)。

    不落在 user 上就会把一轮切开,连带把 tool 与它的 assistant 拆到两层里。
    """
    h = _history()
    s = _settings()
    new_from = layers.degrade(
        h, summary_upto_msg_id=0, layer1_from_msg_id=0, layer1_budget=1, settings=s
    )
    assert new_from >= 0
    if new_from:
        moved = next(m for m in h if m.id == new_from)
        assert moved.role == "user"


def test_degrade_returns_the_same_value_when_already_within_budget():
    """装得下就不动 —— 「压缩是成本不是美德」在纯函数层也要成立。"""
    h = _history()
    s = _settings()
    assert layers.degrade(
        h, summary_upto_msg_id=0, layer1_from_msg_id=0,
        layer1_budget=10 ** 6, settings=s,
    ) == 0


def test_degrade_never_crosses_below_summary_upto():
    """`layer1_from >= summary_upto` 是不变量,降级也必须守。"""
    h = _history()
    s = _settings()
    assert layers.degrade(
        h, summary_upto_msg_id=6, layer1_from_msg_id=7,
        layer1_budget=0, settings=s,
    ) >= 6


def test_degrade_loops_until_it_converges():
    """降级是**循环到收敛**,不是一次判断 —— 挪一次会同时改变两层的大小。"""
    h = _history()
    s = _settings()
    tight = _settings(layer2_assistant_chars=1000, layer2_tool_chars=1000)
    loose = layers.degrade(h, summary_upto_msg_id=0, layer1_from_msg_id=0,
                           layer1_budget=10 ** 6, settings=tight)
    tight_from = layers.degrade(h, summary_upto_msg_id=0, layer1_from_msg_id=0,
                                layer1_budget=1, settings=tight)
    assert tight_from >= loose
    assert layers.split(h, summary_upto_msg_id=0, layer1_from_msg_id=tight_from,
                        settings=tight).layer1_tokens <= 1 or tight_from == 9
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_memory_layers.py`
Expected: FAIL —— `ModuleNotFoundError: No module named 'app.memory.layers'`

- [ ] **Step 3: 把 `trim._to_rounds` 改名成公开的 `to_rounds`**

Run: `grep -rn "_to_rounds" app/ tests/`
把 `app/memory/trim.py` 里的定义与调用点、以及测试里的引用一并改。
**保留原 docstring** —— 那段「按 user 边界切轮以免切开 tool 配对」是 ch01 用一次
偶发 400 换来的,ch07 起它才第一次真正进入真链路。

- [ ] **Step 4: 写 `app/memory/layers.py`**

```python
"""ch07 三层切分与层 2 截短。**纯函数,不依赖 LangChain。**

三层与两个锚点见 spec §3.1:锚点是 `messages.id`,不变量是
`0 <= summary_upto_msg_id <= layer1_from_msg_id`,两个动作都**只挪 id、不搬数据**。
"""

from dataclasses import dataclass, field

from app.config import Settings
from app.memory import trim
from app.schemas import Message

#: 截短后追加的省略号。用 `…` 而不是 `...`:它是验收 4b 做**形态匹配**的锚,
#: 而三个点会与正文里本来就有的句号混淆。
ELLIPSIS = "…"


@dataclass(frozen=True)
class Layers:
    layer2: list[Message] = field(default_factory=list)
    layer1: list[Message] = field(default_factory=list)
    #: **截短后**的 token 数 —— 触发摘要看的就是它(spec §3.2)。
    layer2_tokens: int = 0
    #: 原文 token 数 —— 触发降级看的是它。
    layer1_tokens: int = 0


def _after(history, *, lo: int, hi: int) -> list[Message]:
    """取 `(lo, hi]` 区间的消息,按原序。

    `hi == 0` 表示「到末尾」。`id is None` 的消息(手工构造、尚未落库)
    **一律算在区间内** —— 把它们排除掉会静默丢消息,而它们恰恰是本轮新产生的。
    """
    out = []
    for m in history:
        if m.id is None:
            out.append(m)
            continue
        if m.id > lo and (hi == 0 or m.id <= hi):
            out.append(m)
    return out


def truncate(message: Message, *, settings: Settings) -> Message:
    """按角色截短一条消息。

    **只截 `content`,不动结构字段**(`tool_calls` / `tool_call_id`)——
    截断 `tool_calls` 就是把 assistant 与它的 tool 消息拆开 ⇒ 上游 400,
    且只在历史长到触发分层时才复现。
    """
    if message.role == "user":
        return message                      # 原话一个字不动
    if message.role == "tool":
        limit = settings.layer2_tool_chars
        prefix = "[工具结果] "
    else:
        limit = settings.layer2_assistant_chars
        prefix = ""
    if len(message.content) <= limit:
        return message
    return message.model_copy(
        update={"content": f"{prefix}{message.content[:limit]}{ELLIPSIS}"}
    )


def split(
    history, *, summary_upto_msg_id: int, layer1_from_msg_id: int, settings: Settings
) -> Layers:
    """切三层。**层 2 的 token 按截短后的版本数** —— 见模块 docstring 与 spec §3.2。"""
    layer2_raw = _after(history, lo=summary_upto_msg_id, hi=layer1_from_msg_id)
    layer1 = _after(history, lo=layer1_from_msg_id, hi=0)
    layer2 = [truncate(m, settings=settings) for m in layer2_raw]
    return Layers(
        layer2=layer2,
        layer1=layer1,
        layer2_tokens=sum(trim.count_tokens(m.content) for m in layer2),
        layer1_tokens=sum(trim.count_tokens(m.content) for m in layer1),
    )


def degrade(
    history, *, summary_upto_msg_id: int, layer1_from_msg_id: int,
    layer1_budget: int, settings: Settings,
) -> int:
    """层 1 超预算就把边界往后挪,**循环到收敛**,返回新的 `layer1_from_msg_id`。

    为什么是循环不是一次判断:挪一次会同时改变两层的大小(层 1 变小、层 2 变大),
    而层 2 变大**不会**反过来影响层 1 —— 但它决定了摘要的触发,所以收敛后才算数。

    边界**只落在轮的起点**(user 消息)上:否则会把一轮切开,连带把 tool 与它的
    assistant 拆到两层里(`trim.to_rounds` 的既有理由,ch01 的 400 就是它)。
    """
    cur = layer1_from_msg_id
    while True:
        got = split(
            history, summary_upto_msg_id=summary_upto_msg_id,
            layer1_from_msg_id=cur, settings=settings,
        )
        if got.layer1_tokens <= layer1_budget:
            return cur
        # 找到层 1 里**最旧那一轮**的起点,把它踢出去
        rounds = trim.to_rounds(got.layer1)
        if len(rounds) <= 1:
            return cur      # 只剩一轮还超:再挪就空了,停在原地由调用方决定
        dropped = rounds[0]
        last_dropped = next((m for m in reversed(dropped) if m.id is not None), None)
        if last_dropped is None:
            return cur
        cur = last_dropped.id
        if cur <= summary_upto_msg_id:
            return summary_upto_msg_id      # 守住不变量
```

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_memory_layers.py tests/test_trim.py`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add app/memory/layers.py app/memory/trim.py tests/test_memory_layers.py
git commit -m "feat(ch07): 三层切分与层 2 截短(层 2 计数按截短后)+ 降级循环"
```

---

### Task 5: `ChatState.messages` 通道 + 工具结果落表 + 播种

**Files:**
- Modify: `app/agent/state.py`、`app/agent/nodes.py`、`app/prompts.py`、`app/api/chat.py`
- Test: `tests/test_agent_node.py`、`tests/test_api_chat.py`、`tests/test_agent_graph.py`

**Interfaces:**
- Consumes: `app.schemas.Message.id`(T3)、`history.append_turn` 的新返回值(T3)
- Produces:
  - `ChatState.messages: Annotated[list[AnyMessage], add_messages]`
  - `app.prompts.to_lc_messages` 现在**带上稳定 id**(`str(message.id)`,None 则不带)
  - `app.agent.nodes.make_agent_node` 返回里新增 `"messages": [...]`(本轮 ReAct 往返)
  - `app.agent.nodes.make_log_turn_node` 把 ReAct 往返一并落库

- [ ] **Step 1: 写失败测试**

`tests/test_agent_node.py` 追加:

新增两条(本文件的既有形状:`@pytest.mark.anyio` + `_node(...)` + `_state(...)` +
`ScriptedModel`;别自己另起一套):

```python
@pytest.mark.anyio
async def test_agent_node_returns_its_react_exchange_in_messages():
    """agent 返回的 `messages` 必须**含** assistant(tool_calls)、tool、
    以及收尾那条回复 —— 三条,一条都不能少。

    少 tool 那条 ⇒ 工具往返不落库 ⇒ 下一轮的层 2 里没有「大块工具结果」可截,
    而本章正是为它设计的。少收尾那条 ⇒ 完整历史里**没有客服说过的话**。
    """
    tool = FakeTool(name="query_order", content='{"status": "已发货"}')
    model = ScriptedModel([
        [FakeChunk("", tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "call_1"}
        ])],
        [FakeChunk("您的订单已发货。")],
    ])
    out = await _node(model, tools=[tool], registry={"query_order": tool}).__call__(_state())

    assert [type(m).__name__ for m in out["messages"]] == [
        "AIMessage", "ToolMessage", "AIMessage"
    ]
    assert out["messages"][1].tool_call_id == "call_1"       # 配对没断
    assert out["messages"][2].content == "您的订单已发货。"   # 收尾回复在里面


@pytest.mark.anyio
async def test_log_turn_persists_the_react_exchange_including_tool_rows(monkeypatch):
    """落库的历史必须含 `role='tool'` 的行。

    改动前 production **零处**写这种行(grep 过全仓,只有 `test_history.py`
    为验往返写过),所以这条断言是本仓库第一次真的要求它存在。
    """
    captured: list[Message] = []

    async def _capture(*, session, conversation_id, messages):
        captured.extend(messages)
        return list(range(1, len(messages) + 1))

    monkeypatch.setattr(nodes, "append_turn", _capture)
    node = make_log_turn_node(session=object(), emit=lambda p: None)
    await node({
        "conversation_id": "c1",
        "user_input": "订单 1001 发货了吗",
        "reply": "已发货。",
        "messages": [
            AIMessage(content="", tool_calls=[{
                "name": "query_order", "args": {"order_id": "1001"},
                "id": "call_1", "type": "tool_call",
            }]),
            ToolMessage(content='{"status": "已发货"}', tool_call_id="call_1"),
            AIMessage(content="已发货。"),
        ],
    })

    assert [m.role for m in captured] == ["user", "assistant", "tool", "assistant"]
    tool_row = next(m for m in captured if m.role == "tool")
    assert tool_row.tool_call_id == "call_1"        # schemas.Message 的护栏要求它
    assert "已发货" in tool_row.content
    # **只写本轮**:播种进来的历史已经在库里了,写重了就是历史翻倍,
    # 而每一轮的回复看起来都正常。
    assert len(captured) == 4
```

(文件顶部 import 补 `AIMessage` / `ToolMessage` / `nodes` / `make_log_turn_node` /
`from app.schemas import Message`。)`test_api_chat.py` 追加两条,注意**本文件是同步的**:

```python
def test_second_turn_on_the_same_session_does_not_duplicate_history(client_factory):
    """**播种幂等** —— `add_messages` append-only 那个坑的落点。

    每轮无条件播种 ⇒ 重新构造的消息没有 id ⇒ 当场被赋全新 uuid ⇒
    整段历史被**再追加一遍**。第三轮时历史是三份,**而每一轮的回复看起来都正常**。

    观测点选**模型实际收到的消息条数**,因为那是唯一能把它区分开的地方 ——
    帧、落库、HTTP 状态码在两种实现下**完全一样**。
    """
    client, model = client_factory(batches=[[FakeChunk("好")], [FakeChunk("的")]])
    with client as c:
        sid = _parse_sse(c.post("/api/chat/stream", json={"message": "第一句"}).text)
        sid = [d for e, d in sid if e == "meta"][0]["session_id"]
        c.post("/api/chat/stream", json={"session_id": sid, "message": "第二句"})

    # 第二轮模型收到的 human 消息:上一轮 1 条 + 本轮 1 条 = 2。
    # 重复播种会让上一轮那条()被再追加一次 ⇒ 3 条。
    humans = [m for _, msgs in model.calls for m in msgs
              if type(m).__name__ == "HumanMessage"]
    assert len(humans) == 2
```

**⚠️ 替身要先扩**:`FakeSession` 现在只认 `Conversation` 与 `MessageRecord` 两个实体,
而 T10 的端点会发 `update(...)` 语句、T3 的锚点列也在 `Conversation` 上。
先按下面这段扩 `FakeSession`(**改替身**是本任务的一部分,不是意外):

```python
    async def execute(self, stmt, *args, **kwargs):
        # 新增:锚点推进用的 UPDATE。**必须真的改内存里的那条 Conversation** ——
        # 让它变成 no-op 的替身会让「降级真的持久化了吗」恒真。
        if isinstance(stmt, Update):
            value = stmt.whereclause.right.value
            conversation = self.conversations[value]
            for col, val in stmt._values.items():
                setattr(conversation, col.key, val)
            return _Result([])
        entity = stmt.column_descriptions[0]["entity"]
        # ↓↓↓ 从这里往下**原样保留**(Conversation 与 MessageRecord 两个分支),
        #     一行都不用改 —— 上面插入的 Update 分支是唯一的新增。
```

**并且加一条替身自检**:`_where_value` 现在对不支持的查询形态**直接抛**,
这个设计要保住 —— 扩 `update` 分支时**不要**顺手放宽它,否则「降级有没有写对行」
就再也观测不到了。

另外,`to_lc_messages` 的稳定 id **用一条纯单测钉**,不要绕端点(端点上它被 SSE
和 checkpointer 单例裹住,而这条逻辑本身是纯的):

```python
def test_to_lc_messages_uses_the_mysql_id_as_the_langchain_id():
    """稳定 id 是「重播种幂等」的前提。

    没有它,`add_messages` 会给每条重新构造的消息赋一个新 uuid,
    于是「重新播种同一批消息」= 再追加一遍。
    """
    from app.prompts import to_lc_messages

    msgs = to_lc_messages([
        Message(id=7, role="user", content="你好"),
        Message(id=8, role="assistant", content="你好呀"),
        Message(role="assistant", content="还没有 id"),      # 手工构造的
    ])
    assert [m.id for m in msgs] == ["7", "8", None]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_node.py tests/test_api_chat.py`
Expected: FAIL

- [ ] **Step 3: 改 `app/prompts.py::to_lc_messages`**

给每条转换出来的消息带上 `id`:

```python
def to_lc_messages(history: Sequence[Message]) -> list:
    """把纯数据 Message 转成 LangChain 消息。本模块是唯一的转换点。

    ch07:`id` 用 **MySQL 主键**做稳定 id(`str(message.id)`;None 则不带)。
    这不是修饰 —— `add_messages` 是 append-only,无 id 的消息会被当场赋一个新
    uuid,于是「重新播种同一批消息」= **再追加一遍**。稳定 id 让重播种幂等。
    """
    converted = []
    for message in history:
        lc_id = str(message.id) if message.id is not None else None
        if message.role == "user":
            converted.append(HumanMessage(message.content, id=lc_id))
        elif message.role == "tool":
            converted.append(
                ToolMessage(
                    content=message.content,
                    tool_call_id=message.tool_call_id or "",
                    id=lc_id,
                )
            )
        else:
            converted.append(
                AIMessage(
                    content=message.content,
                    tool_calls=message.tool_calls or [],
                    id=lc_id,
                )
            )
    return converted
```

- [ ] **Step 4: 改 `app/agent/state.py`**

文件顶部**新增两行 import**(其余 import 原样不动):

```python
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
```

在 `ChatState` 里**新增**这一段(放在「输入」段之后、`conversation_id` 附近):

```python
    # ---- 完整历史(跨轮,ch07)----
    # add_messages 是 **append-only** reducer:各节点只管吐**新**消息,框架按序并入。
    # ⚠️ 它**不进** `resolve_references` 的逐轮重置清单 —— 把 messages 加进那份
    # 清单等于**每轮清空完整历史**,而单轮测试完全看不出来。
    messages: Annotated[list[AnyMessage], add_messages]

    #: 注入给模型的**梗概全文**(逐轮覆写,由端点播种)。空串 = 还没有梗概。
    #: 与 `history` 同族:覆写语义 + 每轮重新播种,所以**进的是 `stream_input`**,
    #: 不进 `resolve_references` 的重置清单。
    summary_text: str

    #: **本轮**新产生的消息 —— `log_turn` 落库的唯一依据。
    #:
    #: 为什么不直接用 `messages`:那个是**累积**通道(全量),拿它落库 = 每轮把
    #: 整段历史再写一遍 ⇒ **历史翻倍**,而每一轮的回复看起来都正常。
    #: 这个是**覆写**通道,各节点写它时连同 `messages` 一起写(两个都写)。
    turn_messages: list[AnyMessage]
```

**同时改 `app/agent/nodes.py::make_resolve_references_node` 的逐轮重置清单**,
加一行 `"turn_messages": []`:

```python
            "tool_calls_made": [],
            # 覆写通道,不重置的话「这轮没产生消息」的路径(闲聊/兜底)会继承
            # 上一轮的值 ⇒ 上一轮的消息被再写一遍。
            "turn_messages": [],
            "order_no": "",
```

(这是 ch05–ch06「**通道与它的清零必须同处一地**」的第三次应用。)

**同一个节点还要多做一件事:把本轮的 user 消息吐进 `messages`。**

漏了它的后果**完全静默**:`messages` 里只有播种进来那一段历史 + 本轮新增的
assistant/tool,**而每一轮的用户原话都不进去** ⇒ 从第二轮起 `messages` 与 MySQL 分叉,
spec §7.4 承诺的读边(「端点用快照里的 `messages` 派生精简版」)**兑现不了**,
派生出来的精简版会**缺掉每一轮的提问** —— 而每轮回复正常、落库正常、现有断言全绿。

```python
        return {
            "resolved_input": resolved,
            # 本轮的用户原话进完整历史。
            # **只在这里加**:它是每轮第一个执行节点(START 的唯一出边),
            # 而 resume 路径**不会重跑它**(图从挂起的那个节点继续),
            # 所以续跑不会凭空多出一条 user 消息 —— 而那正是想要的:
            # 续跑续的是**同一轮**,不是新的一轮。
            "messages": [HumanMessage(user_input)],
            "trace": ["resolve_references"],
            ...(其余通道原样)
        }
```

**不要**改成在端点里塞进 `stream_input`:需求 5 的原话是「**各节点只管吐新消息**、
框架自动按顺序并入」,这正是那个形状;放进端点会让「谁负责往里加」有两处答案。

- [ ] **Step 5: 改 `app/agent/nodes.py::make_agent_node`**

`msgs` 的组装改为**从 state 取**(`state.get("history")` 仍是精简版,见 T10;
本任务先保留现状),并在返回里带上本轮新增的 ReAct 消息:

```python
        # 本轮新产生的消息(不含播种进来的历史):tool 往返 + 最终回复。
        # 交给 add_messages 并入 state,再由 log_turn 落库。
        new_messages: list = []      # ← 本任务新增:本轮产生的、要并入 state 的消息

        for step in range(1, settings.max_agent_steps + 1):
            steps = step
            acc, used = await _stream_round(bound, msgs, parts)
            usage_total += used
            tool_calls = list(getattr(acc, "tool_calls", None) or [])

            if not tool_calls:
                needs_final = False
                msgs.append(acc)
                new_messages.append(acc)     # ← 这一轮的输出就是最终回复,收下
                break

            needs_final = True
            msgs.append(acc)
            new_messages.append(acc)         # ← 带 tool_calls 的 assistant
            for call in tool_calls:
                emit({"frame": "tool_call", "name": call["name"],
                      "args": call["args"], "tool_call_id": call["id"]})
                outcome = await execute_tool(
                    tool_call=call, registry=registry, settings=settings
                )
                emit({"frame": "tool_result", "tool_call_id": outcome.tool_call_id,
                      "ok": outcome.ok, "summary": outcome.summary})
                tool_msg = ToolMessage(content=outcome.content, tool_call_id=call["id"])
                msgs.append(tool_msg)
                new_messages.append(tool_msg)      # ← 工具结果,层 2 要截的就是它
                made.append({"name": call["name"], "ok": outcome.ok})
                trace.append(f"agent:step{step} tool={call['name']}")

            if usage_total > settings.agent_token_budget:
                break

        if needs_final:
            # 收尾:不绑 tools。这一轮的输出**不在上面任何一条消息里**,
            # 必须自己收下来,否则下一轮的完整历史里**没有客服说过的话**。
            final_acc, used = await _stream_round(model, msgs, parts)
            usage_total += used
            new_messages.append(
                final_acc if final_acc is not None else AIMessage(content="".join(parts))
            )

        trace.append("agent:converged")
        return {
            "reply": "".join(parts),
            "messages": new_messages,        # → add_messages(累积进完整历史)
            "turn_messages": new_messages,   # → log_turn 落库(只写本轮)
            "agent_steps": steps,
            "tool_calls_made": made,
            "usage": {"total_tokens": usage_total},
            "trace": trace,
        }
```

**两个键值相同、但语义不同,别只写一个**:`messages` 进 `add_messages` 累积,
`turn_messages` 是逐轮覆写、供落库。只写前者 ⇒ 落库恒空 / 写重;
只写后者 ⇒ 完整历史里没有本轮。

**⚠️ 不要**写成 `"messages": [*new_messages, AIMessage(content=reply)]` ——
无工具那一轮 `acc` **已经**在 `new_messages` 里了,再补一条就是**同一句回复出现两次**,
而它只在「模型一轮直接答完」时发生(带工具的轮次不会),很容易漏测。

**注意**:收尾那一轮(len 模型输出)也要作为新消息带上,否则下一轮的完整历史里
**没有客服说过的话**。写的时候把收尾轮的 `AIMessage(content=reply)` 一并 append。

- [ ] **Step 6: 改 `app/agent/nodes.py::make_log_turn_node`**

`append_turn` 改为把 ReAct 往返一并写入:

**⚠️ 先做一件容易被漏掉的事:让四个固定话术出口也写 `turn_messages`。**

`log_turn` 落库改成只读 `turn_messages` 之后,凡是**不写这个通道**的节点,
它的那轮就**只落用户那一句、客服回复永远不落库** —— 而每一轮看起来都正常。
所以下面四个节点必须各补一行:

```python
# chitchat_reply / complaint_reply / fallback_reply:
return {
    "reply": CHITCHAT_REPLY,
    "messages": [AIMessage(content=CHITCHAT_REPLY)],
    "turn_messages": [AIMessage(content=CHITCHAT_REPLY)],
    "choices": [], "trace": ["chitchat_reply"],
}
```

退款子流程的两个出口(`refund_offer` / `refund_explain`)同理 —— 它们各自
产出一句话术,照同一个形状补。

**不要**在 `log_turn` 里写「`turn_messages` 为空就退回用 `reply`」的兜底分支:
那会让「某个节点忘了写」**静默退化成看起来正常的旧行为**,
而本仓的记性里,这类兜底最后都变成了缺陷的藏身处。宁可让漏写的那轮
落出一条空的 assistant 行(现有那几条「落库是 user+assistant 两条」的
测试会当场变红),也不要它自己悄悄补上。

在 `log_turn` 里加一个模块级转换函数(**不**复用 `to_lc_messages` —— 那是
反方向的、且依赖 LangChain;`memory/` 与 `services/` 不依赖 LC 是既有约定):

```python
def _lc_to_records(messages) -> list[Message]:
    """把本轮 ReAct 的 LangChain 消息转成落库用的纯数据 `Message`。

    - `AIMessage` → `role="assistant"`,带 `tool_calls`(可能为空)
    - `ToolMessage` → `role="tool"`,带 `tool_call_id`
    """
    out: list[Message] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            out.append(Message(role="tool", content=m.content,
                               tool_call_id=m.tool_call_id))
        else:
            out.append(Message(role="assistant", content=m.content or "",
                               tool_calls=m.tool_calls or None))
    return out
```

然后 `log_turn` 里的落库改成:

```python
        await append_turn(
            session=session,
            conversation_id=state["conversation_id"],
            messages=[
                Message(role="user", content=state["user_input"]),
                *_lc_to_records(state.get("turn_messages") or []),
            ],
        )
```

**这里必须用 `turn_messages`,不能用 `messages`** —— 这条是本章最容易写错、
且**写错了完全看不出来**的地方:

- `state["messages"]` 是 `add_messages` 通道,**累积的是全量**(播种进来的历史
  + 本轮新增)。拿它落库 = 每一轮都把整段历史再写一遍 ⇒ **历史翻倍**,
  而每一轮的回复看起来都正常、每条单测只要不数字数就全绿。
- 所以另立一个**逐轮覆写**的通道 `turn_messages`,只装**本轮新产生的**消息。
  各节点写它(`messages` 那份照旧给 `add_messages` 用,**两个都写**)。
- **它必须进 `resolve_references` 的逐轮重置清单**(与 `gate_passed` 等并列):
  它是覆写通道,不重置的话「这轮没产生消息」的路径(闲聊/兜底)
  会继承上一轮的值 ⇒ 上一轮的消息被再写一遍。这正是 ch05–ch06 那条
  「通道与它的清零必须同处一地」的第三次应用。

另外:`Message(role="tool")` 必须带非空 `tool_call_id`,否则 `schemas.Message` 的
validator 会抛 —— 这是 ch01 加的护栏,本章第一次真的用到。

- [ ] **Step 7: 改 `app/api/chat.py`**

`stream_input` 里加上播种(仅在快照无 `messages` 时):

插在**既有的待续检查旁边** —— 端点**已经在取快照**了(ch06 那段
`snapshot = await graph.aget_state(...)` 与 `if not snapshot.next: 409`),
复用同一次调用,**不额外加一次 IO**。`request.resume is not None` 那个分支的既有代码
**一行都不改**。

```python
        snapshot = await graph.aget_state({"configurable": {"thread_id": session_id}})
        # 这一线程当前的 state。`messages` 没播种过时为空列表。
        # ⚠️ 是 `snapshot.values`,不是 `snapshot.next` —— 后者是「待续节点名」,
        # 与 state 内容无关,取错了会恒得空列表 ⇒ **每轮都播种** ⇒ 历史翻倍。
        seeded = list(snapshot.values.get("messages") or [])
        <既有 resume 分支代码原样保留,不动>
        else:
            stream_input = {
                "conversation_id": session_id,
                "user_input": request.message,
                "history": history,
                "trace": [],
            }
            # **只在 state 没有 messages 时播种** —— add_messages 是 append-only,
            # 每轮无条件播种会把整段历史重复追加,而每轮回复看起来都正常。
            if not seeded:
                stream_input["messages"] = to_lc_messages(
                    await load_history(session=session, conversation_id=session_id)
                )
```

注意 `request.resume is not None` 的续跑分支**不播种**(`stream_input` 是 `Command`)——
续跑续的是同一轮,上下文该从 checkpoint 还原。

- [ ] **Step 8: 跑测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_node.py tests/test_api_chat.py tests/test_agent_graph.py -m "not db"`
Expected: PASS

- [ ] **Step 9: 提交**

```bash
git add app/agent/state.py app/agent/nodes.py app/prompts.py app/api/chat.py tests/
git commit -m "feat(ch07): messages 通道(add_messages)+ 工具结果落表 + 播种幂等"
```

---

### Task 6: `prompts.build_context_messages` —— 定序组装

**Files:**
- Modify: `app/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `trim_messages`(LangChain)、`app.memory.trim.count_tokens`
- Produces: `app.prompts.build_context_messages(*, brand_name, history, user_input, summary, evidence) -> list[BaseMessage]`

- [ ] **Step 1: 先用 Context7 核一遍 `trim_messages` 的当前签名**

Run: 用 Context7 MCP 查 `reference.langchain.com` 的 `trim_messages`。
Expected: 确认 `strategy` / `start_on` / `token_counter` / `allow_partial` 四个参数名与语义**没变**。
**spec §2.3 记的是 2026-09-22 核对的版本**;版本对不上的 API 是返工重灾区,
而这个函数本章只调一处、错了却会静默少给模型几轮历史(不报错)。

- [ ] **Step 2: 写失败测试**

`tests/test_prompts.py` 追加:

```python
def test_context_messages_put_the_only_system_message_first():
    """`system` 只有一条,而且必须是第 0 条。

    上游模板会把**所有** system 上提合并渲染;多一条 system,工具定义就被挤到
    可变内容**之后**,前缀缓存整段作废。
    """
    msgs = build_context_messages(
        brand_name="本店", history=[], user_input="你好", summary="", evidence=None
    )
    assert isinstance(msgs[0], SystemMessage)
    assert sum(isinstance(m, SystemMessage) for m in msgs) == 1


def test_context_messages_put_history_in_order_and_the_user_turn_last():
    """定序:已经分好层的历史按原序 → 用户的当前这句话**必须是最后一条**。"""
    history = [
        Message(id=1, role="user", content="第一句"),
        Message(id=2, role="assistant", content="第一答"),
    ]
    msgs = build_context_messages(
        brand_name="本店", history=history, user_input="第二句",
        summary="", evidence=None,
    )
    assert [type(m).__name__ for m in msgs] == [
        "SystemMessage", "HumanMessage", "AIMessage", "HumanMessage",
    ]
    assert msgs[-1].content == "第二句"


def test_summary_and_evidence_ride_inside_the_user_message():
    """梗概与证据**并进用户那条消息**,附在原话之后,不另起一条。

    另起一条会凭空多出一轮「谁说的」;而并进去与现有 `build_messages` 同形状。
    """
    msgs = build_context_messages(
        brand_name="本店", history=[],
        user_input="这个能退吗", summary="用户问过订单 1002", evidence=None,
    )
    assert len(msgs) == 2
    assert msgs[-1].content.startswith("这个能退吗")
    assert "订单 1002" in msgs[-1].content


def test_no_summary_no_evidence_keeps_user_text_verbatim():
    """两者都没有时,用户原话**逐字**就是那条消息 —— 不许多一个换行。"""
    msgs = build_context_messages(
        brand_name="本店", history=[], user_input="你好", summary="", evidence=None
    )
    assert msgs[-1].content == "你好"


def test_lc_token_counter_counts_content_only_not_structure():
    """`trim_messages` 的 counter 必须**只数 content**,不数 `tool_calls`。

    与 `trim.select_history` 的既有口径一致(结构性元数据不占预算)。
    口径不一致的症状是「预算说是够的、实际发出去超了」—— 两处都各自「看起来合理」。

    对应地,**层 1 的选择本身(`trim_messages(strategy="last", start_on="human")`)
    不在本任务测**:T6 只做「拿到已分好层的历史之后怎么拼」,选择在 T10 接入,
    它的 `start_on` 断言也在 T10(那边才有真实的分层输入)。
    """
    from app.prompts import _lc_token_counter

    a = AIMessage(content="", tool_calls=[{
        "name": "query_order", "args": {"order_id": "1001"},
        "id": "c1", "type": "tool_call",
    }])
    assert _lc_token_counter([a]) == 0

    b = HumanMessage("四个字")
    assert _lc_token_counter([b]) == count_tokens("四个字")   # 与全仓同一把尺子
```

- [ ] **Step 3: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_prompts.py`
Expected: FAIL —— `ImportError: cannot import name 'build_context_messages'`

- [ ] **Step 4: 实现**

```python
def _lc_token_counter(messages: list) -> int:
    """`trim_messages` 要 `Callable[[list[BaseMessage]], int]`,本仓的
    `trim.count_tokens` 收 `str` —— 这就是那个适配器。

    `tool_calls` **不计入**(与 `trim.select_history` 的既有口径一致:
    结构性元数据不占预算)。
    """
    total = 0
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total += trim.count_tokens(content)
    return total


def build_context_messages(
    *, brand_name: str, history: Sequence[Message], user_input: str,
    summary: str, evidence: list[dict] | None = None,
) -> list:
    """组装本轮发给模型的上下文。**定序见 spec §7.3。**

        [0] SystemMessage(人设 + 红线)   ← 每轮逐字节相同,前缀缓存命中区
        [..] 层 2 截短 / 层 1 原文        ← 已经分好层,这里只负责选与拼
        [-1] HumanMessage(用户原话 + 梗概 + 证据)

    ⚠️ **`system` 只有第 0 条这一条**。多一条,上游模板会把所有 system 上提合并,
    工具定义被挤到可变内容之后,前缀缓存整段作废(spec 需求 3)。

    ⚠️ 梗概与证据**并进用户那条消息、附在原话之后**,不另起一条 ——
    与 `build_messages` 同形状,只是把证据的位置从原话**前**改到原话**后**。
    """
    msgs = [SystemMessage(render_system_prompt(brand_name))]
    msgs.extend(to_lc_messages(history))
    tail = _render_tail(summary=summary, evidence=evidence)
    msgs.append(HumanMessage(f"{user_input}\n\n{tail}" if tail else user_input))
    return msgs
```

`_render_tail` 把梗概与证据合成一段文本(复用 `render_evidence`);
两者都空时返回空串。**层 1 的选择**(`trim_messages`)在 T10 由端点接入 ——
本任务只做「拿到已经分好层的历史之后怎么拼」,这样它可单测。

- [ ] **Step 5: 跑测试并提交**

Run: `.venv/Scripts/python.exe -m pytest tests/test_prompts.py`
Expected: PASS

```bash
git add app/prompts.py tests/test_prompts.py
git commit -m "feat(ch07): build_context_messages —— 定序组装,梗概证据并进用户消息"
```

---

### Task 7: 日志落盘 + `memory/journal.py`

**Files:**
- Create: `app/logging_setup.py`、`app/memory/journal.py`
- Modify: `app/main.py`
- Test: `tests/test_memory_journal.py`

**Interfaces:**
- Consumes: `app.memory.budget.ContextBudget`(T1)、`app.memory.layers.Layers`(T4)
- Produces:
  - `app.logging_setup.setup_logging(log_dir: str = "log") -> None`
  - `app.memory.journal.model_ctx(*, conversation_id, summary, layers, evidence_tokens, budget) -> None`
  - `app.memory.journal.history_ctx(*, conversation_id, summaries, history, budget) -> None`

- [ ] **Step 1: 写失败测试**

```python
```python
def test_model_ctx_logs_segmented_tokens_not_only_a_total(caplog):
    """**必须能在日志里分开读到各段的 token 数。**

    只打一个总计的话,「层 2 按截短后计数」**失效与生效长得一模一样** ——
    这是 spec §7.6 专门为那处洞加的观测面,也是验收 4b 的判据。

    断言的是**JSON 契约**(键在不在、值对不对),不是文案 —— 所以它不违反
    spec §10.3「不测日志格式」那一条。
    """
    s = _settings()
    b = budget.derive(settings=s, system_prompt="你是客服。")
    got = layers.split(
        [Message(id=1, role="user", content="你好")],
        summary_upto_msg_id=0, layer1_from_msg_id=0, settings=s,
    )
    with caplog.at_level(logging.INFO):
        journal.model_ctx(
            conversation_id="c1", summary="用户问过订单 1002",
            layers=got, evidence_tokens=0, budget=b,
        )

    payload = _last_payload(caplog, "model_ctx")
    assert {"layer1", "layer2", "summary", "evidence", "total"} <= set(payload["tokens"])
    assert payload["tokens"]["layer1"] == got.layer1_tokens     # ← 真的按截短后
    assert payload["bounds"]["layer1_from_msg_id"] == 0


def test_history_ctx_is_logged_even_for_the_fallback_branch(client_factory, caplog):
    """每轮必打,**包括不进 Agent 的那几轮**(闲聊/投诉/兜底/退款子流程)。

    那几轮恰恰最容易「看起来正常、其实上下文是错的」—— ch06 的 T4 就是这么丢的
    (节点写了 `confidence`、通道根本不存在、**单测全绿而生产恒为 `None`**),
    而它当时**没有任何观测面**。

    这条走**端点**、用 `intent="闲聊"`(路由不进 Agent),断言 `history_ctx`
    照样出现。**不走端点的话**,「每轮必打」这件事其实没被验到 ——
    直接调 `journal.history_ctx` 只能证明函数本身能打。
    """
    client, _ = client_factory(batches=[[FakeChunk("你好呀")]], intent="闲聊")
    with caplog.at_level(logging.INFO):
        with client as c:
            c.post("/api/chat/stream", json={"message": "你好"})
    _last_payload(caplog, "history_ctx")      # 取不到就抛 ⇒ 红
```

辅助(放在文件顶部,两个用例共用):

```python
def _last_payload(caplog, prefix: str) -> dict:
    """从日志里取出最后一条 `"<prefix> <json>"` 的 JSON 体。

    取不到**直接抛** —— 退化成「返回 {}」会让上面的 `<=` 断言恒真,
    而这条用例的全部价值就在于「那一行到底有没有」。
    """
    for record in reversed(caplog.records):
        if record.message.startswith(f"{prefix} "):
            return json.loads(record.message.split(" ", 1)[1])
    raise AssertionError(f"日志里没有 {prefix} 行")
```
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_memory_journal.py`
Expected: FAIL

- [ ] **Step 3: 写 `app/logging_setup.py`**

```python
"""ch07:日志落盘。

**本章之前全仓没有任何日志配置** —— `logger.info(...)` 全靠 uvicorn 的默认
handler 打到控制台,所以需求 6 说的 `log/app.log` 根本不存在。
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: str = "log") -> None:
    """把 root logger 接一个轮转文件 handler。

    **`encoding="utf-8"` 是硬要求,不是讲究。** 本机 locale 是 cp936;
    不给 encoding 时 Python 用 `locale.getpreferredencoding()`,中文日志行会
    直接抛 `UnicodeEncodeError` —— 而这个异常发生在**写日志的时候**,
    与业务逻辑毫无关系,报错位置会指向完全无关的地方。
    """
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path / "app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        root.addHandler(handler)
```

- [ ] **Step 4: 改 `app/main.py` 的 lifespan**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    _startup_budget_self_check()
    _warm_embedding_in_background()
    yield
```

`_startup_budget_self_check`:

```python
def _startup_budget_self_check() -> None:
    """启动时自检:历史预算连**一轮**都装不下就报警。

    **报警但不拒绝启动**(spec §7.2):需求原文是「报警」,而写成拒绝启动会把
    一个纯算术问题变成服务起不来。两者只差一行,是一句话可翻的选择。
    """
    settings = get_settings()
    b = budget.derive(
        settings=settings, system_prompt=render_system_prompt(settings.brand_name)
    )
    if not b.fits_one_round:
        logger.error(
            "上下文预算不足:历史预算 %s < 每轮稳态 %s —— 连一轮都装不下。"
            "查 model_context_window / max_output_tokens / max_agent_steps 等项。",
            b.history_budget, settings.per_round_steady,
        )
    else:
        logger.info(
            "上下文预算:窗口 %s − 固定开销 %s − 单轮峰值 %s = 历史 %s(层1 %s / 层2 %s)",
            b.window, b.fixed_overhead, b.peak,
            b.history_budget, b.layer1_budget, b.layer2_budget,
        )
```

**pytest 下要跳过**:与 `_warm_embedding_in_background` 同款(`"pytest" in sys.modules`),
理由相同 —— TestClient 会跑 lifespan,而单测不该因为一个算术配置去打 error 日志。

- [ ] **Step 5: 写 `app/memory/journal.py`**

两个函数各打**一行 JSON**,前缀固定:

```python
logger.info("model_ctx %s", json.dumps(payload, ensure_ascii=False))
```

**为什么是 JSON 不是自由文本**:spec §10.3 明确「不测日志格式 —— 它是给人看的」,
但 §10.2 又要求「层 2 的计数确实按截短后」。两条要同时成立,唯一办法是把
**契约**(键与值)与**文案**(措辞)分开 —— JSON 让单测断前者、人读后者。
用 `ensure_ascii=False` 让中文在 `log/app.log` 里可读(T7 的 handler 已钉 utf-8)。

`model_ctx` 的 payload:

```python
{
  "conversation_id": ...,
  "summary": "…梗概全文…",
  "sliding": [{"role": ..., "content": ...}, ...],   # 截短后的实际形态
  "rounds": len(sliding),
  "tokens": {"layer1": …, "layer2": …, "summary": …, "evidence": …, "total": …},
  "budgets": {"layer1": …, "layer2": …},
  "bounds": {"summary_upto_msg_id": …, "layer1_from_msg_id": …},
}
```

`history_ctx` 的 payload 同形,但 `summary` 换成**摘要行列表**
(`[{"seq": 1, "upto_msg_id": 12, "content": "…"}, ...]`),`sliding` 是滑窗。

**四条硬要求**:
1. `tokens` **必须分段**,不能只给 total(否则截短失效无从观测);
2. `sliding` 装的是**截短后**的消息,不是原文 —— 它要能一眼看出 `…` 与 `[工具结果] `;
3. `summary` 在 `model_ctx` 里是**全文**,不是条数;
4. **不记 prompt / response 原文**(那是密钥泄漏面与日志膨胀源,spec §8)。
   异常一律先过 `redact_api_key` 再记。

- [ ] **Step 6: 跑测试并提交**

Run: `.venv/Scripts/python.exe -m pytest tests/test_memory_journal.py`
Expected: PASS

```bash
git add app/logging_setup.py app/memory/journal.py app/main.py tests/test_memory_journal.py
git commit -m "feat(ch07): 日志落盘(log/app.log,显式 utf-8)+ model_ctx/history_ctx"
```

---

### Task 8: 摘要 —— prompt + 触发判定 + 原子落库

**Files:**
- Create: `app/memory/summarize.py`
- Test: `tests/test_memory_summarize.py`

**Interfaces:**
- Consumes: `app.memory.layers.split`(T4)、`history.append_summary_and_advance`(T3)
- Produces:
  - `app.memory.summarize.SUMMARY_SYSTEM_PROMPT: str`
  - `app.memory.summarize.build_summary_messages(*, turns: Sequence[Message]) -> list`
  - `app.memory.summarize.should_summarize(layers: Layers, *, layer2_budget: int) -> bool`
  - `app.memory.summarize.render_turns(turns: Sequence[Message]) -> str`
  - `app.memory.summarize.summarize_range(*, model, session, conversation_id, turns, upto_msg_id) -> str | None`
  - `app.memory.summarize.join_summaries(summaries: Sequence[tuple[int, str]]) -> str`
    (把 `load_summaries` 的 `(seq, content)` 列表拼成**一段**背景文本;
    空列表 → 空串。T10 消费它,`summary_text` 通道的值就是它)

- [ ] **Step 1: 写失败测试**

```python
def _layers_with(layer2_tokens: int) -> layers.Layers:
    """只关心 layer2_tokens 这一个字段时的最小构造。"""
    return layers.Layers(layer2=[], layer1=[], layer2_tokens=layer2_tokens,
                         layer1_tokens=0)


def test_should_summarize_compares_against_the_layer2_budget():
    """触发看**层 2 截短后**的用量,不数条数。

    数条数看不见工具结果涨 —— 而工具结果是本章最大的一块。
    """
    assert should_summarize(_layers_with(101), layer2_budget=100) is True
    assert should_summarize(_layers_with(99), layer2_budget=100) is False


def test_should_summarize_is_false_when_exactly_at_budget():
    """`>` 不是 `>=` —— 「装得下就不压」(验收 3 的反向断言)。

    边界上差一个等号,就是「每轮都压一次」与「一次都不压」的区别。
    """
    assert should_summarize(_layers_with(100), layer2_budget=100) is False


def test_render_turns_includes_tool_rows():
    """渲染给摘要模型看的**原文**必须含工具结果。

    否则梗概里存不下「用户报过的订单号」—— 而验收 2 的落点正是它
    (「最开始那个订单后来怎么说」要能靠梗概答对)。

    注意这里喂的是**原文**(`role="tool"` 那条的完整 JSON),不是截短版:
    拿截短文本去提炼,等于把截断损失焊进梗概,而梗概是**永久背景**。
    """
    text = render_turns([
        Message(id=1, role="user", content="订单 1002 能退吗"),
        Message(id=5, role="tool", content='{"order_no":"1002","status":"已取消"}',
                tool_call_id="call_1"),
    ])
    assert "1002" in text
    assert "已取消" in text          # 截短版只留 60 字也会含它,但完整 JSON 也含
    assert "{" in text               # ← 这条才把「原文 vs 截短版」区分开


def test_summarize_range_returns_none_for_an_empty_range():
    """空区间不开任务、不落空梗概。"""
    assert asyncio_run(summarize_range(
        model=None, session=None, conversation_id="c1", turns=[], upto_msg_id=0,
    )) is None


def test_summarize_range_does_not_advance_the_anchor_on_failure():
    """**失败等于什么都没发生。**

    边界只在**成功后**推进 ⇒ 失败时层 2 原封不动、下次再触发即可,所以不重试。
    反过来(先推边界再落库)会让那段历史**永久消失**。

    注入的模型**真的抛**(不是「返回空串」那种)—— 返回空串是「在处理之后注入」
    的变体:它跳过了「模型调用失败」那一步,而那一步正是本用例要验的。

    `session=None` 也是**故意的**:断言的是「抛在落库**之前**」。真去落库会撞
    `AttributeError`,而那个错会盖掉 `RuntimeError` —— 于是这条用例就红在
    一个指向脚手架的地方,盖住了真问题。
    """
    class _Boom:
        async def ainvoke(self, messages):
            raise RuntimeError("上游炸了")

    with pytest.raises(RuntimeError):
        asyncio_run(summarize_range(
            model=_Boom(), session=None, conversation_id="c1",
            turns=[Message(id=1, role="user", content="你好")], upto_msg_id=1,
        ))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_memory_summarize.py`
Expected: FAIL

- [ ] **Step 3: 写 `SUMMARY_SYSTEM_PROMPT`**

四样提炼物 + 三条硬约束,**逐条写进 prompt**(spec §4.1):

```python
SUMMARY_SYSTEM_PROMPT = """你要把一段电商客服对话压缩成一段**事实备忘**。

提炼四样,其他的都不要:
1. 用户问过的商品 / 款式;
2. 用户报过的订单号、手机号等标识(**原样抄写,不要改写**);
3. 用户明确的诉求;
4. **还没解决的问题**。

硬约束:
- **对话里没有出现的内容,一个字都不许编。** 没有订单号就不要写订单号。
- 寒暄、客套、致谢、对话状态(「用户发了一条消息」)一律不留。
- 长度控制在几十到一两百字。

输出这一段备忘本身,不要解释、不要加标题、不要输出 JSON。"""
```

**为什么必须写「什么不算」**:ch03 的挖知识 prompt 首版没写,把客服「抱歉查不到运费」
这种**非答案**挖成了知识,直接把验收的正确答案挤下 top-1 ——
**模型的失败被挖进知识库,再教它下次继续失败。** 梗概有同一个失效模式:
把寒暄与自我否定压进梗概,然后**每轮都注入一遍**。

- [ ] **Step 4: 实现其余部分**

`should_summarize` / `render_turns` / `build_summary_messages` 都是纯函数;
`summarize_range` 调模型后**只调用一次** `append_summary_and_advance`(原子)。

- [ ] **Step 5: 跑测试并提交**

```bash
git add app/memory/summarize.py tests/test_memory_summarize.py
git commit -m "feat(ch07): 摘要 prompt(四样提炼物 + 不编不留寒暄)+ 触发判定 + 原子落库"
```

---

### Task 9: 后台摘要执行体

**Files:**
- Create: `app/memory/tasks.py`
- Test: `tests/test_memory_tasks.py`

**Interfaces:**
- Consumes: `app.memory.summarize.summarize_range`(T8)
- Produces:
  - `app.memory.tasks.run_summary_in_background(*, conversation_id, settings, model_factory) -> bool`
    (**True = 起了**,False = 因已有任务而跳过 —— 返回值是 T10 与 §10.5 验收 4 的观测面)
  - `app.memory.tasks._run_body(*, conversation_id, settings, model_factory) -> None`(真正干活的部分,单测的接缝)
  - `app.memory.tasks._INFLIGHT: set[str]`(进程内在跑的会话;**测试可以读它,生产不该读**)
  - `app.memory.tasks._reload_state(engine, conversation_id) -> tuple[int, int, list[Message]]`
    (重读 `(summary_upto, layer1_from, 区间内的原文)`;单测的接缝)

- [ ] **Step 1: 先读 ch04 的既有实现**

Run: 读 `app/kb/orchestrate.py`。
**必须照抄它的三条**:专用线程、线程内 `asyncio.run`、**自建 engine 且任务结束 dispose**。
理由写在 CLAUDE.md:`get_engine()` 的 lru_cache 单例绑在**首次使用它的主事件循环**上,
后台线程复用会出跨循环的异步连接问题 —— 这是 ch04 实测踩出来的。

- [ ] **Step 2: 写失败测试**

```python
为此 `run_summary_in_background` **返回 `bool`**(起了 = True,因已有任务而跳过 = False),
并把「真正干活」抽成一个模块级的 `_run_body(*, conversation_id, settings, model_factory)`
—— 下面三条都靠这两个接缝,不然就得去测线程时序,而那是**不稳定断言**的来源。

```python
def test_only_one_summary_task_per_conversation(monkeypatch):
    """同一会话同一时刻只允许一个任务。

    两个任务并发会把同一段原文压两遍(或撞 `(conversation_id, seq)` 唯一键),
    而**重复梗概里每一段单独看都正常** —— 它与「压对了」在数据上只差一段,
    没有任何东西会报错。
    """
    release = threading.Event()

    def _blocking(*, conversation_id, settings, model_factory):
        release.wait(timeout=5)

    monkeypatch.setattr(tasks, "_run_body", _blocking)
    tasks._INFLIGHT.clear()

    first = tasks.run_summary_in_background(
        conversation_id="c1", settings=_settings(), model_factory=lambda s: None
    )
    second = tasks.run_summary_in_background(
        conversation_id="c1", settings=_settings(), model_factory=lambda s: None
    )
    assert first is True
    assert second is False            # ← 第二个必须被挡下

    # 另一个会话**不受影响** —— 全局一把锁会把所有会话串起来
    third = tasks.run_summary_in_background(
        conversation_id="c2", settings=_settings(), model_factory=lambda s: None
    )
    assert third is True

    release.set()
    tasks._INFLIGHT.clear()


def test_task_rereads_anchors_instead_of_trusting_the_trigger_snapshot(monkeypatch):
    """任务开头必须**重读**两个锚点 —— 起任务到真跑之间边界可能已经变了。

    用触发时的快照去压,压出来的区间可能与当前层 2 对不上,
    结果是**一段历史被跳过**:既不在层 2、也不在梗概里。
    """
    seen = {}

    async def _fake_summarize(*, model, session, conversation_id, turns, upto_msg_id):
        seen["turns"] = [m.id for m in turns]
        seen["upto"] = upto_msg_id
        return "梗概"

    monkeypatch.setattr(tasks, "summarize_range", _fake_summarize)
    monkeypatch.setattr(tasks, "_reload_state", lambda engine, conv: (3, 7, _HISTORY()))
    # ↑ 重读回来的是 (summary_upto=3, layer1_from=7),与「触发时的快照」不同

    tasks._run_body(conversation_id="c1", settings=_settings(),
                    model_factory=lambda s: None)

    assert seen["turns"] == [4, 5, 6]        # ← 用的是**重读**到的 (3, 7]
    assert seen["upto"] == 7


def test_task_never_raises_into_the_caller(monkeypatch):
    """后台任务失败**不冒泡到请求路径** —— 它在**自己的线程**里。

    失败只留日志(spec §7.6 的 `summary fail`),边界不动。
    这条形状本身就保证了不冒泡(线程里抛不会传到请求),所以真正的断言是
    **`_run_body` 内部把它接住了** —— 否则线程会打一条
    `Exception in thread` 的噪音,而那是「没人处理」的样子。
    """
    async def _boom(**kwargs):
        raise RuntimeError("上游炸了")

    monkeypatch.setattr(tasks, "summarize_range", _boom)
    monkeypatch.setattr(tasks, "_reload_state", lambda engine, conv: (0, 5, _HISTORY()))

    tasks._run_body(conversation_id="c1", settings=_settings(),
                    model_factory=lambda s: None)   # ← **不抛**才算过

    # 并且锚点**没有被推进**(它只在成功后推)
    assert "c1" not in tasks._INFLIGHT
```
```

- [ ] **Step 3: 跑测试确认失败,然后实现**

结构对齐 `app/kb/orchestrate.py`:
- 进程内 `set` 做同会话去重(模块级,加锁)
- 线程内 `asyncio.run(_run())`
- `_run` 里 `create_async_engine(settings.database_url)` + `try/finally: await engine.dispose()`
- 五个生命周期日志点(spec §7.6),异常走 `redact_api_key`

- [ ] **Step 4: 跑测试并提交**

```bash
git add app/memory/tasks.py tests/test_memory_tasks.py
git commit -m "feat(ch07): 后台摘要执行体(专用线程 + 自建 engine,复刻 ch04 模式)"
```

---

### Task 10: 端点接线 —— 降级 + 起任务 + 播种精简版

**Files:**
- Modify: `app/api/chat.py`、`app/services/chat.py`、`app/prompts.py`、`app/agent/nodes.py`
- Test: `tests/test_api_chat.py`、`tests/test_chat_service.py`、`tests/test_agent_node.py`

**Interfaces:**
- Consumes: T1/T4/T6/T7/T8/T9 的全部产物
- Produces: 端点在流开始前完成:预算 → 降级 → 起摘要任务 → 组装精简版 → 播种

- [ ] **Step 1: 写失败测试**

```python
**注意 `tests/test_api_chat.py` 是同步文件**(`def test_x(client_factory)` + `with client as c`),
不要写成 `async def`。四条都用一个把历史塞满的 `FakeSession` 造出「层 1 超预算」的场景:

```python
SCRATCH_CONV = "c" * 32


def _stuffed_session(rows: int = 20):
    """造一个已存在、且历史长到会触发降级的会话。"""
    db = FakeSession()
    db.conversations[SCRATCH_CONV] = Conversation(
        id=SCRATCH_CONV, user="demo-user", status="active",
        summary_upto_msg_id=0, layer1_from_msg_id=0,
    )
    for i in range(rows):
        db.add(MessageRecord(
            conversation_id=SCRATCH_CONV,
            role="user" if i % 2 == 0 else "assistant",
            content="很长的历史内容" * 20,
        ))
    return db


def test_oversized_user_input_returns_400_json_before_streaming(client_factory):
    """超 `max_user_input_tokens` ⇒ 400 且**是普通 JSON 不是 SSE**。

    一旦 yield 过首帧,响应头就发出去了、状态码再也改不了 ——
    这正是预算校验必须在流开始前的原因,也是这条断言存在的理由。
    """
    client, _ = client_factory(batches=[[FakeChunk("好")]], max_user_input_tokens=5)
    with client as c:
        resp = c.post("/api/chat/stream", json={"message": "这是一句明显超过五个 token 的话"})

    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("application/json")   # ← 不是 SSE
    assert "event:" not in resp.text          # 确认真的没走流


def test_degrade_persists_the_new_anchor_and_does_not_touch_messages(client_factory):
    """降级只写一个整数,**一行 messages 都不动**。

    「不搬数据」是本章的核心卖点,必须有断言钉住 —— 否则「顺手把旧消息截短了
    写回 messages 表」这种实现能让其余**所有**用例照样通过。
    """
    db = _stuffed_session()
    before = [(m.role, m.content) for m in db.messages]
    client, _ = client_factory(batches=[[FakeChunk("好")]], session=db)

    with client as c:
        c.post("/api/chat/stream", json={"session_id": SCRATCH_CONV, "message": "现在这句"})

    conv = db.conversations[SCRATCH_CONV]
    assert conv.layer1_from_msg_id > 0                      # ← 降级真的写进去了
    after = [(m.role, m.content) for m in db.messages[:len(before)]]
    assert after == before                                  # ← 旧行一行没改
    # 新增的两行是本轮的 user + assistant,不是被搬过来的历史
    assert len(db.messages) == len(before) + 2


def test_summary_task_is_fired_without_blocking_the_reply(client_factory, monkeypatch):
    """摘要任务被起起来了,而**回复不 await 它**。

    spec 验收 4「摘要生成没有阻塞该轮用户回复」。测法是**换掉起任务的函数**,
    让它记录调用后立刻返回,再断言 done 帧照样到 ——
    而不是去测时间差(那会退化成一条不稳定断言,且在快机器上恒真)。
    """
    fired: list[str] = []
    monkeypatch.setattr(
        chat_api, "run_summary_in_background",
        lambda **kw: (fired.append(kw["conversation_id"]), True)[1],
    )
    # ↑ 这要求 `app/api/chat.py` 里是 `from app.memory.tasks import
    #   run_summary_in_background`(模块级名字),而不是 `from app.memory import
    #   tasks` 再 `tasks.run_summary_in_background(...)`。后者 patch 不到,
    #   而红法会是「明明起了、断言说没起」—— 指向测试而不是实现。
    #   同理,T4 的 `layers` 与 T8 的 `summarize` 在端点里也要是模块级名字。
    db = _stuffed_session()
    client, _ = client_factory(batches=[[FakeChunk("好")]], session=db)

    with client as c:
        resp = c.post("/api/chat/stream", json={"session_id": SCRATCH_CONV, "message": "现在这句"})

    assert fired == [SCRATCH_CONV]                    # ← 起了
    assert _parse_sse(resp.text)[-1][0] == "done"     # ← 而回复没被它挡住


def test_layer1_within_budget_triggers_neither_degrade_nor_summary(client_factory, monkeypatch):
    """**验收 3 的单元版**:装得下就一个动作都不做。

    「压缩是成本不是美德」。这条防的是「保守起见每次都压一点」的实现 ——
    那种实现能让验收 1/2/4 **全部通过**,而它在默认窗口下白白把历史压没了。
    """
    fired = []
    monkeypatch.setattr(
        chat_api, "run_summary_in_background",
        lambda **kw: (fired.append(kw), False)[1],
    )
    db = FakeSession()
    db.conversations[SCRATCH_CONV] = Conversation(
        id=SCRATCH_CONV, user="demo-user", status="active",
        summary_upto_msg_id=0, layer1_from_msg_id=0,
    )
    db.add(MessageRecord(conversation_id=SCRATCH_CONV, role="user", content="你好"))
    db.add(MessageRecord(conversation_id=SCRATCH_CONV, role="assistant", content="你好呀"))
    client, _ = client_factory(batches=[[FakeChunk("好")]], session=db)   # 默认窗口

    with client as c:
        c.post("/api/chat/stream", json={"session_id": SCRATCH_CONV, "message": "现在这句"})

    assert fired == []                                        # 没起任务
    assert db.conversations[SCRATCH_CONV].layer1_from_msg_id == 0   # 也没降级
```
```

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 实现**

**⚠️ 三件事必须一起做,少一件就是一个静默缺口:**

**(a) 端点**继续调 `prepare_turn`(它现在**已经**在 `app/api/chat.py:118` 被调),
不要把它换掉 —— 换了它就成了**死代码**,而 `tests/test_chat_service.py` 那几条
会变成孤儿测试(`tests/test_api_chat.py:703` 还 monkeypatch 着这个名字)。

**(b) 「本轮输入超 `max_user_input_tokens` → 400」的检查移进 `prepare_turn`。**
T2 之后这个检查**不存在了**:旧口径把 `count_tokens(user_input)` 算进「已用」,
新口径把它归进峰值的 `max_user_input_tokens` 那一项 —— 于是**实际输入长度
再也没人比过**,50k token 的一句话会一路送到上游(`prepare_turn` 现在因此
有一个**读了不用的 `user_input` 参数**)。在 `prepare_turn` 里补:

```python
    if trim.count_tokens(user_input) > settings.max_user_input_tokens:
        raise trim.ContextOverflowError(
            used=trim.count_tokens(user_input), budget=settings.max_user_input_tokens
        )
```

放在预算判据**之前**(输入本身超限与历史装不下是两回事,但都归 400)。
`resume` 分支递进来的 `user_input` 是 `None` —— 那条路**不调** `prepare_turn`
(端点既有注释已写明),所以这里不必处理 `None`;**但不要假设它永远不是 None**,
真拿不准就 `if user_input is not None and ...`。

**(b2) 判据从 `< 0` 改成 `not b.fits_one_round` —— 这是一条 spec 分歧的收口。**

spec §8 那张表写着「**历史预算装不下一轮** → 400」,对应 `not b.fits_one_round`;
而 T2 实现的是 `b.history_budget < 0`。两者在「预算为正但小于 `per_round_steady`」
时**结论不同**:比如 `history_budget = 5`,按 spec 该 400,按现实现却**静默带着
空历史往下走**(用户拿到一个没有任何上下文的回答,而没有任何东西报错)。
`fits_one_round` 因此在本章里**只有 T7 的启动自检一个消费者**,而它在请求路径上
无人问津 —— 这是「有写无读」的变体。

改 `prepare_turn` 里的判据:

```python
    if not b.fits_one_round:            # spec §8:「装不下一轮」就是 400 的判据
        raise trim.ContextOverflowError(used=b.fixed_overhead, budget=b.window)
```

`< 0` 被它完全覆盖(`0 < per_round_steady` 恒成立),所以是收紧不是放宽。

**(b3) 400 的文案现在指错了地方 —— 顺手改对。**

`ContextOverflowError(used=b.fixed_overhead, budget=b.window)` 拼出来的是
「本轮输入需要 5463 tokens,超出可用预算 1024 tokens」:`used` 其实是**固定开销**、
`budget` 是**整个窗口**。故障在**配置**(窗口比 开销+峰值 还小),而这句话把
运维指向了**用户输入**。改 `app/memory/trim.py:17` 那句模板,让它在配置故障下说得像
配置故障(如「上下文预算不足:固定开销与单轮峰值已占满窗口」)。
**不必担心破坏测试**:`tests/test_trim.py` 自己构造异常、只断言两个数字出现在
`str(err)` 里;`tests/test_api_chat.py` 只要求消息里有 `"tokens"`。措辞可以随便改。

**(b4) 把 `app/services/chat.py` docstring 里那句「由端点在 T10 接线」改掉** ——
检查落在**这个文件**,不是端点。一句把人指到错文件去的注释,
与本仓的「报错指向别处」是同一类毛病,只是印刷在注释里。

**(c) T2 的空档已经存在(承诺修复点就是这里)**:T2 到 T10 之间,超长输入
不再被拒。这是**计划没写明的空档,不是 T2 的实现缺陷**(它照计划写的,
并在 docstring 里如实记了这一点)。本步是它的关闭点。

端点里(持锁内、流开始前)按序串起来:

```python
        b = budget.derive(settings=settings, system_prompt=render_system_prompt(settings.brand_name))
        if b.history_budget < 0:
            raise HTTPException(status_code=400, detail=CONTEXT_BUDGET_TOO_SMALL)
        # ↑ 超长输入的 400 在 prepare_turn 里(上面 (b)),不在这儿

        layer1_from = layers.degrade(
            history, summary_upto_msg_id=conv.summary_upto_msg_id,
            layer1_from_msg_id=conv.layer1_from_msg_id,
            layer1_budget=b.layer1_budget, settings=settings,
        )
        if layer1_from != conv.layer1_from_msg_id:
            await history_service.advance_anchors(
                session=session, conversation_id=session_id, layer1_from=layer1_from
            )

        got = layers.split(
            history, summary_upto_msg_id=conv.summary_upto_msg_id,
            layer1_from_msg_id=layer1_from, settings=settings,
        )
        if summarize.should_summarize(got, layer2_budget=b.layer2_budget):
            run_summary_in_background(
                conversation_id=session_id, settings=settings,
                model_factory=create_extract_model,
            )   # 不 await
        trimmed = got.layer2 + got.layer1
        summaries = await load_summaries(session=session, conversation_id=session_id)
        summary_text = summarize.join_summaries(summaries)
```

四个名字(`layers` / `summarize` / `run_summary_in_background` / `load_summaries`)
都必须是**模块级导入的名字**,不能写成 `tasks.run_summary_in_background(...)` ——
单测靠 `monkeypatch.setattr(chat_api, "<名字>", ...)` 拦住它们,patch 打不中时
红法会指向测试而不是实现。

**- [ ] Step 3b: agent 节点改调 `build_context_messages`,并把两个通道播种进去**

这一步**必须做** —— 不做的话 T6 造出来的 `build_context_messages` **没有任何消费者**,
「定序组装 / system 只有一条 / 梗概并进用户消息」这套机制**在生产里根本不跑**,
而 T6 的单测**全绿**(它们直接调那个函数)。

`app/agent/nodes.py::make_agent_node` 的消息组装换成:

```python
        msgs = build_context_messages(
            brand_name=settings.brand_name,
            history=state.get("history") or [],          # 已是精简版(层2+层1)
            user_input=state["resolved_input"],
            summary=state.get("summary_text") or "",
            evidence=state.get("evidence") or [],
        )
```

并把 `build_messages` 的 import 换成 `build_context_messages`
(**不要删** `build_messages` —— 它的 4 条 `tests/test_prompts.py` 用例还在,
而 `build_context_messages` 内部复用它的 evidence 渲染段)。

播种侧(`stream_input`)加两行:

```python
                "history": trimmed,                       # 精简版
                "summary_text": summary_text,             # 梗概全文
```

**`summary_text` 用 `summarize.join_summaries(summaries)` 把多段拼成一段** ——
它是**背景**,不是逐段清单;拼法在 `summarize.py` 里给出(空列表 → 空串)。

两个 400 文案是**固定文案**,与既有 `RESUME_WITHOUT_PENDING` 同款:
不得出现 Python 标识符,且要过 `redact_api_key`。

- [ ] **Step 4: `journal` 接上**

**四件事,每一项单独就能让一条验收标准落空:**

**(1)`history_ctx` 在 `resolve_references` 之前打 —— 而 T7 已经接好了。**
**不要重复接。** 但**必须把那个调用的 `history=` 从 `prepare_turn` 的输出换成
分层后的 `trimmed`(`got.layer2 + got.layer1`)** —— 否则它传的是
`trim.select_history` 的输出,而那个函数**只整轮丢弃、从不标注内容**,
`…` 与 `[工具结果] ` **不可能出现** ⇒ **验收 4b 指定的那条线结构上承载不了 4b**。
T7 为此留了一条**故意会红的 tripwire**(`assert "bounds" not in payload`):
你要动那条线时它会变红,那是设计如此 —— 读它的 docstring 再决定,不要直接删。

**(2)`model_ctx` 在 `agent` 节点组装完消息之后打,而且必须用【降级之后的锚点】
重新 `split` 一次。** 锚点已由 §7.5 播进 `stream_input`,但 state 里放的是
**扁平的 `trimmed` 列表**;`journal.model_ctx` 要的是一份 `Layers`
(它从中读 `sliding` 与 `bounds`)。不重新 split 的话,`model_ctx.sliding`
描述的可能是**与真正发出去的消息不同的一次切分** —— 而**没有任何断言覆盖这种分叉**。

**(3)必须调 `log_trigger(...)`(T9 提供)。** `summary trigger` 那行日志
**只能在这里发**(那个接缝上才有 `layer2_tokens` 与 `layer2_budget`)。
不调的话,**验收 2 的 grep 无物可命中** —— 级联的第一环
(`summary trigger 层2 约 N token > 预算 M`)在日志里根本不存在。

**(4)层 1 的选择(`trim_messages`)走 `prompts.select_layer1(history, *, max_tokens)`。**
**不要在端点里直接 import `trim_messages`** —— `app/prompts.py` 是 LangChain 的
唯一面,端点直接调用会把 LLM 库的调用点散出去,而那正是本章花力气避免的。

两处日志都带 `conversation_id`。

- [ ] **Step 5: 跑测试并提交**

```bash
git add -u
git commit -m "feat(ch07): 端点接线 —— 降级/起后台任务/播种精简版,三个 400 守卫"
```

---

### Task 11: 两个只读端点

**Files:**
- Create: `app/api/conversations.py`
- Modify: `app/main.py`(include_router,**必须在 `mount("/")` 之前**)
- Test: `tests/test_api_conversations.py`

**Interfaces:**
- Produces:
  - `GET /api/conversations` → `{"items": [{"id", "created_at", "preview", "summarized"}]}`
  - `GET /api/conversations/{id}/messages` → `{"items": [{"role", "content", "created_at"}]}` / 404

- [ ] **Step 1: 写失败测试**

**本文件不复用 `test_api_chat.py` 的 `client_factory`** —— 这两个端点不碰模型、
不碰工具、不碰图,只需要一个 DB 会话替身。复用它会把「单测不联网」这条硬规矩
寄存在另一个文件的实现细节上(那个 `client_factory` 默认还换掉了检索器,
正是为了不联网)。自带一个最小 fixture:

```python
"""GET /api/conversations 与 .../messages。同步用例 + TestClient。"""

import pytest
from fastapi.testclient import TestClient

from app.db.models import Conversation, MessageRecord
from app.db.session import get_session
from app.main import app

CONV_A = "a" * 32
CONV_B = "b" * 32


class _StubSession:
    """只支撑这两个端点要用的两种读到的东西:LIST 与 (LIST, 单条)。

    刻意**不做成通用替身** —— 不支持的查询形态直接抛,而不是返回空,
    否则「端点查错了表」会退化成「返回空列表」,而那是这条用例最该发现的事。
    """

    def __init__(self, conversations, messages):
        self.conversations = conversations
        self.messages = messages

    async def execute(self, stmt, *args, **kwargs):
        cols = stmt.column_descriptions
        entity = cols[0]["entity"]
        if entity is Conversation:
            # 有 where 子句 = 按 id 查单条(→ 404 那条路);没有 = 列表查询。
            # **不要把这两种混成一种**:混了之后「列表接口串了另一个用户的会话」
            # 与「详情接口查不到就 404」两条断言会互相掩盖。
            if stmt.whereclause is not None:
                cid = stmt.whereclause.right.value
                row = self.conversations.get(cid)
                return _Result([row] if row is not None else [])
            rows = list(self.conversations.values())
            return _Result(sorted(rows, key=lambda c: c.created_at, reverse=True))
        if entity is MessageRecord:
            cid = stmt.whereclause.right.value
            rows = [m for m in self.messages if m.conversation_id == cid]
            return _Result(sorted(rows, key=lambda m: m.id))
        raise AssertionError(f"替身不支持的实体:{entity}")


class _Result:
    """`session.execute(...)` 的返回物:只需 `.scalars()` 后接 `.all()` / `.one_or_none()`。

    **本文件自带一份**,不从 `test_api_chat.py` 导入 —— 跨测试文件互相 import
    会让「这个替身到底支持什么」变得谁也说不清,而它的全部价值恰恰是
    **支持的东西被显式列出来**。
    """

    def __init__(self, rows):
        self._rows = list(rows)

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)

    def one_or_none(self):
        if len(self._rows) > 1:
            raise AssertionError(f"替身期望至多一行,拿到 {len(self._rows)}")
        return self._rows[0] if self._rows else None

    def one(self):
        if len(self._rows) != 1:
            raise AssertionError(f"替身期望恰好一行,拿到 {len(self._rows)}")
        return self._rows[0]


@pytest.fixture
def conv_client():
    """造会话与消息 → 返回 (client, session 替身)。"""
    def make(*, conversations, messages):
        session = _StubSession(conversations, messages)
        async def _override():
            yield session
        app.dependency_overrides[get_session] = _override
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()


def test_list_filters_by_demo_user_and_orders_newest_first(conv_client):
    """固定 `user='demo-user'`(无认证,前端从来不传 user_id),新在前。

    **必须放一个别的 user 的会话进去** —— 不放的话「有没有 WHERE user」
    在输出上完全一样,这条用例就恒真。
    """
    convs = {
        CONV_A: Conversation(id=CONV_A, user="demo-user", status="active"),
        CONV_B: Conversation(id=CONV_B, user="someone-else", status="active"),
    }
    client = conv_client(conversations=convs, messages=[])
    with client as c:
        items = c.get("/api/conversations").json()["items"]
    assert [i["id"] for i in items] == [CONV_A]        # ← 别人的那个不在里面


def test_preview_comes_from_the_first_user_message(conv_client):
    """预览取**第一条 user 消息**前 30 字 —— 不是最后一条,也不是条数。

    放**两条** user 消息进去,取值取错(取最后一条)就会红。
    """
    convs = {CONV_A: Conversation(id=CONV_A, user="demo-user", status="active")}
    msgs = [
        MessageRecord(conversation_id=CONV_A, role="user", content="第一个问题"),
        MessageRecord(conversation_id=CONV_A, role="assistant", content="第一个回答"),
        MessageRecord(conversation_id=CONV_A, role="user", content="后面又问的那个"),
    ]
    client = conv_client(conversations=convs, messages=msgs)
    with client as c:
        items = c.get("/api/conversations").json()["items"]
    assert items[0]["preview"] == "第一个问题"


def test_summarized_flag_reflects_the_anchor_not_the_row_count(conv_client):
    """`summarized` 读的是 `summary_upto_msg_id > 0`,**不是**「有没有梗概行」。

    两者在正常情况下一致 —— 而「正常情况下一致」正是假绿测试最爱藏身的地方。
    这条构造一个**只有锚点为 0 才是正确答案**的输入:两个会话里,
    A 的锚点是 0、B 的是 12;若实现改去数梗概行数,两个都会是 False,
    于是 B 那条断言变红。
    """
    convs = {
        CONV_A: Conversation(id=CONV_A, user="demo-user", status="active",
                             summary_upto_msg_id=0, layer1_from_msg_id=0),
        CONV_B: Conversation(id=CONV_B, user="demo-user", status="active",
                             summary_upto_msg_id=12, layer1_from_msg_id=20),
    }
    client = conv_client(conversations=convs, messages=[])
    with client as c:
        items = {i["id"]: i for i in c.get("/api/conversations").json()["items"]}
    assert items[CONV_A]["summarized"] is False
    assert items[CONV_B]["summarized"] is True       # ← 这条把「数行数」判死


def test_messages_endpoint_returns_raw_text_not_truncated(conv_client):
    """回载的是**原文** —— 侧栏切回来要看的就是当初聊了什么。

    放一条长到**必然**会被层 2 截短的消息进去。拿截短版回载的话,
    `…` 会出现在响应里 —— 层 2 的渲染就泄漏到 UI 上了。
    """
    convs = {CONV_A: Conversation(id=CONV_A, user="demo-user", status="active")}
    long_reply = "这是一条很长的客服答复。" * 30
    msgs = [MessageRecord(conversation_id=CONV_A, role="assistant", content=long_reply)]
    client = conv_client(conversations=convs, messages=msgs)
    with client as c:
        items = c.get(f"/api/conversations/{CONV_A}/messages").json()["items"]
    assert items[0]["content"] == long_reply          # ← 逐字相等
    assert "…" not in items[0]["content"]


def test_messages_endpoint_404s_for_unknown_conversation(conv_client):
    client = conv_client(conversations={}, messages=[])
    with client as c:
        resp = c.get(f"/api/conversations/{CONV_A}/messages")
    assert resp.status_code == 404
```
```
```

- [ ] **Step 2: 跑测试确认失败,实现,跑通**

- [ ] **Step 3: 提交**

```bash
git add app/api/conversations.py app/main.py tests/test_api_conversations.py
git commit -m "feat(ch07): GET /api/conversations 与 .../messages 两个只读端点"
```

---

### Task 12: 前端会话侧栏(Vibe Coding)

**Files:**
- Modify: `app/static/index.html`

- [ ] **Step 1: 读现有前端结构**

Run: 读 `app/static/index.html` 里 `sessionId` / `streamInto` / `ctx` 的用法。
**本章不引入构建工具链**(与 ch01–ch06 一致)。

- [ ] **Step 2: 实现侧栏**

四件事:
1. 左侧会话列表(新在前、首问预览、已摘要标记);
2. 点击切换 → `GET .../messages` 回载原文 → 替换消息区;
3. 「新对话」= 开新会话(`sessionId = null`),**旧会话仍在侧栏**;
4. **`sessionId` 存 `localStorage`** —— 它现在只活在 JS 变量里,刷新即丢;
   有了侧栏之后这会很难解释(侧栏列着会话,而当前那个每次刷新都变成新的)。

**侧栏加载失败必须静默降级**:不弹错、不挡住聊天区。
catch 住、留一个 console 警告即可 —— 侧栏是增强,聊天是主功能。

- [ ] **Step 3: 手动过一遍**

Run: 起服务,浏览器开 `http://localhost:8000`
Expected: 开三个会话 → 侧栏三个条目 → 点回第一个 → 历史完整回载 → 接着聊 → 刷新页面仍在原会话

- [ ] **Step 4: 提交**

```bash
git add app/static/index.html
git commit -m "feat(ch07): 会话侧栏 + 切换回载 + sessionId 持久化(Vibe Coding)"
```

---

### Task 13: 摘要评估集 + 端到端验收 + 章级收尾

**Files:**
- Create: `evals/summary_cases.jsonl`、`scripts/run_summary_eval.py`、`scripts/acceptance_ch07.sh`
- Modify: `CLAUDE.md`、`dev-notes/ch07.md`、spec §12

- [ ] **Step 1: 写 `evals/summary_cases.jsonl`**

四类样例(spec §10.4):正例(含订单号/商品/未解决问题)、负例(纯寒暄)、
**幻觉探针**(对话里没有订单号的,梗概里不得出现 `\d{4,32}`)。

⚠️ 幻觉探针的数字形态必须是**真能出现**的形态。ch06 的 T1 有一条被实现者
用变异运行抓出来的**同义反复断言**:它用 `\d{4,32}` 去匹配「99」,
而 `99` 只有两位、永远匹配不上 —— 那条断言恒真,零判别力。

- [ ] **Step 2: 写 `scripts/run_summary_eval.py`**

按既有 `scripts/run_expand_eval.py` / `run_resolve_eval.py` 的形状。
**打印非 ASCII 要钉输出边界**(`sys.stdout.buffer.write(...encode("utf-8"))`),
不要依赖控制台 codec —— `✓`/`✗` 不在 GBK 里,`print` 会直接崩。

- [ ] **Step 3: 写 `scripts/acceptance_ch07.sh`**

五条验收(spec §10.5)。三条硬要求:
1. **窗口参数显式覆盖**(§9.4 那组),**断言只读日志里实际算出来的数,不硬编码** ——
   预算里的几个估算值一改,硬编码的断言就集体失效;
2. **验收 2 的模型判定用 `warn()` 档**:显式计数并打印,**不当作通过**。
   ch06 写死过理由:「必须显式计数并打印,否则它就退化成一条悄悄跳过的检查」;
3. **验收 3 是反向断言**(纯聊天二十轮,一次降级与摘要都不该发生)。
   它防的是「保守起见每次都压一点」—— 这类实现能让验收 1/2/4 全绿。

**含中文的请求体一律走 stdin heredoc 或 httpx,不走 `curl` argv。**
**SSE 断言不能直接 grep 原始流**:逐 token 推送会把 `20240915` 切成多个帧,
用 `join_tokens` 拼回后再比对。

- [ ] **Step 4: 跑验收**

Run: `bash scripts/acceptance_ch07.sh`(需服务已启动 + 真实 key + MySQL + Milvus)
**落盘时不要接管道截断**(`| tail -N`)—— ch06 就因为加了它,后台文件只留了尾巴,
11 条失败的内容全部没留下,只能重跑一次。

- [ ] **Step 5: 章级收尾**

- `CLAUDE.md`:ch07 条目、架构段、硬约束(至少加:`add_messages` 的 append-only 语义、
  `InMemorySaver` 是纯内存、日志必须显式 utf-8)
- spec §12 实现订正:记录所有与设计的偏离
- `dev-notes/ch07.md`:每阶段四样如实补(不许收尾一次性补记 —— 但本步是补完**最后**一段)
- 全量 `.venv/Scripts/python.exe -m pytest`

- [ ] **Step 6: 提交**

---

## 附:明确不做(与 spec §11 一致)

- 不改 `scripts/acceptance.sh`(ch01–ch06 老回归网)。它当前是红的:8 条因 ch06
  改路由而失败(题面走退款子流程并挂起,无 done 帧)、1 条 KB 漂移
  (`scripts/acceptance.sh:824` 每次跑写一份 PID 命名的
  `knowledge/超大件运费-$$.md` 且不清理)。**是 ch06 的欠账,不在本章范围。**
- 不做跨会话长期记忆、用户画像、语义检索捞历史、主题重要度。
- 不做摘要淘汰与清理(表只追加)。
- `done` 帧的 `usage` 仍是死值 `None`(接它必须连每轮重置一起做)。
