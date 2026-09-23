# ch09 实现计划 —— 数据观测(Langfuse)与低置信度数据飞轮

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给客服系统接上 Langfuse 全链路观测,并把「答不上来的问题」通过三个入池入口送进一条「标准化 → 查重 → 人工审核 → 写回知识库」的飞轮,再加一条评估趋势流水线。

**Architecture:** 观测走一个**唯一的边界模块** `app/observability.py`(关掉即全 no-op);飞轮走**三个纯步骤 + 一个 ch04 同款的专用线程编排**;知识路径的作答在 `agent` 节点里**多挂一条 JSON 协议消息**,文本流出经过一个**三态增量解码器**(`lead → protocol / plain`),`useful=false` 时停吐、走兜底并落池。

**Tech Stack:** Python 3.13 / FastAPI / SQLAlchemy 2 / LangGraph 1.2.11 / langchain-openai 1.6.2 / langfuse 4.15.4(对本项目网关实测过,见 spec §2.2、§3.4、§3.5)/ MySQL + Milvus。

**Spec:** `docs/superpowers/specs/2026-09-23-ecommerce-cs-ch09-observe-flywheel-design.md`

> ⚠️ **本计划里凡是与外部系统交互的结论,都来自实测,不是文档。** 具体地:
> - `response_format` 与 `bind_tools` **互斥**(spec §2.2)⇒ 协议只能纯提示词驱动。
> - `LangfuseSpan.update(**kwargs)` **静默丢弃**(spec §3.4-1)⇒ 不许用它设 trace 属性。
> - Metrics v2 的 `query` 必须是**一个 JSON 字符串参数**,`sum_totalCost` **恒为 0**(spec §3.5)。
> **不要凭记忆"纠正"这些地方。**

---

## Global Constraints

以下逐条来自 spec 与 `CLAUDE.md`,**每个任务的要求都隐含包含它们**:

- **单测全程不联网。** 评估脚本与验收脚本才允许打真实网络。
- `tests/` 里的 `Settings(...)` **必须传 `_env_file=None`**(仓库根有真实 `.env`,不传会让"缺字段应报错"的测试静默通过)。
- db 测试读真实 `.env`,**模块级** `pytestmark = pytest.mark.db`。
- 异步用例用 `@pytest.mark.anyio`(backend 由 `tests/conftest.py` 固定为 asyncio),**不用** pytest-asyncio。
- **不要再往命令行加 `-q`**(`pytest.ini` 的 `addopts` 已有一个,叠加会整行不打印 `N passed`)。
- 所有**出站**错误文本(SSE `error` 帧、`tool_result` 的失败 `summary`、422/502 的 detail)必须过 `app/sanitize.py:redact_api_key`。
- **含中文的请求体不许走 `curl` 的 argv**(MSYS2 按 CP936 重编码)⇒ 一律 stdin heredoc 或 httpx。
- **`init_db.py` 永不加列**(`create_all` 对已存在的表是空操作)⇒ `db/ch09.sql` 必须**单独执行**。
- `db/chNN.sql` **不幂等是刻意的**,重复执行要**响亮地失败**。
- **不改历史章节的产物**:`db/ch01..ch08.sql`、`app/mcp/**`、`app/tools/**` 本章一字不动。
- **不要给 `app/prompts.py` 的 `render_system_prompt` 加东西** —— 它是 `budget.derive` 的入参,改了会连带改预算推导与一批既有测试(spec §5.1)。
- 本项目的头号风险是**假绿测试**。写断言前先问:**如果实现改错了,这条断言的输出会不会不同?**

---

## 文件结构

### 新增

| 路径 | 职责 |
|---|---|
| `app/observability.py` | **唯一**的 Langfuse 边界。`enabled` / `make_handler` / `trace_scope` / `intent_scope` / `span`。关掉即全 no-op,且**不在模块顶层 import langfuse** |
| `app/agent/json_stream.py` | 三态增量 JSON 解码器。**纯状态机**,零 IO 零依赖 |
| `app/kb/evidence.py` | `evidence_confidence` / `evidence_detail`。**纯函数**,不依赖 LangChain |
| `app/flywheel/__init__.py` | 包标记 |
| `app/flywheel/normalize.py` | 口语原话 → 标准 FAQ 式问题 + 示例答案(模型调用) |
| `app/flywheel/dedupe.py` | 语义查重(模型调用) |
| `app/flywheel/pipeline.py` | 一次飞轮运行的编排(纯逻辑) |
| `app/flywheel/tasks.py` | 后台任务执行体(专用线程 + 自建 engine) |
| `app/api/feedback.py` | `POST /api/feedback` |
| `app/api/review.py` | 待审队列的四个端点 |
| `db/ch09.sql` | 两列 ALTER + 两张新表 |
| `scripts/calibrate_evidence.py` | 阈值标定 |
| `scripts/intent_cost.py` | 按意图的 token 统计 |
| `scripts/eval_trend.py` | 评估趋势表 |
| `scripts/acceptance_ch09.sh` | 六条端到端验收 |
| `evals/flywheel_cases.jsonl` | 标准化/查重的标注样例 |

### 改动

| 路径 | 改什么 |
|---|---|
| `app/config.py` | Langfuse 三键 + 置信度权重/阈值 + 快照两条 + 飞轮一条 |
| `app/db/models.py` | `LowConfidenceQuestion` 加两列;新增 `ReviewQueue`、`EvalRun` |
| `app/kb/assess.py` | `record_low_confidence` 加 `evidence_snapshot` 参数并**返回新行 id** |
| `app/agent/nodes.py` | ① 闸换判据 ② 检索 span ③ `agent` 的知识轮挂协议 + `_stream_round` 文本出口 ④ `useful=false` 落池 |
| `app/api/chat.py` | `config` 挂 callbacks;外层 `trace_scope`;中途 `intent_scope` |
| `app/main.py` | include 两个新 router |
| `app/static/index.html` | 👎 接后端 |
| `app/static/admin.html` | 「待审」标签页 |
| `scripts/run_eval.py` | `--limit` / `--trigger` + 写 `eval_runs` |
| `requirements.txt` / `.env.example` | 钉 langfuse / 补三个键 |

---

## Task 1: 依赖、配置项、`.env.example`

**Files:**
- Modify: `requirements.txt`
- Modify: `app/config.py`
- Modify: `.env.example`
- Test: `tests/test_config_ch09.py`

**Interfaces:**
- Consumes: 无
- Produces: `Settings` 上新增 `langfuse_public_key` / `langfuse_secret_key` / `langfuse_base_url` / `evidence_confidence_threshold` / `evidence_min_score` / `evidence_max_count` / `w_evidence_top1` / `w_evidence_count` / `w_evidence_gap` / `snapshot_top_n` / `snapshot_answer_chars` / `flywheel_batch_size`

> ⚠️ **`langfuse` 已经装进 `.venv` 了**(为写本计划做过实测),但 `requirements.txt` 里还没有 —— 本任务把它补上,让仓库与 venv 一致。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_config_ch09.py
"""ch09 新增配置项:默认值 + 下界。"""

import pytest

from app.config import Settings


def _settings(**over):
    base = {
        "openai_base_url": "http://x", "openai_api_key": "k",
        "openai_model": "m", "database_url": "mysql://x",
    }
    return Settings(**{**base, **over}, _env_file=None)   # 硬约束:必须传 _env_file=None


def test_langfuse_defaults_are_empty_so_tests_never_go_online():
    s = _settings()
    assert s.langfuse_public_key == ""
    assert s.langfuse_secret_key == ""
    assert s.langfuse_base_url == "https://us.cloud.langfuse.com"


def test_evidence_weights_are_bounded():
    s = _settings()
    assert s.w_evidence_top1 + s.w_evidence_count + s.w_evidence_gap == pytest.approx(1.0)
    assert 0.0 <= s.evidence_confidence_threshold <= 1.0
    assert s.evidence_max_count >= 1


def test_snapshot_and_flywheel_bounds():
    s = _settings()
    assert s.snapshot_top_n >= 1
    assert s.snapshot_answer_chars >= 1
    assert s.flywheel_batch_size >= 1


def test_out_of_range_is_rejected():
    with pytest.raises(Exception):
        _settings(evidence_confidence_threshold=2.0)
    with pytest.raises(Exception):
        _settings(snapshot_top_n=0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config_ch09.py`
Expected: FAIL —`Settings` 没有这些字段(pydantic-settings 默认 `extra="ignore"` 会**静默忽略**传入的未知键,所以第一条会红在属性不存在上)

- [ ] **Step 3: 加 `requirements.txt`**

在 `jsonschema==4.26.0` 之后追加一行(**钉死版本**):

```
langfuse==4.15.4
```

- [ ] **Step 4: 加配置项**

`app/config.py` 的 `Settings` 里追加(位置:放在 `tool_*` 那一组之后):

```python
    # ---- ch09 · Langfuse 观测。三个值任一为空 ⇒ 整套观测 no-op ----
    # **单测"全程不联网"这条硬约束就靠它守**:测试的 Settings 不传这三个键,
    # 于是 observability 全部走空壳,一个字节都不出网。
    # ⚠️ 默认值是 **Langfuse Cloud(美国区)**;要"链路数据不出自家服务器"
    #    就换成自部署地址 —— 只改这一个值,代码不动。
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://us.cloud.langfuse.com"

    # ---- ch09 · 置信度闸(spec §4)----
    # 判据 = w_top1*top1 + w_count*min(条数/max_count,1) + w_gap*clamp(top1-top2,0,1)
    # 三个权重之和为 1 ⇒ 输出天然落在 0–1。
    #
    # ⚠️ `evidence_confidence_threshold` 的默认值**是占位,不是标定值**。
    #    真值由 `scripts/calibrate_evidence.py` 在 evals/测试集.md(300 条 / 含
    #    D_absent 应拒答桶 60 条)上扫出来,依据是"应拒答桶拦截率 vs 正常桶误杀率"。
    #    标定前**不许**把这个数写进任何文档当已验证值。
    evidence_confidence_threshold: float = Field(default=0.42, ge=0.0, le=1.0)
    evidence_min_score: float = Field(default=0.15, ge=0.0, le=1.0)
    evidence_max_count: int = Field(default=3, ge=1)
    w_evidence_top1: float = Field(default=0.6, ge=0.0, le=1.0)
    w_evidence_count: float = Field(default=0.2, ge=0.0, le=1.0)
    w_evidence_gap: float = Field(default=0.2, ge=0.0, le=1.0)

    # ---- ch09 · 召回片段快照(spec §6)----
    snapshot_top_n: int = Field(default=5, ge=1)
    snapshot_answer_chars: int = Field(default=400, ge=1)

    # ---- ch09 · 飞轮(spec §8)----
    flywheel_batch_size: int = Field(default=10, ge=1)
```

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config_ch09.py`
Expected: PASS(4 passed)

- [ ] **Step 6: 补 `.env.example`**

在文件末尾追加(带注释说明为什么默认是 Cloud):

```
# ---- Langfuse(ch09)。三个值任一为空 ⇒ 整套观测关闭。----
# 默认是 Langfuse Cloud 美国区。要"链路数据不出自家服务器"就换成自部署地址。
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL="https://us.cloud.langfuse.com"
```

- [ ] **Step 7: 跑全量非 db 测试,确认没碰坏东西**

Run: `.venv/Scripts/python.exe -m pytest -m "not db"`
Expected: 全绿(数量 = 原来的通过数 + 4)。**记下这个数字**,后面每个任务都拿它比。

- [ ] **Step 8: Commit**

```bash
git add requirements.txt app/config.py .env.example tests/test_config_ch09.py
git commit -m "ch09 T1: 钉 langfuse 4.15.4 + 观测/置信度/飞轮的配置项"
```

---

## Task 2: `app/observability.py` —— 唯一的 Langfuse 边界

**Files:**
- Create: `app/observability.py`
- Test: `tests/test_observability.py`

**Interfaces:**
- Consumes: `app.config.Settings`(T1 的三个 langfuse 字段)
- Produces:
  - `enabled(settings) -> bool`
  - `make_handler(settings) -> object | None`
  - `trace_scope(*, conversation_id: str, settings) -> ContextManager[None]`
  - `intent_scope(intent: str, *, settings) -> TagScope`(对象有 `.enter()` / `.exit()`)
  - `span(name: str, *, as_type: str = "span", input=None, settings, output=None) -> ContextManager[object | None]`

**为什么是这五个形状**(spec §3):`trace_scope` 包住整段流;`intent_scope` **不能**是 `with`
—— 意图是流**中途**才知道的,所以要能手动 `enter()/exit()`;`span` 给两个 LangChain 覆盖不到的
地方(工具执行、知识检索)手工开观测。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_observability.py
"""观测边界:关掉时全 no-op,且**不 import langfuse**。

这是"单测全程不联网"这条硬约束的守卫。测试的 Settings 不传三个 LANGFUSE_*
⇒ 任何一条用例都不该让 langfuse 被 import 进来。
"""

import sys

import pytest

from app.config import Settings
from app.observability import enabled, intent_scope, make_handler, span, trace_scope


def _settings(**over):
    base = {
        "openai_base_url": "http://x", "openai_api_key": "k",
        "openai_model": "m", "database_url": "mysql://x",
    }
    return Settings(**{**base, **over}, _env_file=None)


def test_disabled_when_any_key_missing():
    assert enabled(_settings()) is False
    assert enabled(_settings(langfuse_public_key="pk")) is False
    assert enabled(_settings(langfuse_public_key="pk", langfuse_secret_key="sk")) is True


def test_disabled_means_no_handler_and_no_langfuse_import():
    s = _settings()
    assert make_handler(s) is None

    with trace_scope(conversation_id="c1", settings=s):
        scope = intent_scope("商品咨询", settings=s)
        scope.enter()
        with span("tool:x", as_type="tool", input={"a": 1}, settings=s) as sp:
            assert sp is None
        scope.exit()

    assert "langfuse" not in sys.modules, "关掉时不许把 langfuse import 进来"


def test_span_never_raises_even_if_body_raises():
    """观测挂掉绝不许影响业务 —— 与 app/tools/audit.py 的 record_audit 同族。"""
    s = _settings()
    with pytest.raises(ValueError):
        with span("tool:x", settings=s):
            raise ValueError("业务异常必须原样穿出去")


def test_intent_scope_is_idempotent_and_safe_without_enter():
    s = _settings()
    scope = intent_scope("物流", settings=s)
    scope.exit()          # 没 enter 就 exit,不许炸
    scope.enter()
    scope.enter()         # 重复 enter,不许炸
    scope.exit()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_observability.py`
Expected: FAIL —`ModuleNotFoundError: No module named 'app.observability'`

- [ ] **Step 3: 写实现**

```python
# app/observability.py
"""Langfuse 观测 —— **全章唯一**的边界。

三条纪律(每条都有测试):

1. **`app/` 下除本模块外,零处 import langfuse。** 别的模块只认
   `trace_scope` / `intent_scope` / `span` / `make_handler` 四个名字。
2. **关掉时全 no-op,且不 import langfuse。** 三个 `LANGFUSE_*` 任一为空
   ⇒ `enabled()` 为假 ⇒ 所有函数走空壳分支。**langfuse 的 import 一律放在
   函数体内**,顶层 import 会让"没配"的环境在导入期就炸。
3. **本模块不抛异常**(`span` 的 `__exit__` 吞掉一切并 `logger.warning`)——
   与 `app/tools/audit.py:record_audit` 的"永不抛"同族。观测不许影响业务。

⚠️ **两条实测出来的坑,改动前先读 spec §3.4:**

- `LangfuseSpan.update(**kwargs)` 的 kwargs 被**静默丢弃**
  (源码 docstring 逐字:`**kwargs: Additional keyword arguments (ignored)`)。
  ⇒ **不要**用 `span.update(**{"langfuse.trace.tags": [...]})` 设 trace 属性。
- `langfuse.propagate_attributes(...)` 返回的 `_AgnosticContextManager`
  **没有 `__aenter__`** ⇒ 中途进入只能用**同步** `__enter__`。
"""

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from app.config import Settings

logger = logging.getLogger(__name__)


def enabled(settings: Settings) -> bool:
    """三个 LANGFUSE_* 齐了才算开着。缺一个 ⇒ 整套观测 no-op。"""
    return bool(
        settings.langfuse_public_key
        and settings.langfuse_secret_key
        and settings.langfuse_base_url
    )


def _setup_env(settings: Settings) -> None:
    """把配置灌进环境变量 —— langfuse 客户端只认 env。"""
    import os

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_base_url)


def make_handler(settings: Settings) -> Any | None:
    """建一个 LangChain 回调。关掉时返回 None。

    ⚠️ 实测 4.15.4 的签名只有 `(*, public_key=None, trace_context=None)` ——
    文档里那套 `session_id=` / `user_id=` / `tags=` 构造参数是 **JS SDK** 的。
    会话与标签一律走 `trace_scope` / `intent_scope`,不要往这里塞。
    """
    if not enabled(settings):
        return None
    try:
        _setup_env(settings)
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception:  # noqa: BLE001 —— 观测建不起来不许拦住业务
        logger.warning("langfuse handler 构造失败,观测本次关闭", exc_info=True)
        return None


@contextmanager
def trace_scope(*, conversation_id: str, settings: Settings) -> Iterator[None]:
    """包住整段流,把会话 id 挂到 trace 上。

    实测:`propagate_attributes(session_id=...)` 在**这一层**是有效的,
    而且它覆盖**全部**观测(包括进入之前创建的 —— 因为它是外层)。
    """
    if not enabled(settings):
        yield
        return
    try:
        _setup_env(settings)
        from langfuse import propagate_attributes

        with propagate_attributes(trace_name="cs-chat", session_id=conversation_id):
            yield
    except Exception:  # noqa: BLE001
        logger.warning("langfuse trace_scope 失败,降级为无观测", exc_info=True)
        yield


class TagScope:
    """**可以中途进入**的标签作用域。

    `propagate_attributes` 是个上下文管理器,而意图要 `classify_intent` 跑完
    才知道(那时 `astream` 已经在跑了)⇒ 必须能手动 enter/exit。

    实测语义:**进入之后新建的观测带 tag,之前的没有**(spec §3.4-2)。
    开销的大头是意图分类**之后**的 agent 轮次,所以这个语义够用。
    """

    def __init__(self, cm: Any | None = None) -> None:
        self._cm = cm
        self._entered = False

    def enter(self) -> None:
        if self._cm is None or self._entered:
            return
        try:
            self._cm.__enter__()          # ← 同步,没有 __aenter__
            self._entered = True
        except Exception:  # noqa: BLE001
            logger.warning("langfuse intent tag 进入失败", exc_info=True)

    def exit(self) -> None:
        if self._cm is None or not self._entered:
            return
        try:
            self._cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            logger.warning("langfuse intent tag 退出失败", exc_info=True)
        finally:
            self._entered = False


def intent_scope(intent: str, *, settings: Settings) -> TagScope:
    """给 trace 打 `intent:<x>` 标签 —— 按意图统计花销就靠它(spec §3.5)。"""
    if not enabled(settings) or not intent:
        return TagScope(None)
    try:
        _setup_env(settings)
        from langfuse import propagate_attributes

        return TagScope(propagate_attributes(tags=["ch09", f"intent:{intent}"]))
    except Exception:  # noqa: BLE001
        logger.warning("langfuse intent_scope 构造失败", exc_info=True)
        return TagScope(None)


@contextmanager
def span(
    name: str, *, as_type: str = "span", input: Any = None, settings: Settings
) -> Iterator[Any | None]:
    """手工开一个观测。关掉时 yield None。

    **它存在的理由**:Langfuse 的 LangChain 回调只覆盖 LangChain 的 run。
    本项目的**工具执行**(`app/tools/executor.py:execute_tool`)与**知识检索**
    (`app/retrieval/search.py:KnowledgeRetriever`)都不是 LangChain run,
    一个 span 都不会自动出现 —— 而"每个节点的工具调用、检索结果都能铺开看"
    这条需求,只有这里能落地。

    实测可用的 `as_type`:`span` / `generation` / `agent` / `tool` / `chain` /
    `retriever` / `evaluator` / `guardrail` / `embedding`。
    """
    if not enabled(settings):
        yield None
        return
    cm = None
    try:
        _setup_env(settings)
        from langfuse import get_client

        cm = get_client().start_as_current_observation(
            name=name, as_type=as_type, input=input
        )
        handle = cm.__enter__()
    except Exception:  # noqa: BLE001
        logger.warning("langfuse span 进入失败", exc_info=True)
        yield None
        return

    exc_info = (None, None, None)
    try:
        yield handle
    except BaseException:  # noqa: BLE001 —— 业务异常原样穿出去
        exc_info = __import__("sys").exc_info()
        raise
    finally:
        try:
            cm.__exit__(*exc_info)
        except Exception:  # noqa: BLE001
            logger.warning("langfuse span 退出失败", exc_info=True)
```

> ⚠️ `span` 的 `settings` 是**关键字参数**,没有默认值 —— 与 `budget.derive` 同款,
> 本仓的规矩是**依赖走显式注入**,不留会自己兜底的参数。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_observability.py`
Expected: PASS(4 passed)

- [ ] **Step 5: 验证「关掉时真的不 import」这条断言有判别力**

把 `make_handler` 里的 `if not enabled(settings): return None` **删掉**再跑
`test_disabled_means_no_handler_and_no_langfuse_import`。
Expected: **红**(两条断言至少红一条)。

改回来。**这一步不是走过场** —— 本仓已记过多次"变异后没红"其实是断言没有判别力。

- [ ] **Step 6: Commit**

```bash
git add app/observability.py tests/test_observability.py
git commit -m "ch09 T2: observability 边界(关掉即 no-op,不 import langfuse)"
```

---

## Task 3: 端点接线 + 两个手动 span + **真机冒烟**

**Files:**
- Modify: `app/api/chat.py:453-457`(config 那一处)与它外面
- Modify: `app/agent/nodes.py`(检索 span;工具 span 在 `execute_tool` 调用点)
- Modify: `app/agent/confirm_nodes.py`(决议节点的工具 span)

**Interfaces:**
- Consumes: T2 的 `trace_scope` / `intent_scope` / `span` / `make_handler`
- Produces: 无新符号 —— 这一任务只做接线

> **本任务是全章风险最集中的一处。** spec §12.3-0 那条**唯一挡路的未知**在这里结清:
> 中途进入的 `propagate_attributes` 能不能穿透 LangGraph 的上下文,让 **graph 内部**的
> generation 带上 `intent:<x>` 标签。探针里模型调用是直接 `ainvoke` 的,真实链路上它在
> `graph.astream` 内部 —— **没验过**。

- [ ] **Step 1: 接线(端点)**

`app/api/chat.py`,找到 `graph.astream(...)` 那一处。改成(保持原有 `configurable` 不变):

```python
            handler = observability.make_handler(settings)
            extra: dict = {}
            if handler is not None:
                extra["callbacks"] = [handler]

            intent_cm: observability.TagScope | None = None
            try:
                async for mode, chunk in graph.astream(
                    stream_input,
                    config={
                        "configurable": {"thread_id": session_id},
                        **extra,
                    },
                    stream_mode=["custom", "updates"],
                ):
                    # ---- ch09:意图一出现就给它打 tag(中途进入)----
                    if (
                        intent_cm is None
                        and mode == "updates"
                        and isinstance(chunk, dict)
                        and "classify_intent" in chunk
                    ):
                        got = (chunk["classify_intent"] or {}).get("intent") or ""
                        intent_cm = observability.intent_scope(got, settings=settings)
                        intent_cm.enter()
                    ...   # 原有分支一字不改
            finally:
                if intent_cm is not None:
                    intent_cm.exit()
```

外面包一层 `trace_scope`(在 `async def generate()` 的模型调用段之外,`astream` 之前):

```python
        with observability.trace_scope(
            conversation_id=session_id, settings=settings
        ):
            return await _run_stream()      # 把原来那段搬运逻辑原样挪进来
```

> ⚠️ **别改 `configurable` 与 `stream_mode`** —— `stream_mode` 必须带 `updates`
> (langgraph 1.2.11 下只给 `custom` 会把 `interrupt()` 整个吞掉,ch06 已记)。

- [ ] **Step 2: 接线(检索 span)**

`app/agent/nodes.py` 的 `make_retrieve_knowledge_node` 里,`retriever.search(...)` 那一步
包上 span:

```python
        with observability.span(
            "retrieval", as_type="retriever",
            input={"query": state["resolved_input"]}, settings=settings,
        ) as sp:
            chunks = await retriever.search(state["resolved_input"])
            if sp is not None:
                sp.update(output={
                    "chunks": [
                        {"id": c.chunk_id, "score": round(c.score, 4),
                         "section_path": c.section_path}
                        for c in chunks
                    ]
                })
```

> `make_retrieve_knowledge_node` 现在的签名是 `(*, retriever, emit)` —— **加一个
> `settings` 参数**,并在 `graph.py` 的注册处传进去。这是本任务唯一改到 `graph.py` 的地方。

- [ ] **Step 3: 接线(工具 span)**

`app/agent/nodes.py` 里 `await execute_tool(...)` 那一处(`nodes.py:372` 附近),
包上 `observability.span(f"tool:{call['name']}", as_type="tool", input=call["args"], ...)`,
并在拿到 `outcome` 后 `sp.update(output={"ok": outcome.ok, "summary": outcome.summary,
"error_kind": outcome.error_kind})`。

`app/agent/confirm_nodes.py` 的 `apply_write_decision` 里那次 `execute_tool` 同样处理。

- [ ] **Step 4: 跑全量非 db 测试**

Run: `.venv/Scripts/python.exe -m pytest -m "not db"`
Expected: 全绿。**因为是 no-op 路径**(测试的 Settings 没有 LANGFUSE_*),接线不该改变任何行为。

- [ ] **Step 5: ⚠️ 真机冒烟 —— 本任务的核心,不许跳过**

先确认没有残留服务,再起服务(真实 `.env`,带 Langfuse 凭据):

```bash
# 清端口
for p in 8000; do pid=$(netstat -ano | grep ":$p " | grep LISTENING | awk '{print $5}' | head -1); \
  [ -n "$pid" ] && taskkill //F //PID $pid; done
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000 &
sleep 25
```

发一次**商品咨询**请求(**走 httpx,不走 curl argv** —— 中文会被 CP936 重编码):

```bash
.venv/Scripts/python.exe -X utf8 -c "
import httpx, json, sys
sys.stdout.reconfigure(encoding='utf-8')
r = httpx.post('http://127.0.0.1:8000/api/chat/stream',
               json={'message':'退货政策是什么'}, timeout=120)
print(r.status_code)
print(r.text[:400])
"
```

记下这次响应的 `session_id`(`meta` 帧里有)。**等约 60 秒**(ingestion 有延迟),然后:

```bash
.venv/Scripts/python.exe -X utf8 scripts/intent_cost.py --minutes 15
```

(这个脚本在 T4 才建 —— **本步先手工打一次 Metrics API**;见 T4 Step 5 的等价命令。)

**判据(三条都要成立)**:
1. 输出里**出现了** `tags` 含 `intent:商品咨询` 的一行;
2. 那一行的 token 数 **> 0**;
3. 去 Langfuse UI 用那个 `session_id` 搜到这条 trace,**人眼确认**巢状结构:
   `chat`(根)→ 若干个 `ChatOpenAI` generation + `retrieval` span。

**第 1 条不成立**(即只有 `tags: []` 或没有 intent 行)⇒ **走 spec §3.4 末尾的退路**:
把意图写进我们自己开的观测(`retrieval` / `tool:*` 的 `update(metadata={"intent": …})`),
并把 `intent_cost.py` 改成读 `GET /api/public/v2/observations` 自己聚合 token。
**把这次实测的结果(成或不成)写进 dev-notes 与 spec §15。**

- [ ] **Step 6: 收服务**

```bash
pid=$(netstat -ano | grep ":8000 " | grep LISTENING | awk '{print $5}' | head -1)
[ -n "$pid" ] && taskkill //F //PID $pid
```

- [ ] **Step 7: Commit**

```bash
git add app/api/chat.py app/agent/nodes.py app/agent/confirm_nodes.py app/agent/graph.py
git commit -m "ch09 T3: 端点挂 Langfuse 回调 + 检索/工具手动 span(真机冒烟见 dev-notes)"
```

---

## Task 4: `scripts/intent_cost.py` —— 按意图的 token 统计

**Files:**
- Create: `scripts/intent_cost.py`

**Interfaces:**
- Consumes: T3 打上的 `intent:*` tag
- Produces: CLI;验收 5 直接跑它

- [ ] **Step 1: 写脚本**

```python
"""按意图统计 token 花销 —— 打 Langfuse Metrics API v2。

用法:
    .venv/Scripts/python.exe scripts/intent_cost.py
    .venv/Scripts/python.exe scripts/intent_cost.py --minutes 120
    .venv/Scripts/python.exe scripts/intent_cost.py --intent 物流

⚠️ 三条**实测**出来的坑(spec §3.5),别凭文档改:

1. `query` 必须是**一个 JSON 字符串参数**:`params={"query": json.dumps({...})}`。
   把 view/metrics 平铺成独立 query 参数会 400。
2. **`sum_totalCost` 恒为 0** —— 本项目的模型没在 Langfuse 里配价格。
   所以这张表报 **token**,不报钱;cost 只在非零时才打印。
   **不许把 0 当成本报出去。**
3. 按 `tags` **过滤**时 `type` 必须是 `"arrayOptions"`(不是 `"string"`),
   值给**数组**。见 --intent。

ingestion 有延迟,所以这里**轮询**到有数据为止,而不是读一次就下结论。
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]


def emit(text: str) -> None:
    """cp936 陷阱:非 ASCII 一律走 buffer,别依赖控制台 codec。"""
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()


def _dotenv(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{name} 未在 .env 里配置")


async def _fetch(query: dict) -> dict:
    base = _dotenv("LANGFUSE_BASE_URL").rstrip("/")
    auth = (_dotenv("LANGFUSE_PUBLIC_KEY"), _dotenv("LANGFUSE_SECRET_KEY"))
    last = None
    for i in range(4):
        try:
            async with httpx.AsyncClient(timeout=40) as http:
                resp = await http.get(
                    f"{base}/api/public/v2/metrics",
                    params={"query": json.dumps(query)},
                    auth=auth,
                )
                resp.raise_for_status()
                return resp.json()
        except (httpx.ConnectError, httpx.ReadError) as exc:
            last = exc
            await asyncio.sleep(1.5 * (i + 1))
    raise SystemExit(f"连不上 Langfuse:{last}")


async def main() -> None:
    ap = argparse.ArgumentParser(description="按意图统计 token 花销")
    ap.add_argument("--minutes", type=int, default=60, help="回看窗口(分钟)")
    ap.add_argument("--intent", default="", help="只看某一个意图标签(如 物流)")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    query = {
        "view": "observations",                       # v2 只支持 observations / scores-*
        "metrics": [
            {"measure": "totalTokens", "aggregation": "sum"},
            {"measure": "count", "aggregation": "count"},
        ],
        "dimensions": [{"field": "tags"}],
        "filters": [],
        "fromTimestamp": (now - timedelta(minutes=args.minutes)).isoformat(),
        "toTimestamp": (now + timedelta(minutes=5)).isoformat(),
        "config": {"row_limit": 100},
    }
    if args.intent:
        query["dimensions"] = []
        query["filters"] = [{
            "column": "tags", "operator": "contains",
            "value": [f"intent:{args.intent}"],       # ← 数组,不是字符串
            "type": "arrayOptions",                   # ← 不是 "string"
        }]

    rows = (await _fetch(query)).get("data", [])
    usage = [r for r in rows if any(str(t).startswith("intent:") for t in (r.get("tags") or []))]

    emit(f"\n===== 按意图的 token 花销(近 {args.minutes} 分钟)=====\n")
    if not usage:
        emit("没有找到任何带 intent:* 标签的观测。")
        emit("可能原因:① 窗口内没有新请求 ② Langfuse ingestion 还没到"
             "(v2 端点有延迟,等 1–2 分钟再试) ③ 意图标签没打上(spec §3.4)")
        return

    table = []
    for r in usage:
        labels = [t.split(":", 1)[1] for t in r["tags"] if str(t).startswith("intent:")]
        if not labels:
            continue
        table.append({
            "intent": "/".join(labels),
            "observations": int(r.get("count_count") or 0),
            "tokens": int(r.get("sum_totalTokens") or 0),
            "cost": float(r.get("sum_totalCost") or 0),
        })
    table.sort(key=lambda x: -x["tokens"])

    emit(f"{'意图':<12}{'观测数':>8}{'token':>12}")
    emit("-" * 34)
    for row in table:
        emit(f"{row['intent']:<12}{row['observations']:>8}{row['tokens']:>12}")
    top = table[0]
    emit(f"\n最烧 token 的意图:**{top['intent']}**({top['tokens']} tokens)")

    total_cost = sum(r["cost"] for r in table)
    if total_cost > 0:
        emit(f"\n(成本合计 {total_cost:.6f})")
    else:
        emit("\n(cost 恒为 0:本项目模型未在 Langfuse 里配置价格 —— 只报 token,不报钱)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
```

- [ ] **Step 2: 真机跑一次**

Run: `.venv/Scripts/python.exe scripts/intent_cost.py --minutes 60`
Expected: 至少一行 `intent:` 行(T3 冒烟产生的)。若 T3 冒烟判定走了退路,这里要同步改成
读 observations 自己聚合。

- [ ] **Step 3: Commit**

```bash
git add scripts/intent_cost.py
git commit -m "ch09 T4: 按意图的 token 统计(打 Metrics API v2)"
```

---

## Task 5: `db/ch09.sql` + ORM

**Files:**
- Create: `db/ch09.sql`
- Modify: `app/db/models.py`
- Test: `tests/test_ch09_orm.py`(db 标记)、`tests/test_ch09_ddl.py`(非 db,只读文件比对)

**Interfaces:**
- Produces:
  - `LowConfidenceQuestion.evidence_snapshot: dict | None`
  - `LowConfidenceQuestion.matched_review_id: int | None`
  - `ReviewQueue`(`id` / `standard_question` / `example_answer` / `occurrences` / `status` / `approved_answer` / `first_raw_question` / `source_conversation_id` / `created_at` / `reviewed_at`)
  - `EvalRun`(`id` / `trigger_by` / `case_count` / `metrics` / `created_at`)

- [ ] **Step 1: 写 `db/ch09.sql`**

照 spec §7.1 **逐字**抄(ALTER 两列 + `review_queue` + `eval_runs`)。三条要点:

- **不写 `IF NOT EXISTS`** —— 不幂等是刻意的。
- **`review_queue` 上不加"标准化问题"的唯一键** —— 查重是**语义**判断(模型判是不是
  同一个意思),唯一键只能管**字面全等**;两者不是同一条规则,加了会在一次合理的
  语义归并上**响亮地 1062**。
- 文件头写明"必须单独执行,`init_db.py` 不加列"。

- [ ] **Step 2: 写测试(文件比对,非 db)**

```python
# tests/test_ch09_ddl.py
"""db/ch09.sql 的形状 —— 只读文件,不连库。

这些断言守的是 spec §7.1 里**逐条写明**的设计选择;它们变红时先回去读 spec,
别顺手改成"看起来更合理"的样子。
"""

from pathlib import Path

SQL = Path(__file__).resolve().parents[1] / "db" / "ch09.sql"


def _sql() -> str:
    return SQL.read_text(encoding="utf-8")


def test_file_exists_and_is_not_idempotent_by_design():
    s = _sql()
    assert "ALTER TABLE low_confidence_questions" in s
    assert "IF NOT EXISTS" not in s, "不幂等是刻意的(重复执行要响亮地失败)"


def test_two_new_columns():
    s = _sql()
    assert "evidence_snapshot" in s and "JSON" in s
    assert "matched_review_id" in s


def test_review_queue_has_no_unique_key_on_standard_question():
    s = _sql()
    assert "CREATE TABLE review_queue" in s
    assert "UNIQUE" not in s.upper(), (
        "查重是语义判断,唯一键管不了 —— 加了会在合理的语义归并上 1062"
    )


def test_eval_runs_shape():
    s = _sql()
    for col in ("trigger_by", "case_count", "metrics", "created_at"):
        assert col in s
```

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch09_ddl.py`
Expected: FAIL 前先确认文件已写 → 应该 PASS(先写文件再写测试是本任务的顺序;
**若你想按 TDD,就先写测试看它红**,两种都行,但不能不看红就直接绿)。

- [ ] **Step 3: ORM**

`app/db/models.py`:

```python
class LowConfidenceQuestion(Base):
    # ... 现有字段不动 ...
    evidence_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # ⚠️ **一个列担两个语义**:既记"这条问题归并到了 review_queue 的哪一行",
    # 又是飞轮流水线的**待处理标记**(`WHERE matched_review_id IS NULL`)。
    # ⇒ 流水线天然幂等,重跑不会重复归并。别只当成外键用。
    matched_review_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)


class ReviewQueue(Base):
    """待审队列(ch09,DDL: db/ch09.sql)。

    一行 = 一个**去重后**的知识缺口。查重命中时累加 `occurrences` 而不是新建行。
    """

    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    standard_question: Mapped[str] = mapped_column(String(512), nullable=False)
    example_answer: Mapped[str] = mapped_column(Text, nullable=False)
    occurrences: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    first_raw_question: Mapped[str] = mapped_column(Text, nullable=False)
    source_conversation_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class EvalRun(Base):
    """评估流水线的一轮(ch09,DDL: db/ch09.sql)。一行一轮,按时间连成趋势。"""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trigger_by: Mapped[str] = mapped_column(String(32), nullable=False)
    case_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
```

`Integer` / `JSON` 若未 import,补进 `app/db/models.py` 顶部的 `sqlalchemy` import 行。

- [ ] **Step 4: 建表(手工,本任务不做自动化)**

```bash
.venv/Scripts/python.exe scripts/init_db.py     # 只建**不存在**的表 ⇒ 会建出 review_queue / eval_runs
mysql -u root -p <库名> < db/ch09.sql           # ⚠️ 这一步是**必须**的:两列 ALTER 只有它能做
```

> 顺序:先 `init_db.py` 再 `db/ch09.sql`。后者在前者建过的库上会对两张新表报
> `ERROR 1050` —— **那是预期的**(与 `db/ch08.sql` 同一个已知取舍)。
> 但**两列 ALTER 一定会成功**,因为 `create_all` 不加列。
> 核对:`SHOW CREATE TABLE low_confidence_questions\G` 里要出现 `evidence_snapshot`。

- [ ] **Step 5: db 测试**

```python
# tests/test_ch09_orm.py
"""ch09 两张新表 + 两列 —— 真库往返。"""

import pytest
from sqlalchemy import text

from app.db.base import get_sessionmaker
from app.db.models import EvalRun, LowConfidenceQuestion, ReviewQueue

pytestmark = pytest.mark.db


@pytest.mark.anyio
async def test_new_columns_round_trip():
    maker = get_sessionmaker()
    async with maker() as s:
        row = LowConfidenceQuestion(
            question="T9 往返探针", entry_point="生成自评", reject_reason="r",
            evidence_snapshot={"chunks": [{"id": 1, "score": 0.5}]},
        )
        s.add(row)
        await s.commit()
        rid = row.id
        await s.refresh(row)
        assert row.evidence_snapshot == {"chunks": [{"id": 1, "score": 0.5}]}
        assert row.matched_review_id is None
        await s.execute(
            text("DELETE FROM low_confidence_questions WHERE id = :i"), {"i": rid}
        )
        await s.commit()


@pytest.mark.anyio
async def test_review_queue_round_trip():
    maker = get_sessionmaker()
    async with maker() as s:
        rq = ReviewQueue(
            standard_question="T9 标准问题", example_answer="示例",
            occurrences=1, first_raw_question="原话",
        )
        s.add(rq)
        await s.commit()
        rid = rq.id
        assert rq.status == "pending" and rq.occurrences == 1
        await s.execute(text("DELETE FROM review_queue WHERE id = :i"), {"i": rid})
        await s.commit()


@pytest.mark.anyio
async def test_eval_run_round_trip():
    maker = get_sessionmaker()
    async with maker() as s:
        r = EvalRun(trigger_by="manual", case_count=3, metrics={"top_k": 10})
        s.add(r)
        await s.commit()
        rid = r.id
        await s.refresh(r)
        assert r.metrics == {"top_k": 10}
        await s.execute(text("DELETE FROM eval_runs WHERE id = :i"), {"i": rid})
        await s.commit()
```

Run: `.venv/Scripts/python.exe -m pytest tests/test_ch09_orm.py`
Expected: PASS(3 passed)。**MySQL 没起时会失败 —— 那是环境问题不是代码问题**,
如实报,不要跳过。

- [ ] **Step 6: Commit**

```bash
git add db/ch09.sql app/db/models.py tests/test_ch09_ddl.py tests/test_ch09_orm.py
git commit -m "ch09 T5: review_queue / eval_runs 两张表 + 池子两列"
```

---

## Task 6: `app/kb/evidence.py` —— 置信度(纯函数)

**Files:**
- Create: `app/kb/evidence.py`
- Test: `tests/test_kb_evidence.py`

**Interfaces:**
- Consumes: `app.retrieval.search.RetrievedChunk`(`question` / `answer` / `category` / `chunk_id` / `section_path` / `score`)
- Produces:
  - `evidence_detail(chunks, *, settings) -> dict` 键:`top1` / `top2` / `count` / `gap` / `confidence`
  - `evidence_confidence(chunks, *, settings) -> float`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_kb_evidence.py
"""置信度合成的三条信号(spec §4.1)。

⚠️ 用例的输入要**真的走到那条分支**:条数信号封顶用 evidence_max_count=3,
所以"封顶"的用例至少要有 4 条;**分差**的用例要真的建出 top1/top2 两个不同分。
"""

from app.config import Settings
from app.kb.evidence import evidence_confidence, evidence_detail
from app.retrieval.search import RetrievedChunk


def _settings(**over):
    base = {
        "openai_base_url": "http://x", "openai_api_key": "k",
        "openai_model": "m", "database_url": "mysql://x",
        "evidence_min_score": 0.15, "evidence_max_count": 3,
        "w_evidence_top1": 0.6, "w_evidence_count": 0.2, "w_evidence_gap": 0.2,
    }
    return Settings(**{**base, **over}, _env_file=None)


def _c(score, cid=1):
    return RetrievedChunk(question="q", answer="a", category="c",
                          chunk_id=cid, score=score)


def test_empty_evidence_is_zero():
    s = _settings()
    assert evidence_confidence([], settings=s) == 0.0
    assert evidence_detail([], settings=s)["confidence"] == 0.0


def test_single_chunk_has_no_gap_signal():
    """只有一条时 top2 记 0 ⇒ gap 就是 top1。这条**不是**"没有信息",是设计。"""
    s = _settings()
    d = evidence_detail([_c(0.8)], settings=s)
    assert d["top1"] == 0.8 and d["top2"] == 0.0 and d["gap"] == 0.8
    assert d["count"] == 1


def test_ties_make_gap_zero_and_drag_confidence_down():
    """两条同分 ⇒ gap=0,置信度**明显低于**只有一条同分的情况。"""
    s = _settings()
    tie = evidence_confidence([_c(0.8, 1), _c(0.8, 2)], settings=s)
    lone = evidence_confidence([_c(0.8, 1)], settings=s)
    assert tie < lone


def test_count_signal_saturates_at_max_count():
    """4 条 ≥ max_count=3 ⇒ 条数信号打满,再加条目**不再变**。"""
    s = _settings()
    three = evidence_confidence([_c(0.5, i) for i in range(3)], settings=s)
    four = evidence_confidence([_c(0.5, i) for i in range(4)], settings=s)
    assert three == four


def test_low_scores_do_not_count_toward_the_count_signal():
    """低于 evidence_min_score 的块**不计入**条数 —— 否则一堆噪声会把置信度抬起来。"""
    s = _settings()
    noisy = evidence_confidence([_c(0.5, 1)] + [_c(0.01, i) for i in range(2, 6)], settings=s)
    clean = evidence_confidence([_c(0.5, 1)], settings=s)
    assert noisy == clean


def test_confidence_is_always_in_unit_interval():
    s = _settings()
    assert 0.0 <= evidence_confidence([_c(1.0, i) for i in range(10)], settings=s) <= 1.0


def test_weights_are_read_from_settings_not_hardcoded():
    """把权重改成"只看 top1",两种证据的排序应当跟着变 —— 守"权重真的被读了"。"""
    a = [_c(0.9, 1), _c(0.88, 2), _c(0.87, 3)]      # top1 高但分差小
    b = [_c(0.6, 1)]                                  # top1 低但只有一条
    default = _settings()
    top1_only = _settings(w_evidence_top1=1.0, w_evidence_count=0.0, w_evidence_gap=0.0)
    assert evidence_confidence(a, settings=default) > evidence_confidence(b, settings=default)
    assert evidence_confidence(a, settings=top1_only) > evidence_confidence(b, settings=top1_only)
```

> ⚠️ 最后那条用例只证明"权重被读了",**不证明"权重用对了"** —— 别把它读成后者的证据。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_evidence.py`
Expected: FAIL —`No module named 'app.kb.evidence'`

- [ ] **Step 3: 写实现**

```python
# app/kb/evidence.py
"""证据置信度 —— 把"检索 + 精排的结果"压成一个 0–1 的分(ch09 spec §4.1)。

**纯函数、零 IO、不依赖 LangChain。** 与 `app/kb/assess.py` 同层。

三个信号:
  top1  精排最高分(`RetrievedChunk.score` **已经是重排 sigmoid 分**)
  count 有效证据条数(分数 ≥ `evidence_min_score` 的那些,封顶 `evidence_max_count`)
  gap   top1 − top2(只有一条时 top2 记 0)

为什么要有 count 与 gap:单条高分可能是巧合;**够多条中高分**才叫"知识库覆盖了";
分差大说明那条明确对口,分差小说明几条都不对口。

⚠️ **阈值不是拍出来的**:`evidence_confidence_threshold` 由
`scripts/calibrate_evidence.py` 在 `evals/测试集.md` 上标定(spec §4.2)。
本模块的**权重初值是拍的**,标定只负责阈值 —— 这两件事别混。
"""

from typing import Sequence

from app.config import Settings
from app.retrieval.search import RetrievedChunk


def evidence_detail(
    chunks: Sequence[RetrievedChunk], *, settings: Settings
) -> dict:
    """三个信号 + 合成分。空证据返回全 0。"""
    if not chunks:
        return {"top1": 0.0, "top2": 0.0, "count": 0, "gap": 0.0, "confidence": 0.0}

    scores = sorted((c.score for c in chunks), reverse=True)
    top1 = float(scores[0])
    top2 = float(scores[1]) if len(scores) > 1 else 0.0
    gap = max(0.0, top1 - top2)
    count = sum(1 for s in scores if s >= settings.evidence_min_score)

    count_signal = min(count / settings.evidence_max_count, 1.0)
    confidence = (
        settings.w_evidence_top1 * top1
        + settings.w_evidence_count * count_signal
        + settings.w_evidence_gap * gap
    )
    return {
        "top1": round(top1, 4),
        "top2": round(top2, 4),
        "count": count,
        "gap": round(gap, 4),
        "confidence": round(min(max(confidence, 0.0), 1.0), 4),
    }


def evidence_confidence(
    chunks: Sequence[RetrievedChunk], *, settings: Settings
) -> float:
    return evidence_detail(chunks, settings=settings)["confidence"]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_kb_evidence.py`
Expected: PASS(8 passed)

- [ ] **Step 5: Commit**

```bash
git add app/kb/evidence.py tests/test_kb_evidence.py
git commit -m "ch09 T6: evidence_confidence 三信号合成(纯函数)"
```

---

## Task 7: 闸换判据 + `reject_reason` 新文案

**Files:**
- Modify: `app/agent/nodes.py:198-231`(`make_confidence_gate_node`)
- Test: `tests/test_agent_gate.py`(改既有)、`tests/test_agent_gate_ch09.py`(新增)

**Interfaces:**
- Consumes: T6 的 `evidence_detail`
- Produces: 无新符号;`reject_reason` 文案变更

> ⚠️ **`reject_reason` 的文案会变**,而 `tests/test_agent_gate.py` 里断的是
> `"0.31" in reject_reason`。那条要**按新文案改** —— **改的是文案不是判据**,
> 三个信号都要出现在里面。这是一处**必须显式处理的既有测试**。

- [ ] **Step 1: 改既有测试 + 加新用例(先红)**

在 `tests/test_agent_gate.py` 里把断旧文案的那条改成:

```python
    # ch09:判据从"取最高分"换成 evidence_confidence,**三个信号都要写进 reason**,
    # 审核人才看得出为什么被拦。文案变了,判据没变。
    assert "置信度" in row.reject_reason
    assert "低于阈值" in row.reject_reason
    assert "top1=" in row.reject_reason
    assert "条数=" in row.reject_reason
    assert "分差=" in row.reject_reason
```

新增 `tests/test_agent_gate_ch09.py`:

```python
"""ch09 闸:判据换成 evidence_confidence 之后的通过/不通过两支。"""

import pytest

from app.agent.nodes import make_confidence_gate_node
from app.config import Settings
from app.retrieval.search import RetrievedChunk


def _settings(**over):
    base = {
        "openai_base_url": "http://x", "openai_api_key": "k",
        "openai_model": "m", "database_url": "mysql://x",
        "evidence_confidence_threshold": 0.5,
    }
    return Settings(**{**base, **over}, _env_file=None)


class _FakeSession:
    def __init__(self):
        self.added = []
        self.commits = 0

    def add(self, row):
        self.added.append(row)

    async def commit(self):
        self.commits += 1


def _gate(session, settings):
    return make_confidence_gate_node(
        settings=settings, session=session, conversation_id="c1"
    )


@pytest.mark.anyio
async def test_weak_evidence_is_blocked_and_reason_names_all_three_signals():
    s = _settings()
    sess = _FakeSession()
    ev = [RetrievedChunk(question="q", answer="a", category="c", chunk_id=1, score=0.2)]
    out = await _gate(sess, s)({"user_input": "猫砂盆多少钱", "evidence": ev})
    assert out["gate_passed"] is False
    assert sess.commits == 1 and sess.added[0].entry_point == "置信度闸"
    r = sess.added[0].reject_reason
    assert "top1=" in r and "条数=" in r and "分差=" in r


@pytest.mark.anyio
async def test_strong_evidence_passes_and_writes_nothing():
    s = _settings()
    sess = _FakeSession()
    ev = [RetrievedChunk(question="q", answer="a", category="c", chunk_id=i, score=0.9)
          for i in range(3)]
    out = await _gate(sess, s)({"user_input": "退货政策", "evidence": ev})
    assert out["gate_passed"] is True
    assert sess.commits == 0 and sess.added == []


@pytest.mark.anyio
async def test_empty_evidence_is_blocked_with_readable_reason():
    s = _settings()
    sess = _FakeSession()
    out = await _gate(sess, s)({"user_input": "x", "evidence": []})
    assert out["gate_passed"] is False
    assert "检索为空" in sess.added[0].reject_reason
```

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_gate.py tests/test_agent_gate_ch09.py`
Expected: **红**(旧文案 + `evidence_confidence` 还没接)

- [ ] **Step 2: 改实现**

```python
    async def confidence_gate(state) -> dict:
        evidence = state.get("evidence") or []
        detail = evidence_detail(evidence, settings=settings)
        passed = bool(evidence) and detail["confidence"] >= settings.evidence_confidence_threshold

        if not passed:
            # 三个信号**都写进 reason** —— 审核页旁边就是这段文字,
            # 它要回答的是"为什么这条被判成答不了",不是一个分数。
            reason = (
                "检索为空"
                if not evidence
                else (
                    f"置信度 {detail['confidence']} 低于阈值 "
                    f"{settings.evidence_confidence_threshold}"
                    f"(打分:top1={detail['top1']} 条数={detail['count']} "
                    f"分差={detail['gap']})"
                )
            )
            await record_low_confidence(
                session,
                question=state["user_input"],
                source_conversation_id=conversation_id,
                entry_point="置信度闸",
                reject_reason=reason,
            )
        return {"gate_passed": passed, "trace": [...]}   # trace 原样
```

import 加 `from app.kb.evidence import evidence_detail`。

- [ ] **Step 3: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_gate.py tests/test_agent_gate_ch09.py tests/test_agent_graph.py`
Expected: PASS。**`test_agent_graph.py` 那条"弱证据 → 不进 Agent、落池"** 若断言措辞也变了,一并按新文案改。

- [ ] **Step 4: 跑全量非 db**

Run: `.venv/Scripts/python.exe -m pytest -m "not db"`
Expected: 全绿

- [ ] **Step 5: Commit**

```bash
git add app/agent/nodes.py tests/test_agent_gate.py tests/test_agent_gate_ch09.py tests/test_agent_graph.py
git commit -m "ch09 T7: 闸的判据换成 evidence_confidence,reject_reason 写全三个信号"
```

---

## Task 8: `scripts/calibrate_evidence.py` —— 阈值标定

**Files:**
- Create: `scripts/calibrate_evidence.py`
- Modify: `app/config.py`(把标定值写回 `evidence_confidence_threshold`)
- Modify: `dev-notes/ch09.md`(记读数)
- Modify: `docs/superpowers/specs/…ch09…-design.md` §15(记标定结果)

**Interfaces:**
- Consumes: `KnowledgeRetriever.search`(真实链路)、`evidence_confidence`
- Produces: CLI;把选中的阈值打印出来(**不自动改配置** —— 由人写回)

> **为什么要标定而不是拍**:阈值决定了"多少问题会被拦下送进飞轮"。
> 拍高了 → 正常问题被拦成兜底(用户体验直接变差);拍低了 → 该拦的没拦,
> 飞轮永远吃不饱。**两个方向都能从评估集上量出来。**

- [ ] **Step 1: 写脚本**

```python
"""在 evals/测试集.md 上标定 evidence_confidence_threshold。

用法:
    .venv/Scripts/python.exe scripts/calibrate_evidence.py
    .venv/Scripts/python.exe scripts/calibrate_evidence.py --save   # 只打印,不写

口径:
  「应拒答=是」的桶(D_absent,60 条)**拦下**才算对 ⇒ 拦截率
  其余桶(A/B/C/E,240 条)**放行**才算对 ⇒ 误杀率
  在「拦截率 ≥ 0.6」的阈值里挑**误杀率最低**的;若无解,退化为
  「误杀率 ≤ 0.05 里拦截率最高」并**响亮记账**(spec §4.2)。

⚠️ 前置:Milvus + BGE-M3 + 真实 .env。它走**真实检索链路**
(`KnowledgeRetriever`),不是 ch04 那个脚本的自建复刻 —— 标定的必须是
**线上那条路**。
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.config import get_settings
from app.db.base import get_sessionmaker
from app.kb.evidence import evidence_confidence
from app.retrieval.search import KnowledgeRetriever
from app.tools.registry import build_retriever

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / "evals" / "测试集.md"


def emit(text: str) -> None:
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()


def load_cases() -> list[dict]:
    rows = []
    for line in CASES.read_text(encoding="utf-8").splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        rows.append({"bucket": parts[1], "query": parts[2],
                     "should_refuse": parts[5] == "是"})
    return rows


async def main() -> None:
    ap = argparse.ArgumentParser(description="置信度阈值标定")
    ap.add_argument("--grid", default="0.05,0.95,0.025", help="起,止,步长")
    args = ap.parse_args()

    start, stop, step = (float(x) for x in args.grid.split(","))
    settings = get_settings()
    cases = load_cases()
    emit(f"用例 {len(cases)} 条(应拒答 {sum(c['should_refuse'] for c in cases)} 条)")

    maker = get_sessionmaker()
    async with maker() as session:
        retriever: KnowledgeRetriever = build_retriever(session, settings)
        scores: list[tuple[bool, float]] = []
        for i, case in enumerate(cases, 1):
            chunks = await retriever.search(case["query"])
            scores.append((case["should_refuse"],
                           evidence_confidence(chunks, settings=settings)))
            if i % 50 == 0:
                emit(f"  …{i}/{len(cases)}")

    emit(f"\n{'阈值':>8}{'拦截率':>10}{'误杀率':>10}{'拦截数':>8}{'误杀数':>8}")
    emit("-" * 46)
    best = None
    t = start
    grid = []
    while t <= stop + 1e-9:
        blocked = [(ref, s) for ref, s in scores if s < t]
        hit = sum(1 for ref, _ in blocked if ref)
        miss_block = sum(1 for ref, _ in blocked if not ref)
        n_ref = sum(1 for ref, _ in scores if ref)
        n_ok = len(scores) - n_ref
        catch = hit / n_ref if n_ref else 0.0
        kill = miss_block / n_ok if n_ok else 0.0
        grid.append({"t": round(t, 4), "catch": catch, "kill": kill})
        if catch >= 0.6 and (best is None or kill < best["kill"]):
            best = grid[-1]
        t += step
    for row in grid:
        emit(f"{row['t']:>8.3f}{row['catch']:>10.3f}{row['kill']:>10.3f}")

    if best is None:
        cand = [r for r in grid if r["kill"] <= 0.05]
        best = max(cand, key=lambda r: r["catch"]) if cand else max(grid, key=lambda r: r["catch"])
        emit("\n!!! 没有"拦截率≥0.6"的解 —— 退化为"误杀率≤0.05 里拦截率最高"。")
        emit("!!! 这条退化**必须记进 spec §15 与 dev-notes**,不许当成正常结果。")

    emit(f"\n选中阈值:{best['t']}(拦截率 {best['catch']:.3f},误杀率 {best['kill']:.3f})")
    emit("请人工确认后写回 app/config.py 的 evidence_confidence_threshold。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
```

- [ ] **Step 2: 真机跑**

Run: `.venv/Scripts/python.exe scripts/calibrate_evidence.py`
Expected: 打印一张阈值表 + 选中一个值。

- [ ] **Step 3: 写回配置并记账**

把选中的值写进 `app/config.py` 的 `evidence_confidence_threshold`,并把
**整张表的关键读数 + 选中理由**写进 `dev-notes/ch09.md` 与 spec §15。

> **若走了退化分支**,那一段要写清楚"没找到满足拦截率≥0.6 的解",并把整张表贴出来 ——
> 那是本章配置里唯一一个**没达到设计目标**的数。

- [ ] **Step 4: 跑全量非 db,确认改默认值没碰坏测试**

Run: `.venv/Scripts/python.exe -m pytest -m "not db"`
Expected: 全绿。**若有测试因为默认阈值变了而红**,先看它是不是在断"默认值 = 0.42" ——
那种测试应当改成**显式传阈值**,而不是把默认值改回去。

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_evidence.py app/config.py dev-notes/ch09.md \
        docs/superpowers/specs/2026-09-23-ecommerce-cs-ch09-observe-flywheel-design.md
git commit -m "ch09 T8: 置信度阈值标定(evals/测试集.md 300 条)并写回配置"
```

---

## Task 9: `app/agent/json_stream.py` —— 三态增量解码器

**Files:**
- Create: `app/agent/json_stream.py`
- Test: `tests/test_json_stream.py`

**Interfaces:**
- Produces:
  - `Event(kind: str, value: object = None)` —— `kind ∈ {"useful","confidence","answer_delta","done","violation"}`
  - `JsonAnswerDecoder` —— `feed(fragment) -> list[Event]`;属性 `mode`(`"lead"|"protocol"|"plain"`)、`useful`、`confidence`、`answer`、`done`、`raw`、`violation`

> **这是全章最该被穷举测试的一处。** spec §5.3 列了 13 条边界;
> `answer` 的值必然含中文,而 `中` 被切成 `\u4e` + `2d` 是**必现**的。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_json_stream.py
"""增量 JSON 解码器 —— 13 条边界 + 穷举切点 fuzz。

判据(本仓血的教训):**构造输入时先算一遍这个输入真的会走到那条分支吗**。
例如"转义跨 chunk"这条,必须把 `\\u4e2d` 切开,不能整段喂进去。
"""

import json

import pytest

from app.agent.json_stream import JsonAnswerDecoder


def _feed_all(dec, fragments):
    out = []
    for f in fragments:
        out.extend(dec.feed(f))
    return out


def _deltas(events):
    return "".join(e.value for e in events if e.kind == "answer_delta")


# ---- 1–6:正常路径与跨 chunk 的转义 ----------------------------------------

def test_1_unicode_escape_split_across_chunks():
    d = JsonAnswerDecoder()
    ev = _feed_all(d, ['{"useful": true, "confidence": 0.9, "answer": "a\\u4e', '2d b"}'])
    assert d.answer == "a中 b"
    assert _deltas(ev) == "a中 b"


def test_2_backslash_split_across_chunks():
    d = JsonAnswerDecoder()
    d.feed('{"useful": true, "confidence": 0.9, "answer": "a\\')
    d.feed('"b"}')
    assert d.answer == 'a"b'


def test_3_structural_chars_inside_answer_are_not_structure():
    d = JsonAnswerDecoder()
    _feed_all(d, ['{"useful": true, "confidence": 0.9, ',
                  '"answer": "含 { } , : 与 \\"引号\\" 的文本"}'])
    assert d.answer == '含 { } , : 与 "引号" 的文本'


def test_4_answer_is_last_key_and_closing_brace_is_separate():
    d = JsonAnswerDecoder()
    ev = _feed_all(d, ['{"useful": true, "confidence": 0.9, "answer": "你好"',
                       '}'])
    assert d.done is True
    assert d.answer == "你好"
    assert [e.kind for e in ev][-1] == "done"


def test_5_float_split_across_chunks():
    d = JsonAnswerDecoder()
    _feed_all(d, ['{"useful": true, "confidence": 0.', '85, "answer": "x"}'])
    assert d.confidence == pytest.approx(0.85)


def test_6_boolean_split_across_chunks():
    d = JsonAnswerDecoder()
    _feed_all(d, ['{"useful": tru', 'e, "answer": "x"}'])
    assert d.useful is True


# ---- 8/13:lead 态的退出 ----------------------------------------------------

def test_8_leading_text_exits_to_plain():
    """首个非空白不是 `{` ⇒ plain(今天的行为),**不是** violation。"""
    d = JsonAnswerDecoder()
    ev = _feed_all(d, ['好的,', '我来回答:退货政策是…'])
    assert d.mode == "plain"
    assert _deltas(ev) == "好的,我来回答:退货政策是…"
    assert not any(e.kind == "violation" for e in ev)


def test_13_whitespace_only_stays_in_lead():
    d = JsonAnswerDecoder()
    ev = _feed_all(d, ["", "  ", "\n"])
    assert d.mode == "lead"
    assert ev == []


def test_12_json_fence_falls_back_to_plain():
    """```json 围栏 ⇒ plain(刻意不剥壳:剥壳要猜模型意图,剥错了更糟)。"""
    d = JsonAnswerDecoder()
    ev = _feed_all(d, ['```json\n{"useful": true, "answer": "x"}```'])
    assert d.mode == "plain"
    assert d.useful is None


# ---- 7:protocol 态下的首键违规 --------------------------------------------

def test_7_wrong_first_key_is_violation_and_nothing_emitted():
    d = JsonAnswerDecoder()
    ev = _feed_all(d, ['{"answer": "先答后判"}'])
    assert d.mode == "protocol"
    assert [e for e in ev if e.kind == "violation"]
    assert _deltas(ev) == "", "违规时一个 answer_delta 都不许出去"


# ---- 9/10/11:终止条件 ------------------------------------------------------

def test_9_incomplete_stream_is_not_done():
    d = JsonAnswerDecoder()
    _feed_all(d, ['{"useful": true, "answer": "半截'])
    assert d.done is False


def test_10_useful_false_with_empty_answer_emits_nothing():
    d = JsonAnswerDecoder()
    ev = _feed_all(d, ['{"useful": false, "confidence": 0.1, "answer": ""}'])
    assert d.useful is False
    assert _deltas(ev) == ""


def test_11_useful_false_with_nonempty_answer_stops_immediately():
    """⚠️ **这条是本任务最重要的用例**。

    协议要求 useful=false 时 answer 为空串,所以"停吐"这条正常路径**没有可吐的东西**
    ⇒ 用正常输入测它**等于没测**。必须构造**违规流**:useful 为 false 但 answer 非空,
    才能真的验到"停吐"这一步。
    """
    d = JsonAnswerDecoder()
    ev = _feed_all(d, [
        '{"useful": false, ',
        '"answer": "这段文本一个字都不该出现在用户面前"}',
    ])
    assert d.useful is False
    assert _deltas(ev) == ""


# ---- fuzz:穷举切点 ---------------------------------------------------------

FULL = json.dumps(
    {"useful": True, "confidence": 0.75,
     "answer": '中文答案:含 "引号"、\\反斜杠、\n换行、中文与 { } 结构符'},
    ensure_ascii=False,
)


def _run(fragments):
    d = JsonAnswerDecoder()
    ev = []
    for f in fragments:
        ev.extend(d.feed(f))
    return d, ev


def test_fuzz_every_two_way_split_matches_unsplit():
    """把完整 JSON 按**每一个可能的切点**切成两段 —— 结论必须与不切时逐字节相同。"""
    base_d, base_ev = _run([FULL])
    for i in range(1, len(FULL)):
        d, ev = _run([FULL[:i], FULL[i:]])
        assert d.answer == base_d.answer, f"切点 {i} 的 answer 不一致"
        assert d.useful == base_d.useful
        assert d.confidence == base_d.confidence
        assert _deltas(ev) == _deltas(base_ev), f"切点 {i} 的增量流不一致"
        assert d.done is True


def test_fuzz_escaped_unicode_split_at_every_point():
    """专钉 `\\uXXXX` 与 `\\` 的**每一个**内部切点。"""
    payload = '{"useful": true, "confidence": 0.5, "answer": "中\\u4e2d\\"x\\\\y"}'
    base_d, _ = _run([payload])
    for i in range(1, len(payload)):
        d, _ = _run([payload[:i], payload[i:]])
        assert d.answer == base_d.answer, f"切点 {i}: {d.answer!r} != {base_d.answer!r}"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe pytest tests/test_json_stream.py`
Expected: FAIL —`No module named 'app.agent.json_stream'`

- [ ] **Step 3: 写实现**

```python
# app/agent/json_stream.py
"""增量 JSON 解码器 —— 边收 token 边解出 useful / confidence / answer。

ch09 spec §5.3。**纯状态机**:输入 str 片段序列,输出事件序列,
零 IO、零依赖、完全确定性 ⇒ 可以用 chunk 边界 fuzz 穷举钉死。

**为什么自写而不用 partial-json-parser / ijson / json-stream**:
本章只需要两件事 —— ①从头部取出 `useful`;②把 `answer` 的值边收边吐。
而 `answer` 的值**必然含中文**,必然出现 `\\"` `\\\\` `\\n` `\\uXXXX`,而
**转义序列跨 chunk 边界**恰恰是这类通用库最难验的地方(一个 `中` 被切成
`\\u4e` + `2d` 是**必现**的,不是边角)。自写的东西能被穷举切点 fuzz 钉死,
通用库只能信它。

三态(spec §5.4):
    lead     还没定形态。首个**非空白**字符是 `{` ⇒ protocol;不是 ⇒ plain。
    protocol 在解协议对象。首键不是 `useful` ⇒ violation(调用方据此重试一次)。
    plain    **今天的行为**:每个片段原样透出。它存在的意义是让协议违规
             **不产生任何回归**。
"""

from dataclasses import dataclass

#: 三态的取值,别写成裸字符串(调用方与测试都要引用)
LEAD = "lead"
PROTOCOL = "protocol"
PLAIN = "plain"


@dataclass(frozen=True)
class Event:
    kind: str            # useful | confidence | answer_delta | done | violation
    value: object = None


class JsonAnswerDecoder:
    """⚠️ **追加**语义:`feed` 按到达顺序喂,不是替换。

    `answer_delta` 的 `value` 是**本次新增**的那段文本,不是累计。
    """

    def __init__(self) -> None:
        self._mode = LEAD
        self._raw = ""
        self._lead = ""              # lead 态攒下的字节
        self._useful: bool | None = None
        self._confidence: float | None = None
        self._answer = ""
        self._done = False
        self._violation: str | None = None
        # 协议态下的扫描进度
        self._key = ""               # 已解出的键名
        self._seen_keys: list[str] = []
        self._value_buf = ""         # 当前正在读的值(非字符串)
        self._in_string = False
        self._escape = False
        self._unicode_hex = ""       # \uXXXX 收集中
        self._current_key: str | None = None
        self._listening_answer = False

    # ---- 只读视图 ---------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def useful(self) -> bool | None:
        return self._useful

    @property
    def confidence(self) -> float | None:
        return self._confidence

    @property
    def answer(self) -> str:
        return self._answer

    @property
    def done(self) -> bool:
        return self._done

    @property
    def raw(self) -> str:
        return self._raw

    @property
    def violation(self) -> str | None:
        return self._violation

    # ---- 主入口 -----------------------------------------------------------
    def feed(self, fragment: str) -> list[Event]:
        if not fragment or self._done or self._violation:
            return []
        self._raw += fragment
        if self._mode == LEAD:
            return self._feed_lead(fragment)
        if self._mode == PLAIN:
            return [Event("answer_delta", fragment)]
        return self._feed_protocol(fragment)

    # ---- lead -------------------------------------------------------------
    def _feed_lead(self, fragment: str) -> list[Event]:
        self._lead += fragment
        stripped = self._lead.lstrip()
        if not stripped:
            return []                     # 只有空白:留在 lead
        if stripped[0] != "{":
            # 不是协议对象 ⇒ 降级成今天的行为,把**攒下的**与**这一片**一起吐出去
            self._mode = PLAIN
            out = [Event("answer_delta", self._lead)]
            self._lead = ""
            return out
        self._mode = PROTOCOL
        pending, self._lead = self._lead, ""
        return self._feed_protocol(pending)

    # ---- protocol ---------------------------------------------------------
    def _feed_protocol(self, fragment: str) -> list[Event]:
        events: list[Event] = []
        for ch in fragment:
            self._consume(ch)
            if self._violation or self._done:
                break
            if self._current_key == "answer" and ch not in ('"', "\\"):
                pass
        if self._mode == PLAIN:            # 不可能,但别让它悄悄坏
            return [Event("answer_delta", fragment)]
        return events

    def _consume(self, ch: str) -> None:  # noqa: C901 —— 状态机本来就长
        # ---- 字符串内(含转义)----
        if self._in_string:
            if self._escape:
                self._escape = False
                if ch == "u":
                    self._unicode_hex = ""
                    return
                self._emit_char(self._unescape_simple(ch))
                return
            if ch == "\\":
                self._escape = True
                return
            if ch == '"':
                self._in_string = False
                if self._current_key == "answer":
                    self._listening_answer = False
                    events_done = True
                return
            self._emit_char(ch)
            return

        # ---- 字符串外 ----
        if ch == '"':
            self._in_string = True
            self._key = ""
            self._listening_answer = self._current_key == "answer" and self._answer_started
            return
        if self._unicode_hex:
            pass
        ...
```

> ⚠️ **上面这块到 `_consume` 为止是骨架** —— 状态机的完整实现在**实现时按测试驱动写**:
> 先把 §5.3 的 13 条边界 + fuzz 用例写好、逐条跑红、逐条补状态转移。
> **不要照抄这份骨架里没写完的部分**;`_consume` 必须写到**全部用例通过**。
> 判据:`test_fuzz_every_two_way_split_matches_unsplit` 对**每一个切点**都绿。

- [ ] **Step 4: 按测试逐条补全 `_consume`**

要点(实现时对应用例):
- 协议态下**最先出现的键名**必须是 `useful`,否则 `violation="first_key=<键名>"`,**并停止吐任何东西**。
- `answer` 是字符串 ⇒ 其值里的每个字符都进 `_answer` 并产 `answer_delta`;
  `\uXXXX` 要**攒够 4 位**再解码(`chr(int(hex,16))`)。
- `useful` / `confidence` 是非字符串值 ⇒ 用 `_value_buf` 攒到 `,` 或 `}` 再解析。
- 见 `}` 且 `useful` 已解出 ⇒ `done=True`。
- **`useful` 解出为 `False` 的那一刻**设置 `self._done = True` 并返回 ——
  §5.3 边界 11 靠它保证"停吐"。

Run: `.venv/Scripts/python.exe -m pytest tests/test_json_stream.py`
Expected: PASS(全部)

- [ ] **Step 5: 变异验证(必须做)**

把 `useful is False ⇒ done=True` 那两行**注释掉**再跑
`test_11_useful_false_with_nonempty_answer_stops_immediately`。
Expected: **红**。改回来。

- [ ] **Step 6: Commit**

```bash
git add app/agent/json_stream.py tests/test_json_stream.py
git commit -m "ch09 T9: 三态增量 JSON 解码器(13 条边界 + 穷举切点 fuzz)"
```

---

## Task 10: `agent` 节点的知识轮挂协议

**Files:**
- Modify: `app/agent/nodes.py:242-454`(`make_agent_node` 与 `_stream_round`)
- Test: `tests/test_agent_protocol.py`

**Interfaces:**
- Consumes: T9 的 `JsonAnswerDecoder`、`Event`;T5 的 `record_low_confidence(evidence_snapshot=…)`
- Produces: `agent` 节点的知识轮:`useful=false` ⇒ 兜底话术 + (仅知识类)落池

> ⚠️ **本任务动的是全仓最复杂的一段代码。** 改动面**只有两处**:
> ①`msgs` 末尾多一条协议消息;②文本流出多一次解码。
> **工具绑定、工具执行、`pending_write`、`turn_messages`、每轮清零全不碰。**

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_agent_protocol.py
"""agent 节点的协议路径。

⚠️ 本文件里最不能省的是 **test_business_turn_is_byte_identical** ——
它是「业务轮零回归」这句话的**唯一**证据。
"""

import pytest

from app.agent.nodes import make_agent_node
from app.config import Settings


def _settings(**over):
    base = {
        "openai_base_url": "http://x", "openai_api_key": "k", "openai_model": "m",
        "database_url": "mysql://x", "max_agent_steps": 3,
        "agent_token_budget": 999999,
    }
    return Settings(**{**base, **over}, _env_file=None)


class _Chunk:
    """假 chunk。⚠️ `type` 键是硬约束:`BaseTool.ainvoke` 只看它。"""

    def __init__(self, text="", tool_calls=None):
        self.text = text
        self.content = text
        self.tool_calls = tool_calls or []
        self.usage_metadata = None

    def __add__(self, other):
        return self


class _StreamingModel:
    def __init__(self, fragments):
        self._fragments = fragments
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def astream(self, msgs, **kw):
        self.calls += 1
        for f in self._fragments:
            yield _Chunk(text=f)


@pytest.mark.anyio
async def test_knowledge_turn_streams_answer_only_after_useful():
    """useful 之前一个 token 都不许出去。"""
    frames = []
    model = _StreamingModel(['{"useful": tru', 'e, "confidence": 0.9, "answer": "退货',
                             '政策是 7 天"}'])
    node = make_agent_node(model=model, tools=[], registry={}, settings=_settings(),
                           emit=frames.append, context_budget=None)
    out = await node({"conversation_id": "c1", "resolved_input": "退货政策",
                      "intent": "商品咨询", "history": [], "evidence": []})
    texts = [f["text"] for f in frames if f.get("frame") == "token"]
    assert "".join(texts) == "退货政策是 7 天"
    assert out["reply"] == "退货政策是 7 天"
    # 协议字节不许泄漏给用户
    assert "useful" not in "".join(texts)


@pytest.mark.anyio
async def test_business_turn_is_byte_identical():
    """**业务轮的输出与今天逐字节相同** —— 这是「零回归」唯一的硬证据。

    比对的是**帧序列**,不是最终 reply:两者的 reply 可能一样,
    但 token 帧的**切法**不同就是回归。
    """
    model = _StreamingModel(["您的", "订单", "已发货"])
    frames = []
    node = make_agent_node(model=model, tools=[], registry={}, settings=_settings(),
                           emit=frames.append, context_budget=None)
    await node({"conversation_id": "c1", "resolved_input": "我的订单到哪了",
                "intent": "物流", "history": [], "evidence": []})
    assert [f["text"] for f in frames if f.get("frame") == "token"] == ["您的", "订单", "已发货"]
```

再加三条(结构与上面同款,此处给出断言要点):

```python
# 3) useful=false 且 intent=商品咨询 ⇒ 兜底话术 + 落池 entry_point="生成自评"
#    断言:发出去的是兜底话术;sess.added[0].entry_point == "生成自评";
#          sess.added[0].evidence_snapshot 里有 chunks
# 4) useful=false 但 intent=物流 ⇒ 同样兜底话术,**sess.added == []**(不落池)
# 5) 首字符不是 "{"(模型加了前言)⇒ 降级 plain:
#    断言:帧序列 == 逐片原样;"agent:protocol_violation" in out["trace"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_protocol.py`
Expected: **红** —— 现在每个片段都直接 emit,`{"useful": tru` 会被当 token 发出去。

- [ ] **Step 3: 改 `_stream_round`**

```python
    PROTOCOL_MESSAGE = (
        "需要调用工具时照常调用。**不需要工具时,只输出一个 JSON 对象**,"
        "不要任何前言、不要 ``` 围栏。字段与顺序**必须**是:\n"
        "1. useful:布尔值。上面的证据足以回答用户问题为 true,不足为 false。\n"
        "2. confidence:0 到 1 的小数。\n"
        "3. answer:字符串。**useful 为 false 时必须是空字符串**;"
        "证据不足时不得编造、不得用常识补。\n"
        "先判定,后作答。"
    )

    async def _stream_round(target, msgs, parts, *, decode: bool):
        acc = None
        used = 0
        dec = JsonAnswerDecoder() if decode else None
        async for chunk in target.astream(msgs):
            acc = chunk if acc is None else acc + chunk
            used += _total_tokens(chunk)
            if not chunk.text:
                continue
            if dec is None:
                parts.append(chunk.text)
                emit({"frame": "token", "text": chunk.text})
                continue
            for ev in dec.feed(chunk.text):
                if ev.kind == "answer_delta":
                    parts.append(ev.value)
                    emit({"frame": "token", "text": ev.value})
        return acc, used, dec
```

- [ ] **Step 4: 改 `agent_node`**

```python
        # ch09:知识轮挂协议。**业务轮一个字不加** —— 需求说的是「知识不够答」,
        # 而落池也只该发生在知识类(spec §5.5)。
        is_knowledge = state.get("intent") == "商品咨询"
        if is_knowledge:
            msgs = [*msgs, SystemMessage(content=PROTOCOL_MESSAGE)]
```

三处 `_stream_round` 调用点都传 `decode=is_knowledge`。**续跑路径也要带**
(它是同一轮的最终作答)。

轮末处理:

```python
        def _finish_protocol(dec) -> tuple[bool, str]:
            """返回 (是否已判不可用, 要发的兜底话术)。"""
            if dec is None:
                return False, ""
            if dec.useful is False:
                return True, FALLBACK_TEXT
            if dec.violation or not dec.raw.lstrip().startswith("{"):
                trace.append("agent:protocol_violation")
            return False, ""
```

- 若 `useful is False`:把这一轮的 `parts` **清空**、改成发兜底话术的 token 帧、
  并以 `intent == "商品咨询"` 为条件调 `record_low_confidence(entry_point="生成自评",
  evidence_snapshot=…, reject_reason=…)`。
- **快照**取自 `state["evidence"]`,按 `settings.snapshot_top_n` /
  `snapshot_answer_chars` 截:

```python
def _snapshot(evidence, *, settings) -> list[dict]:
    return [
        {"chunk_id": c.get("chunk_id"), "score": c.get("score"),
         "section_path": c.get("section_path"),
         "answer": (c.get("answer") or "")[: settings.snapshot_answer_chars]}
        for c in (evidence or [])[: settings.snapshot_top_n]
    ]
```

> ⚠️ **`_snapshot` 在 T10 与 T12 里长得几乎一样,但吃的东西形状不同,别复制粘贴**:
> - **T10(节点内)**:吃 `state["evidence"]`,元素是**已经序列化进 state 的 dict**
>   ⇒ 用 `c.get("chunk_id")`。
> - **T12(端点内)**:吃 `retriever.search()` 刚返回的 **`RetrievedChunk` 对象**
>   ⇒ 用 `c.chunk_id`。
>
> `state["evidence"]` 的确切键名由 `retrieve_knowledge` 决定 —— **先去读它**再对齐,
> 别按猜的写。(两处都按同一组键落库:chunk_id / score / section_path / answer。)

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_protocol.py`
Expected: PASS

- [ ] **Step 6: 跑全量非 db —— 看有没有既有测试因为 `agent` 变了而红**

Run: `.venv/Scripts/python.exe -m pytest -m "not db"`
Expected: 全绿。**红了先看那条用例的 intent**:若它是商品咨询意图,它本来就该改形态;
若它是业务意图而红了,**那是真的回归**,回去修实现,不要改测试。

- [ ] **Step 7: 真机量违约率(§12.3-1)**

起服务,连发 **10 次**商品咨询问题,数 `log/app.log` 里 `agent:protocol_violation` 出现几次。
**把这个数如实写进 `dev-notes/ch09.md`** —— 它是 spec §5.7-1 那条「没有硬保证」的唯一证据。

- [ ] **Step 8: Commit**

```bash
git add app/agent/nodes.py tests/test_agent_protocol.py dev-notes/ch09.md
git commit -m "ch09 T10: agent 知识轮挂 JSON 协议(业务轮零回归)"
```

---

## Task 11: `record_low_confidence` 支持快照 + 幂等

**Files:**
- Modify: `app/kb/assess.py:63-74`
- Test: `tests/test_kb_assess.py`(扩)

**Interfaces:**
- Produces: `record_low_confidence(session, *, question, source_conversation_id, entry_point, reject_reason, evidence_snapshot=None) -> int`(**返回新行 id**)

- [ ] **Step 1: 写失败的测试**

```python
@pytest.mark.anyio
async def test_record_low_confidence_returns_id_and_stores_snapshot():
    ...
    rid = await record_low_confidence(
        session, question="q", source_conversation_id="c1",
        entry_point="生成自评", reject_reason="r",
        evidence_snapshot=[{"chunk_id": 1, "score": 0.5, "answer": "a"}],
    )
    assert isinstance(rid, int) and rid > 0
    row = (await session.execute(
        text("SELECT evidence_snapshot FROM low_confidence_questions WHERE id=:i"),
        {"i": rid})).scalar_one()
    assert row == [{"chunk_id": 1, "score": 0.5, "answer": "a"}]
    ...
```

- [ ] **Step 2: 跑测试确认失败** → `TypeError: unexpected keyword argument 'evidence_snapshot'`

- [ ] **Step 3: 改实现**

```python
async def record_low_confidence(
    session, *, question: str, source_conversation_id: str | None,
    entry_point: str, reject_reason: str,
    evidence_snapshot: list | None = None,
) -> int:
    """问题落低置信度池。**返回新行 id**(ch09:飞轮与测试都要它)。

    `evidence_snapshot` 是 ch09 加的:落池当轮的召回片段(Top-N 的 id/得分/原文),
    审核人靠它判「知识库真缺这块,还是有但没检到」。
    """
    row = LowConfidenceQuestion(
        question=question, source_conversation_id=source_conversation_id,
        entry_point=entry_point, reject_reason=reject_reason,
        evidence_snapshot=evidence_snapshot,
    )
    session.add(row)
    await session.commit()
    return row.id
```

- [ ] **Step 4: 跑测试** → PASS

- [ ] **Step 5: Commit**

```bash
git add app/kb/assess.py tests/test_kb_assess.py
git commit -m "ch09 T11: record_low_confidence 支持召回片段快照并返回行 id"
```

---

## Task 12: `POST /api/feedback` —— 👎 落池

**Files:**
- Create: `app/api/feedback.py`
- Modify: `app/main.py`(include)
- Test: `tests/test_api_feedback.py`

**Interfaces:**
- Consumes: T6 的 `evidence_confidence`;T11 的 `record_low_confidence`;`build_retriever`
- Produces: `POST /api/feedback`,请求 `{conversation_id, question, message_id?, value}`

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_api_feedback.py
"""👎 落池。up 不落;down 落且尽力回捞召回片段;重复 down 幂等。"""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings


def _settings(**over):
    base = {"openai_base_url": "http://x", "openai_api_key": "k",
            "openai_model": "m", "database_url": "mysql://x"}
    return Settings(**{**base, **over}, _env_file=None)


@pytest.mark.anyio
async def test_down_writes_pool_with_user_feedback_entry_point(...):
    ...
    assert resp.status_code == 200
    rows = ...   # SELECT ... WHERE source_conversation_id = :c
    assert len(rows) == 1
    assert rows[0]["entry_point"] == "用户反馈"
```

三条用例:`up` 不落池(池里 0 行)、`down` 落 1 行且 `entry_point="用户反馈"`、
**同一个 `message_id` 连发两次 `down` 仍然只有 1 行**。

> **断言一律按 `source_conversation_id` 过滤** —— 池子是共享的,不过滤会被上一次
> 运行留下的行污染(本仓已记过的形态)。

- [ ] **Step 2: 跑测试确认失败** → 404

- [ ] **Step 3: 写实现**

```python
# app/api/feedback.py
"""用户满意度反馈(ch09 spec §6)。

`up` 只记日志不落池;`down` 把**该轮的用户问题**送进低置信度池,并
**按该轮问题重跑一次检索尽力回捞**召回片段。

⚠️ **回捞是「重跑」不是「当轮」** —— 本仓**没有任何地方持久化每轮的检索结果**
(`evidence` 只在图 state 里,intent 也没落库)。所以 `evidence_snapshot` 是
**事后重跑**的结果。为什么这样反而更好用:重跑召得到 ⇒「知识库有、当时没检到」;
召不到 ⇒「真缺这块」—— 正是审核人要判的那件事。
语义偏差已记在 spec §6.2。
"""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.config import Settings, get_settings
from app.db.models import LowConfidenceQuestion
from app.db.session import get_session
from app.kb.assess import record_low_confidence
from app.retrieval.search import KnowledgeRetriever
from app.tools.registry import build_retriever

logger = logging.getLogger(__name__)
router = APIRouter()


class FeedbackRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=32)
    question: str = Field(min_length=1, max_length=2000)
    message_id: int | None = None
    value: str = Field(pattern="^(up|down)$")


def _snapshot(chunks, *, settings: Settings) -> list[dict] | None:
    if not chunks:
        return None
    return [
        {"chunk_id": c.chunk_id, "score": round(c.score, 4),
         "section_path": c.section_path,
         "answer": (c.answer or "")[: settings.snapshot_answer_chars]}
        for c in chunks[: settings.snapshot_top_n]
    ]


@router.post("/api/feedback")
async def post_feedback(
    body: FeedbackRequest,
    settings: Settings = Depends(get_settings),
    session=Depends(get_session),
) -> dict:
    if body.value == "up":
        return {"ok": True, "pooled": False}

    # 幂等:同一条消息的 down 重复提交只落一行(池子上没有唯一键,
    # 而查重是**语义**判断,不该靠唯一键 —— spec §7.1)。
    existing = (
        await session.execute(
            select(LowConfidenceQuestion.id).where(
                LowConfidenceQuestion.source_conversation_id == body.conversation_id,
                LowConfidenceQuestion.entry_point == "用户反馈",
                LowConfidenceQuestion.question == body.question,
            )
        )
    ).first()
    if existing is not None:
        return {"ok": True, "pooled": False, "reason": "already_pooled"}

    # 尽力回捞:重跑一次真实检索。**召不到就留空**(那正是"真缺这块"的信号)。
    snapshot = None
    try:
        retriever: KnowledgeRetriever = build_retriever(session, settings)
        chunks = await retriever.search(body.question)
        snapshot = _snapshot(chunks, settings=settings)
    except Exception:  # noqa: BLE001
        # 检索挂掉不许拦住落池 —— 落池才是这个端点的职责。
        # ⚠️ 但**必须响亮地记**:快照为空与"检索故障"在数据上长得一样。
        logger.warning("feedback 回捞召回片段失败,快照留空", exc_info=True)

    await record_low_confidence(
        session, question=body.question,
        source_conversation_id=body.conversation_id,
        entry_point="用户反馈",
        reject_reason="用户点了 👎(未解决)",
        evidence_snapshot=snapshot,
    )
    return {"ok": True, "pooled": True, "snapshot_chunks": len(snapshot or [])}
```

`app/main.py` 加 `app.include_router(feedback.router)`(**必须在 `mount("/")` 之前**)。

- [ ] **Step 4: 跑测试** → PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/feedback.py app/main.py tests/test_api_feedback.py
git commit -m "ch09 T12: POST /api/feedback(👎 落池 + 尽力回捞召回片段)"
```

---

## Task 13: 飞轮三步 —— `normalize` / `dedupe` / `pipeline`

**Files:**
- Create: `app/flywheel/__init__.py`、`app/flywheel/normalize.py`、`app/flywheel/dedupe.py`、`app/flywheel/pipeline.py`
- Create: `evals/flywheel_cases.jsonl`
- Test: `tests/test_flywheel_pipeline.py`(db)

**Interfaces:**
- Produces:
  - `async def normalize_question(question: str, *, model) -> dict` → `{"standard_question": str, "example_answer": str}`
  - `async def find_duplicate(candidate: str, pending: Sequence[ReviewQueue], *, model) -> ReviewQueue | None`
  - `async def run_flywheel(*, session, model, batch_size: int) -> dict` → `{"processed","merged","created","failed"}`

- [ ] **Step 1: 写标注样例 `evals/flywheel_cases.jsonl`**

每行一条:

```json
{"raw": "我买的那个猫砂盆寄到新疆要不要另外加钱啊", "expect_contains": ["新疆", "运费"], "expect_not_contains": ["猫砂盆寄到"], "max_chars": 60, "note": "口语 → FAQ 式;要点在'偏远地区运费',不是复述原话"}
```

**这份样例是 `normalize` 的"评估集"**(纯 Prompt,按项目规矩不套 TDD,用标注样例验)。至少 8 条,覆盖:
口语长句、带错别字、含指代(「它」「那个」)、已经是标准问法(应当基本不变)、纯业务问题(**不该**被标准化成知识问题)。

- [ ] **Step 2: 写 `normalize.py`**

```python
# app/flywheel/normalize.py
"""口语原话 → 标准 FAQ 式问题 + 示例答案。

**纯 Prompt ⇒ 按项目规矩不套 TDD**,用 `evals/flywheel_cases.jsonl` 的标注样例验证
(见 `scripts/run_flywheel_eval.py`,与 `run_resolve_eval.py` 同款)。

⚠️ 提示词里必须出现字面 `JSON` 字样(json_mode 硬约束,spec §2.3),
且**不得使用裸花括号**(ChatPromptTemplate 按 f-string 解析)。
"""
```

函数体:与 `app/kb/assess.py` 的 `assess_sufficiency` 同款(structured output + json_mode),
解析失败**退化为 `{"standard_question": question, "example_answer": ""}`**
(原话入库 = 至少不丢问题,而不是抛出去让整批挂掉)。

- [ ] **Step 3: 写 `dedupe.py`**

模型判「candidate 与 pending 里的某一条是不是同一个意思」,输出 `{"match_index": int}`(`-1` = 都不匹配)。
**解析失败按「不匹配」处理**(宁可多建一行,也不要把两个不同问题并成一个)。

- [ ] **Step 4: 写 `pipeline.py` + 测试**

```python
# tests/test_flywheel_pipeline.py
"""飞轮编排:幂等、命中累加、逐行失败不拖垮整批。"""

import pytest

pytestmark = pytest.mark.db


@pytest.mark.anyio
async def test_second_run_is_a_noop():
    """跑两遍,review_queue 行数不变 —— 幂等由 matched_review_id IS NULL 单独保证。"""
    ...


@pytest.mark.anyio
async def test_duplicate_hit_accumulates_occurrences_and_backfills_matched_review_id():
    ...


@pytest.mark.anyio
async def test_one_row_failing_does_not_kill_the_batch():
    """模型对第 2 行抛异常 ⇒ 第 1、3 行照常入队,第 2 行 matched_review_id 仍是 NULL。"""
    ...
```

实现要点:

```python
async def run_flywheel(*, session, model, batch_size: int) -> dict:
    rows = (await session.execute(
        select(LowConfidenceQuestion)
        .where(LowConfidenceQuestion.matched_review_id.is_(None))
        .order_by(LowConfidenceQuestion.id)
        .limit(batch_size)
    )).scalars().all()

    stats = {"processed": 0, "merged": 0, "created": 0, "failed": 0}
    for row in rows:
        stats["processed"] += 1
        try:
            norm = await normalize_question(row.question, model=model)
            pending = (await session.execute(
                select(ReviewQueue).where(ReviewQueue.status == "pending")
            )).scalars().all()
            hit = await find_duplicate(norm["standard_question"], pending, model=model)
            if hit is not None:
                hit.occurrences += 1
                row.matched_review_id = hit.id
                stats["merged"] += 1
            else:
                rq = ReviewQueue(
                    standard_question=norm["standard_question"],
                    example_answer=norm["example_answer"], occurrences=1,
                    first_raw_question=row.question,
                    source_conversation_id=row.source_conversation_id,
                )
                session.add(rq)
                await session.flush()          # 拿自增主键
                row.matched_review_id = rq.id
                stats["created"] += 1
        except Exception:  # noqa: BLE001
            # 一行失败**不拖垮整批**:那一行的 matched_review_id 留 NULL,
            # 下次重跑还会处理它。失败原因进日志。
            stats["failed"] += 1
            logger.warning("飞轮处理失败 row id=%s", row.id, exc_info=True)
    await session.commit()
    return stats
```

- [ ] **Step 5: 跑测试** → PASS

- [ ] **Step 6: 跑 `normalize` 的标注样例评估**

写 `scripts/run_flywheel_eval.py`(照 `scripts/run_resolve_eval.py` 的样子),
跑一遍并把结果**如实记进 dev-notes**(它是纯 Prompt,没有 TDD 可依靠)。

- [ ] **Step 7: Commit**

```bash
git add app/flywheel evals/flywheel_cases.jsonl scripts/run_flywheel_eval.py tests/test_flywheel_pipeline.py
git commit -m "ch09 T13: 飞轮三步(标准化/查重/编排)+ 标注样例"
```

---

## Task 14: 飞轮后台任务 + 触发端点

**Files:**
- Create: `app/flywheel/tasks.py`
- Modify: `app/api/kb.py`(加 `POST /api/kb/jobs/flywheel`)
- Test: `tests/test_flywheel_task.py`

**Interfaces:**
- Produces: `start_flywheel_job(*, settings, model_factory) -> str`(返回 job id);`POST /api/kb/jobs/flywheel`

- [ ] **Step 1: 写实现**

照 `app/kb/orchestrate.py` **逐条对齐**(那是 ch04 验过的模板),三处**一个字都不能省**:

1. **自建 engine**(`create_async_engine` + 任务结束 `dispose()`)—— `get_engine()` 的
   lru_cache 单例绑在**首次使用它的事件循环**上,后台线程里复用会出跨循环问题。
2. **专用线程 + 线程内 `asyncio.run`**。
3. **在跑标记的摘除必须在 `finally` 里** —— 漏掉不是"多跑一次",是那个会话
   **再也跑不了**,而用户侧每一轮看起来都完全正常(ch07 记过)。

```python
# app/flywheel/tasks.py
JOB_KIND = "flywheel"

def start_flywheel_job(*, settings, model_factory) -> str:
    """起一个后台飞轮任务,返回 job id。落池之后 fire-and-forget 调它。"""
    ...
```

`app/api/kb.py` 加一个端点,形状与既有的 `POST /api/kb/jobs/vectorize` **完全一致**
(含 409「已有任务在跑」那条)。

- [ ] **Step 2: 测试**

```python
# tests/test_flywheel_task.py
@pytest.mark.anyio
async def test_job_completes_and_clears_the_running_flag():
    """跑一次 → 状态 done;**且跑完标记被摘掉**(不摘的话第二次永远 409)。"""
    ...

@pytest.mark.anyio
async def test_second_job_while_running_gets_409():
    ...
```

- [ ] **Step 3: 在落池之后 fire-and-forget 起任务**

`confidence_gate` 与 `agent` 的知识轮落池之后,调 `start_flywheel_job`(不 await)。
**两处都要**,否则入口 ①② 落池后不会自动进飞轮。

> ⚠️ 这两处**不 await** ⇒ 测试里必须能关掉它,否则单测会真的起线程。
> 用 `settings` 上的一个开关或 monkeypatch `app.agent.nodes.start_flywheel_job`;
> **在测试里一律 patch 成 no-op**,并在落池的用例里断"被调用了一次"。

- [ ] **Step 4: 跑测试 + 全量非 db** → 全绿

- [ ] **Step 5: Commit**

```bash
git add app/flywheel/tasks.py app/api/kb.py app/agent/nodes.py tests/test_flywheel_task.py
git commit -m "ch09 T14: 飞轮后台任务(专用线程 + 自建 engine)+ 触发端点"
```

---

## Task 15: 审核端点 + 通过即写知识库

**Files:**
- Create: `app/api/review.py`
- Modify: `app/main.py`
- Test: `tests/test_api_review.py`

**Interfaces:**
- Produces: `GET /api/review/queue`、`GET /api/review/{id}`、`POST /api/review/{id}/approve`、`POST /api/review/{id}/reject`

- [ ] **Step 1: 写失败的测试**

四条:列表只回 `pending`、详情带**用户原话清单**与**快照**、approve 调
`write_chunks` + `vectorize_rows`(替身计数)、reject 不碰知识库。

- [ ] **Step 2: 写实现**

```python
@router.post("/api/review/{review_id}/approve")
async def approve(review_id: int, body: ApproveRequest, ...):
    rq = await session.get(ReviewQueue, review_id)
    if rq is None or rq.status != "pending":
        raise HTTPException(404, "待审问题不存在或已处理")

    answer = (body.approved_answer or rq.example_answer).strip()
    if not answer:
        raise HTTPException(422, "核准答案不能为空")

    # **立刻写知识库并向量化** —— 不这么做的话,「同一个问题再问就能答对」
    # 要等下一次 scripts/build_kb.py,验收 3 直接落不了地(spec §9.2)。
    chunk = Chunk(category="faq", questions=rq.standard_question, answer=answer,
                  section_path=None, content_type="faq", is_key_clause=False)
    added = await write_chunks(session, [chunk])
    if added:
        rows = ...   # 取刚写进去的行
        await vectorize_rows(session, store, embedder, rows)
    # Milvus 不在线 ⇒ vectorize_rows 抛 ToolInfrastructureError → 502。
    # **不许静默降级成「写进 MySQL 了但检索不到」** —— 那是本仓那条
    # 「基础设施故障绝不伪装成成功」的反面。

    rq.status = "approved"
    rq.approved_answer = answer
    rq.reviewed_at = func.now()
    await session.commit()
    return {"ok": True, "chunks_added": added}
```

> ⚠️ `write_chunks` 自带 `(category, questions, answer)` 三元组查重 ⇒
> `chunks_added` **可能是 0**(知识库里早有这条)。那种情况**不算失败**,
> 但响应里要说明,验收脚本据此给出可读的失败信息(spec §13-9)。

- [ ] **Step 3: 跑测试** → PASS

- [ ] **Step 4: Commit**

```bash
git add app/api/review.py app/main.py tests/test_api_review.py
git commit -m "ch09 T15: 待审队列端点(通过即写知识库并向量化)"
```

---

## Task 16: 前端(纯 UI,按项目规矩 Vibe Coding,不套 TDD)

**Files:**
- Modify: `app/static/index.html`
- Modify: `app/static/admin.html`

- [ ] **Step 1: 聊天页 👎 接后端**

现在的满意度反馈是**纯前端采集**(点了只变个样式)。改成点击时
`POST /api/feedback {conversation_id, question, message_id, value}`。
`question` 取该轮渲染的那条用户消息的文本;`message_id` 取本轮 assistant 消息的 id
(拿不到就不传)。

- [ ] **Step 2: `admin.html` 加「待审」标签页**

沿用现有 `data-tab` + `style.display` 的切换方式与 CSS 变量体系
(`--brand` / `--brand-dim` / `--ok` / `--warn` / `--badge`,像素硬边框风格)。
列表:标准化问题 / 出现次数 / 示例答案 + 「通过」「驳回」按钮。行可展开 ⇒
**归并进来的用户原话清单** + **召回片段快照**。

- [ ] **Step 3: 手工点一遍**

起服务,浏览器里:问一个知识库没有的问题 → 待审页看到它 → 展开看到片段 →
点通过 → 回聊天页重问 → 答对。**把截图或文字记录写进 dev-notes。**

---

## Task 17: 评估流水线 —— `--limit` / `--trigger` / `eval_trend.py`

**Files:**
- Modify: `scripts/run_eval.py`
- Create: `scripts/eval_trend.py`
- Test: `tests/test_eval_runs_write.py`(db,只测写库那一段)

**Interfaces:**
- Consumes: T5 的 `EvalRun`
- Produces: `eval_runs` 里的一行;`eval_trend.py` 的趋势表

- [ ] **Step 1: 给 `run_eval.py` 加两个参数**

```python
    parser.add_argument("--limit", type=int, default=0,
                        help="只跑前 N 条(0 = 全跑)。给验收脚本用;"
                             "**case_count 记的是实际条数**,别让不同规模的轮次看起来可比")
    parser.add_argument("--trigger", default="manual",
                        help="触发方式,写进 eval_runs.trigger_by")
```

`--limit 0` 时行为**一字不变**。`--limit N` 时 `cases = cases[:N]`。

- [ ] **Step 2: 跑完写一行 `eval_runs`**

在写 `evals/results/latest.json` **之后**追加(不改那个文件的结构与路径 ——
`admin.html` 的评测页读它):

```python
    await _record_eval_run(
        trigger_by=args.trigger,
        case_count=len(cases),
        # **现有 latest.json 的内容原样进 metrics**,外面包一层 ——
        # 这样"评估集的形状"与"这一轮跑了多少条"是两个独立的键,
        # 不会因为 --limit 而在同一个键上出现两种含义(spec §10.3)。
        metrics={"top_k": args.top_k, "case_count": len(cases),
                 "strategies": results["strategies"]},
    )
```

`_record_eval_run` 照 `app/kb/orchestrate.py` 那套**自建 engine**(这个脚本不在
请求路径上,但它跑在事件循环里,用 `get_engine()` 的 lru_cache 单例没问题 —— 
**若它已经在用,就直接用**;新起 engine 也行,只要 dispose)。

- [ ] **Step 3: 写 `eval_trend.py`**

读 `eval_runs` 按 `created_at` 升序,打印:

```
===== 评估趋势 =====
轮次   时间                  条数   混合+Rerank A_policy        D_absent
 1     2026-09-23 21:03       20    recall@10=0.750 mrr=0.620   recall@10=0.900
 2     2026-09-23 21:15       20    recall@10=0.700 mrr=0.610   recall@10=0.900
                                    ↓ -0.050        ↓ -0.010     = 0.000
```

**必须标出每个指标相对上一轮的增减**(需求原话:「哪个指标在下滑要能一眼看出来」)。
第一轮没有上一轮可比时不打箭头。

- [ ] **Step 4: 测试**

```python
# tests/test_eval_runs_write.py
pytestmark = pytest.mark.db

@pytest.mark.anyio
async def test_eval_run_row_carries_case_count_separately():
    """`--limit` 跑的那一轮,case_count 必须是**实际**条数,不是 300。"""
    ...
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_eval.py scripts/eval_trend.py tests/test_eval_runs_write.py
git commit -m "ch09 T17: 评估流水线落 eval_runs + 趋势表"
```

---

## Task 18: `scripts/acceptance_ch09.sh` —— 六条端到端验收

**Files:**
- Create: `scripts/acceptance_ch09.sh`
- Modify: `CLAUDE.md`(命令一节)

**前置:MySQL + Milvus + 两个 MCP Server + 真实 key + Langfuse 可达**(比 ch08 多了
Milvus 与 Langfuse)。**起服务前先清端口。**

**沿用 ch08 脚本的既有工程约定(逐条照做,别重新发明):**

- `fail_exit()` **先** `KEEP=1` 再退;
- **分离** `EXIT` 与 `INT TERM HUP` 陷阱;
- needle 用 `chr()` 拼(**不写进源码字面量**,避开编辑器的编码歧义);
- **含中文的请求体一律走 stdin heredoc 或 httpx**,**不走 `curl` 的 argv**;
- 回复断言前先 `join_tokens` 把逐 token 的 SSE 帧拼回去;
- **凡是对 `low_confidence_questions` / `review_queue` / `tool_audit_logs` 的断言,
  一律按 `conversation_id` 或本轮生成的主键过滤**;
- 端口可用 `PORT` 覆盖;跑完自己收干净。

- [ ] **Step 1–6: 逐条写六个验收**

| # | 验收点 | 断言 |
|---|---|---|
| 1 | Langfuse 里能点开任意一条请求看到完整链路 | 发一次 chat → 记下 `session_id` → **等 ≥45s**(ingestion 延迟)→ 打 `GET /api/public/v2/observations?traceId=…` → 断观测数 ≥ 3,且**同时**出现 `retrieval`、`ChatOpenAI`(或工具名)。**「界面能点开」那一半靠人,脚本断的是数据在不在** |
| 2 | 知识库没有的问题 → 兜底话术 + 待审队列 | 发问 → `join_tokens` 后断含兜底话术 → **轮询** `POST /api/kb/jobs/flywheel` + `GET /api/review/queue`(容忍 spec §8.4 的可见性窗口)→ 断出现该问题 → `GET /api/review/{id}` 断**用户原话**与**召回片段快照**都在 |
| 3 | 审核页点通过 → 再问就答对 | 先断 `GET /api/kb/stats` 的 `milvus_count` **增长**(否则「答对了」可能原来就答得对 —— spec §13-8),再重问**逐字同一问题** → 断含核准答案的特征串、且**不是**兜底话术 |
| 4 | 👎 → 落池 → 进待审队列 | `POST /api/feedback {value:"down"}` → 轮询飞轮 + 待审队列 → 断该行原话就是那一轮的问题 |
| 5 | 按意图的 token 花销 | 跑 `scripts/intent_cost.py --minutes 30` → 断输出里**≥2 个不同意图**的行 |
| 6 | 评估两轮趋势 | `run_eval.py --limit 5 --trigger manual` 跑两次 → `eval_trend.py` → 断输出里有**两轮**,且第二轮那一行**有增减标记** |

- [ ] **Step 7: 跑一遍**

Run: `bash scripts/acceptance_ch09.sh`
Expected: **6/6 通过**。有红的就修,修不动就**如实报**并在 dev-notes 记下未达成项 ——
**不许把红的写成绿的**。

- [ ] **Step 8: Commit**

```bash
git add scripts/acceptance_ch09.sh CLAUDE.md
git commit -m "ch09 T18: 端到端验收 1–6"
```

---

## Task 19: 章级文档同步

**Files:**
- Modify: `CLAUDE.md`、`AGENTS.md`、`dev-notes/ch09.md`
- Modify: `docs/superpowers/specs/…ch09…-design.md`(§15)

- [ ] **Step 1: `CLAUDE.md`**

加:章节条目(ch09 一段)、架构一节里 `app/observability.py` / `app/flywheel/` /
`app/agent/json_stream.py` 的职责、**本章的硬约束**(逐条从 spec 里搬):

- `span.update(**kwargs)` 的 kwargs **被静默丢弃**(源码 docstring 逐字),改 trace 属性用
  `propagate_attributes`,且它**没有 `__aenter__`**。
- `response_format` 与 `bind_tools` 在 langchain-openai 1.6.2 上**互斥**(走的 beta 路径
  只接受 strict 工具)。
- Metrics v2 的 `query` 是**一个 JSON 字符串参数**;`sum_totalCost` 在没配价格时**恒为 0**;
  按 `tags` 过滤要 `arrayOptions`。
- `evidence_confidence_threshold` 是**标定值**,改它要重跑 `scripts/calibrate_evidence.py`。
- `matched_review_id` **一个列担两个语义**(归并落点 + 待处理标记)。
- `db/ch09.sql` 必须**单独执行**(`init_db.py` 不加列)。

命令一节加:`scripts/calibrate_evidence.py`、`scripts/intent_cost.py`、
`scripts/eval_trend.py`、`bash scripts/acceptance_ch09.sh`(含前置:需 Milvus + Langfuse)。

- [ ] **Step 2: `AGENTS.md`** 同步章节条目与架构。

- [ ] **Step 3: `dev-notes/ch09.md` 收尾段**

记四样:用户关键原话、关键产出、被拒绝/纠偏了什么、翻车与返工。**特别是**:
- T3 冒烟的结果(中途 tag 到底穿透没有);
- T8 标定的读数(**有没有走退化分支**);
- T10 真机量出来的**协议违约率**;
- T16 手工点一遍的记录。

- [ ] **Step 4: spec §15 补实现订正**

实现期所有偏离逐条追加(格式:「最初写的 / 实际是什么 / 为什么会写错 / 证据」)。

- [ ] **Step 5: 跑全量测试(含 db)+ 验收**

```bash
.venv/Scripts/python.exe -m pytest          # 全量,需 MySQL
bash scripts/acceptance_ch09.sh             # 六条
```

**把两个数字都如实记进 dev-notes。**

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md AGENTS.md dev-notes/ch09.md docs/superpowers/specs/
git commit -m "ch09 T19: 章级文档同步 + 实现订正"
```

---

## 自检(spec 覆盖表)

| spec 节 | 落在哪个任务 |
|---|---|
| §2.1 依赖与版本 | T1 |
| §2.2 `response_format`×`bind_tools` | T9/T10(选型的依据) |
| §2.3 字面 `JSON` 字样 | T13(normalize prompt)、T10(协议消息) |
| §3.1–3.3 observability 边界 + 手动 span | T2 / T3 |
| §3.4 意图 tag | T3(+ 退路) |
| §3.5 按意图 token 统计 | T4 |
| §4.1 `evidence_confidence` | T6 |
| §4.2 阈值标定 | T8 |
| §4.3 闸换判据 | T7 |
| §5.1–5.2 协议与位置 | T10 |
| §5.3 解码器 13 条边界 | T9 |
| §5.4 三态行为 | T9 / T10 |
| §5.5 落池范围 | T10 |
| §5.6 违规处理 | T10 |
| §6 👎 落池 | T12 |
| §7 数据 | T5 |
| §8 飞轮流水线 | T13 / T14 |
| §9 审核页与端点 | T15 / T16 |
| §10 评估流水线 | T17 |
| §11 配置项 | T1 / T8 |
| §12 测试与验收 | 每个任务的测试步 + T18 |
| §13 风险 | T3(风险 1)、T8(风险 3 的读数) |
| §14 交付物 | 全部 |
