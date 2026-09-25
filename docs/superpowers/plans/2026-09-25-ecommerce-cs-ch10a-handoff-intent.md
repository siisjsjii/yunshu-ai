# ch10-A:把「转人工」补成第九类意图 —— 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让「我要转人工」成为一条真正的意图:分类器认得它、路由把它送进主力 Agent、Agent 调一个**模拟的**转人工接口把工号与等待时长交给用户。

**Architecture:** ch06 的骨架是「模型只决定**意图标签**,走向由纯函数 `route_by_intent` 写死」。本计划给 `INTENT_TO_ROUTE` 加第九个键 `转人工 → HANDOFF`,并在 `graph.py` 那张条件边映射里让 `HANDOFF` 指向**已有的 `agent` 节点**(不开新出口)。工具体系走 ch08 的包内自动发现:在 `app/tools/builtin/` 新增一个文件就是一个新工具。**唯一一处破例**:转人工发生不发生取决于模型记不记得调工具 —— 这是本章唯一一处「模型决定走向」,已获用户知情同意,§任务 3 把它落成一个**可测的数**。

**Tech Stack:** Python 3.13 / LangGraph 1.2.11 / langchain-core 1.6.3 / pytest 9.1.1 / FastAPI(前端静态页无构建工具链)

**Spec:** `docs/superpowers/specs/2026-09-25-ecommerce-cs-ch10-topic-classifier-design.md`(**§11 是任务来源**,§2.5 是沿革依据)

## Global Constraints

- **意图标签的唯一权威表是 `app/agent/routing.py` 的 `INTENT_TO_ROUTE`**。任何一处再手写一份标签清单都是漂移的开始 —— 本计划的任务 1 会**删掉**其中一份(见 Step 1.6)。
- **路由是纯函数**:`route_by_intent` 无 IO、无模型调用、不依赖 `state` 里除 `intent` 外的任何键。清洗(不 strip、不大小写归一)**不做** —— 那是刻意的,见它的 docstring。
- **伪随机必须用 `hashlib.sha256` 种子,不能用内置 `hash()`**。`hash()` 对 str 每进程随机化,「同一入参永远返回同样数据」会在重启后失效,而**同进程内的测试完全测不出来**。复用 `app/tools/mock_data.rng`。
- **未声明的工具按只读放行**(`app/tools/policy.py`)。转人工**算只读**(用户 2026-09-25 拍板)—— 但声明仍要**显式写出来**并配注释,见任务 2。
- **写测试的硬规矩**:断言一个字段之前,先读它是怎么被赋值的;不要写「在**处理之后**注入」的用例(本仓最高频的假绿形态);计数类断言放在 `ainvoke` 边界。
- **`Settings(...)` 在测试里必须传 `_env_file=None`**;db 测试读真实 `.env` 并打 `@pytest.mark.db`;单测**全程不联网**。
- **不要再往命令行加 `-q`** —— `pytest.ini` 的 `addopts` 已有一个,叠加成 `-qq` 后连 `N passed` 都不打印。
- **含中文的请求体不能走 `curl` 的 argv**(MSYS2 按 CP936 重编码);一律 stdin heredoc 或 httpx。
- **所有出站错误文本必须过 `app/sanitize.py:redact_api_key`**。
- 前端按用户给的例外走 **Vibe Coding**(不套 brainstorm / TDD / code review)。

---

## 文件结构

| 文件 | 职责 | 本计划怎么动 |
|---|---|---|
| `app/agent/routing.py` | 意图 → 出口的映射,**唯一权威标签表** | 加 `HANDOFF` 常量 + `"转人工"` 键;三处「八类」措辞 |
| `app/agent/graph.py` | 建图 | 条件边映射加**一行** `HANDOFF: "agent"` |
| `app/prompts.py` | 意图提示词 | 八类→九类 + 转人工枚举行 + **投诉/转人工 边界样例** |
| `app/agent/state.py` | `ChatState` / `IntentResult` | 两处措辞;`IntentResult.intent` 的第三份标签清单**删掉换成指向权威表** |
| `app/agent/nodes.py` | 意图识别节点 | 一处 docstring 措辞 |
| `app/tools/builtin/handoff.py` | **新建**:模拟的转人工工具 | 全新 |
| `app/tools/policy.py` | 权限声明表 | 显式声明 + 注释(行为上是空操作,见任务 2 Step 3) |
| `tests/test_agent_routing.py` | 路由穷举 | 加第九类;加 `_OUTLETS` 长度锁 |
| `tests/test_agent_intent.py` | 提示词↔路由表对齐守卫 | **改 `_enumerated_labels` 的锚点** + 加边界样例断言 |
| `tests/test_handoff_tool.py` | **新建**:工具的确定性 + 权限 | 全新 |
| `evals/intent_cases.jsonl` | 意图评估集(38 条) | 加转人工用例 |
| `scripts/acceptance_ch06.sh` | ch06 验收(回归网) | 四处「八类」措辞 |
| `app/static/index.html` | 聊天页 | 投诉出口的「转人工」按钮改走真实路径 |
| ch06 spec | 历史设计文档 | **只追加一节后记**,不改历史章节 |

---

## Task 1: 第九类进路由表与提示词(同源一起改)

> **为什么这两件事必须在同一个任务里**:`tests/test_agent_intent.py` 的 `test_every_label_in_the_prompt_matches_the_routing_table` 拿 `INTENT_LABELS` 去比提示词的**枚举行**。只改 `routing.py` 不改提示词 ⇒ 那条测试**立刻红**(缺 `转人工`),仓库停在一个红的状态。路由表与提示词按设计就是**同源**的(`routing.py:33` 的注释原文:「提示词的标签表与这张表同源,少一行,模型就永远学不到「其他」这个词」)。

**Files:**
- Modify: `app/agent/routing.py`
- Modify: `app/agent/graph.py:41-49`(import 块)、`app/agent/graph.py:188-198`(条件边)
- Modify: `app/prompts.py:278-305`
- Modify: `app/agent/state.py:21-29`、`app/agent/state.py:133`
- Modify: `app/agent/nodes.py:44`
- Modify: `tests/test_agent_routing.py`
- Modify: `tests/test_agent_intent.py`

**Interfaces:**
- Consumes: 无(本任务是最上游)
- Produces:
  - `app.agent.routing.HANDOFF: str = "handoff"` —— 新的路由值
  - `INTENT_TO_ROUTE["转人工"] == HANDOFF`;`INTENT_LABELS` 变成 **9 元组**(顺序:商品咨询/退款退货/物流/订单/售后/投诉/闲聊/其他/转人工 —— 即 `INTENT_TO_ROUTE` 的插入序,`转人工` 加在末尾)
  - `route_by_intent({"intent": "转人工"}) == "handoff"`

- [ ] **Step 1: 先写失败的测试(路由侧)**

在 `tests/test_agent_routing.py` 的 import 块里加 `HANDOFF`:

```python
from app.agent.routing import (
    BUSINESS,
    CHITCHAT,
    COMPLAINT,
    FALLBACK,
    HANDOFF,
    INTENT_LABELS,
    INTENT_TO_ROUTE,
    KNOWLEDGE,
    OTHER,
    REFUND,
    route_by_intent,
)
```

在 `CASES` 列表**末尾**(`OTHER` 那一行之后)追加:

```python
    # ch10-A:「转人工」是**第九类**。它由主力 Agent 调模拟接口完成,
    # 所以路由值是 HANDOFF、目标节点是 agent —— **不开新出口**。
    ("转人工", HANDOFF),
```

把 `test_every_intent_is_covered` 的 docstring 改成:

```python
def test_every_intent_is_covered():
    """九类一个不漏 —— 少一类会静默落进兜底,而兜底不调模型,问题就永远答不上。"""
```

在同一文件**末尾**追加两条新测试:

```python
def test_outlets_are_still_exactly_five():
    """出口数**锁死在 5**。ch10-A 加第九类意图时**没有**开新出口,这条是那个决定的守卫。

    加出口是个需要被看见的决定(同 `policy.WRITE_TOOLS` 那条精确相等断言的
    先例):改这里就必须改这条测试,改的时候你会被迫想一遍「真的需要第六个出口吗」。
    """
    from app.agent.graph import _OUTLETS

    assert len(_OUTLETS) == 5


def test_handoff_route_value_is_not_reused():
    """`HANDOFF` 必须是**新的**路由值,不能借用 BUSINESS。

    借用的后果:`log_turn` 的 trace 帧与 `route_by_intent` 的返回值里,
    「转人工」与「物流/订单」长得一模一样 —— 事后想统计「有多少轮真的走了转人工」
    时,这个数**永远取不出来**,而没有任何东西报错。
    """
    assert HANDOFF != BUSINESS
    assert HANDOFF not in (KNOWLEDGE, COMPLAINT, CHITCHAT, FALLBACK, REFUND)
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_routing.py`
Expected: **FAIL / collection error** —— `ImportError: cannot import name 'HANDOFF' from 'app.agent.routing'`

- [ ] **Step 3: 改 `app/agent/routing.py`**

模块 docstring 第 5 行「穷举八类 + 越界 + 缺字段」→「穷举九类 + 越界 + 缺字段」。

在 `REFUND = "refund"` 那行下面加常量(位置紧随 REFUND,并在注释里点名它特殊在哪):

```python
#: ch10-A:「转人工」的路由值。它**不是**新出口 —— `graph.py` 把它指向**已有的
#: `agent` 节点**,由 Agent 调模拟的转人工接口完成(用户 2026-09-25 拍板)。
#: 单独立一个值而不是复用 BUSINESS,是为了让 trace 与统计能把它与「物流/订单」分开。
HANDOFF = "handoff"
```

`INTENT_TO_ROUTE` 的注释块里「**八类意图 → 五出口。**」→「**九类意图 → 五出口。**」,并在表里 `OTHER: FALLBACK` **之后**加一行(**必须在末尾**:`INTENT_LABELS` 是 `tuple(INTENT_TO_ROUTE)`,插入位置会改顺序):

```python
    # ch10-A 加的第九类。它**不改变既有八类任何一条的行为**。
    "转人工": HANDOFF,
```

`route_by_intent` 的 docstring 第 1 行「八类之一 → 五出口」→「九类之一 → 五出口」。

- [ ] **Step 4: 改 `app/agent/graph.py`**

import 块加 `HANDOFF`(按字母序放在 `FALLBACK` 与 `KNOWLEDGE` 之间):

```python
from app.agent.routing import (
    BUSINESS,
    CHITCHAT,
    COMPLAINT,
    FALLBACK,
    HANDOFF,
    KNOWLEDGE,
    REFUND,
    route_by_intent,
)
```

条件边那张 dict 里加一行:

```python
            REFUND: "refund_pick_order",
            # ch10-A:转人工走**主力 Agent**(不是新出口)。Agent 会调
            # `transfer_to_human` 把工号与等待时长交给用户。
            HANDOFF: "agent",
```

- [ ] **Step 5: 跑路由测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_routing.py`
Expected: **PASS**

- [ ] **Step 6: 跑意图提示词守卫,确认它**红**了(这是本任务最要紧的一步)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_intent.py`
Expected: **FAIL**,`test_every_label_in_the_prompt_matches_the_routing_table` 报
`Prompt 的枚举行里缺这些标签:['转人工']`

> ⚠️ **这一步不是走形式。** 它证明那条守卫**真的有判别力** —— 如果它在这一步就是绿的,说明守卫已经失效(比如锚点没匹配上、解析成空集),那么后面所有关于「标签同源」的说法都是空的。**看到红再往下。**

- [ ] **Step 7: 改 `app/prompts.py` 的提示词**

三处「八类」→「九类」(介绍句、段标题、输出契约段的说明),并在枚举段**末尾**(「其他」那一行之后)加一行:

```
- 转人工:明确要求转接人工客服、找真人处理
```

在「边界样例(照着判)」那一段的**末尾**追加两条(**顺序有意义:先给「要转人工」的正例,再给「别把投诉当成转人工」的反例**):

```
- 「我要转人工」「找真人客服」→ 转人工
- 「你们的客服太差了,我要投诉」→ 投诉(**不是**转人工 —— 投诉出口本身会给出转人工的选项,别抢它的活)
```

- [ ] **Step 8: 改 `tests/test_agent_intent.py` 的锚点**

`_enumerated_labels` 里的 `partition("八类")` → `partition("九类")`,两条 assert 的文案同步改成「九类」;函数 docstring 里最后一段关于「下边界命中的是介绍句里的八类」→「九类」。

`test_seven_labels_pass_through` 的参数化列表加 `"转人工"`,并把函数名改成 `test_every_declared_label_passes_through`:

```python
@pytest.mark.anyio
@pytest.mark.parametrize(
    "intent", ["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "转人工"]
)
async def test_every_declared_label_passes_through(intent):
```

> ⚠️ **改名会让 pytest 的 node id 变**,`docs/` 或脚本里若引用过旧 node id 需要一起改。本仓已有一条「node id 过期 → pytest exit 4 而输出被 `tail` 切掉 → 被读成假绿」的教训,改完**先 `grep -rn "test_seven_labels" .` 确认没有残留引用**。

在文件末尾追加:

```python
def test_complaint_and_handoff_are_kept_apart():
    """「投诉」与「转人工」不能揉成一个 —— 揉了的后果是**投诉出口被绕过**。

    投诉出口除了安抚话术,还会发一个 `choices` 帧把「转人工 / 建工单」两个选项
    交给用户。如果模型把「我要投诉」直接判成转人工,用户就**再也拿不到那两个选项**,
    而 `complaint_reply` 那段固定话术永远不会出现 —— 没有任何东西会报错。

    这条只断「提示词里同时写着这两条边界样例」,不断模型行为(模型行为由
    `evals/intent_cases.jsonl` 的边界负例覆盖,见任务 3)。
    """
    from app.prompts import INTENT_SYSTEM_PROMPT

    assert "不是**转人工" in INTENT_SYSTEM_PROMPT or "不是转人工" in INTENT_SYSTEM_PROMPT, (
        "提示词里缺少「投诉 ≠ 转人工」的边界样例"
    )
```

- [ ] **Step 9: 改 `app/agent/state.py` 的三处**

① `intent: str  # 八类之一(含「其他」)` → `# 九类之一(含「其他」)`

② 文件顶部关于通道声明的那段注释(在 `confidence` 上方)里若有「八类」,同步改。

③ **`IntentResult.intent` 的 `Field(description=...)`**:把那份手写的标签清单**删掉**,换成指向权威表:

```python
    # ⚠️ **这里刻意不再列举标签名**(ch10-A)。原先它手写了八类,是**第三份**
    # 标签表 —— 而 `tests/test_agent_intent.py` 的标签守卫只看
    # `routing.INTENT_TO_ROUTE` 与提示词那两处,**看不见这里**(旧注释自己
    # 写着这句)。加第九类时它就会静默过期,且**没有任何测试会红**。
    # 唯一权威表是 `routing.INTENT_TO_ROUTE`;`json_mode` 这条路本来就
    # 不把 schema 描述发给模型,所以这段文字只服务于读代码的人 ——
    # 那就让它指向权威表,而不是再抄一份。
    intent: str = Field(
        description="意图标签之一;取值域见 app.agent.routing.INTENT_TO_ROUTE。"
        "无法归入任何一类时为「其他」。"
    )
```

- [ ] **Step 10: 改 `app/agent/nodes.py:44`**

`"""意图识别:一次 LLM(json_mode),输出八类之一。` → `输出九类之一。`

- [ ] **Step 11: 跑全套意图相关测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_routing.py tests/test_agent_intent.py`
Expected: **PASS**,且**打印出通过数**(不许加 `-q`)

- [ ] **Step 12: 确认没有残留的旧 node id 引用**

Run: `grep -rn "test_seven_labels" . 2>/dev/null | grep -v __pycache__`
Expected: 无输出

- [ ] **Step 13: Commit**

```bash
git add app/agent/routing.py app/agent/graph.py app/prompts.py \
        app/agent/state.py app/agent/nodes.py \
        tests/test_agent_routing.py tests/test_agent_intent.py
git commit -m "ch10-A T1: 转人工进路由表与提示词(第九类,走主力 agent 不开新出口)"
```

---

## Task 2: 模拟的转人工工具 `transfer_to_human`

**Files:**
- Create: `app/tools/builtin/handoff.py`
- Create: `tests/test_handoff_tool.py`
- Modify: `app/tools/mock_data.py`(加一个确定性的转人工记录函数,与其余 mock 数据同源)
- Modify: `tests/test_builtin_discovery.py`、`tests/test_registry.py`、`tests/test_api_chat.py` ——
  ⚠️ **这三个不在原计划的清单里,是实现时补上的**(T2 执行时发现,已核实):
  加一个内置工具会打破**三个文件里的四条精确相等断言**
  (`test_builtin_discovery.py` 的内置名集合、`test_registry.py` 的 `build_tools` 集合、
  `test_api_chat.py` 的两条 `bound_tools` 集合)。**不改它们分支就是红的**。
  每处**只加 `"transfer_to_human"` 一项,不许削弱或删掉别的断言**。
  另外 `test_builtin_discovery.py` 那条测试的名字要跟着改(它叫 `test_four_builtin_tools_...`,
  而 ch08 已有按数量改名的先例)。**改名后 grep 一遍旧 node id 有没有被别处引用。**

**Interfaces:**
- Consumes: `app.tools.mock_data.rng`(sha256 种子工厂)
- Produces:
  - `app.tools.mock_data.handoff_record(reason: str) -> dict` —— 返回 `{"agent_no": str, "queue_position": int, "eta_minutes": int, "status": "connected"}`。
    ⚠️ 参数名是 **`reason`**(有用户原话来源),不是 `agent_pool` —— 后者是坐席池常量 `AGENT_POOL`。
    这两者一度在本计划的 Interfaces 块与 Step 3 之间不一致,以 **Step 3 的代码为准**。
  - `app.tools.builtin.handoff.build(*, session, conversation_id, retriever) -> list` —— 与其他 builtin 模块同签名(包内自动发现按这个签名调用)
  - 工具名 **`transfer_to_human`**,`ainvoke({"reason": ...})` 返回 JSON 字符串

- [ ] **Step 1: 写失败的测试**

Create `tests/test_handoff_tool.py`:

```python
"""转人工工具(ch10-A):确定性 + 权限声明 + 入参回显截断。"""

import json

import pytest

from app.tools.builtin.handoff import build
from app.tools.policy import kind_of


def _tool():
    """`build()` 的签名与其他 builtin 模块一致;转人工不碰会话,故传 None。"""
    return build(session=None, conversation_id="conv-test", retriever=None)[0]


@pytest.mark.anyio
async def test_same_reason_always_gives_the_same_agent():
    """同一入参必须永远得到同样的工号 —— **跨进程**也要稳定。

    这条守的是「伪随机不能用内置 hash()」那条硬约束:`hash()` 对 str 每进程
    随机化(PYTHONHASHSEED),同进程内的测试**完全测不出来**,只有重启后
    用户会发现「同一个问题每次转给不同的人」。跨进程那一半由
    `tests/test_seed_tools_random.py` 的既有跨进程用例守着(它扫的是
    `mock_data.rng` 的使用),这里守的是**本工具确实用了它**。
    """
    tool = _tool()
    a = json.loads(await tool.ainvoke({"reason": "我要找真人"}))
    b = json.loads(await tool.ainvoke({"reason": "我要找真人"}))
    assert a == b


@pytest.mark.anyio
async def test_different_reasons_can_give_different_agents():
    """不是恒等函数 —— 否则上面那条测试对一个常量返回也成立。"""
    tool = _tool()
    reasons = ["我要找真人", "转人工", "帮我接客服", "人工客服在哪", "我要投诉"]
    seen = {json.loads(await tool.ainvoke({"reason": r}))["agent_no"] for r in reasons}
    assert len(seen) > 1


@pytest.mark.anyio
async def test_result_carries_what_the_user_needs():
    """回复里必须**真的有**工号与等待时长 —— 任务 3 的端到端断言就断这两个键。"""
    payload = json.loads(await _tool().ainvoke({"reason": "转人工"}))
    assert payload["agent_no"].startswith("A")
    assert isinstance(payload["queue_position"], int)
    assert payload["eta_minutes"] >= 0


@pytest.mark.anyio
async def test_long_reason_is_truncated_not_rejected():
    """超长入参夹到上限,而不是抛错。

    理由同 `create_ticket` 的 `ticket_type`:这是把用户的诉求交给人工,
    一个被模型撑爆的字段不该让整次转人工失败。`ECHO_LIMIT` 是
    `mock_data` 里已有的常量(32),复用而非新写一个。
    """
    payload = json.loads(await _tool().ainvoke({"reason": "转" * 500}))
    assert payload["agent_no"]


@pytest.mark.anyio
async def test_blank_reason_is_rejected():
    tool = _tool()
    with pytest.raises(Exception):
        await tool.ainvoke({"reason": "   "})


def test_transfer_to_human_is_declared_read():
    """⚠️ 这条测试的**判别力**要说清楚。

    `policy.kind_of` 的规矩是「未声明 = 只读」,所以这条断言对一个
    **漏声明**的实现同样会通过 —— 它守的不是「有没有声明」,而是
    「**将来有人把它改成 write 时会被拦下来**」。声明本身在行为上是空操作,
    它的价值是让这个决定可被 grep 到、可被 review。
    """
    assert kind_of("transfer_to_human") == "read"


def test_transfer_to_human_is_not_a_write_tool():
    """它**不许**进 `WRITE_TOOLS` —— 进了就会被确认流拦住,而用户说的是
    「转人工」不是「请再点一次确认」(用户 2026-09-25 拍板:算只读、直调不确认)。
    """
    from app.tools.policy import WRITE_TOOLS

    assert "transfer_to_human" not in WRITE_TOOLS
```

- [ ] **Step 2: 跑测试,确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_handoff_tool.py`
Expected: **FAIL** —— `ModuleNotFoundError: No module named 'app.tools.builtin.handoff'`

- [ ] **Step 3: 在 `app/tools/mock_data.py` 末尾加确定性数据源**

```python
#: 转人工的模拟坐席池(ch10-A)。工号形如 `A123`。
AGENT_POOL = tuple(f"A{n}" for n in (102, 118, 205, 233, 247, 311, 356, 402, 429, 470))


def handoff_record(reason: str) -> dict:
    """模拟的转人工结果:**由入参决定性派生**(与其余 mock 数据同规矩)。

    用 `rng(...)` 而不是内置 `hash()` —— 理由见 `rng` 的 docstring:
    同进程内测不出来,重启后才炸。
    """
    r = rng("handoff", reason)
    agent_no = r.choice(AGENT_POOL)
    queue_position = r.randint(0, 6)
    return {
        "agent_no": agent_no,
        "queue_position": queue_position,
        # 队首(0)也要给一个非零的等待,否则前端会显示「预计等待 0 分钟」,
        # 读起来像没接上。
        "eta_minutes": max(1, queue_position * 2 + r.randint(1, 3)),
        "status": "connected",
    }
```

- [ ] **Step 4: 写工具 `app/tools/builtin/handoff.py`**

```python
"""转人工工具(ch10-A)—— **模拟**接人工,不接真人系统。

设计要点(都对应一次讨论,别静默改掉):

1. **它是只读的。** `app/tools/policy.py` 里显式声明成 `read`,`kind_of` 因此
   放行直调,**不过 ch08 的确认流**。用户 2026-09-25 拍板:它是模拟接口、
   无真实副作用,而要求用户对「转人工」再点一次确认是坏体验。
   注意「未声明 = 只读」意味着这行声明**在行为上是空操作** ——
   它的价值是可 grep、可 review;真正的守卫是 `tests/test_handoff_tool.py`
   里那条「将来有人改成 write 会被拦下」的断言。

2. **不加 `session` 依赖。** 它不落库 —— 转人工的留痕由 `conversations.status`
   与既有的建单路径负责(ch08 的 `create_ticket` 会把会话置 `pending_human`)。
   本工具刻意**不做**那件事:用户要的是转人工,不是建工单。

3. **`build()` 的签名与其他 builtin 模块一致**,包内自动发现按这个签名调用;
   不用的参数照样要收,否则发现会失败。
"""

import json

from langchain.tools import tool

from app.tools.errors import ToolNotFound
from app.tools.mock_data import ECHO_LIMIT, handoff_record


def build(*, session, conversation_id, retriever):
    @tool
    async def transfer_to_human(reason: str) -> str:
        """把用户转接给人工客服。用户明确要求转人工、找真人、要人工处理时使用。"""
        cleaned = reason.strip()
        if not cleaned:
            raise ToolNotFound("请说明需要人工处理的什么问题")
        return json.dumps(handoff_record(cleaned[:ECHO_LIMIT]), ensure_ascii=False)

    return [transfer_to_human]
```

- [ ] **Step 5: 在 `app/tools/policy.py` 里显式声明(加注释)**

在 `WRITE_TOOLS` 定义**上方**加:

```python
#: ch10-A:`transfer_to_human` **显式**声明为只读。
#:
#: ⚠️ 按本模块的规矩「未声明 = 只读」,这行注释与那条声明**在行为上是空操作**。
#: 留着它是为了让这个决定**可被 grep、可被 review** —— 见到「转人工」三个字,
#: 下一个人的第一反应是「这该不该过确认流」,而答案(不过)必须写在它能被
#: 找到的地方。真正的守卫是 `tests/test_handoff_tool.py` 里那条
#: 「有人把它改成 write 就会被拦下」的断言。
#:
#: 用户 2026-09-25 拍板:**转人工算只读、直调不确认**。理由:它是模拟接口、
#: 无真实副作用;而要求用户对「转人工」再点一次确认是坏体验。
```

- [ ] **Step 6: 跑测试,确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_handoff_tool.py`
Expected: **PASS**(且打印通过数)

- [ ] **Step 7: 跑工具系统的既有测试,确认没打破别的**

Run: `.venv/Scripts/python.exe -m pytest tests/ -k "tool or policy or registry" -m "not db"`
Expected: **PASS**

> 特别留意 `test_write_tools_is_exactly_create_ticket` —— 那条断言的是**精确相等**,新工具若被误加进 `WRITE_TOOLS` 它会红。

- [ ] **Step 8: Commit**

```bash
git add app/tools/builtin/handoff.py app/tools/mock_data.py \
        app/tools/policy.py tests/test_handoff_tool.py
git commit -m "ch10-A T2: 新增模拟转人工工具 transfer_to_human(显式声明只读、不过确认流)"
```

---

## Task 3: 把「转人工靠不靠得住」落成一个**可测的数**

> **为什么要有这个任务。** 转人工走主力 Agent 意味着**它发生不发生,取决于模型记不记得调工具** —— 这是 ch06 骨架立身之本(「模型只决定意图标签,不决定走向」)的**唯一一处例外**,已当面告知用户、用户知情选择。结构上拦不住它,但**可以测出来**。
>
> 本仓先例:ch08 的「缺必填项就主动追问」当时**没测也测不出**,交付时只能写「未达成」。这次至少把它变成一个读数。

**Files:**
- Modify: `evals/intent_cases.jsonl`
- Create: `scripts/acceptance_ch10.sh`

> `scripts/run_intent_eval.py` **不需要改**:它本来就算「按标签分组」的准确率(`per_label`)与三个口径(`fewshot` 标记)。动手前先跑它确认这一点,别凭计划的一行字就去改一个不用改的文件。

**Interfaces:**
- Consumes: 任务 1 的 `HANDOFF` / 第九类;任务 2 的 `transfer_to_human`
- Produces: `scripts/acceptance_ch10.sh` 里「转人工」那一节的判词;`evals/intent_cases.jsonl` 中 `kind` 字段的取值域新增 `"handoff"`

- [ ] **Step 1: 先读评估集的字段约定(别照抄本计划的猜测)**

Run: `.venv/Scripts/python.exe -c "import json,collections; rows=[json.loads(l) for l in open('evals/intent_cases.jsonl',encoding='utf-8') if l.strip()]; print(collections.Counter(r.get('fewshot') for r in rows)); print(rows[0])"`

Expected: 每条只有 **`text` / `expected` / `fewshot`** 三个字段(`fewshot` 可缺省)。

> ⚠️ **字段是 `fewshot`,不是 `kind`。** `scripts/run_intent_eval.py` 的 docstring 写着它的用法与**为什么**:用例里有一部分与 `INTENT_SYSTEM_PROMPT` 的 few-shot **逐字或近乎重合**,那些行测的是「模型会不会照抄 few-shot」,**不是泛化** —— 所以脚本**分别算三个数**(全部 / 去掉逐字重合 / 去掉逐字+近乎重合)。
> 取值域:`verbatim`(与某条 few-shot 逐字相同)/ `near`(同句改写,共用主干词)/ 缺省(不重合)。
> **新用例必须照这个规矩打标**,否则它们会污染「全部」那个数,而这正是本章最该避免的虚高。

- [ ] **Step 2: 给评估集加转人工用例**

在 `evals/intent_cases.jsonl` **末尾**追加 6 条(JSONL,一行一对象):

```json
{"text": "我要转人工", "expected": "转人工", "fewshot": "verbatim"}
{"text": "帮我转接一下人工客服", "expected": "转人工"}
{"text": "有真人在吗,这问题我想找人处理", "expected": "转人工"}
{"text": "你们的客服太差了,我要投诉", "expected": "投诉", "fewshot": "near"}
{"text": "这个我要投诉到消费者协会", "expected": "投诉"}
{"text": "转人工之前先问一句,退货政策是什么", "expected": "退款退货"}
```

**每条为什么这么标(别改标法):**

- 第 1 条 `verbatim` —— 任务 1 Step 7 给提示词加的 few-shot 就是「我要转人工」,所以这条**逐字重合**,它测的是「会不会照抄 few-shot」,**要单独摘出去看**;
- 第 2、3 条**不打标** —— 它们是**没见过的说法**,这才是泛化读数。故意不用「我要转人工」的措辞;
- 第 4 条 `near` —— 提示词里既有的投诉 few-shot 是「你们的客服太差了,我要投诉」,近乎重合;
- 第 5 条**不打标** —— 同样是投诉、但措辞不同,测泛化;
- 第 6 条**不打标,而且它是本组最有用的一条**:句子里**出现了「转人工」三个字**,但用户的**主诉求**是退货政策。分类器若被字面词带走就会判成转人工 —— 这正是「字面提到 ≠ 意图」的一类句子,也是「投诉/转人工」边界之外**第二处**容易被揉掉的地方。

> ⚠️ 后 3 条断的是**投诉不要被抢走**。揉掉的后果不是分类错,而是**投诉出口被绕过** —— 用户再也拿不到「转人工 / 建工单」那两个选项,而 `complaint_reply` 那段固定话术永远不会出现。

- [ ] **Step 3: 跑意图评估集,拿到转人工的**分类**读数**

Run: `.venv/Scripts/python.exe scripts/run_intent_eval.py`
Expected: 打印逐条结果、总准确率、**按标签分组**的准确率、以及**三个口径**(全部 / 去掉逐字重合 / 去掉逐字+近乎重合)。**把「转人工」那一行的三个数抄进任务 5 的 `dev-notes/ch10.md` 记账**。

> ⚠️ **加用例会改变分母。** 用例数从 38 变 44,所以**总准确率与 ch06/CLAUDE.md 里记录的那些数不再是同一个东西**。任务 5 记账时必须写明「38 条口径」与「44 条口径」两个数**不能直接比较**,而不是把新数悄悄替掉旧数 —— 本仓编目过的教训正是「一个看起来已经验过、其实描述的是另一套配置的数」。

> ⚠️ 这一步只测**分类**。分类对了不等于转人工发生了 —— 那正是下一步要测的。

- [ ] **Step 4: 写验收脚本(转人工那一节)**

Create `scripts/acceptance_ch10.sh`。**照 `scripts/acceptance_ch09.sh` 的既有形状写**(它自己起服务、跑完自己收干净、把本轮输出转录一份再自检)。

**先用 `grep -nE "^[a-z_]+\(\)" scripts/acceptance_ch09.sh` 把可复用的 helper 名抄下来** —— 下面是它真实提供的(已核,2026-09-25):

| helper | 用途 |
|---|---|
| `ok` / `bad` / `warn` | 判词计数(`PASS`/`FAIL`/`WARN`) |
| **`boom`** | **装置故障**(与 `bad` 分开计)—— 起不了服务、转录不完整这类 |
| **`ask <sid> <消息> <输出文件>`** | **发一条中文消息并录 SSE**。中文请求体走 stdin heredoc,**不走 argv** —— 现成的,直接用 |
| `frames_sane <file>` | **每个断言块都要先跑它** —— 「没有这一帧」这类断言在一份空文件上恒真 |
| `join_tokens` | 从 stdin 读 SSE,把所有 `token` 帧拼回整段回复 |
| `intent_label <file>` | 取 `done` 帧的 `intent`(**注意它吃文件参数,不是 stdin**) |
| `has_needle` / `assert_needle_absent` | 回复里找 needle,**三值退出码**(见下) |
| `cleanup` / `trap` 那套 | 起停服务、跑完自己收干净。**⚠️ `EXIT` 与 `INT TERM HUP` 要分开 trap**(关终端那条路径不经过 `EXIT`),且打了 trap 之后 bash 不会因为收到信号就退出 —— 信号处理里必须自己 `exit` |

⚠️ **`has_needle` 是三个退出码,别把 1 与 2 混成一个非零**:`0` = 针在;`1` = 文件读得动、针确实不在;`2` = **文件读不了/解不开(装置故障)**。混成一个非零的后果很具体:一句 `has_needle ... || ok "不再是兜底话术"` 会把一次 `OSError` **静默判绿**。**凡否定断言一律走 `assert_needle_absent`**,不要自己写 `else` 分支。

先加一个本脚本自己的 helper(`acceptance_ch09.sh` 没有它)—— **「工具有没有被调」要看 `tool_call` 帧,不是看回复的措辞**:

```bash
# 取 SSE 里所有 tool_call 帧的工具名(空格分隔)。
#
# ⚠️ **为什么不看回复里的「预计/等待」**:那是在断**模型的措辞**。
#    模型完全可能把工具返回改写措辞(「客服马上接入,您的号码是 A205」),
#    于是真实故障与措辞差异分不开 —— 而这条断言的全部意义是回答
#    「模型到底调工具了没有」,那是个**结构**问题,SSE 里有直接证据。
called_tools() {   # $1=文件
  "$PYTHON" -c '
import json, sys
lines = open(sys.argv[1], "rb").read().decode("utf-8", "replace").splitlines()
names = []
for i, l in enumerate(lines):
    if l.strip() == "event: tool_call" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        names.append(json.loads(lines[i+1][6:]).get("name", "?"))
sys.stdout.buffer.write(" ".join(names).encode("utf-8"))' "$1"
}
```

转人工那一节的判词骨架:

```bash
# ---- 验收:转人工是**真意图**,而且**真的发生了** ----
#
# 三层,缺一不可 —— 每一层失败的原因**不同**,所以分开报:
#   ① 分类对     :done 帧的 intent == 转人工      (分类器的问题)
#   ② 工具被调了 :SSE 里有 transfer_to_human 的 tool_call 帧 (模型没调工具 —— §11.4 的那个风险)
#   ③ 用户拿到了 :回复里有工号 A###                (调了但没转述)
#
# 只断 ① 的话,「分类对了但模型没调工具、用户什么也没得到」会**全绿通过** ——
# 那正是本仓 ch08「缺必填项追问」翻过车的形状(没测也测不出,只能写"未达成")。
frames_sane "$SSE_HANDOFF" || { bad "SSE 转录不完整,这一节判不了"; }
HANDOFF_INTENT=$(intent_label "$SSE_HANDOFF")
HANDOFF_REPLY=$(join_tokens < "$SSE_HANDOFF")
HANDOFF_TOOLS=$(called_tools "$SSE_HANDOFF")

if [ "$HANDOFF_INTENT" != "转人工" ]; then
  bad "① 意图没落「转人工」:$HANDOFF_INTENT"
elif ! printf '%s' "$HANDOFF_TOOLS" | grep -qw transfer_to_human; then
  bad "② 意图是「转人工」但**模型没调 transfer_to_human** —— 用户什么也没得到。实际调用:[$HANDOFF_TOOLS]"
elif ! printf '%s' "$HANDOFF_REPLY" | grep -qE 'A[0-9]{3}'; then
  bad "③ 工具调了,但回复里没有工号 —— 用户看不到结果。回复:${HANDOFF_REPLY:0:120}"
else
  ok "转人工三层全过:intent=转人工,调了 transfer_to_human,工号=$(printf '%s' "$HANDOFF_REPLY" | grep -oE 'A[0-9]{3}' | head -1)"
fi
```

> ⚠️ 三层分开报是刻意的:**第 ② 层失败就是要写进报告的那个比例读数**(spec §11.4),它与「分类器不行」(第 ① 层)是两件完全不同的事,混成一句判词就再也分不出来了。

> ⚠️ 两条都已核过、都必须遵守:
> ① 含中文的请求体**不能走 `curl` 的 argv**(MSYS2 按 CP936 重编码,服务端回 `error parsing the body`)—— 走 stdin heredoc 或 httpx,照 acceptance_ch09.sh 的既有做法;
> ② 断言**不能直接 grep 原始 SSE 流** —— 回复逐 token 推送,`A205` 会被切成独立帧,必须先 `join_tokens` 拼回来(上面已经这么写了)。
> ③ 上面这两条断言都是**肯定**断言(针必须在),所以用 `grep` 就够;若你后面要加否定断言,**必须**走 `assert_needle_absent`。

- [ ] **Step 5: 跑验收脚本,拿到真实读数**

Run: `bash scripts/acceptance_ch10.sh`
Expected: 转人工那一节打印 `ok` 或 `bad`。**两种结果都如实抄进任务 4 的转录** —— 如果是 `bad`,说明模型没调工具,那**是要写进报告的一个数**,不是要掩盖的失败(修法见 Step 6)。

- [ ] **Step 6: 如果 Step 5 是 `bad`,按下面的顺序排查(不要跳步)**

1. **先看工具在不在注册表里** —— 起服务后 `curl -s localhost:8000/api/kb/... ` 不适用;改成在服务日志里找工具发现那几行,或直接单测 `build_registry` 是否含 `transfer_to_human`。
2. **再看 Agent 的 system prompt 有没有把工具的用途描述带进去** —— 工具描述来自 `ToolSpec` 的 docstring,`transfer_to_human` 的 docstring 必须**明确写「用户要求转人工时使用」**(任务 2 Step 4 已经这么写了)。
3. **最后才考虑换提示词措辞。** 换之前**先把改动前的失败率记下来**,否则改完你也说不清是变好了还是噪声。

- [ ] **Step 7: Commit**

```bash
git add evals/intent_cases.jsonl scripts/acceptance_ch10.sh
git commit -m "ch10-A T3: 转人工的端到端验收(断回复里的工号/等待时长,不只断意图标签)"
```

---

## Task 4: 投诉出口的「转人工」按钮改走真实路径

> **前端按用户给的例外走 Vibe Coding** —— 不套 brainstorm / TDD / code review,起一版看效果。

**Files:**
- Modify: `app/static/index.html:1026-1033`

**Interfaces:**
- Consumes: 任务 1 的第九类意图(前端不需要 import 任何东西,它只是**发一条消息**)
- Produces: 无(纯前端行为)

- [ ] **Step 1: 改 `onChoice`**

现状(`index.html:1026`,源码注释原文:「本章是**纯前端模拟**,不接真人系统、不调后端」):

```javascript
    if (opt.key === "handoff") {
      // 本章是**纯前端模拟**,不接真人系统、不调后端。
      appendSystem("已转接人工客服");
      appendAssistantText("您好,我是客服小猫,请问有什么可以帮您的");
    } else if (opt.key === "ticket") {
```

改成**发一条「转人工」消息**走真实路径:

```javascript
    if (opt.key === "handoff") {
      // ch10-A:不再是纯前端模拟 —— 发一条真实消息,让第九类意图接住它,
      // 由主力 Agent 调 transfer_to_human 给出工号与等待时长。
      //
      // 为什么要改:同一个词在系统里**不能有两套行为**。改之前,点这个按钮
      // 是前端编的「已转接人工客服」,而用户自己打「我要转人工」会走另一条路
      // —— 同一句话两种结果,而两条路都不报错。
      send("我要转人工");
    } else if (opt.key === "ticket") {
```

> 页面里既有的发消息函数是 **`async function send(text)`**(`index.html:656`),**不是** `sendMessage`、也**不吃 event**。它是 async 的,这里**不 await**(与 `submit()` 的既有用法一致)。
> ⚠️ 动手前仍要 `grep -n "function send" app/static/index.html` 复核一次 —— 行号会漂,函数名不该是猜的。

- [ ] **Step 2: 手工验一遍(前端没有自动化测试,靠人看)**

1. 起服务:`.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000`
2. 浏览器开 `http://localhost:8000`,发一句会走投诉出口的话(如「你们的客服太差了,我要投诉」)
3. 页面上会出现「转人工 / 建工单」两个按钮
4. **点「转人工」** —— 期望:出现一条用户消息「我要转人工」,随后的客服回复里**带工号与等待时长**(不再是从前那句前端编的「已转接人工客服」)

- [ ] **Step 3: Commit**

```bash
git add app/static/index.html
git commit -m "ch10-A T4: 投诉出口的转人工按钮改走真实路径(消除同一个词的两套行为)"
```

---

## Task 5: 文档沿革 + 回归网同步 + 章级记账

**Files:**
- Modify: `scripts/acceptance_ch06.sh`(4 处「八类」)
- Modify: `CLAUDE.md`(1 处「八类」)
- Modify: `docs/superpowers/specs/2026-09-20-ecommerce-cs-ch06-intent-router-design.md`(**只追加后记**)
- Modify: `dev-notes/ch10.md`

**Interfaces:**
- Consumes: 任务 1–4 的全部改动
- Produces: 无代码接口

- [ ] **Step 0: 顺手订正 `CLAUDE.md` 里一个**不存在的文件名**(既有陈旧记录,与 ch10 无关)**

`CLAUDE.md` 引了 `tests/test_seed_tools_random.py`,而**该文件不存在** ——
真实的是 `tests/test_tools_random.py`(已核,2026-09-25)。
这是 T2 的实现者撞出来的:它照计划去找那个文件、没找到。

**为什么顺手修**:T5 本来就改 `CLAUDE.md`;而「引用一个不存在的文件」正是本仓最贵的那类教训
(ch09 记账的正是「引用必须逐条解析得开」)。**只改文件名,不动那句话的其余部分。**

- [ ] **Step 0b: 修掉 `app/agent/nodes.py` 里那句因 T4 而变假的注释**

`app/agent/nodes.py:142-143` 的 `make_complaint_reply_node` docstring 写着
「**转人工是前端模拟**,建工单要用户点了按钮才走 `/api/ticket`。用户都不点就继续正常对话。」

T4(`261d573`)之后**这句是假的** —— 前端那个按钮现在发一条真实消息、走第九类意图。
**改这一句,不动那个节点的任何代码**(它的 `CHOICES` 行为一字未变)。

⚠️ 这条是 T4 的实现者报上来的:**它不在任何任务的 Files 清单里** ——
这类「本次改动让别处的注释变假」的缺口,是靠**实现者的自检**捞出来的,不是靠计划。
改完 `grep -rn "前端模拟" app/ docs/ CLAUDE.md` 确认没有第二处还在这么说。

- [ ] **Step 1: 同步 `scripts/acceptance_ch06.sh` 的四处措辞**

`sed -n '152p;463p;473p;475p' scripts/acceptance_ch06.sh` 看到四处「八类」,逐处改成「九类闭集」之类的等价措辞。**只改文案,不动任何断言逻辑。**

- [ ] **Step 2: 同步 `CLAUDE.md` 的那一处**

把 ch05/ch06 段落里写「八类意图」的地方改成「九类意图(2026-09-25 ch10-A 补入「转人工」,走主力 Agent)」。**同一次改动里在 ch10 段落加一小节**,写清:转人工是第九类、走 agent、工具是模拟的、算只读不过确认流、以及**它的可靠性依赖模型调工具**(诚实记账)。

- [ ] **Step 3: 给 ch06 spec 追加后记(不改历史章节)**

在 `docs/superpowers/specs/2026-09-20-ecommerce-cs-ch06-intent-router-design.md` 的**最末尾**追加:

```markdown
---

## 后记:第九类「转人工」(2026-09-25,ch10-A 补入)

**追加,不改上文。** 上文 §3.1 / §3.2 写的是**八类** —— 那是 ch06 交付时的实况,
历史记录保持原样。

ch10-A 把「转人工」补成**真正的第九类**:`INTENT_TO_ROUTE["转人工"] = HANDOFF`,
`graph.py` 把 `HANDOFF` 指向**已有的 `agent` 节点**(不开新出口,`_OUTLETS` 仍 5 个),
由 Agent 调 `app/tools/builtin/handoff.py` 的**模拟**工具 `transfer_to_human`。

**ch06 交付时它在哪**:`app/agent/nodes.py` 的 `CHOICE_HANDOFF`,只由**投诉出口**
发一个 `choices` 帧,`app/static/index.html` 接住后**纯前端模拟**(源码注释原文:
「不接真人系统、不调后端」)。它当时**既不是意图、也不是出口**。

**一处与 ch06 立身之本的冲突,已知情接受**:ch06 的原则是「模型只决定意图标签,
不决定走向」(`routing.py` 模块 docstring)。而转人工走 Agent 之后,**它发生不发生
取决于模型记不记得调工具** —— 这是全仓唯一一处例外。结构上拦不住,只能用
`scripts/acceptance_ch10.sh` 把它测成一个**比例读数**(见 ch10 spec §11.4)。

**沿革的另一半**:投诉出口那个「转人工」按钮同时改成了**发一条真实消息**走这条新路径,
消除「同一个词两套行为」。
```

- [ ] **Step 4: 跑 ch06 相关回归,确认九类没打破既有八类**

Run: `.venv/Scripts/python.exe -m pytest tests/test_agent_routing.py tests/test_agent_intent.py tests/test_agent_graph.py -m "not db"`
Expected: **PASS**。若 `tests/test_agent_graph.py` 不存在,用 `ls tests/ | grep graph` 找到实际文件名再跑 —— **不要因为找不到文件就跳过这一步**。

- [ ] **Step 5: 记账到 `dev-notes/ch10.md`**

追加一段「阶段 1:A 支完成」,四样齐全(用户关键原话 / 关键产出 / 被拒绝或被纠偏 / 翻车与返工)。**必须包含任务 3 Step 5 拿到的转人工成功率读数** —— 无论是多少。如果模型从不调工具,如实写「未达成」,不要包装。

- [ ] **Step 6: Commit**

```bash
git add scripts/acceptance_ch06.sh CLAUDE.md dev-notes/ch10.md \
        docs/superpowers/specs/2026-09-20-ecommerce-cs-ch06-intent-router-design.md
git commit -m "ch10-A T5: 章级文档同步(ch06 后记 + 回归网措辞 + dev-notes 记账)"
```

---

## 完成 A 支时的自检清单

- [ ] `INTENT_TO_ROUTE` 是 9 键;`INTENT_LABELS` 是 9 元组
- [ ] `_OUTLETS` **仍是 5 个**(没有为了转人工开新出口)
- [ ] 提示词的枚举行与 `INTENT_LABELS` 逐条对齐(那条守卫真的红过又真的绿了)
- [ ] 提示词里有「投诉 ≠ 转人工」的边界样例
- [ ] `transfer_to_human` 在注册表里、是 `read`、不在 `WRITE_TOOLS`
- [ ] `evals/intent_cases.jsonl` 加了 6 条(3 正 + 3 边界负例)
- [ ] **转人工的端到端成功率有一个**数\*\*,并已写进 `dev-notes/ch10.md`
- [ ] `IntentResult.intent` 的第三份标签清单已删除(不是更新,是**删掉**)
- [ ] 全量 `pytest -m "not db"` 绿,且**打印出了通过数**
