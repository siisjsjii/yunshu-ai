# ch05 生产级架构(Workflow 确定性编排 + 主力 Agent)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把客服系统从「单轮工具调用」升级成 LangGraph 确定性骨架 + 一个主力 ReAct Agent 节点,前端加「转人工」「建工单」两个独立按钮。

**Architecture:** 一个 `StateGraph` 把确定性步骤固化成节点(指代消解透传 → 意图识别 → **纯函数分流** → 知识类强制预检索 → 置信度闸 → 日志),不确定的部分收进一个手写的 `agent` 节点(ReAct 循环)。四条出口:知识 / 业务数据 / 投诉 / 闲聊+兜底。状态走 `ChatState`(TypedDict + `Annotated` reducer),`InMemorySaver` 做 checkpointer,MySQL `messages` 表仍是跨轮会话的权威源。

**Tech Stack:** LangGraph 1.2.11、langchain 1.4.0 / langchain-core 1.6.3、FastAPI 0.141.1、原生 HTML/CSS/JS、MySQL 8.0、Milvus + BGE-M3 + bge-reranker(复用)。

**Spec:** `docs/superpowers/specs/2026-09-19-ecommerce-cs-ch05-orchestration-design.md`

## Global Constraints

以下约束逐条抄自 spec 与 `CLAUDE.md`,**每个任务都隐含包含**:

- **测试命令**:`.venv/Scripts/python.exe -m pytest -m "not db"`(快路径)。**不要再加 `-q`** —— `addopts` 已有,叠加成 `-qq` 会整行不打印 `N passed`。
- **`Settings(...)` 构造必须传 `_env_file=None`**(`tests/` 里统一如此);不传的话 pydantic-settings 会读仓库根的 `.env`,「缺字段应报错」的测试会静默通过。
- **单测全程不联网、不加载模型、不碰 Milvus**。db 测试打 `@pytest.mark.db`,读真实 `.env`(不加 `_env_file=None`)。
- **异步测试用 `@pytest.mark.anyio`**(backend 由 `tests/conftest.py` 固定为 asyncio),不用 pytest-asyncio。
- **工具调用的替身必须带 `"type": "tool_call"` 键**。`BaseTool.ainvoke` 判「是不是工具调用」**只看** `x.get("type") == "tool_call"`;缺键时它把整个 dict 当**参数**去校验,于是每次调用都返回「参数不合法」的可恢复失败 —— 测试会看似全绿却一条都没走到真实路径。
- **计数类断言必须放在 `ainvoke` 边界**,不能写在工具函数体里(参数非法时函数体根本不执行,恒为 0,区分不出任何实现)。
- **`tool_call` 的参数校验发生在函数体之外**;`@tool` 的参数不合法时函数体不跑。
- **错误语义(不许改)**:`ToolInfrastructureError` 必须向上抛(→502),**绝不**回灌给模型伪装成「查不到」;`422` 只表示模型输出无法解析;所有出站错误文本过 `app/sanitize.py:redact_api_key`。
- **重试白名单**:只有 `query_order`/`query_product`/`query_logistics`/`query_faq` 可重试;**`create_ticket` 永不重试**。
- **`prompts.py` 是 `Message` → `BaseMessage` 转换的唯一出口**;`memory/` 与 `services/history.py` 不依赖 LangChain。
- **抽取出参只能用 `method="json_mode"`**;提示词里**必须出现字面 `JSON` 字样**,且**不得使用裸花括号**(`ChatPromptTemplate` 按 f-string 解析)。
- **流式取文本用 `chunk.text`,不是 `chunk.content`**(1.x 里后者是 content block 列表)。
- **`mount("/")` 必须在 `include_router` 之后**。
- **工具伪随机必须用 `hashlib.sha256` 种子**,不能用内置 `hash()`。
- **Windows + cp936**:含中文的请求体不走 `curl` argv(`error parsing the body`);脚本打印非 ASCII 用 `sys.stdout.buffer.write(...encode("utf-8"))`;验收断言不直接 grep 原始 SSE(逐 token 推送会把 `1001` 切开),拼回后再比。
- **起服务前先查端口**,残留僵尸进程会让你 curl 到旧代码得出假红。
- **平台**:Windows 11 / Git Bash;解释器一律 `.venv/Scripts/python.exe`。

---

## 文件结构

| 文件 | 职责 | 任务 |
|---|---|---|
| `app/agent/loop.py` | **临时**:最裸 Agent 循环(祛魅热身,任务 8 删除) | 1 |
| `tests/test_agent_loop.py` | 上者的测试(任务 8 一并删除) | 1 |
| `app/agent/state.py` | `ChatState`(TypedDict)+ `IntentResult` | 2 |
| `app/agent/routing.py` | 七类 → 四出口的**纯函数**路由表 | 2 |
| `app/agent/emit.py` | `make_emitter(collector=None)`:图内=真 writer,图外=退化到 `collector`(可调用) | 2 |
| `app/agent/nodes.py` | 全部节点工厂 | 3,4,5,6,7 |
| `app/agent/graph.py` | 组装 `StateGraph` + 编译;checkpointer 单例 | 7 |
| `app/prompts.py` | + 意图识别 Prompt、+ 带证据块的消息组装 | 3,6 |
| `app/config.py` | + `max_agent_steps`、`agent_token_budget` | 6 |
| `app/services/chat.py` | `prepare_turn` 改返回裁剪后历史;删 `stream_turn` | 8 |
| `app/api/chat.py` | 接图 + 新增 `POST /api/ticket` | 8,9 |
| `app/schemas.py` | + `TicketRequest` | 9 |
| `requirements.txt` | + `langgraph` | 7 |
| `app/static/index.html` | `choices` 帧 → 两个独立按钮 | 10 |
| `evals/intent_cases.jsonl` + `scripts/run_intent_eval.py` | 意图识别标注样例验证 | 3 |
| `scripts/acceptance_ch05.sh` | 五条验收 | 11 |

**依赖方向**(不变):`api → agent → {tools, retrieval, db, prompts, llm}`。

---

## Task 1:祛魅热身 —— 手写最裸的 Agent 循环

先不用任何框架,把「Agent 就是个带工具的循环」写出来跑通。**这个产物在任务 8 会被删除**,
它的价值是:① 看清循环的本质;② 给后面 LangGraph 版一个行为基线。

**Files:**
- Create: `app/agent/loop.py`
- Create: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: `app.tools.executor.execute_tool(tool_call, registry, settings) -> ToolOutcome`
- Produces: `run_agent_loop(model, messages, tools, registry, settings, max_steps) -> tuple[str, int]`(回复文本, 实际步数)

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_agent_loop.py`:

```python
"""祛魅热身:手写最裸 Agent 循环的行为。全部用替身,不联网、不碰 DB。"""

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from app.agent.loop import run_agent_loop
from app.config import Settings

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    # _env_file=None:不传的话 pydantic-settings 会读仓库根的 .env。
    return Settings(_env_file=None, **REQUIRED, **overrides)


class FakeAIMessage:
    """替身 AI 消息:text + tool_calls(形状与真 AIMessage 一致)。"""

    def __init__(self, text="", tool_calls=None):
        self.text = text
        # 必须带 "type": "tool_call" —— 见本文件顶部说明与 CLAUDE.md。
        self.tool_calls = [{"type": "tool_call", **tc} for tc in (tool_calls or [])]


class _BoundModel:
    def __init__(self, owner):
        self._owner = owner

    async def ainvoke(self, messages):
        self._owner.bound_rounds += 1
        # 记录入参:循环里**每一轮都是绑着工具**问的,所以「回灌进去的
        # ToolMessage」只能在绑工具的入口上看到(未绑工具的入口只在步数
        # 用尽收尾时走一次)。
        self._owner.bound_messages = list(messages)
        return self._owner.rounds.pop(0)


class ScriptedModel:
    """按脚本回放 AI 消息,并区分调用走的是「绑了工具」还是「未绑工具」的入口。"""

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.bound_rounds = 0
        self.unbound_rounds = 0

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return _BoundModel(self)

    async def ainvoke(self, messages):
        # 刻意不记入参:未绑工具的入口只在步数用尽收尾时走一次,回灌的
        # ToolMessage 记在这儿永远看不到 —— 要断回灌请读 `bound_messages`。
        self.unbound_rounds += 1
        return self.rounds.pop(0)


class FakeTool:
    """替身工具:在 ainvoke 边界计数(写在函数体里的话区分不出任何实现)。"""

    name = "query_order"

    def __init__(self, content='{"order_id": "1001", "status": "已发货"}'):
        self.content = content
        self.calls = []

    async def ainvoke(self, tool_call):
        self.calls.append(tool_call)
        return type("_R", (), {"content": self.content})()


@pytest.mark.anyio
async def test_text_only_reply_converges_in_one_step():
    """模型不调工具:一步收敛,且**从未**走未绑工具的入口。"""
    model = ScriptedModel([FakeAIMessage(text="你好呀")])
    reply, steps = await run_agent_loop(
        model=model, messages=[HumanMessage("你好")], tools=[],
        registry={}, settings=_settings(),
    )
    assert reply == "你好呀"
    assert steps == 1
    assert model.bound_rounds == 1
    assert model.unbound_rounds == 0


@pytest.mark.anyio
async def test_tool_call_is_executed_and_result_is_fed_back():
    """模型调工具:执行 → 回灌 ToolMessage → 再问一轮 → 收敛。"""
    model = ScriptedModel([
        FakeAIMessage(tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}]),
        FakeAIMessage(text="你的订单已发货。"),
    ])
    tool_obj = FakeTool()
    reply, steps = await run_agent_loop(
        model=model, messages=[HumanMessage("订单 1001 发货了吗")],
        tools=[tool_obj], registry={"query_order": tool_obj}, settings=_settings(),
    )
    assert reply == "你的订单已发货。"
    assert steps == 2
    assert len(tool_obj.calls) == 1
    assert tool_obj.calls[0]["args"] == {"order_id": "1001"}
    # 回灌:第二轮的消息里必须有一条 tool_call_id 对得上的 ToolMessage
    tool_msgs = [m for m in model.bound_messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_msgs] == ["c1"]
    assert "已发货" in tool_msgs[0].content


@pytest.mark.anyio
async def test_step_limit_forces_convergence_without_tools():
    """步数用尽:最后一轮**不绑 tools**,模型在结构上无法再调。"""
    # 脚本必须与消费顺序对齐:循环里绑工具问 2 轮(== max_steps),各弹走一条;
    # 第 3 条留给收尾那一轮(未绑工具)弹。多写的条目只会被弹到前几条。
    looping = [
        FakeAIMessage(tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": f"c{i}"}])
        for i in range(2)
    ]
    model = ScriptedModel(looping + [FakeAIMessage(text="收敛了")])
    tool_obj = FakeTool()
    reply, steps = await run_agent_loop(
        model=model, messages=[HumanMessage("查")], tools=[tool_obj],
        registry={"query_order": tool_obj}, settings=_settings(), max_steps=2,
    )
    assert steps == 2
    assert model.bound_rounds == 2          # 循环里只绑着工具问了 2 轮
    assert model.unbound_rounds == 1        # 收尾那一轮未绑工具
    assert reply == "收敛了"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_loop.py
```

预期:`ModuleNotFoundError: No module named 'app.agent'`。

- [ ] **Step 3: 写最小实现**

创建 `app/agent/loop.py`:

```python
"""祛魅热身:最裸的 Agent 循环(**临时产物**,任务 8 落地图后删除)。

没有图、没有 State、没有 checkpointer —— 只有一个 for:
调模型 → 有 tool_call 就执行并回灌 → 没有就收敛。
它存在的意义只是证明「Agent 就是个带工具的循环」,不是魔法。
"""

from langchain_core.messages import ToolMessage

from app.tools.executor import execute_tool


async def run_agent_loop(
    *,
    model,
    messages: list,
    tools,
    registry: dict,
    settings,
    max_steps: int = 5,
) -> tuple[str, int]:
    """跑一轮最裸的 ReAct。返回 (回复文本, 实际步数)。

    每步 `bind_tools` 调模型:无 tool_calls 即收敛;有则逐个执行并回灌。
    步数用尽时**最后一轮不绑 tools** —— 模型在结构上无法再调,必然收敛。
    这是「停止条件是结构保证、不是提示词约定」的第一处体现。
    """
    bound = model.bind_tools(list(tools))
    msgs = list(messages)
    parts: list[str] = []

    for step in range(1, max_steps + 1):
        ai = await bound.ainvoke(msgs)
        parts.append(getattr(ai, "text", "") or "")
        tool_calls = list(getattr(ai, "tool_calls", None) or [])
        if not tool_calls:
            return "".join(parts), step

        msgs.append(ai)
        for call in tool_calls:
            outcome = await execute_tool(
                tool_call=call, registry=registry, settings=settings
            )
            msgs.append(ToolMessage(content=outcome.content, tool_call_id=call["id"]))

    # 步数用尽:不绑 tools 再问一次,收尾。
    final = await model.ainvoke(msgs)
    parts.append(getattr(final, "text", "") or "")
    return "".join(parts), max_steps
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_loop.py
```

预期:`3 passed`。

- [ ] **Step 5: 提交**

```bash
git add app/agent/loop.py tests/test_agent_loop.py
git commit -m "feat: ch05 祛魅热身 —— 手写最裸 Agent 循环(临时产物,图落地后删)"
```

---

## Task 2:State schema + 路由纯函数 + emit 适配

**Files:**
- Create: `app/agent/state.py`
- Create: `app/agent/routing.py`
- Create: `app/agent/emit.py`
- Create: `tests/test_agent_routing.py`

**Interfaces:**
- Produces:
  - `ChatState`(TypedDict,见下)
  - `INTENT_TO_ROUTE: dict[str, str]`、`INTENT_LABELS: tuple[str, ...]`
  - `KNOWLEDGE/BUSINESS/COMPLAINT/CHITCHAT/FALLBACK: str`
  - `route_by_intent(state) -> str`
  - `make_emitter(collector: Callable[[dict], None] | None = None) -> Callable[[dict], None]`
    —— `collector` 是**可调用**(单测里传 `frames.append`),不是 list。传 list 会
    `TypeError: 'list' object is not callable`。

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_agent_routing.py`:

```python
"""分流规则是确定性骨架的核心,必须表驱动全覆盖 —— 含越界与缺字段。"""

import pytest

from app.agent.routing import (
    BUSINESS,
    CHITCHAT,
    COMPLAINT,
    FALLBACK,
    INTENT_LABELS,
    INTENT_TO_ROUTE,
    KNOWLEDGE,
    route_by_intent,
)

CASES = [
    ("商品咨询", KNOWLEDGE),
    ("退款退货", KNOWLEDGE),
    ("物流", BUSINESS),
    ("订单", BUSINESS),
    ("售后", BUSINESS),
    ("投诉", COMPLAINT),
    ("闲聊", CHITCHAT),
]


@pytest.mark.parametrize("intent,expected", CASES)
def test_seven_intents_map_to_their_outlets(intent, expected):
    assert route_by_intent({"intent": intent}) == expected


def test_all_seven_intents_are_covered():
    """七类一个不漏 —— 少一类会静默落进兜底,而兜底不调模型,问题就永远答不上。"""
    assert set(INTENT_TO_ROUTE) == {c[0] for c in CASES}
    assert INTENT_LABELS == tuple(INTENT_TO_ROUTE)


@pytest.mark.parametrize("bad", ["其他", "", "投诉 ", "COMPLAINT", "退款退货 "])
def test_unknown_or_malformed_intent_falls_back(bad):
    """越界/空串/带空格一律兜底 —— 不是 schema 校验,是**走向**的兜底。"""
    assert route_by_intent({"intent": bad}) == FALLBACK


@pytest.mark.parametrize("state", [{}, {"intent": None}])
def test_missing_intent_falls_back(state):
    assert route_by_intent(state) == FALLBACK


def test_make_emitter_outside_graph_degrades_to_collector():
    """图外调用必须退化到 collector,不能抛 RuntimeError(见 spec §12 订正)。

    (旧名 `..._is_a_noop_collector` 名实不符:走的就是那条**会调用** collector
    的分支 —— 本项目已抓到过一次「名字说回灌、函数体在断言 raises」。)
    """
    from app.agent.emit import make_emitter

    got = []
    emit = make_emitter(got.append)
    emit({"frame": "token", "text": "x"})
    assert got == [{"frame": "token", "text": "x"}]
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_routing.py
```

预期:`ModuleNotFoundError: No module named 'app.agent.routing'`。

- [ ] **Step 3: 写最小实现**

创建 `app/agent/state.py`:

```python
"""ch05 图状态。

用 `TypedDict` + `Annotated` reducer 而非 Pydantic:这是 LangGraph 的惯用法。
节点返回的是**部分**键,由 reducer 合并(`trace` 用 `operator.add` 追加、
其余是**覆盖**语义)—— 搞错会得到「上一轮的值和这一轮混在一起」的诡异状态。
"""

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

from app.schemas import Message


class IntentResult(BaseModel):
    """意图识别的结构化出参。取值越界由 routing 兜底,这里只描述形状。"""

    intent: str = Field(
        description="物流 / 订单 / 商品咨询 / 退款退货 / 售后 / 投诉 / 闲聊 之一;"
        "无法归入任何一类时为「其他」。"
    )


class ChatState(TypedDict):
    """贯穿全图的状态。节点返回的是**部分**键,由 reducer 合并(LangGraph 语义)。"""

    # ---- 输入 ----
    conversation_id: str          # = thread_id = MySQL 会话 id
    user_input: str
    history: list[Message]        # 来自 MySQL(跨轮权威源),转 BaseMessage 只经 prompts.py

    # ---- 指代消解 ----
    resolved_input: str           # 本章 = user_input 原样

    # ---- 意图与检索 ----
    intent: str                   # 七类之一 | "其他"
    evidence: list[dict]          # 知识类:检索到的 chunk(含 score/section_path/chunk_id)
    gate_passed: bool

    # ---- Agent ----
    # 刻意**不**放 ReAct 的消息序列:agent 节点在那一轮内部用局部变量组装
    # (`_stream_round` 的 msgs),落库走 append_turn 的 user+assistant 两条。
    # 放一个没人读写的 state 字段 = 死代码 + 白搭一个 reducer。
    agent_steps: int
    tool_calls_made: list[dict]

    # ---- 输出 ----
    reply: str
    citations: list[dict]
    choices: list[str]            # ["handoff", "ticket"];空则不推帧

    # ---- 日志 ----
    trace: Annotated[list[str], operator.add]
    usage: dict
```

创建 `app/agent/routing.py`:

```python
"""意图 → 出口的映射。

**纯函数、无 IO、无模型调用** —— 这是「确定性骨架」的核心:模型只能决定
**意图标签**,不能决定**走向**。分流规则写死在代码里,故可以表驱动单测
穷举七类 + 越界 + 缺字段。
"""

KNOWLEDGE = "knowledge"
BUSINESS = "business"
COMPLAINT = "complaint"
CHITCHAT = "chitchat"
FALLBACK = "fallback"

#: 意图识别失败或输出越界时统一落这个标签,再由本表送进兜底出口。
OTHER = "其他"

#: 七类意图 → 四个出口。
#: 商品咨询 / 退款退货 → 知识(强制预检索;退款退货的 Agent 仍可自调订单工具);
#: 物流 / 订单 / 售后 → 业务数据(直接进 Agent 调工具,无检索证据故不过置信度闸);
#: 投诉、闲聊各有专属出口。
INTENT_TO_ROUTE: dict[str, str] = {
    "商品咨询": KNOWLEDGE,
    "退款退货": KNOWLEDGE,
    "物流": BUSINESS,
    "订单": BUSINESS,
    "售后": BUSINESS,
    "投诉": COMPLAINT,
    "闲聊": CHITCHAT,
}

#: 意图识别 Prompt 里允许输出的标签(与 INTENT_TO_ROUTE 同源,避免两处各写一份)。
INTENT_LABELS: tuple[str, ...] = tuple(INTENT_TO_ROUTE)


def route_by_intent(state) -> str:
    """七类之一 → 四出口;其余(解析失败 / 越界 / 缺字段)一律兜底。

    这里**不做**任何清洗(不 strip、不大小写归一):清洗会让「投诉 」这种
    近乎正确的输入静默走进投诉出口,而它更可能是一次真正的分类失败。
    宁可让它进兜底,兜底话术会请用户再说一遍。
    """
    return INTENT_TO_ROUTE.get(state.get("intent") or "", FALLBACK)
```

创建 `app/agent/emit.py`:

```python
"""把 `get_stream_writer()` 包一层,让节点能脱离图被单测。

真机验证(langgraph 1.2.11):图运行之外调 `get_stream_writer()` 抛
`RuntimeError: Called get_config outside of a runnable context`。

**关键:必须在「每次发帧时」才去取 writer,不能在 `make_emitter()` 里取一次。**
`make_emitter()` 是在**端点里**调的 —— 那时图还没开始跑,上下文里没有 writer,
一次性的取法会永远拿到 no-op 分支:前端**一帧都收不到**,而所有单测仍然全绿
(单测把 collector 直接注入节点,根本不经过这里)。这正是本项目最怕的那类
「假绿 + 静默故障」。

约定:发出的 payload 形如 `{"frame": <名>, ...字段}`,由 `app/api/chat.py`
逐条翻成 SSE 帧。**节点的出站协议只有这一个形状。**
"""

from collections.abc import Callable

from langgraph.config import get_stream_writer


def make_emitter(collector: Callable[[dict], None] | None = None) -> Callable[[dict], None]:
    """返回一个 emit 函数:图运行中发真帧,图外退化成 collector(或静默丢弃)。

    每次调用都重新取一次 writer —— 这样同一个 emitter 既能被节点在图里用,
    又能在图外被单测直接调,不需要调用方关心自己在不在图里。
    """

    def emit(payload: dict) -> None:
        try:
            writer = get_stream_writer()
        except RuntimeError:
            # 图外:单测路径。没给 collector 就丢弃。
            if collector is not None:
                collector(payload)
            return
        writer(payload)

    return emit
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_routing.py
```

预期:`16 passed`(7 参数化 + 1 + 5 参数化 + 2 参数化 + 1)。

> ⚠️ 上面这条用例只证明了「图外退化」。**图内真的把帧送出去**由任务 7 的
> `test_emitter_sends_frames_through_astream_custom_mode` 覆盖 —— 缺了它,
> 把 emit 写成一次性取 writer 的错误版本也能全绿。

- [ ] **Step 5: 提交**

```bash
git add app/agent/state.py app/agent/routing.py app/agent/emit.py tests/test_agent_routing.py
git commit -m "feat: ch05 图状态 schema + 分流纯函数 + emit 适配层"
```

---

## Task 3:意图识别节点 + Prompt + 标注样例验证

意图识别 Prompt 是**非可单测产出**(用户工作要求 1),TDD 那步换成**拿标注样例跑一遍**。

**Files:**
- Modify: `app/prompts.py`(追加)
- Create: `app/agent/nodes.py`(本任务只放这一个节点工厂)
- Create: `tests/test_agent_intent.py`
- Create: `evals/intent_cases.jsonl`
- Create: `scripts/run_intent_eval.py`

**Interfaces:**
- Consumes: `INTENT_LABELS`、`OTHER`(任务 2)、`ChatState`(任务 2)
- Produces: `make_classify_intent_node(model) -> Callable[[ChatState], Awaitable[dict]]`;`build_intent_messages(text) -> list`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_agent_intent.py`:

```python
"""意图识别节点:解析失败/越界必须降级为「其他」,**不许抛异常**。"""

import pytest
from langchain_core.exceptions import OutputParserException

from app.agent.nodes import make_classify_intent_node
from app.agent.routing import OTHER


class FakeStructuredModel:
    """替身:with_structured_output 返回一个可注入结果或异常的链。"""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def with_structured_output(self, schema, method=None):
        assert method == "json_mode", "抽取类出参只能用 json_mode(本项目硬约束)"
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return self.result


class _Intent:
    def __init__(self, intent):
        self.intent = intent


@pytest.mark.anyio
@pytest.mark.parametrize("intent", ["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"])
async def test_seven_labels_pass_through(intent):
    model = FakeStructuredModel(result=_Intent(intent))
    node = make_classify_intent_node(model=model)
    out = await node({"user_input": "随便问点什么"})
    assert out["intent"] == intent
    assert out["trace"] == [f"classify_intent:{intent}"]


@pytest.mark.anyio
async def test_out_of_vocabulary_intent_degrades_to_other():
    """模型吐了七类之外的标签 → 「其他」,由路由送兜底。"""
    model = FakeStructuredModel(result=_Intent("退款"))
    node = make_classify_intent_node(model=model)
    assert (await node({"user_input": "x"}))["intent"] == OTHER


@pytest.mark.anyio
async def test_parse_failure_degrades_to_other_instead_of_raising():
    """解析失败**不许**抛 —— 意图识别是骨架第一步,它的失败不该毁掉整轮对话。"""
    model = FakeStructuredModel(error=OutputParserException("模型输出不是 JSON"))
    node = make_classify_intent_node(model=model)
    out = await node({"user_input": "x"})
    assert out["intent"] == OTHER
    assert out["trace"] == [f"classify_intent:{OTHER}"]


@pytest.mark.anyio
async def test_prompt_carries_the_user_utterance():
    model = FakeStructuredModel(result=_Intent("闲聊"))
    node = make_classify_intent_node(model=model)
    await node({"user_input": "你好呀"})
    assert "你好呀" in model.calls[0][-1].text
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_intent.py
```

预期:`ImportError: cannot import name 'make_classify_intent_node'`。

- [ ] **Step 3: 写 Prompt**

在 `app/prompts.py` 末尾追加(注意:**必须出现字面 `JSON`,且**不得有裸花括号**):

```python
INTENT_SYSTEM_PROMPT = """你是电商客服的意图识别助手。
判断用户这一句话属于下面七类中的哪一类,并以 JSON 对象输出。

七类:
- 物流:查询包裹位置、发货进度、配送时效
- 订单:查询订单状态、金额、下单时间
- 商品咨询:咨询商品价格、库存、规格、功能
- 退款退货:申请退款、退货、换货,或询问相关政策
- 售后:商品质量问题、维修、补发、安装
- 投诉:表达不满、要求赔偿、要求人工处理
- 闲聊:问候、感谢、与购物无关的闲聊

输出一个 JSON 对象,只有 intent 一个字段,取值为上述七类之一的原文。
无法归入任何一类时,intent 输出「其他」。
不要输出 JSON 以外的任何内容。"""

_INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", INTENT_SYSTEM_PROMPT), ("human", "{text}")]
)


def build_intent_messages(text: str) -> list:
    """组装意图识别的消息。"""
    return _INTENT_PROMPT.format_messages(text=text)
```

- [ ] **Step 4: 写节点实现**

创建 `app/agent/nodes.py`(本任务只放这一个工厂):

```python
"""ch05 各节点的工厂函数。

统一用**闭包工厂**(同 `app/tools/business.py` 的既有做法):模型、工具、
会话、检索器都不是模块级单例,而是每请求经闭包绑定 —— 这样节点既拿到了
依赖,又不会在 import 时做任何 IO。
"""

import logging

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from app.agent.routing import INTENT_TO_ROUTE, OTHER
from app.agent.state import IntentResult
from app.prompts import build_intent_messages

logger = logging.getLogger(__name__)


def make_classify_intent_node(*, model):
    """意图识别:一次 LLM(json_mode),输出七类之一。

    解析失败或越界一律降级为「其他」 —— **不抛异常**。理由:意图识别是骨架
    的第一步,它失败时整轮对话不该跟着崩;降级后由 `route_by_intent` 送进
    兜底出口,用户至少能拿到一句「请再说具体些」。
    """
    chain = model.with_structured_output(IntentResult, method="json_mode")

    async def classify_intent(state) -> dict:
        try:
            result = await chain.ainvoke(build_intent_messages(state["user_input"]))
            intent = result.intent
        except (OutputParserException, ValidationError) as exc:
            logger.warning("意图识别解析失败,降级为「其他」:%s", exc)
            intent = OTHER

        if intent not in INTENT_TO_ROUTE:
            intent = OTHER
        return {"intent": intent, "trace": [f"classify_intent:{intent}"]}

    return classify_intent
```

- [ ] **Step 5: 跑测试确认通过**

```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_intent.py
```

预期:`10 passed`。

- [ ] **Step 6: 建标注样例集**

创建 `evals/intent_cases.jsonl`(每行一条;`expected` 是七类之一或 `其他`):

```jsonl
{"text": "我的包裹什么时候能到", "expected": "物流"}
{"text": "快递到哪了", "expected": "物流"}
{"text": "发货了吗", "expected": "物流"}
{"text": "订单 1001 什么状态", "expected": "订单"}
{"text": "我这个单子多少钱", "expected": "订单"}
{"text": "我什么时候下的单", "expected": "订单"}
{"text": "智能猫砂盆 Pro 多少钱", "expected": "商品咨询"}
{"text": "MH-LP100 有货吗", "expected": "商品咨询"}
{"text": "这个保温杯什么规格", "expected": "商品咨询"}
{"text": "我要退款", "expected": "退款退货"}
{"text": "七天无理由怎么算", "expected": "退款退货"}
{"text": "我想换成另一个尺码", "expected": "退款退货"}
{"text": "收到的杯子是碎的", "expected": "售后"}
{"text": "键盘用了两天就坏了,能修吗", "expected": "售后"}
{"text": "少发了一件,能补发吗", "expected": "售后"}
{"text": "我要投诉你们客服", "expected": "投诉"}
{"text": "太差了,我要找你们领导", "expected": "投诉"}
{"text": "这什么破服务,转人工", "expected": "投诉"}
{"text": "你好", "expected": "闲聊"}
{"text": "谢谢啦", "expected": "闲聊"}
{"text": "今天天气不错", "expected": "闲聊"}
{"text": "帮我写一首关于月亮的诗", "expected": "其他"}
{"text": "asdkjhaskjd", "expected": "其他"}
{"text": "你们老板是谁", "expected": "其他"}
```

- [ ] **Step 7: 写评估脚本并跑**

创建 `scripts/run_intent_eval.py`:

```python
"""意图识别标注样例评估。需真实 key(打网络),不属于单测。

用法:
    .venv/Scripts/python.exe scripts/run_intent_eval.py
报告每条的期望/实际与总准确率;**不做字符串断言**,只出数字
(deepseek 在 temperature=0 下依然非确定)。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.nodes import make_classify_intent_node
from app.config import get_settings
from app.llm import create_extract_model

CASES = Path(__file__).resolve().parents[1] / "evals" / "intent_cases.jsonl"


def emit(line: str = "") -> None:
    # 控制台是 cp936;`✓`/`✗` 不在 GBK 里,直接 print 会崩。
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        print(line)
        return
    stream.write(line.encode("utf-8") + b"\n")
    stream.flush()


async def main() -> int:
    settings = get_settings()
    node = make_classify_intent_node(model=create_extract_model(settings))
    rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]

    hit = 0
    for row in rows:
        got = (await node({"user_input": row["text"]}))["intent"]
        ok = got == row["expected"]
        hit += ok
        emit(f"{'OK ' if ok else 'MISS'} 期望={row['expected']:<6} 实际={got:<6} {row['text']}")

    emit()
    emit(f"准确率 {hit}/{len(rows)} = {hit / len(rows):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

跑:

```bash
.venv/Scripts/python.exe scripts/run_intent_eval.py
```

预期:准确率 reported。**低于 80% 就把误判样例补进 Prompt 或样例集并在
`dev-notes/ch05.md` 记账** —— 不要为了数字好看删样例。

- [ ] **Step 8: 提交**

```bash
git add app/prompts.py app/agent/nodes.py tests/test_agent_intent.py evals/intent_cases.jsonl scripts/run_intent_eval.py
git commit -m "feat: ch05 意图识别节点 + 七类 Prompt + 标注样例评估集"
```

---

## Task 4:三个固定话术出口(闲聊 / 兜底 / 投诉)

**关键断言**:闲聊与兜底**不调模型**(用户需求 3:「不花模型调用」)。
投诉要发 `choices` 帧。

**Files:**
- Modify: `app/agent/nodes.py`(追加)
- Create: `tests/test_agent_fixed_replies.py`

**Interfaces:**
- Produces: `make_chitchat_reply_node(*, emit)`、`make_fallback_reply_node(*, emit)`、`make_complaint_reply_node(*, emit)`(三个都收 `emit` —— 每个出口都必须自己发 `token` 帧,否则前端气泡是空的);常量 `CHITCHAT_REPLY`、`FALLBACK_REPLY`、`COMPLAINT_REPLY`、`CHOICE_HANDOFF`、`CHOICE_TICKET`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_agent_fixed_replies.py`:

```python
"""三个固定话术出口:闲聊/兜底零模型调用,投诉额外发 choices 帧。"""

import pytest

from app.agent.nodes import (
    CHOICE_HANDOFF,
    CHOICE_TICKET,
    CHITCHAT_REPLY,
    COMPLAINT_REPLY,
    FALLBACK_REPLY,
    make_chitchat_reply_node,
    make_complaint_reply_node,
    make_fallback_reply_node,
)


@pytest.mark.anyio
async def test_chitchat_emits_token_frame_so_the_bubble_is_not_blank():
    """**必须发 token 帧** —— 前端靠累积 token 画气泡。

    只写 `state["reply"]` 的话:后端 state 里有话、前端气泡是**空的**,
    而所有断言 `out["reply"]` 的单测全绿。验收 4 会直接失败。
    """
    frames = []
    node = make_chitchat_reply_node(emit=frames.append)
    out = await node({"user_input": "你好"})
    assert out["reply"] == CHITCHAT_REPLY
    assert out["choices"] == []          # 闲聊不给按钮
    assert out["trace"] == ["chitchat_reply"]
    assert frames == [{"frame": "token", "text": CHITCHAT_REPLY}]


@pytest.mark.anyio
async def test_fallback_emits_token_frame():
    frames = []
    node = make_fallback_reply_node(emit=frames.append)
    out = await node({"user_input": "帮我写诗", "intent": "其他"})
    assert out["reply"] == FALLBACK_REPLY
    assert out["choices"] == []
    assert out["trace"] == ["fallback_reply"]
    assert frames == [{"frame": "token", "text": FALLBACK_REPLY}]


@pytest.mark.anyio
async def test_complaint_emits_token_then_choices_frame():
    frames = []
    node = make_complaint_reply_node(emit=frames.append)
    out = await node({"user_input": "我要投诉"})

    assert out["reply"] == COMPLAINT_REPLY
    assert out["choices"] == ["handoff", "ticket"]
    assert frames == [
        {"frame": "token", "text": COMPLAINT_REPLY},
        {"frame": "choices", "options": [CHOICE_HANDOFF, CHOICE_TICKET]},
    ]
    # 两个选项是**两件事**,键必须不同且都在
    assert {o["key"] for o in frames[1]["options"]} == {"handoff", "ticket"}


def test_fixed_copy_factories_take_no_model_at_all():
    """静态保证:这三个工厂的签名里根本没有 model 参数。

    比「实现里记得别调模型」强 —— 参数不存在 = 结构上不可能调。
    """
    import inspect

    for factory in (make_chitchat_reply_node, make_fallback_reply_node,
                    make_complaint_reply_node):
        assert "model" not in inspect.signature(factory).parameters
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_fixed_replies.py
```

预期:`ImportError: cannot import name 'CHITCHAT_REPLY'`。

- [ ] **Step 3: 写实现**

在 `app/agent/nodes.py` 追加(实现里不引入任何模型 —— 不 import、不接收):

```python
# ---- 固定话术出口 ----
#
# 三个出口**都不调模型**:闲聊与兜底是纯文案(零成本、零延迟),投诉的安抚话术
# 同样固定。这三个工厂的签名里因此根本没有 model 参数 —— 那是「不花模型调用」
# 的**结构保证**,不是「我们记得别调」的行为约定。

COMPLAINT_REPLY = (
    "非常抱歉给您带来不好的体验,您反馈的问题我已经记录下来了。"
    "您可以让我转人工客服,或者为您建一张工单跟进,选哪个都可以。"
)
CHITCHAT_REPLY = "你好呀~我是本店客服小猫,有什么可以帮您的吗?"
FALLBACK_REPLY = "抱歉,我没太理解您的意思,可以再说得具体一些吗?"

CHOICE_HANDOFF = {"key": "handoff", "label": "转人工"}
CHOICE_TICKET = {"key": "ticket", "label": "建工单"}


def make_chitchat_reply_node(*, emit):
    """闲聊:固定话术,不推进任何后续动作。

    **必须自己发 token 帧**:前端是「累积 token 画出气泡」的,只写
    `state["reply"]` 的话后端有话、前端空白 —— 而且所有断言 `reply` 的单测
    照样全绿,验收 4 才会暴露。
    """

    async def chitchat_reply(state) -> dict:
        emit({"frame": "token", "text": CHITCHAT_REPLY})
        return {"reply": CHITCHAT_REPLY, "choices": [], "trace": ["chitchat_reply"]}

    return chitchat_reply


def make_fallback_reply_node(*, emit):
    """兜底:意图不属于七类、或分类解析失败时走这里(用户明确要求)。

    也是置信度闸不通过时的落点 —— 那时问题已由 confidence_gate 落进低置信度池。
    """

    async def fallback_reply(state) -> dict:
        emit({"frame": "token", "text": FALLBACK_REPLY})
        return {"reply": FALLBACK_REPLY, "choices": [], "trace": ["fallback_reply"]}

    return fallback_reply


def make_complaint_reply_node(*, emit):
    """投诉:安抚话术 + 把「转人工」「建工单」两个选项交给用户自己选。

    **两者是两回事,分开给** —— 后端不自动执行任何一个:转人工是前端模拟,
    建工单要用户点了按钮才走 /api/ticket。用户都不点就继续正常对话。
    """

    async def complaint_reply(state) -> dict:
        emit({"frame": "token", "text": COMPLAINT_REPLY})
        emit({"frame": "choices", "options": [CHOICE_HANDOFF, CHOICE_TICKET]})
        return {
            "reply": COMPLAINT_REPLY,
            "choices": ["handoff", "ticket"],
            "trace": ["complaint_reply:choices"],
        }

    return complaint_reply
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_fixed_replies.py
```

预期:`4 passed`。

- [ ] **Step 4b: 补一条意图标签的漂移守卫(控制器补充项,见 ledger Ruling 16)**

> 这一条**不是**本任务范围内的自然延伸,是 T3 审查挖出来的缺口,**由控制器授权补做**。
> 审查者已确认它便宜且堵的是静默失效,故不另开任务、并入本次提交。

追加到 `tests/test_agent_intent.py`(文件已存在,T3 建的):

```python
def test_every_label_in_the_prompt_matches_the_routing_table():
    """标签名在两处各写一份,必须自动对齐。

    `INTENT_SYSTEM_PROMPT` 里的七个标签是**手写**的;`INTENT_TO_ROUTE` 是代码里的
    权威表。谁改了 `app/agent/routing.py` 的键名,模型仍会吐**旧名**,
    `nodes.py` 的越界检查把**整个桶**静默降级成「其他」—— 全部进兜底,零报错。

    既有守卫 `tests/test_agent_routing.py:35`(`INTENT_LABELS == tuple(INTENT_TO_ROUTE)`)
    是恒真式(定义即如此),**正好抓不到这个**;这条才是真的守卫。
    """
    from app.agent.routing import INTENT_LABELS
    from app.prompts import INTENT_SYSTEM_PROMPT

    missing = [label for label in INTENT_LABELS if label not in INTENT_SYSTEM_PROMPT]
    assert not missing, f"Prompt 里缺这些标签名,对应整桶会静默降级成「其他」:{missing}"
```

跑一次确认通过(它**一开始就该是绿的** —— Prompt 里的标签名今天是对的,
这是一条守卫,不是 TDD 的驱动测试):

```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_intent.py
```

预期:`12 passed`。

- [ ] **Step 5: 提交**

```bash
git add app/agent/nodes.py tests/test_agent_fixed_replies.py tests/test_agent_intent.py
git commit -m "feat: ch05 闲聊/兜底/投诉三个固定话术出口(零模型调用 + choices 帧)"
```

---

## Task 5:强制预检索节点 + 置信度闸

**Files:**
- Modify: `app/agent/nodes.py`(追加)
- Create: `tests/test_agent_gate.py`

**Interfaces:**
- Consumes: `KnowledgeRetriever.search(query) -> list[RetrievedChunk]`;`record_low_confidence(session, ...)`
- Produces: `make_retrieve_knowledge_node(*, retriever, emit)`(检索到证据时发 `citations` 帧 —— 见 Ruling 4)、`make_confidence_gate_node(*, settings, session, conversation_id)`

**背景**:`RetrievedChunk` 的字段顺序是 `question, answer, category, chunk_id, section_path, score`。

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_agent_gate.py`:

```python
"""强制预检索 + 置信度闸:阈值边界、落池、以及「闸不过就不进 Agent」。"""

import pytest

from app.agent.nodes import make_confidence_gate_node, make_retrieve_knowledge_node
from app.config import Settings
from app.retrieval.search import RetrievedChunk
from app.tools.errors import ToolInfrastructureError

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **REQUIRED, **overrides)


class FakeRetriever:
    def __init__(self, chunks=(), error=None):
        self.chunks = list(chunks)
        self.error = error
        self.calls = []

    async def search(self, query):
        self.calls.append(query)
        if self.error is not None:
            raise self.error
        return list(self.chunks)


class RecordingSession:
    def __init__(self):
        self.added = []
        self.commits = 0

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1


def _chunk(score, answer="七天无理由退货"):
    return RetrievedChunk("怎么退货", answer, "退换货", chunk_id=7,
                          section_path="退换货 > 退货政策", score=score)


@pytest.mark.anyio
async def test_retrieval_uses_the_resolved_input_and_builds_citations():
    frames = []
    retriever = FakeRetriever([_chunk(0.91)])
    node = make_retrieve_knowledge_node(retriever=retriever, emit=frames.append)
    out = await node({"resolved_input": "怎么退货"})
    assert retriever.calls == ["怎么退货"]
    assert out["evidence"][0]["score"] == 0.91
    assert out["citations"] == [{
        "n": 1, "chunk_id": 7, "section_path": "退换货 > 退货政策",
        "question": "怎么退货", "answer": "七天无理由退货", "category": "退换货",
    }]
    assert out["trace"] == ["retrieve_knowledge:1 hits top=0.91"]


@pytest.mark.anyio
async def test_citations_are_emitted_as_a_frame():
    """**citations 必须发帧** —— 前端靠它渲染可点击的引用来源。

    只写进 state 的话:ch04 的引用 UI 静默失效、老的 acceptance.sh 回归,
    而所有断言 `out["citations"]` 的单测**照样全绿**。
    """
    frames = []
    node = make_retrieve_knowledge_node(retriever=FakeRetriever([_chunk(0.91)]),
                                        emit=frames.append)
    out = await node({"resolved_input": "怎么退货"})
    # 载荷键是 `items` —— ch04 前端读的就是 `payload.items`(见 index.html:368)。
    # 断言写成 `{"citations": ...}` 的话,把键改错也照样绿。
    assert frames == [{"frame": "citations", "items": out["citations"]}]


@pytest.mark.anyio
async def test_empty_retrieval_is_recorded_in_trace_not_an_error():
    frames = []
    node = make_retrieve_knowledge_node(retriever=FakeRetriever([]), emit=frames.append)
    out = await node({"resolved_input": "没有的问题"})
    assert out["evidence"] == []
    assert out["citations"] == []
    assert out["trace"] == ["retrieve_knowledge:0 hits"]
    assert frames == []          # 没证据就不发空引用帧


@pytest.mark.anyio
async def test_infrastructure_failure_propagates_to_502():
    """检索器挂了必须抛上去(→502),绝不降级成「没搜到」。"""
    node = make_retrieve_knowledge_node(
        retriever=FakeRetriever(error=ToolInfrastructureError("检索不可用")),
        emit=lambda p: None,
    )
    with pytest.raises(ToolInfrastructureError):
        await node({"resolved_input": "q"})


@pytest.mark.anyio
async def test_gate_passes_at_threshold_boundary_inclusive():
    """0.58 恰好等于阈值 → 通过(与 ch04 的阈值语义一致:>=)。"""
    session = RecordingSession()
    node = make_confidence_gate_node(
        settings=_settings(retrieval_score_threshold=0.58),
        session=session, conversation_id="conv-1",
    )
    out = await node({"user_input": "q", "evidence": [{"score": 0.58}]})
    assert out["gate_passed"] is True
    assert out["trace"] == ["confidence_gate:pass"]
    assert session.added == []          # 通过时不落池


@pytest.mark.anyio
async def test_gate_fails_below_threshold_and_records_low_confidence():
    session = RecordingSession()
    node = make_confidence_gate_node(
        settings=_settings(retrieval_score_threshold=0.58),
        session=session, conversation_id="conv-1",
    )
    out = await node({"user_input": "怎么退货", "evidence": [{"score": 0.31}]})
    assert out["gate_passed"] is False
    assert out["trace"] == ["confidence_gate:fail"]
    assert len(session.added) == 1
    row = session.added[0]
    assert row.entry_point == "置信度闸"
    assert row.question == "怎么退货"
    assert row.source_conversation_id == "conv-1"
    assert "0.31" in row.reject_reason
    assert session.commits == 1


@pytest.mark.anyio
async def test_gate_fails_on_empty_evidence_and_records_that_reason():
    session = RecordingSession()
    node = make_confidence_gate_node(
        settings=_settings(), session=session, conversation_id="c",
    )
    out = await node({"user_input": "q", "evidence": []})
    assert out["gate_passed"] is False
    assert session.added[0].reject_reason == "检索为空"


@pytest.mark.anyio
async def test_gate_uses_max_score_not_top1_position():
    """判据是**最高分**,不是「第一条的分」—— 顺序由重排决定,取 max 更稳。"""
    session = RecordingSession()
    node = make_confidence_gate_node(
        settings=_settings(retrieval_score_threshold=0.58),
        session=session, conversation_id="c",
    )
    out = await node({"user_input": "q", "evidence": [{"score": 0.2}, {"score": 0.9}]})
    assert out["gate_passed"] is True
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_gate.py
```

预期:`ImportError: cannot import name 'make_retrieve_knowledge_node'`。

- [ ] **Step 3: 写实现**

在 `app/agent/nodes.py` 追加(顶部 import 补 `from app.kb.assess import record_low_confidence`):

```python
# ---- 知识类:强制预检索 + 置信度闸 ----


def make_retrieve_knowledge_node(*, retriever, emit):
    """知识类意图的**强制**预检索(确定性骨架的一步,不走 query_faq 工具)。

    检索器的故障语义原样透传:`KnowledgeRetriever` 把 Milvus/嵌入的故障翻成
    `ToolInfrastructureError`,这里**不接** —— 它必须一路抛到 API 层变 502,
    绝不能被伪装成「没搜到」。

    `citations` **同时**写进 state 并发一帧:state 那份给 Agent 组装引用编号,
    帧那份给前端渲染可点击的来源。少发帧 = ch04 的引用 UI 静默失效。
    """

    async def retrieve_knowledge(state) -> dict:
        chunks = await retriever.search(state["resolved_input"])
        evidence = [
            {
                "chunk_id": c.chunk_id,
                "section_path": c.section_path,
                "question": c.question,
                "answer": c.answer,
                "category": c.category,
                "score": c.score,
            }
            for c in chunks
        ]
        citations = [
            {"n": i + 1, **{k: e[k] for k in
                            ("chunk_id", "section_path", "question", "answer", "category")}}
            for i, e in enumerate(evidence)
        ]
        if citations:
            # 载荷键必须是 **`items`**,不是 `citations`:ch04 的
            # `app/static/index.html` 里是 `ctx.citations = payload.items || []`。
            # 换个键名 = 帧到了、前端仍渲染不出引用(静默失效),而且
            # ch05 的单测只断言「发了一帧」,照样全绿。
            emit({"frame": "citations", "items": citations})
        top = f" top={evidence[0]['score']:.2f}" if evidence else ""
        return {
            "evidence": evidence,
            "citations": citations,
            "trace": [f"retrieve_knowledge:{len(evidence)} hits{top}"],
        }

    return retrieve_knowledge


def make_confidence_gate_node(*, settings, session, conversation_id):
    """置信度闸:卡在检索之后、进 Agent 之前。

    **为什么必须在这儿**:Agent 的答复是流式吐给用户的,等答完再判就晚了
    (ch04 的自评正是那个位置,本章把它撤掉)。证据弱就直接回兜底话术、
    不进 Agent,同时把问题落池留给后面的数据飞轮。

    判据是纯**检索分数阈值**(取最高分),零额外模型调用 —— 最简版;
    正式的置信度检查留给「可观测」那章。
    """

    async def confidence_gate(state) -> dict:
        scores = [e["score"] for e in (state.get("evidence") or [])]
        passed = bool(scores) and max(scores) >= settings.retrieval_score_threshold

        if not passed:
            await record_low_confidence(
                session,
                question=state["user_input"],
                source_conversation_id=conversation_id,
                entry_point="置信度闸",
                reject_reason=(
                    "检索为空"
                    if not scores
                    else f"最高分 {max(scores):.2f} 低于阈值 {settings.retrieval_score_threshold}"
                ),
            )

        return {
            "gate_passed": passed,
            "trace": [f"confidence_gate:{'pass' if passed else 'fail'}"],
        }

    return confidence_gate
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_gate.py
```

预期:`8 passed`。

> ⚠️ **已知且已裁定,不要"顺手修"(ledger Ruling 15)**:检索器内部**已经**按
> `settings.retrieval_score_threshold` 滤过一遍(`app/retrieval/search.py:72`),
> 而 `build_retriever` 传的就是**闸用的同一个值**(`app/tools/registry.py:41`)。
> 所以真实链路上走到闸的证据**分数必然已达标**,`test_gate_fails_below_threshold...`
> 之所以能过,是因为它用**假检索器**塞了一条 0.31 —— 真实检索器造不出这个状态。
> 也就是说:**生产上唯一会触发的失败是「检索为空」**,`reject_reason` 里那句
> 「最高分 … 低于阈值 …」是死分支。
>
> **这是本章「最简版」的已知边界,不是 bug,不要改检索器、不要传低阈值。**
> 用户要的走向(证据弱 → 直接兜底、不进 Agent)现在就是对的:弱证据被检索器
> 滤成空 → 闸拦下 → 兜底。正式的置信度检查已明确推给「可观测」那章。

- [ ] **Step 5: 提交**

```bash
git add app/agent/nodes.py tests/test_agent_gate.py
git commit -m "feat: ch05 强制预检索节点 + 置信度闸(分数阈值 + 落低置信度池)"
```

---

## Task 6:主力 Agent 节点(ReAct)+ 配置项

**Files:**
- Modify: `app/config.py`(追加两个配置)
- Modify: `app/prompts.py`(追加 `render_evidence`;给 `build_messages` 加 `evidence` 可选参)
- Modify: `app/agent/nodes.py`(追加 `make_agent_node`)
- Modify: `tests/test_config.py`(追加两条)
- Create: `tests/test_agent_node.py`

**Interfaces:**
- Consumes: `execute_tool`(任务 1 在用)、`build_messages`(**扩展 `evidence` 可选参**,不新建平行函数)
- Produces: `make_agent_node(*, model, tools, registry, settings, emit)`;配置 `max_agent_steps`、`agent_token_budget`

- [ ] **Step 1: 写失败的配置测试**

在 `tests/test_config.py` 末尾追加:

```python
def test_agent_step_limit_must_be_positive():
    """<=0 会让 ReAct 循环一次都不跑 —— 必须在启动时炸,不能运行时静默。"""
    import pytest

    with pytest.raises(ValueError):
        Settings(_env_file=None, **REQUIRED, max_agent_steps=0)


def test_agent_token_budget_must_be_positive():
    import pytest

    with pytest.raises(ValueError):
        Settings(_env_file=None, **REQUIRED, agent_token_budget=0)
```

> 注意:`tests/test_config.py` 里已有的 `REQUIRED` 常量与 `_settings` 辅助直接复用;
> 若该文件没有模块级 `REQUIRED`,把任务开头 `tests/test_agent_loop.py` 里的那份抄过来。

- [ ] **Step 2: 跑配置测试确认失败**

```bash
.venv/Scripts/python.exe -m pytest tests/test_config.py
```

预期:两条失败(`max_agent_steps` 不是已知字段 → 被 `extra="ignore"` 吞掉,构造不报错)。

- [ ] **Step 3: 加配置**

在 `app/config.py` 的 `Settings` 里,`ch03` 那段之前插入:

```python
    # ch05 编排。两个数都加了界:写错要在启动时炸,不能等运行时变成
    # 「ReAct 循环一次都不跑」或「预算恒超 → 第一步就强制收敛」这种静默故障。
    max_agent_steps: int = Field(default=5, ge=1)
    agent_token_budget: int = Field(default=20000, ge=1)
```

- [ ] **Step 4: 跑配置测试确认通过**

```bash
.venv/Scripts/python.exe -m pytest tests/test_config.py
```

预期:全绿。

- [ ] **Step 5: 写 Agent 节点的失败测试**

创建 `tests/test_agent_node.py`:

```python
"""主力 Agent 的 ReAct 循环:收敛、工具回灌、停止条件、token 预算、流式。"""

import pytest
from langchain_core.messages import ToolMessage

from app.agent.nodes import make_agent_node
from app.config import Settings
from app.schemas import Message
from app.tools.errors import ToolInfrastructureError

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **REQUIRED, **overrides)


def _text(msg) -> str:
    """1.x 里**流式 chunk** 的 content 是 block 列表;这里的 BaseMessage 是
    我们自己用字符串构造的,content 仍是 str —— 两种都兜住,免得断言靠猜。"""
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


class FakeChunk:
    def __init__(self, text="", tool_calls=None, usage=None):
        self.text = text
        # 必须带 "type": "tool_call" —— 见 CLAUDE.md;缺键时 BaseTool.ainvoke
        # 会把整个 dict 当**参数**去校验 schema,每次调用都变成「参数不合法」。
        self.tool_calls = [{"type": "tool_call", **tc} for tc in (tool_calls or [])]
        self.usage_metadata = usage

    def __add__(self, other):
        # 累积 chunk 时**原样**拼接 tool_calls,**必须保留 `"type": "tool_call"`**。
        # 它是 `BaseTool.ainvoke` 判定「这是一个工具调用」的**唯一**依据
        # (CLAUDE.md 的硬约束);`execute_tool` 则原样把 dict 透传给
        # `tool.ainvoke`(`app/tools/executor.py:65`)。丢了它,真实 `@tool` 会把
        # 整个 dict 当**参数**去校验 schema,每次调用都退化成「参数不合法」的
        # 可恢复失败 —— 而本文件的替身工具不查这个键,所以**测试照样全绿**。
        #
        # 说明:本替身**不模拟** LangChain 真实的按 index 合并 + 分片 args 拼接
        # (真实模型会把一次工具调用拆成多个 chunk 流式吐出来),因为那些测试
        # 里每次工具调用都写在**单个** chunk 里,`__add__` 的合并分支用不到。
        # 真实多 chunk 的累积由 T11 的验收 5(真模型、要求 ReAct 不止一步)兜底。
        return FakeChunk(
            text=self.text + other.text,
            tool_calls=self.tool_calls + other.tool_calls,
            usage=other.usage_metadata or self.usage_metadata,
        )


class _BoundModel:
    def __init__(self, owner):
        self._owner = owner

    async def astream(self, messages):
        self._owner.bound_rounds += 1
        self._owner.bound_messages = list(messages)   # 断言入参用
        for chunk in self._owner.rounds.pop(0):
            yield chunk


class ScriptedModel:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.bound_rounds = 0
        self.unbound_rounds = 0

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return _BoundModel(self)

    async def astream(self, messages):
        # 这里**刻意不记**入参:agent 每一轮都绑着工具问,未绑工具的入口只在
        # 步数用尽收尾时走一次 —— 把「回灌的 ToolMessage」记在这儿会永远看不到。
        # 要断回灌就读 `bound_messages`(Task 1 已踩过一次,见 ledger Ruling 6)。
        self.unbound_rounds += 1
        for chunk in self.rounds.pop(0):
            yield chunk


class FakeTool:
    name = "query_order"

    def __init__(self, content='{"status": "已发货"}', error=None):
        self.content = content
        self.error = error
        self.calls = []

    async def ainvoke(self, tool_call):
        self.calls.append(tool_call)
        if self.error is not None:
            raise self.error
        return type("_R", (), {"content": self.content})()


def _node(model, tools=(), registry=None, settings=None, frames=None):
    return make_agent_node(
        model=model, tools=list(tools), registry=registry or {},
        settings=settings or _settings(),
        emit=(frames.append if frames is not None else (lambda p: None)),
    )


def _state(**over):
    base = {"conversation_id": "c1", "user_input": "订单 1001 发货了吗",
            "resolved_input": "订单 1001 发货了吗", "history": []}
    base.update(over)
    return base


@pytest.mark.anyio
async def test_direct_text_answer_converges_without_tools():
    frames = []
    model = ScriptedModel([[FakeChunk("你的"), FakeChunk("订单已发货。")]])
    out = await _node(model, frames=frames).__call__(_state())
    assert out["reply"] == "你的订单已发货。"
    assert out["agent_steps"] == 1
    assert out["tool_calls_made"] == []
    assert frames == [
        {"frame": "token", "text": "你的"},
        {"frame": "token", "text": "订单已发货。"},
    ]


@pytest.mark.anyio
async def test_tool_call_round_emits_frames_and_feeds_result_back():
    frames = []
    tool_obj = FakeTool()
    model = ScriptedModel([
        [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}])],
        [FakeChunk("已发货。")],
    ])
    out = await _node(model, [tool_obj], {"query_order": tool_obj}, frames=frames).__call__(_state())

    assert out["reply"] == "已发货。"
    assert out["agent_steps"] == 2
    assert out["tool_calls_made"] == [{"name": "query_order", "ok": True}]
    assert {"frame": "tool_call", "name": "query_order",
            "args": {"order_id": "1001"}, "tool_call_id": "c1"} in frames
    assert {"frame": "tool_result", "tool_call_id": "c1", "ok": True,
            "summary": '{"status": "已发货"}'} in frames
    # 回灌的 ToolMessage 必须与 tool_call_id 配对。
    # 注意记的是**绑工具**那个入口:agent 每一轮都绑着工具问,未绑工具的入口
    # 只在步数用尽收尾时走一次。
    fed = [m for m in model.bound_messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in fed] == ["c1"]
    # 钉住 `msgs.append(acc)` 那一行:删掉它,上面 fed 的断言**依然全绿**,
    # 而真实链路上会退化成「有 tool 消息、没有前置的 assistant(tool_calls)消息」——
    # OpenAI 兼容 API 直接 400。这正是 CLAUDE.md 点名的「假绿」形态。
    calls_idx = [i for i, m in enumerate(model.bound_messages)
                 if getattr(m, "tool_calls", None)]
    tool_idx = [i for i, m in enumerate(model.bound_messages)
                if isinstance(m, ToolMessage)]
    assert calls_idx and tool_idx, "第二轮入参缺 assistant(tool_calls) 或 ToolMessage"
    assert tool_idx[0] == calls_idx[0] + 1   # 必须**紧邻**
    # trace 让验收 5「ReAct 不止一步」机械可断言
    assert out["trace"] == ["agent:step1 tool=query_order", "agent:converged"]


@pytest.mark.anyio
async def test_step_limit_converges_without_tools_bound():
    """步数用尽 → 最后一次调用**不绑 tools**,结构上保证收敛。"""
    # 脚本按消费顺序对齐:绑工具的 2 轮(== max_agent_steps)各弹一条,
    # 第 3 条留给收尾那一轮(未绑工具)。
    looping = [
        [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": str(i)}, "id": f"c{i}"}])]
        for i in range(2)
    ]
    model = ScriptedModel(looping + [[FakeChunk("收敛了")]])
    tool_obj = FakeTool()
    out = await _node(model, [tool_obj], {"query_order": tool_obj},
                      settings=_settings(max_agent_steps=2)).__call__(_state())
    assert out["agent_steps"] == 2
    assert model.bound_rounds == 2
    assert model.unbound_rounds == 1
    assert out["reply"] == "收敛了"


@pytest.mark.anyio
async def test_token_budget_stops_further_tool_rounds():
    """预算超限后不再进入下一轮工具调用。"""
    big = {"total_tokens": 99999}
    model = ScriptedModel([
        [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}], usage=big)],
        [FakeChunk("收敛了")],
    ])
    tool_obj = FakeTool()
    out = await _node(model, [tool_obj], {"query_order": tool_obj},
                      settings=_settings(agent_token_budget=100)).__call__(_state())
    assert out["usage"]["total_tokens"] >= 99999
    assert model.bound_rounds == 1      # 只走了一轮绑工具的
    assert out["reply"] == "收敛了"


@pytest.mark.anyio
async def test_unknown_tool_error_propagates_as_infrastructure_error():
    """未分类的异常由 executor 判成基础设施故障,**必须向上抛**(→502),
    绝不能被回灌给模型、伪装成「你的订单号查不到」。"""
    tool_obj = FakeTool(error=RuntimeError("boom"))
    model = ScriptedModel([
        [FakeChunk(tool_calls=[{"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}])],
    ])
    with pytest.raises(ToolInfrastructureError):
        await _node(model, [tool_obj], {"query_order": tool_obj}).__call__(_state())


@pytest.mark.anyio
async def test_history_and_evidence_reach_the_first_model_call():
    """历史经 `prompts.to_lc_messages` 转、证据并进本轮 human 消息。

    **断言的是发给模型的真实入参**,不是「函数没抛异常」——后者恒真。
    """
    model = ScriptedModel([[FakeChunk("好")]])
    await _node(model).__call__(
        _state(
            history=[Message(role="user", content="在吗")],
            evidence=[{"section_path": "退货政策", "answer": "七天无理由", "category": "退换货"}],
        )
    )
    sent = model.bound_messages
    assert sent[0].type == "system"            # 品牌 system prompt
    assert _text(sent[1]) == "在吗"             # 历史转过来了(只经 to_lc_messages)
    assert "七天无理由" in _text(sent[-1])      # 证据块
    assert "订单 1001 发货了吗" in _text(sent[-1])   # 用户原话
    assert sent[-1].type == "human"


@pytest.mark.anyio
async def test_no_evidence_means_no_evidence_block_in_the_prompt():
    """业务数据类没有证据块 —— 不能留一个空标题在那儿诱导模型。"""
    model = ScriptedModel([[FakeChunk("好")]])
    await _node(model).__call__(_state(evidence=[]))
    assert "以下是知识库中" not in _text(model.bound_messages[-1])
    assert _text(model.bound_messages[-1]) == "订单 1001 发货了吗"
```

- [ ] **Step 6: 跑测试确认失败**

```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_node.py
```

预期:`ImportError: cannot import name 'make_agent_node'`。

- [ ] **Step 7: 写 `render_evidence` 并给 `build_messages` 加 `evidence` 可选参**

在 `app/prompts.py` 追加(并用它组装,**不新增第二个转换点**):

```python
def render_evidence(evidence: list[dict]) -> str:
    """把检索证据渲染成一段文本,编号与 citations 的 n 对齐。"""
    lines = [
        f"[{i + 1}] ({e['section_path'] or e['category']}) {e['answer']}"
        for i, e in enumerate(evidence)
    ]
    return "以下是知识库中与该问题相关的资料,回答时请在对应信息处标注编号:\n\n" + "\n\n".join(lines)
```

**然后给已有的 `build_messages` 加一个可选参**(不要新建平行函数):

```python
def build_messages(
    *,
    brand_name: str,
    history: Sequence[Message],
    user_input: str,
    evidence: list[dict] | None = None,
) -> list:
    """组装本轮要发给模型的消息。

    `evidence` 只在**知识类意图且过了置信度闸**时有值:把它并进本轮 human
    消息,而不是插一条中段 system 消息 —— 中段 system 在多家兼容网关上的
    支持不如并进 human 稳。

    **不要另建一个 `build_agent_messages`**:本函数是 `prompts.py` 唯一的消息
    组装出口,多一个平行函数会立刻变成死代码(节点只用新的那个,这里的调用点
    在 ch05 被 `prepare_turn` 让出来),并连带 `tests/test_prompts.py` 的 4 条
    用例变成孤儿。加一个带默认值的参数则零破坏。
    """
    messages = [SystemMessage(render_system_prompt(brand_name))]
    messages.extend(to_lc_messages(history))
    text = user_input if not evidence else f"{render_evidence(evidence)}\n\n用户问题:{user_input}"
    messages.append(HumanMessage(text))
    return messages
```

> 改完请跑 `.venv/Scripts/python.exe -m pytest tests/test_prompts.py` 确认那 4 条老用例
> **一字不改仍然通过** —— 那正是「加了可选参数」而非「另起炉灶」的证据。
> (已核实确实会通过:`messages[0]` 仍是 `SystemMessage`、`messages[-1]` 仍是以
> `user_input` 为内容的 `HumanMessage`、空历史 2 条 / 两条历史 4 条 —— 重写前后
> 逐条对得上。)

**同时删掉因这次重写而变成死代码的两处**(评审一定会提,现在顺手做掉):

- `app/prompts.py:60` 的 `CHAT_PROMPT = ChatPromptTemplate.from_messages([...])`
  —— `build_messages` 是它**唯一**的使用者(`grep -rn "CHAT_PROMPT" app/ tests/
  scripts/ evals/` 只剩它自己那一行)。重写后它不再被引用。
  删它**不改变任何行为**:它的 system 段就是 `SYSTEM_PROMPT_TEMPLATE`,而
  `render_system_prompt()` 走的是同一个模板(`_SYSTEM_PROMPT`)——
  两条路径产出**同一个字符串**。
- `app/prompts.py` 顶部 `from langchain_core.prompts import ... MessagesPlaceholder`
  里,**只删 `MessagesPlaceholder`**(`CHAT_PROMPT` 的 `MessagesPlaceholder("history"
  , optional=True)` 是它唯一用处)。`ChatPromptTemplate` **要留着** ——
  `_SYSTEM_PROMPT` 与 `EXTRACT_PROMPT` 还在用。

改完**再跑一次** `tests/test_prompts.py` 与 `tests/test_chat_service.py`,确认仍然全绿。

- [ ] **Step 8: 写 Agent 节点实现**

在 `app/agent/nodes.py` 追加(顶部补 `from langchain_core.messages import ToolMessage`、`from app.prompts import build_messages`、`from app.tools.executor import execute_tool`):

```python
# ---- 主力 Agent:手写 ReAct ----


def _total_tokens(chunk) -> int:
    meta = getattr(chunk, "usage_metadata", None) or {}
    return int(meta.get("total_tokens") or 0)


def make_agent_node(*, model, tools, registry, settings, emit):
    """主力 Agent 的 ReAct 循环。

    **不用 ToolNode / create_react_agent**:工具执行必须走 `execute_tool`,
    它承载本项目的错误语义 —— 基础设施故障抛 `ToolInfrastructureError`(→502)、
    重试用白名单(`create_ticket` 永不重试)、`ValidationError`/`ToolNotFound`
    不重试、10s 超时。`ToolNode` 直接 `tool.ainvoke`,这些语义全丢。

    停止条件是**结构保证**:循环里绑着 tools 走 `max_agent_steps` 轮;步数用尽
    或预算超限后,收尾那一轮**不绑 tools**,模型在结构上无法再调。
    """
    bound = model.bind_tools(list(tools))

    async def _stream_round(target, msgs, parts) -> tuple[object, int]:
        acc = None
        used = 0
        async for chunk in target.astream(msgs):
            acc = chunk if acc is None else acc + chunk
            used += _total_tokens(chunk)
            if chunk.text:
                parts.append(chunk.text)
                emit({"frame": "token", "text": chunk.text})
        return acc, used

    async def agent_node(state) -> dict:
        msgs = build_messages(
            brand_name=settings.brand_name,
            history=state.get("history") or [],
            evidence=state.get("evidence") or [],
            user_input=state["resolved_input"],
        )
        parts: list[str] = []
        made: list[dict] = []
        trace: list[str] = []
        steps = 0
        usage_total = 0
        needs_final = False

        for step in range(1, settings.max_agent_steps + 1):
            steps = step
            acc, used = await _stream_round(bound, msgs, parts)
            usage_total += used
            tool_calls = list(getattr(acc, "tool_calls", None) or [])

            if not tool_calls:
                needs_final = False
                break

            needs_final = True
            msgs.append(acc)
            for call in tool_calls:
                emit({"frame": "tool_call", "name": call["name"],
                      "args": call["args"], "tool_call_id": call["id"]})
                outcome = await execute_tool(
                    tool_call=call, registry=registry, settings=settings
                )
                emit({"frame": "tool_result", "tool_call_id": outcome.tool_call_id,
                      "ok": outcome.ok, "summary": outcome.summary})
                msgs.append(ToolMessage(content=outcome.content, tool_call_id=call["id"]))
                made.append({"name": call["name"], "ok": outcome.ok})
                trace.append(f"agent:step{step} tool={call['name']}")

            if usage_total > settings.agent_token_budget:
                break

        if needs_final:
            # 收尾:不绑 tools。预算已超也照做一次 —— 它是**唯一**能产出
            # 用户可见答复的调用,不做的话这一轮就是「有工具调用、没有回答」。
            _, used = await _stream_round(model, msgs, parts)
            usage_total += used

        trace.append("agent:converged")
        return {
            "reply": "".join(parts),
            "agent_steps": steps,
            "tool_calls_made": made,
            "usage": {"total_tokens": usage_total},
            "trace": trace,
        }

    return agent_node
```

- [ ] **Step 9: 跑测试确认通过**

```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_node.py
```

预期:`7 passed`。

- [ ] **Step 10: 跑全量快路径**

```bash
.venv/Scripts/python.exe -m pytest -m "not db"
```

预期:全绿。

- [ ] **Step 11: 提交**

```bash
git add app/config.py app/prompts.py app/agent/nodes.py tests/test_agent_node.py tests/test_config.py
git commit -m "feat: ch05 主力 ReAct Agent 节点 + token 预算/步数停止条件 + 配置项"
```

---

## Task 7:日志节点 + 图组装

**Files:**
- Modify: `app/agent/nodes.py`(追加 `make_log_turn_node`、`make_resolve_references_node`)
- Create: `app/agent/graph.py`
- Modify: `requirements.txt`
- Create: `tests/test_agent_graph.py`

**Interfaces:**
- Produces: `make_log_turn_node(*, session, emit)`、`make_resolve_references_node()`、`get_checkpointer()`、`build_graph(*, model, intent_model, tools, registry, settings, retriever, session, conversation_id, emit, checkpointer)`

> 签名以 **Step 3 的代码块**为准。此前这行把 `make_log_turn_node` 写成 `(*, session)`,
> 漏了 `emit` —— 而节点体里要用它发 trace 帧。实现者按代码块写是对的。

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_agent_graph.py`:

```python
"""整图行为:五条出口各走一遍,断言 trace(验收 1/5 的可检查性来源)。"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.emit import make_emitter
from app.agent.graph import build_graph, get_checkpointer
from app.config import Settings
from app.retrieval.search import RetrievedChunk


class FakeChunk:
    def __init__(self, text="", tool_calls=None, usage=None):
        self.text = text
        self.tool_calls = [{"type": "tool_call", **tc} for tc in (tool_calls or [])]
        self.usage_metadata = usage

    def __add__(self, other):
        # 与 T6 的替身同一条理由:**累积时不能丢 tool_calls,更不能丢
        # `"type": "tool_call"` 键**(见 Ruling 21)。丢了前者,「模型在一轮里
        # 分多次吐完一次工具调用」在替身里就永远累积不起来;丢了后者,
        # `BaseTool.ainvoke` 会把 dict 当**参数**去校验 schema(CLAUDE.md 的硬约束)。
        # 本文件的用例目前每轮只有一个 chunk,所以这条分支**走不到** ——
        # 但替身必须忠实于真实 chunk 的形状,否则它会教后来的人写错的形状,
        # 而且错得**不会红**。真实的多 chunk 累积由 T11 的验收 5 兜底。
        return FakeChunk(
            self.text + other.text,
            tool_calls=self.tool_calls + other.tool_calls,
        )


class ScriptedModel:
    """每次 astream 都从 next(脚本)取一批。"""

    def __init__(self, rounds):
        self.rounds = list(rounds)

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, method=None):
        return self

    async def ainvoke(self, messages):
        batch = self.rounds.pop(0)
        return batch[0] if isinstance(batch, list) else batch

    async def astream(self, messages):
        batch = self.rounds.pop(0)
        for chunk in (batch if isinstance(batch, list) else [batch]):
            yield chunk


class FakeRetriever:
    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.calls = []

    async def search(self, query):
        self.calls.append(query)
        return list(self.chunks)


class RecordingSession:
    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass


def _settings(**over):
    return Settings(
        _env_file=None,
        openai_base_url="https://example.invalid/v1", openai_api_key="sk-test",
        openai_model="m", database_url="mysql+asyncmy://u:p@h:3306/db", **over,
    )


def _graph(intent, *, retriever=None, rounds=None, settings=None, frames=None):
    session = RecordingSession()
    # rounds is None 才用默认;显式传 [] 要保留成空脚本 ——
    # 那是「碰模型就炸」的探针,`rounds or [...]` 会把空列表换掉、探针失效。
    graph = build_graph(
        model=ScriptedModel(rounds if rounds is not None else [[FakeChunk("模型回复")]]),
        intent_model=ScriptedModel([[type("_I", (), {"intent": intent})()]]),
        tools=[], registry={}, settings=settings or _settings(),
        retriever=retriever or FakeRetriever(),
        session=session, conversation_id="conv-1",
        emit=(frames.append if frames is not None else (lambda p: None)),
        # **每个测试一个全新的 checkpointer**,不用 `get_checkpointer()` 那个
        # 进程级单例 —— 真机已验证:同一个 thread 上重复 ainvoke,带
        # `operator.add` 的 `trace` 会**跨调用累积**(实测 `['a','b','a','b']`)。
        # 而所有测试都用默认 thread "t",于是**测试之间互相污染**:
        # `test_business_route_skips_retrieval_and_gate` 断言的
        # `"confidence_gate:pass" not in out["trace"]` 会被前面那个知识类测试
        # 留下的同一句打红 —— 而它红得毫无道理,指向的是测试脚手架而非实现。
        # 单例本身由 `test_checkpointer_is_a_process_level_singleton` 单独覆盖,
        # 生产路径由 T8 的端点测试覆盖。
        checkpointer=InMemorySaver(),
    )
    return graph, session


async def _run(graph, user_input, thread="t"):
    # `agent_steps` 必须在入参里**显式给初值**:真机已验证,一个从未被写过的
    # 通道在返回的 state 里**根本不存在**(实测 `"agent_steps" in out` 为 False),
    # 于是 `assert out["agent_steps"] == 0` 抛的是 KeyError ——
    # 那个红指向「测试写错了」,而不是「闸没拦住」,把真问题盖掉。
    # 给了初值之后,这条断言才真的在问「Agent 到底有没有跑」。
    return await graph.ainvoke(
        {"conversation_id": "conv-1", "user_input": user_input, "history": [],
         "agent_steps": 0, "trace": []},
        config={"configurable": {"thread_id": thread}},
    )


@pytest.mark.anyio
async def test_knowledge_intent_forces_retrieval_before_agent():
    """验收 1:政策类问题,trace 里必须看到强制检索节点被走到。"""
    retriever = FakeRetriever([RetrievedChunk("怎么退货", "七天无理由", "退换货",
                                              chunk_id=1, section_path="退货政策", score=0.9)])
    graph, _ = _graph("商品咨询", retriever=retriever)
    out = await _run(graph, "怎么退货")
    assert retriever.calls == ["怎么退货"]
    assert "retrieve_knowledge:1 hits top=0.90" in out["trace"]
    assert "confidence_gate:pass" in out["trace"]


@pytest.mark.anyio
async def test_weak_evidence_skips_agent_and_records_low_confidence():
    retriever = FakeRetriever([RetrievedChunk("q", "a", "c", chunk_id=1,
                                              section_path=None, score=0.10)])
    frames = []
    graph, session = _graph("商品咨询", retriever=retriever, frames=frames)
    out = await _run(graph, "冷门问题")
    assert out["gate_passed"] is False
    assert out["agent_steps"] == 0          # **没进 Agent**
    assert session.added[0].entry_point == "置信度闸"


@pytest.mark.anyio
async def test_business_route_skips_retrieval_and_gate():
    """业务数据类不预检索、不过闸(没有检索证据,证据强弱无从谈起)。"""
    retriever = FakeRetriever()
    graph, _ = _graph("物流", retriever=retriever)
    out = await _run(graph, "订单 1001 的物流到哪了")
    assert retriever.calls == []
    # 精确表述:**闸一次都没跑**,而不是"闸跑了但没过"。
    # 旧的 `"confidence_gate:pass" not in out["trace"]` 会收下一个把物流也送进闸的
    # 图 —— 那种图留下的是 `confidence_gate:fail`,照样绿。
    assert not any(t.startswith("confidence_gate") for t in out["trace"])
    assert "agent:converged" in out["trace"]


@pytest.mark.anyio
async def test_chitchat_returns_fixed_copy_and_never_calls_the_model():
    graph, _ = _graph("闲聊", rounds=[])     # 脚本为空:碰模型就会 IndexError
    out = await _run(graph, "你好")
    assert "客服小猫" in out["reply"]
    assert "classify_intent:闲聊" in out["trace"]


@pytest.mark.anyio
async def test_complaint_emits_choices_and_returns_soothing_copy():
    frames = []
    graph, _ = _graph("投诉", rounds=[], frames=frames)
    out = await _run(graph, "我要投诉")
    assert out["choices"] == ["handoff", "ticket"]
    assert {"frame": "choices", "options": [
        {"key": "handoff", "label": "转人工"}, {"key": "ticket", "label": "建工单"}]} in frames


@pytest.mark.anyio
async def test_fallback_route_for_out_of_vocabulary_intent():
    graph, _ = _graph("其他", rounds=[])
    out = await _run(graph, "帮我写首诗")
    assert "没太理解" in out["reply"]
    assert "classify_intent:其他" in out["trace"]


@pytest.mark.anyio
async def test_successful_turn_is_persisted_to_mysql_history():
    graph, session = _graph("闲聊", rounds=[])
    out = await _run(graph, "你好")
    roles = [m.role for m in session.added]
    assert roles == ["user", "assistant"]
    # **断言内容,不只断言角色。** 只断角色时,把两条消息的**内容对调**
    # (把 user_input 存成 assistant)照样绿 —— 闲聊是固定话术、确定性,
    # 可以安全地断内容。
    assert session.added[0].content == "你好"
    assert session.added[1].content == out["reply"]


@pytest.mark.anyio
async def test_checkpointer_is_a_process_level_singleton():
    """每请求新建 checkpointer 的话,下一轮 thread 状态就没了。"""
    assert get_checkpointer() is get_checkpointer()


async def test_weak_evidence_skips_agent_and_records_low_confidence():
    """弱证据 → 不进 Agent、落低置信度池。"""
    # ...(既有断言略:不进 Agent、落库、trace 里没有 confidence_gate:pass)
    # ↓ **新补的这一行**是 Ruling 22 不变量**唯一**的守卫。
    # `build_graph` 把 `conversation_id` 传给 `make_confidence_gate_node`,
    # 而闸**从不读** `state["conversation_id"]` —— 同一个事实两个来源,
    # 测试里两者永远同值,所以不匹配**看不见**
    # (实测:把闭包值改成 `"WRONG-CONV"`,补这行之前 **10/10 全绿**)。
    # `RecordingSession.add` 顺序收集,`record_low_confidence` 是这一轮
    # 唯一写库的动作,所以 `added[0]` 就是那条低置信度记录。
    assert session.added[0].source_conversation_id == "conv-1"


@pytest.mark.anyio
async def test_second_turn_on_same_thread_reports_only_its_own_turn():
    """**跨轮证据链**:同一个 thread 连跑两轮,第二轮只许报第二轮。

    checkpointer 是**进程级单例**、`thread_id = session_id`、而 `trace` 是
    `operator.add` 累积通道、其余通道**未写就保留旧值**。三件事叠起来:
    第 N 轮的 trace 帧是「第 1..N 轮」的拼接,于是**验收 1 的
    「trace 里有 retrieve_knowledge」会在一条根本没检索的轮上通过**(上一轮
    留下的),逐轮的 `gate_passed` / `agent_steps` 同理。这是本章最高价值
    证据链上的假绿通道,所以两条修法**(逐轮重置 + trace 切片)各自都要有
    断言钉住**,不能只修一条。

    本文件其余 10 条用例每条都新建 `InMemorySaver`、只跑一轮 —— **没有一条
    能看见这个**。
    """
    retriever = FakeRetriever([RetrievedChunk("怎么退货", "七天无理由", "退换货",
                                              chunk_id=1, section_path="退货政策", score=0.9)])
    frames = []
    session = RecordingSession()
    graph = build_graph(
        model=ScriptedModel([[FakeChunk("模型回复")]]),
        # 意图替身按**调用顺序**回放(ainvoke 是 rounds.pop(0)):第 1 轮知识类,
        # 第 2 轮闲聊 —— 两轮走**不同分支**,残留才看得见。
        intent_model=ScriptedModel([
            [type("_I", (), {"intent": "商品咨询"})()],
            [type("_I", (), {"intent": "闲聊"})()],
        ]),
        tools=[], registry={}, settings=_settings(),
        retriever=retriever, session=session, conversation_id="conv-1",
        emit=frames.append,
        # 两轮**共用**同一个 checkpointer,且 thread_id 相同 —— 这正是要测的场景。
        checkpointer=InMemorySaver(),
    )

    first = await _run(graph, "怎么退货")
    one_turn = len(frames)
    # 第二轮的入参**刻意不给 `agent_steps` 初值**(第一轮仍走 `_run`)。
    # 原因**实测**得到,不是推理:入参里的值会**覆写**非归约通道,所以
    # `agent_steps: 0` 一给,就等于替实现把上一轮的残留抹掉了 ——
    # 变异实测(只重置 `gate_passed`、`agent_steps` 不清)下:
    #   第二轮入参**给** `agent_steps: 0` → 帧里报 0 → 断言**假绿**;
    #   第二轮入参**不给**         → 帧里报 1(上一轮的值)→ 断言红。
    # 而真机端点(T8)的入参里**没有** `agent_steps`(见计划 T8 Step 5),
    # 残留会一路进 `done` 帧 —— 所以这里必须真的能红。
    second = await graph.ainvoke(
        {"conversation_id": "conv-1", "user_input": "你好", "history": [], "trace": []},
        config={"configurable": {"thread_id": "t"}},
    )
    second_frames = frames[one_turn:]

    # 前置条件:第 1 轮**确实**检索了、过了闸、进了 Agent —— 这样第二轮的
    # 「没有」才是在说「被清掉了」,而不是「第 1 轮本来就没有」。
    assert retriever.calls == ["怎么退货"]
    assert "retrieve_knowledge:1 hits top=0.90" in first["trace"]
    assert "confidence_gate:pass" in first["trace"]
    assert "agent:converged" in first["trace"]

    # 前提:累积通道**确实**累积了 —— 这正是本用例存在的理由。
    # 没有这一条,下面那些断言可能只是因为"根本没累积"而通过。
    assert len(second["trace"]) > len(first["trace"])

    payload = second_frames[-1]                    # 第二轮的 trace 帧
    assert payload["frame"] == "trace"
    assert payload["trace"][0] == "resolve_references"
    assert not any(t.startswith("confidence_gate") for t in payload["trace"])
    assert "retrieve_knowledge" not in " ".join(payload["trace"])
    assert payload["gate_passed"] is None          # 第 1 轮过闸了;第 2 轮没进闸
    assert payload["agent_steps"] == 0             # 第 1 轮进过 Agent;第 2 轮没有


@pytest.mark.anyio
async def test_resolve_references_resets_every_per_turn_channel():
    """每轮开头必须清掉上一轮的**全部**逐轮通道。

    为什么值得单钉:**删除其中一个键不会有任何测试变红** —— 而后果按通道
    不同而不等。`evidence` 尤其真实:它被 agent 节点读进 `build_messages`
    (`app/agent/nodes.py:234` 附近),业务轮跟在知识轮后会把**上一轮的检索
    结果**当本轮知识塞进 prompt —— 用户看到的是上一轮的知识,且完全静默。

    两轮整图用例(闲聊轮)盖不住它:闲聊不读 `evidence`。
    """
    node = make_resolve_references_node()
    out = await node({
        "user_input": "在吗",
        "evidence": [{"answer": "上一轮的旧知识"}],
        "gate_passed": True,
        "agent_steps": 3,
        "reply": "上一轮的回复",
        "choices": ["handoff"],
        "citations": [{"n": 1}],
        "tool_calls_made": [{"name": "query_order"}],
    })

    assert out["resolved_input"] == "在吗"
    assert out["trace"] == ["resolve_references"]
    assert out["evidence"] == []
    assert out["gate_passed"] is None
    assert out["agent_steps"] == 0
    assert out["reply"] == ""
    assert out["choices"] == []
    assert out["citations"] == []
    assert out["tool_calls_made"] == []
```

> **变异验证(7 个键全扫,已实测)**:逐个单删重置字典里的键 → 跑 → 还原。
> `evidence` / `citations` / `reply` / `choices` / `tool_calls_made` 各**只**打红本用例;
> `gate_passed` / `agent_steps` 打红本用例 **+** 两轮整图用例(两条独立用例都钉,预期)。
> **没有任何键是「删掉也不红」的。**
>
> 分工:`gate_passed`/`agent_steps` 由两轮用例覆盖(它们跨轮可见);
> `evidence` **只有本用例**是防线 —— 删它时两轮用例**全绿**(第二轮走闲聊、不读它)。

```python


@pytest.mark.anyio
async def test_log_turn_emits_trace_frame_for_end_to_end_acceptance():
    """`trace` 要能被验收脚本从 SSE 里读到(验收 1/5 靠它机械可断言)。

    `log_turn` 用同一个 emit 通道发一帧 `trace`;端点把它折进 `done` 帧、
    不外推给前端。走同一条发出路径 = 不需要第二个 stream_mode。
    """
    frames = []
    graph, _ = _graph("闲聊", rounds=[], frames=frames)
    await _run(graph, "你好")
    kinds = [p["frame"] for p in frames]
    assert "trace" in kinds
    trace = next(p for p in frames if p["frame"] == "trace")["trace"]
    assert "resolve_references" in trace
    assert "classify_intent:闲聊" in trace
    assert "chitchat_reply" in trace
    assert trace[-1] == "log_turn"


@pytest.mark.anyio
async def test_emitter_sends_frames_through_astream_custom_mode():
    """**这条最关键**:证明 emit 在真实的图运行中把帧送到了 astream 消费端。

    缺了它,把 `make_emitter()` 写成「一次性 `get_stream_writer()`」的错误版本
    也能全绿 —— 而那种写法下前端**一帧都收不到**(图外必然抛 RuntimeError,
    静默退化成 no-op)。这正是本项目最怕的「假绿 + 静默故障」。
    """
    collected = []                       # 图外兜底:真跑到这里就说明没拿到 writer
    emit = make_emitter(collected.append)
    session = RecordingSession()
    graph = build_graph(
        model=ScriptedModel([]),
        intent_model=ScriptedModel([[type("_I", (), {"intent": "投诉"})()]]),
        tools=[], registry={}, settings=_settings(), retriever=FakeRetriever(),
        session=session, conversation_id="conv-emit", emit=emit,
        checkpointer=get_checkpointer(),
    )

    got = []
    async for payload in graph.astream(
        {"conversation_id": "conv-emit", "user_input": "我要投诉", "history": [], "trace": []},
        config={"configurable": {"thread_id": "emit-test"}},
        stream_mode="custom",
    ):
        got.append(payload)

    assert {"frame": "choices", "options": [
        {"key": "handoff", "label": "转人工"}, {"key": "ticket", "label": "建工单"}]} in got
    assert collected == []               # 图内拿到了真 writer,没退化成 collector
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_graph.py
```

预期:`ModuleNotFoundError: No module named 'app.agent.graph'`。

- [ ] **Step 3: 写两个小节点**

在 `app/agent/nodes.py` 追加(顶部补 `from app.schemas import Message`、`from app.services.history import append_turn`):

```python
def make_resolve_references_node():
    """指代消解:**本章原样透传**,正式版留给下一步(用户点名)。

    节点本身先立在这里,是为了把「骨架的第一步」这个位置固定下来 ——
    正式版换实现时,图的拓扑一行都不用动。

    它同时承担**每轮重置**:图是每请求现编译的,但 checkpointer 是**进程级**
    单例(`get_checkpointer` 的 lru_cache),而 thread_id = session_id ——
    所以同一会话的**第二轮**会带着上一轮的通道值进来。未写的通道**保留旧值**
    (LangGraph 不把未写通道重新写成默认值)。不清零,trace 帧与日志行就会把
    上一轮的 `gate_passed` / `agent_steps` 报成本轮的,而且**一路静默**:
    物流轮的 `gate_passed` 会是上一轮知识检索的结论。

    为什么放在这个节点:它是**每轮第一个**执行节点(START 的唯一出边),
    放这儿等于「每轮开头清一次」,不依赖任何调用方记得播种初值。

    `trace` 通道不在此列 —— 它是 `operator.add` 归约通道,写 `[]` 等于没写,
    清不掉;**它靠 `log_turn` 切片取当轮**(见下)。
    """

    async def resolve_references(state) -> dict:
        return {
            "resolved_input": state["user_input"],
            "trace": ["resolve_references"],
            # 每轮归零的**逐轮**通道:它们描述的是「这一轮」,不是「这段会话」。
            "gate_passed": None,
            "agent_steps": 0,
            "reply": "",
            "choices": [],
            "citations": [],
            "evidence": [],
            "tool_calls_made": [],
        }

    return resolve_references


def make_log_turn_node(*, session, emit):
    """日志记录:落一行结构化日志、把 trace 发成帧、把这一轮写回 MySQL。

    `trace` 是本轮**唯一**的确定性证据链:验收 1「走了强制检索节点」与
    验收 5「ReAct 不止一步」都靠它断言,而不是靠模型自由文本。

    它同时以 `trace` 帧发给端点(端点折进 `done`、不外推给前端)—— 走的是
    和 token 帧同一条 emit 通道,所以**不需要第二个 stream_mode**。
    """

    async def log_turn(state) -> dict:
        # **只取当轮**。`trace` 是 operator.add 通道,同一 thread 的第二轮
        # 拿到的是「第一轮 + 第二轮」的拼接 —— 直接发出去,验收 1 的
        # 「trace 里有 retrieve_knowledge」会在一条**根本没检索**的物流轮上
        # 通过(上一轮留下的),这是本章最高价值证据链上的假绿通道。
        #
        # `resolve_references` 是 START 的唯一出边、每轮第一个执行,
        # 所以**最后一次**出现它就是当轮起点。找不到时(直接调 log_turn、
        # 或将来拓扑变了)退化为整段,不抛异常。
        accumulated = list(state.get("trace") or [])
        turn_trace = accumulated
        if "resolve_references" in accumulated:
            start = len(accumulated) - 1 - accumulated[::-1].index("resolve_references")
            turn_trace = accumulated[start:]
        full_trace = [*turn_trace, "log_turn"]
        logger.info(
            "chat_turn conv=%s intent=%s gate=%s steps=%s tools=%s trace=%s",
            state.get("conversation_id"), state.get("intent"), state.get("gate_passed"),
            state.get("agent_steps"),
            [t["name"] for t in (state.get("tool_calls_made") or [])],
            " > ".join(full_trace),
        )
        emit({"frame": "trace", "trace": full_trace,
              "intent": state.get("intent"), "gate_passed": state.get("gate_passed"),
              "agent_steps": state.get("agent_steps") or 0})
        await append_turn(
            session=session,
            conversation_id=state["conversation_id"],
            messages=[
                Message(role="user", content=state["user_input"]),
                Message(role="assistant", content=state.get("reply") or ""),
            ],
        )
        return {"trace": ["log_turn"]}

    return log_turn
```

- [ ] **Step 4: 写图组装**

创建 `app/agent/graph.py`:

```python
"""ch05 图组装。

**两个粒度必须分清**:
- **图按请求组装** —— `query_faq` / `create_ticket` 是每请求闭包(见
  `app/tools/business.py` 的说明),图必须绑到本请求的那批工具上。
  `StateGraph` 构造是纯内存操作,不进请求路径。
- **checkpointer 是进程级单例** —— 每请求新建的话,thread 状态下一轮就没了。
"""

from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    make_agent_node,
    make_chitchat_reply_node,
    make_classify_intent_node,
    make_complaint_reply_node,
    make_confidence_gate_node,
    make_fallback_reply_node,
    make_log_turn_node,
    make_resolve_references_node,
    make_retrieve_knowledge_node,
)
from app.agent.routing import (
    BUSINESS,
    CHITCHAT,
    COMPLAINT,
    FALLBACK,
    KNOWLEDGE,
    route_by_intent,
)
from app.agent.state import ChatState

#: 出口节点 —— 它们统一汇进 log_turn 再结束。
_OUTLETS = ("agent", "complaint_reply", "chitchat_reply", "fallback_reply")


@lru_cache(maxsize=1)
def get_checkpointer() -> InMemorySaver:
    """**进程级单例**。按请求新建 = 每轮都是新 thread,状态全丢。

    **无淘汰、无 TTL**:见过的每个 thread 状态常驻进程(含累积的 `trace`)。
    本章可接受(演示规模);正式版要换成有界的持久化 checkpointer。

    另:`InMemorySaver` 是纯进程内存储,**不含** `asyncio.Lock` 之类
    loop-bound 原语(已实测 grep 无命中)—— 所以它**不适用** ch04 那类
    "lru_cache 单例绑在首个事件循环上"的故障模式。要在测试里换掉它,
    理由是**状态跨测试累积**,不是跨循环。
    """
    return InMemorySaver()


def _gate_route(state) -> str:
    return "agent" if state.get("gate_passed") else "fallback_reply"


def build_graph(
    *,
    model,
    intent_model,
    tools,
    registry,
    settings,
    retriever,
    session,
    conversation_id,
    emit,
    checkpointer,
):
    """组装本请求的图并编译。"""
    graph = StateGraph(ChatState)

    graph.add_node("resolve_references", make_resolve_references_node())
    graph.add_node("classify_intent", make_classify_intent_node(model=intent_model))
    graph.add_node(
        "retrieve_knowledge",
        make_retrieve_knowledge_node(retriever=retriever, emit=emit),
    )
    # ⚠️ 这个 `conversation_id` 必须与 state 里的 `conversation_id` 是**同一个值**。
    # 闸用它写 `low_confidence_questions.source_conversation_id`(即「这问题是从哪段
    # 对话里冒出来的」),而 state 里那个是 `thread_id`、是落库 `append_turn` 的依据。
    # 两者不一致时,问题会被记到**别的会话**名下 —— 而**没有任何测试看得见**:
    # 测试全程给闭包和 state 传同一个字面量,读的也是闭包传的那个。
    # 传入方只有一个(T8 的端点,`conversation_id=session_id`,state 里也是
    # `session_id`),所以今天是一致的;这行注释是给**以后**加调用点的人。
    graph.add_node(
        "confidence_gate",
        make_confidence_gate_node(
            settings=settings, session=session, conversation_id=conversation_id
        ),
    )
    graph.add_node(
        "agent",
        make_agent_node(
            model=model, tools=tools, registry=registry, settings=settings, emit=emit
        ),
    )
    graph.add_node("complaint_reply", make_complaint_reply_node(emit=emit))
    graph.add_node("chitchat_reply", make_chitchat_reply_node(emit=emit))
    graph.add_node("fallback_reply", make_fallback_reply_node(emit=emit))
    graph.add_node("log_turn", make_log_turn_node(session=session, emit=emit))

    graph.add_edge(START, "resolve_references")
    graph.add_edge("resolve_references", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            KNOWLEDGE: "retrieve_knowledge",
            BUSINESS: "agent",
            COMPLAINT: "complaint_reply",
            CHITCHAT: "chitchat_reply",
            FALLBACK: "fallback_reply",
        },
    )
    graph.add_edge("retrieve_knowledge", "confidence_gate")
    graph.add_conditional_edges(
        "confidence_gate", _gate_route, {"agent": "agent", "fallback_reply": "fallback_reply"}
    )
    for outlet in _OUTLETS:
        graph.add_edge(outlet, "log_turn")
    graph.add_edge("log_turn", END)

    return graph.compile(checkpointer=checkpointer)
```

- [ ] **Step 5: 补 requirements**

`requirements.txt` 追加(已装但未登记):

```
langgraph==1.2.11
```

- [ ] **Step 6: 跑测试确认通过**

```bash
.venv/Scripts/python.exe -m pytest tests/test_agent_graph.py
```

预期:`10 passed`。

- [ ] **Step 7: 跑全量快路径**

```bash
.venv/Scripts/python.exe -m pytest -m "not db"
```

预期:全绿。

- [ ] **Step 8: 提交**

```bash
git add app/agent/nodes.py app/agent/graph.py requirements.txt tests/test_agent_graph.py
git commit -m "feat: ch05 日志节点 + 图组装(checkpointer 单例、图按请求组装)"
```

---

## Task 8:接入 SSE 端点 + 删掉手写循环

**Files:**
- Modify: `app/services/chat.py`(`prepare_turn` 改返回裁剪后历史;删 `stream_turn`)
- Modify: `app/api/chat.py`
- Modify: `tests/test_chat_service.py`(改 `prepare_turn` 的断言;删 `stream_turn` 的用例)
- Delete: `app/agent/loop.py`、`tests/test_agent_loop.py`
- Modify: `tests/test_api_chat.py`

**Interfaces:**
- Consumes: `build_graph`、`get_checkpointer`(任务 7)
- Produces: `prepare_turn(*, settings, history, user_input) -> list[Message]`(**改为返回裁剪后历史**)

> ⚠️ **这是 spec §4.3 的落地**:`prepare_turn` 保留(预算校验仍在流开始前),
> 但它的产出从「组装好的消息」变成「裁剪后的历史」,因为消息组装现在由节点做
> (要插证据块)。实现完请在 spec §12 记一条订正。

- [ ] **Step 1: 改测试**

`tests/test_chat_service.py`:删掉所有 `stream_turn` 的用例与 `ScriptedModel`/`FakeChunk`
里只服务于它的部分;把 `prepare_turn` 的断言改成「返回裁剪后的历史」。

```python
@pytest.mark.anyio
async def test_prepare_turn_returns_trimmed_history():
    """预算校验仍在流开始前完成;产出是**裁剪后的历史**,消息组装交给节点。"""
    history = [Message(role="user", content="在吗"),
               Message(role="assistant", content="在的")]
    kept = prepare_turn(settings=_settings(), history=history, user_input="你好")
    assert [m.content for m in kept] == ["在吗", "在的"]


def test_prepare_turn_raises_when_budget_is_exhausted():
    """预算不足仍然抛 ContextOverflowError —— 端点靠它在流开始前返回 400。"""
    huge = "字" * 100_000
    with pytest.raises(ContextOverflowError):
        prepare_turn(settings=_settings(), history=[], user_input=huge)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python.exe -m pytest tests/test_chat_service.py
```

预期:**只有第一条失败**(返回值仍是组装好的消息列表,`[m.content for m in kept]`
拿到的是 system/human 两条而不是原历史)。**第二条本来就通过** —— 预算校验
这一章不变,它是用来钉「别把它改坏」的回归,红不了才对。若第二条也红了,
说明改动越界了。

- [ ] **Step 3: 改 `services/chat.py`**

把 `prepare_turn` 改成返回 `kept`,并**删除 `stream_turn` 整个函数**(连同它顶部
只为它服务的 import:`AIMessage`/`ToolMessage`/`assess_sufficiency`/`execute_tool` 等)。

```python
def prepare_turn(*, settings: Settings, history: Sequence[Message], user_input: str) -> list[Message]:
    """预算校验 + 历史裁剪。返回**裁剪后的历史**(消息组装由 ch05 的节点做)。

    历史由调用方从 MySQL 读出后传入 —— 本函数不做 IO,便于单测。

    预算不足时抛 ContextOverflowError,调用方在响应开始前处理,因此能返回
    400 而不是一个已经开始的 SSE 流。这条约束 ch05 不变:图开始跑之前必须
    已经知道预算够不够。
    """
    system_prompt = render_system_prompt(settings.brand_name)
    available = trim.compute_available_tokens(
        system_prompt=system_prompt,
        user_input=user_input,
        context_budget_tokens=settings.context_budget_tokens,
        reserved_output_tokens=settings.reserved_output_tokens,
        safety_margin_tokens=settings.safety_margin_tokens,
    )
    if available < 0:
        budget = (
            settings.context_budget_tokens
            - settings.reserved_output_tokens
            - settings.safety_margin_tokens
        )
        raise trim.ContextOverflowError(
            used=trim.count_tokens(system_prompt) + trim.count_tokens(user_input),
            budget=budget,
        )
    return trim.select_history(history, available)
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/Scripts/python.exe -m pytest tests/test_chat_service.py
```

预期:全绿。

- [ ] **Step 5: 改端点**

`app/api/chat.py`:

- 顶部 import 调整:删 `stream_turn`;加
  `from app.agent.emit import make_emitter`、
  `from app.agent.graph import build_graph, get_checkpointer`、
  `from app.llm import create_extract_model`。
- **加一个 `get_intent_model` 依赖,与既有的 `get_chat_model` 并排**:

```python
def get_intent_model(settings: Settings = Depends(get_settings)):
    """意图识别用的模型。与 `get_chat_model` **并排**做成依赖,不是为了让生产
    代码多一层 —— 是为了给测试留一个 `dependency_overrides` 的注入点。

    意图节点**每一个请求都会跑**,也就每一个请求都会真的 `.ainvoke` 一次。
    若这里就地 `create_extract_model(settings)`(模块级导入的名字),测试想拦
    它就只能去 monkeypatch `chat_api.create_extract_model` —— 而那是**导入进来的
    名字**,patch 它等于 patch 整个模块的绑定,别的端点(抽取)同用一个名时会
    连带被换掉。`Depends` 是本项目既有的注入缝(CLAUDE.md:services 收 llm 实例
    作参数、FastAPI 侧靠 Depends 注入、测试用 dependency_overrides 替换),
    `tests/test_api_chat.py` 的 `client_factory` 里已经有一行
    `app.dependency_overrides[chat_api.get_chat_model] = lambda: model`,
    加一行同形的即可。**不做这层,端点测试里 25 条都会朝
    `https://example.invalid/v1` 发真实请求,而且不联网这一条是硬规矩。**
    """
    return create_extract_model(settings)
```

  端点签名相应加 `intent_model=Depends(get_intent_model),`。
- `prepare_turn(...)` 的返回值语义变了 —— 现在是**裁剪后的历史**,直接喂给图的
  `history` 字段:`history = prepare_turn(settings=settings, history=history, user_input=request.message)`。
- 把 `stream_turn(...)` 那段 `async for (event, payload) in ...` 换成:

```python
    async def generate():
        try:
            yield _frame("meta", {"session_id": session_id, "model": settings.openai_model})

            # emit 必须在图**之外**创建、在节点里才被调用 —— make_emitter 返回的
            # 是可重复调用的函数,每次发帧时现取 writer(见 app/agent/emit.py)。
            emit = make_emitter()
            graph = build_graph(
                model=model,
                intent_model=intent_model,
                tools=tools,
                registry=registry,
                settings=settings,
                retriever=build_retriever(session),
                session=session,
                conversation_id=session_id,
                emit=emit,
                checkpointer=get_checkpointer(),
            )

            final = {"trace": [], "intent": None, "gate_passed": None, "agent_steps": 0}
            async for payload in graph.astream(
                {
                    "conversation_id": session_id,
                    "user_input": request.message,
                    "history": history,        # prepare_turn 返回的**裁剪后**历史
                    "trace": [],
                },
                config={"configurable": {"thread_id": session_id}},
                stream_mode="custom",
            ):
                event = payload.get("frame")
                if event == "trace":
                    # 内部证据链:折进 done 帧,不外推 —— 前端不认识这个帧。
                    final = payload
                    continue
                data = {k: v for k, v in payload.items() if k != "frame"}
                if event == "tool_result" and not data.get("ok", True):
                    data["summary"] = redact_api_key(
                        data.get("summary", ""), settings.openai_api_key
                    )
                yield _frame(event, data)

            yield _frame("done", {
                "finish_reason": "stop",
                "usage": None,
                "trace": final.get("trace") or [],
                "intent": final.get("intent"),
                "gate_passed": final.get("gate_passed"),
                "agent_steps": final.get("agent_steps") or 0,
            })
        except Exception as exc:
            yield _frame("error", {"message": redact_api_key(str(exc), settings.openai_api_key)})
        finally:
            lock.release()
```

> **为什么 emit 不能在图里建**:`make_emitter()` 在图外调时 `get_stream_writer()`
> 必然抛 `RuntimeError` → 退化成 no-op → **前端一帧都收不到**,而任何只断言
> `done` 帧的测试照样全绿。任务 7 的
> `test_emitter_sends_frames_through_astream_custom_mode` 就是钉这个的。
>
> **`trace` 帧为什么从这里折走**:它是给验收脚本看的,不是给前端看的。
> 它和 token 帧走**同一条** emit 通道,所以不需要 `stream_mode=["custom","values"]`
> ——少一个流模式就少一处「帧会不会被缓冲住」的风险。

- [ ] **Step 6: 改 `tests/test_api_chat.py`(25 条端点用例的接缝;**数字用 `pytest --collect-only` 现数,不要凭印象写**)

这批用例是**对着 ch02 那条直线流**写的:发消息 → 模型 → 工具 → 模型。ch05 在
模型**前面**插了两个节点(指代消解、意图识别),路由又是**按意图分叉**的。
不改这批文件,`tests/test_api_chat.py` 会**整片红**,而且红法与实现无关 ——
所以这一步是**必须的**,不是"顺手清理"。

改动只有两处:

1. `client_factory.make(...)` 里加第二行 override(与既有那行同形):

```python
        app.dependency_overrides[chat_api.get_intent_model] = lambda: intent_model
```

   并让 `make(...)` 收一个新参数 `intent: str = "订单"`,默认值就是它 ——
   替身用 `tests/test_agent_intent.py` 里那个 `FakeStructuredModel` 的形状
   (`with_structured_output` 断言 `method == "json_mode"` 后返回自身、
   `ainvoke` 回放一个带 `.intent` 的对象)。

2. **默认值必须是 `BUSINESS` 那一类(物流/订单/售后),不能是 `其他`/`闲聊`。**
   这批用例把 `ScriptedModel(batches)` 的批次当作"模型一定会被调用"来写;
   意图若是 `其他`/`闲聊`/`投诉`,路由**根本不进 Agent 节点**,那些批次一个都
   消费不到 —— 结果是一堆 `pop from empty list` 之类的**看不出因果的红**。
   默认值定在业务数据类,现有批次脚本就仍由 Agent 节点照常消费。

   需要单独覆盖路由行为的用例(闲聊话术、投诉按钮)显式传 `intent="闲聊"` /
   `intent="投诉"`。

> **别为 `done` 帧的 `usage` 纠结。** Step 5 里它被写成 `None`(ch02 曾报真实
> usage),全仓**没有任何消费方**:`app/static/`、`scripts/`、`evals/` 与
> `tests/test_api_chat.py` 里 grep 不到一处读 `done.usage` 的代码。这是记在案的
> 行为变更,不是遗漏;图里没有一处汇总 token,硬凑一个假数只会更糟。

> **删掉的 ch02 守卫不丢覆盖 —— 逐条填表,不靠"应该有"。** 本步会删掉
> `tests/test_chat_service.py` 里 `stream_turn` 的 **15 条**用例。删测试**永远全绿**,
> 这类损失没有任何自动化信号,所以**逐条**给出新家,填不进表的一律补:
>
> | 被删的守卫 | 新家 | 结论 |
> |---|---|---|
> | `:319` 同轮两个 tool_call 都执行都配对 | `tests/test_agent_node.py:247` | 已在(更严:帧序 + 紧邻 + 步号) |
> | `:449` 回灌全文不是截断摘要 | `tests/test_agent_node.py:311` | 已在(断言 `> SUMMARY_MAX_CHARS`) |
> | `:419` assistant(tool_calls) 紧邻其 tool 消息 | `tests/test_agent_node.py:135` + `:247` | 已在(出现两次) |
> | `:244` 基础设施故障向上抛 | `tests/test_agent_node.py:205` | 已在 |
> | `:142` 第二轮不绑 tools | —— | **按设计不存在**:T6 的节点每轮都绑 tools、靠步数/预算收敛(已验收) |
> | `:508/:527/:551` done 帧的 usage | —— | **本步明写丢弃**(全仓无消费方),记账即可 |
> | `:492` 调工具那一轮**不发 token 帧** | **无** | ⚠️ **本步补** |
> | `:278` 第二轮炸了 → 整轮不落库 | **无** | ⚠️ **本步补** |
>
> 两个缺口**就地补上**,各配一个能打红的变异:
>
> - `nodes.py` 的 `if chunk.text:` 是"不发空 token 帧"的**唯一**屏障,而
>   `test_tool_call_round_emits_frames_and_feeds_result_back` 用的是 `in frames`
>   **不是精确列表** —— 删掉 `if chunk.text:`,空串 token 帧照样全绿。
>   补:`[p["text"] for p in frames if p["frame"] == "token"] == ["已发货。"]`;
>   变异 = 删 `if chunk.text:`(只留 `parts.append`/`emit`)→ 该条红。
> - 「失败不落库」在新架构里**结构上成立**(`log_turn` 是唯一写方且在下游,
>   agent 节点抛异常则图走不到它),但**零断言**:端点侧
>   `test_upstream_error_becomes_sse_error_event` 只断言 error 帧,不断言
>   `session.added == []`。补一条断言(在该用例里加即可);变异 = 把 `log_turn`
>   挪到 agent 之前(或在异常路径上也调 `append_turn`)→ 该条红。
> - 若填表时还发现第三个没家的,**报上来,不要静默丢弃**。
>
> 表与两个补口写进 T8 的报告。

- [ ] **Step 7: 删掉手写循环**

```bash
git rm app/agent/loop.py tests/test_agent_loop.py
```

- [ ] **Step 8: 跑全量快路径 + db**

```bash
.venv/Scripts/python.exe -m pytest -m "not db"
.venv/Scripts/python.exe -m pytest
```

预期:全绿。**db 测试需要 MySQL 起着**;没起就用 `docker start mysql`。

- [ ] **Step 9: 记账**

在 spec §12 追加一条订正,并在 `dev-notes/ch05.md` 记本任务。

- [ ] **Step 10: 提交**

```bash
git add -A
git commit -m "feat: ch05 接入图骨架到 SSE 端点;删掉祛魅用的手写循环"
```

---

## Task 9:`POST /api/ticket`(建工单按钮的后端)

**Files:**
- Modify: `app/schemas.py`(追加 `TicketRequest`)
- Modify: `app/api/chat.py`(追加端点)
- Create: `tests/test_api_ticket.py`

**Interfaces:**
- Produces: `POST /api/ticket`,请求 `{"session_id": str}`,响应 `{"ticket_no": str, "status": "open"}`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_api_ticket.py`:

```python
"""建工单端点:按钮点击是 HTTP 请求,够不到模型工具 —— 故需要这个入口。

**这里刻意不用真库**(否则是本仓第一个「TestClient + 真实 engine」的组合):
`TestClient` 在它自己的 portal 事件循环里跑请求,会把 `get_engine()` 那个
**lru_cache 单例**绑到那个循环上;退出 `with` 后 portal 循环关闭,同进程里
后面所有走 `get_sessionmaker()` 的 db 测试(文件名排在 `test_api_ticket` 之后)
都会拿到跨循环的连接 —— 这是 ch04 记过账的故障形态。端点级测试在本仓
**一律替换 `get_session`**(`tests/test_api_chat.py` 的 `client_factory` 即此),
这里沿用同一条缝。

真库的「写进去了吗」由 `tests/test_tools_db.py::test_create_ticket_writes_row`
独占(它用新 session 回查 `select(Ticket)`)—— 那条钉的是工具,
这条钉的是**端点真的把工具调起来了**。
"""

from fastapi.testclient import TestClient

from app.db.models import Ticket
from app.db.session import get_session
from app.main import app

# 顶层 import(`tests/` 没有 `__init__.py`,pytest 默认的 prepend 导入模式会把
# `tests/` 放进 sys.path)—— **不要**写成 `tests.test_api_chat`:那会让同一份
# 替身以两个不同的模块名被加载两遍,`FakeSession` 变成两个类,
# `isinstance` 判断会莫名其妙地为假。已实测 `import test_api_chat` 可用。
from test_api_chat import FakeSession

SID = "00000000000000000000000000000001"


class _TicketSession(FakeSession):
    """在既有端点替身上补 `Ticket` 支持 —— 建工单端点写的正是它。

    继承而不是另写一份:`execute`/`commit`/`Conversation` 的行为必须与
    对话端点测试**同一套**,否则两个端点的替身会各自漂移。
    """

    def __init__(self):
        super().__init__()
        self.tickets: list[Ticket] = []

    def add(self, obj):
        if isinstance(obj, Ticket):
            self.tickets.append(obj)
        else:
            super().add(obj)


def _client(session):
    async def _override():
        yield session

    app.dependency_overrides[get_session] = _override
    return TestClient(app)


def test_build_ticket_invokes_the_tool_and_returns_its_payload():
    """按钮 → HTTP → **真的**调到 `create_ticket` 并把它的回参原样返回。

    **必须断言落库动作,不能只断言响应体**:把 `execute_tool(...)` 那段换成
    `return {"ticket_no": "T-假", "status": "open"}`,响应断言**全绿** —— 一个
    假 `ticket_no` 与真的一模一样(都是 `T-` 开头)。`session.tickets` 是唯一
    能区分「调了工具」和「编了一个」的东西。
    """
    session = _TicketSession()
    client = _client(session)
    try:
        resp = client.post("/api/ticket", json={"session_id": SID})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["ticket_no"].startswith("T-")
    assert body["status"] == "open"
    # ↓ 判别性断言:工具的写动作真的发生了,且号与响应里的一模一样
    assert [t.ticket_no for t in session.tickets] == [body["ticket_no"]]
    assert session.tickets[0].conversation_id == SID
    assert session.commits >= 1


def test_session_id_length_is_bounded_like_chat_request():
    """上限必须与 conversations.id 的 varchar(32) 对齐 —— 否则 DataError 会被判成 502。

    **同样要替换 `get_session`**,不要图省事写裸 `TestClient(app)`。FastAPI 在
    **422 之前就会进入 yield 依赖** —— 已实测:body 校验失败时依赖的 `enter`/`exit`
    都跑了。所以裸 `TestClient` 照样会把 `get_engine()` 的 lru_cache 单例建在
    portal 循环上,正是文件头警告的那个组合(第一版计划这里就是裸的,已订正)。
    """
    session = _TicketSession()
    client = _client(session)
    try:
        resp = client.post("/api/ticket", json={"session_id": "x" * 33})
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 422
    # 请求**根本没进端点**(校验先于函数体):长 session_id 不该建出工单
    assert session.tickets == []
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/Scripts/python.exe -m pytest tests/test_api_ticket.py
```

预期:**两条都红,红法不同**。第一条 404 或 405(端点还不存在)。`app/main.py`
在 `include_router` 之后 `mount("/")` 了静态目录,Starlette 的 `StaticFiles` 对
非 GET/HEAD 一律回 **405**,所以这里看到 405 是**正常的**,不是「路由顺序错了」。
第二条同样 404/405,而不是它最终要的 422。

> ⚠️ 这个文件**不需要 `@pytest.mark.db`**(改过一轮:原设计用真库,会引入本仓
> 第一个「TestClient + 真实 engine」组合并污染 `get_engine()` 的 lru_cache,
> 详见文件头 docstring)。`-m "not db"` 过滤对它是空操作,直接跑文件即可。

- [ ] **Step 3: 写 schema**

在 `app/schemas.py` 追加:

```python
class TicketRequest(BaseModel):
    """建工单请求(ch05「建工单」按钮)。

    `session_id` 的上限 32 与 `ChatRequest` 一致,理由见那处的 docstring:
    它是 `conversations.id` 的 varchar(32) 主键,放宽会以 DataError 形态复现
    并被错误分类判成不可恢复 → 502。
    """

    session_id: str = Field(min_length=1, max_length=32)
```

- [ ] **Step 4: 写端点**

在 `app/api/chat.py` 追加:

```python
@router.post("/api/ticket")
async def create_ticket_endpoint(
    request: TicketRequest,
    settings: Settings = Depends(get_settings),
    store: SessionStore = Depends(get_store),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """建工单:「建工单」按钮的后端入口。

    为什么需要它:`create_ticket` 是**模型工具**,只能由模型在对话里调;
    而按钮点击是 HTTP 请求,够不到模型工具。不加这个端点,验收 3 的
    「点建工单写 tickets 表」无法达成(spec §8.2)。

    护栏与对话端点一致:同一把会话锁串行化;`create_ticket` 是非幂等写操作,
    executor 的重试白名单不含它,**永不重试**。
    """
    lock = store.lock_for(request.session_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=settings.session_lock_timeout_seconds)
    except TimeoutError as exc:
        raise HTTPException(status_code=409, detail="该会话正在处理另一条消息,请稍后重试") from exc

    try:
        await ensure_conversation(
            session=session, session_id=request.session_id, user_id="demo-user"
        )
        tools = build_tools(session=session, conversation_id=request.session_id)
        registry = registry_for(tools)
        outcome = await execute_tool(
            tool_call={
                "name": "create_ticket",
                "args": {"description": "用户在会话中主动点击「建工单」", "ticket_type": "其他"},
                "id": "manual-ticket",
                "type": "tool_call",
            },
            registry=registry,
            settings=settings,
        )
        if not outcome.ok:
            raise HTTPException(
                status_code=502,
                detail=redact_api_key(outcome.summary, settings.openai_api_key),
            )
        return json.loads(outcome.content)
    finally:
        lock.release()
```

顶部 import 补:`from app.schemas import ChatRequest, TicketRequest`、
`from app.tools.executor import execute_tool`。

- [ ] **Step 5: 跑测试确认通过**

```bash
.venv/Scripts/python.exe -m pytest tests/test_api_ticket.py
```

预期:**2 passed**,不需要 MySQL。

**变异验证(必须做)**:把端点里的 `outcome = await execute_tool(...)` 那一段
连同 `return json.loads(outcome.content)` 一起换成
`return {"ticket_no": "T-20260101000000-DEAD", "status": "open"}` ——
`test_build_ticket_invokes_the_tool_and_returns_its_payload` 必须**恰好**打红
(红在 `session.tickets == []` 那条),第二条不受影响。然后还原。

- [ ] **Step 6: 随带的三处文字收口(T8 审查遗留的 1 Important + 2 Minor)**

**不新增任何行为,只改注释与 docstring。** 这三处都不是本任务引入的,是 T8
任务审查的遗留项;裁定由编排者做,并进本任务的提交(其中一处就在
`app/api/chat.py`,而另两处各只有一两行 —— 不值得为纯注释单开一轮实现+审查)。

**(a)`28 条` → `27 条`(三处)。** `app/api/chat.py:54` 的 docstring 写
「端点测试里 28 条都会朝 `https://example.invalid/v1` 发真实请求」;
`tests/test_api_chat.py:113` 与 `:240` 的注释里各有一处同形表述。
**实测真值是 27**(`.venv/Scripts/python.exe -m pytest tests/test_api_chat.py
--collect-only` 现数),三处一并改成 27。
`tests/test_api_chat.py:901` 的「上面 25 条」指的是**该行以上**的 25 条
(25 + 下面 2 条 = 27),它是对的,**不要动**。

> 数字一律用 `--collect-only` 现数,不要从任何人的报告里抄 —— 本条的来源
> (审查报告)自己就把 27 写成了 25。

**(b)`app/kb/assess.py` 的 `assess_sufficiency` 标注「不在请求路径上」。**
它是 ch04 spec §5.5 的交付物。T8 删掉 `services/chat.py:stream_turn` 之后,
它的**唯一**调用方只剩 `tests/test_kb_assess.py` —— 而那个文件至今全绿,
于是没有任何东西会提示「这条路已经没有任何请求能走到」。
ch05 spec §50 用**事前置信度闸**取代了它的事后自评,但它本身没有坏,
且在 ch05 里删它等于单方面改掉 ch04 已交付的接口面。
**裁定:保留,但把「不在请求路径上」写进 docstring 与测试文件头**,让下一个
读者一眼看到,而不是靠 grep 反推。

在 `assess_sufficiency` 的 docstring 末尾(现有那段的后面)追加:

```python
    **当前不在请求路径上**(ch05 spec §50):ch05 起「召回够不够」改由
    `app/agent/nodes.py` 的**置信度闸在事前**判定,ch04 这套「生成后再自评」
    被整段替换。T8 删掉 `services/chat.py:stream_turn` 后,本函数的生产调用方
    为零,只剩 `tests/test_kb_assess.py` 在跑它 —— 那个文件全绿**不代表**线上
    有这条链路。函数本身没坏,也不删(删它等于在 ch05 里改掉 ch04 已交付的
    接口面);留着是为了将来需要「生成后二次自评」时有现成的、有测试的实现。
```

并在 `tests/test_kb_assess.py` 的模块 docstring(第 1 行)末尾追加一句:

```python
注意:被测的 `assess_sufficiency` 当前**不在请求路径上**(ch05 spec §50 用事前
置信度闸取代了它的事后自评),本文件全绿不代表线上有这条链路;理由见该函数 docstring。
```

**(c)`tests/test_api_chat.py:903-905` 的 `choices` 帧注释改成将来时。**
现注释说前端的「转人工 / 建工单」两个按钮「就靠它」,但
`app/static/index.html` 此刻**没有**任何 `choices` / `handoff` / `ticket` 处理
—— 那要等 T10。用例本身没错(它钉的是 T10 将要消费的帧契约),错的是把
「将来会依赖」写成了「现在就依赖」。把措辞改成「T10 的前端将消费这个帧」。

```bash
.venv/Scripts/python.exe -m pytest tests/test_api_chat.py
.venv/Scripts/python.exe -m pytest tests/test_kb_assess.py
```

预期:两个文件全 passed,且**数字与本轮开始前一字不差**(纯注释改动:
`test_api_chat.py` 仍是 27,`test_kb_assess.py` 不变)。有任何一个数字变了,
说明改到了行为而不是注释,停下来报告。

- [ ] **Step 7: 提交**

```bash
git add app/schemas.py app/api/chat.py tests/test_api_ticket.py         tests/test_api_chat.py app/kb/assess.py tests/test_kb_assess.py
git commit -m "feat: ch05 新增 POST /api/ticket —— 建工单按钮的后端入口"
```

---

## Task 10:前端两个独立按钮(Vibe Coding,不套 TDD)

**Files:**
- Modify: `app/static/index.html`

**现状(504 行)**:`handleBlock(block, ctx)` 在 **341–379 行**用 switch 分发事件;
`addAssistant()` 在 **287–312 行**建 `.badges` / `.body` / `.feedback`;`.fb` 胶囊样式在 **164–166 行**;
`ctx` 是单轮局部对象;**流结束后**在 `finally`(431 行)调 `makeCitesClickable`。

- [ ] **Step 1a: 让 `addAssistant()` 把气泡元素**也**返回出来**

`addAssistant()` 现在的末行是 `return { badges, body, citations: [] };` ——
`bubble` 只是**局部变量**,没进返回对象。而下面 `renderChoices(ctx.bubble, ...)`
要的正是它:不改这一行,`ctx.bubble` 是 `undefined`,
`bubble.appendChild(bar)` 直接抛 TypeError,**两个按钮一个都不出现**,
验收 3 卡在第一步。

**这一行还顺带修掉一个既有 bug,不是可选的清理**:`ctx.bubble` 在改动前
**从来没被赋过值**,而 `index.html` 的 error 帧分支与 `catch` 分支都在读它 ——
今天任何一次 error 帧都会让页面**砖掉**(详见 Step 4 的说明框)。改末行:

```js
    return { bubble, badges, body, citations: [] };
```

- [ ] **Step 1b: 加 `choices` 分支**

在 `handleBlock` 的 switch 里(`citations` 分支之后)加:

```js
      case "choices":
        renderChoices(ctx.bubble, payload.options || []);
        break;
```

- [ ] **Step 2: 写渲染与交互函数**

在 `makeCitesClickable` 附近加(样式沿用 `.fb` 的胶囊风格与 `--brand` 配色):

```js
  // 「转人工」「建工单」是**两件事**,分开渲染成两个独立按钮,互不绑定。
  // 都不点、继续发消息 → 一切照常,后端不会被调用。
  function renderChoices(bubble, options) {
    const bar = document.createElement("div");
    bar.className = "choices";
    options.forEach((opt) => {
      const btn = document.createElement("button");
      btn.className = "choice-btn";
      btn.dataset.key = opt.key;
      btn.textContent = opt.label;
      btn.addEventListener("click", () => onChoice(opt, btn, bar, bubble));
      bar.appendChild(btn);
    });
    bubble.appendChild(bar);
  }

  function onChoice(opt, btn, bar, bubble) {
    if (btn.disabled) return;          // 一次性锁定:同 👍/👎 的既有语义
    btn.disabled = true;
    if (opt.key === "handoff") {
      // 本章是**纯前端模拟**,不接真人系统、不调后端。
      appendSystem("已转接人工客服");
      appendAssistantText("您好,我是客服小猫,请问有什么可以帮您的");
    } else if (opt.key === "ticket") {
      createTicket(btn, bar);
    }
  }

  async function createTicket(btn, bar) {
    btn.textContent = "建单中…";
    try {
      const resp = await fetch("/api/ticket", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const data = await resp.json();
      btn.textContent = "已建单 " + data.ticket_no;
    } catch (err) {
      btn.textContent = "建单失败,请重试";
      btn.disabled = false;            // 失败允许重试
    }
  }

  function appendSystem(text) {
    const row = document.createElement("div");
    row.className = "row system";
    const b = document.createElement("div");
    b.className = "bubble";
    b.textContent = text;
    row.appendChild(b);
    logEl.appendChild(row);
    scrollToEnd();
  }

  function appendAssistantText(text) {
    const ctx = addAssistant();
    ctx.body.textContent = text;
    scrollToEnd();
  }
```

- [ ] **Step 3: 加样式**

在 `.fb` 那段附近加:

```css
  .choices { display: flex; gap: 8px; margin-top: 10px; }
  .choice-btn {
    background: #fff; color: var(--brand); border: 1.5px solid var(--brand);
    border-radius: 999px; padding: 5px 14px; font-size: 13px; cursor: pointer;
  }
  .choice-btn:hover:not(:disabled) { background: #dcebf9; }
  .choice-btn:disabled { opacity: .6; cursor: default; }
  .row.system .bubble { background: #eef4fa; color: #5b7186; border-style: dashed; }
```

- [ ] **Step 4: 人工验证(纯 UI 例外,不写自动化测试)**

起服务后:

```bash
netstat -ano | grep ":8000"     # 先查僵尸进程,再起
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

浏览器开 `http://localhost:8000`,发「我要投诉」,逐条确认:

1. 出现**两个独立按钮**(转人工 / 建工单),点一个不禁用另一个。
2. 点「转人工」→ 出现「已转接人工客服」+「您好,我是客服小猫,请问有什么可以帮您的」。
3. 点「建工单」→ 按钮文字变成 `已建单 T-...`;`tickets` 表新增一行。
4. 刷新页面重来,**都不点**,直接继续发消息 → 正常对话,`tickets` 表无新增。

> **Step 1a 顺带修掉一个既有 bug,必须单独确认 —— 它比"按钮不出现"严重得多。**
>
> `ctx.bubble` 在改动前的现场代码里**根本没有被赋值**:`addAssistant()` 的返回值
> 只有 `{ badges, body, citations }`,而 `send()` 用 `{ ...addAssistant(), ... }` 造 ctx ——
> 可是 `index.html` 有**两处**在读 `ctx.bubble`(error 帧分支与 `catch` 分支)。
> 后果是**级联的**,不是"报个错而已":
> error 帧一来,error 分支先抛 `TypeError` → 落进外层 `catch` → `catch` 里**还是**
> `ctx.bubble.appendChild` → **再抛一次且无人接** → `finally` 不执行 → `busy` 永远为
> `true`、输入框与发送键永远禁用 —— **页面当场砖掉,只能刷新**。
> (以上是从代码读出来的结论,**没有在浏览器里复现过**;Step 1a 之后这条路径才第一次
> 真正可用。)
>
> **确认方式(必须实测,不许只读代码)。**
>
> ⚠️ **这一版计划原本给的是"在 DevTools 控制台里跑 `Object.keys(addAssistant())` /
> `handleBlock(..., ctx)`" —— 已订正,那样跑必红。** 整个脚本包在
> `(() => { ... })();` 里(`index.html:248` 开、`:501` 闭),带 `"use strict"`,
> 且**全文件零个 `window.` 赋值** —— 所以 `addAssistant`、`handleBlock` **从控制台
> 一律够不到**,只会得到 `ReferenceError`;而 `ctx` 本身是 `send()` 里的 `const`
> (`:386`),函数作用域,任何情况下都拿不到。
>
> 改用**网络层**触发:确定性、不改代码、不用重启,而且打到的是读 `ctx.bubble` 的
> 两处之一(`catch` 分支):
>
> 1. 起服务,浏览器开 `http://localhost:8000`(先 `netstat -ano | grep ":8000"` 查僵尸)。
> 2. DevTools → Network → Throttling → **Offline**。
> 3. 发一条消息(`fetch` 会 reject → 落进 `catch` → `ctx.bubble.appendChild`)。
> 4. 期望(**改后**):出现 ⚠️「连接中断(...)」,**且输入框与发送键仍然可用**,
>    可以接着发下一条。
> 5. 把 Throttling 调回 **No throttling**。
>
> **必须再做一次"改前"的对照**,否则「Step 1a 修掉了砖机」就只是读代码读出来的结论:
> 把 `return { bubble, badges, body, citations: [] };` 里的 **`bubble,` 临时删掉**
> → 重复 2–4 → 这次**必须**看到页面砖掉(⚠️ 出现之后输入框与发送键**永久禁用**,
> 只能刷新恢复)→ 还原那一行。
>
> 两次现象都写进报告。**"改前"那一次是这条结论唯一的实测依据** ——
> 没做就只能写"从代码读出",那不算验证。

- [ ] **Step 5: 提交**

```bash
git add app/static/index.html
git commit -m "feat: ch05 聊天页渲染「转人工」「建工单」两个独立按钮"
```

---

## Task 11:验收脚本(五条)

**Files:**
- Create: `scripts/acceptance_ch05.sh`

**前置**:服务已启动、MySQL 与 Milvus 起着、`.env` 有真实 key。

- [ ] **Step 1: 写脚本**

创建 `scripts/acceptance_ch05.sh`(含中文的请求体走 stdin heredoc —— MSYS2 会按 CP936
重编码 argv,服务端只回 `error parsing the body`):

```bash
#!/usr/bin/env bash
# ch05 验收 1–5。前置:服务已在 8000 启动(docker start mysql milvus-standalone)。
#
# 断言依据:done 帧里的 trace / intent / agent_steps —— 本轮**确定性的**证据链。
# 不靠模型自由文本(deepseek 在 temperature=0 下依然非确定),也不 grep 原始
# SSE 流(逐 token 推送会把 "1001" 切成三帧)。
set -uo pipefail
BASE="${BASE:-http://localhost:8000}"
# 与 scripts/acceptance.sh 一致:一律用 venv 里的解释器。裸 `python` 在本机
# 不保证存在、也不保证是 venv 那个 —— 那样 new_sid 会产出空串,整个脚本静默走偏。
PYTHON="${PYTHON:-.venv/Scripts/python.exe}"
PASS=0; FAIL=0

ok()  { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad() { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }

# 发一条消息。请求体走 stdin heredoc —— 含中文的请求体不能走 curl 的 argv,
# MSYS2 会按 CP936 重编码,服务端只回 error parsing the body。
ask() {
  local sid="$1" msg="$2"
  curl -s -N -X POST "$BASE/api/chat/stream" \
    -H "Content-Type: application/json" \
    --data-binary @- <<JSON
{"session_id":"$sid","message":"$msg"}
JSON
}

new_sid() { "$PYTHON" -c "import uuid;print(uuid.uuid4().hex)"; }

# 把 token 帧拼回整段回复(逐 token 推送,不能直接 grep)
join_tokens() {
  "$PYTHON" -c '
import json, sys
lines = sys.stdin.read().splitlines()
parts = []
for i, l in enumerate(lines):
    if l.strip() == "event: token" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        parts.append(json.loads(lines[i+1][6:]).get("text", ""))
sys.stdout.buffer.write("".join(parts).encode("utf-8"))'
}

# 从最后一帧(done)的 data 里取一个字段
done_field() {
  "$PYTHON" -c '
import json, sys
lines = sys.stdin.read().splitlines()
data = [l[6:] for l in lines if l.startswith("data: ")]
v = json.loads(data[-1]).get(sys.argv[1])
sys.stdout.buffer.write(
    (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)).encode("utf-8"))' "$1"
}

# 数一数有多少个 tool_result 帧是**失败**的(ok 为假)。
# 存在的理由见验收 5 那段注释:它是 `"type": "tool_call"` 键在真实链路上的**唯一**探针。
# 逐帧解析而不是 grep —— `_frame` 的 JSON 分隔符格式不属于本脚本的契约,
# 用空格敏感的字符串匹配去断帧内容是自找假红。
bad_tool_results() {
  "$PYTHON" -c '
import json, sys
bad = 0
lines = sys.stdin.read().splitlines()
for i, l in enumerate(lines):
    if l.strip() == "event: tool_result" and i + 1 < len(lines) and lines[i+1].startswith("data: "):
        if not json.loads(lines[i+1][6:]).get("ok", True):
            bad += 1
print(bad)'
}

echo "== 验收 1:政策类问题走到强制检索节点 =="
SID=$(new_sid)
ask "$SID" "退货政策是怎么规定的" > /tmp/ch05_1.sse
T=$(done_field trace < /tmp/ch05_1.sse)
G=$(done_field gate_passed < /tmp/ch05_1.sse)
case "$T" in
  *"retrieve_knowledge"*) ok "trace 有强制检索节点:$T(gate_passed=$G)";;
  *) bad "trace 里没有强制检索节点:$T";;
esac

echo "== 验收 2:Agent 自己调工具作答 =="
SID=$(new_sid)
ask "$SID" "订单 1001 的物流到哪了" > /tmp/ch05_2.sse
if grep -q "event: tool_call" /tmp/ch05_2.sse; then
  ok "Agent 自己调了工具(trace=$(done_field trace < /tmp/ch05_2.sse))"
else bad "没有 tool_call 帧"; fi

echo "== 验收 3:投诉 → 两个独立选项 =="
SID=$(new_sid)
ask "$SID" "我要投诉" > /tmp/ch05_3.sse
if grep -q '"handoff"' /tmp/ch05_3.sse && grep -q '"ticket"' /tmp/ch05_3.sse; then
  ok "choices 帧含 handoff 与 ticket 两个独立选项"
else bad "choices 帧缺失或不完整"; fi

echo "== 验收 4:闲聊拿到固定话术 =="
SID=$(new_sid)
ask "$SID" "你好" > /tmp/ch05_4.sse
TXT=$(join_tokens < /tmp/ch05_4.sse)
case "$TXT" in *"客服小猫"*) ok "闲聊固定话术:$TXT";; *) bad "闲聊话术不符(可能空了):$TXT";; esac

echo "== 验收 5:复杂问题 ReAct 不止一步 =="
SID=$(new_sid)
# 问题必须是**强制串行**的:第二次工具调用的入参**只能**来自第一次的返回。
# `query_order` 与 `query_logistics` **都直接收 order_id**,所以
# 「订单 1001 的物流到哪了」这类问法,模型完全可以在**同一轮里并发**发两个
# tool_call(T6 审查实测过:那次 `agent_steps` 是 2 —— `agent_steps` 数的是
# **绑工具的轮数,含收敛的那一轮**,并发一轮 + 收敛一轮 = 2,所以它**也能过**)。
# 也就是说那个问法不会假红,但它**过的没有道理**:两步之间没有数据依赖,
# 「不止一步」成立只是因为**收敛那一轮也被算了一步**。
# 「订单 1001 买的是什么商品?那件商品现在还有货吗」在结构上不可能并发:
# 订单里有 `product` 字段(`app/tools/business.py:79`),商品名**只能**先查订单
# 才知道,所以「不止一步」是**问题本身的形状**保证的,与轮数怎么数无关。
ask "$SID" "订单 1001 买的是什么商品?那件商品现在还有货吗" > /tmp/ch05_5.sse
STEPS=$(done_field agent_steps < /tmp/ch05_5.sse)
N=$(grep -c "event: tool_call" /tmp/ch05_5.sse)
BAD=$(bad_tool_results < /tmp/ch05_5.sse)
if [ "$STEPS" -ge 2 ] && [ "$N" -ge 2 ] && [ "$BAD" -eq 0 ]; then
  ok "ReAct 走了 $STEPS 步、$N 次工具调用、0 次工具失败(trace=$(done_field trace < /tmp/ch05_5.sse))"
else bad "步数不足或工具失败:agent_steps=$STEPS tool_calls=$N 失败工具数=$BAD"; fi

# ⚠️ `BAD -eq 0` 这一条**不是补充,是唯一的探针**,别删。
# `"type": "tool_call"` 键的丢失在**单测层捕获不到**:替身 `FakeTool` 不查这个键、
# `execute_tool` 原样透传 —— T6 审查实测,把该键从 `FakeChunk.__init__` 与 `__add__`
# 同时删掉,`tests/test_agent_node.py` **仍然 7 passed**。而真实链路上,
# `BaseTool.ainvoke` 判「这是不是工具调用」**只看这一个键**,缺了它就把整个 dict 当
# **参数**去校验 schema → `ValidationError` → `app/tools/executor.py:77` 转成
# `ok=False` +「工具参数不合法」。**不抛异常、不报错、不写日志**,
# 只是模型永远拿不到数据、于是开始编 —— 本项目最怕的那类静默故障。
# `BAD` 就是它的探针:真模型 + 真 `@tool` + 真执行器,缺键则必然非 0。

echo
echo "结果:$PASS 通过,$FAIL 失败"
[ "$FAIL" -eq 0 ]
```

> **验收 3 的后半段(点按钮)没法脚本化** —— 转人工是纯前端模拟,建工单要真的
> 点一下浏览器按钮。那部分在任务 10 的 Step 4 人工验,并在此脚本之外留一份
> 手工记录:`tickets` 表行数前后对比。

- [ ] **Step 2: 跑**

```bash
netstat -ano | grep ":8000"          # 先查僵尸
bash scripts/acceptance_ch05.sh
```

预期:`结果:5 通过,0 失败`。

- [ ] **Step 3: 补一条对 trace 的直接断言**

验收 1 与 5 的**权威证据是服务端日志里的 `trace`**(`log_turn` 打的那行),
上面的脚本用的是端到端替代信号。用 Python 直接驱动图再验一次:

```bash
.venv/Scripts/python.exe - <<'PY'
import asyncio, logging, sys
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
from app.agent.graph import build_graph, get_checkpointer
from app.config import get_settings
from app.llm import create_chat_model, create_extract_model
from app.db.base import get_sessionmaker
from app.tools.registry import build_tools, registry_for, build_retriever
from app.agent.emit import make_emitter
from app.services.history import ensure_conversation

async def main():
    s = get_settings()
    async with get_sessionmaker()() as session:
        await ensure_conversation(session=session, session_id="000000000000000000000000000000aa", user_id="demo")
        tools = build_tools(session=session, conversation_id="000000000000000000000000000000aa")
        graph = build_graph(
            model=create_chat_model(s), intent_model=create_extract_model(s),
            tools=tools, registry=registry_for(tools), settings=s,
            retriever=build_retriever(session), session=session,
            conversation_id="000000000000000000000000000000aa",
            emit=make_emitter(), checkpointer=get_checkpointer(),
        )
        out = await graph.ainvoke(
            {"conversation_id": "000000000000000000000000000000aa",
             "user_input": "退货政策是怎么规定的", "history": [], "trace": []},
            config={"configurable": {"thread_id": "000000000000000000000000000000aa"}},
        )
        sys.stdout.buffer.write(("trace = " + " > ".join(out["trace"]) + "\n").encode("utf-8"))

asyncio.run(main())
PY
```

预期:输出的 `trace` 里能看到 `retrieve_knowledge:... > confidence_gate:pass > ...` ——
这就是验收 1 的权威证据(强制检索节点被走到)。

> ⚠️ **两点口径**(T7 复评提出):
> 1. `out["trace"]` 是**未切片的累积通道**。这里只跑一次、且 thread 是这个脚本
>    新建的,所以它等于当轮 —— **但别把它当通用做法**。权威性来自 `log_turn`
>    打的那行**日志**(那行是切过片的,`nodes.py:358`),不是这个返回值。
>    要在同一 thread 上跑第二次,必须改用日志行或 `stream_mode="custom"` 的
>    `trace` 帧,否则读到的是两轮的拼接。
> 2. 这个片段**照抄自 Step 1 的脚本骨架**,`build_graph` 的签名若在 T7/T8 落地时
>    有出入,以 `app/agent/graph.py` 的实况为准(先读再用)。

- [ ] **Step 4: 提交**

```bash
git add scripts/acceptance_ch05.sh
git commit -m "test: ch05 验收脚本 1–5"
```

---

## 收尾检查

- [ ] `.venv/Scripts/python.exe -m pytest` 全绿(含 db,需 MySQL)
- [ ] `bash scripts/acceptance.sh`(ch01–ch04 的老验收)**仍然全绿** —— 本章改了 `/api/chat/stream` 的实现,老验收是回归网
- [ ] 确认 `app/agent/loop.py` 与 `tests/test_agent_loop.py` 已删除
- [ ] `dev-notes/ch05.md` 补齐每个阶段(不许收尾一次性补记)
- [ ] spec §12 记齐实现订正(至少:`prepare_turn` 语义变更、`emit` 适配层)
- [ ] 交付:演示命令、测试结果、dev-notes 路径
