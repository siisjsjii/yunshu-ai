# 电商智能客服系统 · ch05:Workflow 确定性编排 + 主力 Agent — 设计文档

> 设计经用户逐点确认(2026-09-19:八个功能需求、技术栈、五条验收、五条工作要求)。
> 两个选型点经用户「开干吧」批准(手写 Agent 节点 + `stream_mode=["messages","custom"]`)。
> 澄清三问的答复见 §5.0。
> 本文是本章权威设计文档;实现与设计的偏离记录在 §12「实现订正」。

## 1. 目标与验收

把客服系统从「单轮工具调用」升级成**生产级编排架构**:用 LangGraph 的 Workflow 把
**确定性的骨架**(指代消解 → 意图识别 → 分流 → 检索 → 置信度闸 → Agent → 日志)
固化下来,把**不确定的部分**(推理与多步工具调用)收进骨架里的一个核心节点 —— 主力
ReAct Agent。先手写一遍最裸的 Agent 循环祛魅,再用 LangGraph 重构。

验收标准(用户给定):

1. 问政策类问题,日志里能看到**强制检索节点被走到**。
2. 问「订单 1001 的物流到哪了」,Agent 自己调工具作答。
3. 说「我要投诉」,前端出现「转人工」「建工单」两个**独立**按钮;点「转人工」前端显示
   「已转接人工客服」并蹦出客服小猫的问候,点「建工单」才写 `tickets` 表;两者分开,
   都不点就接着正常对话、不做任何动作。
4. 闲聊拿到固定话术。
5. 一个要先查订单再查物流的复杂问题,能看到 ReAct 走了不止一步。

## 2. 技术栈与版本

| 组件 | 版本 | 说明 |
|---|---|---|
| LangGraph | 1.2.11(已装,本章补进 requirements.txt) | 图编排 + State + checkpointer |
| langgraph-checkpoint | 4.2.0(已装,随上) | `InMemorySaver` |
| LangChain | 1.4.0 / langchain-core 1.6.3(现有) | `bind_tools` / `astream` |
| FastAPI | 0.141.1(现有) | 新增 `/api/ticket` 端点 |
| 前端 | 原生 HTML/CSS/JS(无框架,用户点名) | 改 `app/static/index.html` |

MySQL / Milvus / BGE-M3 / bge-reranker 沿用,本章不新增存储。

## 3. 现状与复用

| 复用物 | 位置 | 本章用途 |
|---|---|---|
| `KnowledgeRetriever`(混合+重排) | `app/retrieval/search.py` | 知识类意图的**强制预检索** |
| 五个 `@tool` | `app/tools/business.py` | Agent 的工具集,**不新写业务工具** |
| `execute_tool`(超时/重试/错误分类) | `app/tools/executor.py:41` | Agent 每一步的工具执行 |
| `assess.py` 的 `record_low_confidence` | `app/kb/assess.py:56` | 置信度闸失败落池 |
| `render_system_prompt` / `to_lc_messages` | `app/prompts.py` | System Prompt 与唯一消息转换点 |
| `load_history` / `append_turn` | `app/services/history.py` | **跨轮会话权威源**(MySQL) |
| 锁 / 脱敏 / 预算校验 | `app/api/chat.py` | SSE 端点的既有护栏,原样保留 |

**被取代的**:`app/services/chat.py` 的 `stream_turn`(单轮编排)。它的「只做单轮是结构
保证」让位于 ReAct 的多轮循环;它的 ch04 事后自评被 §5.4 的**事前置信度闸**取代。

## 4. 架构

### 4.1 图拓扑

```
START
 └→ resolve_references    指代消解(本章:原样透传,正式版留后)
     └→ classify_intent    LLM 一次,输出七类 JSON
         └→ route_by_intent   ← 条件边,**规则写死在代码里**(纯函数)
             ├─ knowledge  → retrieve_knowledge → confidence_gate ─┬ pass → agent
             │                                                     └ fail → fallback_reply
             ├─ business   → agent
             ├─ complaint  → complaint_reply   (安抚 + choices 帧)
             ├─ chitchat   → chitchat_reply    (固定话术,不调模型)
             └─ fallback   → fallback_reply    (固定话术,不调模型)
                                     ↓ 五出口汇合
                                  log_turn
                                     ↓
                                    END
```

### 4.2 七类意图 → 四个出口的映射(写死在代码里)

| 意图 | 出口 | 行为 |
|---|---|---|
| 商品咨询、退款退货 | **knowledge** | 强制预检索 → 置信度闸 → 证据够才连证据一起交 Agent |
| 物流、订单、售后 | **business** | 不预检索,直接进 Agent 自己调工具 |
| 投诉 | **complaint** | 不进 Agent,回安抚话术 + 推 `choices` 帧 |
| 闲聊 | **chitchat** | 固定话术,不花模型调用 |
| 其余 / 分类解析失败 / 越界 | **fallback** | 固定兜底话术,不花模型调用 |

**退款退货的特殊性**:它属知识类(先检索政策),但 Agent 仍可自己调订单工具 —— 这是
「预检索」与「Agent 自调工具」的唯一交叠点,不进 `business` 分支,Agent 的工具集也不
按意图裁剪(五个工具恒定绑定)。

### 4.3 组件划分

```
app/agent/state.py        ChatState(TypedDict)+ 常量
app/agent/routing.py      七类 → 四出口的**纯函数**路由表
app/agent/nodes.py        各节点实现(职责见 §5)
app/agent/graph.py        组装 StateGraph + 编译(InMemorySaver)
app/agent/loop.py         **临时**:手写的最裸 Agent 循环(祛魅热身,重构后删)
app/api/chat.py           改:接图 + 新增 POST /api/ticket
app/schemas.py            + TicketRequest(遵循「唯一类型源」约定)
app/prompts.py            + 带证据块的消息组装(唯一转换点不变)
app/static/index.html     改:choices 帧 → 两个独立按钮 + 交互
```

**`app/services/chat.py` 的处置**:删 `stream_turn`(单轮编排语义已无处可留),
**保留 `prepare_turn`** —— SSE 端点的「预算校验必须在流开始前完成」这条硬约束仍然需要它,
它产出的裁剪后历史进 state(见 §5.1 的 `history`)。

依赖方向不变:`api → agent → {tools, retrieval, db, prompts, llm}`,仍是单向。

## 5. 关键设计决策

### 5.0 澄清结论(用户答复)

1. **checkpointer = `InMemorySaver`,MySQL 仍是会话状态权威**。图状态(意图/检索结果/
   中间步骤)只在单次请求内贯穿;跨轮历史仍走 `load_history` 读 MySQL、流完
   `append_turn` 落库。**不产生第二份真相**。
2. **手写 Agent 循环先跑通,再用 LangGraph 重构时删掉**。它是中间产物不是交付物;
   运行证据留 `dev-notes/ch05.md`。
3. **置信度闸 = 纯检索分数阈值,零额外模型调用**。

### 5.1 State schema

```python
class ChatState(TypedDict):
    conversation_id: str          # = thread_id = MySQL 会话 id
    user_input: str
    history: list[Message]        # 来自 MySQL(权威源),转 BaseMessage 只经 prompts.to_lc_messages

    resolved_input: str           # 本章 = user_input 原样
    intent: str                   # 七类之一 | "fallback"
    evidence: list[dict]          # 知识类:检索到的 chunk(含 score / section_path / chunk_id)
    gate_passed: bool

    messages: Annotated[list, add_messages]   # Agent 内部 ReAct 消息序列
    agent_steps: int
    tool_calls_made: list[dict]   # 供日志与验收断言

    reply: str                    # 最终回答文本(= 用户实际看到的拼接)
    citations: list[dict]
    choices: list[str]            # ["handoff", "ticket"];空则不推帧

    trace: Annotated[list[str], operator.add]  # 节点留痕,验收 1/5 靠它
    usage: dict
```

用 `TypedDict` + `Annotated` reducer(而非 Pydantic):这是 LangGraph 的惯用法,
且 `add_messages` 是官方为消息序列提供的 reducer,自己实现会重复造轮子。

### 5.2 路由是纯函数

`route_by_intent(state) -> Literal["knowledge","business","complaint","chitchat","fallback"]`
不碰 IO、不调模型、不含分支外的逻辑 —— 因此可以**表驱动单测**覆盖七类 + 兜底 + 越界。
这是「确定性骨架」的核心:分流规则写死在代码里,模型只能决定**意图标签**,不能决定**走向**。

### 5.3 强制预检索(knowledge 出口)

`retrieve_knowledge` 直接调 `KnowledgeRetriever.search(resolved_input)`,**复用 ch03/ch04
的混合检索 + 重排**,不走 `query_faq` 工具(工具是给 Agent 用的,这里是骨架的确定性一步)。
产出 `evidence` 与 `citations`(供前端点引用)。

### 5.4 置信度闸

- **位置**:`retrieve_knowledge` 之后、`agent` 之前。理由就是用户说的「Agent 的答复是
  流式吐给用户的,**答完再判就晚了**」。
- **判据**:`evidence` 非空 且 `max(score) >= settings.retrieval_score_threshold`(0.58,
  ch04 实测得出)。
- **不通过**:回兜底话术、**不进 Agent**、问题落 `low_confidence_questions`
  (`entry_point="置信度闸"`)留给后续数据飞轮。
- **business 出口不走这道闸**:业务数据类没有检索证据,证据强弱无从谈起(用户明确)。

### 5.5 主力 Agent(ReAct)

```
组装消息:system(render_system_prompt)+ history(进 state 的裁剪后历史)
         + [可选:证据块] + user   ← 组装仍走 app/prompts.py,它是唯一转换点
loop:
  ① model.bind_tools(五个工具).astream(msgs) → 文本 token **实时外推**
  ② 累积出 AIMessage
  ③ 无 tool_calls → 收敛,退出
  ④ 有 tool_calls → 逐个 execute_tool,推 tool_call / tool_result 帧
                   → 回灌 ToolMessage → 回到 ①
停止条件(任一触发即强制收敛):
  - agent_steps > settings.max_agent_steps(默认 5)
  - 累计 usage.total_tokens > settings.agent_token_budget(默认 20000)
```

**不用 `ToolNode` / `create_react_agent`**:工具执行必须走 `app/tools/executor.py`,它承载
CLAUDE.md 点名的错误语义 —— `ToolInfrastructureError` 必须向上抛(绝不伪装成「查不到」)、
重试用白名单(`create_ticket` 永不重试)、`ValidationError`/`ToolNotFound` 不重试、10s 超时。
`ToolNode` 直接 `tool.ainvoke`,这些语义全丢;`create_react_agent` 还把停止条件与 token
预算挡在外面。

**「实时外推 token」是乐观策略**:模型决定调工具的那轮通常不产出文本;若产出前言
(如「我帮你查一下」),推给用户正是想要的。ch04 已在 `stream_turn` 里沿用同一实测结论。
但**它是模型行为、不是结构保证** —— 见 §9 待实测项。

### 5.6 帧协议

沿用 ch04 的 SSE 帧:`meta` / `token` / `tool_call` / `tool_result` / `citations` / `done` / `error`。

**新增一帧** `choices`:

```json
{"options": [{"key": "handoff", "label": "转人工"}, {"key": "ticket", "label": "建工单"}]}
```

由 `complaint_reply` 节点用 `get_stream_writer()` 发出(`stream_mode="custom"`),也可由
Agent 在判断合适时发出(需求 8:「Agent 判断合适时」)。前端按帧渲染按钮,**不反解文本**。

**token 也走 `custom` 帧**:`agent` 节点自己在每一轮流式读模型时 `emit({"frame":"token","text":...})`。
所有帧因此只有**一条发出路径**(见 §12 的订正 —— 原设计用的 `messages` 流模式在真机验证时被否掉)。

### 5.7 日志与验收可检查性

每个节点往 `state["trace"]` 追加一条留痕,`log_turn` 汇总成一行结构化日志:

```
trace = ["resolve_references",
         "classify_intent:knowledge",
         "retrieve_knowledge:5 hits top=0.71",
         "confidence_gate:pass",
         "agent:step1 tool=query_order",
         "agent:step2 tool=query_logistics",
         "agent:converged",
         "log_turn"]
```

- **验收 1**(政策类走到强制检索)↔ trace 含 `retrieve_knowledge`。
- **验收 5**(ReAct 不止一步)↔ trace 含 `agent:step1` 与 `agent:step2`。

验收因此是**机械可断言**的,不依赖模型自由文本。

## 6. 数据层

**不新增表**。复用:

- `low_confidence_questions`(ch04)—— 置信度闸失败落池,`entry_point="置信度闸"`。
- `tickets`(ch02)—— 「建工单」按钮写入。
- `messages`(ch01)—— `log_turn` 调 `append_turn` 落库,仍是会话权威源。

## 7. 前端(Vibe Coding,不套 TDD)

改 `app/static/index.html`(504 行,原生 JS IIFE):

- `handleBlock` 的 switch 增 `choices` 分支:在气泡下方渲染**两个独立按钮**
  (沿用 `.fb` 胶囊样式与 `--brand` 配色)。
- 点「转人工」:**纯前端**,插入两条消息 —— 「已转接人工客服」+「您好,我是客服小猫,
  请问有什么可以帮您的」。不接真人系统,不调后端。
- 点「建工单」:调 `POST /api/ticket`,成功后展示返回的 `ticket_no`。
- 两个按钮**互不绑定**:点一个不禁用另一个;都不点、继续发消息 → 一切照常,后端不做
  任何动作(后端本来就没被调用)。
- 按钮点击后自身置为「已处理」并锁定(同 👍/👎 的既有一项式锁定语义)。

## 8. API 变更

### 8.1 改:`POST /api/chat/stream`

端点骨架(锁、预算校验、`ensure_conversation`、`EventSourceResponse` 手工构造、脱敏、
`finally` 释放锁)全部保留,只把 `stream_turn(...)` 换成**驱动图**:

```python
async for chunk in graph.astream(state, config={"configurable": {"thread_id": session_id}},
                                 stream_mode="custom"):
    # chunk 形如 {"frame": "token"|"tool_call"|"tool_result"|"choices", ...}
    ...
```

**锁的语义不变**:整轮仍由 `store.lock_for(session_id)` 串行化。

**图的组装粒度**:`query_faq` / `create_ticket` 是**每请求闭包**(见 `business.py` 的说明),
所以**图按请求组装**(`build_graph(tools, registry, ...)`,`StateGraph` 构造是纯内存操作,
不进请求路径);**`InMemorySaver` 是进程级单例** —— 它必须跨请求存活,否则 thread 状态
在下一轮就没了。

### 8.2 新增:`POST /api/ticket`

```json
请求  {"session_id": "..."}
响应  {"ticket_no": "T-20260919...", "status": "open"}
```

内部调 `make_create_ticket(session, session_id)`,沿用既有护栏(锁、脱敏、
`create_ticket` 永不重试)。**理由**:`create_ticket` 是模型工具,按钮点击是 HTTP 请求,
够不到模型工具;不加这个端点,验收 3 的「点建工单写 tickets 表」无法达成。

`session_id` 用现有 `ChatRequest` 的同一套约束(varchar(32) 主键,`max_length=32`)——
不一致会以 DataError 形态复现,被错误分类判成不可恢复 → 502。

## 9. 待实测项与风险

| 项 | 处置 |
|---|---|
| 「调工具的轮次不出文本」是模型行为非结构保证 | 真机冒烟;若模型前言与 `reply` 不一致,接受「推给用户的是拼接文本」这一口径并在 §12 记账 |
| 意图识别准确率(七类 + 兜底) | 拿标注样例跑一遍(用户工作要求 1);不用字符串断言 |
| `stream_mode="messages"` 的 token 过滤是否可靠 | 真机验证 `metadata["langgraph_node"]` 取值 |
| 图 + checkpointer 与既有锁的叠加 | 锁在 API 层、thread_id 用 session_id,两者不冲突;补一条并发测试 |
| 冷启动(双模型 + 图编译) | 沿用 lifespan 预热;图编译是纯内存操作,不进请求路径 |
| ReAct 多步的延迟叠加 | 上界 = `max_agent_steps` × `tool_timeout_seconds` = 5×10s;本章**不加**整轮总超时,超限由步数封顶 |

## 10. 本章不做

- 意图识别与指代消解的正式版(用户点名留给下一步)。指代消解本章=透传。
- 上下文管理策略升级、MCP 接入、数据飞轮入库(用户点名留给后面)。
- 置信度闸的正式版(正式的置信度检查留给「可观测」那章);本章只做分数阈值。
- 真实人工客服系统接入(「转人工」是前端模拟)。
- Agent 的并行工具调用优化、跨轮记忆压缩。

## 11. 配置项

新增(`app/config.py`,数值项带下界):

| 配置 | 默认 | 说明 |
|---|---|---|
| `max_agent_steps` | 5 | ReAct 最大步数,超限强制收敛 |
| `agent_token_budget` | 20000 | Agent 累计 token 上限,超限强制收敛 |

复用:`retrieval_score_threshold`(0.58,置信度闸)、`retrieval_top_k`、
`tool_timeout_seconds`、`session_lock_timeout_seconds`。

`requirements.txt` 补 `langgraph==1.2.11`(已装但未登记)。

## 12. 实现订正

(实现过程中与本文的偏离,连同原因记录于此。)

### §5.6/§8.1 之订正:改 `stream_mode="custom"` 单模式,弃用 `messages` 流模式(2026-09-19,写计划期真机验证)

原设计用 `stream_mode=["messages","custom"]`,靠 `metadata["langgraph_node"]` 过滤出
最终回答节点的 token。真机验证(`langgraph 1.2.11`)发现两条硬事实:

1. **`messages` 流模式只对真的 LangChain Runnable 生效** —— 用普通对象替身(本项目
   `tests/test_chat_service.py` 的 `ScriptedModel` 就是这类)调 `astream`,**一条都不流出来**。
2. **langchain 自带的 `GenericFakeChatModel` 虽然能用 `messages` 模式,但 `bind_tools` 抛
   `NotImplementedError`** —— 而 ReAct 节点每轮都要 `bind_tools`。要用它测,就得手写一个
   同时实现 `bind_tools` 与 `_astream` 的完整 `BaseChatModel` 子类。

第 2 条是决定性的:本项目最怕「复杂替身掩盖真实行为」(CLAUDE.md 头号风险是假绿测试),
为测一个流模式而引入复杂替身,方向反了。改为**只走 `custom` 单模式、`agent` 节点自己
`emit` token 帧** —— 所有帧一条发出路径,节点能用普通对象替身直接单测。
对前端的帧协议**一字未变**,技术选型(LangGraph 图 + State + checkpointer)未动。

### §5.3/§5.6 之订正:`get_stream_writer()` 在图外抛 `RuntimeError`,需包一层 emit(2026-09-19,写计划期真机验证)

真机验证:`get_stream_writer()` 在非图运行上下文里抛
`RuntimeError: Called get_config outside of a runnable context`。而节点要能被**直接单测**
(不经图),就必须容忍这个异常。故统一包一个 `app/agent/emit.py:make_emitter()`,
图内发真帧、图外退化为 no-op 收集器(测试断言用)。原设计直接调
`get_stream_writer()`,节点将无法脱离图测试。

**`make_emitter()` 返回的必须是「每次发帧时现取 writer」的函数,不能进去取一次。**
`make_emitter()` 是在**端点里**调的 —— 那时图还没开始跑,上下文里没有 writer,
一次性的取法会永远落到 no-op 分支:**前端一帧都收不到**,而所有单测仍然全绿
(单测把 collector 直接注入节点,根本不经过这条路径)。这是写计划期自检抓到的,
属本项目 CLAUDE.md 点名的头号风险(假绿测试 + 静默故障)。钉它的用例是
`tests/test_agent_graph.py::test_emitter_sends_frames_through_astream_custom_mode`:
在真实的 `graph.astream(stream_mode="custom")` 下断言帧到了消费端、且 collector
**没被碰过**。

### §4.3 之订正:`prepare_turn` 的**产出语义变更**(2026-09-20,T8 落地)

ch01–ch04 的 `prepare_turn` 返回的是**组装好的消息列表**(system + 裁剪后历史 + 本轮 user)。
ch05 起它**只返回裁剪后的历史** —— 消息组装搬进了图里:节点要在**检索之后**往 user
消息里插**证据块**(§5.3),而那一步在端点里做不到。

保留它的理由不变,而且更硬:「**预算校验必须在流开始前完成**」这条约束仍然需要它
(SSE 一旦 yield 过第一帧,响应头就发出去了,状态码再也改不了,溢出只能在此之前变 400)。
签名收窄为 `-> list[Message]`(裁剪后历史),**它是唯一知道「这轮能不能跑」的地方**。

连带:删 `stream_turn`(单轮编排语义已无处可留,改由图的节点承担);
`tests/test_chat_service.py` 的断言从「消息组装正确」改为「裁剪后历史正确」。

### §8.2 之订正:建工单端点必须**自己**捕 `ToolInfrastructureError` 才是 502(2026-09-20,T9 审查探针实测)

§8.2 给的端点骨架只处理了 `if not outcome.ok` 那条 502 分支。T9 审查用探针实测:
**那条分支今天不可达,而真会发生的故障会变成 500。**

- **不可达**:`create_ticket` 必在 registry 里、args 是硬编码合法值、
  `ToolNotFound` 要求 description 为空 —— 三者都不可能;
- **真会发生的**:`create_ticket` 写库时抛 `SQLAlchemyError`
  → `app/tools/executor.py:91` 抛 `ToolInfrastructureError("数据服务暂时不可用")`
  → 端点**没有** `except` ⇒ FastAPI 默认 **500**(修复前探针实测 `{"status": 500}`)。

这与 CLAUDE.md 的错误语义边界相悖(基础设施故障一律 502 + 固定文案;
`ToolInfrastructureError` 必须向上抛、不许被伪装成「查不到」)。
修法:在 `finally` 之前插 `except ToolInfrastructureError` →
`HTTPException(502, redact_api_key(str(exc), settings.openai_api_key))`。

**这是本仓第一个「非流式 + 会跑工具」的端点**,所以那条约定第一次有了真实后果。
钉它的用例:`tests/test_api_ticket.py::test_infrastructure_failure_returns_502_not_500`,
连同失败退出路径**放锁**(漏放 = 该会话永久 409,且 ticket 与 chat 共用同一个
进程级 `_store`,会连带毒掉聊天)与 **409 分支**两条,本文件 2 → 5 条。
