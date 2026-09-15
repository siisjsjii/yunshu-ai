# 电商智能客服系统 · 第 01 章:纯对话 — 设计文档

- 日期:2026-09-15
- 状态:待用户评审
- 范围:仅后端 API。前端聊天页面不在本章范围。

## 1. 目标与验收

本章跑通"纯对话"这条最小链路,不涉及工具调用与 Agent 循环。

验收标准(用户给定,逐字):

1. `curl` 调对话接口能看到流式回复
2. 连续问两轮,第二轮能接住第一轮的上下文
3. 发一段售后描述,能拿到结构化 JSON

## 2. 技术栈与版本

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.13.14 | 已存在的 `.venv` |
| FastAPI | 0.141.1 | 原生 `fastapi.sse`(见 4.3) |
| LangChain | 1.4.0 | 1.x 架构 |
| langchain-openai | 1.6.2 | `ChatOpenAI` |
| pydantic-settings | latest | 读 `.env` |
| tiktoken | 0.14.0 | token 估算,已有 3.13 wheel |
| pytest | latest | 测试运行器 |

版本均已通过 `pip index versions` 与 Context7/源码核实,非凭记忆。

### 2.1 硬约束:`use_responses_api=False`

LangChain 1.x 的 OpenAI provider **默认走 Responses API**。DeepSeek 等 OpenAI 兼容网关只实现 Chat Completions API,不显式关闭会导致调用失败,且报错信息会指向"模型不存在",极难定位。

因此 `app/llm.py` 中构造 `ChatOpenAI` 时必须传 `use_responses_api=False`。此约束需有单测守护(见 7.1)。

## 3. 架构

分层薄封装,**不使用 LCEL,不使用 LangGraph**。

```
app/
  config.py       # pydantic-settings,读 .env
  llm.py          # ChatOpenAI 工厂,收口 use_responses_api=False
  prompts.py      # System Prompt + 模板渲染,Message -> BaseMessage 转换
  schemas.py      # Pydantic 模型
  memory/
    store.py      # 会话表:TTL + LRU + per-session 锁
    trim.py       # token 预算与裁剪
  services/
    chat.py       # 对话编排
    extract.py    # 结构化抽取
  api/
    chat.py       # POST /api/chat/stream
    extract.py    # POST /api/extract
  main.py         # FastAPI app、lifespan、路由挂载
```

依赖方向单向:`api → services → {memory, prompts, llm} → config`。`memory/` 不依赖 LangChain(见 5.3)。

### 3.1 依赖注入约束

`services/` 中的函数**接收 llm 实例作为参数**,不在模块层创建全局单例。FastAPI 侧在 lifespan 中创建实例并通过 dependency 注入。

理由:若在模块层实例化,单测 `chat_service` 就必须 patch 真实网络调用,测试会退化成"测试 mock 本身"。

### 3.2 对话数据流

```
POST /api/chat/stream {session_id?, message}
  ├─ api/chat.py        校验请求
  ├─ memory/store.py    取该 session 历史(持锁)
  ├─ memory/trim.py     按 token 预算裁剪历史
  ├─ prompts.py         渲染 system + 历史 + 本轮输入
  ├─ llm.py             ChatOpenAI.astream() 逐 chunk 产出
  ├─ → ServerSentEvent  逐 token 推送
  └─ 流正常结束后,才把本轮 user + assistant 写回 store(释放锁)
```

**流正常结束后才写历史**:流中途断掉(客户端断开 / 上游超时)时,半截的 assistant 回复不写入历史,否则下一轮会拿着半句话当上下文,产生难以复现的脏数据。

## 4. 接口契约

### 4.1 `POST /api/chat/stream`

请求:

```json
{"session_id": "可选", "message": "我上周买的鞋还没发货"}
```

响应:`text/event-stream`

### 4.2 SSE 事件协议

| event | data | 时机 |
|---|---|---|
| `meta` | `{"session_id": "...", "model": "..."}` | 首帧 |
| `token` | `{"text": "..."}` | 每个 token 一帧 |
| `done` | `{"finish_reason": "...", "usage": {...}}` | 正常结束 |
| `error` | `{"message": "..."}` | 出错(含上游异常) |

`session_id` 省略时由服务端生成,在 `meta` 事件中返回。

`data` 由 FastAPI 自动 JSON 序列化:`ServerSentEvent(data="hello")` 在线上是 `data: "hello"`(带引号)。客户端每帧需 `JSON.parse`。

选用该序列化方式而非手拼 `data:` 字符串,是因为 token 中可能含换行符,手拼会破坏 SSE 帧结构。

### 4.3 SSE 实现方式

使用 FastAPI 原生 `fastapi.sse.EventSourceResponse` 与 `ServerSentEvent`(已读 `fastapi/sse.py` 源码核实 0.141.1 中存在)。

```python
from fastapi.sse import EventSourceResponse, ServerSentEvent

@app.post("/api/chat/stream", response_class=EventSourceResponse)
async def chat_stream(req: ChatRequest) -> AsyncIterable[ServerSentEvent]:
    ...
```

`EventSourceResponse` 自动设置 `text/event-stream`、`cache-control: no-cache`、`x-accel-buffering: no`。

### 4.4 `POST /api/extract`

请求:

```json
{"text": "订单 20240915 的鞋码不对，我想换大一码"}
```

响应:

```json
{
  "order_id": "20240915",
  "request_type": "换货",
  "expected_solution": "换成大一码"
}
```

`request_type` 为枚举,取值:`退货退款` / `换货` / `物流异常` / `发票问题` / `商品咨询` / `投诉` / `其他`。

用枚举而非自由文本,是为了让下游可做统计与路由,也让评估集能计算准确率。

`order_id` 允许为 `null`(用户经常不带订单号),但**不允许模型编造**:抽不到即为 `null`。此约束写入 System Prompt,并在评估集中用诱饵样例验证(见 7.2)。

## 5. 关键设计决策

### 5.1 裁剪策略

每轮请求的 token 组成与处置:

| 部分 | 可否裁 | 说明 |
|---|---|---|
| System Prompt | 否 | 恒定,优先从预算中扣除 |
| 本轮用户输入 | 否 | 裁了即失去意义;放不下直接 400 |
| 历史消息 | 是 | 从最老的**整轮**开始丢弃 |

**按"轮"裁,不按"条"裁。** 一轮 = (user, assistant) 两条。本章无工具调用,不必考虑 tool/assistant 配对;按轮裁保证历史中不出现"有问无答"的孤立消息(会让模型以为上一轮它没回复)。

算法:

```
budget    = CONTEXT_BUDGET_TOKENS - RESERVED_OUTPUT_TOKENS - SAFETY_MARGIN_TOKENS
available = budget - tokens(system_prompt) - tokens(本轮输入)
if available < 0:
    raise ContextOverflowError   → 400
history = 从最新往最老累加整轮,直到超出 available
```

**不强制保留最近 1 轮。** 历史轮数由预算决定,可以为 0 轮。理由:强制保留会打破"请求 token 数有硬上界"这一不变量,而该不变量是防止上游报错的唯一保障。丢历史只是体验降级,超上下文是直接报错。

### 5.2 token 计数

使用 tiktoken `cl100k_base` 估算。

这是**近似**,但对 DeepSeek 偏保守:cl100k 处理中文效率差(1 个中文字符常消耗 1–2 token),而 DeepSeek 自身 tokenizer 中文约 0.6 token/字,因此 tiktoken 会**高估** token 数、更早触发裁剪。估算偏差的方向落在安全一侧。

仍保留 `SAFETY_MARGIN_TOKENS` 以兜住更换模型时的反向偏差。

计数收口为单一函数 `count_tokens(text: str) -> int`,`memory/trim.py` 中定义。更换为精确 tokenizer 时只需改动此一处。

### 5.3 会话存储

存储**纯数据**,不存 LangChain 的 `BaseMessage`:

```python
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class SessionStore:
    _sessions: dict[str, list[Message]]
    _locks:    dict[str, asyncio.Lock]
```

`memory/` 层因此完全不依赖 LangChain —— 单测它既不需要安装 LangChain,也不需要 mock 任何东西。转换为 `BaseMessage` 是 `prompts.py` 的职责。

**并发**:同一 session 的并发请求会互相覆盖历史(两个请求都读到相同历史,各自追加,后写覆盖先写)。为每个 session 配一把 `asyncio.Lock`,同 session 串行化。代价是第二个请求需等待第一个流式结束 —— 客服场景下一个用户一个会话,该代价可接受。锁获取超时返回 `409`,避免卡死的流永久锁死会话。

**回收**:惰性 TTL(默认 30 分钟未活跃)+ `max_sessions` LRU 上限(默认 1000)。

**不做后台清理任务**:LRU 上限已给出内存硬上界,后台任务只改变"何时释放",不改变"是否释放",本章不值得为此引入生命周期管理。

### 5.4 结构化输出

使用 `with_structured_output(ExtractResult, method="function_calling")`。

选择 `function_calling` 而非 `json_schema`,因为 DeepSeek 明确支持 function calling,而 strict `json_schema` 的支持情况未经确认。**这是本文档唯一的待实测项**,详见第 9 节。

抽取使用**独立的 ChatOpenAI 实例**,`temperature=0`;对话实例 `temperature=0.7`。两者均由 `app/llm.py` 的同一工厂产出,避免密钥与 base_url 配置散落。

抽取失败(模型输出不符合 schema)直接返回 `422`,不返回部分结果,不做自动重试 —— 重试次数应由评估数据决定,不应凭感觉设定。

## 6. 错误处理

| 场景 | 行为 |
|---|---|
| 上游 401 / 403 | SSE `error` 事件,提示密钥配置问题,**不回显 key 内容** |
| 上游超时 / 限流 | SSE `error` 事件,已推送的 token 保留 |
| 客户端中途断开 | 丢弃本轮,不写历史,不记错误 |
| `session_id` 不存在 | 静默新建,`meta` 事件返回实际生效的 session_id |
| 单轮输入超预算 | `400`,附带实际 token 数与预算 |
| 抽取 schema 不符 | `422` |
| 同 session 并发 | 等待锁;超时返回 `409` |

`session_id` 不存在时静默新建而非 404:因为 `meta` 会回传实际 ID,客户端能自行发现不一致 —— 既不让首次调用必须分两步,又不掩盖客户端 bug。

## 7. 测试与验收

三类产出走三条不同的验证路径。

### 7.1 可单测代码 → TDD

| 模块 | 测什么 |
|---|---|
| `memory/store.py` | TTL 过期、LRU 淘汰、锁串行、同 session 并发不丢消息 |
| `memory/trim.py` | 预算计算、按整轮裁剪、0 轮边界、超预算判定、`count_tokens` 高估行为 |
| `prompts.py` | 渲染结果快照(历史为空 / 历史多轮两种) |
| `llm.py` | 工厂产出的实例 `use_responses_api is False`(守护 2.1 的硬约束) |
| `api/chat.py` | `TestClient` 打接口,注入假 llm,断言 SSE 事件序列 |
| `api/extract.py` | `TestClient` 打接口,注入假 llm,断言 200 / 422 分支 |

单测全程不联网。

### 7.2 纯 Prompt → 评估集

`evals/extract_cases.jsonl`,约 12 条售后描述,每条标注三个字段的期望值。覆盖:

- 订单号:明确给出 / 口语化变体("单号是…")/ 完全没给(期望 `null`)
- 诉求类型:三类各若干(退货退款 / 换货 / 物流异常)
- 诱饵:文本中出现数字但并非订单号(如"买了 2 双"、"9 月 15 号下单")

`evals/run_extract_eval.py` 逐条调用真实模型,**按字段分别计算准确率**(非整体对/错),输出表格。

分字段计算的原因:`order_id` 抽错与 `expected_solution` 表述不佳,严重程度完全不同,混在一起会掩盖问题。

规模说明:12 条只能说明"大致能跑",无法给出置信区间。需要更硬结论时应扩到 30 条以上。

### 7.3 端到端 → 验收命令

`scripts/acceptance.sh`,逐条对应第 1 节的验收标准。

验收 1(流式):

```bash
curl -N -X POST localhost:8000/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"你好，我想咨询退货"}'
# 期望:meta → token 逐帧滚动 → done
```

验收 2(两轮上下文,同一 session_id):

```bash
curl -N -X POST localhost:8000/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"s1","message":"我的订单 20240915 还没发货"}'

curl -N -X POST localhost:8000/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"s1","message":"我刚才说的订单号是多少？"}'
# 期望:第二轮回复中出现 20240915
```

第二条刻意问的是**上一轮说过的信息**,而非"你还记得吗"。模型无法靠猜,必须真的拿到历史。该断言不易假阳性。

验收 3(结构化):

```bash
curl -X POST localhost:8000/api/extract \
  -H 'Content-Type: application/json' \
  -d '{"text":"订单 20240915 的鞋码不对，我想换大一码"}'
# 期望:{"order_id":"20240915","request_type":"换货","expected_solution":"..."}
```

### 7.4 运行命令

```bash
.venv/Scripts/python.exe -m pytest -q                 # 单测,不联网
.venv/Scripts/python.exe evals/run_extract_eval.py    # 评估集,需真实 key
bash scripts/acceptance.sh                            # 端到端,需真实 key
```

## 8. 本章不做

- 工具调用、Agent 循环
- 前端聊天页面
- Redis / 数据库持久化(会话存进程内存)
- 抽取结果驱动任何路由(仅作为字段返回)
- 抽取失败的自动重试
- System Prompt 的红线检查评估集
- 多进程 / 多 worker 部署(进程内存储不跨 worker 共享)

## 9. 待实测项与风险

| 项 | 说明 | 处置 |
|---|---|---|
| `with_structured_output` 的 method | `function_calling` vs `json_schema` 在 DeepSeek 上的实际表现 | 实现时两种各试一次,以实测为准并回写本文档 |
| tiktoken 对 DeepSeek 的偏差方向 | 假设为高估(安全侧),未在真实模型上验证 | 首次评估集运行时记录实际 `usage.prompt_tokens` 与估算值对比 |
| 锁超时阈值 | 默认 60s 是估计值。太小会让正常的长回复产生假 409,太大则失去"防卡死"的意义 | 先用 60s;拿到真实流式耗时分布后再调整 |

## 10. 配置项(.env)

**必填,无默认值**:

```
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=
OPENAI_MODEL=deepseek-chat
```

`OPENAI_MODEL` **刻意不设默认值**。若默认成 `deepseek-chat`,用户换模型(如换 Ollama)却忘改配置时,应用会拿错误的模型名去请求,报错指向"模型不存在" —— 与 2.1 节 `use_responses_api` 属同类难以定位的故障。必填则启动即报 `OPENAI_MODEL is required`,一眼可辨。

**可选,均有默认值**:

```
CHAT_TEMPERATURE=0.7
EXTRACT_TEMPERATURE=0.0
CONTEXT_BUDGET_TOKENS=8192
RESERVED_OUTPUT_TOKENS=1024
SAFETY_MARGIN_TOKENS=512
SESSION_TTL_SECONDS=1800
MAX_SESSIONS=1000
SESSION_LOCK_TIMEOUT_SECONDS=60
```

`.env.example` 提交入库,`.env` 不入库。

配置显式传入 `ChatOpenAI`,不依赖 langchain-openai 的环境变量自动读取 —— 避免与其它工具的 `OPENAI_*` 变量冲突,行为更可预测。
