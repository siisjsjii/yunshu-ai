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
  main.py         # FastAPI app、路由挂载
```

依赖方向单向:`api → services → {memory, prompts, llm} → config`。`memory/` 不依赖 LangChain(见 5.3)。

### 3.1 依赖注入约束

`services/` 中的函数**接收 llm 实例作为参数**,不在模块层创建全局单例。FastAPI 侧通过 dependency 注入。

**实现订正**:本节原写"在 lifespan 中创建实例"。实际交付**没有 lifespan** —— `app/api/chat.py` 用 `Depends` 里的工厂**每请求构造** `ChatOpenAI`。这不违反本节的目的(可测性由"services 接收实例作为参数"满足,测试可用 `dependency_overrides` 替换),但成本与本节的原始设想不同:

- `langchain_openai` 对底层 httpx 客户端做了 `lru_cache`(键为 `base_url`/`timeout`/`socket_options`),故并未每请求重建连接池;
- 但 pydantic 校验、同步与异步两个根客户端仍是每请求构造。

若 ch02 关注这部分开销,改成 lifespan 单例即可,不必动 `services/` 一层。

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

**字段约束**(实现后补):`session_id` 为 `None` 或 1–128 字符的字符串。空字符串**非法**(返回 422),不再被静默当作"新建会话" —— 否则客户端传 `""` 会得到一个与预期不符的新会话且毫无提示。上限 128 同时也限制了它作为 `_sessions`/`_locks`/`_touched` 字典键的长度(见第 5.3 节关于 `_locks` 无硬数量上限的说明)。

`data` 由 FastAPI 自动 JSON 序列化:`ServerSentEvent(data="hello")` 在线上是 `data: "hello"`(带引号)。客户端每帧需 `JSON.parse`。

选用该序列化方式而非手拼 `data:` 字符串,是因为 token 中可能含换行符,手拼会破坏 SSE 帧结构。

### 4.3 SSE 实现方式

使用 FastAPI 原生 `fastapi.sse.EventSourceResponse` 与 `format_sse_event`(已读 `fastapi/sse.py` 源码核实 0.141.1 中存在)。**不使用路由层的隐式编码。**

**实现订正** —— 本节原写"用 `response_class=EventSourceResponse` + `yield ServerSentEvent`"。实际交付**不是这个形状**,原因是本设计第 6 节的一条硬需求:

> 端点必须在流开始**之前**返回 400(超长输入)与 409(会话占用)。

而一旦 SSE 生成器 yield 过第一帧,响应头就已发出,状态码再也改不了。因此端点被写成**普通 `async def`**,在函数体里完成取锁与预算校验,然后返回一个显式构造的 `EventSourceResponse`;只有通过校验的生成器才开始产帧。帧由 `format_sse_event(event=..., data_str=json.dumps(payload, ensure_ascii=False))` 手工组装(`data_str` 是**已序列化**的字符串,不是对象),直接产 bytes。

代价与收益:放弃了路由层的自动编码,换来了"校验失败能返回真实 HTTP 状态码"这一能力。另需自行设置 `Cache-Control: no-cache` 与 `X-Accel-Buffering: no`(路由层不再代劳)。

注:原设计里"token 含换行符会破坏 SSE 帧结构"的担忧在本实现下**不成立** —— `json.dumps` 会把换行转义成 `\n`,`format_sse_event` 再做一次分行,故多行 token 不会破坏帧。

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

**但该上界只约束了输入侧** —— 最终 code review 指出:`reserved_output_tokens` 只是从上下文预算里**扣掉**一个数字,**没有任何东西真的限制了模型的输出长度**(未传 `max_tokens`)。所以这个不变量应表述为"**输入 token 有硬上界**",而非"请求 token 有硬上界"。ch01 的实际风险很低(真实上下文窗口远大于 8192),但原表述强于代码。ch02 要么真的传输出上限,要么保持这个更弱的准确表述。

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

**内存上界的确切构成** —— 实现后经 code review 核实,此处原先写的"LRU 上限已给出内存硬上界"是**不准确的**,订正如下:

| 结构 | 上界 | 由谁约束 |
|---|---|---|
| 会话历史 `_sessions` | 硬上界 `max_sessions` 条 | LRU 淘汰 |
| 锁与时间戳 `_locks` / `_touched` | **无硬数量上限** | 约「TTL 时间窗内的不同 session_id 到达数」 |
| 被持锁的孤儿条目 | 永不清扫 | 并发在途流数量 |

`_locks` / `_touched` 不受 `max_sessions` 约束的原因:`lock_for` 会为**任何** session_id 创建条目,而端点在校验失败时(例如超长输入返回 400)该 session 不会进入历史表,条目遂成孤儿。孤儿由 `_purge` 在"超过 TTL **且**锁未被持有"时清扫,故其上界是到达速率 × TTL,而非常数。

**本章接受此风险**:端点在本章本就没有任何认证,内存耗尽只是诸多 DoS 向量之一,为此给锁对象再加第二套淘汰策略属过度工程。**若 ch02 引入认证,应同时给这两个 dict 加数量上限。**

**不做后台清理任务**:后台任务只改变"何时释放",不改变"是否释放",本章不值得为此引入生命周期管理。注意它**不是**内存上界的来源。

### 5.4 结构化输出

使用 `with_structured_output(ExtractResult, method="json_mode")`。

**这一条经实测后已从 `function_calling` 改为 `json_mode`**,原因是原选择在本项目实际使用的模型上完全不可用(见第 9 节第 1 行的实测记录)。`json_mode` 走 `response_format={"type": "json_object"}`,仍然由 `with_structured_output` 提供 schema 校验与解析 —— **"用 with_structured_output 实现"这一要求未变**,变更的只是其内部 `method` 参数,而该参数在本设计中被预先标注为待实测项。

`json_mode` 有一个端点强制的附加要求:**提示词中必须出现 "json" 字样**,且需自行描述字段结构。因此 `EXTRACT_SYSTEM_PROMPT` 中写明了三个字段的 JSON 结构。

**注意**:`EXTRACT_PROMPT` 是 `ChatPromptTemplate`,默认按 f-string 解析,**提示词中的字面花括号会被当成模板变量并报错**。提示词描述结构时不得使用裸 `{` / `}`(需转义为 `{{` `}}`,或改用无花括号的描述方式)。

抽取使用**独立的 ChatOpenAI 实例**,`temperature=0`;对话实例 `temperature=0.7`。两者均由 `app/llm.py` 的同一工厂产出,避免密钥与 base_url 配置散落。

抽取失败(模型输出不符合 schema)直接返回 `422`,不返回部分结果,不做自动重试 —— 重试次数应由评估数据决定,不应凭感觉设定。

## 6. 错误处理

| 场景 | 行为 |
|---|---|
| 上游 401 / 403 | SSE `error` 事件,提示密钥配置问题,**不回显 key 内容** |
| 上游超时 / 限流 | SSE `error` 事件,已推送的 token 保留 |
| 客户端中途断开 | 丢弃本轮,不写历史,不记错误 |
| `session_id` 不存在 | 静默新建,`meta` 事件返回实际生效的 session_id |
| `session_id` 为空串 / 超 128 字符 | `422`(不再静默当作新建) |
| 单轮输入超预算 | `400`,附带实际 token 数与预算 |
| 抽取 schema 不符 | `422` |
| **`/api/extract` 的上游故障**(401 / 超时 / 限流 / 连接失败) | **`502`**,返回固定文案「抽取服务暂时不可用」;原始异常只进服务端日志 |
| 同 session 并发 | 等待锁;超时返回 `409` |

**关于"不回显 key"—— 实现后补正。** 本节最初只写了要求,没写实现方式,而实现确实缺失了一段:所有出口错误文本此前都是 `str(exc)` 原样透出,而 OpenAI SDK 异常的 `str()` 形如 `"Error code: {status} - {response_body}"`,即**上游响应体原文** —— 认证失败时可能含掩码后的 key 片段。

现已收口为一个统一的脱敏函数,作用于**两条**出口:对话端的 SSE `error` 帧,以及抽取端的 `HTTPException` detail(422 与 502 两处)。`/api/extract` 的上游故障不再回显原始异常文本,只回固定文案,原始异常进日志。

**注意 `422` 的语义边界**:它**只**表示"模型输出无法解析为约定结构"。上游故障一律 `502`,不再伪装成"用户输入不符合 schema"。这条边界是最终审查发现的缺陷,此前实现把所有异常都判成 422。

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

**两个字段的评分方式不同,这是刻意的**:

- `order_id` / `request_type` 是**闭式**的,用精确相等评分。
- `expected_solution` 是**自由文本**,精确相等没有意义 —— 首次运行时它报出 0/12,而输出实际语义正确。改为**关键词命中**(标注里为每条用例给出若干个正确摘要必然包含的短子串,去除空白与常见标点后判包含),并**同时打印精确匹配作为对照基线**,以免掩盖真实情况。

该指标的实际强度**弱于设计意图**:交付的关键词集合是 1–2 个、且多为单个词,部分弱到只剩单个字(如 `退`、`质量`、`具体`、`核实`)。这不是笔误,而是上面第 1 条(反复放宽)的必然结果。

**但 `expected_solution` 的评分在本章不具备可信度,不应作为准确率结论引用。** 原因有三,均由实现过程实测暴露:

1. 关键词集合是**观察到模型输出的措辞之后才放宽的**(首轮 9/11,两处未命中语义均正确)。这是对样本的拟合,不是对能力的度量。
2. 放宽后有若干组关键词弱到只剩单个字(如 `退`、`质量`、`具体`),几乎任何相关摘要都能命中。
3. 分母是 11 而非 12 —— 有一条用例的标注与模型输出之间不存在有区分度的共同子串,该条被排除(分母已在输出中打印)。

**结论:`order_id` 与 `request_type` 的 100% 是可信的(闭式、跨多次运行稳定);`expected_solution` 的分数仅作指示,真实质量需人工审阅这 12 条输出,或引入带自身校验的 LLM 评判。** ch02 若需要该字段的可信度量,应改为 LLM-as-judge 并对评判者本身做一致性校验。

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
| ~~`with_structured_output` 的 method~~ | **已实测,已解决。** 在本项目可用端点(仅有 `deepseek-flash` 与 `deepseek-v4-pro`,均为 thinking 模型)上:`function_calling` → 400 `Thinking mode does not support this tool_choice`;`json_schema` → 400 `This response_format type is unavailable now`。**两者全部不可用,换模型也解决不了。** `json_mode` 可用,但端点强制要求提示词含 "json" 字样。 | 已改为 `json_mode` 并回写第 5.4 节。 |
| ~~tiktoken 对 DeepSeek 的偏差方向~~ | **已实测:假设成立。** `count_tokens` 对 DeepSeek 实际输入 token **高估约 1.32–1.34 倍**,偏差落在安全一侧(更早触发裁剪,不会超出上下文)。 | 无需处置。第 5.2 节的安全论证得到实测支持。 |
| `expected_solution` 的评分方式 | 自由文本字段无法用精确字符串相等评分 —— 首次评估集因此报出 0/12,而输出实际语义正确("将鞋子换成大一码" vs 标注"换成大一码")。改为关键词命中后得 11/11,**但该分数不可信** —— 关键词是看到输出措辞后才放宽的,属样本拟合。 | 已在 §7.2 记录其不可引用性。ch02 若需该字段的可信度量,应改用带一致性校验的 LLM-as-judge。 |
| `deepseek-flash` 的自由文本非确定性 | 即使 `temperature=0`,同一输入在 4 次运行中产出明显不同的自由文本措辞。`order_id` / `request_type` 这类闭式字段不受影响(始终 12/12)。 | 接受。这意味着任何针对 `expected_solution` 措辞的回归测试都会脆弱 —— 不要为自由文本字段写字符串断言。 |
| 锁超时阈值 | 默认 60s 是估计值。太小会让正常的长回复产生假 409,太大则失去"防卡死"的意义 | 先用 60s;拿到真实流式耗时分布后再调整 |
| 锁获取的取消竞态 | `asyncio.wait_for(lock.acquire(), timeout)` 存在极窄窗口:锁已获取,但 `wait_for` 抛出取消(3.12+ 的 `wait_for` 在当前 task 内运行 `acquire`,故取消可能落在 future 已决议、端点尚未恢复之间)。此时该 session 的锁无人释放,而**持锁会话不被 TTL 或 LRU 淘汰**,故泄漏是**永久**的(该 session_id 之后每次请求都 409) | **ch01 未关闭,已接受并记录。** 实现时试过 guard flag 与 `except BaseException` 两种形状,均无法关闭。**但最终审查纠正了本文档原先"无法用合理形状关闭"的说法** —— 该说法过强:给 `SessionStore` 的锁记录**获取它的 task**,在 409 处理里比对 `lock.owner is asyncio.current_task()` 后释放,约 10 行即可关闭,且不改变正常路径。ch02 若需要,这就是该用的形状 |

## 10. 配置项(.env)

**必填,无默认值**:

```
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=
OPENAI_MODEL=deepseek-flash
```

(上面的模型名是本项目实际使用的值。写文档时曾假设为 `deepseek-chat` —— 实测发现该账号下用的是 `deepseek-flash`。这正是 `OPENAI_MODEL` 设为必填、不给默认值的原因:默认值会拿一个可能不存在的模型名去请求,报错指向"模型不存在",而真实原因是配置。) 

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
