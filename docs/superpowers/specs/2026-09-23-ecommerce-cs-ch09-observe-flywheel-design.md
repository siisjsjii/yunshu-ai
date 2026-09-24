# ch09 设计:给客服系统装上眼睛,再把答不上的问题变成改进燃料

| | |
|---|---|
| 日期 | 2026-09-23 |
| 分支 | `ch09-observe-flywheel`(待建) |
| 状态 | **设计定稿**(2026-09-23 用户批准);实现与订正见 §15 |
| 前置 | ch01–ch08 全部已合并 `main`,`main` = `b2a25b9` |
| 权威性 | 本文是本章的**设计源**。代码与本文冲突时以本文为准;实现期的偏离一律记进 §15,不改历史章节 |

---

## §1 目标与非目标

### 目标

把「客服系统答不上来的问题」从**丢掉的流量**变成**知识库的增量**,并让整条链路第一次**可被看见**。

1. **可观测**:接 Langfuse,每请求一条完整 trace(每个节点的 prompt、工具调用、检索结果、token 与耗时),界面可铺开看。
2. **成本归因**:意图识别结果进 trace 元数据,能按意图维度统计 token 花销。
3. **数据飞轮闭环**:三个入口把问题送进 ch04 建的 `low_confidence_questions` 池,**各自标好入池入口**,落池时一并存下**当轮召回片段快照**。
4. **飞轮流水线**:问题标准化 → 查重归并 → 进 `review_queue` 待审 → 人工审核 → 通过的写进知识库。
5. **自动化评估流水线**:复用 ch04 的评估集与指标定期跑,每轮落 `eval_runs`,按时间连成趋势。
6. **前端配套**:待审队列后台管理页(列表 + 通过/驳回 + 详情看用户原话与召回片段)。

### 非目标(明确不做)

- **低置信度问题按主题归类的微调分类器** —— 用户点名「下一步的事」(2026-09-23 原话)。
- **Faithfulness 之类的生成段 LLM-as-judge 指标** —— 用户 2026-09-23 明确「faithfulness 应该指的是置信度池之类的」,即需求文本里那半个词指的是**置信度兜底机制**,不是一个评分指标。`eval_runs` 只落检索段指标(§10)。
- **不改 ch08 的工具系统、确认流、MCP 接入** —— `agent` 节点本章**一行不动**(§5.6)。
- **不做跨会话长期记忆、用户画像**(ch07 起就不做,本章不改变)。

---

## §2 设计依据(实测事实,不是推断)

本章有三个决策完全由实测决定,先摆事实。**所有探针脚本留在 `.superpowers/`(未跟踪),结论抄在这里。**

### 2.1 Langfuse:用 Cloud,不是自部署(用户 2026-09-23 拍板)

需求的技术栈一栏写的是「开源、自部署,链路数据不出自家服务器」,但**用户 `.env` 里已经配好的是**:

```
LANGFUSE_SECRET_KEY=…              (非空)
LANGFUSE_PUBLIC_KEY=…              (非空)
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

**这是 Langfuse Cloud 美国区,不是自部署。** 两条冲突的事实都摆给了用户,用户 2026-09-23 拍板:**就用云端**。

- 服务端版本实测:`GET /api/public/ready` → `4.41.0`;同一天晚些时候再测是 **`4.42.0`**
  ⇒ **Cloud 会自己升级,服务端版本不是固定值**。引用版本号时要带日期。
- SDK 取 PyPI `latest` = **`langfuse==4.15.4`**(`requires_python: >=3.10`)。
- **装它零漂移**:`pip install --dry-run langfuse==4.15.4` 实测只新增 11 个包
  (`backoff`、`googleapis-common-protos`、`opentelemetry-{api,sdk,proto,semantic-conventions,
  exporter-otlp-proto-common,exporter-otlp-proto-http}`、`wrapt`),**不升级任何现有依赖** ——
  `httpx==0.28.1` / `pydantic==2.13.5` 一字不动。§13-10 那条风险随之解除。
- **代价如实记**:对话原文、召回片段、用户问题会出境到 Langfuse Cloud,与需求里写的「链路数据不出自家服务器」不符。换成自部署只需改 `LANGFUSE_BASE_URL` 一个值(§11)。
- **`.env.example` 补上三个键**(现在没有),并把 `LANGFUSE_BASE_URL` 的默认值写成 Cloud 地址 + 一行注释说明可改自部署。

### 2.2 ⚠️ `response_format` 与 `bind_tools` **不能同时用** —— 四条路逐条实测

**结论先说**:本章**不用 `response_format`**,`useful` 协议**纯提示词驱动 + §5.3 的增量解析器**。这一节是那个结论的依据,也是它为什么**没有硬保证**的依据(§5.7-1)。

探针 `.superpowers/probe_ch09_json.py`,模型 `deepseek-flash`、`base=https://api.deepseek.com/v1`。

| 组合 | 实测结果 |
|---|---|
| `response_format={"type":"json_object"}` + **不绑 tools** | ✅ 通。28–41 个碎片,**原始 JSON 文本增量**,字段顺序就是提示词里要求的 `useful, confidence, answer` |
| `bind_tools([t]).bind(response_format=…)`,非 strict 工具 | ❌ **客户端** `ValueError: \`get_weather\` is not strict. Only \`strict\` function tools can be auto-parsed` |
| `bind_tools([t], strict=True).bind(response_format=…)`,作答提示词 | ⚠️ 通(74 个碎片,JSON 完整) |
| `bind_tools([t], strict=True).bind(response_format=…)`,要求调工具 | ❌ 网关 400:`Prompt must contain the word 'json' in some form to use 'response_format' of type 'json_object'` |

**根因**(读 `.venv/Lib/site-packages/langchain_openai/chat_models/base.py:2085-2097`):

```python
if "response_format" in payload:
    payload.pop("stream")
    response_stream = self.root_async_client.beta.chat.completions.stream(**payload)
```

只要 payload 里有 `response_format`,langchain-openai 1.6.2 就**改走 openai SDK 的 beta 解析路径**,并顺手把 `stream` 弹掉;而那条路径 `_validate_input_tools` **只接受 strict 工具**。**没有开关能绕开。**

**因此 `strict=True` 是唯一的硬组合方式,而它被否掉了**:strict 会改写**发给网关的每一个工具 schema**;ch08 的工具系统里有**两个 MCP Server 原样透传的 `inputSchema`**(对方写的、我们没有编辑权),它们未必满足 strict 的结构要求(顶层 `additionalProperties: false`、全部字段进 `required` 等)。为了一个字段去动整个工具系统的序列化形状,风险与收益不成比例。

> ⚠️ **上表第四行那条 400 曾被我记成「strict 路径调工具不通」。那是错的** —— 400 的成因是**我那条提示词里没有 `JSON` 字样**,不是 strict 与工具冲突。真结论是**未验**,不是「不通」。这条错误连同它的成因留在 §15.1,因为它正是本仓那条「先写结论后没跑」的老毛病。

### 2.3 硬约束:提示词里**必须出现字面 `JSON` 字样**

2.2 第四行那个 400 是网关的原话。**与 `app/services/extract.py` 那条 `json_mode` 硬约束同源**,本章第二次撞上:

- `KNOWLEDGE_ANSWER_SYSTEM_PROMPT` 里必须**逐字出现 `JSON`**(提示词自己会说明「只输出一个 JSON 对象」,天然满足,但**要有一条测试钉它** —— 别人删掉那句话时,红在单测而不是红在线上)。
- 同一处还有 ch01 的老坑:**描述结构时不得使用裸花括号**(`ChatPromptTemplate` 按 f-string 解析)。

### 2.4 `KNOWLEDGE` 出口**只来自「商品咨询」** —— 这是 §5 落池范围的前提

`app/agent/routing.py:25-36` 的 `INTENT_TO_ROUTE` 是全局唯一的意图→出口映射表:

```
商品咨询 → KNOWLEDGE      退款退货 → REFUND        物流 → BUSINESS
订单     → BUSINESS       售后     → REFUND        投诉 → COMPLAINT
闲聊     → CHITCHAT       其他     → FALLBACK
```

**订单 / 物流 / 退款 / 售后全部不走 `KNOWLEDGE`。**

> ⚠️ **这条曾被我用错了地方。** 我据它论证「知识路径不需要工具,所以可以不绑 tools」——
> 而用户 2026-09-23 后半段**否掉了那个论证**:「知识路径作答还是要绑定 tools……多路检索
> 之后进主力 agent 可能还是要调工具」。用户的例子(退款售后)其实走 `REFUND` 子流程、
> 落在 `KNOWLEDGE` 之外,**但方向是对的:这张表只说明「走哪个出口」,不说明「那个出口
> 需不需要工具」**。一个出口要不要工具,该由那句需求与实现者的判断定,不该由这张
> 映射表**倒推**。
>
> 结论:这张表的**真正**用处是 §5.5 的**落池范围**(只有 `KNOWLEDGE` 那一类问题
> 才该进知识池),**不是**「要不要绑 tools」论据。`agent` 保留 `bind_tools`(§5.1)。

实测证据的边界也如实说:`log/app.log` 里 7 条 `chat_turn` **全是 ch08 验收留下的物流轮**,
**没有任何一条商品咨询轮**,所以「知识类今天会不会真的调工具」**仍然没有证据** ——
这也正是采纳用户方向的原因:**没有证据支持去掉一个能力时,就不去掉。**

### 2.5 现状盘点:低置信度池**只写不读**

- 唯一写入点 `app/kb/assess.py:63` 的 `record_low_confidence`,唯一生产调用方 `app/agent/nodes.py:214` 的置信度闸。
- **全仓 `app/` 下 0 个 `select(LowConfidenceQuestion)`**,无端点、无 UI、无脚本读它。
- **无去重、无节流**:闸每失败一次落一行,同一问题反复问反复落。
- `entry_point` 生产取值**只有一个字面量 `"置信度闸"`**;DDL 注释里列过的 `自评不足` 在今天**没有生产写入方**(§2.6)。

### 2.6 ⚠️ 需求里「生成阶段自评」这条链**在生产上不存在**

需求原文:「生成阶段模型自评知识不够答:**ch04 已经在落池**,确认接进飞轮即可」。

**这半句是错的。** `app/kb/assess.py` 的 `assess_sufficiency`(ch04 那套「生成后自评」)**自 ch05 起生产调用方为零** —— 它自己的 docstring 就写着这件事:

> **当前不在请求路径上**(ch05 spec §50):ch05 起「召回够不够」改由 `app/agent/nodes.py` 的**置信度闸在事前**判定,ch04 这套「生成后再自评」被整段替换。

今天在线的判据只有一行(`app/agent/nodes.py:210-211`):

```python
scores = [e["score"] for e in (state.get("evidence") or [])]
passed = bool(scores) and max(scores) >= settings.retrieval_score_threshold
```

⇒ **「生成阶段自评」是本章要新建的东西,不是「确认接上」**。§5 全章都在做这件事。用户 2026-09-23 拍板了它的形态(`useful` 字段 + 与回答同一次调用 + 提示词约束拒答)。

### 2.7 现状盘点:每轮的 `intent` 与 `evidence` **都没有持久化**

`log_turn`(`app/agent/nodes.py:679`)只做三件事:写日志行、发 `trace` 帧(端点**折进 `done`、不外推**,前端不读)、`append_turn` 落 messages。**`intent` 与 `evidence` 出不了这一次请求。**

⇒ 用户需求里「👎 时从**该会话当轮**的检索结果里尽力回捞」**没有现成来源**。§6 给了解法与其语义偏差的记账。

### 2.8 `Faithfulness` 全仓不存在(需求措辞订正)

`scripts/run_eval.py` 的指标是 `recall@5 / recall@10 / mrr / conf / answer_ok` 五个,**没有 Faithfulness**;ch04 spec 提过它但没实现。用户 2026-09-23 明确它指的是置信度池那套 ⇒ **§10 只落检索段指标**。

---

## §3 可观测(Langfuse)

### 3.1 `app/observability.py` —— 全章**唯一**的 Langfuse 边界

沿用本仓既有的边界写法(`app/sanitize.py` 之于密钥、`app/tools/errors.py` 之于错误词汇表):

```python
def enabled(settings) -> bool                       # 三个 LANGFUSE_* 缺任一 ⇒ False
def handler(settings)                       -> Any  # 关掉时返回 None
def trace_metadata(*, conversation_id, intent, confidence) -> dict
@contextmanager
def span(name, *, as_type="span", input=None)       # 关掉时 yield 一个空壳
```

**纪律(三条,都要有测试):**

1. **`app/` 下除本模块外,零处 import langfuse。** 别的模块只认 `observability.span(...)` 这一个形状 ⇒ 将来换观测后端,改一处。
2. **关掉时全部 no-op,且不 import langfuse。** `langfuse` 的 import 放在函数体内 —— 顶层 import 会让「没装/没配」的环境**在导入期**就炸。
3. **本模块不抛异常。** 观测挂掉绝不许影响业务 —— 与 `app/tools/audit.py` 的 `record_audit`「永不抛」同族。`span` 的 `__exit__` 吞掉一切异常并 `logger.warning`。

### 3.2 挂点:**一处**

`app/api/chat.py:453` 的

```python
async for mode, chunk in graph.astream(
    stream_input,
    config={"configurable": {"thread_id": session_id}},
    stream_mode=["custom", "updates"],
):
```

改成 `config={**config, "callbacks": [...], "metadata": {...}}`。**图的编译处(`graph.py:225`)不动** —— `compile(checkpointer=...)` 不传 callbacks,回调跟着**调用**走,这样每请求的 `session_id` / 意图才能进 trace,而不是被编译期冻死。

### 3.3 ⚠️ callback **只覆盖 LangChain 的 run** —— 工具与检索一个 span 都不会自动出现

这是本章最容易做错的一处。Langfuse 的 `CallbackHandler` 挂在 **LangChain 的 Runnable** 上,**自动**拿到的是:

- ✅ 模型调用(intent 分类、resolve、agent 的每一轮、退款判定)—— prompt、token、耗时、模型名
- ✅ 走 LangChain 的 retriever(本项目**没有**:`KnowledgeRetriever` 是自写的普通类)

**拿不到**的是:

- ❌ **工具执行** —— 走 `app/tools/executor.py` 的 `execute_tool`,不是 LangChain tool run
- ❌ **知识检索** —— 走 `app/retrieval/search.py` 的 `KnowledgeRetriever.search`

⇒ **手工挂两个 `span`**,这是「每个节点的工具调用、检索结果都能铺开看」这条需求的**唯一**落地方式:

| 位置 | span 名 | as_type | input | output |
|---|---|---|---|---|
| `execute_tool` 的调用点(`nodes.py:372`、`confirm_nodes.py` 的决议节点) | `tool:<name>` | `tool` | `args` | `{ok, summary, error_kind}` |
| `retrieve_knowledge` 节点(`nodes.py:164`) | `retrieval` | `retriever` | `query` | 召回片段清单(id / score / section_path) |

**放行的是「渲染后的证据」而不是「裸 answer」** —— 与 `journal.model_ctx` 的口径一致(那里数的是 `render_evidence(evidence)`)。

### 3.4 意图 → trace tag(实测后重写,见 §15.3)

**三段实测事实**(探针 `.superpowers/probe_ch09_langfuse2.py`):

1. **`LangfuseSpan.update(**kwargs)` 的 kwargs 被静默丢弃** —— 源码 docstring 逐字:
   `**kwargs: Additional keyword arguments (ignored)`。⇒
   `root.update(**{"langfuse.trace.tags": [...]})` **什么都不做、也不报错**。
   (**本仓「静默无效」家族的新成员**,§15.3 记账。)
2. **`propagate_attributes(tags=[...])` 有效,且「中途进入」也有效**:包住整段的观测拿到 tag;
   **中途 `__enter__` 之后新建的观测拿到 tag,之前的没有**。实测:
   - 整段包裹 → `tags: ['intent:AAA','ch09']`,3 条观测
   - 中途进入 → `tags: ['intent:BBB']`,**2 条**(进入前那次调用不在内)
3. **`_AgnosticContextManager` 没有 `__aenter__`** ⇒ 中途进入只能用**同步** `__enter__`。

**做法(定稿)**:

```python
with propagate_attributes(trace_name="cs-chat", session_id=conversation_id):
    cm = None
    async for mode, chunk in graph.astream(...):      # 端点现有的循环
        ...
        if cm is None and <这一批 updates 里出现了 classify_intent>:
            cm = propagate_attributes(tags=[f"ch09", f"intent:{state}"])
            cm.__enter__()                            # 同步,不是 __aenter__
        ...
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)
```

- **session_id 走外层**:`session_id` 在 `astream` 之前就已知,外层包裹能覆盖**全部**观测
  (实测 A/B 两条 trace 的 `sessionId` 都落对了)。
- **intent 走中途**:意图要 `classify_intent` 跑完才知道,**不可能**在 `astream` 之前拿到。
  中途进入意味着**意图分类那一次调用本身不带 tag**;这可以接受 —— 开销的大头在它**之后**
  的 agent 轮次,而验收 5 问的正是「钱花在哪类**问题**上」。
- ⚠️ **待冒烟**:上面这段在**真实图**里能不能让 tag 落到 graph 内部的 generation 上
  **尚未验证** —— 探针里 `model.ainvoke` 是直接调的,真实链路上模型调用发生在
  `graph.astream` **内部**(LangGraph 可能自带 OTel 上下文)。**这是 T1 冒烟项。**
  不行的话退路是:把意图写进**我们自己开的观测**(`retriever` / `tool:*` 两个手动 span 的
  `update(metadata=...)`),`intent_cost.py` 改从这些观测自己聚合 —— 代价是脚本变重。

> **不用 `config["metadata"]["langfuse_tags"]`** 那条路吗?它**实测也有效**
> (C 组:`tags: ['intent:CCC']`)。但 `config` 在 `astream` **之前**就固定了,
> 而意图在那之后才知道 —— 这条路解决不了「事后才知道」这件事。留着它给**已知**的标签用。

### 3.5 按意图的 token 花销(实测后订正)

`scripts/intent_cost.py`:打 Langfuse **Metrics API v2**,按 `tags` 维度分组,打印
「意图 / 观测数 / token 总量」表。**请求形状是实测出来的,不是照文档拼的**:

```python
query = {
    "view": "observations",                        # v2 只支持 observations / scores-*
    "metrics": [{"measure": "totalTokens", "aggregation": "sum"},
                {"measure": "count", "aggregation": "count"},
                {"measure": "totalCost", "aggregation": "sum"}],   # ← 见下面第 2 条,漏了它 cost 那句是断言
    "dimensions": [{"field": "tags"}],             # ← 按 tag 分组,实测可用
    "filters": [],
    "fromTimestamp": …,  "toTimestamp": …,         # ← **两个都必填**
    "config": {"row_limit": 50},
}
resp = await http.get(f"{base}/api/public/v2/metrics",
                      params={"query": json.dumps(query)},   # ← 整个 query 是一个字符串参数
                      auth=(pk, sk))
```

**五条实测出来的坑**,逐条记(第 2、4、5 条是 T4 的实现者对着**真实网关**补的,见 §15.7):

1. **`query` 必须是 JSON 字符串参数**,把 `view`/`metrics` 平铺成独立 query 参数会 400
   (`Invalid input: expected string, received undefined` 指向 `query`)。
2. **`metrics` 里必须显式请求 `totalCost`** —— 不请求的话 `sum_totalCost` 这个键
   **根本不在响应行里**,`r.get("sum_totalCost")` 恒为 `None` ⇒ 那句「cost 恒为 0」
   就退化成**断言**而不是**读数**,而脚本里 `total_cost > 0` 那条分支是**死代码**。
   (实测:键名就是 `sum_totalCost`,**值是数字 `0`,不是字符串**。)
3. **`sum_totalCost` 恒为 0** —— 模型 `deepseek-flash` 没在 Langfuse 里配价格。
   ⇒ **统计表报 token,不报钱**;脚本对 cost 字段**只在非零时才打印**,并且在文档里写清
   「为什么是 0」。**不许把 0 当成本报出去。**
4. **按 `tags` 过滤时三个字段都有约束,缺一个就 400**(本章的分组统计不需要 filter,
   但脚本的 `--intent` 可选参数会用到)。网关原话逐字:
   - **`operator` 必须是 `"any of"` / `"none of"` / `"all of"`** ——
     **`"contains"` 不是 arrayOptions 的合法算子**(那是 `"string"` 那一组的,该组的合法值是
     `"="` / `"contains"` / `"does not contain"` / `"starts with"` / `"ends with"` / `"is not empty"`,
     与 arrayOptions 那组**不重叠**)。给 `"contains"` 回:
     `Invalid option: expected one of "any of"|"none of"|"all of"`。
   - **`type` 必须是 `"arrayOptions"`**(不是 `"string"`):`Filter type 'string' is not
     supported for dimension type 'string[]'. Expected 'arrayOptions'`。
     而且**这个键不能省** —— 少给时回
     `{"code":"invalid_union","note":"No matching discriminator","discriminator":"type"}`。
   - **`value` 必须是数组**:给字符串回 `expected array, received string`。
5. **`dimensions` 被清空时,响应行里没有 `tags` 键** —— 那一版回来的是**一行总聚合**
   (实测 `{"sum_totalTokens": "4850", "count_count": "26"}`)。
   ⇒ 拿 `r.get("tags") or []` 去挑 intent 行的写法会**恒判 False**,于是脚本
   **永远打印「没有找到任何 intent 观测」,而且不报任何错**。过滤其实已经在**服务端**
   做完了,**按 tag 挑行的分支在 `--intent` 这条路上根本不该跑**。
   同理:**过滤命中 0 条时网关回的是一行 `0/0`**(`{"sum_totalTokens": "0", "count_count": "0"}`),
   **不是空数组** ⇒ 还要滤掉 count 为 0 的空桶,否则表里会多一行「最烧 token 的意图:XXX(0 tokens)」。

其他实测事实:

- **`GET /api/public/v2/observations`**(v1 的 `/traces` 已弃用)按 `traceId` +
  `fromStartTime` / `toStartTime` 读回,**行里的 `tags` 字段是 `None`** ——
  单体观测读不回 tag,**只有 Metrics 聚合看得见**。所以「验证 tag 落没落」只能靠 Metrics。
- **ingestion 有延迟**:首次读回 0 条、隔一会儿再读就有了。⇒ 脚本与验收**一律轮询**,
  不许读一次就断言。
- 凭证从 `.env` 读,走 `httpx.AsyncClient`(**不引新依赖**);出站错误文本过 `redact_api_key`。
  ⚠️ **传的是这个脚本自己的凭据 `LANGFUSE_SECRET_KEY`,不是 `openai_api_key`** ——
  本仓**其余**那些调用传的都是 `openai_api_key`,那是因为它们处理的是**上游 openai SDK** 的异常文本;
  这条规矩的**目的**是「出站文本不许回显凭据」,所以按**碰的是哪把密钥**来传。
  > ⚠️ **订正(2026-09-25,T19)**:这句话原写「本仓那 **17** 处调用」,**17 这个数没有数过** ——
  > 评审在 T4 时用 AST 数出来是 **11**(chat.py 6 / extract.py 2 / refund.py 1 /
  > `kb/orchestrate.py` 1 / `memory/tasks.py` 1)。**11 也是当时那一刻的读数**:那之后
  > T15 加了 `api/review.py` 1 处、T14 加了 `flywheel/tasks.py` 1 处 ⇒ 本章结束时 `app/` 下
  > 共 **12** 处传 openai 那把(plus `scripts/intent_cost.py` 的 **2** 处传 **Langfuse** 那把,
  > 那正是本条规矩的应用)。原话里「17」保留在此处不删,便于对账;
  > **要引用计数就现数一遍**(判据:`redact_api_key(...)` 的第二次实参提到 openai)。
  覆盖面:**每一个**会把响应体或异常字符串打出去的地方(`连不上 Langfuse` 那条、
  以及 400 分支里回显 `resp.text` 的那一段)。**不要**因此引入 `OPENAI_API_KEY` 的依赖。
- 命令行的两条流**都要钉编码**:`emit()` 走 stdout,而 `SystemExit` / traceback 走 **stderr**
  —— 只钉 stdout 时,那几条中文报错在 cp936 管道上会输出成乱码(**不崩,但一个字都读不出来**)。
- **不写进任何端点** —— 它是「拿出统计」的交付物,不是在线功能。
- 验收 5 就断它:输出里必须**至少两个不同意图的行**,且能看出哪个 token 最多。

### 3.6 单测「全程不联网」怎么守

`observability.enabled()` 在 `LANGFUSE_*` 缺任一或值为空时返回 `False`;测试的 `Settings(...)` 一律 `_env_file=None`(本仓硬约束)⇒ 只要测试不显式传三个键,整套观测就是 no-op,**一个字节都不会出网**。**要有一条测试钉这个**:`enabled()` 为假时 `handler()` 是 `None`、`span()` 里跑代码不 import langfuse。

---

## §4 事前置信度闸(飞轮入口 ①)

### 4.1 判据:从「取最高分」换成 `evidence_confidence`

新模块 `app/kb/evidence.py`(**纯函数,不依赖 LangChain**,与 `app/kb/assess.py` 同层):

```python
def evidence_confidence(chunks: Sequence[RetrievedChunk]) -> float:
    """把「检索+精排的结果」压成一个 0–1 的置信度。

    空证据、或全部块都低于 `evidence_min_score`,返回 0.0(同一个出口)。
    """
```

三个信号(**三个一律只在那批「分数 ≥ `evidence_min_score`」的块上算**,见 §15.8;
用户 2026-09-23 点名):

| 信号 | 取法 | 为什么要它 |
|---|---|---|
| 精排 Top1 的相关性分 | 过滤后的 `max(c.score)` | 主判据。`RetrievedChunk.score` **已经是重排 sigmoid 分**(不是 RRF、不是余弦) |
| 有效证据数 | 分数 ≥ `evidence_min_score` 的条数,封顶 `evidence_max_count` | 单条高分可能是巧合;**够多条中高分**才叫「知识库覆盖了」 |
| Top1 与 Top2 的分差 | 过滤后的 `top1 - top2`(只剩一条时按 `top1 - 0`) | 分差大 ⇒ 那条明确对口;分差小 ⇒ 几条都差不多,**可能都不对口** |

**全被过滤掉时返回全 0**(`top1` / `top2` / `count` / `gap` / `confidence` 五个键全 0),
与「空证据」走**同一个出口**。理由见 §15.8:**不是证据的块,不该影响任何一个信号**;
而且这是闸的 **fail-closed 一侧** —— 一堆纯噪声的块必须被拦。

**合成式**(权重与形状在 §11 里是配置项,便于标定时调):

```
confidence = w_top1 * top1
           + w_count * min(有效条数 / evidence_max_count, 1.0)
           + w_gap   * clamp(top1 - top2, 0, 1)
```

权重初值 `w_top1=0.6, w_count=0.2, w_gap=0.2`(和为 1 ⇒ 输出天然落在 0–1)。**初值是拍的,阈值是标出来的** —— 这两件事必须分开说,§4.2 只负责后者。

### 4.2 阈值 = **标定出来的**,不拍脑袋(用户点名)

标定脚本 `scripts/calibrate_evidence.py`:

- **评估集用 `evals/测试集.md`(300 条 / 5 桶)**,不用 `evals/retrieval_cases.jsonl`(23 条)。理由是**它自带 `D_absent` 应拒答桶 60 条** —— 没有负例就标不出「拦得对不对」,而 23 条那份只有 4 条干扰项,ch03 已记过它「不构成压力」。
- 对每条用例跑**真实链路**(`retriever.search` → `evidence_confidence`),扫阈值网格,输出混淆表:
  - **应拒答桶(D_absent)的拦截率** —— 越高越好
  - **正常桶(A/B/C/E)的误杀率** —— 越低越好
- **取折中点**:在「拦截率 ≥ 0.6」的阈值集合里,挑**误杀率最低**的那个;若无解,退化为「误杀率 ≤ 0.05 里拦截率最高」并**响亮记账**。
  > ⚠️ **T8 已结清(2026-09-23):这条"取折中点"在本次评估集上落不下来。** 实测
  > 300 条里没有一条的置信度落在 `(0, 0.2894)` 内 ⇒ 规则挑出的其实是**一段平台**,
  > 平台内任何取值等价;"折中点"**不是一条仍然有效的指令**,`0.2` 是平台内的
  > **一次判断**。详见 **§15.9**(别在这里就地改口径 —— 本节是设计期原文,
  > 改的是 §15.9 与 `app/config.py`)。
- 脚本打印全表 + 选中的那个,人工确认后写回 `app/config.py`(**标定值属设计授权内的调参**,按工作要求第 4 条记账并告知可一句话回退)。
- 标定用的 `D_absent` 是**闭式判据**(「应拒答=是」这一列),不是主观打分。

### 4.3 闸的**位置不动**

`retrieve_knowledge → confidence_gate → (agent | fallback_reply)` 的拓扑一字不改,只换判据:

```python
# app/agent/nodes.py, make_confidence_gate_node
conf = evidence_confidence(state.get("evidence") or [])
passed = conf >= settings.evidence_confidence_threshold
```

`reject_reason` 从「最高分 X 低于阈值 Y」换成**三个信号都写出来**的形态(审核人要看得懂):

```
置信度 0.31 低于阈值 0.42(打分:top1=0.29 条数=1 分差=0.04)
```

**不通过时不变**:`record_low_confidence(entry_point="置信度闸", …)` + 走 `fallback_reply`。

> `retrieval_score_threshold`(0.25)在本章之后**仍是检索器内部的过滤阈值**(`build_retriever` 用它筛块),它在**闸**里的第二个用处被本次替换掉 —— §11 记这条。

---

## §5 生成阶段自评(飞轮入口 ②)

### 5.1 图拓扑**一字不改**,协议加在 `agent` 节点的知识轮上

```
retrieve_knowledge → confidence_gate ─通过→ agent ─→ log_turn
                                     └不通过→ fallback_reply ─→ log_turn
```

- **`agent` 保留 `bind_tools`,工具可用性零损失。**(用户 2026-09-23 后半段拍板:
  「知识路径作答还是要绑定 tools……多路检索之后进主力 agent 可能还是要调工具」。)
- `confidence_gate` 的出边不动,`_OUTLETS` 不动,图**一个字都不改**。
- 只在 `make_agent_node` **组装 `msgs` 时**按 `state["intent"]` 追加一条**协议消息**,
  并让**文本流出**走一次解码(§5.4)。

**协议只加在知识轮(`intent == "商品咨询"`)。** 理由:

1. 需求说的是「**知识**不够答」;落池也只该发生在知识类(§5.5)。
2. **业务 / 退款 / 闲聊三条路径的输出形状与今天逐字节相同** ⇒ ch08 的写确认流
   (挂起 / 续跑 / `turn_messages` 覆写 / `pending_write` 每轮清零)**零风险** ——
   这是上一版方案(新节点)想要的收益,现在用一个 `if` 就拿到了。

**协议消息是追加的第三条 system 消息,不进 `render_system_prompt`。**
后者是 `budget.derive(...)` 的入参之一,动它就会连带改**预算推导**,并让一批
既有的预算/分层测试跟着动 —— 收益为零、风险不小。

### 5.2 一次调用,三个字段,**顺序是协议的一部分**

```python
class KnowledgeAnswer(BaseModel):        # 只作**协议文档**,不作解析器
    useful: bool        # 证据是否足以回答
    confidence: float   # 0–1,自评置信度
    answer: str         # useful 为 false 时必须是空串
```

⚠️ **不用 `response_format`,也不用 `with_structured_output`** —— §2.2 实测:

- 本节点的轮次**绑着 tools**(§5.1),而 `response_format` 与 `bind_tools` 在
  langchain-openai 1.6.2 上**互斥**(非 strict 工具直接客户端 `ValueError`)。
- 唯一的硬组合方式是 `strict=True`,而它**会改写每一个工具的 schema**,
  含两个 MCP Server 原样透传的 `inputSchema`。**否。**

⇒ 协议**纯提示词驱动**,`KnowledgeAnswer` 这个 Pydantic 类**只用来写文档与测试**,
**不进运行时**。解析由 §5.3 的增量状态机负责。

**协议消息必须做到三件事:**

1. 逐字出现 **`JSON`** 字样(§2.3 的网关硬约束;`response_format` 虽然不发了,
   但这句约束在本仓是**通行纪律**,而且协议文本本来就要说「JSON 对象」)。
2. **顺序必须是 `useful` → `confidence` → `answer`**,并说明理由(「先判定,后作答」)。
   **顺序不是风格,是协议** —— §5.4 的「`useful=false` 时一个 token 都不放出去」
   完全靠它成立。
3. **证据不足以回答时,`useful` 必须为 `false` 且 `answer` 必须为空字符串**;
   不得编造、不得用常识补、**不得在前言里先给答案再说 useful=false**。
4. **需要调工具时照常调**;不需要工具时**只输出那一个 JSON 对象,不要任何前言、不要 ``` 围栏**。

### 5.3 `app/agent/json_stream.py` —— 增量解析器(**自写**,理由见下)

**为什么自写而不是用 `partial-json-parser` / ijson / json-stream**(用户 2026-09-23 提名了这类库,并授权按推荐走):

本章只需要两件事 —— ①**从头部取出 `useful`**;②**把 `answer` 的值边收边吐**。而 `answer` 的值**必然含中文**,必然出现 `\"` `\\` `\n` `\uXXXX` —— **转义序列跨 chunk 边界**恰恰是这类通用库最容易出错、也最难验的地方(一个 `中` 被切成 `\u4e` + `2d` 是**必现**的,不是边角)。

自写的东西是一台**纯状态机**:输入 `str` 片段序列,输出事件序列,**零 IO、零依赖、完全确定性** ⇒ 可以用 chunk 边界 fuzz **穷举钉死**。通用库我们只能信它。

**接口**:

```python
@dataclass(frozen=True)
class Event:
    kind: str            # "useful" | "confidence" | "answer_delta" | "done" | "violation"
    value: object = None

class JsonAnswerDecoder:
    """吃 token 片段,吐事件。**纯函数式**:没有 await、没有 IO、没有随机。"""
    def feed(self, fragment: str) -> list[Event]: ...
    @property
    def mode(self) -> str: ...            # "lead" | "protocol" | "plain" —— 见 §5.4
    @property
    def useful(self) -> bool | None: ...
    @property
    def confidence(self) -> float | None: ...
    @property
    def answer(self) -> str: ...          # 已解出的 answer 全文(供落库)
    @property
    def done(self) -> bool: ...
    @property
    def raw(self) -> str: ...             # 收到过的**全部**原文(降级路径要用)
```

解码器自带一个**三态**状态机:**`lead`(还没定形态)→ `protocol`(在解协议对象)
/ `plain`(不是协议对象)**。三态的**退出条件与行为**在 §5.4;这里只强调一件事 ——
**`plain` 态是「今天的行为」的原样保留**,它存在的意义是让协议违规**不产生任何回归**。

**必须钉死的边界(每条一个用例)**:

| # | 情形 | 期望 |
|---|---|---|
| 1 | **转义跨 chunk**:`"a\u4e` / `2d b"` | `answer` == `"a中 b"` |
| 2 | **反斜杠跨 chunk**:`"a\` / `"b"` | `answer` == `'a"b'` |
| 3 | `answer` 值里含 `{` `}` `,` `:` | 不被当成结构字符 |
| 4 | `answer` 是**最后一个键**,收尾 `"` 与 `}` 分在两个 chunk | `done` 在收到 `}` 后才为真 |
| 5 | **小数跨 chunk**:`"confidence": 0.` / `85` | `confidence == 0.85` |
| 6 | `useful` 的值跨 chunk:`tru` / `e` | `useful is True` |
| 7 | **首键不是 `useful`**(先出现 `answer`) | `protocol` 态下 `violation("first_key=answer")`,**且一个 token 都没吐过** |
| 8 | **前导空白后第一个非空白字符不是 `{`** | `mode` 变 `plain`(§5.4),**不是 violation** |
| 9 | 流提前结束(没有 `}`) | 收尾时 `done is False` ⇒ 调用方按「不完整」处理 |
| 10 | `useful=false` 且 `answer=""` | `useful=False` 后立刻 `done`,**零个 `answer_delta`** |
| 11 | `useful=false` **但 `answer` 非空**(违规流) | 解出 `useful=False` 的**那一刻**停,**后续 answer 一个 delta 都不再吐** |
| 12 | 模型加 ```json 围栏 | 前缀 `` ```json `` 的首个非空白是 `` ` `` ⇒ **落 `plain`**(§5.4)。**这是刻意的**:围栏意味着模型没守协议,按今天的纯文本行为处理并把 JSON 原样显示,比猜着剥壳**更可预测、更好记账** |
| 13 | 前导空白(片段是 `""` / `"  "` / `"\n"`) | 留在 `lead` 态,不判定 |

**语义约束**:`feed` 是**追加**语义(片段按到达顺序喂),不是替换;`answer_delta` 事件里的 `value` 是**本次新增**的那段文本,不是累计。`feed("")` 必须是**空事件表**,不许因为空片段而误判 `plain`。

### 5.4 三态状态机:一轮文本怎么变成 token 帧

`_stream_round` 今天做的事是 `if chunk.text: emit({"frame":"token", ...})`。
知识轮改成喂解码器,**业务轮一个字不改**。

```
lead(缓冲中,还没定形态)
 ├ 首个**非空白**字符是 `{`        → protocol
 ├ 首个非空白字符不是 `{`          → plain   ← 把缓冲与后续**全部实时 emit**
 └ 片段全是空白 / 空串             → 留在 lead

protocol(在解协议对象)
 ├ 解出 useful=true   → 先 flush 缓冲里 `{` 之前的字节(若有),此后每个
 │                      answer_delta **立刻** emit 一个 token 帧
 ├ 解出 useful=false  → **立刻停止消费**、丢弃 remainder、emit 兜底话术的 token 帧
 │                      (§5.5 决定要不要落池)
 ├ 首键不是 useful    → **中止本轮 → 重试一次**(此时零 emit,见 §5.6)
 └ 流结束仍没 useful  → 与 plain 同路:把全部原文当普通答复 emit + 记账

plain → 每个 chunk.text **立刻** emit(↔ 今天的行为,**零回归**)
```

**两条不变量,都要有测试:**

1. **`lead` 态下什么都不 emit。** 用户此刻看不到任何字。
2. **`useful=false` 之后,一个 `answer_delta` 都不许再出去。** 协议第 3 条
   (`answer` 必须为空串)是这条的**正常路径**保证,而 §5.3 的边界 11
   (`useful=false` 但 `answer` 非空)是它的**违规路径**保证 —— 两条都要测。

**「`useful` 之前不 emit」是协议顺序的直接推论**:`answer` 排在第三位,
所以解出 `useful` 之前**不可能**有 `answer` 的字节。

`confidence` **只记录,不做第二个阈值**(用户只点名了 `useful` 一个判据):
它进 state、进 `log_turn` 的日志行、进 `trace` 帧。

### 5.5 落池范围:`useful=false` 一律兜底,但**只有知识类落池**

| 情形 | 用户看到 | 落池? |
|---|---|---|
| `useful=false`,且 `intent == "商品咨询"` | 兜底话术 | ✅ `entry_point="生成自评"` + 存召回片段快照 |
| `useful=false`,其它意图 | 兜底话术 | ❌ **不落** |

**为什么业务类不落池**:池子是**知识缺口**的池子,它的下游是「标准化 → 审核 →
写进知识库」。一笔查不到的物流单**不是知识缺口**,写进知识库只会污染它。
业务类 `useful=false` 只记日志行。

### 5.6 协议违规:**一律降级成「今天的行为」**(早期那处「重试一次」已取消 —— 见表下)

| 违规形态 | 处理 | 为什么 |
|---|---|---|
| `protocol` 态下**首键不是 `useful`** | **中止本轮 ⇒ 降级**(此时**零 emit**,用户完全看不见);`trace` 记 `agent:protocol_violation` | 早期设计在这里写的是「重试一次」,**已取消**(2026-09-23,契约固化之后)—— 理由与代价见**下面那段** |
| `lead` 态首个非空白不是 `{`(含 ```json 围栏、含前言) | **不重试**,直接 `plain` | 它常常意味着「模型先说了句『让我查一下』再调工具」—— 那是**正常行为不是违规**;重试会把这种最常见的形态也打成一轮额外的模型调用 |
| 流结束仍没 `useful` | 不重试,按 `plain` 收尾 + `logger.warning`;`trace` **记不记 `agent:protocol_violation` 看这一轮有没有 `tool_calls`**(有 ⇒ **不记**,没有 ⇒ **记**) | 同上;判据见**下面那一段** |

> ⚠️ **订正(2026-09-23,控制器裁定):「只在一处重试」—— 那一处取消了,本节现在是「一律降级」。**
>
> **最初写的是什么**:本节第一行原来是「**中止本轮,重试一次**(零 emit,用户完全看不见);
> 再违规 ⇒ 降级」。那是**写本节时**设计的 —— 那时调用方契约还不存在。
>
> **实际是什么**:T9 用**六轮**把调用方契约固化成**三行表**(task-9-report.md §11-③),
> 而表里**第 2 行 = 降级,没有重试**。T10 落地的是表。
>
> **为什么取消**:①**第 2 行的降级路径本身是安全的** —— 它退回 ch08 的纯文本行为,
> 不会让任何一条既有路径变差;②重试要在一条**频率未测**的路径上**再加一次模型调用**
> (本章从头到尾没有量过首键违规有多常见),而 §12.3-1 要量的正是这类频率 ——
> 用一个没量过的动作去换一个本来就安全的退路,收益说不清。
>
> **取消的代价如实记**:首键违规那一支会把 **`raw`(即协议 JSON 原文)当纯文本显示给
> 用户** —— 丑,但可预测,而且 `trace` 里有 `agent:protocol_violation`、日志里有告警。
> **这条是用户可见的质量取舍**,已写进 §13 的风险表(用户日后若要求「更好看」,
> 首选修法是把重试加回来)。
>
> **归因(按 §15 体例)**:**重试是设计方写下的,却没让它在契约里活下来** ——
> 契约表被更狠地验过(六轮 + 七个变异),而重试那条从未实现、也从未被测。
> 记在 §15.11。
>
> ⚠️ **订正(2026-09-23,T10 实现时发现):本节原先自相矛盾,`trace` 记不记以 `tool_calls` 为准。**
>
> 本节原先写「降级 ⇒ `trace` 记 `agent:protocol_violation`」,而**同一张表**上一行又说
> `lead`→`plain`「常常意味着『模型先说了句让我查一下再调工具』—— 那是**正常行为不是违规**」。
> **两句话都出自本节,而只有一句能活。**
>
> 定语是:`agent:protocol_violation` 的**唯一用途**是量「模型不守协议的比例」
> (§12.3-1 要的就是那个数)。按字面在**每一个** `plain` 轮上打标,那么
> **每一次调工具的正常轮都会被记成违规** —— 这个数就**废了**。
> 真违规只有一种:**这一轮本该作答,却没有 JSON**(即这一轮**没有** `tool_calls`,
> 却吐了散文或 ``` 围栏)。
>
> ⇒ **判据是「这一轮有没有 `tool_calls`」,不是「是不是 `plain`」。**
> 帧**不重发**这条不受影响:`raw` 已经逐片透出去了,再补一遍就是重说一次。
>
> **本处的自相矛盾是实现者发现的**(T10),不是设计期就写清的 —— 按 §15 体例记账。

**降级 = 今天的行为**,这是本章的**兜底不变量**:协议怎么坏,最差也就是回到
ch08 的纯文本流,**不会让任何一条既有路径变差**。`trace` 里的标记供验收与事后排查。

**为什么降级是 fail-open**(与闸的 fail-closed 相反):闸拦下的是「**还没生成**的回答」,
代价是用户再看一次兜底话术;这里面对的是「**已经生成完、只是包装不合协议**的回答」,
按 `useful=false` 处理等于**把一段可能完全正确的回答扔掉**并落一条**假的池记录**。
两处的代价不对称,所以方向相反。**这条方向差异写死在本节**,否则后来人会
「为了一致性」把它改反。

**不做围栏剥离。** 上一版曾打算「取第一个 `{` 到最后一个 `}`」。**去掉了**:
剥壳意味着我们要猜模型的意图,而剥错了的后果是把一段残缺 JSON 当答案推给用户;
`plain` 降级把 JSON 原样显示**丑但可预测**,且 `trace` 里有标记、日志里有告警 ——
对一个**本就不该发生**的形态,可预测比好看重要。

### 5.7 ⚠️ 代价与已知取舍(如实记账,不许读成「没损失」)

1. **协议没有硬保证。** 因为它与 `bind_tools` 互斥(§2.2),只能靠提示词。
   ⇒ 模型不守协议时,降级成纯文本(§5.6),**`useful` 那一轮就丢了**。
   **这是本章最可能被事后认为「不稳」的一处。**
   - 兜底是结构性的:降级路径 = 今天的 ch08 行为,**不会更差**;`trace` 里有标记,
     违约率是可观测的(Langfuse 上按 `agent:protocol_violation` 一搜就有)。
     注意这个数的**口径**(§5.6 的订正段):它数的是「**本该作答却没有 JSON**」,
     不含那些「先说一句『让我查一下』再调工具」的正常轮 —— 否则它就没有意义。
2. **知识轮多一段协议消息的 token 开销**(约 150 token/轮)。它**只进 `msgs`,
   不进 `render_system_prompt`** ⇒ **不影响 `budget.derive` 的推导**,也不动
   既有的预算/分层测试(`prompt 预算`那套是照 system prompt 算的)。
3. **`usage` 统计不受影响。** 上一版那条「`response_format` 会让 langchain 走 beta
   路径、`stream_options.include_usage` 可能丢」的风险**随方案一起消失了** ——
   现在走的是**普通 `astream` 路径**,与 ch05–ch08 每一轮完全相同。
4. **知识类问题若在某一轮里既想调工具、又想作答**:协议的写法是「需要工具就照常调」,
   模型会先发 `tool_calls`(那一轮没有文本或只有前言),工具执行完在**下一轮**作答。
   这与今天的 ReAct 行为一致。

---

## §6 用户反馈落池(飞轮入口 ③)

### 6.1 `POST /api/feedback`

```
请求:{ "conversation_id": str, "question": str, "message_id": int?, "value": "up" | "down" }
响应:200 {"ok": true}
```

- `value="up"` ⇒ **只记录,不落池**。
- `value="down"` ⇒ 落池(`entry_point="用户反馈"`)+ 尽力回捞召回片段(§6.2)。
- **幂等**:同一 `message_id` 的 `down` 重复提交只落一行(靠 `low_confidence_questions` 上加不了唯一键 ⇒ 在服务层查一次「该会话 + 该问题 + 入口=用户反馈」是否已有行,有就跳过)。

> 为什么带 `question`:需求说「用户点了就把**该轮的用户问题**落池」。前端每轮都渲染了那条用户消息,自己就拿着这个值 —— 不必让后端从 `messages` 里反查(反查要面对「一条问题被问两次」的歧义)。

### 6.2 ⚠️ 召回片段:后端**重跑一次检索**尽力回捞,语义偏差记账

用户需求原文:「👎 这种事后落池的,**从该会话当轮的检索结果里尽力回捞**,那一轮真没走检索(闲聊、业务数据类)才空着」。

**事实**:§2.7 —— 每轮的 `intent` 与 `evidence` **都没有持久化**,「那一轮的检索结果」**不存在于任何地方**。

**两条路,以及为什么选后者**:

| 方案 | 代价 |
|---|---|
| 新增一张「每轮上下文快照」表(会话 id + 用户消息 id + intent + 召回片段 JSON),当轮写入 | 多一张表、多一次当轮写;但语义与需求**严格一致** |
| **后端按该轮问题重跑一次 `KnowledgeRetriever.search`**(选中) | 不加表;但「那一轮真没走检索才空着」是靠「**重跑也没召回**」**近似**出来的,不是真的知道那一轮的意图 |

**选后者的第二条理由(不只是省事)**:重跑对**审核人**更好用。审核页要回答的是「知识库是真缺这块,还是有但没检到」—— 重跑**召得到** ⇒ 有但当时没检到;重跑**召不到** ⇒ 真缺。而「那一轮当时的快照」在这件事上给的信息**更少**(它只证明当时没检到)。

**如实记两条偏差**:① 停用阈值过滤后重跑,闲聊类问题**理论上**也可能召回低分块(实践中被 `retrieval_score_threshold` 挡掉,但这是**性质**不是保证);② 重跑用的是**当前**的知识库,而那一轮用的是**当时**的 —— 审核通过后重跑的同一问题,快照会变。

### 6.3 前端

`app/static/index.html` 的满意度反馈**目前是纯前端采集**(点了只在本地变个样式)。本章把它接到后端:点击时带 `conversation_id` / 该轮用户问题 / 该条 assistant 消息 id POST 上去。**纯 UI 部分按项目规矩用 Vibe Coding 直做,不套 TDD。**

---

## §7 数据

### 7.1 `db/ch09.sql`

⚠️ **`init_db.py` 永不加列**(本仓硬约束)。所以**老库升级必须手工执行这份 DDL**;`ALTER TABLE` **不幂等是刻意的**(静默跳过会让「表已存在但形状不对」永远补不上),与 `db/ch03~ch08.sql` 同款。

```sql
SET NAMES utf8mb4;

-- ① 池子加两列
ALTER TABLE low_confidence_questions
  ADD COLUMN evidence_snapshot JSON          NULL COMMENT '落池当轮的召回片段快照(Top-N 的 id/得分/原文)',
  ADD COLUMN matched_review_id BIGINT UNSIGNED NULL COMMENT '归并到的 review_queue.id;NULL = 尚未进流水线',
  ADD KEY idx_matched_review (matched_review_id);

-- ② 待审队列
CREATE TABLE review_queue (
  id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  standard_question   VARCHAR(512)    NOT NULL COMMENT '标准化后的问题(FAQ 式)',
  example_answer      TEXT            NOT NULL COMMENT '模型给的示例答案(未核准)',
  occurrences         INT             NOT NULL DEFAULT 1 COMMENT '归并进来的问题条数',
  status              VARCHAR(16)     NOT NULL DEFAULT 'pending' COMMENT 'pending|approved|rejected',
  approved_answer     TEXT            NULL COMMENT '人工核准后的答案(通过时必填)',
  first_raw_question  TEXT            NOT NULL COMMENT '第一条用户原话(详情页展示)',
  source_conversation_id VARCHAR(32)  NULL COMMENT '首个来源会话',
  created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  reviewed_at         DATETIME        NULL,
  PRIMARY KEY (id),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='低置信度问题待审队列(ch09)';

-- ③ 评估轮次
CREATE TABLE eval_runs (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  trigger_by  VARCHAR(32)     NOT NULL COMMENT '触发方式:manual|scheduled',
  case_count  INT             NOT NULL COMMENT '本轮评估集条数',
  metrics     JSON            NOT NULL COMMENT '各指标分数(按策略/按桶)',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='评估流水线轮次(ch09)';
```

**两条刻意的设计选择**:

- **`review_queue` 上没有「标准化问题」的唯一键。** 查重是**语义判断**(模型判是不是同一个意思),而唯一键只能管**字面全等** —— 两者不是同一条规则,加了唯一键会在一次合理的语义归并上**响亮地 1062**。
- **`matched_review_id` 一个列担两个语义**:既记「这条问题归并到了哪一行」,又是流水线的**待处理标记**(`WHERE matched_review_id IS NULL`)。⇒ 流水线天然幂等,重跑不会重复归并。这个双语义在 ORM 的 docstring 里要写出来。

⚠️ **上面 ALTER 里的 `ADD KEY idx_matched_review` 是 T5 修复轮补的**(2026-09-23,评审裁定):流水线的选择谓词 `WHERE matched_review_id IS NULL ORDER BY id LIMIT n` 就是它**唯一的热路径**,而原版 DDL 不建索引、ORM 侧却声明了 `index=True` —— 「ORM 有 / DDL 没有」这个方向不该留。实况库在**同一次修复里用同一条语句、同一个索引名**补过,所以实况库 ≡ 这份文件。

⚠️ **这份 DDL 是给「已有 `low_confidence_questions`、但缺这两列」的库升级用的**;在**全新空库**上它是错的工具(实测:先跑它 ALTER 报 1146;先跑 `init_db.py` 则 create_all 会连同两列一起建,再跑它 1060 + 1050)。全新库只跑 `init_db.py`,形状差异见 §7.3。

### 7.2 `entry_point` 的三个取值

| 取值 | 产生处 | 状态 |
|---|---|---|
| `置信度闸` | `confidence_gate`(§4.3) | **沿用旧值**,不重命名(ch04/ch05 的测试与验收都断它) |
| `生成自评` | `agent` 节点的知识轮(`useful=false` 且 `intent=商品咨询`,§5.5) | 新增 |
| `用户反馈` | `POST /api/feedback`(§6.1) | 新增 |

列宽 `VARCHAR(32)`,三个取值都放得下(MySQL 的 VARCHAR 长度按**字符**算)。

`db/ch04.sql` 里 `entry_point` 的列注释写着 `检索为空|自评不足|低分拒绝` —— 那三个值**今天一个都不是生产取值**(生产只有「置信度闸」,见 §2.5)。**不回去改 ch04 的 DDL**(不改历史章节的产物),在 §15 记账。

### 7.3 ORM 侧

`app/db/models.py`:

- `LowConfidenceQuestion` 加 `evidence_snapshot: Mapped[dict | None] = mapped_column(JSON)` 与 `matched_review_id: Mapped[int | None] = mapped_column(BigInteger, index=True)`。
- 新 `ReviewQueue`、`EvalRun` 两个模型。
- **ORM 与 DDL 的形状差异照 ch08 的记法逐条对比并记账**(索引名、COMMENT、`unsigned` 有无、**列序**、`created_at` 的默认值措辞)—— 本项目已经栽过一次「只剩两处」是源不支持的绝对断言。

⚠️ **T5 修复轮的订正(2026-09-23,评审实测)**:

- **`occurrences` / `status` 的 `server_default=` 必须写**(照 `ToolAuditLog.retry_count` 的先例)。只写 `default=` 时它**只是 Python 侧默认**,create_all 建出的表**没有列级 DEFAULT** ⇒ 一条省略这两列的**裸 INSERT** 在 create_all 那条路径上失败、在 DDL 那条路径上成功(行为变成「看谁建的库」)。⚠️ 字符串默认值写**不带引号**的 `"pending"`;写成 `"'pending'"` 会渲染成 `DEFAULT '''pending'''`,而它**看起来像对的**。
- **索引覆盖两个方向都拉平**:ORM 侧 `status` / `created_at` 补 `index=True`,DDL 侧补 `idx_matched_review`(见 §7.1 的 ⚠️)。拉平后**索引覆盖三处一致,只有名字不同**(create_all 自动生成的 `ix_*` vs DDL 手写的 `idx_*`,ch08 同款已知差异)。

---

## §8 飞轮流水线

### 8.1 三个纯步骤 + 一个编排

```
app/flywheel/normalize.py   口语原话 → 标准 FAQ 式问题 + 示例答案(prompt)
app/flywheel/dedupe.py      模型判「和待审队列里已有问题是不是同一个意思」
app/flywheel/pipeline.py    取待处理 → 标准化 → 查重 → 建行/累加 + 回填 matched_review_id
```

`normalize` 与 `dedupe` 都是**模型调用 ⇒ 非可单测产出**,按项目规矩(工作要求第 1 条)**用标注样例验证,不套 TDD**:`evals/flywheel_cases.jsonl`,每条给一个口语原话与期望的标准化问题要点(闭式:必须出现/不得出现的词、字数上限),以及一组「同义/不同义」的问题对供 `dedupe` 判。

`pipeline` 是纯编排,**按 TDD 走**。

### 8.2 流水线的一次运行

```
rows = SELECT * FROM low_confidence_questions WHERE matched_review_id IS NULL ORDER BY id LIMIT n
for row in rows:
    norm = await normalize(row.question)            # → {standard_question, example_answer}
    hit  = await dedupe(norm.standard_question, pending_review_rows)
    if hit:
        hit.occurrences += 1
        row.matched_review_id = hit.id
    else:
        rq = ReviewQueue(standard_question=…, example_answer=…, occurrences=1,
                         first_raw_question=row.question, status="pending")
        session.add(rq); await session.flush()
        row.matched_review_id = rq.id
await session.commit()
```

**幂等由 `matched_review_id IS NULL` 单独保证** —— 重跑不会重复归并,不需要额外状态。

**逐行 try/except**:一行失败(模型吐空、解析失败)**不拖垮整批**,那一行的 `matched_review_id` 留 `NULL`(下次重跑还会处理它),失败原因进日志行。

### 8.3 触发方式:照 ch04 的 `orchestrate.py` 那套

- **落池后 fire-and-forget 起一个后台任务**(`app/kb/jobs.py` 的 `JobStore` + `app/flywheel/` 的专用线程 + **自建 engine**),理由与 ch04/ch07 完全相同:`get_engine()` 的 lru_cache 单例绑在**首次使用它的事件循环**上,后台线程里 `asyncio.run` 复用会出跨循环问题。
- **另给 `POST /api/kb/jobs/flywheel`** 手动强制跑一轮 —— 验收脚本靠它,不必等后台任务。
- **`in-flight` 标记的摘除必须在 `finally` 里**(ch07 记过:漏掉不是「多跑一次」,是那个会话**再也跑不了**,而用户侧一切正常)。

### 8.4 落池后触发的时序(如实记)

「落池」发生在 `confidence_gate`(闸拦下)与 `agent` 的知识轮(`useful=false`)里,而**那时请求还没结束**。后台任务在**同一进程**里另起线程 + 自建 engine 读同一张表。⇒ 存在一个**短暂的可见性窗口**:落池那一行提交之后、后台任务读它之前。这**不构成正确性问题**(流水线只看 `matched_review_id IS NULL`,晚一轮也会被处理到),但**验收脚本必须容忍这个窗口** —— 脚本一律用「轮询 + 超时」而不是「落池后立刻断言队列里有」。

---

## §9 审核页与端点

### 9.1 **四个**端点(`app/api/review.py`,与 `kb_router` 并列 include)

| 方法 | 路径 | 作用 |
|---|---|---|
| `GET` | `/api/review/queue?status=pending` | 列表:标准化问题、出现次数、示例答案、状态 |
| `GET` | `/api/review/{id}` | 详情:上面那些 + **归并进来的用户原话清单** + **每条的召回片段快照** |
| `POST` | `/api/review/{id}/approve` | body `{"approved_answer": str?}`;不传就用 `example_answer` |
| `POST` | `/api/review/{id}/reject` | 置 `status='rejected'` |

详情页的两块数据来源:用户原话 = `review_queue.id == low_confidence_questions.matched_review_id` 的那些行的 `question`;召回片段 = 同一批行的 `evidence_snapshot`。**这正是 §7.1 说 `matched_review_id` 是「归并落点」那一面的用途。**

### 9.2 ⚠️ 通过 ⇒ **立刻写知识库并向量化**

```python
chunk = Chunk(category="faq", questions=rq.standard_question,
              answer=rq.approved_answer, section_path=None,
              content_type="faq", is_key_clause=False)
await write_chunks(session, [chunk])          # MySQL,pending
await vectorize_rows(session, store, embedder, rows)   # 嵌入 + Milvus upsert
```

**不这么做的话**,「同一个问题再问就能答对」要等下一次 `scripts/build_kb.py` —— **验收 3 直接落不了地**。

- `write_chunks` 自带 `(category, questions, answer)` 三元组查重 ⇒ 重复通过同一个问题不会写第二遍(返回 0)。
- `vectorize_rows` 要走**共享的 embedder / Milvus store 单例**(`app/tools/registry.py` 的 `build_retriever` 用的是同一批)—— 否则白写一个连接。
- **同步做、不后台**:审核人是**等着看结果**的,后台化只会让「点通过之后到底成没成」变成一个说不清的状态。代价是一次嵌入 + 一次 upsert 的延迟(数百毫秒级)。
- Milvus 不在线 ⇒ 抛 `ToolInfrastructureError`(502),**不许静默降级成「写进 MySQL 了但检索不到」** —— 那是本仓那条「基础设施故障绝不伪装成成功」的反面。

### 9.3 前端(`admin.html` 加第三个标签页)

现有两个标签页(`入库` / `评测`)靠 `data-tab` + `style.display` 切换,新增 `待审` 沿用同一套。复用现有 CSS 变量体系(`--brand` / `--brand-dim` / `--ok` / `--warn` / `--badge`,像素硬边框风格)。

**纯 UI,按项目规矩 Vibe Coding 直做,不套 brainstorm/TDD/code review。**

---

## §10 自动化评估流水线

### 10.1 复用,不重写

`scripts/run_eval.py` 的四种策略、五种指标、五桶拆分**一字不改**;`evals/results/latest.json` 的**结构与路径不变**(`admin.html` 的评测页读它)。

### 10.2 加三样

1. **`--trigger` 参数**(默认 `manual`):跑完把本轮结果**追加**一行进 `eval_runs`(`trigger_by` / `case_count` / `metrics`)。
2. **`--limit N`**:只跑前 N 条用例。**给验收脚本用** —— 300 条 × 4 策略 × 重排的完整一轮不适合放进端到端验收。**默认不限制**(`--limit 0` = 全跑),所以人手工跑还是全量。**加了 `--limit` 的那一轮,`case_count` 记的是实际条数** —— 别让趋势图上两条不同规模的轮次看起来可比。
3. **`scripts/eval_trend.py`**:读 `eval_runs` 按时间打印「轮次 / 时间 / 条数 / 各桶 Recall@10 / MRR」对照表,**并标出每个指标相对上一轮的增减**(哪个指标在下滑要一眼看得出来 —— 需求原话)。

### 10.3 `eval_runs.metrics` 的形状(闭式)

```json
{"top_k": 10, "case_count": 20,
 "strategies": {"混合+Rerank": {"A_policy": {"n": 4, "recall@5": 0.75, "recall@10": 1.0,
                                             "mrr": 0.62, "conf": 0.41, "answer_ok": 0.75}}}}
```

即**现有 `latest.json` 的内容原样进 `metrics`**,外面包 `case_count` —— 这样「评估集的形状」与「这一轮跑了多少条」是两个独立的键,**不会因为 `--limit` 而在同一个键上出现两种含义**。

---

## §11 配置项

### 11.1 新增(`app/config.py`)

```python
# ---- Langfuse(ch09)。三个值任一为空 ⇒ 整套观测 no-op(单测不联网靠它守)。
# ⚠️ 默认是 **Langfuse Cloud(美国区)**;要「链路数据不出自家服务器」就换成自部署地址。
langfuse_public_key: str = ""
langfuse_secret_key: str = ""
langfuse_base_url: str = "https://us.cloud.langfuse.com"

# ---- 置信度闸(ch09 §4)。阈值是**标定**出来的,不是拍的:
# scripts/calibrate_evidence.py 在 evals/测试集.md 上扫出来的折中点,
# 依据是「D_absent 应拒答桶的拦截率 vs 正常桶的误杀率」。改动属设计授权,记 §15。
evidence_confidence_threshold: float = Field(default=0.42, ge=0.0, le=1.0)
evidence_min_score: float = Field(default=0.15, ge=0.0, le=1.0)   # 「什么算一条证据」的下界,三个信号共用(§15.8)
evidence_max_count: int = Field(default=3, ge=1)                  # 条数信号的封顶
w_top1: float = Field(default=0.6, ge=0.0, le=1.0)
w_count: float = Field(default=0.2, ge=0.0, le=1.0)
w_gap: float = Field(default=0.2, ge=0.0, le=1.0)

# ---- 召回片段快照(ch09 §6)。存几条、每条的原文最多留多少字。
snapshot_top_n: int = Field(default=5, ge=1)
snapshot_answer_chars: int = Field(default=400, ge=1)

# ---- 飞轮(ch09 §8)
flywheel_batch_size: int = Field(default=10, ge=1)
```

**`evidence_confidence_threshold` 的默认 `0.42` 是占位**,必须由 `scripts/calibrate_evidence.py` 跑出真值后替换 —— **标定前不许把它写进任何文档当已验证值。**
> **T8 已结清(2026-09-23):`0.42` → `0.2`**(见 §15.9:**平台内的一次判断,不是标定出的最优点**)。上面这段是**设计期原文**,不再改动;
> 标定的实际形态、平台段这条读数与选值理由见 **§15.9**。

### 11.2 改值

无。`retrieval_score_threshold`(0.25)**不变** —— 它仍是检索器内部的过滤阈值;只是它在**闸**里的第二个用处被 §4.3 替换掉了。

> **T8 追加一处改值(2026-09-23)**:`evidence_confidence_threshold` 由占位 **0.42**
> 改为 **0.2**(读数:拦截率 0.967 / 误杀率 0.175,300 条)。
> ⚠️ **是"平台内的一次判断",不是"标定出的最优点"** —— 标定定出的是平台
> `(0, 0.2894]`(该区间内任何取值在这 300 条上读数逐位相同;两个实测数字:
> 正常类非零最小 **0.2894** / `D_absent` 非零最小 **0.5420**),`0.2` 是落在里面的
> 一个选择。**要引用就引平台与这两个数,别把 `0.2` 引成"最优"**;详见 §15.9。
> 属设计授权内的调参,**一句话回退**见 §15.9 末条。

### 11.3 `.env.example`

补上三个 `LANGFUSE_*` 键(现在完全没有)。

---

## §12 测试与验收口径

### 12.1 单测(全程不联网)

**按 TDD 走的**(确定性逻辑):

- `app/agent/json_stream.py` —— §5.3 表里 **13 条边界,每条一个用例**,另加一组 **chunk 边界 fuzz**(把一段完整 JSON 按**每一个可能的切点**切成两段,以及随机切成 3–8 段,**结论必须与不切时逐字节相同**)。这是本章测试价值最高的一处。
- `app/kb/evidence.py` —— 三个信号的取值、空证据、只有一条、分数并列(top1−top2=0)、条数封顶、
  **全部块低于 `evidence_min_score` ⇒ 五个键全 0 且与「空证据」返回同一个 dict**(§15.8)。
- `confidence_gate` —— 判据换成 `evidence_confidence` 后:通过/不通过两支、`reject_reason` 含三个信号、落池的 `entry_point` 仍是 `置信度闸`。
- **`agent` 节点的协议路径**(注入一个**可控的假流**,按需吐片段):
  - **知识轮**走 `protocol`:首键 `useful` 正常 ⇒ line-by-line 的 token 帧与 answer 逐字节一致;**中途的分片不影响结果**(拿 §5.3 的 fuzz 再来一遍,这次在**节点层**)。
  - `useful=false`(**且 `answer` 非空**的违规流)⇒ 一个 answer 帧都没发出去、发了兜底话术、`intent=商品咨询` 时落池 + 存快照。
  - `useful=false` 且 `intent=物流` ⇒ 同样的兜底话术,**但不落池**(§5.5 的两行**各一个用例**)。
  - **首键违规 / 首字符不是 `{` / 流结束仍没 `useful`** ⇒ 一律降级:`raw` 当纯文本发一遍(**不重发**)。
    ⚠️ **这里原先写的是「首键违规 ⇒ 恰好重试 1 次」,已删**(2026-09-23,控制器裁定;理由与代价见 §5.6 的订正段)——
    **重试没有实现,也不实现**;首键违规与其余违规走**同一条**降级路。
  - 降级那一支的 `trace`:**记不记** `agent:protocol_violation` **看这一轮有没有 `tool_calls`**(§5.6 的订正段:**没有 ⇒ 记**(本该作答却吐了散文)、**有 ⇒ 不记**(那是「让我查一下再调工具」的正常轮))。**两种各一个用例。**
  - **业务轮(`intent=物流`)** ⇒ **逐字节等于今天的输出**(这是「零回归」这条不变量的**唯一**硬证据,必须单独一个用例)。⚠️ 用例的脚本**必须是协议形状的输入**(`{` 打头)才有判别力:解码器在 `plain` 态本来就是**逐片原样**透出,拿一句普通中文当脚本时,**「协议泄漏到业务轮」这个改动在帧序列上完全看不出来**(T10 实测)。
  - **落库的那条 assistant** ⇒ 是**答案文本**,不是协议 JSON(真机实测:回载接口
    `GET /api/conversations/{id}/messages` 会把它**原样显示给用户**)。`useful=true` ⇒ `dec.answer`;
    `useful=false` ⇒ 兜底话术;降级 ⇒ 已经发出去的那段文本。**`messages` 与 `turn_messages` 两个通道都要断。**
- `observability` —— §3.6 那条 no-op 测试(关掉时不 import langfuse、`handler()` 为 `None`、`span()` 里跑代码不炸)。
- `flywheel/pipeline.py` —— 幂等(同批跑两次,`review_queue` 行数不变)、查重命中累加 `occurrences` 且回填 `matched_review_id`、逐行失败不拖垮整批。
- `POST /api/feedback` —— `down` 落池、`up` 不落、重复 `down` 幂等。
- 审核端点 —— approve 会调 `write_chunks` + `vectorize_rows`(替身计数)、reject 不改知识库。

**db 标记**:`tests/test_review_api_db.py`、`tests/test_flywheel_db.py` 等带真实 MySQL 的用例一律 `pytestmark = pytest.mark.db`。

### 12.2 假绿防线(本章最容易长出来的四条)

本仓的头号风险是假绿,下面四条按**本章的具体形状**写:

1. **增量解析器的测试输入切得太粗** ⇒ 转义**不跨边界**,于是「跨 chunk 转义」这条根本不被验。⇒ 必须有**穷举切点**的用例(§12.1),不许只测「整段喂进去」。
2. **「业务轮零回归」用一句 `assert reply == ...` 断不出来** —— 业务轮与知识轮走的是**同一个节点**,而「零回归」说的是**推出去的 token 帧序列**逐字节相同。⇒ 用例必须比**帧序列**,不比对最终 `reply`。
3. **`useful=false` 的用例里 `answer` 本来就是空的** ⇒ 「停吐」这一步**不被验**(没有可吐的东西)。⇒ 用例必须构造 **`useful=false` 但 `answer` 非空**的违规流,断「一个 answer_delta 都没发出」。
4. **落池断言不按会话过滤** ⇒ 被上一次运行留下的行污染(ch08 已记过:审计表每轮 +55 行)。⇒ 凡是对 `low_confidence_questions` / `review_queue` 的断言**一律按 `source_conversation_id` 或本轮生成的主键过滤**。

### 12.3 真机冒烟(单测绿了也要跑)

0. **⚠️ 头一条,也是唯一挡路的**:跑一次**真实的 `/api/chat/stream`**,然后打
   `GET /api/public/v2/metrics`(按 `tags` 分组),确认**出现了 `intent:<那一轮的意图>` 这一行**。
   探针里模型调用是直接 `ainvoke` 的,真实链路上它在 `graph.astream` **内部** ——
   中途进入的 `propagate_attributes` 能不能穿透 LangGraph 的上下文,**这是唯一没验的一件事**。
   不行就走 §3.4 末尾的退路。
1. **真实网关上的回答轮真的会吐协议 JSON** —— 单测里的假流是我们喂的,**测不出模型守不守协议**。跑 N 次真实的商品咨询问题,统计 `agent:protocol_violation` 出现几次。**这个数要如实进 dev-notes**,它是 §5.7-1 那条「没有硬保证」的**唯一**证据。
2. **Langfuse trace 真能落在 Cloud 上**:去 Langfuse 用 `session_id` 搜到那条 trace,确认**巢状结构**(模型 span 在节点 span 下)、工具 span 与 retrieval span 都在,**人眼看见**。**探针跑通不算数。**(tag 那半条见 0。)
3. **`scripts/intent_cost.py` 打真实 Metrics API** 拿到非空结果(验收 5 的前置)。

### 12.4 端到端验收:`scripts/acceptance_ch09.sh`

前置:**MySQL + Milvus + 真实 key + Langfuse 可达**(比 ch08 多了 Milvus 与 Langfuse)。沿用 ch08 脚本的既有工程约定:`fail_exit()` 先置 `KEEP=1`、分离 `EXIT` / `INT TERM HUP` 陷阱、`chr()` 拼 needle、**含中文的请求体一律走 stdin heredoc 或 httpx**(不走 `curl` argv)、`join_tokens` 拼回 token 帧后再比对、审计/池表断言**按 `conversation_id` 过滤**、起服务前清端口。

| # | 验收点 | 怎么断(闭式) |
|---|---|---|
| 1 | Langfuse 里能点开任意一条请求,看到完整链路 | 起服务 → 发一次 chat → **打 Langfuse API 按 `session_id` 查那条 trace**,断:存在、且子观测里**同时**有 `retrieval`、`tool:*`(物流那轮)、generation。**「界面能点开」这一半靠人** —— 脚本能断的是数据在不在 |
| 2 | 问一个知识库没有的问题 → 兜底话术 + 出现在待审队列 | 发问 → 断回复含兜底话术 → **轮询** `review_queue`(容忍 §8.4 的可见性窗口)出现该问题的标准化行 → `GET /api/review/{id}` 断详情里有**用户原话**与**召回片段快照** |
| 3 | 审核页点通过 → 同一个问题再问就能答对 | `POST /api/review/{id}/approve` → **同一个会话**(或新会话)重问**逐字同一问题** → 断回复含核准答案里的特征串、且**不再**是兜底话术 |
| 4 | 聊天页点 👎 → 落池 → 标准化查重后进待审队列 | `POST /api/feedback {value:"down"}` → 轮询待审队列出现 → 断该行的用户原话是那一轮的问题 |
| 5 | 按意图汇总的 token 花销 | 跑 `scripts/intent_cost.py` → 断输出里有**至少两个不同意图**的行,且能看出最大值 |
| 6 | 评估流水线至少两轮,能看到趋势对比 | `run_eval.py --limit N --trigger manual` 跑两次 → `eval_trend.py` → 断输出里有**两轮**、且**标出了增减** |

**验收 3 的一个坑**:「同一个问题再问就能答对」依赖**新写的那条知识被检索到**。写过之后要确认 Milvus 里真有那条(验收脚本用 `GET /api/kb/stats` 断 `milvus_count` 增长),否则「答对了」可能只是**原来就答得对**(知识库覆盖不够时的假绿)。

### 12.5 老回归网

ch01–ch08 的既有测试**不改判据、不放宽**。特别地:

- `tests/test_agent_gate.py` 断的是 `entry_point == "置信度闸"` 与 `"0.31" in reject_reason` —— **`reject_reason` 的文案变了**(§4.3),这条测试**要按新文案改**,而**改的是文案不是判据**(仍要断三个信号都在)。这是一处**必须显式处理的既有测试**,不许绕过。
- **图拓扑没变** ⇒ 没有一条既有测试因为「出口换了」而要改。这是 §5.1 选「不动图」的直接收益。
- **但要逐个审「跑 `agent` 节点的用例里有没有用商品咨询意图的」**:那些轮的 token 帧会经过解码器。业务意图的用例**预期逐字节不变**,商品咨询意图的用例**预期改形态** —— 后者要显式改成协议用例(§12.1),**不许为了让老测试绿而把协议关掉**。
- `app/agent/nodes.py` 的 `_stream_round` 签名会变(多一个文本出口)。全仓搜它的调用点(3 处)与测试替身,逐个确认。

---

## §13 风险与已知取舍

| # | 风险 | 应对 |
|---|---|---|
| 1 | ⚠️ **中途 `propagate_attributes` 在真实图里能不能给 graph 内部的 generation 打上 tag —— 未验**(探针里模型调用是直接调的,真实链路上它在 `graph.astream` 内部) | 已按 wheel 源码核过签名、已用探针验过机制(§3.4);**剩下这一条是 T1 的冒烟项**。不行就走 §3.4 末尾写明的退路:`intent_cost.py` 改从我们自己的观测聚合 |
| 1b | ~~Langfuse API 未按 wheel 核对~~ | **已解除**:4.15.4 已装并逐字读过 `langfuse/__init__.py`、`langchain/CallbackHandler.py:496-520`、`_client/span.py:637-675`;两条探针跑通(§3.4 / §3.5) |
| 2 | `.env` 指向 **Cloud**,与需求写的自部署不符 | 已由用户 2026-09-23 拍板;§2.1 记账;换自部署只需改一个值 |
| 3 | **协议没有硬保证**(`response_format` 与 `bind_tools` 互斥,§2.2)⇒ 模型不守协议时该轮拿不到 `useful` | §5.6 降级 = 今天的 ch08 行为,**不会更差**;`trace` 记 `agent:protocol_violation`(**口径见 §5.6 订正段**:只数「本该作答却没有 JSON」的轮),违约率可观测;§12.3-1 真机跑 N 次把违约率量出来 |
| 4 | **改动落在 `agent` 节点上**(全仓最复杂的一段,ch08 的确认流在里面) | 改动面**只有两处**:`msgs` 多一条协议消息、文本出口多一次解码。**工具绑定、工具执行、`pending_write`、`turn_messages`、每轮清零全不碰**;§12.1 的「业务轮逐字节不变」用例是这条的硬证据 |
| 5 | **增量解析器的转义跨 chunk** | §5.3 的 13 条边界 + §12.1 的穷举切点 fuzz |
| 6 | **👎 回捞是「重跑」不是「当轮」** | §6.2 的两条偏差如实记 |
| 7 | **`eval_runs` 两轮的规模可能不同**(`--limit`) | §10.2-2:`case_count` 单独一列,趋势表把它打出来 |
| 8 | **验收 3 的「答对了」可能是假绿** | §12.4 末尾那条:先断 Milvus 条数增长 |
| 9 | `write_chunks` 的三元组查重可能让「审核通过」写不进去(同一 category+question+answer 已存在) | 端点要**返回** `chunks_added`;为 0 时**不算失败**但在响应里说明(那时说明知识库早有这条,「答不对」是别的原因)—— 验收脚本据此给出可读的失败信息 |
| 10 | Langfuse SDK 引入 **opentelemetry-\*** 一串新依赖 | `requirements.txt` 钉 `langfuse==4.15.4`;装完核对是否与既有 `httpx==0.28.1` / `pydantic==2.13.5` 冲突 |
| **11** | ⚠️ **每 6 条口语化提问就有 1 条拿不到任何证据、直接吃兜底话术(17.5%),而这条与阈值无关。** T8 实测:`evals/测试集.md` 240 条正常用例里 **42 条检索返回空** ⇒ 置信度 0.0 ⇒ 闸的 `bool(evidence)` 在**任何**阈值下都拦它们(**连 0 也拦**)。**集中在口语桶:`C_colloquial` 21/60 = 35% 被误杀**(对照 `B_model` 只有 3/60)。⇒ 退化判据里的「误杀率 ≤ 0.05」在这份评估集上**不可达**。 | **这是本章量出来的一个既有缺陷,不是本章引入的,也不是旋钮能治的** —— 修法在**检索覆盖率 / 口语问法的 Query 改写**上(spec §4.2 的标定越准也救不了它)。如实记账,别淡化;要不要动检索由用户拍板。读数见 §15.9 |
| **12** | ⚠️ **置信度公式(`evidence_confidence` 的三信号)的判别带,在本章评估集上零覆盖。** T8 实测:60 条 `D_absent` 里 **58 条是空证据**,只有 2 条有证据(0.6978 / 0.542);300 条里**没有一条**的置信度落在 `(0, 0.2894)`。⇒ 拦截率 0.967 **几乎全部由「检索返回空」挣来**,而不是公式把「弱证据」分辨出来的。 | **本章验证的是检索器,不是那个三信号公式。** 要验公式本身,需要一份「知识库**有**、但**不该答**」的**近域硬负例**用例集,现在这份(`D_absent` = 知识库根本没有的内容)没有这条分辨力。属于评估集覆盖面的缺口,与 §13-11 是同一枚硬币的两面;读数见 §15.9 |
| **13** | ⚠️ **首键违规那一支,用户会看到一段协议 JSON 原文。** 模型没按 `useful → confidence → answer` 的顺序开头时,走的是 §5.6 的降级:**`raw`(整段 JSON)当纯文本显示给用户**。 | **这是「取消重试」的代价,不是缺陷**(§5.6 的订正段已写清理由)。已拍板接受:**丑但可预测**,`trace` 里有 `agent:protocol_violation`、日志里有告警。**用户日后若要求「更好看」,首选修法是把重试加回来**(次选是围栏剥离,§5.6 已明确不做)。**别把它读成「未处理的 bug」** —— 它是一个**有意接受**的质量取舍 |

---

## §14 交付物清单

| 类别 | 路径 |
|---|---|
| 新增 | `app/observability.py`、`app/agent/json_stream.py`、`app/kb/evidence.py`、`app/flywheel/{__init__,normalize,dedupe,pipeline}.py`、`app/api/review.py`、`db/ch09.sql` |
| 新增(脚本) | `scripts/calibrate_evidence.py`、`scripts/intent_cost.py`、`scripts/eval_trend.py`、`scripts/acceptance_ch09.sh` |
| 新增(评估) | `evals/flywheel_cases.jsonl` |
| 新增(端点) | `app/api/feedback.py` —— `POST /api/feedback`(不塞进 `app/api/chat.py`,那个文件已经 600 行) |
| 改动 | `app/config.py`、`app/db/models.py`、`app/agent/nodes.py`(**三处**:闸的判据、检索 span、`agent` 的协议消息 + `_stream_round` 的文本出口)、`app/api/chat.py`(**只加 `config` 里的 callbacks/metadata**)、`app/main.py`(include 两个新 router)、`app/static/index.html`(👎 接后端)、`app/static/admin.html`(待审标签页)、`scripts/run_eval.py`(两个参数 + 写 `eval_runs`)、`requirements.txt`、`.env.example` |
| **不改** | `app/agent/graph.py`(拓扑一字不动)、`app/prompts.py`(**协议消息不进 `render_system_prompt`**)、`app/tools/**`(ch08 零改动) |
| 文档 | `CLAUDE.md`(章节条目 + 硬约束 + 命令)、`AGENTS.md`、`dev-notes/ch09.md` |

---

## §15 实现订正

**（本节由实现期填写。原文一律不改,订正逐条追加,写明「最初写的是什么 / 实际是什么 / 为什么会写错 / 证据」。）**

### 15.1 待订正:§2.2 的表里第四行

- **最初写的**:「`bind_tools(strict=True)` + `response_format`,要求调工具 → ❌ 网关 400」,并据此在正文里声称 strict 路径**调工具不通**。
- **实际是**:那条 400 的成因是**探针的提示词里没有 `JSON` 字样**(网关原话:`Prompt must contain the word 'json' in some form`),**与 strict / 工具无关**。strict 路径**调不调得动工具,未验**。
- **写错的原因**:把「一次调用失败」直接读成了「这个组合不行」,没有看错误文本说的是什么。**本仓那条「先写结论后没跑」的老毛病换了个壳**。
- **影响**:不影响本章选型(§5 本来就不走 strict 路径),但**结论句是错的**,按本仓规矩必须订正而不是删掉。
- **另**:同一次探针的**首版**还有一个更严重的假绿 —— `kwargs={}` 那两个分支**压根没把 `response_format` 挂上去**,把「裸 `bind_tools`」当成了「两者同时用」,输出看起来完全正常。**验证装置自己产假绿,这是本仓记过的形态,这次又中了。**

### 15.2 待订正:§5 整节 —— 「新节点 + 不绑 tools」被用户否掉,改成「`agent` 内挂协议」

- **最初写的**:知识路径的作答换成一个**新节点 `knowledge_answer`**(`app/agent/answer_nodes.py`),
  不绑 tools + `response_format=json_object`,拿硬保证;`confidence_gate` 的出边改指向它;
  `_OUTLETS` 加一个成员。代价栏写的是「商品咨询失去 `query_product`」,并给了退路。
- **实际是**:用户 2026-09-23 后半段否掉了它 ——
  > 「知识路径作答还是要绑定 tools 我觉得,比如退款售后问题的时候多路检索之后进主力 agent
  > 可能还是要调工具」
- **为什么会写错**:我用 §2.4 的 `INTENT_TO_ROUTE` 表去**倒推**「知识路径不需要工具」。
  那张表只决定**走哪个出口**,不决定**那个出口需不需要工具**;而且我**自己写着**
  「`log/app.log` 里没有任何商品咨询样本,所以知识类今天会不会真的调工具**没有证据**」——
  **证据没有,结论却下了**。这与 §15.1 是同一种毛病的两种壳:**把「看起来合理」当成「已成立」。**
  用户举的例子(退款售后)其实落在 `KNOWLEDGE` 之外,但**他的方向是对的**。
- **订正后的形态**(本节起,§5 全节以新形态为准):
  - 图拓扑**一字不改**;`agent` **保留 `bind_tools`**;协议只加在**知识轮**
    (`intent == "商品咨询"`)的 `msgs` 里,文本流出走三态解码器。
  - `response_format` **用不上了**(与 `bind_tools` 互斥,§2.2)⇒ 协议**纯提示词驱动**,
    §5.7-1 如实记「没有硬保证」,§12.3-1 真机量违约率。
  - `app/agent/answer_nodes.py` **不建**;`app/agent/graph.py` **不改**;
    `_OUTLETS` **不动**;`app/prompts.py` **不动**(协议消息不进 `render_system_prompt`,
    以免连带改 `budget.derive` 的推导)。
  - **上一版那条「`response_format` 走 beta 路径、usage 可能丢」的风险随之消失**(§5.7-3)。
  - **业务 / 退款 / 闲聊三条路径的输出形状与今天逐字节相同** —— 上一版方案想要的
    「ch08 确认流零风险」,新形态用一个 `if` 就拿到了,而且比新节点更小。
- **代价(如实记)**:新形态把改动落到了 `agent` 节点上(全仓最复杂的一段)。
  缓解是改动面**只有两处**(`msgs` 多一条消息、文本出口多一次解码),
  且 §12.1 加了「业务轮 token 帧逐字节不变」这条硬证据。

### 15.3 待订正:§3.4 / §3.5 —— Langfuse 的三条接口事实**全和文档不一样**

**最初写的**(§3.4 原文):用 `propagate_attributes(trace_name=…, session_id=…, tags=…)`
包住 `astream`,意图「用 `update_current_trace` 补写」。

**实际是**(实测,探针 `.superpowers/probe_ch09_langfuse.py` / `…2.py`):

| 我写的 / 文档说的 | 实测 |
|---|---|
| `langfuse.update_current_trace` 存在 | ❌ **不存在**。模块级没有,`Langfuse` 客户端上也没有(它有 `score_current_trace` / `set_current_trace_io`,没有 update) |
| `span.update(**{"langfuse.trace.tags": …})` | ❌ **静默丢弃** —— 源码 docstring 逐字 `**kwargs: Additional keyword arguments (ignored)` |
| `CallbackHandler(session_id=…, user_id=…, tags=…)`(文档 JS 页的写法) | ❌ Python 4.15.4 的签名只有 `(*, public_key=None, trace_context=None)` |
| `client.api.trace.get(id)` 读回 | ❌ 已弃用,换 `GET /api/public/v2/observations`(按 `traceId` 过滤) |
| Metrics v2 把参数平铺进 query string | ❌ 必须 `params={"query": json.dumps({...})}` |
| Metrics 能报成本 | ❌ `sum_totalCost` **恒为 0**(模型没配价格)⇒ 统计只报 **token** |
| 按 `tags` 过滤用 `type="string"` | ❌ 网关要求 `arrayOptions` |

**为什么会写错**:§3.4 那一版是**照文档写的**,而文档里那页是 **JS/TS 示例与 Python 示例混排**,
我按 JS 的构造参数写了 Python 的代码。**ch08 已经记过「Context7 整站迁 v2、以轮子为准」**,
这次**没等读轮子就把结论写进了 spec** —— 同一个教训的第二次。

**订正后的形态**:见 §3.4(外层 `propagate_attributes` 管 session/trace_name,
**中途同步 `__enter__` 一个只带 `tags` 的 `propagate_attributes`** 管意图)与
§3.5(请求形状、只报 token、过滤要 `arrayOptions`)。

**顺带记一条「静默无效」的新成员**:`span.update(**kwargs)` 不报错、不生效。
本仓已经收过「未声明的通道写入被静默丢弃」(ch06)、「`add_messages` 给无 id 消息赋 uuid4」
(ch07)、「`response_format` 走 beta 路径」(ch09 §2.2)。**这一次是第四次同一个形状**:
**一个看起来会生效的赋值,什么都没做,而且不报错。**

### 15.4 订正:§12.3-0 的「唯一挡路的未知」**已结清(成)**,而 §3.3 的前提**被实测推翻了一半**

（T3 的真机冒烟结果,2026-09-23。**§3.4 / §12.3-0 原文一律不改,订正追加在这里。**）

**① §12.3-0 问的那件事:成。**

「中途进入的 `propagate_attributes` 能不能穿透 LangGraph 的上下文,让 **graph 内部**的
generation 带上 `intent:<x>` 标签」——探针里模型调用是直接 `ainvoke` 的,没验过;真机跑一次
商品咨询请求后,`GET /api/public/v2/metrics` 按 `tags` 分组读回:

```
{"tags": ["ch09", "intent:商品咨询"], "sum_totalTokens": "1485", "count_count": "8"}
{"tags": [],                          "sum_totalTokens": "646",  "count_count": "8"}
```

- tag 行**出现了**,token **1485 > 0**;**`intent:*` 那一行没落空 ⇒ §3.4 末尾那条退路不启用**。
- **8 / 8 这个切法本身就是证据**:整条 trace 16 条观测,进入作用域**之前**恰好 8 条、
  **之后**也恰好 8 条,边界**正好**落在 `classify_intent` 上 —— 与 §3.4 的设计语义逐条对上,
  不是"整体都带上了 tag"那种分不出判别力的读数。
- 第二个请求(订单意图)复现:`intent:订单` / 2207 tokens / 7 条。
  **顺带把验收 5 的前提也验到了:两行不同意图、token 不同,可分。**

**② §3.3 的前提「工具执行与知识检索一个 span 都不会自动出现」——只对了一半。**

- **对的一半**:`retrieve` 走的是自写的 `KnowledgeRetriever`,**确实**一个 span 都不会有,
  手工 `retrieval` span 是它唯一的落点。
- **错的一半**:**内置工具**的执行(`app/tools/executor.py:execute_tool` → `spec.tool.ainvoke`)
  **是**一个 LangChain run ⇒ 回调**已经**给了它一条 `TOOL 'query_order'`,而且**嵌套正确**
  (在 `agent` 之下)。手工那条 `tool:*` 的价值因此**不在"有没有"**,而在它带
  `ok` / `summary` / `error_kind` 这三个 LangChain run 拿不到的字段。

**③ 新发现(比 ② 更要紧):手工 span 落进的是**另一条 trace**,不是请求那条。**

按 `sessionId` 读回那次商品咨询请求的 16 条观测,里面有 **2 个 `traceId`**:

| 观测 | `traceId` | `parentObservationId` |
|---|---|---|
| `LangGraph` 根及其下全部(包括 `ChatOpenAI` generation 与 `TOOL 'query_order'`) | `879bf705…` | —(根) |
| 手工 `retrieval`(RETRIEVER) | `6ff9033e…` | **null** |
| 手工 `tool:query_order`(TOOL) | `07ab51df…` | **null** |

- **成因**:`start_as_current_observation` 的父级取自 **OTel 当前 span**,而 Langfuse 的
  LangChain 回调**不把观测挂成 current**(它靠 LangChain 的 run tree 定父子)⇒
  `graph.astream` 内部**没有"当前 span"**,手工 span 只能**自己开一条新 trace**。
  两条 trace 的 `sessionId` 与 `tags` 都是对的(所以 §3.5 的按 tag 聚合**照样是对的**),
  但它们**不在同一条 trace 上**。
- **后果**:§12.3-2 与 §12.4 验收 1 那句「**子观测**里同时有 `retrieval`、`tool:*`、generation」
  **按字面断不出来** —— `generation` 在请求那条 trace 上,两个手工 span 在另外两条上。
  用户需求里「点开任意一条请求,看到完整链路」也**只满足了大半**。
- **候选修法(未实施,待主控裁定)**:在 `app/api/chat.py` 把 `graph.astream` 再包一层
  `with observability.span("chat", as_type="span", settings=settings):` —— 有了 current span,
  手工 span 会挂进去、LangChain 的 `LangGraph` 根也会挂进去,顺带对上 §12.3-2 里
  「`chat`(根)」那个描述。**代价**:改的是**请求路径上的 trace 拓扑**,
  而 §3.2「挂点:**一处**」与 R5 的授权都只到「多传一个 `settings=`」为止 ——
  所以 T3 **只报不改**。

**④ 顺带记一条环境事实(不是设计问题,但会让人误判成数据坏)**:本机 Milvus 容器
(`milvus-standalone`,`ETCD_USE_EMBED=true`,1s 选举超时)**在 CPU 被占满时起不来** ——
实测症状是 `panic: etcdserver: leader changed` + SIGABRT(exit 134)。成因是嵌入 etcd 没等到
ReadIndex 就换了 leader。**机器空下来再 `docker start`,一次就起,存储卷完好**(实测:集合
`knowledge` 90 行,一字未丢)。**别急着删数据。**

### 15.5 订正:手工 span 的**嵌套**已修 —— `trace_scope` 补开根观测;手工 `tool:*` span 删除

（T3 的修复轮,2026-09-23。**§15.4 ③ 那条「未实施 / 待主控裁定」以本节为准。**
§15.4 保留原文,记的是当时**只报不改**的那个状态。)

**主控裁定(原话摘要)**:「`trace_scope` 现在**只调了 `propagate_attributes`**,没有开任何
**当前 span**……**是我把这个关键动作漏在了 `trace_scope` 的实现里**」⇒ 两件事一起做。

**① `app/observability.py:trace_scope` 补开根观测(`chat`)**

- 新增接缝 `_root_cm(settings, conversation_id)`(照 `_outer_cm` / `_observation_cm` 的样子
  单独抽出来,**唯一目的是让 enabled 路径能在不联网、不 import langfuse 的前提下被验到**)。
- 形状:`propagate_attributes(...)` 在**外层**、根观测在**里层**;退出顺序相反。
- 根观测:`name="chat"` / `as_type="span"` / `input={"conversation_id": …}`。
  **用户原话刻意不在这里塞**(谁手里有谁放,归端点决定)——
  在真正的 `chat` 根观测上补 `input`,是上下文那半章的事。
- 进入失败:**进了一半要 unwind 外层**(否则 `propagate_attributes` 的 token
  一直挂在当前任务上,后面的观测会被当成它的孩子 —— 跨请求串味,且**没有任何日志**)。
- 退出走 `_safe_exit`,根与外层各吞各的异常。
- **对外签名不变**(`trace_scope(*, conversation_id, settings)`),端点那一侧一字未动。

**② 删掉手工的 `tool:*` span(`app/agent/nodes.py` 与 `app/agent/confirm_nodes.py`)**

- 依据:§15.4 ② 的实测(内置工具**已经**有一条嵌套正确的 `TOOL` 观测)+ Langfuse 自己的
  最佳实践原话 **`Don't emit duplicate dispatch + execution nodes. … it double-represents
  one event.`** ⇒ 一次 `query_product` 产出两条观测是**同一个事件表示两遍**。
- 那两处**包着的业务代码一行不动**,只是不再开手工 span;
  `app/agent/confirm_nodes.py` 的 `observability` import 随之删除。
- **`retrieval` 那条保留**:`KnowledgeRetriever` 是自写的普通类,**不是** LangChain run,
  没有它知识检索在界面上是空的 —— 这一条不能被自动覆盖,手工 span 的存在理由成立。

**③ 修复后的真机复验(三个请求,逐个查 `traceId` 与 `parentObservationId`)**

| 请求 | 意图 | 观测数 | **distinct `traceId`** | 树里有没有 |
|---|---|---|---|---|
| A(知识路径 + 闸不过) | 商品咨询 | 15 | **1** | `chat` 根 / 3×generation / `retrieval` |
| B(业务路径) | 订单 | 15 | **1** | `chat` 根 / 4×generation / **`TOOL 'query_order'`** |
| C(知识路径 + 调工具) | 商品咨询 | 19 | **1** | `chat` 根 / 5×generation / **`TOOL 'query_product'`** / `retrieval` |

修复前的对照值(§15.4 ③):同一次请求里 **2 个 `traceId`**,手工 span 的
`parentObservationId` 是 **`null`**。现在三个请求**各自只有一个 `traceId`**,
根是 `SPAN 'chat'`(`parentObservationId=null`),`LangGraph` chain 与 `retrieval` 都挂在它下面。

**④ tag 没有回归**:同一窗口的 Metrics v2 按 `tags` 分组(修复后重跑)

```
{"tags": ["ch09", "intent:商品咨询"], "sum_totalTokens": "3365", "count_count": "18"}
{"tags": [],                          "sum_totalTokens": "2444", "count_count": "24"}
{"tags": ["ch09", "intent:订单"],     "sum_totalTokens": "2198", "count_count": "7"}
```

24 = 3 个请求各 8 条未 tag 的观测(`chat` 根 + `LangGraph` + `resolve_references` +
它的 generation + `classify_intent` + `RunnableSequence` + generation + `PydanticOutputParser`),
7 + 18 = 25 条已 tag;`24 + 7 + 18 = 49` = 三个请求的观测总数,**逐条对得上**。
⇒ **边界仍然落在 `classify_intent` 上,根观测的引入没有把 tag 弄丢。**

### 15.6 订正:退款子流程的 `refund_expand_retrieve` 补上 `retrieval` span;`retrieval` 与 `LangGraph` 平级**接受**

（T3 修复轮 2,2026-09-23。两条裁定各记一条。）

**① 补 span(需求 1 的一个洞,不是「不在本章范围」)**

§14 那张表里手工 span 只列了「`execute_tool` 的调用点」与「`retrieve_knowledge` 节点」两处 ——
**漏了 `refund_expand_retrieve`**(`app/agent/refund_nodes.py`),而它**也调 retriever**
(`multi_search`)。后果是「每个节点的 prompt、工具调用、**检索结果**、token 消耗和耗时都能
铺开看」这条需求对**退款请求**不成立:界面上那一步是空的,验收 1「点开任意一条请求,看到
完整链路」对退款请求断不出来。这是**请求路径上最后一处**自动覆盖不到、又确实漏了的检索点。

补法与知识那一路**同形**(键名与字段逐字一致,两条检索在界面上必须是同一种读法):

| 位置 | span 名 | as_type | input | output |
|---|---|---|---|---|
| `refund_expand_retrieve`(`multi_search` 那一步) | `retrieval` | `retriever` | `{"query": queries}` —— **扩写之后的多路查询列表**,不是 `state["resolved_input"]` | `{"chunks": [{id, score, section_path}]}` |

**真机复验**(退款请求 `订单 20240915 这个能退吗`,`intent=退款退货`,
`refund_expand_retrieve:3 路 9 命中 top=0.97`):25 条观测、**1 个 `traceId`**,根 `SPAN 'chat'` →
`LangGraph` → `refund_pick_order` / `refund_fetch_order`(内有 `TOOL 'query_order'`)/
`refund_expand_retrieve` / `refund_judge` / `refund_offer` / `log_turn`,
外加 `RETRIEVER 'retrieval'`(`parentObservationId` = `chat` 那个 SPAN)。
⇒ **两条检索现在都落进请求那条 trace**,退款请求的完整链路成立。

**② `retrieval` 挂在根 `chat` 下、与 `LangGraph` 平级 —— 接受,不再是缺陷**

拿不到 `LangGraph` 那条 chain 观测的 observation id(它是 Langfuse 的 LangChain 回调建的),
LangGraph 也不给节点级 span 钩子;要硬做就得自己把整个图重包一遍。

**判据(裁定原话)**:**「同一条 trace」是需求的实质(能点开一条请求看完整链路),
「挂在哪一层的缩进」不是。** 前者两条链路都已验到,后者不再动代码。

**③ 顺带钉住的不变量**:手工 span 现在**只有一种形状**(`retrieval`)。`tool:*` 已按 §15.5 删除,
**不许以任何理由加回来** —— `tests/test_agent_refund.py` 的新用例里有一条
`spy.calls == [ {...} ]` 顺带钉住「整轮里只有这一条手工 span」。

**④ 两条检索 span 的守卫(本轮新增,此前**一条都没有**)**

变异实测(改错实现后能不能红):

| 变异 | 结果 |
|---|---|
| 删掉 `refund_expand_retrieve` 的 span(**本轮之前**) | 全量 709 **全绿** ⇒ 它没被任何东西守着 |
| 删掉 `refund_expand_retrieve` 的 span(加用例之后) | `tests/test_agent_refund.py` **1 failed** |
| 删掉 `retrieve_knowledge` 的 span | `tests/test_agent_gate.py` **1 failed** |
| 退款那条 `input` 记成原话(`text`)而不是扩写后的 `queries` | **1 failed** |
| 知识那条 `input` 记成 `user_input` 原话而不是 `resolved_input` | **1 failed** |

两条用例都建在 `observability.span` 这个**边界**上(捕获实参 + 一个记录 `update` 的假 handle):
`name` / `as_type` / `input` 的形状与**取值**、`output` 的字段 —— 这些在节点返回的
dict 里一个字都看不见。

### 15.7 订正:§3.5 的「按 tags 过滤」与 cost 两句 —— **计划里那段代码没有对着真实网关跑过**

（T4 的真机验证结果,2026-09-23。**§3.5 原文的 1/3 两条保留,其余在 §3.5 就地改了。**）

**最初写的**(T4 的 brief 与 §3.5 原文):`--intent` 的 filter 只记了 `type` 与 `value` 两个字段
—— `{"column": "tags", "operator": "contains", "value": [...], "type": "arrayOptions"}`;
`metrics` 里只请求 `totalTokens` 与 `count`。

**实际是**(实测,探针 `.superpowers/probe_t4_filter.py` / `probe_t4_cost.py`,两个都**已删**;
网关原话逐字抄在 §3.5 第 4/5 条):

| 我写的 | 实测 |
|---|---|
| `operator: "contains"` | ❌ **400** —— arrayOptions 那组的合法算子是 `any of` / `none of` / `all of`,**`contains` 是 string 那一组的**。改 `"any of"` 之后 200 |
| `type` / `value` 两个字段我记对了 | ✅ 网关逐字背书:`Expected 'arrayOptions'`、`expected array, received string` |
| `--intent` 拿到响应后按 `r.get("tags")` 挑行 | ❌ `dimensions` 清空时行里**没有 `tags` 键** ⇒ 恒判 False ⇒ **永远打印「没有找到任何 intent 观测」,不报错** |
| `metrics` 不含 `totalCost`,却打印「cost 恒为 0」 | ❌ 该键**不在响应里** ⇒ 那句话是**断言**;`total_cost > 0` 是**死代码** |

**为什么会写错**:这三处**都不是「记错了」**,而是**记漏了**/**没跑过**——
`type` 与 `value` 我记对了,只是把**三个字段里的两个**当成了全部;
而「cost 恒为 0」那句我实测过一次(单独发过一次带 `totalCost` 的查询),
但**写进 brief 的代码里没有带上那个 measure**,于是代码里的 0 退化成空值兜底。
**根因是同一句话:计划里的代码片段没有对着真实网关端到端跑过一遍。**
本仓已两次记过「先写结论、后没跑」(ch05 的 `confidence_gate`、ch07 的「冷启动必然超时」),
**这是第三次**,而且这次错的是**交给实现者的代码本身** —— 实现者照抄就必踩,
只有他去真机跑一次才会发现。

**订正后的形态**:见 §3.5 第 2/4/5 条与订正后的代码块。
脚本侧的三处修改在 `scripts/intent_cost.py`(T4),**逐字对得上网关**:
`--intent 商品咨询` → `26 / 4850`、`--intent 订单` → `14 / 4405`,
与**不分组那一版的同名行逐字节一致** —— 过滤既不漏也不多。

**顺带钉住的两条(都被实测撞到)**:

- **`raise_for_status()` 的异常文本只带 URL、不带 body** —— 400 的原因(哪个字段、合法值是什么)
  在**响应体**里。计划里那段代码直接把它甩给 traceback,于是第一步排查看到的是 30 行栈,
  **一个字的原因都没有**。⇒ `_fetch` 里把 `resp.text[:800]` 抬上来。
- **`SystemExit` 与 traceback 走 stderr** —— 计划里只 `reconfigure` 了 stdout,
  于是三条中文报错在 cp936 管道上是乱码(**不崩**,比崩更难发现)。⇒ 两个流都钉。

**顺带记一条「静默错」的新成员**:`r.get("tags")` 在 `dimensions` 清空时**恒为空**
⇒ 脚本**输出一句看起来完全合理的「没有找到任何观测」**,退出码 0,没有任何异常。
与 §15.3 那条 `span.update(**kwargs)` 是同一个形状(「做了,但什么都没发生,而且不报错」),
**这是本章第二次**,也是全仓第五次。它比 400 **危险得多** —— 400 会逼你去查,
它只会让你得到「窗口内确实没数据」这个**错误结论**。

### 15.8 订正:§4.1 —— `evidence_min_score` 必须也用在 top1/top2 上(骨架与它自己的测试不一致)

- **最初写的**(本节 4.1 的表 + T6 的实现骨架):`evidence_min_score` **只**用在
  「有效证据数」这一个信号上;`top1` 写作 `max(c.score)`、`top2` 写作"第二大的 score",
  两者都隐含地取在**全部**分数上。
- **实际是**:**同一批块上算三个信号**。T6 的实现骨架逐字落地后,它**自己 brief 里那条**
  `test_low_scores_do_not_count_toward_the_count_signal` 就是**红的**:

  ```
  tests/test_kb_evidence.py:68: in test_low_scores_do_not_count_toward_the_count_signal
  >   assert noisy == clean
  E   assert 0.4647 == 0.4667
  1 failed, 7 passed
  ```

  根因:那条 `0.01`(低于 `evidence_min_score=0.15`)的噪声**没被算进条数**,却
  **占住了 top2 的位子**,把 gap 从 `0.50` 压到 `0.49` ⇒ 置信度差 `0.2 × 0.01 ≈ 0.002`。
  即「不是证据的块**换了条路**影响了另一个信号」。更糟的一侧:全部块都低于下限时,
  骨架仍会拿那条噪声分当 `top1` ⇒ 纯噪声拿到 `confidence = 0.084` 的**非零**值。
- **为什么会写错**:**计划里的实现骨架与它自己的测试不一致 —— 骨架是错的,测试是对的。**
  §4.1 那张表其实**已经承认了这条下限**(它把"有效证据数"定义成「分数 ≥
  `evidence_min_score` 的条数」),我只是**没把同一个下限推广到另外两个信号上**;
  而 T6 的用例作者(同一个我)在写用例时用的语义是"不是证据的块对**置信度完全不可见**",
  两边**没有对过**。⇒ 与本仓已记过的形态同族:§15.1「把看起来合理当成已成立」、
  §15.7「交给实现者的代码本身没跑过」;**这一条是第三次**,错因是**同一份计划里
  两处文本各自成立、互相矛盾,而两处都没有被拿着对读**。
  这一次是**实现者去跑了才照出来的** —— 若他照抄骨架交差,那句"8 passed"会被
  用例**自己**揭穿,不会静默通过。
- **订正后的形态**(§4.1 表已改):`valid = [分数 ≥ evidence_min_score 的块]`;
  三个信号(`top1` / `count` / `gap`)**一律在 `valid` 上算**;`valid` 为空时
  与"空证据"走**同一个出口**,五个键全 0。
- **证据**:T6 的变异检查 —— 把过滤那一步去掉(退回骨架写法)后,
  **两条**用例同时变红:`test_low_scores_do_not_count_toward_the_count_signal`
  (`0.4647 == 0.4667`)**与**新补的
  `test_all_below_min_score_is_exactly_the_empty_evidence_result`
  (`{'top1': 0.14, 'top2': 0.14, 'confidence': 0.084, …}` ≠ 全 0)。
- **记账一条范围限定**:`confidence = 0.084` 在**当前占位阈值 `0.42`** 下**仍然会被拦**,
  所以这不是"今天就漏放了"的活故障;订正的是**语义** —— 「判据的取值不该依赖
  "噪声分恰好算出来不大"」,而这个性质在 T8 把阈值标定成**更小**的值时就会失效。

### 15.9 订正:§4.2 / §11.1 —— 占位阈值 `0.42` → **`0.2`**(**平台内的一次判断,不是标定出的最优点**)

> 标题就是结论,别读成「标定得 0.2」。标定定出来的是**一段平台**(下面第 1 条),
> `0.2` 是落在平台里的一个选择。**要引用本章的阈值,引"平台"与那两个实测数字,不要引 0.2。**

- **原来写的**:§4.2 说「取折中点」,§11.1 写着 `default=0.42`,并注明「**0.42 是占位,
  不是标定值,标定前不许把它写进任何文档当已验证值**」。T8 跑完后它被替换 ——
  但替换它的**不是**一个"量出来的最优点",而是下面第 1 条那个平台里的一个取值。
- **实际是什么**(`scripts/calibrate_evidence.py`,`evals/测试集.md` 300 条 = A/B/C/E
  各 60 + `D_absent` 60,走**真实链路** `retriever.search` → `evidence_confidence`):

  | 阈值 | 拦截率 | 误杀率 | 拦截数 | 误杀数 |
  |---|---|---|---|---|
  | 0.050 – 0.275(网格上这一段读数逐位相同) | 0.967 | 0.175 | 58/60 | 42/240 |
  | 0.300 | 0.967 | 0.188 | 58 | 45 |
  | 0.425 | 0.967 | 0.263 | 58 | 63 |
  | 0.550 | 0.983 | 0.333 | 59 | 80 |
  | 0.700 | 1.000 | 0.458 | 60 | 110 |
  | 0.950 | 1.000 | 1.000 | 60 | 240 |

  **写回 `0.2`**(`app/config.py`)—— 平台内的一次判断,不是量出来的最优(见下)。
  分支:**正常**(存在拦截率 ≥ 0.6 的解)。**注:走"正常"分支这件事本身没什么分量**,
  两列读数在整个平台段上都是常数(见第 1 条)。
- **⚠️ 本节最重要的一条:标定没定出一个"点",只定出了一段平台。**
  逐条读数(`--dump`)显示:**300 条里没有一条的置信度落在 `(0, 0.2894)` 这个开区间内**
  —— 正常桶非零最小 **0.2894**、`D_absent` 非零最小 **0.5420**,其余全是 **0.0**。
  所以 `(0, 0.2894]` 里的**任何**阈值在这 300 条上读数**逐位相同**。
  "折中点"这个说法在这份评估集上**没有落点**。
  ⇒ 写回值因此**不是量出来的**,是**设计判断**,已按判断记账:
  取 `0.2` 而不是扫描网格的**下沿 0.05** —— 后者是 `--grid` 默认起点(0.05)的产物,
  换一个起点它就会跟着变(0.001 也一样),**不能当标定值**;`0.2` 在两个方向各留余量
  (离 0 有 0.2,离正常桶下沿 0.2894 有 0.089),且不与 `retrieval_score_threshold`(0.25)
  / `evidence_min_score`(0.15)**同值**,免得三个旋钮看起来像一个。
  另:`0.42` **不在**平台段内 —— 网格上最接近它的一点(0.425)误杀率是 **0.263**(63 条),
  比 0.175 差,**所以 0.42 这个占位值确实是错的,这次改动有实质效果**(不是换个数字)。

  > 平台段的**下沿是 0、上沿是 0.2894**:`(0, 0.2894]` 内的取值在两列读数上完全等价。
  > 取 `0.2` 取的是**平台内**一点,不是"最优解"—— 这份评估集给不出最优解。
- **第二条:误杀率的 0.175 与阈值无关。** 42 条被误杀的正常问题**检索返回空**,
  闸的 `bool(evidence)` 在**任何**阈值下都拦它们(连 `0` 也拦)。
  ⇒ 「**每 6 条口语化提问就有 1 条拿不到任何证据、直接吃兜底话术(17.5%)**」是
  **检索覆盖率/口语问法**的问题,**不是这个旋钮的问题** —— 阈值标定得再准也救不了它。
  退化判据里的「误杀率 ≤ 0.05」在这份评估集上**不可达**。
  分布上它**集中在口语桶**:`C_colloquial` 21/60 = **35%** 被误杀,而 `B_model` 只有 3/60。
  ⇒ **已单独进 §13 风险表第 11 条**(它是本章**量出来的一个既有缺陷**,不是"本章没做好")。
- **第三条:拦截率的 0.967 几乎全部由「检索为空」挣来 —— 公式的判别带根本没被行使。**
  60 条 `D_absent` 里 **58 条空证据**;只有 2 条有证据(D7 conf 0.6978 / D28 conf 0.542)。
  要把拦截率从 0.967 提到 1.000 得把阈值抬到 **0.7 以上**,代价是误杀 110/240 = **45.8%**
  —— 这个交换比不成立。置信度公式真正该分的那一段(弱但有证据)在这份评估集里**零覆盖**
  (见上面的平台段)。⇒ **本章验证的是检索器,不是那个三信号公式。**
  与第二条是同一枚硬币的两面,**一并进 §13 风险表第 12 条**。
- **与 §15.8 的关系(闭环一条)**:§15.8 记账说「`confidence = 0.084` 在占位 0.42 下仍被拦,
  这个 fail-closed 性质在**阈值标定成更小的值时**就会失效」。**这条没有兑现** ——
  0.2 仍远高于 0.084,纯噪声分照样被拦。**保持不变**(fail-closed 的那道防线**仍不是承重的**)。
- **brief 里那段脚本不能直接跑,三处都改了** —— **完整记账见 §15.10**(不在此处重复,
  免得同一条规则两处漂移)。一句摘要:`split(",")` 把 125 条带引号逗号的用例解析歪、
  让负例从 **60 条静默变成 44 条**;`emit` 那行是 SyntaxError;缺 `sys.path.insert`。
  改动都在 `scripts/calibrate_evidence.py`(另:删掉 docstring 里**并不存在**的 `--save`,
  补齐表头承诺的「拦截数/误杀数」两列)。
- **测试连带改了一处**:`tests/test_config_ch09.py::test_ch09_fields_have_defaults` 里
  `== 0.42` 改成 `== 0.2`。**这不是"为了绿而放松断言"** —— 那条 `==` 自己上面的注释就写着
  「真值由 calibrate_evidence.py 扫出来,**届时这个 `==` 要一起改**」;断言形式一字未动
  (仍是逐字钉住一个数),守的性质(默认值不被静默改掉)没有变弱。变异验证过:
  把它改回 `0.42`,该用例 **1 failed**(`assert 0.2 == 0.42`),锚点命中数恰好 1。
- **全量非 db**:`735 passed, 61 deselected`(与改动前基线一致,无回归)。
- **一句话回退**:把 `app/config.py` 的 `default=0.2` 改回 `0.42`(连同上面那条 `==`)。
  按平台段的读法,回退**不影响**本章评估集上的任何读数。

### 15.10 订正:T8 brief 里那段标定脚本**没对着真实数据跑过** —— 三处缺陷(一处静默、两处当场就炸)

- **计划里怎么写的**:T8 的 brief 直接给了一段可复制的 `scripts/calibrate_evidence.py`,
  并说「跑它,把选中的值写回配置」。**那段脚本逐字跑不了**,三处都实测过:
  1. **`line.split(",")` 把用例文件解析错了(静默错,最重的一处)。**
     `evals/测试集.md` 是**带引号逗号的真 CSV** —— 300 条里 **125 条**的 query 形如
     `"满多少钱包邮,不满怎么收运费"`。`split(",")` 把它们截断成 `"满多少钱包邮`
     (带引号、少一半),并把**第 6 列错位到第 7 列** ⇒ 「应拒答=是」被数成 **44 条**
     (真值 **60**)。
     **后果:16 条负例样本静默消失,且全部混进正常桶** ⇒ 同时**压低拦截率、抬高误杀率**,
     两个数都错,而**脚本不报任何错**、输出格式完全正常。本仓那条
     「假绿测试」的头号形态在这里换了个壳:**不是断言没判别力,是量具把分母读错了**。
     改用 `csv` 模块 + **表头逐字校验**(列一换位置就响亮地炸)+ 一条
     「负例只该来自 `D_absent`」的守卫。
  2. **`emit("没有"拦截率≥0.6"的解…")` 是 `SyntaxError`** —— 双引号串里裸的 ASCII
     双引号(用 `compile()` 抽出代码块逐字验证过:报错落在 `≥`,而 `≥` 本身合法,
     错的是引号 —— **报错位置指向别处**这一条也一并复现了)。
  3. **缺 `sys.path.insert(0, repo_root)`** —— `python scripts/x.py` 时 `sys.path[0]`
     是 `scripts/`、不是仓库根,`from app.config import …` 直接 `ImportError`。
     本仓 `scripts/` 下**每一个**脚本都有这一行(`scripts/intent_cost.py:47` 还专门写了注释)。
- **归因(按体例)**:**计划里的脚本没对着真实数据跑过** —— 第 1 条只要拿真文件跑一次
  (甚至只要数一下 44 ≠ 60)就会暴露,而它**不是"写错了"、是"没跑过"**:同一份计划里
  §4.2 明写「D_absent 应拒答桶 60 条」,而脚本读出来是 44 条,两处**没有被拿着对读**。
  与 §15.8 同族(§15.1 / §15.7 之后**第四次**),触发机制也相同:**写下来的是"看起来对",
  没跑;**而且**同一份计划里两处数字/语义互相矛盾,没人对读。
- **三处里最该记的是第 1 条**:**它是静默错的又一次**,与 T4 那条 `r.get("tags")`
  (值长得像对的、只是永远拿不到)是**同一个形状** —— 失败方式不是"报错让人看见",
  而是"**跑起来了、数变了**"。
  **判据**(本仓已有):凡是"计划里给的、可复制的代码",**要么对着真实输入跑过,
  要么显式标注'未跑'**。
  > ⚠️ **分清"静默"与"响亮"再引用**:上面三条里**只有第 1 条是静默错**。
  > 第 2 条是 `SyntaxError`、第 3 条是 `ImportError` —— **两条都是当场就炸的**,
  > 不存在"悄悄把数读歪"。把三条笼统写成"都是静默错"**是错的**,此处按条记清。
  >
  > **留痕这一条为什么值得写**:复盘时**审查者一度把这三条统称为"静默错"**
  > (原话写作"第三条是又一次静默错"),实现者按证据顶回去、只把 T4 的亲缘关系
  > 挂到第 1 条上,双方核对后**审查者确认按实现者这版为准**。
  > 记这一笔的用处**不是分对错**:是**给上面那段加一道免疫** —— 免得后来的读者
  > (或下一次复盘)照着那句错话把三条重新并成一种记法。
  > 本仓同类先例:CLAUDE.md 记过「③ 曾被写小成『六个』(审查员列清单时漏了
  > `tool_call_id`)」—— **审查者的清单也会错,而错的清单必须被订正本身记下来。**
- **证据**:`csv` vs `split` 的字段数分布实测 `{6: 175, 7: 125}`、「应拒答=是」实测
  **44**(真值 60);brief 代码块的 `compile()` 报 `SYNTAXERROR at code line 94`;
  改后脚本(`scripts/calibrate_evidence.py`)三次运行读数逐位相同。
- **范围限定(要不要外扩,不在本章)**:`scripts/run_eval.py`(ch04)里也有同款的
  `line.split(",")`。**它是否也吃引号逗号、是否也把某个分母读错,本章未核** ——
  已交给 T17 复核。这条缺陷面**可能比本章更值钱**,但**不许**在没跑之前当成结论。

### 15.11 订正:§5.6 的「只在一处重试」被**取消** —— 设计方写下的重试,**没让它在契约里活下来**

- **最初写的是什么**:§5.6 的标题是「协议违规:**只在一处重试**」,表第一行写
  「首键不是 `useful` ⇒ **中止本轮,重试一次**(零 emit);再违规 ⇒ 降级」。
- **实际是什么**:T10 落地的是 **T9 的三行调用方契约表**(task-9-report.md §11-③)
  —— **第 2 行 = 降级,没有重试**。首键违规与其余违规走**同一条**降级路。
  **重试从未实现、也从未被测**(全仓 `grep` 不到任何重试逻辑)。
- **为什么会写错(按体例归因:这是设计方的账)**:
  「重试一次」是**写 §5.6 时**设计的,那时**调用方契约还不存在**;而 T9 用**六轮评审
  + 七个变异**把契约固化成了三行表,重试没有被收进去。
  ⇒ 一份**被更狠地验过**的契约,盖过了一句**从未落地也从未被验**的设计。
  **不是实现者偏离设计,是设计的两半打架、而且只有一半被验过。**
- **裁定(控制器 2026-09-23):删掉重试,按表来。** 理由:①第 2 行的降级路径**本身是安全的**
  (退回 ch08 的纯文本行为);②重试要在一条**频率未测**的路径上**再加一次模型调用** ——
  用一个没量过的动作去换一个本来就安全的退路,收益说不清。
- **代价如实记(用户可见)**:首键违规那一支会把 **`raw`(协议 JSON 原文)当纯文本显示
  给用户**。丑,但可预测,`trace` 里有 `agent:protocol_violation`、日志里有告警。
  **已写进 §13 的风险表(第 13 行)** —— 用户日后若要求「更好看」,首选修法是把重试加回来。
- **证据**(两半,都不是推断):
  1. **文法上**:`app/agent/nodes.py` 的 `_finish_verdict` 里**没有**任何重试分支
     (契约表第 2 行 = `parts.clear()` + 发一帧 `raw` + `trace` 记违约);
     `_stream_round` 全仓只有**三个**调用点(续跑 / 循环内 / 收尾轮),
     `grep -n "重试" app/agent/nodes.py` 的三处命中**全是工具重试的注释**、
     与协议无关。
  2. **实测**(探针 `.superpowers/probe_t10_noretry.py`,数 `astream` 被调了几次):

     ```
     首键违规(first_key=answer): astream 调用 1 次;trace=['agent:protocol_violation', 'agent:converged']
     首字符不是 { (plain):      astream 调用 1 次;trace=['agent:protocol_violation', 'agent:converged']
     合规协议:                  astream 调用 1 次;trace=['agent:converged']
     ```

     ⇒ **首键违规只跑一轮、直接降级**(重试若存在,第一个数会是 2)。
     顺带把 §5.6 那条 `tool_calls` 订正也再证一遍:第三条(合规)不记违约,
     第二条(PLAIN 且**没有** `tool_calls`)记。
  3. 用例侧:`tests/test_agent_protocol.py::test_protocol_violation_degrades_to_the_raw_text_flushed_once`
     断的正是「一次降级、恰好一帧 `raw`」。
- **一处被这一裁定顺带订正的**:§12.1 原先那条「首键违规 ⇒ 恰好重试 **1** 次
  (模型替身记录调用次数)」**已删** —— 它描述的是一个**从未实现**的行为,
  留着的唯一后果是**让后来人以为它测过**。

### 15.12 订正:T16b —— 飞轮的**两个墙钟上界**,而「单槽」只关了一半

（T16 走查照出来的运行时故障 + T16b 的修复,2026-09-24。**§8.3 提到的「照 ch04 那套」原文不改。**）

- **最初的样子(实现期第 11 阶段之前)**:`app/llm.py` 的 `ChatOpenAI(...)` **没有传 `timeout`**,
  `_run_flywheel` 也没有任何**整条任务**的寿命上界 —— 唯一终结一个任务的东西是「它自己跑完」。
- **实际发生了什么**(T16 走查现场,原文见 `task-16-report.md` §6-C1):**9 个任务里卡了 3 个**
  (`7e24193e0520` / `21f7dad5859d` / `f40f9075cb2e`),分别盯到 **666 / 245 / 382 秒**
  仍是 `running`、窗口内**零日志**;MySQL 那侧是一条**开着却空转的事务**
  (`trx_rows_locked=0`、连接状态 `Sleep`)⇒ 不是锁等待、不是慢查询,是**在等一个永远不回来的
  模型响应**(同一时刻服务对 443 有 `CLOSE_WAIT`)。**后果**:运行槽被永久占着,
  此后**每一次**手动触发都是 409 —— 连「手动那根杠杆」也拿不到槽,**唯一出路是重启客服服务**。
- **根因(两条,缺一不成)**:
  1. **不传 `timeout` ≠ 用 SDK 的默认值。** langchain-openai 1.6.2 把 `request_timeout=None`
     **原样**交给 openai SDK,而 SDK 对「**显式给的 None**」的处理是**不设超时** ——
     不是它自己的 `DEFAULT_TIMEOUT = Timeout(connect=5.0, read=600, ...)`。
     实测 `model.root_async_client._client.timeout` 是 `Timeout(timeout=None)`,**四相全 None**,
     连 DNS 与握手都没有上界(再下一层 httpcore2 同样把 `None` 直接交给 `connect_tcp`/`start_tls`)。
  2. **`finally` 只在异常时执行。** 挂起**不是异常** ⇒ 取消/收尾代码永不运行 ⇒ 槽永不释放。
- **为什么会写错**:**「用了库的默认值」这个假设从来没被验证过**,而它恰好是反的
  (库把「没给」翻译成了「不要超时」)。这是本仓那条「**语言/库 X 在情况 Z 下表现 Y,要么带证据
  要么标注未验证**」的第 N 次复现 —— 只是这一次,错误的代价是**服务不可用**,而不是一个读数。
- **订正后的形态**(两处,各一个设置项):
  - `app/llm.py` 显式传 `timeout=settings.llm_timeout_seconds`(默认 **60**,**每一次往返**的上界);
  - `app/flywheel/tasks.py` 的 `_run_flywheel` 用 `asyncio.timeout(settings.flywheel_job_timeout_seconds)`
    (默认 **300**,**整条任务**的上界;被它杀掉的那一批**一行都不落地** —— 池子靠
    `matched_review_id IS NULL` 幂等,下一轮会重新吃到)。
  - ⚠️ 一个**标量**会把 SDK 原本的 `connect=5` 一并换成 60 ⇒ 连接阶段**放松 12 倍**
    (修前是 ∞,所以不是回归);要保住它得写成**四元组**(**二元组会让 write/pool 落回 `None`**)。
- **真机复验**(黑洞监听端口 + 真 `ChatOpenAI`,证据见 `task-16b-debug-report.md`):
  | 配置 | 结果 |
  |---|---|
  | `timeout=1.0` | **4.25s** 抛 `OpenAITimeoutError`(3 次尝试 + 两次退避) |
  | `timeout=1.0, max_retries=0` | **1.01s** 抛同款 |
  | 黑洞 + `llm_timeout=1.0` | job 寿命 **5.94s** → `done`,行留池、下一轮重试(**基础设施故障没有伪装成成功**) |
  | 黑洞 + `llm_timeout=5.0` + `job_timeout=2.0` | job 寿命 **3.45s** → `failed`,**槽即刻可用** |

- **⚠️ 范围限定(必须与上面一起读):那个「单槽」只关了一半。**
  `JobStore`(ch04)只有**一个** `running` 槽,**三个任务共用**:`vectorize` / `mine`(ch04)与
  `flywheel`(本章)。本章只给**其中一个**加了整条任务的墙钟上界;**`vectorize` 与 `mine`
  至今没有任何死线**,ch04 管理台的 `pollJob` 也**没有轮询上界** ⇒
  「任务卡住 ⇒ 槽位永久占死 ⇒ 只能重启服务」在那两个任务上**依然敞着**。
  **不许把本节读成「单槽问题已修复」** —— 准确的说法是
  **「飞轮这一个任务有了寿命上界」**。
- **一处顺带被证否的旧说法**:T16 报告 §6-C1 里那句「`app/llm.py` 没有配任何 `timeout`/`max_retries`」
  在修复后**不再成立**(它记的是修复前的状态,保留原文)。

### 15.13 订正:T18b —— 置信度闸**漏传了落池快照**,以及一条被推翻的「iff」

（T18b,2026-09-24。**§4.3 与 §7.1 的正文不改。**）

- **最初的样子**:§7.1 要求「落池那一行带上当轮召回片段快照」,三处入口里
  `生成自评`(`app/agent/nodes.py:_finish_verdict`)与 👎(`app/api/feedback.py`)都做了,
  **置信度闸那一处没做** —— 它的 `record_low_confidence(...)` **只传到 `reject_reason` 为止**,
  `evidence_snapshot` 走 `None` 默认值。**不是做不到**(形参与 `_snapshot()` 早就都在),
  **是漏写的**一处调用。
- **一处必须订正的旧说法**(它是 T18 复审的产物):有一份报告断言
  「闸 **iff** 召回为空才拦(证据非空 ⇒ confidence ≥ 0.2667 > 0.2)」。**这个断言过强,是错的。**
  反例(生产默认值):单条 `score = 0.16` ⇒ 过得了 `evidence_min_score`(0.15)、
  过不了 `evidence_confidence_threshold`
  (`0.6×0.16 + 0.2×min(1/3,1) + 0.2×0 = 0.163 < 0.2`)⇒ **「手里有一个块、却被拦」真实可达**。
  那一刻今天做的是**把这个块的原文与得分扔掉**,池子里只剩一句问题 ——
  而审核页上「知识库真缺这一块」与「有、但没检到」长得**一模一样**,分开它俩正是快照的用途。
- **为什么会写错**:把**两个独立旋钮**当成一个 —— `evidence_min_score` 管「哪些块算证据」,
  合成分管「分够不够」,**两者可以一个过一个不过**。订正后那段算术**写进了生产代码的注释**。
- **订正后的形态**:闸的落池调用与另两处**逐字同形** ——
  `evidence_snapshot=(_snapshot(evidence, settings=settings) if evidence else None)`;
  空证据传 **`None`(不是 `[]`)** —— 本仓约定 `None` = 当轮确实零召回。
- **证据**:`tests/test_agent_gate_ch09.py` 新增判别性用例(把过滤/kwarg 去掉即红);
  真机读数见验收 ②(零召回那一支)与 T18b 报告。
- **⚠️ 端到端验收对**这一列**没有覆盖**(如实记):验收 ② 问的是**零召回**的问题 ⇒
  没有片段可快照;而零召回那一支里「闸写了这一列」与「闸漏了 `evidence_snapshot=`」落出来的
  **都是 JSON `null`** ⇒ ② 那条断言对它本该抓的那个 bug **不变**。
  守它的是上面那条单测,**不在端到端覆盖内**(验收脚本自己在结尾的「局限」里打这条)。
  探针读数(供对账,别当覆盖证据):修之前闸行 **40 条、非空 0**;接上之后 **43 条、非空 14**。

### 15.14 订正:T19 章级文档同步 —— 四条错账与九处过时

（T19,2026-09-25。**本节是本章最后一次订正,记的全是「文档里写了一个与实测不符的事实」。**）

**① `app/db/models.py` 的 `created_at` 差异,被一次复审判过一次「不是差异」—— 那个判定不可复现。**

- **最初写的**:T5 修复轮在差异清单里加了「③ `created_at` 的默认值措辞:create_all 建出来是
  `DEFAULT (now())`,DDL 是 `DEFAULT CURRENT_TIMESTAMP`」。
- **那次复审说**:实测 `CreateTable` 发的是 `DEFAULT now()`,在 MySQL 8.0.46 上执行得到
  `DEFAULT CURRENT_TIMESTAMP`,**与 DDL 相同** ⇒ 「把不是差异记成了差异」。
- **实际是(2026-09-25 重测,方法:临时库 + 一条路走 `create_all`、另一条路走 `db/ch09.sql`
  原文的 `CREATE TABLE`,核对 `SHOW CREATE TABLE`)**:

  | 路径 | `created_at` 那一行 |
  |---|---|
  | `create_all` | `` `created_at` datetime NOT NULL DEFAULT (now()) `` |
  | `db/ch09.sql` | `` `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP `` |

  **两条路逐字不同 ⇒ 原判(是差异)是对的,那次「订正」才是不可复现的那个。**
- **两边各对一半、所以容易读错**:`CreateTable` **发**出去的确是 `DEFAULT now()`,
  是 **MySQL 把它规范化成 `(now())`**。**「发出去的文本」不等于「`SHOW CREATE TABLE` 读回来的文本」** ——
  那次复审正是从前者往上推后者。判据:**凡「外部系统会怎么规范化我这条声明」,必须跑一遍
  `SHOW CREATE TABLE`。**
- **仍是纯文本差异**(语义都是「插入时取当前时间」),所以**不影响行为**——
  这一条要**两件事分开说**:语义对齐、措辞不同。原先 `EvalRun` 的 docstring 把两者写在同一段里,
  读起来**自相矛盾**(已拆开)。

**② §3.5 的「本仓那 17 处调用」—— 17 这个数**没有数过**。**
评审在 T4 时用 AST 数出来是 **11**(chat.py 6 / extract.py 2 / refund.py 1 / `kb/orchestrate.py` 1 /
`memory/tasks.py` 1)。**11 也是当时的读数**:那之后 T15 加了 `api/review.py` 1 处、
T14 加了 `flywheel/tasks.py` 1 处 ⇒ 本章结束时 `app/` 下共 **12** 处传 openai 那把,
另有 `scripts/intent_cost.py` 的 **2** 处传 **Langfuse** 那把(那正是本条规矩的应用)。
**为什么写错**:引用了一个没数过的数 —— 本仓已有禁令。已在 §3.5 就地订正并留原文备查,
**引用计数时现数一遍**。

**③ 两处**方向相反的**过时 docstring(`app/observability.py`)。**
- `_root_cm` 的 docstring 与 `span()` 的 docstring 都还写着「**工具执行**与知识检索一个 span
  都不会自动出现」—— 而 §15.4 ② 的实测**推翻了前半句**(`spec.tool.ainvoke` **是** LangChain run,
  内置工具**早就**有一条嵌套正确的 `TOOL` 观测),§15.5 又据此**删掉了**手工 `tool:*`。
  ⇒ 「过时」的方向与常见的相反:**不是漏记了新东西,是留着一条已被推翻的旧理由。**
  已改成:**手工 span 今天只剩 `retrieval` 一种形状**(且它的存在理由只剩「检索器不是 LangChain run」)。

**④ `app/db/models.py:147` 的「它今天还不存在」已过时。**
`evidence_snapshot` 的 docstring 里写着另一处写入方是「T12 起的 `POST /api/feedback`
(**它今天还不存在**,形状按这一条对齐)」。**T12 早已交付**(`app/api/feedback.py`,
`POST /api/feedback` 在 `main.py` 里 include),而且 T18b 又**加了第三处**写入方(置信度闸)。
已按「**三处写入方都已在线**」重写。

**⑤ `tests/test_config_ch09.py` 的四条被取代的弱断言。**
`test_langfuse_defaults_are_empty_so_tests_never_go_online` / `test_evidence_weights_are_bounded` /
`test_snapshot_and_flywheel_bounds` / `test_out_of_range_is_rejected` 是 T1 初版写的;
同一文件下半段的 `test_ch09_fields_have_defaults`(逐条 `==`)、两个 parametrize 的越界用例
(且**断出错信息里出现字段名**)**严格更强**,`>= 1` / `0.0 <= x <= 1.0` / `pytest.raises(Exception)`
对一个**没有声明任何边界**的实现同样成立。**处置:保留 + 逐条标注「已被取代,不得引用」**
(不删 —— 本仓的记法是留痕);四条都还在跑,但**不许拿它们当证据**。

**⑥ §9.1 的小标题写「三个端点」,而它**自己的表**里有四行。**
`app/api/review.py` 实测 **4** 条路由(`GET /api/review/queue`、`GET /api/review/{id}`、
`POST .../approve`、`POST .../reject`),模块 docstring 也写「四个」——
**表是对的,标题少了一个**。已就地改标题(原标题的读数保留在此处备查)。
同一形状的还有 §14 里「`app/api/review.py`」的措辞与 `include` 处,核过无误。

**⑦ 两处报告/留痕的出处错账**(不在 spec 里,记在此处备查):
`dev-notes/ch09.md` 里两个标记曾写作「**证据独有的一行**」,而同一段 11 行之下的订正
**自己就否掉了那个理由**(那个命令块**逐字引用了**这两个标记 ⇒ 「引用里不会出现」当场是假的);
`task-17-report.md` §7 的证据索引**只列了 19 个文件**,而 `.superpowers/t17/` 下被 git 跟踪的是
**23** 个(`git ls-files .superpowers/t17/ | wc -l`)。**两处都已订正。**

**⑧ 验收脚本的「候选预算」与「特征串不许含数字」。**
② 每跑一轮会消耗一条 `CAND_Q*`,旧 8 条在 T18 的八次运行里耗尽,2026-09-25 T19 那次跑
用的是**最后一条(`#8`)**。T19 补了 `CAND_Q9..Q16`(现 16 条),并把两条选材规矩写进脚本头:
① marker 必须真的出现在它自己的答案里,**且要用脚本自己的解码器核**
(那是 `chr(int(h,16))` **码点**,不是 UTF-8 字节 —— 极易写混);
② **marker 里不许出现数字** —— 实测:`#8` 的核准答案写「一点八米」,模型转述成「**1.8 米**」,
③ 那句「回复含核准答案的特征串」因此判红,而**产品那一侧每一步都是对的**
(写块 / 向量化 / 召回 / 过闸 / 带引用 / 不再是兜底话术)。**数字是这一族**回声断言**
唯一不稳定的一类 token**,原注释只防了反方向(`12 毫米` → `12毫米`)。
**断言本身一个字没动** —— 改的是**夹具**;`CAND_Q8 / M8` **原样留着当证据**。
补候选时还要**先用生产同款探针核 `hits == 0`**:实测起草的 **22** 条里**有 10 条召得到**
(其中 3 条只比阈值 0.25 高一点点:0.2507 / 0.2615 / 0.2757)⇒
**「看起来库里没写」靠眼睛判不出来。**
