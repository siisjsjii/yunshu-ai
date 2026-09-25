# ch06 设计:把占位的分流器做成正式版

> 目标流程图:`asserts/ch06workflow.png`(本章的验收形态以它为准)
> 前置:ch05 已把 `/api/chat/stream` 换成 LangGraph 确定性骨架(已合并 `main`)。

## §1 目标与非目标

**目标**:让 Workflow 的关键节点从"占位"变成"正式版" —— 意图识别用 LLM prompt 路线做准,
指代消解/改写/扩写补齐上下文与召回,退款退货与售后走一条**确定性子流程**,
槽位缺失时用**前端卡片回填**而不是让模型猜。

**非目标(本章不做)**:微调小模型 / BERT 做意图分类;跨会话记忆;检索质量(那 10 条与阈值
无关的漏召回)在本章仍不修。

## §2 设计依据:四条实测事实

以下均为 **2026-09-20 在本机 langgraph 1.2.11 上实跑得出**(探针见 `dev-notes/ch06.md`),
**不是从文档推的** —— 它们直接决定了下面的接口形态:

| # | 事实 | 对设计的影响 |
|---|---|---|
| F1 | `astream(stream_mode="custom")` **会把 interrupt 整个吞掉**(一个帧都不吐,run 直接结束,`state.next` 停在待续节点) | 端点必须改成 `["custom","updates"]`。照 ch05 原样写,订单卡片**永远不会出现** |
| F2 | interrupt 从 `updates` 模式浮出:`{'__interrupt__': (Interrupt(value=…, id=…),)}` | 端点在 updates 分支里认这个键,转成前端帧 |
| F3 | resume 时**节点从头重跑**(`interrupt()` 之前的代码会再执行一遍) | `refund_pick_order` 节点里 **interrupt 之外不干任何事**;取订单数据放它**之后**的节点 |
| F4 | 每请求**重新 compile 图** + 同一 checkpointer,resume **能接上**;有 pending interrupt 时改发**普通新消息**,图**从 START 重开**、旧的挂起被丢弃 | 端点的 `build_graph` 每请求重建是可行的;「用户不理卡片直接问别的」是安全路径 |

## §3 图结构

### 3.1 主图

```
START → resolve_references        指代消解 + Query 改写(合成一次 LLM 调用)
      → classify_intent           LLM,强制 JSON {intent, confidence}
      → route_by_intent           纯函数,8 类 → 五出口
          ├ 投诉          → complaint_reply
          ├ 闲聊 / 其他   → fallback_reply
          ├ 商品咨询      → retrieve_knowledge → confidence_gate ─┬ strong → agent
          │                                                      └ weak   → fallback_reply
          ├ 物流 / 订单   → agent
          └ 退款退货 / 售后 → refund_pick_order(子流程入口)
                                        └──────── all → log_turn → END
```

### 3.2 意图八类 → 五出口(路由表写死在代码里)

| 意图 | 出口 | 与 ch05 的差异 |
|---|---|---|
| 物流 | BUSINESS | 不变 |
| 订单 | BUSINESS | 不变 |
| 商品咨询 | KNOWLEDGE | 不变 |
| **退款退货** | **REFUND** | 从 KNOWLEDGE 改走子流程 |
| **售后** | **REFUND** | 从 BUSINESS 改走子流程 |
| 投诉 | COMPLAINT | 不变 |
| 闲聊 | CHITCHAT | 不变 |
| **其他** | FALLBACK | 成为**显式标签**(ch05 已作为解析失败的兜底标签存在) |
| (越界 / JSON 解析失败) | FALLBACK | 不变,且**不得**抛异常给用户 |

`route_by_intent` 仍是**纯函数、无 IO、无模型调用**,可穷举单测(8 类 + 越界 + 缺字段)。

### 3.3 子流程 `refund_flow`

```
refund_pick_order ──缺订单号── interrupt({frame:"order_choice", options:[...]})
      │                              ↑ 前端渲染订单卡片
      │                              └── resume(选中订单号)──┐
      │ 已有订单号                                          │
      ↓                                                     │
refund_fetch_order ←────────────────────────────────────────┘
      ↓  调 query_order 拿这一单(工具,不新写业务能力)
refund_expand_retrieve     Query 扩写 → 多条检索 → 去重合并
      ↓
refund_judge               同一个主力 Agent,只判一次:「这一单能不能退」
      ├ 能退   → refund_offer     发帧:{categories:[...], order_no} → 前端表单
      └ 不能退 → refund_explain   说明具体原因 + 建议联系人工(以 token 帧流出)
```

**节点职责**

| 节点 | 干什么 | 约束 |
|---|---|---|
| `refund_pick_order` | 决定订单号槽位是否有值;**没有就 `interrupt()` 弹卡片** | **除 interrupt 外不干任何事**(F3) |
| `refund_fetch_order` | 用 `query_order` 工具取这一单 | **查不到单** → 走 `refund_explain` 说明并请用户核对订单号,**不进判定**;基础设施故障仍上抛 502 |
| `refund_expand_retrieve` | 扩写 → 多路检索 → 按 `chunk_id` 去重合并 | 扩写失败**降级为单路原问题检索**,不阻断 |
| `refund_judge` | **不新建 Agent**:复用 `app/prompts.py` 的模型消息组装 + 同一个 `model`,做**一次** `ainvoke`(不绑工具、不进 ReAct 循环),产出判定与话术 | 判据只回答「这一单能不能退」;输出解析失败 → 落 `refund_explain` 并如实说明判不了 |
| `refund_offer` / `refund_explain` | 各发一帧或一段文本 | `refund_offer` 只发帧,**不写库**(用户确认后才建单) |

**订单号槽位来源顺序**(`refund_pick_order` 的判据):
① 本轮 `resolved_input` 里含合法订单号 → 直接用;② 会话历史里出现过 → 用最近一个;
③ 都没有 → `interrupt` 弹卡片。

## §4 三个 Prompt(非可单测产出 → 用标注样例/评估集验证)

按项目规矩,这三件**不走 TDD**,各自配一组标注样例跑验证。

### 4.1 指代消解 + Query 改写(合成一次调用)

- 输入:对话历史 + 本轮原话。
- 输出:**一句不依赖上下文也能看懂的完整问题**。
- **关键行为**:问题本身已完整、指代已明确时**原样透传**,不强行改写。
- 失败/解析不出时**原样透传原话**(绝不阻断对话)。

### 4.2 意图识别(prompt 四件套)

1. **七类 + 其他,枚举成选择题**让模型选;
2. **强制 JSON 输出**,字段**只有** `intent` 与 `confidence` 两个;
3. 给**边界 few-shot 样例**(易混对:投诉 vs 售后、商品咨询 vs 订单、闲聊 vs 其他);
4. **留「其他」兜底**,拿不准就归它,**不硬塞进业务意图**。

模型配额:**默认大模型**;`intent_escalation_model` 留空 = 不降级 —— 这是"只留接口"的落点。
`confidence` 照常输出并进 done 帧,用于日志与后续降级路。

### 4.3 Query 扩写(只在退款子流程)

- 输入:已消解的问题 + 这一单的上下文。
- 输出:**强制 JSON,字段只有 `queries` 一个数组**。
- 多条一起检索 → **按 `chunk_id` 去重合并**。
- **只对退款退货/售后做**;商品咨询那条知识路由**不扩**(简单 FAQ 不扩)。
- **扩写发生在检索侧、现查现用**:库里知识只留一份,**不在入库侧拆存多份**。

## §5 接口契约

### 5.1 `/api/chat/stream`(改造)

请求体新增**可选**字段:

```jsonc
{ "session_id": "…", "message": "…" }              // 现行:开一轮
{ "session_id": "…", "resume": {"order_no": "1002"} } // 新增:从挂起点续跑
```

- 带 `resume` → `astream(Command(resume=...), config={thread_id})`;
  不带 → 现行一轮。
- 流模式 `custom` → **`["custom","updates"]`**(F1);`updates` 只用于**识别 interrupt**,
  不把原始 updates 直接透给前端。
- 收到 `__interrupt__` → 转出一帧 `order_choice` → **结束本次响应**。
- **挂起的那一轮不落库**(`log_turn` 未跑);resume 走完才落库。
- **锁**:挂起即释放;resume 时重新取(与 `/api/ticket` 同一把会话锁)。

### 5.2 新增帧

| 帧 | 载荷 | 谁发 |
|---|---|---|
| `order_choice` | `{options: [{order_no, status, product, amount}]}` | 端点由 interrupt 转出 |
| `refund_offer` | `{order_no, categories: [...]}` | `refund_offer` 节点 |

### 5.3 `POST /api/refund`(新增)

```jsonc
// 请求
{ "session_id": "…", "order_no": "1002", "reason_category": "商品质量问题" }
// 响应:落库后的行
{ "id": 1, "conversation_id": "…", "order_no": "1002",
  "reason_category": "商品质量问题", "status": "pending", "created_at": "…" }
```

- `reason_category` **不在固定类目集内 → 422**(请求语义错)。
- 基础设施故障 → **502 + 固定文案**(与既有一致);出站文本一律过 `redact_api_key`。
- 锁纪律、`ensure_conversation` 顺序与 `/api/ticket` 一致。

## §6 数据

### 6.1 `db/ch06.sql` — 新表 `refund_requests`

```sql
CREATE TABLE refund_requests (
  id              BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
  conversation_id VARCHAR(32)  NOT NULL,
  order_no        VARCHAR(32)  NOT NULL,
  reason_category VARCHAR(64)  NOT NULL,
  status          VARCHAR(32)  NOT NULL DEFAULT 'pending',
  created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_refund_conv (conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

配 ORM 模型 `RefundRequest`(与 `db/ch04.sql` + models 的既有做法一致)。
模型的 `status` **同时**写 Python 侧 `default="pending"` 与 `server_default="pending"`。

> ⚠️ **这两层都要留,别"对齐"成一层**(2026-09-21 复评指出):本仓其余模型
> (`qa_extraction_staging` / `knowledge_chunks.vectorize_status`)**只有** Python 侧
> default,没有 `server_default`。`RefundRequest` **是有意的超集** ——
> 正因为**两层都写**,「`init_db.py` 的 `create_all`」与「照 `db/ch06.sql` provision」
> 两条路径产出的列**才一致**。把 `server_default` 删掉,分歧会原样回来。

> **订正(2026-09-21,T2 审查发现)**:最早这里写的是 **`CREATE TABLE IF NOT EXISTS`**,
> 被裁定改掉。原因是一个**静默**的分歧:表实际由 `scripts/init_db.py` 的
> `create_all` 建出,而模型只有 Python 侧 default ⇒ **live 列没有 SQL DEFAULT**;
> 此时 `IF NOT EXISTS` 让 `.sql` 的每一次应用都成为**无声 no-op**,
> 于是 `db/ch06.sql` 成了一张**哪儿都不存在的表**的文档,行为还变成**环境相关**
> (dev 里裸 SQL 插入 `status` 会 1364,照本文件 provision 的环境却没事)。
> `db/ch03.sql` / `db/ch04.sql` 用的都是**普通 `CREATE TABLE`** —— 重复执行会**响亮失败**
> 而不是静默 no-op。本章改回普通建表,**并把 `server_default` 补进模型**,
> 让三条路径(模型 / SQL / live 表)一致。

### 6.2 固定退款原因类目的**单一来源**

放 `app/refund/categories.py`,**两端共用**:`/api/refund` 用它做校验,
`refund_offer` 用它下发。**前端不硬编码类目**(选项从帧里来)。

### 6.3 候选订单(补图里没有的一环)

系统里**没有 orders 表** —— 订单是 `hashlib` 按订单号现算的,不存在"某用户的订单"。
故新增 `app/refund/orders.py`:

- ① 扫本会话历史,取出出现过的合法订单号;
- ② 一个号码都没有时,给**会话固定的一组演示订单**(订单由哈希派生,固定号码可稳定复现);
- **代码注释必须如实写明这是演示数据,不是真实用户订单。**

## §7 模块布局

```
app/agent/state.py        新增通道:order_no / order_data / refund_decision …
app/agent/routing.py      REFUND 出口 + 8 类路由表
app/agent/nodes.py        resolve_references 改造(消解+改写);classify_intent 改八类
app/agent/refund_nodes.py 子流程五个节点
app/agent/graph.py        挂子流程 + 路由表接线
app/refund/categories.py  固定类目(单一来源)
app/refund/orders.py      候选订单(历史扫描 + 演示集)
app/retrieval/expand.py   多路检索 + 去重合并(扩写在检索侧)
app/api/chat.py           resume 分支 + updates 流模式 + interrupt 转帧
app/api/refund.py         POST /api/refund
app/prompts.py            三件 prompt(app/prompts.py 是唯一的消息组装出口)
app/static/index.html     订单卡片 + 退款表单(Vibe Coding)
```

依赖方向仍单向:`api → services → {tools, db, memory, prompts, llm}`,
`agent → refund`(新增边,单向);`refund_nodes` 在 `app/agent/` 内,避免 `refund → agent` 反边。

## §8 错误语义(沿用既有边界,不新立规矩)

- **422** = 请求本身不合约定(类目不在固定集、缺必填字段);
- **502** = 上游/基础设施故障 + 固定文案;
- `ToolInfrastructureError` **必须上抛**,绝不回灌给模型;
- 出站文本(SSE `error` 帧、`tool_result` 失败 `summary`、422/502 detail)一律过 `redact_api_key`;
- 指代消解/扩写/意图识别**失败都不阻断对话** —— 各自降级(原样透传 / 单路检索 / 落其他)。

## §9 配置项

| 项 | 默认 | 说明 |
|---|---|---|
| `intent_model` | 与主力同款大模型 | 意图识别模型 |
| `intent_escalation_model` | **空** | 留空 = 不降级。接口在此,降级路下一章再说 |
| `intent_confidence_threshold` | 待标注集实测 | 低置信的判据(本章用于日志) |
| `query_expansion_max_queries` | 3 | 扩写上限;越界在启动时拒 |

## §10 测试与验收口径

| 层 | 覆盖什么 | 手段 |
|---|---|---|
| 单测 | 路由表 8 类 + 越界 + 缺字段;槽位提取;类目校验;扩写结果去重合并 | 纯函数,TDD |
| 评估集/标注样例 | 指代消解、意图识别、Query 扩写三件 prompt | 项目规矩:非可单测产出把 TDD 换成这个 |
| 图级 | 子流程每步;**interrupt→resume 往返**;**pending 时改发普通消息 = 放弃旧流程** | 替身模型 + 真 checkpointer,不联网 |
| 人工 | 订单卡片点选、退款表单提交 | 浏览器 |

**四条验收标准的落点**

| 验收 | 落点 |
|---|---|
| 1. 多轮用例(物流→退款→物流)每轮意图对、指代补全对 | 指代消解 + 意图识别的标注集 |
| 2. JSON 稳定可解析,怪问题落「其他」 | 意图识别标注集(含刻意构造的怪问题) |
| 3. 「这个能退吗」先补全指代、再走子流程拿订单和政策 | 图级测试 + 端到端 |
| 4. 不带订单号问退款 → 弹订单选择器 → 点选后走完 | 端到端 + 人工;落库证据查 `refund_requests` |

## §11 风险与已知取舍

| 风险 | 处置 |
|---|---|
| `updates` 模式带出的原始载荷可能含模型自由文本 | **只认 `__interrupt__` 键**,其余一律不外推 |
| 挂起期间用户改问别的 → 旧子流程被丢弃 | **实测如此且可接受**(F4);spec 明记为设计行为,不是缺陷 |
| `confidence` 本章不驱动路由 | 明记:它进 done 帧供日志与下一章降级路,本章**不改变路由结果** |
| 子流程与商品咨询的检索重复 | 两者共用同一个 `KnowledgeRetriever`,不新造检索器 |

## §12 实现订正

本章进行中若代码与本文偏离,**逐条记在这里**并写明原因(项目惯例,与 ch03–ch05 同)。

---

## §12 实现订正(2026-09-22 收尾时汇总)

逐条记代码与本文的偏离及原因(项目惯例,与 ch03–ch05 同):

1. **`query_expansion_max_queries` 进了 `app/config.py`,带 `Field(ge=1)`**。
   实测边界:`0 → []`(整条契约要防的**空检索**形状)、**`-1 → ['a']`**
   (Python 负切片**静默丢一条**,不报错)。这是本章唯一一个「不夹住就会静默走偏」的配置项。
2. **`ToolOutcome` 加了 `error_kind`**(`not_found`/`timeout`/`invalid_args`/`tool_missing`)。
   起因:取数节点把**每一个** `ok=False` 都当成「查无此单」,于是**超时**会让用户看到
   「请核对订单号后再试一次」—— **服务端故障被包装成用户输入错**。
   修法是让**唯一知道失败原因的那层**(执行器)记录原因,而不是让调用方按文案反推。
   `app/tools/executor.py` 因此超出 T7 的 Files 块(已披露);**既有调用方一行未动**。
3. **`app/retrieval/search.py::_load_rows` 补了错误翻译**。
   它是该模块**唯一**未翻译的出站口,而模块 docstring 早就承诺「本模块是翻译边界…绝不降级成没搜到」
   —— 属**对自身契约的违约**,不是新设计。不补的话,Milvus/MySQL 故障会被
   `multi_search` 的 `except Exception` 吞成「这条查询失败」。
4. **`POST /api/chat/stream` 对「无待续流程的 resume」返回 409**,本文原本没有这条。
   不加的话用户会看到**裸露的 Python 键名 `'user_input'`**。
   选 409 而非 422:请求体完全合约定,**冲突的是会话状态**(与既有的「会话忙」同族)。
   为此把 `build_graph` 前移到响应之前(`CheckpointTuple` 无 `next` 字段,查待续状态需要编好的图)。
5. **`resume` 载荷接受三种形状**(`str` / `{"order_no": …}` / 对象)。
   本文只写「端点发其中一种」;实测**前端发的是 `{"order_no": …}`**,
   只收裸字符串会让用户看到乱码「没能查到订单 {'order_no': …}」。
6. **done 帧的 `usage` 仍写死 `None`**,未接 `ChatState.usage`。
   接它必须**同时**把 `usage` 加进每轮清零,否则非 Agent 轮会报**上一轮**的 token 数
   (checkpointer 进程级单例 + 未写通道保留旧值)。本章无读者,故保持写死并写明理由。
7. **前端(规格中属 Vibe Coding)两处实现细节**:`streamInto(ctx, body)` 抽出以便 **resume 续进同一气泡**;
   以及**挂起轮不写「(没有返回内容)」占位符** —— 否则那句会粘在 resume 后的回复前面
   (`ctx.body` 在挂起轮恒空,而 `finally` 无条件写它)。
8. **`refund_requests` 的 DDL 去掉 `IF NOT EXISTS`**(§6.1 已就地订正)。
   原因是一个**静默**分歧:`create_all` 建的表没有 SQL DEFAULT,而 `IF NOT EXISTS` 让
   `.sql` 的每次应用都是无声 no-op ⇒ 本文件成了一张**哪儿都不存在的表**的文档。

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
