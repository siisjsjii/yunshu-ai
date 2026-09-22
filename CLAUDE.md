# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目

电商智能客服系统。

- **ch01(纯对话)** 已合并到 `main`:SSE 流式对话 + 结构化抽取。
- **ch02(Function Calling 查数据)** 交付:模型单轮选工具 → 后端执行 → 回灌 → 作答。含四张 MySQL 表、五个 `@tool`、工具执行器、评估集、端到端验收、聊天页。
- **ch03(知识库 + 向量检索)** 交付(分支 `ch03-kb`):`query_faq` 内部实现从关键词查 `faq` 表换成 **BGE-M3 + Milvus 的语义检索**(工具契约一字未改)。含结构感知切分、语料导入、双写幂等、对话挖知识、检索评估集、端到端验收 7 项。设计源见 ch03 spec(§12 订正最多的一章)。
- **ch04(知识库管理台)** 交付(分支 `ch04-kb-console`):文档查看/在线上传、后台触发向量化与从会话挖知识,独立管理页 `admin.html`。核心是 `app/kb/jobs.py`(JobStore)+ `app/kb/orchestrate.py`(后台任务:专用线程 + **自建独立 engine**)+ `app/api/kb.py`(7 端点)。
- **ch04 增补(混合检索 + 重排 + 评估)** 交付:Milvus BM25(`text` 字段 jieba analyzer + BM25 Function)+ `hybrid_search` RRF + bge-reranker-v2-m3 重排(候选池 20,CPU 性能约束);生成 QC(自评 json_mode → 拒答落 `low_confidence_questions` 池 + `citations` 帧 + 负面知识 prompt);四策略评估 `scripts/run_eval.py`。设计源见 ch04 增补 spec §13。
- **ch05(生产级架构:LangGraph 确定性编排 + 主力 ReAct Agent)** 已合并 `main`:把 `/api/chat/stream` 从「模型单轮选工具」换成 **确定性图骨架** —— `resolve_references → classify_intent → route_by_intent`(纯函数,写死在代码里)→ 五出口;知识类走强制预检索 + 置信度闸再进 Agent,业务类直交 Agent;Agent 是骨架里的**一个节点**(手写 ReAct,第二轮**不绑 tools** 是结构保证)。含 `POST /api/ticket`、聊天页两个独立按钮。设计源见 ch05 spec。
- **ch06(分流器正式版)** 交付(分支 `ch06-intent-router`):把 ch05 里占位的前两个节点做成正式版 —— **八类意图**(含「其他」)+ `{intent, confidence}` 强制 JSON;**指代消解 + Query 改写**(失败原样透传);**退款退货/售后走一条确定性子流程**(取订单 → Query 扩写 + 强制检索政策 → 主力 Agent 判「这一单能不能退」→ 给退款表单或说明原因);**缺订单号时 `interrupt()` 真挂起**,前端渲染订单卡片,点选后 `Command(resume=...)` 同 thread 续跑;`POST /api/refund` + `refund_requests` 表。设计源见 ch06 spec。

**ch03 不做**:关键词召回、混合检索(BGE-M3 的 sparse/colbert)、重排 —— 只跑 dense 单路。**ch04 不做**:文档删除/编辑、任务持久化、并发任务队列。**全程不做**:多轮 Agent Loop、认证。

文档即设计源:`docs/superpowers/specs/` 下的 spec 是权威设计文档(内有「实现订正」小节,记录代码与最初设计的偏离及原因);`dev-notes/chNN.md` 是按阶段实时记录的开发留痕。改行为前先读 spec 对应章节。

## 高频命令

```bash
.venv/Scripts/python.exe -m pytest                             # 全部测试(含 db,需 MySQL)
.venv/Scripts/python.exe -m pytest -m "not db"                 # 只跑不需要数据库的
.venv/Scripts/python.exe -m pytest tests/test_trim.py          # 单个文件
.venv/Scripts/python.exe -m pytest tests/test_trim.py::test_rounds_never_start_with_a_tool_message

.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000    # 起服务;浏览器开 http://localhost:8000
.venv/Scripts/python.exe evals/run_tool_selection_eval.py       # 工具选择评估集,需真实 key + MySQL
bash scripts/acceptance.sh                                      # 端到端验收 1–9,需服务已启动 + 真实 key
# ch04 管理台:浏览器开 http://localhost:8000/admin.html(文档查看/上传、向量化/挖知识按钮)

# ch03(前置:docker start milvus-standalone)
.venv/Scripts/python.exe scripts/build_kb.py                    # 建库;重跑=幂等补齐(中断了直接再跑)
.venv/Scripts/python.exe scripts/build_kb.py --reindex          # 全表打回 pending + 删集合 + 重算
.venv/Scripts/python.exe scripts/mine_qa.py                     # 对话挖知识(独立脚本 + 外部调度)
.venv/Scripts/python.exe evals/run_retrieval_eval.py            # 检索评估,需 Milvus + BGE-M3
.venv/Scripts/python.exe evals/run_retrieval_eval.py --dist     # 打印相似度分布,用于定阈值
```

`pytest.ini` 已设 `testpaths = tests`、`addopts = -q`。异步测试用 `@pytest.mark.anyio`(backend 由 `tests/conftest.py` 固定为 asyncio),不用 pytest-asyncio。

**不要再往命令行加 `-q`。** `addopts` 里已有一个,叠加成 `-qq` 后 pytest 在 `verbosity < -1` 时**整行不打印 `N passed`** —— 失败仍会报,但通过数没了,等于把「我验过」的证据抹掉。要过滤用 `-m "not db"`。

## 架构

依赖方向严格单向:`api → services → {tools, db, memory, prompts, llm}`。

```
app/config.py     pydantic-settings 读 .env;四个必填字段(三个 OPENAI_* + DATABASE_URL),数值项带下界
app/llm.py        ChatOpenAI 工厂,_build 收口全部硬约束
app/prompts.py    System/抽取 Prompt + 消息组装;Message -> BaseMessage 转换的**唯一**出口
app/schemas.py    纯数据模型,唯一被到处引用的类型源
app/db/           base(引擎/会话工厂)、models(四张表)、session(FastAPI 依赖)
app/tools/        business(五个 @tool)、registry(每请求组装)、executor(超时/重试/错误分类)
app/memory/       store.py(锁注册表)、trim.py(token 预算与按整轮裁剪)—— **不依赖 LangChain**
app/services/     chat.py(单轮编排)、extract.py(抽取)、history.py(会话历史读写)
app/api/          chat.py、extract.py
app/static/       聊天页(单页,无构建工具链)
app/retrieval/    ch03 在线检索:embedder.py(BGE-M3 懒加载)、milvus.py、search.py(KnowledgeRetriever)
app/kb/           ch03 离线管线(不在请求路径上):chunker / ingest / writer / mining
knowledge/        知识语料(3 份 Markdown,首行带 <!--type: ...--> 类型标记)
scripts/          build_kb.py、mine_qa.py(离线建库与挖知识)
```

ch03 把依赖方向扩展为 `tools → retrieval → db` 与 `kb → {db, llm, retrieval}`,仍是单向。

两条贯穿性的结构约定:

- **`memory/` 与 `services/history.py` 不依赖 LangChain**,只碰 `app.schemas.Message` 纯数据类。转 `BaseMessage` 是 `prompts.py:to_lc_messages` 一处的职责。
- **`services/` 的函数接收 llm 实例作为参数**,不在模块层建全局单例;FastAPI 侧靠 `Depends` 注入,测试用 `dependency_overrides` 替换。

### ch03 的检索链路

```
离线:knowledge/*.md ──chunker──→ write_chunks(MySQL,pending)
                                    ↓ vectorize_pending(嵌入 + Milvus upsert)
                                MySQL: vector_id + status=done
在线:query_faq(keyword) → KnowledgeRetriever.search
        = 嵌入 → Milvus Top-K → 阈值过滤 → 按 id 回查 MySQL → 按相似度序组装
```

> **2026-09-20:`faq` 表已废弃删除。** 原先离线管线还有一路
> 「faq 表 12 条 ──迁移──→ write_chunks」,那 12 条**已经迁完**并成为
> `knowledge_chunks` 里 `content_type="faq"` 的块(在线检索查的是它 + Milvus)。
> 该表此后没有读写方,**用户已删表**;`Faq` 模型、`faq_migration()`、
> `build_kb.py` 的迁移分支、seed 里的 FAQ_ROWS 一并删除。
> 注意:`content_type="faq"` 与上传类型 `"faq"` 是**取值**,与那张表无关,保留。

- **Milvus 只当索引,不存文本**:集合只有 `id`(= `str(MySQL id)`)与 `vector`;原文一律回 MySQL 取,所以集合可随时 drop 重建(`build_kb --reindex`)。
- **`retrieval/search.py` 是错误语义的翻译边界**:Milvus/嵌入故障 → `ToolInfrastructureError`(502),绝不降级成「没搜到」。`tools/errors.py` 是零依赖的错误词汇表,retrieval 反向引用它已记账(ch03 spec §12)。

### 两条链路

**对话**:`api/chat.py` → `store.lock_for()` 取锁 → `ensure_conversation` + `load_history`(MySQL) → `prepare_turn` 校验预算 → 显式构造 `EventSourceResponse` → `stream_turn` 单轮编排 → **流完整走完后**才 `append_turn` 落库 → `finally` 释放锁。

**抽取**:`/api/extract` 独立于对话,本章未改动。

### 单轮工具编排(本章核心)

```
第一轮:model.bind_tools(TOOLS).astream(msgs)
         ├ 文本 chunk     → 推 SSE token 帧
         └ tool_call chunk → 累积,不外推
     └ 有 tool_calls → tool_call 帧 → 执行 → tool_result 帧 → 回灌 ToolMessage → 第二轮
        无 tool_calls → 已经流完,单次调用即 done
第二轮:model.astream(msgs)   ← **不绑 tools**
```

**「只做单轮」是结构保证,不是提示词约定。** 第二轮不绑 tools,模型在结构上无法再调。实测过「第二轮仍绑 tools 时模型这次没再调」—— 但那是**模型行为,不是保证**,不采用。

注意:**「单轮」不等于「单工具」**。模型可以在同一轮里并发发多个 `tool_call`,执行器**全部执行**并逐个回灌 —— 这是**必须的**,少回灌一个就构成「有 `tool_calls` 没有对应 tool 消息」,上游直接 400。

SSE 事件协议:`meta` → `token` / `tool_call` → `tool_result` → `done` / `error`。

## 若干不读多文件就会踩的硬约束

每一条都对应一个「报错指向别处、极难定位」的故障。

**`use_responses_api=False`**(`app/llm.py`)。LangChain 1.x 的 OpenAI provider 默认走 Responses API,DeepSeek 等兼容网关只实现 Chat Completions。不关掉会调用失败,而报错指向「模型不存在」。

**`tool_call` 条目必须带 `"type": "tool_call"` 键**(`app/tools/`、所有测试替身)。`BaseTool.ainvoke` 判「这是不是工具调用」**只看** `x.get("type") == "tool_call"` 这一个条件;缺键时它把整个 dict 当成**参数**去校验工具 schema,于是每次调用都返回「参数不合法」的可恢复失败。真实链路上模型流出的 chunk 经 LangChain 解析后**是带这个键的**。测试里的 `FakeChunk` 必须补上。

**抽取只能用 `method="json_mode"`**(`app/services/extract.py`)。本项目端点上 `function_calling` 与 `json_schema` 均返回 400,换模型也一样。连带:提示词里**必须出现字面 `JSON` 字样**,且 `EXTRACT_SYSTEM_PROMPT` 描述结构时**不得使用裸花括号**(`ChatPromptTemplate` 按 f-string 解析)。

**流式取文本用 `chunk.text`,不是 `chunk.content`**。1.x 里后者是 content block 列表。

**`create_ticket` 的 `conversation_id` 用闭包工厂,不用 `InjectedToolArg`**。后者对模型隐藏了字段,但值必须另行注入,而该机制在 langchain-core 1.6.3 上无文档 —— 直接用 `tool_call` 调用会抛 `ValidationError`。闭包让参数**根本不在签名里**。

**工具伪随机必须用 `hashlib.sha256` 种子,不能用内置 `hash()`**。`hash()` 对 str 每进程随机化,会让「同一订单号永远返回同样数据」在重启后失效,而**同进程内的测试完全测不出来**(`test_seed_tools_random.py` 里有一条跨进程测试专钉这个)。

**重试用白名单,不是黑名单**。只重试幂等的 `query_*`;**`create_ticket` 永不重试**(超时后重试会建出两张工单)。`ToolNotFound` 与 `ValidationError` 也都不重试 —— 重放同样的参数只会同样失败。

**错误语义边界**:`422` 只表示「模型输出无法解析为约定结构」;上游故障(401/超时/限流/连接失败)一律 `502` + 固定文案。因为 openai SDK 的 `str(exc)` 是**上游响应体原文**,原样回显会泄漏密钥。所有出站错误文本(SSE `error` 帧、`tool_result` 的失败 `summary`、422/502 的 detail)必须过 `app/sanitize.py:redact_api_key`。

**`ToolInfrastructureError` 必须向上抛,不能回灌给模型** —— 数据库故障绝不能被伪装成「你的订单号查不到」。

**锁必须在所有非流式退出路径上释放**,包括 `except BaseException`(`CancelledError` 是 `BaseException`)。**每一条路径都要有具名测试** —— 漏掉的话持锁会话**永久** 409(持锁会话不被 TTL 也不被 LRU 淘汰)。

**预算校验必须在流开始前完成**。SSE 一旦 yield 过第一帧,响应头就发出了,状态码再也改不了 —— 所以端点是普通 `async def` 手工构造 `EventSourceResponse`。

**ch03 · Milvus/嵌入的七条命门**(每条都对应一次实测故障,细节见 ch03 spec §12 与 `dev-notes/ch03.md`):

**VARCHAR 主键必须显式传 `max_length`**。pymilvus 的快捷建法不会替你补,漏了直接 `1101 type param(max_length) should be specified` 拒建。**单测用假 client 测不出这条**(假 client 只验证"我按我以为的形状调用了"),所以凡是对外部系统的调用,单测绿了还要真机冒烟一次。

**`upsert` 后必须 `flush` 才立查**(默认 Bounded 一致性下 search 返回空);**行数一律用 `query(count(*))`**,`get_collection_stats` 的 `row_count` 数的是 insert 操作、未扣 delete,同 pk 三遍 upsert 会报 9 而真值是 3。

**写 Milvus 成功之后才改 MySQL 状态并提交**。顺序反了,写 Milvus 失败的行会被记成 done、重跑永不补齐 —— 静默的永久缺失,不报错,只让某些知识永远检索不到。

**挖知识的 prompt 必须写明「什么不算知识」**(非答案、个案数据、客套、对话状态)。首版没写,把客服「抱歉查不到运费」这种**非答案**挖成了知识,直接把验收 1 的正确答案挤下 top-1 —— 模型的失败被挖进知识库,再教它下次继续失败。

**去重基准必须在抽取之前取快照**。之后取的话,本轮刚写进 staging 的产物会把自己全判成"已存在",结果是 **0 条保留** —— 而这个错误看起来就像"这轮确实没抽出新东西",没有任何报错。

**冷启动的模型加载不在请求路径上**:2.2GB 权重首次加载十几秒 > `tool_timeout_seconds`(10s),冷进程第一个检索请求必然超时;超时取消协程后会话停在未收尾的事务上,而 `query_faq` **在重试白名单里**,重试复用同一 session 直接撞 `PendingRollbackError` → 用户看到 502。两处保障:`app/main.py` lifespan 起后台线程预热(**pytest 下跳过**,单测不加载模型是硬规矩);retriever 在取消路径 `rollback()` 再抛(`except BaseException` —— `CancelledError` 是 BaseException,只抓 `Exception` 正好漏掉)。

**`_ensure_model` 必须有 `threading.Lock`**:没锁时预热线程与首请求并发会加载**两份 2.2GB** 且两边都成功、不报任何错。

**ch04 · 后台任务的三个命门**(细节见 ch04 spec §12 与 `dev-notes/ch04.md`):

**后台任务必须自建 engine,不能用 `get_engine()` 的 lru_cache 单例** —— 那单例绑在首次使用它的主事件循环上,后台线程 `asyncio.run` 里复用会出跨循环的异步连接问题。`orchestrate.py` 每任务 `create_async_engine` + 任务结束 `dispose()`。

**encode 也要锁(`_encode_lock`)**:后台任务与聊天检索共享同一个 torch 模型实例,并发前向跨线程不保证安全。与 `_load_lock` 分开,串行 encode 的代价是任务批量编码时聊天查询可能等当前一批(数秒)。

**验收/测试的确定性**:聊天里的工具选择与关键词抽取非确定(deepseek 在 temperature=0 下亦然),凡是「召回」「挖知识幂等」这类断言**不要经过 LLM 聊天** —— 召回直接查 retriever、幂等直接查「知识库有无重复三元组」。

**裁剪按 `user` 边界切轮,不按 `assistant`**(`memory/trim.py`)。OpenAI 兼容 API 要求 `tool` 消息前面紧跟着带对应 `tool_call_id` 的 `assistant` 消息;按 `assistant` 收轮会把这对切开,而那**只在历史长到触发裁剪时偶发 400**。

**`mount("/")` 必须在 `include_router` 之后**(`app/main.py`),否则静态目录会抢走 `/api/*`。

## 写测试的规矩(本项目血的教训)

**本项目的头号风险是「假绿测试」,不是错误代码。**

ch01 抓到 4 类;ch02 又抓到 **7 条「在它本该禁止的实现下依然通过」的断言**,其中 4 条出自主控写的计划文本、3 条是实现者**先测了一下**才发现的。**两次全章复盘,没有一处缺陷是实现者引入的。**

写断言前先反问:**如果实现改错了,这条断言的输出会不会不同?** 具体到本仓库:

- **`Settings(...)` 构造必须传 `_env_file=None`**(`tests/test_*` 里统一如此)。仓库根有真实 `.env`,`pydantic-settings` 会自动读它 —— 不传的话,「缺字段应报错」的测试会因为 `.env` 把值补上而**静默通过**。
- **db 测试读真实 `.env`,不加 `_env_file=None`**;它们用 `@pytest.mark.db` 标记。
- **别为自由文本写字符串断言**。`deepseek-flash` 在 `temperature=0` 下依然非确定。
- **计数类断言必须实测能区分**:`@tool` 的参数校验发生在**函数体之外**,所以「这个工具被调用了几次」的计数器**必须放在 `ainvoke` 边界**,写在函数体里在参数非法时恒为 0、区分不出任何实现。
- **变量名不等于语义 —— 断言一个字段之前,先去读它是怎么被赋值的(ch05 最刺眼的一处)**。ch05 的验收 5 写了 `[ "$STEPS" -ge 2 ]` 来断「ReAct 不止一步」,`STEPS` 取自 done 帧的 **`agent_steps`**。那个名字读作「步数」,但 `app/agent/nodes.py` 里它是**绑工具轮次的序号且把收敛那一轮也算进去**,**任何一次工具调用都得到 2** —— 于是这条断言在 `tool_calls >= 2` 之外**零判别力**,而且**漏得掉真回归**:把循环改成单轮、让模型在一轮里并发发两个 `tool_call`,照样绿。更刺眼的是:**计划里那段注释自己就写了「收敛那一轮也被算了一步」** —— 事实写下来了,结论却没用到它。改断 trace 里的 `agent:step2`(那个字符串**只在第 2 轮真的发了工具调用时才追加**)。
- **「在**处理之后**注入」是本项目最高频的假绿形态 —— ch06 一章之内中了三次**。三次的形状完全一样:**测试把「已经被处理好的值」喂给被测对象,于是「处理」那一步永远不被验**。
  - T3:API 端点测 502 时,patch 的函数**直接抛 `ToolInfrastructureError`** —— 跳过了「把 `SQLAlchemyError` 翻译成它」那一步,而**那一步正是缺陷所在**(翻译根本不存在,真故障返回 500)。
  - T4:单测断言 `log_turn` 的帧里有 `confidence`,但它是**直接塞进 state 字典**的 —— LangGraph 全程没参与,而 `ChatState` **根本没有这个通道**,真机上写入被静默丢弃、恒为 `None`。
  - T6:检索的「基础设施故障必须上抛」用例注入的也是**已翻译好**的错误;而 `_load_rows` 的裸 `SQLAlchemyError` 会被 `except Exception` 吞成「这条查询失败」。
  **判据**:写这类测试前先问「我注入的这个值,**在被测对象内部还会被处理一次吗**?」——会,就注入**处理之前**的形态。对照写法见 `tests/test_api_ticket.py:151-212`(注入裸 `OperationalError`,走真分类路径)。
- **「语言/库 X 在情况 Z 下表现 Y」这类断言,要么带可复现证据,要么显式标注「未验证」**。ch05 一个 brief 里写过 `confidence_gate:pass`,实际是 `fail`;另一处断言「修好阈值验收 5 就会稳」,实测那个题面在阈值 0 时 top-1 只有 0.089(知识库**根本没覆盖**)。**两条都是先写结论、后没跑**。裁定「这不归本章管」时同理:**不能只看文本授权,还要算这条缺陷会不会卡住本章自己的验收**。
- **读回数据库的值要用新 session**:SQLAlchemy 身份映射持**弱引用**,同 session 重读是否打到库取决于还有没有东西引用着那个 ORM 对象 —— 会变成「靠 refcount 走运」的断言。
- **复述类断言要对着真实来源验**:让替身**真的把密钥写进异常文本**,否则「响应里没有密钥」是恒真的。
- 单测**全程不联网**;评估集与验收脚本才允许打真实网络。

## 数据与产物

- `evals/tool_selection_cases.jsonl` —— 15 条工具选用例,`expected` 是工具名(闭式精确匹配)或 `null`(不该调工具)。
- **工具选择准确率 13/15 = 86.7% 可引用**(闭式枚举,与 ch01 那个被样本拟合的 `expected_solution` 关键词口径不同)。但引用时须一并说明:**该数字是在没有生产 system prompt 的条件下测得的**,且**用例集偏弱**(非 null 的 13 条里 11 条从不失误,信息量主要来自 2 条诱饵)。
- `evals/extract_cases.jsonl`(ch01)的 `expected_solution` 分数**不可引用** —— 关键词是看到输出措辞后才放宽的。
- `evals/retrieval_cases.jsonl`(ch03)—— 23 条(19 换说法正例 + 4 干扰项),闭式口径(期望片段取自语料**逐字原文**且须在**同一块**里全部出现,不掺主观判断)。**⚠️「23/23」那一版是 ch03 的 dense 单路;ch04 换混合+重排后从未复核过,现链路是 13/23** —— 引用时必须说清是哪条链路。用例自造、4 条干扰项里 3 条离阈值很远、不构成压力;**「卖手机」是唯一有信息量的近域硬负例**。
- 阈值 `retrieval_score_threshold` = **0.25**(2026-09-20 重定,原 0.58)。**0.58 是在 dense 余弦分数上标定的**(正例最低 0.609 / 干扰最高 0.560,区间仅 0.049 宽),ch04 换混合+重排时**原值沿用**,而重排器输出的是 **sigmoid** 分数 —— 两把尺子不可通约,0.58 比可用区间上界还高。现链路实测可用区间 **`(0.114, 0.389]`**(宽 0.275),取中点。`dedupe_threshold` = 0.95 **仍是未实测值** —— 真实数据上从未被触发过,不要当成已验证的。
- **`retrieval_score_threshold` 改一次要动两处,它们是一致的**:`app/tools/registry.py` 传给 retriever(过滤块)与 `app/agent/nodes.py` 的置信度闸(取 max 比阈值)。因为 `retrieve_knowledge` 拿到的块**已经**过同一阈值,闸的 `max(scores) >= threshold` 在有 evidence 时几乎恒真 —— **闸的实际效果约等于「检索是否返回非空」**。
- `evals/results/` 被 gitignore,是历史运行产物。

## 平台陷阱(Windows + Git Bash)

本机 locale 是 **cp936**,这个陷阱在 ch02 咬过**三次**,属**复发型**:

- **含中文的请求体不能走 `curl` 的 argv**。MSYS2 会按 CP936 重编码,服务端只回 `error parsing the body`。一律走 stdin heredoc。
- **子进程输出要显式钉编码**。跨进程测试给子进程加 `-X utf8`,否则管道上的 stdout 按 GBK 编码而父进程按 UTF-8 解码,报错表现为 `proc.stdout is None`。
- **脚本打印非 ASCII 要钉输出边界**,用 `sys.stdout.buffer.write(...encode("utf-8"))`,不要依赖控制台 codec(`✓`/`✗` 不在 GBK 里,`print` 会直接崩)。
- **验收断言不能直接 grep 原始 SSE 流**。回复逐 token 推送,`20240915` 会被切成三个独立帧。用 `join_tokens` 拼回后再比对。
- **不要用 `grep '[一-龥]'` 检查中文完好性**:C locale 下 bracket expression 退化成字节区间,对真实 UTF-8 和 mojibake 全部匹配,是个恒真的假断言。脚本里的 `has_cjk` 按 Python 码点判断。
- **起服务前先查端口**:8000 上残留的僵尸进程会让你 curl 到旧代码,从而得出「新代码坏了」的**假红**。ch02 的最终验证就差点栽在这上面。ch05 又遇到一次,且**一次开了两个 uvicorn** —— 见到多个就全部清掉再起,别猜哪个是新的。
- **LangGraph 的两条实测硬约束(ch06,都是「报错指向别处」的类型)**:
  - **`astream(stream_mode="custom")` 会把 `interrupt()` 整个吞掉** —— 一个帧都不吐、run 直接结束、`state.next` 停在待续节点、**不报任何错**。interrupt 只从 **`updates`** 模式浮出(`{'__interrupt__': (Interrupt(value=…),)}`)。ch06 的订单卡片差点因此「永远不出现且不报错」;端点的流模式因此是 `["custom","updates"]`。
  - **`resume` 时节点会从头重跑**(`interrupt()` 之前的代码再执行一遍)。所以**放 `interrupt()` 的节点里不能有别的事** —— 取订单那类有副作用的活必须在它**之后**的节点。ch06 有一条「取订单恰好一次」的用例,计数器放在 **tool 的 `ainvoke` 边界**上(那正是本项目栽过的边界)。
  - **未在 `ChatState` 里声明的通道,写入被静默丢弃**(只 `logger.warning`,不抛)。ch06 因此丢过一整个交付物:节点写了 `confidence`、通道不存在、**单测全绿而生产恒为 `None`**。加通道时**连带加它的每轮清零**(checkpointer 是进程级单例、thread_id=session_id,未写通道**保留上一轮的值**)。
- **引用一条陷阱 ≠ 免疫于它(ch05 实测)**。ch05 的编排者在给子代理的 dispatch 里**逐字引用过**上面那条「含中文的请求体不能走 curl argv」,几分钟后**自己**用 `curl --data-binary "{\"message\":\"退货政策是什么\"}"` 发中文,拿到 `{"detail":"There was an error parsing the body"}`,12 个样本全部没有 done 帧。**结论不是「下次记得」,而是换掉通道**:含非 ASCII 的请求一律走 **httpx 这类替你处理编码的客户端**(或 stdin heredoc),不要依赖「我记得要避开 argv」。

## 工作方式要求

用户对本项目开发有固定要求(见 memory,勿自行放宽):

1. **全程走 Superpowers 流程**,技能自动触发不跳步。
2. **非可单测产出(Prompt 模板、数据类)把 TDD 换成评估集/标注样例验证**,其余步骤照走;纯 UI 页面例外,用 Vibe Coding 直做。
3. **涉及具体库/框架/API 的用法,先用 Context7 MCP 查最新官方文档再动手**,不许凭记忆写。
4. **用户点名的技术选型是定死的** —— 走不通就停下来问,不许自行换方案。spec §9 中预先标注为「待实测」的参数,按实测结果改动属设计授权,但仍需记账并告知用户可一句话回退。
5. **过程实时留痕到 `dev-notes/chNN.md`**,每完成一个阶段就补一段,记四样:用户关键原话、关键产出、被拒绝/被纠偏了什么、翻车与返工。**明确不允许收尾时一次性补记。**
