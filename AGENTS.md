# AGENTS.md

电商智能客服系统(品牌「云枢」,MewHelp)。ZCode 工作指引。**每条约束的完整理由见 `CLAUDE.md`** —— 它是 Claude Code 时代逐条踩坑积累的权威长文,改敏感区域前先读它和 spec。

## 项目状态与范围

- **ch01(纯对话)**:SSE 流式对话 + 结构化抽取,已并入 main。
- **ch02(单轮 Function Calling 查数据)**:已交付 —— 四张 MySQL 表、五个 `@tool`、工具执行器、评估集、端到端验收、聊天页。
- **ch03(知识库 + 向量检索)**:已交付(T0–T13)—— 结构感知切分、语料导入、BGE-M3 嵌入、Milvus 双写幂等、`query_faq` 换向量语义检索、对话挖知识、检索评估集、端到端验收 7 项。设计源:ch03 spec(含 §12 订正)+ `dev-notes/ch03.md`。
- **ch04(知识库管理台)**:已交付(T1–T8 + 在线检索增补)—— 文档查看/在线上传、在线检索(知识块 + 源文档链接)、后台触发向量化与从会话挖知识,独立管理页 `admin.html`。分支 `ch04-kb-console`。设计源:ch04 spec(含 §12 订正)+ `dev-notes/ch04.md`。
- **ch04 增补(混合检索 + 重排 + 评估)**:已交付 —— Milvus BM25(chinese analyzer)+ hybrid_search RRF + bge-reranker-v2-m3 重排;生成 QC(自评拒答落 `low_confidence_questions` 池 + 引用帧 + 负面知识 prompt);四策略评估 `scripts/run_eval.py` 读 `evals/测试集.md` 出 Recall@K/MRR/置信度;前端菜单分入库/评测 + 聊天页引用可点 + 👍/👎。设计源:ch04 增补 spec(含 §13 订正)。
- **ch08(工具系统:注册中心 + MCP + 写操作确认流)**:已交付(分支 `ch08-tool-registry`,T1–T12)—— 写死的五个 `@tool` 换成**即插即用的工具系统**:`ToolSpec`(名/用途/**原始 JSON Schema**/读写/来源)+ 内置**包内自动发现** + 唯一 JSON Schema 校验器 + 三态权限闸 + 唯一执行点 + `tool_audit_logs` 审计表;两个**自建业务 MCP Server**(8101/8102,Streamable HTTP)+ 每请求现问现拿、单 Server 挂了降级的客户端;建工单改成**确认流**(`agent` 停循环 → `interrupt()` 弹卡片 → `resume` 续跑 → 执行/拒绝)。设计源:ch08 spec(含 §15 订正)+ `dev-notes/ch08.md`。**⚠️ 未达成项见 CLAUDE.md「已知问题与未达成项」**(「缺必填项就主动追问」没实现;写路径超时会往 error 帧吐裸 SQLAlchemy 文本)。
- **明确不做**:多轮 Agent Loop、认证;ch03 不做关键词召回/混合检索/重排(spec §10 定死只跑 dense 单路);ch04 不做文档删除/编辑、任务持久化、并发任务队列;ch08 不做 Skill 机制、接更多外部系统、工具的**热重载**(改我们自己的代码不重启 —— 只到「新增一个内置文件」为止)。
- 技术栈:Python 3.13 + FastAPI + LangChain 1.4(`langchain-openai`)+ DeepSeek(OpenAI 兼容网关)+ MySQL(`asyncmy`)+ ch03 新增 **BGE-M3 本地权重**(`models/bge-m3/`,2.2GB,已 gitignore)+ **Milvus 2.6 standalone**。`.venv` 已建好,一律用 `.venv/Scripts/python.exe`。

## 高频命令

```bash
.venv/Scripts/python.exe -m pytest                # 全部测试(含 db 标记,需 MySQL)
.venv/Scripts/python.exe -m pytest -m "not db"    # 跳过需要数据库的
.venv/Scripts/python.exe -m pytest tests/test_trim.py::test_xxx   # 单条
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000      # 起服务
.venv/Scripts/python.exe evals/run_tool_selection_eval.py         # 需真实 key + MySQL
.venv/Scripts/python.exe scripts/build_kb.py                      # ch03 建库(幂等可重跑)
.venv/Scripts/python.exe scripts/build_kb.py --reindex            # 重建向量索引
.venv/Scripts/python.exe scripts/mine_qa.py                       # ch03 对话挖知识
.venv/Scripts/python.exe evals/run_retrieval_eval.py              # 检索评估(需 Milvus)
.venv/Scripts/python.exe evals/run_retrieval_eval.py --dist       # 看相似度分布定阈值
bash scripts/acceptance.sh                        # 端到端验收 1–9,需服务已启动 + 真实 key
bash scripts/acceptance_ch08.sh                   # ch08 验收 1–6(脚本自己起停两个 MCP Server + 客服服务)
```

ch08 手工起两个业务 MCP Server(验收脚本**不需要**,它自己起):`.venv/Scripts/python.exe -m mcp_servers.logistics`(8101)/ `-m mcp_servers.aftersales`(8102)。ch08 前置:MySQL + 真实 key,**不需要** Milvus。

ch04 管理台:浏览器开 `http://localhost:8000/admin.html`(文档查看/上传、向量化与挖知识按钮)。
ch03 前置:`docker start milvus-standalone`(容器名固定,重建时必须带 `-e DEPLOY_MODE=STANDALONE`,完整命令见 ch03 spec §12)。

ch03 前置:`docker start milvus-standalone`(容器名固定,重建时必须带 `-e DEPLOY_MODE=STANDALONE`,完整命令见 ch03 spec §12)。

- `pytest.ini` 已有 `addopts = -q`,**别再手动加 `-q`**(`-qq` 会隐藏 `N passed`,抹掉验证证据);过滤用 `-m "not db"`。
- 异步测试用 `@pytest.mark.anyio`(backend 由 `tests/conftest.py` 固定为 asyncio),不用 pytest-asyncio。
- MySQL 由 Docker 提供(3307 端口),容器不在仓库管理范围内;`.env` 四个必填:三个 `OPENAI_*` + `DATABASE_URL`。

## 架构边界

依赖方向严格单向:`api → services → {tools, db, memory, prompts, llm}`;ch03 扩展为
`tools → retrieval → db` 与 `kb(离线管线) → {db, llm, retrieval}`。

- `app/retrieval/`(在线):`embedder.py`(BGE-M3 懒加载单例)、`milvus.py`(`MilvusVectorStore`)、`search.py`(`KnowledgeRetriever`,**错误语义的翻译边界**:Milvus/嵌入故障 → `ToolInfrastructureError`)。
- `app/kb/`(离线,不在请求路径上):`chunker.py`(纯函数)、`ingest.py`、`writer.py`(双写幂等)、`mining.py`(挖知识 + `mine_knowledge` 编排)。
- ch04 新增:`app/kb/jobs.py`(JobStore 内存任务注册表)、`app/kb/orchestrate.py`(后台任务:专用线程 + **自建独立 engine**,不用 `get_engine()` 单例)、`app/api/kb.py`(7 端点)。
- **Milvus 只当索引**:集合 `knowledge` 只有 `id VARCHAR(= str(MySQL id))` + `vector`;原文一律回 MySQL 查,故集合可随时 drop 重建。

- **ch08 新增**:`app/tools/` 从「五个 `@tool` 放一个文件」变成工具系统 —— `spec.py`(`ToolSpec` + 唯一校验器)/ `policy.py`(权限表)/ `audit.py`(唯一写口)/ `builtin/`(自动发现,一文件一工具)/ `mock_data.py`(种子数据源)/ `registry.py`(每请求组装)/ `executor.py`(唯一执行点);`app/mcp/client.py`(每请求发现两个业务 Server);`mcp_servers/`(**独立进程**的 Server,不在请求路径上)。
- **`app/tools/` 不依赖 `app/mcp/`,两者靠 `ToolSpec` 交接**:内置与 MCP 工具在注册表里长得一模一样,执行器/权限闸/校验器/审计都不知道手里这条从哪来。想加 MCP 就改 `app/mcp/client.py`,不是执行器。
- `app/memory/`(锁注册表、token 裁剪)与 `app/services/history.py` **不依赖 LangChain**,只碰 `app.schemas.Message` 纯数据类;转 `BaseMessage` 只在 `prompts.py:to_lc_messages` 一处。
- `services/` 的函数**接收 llm 实例作为参数**,不建模块级单例;FastAPI 侧 `Depends` 注入,测试用 `dependency_overrides`。
- `app/schemas.py` 是唯一被到处引用的类型源;`app/sanitize.py:redact_api_key` 是所有出站错误文本的必经出口。
- 聊天页在 `app/static/`(单页,无构建工具链);`main.py` 里 `mount("/")` **必须在 `include_router` 之后**,否则静态目录抢走 `/api/*`。

## 单轮工具编排(核心机制)

```
第一轮:model.bind_tools(TOOLS).astream(msgs)
     ├ 文本 chunk → SSE token 帧
     └ tool_call chunk → 累积
   有 tool_calls → 执行 → tool_result 帧 → 回灌 ToolMessage → 第二轮
第二轮:model.astream(msgs)   ← 不绑 tools
```

- 「只做单轮」靠**第二轮不绑 tools 的结构保证**,不是提示词约定。
- 「单轮」≠「单工具」:模型可并发发多个 `tool_call`,执行器**全部执行、逐个回灌**,少一个上游直接 400。
- SSE 事件协议:`meta` → `token` / `tool_call` → `tool_result` → `done` / `error`。

## 硬约束(不读多文件必踩)

- `app/llm.py` 必须 `use_responses_api=False`:LangChain 1.x 默认走 Responses API,DeepSeek 等兼容网关只实现 Chat Completions;不关会报「模型不存在」的误导性错误。
- `tool_call` 条目必须带 `"type": "tool_call"` 键,否则 `BaseTool.ainvoke` 把整个 dict 当参数去校验 schema,恒报「参数不合法」。测试替身(`FakeChunk` 等)必须补上这个键。
- 抽取只能 `method="json_mode"`(本端点上 `function_calling`/`json_schema` 均 400);提示词必须含字面 `JSON`,且模板里**不得用裸花括号**(会被 f-string 解析)。
- 流式取文本用 `chunk.text`,不是 `chunk.content`(1.x 里后者是 content block 列表)。
- `create_ticket` 的 `conversation_id` 用闭包工厂注入,不用 `InjectedToolArg`(在 langchain-core 1.6.3 上未文档化,直接调用会抛 `ValidationError`)。
- 工具伪随机必须用 `hashlib.sha256` 种子,**禁用内置 `hash()`**(str 每进程随机化,同进程测试测不出来,重启后行为漂移)。
- 重试**由工具的 `kind` 推导,不再是写死的白名单**(ch08):只读工具可重试、写工具**结构性不可重试**(超时未必没执行,重放会建出两张工单);`ToolNotFound`/`ValidationError` 也不重试。默认 `tool_retry_attempts=2`(共 3 次尝试)是 **ch08 起改的全局值**(ch03–ch07 一并变慢),回退:`.env` 里 `TOOL_RETRY_ATTEMPTS=1`。
- 错误语义:`422` 只表示「模型输出无法解析」;上游故障一律 `502` + 固定文案(openai SDK 的 `str(exc)` 是上游响应体原文,原样回显会泄漏密钥)。所有 SSE `error` 帧、`tool_result` 失败 summary、422/502 detail 必须过 `redact_api_key`。
- `ToolInfrastructureError` 必须向上抛,**不能回灌给模型**(数据库故障不能伪装成「订单号查不到」)。
- 会话锁必须在**所有**非流式退出路径释放,包括 `except BaseException`(`CancelledError` 是 `BaseException`);每条路径要有具名测试 —— 漏一条就是永久 409(持锁会话不被 TTL/LRU 淘汰)。
- 预算校验必须在流开始前完成(SSE 首帧 yield 后状态码就改不了了),所以端点是普通 `async def` 手工构造 `EventSourceResponse`。
- 历史裁剪按 `user` 边界切轮(`memory/trim.py`),按 `assistant` 切会拆开 `assistant(tool_calls)`+`tool` 消息对,只在历史变长后偶发 400。

**ch03 新增(每条都对应一次实测故障)**:

- Milvus **VARCHAR 主键必须显式传 `max_length`**,否则报 1101 拒建 —— pymilvus 的快捷建法不会替你补(spec §12)。
- **upsert 后必须 flush 才立查**(默认 Bounded 一致性);**行数一律用 `query(count(*))`**,`get_collection_stats` 的 row_count 未扣 delete、不可信。
- 写 Milvus **成功之后**才改 MySQL 状态并提交;顺序反了,写失败的行会被记成 done、重跑永不补(静默永久缺失)。
- 挖知识的 prompt 必须写明**什么不算知识**(非答案/个案数据/客套/对话状态)。首版没写,把客服「查不到运费」这种非答案挖成了知识,直接把验收 1 的正确答案挤下 top-1。
- 去重基准必须在抽取**之前**取快照 —— 之后取的话本轮产物会把自己全判重(表现为「0 条保留」,像"真没抽出东西")。
- 冷启动:2.2GB 权重首次加载 > 工具超时(10s),故 `app/main.py` lifespan 起后台线程预热(**pytest 下跳过**);retriever 在取消路径上必须 `rollback()` 再抛,否则重试撞 `PendingRollbackError` 被升级成 502。
- `_ensure_model` 有 `threading.Lock`:没锁时预热与首请求并发会**加载两份 2.2GB 且不报错**。

**ch08 新增(工具系统 / MCP,每条都对应一次实测故障)**:

- **`mcp>=1.24,<2` 是 `langchain-mcp-adapters==0.3.2` 的 `Requires-Dist` 钉死的**(解析到 1.30.0);**Context7 整站已迁 v2,这一处以轮子源码为准**。1.x 入口是 `mcp.server.FastMCP`,传输参数是**直接关键字参数**(不是 `settings=Settings(...)`;同级那个同名 `Settings` 字段**全无默认值**)。参考 `mcp_servers/logistics.py`。
- `convert_mcp_tool_to_langchain_tool` 要传 **`connection=`**(传 `session=` 的话 session 一关工具就废);**`handle_tool_errors=False` 必须显式关**,否则 MCP 故障被包成「正常返回」⇒ 执行器把「物流连不上」当**成功**。
- **写操作决议是三态**(`pending`/`approved`/`denied`),不是布尔 —— 布尔会让取消路径再拿到一次 `confirmation_required`;认不出的决议值**响亮地抛且不审计**(不许写一条谎报「用户点了取消」的行)。
- **`retry_count` 记真实重试次数(`attempts_made − 1`),不是配置值。**
- **`turn_messages` 是覆写通道**,承载本轮**全部**消息;续跑/决议节点必须**读旧值再追加**,只返回新那条会**丢掉带 `tool_calls` 的 AIMessage** 而那一轮看起来正常。
- **`agent` 不再是 `_OUTLETS` 成员**(条件出口 `route_after_agent`);无条件接回 `log_turn` 会让挂起那轮**一半写库一半没写**。
- **`pending_write` / `write_decision` 必须连同每轮清零一起落地**(checkpointer 是进程级单例、thread_id=session_id,漏了清零 ⇒ **上一轮批准过的写操作这一轮自动放行**)。
- **`FastMCP.call_tool()` 返回 2-tuple `(list[ContentBlock], dict)`**,而它的**返回注解写的是 `Sequence[ContentBlock] | dict` —— 注解是陷阱**,照它写 `result[0].text` 会 `TypeError`。
- **`isError: true` 在 mcp 1.30.0 里是通用的**:工具名不存在 / 入参校验失败 / 出参 schema 不匹配全汇进同一个 `_make_error_result`,形状一样 ⇒ **不能当「业务性未找到」的同义词**(只能靠文案分辨)。
- **新增**一个内置工具文件**当场生效**;**修改**已有内置模块**仍要重启**(`sys.modules` 缓存)。
- **审计写口让整套测试每轮往 `tool_audit_logs` 写只追加记录**(2026-09-23 实测:一轮全量 **+55 行**)⇒ 对那张表的断言**必须按 `conversation_id` 过滤**,否则偶尔红偶尔绿。
- **本机对已关闭的回环端口要 ~2.05s 才拒绝**(2026-09-23 复测,6 个端口 2.036–2.055s),监听中的端口毫秒级 ⇒ 「两 Server 都没起 ≈4.8s」是**本机性质**,引用秒数必须带「本机实测」;可移植上界只有 `mcp_discovery_timeout_seconds × 2`。

## 测试规矩(头号风险是「假绿测试」)

写断言前先反问:**实现改错了,这条断言的输出会不会不同?**

- 单测构造 `Settings(...)` 必须传 `_env_file=None`(否则根目录真实 `.env` 会把「缺字段应报错」静默补齐);db 测试相反,读真实 `.env`,加 `@pytest.mark.db`。
- 别为自由文本写字符串断言(`temperature=0` 下依然非确定)。
- 工具调用计数器必须放在 `ainvoke` 边界,写在函数体里在参数非法时恒为 0。
- 读回数据库的值要用**新 session**(SQLAlchemy 身份映射是弱引用,同 session 重读可能不打库)。
- 测「响应里没有密钥」时,替身要**真的把密钥写进异常文本**,否则断言恒真。
- 单测全程不联网;只有评估集与验收脚本允许打真实网络。

## 评估与数据口径

- `evals/tool_selection_cases.jsonl`:15 条,**13/15 = 86.7% 可引用**,但引用时须说明:无生产 system prompt 条件下测得、用例集偏弱。**⚠️ 追加限定(ch08,两层)**:① 该数字在**旧工具顺序**下测得,ch08 起顺序变成 `(模块名, 工具名)` 排序;② `query_logistics` 已下线内置、改由 MCP Server 提供,而评估脚本原先只用内置那一半 ⇒ 那 3 条物流用例**在结构上不可能通过**。脚本已改成与生产同款(含 MCP 发现),**但改完之后从未重跑**。⇒ 这是一个「描述旧配置、且其测量脚本已改而从未执行」的数。
- `evals/extract_cases.jsonl` 的 `expected_solution` 分数**不可引用**(关键词是看到输出后才放宽的)。
- `evals/retrieval_cases.jsonl`(ch03):23 条,**闭式口径**(期望片段取自语料逐字原文且须同块命中),不掺主观判断。**⚠️「23/23」是 ch03 的 dense 单路;ch04 换混合+重排后从未复核,现链路 13/23** —— 引用必须说清是哪条链路。**⚠️ 追加限定(ch07 终审):13/23 与下面那个区间都是 `rerank_top_k = 3` 下测的**,而 ch07 把默认值改成了 **5** —— 它们描述的不是现在这条链路,且从未在新默认值下复核。用例自造、干扰项 4 条中 3 条离阈值很远。
- 阈值 `retrieval_score_threshold` = **0.25**(2026-09-20 重定,原 0.58)。0.58 在 **dense 余弦**上标定,ch04 换混合+重排后原值沿用,而重排器输出的是 **sigmoid** 分数,两把尺子不可通约。现链路可用区间 `(0.114, 0.358]`(上界一度写成 `0.389` —— 那是 **top-1 代理量**,偏乐观;真值取「含齐期望片段那块」的分,订正见 ch05 dev-notes 阶段 21),取中点。改它要同时想到两处消费者:`app/tools/registry.py`(过滤块)与 `app/agent/nodes.py` 的置信度闸。
- `evals/results/` 已 gitignore,是历史运行产物。
- **ch05/ch06 的关键实测约束**:`astream(stream_mode="custom")` **会静默吞掉 `interrupt()`**(只从 `updates` 浮出);**resume 会让节点从头重跑**(放 `interrupt()` 的节点里不能有别的事);**未在 `ChatState` 声明的通道写入被静默丢弃**(ch06 因此丢过一整个交付物)。


## Windows + Git Bash 平台陷阱(本机 locale cp936,复发型)

- 含中文的请求体**不能走 curl argv**(MSYS2 按 CP936 重编码),一律 stdin heredoc。**引用这条 ≠ 免疫于它**(ch05 实测:编排者刚在 dispatch 里逐字引用过它,转手自己就踩了)→ 含非 ASCII 的请求优先走 **httpx**,别依赖「我记得避开 argv」。
- 子进程输出显式钉编码:测试给子进程加 `-X utf8`;脚本打印非 ASCII 用 `sys.stdout.buffer.write(...encode("utf-8"))`(`✓`/`✗` 不在 GBK 里,`print` 会崩)。
- 验收不能直接 grep 原始 SSE 流(token 被切开),用 `join_tokens` 拼回再比;也不能用 `grep '[一-龥]'` 检查中文(C locale 下是恒真假断言),按 Python 码点判断。
- **起服务前先查 8000 端口**:残留僵尸进程会让你 curl 到旧代码,得出假红/假绿。需要并行实例时用 `BASE` 覆盖换端口(如 8010),不要 kill 用户自己起的进程。

## 文档即设计源

- `docs/superpowers/specs/*.md`:权威设计文档,内有「实现订正」小节记录代码与设计的偏离;**改行为前先读对应章节**。
- `dev-notes/chNN.md`:按阶段实时留痕(用户原话、产出、被纠偏、翻车返工),**不许收尾一次性补记**。
- `CLAUDE.md`:所有硬约束的完整理由与事故记录。

## 用户的工作方式要求(勿自行放宽)

1. **全程走 Superpowers 流程**,技能自动触发不跳步(本仓库由 Claude Code + Superpowers 开发而来,流程延续)。
2. 非可单测产出(Prompt 模板、数据类)把 TDD 换成**评估集/标注样例验证**;纯 UI 页面例外,Vibe Coding 直做。
3. 涉及具体库/框架/API 的用法,**先用 Context7 MCP 查最新官方文档**,不许凭记忆写。
4. **用户点名的技术选型是定死的**,走不通就停下来问;spec 里标「待实测」的参数按实测改属设计授权,但要记账并告知可回退。
5. 过程实时留痕到 `dev-notes/chNN.md`,每完成一个阶段补一段。
