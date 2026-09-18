# 电商智能客服系统 · 第 04 章:知识库管理台 — 设计文档

> 设计经用户逐点确认(2026-09-17,三点:文档=源 Markdown、触发=后台任务+轮询、前端=独立管理页)。
> 本文是本章权威设计文档;实现与设计的偏离记录在 §12「实现订正」。

## 1. 目标与验收

给 ch03 的知识库加一个**在线管理台**,把原本只靠离线脚本做的三件事搬上网页:

1. **查看文档**:列出 `knowledge/` 下的源 Markdown 文档,点开看原文。
2. **在线上传文档**:网页上传一份 Markdown(选类型),系统注入 `<!--type-->` 标记、落盘、切分入库。
3. **从会话挖知识**:网页一键触发 ch03 的挖知识管线(qa 抽取 + 两级去重),看进度与结果。

验收标准:

1. 上传一份带运费说明的新 Markdown,随后「向量化」,「邮费是多少」能召回新文档里的运费内容(语义检索,不是关键词)。
2. 点「挖知识」,能看到任务进度,结束后展示「抽 N / 保留 K / 丢弃 D」以及保留的问答对;重复触发不会重复入库(幂等)。
3. 后台任务运行时,**聊天接口不被冻结**(请求路径与后台任务隔离)。

## 2. 现状与复用

ch03 已交付(见 ch03 spec / `dev-notes/ch03.md`),本章直接复用,不重写:

| 复用物 | 位置 | 用途 |
|---|---|---|
| `ingest.parse_corpus_file` | `app/kb/ingest.py` | 单文件 Markdown → Chunk 行 |
| `writer.write_chunks` | `app/kb/writer.py` | Chunk 行 → `knowledge_chunks`(pending,三元组幂等) |
| `writer.vectorize_pending` | `app/kb/writer.py` | 补齐全部 pending 行(嵌入 + Milvus 回填) |
| `mining.*`(mine_batch / load_turns / dedupe_pairs / keep_pairs / finalize_staging / …) | `app/kb/mining.py` | 挖知识各步骤 |
| `get_embedder` / `get_vector_store` | `app/retrieval/` | 嵌入 / Milvus 封装(懒加载单例) |
| `create_extract_model` | `app/llm.py` | 挖知识抽取用模型(temperature 0) |

**唯一重构**:把 `scripts/mine_qa.py` 的编排逻辑(`_run`)抽成可复用的异步函数 `app/kb/mining.py:mine_knowledge(...)`,脚本 `mine_qa.py` 与 web 任务共用同一份,消除两处复制。`build_kb` 的向量化部分已是 `writer.vectorize_pending`,无需再抽。

## 3. 架构

### 3.1 依赖边(本章新增一条)

```
api → services → tools → retrieval → db
api ──kb──▶ {db, llm, retrieval}     ← 本章新增:管理面直接调 kb 离线组件
kb ─▶ {db, llm, retrieval}           (既有)
```

- **新增 `api/kb.py → app/kb/`**:ch03 把 `kb` 定义成「离线、不在请求路径上」。本章为管理面打开这条边 —— 但**聊天请求路径不受影响**:上传的切分入库不加载模型;向量化/挖知识都在**后台线程**跑,不在请求协程里。这条边是用户明确授权的(「在线上传 / 在线挖知识」),记录在 §12 订正。

### 3.2 组件

```
app/kb/jobs.py        JobStore:内存任务注册表(threading.Lock 保护),同时只跑一个任务
app/kb/orchestrate.py 后台任务编排:run_vectorize_job / run_mine_job(专用线程 + 独立 engine)
app/api/kb.py         路由:文档 3 个端点 + 任务 4 个端点
app/kb/mining.py      + mine_knowledge(...)(从 scripts/mine_qa.py 抽出)
scripts/mine_qa.py    改成薄 CLI 壳,调 mine_knowledge
app/static/admin.html 独立管理页(vanilla,无构建工具链,与 index.html 同风格)
```

### 3.3 后台任务模型(本章最关键设计)

向量化/挖知识会**同步**加载/使用 2.2GB BGE-M3 与 Milvus,若跑在事件循环里会冻结聊天接口(ch03 记账过的冷启动阻塞的同类问题)。做法:

1. **专用后台线程**:每个任务 `threading.Thread(daemon=True)` 起一个线程,线程内 `asyncio.run(job_coro)`。
2. **独立 async engine**:任务线程**自建** `create_async_engine(settings.database_url)` + `async_sessionmaker`,不用 `get_engine()` 的 lru_cache 单例 —— 那个单例绑在首次使用它的主事件循环上,跨线程复用会出异步连接问题。任务结束 `dispose()`。
3. **串行**:`JobStore` 记当前运行任务;忙时新任务返回 409「已有任务在跑」。同时只跑一个,天然避免两份 2.2GB 权重与 Milvus 并发写。
4. **进度**:任务函数通过可选 `progress(message)` 回调更新 `JobStore` 里的 message/result;API 轮询读回。
5. **生命周期**:任务在内存里,进程重启即消失(管理/演示工具,与 `SessionStore` 同一模式,不持久化)。

## 4. 接口契约

统一前缀 `/api/kb`,JSON 请求/响应。

### 4.1 统计

```
GET /api/kb/stats
   → 200 {total, done, pending, milvus_count}
      total/pending/done = knowledge_chunks 计数;milvus_count = Milvus 集合条数
      (用 MilvusVectorStore.count() 走 query(count(*)) —— ch03 命门,stats 的 row_count 不可信)
```

### 4.2 文档

```
GET  /api/kb/documents
   → 200 [{name, type, title, chunk_count}]
      name=文件名(如 "退货政策.md"), type=content_type, title=首个 H1,
      chunk_count=该文件切出的块数(跑纯函数 chunker 现算,不查库)

GET  /api/kb/documents/{name}
   → 200 {name, type, content}          content=原文(含类型标记)
   → 404 无此文档

POST /api/kb/documents
   请求 {filename, type, content}
   → 201 {name, type, title, chunks_added, chunks_skipped}
   → 409 同名文档已存在 | 422 文件名/类型非法
```

约束:

- `filename` 必须匹配 `^[\w\-]+\.md$`(防路径穿越,只收 Markdown);同名已存在则 409(不覆盖)。
- `type` ∈ {policy, faq, manual}(ch03 spec §6.4 的三类);系统把 `<!--type: {type}-->` 作为首行注入。
- 上传即:落盘 → `parse_corpus_file` → `write_chunks`。**只入库(pending),不向量化** —— 向量化是单独的后台任务(见 4.3)。

### 4.3 任务

```
POST /api/kb/jobs/vectorize    → 201 {job_id}  | 409 已有任务在跑
POST /api/kb/jobs/mine         → 201 {job_id}  | 409 已有任务在跑

GET  /api/kb/jobs              → 200 [{id, type, status, message, created_at, finished_at}]
GET  /api/kb/jobs/{id}         → 200 {id, type, status, message, result, created_at, finished_at}
                                → 404 无此任务
```

`status` ∈ {running, done, failed}。`result` 仅在 done 时有值:

- vectorize:`{processed, total_pending_before, milvus_count, done_rows}`
- mine:`{extracted, kept, discarded, near_dup_dropped, inserted, kept_qa:[{question, answer, category}]}`

任务失败:`status=failed`,`message` 为经 `redact_api_key` 处理的错误文案(与 ch02/ch03 一致,不泄漏上游响应体)。

## 5. 后台任务内部流程

### 5.1 vectorize 任务

```
自建 engine/sessionmaker → 数 pending → writer.vectorize_pending(进度回调逐批报)
→ 数 done + Milvus count → result
```

### 5.2 mine 任务(= scripts/mine_qa.py 的 _run,抽出为 mine_knowledge)

```
自建 engine → 去重基准快照(existing + staging,在抽取**之前**取 —— ch03 命门)
→ batch_conversations 分批 → mine_batch(json_mode)逐批写 staging(progress 报第 N 批)
→ 字面 dedupe_pairs → find_near_duplicate 向量近重复 → keep_pairs 入库
→ finalize_staging 回填 kept/discarded → result(含 kept_qa 列表)
```

每一步的语义与 ch03 完全一致(排除规则 prompt、去重快照时机、kept 语义、幂等),不做改动,只是把编排从脚本搬进可调用函数并接入 progress。

挖知识只入库(kept 行落成 pending),**不向量化** —— 与脚本版一致;前端在 mine 完成后提示「入库 K 条,记得点向量化」,或由用户手动触发 vectorize 任务。

## 6. 前端(app/static/admin.html)

独立单页,复用 `index.html` 的网格纸配色与无构建工具链风格:

- **顶部**:标题「知识库管理」+ 回聊天页链接 + 全局知识库状态(总块数 / done / pending / Milvus 条数,来自 `GET /api/kb/stats`)。
- **文档区**:列表(名/类型徽章/标题/块数),点开展示原文;上传表单(文件名输入 + 类型下拉 + 文件选择或文本粘贴)。
- **任务区**:两个按钮「向量化」「挖知识」;发起后轮询 `GET /api/kb/jobs/{id}`(约 1s 间隔),显示 status/message;done 后渲染 result(挖知识的 kept 问答对列表)。
- 纯静态,无构建;`index.html` 顶栏加「知识库」链接跳 `/admin.html`。

## 7. 数据层

**不改任何表结构**(ch03 DDL 是既定事实,`knowledge_chunks` 无 source 列 —— 文档↔块的映射不靠 DB,而是:文档列表从 `knowledge/` 文件系统来,块数用 chunker 现算)。

- 上传写入 `knowledge/<filename>.md`(文件系统是文档库的权威源)。
- 挖知识沿用 `qa_extraction_staging` + `knowledge_chunks`,写入路径与 ch03 相同。

## 8. 测试与验收

| 层 | 内容 |
|---|---|
| 单测 | `JobStore` 状态机(串行/忙拒/状态流转/404);上传文件名与类型校验;`mine_knowledge` 编排用 fake LLM + fake store(复用 `test_kb_mining` 替身);文档列表/读取用 tmp 目录 |
| db 集成 | 上传 → `write_chunks` 真落库(fake embedder/store);`mine_knowledge` 真 MySQL 端到端(fake LLM + fake store);后台线程独立 engine 正确性(真 MySQL) |
| 冒烟 | 后台任务运行时并发打一次聊天接口,确认不被冻结(encode 锁的代价可接受) |
| 端到端验收 | `scripts/acceptance.sh` 增补:上传文档 → 向量化 → 改说法问题召回新内容(验收 1);挖知识幂等(验收 2) |

## 9. 待实测项与风险

| 项 | 处置 |
|---|---|
| torch CPU 推理跨线程并发(聊天 encode 与后台 encode 同用单例模型) | **给 embedder 加 encode 锁**串行化。代价:任务批量 encode 时,聊天查询可能等当前一批(~数秒)。实测可接受则保留。 |
| pymilvus MilvusClient 跨线程 | 任务串行 + 冒烟验证;若炸则给 `MilvusVectorStore` 加锁。 |
| 后台线程独立 engine 是否干净 | db 集成测试 + 冒烟;任务结束必须 dispose。 |
| 上传后立即可检索 | 依赖 ch03 已解决的「upsert 后 flush」;验收 1 覆盖。 |

## 10. 本章不做

- 认证(沿袭全程)。
- 文档删除 / 编辑 / 重命名、非 Markdown 上传、富文本编辑器。
- 任务持久化(重启即失)、并发任务队列、任务取消。
- 挖知识的 staging 自动清空(ch03 已定留给人工)。

## 11. 配置项

无新增 `.env` 字段。全部复用 ch03 的 `embedding_*` / `milvus_*` / `chunk_*` / `mine_batch_conversations`。

## 12. 实现订正

(实现过程中与本文的偏离,连同原因记录于此。)

### §9 之订正:embedder 加了 `_encode_lock`,encode 全程串行(2026-09-18)

§9 预告「给 embedder 加 encode 锁串行化」。实现时确认:**后台任务与聊天检索共享同一个 torch 模型实例**,并发调同一模型前向(跨线程)不保证安全。故 `encode` 主体包进 `_encode_lock`(与 `_load_lock` 分开),可证伪测试:去锁后 4 线程并发峰值 4,加锁后峰值 1。代价:任务批量 encode 时聊天查询可能等当前一批(约数秒),实测可接受。

### §3.3 之订正:后台任务「独立 engine」落地的形态(2026-09-18)

§3.3 写「任务线程自建 `create_async_engine`,不用 `get_engine()` 的 lru_cache 单例」。实现:`app/kb/orchestrate.py` 里 `_fresh_factory(settings)` 每次任务新建 engine + `async_sessionmaker(expire_on_commit=False)`,任务结束 `dispose()`;`start_job` 的 `store`/`embedder`/`model` 缺省用真实单例、测试注入 fake。db 测试(`test_kb_orchestrate.py`)真 MySQL + fake 三件套验证线程+engine 胶水。

### §4.3 之订正:挖知识结果计数「kept=最终入库数」(2026-09-18)

`mine_knowledge` 返回的 `kept` 定义为**最终保留并入库**的问答对数(不是「活过字面去重」的数),与 `inserted` 一致;`discarded` = 字面去重 + 向量近重复总共丢掉的。`kept_qa` 是 kept 的 `{question,answer,category}` 列表。

### §8 之订正:验收 8/9 的断言都改成了确定性写法(2026-09-18)

- **验收 8(召回)**:初版走「聊天 → grep 工具结果」,但**聊天里的工具选择/关键词抽取非确定**(deepseek 在 temperature=0 下依然非确定,CLAUDE.md 记过),会偶发假红。改成 `kb_recall_hit` 直接用查询原文喂 retriever,确定性验证「新内容进索引且能召回」。
- **验收 9(幂等)**:初版断言「第二次 inserted==0」,但 LLM 每次抽的问答对**不同**(非确定),第二次会抽出第一次没抽到的新对 → inserted>0,这不是重复入库、是新内容。幂等的真实含义是「重复触发不重复入库」= 知识库不产生重复三元组。改成确定性的 `kb_duplicate_triples == 0`(查 `knowledge_chunks` 有无重复 `(category, questions, answer)`)。

### 测试之订正:orchestrate 测试不绑定全局 pending 数(2026-09-18)

`test_kb_orchestrate.py` 后台任务的 `vectorize_pending` 扫**全表** pending,而断言 `processed == 1` / `milvus_count == 1` 假设干净库 —— 库里残留上一轮验收挖矿留下的 pending 行时,全套跑会 flaky 红。改成只断言「至少处理了本测试插入的那行」+ 该行确实 done + vector_id 正确,不绑全局计数。
