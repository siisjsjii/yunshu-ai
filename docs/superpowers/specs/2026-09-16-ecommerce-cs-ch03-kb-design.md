# 电商智能客服系统 · 第 03 章:知识库与向量语义检索 — 设计文档

> 分段设计已经用户逐段确认通过(2026-09-16,四段:数据层与双写 / 文档切分与语料 / 对话挖知识 / 在线检索与验收)。
> 本文是本章权威设计文档;实现与设计的偏离记录在 §12「实现订正」。

## 1. 目标与验收

把 `query_faq` 的内部实现从关键词查表升级为向量语义检索,**工具的入参出参契约保持不变**。

功能需求(用户原话归纳):

1. **离线建库·文档处理**:知识文档(退货政策、商品 FAQ、售后手册这类 Markdown)按标题层级做结构感知切分;超长内容递归切;块间加重叠且裁到最近句号,不留半截话;大表格按行切时每块都复制表头。
2. **离线建库·对话挖知识**:从历史客服对话挖知识的定时任务,分批喂 LLM 抽取问答对,先进暂存表、再整体去重入库。
3. **落库结构**:每条知识带 `category`、`questions`、`answer` 三字段,拼成一段文本做向量化;商品 FAQ 和挖出来的问答对 questions 填真实问法;政策手册这类没有天然问题的,questions 填所在章节标题、category 填上级标题路径;另带章节路径、内容类型、是否关键条款、前后块指针四类元数据,**只存不进向量**。
4. **双写落库**:MySQL `knowledge_chunks` 当原文权威源、Milvus `knowledge` 集合存向量;先写 MySQL 记「待向量化」,再写 Milvus 拿 vector_id 回填、状态转「已向量化」;按主键幂等,挂了能重跑。
5. **在线检索**:问题向量化后到 Milvus 按相似度取 Top-K,替换 `query_faq` 的关键词查表实现。

验收标准:

1. 「邮费是多少」这类换说法的问题,能召回运费说明并答对。
2. 故意中断建库任务再重跑,漏向量化的块能被捡起补齐。

## 2. 技术栈与版本

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.13.14(既有 venv) | 解释器一律 `.venv/Scripts/python.exe` |
| BGE-M3 | 本地权重 `models/bge-m3/` | **用户已放好完整 HF 快照**(pytorch_model.bin + tokenizer,2.2GB);dense 输出 **1024 维**,官方 model card 核实 |
| 加载库 | FlagEmbedding(版本安装时锁定) | 官方推荐的 BGE-M3 加载方式;若 Py3.13 无法安装,退化方案见 §9(预授权) |
| Milvus | 2.6.x standalone(单容器,内嵌 etcd) | 部署方式经官方文档核实:端口 19530、数据卷持久化;具体镜像 tag 安装时锁定并记账 |
| pymilvus | 2.6.x | 官方兼容表:Milvus 2.6.x ↔ pymilvus 2.6.x;3.0 服务端才需 SDK 3.0.x,保守走 2.6 对 |
| MySQL | 8.0(既有 3307 实例) | 原文权威源,不改 |
| LangChain | 1.4.0(既有) | 对话链路不动 |

**用户裁决(定死)**:嵌入模型 BGE-M3、向量库 Milvus、MySQL 权威源、双写结构、dense 单路。

## 3. 已完成的实测与文档核实(本章设计的地基)

| # | 项 | 结论 |
|---|---|---|
| 1 | `db/ch03.sql` 两张表 | 用户已建:`knowledge_chunks` + `qa_extraction_staging`,SHOW TABLES 确认、均 0 行;DDL 是既定事实,**本章不改表** |
| 2 | `query_faq` 现契约 | `make_query_faq(session)` 闭包工厂;入参 `keyword: str`;出参 `{keyword, count, items: [{question, answer, category}]}` |
| 3 | `.env` / venv | 无任何 embedding 配置;无 pymilvus / FlagEmbedding / torch —— 本章新装 |
| 4 | Milvus 部署 | 未部署(19530 未监听);**用户授权我直接用 Docker 起 standalone** |
| 5 | BGE-M3 官方用法 | `BGEM3FlagModel(路径, use_fp16=...)` 吃本地权重目录;`encode(..., return_dense=True)['dense_vecs']` → 已归一化的 1024 维稠密向量(官方 model card,经 hf-mirror 核实) |
| 6 | Milvus standalone 部署 | 官方文档:单容器内嵌 etcd(`ETCD_USE_EMBED=true` 等)、端口 19530、数据卷 `volumes/milvus` 持久化 |
| 7 | 版本配对 | pymilvus GitHub 官方兼容表:2.6 服务端 ↔ 2.6 SDK;跨大版本不兼容 |
| 8 | Context7 MCP | 本环境没有;以官方文档在线核实替代,已向用户声明 |

## 4. 架构

依赖方向在 ch02 基础上扩展,仍严格单向:

```
api → services → tools → retrieval → {db}
                  ↑
kb(离线管线)→ {db, llm, retrieval}
```

```
app/kb/            离线建库管线(不在请求路径上)
  chunker.py       Markdown 结构感知切分(纯函数,可完全单测)
  ingest.py        语料 → chunk 行(faq 迁移 + Markdown,三元组查重)
  writer.py        双写落库:insert pending → 向量化 → 回填(幂等核心)
  mining.py        对话挖 QA 的 LLM 抽取(json_mode)
app/retrieval/     在线检索组件
  embedder.py      BGE-M3 封装(懒加载单例,进程内复用)
  milvus.py        MilvusClient 封装(ensure_collection / upsert / search / count)
  search.py        KnowledgeRetriever:嵌入 → Milvus Top-K → MySQL 回查 → 阈值过滤
scripts/build_kb.py    离线建库编排(手动跑 / 可挂计划任务)
scripts/mine_qa.py     对话挖知识编排(定时任务本体,独立脚本 + 外部调度 —— 用户裁决)
knowledge/         知识语料(3 份 Markdown,本章新写)
```

- `tools/registry.py` 组装请求工具集时构造 retriever 并注入 `make_query_faq(session, retriever)` —— **tools → retrieval 是新增的合法依赖边**;`services/`、`api/` 不直接碰 retrieval。
- **embedder 与 Milvus 连接必须懒初始化**(首次使用才连):单测不联网不连 Milvus 是硬规矩,`import` 与构造都不能触发加载/连接。

### 4.1 与 ch02 既定决策的关系

- 工具错误分类、重试白名单、`ToolInfrastructureError` 向上抛的语义**全部沿用**:Milvus 连不上、嵌入服务故障都属基础设施故障 → 502,绝不伪装成「查无此知识」回灌模型。
- SSE 协议、对话编排、抽取端点不动。
- ch02 的 `Faq` 表与种子**保留不再被查询**;其 12 条内容迁入 `knowledge_chunks` 后回复面不缩小(§6.4)。

## 5. 接口契约

### 5.1 `query_faq`(对模型,不变)

```
入参:keyword: str(模型从用户原话摘取,语义不变)
出参:JSON 字符串 {"keyword": str, "count": int, "items": [{"question": str, "answer": str, "category": str}]}
```

- `count` = 返回块数,≤ `retrieval_top_k`(默认 3,与 ch02 `FAQ_LIMIT=3` 对齐)。
- **字段名与 JSON 结构一字不变**;语义从「关键词命中的 FAQ 行」变为「语义召回的知识块」:`items[].question` = 该块 `questions` 全文(多个问法含换行),`answer` = 块正文,`category` = 块分类。
- 空关键词 → `ToolNotFound`(文案不变);检索后全被阈值滤掉 → `ToolNotFound`,**保留「如实告知未收录,不要自行编造答案」防线**。

### 5.2 内部组件契约

```
chunker.chunk(markdown_text, *, doc_kind) -> list[Chunk]
Chunk:category / questions / answer / section_path / content_type / is_key_clause

embedder.encode(texts: list[str]) -> list[list[float]]   # 1024 维,同配置下确定
milvus.ensure_collection(dim) / upsert(ids, vectors) / search(vector, top_k) -> [(id, score)] / count()
writer.write_chunks(rows) -> 写 MySQL(pending),三元组查重幂等
writer.vectorize_pending(store, embedder) -> 补齐所有 pending 行(可重跑)
KnowledgeRetriever.search(query) -> list[RetrievedChunk]
```

### 5.3 脚本契约

```
.venv/Scripts/python.exe scripts/build_kb.py    # 建库:语料导入 + faq 迁移 + 向量化补齐;重跑 = 幂等补齐
.venv/Scripts/python.exe scripts/mine_qa.py     # 挖知识:读会话 → LLM 抽 QA → staging → 去重 → 入库(→ 向量化复用 build_kb 的补齐步)
```

## 6. 关键设计决策

### 6.1 Milvus 只当索引,不存文本

集合 `knowledge` 仅两个字段:`id VARCHAR(64)`(主键,值 = `str(MySQL id)`)+ `vector FLOAT_VECTOR(1024)`,IP 度量(dense 已归一化,IP 等价余弦)。**不存 category/questions/answer** —— 检索命中后拿 id 回 MySQL 查原文。理由:DDL 注释明确 MySQL 是「原文权威源」;Milvus 可随时 drop 重建(全表回到 pending → 重跑补齐),不存在双份真相漂移。

### 6.2 主键对齐与幂等(验收 2 的结构基础)

- `vector_id` 回填值 = `str(knowledge_chunks.id)` —— 与 DDL 注释「chunk 主键,与 Milvus 集合主键对齐」一致。
- 双写时序:`INSERT (pending)` → 嵌入 → `Milvus upsert(同 pk 覆盖)` → `UPDATE vector_id + status=done`。
- 两个中断窗口都安全:(a) Milvus 写完、MySQL 回填前挂 → 行仍 pending,重跑重写**同 pk**(upsert 幂等)再回填;(b) MySQL 已 done → 重跑跳过。**重跑 = 重扫 pending 行**,这就是验收 2 的实现。
- 文档重复导入的幂等:`knowledge_chunks` 无 source 列(DDL 已定,不改),以 `category + questions + answer` 三元组全等查重 —— 语料量级小,应用层查重足够;挖知识入库同走此查重。

### 6.3 切分算法(chunker,纯函数)

- **标题层级**:按 `#/##/###/...` 切,维护标题栈;`section_path` = 根到当前节的标题路径(如 `退货政策 > 运费说明`);政策/手册类 `questions` = 所在章节标题,`category` = 上级标题路径(根节自身 → category = 文档标题)。
- **超长递归**:块正文超 `chunk_max_chars`(默认 800,待实测)时,先按空行分段,段仍超长再按句子切;每段/句组独立成块,`section_path` 不变,块间用 prev/next 链接。
- **重叠**:相邻块重叠 `chunk_overlap_chars`(默认 100,待实测);重叠区的**起点必须回退到最近句号之后**(句末标点集:`。!?;` 及省略号),裁不出句号(表格/无标点)则不重叠。不留半截话。
- **表格**:连续 `|` 开头的行是表格;表格块超长按**数据行**切,每块 = 表头行 + 分隔行 + 若干数据行(表头每块复制);表格块不参与句号重叠。
- **关键条款**:源文档中 `<!--key-->` 注释标记的段落,其块 `is_key_clause=1`,其余 0(显式标注,不靠关键词猜)。
- **向量化文本**:`category + questions + answer` 三字段拼接(拼接符与顺序固定,离线在线同一条路径),元数据不进向量。

### 6.4 初始语料(用户裁决:faq 迁移 + 新写文档)

- `knowledge/` 三份新写 Markdown:**退货政策**(含运费说明 —— ch02 种子刻意不含邮费,验收 1 依赖此文档)、**商品 FAQ**、**售后手册**;含若干 `<!--key-->` 标注与一个宽表(表格切块的演示面)。
- 退换货类文档 `content_type=policy`,手册 `manual`,FAQ 类 `faq`。
- faq 迁移:12 条种子 → `questions=question`、`answer`/`category` 沿用、`content_type=faq`、`section_path=NULL`(无章节结构)。

### 6.5 对话挖知识(mine_qa,独立脚本 + 外部调度)

- 源:`messages` 按会话分组取 user 提问与 assistant 应答;不足一轮的会话跳过。
- 分批:每批 `mine_batch_conversations`(默认 5)个会话喂一次 LLM —— 分批防串味、按批追溯(`batch_no`)。
- 抽取:json_mode,**沿用 ch02 两条血泪**:prompt 必须含字面 `JSON`;描述结构不得用裸花括号(`ChatPromptTemplate` 按 f-string 解析)。LLM 同时输出 `category`;`answer` 必须来自对话原文,不许模型补写。解析失败的批次记数不中断整跑。
- 落 staging:`status=extracted`、`source_ref=conversation_id`。
- **整体去重两级**(全部批次抽完后做):① 归一化(去空白、去标点、lower)后 sha256,对 staging 内与已入库 `knowledge_chunks` 精确去重;② 向量近重复:kept 候选的 question 向量对 `knowledge_chunks` 检索,余弦 ≥ `dedupe_threshold`(默认 0.95,待实测)判重复丢弃。
- kept 行 → `content_type=faq` 入 `knowledge_chunks`(复用 writer,自动获得向量化补齐);discarded 留痕;脚本**不**自动清 staging(DDL 注释「可清空」留给人工)。

### 6.6 检索阈值与防线

- score < `retrieval_score_threshold`(默认 0.5,**待实测**)的命中丢弃;全部被滤空 → `ToolNotFound`。dense 单路没有重排兜底,阈值是「不相关也硬凑答案」的唯一闸门,验收时按实测分布调。
- Top-K:`retrieval_top_k=3`(契约与 ch02 对齐)。

### 6.7 错误语义(沿用 ch02,映射到新组件)

| 故障 | 分类 | 表现 |
|---|---|---|
| Milvus 连不上 / 集合缺失 / 搜索失败 | `ToolInfrastructureError` | 502 + 固定文案,过 `redact_api_key` |
| 嵌入模型加载失败 / 推理异常 | `ToolInfrastructureError` | 同上 |
| 检索为空 / 全被阈值滤掉 | `ToolNotFound` | 可恢复,模型如实告知 |
| 建库/挖库脚本内故障 | 直接崩 | 离线任务,人工重跑(幂等保证安全) |

### 6.8 embedder 单例

BGE-M3 权重 2.2GB,进程内只加载一次(`lru_cache` 惰性单例,同 `get_settings` 模式);离线脚本与在线检索共用同一配置(`embedding_max_length`、`embedding_batch_size`),保证离线向量与在线查询向量在同一空间。

## 7. 数据层

### 7.1 两张表(DDL 已由用户建好:`db/ch03.sql`,ORM 严格对齐)

- `knowledge_chunks`:`id`(自增主键)、`category/questions/answer`(进向量)、`section_path/content_type/is_key_clause/prev_chunk_id/next_chunk_id`(元数据,不进向量)、`vector_id`(回填)、`vectorize_status ENUM('pending','done')`、时间戳。ORM 用 `BigInteger` / `Boolean`(TINYINT(1))/ `Enum("pending","done")`。
- `qa_extraction_staging`:`batch_no/source_ref/question/answer/status ENUM('extracted','kept','discarded')`。
- 建表方式:**信任既有 DDL**,ORM 只做映射;db 测试直接对真实表读写(沿用 test_db_models 的 scratch 行 + 按名清理模式)。

### 7.2 Milvus 集合

- 名称 `knowledge`(可配);schema 如 §6.1;索引 AUTOINDEX / IP;`create_collection` 幂等(存在即跳过)。
- Milvus standalone 数据卷持久化;容器与 MySQL 同属「实例由 Docker 提供」,但**本次由我经用户授权直接起**(用户裁决),起法与 tag 记账到 dev-notes。

### 7.3 预期库内容

12 条 faq 迁移 + 3 份 Markdown 切块(估计 30-60 块,以实测为准)+ 挖掘入库若干;验收前 `vectorize_status` 全 `done`。

## 8. 测试与验收

### 8.1 分三层

| 层 | 依赖 | 内容 |
|---|---|---|
| 单元测试 | 无网络、无 MySQL、无 Milvus | chunker 全部规则、ingest 查重逻辑、writer 幂等逻辑(fake store + fake embedder)、retriever 阈值/过滤、mining 的 prompt 结构与解析、契约回归(fake retriever) |
| db 集成(`@pytest.mark.db`) | 真实 MySQL | 两张新表 ORM 往返、ENUM/FK、writer 真实落库 + 中断重跑(Milvus 用 fake 注入) |
| 评估与验收 | 真实 Milvus + BGE-M3 + DeepSeek | 检索评估集、acceptance.sh 验收 5/6 |

### 8.2 必须能「区分正确与错误实现」的断言(假绿防线)

- 重叠起点是句号**之后**(喂一个重叠区中间是句中的样例,断言块首字符恰为句首)。
- 表格切块**每一块**都含表头行(数表头出现次数 == 块数)。
- 递归切分后无块超上限;标题路径层级正确(嵌套三层标题的样例)。
- 幂等:同一语料跑两遍,行数不变、Milvus fake 收到的 pk 集合不变;中断模拟(fake store 在第 N 块抛错)后重跑,pending 清零且每个 pk 恰好 upsert 一次。
- 契约回归:出参 JSON 的 key 集合与类型逐项断言(结构变了要红)。
- 计数器放 `encode` / `upsert` 边界,不进函数体(ch02 教训)。
- 单测中的 `Settings(...)` 一律 `_env_file=None`(ch01 教训)。

### 8.3 检索评估集(替代 TDD 的那一步)

- `evals/retrieval_cases.jsonl`:`{query, expect_contains(章节路径或关键词条), expect_may_query(干扰问法)}`;含换说法案例(「邮费是多少」「寄东西到付谁出钱」「不想要了怎么退」…)与应拒答的干扰项(超纲问题,期望空/阈值滤掉)。
- `evals/run_retrieval_eval.py`:真实链路(Milvus + BGE-M3),hit@K 闭式口径;结果只记 dev-notes,不可引用前先看样本量。

### 8.4 端到端验收(scripts/acceptance.sh 增补)

- 验收 5:「邮费是多少」→ SSE 流拼回,断言回复含运费要点且带工具徽章链路(query_faq 被调)。
- 验收 6(中断重跑):`timeout` 杀 build_kb 于半途 → 重跑至完成 → `SELECT COUNT(*) WHERE vectorize_status='pending'` == 0 且 Milvus `count()` == MySQL done 数。**起服务前先查 8000 端口**(ch02 教训);Milvus 起容器后要等健康再连。

## 9. 待实测项与风险

| 项 | 默认 | 备注 |
|---|---|---|
| `retrieval_score_threshold` | 0.5 | 验收时看真实分布调,改动属设计授权、记账 |
| `dedupe_threshold` | 0.95 | 同上 |
| 块大小 / 重叠 | 800 / 100 字符 | 影响召回粒度,评估集上可调 |
| FlagEmbedding on Py3.13 + Windows | 可装 | **若装不上:退化 = transformers 直载**(AutoModel + CLS 池化 + L2 归一化,与 FlagEmbedding dense 输出数学等价,仍是 BGE-M3、仍是 1024 维)—— 属预授权替代,记账即可,不算换选型 |
| torch cp313 Windows wheel | 有(≥2.6) | 安装时核实 |
| Milvus 镜像 tag | 2.6.x 最新 patch | 安装时锁定记账;embedded etcd 单容器在 Windows Docker Desktop 的卷用 named volume 防 bind mount 权限问题 |
| Milvus 写后立即可查? | 默认假设需要等 | 脚本末尾显式 flush/等待,实测确认 |
| 嵌入 CPU 速度 | 分钟级 | 语料小,可接受 |
| LLM 抽 QA 幻觉 | prompt 约束 + 评估抽查 | answer 必须来自对话原文 |

## 10. 本章不做

- 关键词召回、混合检索(BGE-M3 的 sparse/colbert 输出)、重排 —— **只跑 dense 单路**(用户定死)。
- 多轮 Agent Loop、认证(沿袭 ch02)。
- LangGraph、Langfuse(用户工作要求里点名了,但本章需求未用到,明确不做)。
- staging 自动清空、Milvus 分布式/高可用、增量文档监听(文件 watch)。
- 前端改造(聊天页不动)。

## 11. 配置项(.env 新增,全部可选带默认值)

```bash
# 嵌入与向量库
EMBEDDING_MODEL_PATH=models/bge-m3
EMBEDDING_MAX_LENGTH=1024
EMBEDDING_BATCH_SIZE=16
MILVUS_URI=http://127.0.0.1:19530
MILVUS_COLLECTION=knowledge
# 检索
RETRIEVAL_TOP_K=3
RETRIEVAL_SCORE_THRESHOLD=0.5
DEDUPE_THRESHOLD=0.95
# 切分
CHUNK_MAX_CHARS=800
CHUNK_OVERLAP_CHARS=100
# 挖知识
MINE_BATCH_CONVERSATIONS=5
```

## 12. 实现订正

(实现过程中与本文的偏离,连同原因记录于此。)

### §4 之订正:retrieval 反向引用 `app/tools/errors.py`(2026-09-16,T9)

§4 写的依赖方向是 `tools → retrieval → {db}`,但 T9 要让 retriever 把
Milvus/嵌入故障翻成 `ToolInfrastructureError`,这个类落在 `app/tools/errors.py`。

权衡后**接受这条反向引用**,理由:`tools/errors.py` 是零依赖的**错误词汇表**,
不是工具层逻辑 —— `executor`、`api`、`services` 都在对着它分类,把它当成
「工具层」才是不准确的。反向引用不产生环(pymilvus/Milvus 的导入仍在函数内)。
若要彻底消掉这条边,做法是把两个异常类挪到 `app/errors.py` 再由
`tools/errors.py` 转出(既有 import 全不用改)—— 已列为可选重构,未做。

### §8.2 之订正:三条 LIKE 通配符用例随关键词查表一起删除(2026-09-16,T9)

`test_tools_db.py` 里 `test_query_faq_wildcard_keyword_cannot_match_everything`
/ `test_query_faq_matches_literal_percent_not_everything` /
`test_query_faq_matches_literal_underscore_not_single_char` 三条,钉的是
`make_query_faq` 里 `LIKE ... ESCAPE` 的转义正确性。T9 换实现后该 SQL 路径
**不再存在**,断言失去对象,故删除。

**它们原本防的风险没有消失,而是换了防线**:那三条守的是「工具返回一堆与
问题无关的答案却报 ok=true,模型照着自己编」;现在由 **相似度阈值**
(spec §6.6)承担,由 `test_retrieval_search.py` 的阈值用例与 T11 的检索评估集
继续守。同批搬走的是契约类断言(种子命中/漏召回/文案有界),它们进了
`tests/test_tools_query_faq.py` —— 用替身检索器,不依赖 MySQL,因此能在
`-m "not db"` 的快路径里跑到(留在 db 文件里会被默认跳过,最容易破的接口
反而最少被验)。

### §5.2 之订正:`make_query_faq(session, retriever)` 的 `session` 已无用途(2026-09-16,T9)

`session` 在本工具里已不再被使用(原文回查由 retriever 承担)。按 plan
「工厂签名扩参」的要求保留该形参,以维持与 `make_create_ticket(session, ...)`
一致的工厂形态;若认为冗余,可一句话改成 `make_query_faq(retriever)`
(同时改 `registry.build_tools` 与 `tests/test_api_chat.py` 的 `boom` 桩签名)。

### §8.4 / §7.2 之订正:Milvus 行数核对必须用 `query(count(*))`,不能用 `get_collection_stats`(2026-09-16,T0 实测)

同 pk 三遍 upsert 后:`get_collection_stats` 的 `row_count` 报 9(它数的是 insert 操作,未扣 delete,compaction 前不真实),`query(output_fields=["count(*)"])` 报 3(真值),search 也只返回 3 个唯一 id。**upsert 按 pk 覆盖的幂等语义成立**;验收 6 的「Milvus 数 == MySQL done 数」一律用 `count(*)`。

另两处 T0 实测事实:①upsert 后**必须 `flush`** 才能立查(默认 Bounded 一致性下 search 返回空)——离线写入路径每批 flush;在线检索读的是久远数据,不受影响。②Milvus 容器(2.6.24,embed etcd 单容器)必须显式 `-e DEPLOY_MODE=STANDALONE`,否则启动即 panic「embedded etcd can not be used under distributed mode」(v2.6.24 源码 service_param.go:145 判据即此环境变量;milvus.io 文档上的新版启动脚本默认有入口脚本代设,本镜像入口只有 tini)。
