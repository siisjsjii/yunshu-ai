# ch03 知识库与向量语义检索 · 实施计划

> **For agentic workers:** 按任务序执行,每任务先写测试(红)再实现(绿);ZCode 环境无 superpowers 子代理技能,由主控直接逐任务执行并产出 task brief/report 到 `.superpowers/sdd/2026-09-16-ecommerce-cs-ch03-kb/`。

**Goal:** 把 `query_faq` 从关键词查表升级为向量语义检索(BGE-M3 dense → Milvus Top-K → MySQL 回查原文),工具契约不变;配套离线建库(结构感知切分 + faq 迁移)与对话挖知识管线,双写幂等可重跑。

**Architecture:** 新增 `app/kb/`(离线管线)与 `app/retrieval/`(在线组件);依赖方向 `tools → retrieval → db`、`kb → {db, llm, retrieval}`;Milvus 只当索引(pk=MySQL id + 1024 维向量),原文一律回 MySQL 查。`query_faq` 换内部实现,模型侧契约一字不变。

**Tech Stack:** Python 3.13.14 / FlagEmbedding(BGE-M3 本地权重)/ Milvus 2.6.x standalone(Docker)/ pymilvus 2.6.x / MySQL 8.0(asyncmy)/ LangChain 1.4.0

**Spec:** `docs/superpowers/specs/2026-09-16-ecommerce-cs-ch03-kb-design.md`

## Global Constraints

- 解释器一律 `.venv/Scripts/python.exe`(Windows + Git Bash,locale cp936)。
- **新依赖版本安装时锁定进 `requirements.txt`**:`pymilvus==2.6.*`(与 Milvus 2.6 服务端配对)、`FlagEmbedding`(+ 传递依赖 torch/transformers);若 FlagEmbedding 在 Py3.13 装不上 → 退化 transformers 直载(spec §9 预授权),记账。
- **单测不联网、不连 Milvus、不加载 BGE-M3**:embedder/milvus 一律接口注入,生产实现懒初始化。
- `Settings(...)` 测试构造一律 `_env_file=None`(ch01 教训)。
- 异步测试 `@pytest.mark.anyio`;db 测试 `@pytest.mark.db` 读真实 `.env`。
- **跑测试不加 CLI `-q`**(addopts 已有,叠加成 `-qq` 隐藏通过数)。
- 工具错误语义沿袭 ch02:基础设施故障 → `ToolInfrastructureError`(502 + 固定文案 + `redact_api_key`);查无知识 → `ToolNotFound`。
- 脚本输出非 ASCII 一律 `sys.stdout.buffer.write(...encode("utf-8"))`;含中文的 HTTP 请求体走 stdin heredoc(cp936 教训)。
- 提交信息中文,`type: 描述`;每任务一提交,先红后绿的证据留在测试里。
- **不修改 `db/ch03.sql` 与已建表结构**(DDL 是用户给的既定事实)。
- 本章不做:关键词召回/混合检索/重排、LangGraph、Langfuse(spec §10)。

## Tasks

### T0 · 环境实测与容器(无代码,产出为事实)
- [ ] 开分支 `ch03-kb`。
- [ ] 安装依赖并锁定版本;实测:BGEM3FlagModel 能加载 `models/bge-m3`、encode 输出 1024 维、已归一化(模长≈1)、同文本两次编码一致;FlagEmbedding 装不上则退化 transformers 直载(记账)。
- [ ] 起 Milvus standalone 容器(named volume 持久化),等健康,pymilvus 连通;镜像 tag 记账;`docker start/stop` 命令写进 dev-notes(演示要用手册)。
- [ ] 实测 Milvus:建集合(VARCHAR pk + FLOAT_VECTOR 1024 + IP)、upsert、search、写后可查性(是否需 flush)。

### T1 · Settings 扩展
- [ ] 红:`test_config.py` 新增 —— 13 个新字段全部有默认值、类型正确、`retrieval_top_k>=1`、`retrieval_score_threshold∈[0,1]`、阈值/批量数值项带下界(配置写错要启动时炸)。
- [ ] 绿:`app/config.py` 增字段(spec §11)。

### T2 · ORM 两模型
- [ ] 红:`test_db_models.py` 追加 —— KnowledgeChunk 全字段往返(含 is_key_clause 布尔、prev/next 自引用 FK、ENUM 默认 pending)、QaExtractionStaging 往返(status 三态)。
- [ ] 绿:`app/db/models.py` 增两模型,严格对齐 ch03.sql(列名/类型/默认值)。

### T3 · chunker(纯函数,TDD 主战场)
- [ ] 红:`tests/test_kb_chunker.py` —— 标题栈与 section_path(嵌套三层)、政策类 questions=章节标题/category=上级路径、超 800 字递归切且无块超限、相邻块重叠起点在句号后(喂重叠区句中的样例)、表格按行切且**每块含表头**、`<!--key-->` 置 is_key_clause、faq 类 questions=真实问法、Chunk 字段完备。
- [ ] 绿:`app/kb/chunker.py`。

### T4 · ingest(语料 → 行,幂等查重)
- [ ] 红:`tests/test_kb_ingest.py` —— Markdown 文件 → Chunk 行(文件级 doc_kind);faq 行迁移映射(questions=question、content_type=faq);三元组查重:同输入两遍只留一份。
- [ ] 绿:`app/kb/ingest.py`。
- [ ] 语料:`knowledge/` 三份 Markdown(退货政策含运费说明与 `<!--key-->` 标注与宽表、商品FAQ、售后手册)—— 内容直做,质量由评估集背书。

### T5 · embedder 封装
- [ ] 红:`tests/test_retrieval_embedder.py` —— encode 接口签名、维度 1024(用 tiny fake 权重或 monkeypatch,不加载真模型);懒加载:import 与构造不触发加载;单例复用。
- [ ] 绿:`app/retrieval/embedder.py`(FlagEmbedding 路线,懒加载 `lru_cache`)。

### T6 · milvus 封装
- [ ] 红:`tests/test_retrieval_milvus.py` —— ensure_collection 幂等、upsert/search/count 参数与返回形状(fake client 注入);search 返回 [(id, score)]。
- [ ] 绿:`app/retrieval/milvus.py`(MilvusClient 包一层,懒连接)。

### T7 · writer 双写与幂等(验收 2 的核心)
- [ ] 红:`tests/test_kb_writer.py`(单测,fake embedder/store)+ `tests/test_kb_writer_db.py`(db 标记,真 MySQL + fake store)—— write_chunks 两遍行数不翻倍;vectorize_pending 后状态 done、vector_id=str(id);**fake store 第 N 块抛错模拟中断 → 重跑 → pending 清零且每个 pk 恰 upsert 一次**(计数器在 upsert 边界)。
- [ ] 绿:`app/kb/writer.py`(write_chunks + vectorize_pending)。

### T8 · build_kb 脚本
- [ ] 实现:`scripts/build_kb.py` 编排 ingest → write_chunks → vectorize_pending;UTF-8 输出钉死;异常直接崩(离线任务);跑通真实建库,行数/耗时记账。
- [ ] 手动验证:重复跑 = 幂等;`--source` 可选限定单个文件。

### T9 · retriever 与 query_faq 换实现(契约红线)
- [ ] 红:`tests/test_retrieval_search.py` —— KnowledgeRetriever:embed→search→MySQL 回查→阈值过滤→字段组装(question=questions 全文含换行);全滤空返回空列表;Milvus 抛错 → ToolInfrastructureError。
- [ ] 红:`tests/test_tools_db.py` 现有 query_faq 用例改为注入 fake retriever,断言出参结构逐 key 不变;新增:空 keyword 不触检索直接 ToolNotFound、全滤空 → ToolNotFound 文案不变。
- [ ] 绿:`app/retrieval/search.py`;`business.py:make_query_faq(session, retriever)` 换内部实现(工厂签名扩参,工具 schema/文案不变);`registry.build_tools` 注入。

### T10 · 挖知识管线
- [ ] 红:`tests/test_kb_mining.py` —— prompt 含字面 JSON、**无裸花括号**(字符串级断言)、json_mode;解析 LLM 输出 → 行;坏 JSON 记数不中断;归一化 sha256 精确去重(staging 内 + 对已入库);db 测试:staging 写入/状态推进、kept 入库复用 writer。
- [ ] 绿:`app/kb/mining.py`(先读 `services/extract.py` 复用其 json_mode 模式)。
- [ ] 实现:`scripts/mine_qa.py` 编排(分批 → staging → 整体去重 → kept 入库 → 提示跑 build_kb 补向量化);手动对种子会话跑通。

### T11 · 评估集(替代 TDD 的那一步)
- [ ] `evals/retrieval_cases.jsonl`:`{query, expect_contains,干扰项}`;换说法案例为主(邮费/到付/退货时限/保修…)。
- [ ] `evals/run_retrieval_eval.py`:真实 Milvus + BGE-M3,hit@K 闭式口径;跑三遍看稳定性,结果与未命中分析记 dev-notes;阈值按分布定夺(设计授权内,记账)。

### T12 · 端到端验收增补
- [ ] `scripts/acceptance.sh` 增验收 5(「邮费是多少」换说法,join_tokens 后断言运费要点)与验收 6(timeout 杀 build_kb 半途 → 重跑 → pending=0 且 Milvus count == MySQL done);起服务前查端口。

### T13 · 收尾
- [ ] 全量测试回归;`dev-notes/ch03.md` 各阶段补记完整;spec §12 实现订正回填;CLAUDE.md / AGENTS.md 状态行更新;最终 fix 波 + 合并 main。

## 依赖与顺序

T0 → T1/T2(并行)→ T3 → T4 → T5/T6(并行)→ T7 → T8 → T9 → T10 → T11 → T12 → T13。
T9 是契约红线任务;T11/T12 是真实链路关闸(T0 的容器与嵌入实测是它们的前置)。

## 风险与回退

- FlagEmbedding 装不上 → transformers 直载(spec §9 预授权,记账)。
- 阈值默认值不合适 → 评估集上实测调整(设计授权内,记账)。
- Milvus 单容器 Windows 卷问题 → named volume;仍不行再议 compose 三容器(**停下来问**,不自行换方案)。
- 嵌入 CPU 太慢 → 只影响建库时长,不换模型。
