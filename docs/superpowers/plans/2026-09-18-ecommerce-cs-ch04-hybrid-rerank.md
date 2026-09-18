# ch04 混合检索 + 重排 + 评估体系 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 检索从 dense 单路升级为 BM25+dense 混合(RRF)再重排(bge-reranker-v2-m3),生成端带引用/自评/拒答/落池,配四策略评估框架 + 前端评测栏。

**Architecture:** Milvus 集合加 `text`(BM25)+`category`(过滤)字段 + BM25 Function;`hybrid_search` RRF 融合 → `FlagReranker` 精排 → 回查 MySQL。生成管线加自评(拒答落 `low_confidence_questions` 池)+ 引用帧。评估用用户 `evals/测试集.md` 跑四策略。保留 tool-calling 骨架,只升级 `query_faq` 内部与生成。

**Tech Stack:** Milvus 2.6.24(BM25/hybrid_search/RRF)+ pymilvus 2.6.17 + BGE-M3 dense + bge-reranker-v2-m3(FlagReranker)+ MySQL + LangChain 1.4。

**Spec:** `docs/superpowers/specs/2026-09-18-ecommerce-cs-ch04-hybrid-rerank-design.md`

## Global Constraints

- 解释器 `.venv/Scripts/python.exe`(Windows + Git Bash,locale cp936)。
- 单测不联网/不连 Milvus/不加载 BGE-M3 与 reranker 真模型;`Settings(...)` 一律 `_env_file=None`;db 测试 `@pytest.mark.db`;异步 `@pytest.mark.anyio`。
- 跑测试不加 `-q`;非 ASCII 输出用 `sys.stdout.buffer.write(...encode("utf-8"))`;含中文 HTTP 体走 stdin heredoc。
- 出站错误文本过 `app/sanitize.py:redact_api_key`。
- 不改 `knowledge_chunks` 现有表结构(只新增 `low_confidence_questions`)。
- 已核验的库 API(Milvus `Function(FunctionType.BM25, ...)`/`hybrid_search(reqs, RRFRanker(k=60), limit)`/`AnnSearchRequest(data, anns_field, param, limit, expr)`;FlagEmbedding `FlagReranker(path, use_fp16=False).compute_score(pairs, normalize=True)` → 0-1 分数)。**BM25 中文 analyzer 的精确 `analyzer_params` 值 + BM25 腿 `data` 形状由 T0 真机冒烟定,不在单测里猜。**
- 提交信息中文 `type: 描述`;每任务一提交,先红后绿。

## File Structure

- **Modify** `app/retrieval/milvus.py` — schema 加 text/category + BM25 function + `hybrid_search`。
- **Create** `app/retrieval/reranker.py` — `FlagReranker` 懒加载单例。
- **Create** `app/retrieval/query_understanding.py` — LLM 改写 + 同义词扩展。
- **Modify** `app/retrieval/search.py` — `KnowledgeRetriever` 换内部(hybrid→rerank→回查),`RetrievedChunk` 加 `chunk_id`/`section_path`。
- **Create** `db/ch04.sql` — `low_confidence_questions` DDL。
- **Modify** `app/db/models.py` — `LowConfidenceQuestion` ORM。
- **Create** `app/kb/assess.py` — 自评 + 拒答落池。
- **Modify** `app/services/chat.py` — 生成管线加自评 + citations 帧 + 负面 prompt。
- **Modify** `app/prompts.py` — 负面知识禁令 + 引用说明。
- **Modify** `app/kb/writer.py` — 写 Milvus 时加 text/category。
- **Modify** `scripts/build_kb.py` — 用新 schema 重建(复用 `--reindex`)。
- **Create** `scripts/run_eval.py` — 四策略评估 + 报告 JSON。
- **Modify** `app/static/admin.html` — 菜单分「落库/评测」+ 评测结果渲染。
- **Modify** `app/static/index.html` — 聊天页引用可点 + 👍/👎。
- **Test** `tests/test_retrieval_milvus.py`(扩)、`tests/test_retrieval_reranker.py`、`tests/test_query_understanding.py`、`tests/test_retrieval_search.py`(扩)、`tests/test_db_models.py`(扩)、`tests/test_kb_assess.py`、`tests/test_api_chat.py`(扩)。

---

### Task 1: Milvus schema 变更 + hybrid_search 封装

**Files:**
- Modify: `app/retrieval/milvus.py`
- Test: `tests/test_retrieval_milvus.py`(扩)

**Interfaces:**
- Consumes: 现有 `MilvusVectorStore`(ensure_collection/upsert/search/count/drop_collection)。
- Produces:
  - `MilvusVectorStore.ensure_collection()` 改为建含 `text`(VARCHAR, analyzer)+`category`(VARCHAR)+ BM25 function 的集合。
  - `MilvusVectorStore.upsert(ids, texts, categories, vectors)`(签名扩:加 text/category)。
  - `MilvusVectorStore.hybrid_search(vector, text, top_k, *, category=None) -> list[(id, score)]`。
  - 常量 `_TEXT_MAX_LENGTH = 4096`。

- [ ] **Step 1: 写失败测试**(fake client,验 ensure_collection 带 text/category/BM25 function 参数、upsert 行带 text/category、hybrid_search 调 MilvusClient.hybrid_search 传两条 AnnSearchRequest + RRFRanker)

```python
def test_ensure_collection_creates_text_category_and_bm25_function():
    fake = _FakeClient(exists=False)
    _store(fake).ensure_collection()
    # 断言 create_collection 或 create_schema 带 text/category 字段 + add_function(BM25)
    # 断言 analyzer_params 传给 text 字段
```

```python
def test_hybrid_search_sends_dense_and_bm25_legs_with_rrf():
    fake = _FakeClient(hybrid_hits=[{"id": "7", "distance": 0.9}])
    result = _store(fake).hybrid_search([0.1]*1024, "猫砂盆", top_k=50, category="商品规格手册")
    assert result == [("7", 0.9)]
    # 断言两条 AnnSearchRequest:anns_field 分别是 "vector" 与 "text_bm25";ranker 是 RRFRanker;category 过滤在两条腿的 expr 上
```

- [ ] **Step 2: 跑测试确认红** → `pytest tests/test_retrieval_milvus.py -v`

- [ ] **Step 3: 实现**(用 T0 定下的 analyzer 参数;`hybrid_search` 用 `MilvusClient.hybrid_search` 或 `Collection.hybrid_search`)

- [ ] **Step 4: 跑测试绿 + 真机冒烟**(见 Task 0 的冒烟脚本复用)

- [ ] **Step 5: 提交** `feat: ch04 T1 Milvus schema 加 text/category + BM25 function + hybrid_search 封装`

---

### Task 0(前置,先做):环境实测 BM25 + chinese analyzer + hybrid_search 真机冒烟

**产出为事实,无提交代码。** 用真实 Milvus 2.6.24 容器跑一段脚本:

- 建一个临时集合,`text` VARCHAR 字段带 chinese analyzer(`analyzer_params` 精确值在此确定:试 `{"tokenizer":"jieba"}` 与 analyzer 名「chinese」)。
- 加 BM25 Function;插几条中文,`hybrid_search` 用 dense 腿 + BM25 腿,确认 **BM25 腿的 `data` 形状**(raw 文本 list vs 需客户端算稀疏向量)。
- 记录:analyzer 值、BM25 腿 data 形状、`MilvusClient.hybrid_search` 是否存在(否则用 `Collection`)。结论写进 dev-notes 与 Task 1 实现。

---

### Task 2: reranker 封装

**Files:**
- Create: `app/retrieval/reranker.py`
- Test: `tests/test_retrieval_reranker.py`

**Interfaces:**
- Produces: `Reranker.__init__(model_path)`、`rerank(query: str, chunks: list[tuple[str, str]]) -> list[float]`(chunks 是 [(id, text)],返回同序 0-1 分数)、`get_reranker(model_path) -> Reranker`(lru_cache 单例)。

- [ ] **Step 1: 写失败测试**(fake FlagReranker 模块,验懒加载:构造不 import、首次 rerank 才加载、compute_score 传 [(query, text)] 对、normalize=True)

```python
def test_construction_does_not_import_flagembedding():
    Reranker("fake-path")
    assert "FlagEmbedding" not in sys.modules
```

- [ ] **Step 2-4: 红→绿**(同 embedder 模式,`_install_fake` 替身)

- [ ] **Step 5: 提交** `feat: ch04 T2 reranker 封装(FlagReranker 懒加载单例)`

---

### Task 3: Query 理解(改写 + 同义词)

**Files:**
- Create: `app/retrieval/query_understanding.py`
- Test: `tests/test_query_understanding.py`

**Interfaces:**
- Produces: `QueryUnderstanding.__init__(model)`、`async rewrite(query: str) -> QueryVariant`(dataclass:`normalized: str`、`synonyms: list[str]`)。prompt 走 json_mode(字面 JSON、无裸花括号)。

- [ ] **Step 1: 写失败测试**(prompt 含字面 JSON、无裸花括号;fake model 返 `{"normalized": "...", "synonyms": ["..."]}` 验解析;坏 JSON 抛约定异常)

- [ ] **Step 2-4: 红→绿**

- [ ] **Step 5: 提交** `feat: ch04 T3 Query 理解(LLM 改写 + 同义词扩展,json_mode)`

---

### Task 4: retriever 换内部(hybrid → rerank → 回查)

**Files:**
- Modify: `app/retrieval/search.py`
- Test: `tests/test_retrieval_search.py`(扩)

**Interfaces:**
- Consumes: `MilvusVectorStore.hybrid_search`(T1)、`Reranker`(T2)、`QueryUnderstanding`(T3)。
- Produces: `RetrievedChunk` 加 `chunk_id: int`、`section_path: str | None`;`KnowledgeRetriever.search(query) -> list[RetrievedChunk]` 内部 = 改写 → hybrid_search(Top-50)→ rerank(Top-10)→ MySQL 回查 → 按 rerank 序组装。

- [ ] **Step 1: 写失败测试**(fake store/embedder/reranker/rewriter:验改写被调、hybrid_search 被调(传改写后 text)、rerank 分数决定最终顺序、RetrievedChunk 带 chunk_id/section_path、全滤空返回空、store 抛错 → ToolInfrastructureError)

- [ ] **Step 2-4: 红→绿**

- [ ] **Step 5: 提交** `feat: ch04 T4 retriever 换 hybrid+rerank 内部`

---

### Task 5: low_confidence_questions 表 + ORM

**Files:**
- Create: `db/ch04.sql`
- Modify: `app/db/models.py`
- Test: `tests/test_db_models.py`(扩)

**Interfaces:**
- Produces: `LowConfidenceQuestion` ORM(question/source_conversation_id/entry_point/reject_reason/created_at)。

- [ ] **Step 1: 写 DDL + 建表**(`db/ch04.sql`,用户授权我建)
- [ ] **Step 2: 写失败测试**(往返:question/entry_point/reject_reason + source_conversation_id 可空)
- [ ] **Step 3-4: 红→绿(ORM 严格对齐 DDL)**
- [ ] **Step 5: 提交** `feat: ch04 T5 low_confidence_questions 表 + ORM`

---

### Task 6: 生成 QC(自评 + 拒答落池 + citations 帧 + 负面 prompt)

**Files:**
- Create: `app/kb/assess.py`
- Modify: `app/services/chat.py`、`app/prompts.py`
- Test: `tests/test_kb_assess.py`、`tests/test_api_chat.py`(扩)

**Interfaces:**
- Produces: `assess.py:async assess_sufficiency(question, chunks, model) -> {"sufficient": bool, "reason": str}`(json_mode);`record_low_confidence(session, ...) -> None`。
- `chat.py` 生成管线:query_faq 返回 chunk 后 → 自评 → 不足则拒答(固定文案)+ 落池 + 不生成;足够则流式生成带引用,SSE 加 `citations` 帧(引用编号 → chunk_id/section_path)。
- `prompts.py` 加负面知识禁令 + 引用说明。

- [ ] **Step 1: 写失败测试**(assess:prompt 两禁 + fake model 解析 + 落池 db 测试;chat:自评不足 → 拒答 + 落池 + 有 citations 帧,自评足够 → 有引用)

- [ ] **Step 2-4: 红→绿**

- [ ] **Step 5: 提交** `feat: ch04 T6 生成 QC(自评拒答落池 + 引用帧 + 负面 prompt)`

---

### Task 7: build_kb 用新 schema 重建

**Files:**
- Modify: `app/kb/writer.py`、`scripts/build_kb.py`

**Interfaces:**
- Consumes: T1 的 `upsert(ids, texts, categories, vectors)`。
- Produces: `vectorize_rows` 写 Milvus 时带上 text(=vector_text)与 category。

- [ ] **Step 1: 改 writer 写 text/category;build_kb `--reindex` 走新 schema**
- [ ] **Step 2: 真机 `build_kb --reindex` 重建,验 Milvus 集合含 text/category + BM25 function、55 行、BM25 检索能命中型号**
- [ ] **Step 3: 提交** `feat: ch04 T7 build_kb 写 text/category 到新 schema`

---

### Task 8: 评估框架(run_eval.py,四策略)

**Files:**
- Create: `scripts/run_eval.py`
- (输出) `evals/results/*.json`

**Interfaces:**
- Consumes: T4 的 retriever(可切四策略:纯 dense / 纯 BM25 / 混合 / 混合+Rerank)。
- Produces: 读 `evals/测试集.md`(CSV)→ 每策略算 Recall@K(K=5/10)、MRR、平均置信度 → 按桶(5 类)分桶 → 写 `evals/results/latest.json`(供前端评测栏读)+ 控制台表格。

- [ ] **Step 1: 实现四策略开关(纯 dense/纯 BM25/混合/混合+Rerank,复用 T1/T2/T4)**
- [ ] **Step 2: 实现读 CSV + Recall@K/MRR/置信度 计算 + 分桶**
- [ ] **Step 3: 真机跑通,产出数字(验收 1),记 dev-notes**
- [ ] **Step 4: 提交** `feat: ch04 T8 评估框架(四策略 Recall@K/MRR/置信度)`

---

### Task 9: 前端(菜单 落库/评测 + 评测结果 + 聊天页引用/反馈)

**Files:**
- Modify: `app/static/admin.html`、`app/static/index.html`

- [ ] **Step 1: admin.html 加菜单,分「落库」(文档/检索/上传/向量化/挖知识)与「评测」(运行评测按钮 + 读 `evals/results/latest.json` 渲染四策略对照表)**
- [ ] **Step 2: index.html 聊天页:引用 `[n]` 可点(用 citations 帧数据弹原文+章节路径);每条回复左下角 👍/👎(点亮 + 已反馈 + 一次性锁定,纯前端)**
- [ ] **Step 3: 起服务手验两页**
- [ ] **Step 4: 提交** `feat: ch04 T9 前端(菜单分落库/评测 + 评测结果 + 聊天页引用/反馈)`

---

### Task 10: 验收增补 + 收尾

**Files:**
- Modify: `scripts/acceptance.sh`、`dev-notes/ch04.md`、`docs/superpowers/specs/...ch04-hybrid-rerank-design.md`(§13)、`AGENTS.md`、`CLAUDE.md`

- [ ] **Step 1: acceptance.sh 增验收(型号问法 BM25 命中、引用定位、拒答落池)**
- [ ] **Step 2: 全量 pytest + acceptance 全绿**
- [ ] **Step 3: dev-notes 补记 + spec §13 订正回填 + AGENTS/CLAUDE 状态行**
- [ ] **Step 4: 提交 + finish**

---

## 依赖与顺序

T0(前置)→ T1 → T2 → T3 → T4 → T5(可并行于 T2-T4)→ T6 → T7 → T8 → T9 → T10。T4 是契约核心;T8 依赖 T4(四策略);T9 依赖 T8 的输出格式。

## 风险与回退

- BM25 中文 analyzer 不达预期(jieba 分词不准)→ 停止问用户,不换方案(定死)。
- BM25 腿 data 形状若需客户端算稀疏向量 → 按 T0 结论实现,不猜。
- 冷启动双模型(2.2GB×2)→ lifespan 预热 + 锁;单测不加载。
- Faithfulness 若 LLM judge 非确定 → 报告只记数不引用单次。
