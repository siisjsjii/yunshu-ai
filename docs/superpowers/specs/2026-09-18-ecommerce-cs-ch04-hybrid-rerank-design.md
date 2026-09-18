# 电商智能客服系统 · ch04 增补:混合检索 + 重排 + 评估体系 — 设计文档

> 设计经用户逐点确认(2026-09-18:混合检索/重排/元数据过滤/Query 理解/生成质量控制/评估体系/前端配套,七条功能需求 + 四条验收)。
> 本文是本章权威设计文档;实现与设计的偏离记录在 §13「实现订正」。

## 1. 目标与验收

把客服系统的检索质量做上一个台阶:在 ch03 的 dense 单路检索之上,加 **BM25 关键词召回 + RRF 混合 + bge-reranker 重排**,配一套**可量化对比的评估体系**,并把生成端升级为**带引用、会拒答、可自评**的受控生成。

验收标准:

1. 四策略对比报告能跑出数字(纯 dense / 纯 BM25 / 混合 / 混合+Rerank)。
2. 问带具体型号的问题(如「MH-LP100」),BM25 那一路能命中(关键词精确匹配)。
3. 答案引用编号能定位回原文,聊天页上点引用能看到来源原文 + 章节路径。
4. 问知识库没有的内容,得到明确拒答,且问题进了低置信度池。

## 2. 技术栈与版本

| 组件 | 版本 | 说明 |
|---|---|---|
| Milvus | 2.6.24(现有容器) | BM25 Function + hybrid_search RRF(≥2.5 原生) |
| pymilvus | 2.6.17(锁定) | `Function` / `AnnSearchRequest` / `RRFRanker` |
| BGE-M3 | models/bge-m3(现有) | dense 1024 维 |
| bge-reranker-v2-m3 | models/bge-reranker-v2-m3(用户提供) | 重排,FlagEmbedding `FlagReranker` |
| FlagEmbedding | 1.4.2 | 加载 dense + reranker |
| MySQL | 8.0(现有 3307) | `low_confidence_questions` 新表 |

## 3. 现状与复用

ch03/ch04 已交付,本章复用:

| 复用物 | 位置 | 用途 |
|---|---|---|
| `KnowledgeRetriever` / `MilvusVectorStore` / `BgeM3Embedder` | `app/retrieval/` | 在线检索(本章替换内部实现) |
| `chunker` / `ingest` / `writer` | `app/kb/` | 离线切分/导入/双写(本章扩展 schema) |
| `query_faq` 工具 + tool-calling 骨架 | `app/tools/` | **保留骨架,只升级内部** |
| `mine_knowledge` / 挖知识管线 | `app/kb/mining.py` | 不涉及 |
| 检索评估集 | `evals/retrieval_cases.jsonl` | 本章扩展为四策略对比 |

## 4. 架构

### 4.1 检索管线(query_faq 内部升级)

```
模型选 query_faq(keyword)
 1. Query 理解:LLM 把口语/模糊 keyword 改写成标准问法 + 生成同义词变体(仅检索侧,不入库)
 2. 混合检索:hybrid_search
      dense 腿:embed(改写后 query) → Top-50
      BM25 腿:改写后 query 原文 → Top-50
      RRF(k=60) 融合;category 过滤在两条腿的 expr 上先加
 3. 重排:bge-reranker-v2-m3 对融合结果精排 → Top-10
 4. 回 MySQL 取原文,组装(带 chunk_id + section_path + category)
```

### 4.2 生成管线(知识路径)

```
query_faq 返回 Top-10 chunk(带引用编号 [1]..[10])
 → 自评(结构化 json_mode):{sufficient, reason}
    不足 → 显式拒答 + 问题落 low_confidence_questions
    足够 → 流式生成带引用的答案(chunk 最相关放首尾,System Prompt 列负面知识禁令)
 → SSE 额外推 citations 帧(引用编号 → chunk 元数据),供前端点开
```

**保留 tool-calling 骨架**:模型仍选 `query_faq` 决定「这是知识问题」;订单/物流等其它工具零改动。自评是生成阶段新增的一步。

### 4.3 组件

```
app/retrieval/milvus.py    加 text/category 字段 + BM25 function + hybrid_search 封装
app/retrieval/search.py    换内部:hybrid → rerank → 回查,返回带元数据的 chunk
app/retrieval/reranker.py  bge-reranker-v2-m3 封装(懒加载单例)
app/retrieval/query_understanding.py  LLM 改写 + 同义词扩展
app/services/chat.py       生成管线加自评 + citations 帧 + 拒答落池
app/db/models.py           + LowConfidenceQuestion ORM
db/ch04.sql                low_confidence_questions DDL
evals/run_hybrid_eval.py   四策略对比报告
```

## 5. 关键设计决策

### 5.1 Milvus schema 变更(逆转 ch03「只当索引」)

现有集合 `knowledge` 只有 `id + vector`。为支持 BM25 与元数据过滤,集合改为:

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | VARCHAR(64) pk | = str(MySQL id),不变 |
| `text` | VARCHAR(4096) | `category+questions+answer` 拼接原文,**BM25 输入**(挂 chinese analyzer) |
| `category` | VARCHAR(255) | 元数据过滤用 |
| `vector` | FLOAT_VECTOR(1024) | dense,不变 |

加 BM25 Function:`Function(name="text_bm25", type=BM25, input="text", output="text_bm25")`。

**这逆转了 ch03「Milvus 只当索引、不存文本」的决定** —— 用户需求「text 字段挂 BM25 函数」明确授权。代价:集合需 drop 重建(已有 `--reindex` 路径);`text` 与 MySQL 原文出现第二份文本(以 MySQL 为权威源,Milvus 的 text 只服务 BM25 分词,回查仍走 MySQL)。

### 5.2 BM25 + RRF

- BM25 的 chinese analyzer 通过 `text` 字段的 `analyzer_params` 配置(jieba 分词);**精确参数值真机冒烟后定**(§12 待实测)。
- 检索:`hybrid_search(reqs=[dense 腿, BM25 腿], ranker=RRFRanker(k=60), limit=N)`。
- dense 腿 `AnnSearchRequest(data=[向量], anns_field="vector", param={metric_type:"IP"}, limit=50)`;BM25 腿 `AnnSearchRequest(data=[改写后 query 原文], anns_field="text_bm25", param={}, limit=50)`。
- **BM25 腿的 data 形状(文本 vs 需自己算 BM25)待真机确认**(§12)。

### 5.3 重排(bge-reranker-v2-m3)

`FlagReranker(model_path)` 懒加载单例;对 RRF 融合结果逐对 `compute_score([(query, chunk_text)...])` → 按分数精排 Top-10。**冷启动加载同 bge-m3,走 lifespan 预热 + 锁**。

### 5.4 Query 理解

- LLM 改写:口语模糊问法 → 标准问法(如「猫砂盆 pro 多少钱」→「智能猫砂盆 Pro 价格」)。
- 同义词扩展:改写时顺带生成 1-2 个同义变体,**只做检索侧扩展,不在入库侧拆存多份**。
- 一个可开关的步骤(离线评估时可直接关闭,做纯原始 query 的基线)。

### 5.5 生成质量控制

- **引用编号**:chunk 组装成 `[1]..[10]`,答案里引 `[n]`;`citations` 帧把 `n → {chunk_id, section_path}` 发给前端。
- **拒答**:检索为空(全被阈值滤掉)或自评 `sufficient=false` → 显式拒答,不硬编;问题落池。
- **自评**:结构化 json_mode 一步,输入=用户原话 + Top-10 chunk,输出 `{sufficient, reason}`。
- **负面知识**:System Prompt 列明禁止承诺(如「不承诺到账时间、不承诺具体时效」),以政策原文为准。
- **chunk 顺序**:最相关的放首尾,次相关的放中间(防 lost-in-the-middle)。

### 5.6 元数据过滤

`category` 字段支持 `expr` 先过滤再检索。本章先支持「按 category 过滤」这一条路径(Query 理解阶段不自动抽取品类,前端/调用方可显式传)。

## 6. 数据层:low_confidence_questions

```sql
CREATE TABLE low_confidence_questions (
  id                      BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  question                TEXT NOT NULL COMMENT '用户原话',
  source_conversation_id  VARCHAR(32) NULL COMMENT '来源会话',
  entry_point             VARCHAR(32) NOT NULL COMMENT '入池入口:检索为空|自评不足|低分拒绝',
  reject_reason           TEXT NOT NULL COMMENT '判不能的原因',
  created_at              DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_entry_point (entry_point)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='低置信度问题池';
```

我写 `db/ch04.sql` 并建表(用户已授权)。

## 7. 评估体系

### 7.1 评估集

`evals/hybrid_cases.jsonl`:带难度梯度(易/中/难)+ ground-truth(每个 query 标注应命中的 chunk 的 category/章节路径,或「应拒答」)。含**具体型号问法**(如「MH-LP100 多大容量」,验证 BM25 命中)与**换说法**、**超纲拒答**。

### 7.2 指标与策略

- 检索段:`Recall@K`(K=5/10)、`MRR`(ground-truth chunk 命中)。
- 生成段:`Faithfulness`(LLM judge 判「答案是否只基于召回证据、有没有编造」)。
- 四策略:**纯 dense / 纯 BM25 / 混合(RRF) / 混合+Rerank**。
- 报告:按 query 类型(易/中/难/型号/拒答)分桶,输出每策略的 Recall@K / MRR / Faithfulness 表。

## 8. 前端(Vibe Coding,不套 TDD/brainstorm)

- 聊天页:答案里的 `[n]` 引用做成可点 → 弹出该 chunk 原文 + 章节路径(用 `citations` 帧数据)。
- 每条回复左下角 👍/👎,点一下点亮 + 「已反馈」+ 一次性锁定,纯前端采集(数据飞轮入口)。

## 9. 测试与验收

| 层 | 内容 |
|---|---|
| 单测 | reranker 封装懒加载/接口(fake 权重不加载真模型);query_understanding 改写/同义词(prompt 两禁:字面 JSON/无裸花括号);hybrid_search 封装(fake client 注入,验 RRF/BM25 腿参数);引用组装 |
| db 集成 | `low_confidence_questions` ORM 往返;拒答落池路径 |
| 评估 | 四策略对比脚本跑出数字(验收 1);型号问法 BM25 命中(验收 2) |
| 端到端 | 验收脚本增补:引用定位(验收 3)、拒答落池(验收 4) |

## 10. 待实测项与风险

| 项 | 处置 |
|---|---|
| BM25 chinese analyzer 精确参数 | Context7 + 真机冒烟(§12 记账) |
| BM25 腿 data 形状(文本 vs 需自算) | 同上 |
| bge-reranker-v2-m3 加载方式/接口 | FlagEmbedding `FlagReranker`,真机冒烟 |
| 冷启动(双模型 2.2GB×2) | lifespan 预热 + encode/rerank 锁;单测不加载模型 |
| schema 变更需 drop 重建 | `--reindex` 路径;MySQL 权威源不丢 |
| Faithfulness LLM judge 非确定 | 评估只记数不引用单次,样本量说明 |

## 11. 本章不做

- 指代消解、多轮改写(用户点名)。
- 混合检索的 sparse/colbert(BGE-M3 自带)、WeightedRanker(只做 RRF)。
- 重排模型的量化/GPU 加速。
- 低置信度池的消费端(只落池,不自动回流)。
- 满意度的后端落库(纯前端采集)。

## 12. 配置项

`retrieval_score_threshold`(0.58)、`retrieval_top_k` 沿用;新增可加 `rerank_top_k`(默认 10)、`hybrid_top_k`(默认 50)。其余复用 ch03/ch04。

## 13. 实现订正

(实现过程中与本文的偏离,连同原因记录于此。)
