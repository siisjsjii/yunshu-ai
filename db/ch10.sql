-- =============================================================
-- ch10-B · 多标签主题分类器的结果表 `topic_classifications`(**新表,只有 CREATE**)
-- 设计源:docs/superpowers/specs/2026-09-25-ecommerce-cs-ch10-topic-classifier-design.md §9.3
-- =============================================================

-- 本机 locale 是 cp936,不钉这一行中文 COMMENT 会在客户端侧被重编码(本仓记过这条)。
SET NAMES utf8mb4;

-- ⚠️ **走法(与 db/ch09.sql 相反,因为它是一张全新表)**
--
--   本表是**全新的** ⇒ `scripts/init_db.py` 的 `create_all` **会**把它建出来
--   (`init_db.py` 那句「永不加列」管的是 ALTER 的活,建表它照建)。所以:
--     · **全新库:只跑 `scripts/init_db.py`,不要跑这份文件** —— 跑了会在这条
--       `CREATE TABLE` 上响亮地报 **1050**(表已存在)。**这是刻意的已知取舍**,
--       与 `db/ch08.sql` 的 `tool_audit_logs`、`db/ch06.sql` 的 `refund_requests`
--       完全同款(那两张的 ORM 侧也有同名模型)。
--     · **想让 DDL 成为权威形状**(要手工核 `SHOW CREATE TABLE` 时):
--       `DROP TABLE topic_classifications;` → 跑这份文件 → 核对。T12 就是这么做的。
--     · **别的库升级**(clone 出来、从未跑过 `init_db.py` 的库):跑这份文件即可。
--   两条路径建出来的形状差异**逐条记在** `app/db/models.py` 的
--   `TopicClassification` docstring 里(已对齐的与未对齐的分开写,数过再写)。
--
-- ⚠️ **本文件刻意不幂等**:不带任何幂等守卫(与 db/ch03 / ch04 / ch06 / ch07 / ch08 / ch09
--    同规矩)。重复执行要**响亮地失败**(1050),静默跳过会让「表已存在但形状不对」
--    永远补不上 —— 那是本仓记过的、只让某些知识永远检索不到却不报错的那一类故障。

-- 一行 = 一条池子行(`low_confidence_questions.id`)的归类结果。写入方是**离线批处理**
-- (`scripts/classify_topics.py`),请求路径上没有任何东西读写它。
--
-- ⚠️ **唯一键 `uk_pool_question` 是「重跑 = 覆盖,不是追加」的保证**(spec §9.3)。
--    它与 ch09 `review_queue` **刻意不加唯一键**的规矩相反,而这是对的:
--    `review_queue` 判的是**语义**(模型判两句话是不是一个意思),字面唯一键会在一次
--    **合理的语义归并**上响亮地 1062;本表判的是**确定性重算**,同一条池子行重算就该
--    覆盖旧值 ⇒ 唯一键是 upsert 那种写法的前提。**两处规矩相反,是因为两件事性质相反。**
--    没有它,重跑会在池子旁边静静堆出第二份结果、分布页条数跟着翻倍,而不报任何错。
--
-- ⚠️ 不挂外键(与 `tool_audit_logs.conversation_id` 同款):这是**结果表**。挂了外键,
--    池子那一侧的清理会受约束,甚至反过来影响主流程。代价如实记账 —— 「悬空的池子 id」
--    数据库不拦(测试探针正是这样,所以它不必真的在池子里)。
--
-- ⚠️ `labels` / `scores` 是 JSON 列:读它们一律用 `JSON_TYPE()`,**不用 `IS NULL` /
--    `IS NOT NULL`**(`none_as_null=False` ⇒ Python 的 `None` 落库是字面 JSON `null`,
--    SQL 上不是 NULL;ch09 T19 拿 `IS NOT NULL` 数「有快照的行」把 JSON `null` 数成了非空)。
--    两列都 NOT NULL —— 空标签该让批处理**响亮失败**(spec §9.2),而不是落一行
--    「没有主题」的结果,让分布页把「推理服务挂了」读成「这些问题没有主题」。
CREATE TABLE topic_classifications (
  id                         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  low_confidence_question_id BIGINT UNSIGNED NOT NULL COMMENT '关联 low_confidence_questions.id',
  labels                     JSON            NOT NULL COMMENT '多标签结果,如 ["尺码","退换货"]',
  scores                     JSON            NOT NULL COMMENT '逐标签概率,如 {"尺码":0.93}',
  model_version              VARCHAR(128)    NOT NULL COMMENT '训练产物指纹:这个结果是谁算的',
  classified_at              DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '归类时间',
  PRIMARY KEY (id),
  -- 索引名与 ORM 侧**同名**(`UniqueConstraint(..., name="uk_pool_question")`)——
  -- 刻意把 ch08/ch09 那种「索引名不同」的差异消掉,照 ch07 `uk_conv_seq` 的做法。
  UNIQUE KEY uk_pool_question (low_confidence_question_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='低置信度问题的多标签主题归类结果(ch10-B)';
