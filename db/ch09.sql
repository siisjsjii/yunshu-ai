-- =============================================================
-- ch09 · 低置信度数据飞轮(池子两列 + 待审队列 + 评估轮次)
-- 设计源:docs/superpowers/specs/2026-09-23-ecommerce-cs-ch09-observe-flywheel-design.md §7.1
-- =============================================================

-- 本机 locale 是 cp936,不钉这一行中文 COMMENT 会在客户端侧被重编码(本仓记过这条)。
SET NAMES utf8mb4;

-- ⚠️ **这份文件是给「已经有 `low_confidence_questions`、但还没有这两列」的库升级用的**
--    (老库,或 `db/ch04.sql` 建出来的库)。本机实测(2026-09-23,MySQL 8.0.46,临时库上跑):
--    · **已有那张表、缺这两列的库**:下面的 ALTER **一定会成功** —— `create_all` 不加列,
--      这两列只有它加得上(本仓硬约束:「init_db.py 永不加列」)。
--    · **全新空库**:先跑这份文件会在 ALTER 上报 **1146**(表还不存在);而先跑
--      `scripts/init_db.py` 呢,`create_all` 会**连同这两列一起**把表建出来,这份文件
--      再来一遍就 1060(列已存在)+ 1050(两张表已存在)三条全红。
--      ⇒ **全新库的正确走法:只跑 `scripts/init_db.py`,不要跑这份文件**;
--        它建出来的形状与这里略有出入(索引名 / `unsigned` / COMMENT / 列序,
--        逐条记在 `app/db/models.py` 的差异清单里,都不影响行为)。
--    ⇒ 一句话:**升级老库跑它,新建库不跑它。**
--    核对:`SHOW CREATE TABLE low_confidence_questions\G` 里要出现 evidence_snapshot。
--
-- ⚠️ **本文件刻意不幂等**:不带任何幂等守卫(与 db/ch03 / ch04 / ch06 / ch07 / ch08 同规矩)。
--    重复执行要**响亮地失败**(1060 列已存在 / 1050 表已存在),静默跳过会让
--    「表已存在但形状不对」永远补不上 —— 那是本仓记过的、只让某些知识永远检索不到
--    却不报错的那一类故障。
--
-- 升级一个老库的三步:① 跑这份文件;② 跑 `scripts/init_db.py`;③ `SHOW CREATE TABLE` 核对。

-- ① 池子加两列。两列都是**落池那一刻的留痕**,供飞轮消费。
--
--    evidence_snapshot:落池当轮的召回片段快照(Top-N 的 id / 得分 / 原文)。
--      用户点了「没用」时后端**重跑一次检索**尽力回捞(§6.2),那一份就是它。
--      可空 —— 置信度闸那条路径是先检索后落池,未必总有快照。
--
--    matched_review_id:**一个列担两个语义**(spec §7.1,ORM 的 docstring 里也写了一份)——
--      ① 记「这条问题归并到了 review_queue 的哪一行」;
--      ② 又是流水线的**待处理标记**(`WHERE matched_review_id IS NULL`)。
--      ⇒ NULL 是**有含义的值**(尚未进流水线),所以这一列必须可空;
--        流水线的幂等**只靠它**保证:重跑不会重复归并,不需要额外的状态列。
--
--    ⚠️ 两列会被**追加到 `created_at` 之后**(ALTER 只能往末尾加),而 ORM / create_all
--       按声明顺序建表、把这两列排在 `created_at` **之前** ⇒ 两条路径**列序不同**。
--       SQLAlchemy 一律**按名取列**(不按位置)⇒ 不影响行为,只记账。
ALTER TABLE low_confidence_questions
  ADD COLUMN evidence_snapshot JSON          NULL COMMENT '落池当轮的召回片段快照(Top-N 的 id/得分/原文)',
  ADD COLUMN matched_review_id BIGINT UNSIGNED NULL COMMENT '归并到的 review_queue.id;NULL = 尚未进流水线',
  -- 流水线的选择谓词 `WHERE matched_review_id IS NULL ORDER BY id LIMIT n` 是它唯一的热路径。
  -- ORM 侧同一列写了 `index=True`(create_all 建的名字不同:`ix_low_confidence_questions_matched_review_id`)。
  ADD KEY idx_matched_review (matched_review_id);

-- ② 待审队列。一行 = 一个**去重后**的知识缺口;查重命中时累加 occurrences,不新建行。
--
--    ⚠️ **刻意不给 standard_question 加唯一键**(spec §7.1 明写):
--      查重是**语义判断** —— 由模型判「两句话是不是同一个意思」;而唯一键只能管
--      **字面全等**。两者**不是同一条规则**,加了唯一键会在一次**合理的语义归并**上
--      响亮地 1062(该报的是「归并成功」,得到的却是「插入失败」)。
--      ⇒ 「同义不同字」的两行**本来就该能共存**;归并是代码(模型)的活,不是约束的活。
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

-- ③ 评估轮次。一行一轮,按时间连成趋势。
CREATE TABLE eval_runs (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  trigger_by  VARCHAR(32)     NOT NULL COMMENT '触发方式:manual|scheduled',
  case_count  INT             NOT NULL COMMENT '本轮评估集条数',
  metrics     JSON            NOT NULL COMMENT '各指标分数(按策略/按桶)',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='评估流水线轮次(ch09)';
