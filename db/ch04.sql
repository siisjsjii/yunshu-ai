-- =============================================================
-- ch04 · 低置信度问题池
-- 检索不到 / 自评不足时,问题落此池,供后续数据飞轮消费(本章只落池不消费)
-- =============================================================

SET NAMES utf8mb4;

CREATE TABLE low_confidence_questions (
  id                     BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  question               TEXT            NOT NULL                COMMENT '用户原话',
  source_conversation_id VARCHAR(32)     NULL                    COMMENT '来源会话(conversations.id)',
  entry_point            VARCHAR(32)     NOT NULL                COMMENT '入池入口:检索为空|自评不足|低分拒绝',
  reject_reason          TEXT            NOT NULL                COMMENT '判不能的原因',
  created_at             DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '入池时间',
  PRIMARY KEY (id),
  KEY idx_entry_point (entry_point)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='低置信度问题池';
