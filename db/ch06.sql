-- =============================================================
-- ch06 · 退款单
-- 用户在前端退款表单里确认后才写入(端点: /api/refund)
-- =============================================================

-- 本机 locale 是 cp936,不钉这一行中文 COMMENT 会在客户端侧被重编码(本仓记过这条)。
SET NAMES utf8mb4;

-- ch06:退款单。用户在前端退款表单里确认后才写入。
-- 不带 IF NOT EXISTS(与 db/ch03.sql / db/ch04.sql 同规矩):重复执行要**响亮地失败**,
-- 否则「表已存在但形状不对」会被静默咽掉(本表就踩过:create_all 先建了没有
-- DEFAULT 的版本,IF NOT EXISTS 让这份 DDL 永远补不上)。
CREATE TABLE refund_requests (
  id              BIGINT      NOT NULL AUTO_INCREMENT PRIMARY KEY,
  conversation_id VARCHAR(32) NOT NULL,
  order_no        VARCHAR(32) NOT NULL,
  reason_category VARCHAR(64) NOT NULL,
  status          VARCHAR(32) NOT NULL DEFAULT 'pending',
  created_at      DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_refund_conv (conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='退款单(ch06)';
