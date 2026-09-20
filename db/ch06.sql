-- =============================================================
-- ch06 · 退款单
-- 用户在前端退款表单里确认后才写入(端点: /api/refund)
-- =============================================================

-- 本机 locale 是 cp936,不钉这一行中文 COMMENT 会在客户端侧被重编码(本仓记过这条)。
SET NAMES utf8mb4;

-- ch06:退款单。用户在前端退款表单里确认后才写入。
CREATE TABLE IF NOT EXISTS refund_requests (
  id              BIGINT      NOT NULL AUTO_INCREMENT PRIMARY KEY,
  conversation_id VARCHAR(32) NOT NULL,
  order_no        VARCHAR(32) NOT NULL,
  reason_category VARCHAR(64) NOT NULL,
  status          VARCHAR(32) NOT NULL DEFAULT 'pending',
  created_at      DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_refund_conv (conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='退款单(ch06)';
