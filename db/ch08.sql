-- =============================================================
-- ch08 · 工具调用审计
-- 每次工具调用一行;被权限拒、被校验拦的同样要落。
-- =============================================================

-- 本机 locale 是 cp936,不钉这一行中文 COMMENT 会在客户端侧被重编码(本仓记过这条)。
SET NAMES utf8mb4;

-- 不带 IF NOT EXISTS(与 db/ch03.sql / ch04.sql / ch06.sql / ch07.sql 同规矩):
-- 重复执行要**响亮地失败**,否则「表已存在但形状不对」会被静默咽掉。
--
-- **刻意不挂外键**(要求明写):审计是旁路记录。挂了外键的话,
-- 删会话/删工单会受约束,甚至反过来影响主流程 —— 而审计的职责是**只记不拦**。
CREATE TABLE tool_audit_logs (
  id              BIGINT       NOT NULL AUTO_INCREMENT,
  conversation_id VARCHAR(32)  NOT NULL COMMENT '所属会话',
  tool_call_id    VARCHAR(128) NOT NULL COMMENT '本次调用的 id(模型给的)',
  tool_name       VARCHAR(64)  NOT NULL,
  source          VARCHAR(64)  NOT NULL COMMENT 'builtin | mcp:logistics | mcp:aftersales',
  args            TEXT         NOT NULL COMMENT '调用参数(JSON)',
  result_summary  VARCHAR(500) NOT NULL DEFAULT '' COMMENT '结果摘要',
  status          VARCHAR(32)  NOT NULL COMMENT 'success|failed|timeout|invalid_args|permission_denied',
  error_detail    VARCHAR(500) NOT NULL DEFAULT '',
  retry_count     INT          NOT NULL DEFAULT 0 COMMENT '真实发生过的重试次数(不是配置值)',
  duration_ms     INT          NOT NULL DEFAULT 0,
  created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_conv (conversation_id),
  KEY idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='工具调用审计(ch08)';
