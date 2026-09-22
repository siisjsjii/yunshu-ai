-- =============================================================
-- ch07 · 上下文管理(三层 + 两个锚点)
-- =============================================================

-- 本机 locale 是 cp936,不钉这一行中文 COMMENT 会在客户端侧被重编码(本仓记过这条)。
SET NAMES utf8mb4;

-- ch07:会话梗概。只追加,不删除、不重写(seq 从 1 起,只增不改)。
CREATE TABLE IF NOT EXISTS conversation_summaries (
  id              BIGINT       NOT NULL AUTO_INCREMENT,
  conversation_id VARCHAR(32)  NOT NULL,
  seq             INT          NOT NULL COMMENT '第 N 段,从 1 起,只增不改',
  upto_msg_id     BIGINT       NOT NULL COMMENT '这一段覆盖到哪条 messages.id(含)',
  content         TEXT         NOT NULL COMMENT '梗概正文,几十到一两百字',
  created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  -- 并发保护的第二道:同一会话两个摘要任务同时提交时,后者撞唯一键
  -- ⇒ 失败 ⇒ 锚点不推进 ⇒ 下次重来。内存锁挡不住多进程,这个能。
  UNIQUE KEY uk_conv_seq (conversation_id, seq),
  KEY idx_conv (conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 两个锚点必须**手写 ALTER**:scripts/init_db.py 跑的是
-- `Base.metadata.create_all`,它只建不存在的**表**,不改已有表 ——
-- 靠它加列会静默什么也不做,而代码里已经在读那两列。

-- 0 = 尚无梗概(不含任何消息)
-- 0 = 层 1 起于最早,层 2 为空
-- 不变量:0 <= summary_upto_msg_id <= layer1_from_msg_id
ALTER TABLE conversations
  ADD COLUMN summary_upto_msg_id BIGINT NOT NULL DEFAULT 0,
  ADD COLUMN layer1_from_msg_id  BIGINT NOT NULL DEFAULT 0;
