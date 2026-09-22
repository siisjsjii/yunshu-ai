-- =============================================================
-- ch07 · 上下文管理(三层 + 两个锚点)
-- =============================================================

-- 本机 locale 是 cp936,不钉这一行中文 COMMENT 会在客户端侧被重编码(本仓记过这条)。
SET NAMES utf8mb4;

-- 执行顺序:**先这份文件,再 `scripts/init_db.py`**(反过来也行,只要保持本文件
-- 自己「先 ALTER 后 CREATE」的次序 —— 理由见 ①)。
-- `scripts/init_db.py` 跑的是 `Base.metadata.create_all`,它只建**表**、**不加列** ——
-- 两个锚点列只能靠下面这句 ALTER。

-- ① 先加两个锚点列。
--    顺序是**刻意的**:mysql 客户端**遇到第一个错误就中止整个脚本**,而
--    `ALTER ... ADD COLUMN` 不是幂等的(列已在就报 1060)。ALTER 放前面,
--    「表已存在但列还没加」那条升级路径(先跑过 init_db.py 的 create_all、
--    或这份脚本上次跑到一半)才跑得到建表那一步;反过来放,CREATE 先以 1050
--    中止,**两列就永远加不上**,而报错信息读起来像「表已存在 ⇒ 已经装好了」。
--    镜像的那条(列已加、表没建)会停在 1060 上,代价小得多:缺梗概表只是
--    摘要功能不可用,缺列则是**每个请求都要读**的字段。
--
-- 0 = 尚无梗概(不含任何消息)
-- 0 = 层 1 起于最早,层 2 为空
-- 不变量:0 <= summary_upto_msg_id <= layer1_from_msg_id
ALTER TABLE conversations
  ADD COLUMN summary_upto_msg_id BIGINT NOT NULL DEFAULT 0,
  ADD COLUMN layer1_from_msg_id  BIGINT NOT NULL DEFAULT 0;

-- ② 再建梗概表。只追加,不删除、不重写(seq 从 1 起,只增不改)。
-- 不带 IF NOT EXISTS(与 db/ch03.sql / db/ch04.sql / db/ch06.sql 同规矩):重复执行要
-- **响亮地失败**,否则「表已存在但形状不对」会被静默咽掉(本表就踩过:create_all 若
-- 先建了表,这份 DDL 用 IF NOT EXISTS 会安静地什么都不做,唯一键于是永远补不上;
-- 现在 ORM 侧也声明了同一个约束,见 `app/db/models.py::ConversationSummary.__table_args__`)。
CREATE TABLE conversation_summaries (
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
