-- 认证(2026-09-27):login 用的账号表。
--
-- ⚠️ **刻意不幂等**(与 db/ch03/04/06/07/08/ch09 同规矩):重复执行会在
--    CREATE TABLE 上响亮地报 1050。那是有意的 —— 静默跳过会让
--    「表已存在但形状不对」永远补不上。
--
-- ⚠️ **ORM 侧有同名模型**(`app/db/models.py` 的 `User`,本任务 Step 4 加的)⇒
--    `init_db.py` 的 create_all **会把这张表建出来**(它建「不存在的表」)——
--    ⇒ **正常路径只需要 `init_db.py`,不需要跑这一份**;在表已存在时跑它会在
--    那条 CREATE 上响亮地报 **1050**。**那是刻意的,不是脏库** ——
--    与 `db/ch10.sql` 的 `topic_classifications`、`db/ch08.sql` 的 `tool_audit_logs`
--    是**同一个**已知取舍(本仓既有的三处同族;写法与理由见 CLAUDE.md 的建库一段)。
--    想让**这份 DDL 成为形状的权威**就先 `DROP TABLE users;` 再跑一遍,
--    然后用 `SHOW CREATE TABLE users\G` 核对。
--
-- ⚠️ 两条路径的形状差异**只有文本**(纯注释与列序):列定义两边逐字一致。
--    这份文件的价值是「形状肉眼可读 + 有 COMMENT」,不是「建表的那一步」。
--
-- ⚠️ **不给 `conversations.user` 加外键**(spec §5.1 的取舍):那一列已有
--    581 行值、宽度 varchar(128),加 FK 要一条迁移,而收益只是「写错的 user
--    被数据库拦住」—— 写入方只有一个端点,且值来自 token。如实记账。
--
-- 账号本身**不在这份文件里**:scrypt 的盐是随机的,写进 SQL 就得把某一轮的盐
-- 焊死在文件里,且改密码要人来重算。见 `scripts/seed_users.py`(幂等、可重跑)。

CREATE TABLE users (
  id            BIGINT       NOT NULL AUTO_INCREMENT,
  username      VARCHAR(128) NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  role          VARCHAR(16)  NOT NULL,
  created_at    DATETIME     NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uk_users_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='登录账号(认证功能,2026-09-27)';
