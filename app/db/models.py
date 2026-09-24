from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Conversation(Base):
    """会话壳。id 复用 ch01 的 session_id(uuid4().hex,32 字符)。"""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    # ch07 两个锚点(`ALTER TABLE` 手写在 db/ch07.sql 里,create_all **不加列**)。
    # **两侧默认值都要**:`default` 让 ORM 插入时补值,`server_default` 让表本身
    # 有 DEFAULT(裸 SQL 省略也不至于 1364)—— 只留前者会让 create_all 建的表与
    # db/ch07.sql 建的表**形状不同**,行为变成「看谁建的库」。
    # 不变量:0 <= summary_upto_msg_id <= layer1_from_msg_id。
    summary_upto_msg_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default="0"
    )
    layer1_from_msg_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class MessageRecord(Base):
    """消息流水。类名不叫 Message —— ch01 的 app.schemas.Message 已占用该名字。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("conversations.id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # assistant 的「工具调用申请」是**数组**(可能一次申请多个),故用 JSON 列。
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Ticket(Base):
    """工单。按用户要求用业务工单号做主键,不用自增 id。"""

    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("conversations.id"), nullable=False, index=True
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class KnowledgeChunk(Base):
    """知识库 chunk 原文权威源(DDL: db/ch03.sql,表已由用户建好,ORM 只映射)。

    category / questions / answer 三字段拼成一段文本进向量;其余列都是
    「只存不进向量」的元数据。vectorize_status 是双写幂等的状态机:
    pending(待向量化)→ done(已向量化,/vector/ 已回填)。
    """

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    questions: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_key_clause: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("knowledge_chunks.id"), nullable=True
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("knowledge_chunks.id"), nullable=True
    )
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done", name="vectorize_status"),
        nullable=False,
        default="pending",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class LowConfidenceQuestion(Base):
    """低置信度问题池(ch04 建表:db/ch04.sql;ch09 加两列:db/ch09.sql)。

    检索为空 / 自评不足时,问题落此池留痕。ch04/ch05 只落不消费,**ch09 起它是
    数据飞轮的入口之一** —— 下面两列就是为此加的。`entry_point` 的三个生产取值
    见 ch09 spec §7.2(旧值 `置信度闸` 沿用,另有 `生成自评` / `用户反馈`)。

    ⚠️ **两列只能靠 db/ch09.sql 的 ALTER 加上**:本仓硬约束「`init_db.py` 永不加列」
    —— `create_all` 对**已存在**的表是空操作,它既不比形状也不报错。
    """

    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    source_conversation_id: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )
    entry_point: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    reject_reason: Mapped[str] = mapped_column(Text, nullable=False)
    # ---- 以下两列是 ch09 加的(db/ch09.sql 的 ALTER,create_all 不会加)----
    # 落池当轮的召回片段快照(Top-N 的 id / 得分 / 原文)。可空 —— 置信度闸那条
    # 路径未必总有快照;用户点「没用」时后端会重跑一次检索尽力回捞(ch09 §6.2)。
    #
    # ⚠️ **形状是 `list[dict]`,不是 `dict`**(2026-09-23 订正):写它的
    # `app/agent/nodes.py:_snapshot` 返回**列表**,每项是
    # `{"chunk_id", "score", "section_path", "answer"}`。注解原先写的是
    # `dict | None` —— 运行时无害(JSON 列照收),但**注解与事实不符会误导后来人**
    # (审核页 T15/T16 是照这个形状读的)。**另两处写入方**:`POST /api/feedback`
    # (T12 起,**已在线**,`app/api/feedback.py` 的 `_snapshot` 吃的是检索器刚返回的
    # `RetrievedChunk` 对象)与置信度闸(T18b 起,`_snapshot(evidence) if evidence else None`,
    # **已在线** —— 它改之前闸那一处**根本没传这个 kwarg**,那一列恒是「没快照」)。
    #
    # ⚠️ **别拿这两条反推形状**(都核过、都写准):
    # - `tests/test_ch09_orm.py` 的往返用例塞的是 **dict**(`{"chunks": [...]}`)——
    #   JSON 列两种都存得下,那条用例**说不出来**生产写的是哪种;别读成
    #   「dict 才是历来唯一的形状」;
    # - 同为 JSON 列的 `EvalRun.metrics` 是 **`Mapped[dict]`**(确实是对象);
    #   而 `MessageRecord.tool_calls` 是 **`Mapped[list]`** ⇒ **不能按表推**,一列一核。
    #
    # ⚠️⚠️ **查这一列「有没有快照」不许用 `IS NOT NULL`**(2026-09-25 T19 复审实测)。
    # `JSON` 类型的默认是 `none_as_null=False` ⇒ Python 的 `None` **落库是字面 JSON `null`**,
    # 它 **SQL 上不是 NULL** ⇒ `evidence_snapshot IS NOT NULL` 对「没记快照」的行**同样为真**。
    # 实测(闸的 46 行):SQL NULL **29** / 字面 JSON `null` **17** /
    # **`JSON_TYPE(...)='ARRAY'`(真的带快照)0** ⇒ 「闸那一列被行使过」是**假的**。
    # **判据:一律用 `JSON_TYPE(evidence_snapshot)`**;`NULL`(SQL)与 `null`(JSON)
    # 在这一列上是**两个不同的东西**(前者 = 加列之前的老行,后者 = 写了但没快照)。
    evidence_snapshot: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # ⚠️ **一个列担两个语义**(ch09 §7.1):既记「这条问题归并到了 review_queue 的
    # 哪一行」,**又**是飞轮流水线的**待处理标记**(`WHERE matched_review_id IS NULL`)。
    # ⇒ 流水线天然幂等,重跑不会重复归并;别只把它当外键用 —— 谁把它写成 NOT NULL,
    # 谁就抹掉了「尚未处理」这个状态(而且不会有任何东西报错)。
    # 这一列**必须两条路径都建索引**:流水线的选择谓词就是
    # `WHERE matched_review_id IS NULL ORDER BY id LIMIT n`,那是它唯一的热路径。
    # `index=True` ⇒ create_all 建 `ix_low_confidence_questions_matched_review_id`,
    # db/ch09.sql 的 ALTER 建**同列**的 `idx_matched_review` —— 覆盖一致、**名字不同**
    # (与 db/ch08.sql 的 `ix_*` vs `idx_*` 同一处已知差异)。
    # 这一列**刻意不挂 ForeignKey** —— 池子里的行不随 review_queue 的删除而受约束
    # (与 ToolAuditLog 同款理由)。
    matched_review_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class QaExtractionStaging(Base):
    """历史对话抽 QA 的离线中转暂存表(DDL: db/ch03.sql)。

    分批抽取按 batch_no 追溯;整体去重后 kept 行进 knowledge_chunks,
    discarded 行留痕;表本身不做自动清空(DDL 注释「建库完成可清空」留给人工)。
    """

    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded", name="qa_staging_status"),
        nullable=False,
        default="extracted",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class RefundRequest(Base):
    """退款单(ch06,DDL: db/ch06.sql)。用户在前端表单确认后由 /api/refund 写入。

    `reason_category` 必须是 `app/refund/categories.REFUND_REASON_CATEGORIES`
    里的一个 —— 校验在端点层做(DB 不建 CHECK,与既有表的做法一致)。
    """

    __tablename__ = "refund_requests"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    order_no: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_category: Mapped[str] = mapped_column(String(64), nullable=False)
    # 两侧默认值**都要**:`default` 让 ORM 插入时补值,`server_default` 让表本身
    # 有 DEFAULT(裸 SQL 省略 status 也不至于 1364)。只留前者的话,create_all
    # 建出的表与 db/ch06.sql 建出的表**形状不同**,行为变成「看谁建的库」。
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        server_default="pending",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class ConversationSummary(Base):
    """会话梗概(ch07,DDL: db/ch07.sql)。**只追加,不删除、不重写。**

    `seq` 从 1 起(由 `services/history.append_summary_and_advance` 在提交前
    取 `MAX(seq)+1` 算得);`upto_msg_id` 是这一段覆盖到哪条 `messages.id`(含)。

    唯一键 `(conversation_id, seq)` 是并发保护的第二道:同一会话两个摘要任务
    同时提交时,后者撞唯一键 ⇒ 失败 ⇒ 锚点不推进 ⇒ 下次重来。内存锁挡不住
    多进程,这个能。

    它**必须在 DDL 与 ORM 两侧各声明一份**(见 `__table_args__` 那段):
    两条建库路径(`scripts/init_db.py` 的 create_all / `db/ch07.sql` 的手工执行)
    建出来的表形状必须相同。
    """

    __tablename__ = "conversation_summaries"

    __table_args__ = (
        # 必须在这里也声明一份:DDL 与 ORM 是**两条建库路径**
        # (`scripts/init_db.py` 跑 create_all / `db/ch07.sql` 手工执行)。
        # 只在一侧声明,两条路径建出来的表**形状不同** —— 行为变成
        # 「看谁建的库」,而**没有任何东西会报错**。
        # (同类教训:`RefundRequest.status` 的 default/server_default 两处都要。)
        UniqueConstraint("conversation_id", "seq", name="uk_conv_seq"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    upto_msg_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 故意**不给 default**:空梗概是错误状态,写 None 必须被数据库当场拒
    # (1048),而不是被 Python 侧悄悄补成空串 —— 一张「内容为空」的梗概行
    # 会永久顶掉那段原文(锚点推过去了,替换物却什么也没说)。
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class ToolAuditLog(Base):
    """工具调用审计。**只映射,不被任何业务读写。**

    **刻意不挂外键**(spec §7.1):审计是**旁路记录** —— 挂了外键的话,
    删会话/删工单会受约束,甚至反过来影响主流程。审计的职责是**只记不拦**。
    """

    __tablename__ = "tool_audit_logs"

    # BigInteger 不是 Integer —— 与 db/ch08.sql 的 `BIGINT` 对齐。写 `Integer`
    # 的话两条建库路径建出来的表**形状不同**(create_all 版是 INT),
    # 行为变成「看谁建的库」(同 `RefundRequest.status` / `ConversationSummary`
    # 那两处的教训)。`CreateTable(...)` 编译出来逐列比对过。
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 不是 ForeignKey —— 见类 docstring。
    conversation_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    tool_call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    # `args` 与 `status` **刻意不给 `server_default`** —— db/ch08.sql 里这两列
    # 本来就没有 `DEFAULT`,照抄。多给一个会让两条路径又不一样。
    args: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # 这两列在 DDL 里写着 `DEFAULT ''`,所以**两侧都要**(与 retry_count 同款理由):
    # 只留 Python 侧 `default` 的话,create_all 建出的表缺 DEFAULT。
    result_summary: Mapped[str] = mapped_column(
        String(500), nullable=False, default="", server_default=""
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_detail: Mapped[str] = mapped_column(
        String(500), nullable=False, default="", server_default=""
    )
    # 两侧默认值都要:与 Conversation 的两个锚点同款理由 ——
    # 只留 `default` 会让 create_all 建的表与 db/ch08.sql 建的表**形状不同**。
    retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    duration_ms: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    # `index=True` 对应 db/ch08.sql 的 `KEY idx_created`(验收 5/6 都是
    # 「查最近这几条」)。索引名与 DDL 不同(SQLAlchemy 自动生成
    # `ix_tool_audit_logs_created_at`)—— 那处是**已知差异**,不影响行为。
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )


class ReviewQueue(Base):
    """待审队列(ch09,DDL: db/ch09.sql)。

    一行 = 一个**去重后**的知识缺口。查重命中时累加 `occurrences` 而不是新建行 ——
    **查重是语义判断**(模型判「两句话是不是同一个意思」),所以 DDL 上**刻意没有
    「标准化问题」的唯一键**:唯一键只能管**字面全等**,加了会在一次合理的语义归并上
    响亮地 1062(该报「归并成功」,得到「插入失败」)。

    `status` 取值 `pending|approved|rejected`;通过时 `approved_answer` 必填
    (不传就用 `example_answer`,校验在端点层做,DB 不建 CHECK —— 与既有表一致)。

    **与 db/ch09.sql 的形状差异**(两条建库路径 = `scripts/init_db.py` 的 create_all /
    db/ch09.sql 手工执行)。下面这份清单逐条对过**三处**:① 编译出的 `CreateTable(...)`;
    ② 临时库里 `create_all` 真正建出来的 `SHOW CREATE TABLE`;③ 实况库 ALTER 出来的
    `SHOW CREATE TABLE` / `SHOW INDEX`。

    **已对齐的**(列名、可空性、类型、`occurrences` / `status` 的**列级 DEFAULT**):
    `default=`(ORM 插入时补值)与 `server_default=`(表自己的 DEFAULT)**两侧都写**,
    照 `ToolAuditLog.retry_count` / `duration_ms` 的先例 —— 只写前者的话,create_all
    建出的表**没有列级 DEFAULT**,一条**省略 `status` / `occurrences` 的裸 INSERT**
    在严格模式下会失败,而 DDL 建的同一张表会成功(行为变成「看谁建的库」)。
    **索引覆盖也已拉平**:`status` / `created_at` / `matched_review_id` 三处在两条路径上都有索引
    (只差索引名,见 ⑤)。

    **差异清单(都是已知的,不影响行为)**:
    ① **`unsigned` 有无**:DDL 里三个 `BIGINT`(`review_queue.id` / `eval_runs.id` /
       `low_confidence_questions.matched_review_id`)都带 `UNSIGNED`,ORM 的 `BigInteger`
       编译成**有符号** `BIGINT` ⇒ 列范围 2^64-1 vs 2^63-1(取值上今天碰不到)。
       ⚠️ 但 `id` 只能写 `BigInteger`:ORM 的 `Integer` 编译成 4 字节 `INT`,那是**真的**形状不同。
    ② **COMMENT**:DDL 有表级 + 列级中文注释,ORM 侧没有 `comment=`。
    ③ **`created_at` 的默认值措辞**:DDL 建出来是 `DEFAULT CURRENT_TIMESTAMP`,
       create_all 建出来是 `DEFAULT (now())`(SQLAlchemy 把 `func.now()` 渲染成表达式)。
       两者都是「插入时取当前时间」,行为一致。
       ⚠️ **本条被一次复审判过一次「不是差异」—— 那个判定不可复现,2026-09-25 重测,
       结论是原判。** 方法(可复跑、不碰生产库):建临时库 → 一条路走 ORM 的
       `create_all`、另一条路走 `db/ch09.sql` 原文那句 `CREATE TABLE` →
       核对 `SHOW CREATE TABLE`。读数(MySQL **8.0.46**,即本机那个容器):

       | 路径 | `created_at` 那一行 |
       |---|---|
       | `create_all` | `` `created_at` datetime NOT NULL DEFAULT (now()) `` |
       | `db/ch09.sql` | `` `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP `` |

       **两条路逐字不同。** 注:`CreateTable` **发**出去的是 `DEFAULT now()`,是 **MySQL
       把它规范化成 `(now())`** —— 所以「发的是 `now()`」与「库里存的是 `(now())`」
       两句话**都对**,差别只在下一步。那次复审的依据正是「发的是 `now()` ⇒ 落库就是
       `CURRENT_TIMESTAMP`」的推论:**前半句对,后半句实测不成立**(「发出去的文本」
       不等于「`SHOW CREATE TABLE` 读回来的文本」)。判据:凡「外部系统会怎么规范化我这条
       声明」,必须**跑一遍 `SHOW CREATE TABLE`**,不能从发出去的文本往上推。
       仍是**纯文本差异**(语义都是「插入时取当前时间」)—— 别读成行为分歧。
    ④ **列序**:db/ch09.sql 走的是 ALTER,新列被**追加到末尾**(池子那两列就落在
       `created_at` 之后),而 create_all 按 ORM 的声明顺序建表 ⇒ 两条路径**列序不同**。
       这是 ALTER 路径的必然结果,而 SQLAlchemy 一律**按名取列**(不按位置)⇒ 不影响行为。
    ⑤ **索引名**:create_all 自动生成 `ix_review_queue_status`,DDL 写的是 `idx_status`
       —— 与 db/ch08.sql 的 `ix_tool_audit_logs_*` vs `idx_*` 是同一处已知差异(列相同、语义相同)。
    """

    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    standard_question: Mapped[str] = mapped_column(String(512), nullable=False)
    example_answer: Mapped[str] = mapped_column(Text, nullable=False)
    # `default=`(ORM 插入时补值)与 `server_default=`(**表自己**的列级 DEFAULT)**两侧都要**
    # —— 只写 `default=` 的话,create_all 建出的表没有列级 DEFAULT,一条省略 `status` /
    # `occurrences` 的裸 INSERT 在严格模式下会**失败**,而 DDL 建的同名表会成功:
    # 行为变成「看谁建的库」。照 `ToolAuditLog.retry_count` / `duration_ms` 的先例。
    # ⚠️ 字符串默认值写**不带引号**的 `"pending"` —— SQLAlchemy 把普通字符串当**值**,
    # 自己补引号(编译出来是 `DEFAULT 'pending'`)。**写成 `"'pending'"` 会渲染成
    # `DEFAULT '''pending'''`**,默认值就成了「带引号的字符串」,而它**长得像对的一样**
    # (实测,见 T5 报告)。整数 `"1"` 同理(渲染 `DEFAULT '1'`,MySQL 对 INT 列接受)。
    occurrences: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    # `index=True` 对应 db/ch09.sql 的 `KEY idx_status`(审核列表按 status 过滤)。
    # 索引名不同(SQLAlchemy 自动生成 `ix_review_queue_status`)—— 已知差异,与 db/ch08.sql 同款。
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", server_default="pending", index=True
    )
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    first_raw_question: Mapped[str] = mapped_column(Text, nullable=False)
    source_conversation_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class EvalRun(Base):
    """评估流水线的一轮(ch09,DDL: db/ch09.sql)。一行一轮,按时间连成趋势。

    `metrics` 的形状是**闭式**的(ch09 §10.3),各策略 / 各分桶的分数都在这一个 JSON 里;
    `trigger_by` 取值 `manual|scheduled`(ddl 注释同)。

    与 db/ch09.sql 的形状差异 —— 同 `ReviewQueue` docstring 里那份**五条清单**:
    ① `id` 的 `unsigned` 有无(DDL 带 `UNSIGNED`,ORM 有符号);
    ② DDL 的表/列 COMMENT 在 ORM 侧没有对应物;
    ③ `created_at` 的 `DEFAULT CURRENT_TIMESTAMP`(DDL)vs `DEFAULT (now())`(create_all)
       —— 本表与 `review_queue` 是**同一个读数、同一次实测**(2026-09-25 重测,方法见
       `ReviewQueue` 的 ③);
    ④ 列序不涉及(本表没走 ALTER);
    ⑤ 索引名 `ix_eval_runs_created_at` 对 DDL 的 `idx_created`。
    **已对齐的**:列名、可空性、类型、**索引覆盖**(`created_at` 在两条路径上都建索引),
    以及 `created_at` 默认值的**语义**(两条路都是「插入时取当前时间」)。
    ⚠️ 那句「语义对齐」与上面 ③ 的「**措辞**不同」**不矛盾,但必须分开说**:
    对齐的是行为,不同的是 `SHOW CREATE TABLE` 读回来的文本(一度把这两件事写成
    同一句话,读起来自相矛盾)。
    """

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trigger_by: Mapped[str] = mapped_column(String(32), nullable=False)
    case_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    # `index=True` 对应 db/ch09.sql 的 `KEY idx_created`(趋势查询按时间取)。
    # 索引名不同(SQLAlchemy 自动生成 `ix_eval_runs_created_at`)—— 已知差异,与 db/ch08.sql 同款。
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )
