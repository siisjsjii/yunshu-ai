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
    """低置信度问题池(ch04,DDL: db/ch04.sql)。

    检索为空 / 自评不足时,问题落此池留痕,供数据飞轮消费(本章只落不消费)。
    """

    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    source_conversation_id: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )
    entry_point: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    reject_reason: Mapped[str] = mapped_column(Text, nullable=False)
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

    ⚠️ **唯一键 `(conversation_id, seq)` 不在 ORM 声明里**(本类只声明了
    `index=True`),它只在 db/ch07.sql 里 —— 所以 `create_all` 建出来的表
    **没有这道防线**。这不是疏漏:它是并发保护的第二道(同一会话两个摘要任务
    同时提交时,后者撞唯一键 ⇒ 失败 ⇒ 锚点不推进 ⇒ 下次重来;内存锁挡不住
    多进程)。把库当成 db/ch07.sql 建的,`tests/test_db_models.py` 有哨兵。
    """

    __tablename__ = "conversation_summaries"

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
