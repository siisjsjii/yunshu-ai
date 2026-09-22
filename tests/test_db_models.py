"""DB 集成测试。需要 MySQL 在跑:见 spec §7.4。"""

import pytest
from sqlalchemy import select, text

from app.db.base import get_engine, get_sessionmaker
from app.db.models import (
    Conversation,
    ConversationSummary,
    KnowledgeChunk,
    LowConfidenceQuestion,
    MessageRecord,
    QaExtractionStaging,
    RefundRequest,
    Ticket,
)

pytestmark = pytest.mark.db

SCRATCH_CONVERSATION = "test0000000000000000000000000000"

#: 探针问题必须**不等于**库里任何既有行 —— 下面按名清理是 `DELETE ... WHERE
#: questions = :q`,探针与既有行同串就会把**真数据**一起删掉。
#: 加 `probe` 尾巴后,按名清理只可能命中本测试自己插入的那一行。
#: (原注:这条约束是在打 `faq` 表时踩出来的;改打 `knowledge_chunks` 后同理。)
SCRATCH_CHUNK_QUESTION = "退货政策是什么probe"


@pytest.mark.anyio
async def test_tables_exist_and_chinese_roundtrips():
    """四张表建得出来,且中文与 JSON 列往返不炸。"""
    engine = get_engine()
    async with get_sessionmaker()() as session:
        # 清理上次残留
        await session.execute(
            text("DELETE FROM messages WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.execute(
            text("DELETE FROM tickets WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.execute(
            text("DELETE FROM conversations WHERE id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )

        session.add(
            Conversation(id=SCRATCH_CONVERSATION, user="tester", status="active")
        )
        session.add(
            MessageRecord(
                conversation_id=SCRATCH_CONVERSATION,
                role="assistant",
                # 中文必须是**真插进去再读回来**才有意义:原先这里是空串,
                # 而 docstring 却声称验了 messages.content 的中文往返 ——
                # 空串在什么编码下都能往返,那条声称是空头支票。
                content="我帮您查一下物流",
                tool_calls=[
                    {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "call_1"}
                ],
            )
        )
        session.add(
            Ticket(
                ticket_no="T-TEST-1",
                conversation_id=SCRATCH_CONVERSATION,
                description="鞋码不对想换",
                ticket_type="换货",
                status="open",
            )
        )
        await session.commit()

    async with get_sessionmaker()() as session:
        row = (
            await session.execute(
                select(MessageRecord).where(
                    MessageRecord.conversation_id == SCRATCH_CONVERSATION
                )
            )
        ).scalars().one()
        assert row.tool_calls[0]["name"] == "query_logistics"   # JSON 列往返
        assert row.tool_call_id is None
        assert row.content == "我帮您查一下物流"                  # 中文往返

        conv = (
            await session.execute(
                select(Conversation).where(Conversation.id == SCRATCH_CONVERSATION)
            )
        ).scalars().one()
        assert conv.user == "tester"

    # 收尾清理,不留垃圾数据
    async with get_sessionmaker()() as session:
        await session.execute(
            text("DELETE FROM messages WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.execute(
            text("DELETE FROM tickets WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.execute(
            text("DELETE FROM conversations WHERE id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.commit()

    await engine.dispose()


@pytest.mark.anyio
async def test_like_matches_chinese_substring():
    """中文子串能命中。实测已确认 utf8mb4_0900_ai_ci 正常,此测试防回归。

    pattern 刻意**不是**裸 `%退货%`:库里本来就有含「退货」的行,裸 pattern 加
    `len(hits) == 1` 必挂,而 `>= 1` 又可以被既有行单独满足 —— 那样即使
    本测试的 INSERT 整段失效,断言照样全绿(这正是「假绿」的形态)。把 pattern
    锚到探针独有的 probe 尾巴上,命中数就只可能来自本测试自己插的那一行;
    中文部分仍在 pattern 里,所以编码/排序规则一坏,这里同样会红。

    **2026-09-20 改靶**:原先打的是 `faq` 表,该表已废弃删除。本测试验的是
    **中文 LIKE 的排序规则**,与哪张表无关,于是改打 `knowledge_chunks`
    (现在是知识唯一的权威源)。探针行带 `probe` 尾巴,收尾按名删除,
    不留待向量化的脏行。
    """
    async with get_sessionmaker()() as session:
        session.add(
            KnowledgeChunk(
                questions=SCRATCH_CHUNK_QUESTION,
                answer="七天无理由退货",
                category="退换货",
            )
        )
        await session.commit()
        hits = (
            await session.execute(
                select(KnowledgeChunk).where(
                    KnowledgeChunk.questions.like("%退货%probe%")
                )
            )
        ).scalars().all()
        assert len(hits) == 1
        assert hits[0].questions == SCRATCH_CHUNK_QUESTION
        await session.execute(
            text("DELETE FROM knowledge_chunks WHERE questions = :q"),
            {"q": SCRATCH_CHUNK_QUESTION},
        )
        await session.commit()


# ---- ch03:knowledge_chunks / qa_extraction_staging(DDL: db/ch03.sql)----

SCRATCH_KB_CATEGORY = "ch03-probe-分类"
SCRATCH_KB_BATCH = "ch03-probe-batch"


@pytest.mark.anyio
async def test_knowledge_chunk_defaults_and_roundtrip():
    """最小插入 → 服务器默认值生效:pending / 非关键条款 / 指针与 vector_id 为空。
    中文三字段(向量化文本的组成)必须真插真读。"""
    engine = get_engine()
    async with get_sessionmaker()() as session:
        chunk = KnowledgeChunk(
            category=SCRATCH_KB_CATEGORY,
            questions="运费怎么算\n下单时显示的邮费是多少",
            answer="单笔订单满 99 元包邮,否则收取 8 元运费。",
            content_type="policy",
        )
        session.add(chunk)
        await session.commit()
        chunk_id = chunk.id
        assert chunk_id is not None

    async with get_sessionmaker()() as session:
        row = (
            await session.execute(
                select(KnowledgeChunk).where(KnowledgeChunk.id == chunk_id)
            )
        ).scalars().one()
        assert row.category == SCRATCH_KB_CATEGORY
        assert row.questions.startswith("运费怎么算")          # 中文与换行往返
        assert row.vectorize_status == "pending"               # ENUM 服务器默认
        assert row.is_key_clause is False                      # TINYINT(1) → bool
        assert row.vector_id is None
        assert row.prev_chunk_id is None and row.next_chunk_id is None
        assert row.section_path is None

    await _cleanup_kb_rows(engine)


@pytest.mark.anyio
async def test_knowledge_chunk_flags_and_self_reference():
    """is_key_clause 可置真、vectorize_status 可置 done、vector_id 回填语义、
    prev/next 自引用 FK 双向往返 —— 四件都是双写流程要写的东西。"""
    engine = get_engine()
    async with get_sessionmaker()() as session:
        first = KnowledgeChunk(
            category=SCRATCH_KB_CATEGORY, questions="q1", answer="a1"
        )
        second = KnowledgeChunk(
            category=SCRATCH_KB_CATEGORY,
            questions="q2",
            answer="a2",
            is_key_clause=True,
        )
        session.add_all([first, second])
        await session.flush()
        second.prev_chunk_id = first.id
        first.next_chunk_id = second.id
        second.vectorize_status = "done"
        second.vector_id = str(second.id)
        await session.commit()
        first_id, second_id = first.id, second.id

    async with get_sessionmaker()() as session:
        a = (
            await session.execute(
                select(KnowledgeChunk).where(KnowledgeChunk.id == first_id)
            )
        ).scalars().one()
        b = (
            await session.execute(
                select(KnowledgeChunk).where(KnowledgeChunk.id == second_id)
            )
        ).scalars().one()
        assert a.next_chunk_id == second_id and b.prev_chunk_id == first_id
        assert b.is_key_clause is True and a.is_key_clause is False
        assert b.vectorize_status == "done"
        assert b.vector_id == str(second_id)

    await _cleanup_kb_rows(engine)


@pytest.mark.anyio
async def test_qa_staging_defaults_and_three_states():
    """staging 三态 ENUM:默认 extracted,kept/discarded 可写;source_ref 可空;
    批次号与中文问答对往返。"""
    engine = get_engine()
    async with get_sessionmaker()() as session:
        rows = [
            QaExtractionStaging(
                batch_no=SCRATCH_KB_BATCH,
                source_ref="seed0000000000000000000000000000",
                question="发什么快递",
                answer="默认发中通/圆通。",
            ),
            QaExtractionStaging(
                batch_no=SCRATCH_KB_BATCH, question="q", answer="a", status="kept"
            ),
            QaExtractionStaging(
                batch_no=SCRATCH_KB_BATCH, question="q", answer="a", status="discarded"
            ),
        ]
        session.add_all(rows)
        await session.commit()

    async with get_sessionmaker()() as session:
        got = (
            await session.execute(
                select(QaExtractionStaging)
                .where(QaExtractionStaging.batch_no == SCRATCH_KB_BATCH)
                .order_by(QaExtractionStaging.id)
            )
        ).scalars().all()
        assert [r.status for r in got] == ["extracted", "kept", "discarded"]
        assert got[0].question == "发什么快递"
        assert got[0].source_ref == "seed0000000000000000000000000000"

    async with get_sessionmaker()() as session:
        await session.execute(
            text("DELETE FROM qa_extraction_staging WHERE batch_no = :b"),
            {"b": SCRATCH_KB_BATCH},
        )
        await session.commit()
    await engine.dispose()


async def _cleanup_kb_rows(engine) -> None:
    from sqlalchemy import delete

    async with get_sessionmaker()() as session:
        await session.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.category == SCRATCH_KB_CATEGORY)
        )
        await session.commit()
    await engine.dispose()


SCRATCH_LQ_QUESTION = "ch04-probe-低置信度问题"


@pytest.mark.anyio
async def test_low_confidence_question_roundtrip():
    """落池行往返:原话/入池入口/原因 + source_conversation_id 可空。"""
    engine = get_engine()
    async with get_sessionmaker()() as session:
        row = LowConfidenceQuestion(
            question=SCRATCH_LQ_QUESTION,
            source_conversation_id=None,
            entry_point="检索为空",
            reject_reason="知识库没有相关内容",
        )
        session.add(row)
        await session.commit()
        row_id = row.id

    async with get_sessionmaker()() as session:
        got = (
            await session.execute(
                select(LowConfidenceQuestion).where(
                    LowConfidenceQuestion.id == row_id
                )
            )
        ).scalars().one()
        assert got.question == SCRATCH_LQ_QUESTION
        assert got.entry_point == "检索为空"
        assert got.reject_reason == "知识库没有相关内容"
        assert got.source_conversation_id is None
        await session.execute(
            text("DELETE FROM low_confidence_questions WHERE id = :i"), {"i": row_id}
        )
        await session.commit()
    await engine.dispose()


# ---- ch06:refund_requests(DDL: db/ch06.sql)----

SCRATCH_REFUND_ORDER = "20240915"
SCRATCH_REFUND_CATEGORY = "商品质量问题"


async def _cleanup_refund_rows() -> None:
    """按会话 id 清掉探针行。

    **必须 commit** —— `async with session` 退出时是 rollback,不 commit 的
    DELETE 会原地作废,探针行留库;下一条测试里的 `.one()` 于是撞
    MultipleResultsFound。(原计划文本里的 DELETE 就没有 commit,等于没删。)
    """
    async with get_sessionmaker()() as session:
        await session.execute(
            text("DELETE FROM refund_requests WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.commit()


@pytest.mark.anyio
async def test_refund_request_roundtrips_chinese_and_defaults_status():
    """新表建得出来、中文往返不炸、status 有默认值。

    插入时**不显式传** status / created_at:两者都靠默认值补上。
    断言一律在**新 session** 读回 —— 身份映射持弱引用,同 session 重读是否
    真打库取决于还有没有别的东西引用着那个 ORM 对象,会退化成「靠 refcount 走运」。
    """
    await _cleanup_refund_rows()  # 上一轮跑挂了也不污染本轮
    async with get_sessionmaker()() as session:
        request = RefundRequest(
            conversation_id=SCRATCH_CONVERSATION,
            order_no=SCRATCH_REFUND_ORDER,
            reason_category=SCRATCH_REFUND_CATEGORY,
        )
        session.add(request)
        await session.commit()
        # 自增主键提交后已回填(sessionmaker 是 expire_on_commit=False,读它不会触发隐式 IO)。
        inserted_id = request.id

    async with get_sessionmaker()() as session:
        row = (
            await session.execute(
                select(RefundRequest).where(
                    RefundRequest.conversation_id == SCRATCH_CONVERSATION
                )
            )
        ).scalars().one()
        assert row.reason_category == SCRATCH_REFUND_CATEGORY   # 中文往返
        assert row.order_no == SCRATCH_REFUND_ORDER
        assert row.status == "pending"                          # 默认值
        # 这两条不写 `is not None` —— NOT NULL 列上那是不可能失败的断言(真坏了
        # 会在 INSERT 就 1048/1364 抛,根本到不了断言),读了会误以为是覆盖。
        assert row.id == inserted_id              # 读回的就是刚插的那一行
        assert row.created_at.year >= 2026        # 服务器真的盖了当下的时间戳

    await _cleanup_refund_rows()


@pytest.mark.anyio
async def test_refund_request_persists_expected_columns():
    """order_no / reason_category / status 三列原样落库(新 session 读回)。

    与上一条的分工:上一条验的是「不给值也有默认值」,这条验的是
    「给了值就存住」。两条都用 `.one()` —— 靠上面的清理保证只可能命中
    本测试自己插的那一行,否则 `.one()` 会在残行上撞 MultipleResultsFound,
    而在**没有残行**时又恒真。"""
    await _cleanup_refund_rows()  # 保证下面 `.one()` 只可能命中本测试插的那行
    async with get_sessionmaker()() as session:
        session.add(
            RefundRequest(
                conversation_id=SCRATCH_CONVERSATION,
                order_no=SCRATCH_REFUND_ORDER,
                reason_category=SCRATCH_REFUND_CATEGORY,
            )
        )
        await session.commit()

    async with get_sessionmaker()() as session:      # 新 session 读回
        row = (
            await session.execute(
                select(RefundRequest).where(
                    RefundRequest.conversation_id == SCRATCH_CONVERSATION
                )
            )
        ).scalars().one()
        assert row.order_no == SCRATCH_REFUND_ORDER
        assert row.reason_category == SCRATCH_REFUND_CATEGORY
        assert row.status == "pending"
        await session.execute(
            text("DELETE FROM refund_requests WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.commit()


# ---- ch07:conversation_summaries + conversations 两个锚点(DDL: db/ch07.sql)----

SCRATCH_SUMMARY_CONV = "ch07probe0000000000000000000000"


async def _cleanup_summary_rows() -> None:
    """**必须 commit** —— `async with session` 退出是 rollback,
    不 commit 的 DELETE 原地作废,探针行留库,下一轮的 `.one()` 撞
    MultipleResultsFound。(计划文本里的 DELETE 就没有 commit。)"""
    async with get_sessionmaker()() as session:
        await session.execute(
            text("DELETE FROM conversation_summaries WHERE conversation_id = :c"),
            {"c": SCRATCH_SUMMARY_CONV},
        )
        await session.execute(
            text("DELETE FROM conversations WHERE id = :c"), {"c": SCRATCH_SUMMARY_CONV}
        )
        await session.commit()


@pytest.mark.anyio
async def test_conversation_anchor_columns_default_to_zero():
    """两个锚点列存在、且有默认值 0 —— 这是「还没压过」的哨兵。

    ⚠️ 在**已有**的 `conversations` 上,这两列只能靠 db/ch07.sql 的 `ALTER TABLE`:
    `Base.metadata.create_all` 只建表、**不改表**(全新库上它会把两列一并建出来,
    但任何既有的库都不会)。所以本用例红,最可能的原因是 ch07.sql 没跑到那个库上。
    """
    await _cleanup_summary_rows()
    async with get_sessionmaker()() as session:
        session.add(Conversation(id=SCRATCH_SUMMARY_CONV, user="tester", status="active"))
        await session.commit()

    async with get_sessionmaker()() as session:      # 新 session 读回
        row = (
            await session.execute(
                select(Conversation).where(Conversation.id == SCRATCH_SUMMARY_CONV)
            )
        ).scalars().one()
        assert row.summary_upto_msg_id == 0
        assert row.layer1_from_msg_id == 0

    await _cleanup_summary_rows()


@pytest.mark.anyio
async def test_anchor_columns_have_a_table_level_default_not_only_an_orm_one():
    """裸 SQL 插入(绕开 ORM)也必须拿到 0 —— 钉的是**表自己的 DEFAULT**。

    与上一条的分工:上一条走 ORM 插入,而 `Conversation` 上有 `default=0`,
    ORM 会替你把值填上 —— 于是**表里根本没有 DEFAULT 也照样绿**(本仓库
    记过多次的假绿形态:被断言的值恰好等于兜底值填出来的那个)。

    这条把 ORM 摘掉再插,表没有 DEFAULT 时 MySQL 直接以 1364
    「Field doesn't have a default value」拒掉。判别力是**实测的**:在活库上
    `ALTER TABLE conversations ALTER COLUMN summary_upto_msg_id DROP DEFAULT` 之后,
    本用例红、而上面那条走 ORM 的用例**照样绿**(这正是要分开两条的理由)。

    现状(2026-09-22):两条建库路径**都**带这个 DEFAULT —— DDL 里是 `DEFAULT 0`,
    ORM 里是 `server_default="0"`(实测 create_all 建出的 `conversations`,
    两个锚点列都是 `DEFAULT '0'`)。所以本用例是**防回归的哨兵**,
    而不是「现在有一条路径是坏的」的证据。
    """
    await _cleanup_summary_rows()
    async with get_sessionmaker()() as session:
        await session.execute(
            text(
                "INSERT INTO conversations (id, user, status) "
                "VALUES (:c, 'tester', 'active')"
            ),
            {"c": SCRATCH_SUMMARY_CONV},
        )
        await session.commit()

    async with get_sessionmaker()() as session:      # 新 session 读回
        row = (
            await session.execute(
                select(Conversation).where(Conversation.id == SCRATCH_SUMMARY_CONV)
            )
        ).scalars().one()
        assert row.summary_upto_msg_id == 0
        assert row.layer1_from_msg_id == 0

    await _cleanup_summary_rows()


@pytest.mark.anyio
async def test_summary_rows_roundtrip_and_seq_is_unique_per_conversation():
    """中文往返 + `(conversation_id, seq)` 唯一键真的在拦人。

    唯一键这条**必须实测**:它是并发保护的第二道,而「我以为建了唯一键」
    与「真建了」在并发出问题之前完全没有区别。判别力实测过:在活库上
    `ALTER TABLE conversation_summaries DROP INDEX uk_conv_seq` 之后本用例红。

    ⚠️ 本用例跑在**已经建好的库**上,所以它只证明「这个库里有这道约束」——
    「**两条**建库路径(create_all / db/ch07.sql)是否都声明了它」是
    `tests/test_orm_shape.py`(纯单测,读 ORM 元数据)的活,两条合起来才盖得住。
    2026-09-22 之前 ORM 侧确实没有这个约束、而 DDL 写着
    `CREATE TABLE IF NOT EXISTS`:create_all 抢建之后那句静默跳过,
    **唯一键永远不存在而没有任何东西报错**。现已两侧都声明
    (同类教训:`RefundRequest.status` 的 default / server_default 两侧都要)。
    """
    from sqlalchemy.exc import IntegrityError

    await _cleanup_summary_rows()
    async with get_sessionmaker()() as session:
        session.add(Conversation(id=SCRATCH_SUMMARY_CONV, user="tester", status="active"))
        session.add(
            ConversationSummary(
                conversation_id=SCRATCH_SUMMARY_CONV, seq=1,
                upto_msg_id=12, content="用户问过订单 1002 能不能退,尚未解决。",
            )
        )
        await session.commit()

    async with get_sessionmaker()() as session:
        row = (
            await session.execute(
                select(ConversationSummary).where(
                    ConversationSummary.conversation_id == SCRATCH_SUMMARY_CONV
                )
            )
        ).scalars().one()
        assert row.content.startswith("用户问过订单 1002")   # 中文往返
        assert row.upto_msg_id == 12
        assert row.seq == 1

    async with get_sessionmaker()() as session:
        session.add(
            ConversationSummary(
                conversation_id=SCRATCH_SUMMARY_CONV, seq=1,
                upto_msg_id=99, content="重复的 seq",
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()

    await _cleanup_summary_rows()
