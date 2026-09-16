"""挖知识管线:prompt 两禁、json_mode、解析、归一化去重。

prompt 与解析用手替身(不联网);staging 状态推进、kept 入 knowledge_chunks
打 @pytest.mark.db。真实抽取出题质量由 T11 的评估集与人工抽查背书。
"""

import hashlib
import json

import pytest
from langchain_core.exceptions import OutputParserException
from sqlalchemy import delete, select, text

from app.db.base import get_engine, get_sessionmaker
from app.db.models import Conversation, KnowledgeChunk, MessageRecord, QaExtractionStaging
from app.kb.mining import (
    MINE_SYSTEM_PROMPT,
    MineParseError,
    QaPair,
    Turn,
    batch_conversations,
    build_mine_messages,
    dedupe_pairs,
    existing_fingerprints,
    find_near_duplicate,
    load_turns,
    mark_staging,
    mine_batch,
    normalize_question,
    question_fingerprint,
    render_conversations,
    staging_fingerprints,
    write_staging,
)
from app.schemas import MinedQaBatch, MinedQaItem

SCRATCH_CONVERSATION = "mine0000000000000000000000000000"
SCRATCH_BATCH = "ch03-probe-mine-batch"
SCRATCH_CATEGORY = "ch03-probe-mine-cat"


class _FakeChain:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.result


class _FakeModel:
    def __init__(self, result=None, error=None):
        self.calls: list = []
        self.chain = _FakeChain(result, error)

    def with_structured_output(self, schema, **kwargs):
        self.calls.append((schema, kwargs))
        return self.chain


def _batch(*items) -> MinedQaBatch:
    return MinedQaBatch(
        items=[MinedQaItem(question=q, answer=a, category=c) for q, a, c in items]
    )


# ---- prompt 与调用形态(不联网) ----


def test_system_prompt_contains_literal_json():
    """`json_mode` 链路要求提示词里出现字面 JSON —— ch02 血泪,换模型也一样。"""
    assert "JSON" in MINE_SYSTEM_PROMPT


def test_system_prompt_has_no_bare_braces():
    """描述结构时不得用裸花括号。

    `ChatPromptTemplate` 按 f-string 解析模板,`{` 会被当成变量占位符,
    构造消息时抛 KeyError —— 报错看起来像"模板缺变量",实则提示词写法问题。
    下面那条 `build_mine_messages` 不炸是功能层面的同一断言。
    """
    assert "{" not in MINE_SYSTEM_PROMPT
    assert "}" not in MINE_SYSTEM_PROMPT


def test_messages_render_without_template_error():
    messages = build_mine_messages("用户:怎么退货\n客服:七天无理由。")
    assert messages[0].content == MINE_SYSTEM_PROMPT
    assert "怎么退货" in messages[-1].content


def test_mine_batch_uses_json_mode():
    """抽取只能走 json_mode:该端点上 function_calling / json_schema 均 400。"""
    model = _FakeModel(_batch(("怎么退", "七天无理由", "退换货")))
    import asyncio

    asyncio.run(mine_batch(model=model, conversation_text="对话"))
    assert len(model.calls) == 1
    assert model.calls[0][1]["method"] == "json_mode"
    assert model.calls[0][0] is MinedQaBatch


def test_mine_batch_maps_items_to_pairs():
    model = _FakeModel(_batch(("怎么退", "七天无理由", "退换货"), ("能换吗", "能", "退换货")))
    import asyncio

    pairs = asyncio.run(mine_batch(model=model, conversation_text="对话"))
    assert pairs == [
        QaPair("怎么退", "七天无理由", "退换货"),
        QaPair("能换吗", "能", "退换货"),
    ]


def test_mine_batch_drops_blank_items():
    """模型可能回一条空壳,不能让它带着空问题进库。"""
    model = _FakeModel(_batch(("  ", "有答案", "分类"), ("真问题", "真答案", "分类")))
    import asyncio

    pairs = asyncio.run(mine_batch(model=model, conversation_text="对话"))
    assert pairs == [QaPair("真问题", "真答案", "分类")]


def test_mine_batch_bad_json_raises_parse_error():
    """坏 JSON 抛**约定的** MineParseError,由编排层记数后继续跑下一批。

    不抛裸异常:编排层要能区分"这批模型没吐好"和"上游挂了"—— 后者必须
    让整跑崩掉,而不是被当成一次解析失败悄悄跳过。
    """
    model = _FakeModel(error=OutputParserException("不是 JSON"))
    import asyncio

    with pytest.raises(MineParseError):
        asyncio.run(mine_batch(model=model, conversation_text="对话"))


# ---- 分批与渲染(纯函数) ----


def _turns(*specs) -> list[Turn]:
    return [Turn(cid, q, a) for cid, q, a in specs]


def test_batch_conversations_never_splits_a_conversation():
    """按会话分批:同一会话的轮次必须整批在一起,否则模型读到半段对话。

    下面 4 轮来自 2 个会话,batch_size=1 —— 按"每 1 轮一批"切会切成 4 批,
    每批都是残缺的一段。
    """
    turns = _turns(
        ("c1", "q1", "a1"), ("c1", "q2", "a2"), ("c2", "q3", "a3"), ("c2", "q4", "a4")
    )
    batches = batch_conversations(turns, 1)
    assert [[t.conversation_id for t in b] for b in batches] == [["c1", "c1"], ["c2", "c2"]]
    assert sum(len(b) for b in batches) == 4  # 不丢轮次
    assert batch_conversations(turns, 5) == [turns]


def test_batch_conversations_keeps_turn_order_within_conversation():
    turns = _turns(("c1", "q1", "a1"), ("c1", "q2", "a2"))
    assert [t.question for t in batch_conversations(turns, 5)[0]] == ["q1", "q2"]


def test_render_conversations_separates_sessions_and_labels_roles():
    """会话之间必须有分隔,否则模型会把两段对话读成一段、张冠李戴。"""
    text = render_conversations(_turns(("c1", "怎么退", "七天无理由。"), ("c2", "有货吗", "有。")))
    assert "用户:怎么退" in text and "客服:七天无理由。" in text
    assert text.index("c1") < text.index("c2")
    assert "---" in text            # 会话分隔线
    assert text.count("【会话") == 2


# ---- 归一化与去重(纯函数) ----


def test_normalize_strips_whitespace_punctuation_and_case():
    assert normalize_question(" 怎么  退货?!\n") == normalize_question("怎么退货")
    assert normalize_question("ABC") == normalize_question("abc")


def test_fingerprint_is_sha256_of_normalized_text():
    expected = hashlib.sha256(normalize_question("怎么退货").encode("utf-8")).hexdigest()
    assert question_fingerprint(" 怎么退货? ") == expected
    assert len(question_fingerprint("怎么退货")) == 64


def test_fingerprint_distinguishes_different_questions():
    """证伪「所有输入都返回同一个串」的实现。"""
    assert question_fingerprint("怎么退货") != question_fingerprint("怎么换货")


def test_dedupe_pairs_keeps_first_occurrence_and_reports_dropped():
    pairs = [
        QaPair("怎么退货?", "A", "退换货"),
        QaPair(" 怎么退货 ", "B", "退换货"),   # 归一化后与上一条同
        QaPair("怎么换货", "C", "退换货"),
    ]
    kept, dropped = dedupe_pairs(pairs, seen=set())
    assert [p.question for p in kept] == ["怎么退货?", "怎么换货"]
    assert [p.answer for p in dropped] == ["B"]


def test_dedupe_pairs_respects_seeded_fingerprints():
    """seen 里预置已入库知识的指纹 —— 挖出来的重复项要被丢掉。"""
    seen = {question_fingerprint("怎么退货")}
    kept, dropped = dedupe_pairs(
        [QaPair("怎么退货", "A", "c"), QaPair("新问题", "B", "c")], seen
    )
    assert [p.question for p in kept] == ["新问题"]
    assert [p.question for p in dropped] == ["怎么退货"]


def test_find_near_duplicate_uses_threshold_inclusively():
    import asyncio

    class _Store:
        def __init__(self, hits):
            self.hits = hits
            self.calls = []

        def search(self, vector, top_k):
            self.calls.append((list(vector), top_k))
            return list(self.hits)

    class _Embedder:
        def encode(self, texts):
            return [[0.5, 0.5] for _ in texts]

    assert asyncio.run(
        find_near_duplicate(
            store=_Store([("7", 0.95)]), embedder=_Embedder(), question="q", threshold=0.95
        )
    ) is True
    assert asyncio.run(
        find_near_duplicate(
            store=_Store([("7", 0.94)]), embedder=_Embedder(), question="q", threshold=0.95
        )
    ) is False
    assert asyncio.run(
        find_near_duplicate(
            store=_Store([]), embedder=_Embedder(), question="q", threshold=0.95
        )
    ) is False


# ---- 真 MySQL ----


async def _cleanup() -> None:
    async with get_sessionmaker()() as session:
        await session.execute(
            text("DELETE FROM messages WHERE conversation_id = :c"),
            {"c": SCRATCH_CONVERSATION},
        )
        await session.execute(
            text("DELETE FROM conversations WHERE id = :c"), {"c": SCRATCH_CONVERSATION}
        )
        await session.execute(
            text("DELETE FROM qa_extraction_staging WHERE batch_no = :b"),
            {"b": SCRATCH_BATCH},
        )
        await session.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.category == SCRATCH_CATEGORY)
        )
        await session.commit()


@pytest.mark.db
@pytest.mark.anyio
async def test_load_turns_pairs_user_assistant_and_skips_incomplete():
    """一问一答配对;只有 user 没收到的、以及空内容,都不构成一轮。"""
    await _cleanup()
    try:
        async with get_sessionmaker()() as session:
            session.add(Conversation(id=SCRATCH_CONVERSATION, user="t", status="active"))
            session.add_all(
                [
                    MessageRecord(
                        conversation_id=SCRATCH_CONVERSATION, role="user", content="怎么退货"
                    ),
                    MessageRecord(
                        conversation_id=SCRATCH_CONVERSATION,
                        role="assistant",
                        content="七天无理由。",
                    ),
                    MessageRecord(
                        conversation_id=SCRATCH_CONVERSATION, role="user", content="还没答的问题"
                    ),
                    MessageRecord(
                        conversation_id=SCRATCH_CONVERSATION, role="assistant", content=""
                    ),
                ]
            )
            await session.commit()
        async with get_sessionmaker()() as session:
            turns = [t for t in await load_turns(session) if t.conversation_id == SCRATCH_CONVERSATION]
        assert turns == [Turn(SCRATCH_CONVERSATION, "怎么退货", "七天无理由。")]
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.db
@pytest.mark.anyio
async def test_write_staging_then_mark_status():
    await _cleanup()
    try:
        pairs = [QaPair("怎么退货", "七天无理由。", "退换货")]
        async with get_sessionmaker()() as session:
            rows = await write_staging(
                session, batch_no=SCRATCH_BATCH, source_ref=SCRATCH_CONVERSATION, pairs=pairs
            )
            assert [r.status for r in rows] == ["extracted"]
        async with get_sessionmaker()() as session:
            got = (
                await session.execute(
                    select(QaExtractionStaging).where(
                        QaExtractionStaging.batch_no == SCRATCH_BATCH
                    )
                )
            ).scalars().all()
            assert [(r.question, r.answer, r.source_ref) for r in got] == [
                ("怎么退货", "七天无理由。", SCRATCH_CONVERSATION)
            ]
            await mark_staging(session, got, "kept")
        async with get_sessionmaker()() as session:
            got = (
                await session.execute(
                    select(QaExtractionStaging).where(
                        QaExtractionStaging.batch_no == SCRATCH_BATCH
                    )
                )
            ).scalars().all()
            assert [r.status for r in got] == ["kept"]
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.db
@pytest.mark.anyio
async def test_finalize_staging_marks_kept_and_discarded():
    """本轮结论回填 staging:保留的标 kept,其余标 discarded(不留 extracted)。"""
    from app.kb.mining import finalize_staging

    await _cleanup()
    try:
        async with get_sessionmaker()() as session:
            await write_staging(
                session,
                batch_no=SCRATCH_BATCH,
                source_ref=None,
                pairs=[
                    QaPair("留下的问题", "答", "c"),
                    QaPair("留下的问题", "答（同一问法的第二份抽取）", "c"),
                    QaPair("丢掉的问题", "答", "c"),
                ],
            )
        async with get_sessionmaker()() as session:
            kept_count, discarded_count = await finalize_staging(
                session, batch_no=SCRATCH_BATCH, kept=[QaPair("留下的问题?", "答", "c")]
            )
        # 同一问法只算一条 kept —— kept 的含义是「这行进库了」,不是「活过了去重」
        assert (kept_count, discarded_count) == (1, 2)
        async with get_sessionmaker()() as session:
            rows = (
                await session.execute(
                    select(QaExtractionStaging)
                    .where(QaExtractionStaging.batch_no == SCRATCH_BATCH)
                    .order_by(QaExtractionStaging.id)
                )
            ).scalars().all()
            # 「留下的问题?」与「留下的问题」归一化后同形 —— 问号不影响判定
            assert [(r.question, r.status) for r in rows] == [
                ("留下的问题", "kept"),
                ("留下的问题", "discarded"),
                ("丢掉的问题", "discarded"),
            ]
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.db
@pytest.mark.anyio
async def test_existing_fingerprints_covers_every_line_of_multiline_questions():
    """入库行的 `questions` 可能多行(政策块=章节标题,faq 块=问法);
    每一行都要参与去重,否则挖出来的问法与库里第二条问法重复也检不出。"""
    await _cleanup()
    try:
        async with get_sessionmaker()() as session:
            session.add(
                KnowledgeChunk(
                    category=SCRATCH_CATEGORY,
                    questions="怎么退货?\n退货要多久",
                    answer="正文",
                )
            )
            await session.commit()
        async with get_sessionmaker()() as session:
            prints = await existing_fingerprints(session)
        assert question_fingerprint("怎么退货") in prints
        assert question_fingerprint("退货要多久") in prints
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.db
@pytest.mark.anyio
async def test_keep_pairs_lands_in_knowledge_chunks_as_pending_faq():
    """kept 行复用 writer 入库:content_type=faq、状态 pending(等 build_kb 补向量)、
    无章节路径;跑两遍不重复(三元组查重是 writer 给的)。"""
    from app.kb.mining import keep_pairs

    await _cleanup()
    try:
        pairs = [QaPair("挖出来的问法", "挖出来的答案", SCRATCH_CATEGORY)]
        async with get_sessionmaker()() as session:
            assert await keep_pairs(session, pairs) == 1
        async with get_sessionmaker()() as session:
            assert await keep_pairs(session, pairs) == 0  # 幂等
        async with get_sessionmaker()() as session:
            rows = (
                await session.execute(
                    select(KnowledgeChunk).where(
                        KnowledgeChunk.category == SCRATCH_CATEGORY
                    )
                )
            ).scalars().all()
            assert len(rows) == 1
            row = rows[0]
            assert (row.questions, row.answer) == ("挖出来的问法", "挖出来的答案")
            assert row.content_type == "faq"
            assert row.vectorize_status == "pending"
            assert row.section_path is None
    finally:
        await _cleanup()
        await get_engine().dispose()


@pytest.mark.db
@pytest.mark.anyio
async def test_staging_fingerprints_covers_all_statuses():
    """三态都算数 —— 否则重跑会把上一轮已判为 discarded 的又抽一遍。"""
    await _cleanup()
    try:
        async with get_sessionmaker()() as session:
            rows = await write_staging(
                session,
                batch_no=SCRATCH_BATCH,
                source_ref=None,
                pairs=[QaPair("问题一", "答", "c"), QaPair("问题二", "答", "c")],
            )
            await mark_staging(session, rows[:1], "discarded")
        async with get_sessionmaker()() as session:
            prints = await staging_fingerprints(session)
        assert question_fingerprint("问题一") in prints
        assert question_fingerprint("问题二") in prints
    finally:
        await _cleanup()
        await get_engine().dispose()
