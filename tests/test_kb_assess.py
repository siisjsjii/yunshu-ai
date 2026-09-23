"""生成 QC:自评 prompt 两禁 + json_mode + 解析失败退化;落池 db 测试。

注意:被测的 `assess_sufficiency` 当前**不在请求路径上**(ch05 spec §50 用事前
置信度闸取代了它的事后自评),本文件全绿不代表线上有这条链路;理由见该函数 docstring。
"""

import json

import pytest
from langchain_core.exceptions import OutputParserException
from sqlalchemy import select, text

from app.db.base import get_engine, get_sessionmaker
from app.db.models import LowConfidenceQuestion
from app.kb.assess import (
    ASSESS_SYSTEM_PROMPT,
    AssessResult,
    assess_sufficiency,
    record_low_confidence,
)


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

    def with_structured_output(self, schema, **kw):
        self.calls.append((schema, kw))
        return self.chain


class _C:
    def __init__(self, answer):
        self.answer = answer


def test_prompt_contains_literal_json():
    assert "JSON" in ASSESS_SYSTEM_PROMPT


def test_prompt_has_no_bare_braces():
    assert "{" not in ASSESS_SYSTEM_PROMPT and "}" not in ASSESS_SYSTEM_PROMPT


def _run(model, chunks):
    import asyncio

    return asyncio.run(assess_sufficiency("退货几天", chunks, model))


def test_assess_uses_json_mode_and_returns_result():
    model = _FakeModel(AssessResult(sufficient=False, reason="没有相关内容"))
    result = _run(model, [_C("七天无理由退货")])
    assert result == {"sufficient": False, "reason": "没有相关内容"}
    assert model.calls[0][0] is AssessResult
    assert model.calls[0][1]["method"] == "json_mode"
    # 喂进去的 human 消息含用户问题与知识块正文
    human = model.chain.messages[-1].content
    assert "退货几天" in human and "七天无理由退货" in human


def test_assess_falls_back_to_sufficient_on_parse_error():
    model = _FakeModel(error=OutputParserException("bad"))
    assert _run(model, [_C("x")]) == {"sufficient": True, "reason": ""}


# ---- db ----


@pytest.mark.db
@pytest.mark.anyio
async def test_record_low_confidence_writes_row():
    engine = get_engine()
    async with get_sessionmaker()() as session:
        await record_low_confidence(
            session,
            question="ch04-probe-拒答问题",
            source_conversation_id=None,
            entry_point="自评不足",
            reject_reason="知识库没有相关内容",
        )
    async with get_sessionmaker()() as session:
        row = (
            await session.execute(
                select(LowConfidenceQuestion).where(
                    LowConfidenceQuestion.question == "ch04-probe-拒答问题"
                )
            )
        ).scalars().one()
        assert row.entry_point == "自评不足"
        assert row.reject_reason == "知识库没有相关内容"
        assert row.source_conversation_id is None
        await session.execute(
            text("DELETE FROM low_confidence_questions WHERE question = :q"),
            {"q": "ch04-probe-拒答问题"},
        )
        await session.commit()
    await engine.dispose()


_T11_Q = "ch09-T11-探针-返回行 id"


@pytest.mark.db
@pytest.mark.anyio
async def test_record_low_confidence_returns_new_row_id_and_stores_snapshot():
    """T11:返回**新行 id**,且快照真的落进列里(不是只停在形参上)。

    两半都必须断,因为**每一半都能静默失效**:

    - **返回 `None`**:飞轮流水线(T13+)拿不到「刚写进去的是哪一行」,
      而调用点看起来完全正常 —— 它今天返回 `None`,没有任何地方会红;
    - **形参收了但没写进 ORM**:审核页(T15/T16)拿到 NULL,而函数
      **调用成功、返回正常**。

    写这条之前先把 T10 留下的覆盖度**实测了一遍**(变异 B,见 task-11-report):
    把 `evidence_snapshot=evidence_snapshot` 换成 `=None` 之后 ——
    ① `tests/test_agent_protocol.py::test_useful_false_falls_back_and_records_one_pool_row`
    **会红**(它断的是节点塞进**替身 session** 的 ORM 对象)⇒ 「形参 → ORM 对象」
    这半有覆盖;② 而**没有任何一条用例在真列上读过它** ——
    `test_record_low_confidence_writes_row` 在同一次变异下**照样绿**
    (它根本不传快照,自然读不出差别)。⇒ 缺的是**真库那一环**,补的就是它。

    清理用的是**返回的 id**(这正是本章要这个 handle 的原因:测试不必
    再按问题文本反查)。
    """
    snapshot = [
        {"chunk_id": 7, "score": 0.31, "section_path": "退换货/政策",
         "answer": "七天无理由退货。"},
        {"chunk_id": 8, "score": 0.22, "section_path": "退换货/流程",
         "answer": "点申请即可。"},
    ]
    engine = get_engine()
    rid = None
    try:
        async with get_sessionmaker()() as session:
            rid = await record_low_confidence(
                session,
                question=_T11_Q,
                source_conversation_id="t11-conv",
                entry_point="生成自评",
                reject_reason="ch09-T11 探针",
                evidence_snapshot=snapshot,
            )
        # ① 返回值:必须是可用的主键,不是一个 `None`
        assert isinstance(rid, int) and rid > 0, f"必须返回新行 id,实际是 {rid!r}"

        # ② 快照真的落进了列里;且这个 id **指向刚写的那一行**(而不是别的行)
        #
        # 读回用**新 session**:SQLAlchemy 身份映射持弱引用,同 session 重读
        # 是否打到库取决于还有没有东西引用着那个 ORM 对象(本仓已知的
        # 「靠 refcount 走运」)。这里直接走 SQL,连身份映射都不经过 ——
        # 也正因为走的是**裸 SQL**,JSON 结果处理器不参与(它只作用于带类型
        # 的列),拿到的就是库里那串 JSON 文本,`json.loads` 一下正好说明
        # 「库里存的确实是我给的那份」(实测:裸 SQL 读出来是 `str`)。
        async with get_sessionmaker()() as session:
            row = (
                await session.execute(
                    text(
                        "SELECT question, evidence_snapshot "
                        "FROM low_confidence_questions WHERE id = :i"
                    ),
                    {"i": rid},
                )
            ).one()
            assert row[0] == _T11_Q
            assert json.loads(row[1]) == snapshot
    finally:
        # RED 时 rid 还是 None ⇒ 退化成按问题文本删,免得留垃圾
        if rid is not None:
            stmt, params = (
                text("DELETE FROM low_confidence_questions WHERE id = :i"), {"i": rid})
        else:
            stmt, params = (
                text("DELETE FROM low_confidence_questions WHERE question = :q"),
                {"q": _T11_Q})
        async with get_sessionmaker()() as session:
            await session.execute(stmt, params)
            await session.commit()
            left = (
                await session.execute(
                    text("SELECT COUNT(*) FROM low_confidence_questions WHERE question = :q"),
                    {"q": _T11_Q},
                )
            ).scalar_one()
        # 清理真的生效(id 确实定位到了那一行,不是删了个空气)
        assert left == 0
        await engine.dispose()


@pytest.mark.db
@pytest.mark.anyio
async def test_record_low_confidence_snapshot_defaults_to_null():
    """不传快照 = 落 `null`,**不是空列表**。

    置信度闸那条路径未必总有快照(ch09 §6.2:用户点「没用」时后端重跑检索
    尽力回捞),T12 的反馈端点也会不传。落 `[]` 与落 `null` 在审核页上
    **读起来是两件事**(「当轮零召回」vs「没人记这件事」)。

    ⚠️ 这里断的是**实测到的形状**(2026-09-23 探针):SQLAlchemy 的 `JSON`
    列默认 `none_as_null=False`,Python 的 `None` 被写成 **JSON 的 `null`**,
    **不是 SQL 的 NULL** ⇒ `IS NULL` 是 0、`JSON_TYPE` 是 `'NULL'`。
    两种只要一条断言就够,但**必须断在库里那一层**:只断「ORM 读回是 None」
    的话,「落 JSON null」与「落 SQL NULL」都能过,而它们在 SQL 侧**不等价**
    (后来人拿 `WHERE evidence_snapshot IS NULL` 筛「没快照」会一行都筛不到)。
    """
    engine = get_engine()
    rid = None
    try:
        async with get_sessionmaker()() as session:
            rid = await record_low_confidence(
                session,
                question=_T11_Q,
                source_conversation_id=None,
                entry_point="置信度闸",
                reject_reason="不传快照",
            )
        assert isinstance(rid, int) and rid > 0
        async with get_sessionmaker()() as session:
            raw, is_null, jtype = (
                await session.execute(
                    text("SELECT evidence_snapshot, evidence_snapshot IS NULL, "
                         "JSON_TYPE(evidence_snapshot) "
                         "FROM low_confidence_questions WHERE id = :i"),
                    {"i": rid},
                )
            ).one()
            assert raw == "null" and jtype == "NULL"
            assert is_null == 0          # ← 不是 SQL NULL(别拿 IS NULL 筛它)
    finally:
        async with get_sessionmaker()() as session:
            await session.execute(
                text("DELETE FROM low_confidence_questions WHERE question = :q"),
                {"q": _T11_Q},
            )
            await session.commit()
        await engine.dispose()
