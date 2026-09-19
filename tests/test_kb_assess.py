"""生成 QC:自评 prompt 两禁 + json_mode + 解析失败退化;落池 db 测试。

注意:被测的 `assess_sufficiency` 当前**不在请求路径上**(ch05 spec §50 用事前
置信度闸取代了它的事后自评),本文件全绿不代表线上有这条链路;理由见该函数 docstring。
"""

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
