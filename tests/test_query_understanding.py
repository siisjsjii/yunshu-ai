"""Query 理解:prompt 两禁(json_mode)、解析、失败退化。全程 fake model,不联网。"""

import pytest
from langchain_core.exceptions import OutputParserException

from app.retrieval.query_understanding import (
    QUERY_UNDERSTANDING_PROMPT,
    QueryRewriteResult,
    QueryUnderstanding,
    QueryVariant,
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


def test_prompt_contains_literal_json():
    """json_mode 链路要求提示词里出现字面 JSON(ch02 血泪)。"""
    assert "JSON" in QUERY_UNDERSTANDING_PROMPT


def test_prompt_has_no_bare_braces():
    """描述结构不得用裸花括号(ChatPromptTemplate 按 f-string 解析)。"""
    assert "{" not in QUERY_UNDERSTANDING_PROMPT
    assert "}" not in QUERY_UNDERSTANDING_PROMPT


def _run(model):
    import asyncio

    return asyncio.run(QueryUnderstanding(model).rewrite("猫砂盆 pro 多少钱"))


def test_rewrite_uses_json_mode_and_maps_result():
    model = _FakeModel(QueryRewriteResult(
        normalized="智能猫砂盆 Pro 价格", synonyms=["MH-LP100 售价"]))
    result = _run(model)
    assert result == QueryVariant("智能猫砂盆 Pro 价格", ["MH-LP100 售价"])
    assert model.calls[0][0] is QueryRewriteResult
    assert model.calls[0][1]["method"] == "json_mode"


def test_rewrite_falls_back_to_original_query_on_parse_error():
    """解析失败不中断检索,退化为原始 query。"""
    model = _FakeModel(error=OutputParserException("bad json"))
    result = _run(model)
    assert result == QueryVariant("猫砂盆 pro 多少钱", [])


def test_rewrite_strips_empty_synonyms():
    model = _FakeModel(QueryRewriteResult(normalized="猫砂盆价格", synonyms=[" ", ""]))
    result = _run(model)
    assert result == QueryVariant("猫砂盆价格", [])
