"""强制预检索 + 置信度闸:阈值边界、落池、以及「闸不过就不进 Agent」。"""

import pytest

from app.agent.nodes import make_confidence_gate_node, make_retrieve_knowledge_node
from app.config import Settings
from app.retrieval.search import RetrievedChunk
from app.tools.errors import ToolInfrastructureError

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **REQUIRED, **overrides)


class FakeRetriever:
    def __init__(self, chunks=(), error=None):
        self.chunks = list(chunks)
        self.error = error
        self.calls = []

    async def search(self, query):
        self.calls.append(query)
        if self.error is not None:
            raise self.error
        return list(self.chunks)


class RecordingSession:
    def __init__(self):
        self.added = []
        self.commits = 0

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1


def _chunk(score, answer="七天无理由退货"):
    return RetrievedChunk("怎么退货", answer, "退换货", chunk_id=7,
                          section_path="退换货 > 退货政策", score=score)


@pytest.mark.anyio
async def test_retrieval_uses_the_resolved_input_and_builds_citations():
    frames = []
    retriever = FakeRetriever([_chunk(0.91)])
    node = make_retrieve_knowledge_node(retriever=retriever, emit=frames.append)
    out = await node({"resolved_input": "怎么退货"})
    assert retriever.calls == ["怎么退货"]
    assert out["evidence"][0]["score"] == 0.91
    assert out["citations"] == [{
        "n": 1, "chunk_id": 7, "section_path": "退换货 > 退货政策",
        "question": "怎么退货", "answer": "七天无理由退货", "category": "退换货",
    }]
    assert out["trace"] == ["retrieve_knowledge:1 hits top=0.91"]


@pytest.mark.anyio
async def test_citations_are_emitted_as_a_frame():
    """**citations 必须发帧** —— 前端靠它渲染可点击的引用来源。

    只写进 state 的话:ch04 的引用 UI 静默失效、老的 acceptance.sh 回归,
    而所有断言 `out["citations"]` 的单测**照样全绿**。
    """
    frames = []
    node = make_retrieve_knowledge_node(retriever=FakeRetriever([_chunk(0.91)]),
                                        emit=frames.append)
    out = await node({"resolved_input": "怎么退货"})
    # 载荷键是 `items` —— ch04 前端读的就是 `payload.items`(见 index.html:368)。
    # 断言写成 `{"citations": ...}` 的话,把键改错也照样绿。
    assert frames == [{"frame": "citations", "items": out["citations"]}]


@pytest.mark.anyio
async def test_empty_retrieval_is_recorded_in_trace_not_an_error():
    frames = []
    node = make_retrieve_knowledge_node(retriever=FakeRetriever([]), emit=frames.append)
    out = await node({"resolved_input": "没有的问题"})
    assert out["evidence"] == []
    assert out["citations"] == []
    assert out["trace"] == ["retrieve_knowledge:0 hits"]
    assert frames == []          # 没证据就不发空引用帧


@pytest.mark.anyio
async def test_infrastructure_failure_propagates_to_502():
    """检索器挂了必须抛上去(→502),绝不降级成「没搜到」。"""
    node = make_retrieve_knowledge_node(
        retriever=FakeRetriever(error=ToolInfrastructureError("检索不可用")),
        emit=lambda p: None,
    )
    with pytest.raises(ToolInfrastructureError):
        await node({"resolved_input": "q"})


@pytest.mark.anyio
async def test_gate_passes_at_threshold_boundary_inclusive():
    """0.58 恰好等于阈值 → 通过(与 ch04 的阈值语义一致:>=)。"""
    session = RecordingSession()
    node = make_confidence_gate_node(
        settings=_settings(retrieval_score_threshold=0.58),
        session=session, conversation_id="conv-1",
    )
    out = await node({"user_input": "q", "evidence": [{"score": 0.58}]})
    assert out["gate_passed"] is True
    assert out["trace"] == ["confidence_gate:pass"]
    assert session.added == []          # 通过时不落池


@pytest.mark.anyio
async def test_gate_fails_below_threshold_and_records_low_confidence():
    session = RecordingSession()
    node = make_confidence_gate_node(
        settings=_settings(retrieval_score_threshold=0.58),
        session=session, conversation_id="conv-1",
    )
    out = await node({"user_input": "怎么退货", "evidence": [{"score": 0.31}]})
    assert out["gate_passed"] is False
    assert out["trace"] == ["confidence_gate:fail"]
    assert len(session.added) == 1
    row = session.added[0]
    assert row.entry_point == "置信度闸"
    assert row.question == "怎么退货"
    assert row.source_conversation_id == "conv-1"
    assert "0.31" in row.reject_reason
    assert session.commits == 1


@pytest.mark.anyio
async def test_gate_fails_on_empty_evidence_and_records_that_reason():
    session = RecordingSession()
    node = make_confidence_gate_node(
        settings=_settings(), session=session, conversation_id="c",
    )
    out = await node({"user_input": "q", "evidence": []})
    assert out["gate_passed"] is False
    assert session.added[0].reject_reason == "检索为空"


@pytest.mark.anyio
async def test_gate_uses_max_score_not_top1_position():
    """判据是**最高分**,不是「第一条的分」—— 顺序由重排决定,取 max 更稳。"""
    session = RecordingSession()
    node = make_confidence_gate_node(
        settings=_settings(retrieval_score_threshold=0.58),
        session=session, conversation_id="c",
    )
    out = await node({"user_input": "q", "evidence": [{"score": 0.2}, {"score": 0.9}]})
    assert out["gate_passed"] is True
