"""强制预检索 + 置信度闸:阈值边界、落池、以及「闸不过就不进 Agent」。"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import app.observability as observability
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
    node = make_retrieve_knowledge_node(
        retriever=retriever, emit=frames.append, settings=_settings()
    )
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
                                        emit=frames.append, settings=_settings())
    out = await node({"resolved_input": "怎么退货"})
    # 载荷键是 `items` —— ch04 前端读的就是 `payload.items`(见 index.html:368)。
    # 断言写成 `{"citations": ...}` 的话,把键改错也照样绿。
    assert frames == [{"frame": "citations", "items": out["citations"]}]


@pytest.mark.anyio
async def test_empty_retrieval_is_recorded_in_trace_not_an_error():
    frames = []
    node = make_retrieve_knowledge_node(
        retriever=FakeRetriever([]), emit=frames.append, settings=_settings()
    )
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
        settings=_settings(),
    )
    with pytest.raises(ToolInfrastructureError):
        await node({"resolved_input": "q"})


@pytest.mark.anyio
async def test_gate_passes_at_threshold_boundary_inclusive():
    """**恰好等于阈值** → 通过(语义是 `>=`,边界含等号)。

    ch09 起闸比的是 `evidence_confidence(...)`,所以这里的边界值**由那条算术
    算出来**,不再是「传一个分数进去、拿它当阈值」:

        单条 score=0.5 ⇒ top1=0.5、条数=1(count_signal=min(1/3,1))、gap=0.5
        confidence = 0.6*0.5 + 0.2*(1/3) + 0.2*0.5 = 0.46666… → round(…,4) = 0.4667

    所以 0.4667 是"恰好等于",0.4668 是"高一点点" —— **两个方向都要断**,
    只断"等于时通过"的话,一个写成 `>` 的实现照样绿。
    """
    session = RecordingSession()
    at = make_confidence_gate_node(
        settings=_settings(evidence_confidence_threshold=0.4667),
        session=session, conversation_id="conv-1",
    )
    out = await at({"user_input": "q", "evidence": [{"score": 0.5}]})
    assert out["gate_passed"] is True
    assert out["trace"] == ["confidence_gate:pass"]
    assert session.added == []          # 通过时不落池

    above = make_confidence_gate_node(
        settings=_settings(evidence_confidence_threshold=0.4668),
        session=session, conversation_id="conv-1",
    )
    out = await above({"user_input": "q", "evidence": [{"score": 0.5}]})
    assert out["gate_passed"] is False


@pytest.mark.anyio
async def test_gate_fails_below_threshold_and_records_low_confidence():
    session = RecordingSession()
    node = make_confidence_gate_node(
        settings=_settings(evidence_confidence_threshold=0.5),
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
    # ch09:判据从"取最高分"换成 evidence_confidence,**三个信号都要写进 reason**,
    # 审核人才看得出为什么被拦。文案变了,判据没变。
    assert "置信度" in row.reject_reason
    assert "低于阈值" in row.reject_reason
    assert "top1=" in row.reject_reason
    assert "条数=" in row.reject_reason
    assert "分差=" in row.reject_reason
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
    """判据取的是**最高分**,不是「第一条的分」—— 顺序由重排决定,取 max 更稳。

    ch09 起这条性质落在 `evidence_detail` 的 `top1 = 过滤后的 max(score)` 上。
    阈值 0.5 是**刻意夹在两种实现的输出之间**的:

        取 max:top1=0.9、条数=2、gap=0.7 ⇒ 0.6*0.9 + 0.2*(2/3) + 0.2*0.7 = 0.8133 ⇒ 过
        取第一条:top1=0.2、条数=1、gap=0.2 ⇒ 0.6*0.2 + 0.2*(1/3) + 0.2*0.2 = 0.2267 ⇒ 拦

    所以「按位置取第一条」的实现会让这条用例红。阈值若落在 0.8133 之上或
    0.2267 之下,两种实现都一样 —— 用例就失去判别力。
    """
    session = RecordingSession()
    node = make_confidence_gate_node(
        settings=_settings(evidence_confidence_threshold=0.5),
        session=session, conversation_id="c",
    )
    out = await node({"user_input": "q", "evidence": [{"score": 0.2}, {"score": 0.9}]})
    assert out["gate_passed"] is True


# ---- 观测:检索必须留下 span(ch09 T3)--------------------------------------


class _SpanSpy:
    """捕获 `observability.span` 的实参,并 yield 一个记录 `update` 的假 handle。

    断言打在**这个边界上**: `name` / `as_type` / `input` 键名 / `output` 字段
    是「检索结果都能铺开看」这条需求的落点,而它**在节点返回的 dict 里一个字
    都看不见**(`evidence` 是给 state 与 prompt 用的,与观测无关)。
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.updates: list[dict] = []

    def __call__(self, name, *, as_type="span", input=None, settings):
        self.calls.append({"name": name, "as_type": as_type, "input": input})
        updates = self.updates

        @contextmanager
        def _cm():
            yield SimpleNamespace(update=lambda **kw: updates.append(kw))

        return _cm()


@pytest.mark.anyio
async def test_retrieval_node_opens_a_span_with_the_resolved_input(monkeypatch):
    """检索节点必须手工开 `retrieval` span —— `KnowledgeRetriever` **不是**
    LangChain run,回调一个 span 都不会给它。

    形状与退款那一路(`tests/test_agent_refund.py` 的同名用例)逐字对齐:
    两条检索在界面上必须是同一种读法。`input` 断的是**真正喂给 retriever**
    的那个值(`resolved_input`,不是 `user_input` 原话)。
    """
    spy = _SpanSpy()
    monkeypatch.setattr(observability, "span", spy)
    node = make_retrieve_knowledge_node(
        retriever=FakeRetriever([_chunk(0.91)]), emit=lambda p: None, settings=_settings()
    )
    await node({"resolved_input": "怎么退货"})

    assert spy.calls == [{
        "name": "retrieval",
        "as_type": "retriever",
        "input": {"query": "怎么退货"},
    }]
    assert spy.updates == [{
        "output": {"chunks": [{
            "id": 7, "score": 0.91, "section_path": "退换货 > 退货政策",
        }]}
    }]
