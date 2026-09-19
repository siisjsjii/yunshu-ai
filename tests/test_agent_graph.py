"""整图行为:五条出口各走一遍,断言 trace(验收 1/5 的可检查性来源)。"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.emit import make_emitter
from app.agent.graph import build_graph, get_checkpointer
from app.config import Settings
from app.retrieval.search import RetrievedChunk


class FakeChunk:
    def __init__(self, text="", tool_calls=None, usage=None):
        self.text = text
        self.tool_calls = [{"type": "tool_call", **tc} for tc in (tool_calls or [])]
        self.usage_metadata = usage

    def __add__(self, other):
        # 与 T6 的替身同一条理由:**累积时不能丢 tool_calls,更不能丢
        # `"type": "tool_call"` 键**(见 Ruling 21)。丢了前者,「模型在一轮里
        # 分多次吐完一次工具调用」在替身里就永远累积不起来;丢了后者,
        # `BaseTool.ainvoke` 会把 dict 当**参数**去校验 schema(CLAUDE.md 的硬约束)。
        # 本文件的用例目前每轮只有一个 chunk,所以这条分支**走不到** ——
        # 但替身必须忠实于真实 chunk 的形状,否则它会教后来的人写错的形状,
        # 而且错得**不会红**。真实的多 chunk 累积由 T11 的验收 5 兜底。
        return FakeChunk(
            self.text + other.text,
            tool_calls=self.tool_calls + other.tool_calls,
        )


class ScriptedModel:
    """每次 astream 都从 next(脚本)取一批。"""

    def __init__(self, rounds):
        self.rounds = list(rounds)

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, method=None):
        return self

    async def ainvoke(self, messages):
        batch = self.rounds.pop(0)
        return batch[0] if isinstance(batch, list) else batch

    async def astream(self, messages):
        batch = self.rounds.pop(0)
        for chunk in (batch if isinstance(batch, list) else [batch]):
            yield chunk


class FakeRetriever:
    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.calls = []

    async def search(self, query):
        self.calls.append(query)
        return list(self.chunks)


class RecordingSession:
    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass


def _settings(**over):
    return Settings(
        _env_file=None,
        openai_base_url="https://example.invalid/v1", openai_api_key="sk-test",
        openai_model="m", database_url="mysql+asyncmy://u:p@h:3306/db", **over,
    )


def _graph(intent, *, retriever=None, rounds=None, settings=None, frames=None):
    session = RecordingSession()
    # rounds is None 才用默认;显式传 [] 要保留成空脚本 ——
    # 那是「碰模型就炸」的探针,`rounds or [...]` 会把空列表换掉、探针失效。
    graph = build_graph(
        model=ScriptedModel(rounds if rounds is not None else [[FakeChunk("模型回复")]]),
        intent_model=ScriptedModel([[type("_I", (), {"intent": intent})()]]),
        tools=[], registry={}, settings=settings or _settings(),
        retriever=retriever or FakeRetriever(),
        session=session, conversation_id="conv-1",
        emit=(frames.append if frames is not None else (lambda p: None)),
        # **每个测试一个全新的 checkpointer**,不用 `get_checkpointer()` 那个
        # 进程级单例 —— 真机已验证:同一个 thread 上重复 ainvoke,带
        # `operator.add` 的 `trace` 会**跨调用累积**(实测 `['a','b','a','b']`)。
        # 而所有测试都用默认 thread "t",于是**测试之间互相污染**:
        # `test_business_route_skips_retrieval_and_gate` 断言的
        # `"confidence_gate:pass" not in out["trace"]` 会被前面那个知识类测试
        # 留下的同一句打红 —— 而它红得毫无道理,指向的是测试脚手架而非实现。
        # 单例本身由 `test_checkpointer_is_a_process_level_singleton` 单独覆盖,
        # 生产路径由 T8 的端点测试覆盖。
        checkpointer=InMemorySaver(),
    )
    return graph, session


async def _run(graph, user_input, thread="t"):
    # `agent_steps` 必须在入参里**显式给初值**:真机已验证,一个从未被写过的
    # 通道在返回的 state 里**根本不存在**(实测 `"agent_steps" in out` 为 False),
    # 于是 `assert out["agent_steps"] == 0` 抛的是 KeyError ——
    # 那个红指向「测试写错了」,而不是「闸没拦住」,把真问题盖掉。
    # 给了初值之后,这条断言才真的在问「Agent 到底有没有跑」。
    return await graph.ainvoke(
        {"conversation_id": "conv-1", "user_input": user_input, "history": [],
         "agent_steps": 0, "trace": []},
        config={"configurable": {"thread_id": thread}},
    )


@pytest.mark.anyio
async def test_knowledge_intent_forces_retrieval_before_agent():
    """验收 1:政策类问题,trace 里必须看到强制检索节点被走到。"""
    retriever = FakeRetriever([RetrievedChunk("怎么退货", "七天无理由", "退换货",
                                              chunk_id=1, section_path="退货政策", score=0.9)])
    graph, _ = _graph("商品咨询", retriever=retriever)
    out = await _run(graph, "怎么退货")
    assert retriever.calls == ["怎么退货"]
    assert "retrieve_knowledge:1 hits top=0.90" in out["trace"]
    assert "confidence_gate:pass" in out["trace"]


@pytest.mark.anyio
async def test_weak_evidence_skips_agent_and_records_low_confidence():
    retriever = FakeRetriever([RetrievedChunk("q", "a", "c", chunk_id=1,
                                              section_path=None, score=0.10)])
    frames = []
    graph, session = _graph("商品咨询", retriever=retriever, frames=frames)
    out = await _run(graph, "冷门问题")
    assert out["gate_passed"] is False
    assert out["agent_steps"] == 0          # **没进 Agent**
    assert session.added[0].entry_point == "置信度闸"


@pytest.mark.anyio
async def test_business_route_skips_retrieval_and_gate():
    """业务数据类不预检索、不过闸(没有检索证据,证据强弱无从谈起)。"""
    retriever = FakeRetriever()
    graph, _ = _graph("物流", retriever=retriever)
    out = await _run(graph, "订单 1001 的物流到哪了")
    assert retriever.calls == []
    assert "confidence_gate:pass" not in out["trace"]
    assert "agent:converged" in out["trace"]


@pytest.mark.anyio
async def test_chitchat_returns_fixed_copy_and_never_calls_the_model():
    graph, _ = _graph("闲聊", rounds=[])     # 脚本为空:碰模型就会 IndexError
    out = await _run(graph, "你好")
    assert "客服小猫" in out["reply"]
    assert "classify_intent:闲聊" in out["trace"]


@pytest.mark.anyio
async def test_complaint_emits_choices_and_returns_soothing_copy():
    frames = []
    graph, _ = _graph("投诉", rounds=[], frames=frames)
    out = await _run(graph, "我要投诉")
    assert out["choices"] == ["handoff", "ticket"]
    assert {"frame": "choices", "options": [
        {"key": "handoff", "label": "转人工"}, {"key": "ticket", "label": "建工单"}]} in frames


@pytest.mark.anyio
async def test_fallback_route_for_out_of_vocabulary_intent():
    graph, _ = _graph("其他", rounds=[])
    out = await _run(graph, "帮我写首诗")
    assert "没太理解" in out["reply"]
    assert "classify_intent:其他" in out["trace"]


@pytest.mark.anyio
async def test_successful_turn_is_persisted_to_mysql_history():
    graph, session = _graph("闲聊", rounds=[])
    await _run(graph, "你好")
    roles = [m.role for m in session.added]
    assert roles == ["user", "assistant"]


@pytest.mark.anyio
async def test_checkpointer_is_a_process_level_singleton():
    """每请求新建 checkpointer 的话,下一轮 thread 状态就没了。"""
    assert get_checkpointer() is get_checkpointer()


@pytest.mark.anyio
async def test_log_turn_emits_trace_frame_for_end_to_end_acceptance():
    """`trace` 要能被验收脚本从 SSE 里读到(验收 1/5 靠它机械可断言)。

    `log_turn` 用同一个 emit 通道发一帧 `trace`;端点把它折进 `done` 帧、
    不外推给前端。走同一条发出路径 = 不需要第二个 stream_mode。
    """
    frames = []
    graph, _ = _graph("闲聊", rounds=[], frames=frames)
    await _run(graph, "你好")
    kinds = [p["frame"] for p in frames]
    assert "trace" in kinds
    trace = next(p for p in frames if p["frame"] == "trace")["trace"]
    assert "resolve_references" in trace
    assert "classify_intent:闲聊" in trace
    assert "chitchat_reply" in trace
    assert trace[-1] == "log_turn"


@pytest.mark.anyio
async def test_emitter_sends_frames_through_astream_custom_mode():
    """**这条最关键**:证明 emit 在真实的图运行中把帧送到了 astream 消费端。

    缺了它,把 `make_emitter()` 写成「一次性 `get_stream_writer()`」的错误版本
    也能全绿 —— 而那种写法下前端**一帧都收不到**(图外必然抛 RuntimeError,
    静默退化成 no-op)。这正是本项目最怕的「假绿 + 静默故障」。
    """
    collected = []                       # 图外兜底:真跑到这里就说明没拿到 writer
    emit = make_emitter(collected.append)
    session = RecordingSession()
    graph = build_graph(
        model=ScriptedModel([]),
        intent_model=ScriptedModel([[type("_I", (), {"intent": "投诉"})()]]),
        tools=[], registry={}, settings=_settings(), retriever=FakeRetriever(),
        session=session, conversation_id="conv-emit", emit=emit,
        checkpointer=get_checkpointer(),
    )

    got = []
    async for payload in graph.astream(
        {"conversation_id": "conv-emit", "user_input": "我要投诉", "history": [], "trace": []},
        config={"configurable": {"thread_id": "emit-test"}},
        stream_mode="custom",
    ):
        got.append(payload)

    assert {"frame": "choices", "options": [
        {"key": "handoff", "label": "转人工"}, {"key": "ticket", "label": "建工单"}]} in got
    assert collected == []               # 图内拿到了真 writer,没退化成 collector
