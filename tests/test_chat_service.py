import pytest
from langchain_core.messages import SystemMessage

from app.config import Settings
from app.memory.store import SessionStore
from app.memory.trim import ContextOverflowError
from app.schemas import Message
from app.services.chat import prepare_turn, stream_turn

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **REQUIRED, **overrides)


def _store() -> SessionStore:
    return SessionStore(ttl_seconds=60, max_sessions=10)


class FakeChunk:
    def __init__(self, text: str, usage=None):
        self.text = text
        self.usage_metadata = usage


class FakeModel:
    """替身模型:按脚本产出 chunk,不联网。"""

    def __init__(self, chunks):
        self._chunks = chunks
        self.received = None

    async def astream(self, messages):
        self.received = messages
        for chunk in self._chunks:
            yield chunk


def test_prepare_turn_returns_system_plus_input_for_new_session():
    messages = prepare_turn(
        settings=_settings(),
        store=_store(),
        session_id="s1",
        user_input="你好",
    )
    assert len(messages) == 2
    assert messages[0].content  # 非空 system prompt
    assert messages[-1].content == "你好"
    # 只断长度和末条的话,[user_input, user_input] 这种实现也能过。
    assert isinstance(messages[0], SystemMessage)


def test_prepare_turn_includes_existing_history():
    store = _store()
    store.append(
        "s1",
        [
            Message(role="user", content="我的订单是 20240915"),
            Message(role="assistant", content="好的,我为您查询"),
        ],
    )

    messages = prepare_turn(
        settings=_settings(),
        store=store,
        session_id="s1",
        user_input="我刚才说的订单号是多少？",
    )

    assert len(messages) == 4
    assert messages[1].content == "我的订单是 20240915"
    assert messages[-1].content == "我刚才说的订单号是多少？"


def test_prepare_turn_raises_when_input_alone_exceeds_budget():
    with pytest.raises(ContextOverflowError):
        prepare_turn(
            settings=_settings(
                context_budget_tokens=200,
                reserved_output_tokens=0,
                safety_margin_tokens=0,
            ),
            store=_store(),
            session_id="s1",
            user_input="退" * 5000,
        )


@pytest.mark.anyio
async def test_stream_turn_yields_tokens_then_done():
    store = _store()
    model = FakeModel([FakeChunk("您"), FakeChunk("好")])
    messages = prepare_turn(
        settings=_settings(), store=store, session_id="s1", user_input="你好"
    )

    events = [
        ev
        async for ev in stream_turn(
            settings=_settings(),
            store=store,
            model=model,
            session_id="s1",
            user_input="你好",
            messages=messages,
        )
    ]

    assert events[0] == ("token", {"text": "您"})
    assert events[1] == ("token", {"text": "好"})
    assert events[-1][0] == "done"


@pytest.mark.anyio
async def test_stream_turn_persists_both_messages_on_success():
    store = _store()
    model = FakeModel([FakeChunk("好的")])
    messages = prepare_turn(
        settings=_settings(), store=store, session_id="s1", user_input="你好"
    )

    async for _ in stream_turn(
        settings=_settings(),
        store=store,
        model=model,
        session_id="s1",
        user_input="你好",
        messages=messages,
    ):
        pass

    assert store.history("s1") == [
        Message(role="user", content="你好"),
        Message(role="assistant", content="好的"),
    ]


@pytest.mark.anyio
async def test_stream_turn_discards_history_when_stream_breaks():
    """流中途断掉时,半截回复不写入历史。"""
    store = _store()

    class ExplodingModel:
        async def astream(self, messages):
            yield FakeChunk("前半")
            raise RuntimeError("上游炸了")

    messages = prepare_turn(
        settings=_settings(), store=store, session_id="s1", user_input="你好"
    )

    with pytest.raises(RuntimeError, match="上游炸了"):
        async for _ in stream_turn(
            settings=_settings(),
            store=store,
            model=ExplodingModel(),
            session_id="s1",
            user_input="你好",
            messages=messages,
        ):
            pass

    assert store.history("s1") == []


@pytest.mark.anyio
async def test_stream_turn_done_event_carries_usage_when_available():
    store = _store()
    model = FakeModel(
        [
            FakeChunk("好"),
            FakeChunk("", usage={"input_tokens": 10, "output_tokens": 1}),
        ]
    )
    messages = prepare_turn(
        settings=_settings(), store=store, session_id="s1", user_input="你好"
    )

    events = [
        ev
        async for ev in stream_turn(
            settings=_settings(),
            store=store,
            model=model,
            session_id="s1",
            user_input="你好",
            messages=messages,
        )
    ]

    assert events[-1][1]["usage"] == {"input_tokens": 10, "output_tokens": 1}


@pytest.mark.anyio
async def test_stream_turn_keeps_earlier_usage_when_a_later_chunk_has_none():
    """后一个 chunk 的 usage_metadata 为 None 时,不能把先前的 usage 冲掉。

    真实上游只有最后一帧带 usage;若实现写成 `usage = chunk.usage_metadata`
    (丢掉 `or usage`),这一条会退化成 None。
    """
    store = _store()
    model = FakeModel(
        [
            FakeChunk("好", usage={"input_tokens": 10, "output_tokens": 1}),
            FakeChunk("", usage=None),
        ]
    )
    messages = prepare_turn(
        settings=_settings(), store=store, session_id="s1", user_input="你好"
    )

    events = [
        ev
        async for ev in stream_turn(
            settings=_settings(),
            store=store,
            model=model,
            session_id="s1",
            user_input="你好",
            messages=messages,
        )
    ]

    assert events[-1][1]["usage"] == {"input_tokens": 10, "output_tokens": 1}


@pytest.mark.anyio
async def test_stream_turn_done_usage_is_none_when_absent():
    store = _store()
    model = FakeModel([FakeChunk("好")])
    messages = prepare_turn(
        settings=_settings(), store=store, session_id="s1", user_input="你好"
    )

    events = [
        ev
        async for ev in stream_turn(
            settings=_settings(),
            store=store,
            model=model,
            session_id="s1",
            user_input="你好",
            messages=messages,
        )
    ]

    assert events[-1][1]["usage"] is None


@pytest.mark.anyio
async def test_stream_turn_skips_empty_token_chunks():
    store = _store()
    model = FakeModel([FakeChunk(""), FakeChunk("好")])
    messages = prepare_turn(
        settings=_settings(), store=store, session_id="s1", user_input="你好"
    )

    events = [
        ev
        async for ev in stream_turn(
            settings=_settings(),
            store=store,
            model=model,
            session_id="s1",
            user_input="你好",
            messages=messages,
        )
    ]

    assert [e for e in events if e[0] == "token"] == [("token", {"text": "好"})]
