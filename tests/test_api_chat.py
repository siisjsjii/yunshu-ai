import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.memory.store import SessionStore

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
}


class FakeChunk:
    def __init__(self, text, usage=None):
        self.text = text
        self.usage_metadata = usage


class SlowModel:
    """可控替身:astream 逐条吐出预置 chunk,并记下最后一次收到的消息。"""

    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = 0
        self.last_messages = None

    async def astream(self, messages):
        self.calls += 1
        self.last_messages = messages
        for chunk in self._chunks:
            yield chunk


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """把 SSE 响应体解析成 (event, data) 列表。"""
    events = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        name, payload = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                payload = json.loads(line[len("data: ") :])
        events.append((name, payload))
    return events


@pytest.fixture
def client():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    fake = SlowModel([FakeChunk("您"), FakeChunk("好")])

    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, **REQUIRED
    )
    from app.api import chat as chat_api

    app.dependency_overrides[chat_api.get_store] = lambda: store
    app.dependency_overrides[chat_api.get_chat_model] = lambda: fake

    with TestClient(app) as c:
        c.store = store
        c.fake_model = fake
        yield c

    app.dependency_overrides.clear()


def test_stream_emits_meta_then_tokens_then_done(client):
    resp = client.post("/api/chat/stream", json={"message": "你好"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    assert events[0][0] == "meta"
    assert events[-1][0] == "done"
    tokens = [d["text"] for name, d in events if name == "token"]
    assert tokens == ["您", "好"]


def test_stream_generates_session_id_when_omitted(client):
    resp = client.post("/api/chat/stream", json={"message": "你好"})
    meta = _parse_sse(resp.text)[0][1]
    assert isinstance(meta["session_id"], str)
    assert meta["session_id"]


def test_stream_echoes_provided_session_id(client):
    resp = client.post(
        "/api/chat/stream", json={"session_id": "s1", "message": "你好"}
    )
    assert _parse_sse(resp.text)[0][1]["session_id"] == "s1"


def test_meta_reports_the_model_name(client):
    resp = client.post("/api/chat/stream", json={"message": "你好"})
    assert _parse_sse(resp.text)[0][1]["model"] == "test-model"


def test_second_turn_receives_first_turn_context(client):
    client.post("/api/chat/stream", json={"session_id": "s1", "message": "订单是 20240915"})

    client.post(
        "/api/chat/stream",
        json={"session_id": "s1", "message": "刚才那个订单号是多少？"},
    )

    sent = client.fake_model.last_messages
    contents = [m.content for m in sent]
    assert "订单是 20240915" in contents
    assert contents[-1] == "刚才那个订单号是多少？"


def test_unknown_session_id_is_created_silently(client):
    resp = client.post(
        "/api/chat/stream", json={"session_id": "brand-new", "message": "你好"}
    )
    assert resp.status_code == 200
    assert _parse_sse(resp.text)[0][1]["session_id"] == "brand-new"


def test_empty_message_is_rejected(client):
    assert client.post("/api/chat/stream", json={"message": ""}).status_code == 422


def test_missing_message_is_rejected(client):
    assert client.post("/api/chat/stream", json={}).status_code == 422


def test_oversized_input_returns_400_before_streaming(client):
    from app.api import chat as chat_api

    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        **REQUIRED,
        context_budget_tokens=200,
        reserved_output_tokens=0,
        safety_margin_tokens=0,
    )

    resp = client.post("/api/chat/stream", json={"message": "退" * 5000})

    assert resp.status_code == 400
    assert "tokens" in resp.text


def test_upstream_error_becomes_sse_error_event(client):
    from app.api import chat as chat_api

    class ExplodingModel:
        async def astream(self, messages):
            yield FakeChunk("前半")
            raise RuntimeError("上游超时")

    app.dependency_overrides[chat_api.get_chat_model] = lambda: ExplodingModel()

    resp = client.post("/api/chat/stream", json={"message": "你好"})

    events = _parse_sse(resp.text)
    assert events[-1][0] == "error"
    assert "上游超时" in events[-1][1]["message"]
    assert "sk-test" not in resp.text
