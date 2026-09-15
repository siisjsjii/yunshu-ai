import asyncio
import json

import httpx
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


def test_error_event_does_not_echo_the_configured_key(client):
    """异常文本里带着真实密钥时必须被抹掉。

    与 test_upstream_error_becomes_sse_error_event 里的断言不同,这条
    能真的失败:那个替身抛的是 RuntimeError("上游超时"),而夹具里配的
    密钥是字面量 "sk-test",没有任何代码路径能把它放进响应 —— 断言恒真。
    这里让异常文本**自带**夹具配置的密钥值,才真正检验过滤逻辑。
    """
    from app.api import chat as chat_api

    class LeakyModel:
        async def astream(self, messages):
            raise RuntimeError("Incorrect API key provided: sk-test")
            yield FakeChunk("永不产出")

    app.dependency_overrides[chat_api.get_chat_model] = lambda: LeakyModel()

    resp = client.post("/api/chat/stream", json={"message": "你好"})

    events = _parse_sse(resp.text)
    assert events[-1][0] == "error"
    assert "sk-test" not in resp.text
    assert "***" in events[-1][1]["message"]


@pytest.mark.anyio
async def test_concurrent_same_session_second_request_times_out_with_409():
    """同 session 并发:一个拿到锁走完,另一个等锁超时 → 恰好一个 409。

    设计文档 §6 的最后一行,也是唯一没有测试覆盖的一行。必须真并发 ——
    顺序调用只会得到两个 200。

    用 httpx.AsyncClient + ASGITransport 而不是 TestClient:TestClient
    是同步的,两个线程里跑同一事件循环会带来额外的调度不确定性。
    """
    from app.api import chat as chat_api

    store = SessionStore(ttl_seconds=60, max_sessions=10)

    class SlowModel:
        """持有锁约 0.8s:远长于 0.15s 的等锁超时。"""

        async def astream(self, messages):
            await asyncio.sleep(0.4)
            yield FakeChunk("您")
            await asyncio.sleep(0.4)
            yield FakeChunk("好")

    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, **REQUIRED, session_lock_timeout_seconds=0.15
    )
    app.dependency_overrides[chat_api.get_store] = lambda: store
    app.dependency_overrides[chat_api.get_chat_model] = lambda: SlowModel()

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as c:
            first, second = await asyncio.gather(
                c.post(
                    "/api/chat/stream",
                    json={"session_id": "s1", "message": "你好"},
                ),
                c.post(
                    "/api/chat/stream",
                    json={"session_id": "s1", "message": "在吗"},
                ),
            )
    finally:
        app.dependency_overrides.clear()

    assert sorted([first.status_code, second.status_code]) == [200, 409]
    loser = first if first.status_code == 409 else second
    assert "正在处理另一条消息" in loser.text

    # 输的那个请求没有写历史 —— 恰好一轮(user + assistant)。
    history = store.history("s1")
    assert [m.role for m in history] == ["user", "assistant"]


def test_validation_failure_releases_lock(client, monkeypatch):
    from app.api import chat as chat_api

    def boom(*, settings, store, session_id, user_input):
        raise RuntimeError("组装消息失败")

    monkeypatch.setattr(chat_api, "prepare_turn", boom)

    # 未处理的异常在默认 TestClient(raise_server_exceptions=True)下会被
    # 重新抛出,而不是变成 500 响应 —— 生产环境下它才是 500。这里用
    # pytest.raises 断言"没有得到 200 SSE 流",再断言锁已释放。
    with pytest.raises(RuntimeError):
        client.post(
            "/api/chat/stream", json={"session_id": "s1", "message": "你好"}
        )

    assert client.store.lock_for("s1").locked() is False
