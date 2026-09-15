import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.schemas import ExtractResult, RequestType

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
}


class FakeStructuredModel:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def with_structured_output(self, schema, **kwargs):
        outer = self

        class Chain:
            async def ainvoke(self, messages):
                if outer._error is not None:
                    raise outer._error
                return outer._result

        return Chain()


@pytest.fixture
def client():
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, **REQUIRED
    )
    from app.api import extract as extract_api

    app.dependency_overrides[extract_api.get_extract_model] = lambda: FakeStructuredModel(
        result=ExtractResult(
            order_id="20240915",
            request_type=RequestType.EXCHANGE,
            expected_solution="换成大一码",
        )
    )

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


def test_extract_returns_structured_json(client):
    resp = client.post("/api/extract", json={"text": "订单 20240915 想换大一码"})

    assert resp.status_code == 200
    assert resp.json() == {
        "order_id": "20240915",
        "request_type": "换货",
        "expected_solution": "换成大一码",
    }


def test_extract_allows_null_order_id(client):
    from app.api import extract as extract_api

    app.dependency_overrides[extract_api.get_extract_model] = lambda: FakeStructuredModel(
        result=ExtractResult(
            order_id=None,
            request_type=RequestType.OTHER,
            expected_solution="先了解一下",
        )
    )

    resp = client.post("/api/extract", json={"text": "随便问问"})

    assert resp.status_code == 200
    assert resp.json()["order_id"] is None


def test_extract_rejects_empty_text(client):
    assert client.post("/api/extract", json={"text": ""}).status_code == 422


def test_extract_returns_422_on_schema_mismatch(client):
    from app.api import extract as extract_api

    app.dependency_overrides[extract_api.get_extract_model] = lambda: FakeStructuredModel(
        error=ValueError("解析失败")
    )

    resp = client.post("/api/extract", json={"text": "我的鞋没发货"})

    assert resp.status_code == 422
    assert "解析失败" in resp.text
