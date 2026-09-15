import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.exceptions import OutputParserException

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
    """模型输出解析不出来 → 422。这是 §6 里 422 唯一的适用场景。

    异常类型取自实测:json_mode 链路下 PydanticOutputParser 抛的是
    OutputParserException(不是裸 ValueError,也不是 pydantic.ValidationError)。
    """
    from app.api import extract as extract_api

    app.dependency_overrides[extract_api.get_extract_model] = lambda: FakeStructuredModel(
        error=OutputParserException("解析失败")
    )

    resp = client.post("/api/extract", json={"text": "我的鞋没发货"})

    assert resp.status_code == 422
    assert "解析失败" in resp.text


def _auth_error(message: str):
    """构造上游 401 异常,形态与生产一致(openai SDK 的 AuthenticationError)。

    真实运行时抛的是 langchain_openai.OpenAIAuthenticationError —— openai
    AuthenticationError 的子类,这里用父类即可,catch 行为相同。
    """
    import openai

    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(
        401, request=request, json={"error": {"message": message}}
    )
    return openai.AuthenticationError(
        f"Error code: 401 - {response.text}",
        response=response,
        body={"error": {"message": message}},
    )


def test_upstream_auth_failure_is_502_not_422(client, caplog):
    """上游 401 是服务端故障,不是"你的输入不符合 schema"。

    修前:extract_structured 把一切异常都包成 ExtractionError,于是 401
    被报成 422 —— 把服务端的锅甩给用户的文本。修后是 502。
    """
    from app.api import extract as extract_api

    app.dependency_overrides[extract_api.get_extract_model] = lambda: FakeStructuredModel(
        error=_auth_error("Incorrect API key provided: sk-test")
    )

    resp = client.post("/api/extract", json={"text": "我的鞋没发货"})

    assert resp.status_code == 502
    assert "抽取服务暂时不可用" in resp.text
    # 原始异常不丢:只回给客户端的部分被脱敏,服务端日志里留有全文。
    assert "Incorrect API key" in caplog.text


def test_upstream_error_body_is_not_echoed(client):
    """上游响应体不得回显 —— openai 的 str(exc) 就是 "Error code: N - {body}"。

    而且这个 body 在真实 401 里恰好含密钥,所以顺带验证脱敏。
    """
    from app.api import extract as extract_api

    app.dependency_overrides[extract_api.get_extract_model] = lambda: FakeStructuredModel(
        error=_auth_error("Incorrect API key provided: sk-test")
    )

    resp = client.post("/api/extract", json={"text": "我的鞋没发货"})

    assert "Error code: 401" not in resp.text
    assert "Incorrect API key" not in resp.text
    assert "sk-test" not in resp.text
