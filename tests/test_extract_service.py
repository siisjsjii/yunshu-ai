import httpx
import openai
import pytest
from langchain_core.exceptions import OutputParserException

from app.schemas import ExtractResult, RequestType
from app.services.extract import ExtractionError, extract_structured


class FakeStructuredModel:
    """替身:with_structured_output 返回一个可 await 的链。"""

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.schema = None
        self.method = None
        self.received = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.method = kwargs.get("method")

        outer = self

        class Chain:
            async def ainvoke(self, messages):
                outer.received = messages
                if outer._error is not None:
                    raise outer._error
                return outer._result

        return Chain()


@pytest.mark.anyio
async def test_extract_returns_parsed_result():
    expected = ExtractResult(
        order_id="20240915",
        request_type=RequestType.EXCHANGE,
        expected_solution="换大一码",
    )
    model = FakeStructuredModel(result=expected)

    result = await extract_structured(model=model, text="订单 20240915 想换大一码")

    assert result is expected


@pytest.mark.anyio
async def test_extract_requests_the_extract_result_schema():
    model = FakeStructuredModel(
        result=ExtractResult(
            order_id=None,
            request_type=RequestType.OTHER,
            expected_solution="x",
        )
    )

    await extract_structured(model=model, text="随便问问")

    assert model.schema is ExtractResult
    assert model.method == "json_mode"


@pytest.mark.anyio
async def test_extract_passes_the_text_to_the_model():
    model = FakeStructuredModel(
        result=ExtractResult(
            order_id=None,
            request_type=RequestType.OTHER,
            expected_solution="x",
        )
    )

    await extract_structured(model=model, text="我的鞋还没发货")

    assert model.received[-1].content == "我的鞋还没发货"


@pytest.mark.anyio
async def test_extract_raises_extraction_error_on_schema_mismatch():
    model = FakeStructuredModel(error=OutputParserException("模型输出不符合 schema"))

    with pytest.raises(ExtractionError, match="不符合 schema"):
        await extract_structured(model=model, text="随便说说")


@pytest.mark.anyio
async def test_extract_does_not_wrap_upstream_failures():
    """上游 401 必须原样穿透,不能被包成"输出不符合 schema"。

    修前这里会拿到 ExtractionError(→ 422),客户端以为是自己输入的问题。
    """
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(401, request=request)
    upstream = openai.AuthenticationError(
        "Error code: 401 - Incorrect API key provided",
        response=response,
        body={"error": {"message": "Incorrect API key provided"}},
    )
    model = FakeStructuredModel(error=upstream)

    with pytest.raises(openai.AuthenticationError):
        await extract_structured(model=model, text="随便说说")
