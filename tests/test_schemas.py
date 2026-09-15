import pytest
from pydantic import ValidationError

from app.schemas import ChatRequest, ExtractResult, Message, RequestType


def test_message_accepts_known_roles():
    assert Message(role="user", content="你好").role == "user"
    assert Message(role="assistant", content="您好").role == "assistant"


def test_message_rejects_unknown_role():
    with pytest.raises(ValidationError):
        Message(role="system", content="x")


def test_chat_request_session_id_is_optional():
    assert ChatRequest(message="你好").session_id is None
    assert ChatRequest(session_id="s1", message="你好").session_id == "s1"


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_request_type_values_are_chinese_labels():
    assert RequestType.REFUND.value == "退货退款"
    assert RequestType.EXCHANGE.value == "换货"
    assert RequestType.OTHER.value == "其他"
    assert len(RequestType) == 7


def test_extract_result_allows_null_order_id():
    result = ExtractResult(
        order_id=None,
        request_type=RequestType.LOGISTICS,
        expected_solution="查询物流进度",
    )
    assert result.order_id is None


def test_extract_result_requires_request_type():
    with pytest.raises(ValidationError):
        ExtractResult(order_id="1", expected_solution="x")


def test_extract_result_rejects_unknown_request_type():
    with pytest.raises(ValidationError):
        ExtractResult(order_id=None, request_type="随便", expected_solution="x")
