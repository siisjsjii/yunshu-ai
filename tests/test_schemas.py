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
    """逐个钉住枚举字面量,7 个全钉。

    提示词耦合测试(test_prompts.py)覆盖不到这些取值:它只查提示词里
    是否出现枚举值,而 "发票" 是 "发票问题" 的子串 —— 把枚举改成
    "发票" 它照样通过。这里钉的是契约本身(设计文档 §4.4 列出的取值)。
    """
    assert {member.name: member.value for member in RequestType} == {
        "REFUND": "退货退款",
        "EXCHANGE": "换货",
        "LOGISTICS": "物流异常",
        "INVOICE": "发票问题",
        "PRODUCT": "商品咨询",
        "COMPLAINT": "投诉",
        "OTHER": "其他",
    }
    assert len(RequestType) == 7


def test_chat_request_rejects_empty_session_id():
    """空串不再等同于"新建会话" —— 静默当成新会话会掩盖客户端 bug。"""
    with pytest.raises(ValidationError):
        ChatRequest(session_id="", message="你好")


def test_chat_request_rejects_oversized_session_id():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s" * 129, message="你好")
    assert ChatRequest(session_id="s" * 128, message="你好").session_id == "s" * 128


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
