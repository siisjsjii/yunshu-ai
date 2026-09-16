import pytest
from pydantic import ValidationError

from app.schemas import ChatRequest, ExtractResult, Message, RequestType


def test_message_accepts_known_roles():
    assert Message(role="user", content="你好").role == "user"
    assert Message(role="assistant", content="您好").role == "assistant"


def test_message_rejects_unknown_role():
    with pytest.raises(ValidationError):
        Message(role="system", content="x")


def test_tool_message_requires_a_non_empty_tool_call_id():
    """`role="tool"` 不带 tool_call_id 必须在**构造时**被拒。

    放过去的话,app/prompts.py 会把它转成
    `ToolMessage(content=..., tool_call_id="")`,而实测该形态经
    `convert_to_openai_messages` 真的会发出去:
    `{'role': 'tool', 'tool_call_id': '', 'content': 'x'}` —— 上游一个
    无从解释的 400。空串与 None 都要拒:前者正是 `or ""` 那一步的产物。
    """
    with pytest.raises(ValidationError):
        Message(role="tool", content="已揽件")
    with pytest.raises(ValidationError):
        Message(role="tool", content="已揽件", tool_call_id="")

    assert Message(role="tool", content="已揽件", tool_call_id="c1").tool_call_id == "c1"

    # 其余两个角色不受这条约束(它们的 tool_call_id 本就该是 None)。
    assert Message(role="user", content="你好").tool_call_id is None
    assert Message(role="assistant", content="您好").tool_call_id is None


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
    """上限 32 = conversations.id 的列宽(ch02 由 128 收窄)。

    本用例原先钉的是 128。收窄的理由见 app/schemas.ChatRequest 的
    docstring:33–128 字符的 id 会在 INSERT 处抛 DataError,被错误分类
    判成不可恢复 → 502,把参数问题报成服务端故障。
    """
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s" * 33, message="你好")
    assert ChatRequest(session_id="s" * 32, message="你好").session_id == "s" * 32


def test_chat_request_user_id_is_optional_and_bounded():
    """user_id 上限 128 = conversations.user 的列宽,宽度不一致会复现同一个 DataError。"""
    assert ChatRequest(message="你好").user_id is None
    assert ChatRequest(message="你好", user_id="alice").user_id == "alice"
    with pytest.raises(ValidationError):
        ChatRequest(message="你好", user_id="")
    with pytest.raises(ValidationError):
        ChatRequest(message="你好", user_id="u" * 129)
    assert ChatRequest(message="你好", user_id="u" * 128).user_id == "u" * 128


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
