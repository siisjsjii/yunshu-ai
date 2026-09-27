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


def test_message_id_defaults_to_none_and_accepts_a_mysql_pk():
    """ch07:`Message.id` 是可选的 MySQL 主键。

    默认必须**恰好**是 None,不能是「必填」也不能是别的哨兵:`log_turn` 里
    那两条手工构造的消息在落库前**没有** id,T10 播种时又靠 None 区分
    「这条还没进库」。分层(两个锚点比大小)则要求读回来的消息**带着** id。
    """
    assert Message(role="user", content="你好").id is None
    assert Message(role="user", content="你好", id=7).id == 7


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


def test_chat_request_no_longer_has_a_user_id():
    """`user_id` **已删除**(认证,2026-09-27)—— 身份只从 token 来。

    本条替掉的是原 `test_chat_request_user_id_is_optional_and_bounded`(断上限
    128 / 空串 422):字段没了,那三条断言**不可能再成立**,而它们红的原因
    不是缺陷。这里留一个**正向**守卫,免得日后有人又把这个字段加回来。

    ⚠️ **别把它读成「多带一个 `user_id` 会 422」**:`ChatRequest` 没有
    `extra="forbid"` ⇒ 未知键被 pydantic **静默忽略**、请求照发。归属那一侧
    由 `tests/test_api_chat.py::test_token_user_wins_over_a_smuggled_user_id` 钉。
    """
    assert "user_id" not in ChatRequest.model_fields


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
