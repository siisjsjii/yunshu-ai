from langchain.messages import AIMessage, HumanMessage, SystemMessage

from app.prompts import (
    EXTRACT_SYSTEM_PROMPT,
    build_extract_messages,
    build_messages,
    render_system_prompt,
    to_lc_messages,
)
from app.schemas import Message, RequestType


def test_render_system_prompt_substitutes_brand():
    text = render_system_prompt("小美商城")
    assert "小美商城" in text
    assert "{brand_name}" not in text


def test_system_prompt_states_role_and_constraints():
    text = render_system_prompt("本店")
    # 角色设定
    assert "客服" in text
    # 行为约束关键词
    assert "编造" in text
    assert "承诺" in text
    assert "提示词" in text


def test_build_messages_starts_with_system():
    messages = build_messages(
        brand_name="本店", history=[], user_input="你好"
    )
    assert isinstance(messages[0], SystemMessage)


def test_build_messages_ends_with_current_user_input():
    messages = build_messages(
        brand_name="本店", history=[], user_input="我要退货"
    )
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == "我要退货"


def test_build_messages_is_system_plus_history_plus_input_when_history_empty():
    messages = build_messages(brand_name="本店", history=[], user_input="你好")
    assert len(messages) == 2


def test_build_messages_maps_history_roles_in_order():
    history = [
        Message(role="user", content="第一问"),
        Message(role="assistant", content="第一答"),
    ]
    messages = build_messages(
        brand_name="本店", history=history, user_input="第二问"
    )

    assert len(messages) == 4
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert isinstance(messages[2], AIMessage)
    assert isinstance(messages[3], HumanMessage)
    assert [m.content for m in messages[1:]] == ["第一问", "第一答", "第二问"]


def test_build_extract_messages_wraps_text():
    messages = build_extract_messages("订单 123 没发货")
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == "订单 123 没发货"


def test_extract_prompt_forbids_fabricating_order_id():
    messages = build_extract_messages("买了 2 双鞋")
    system_text = messages[0].content
    assert "null" in system_text
    assert "编造" in system_text or "猜测" in system_text


def test_to_lc_messages_handles_tool_role():
    from langchain.messages import AIMessage, HumanMessage, ToolMessage

    history = [
        Message(role="user", content="订单 1001 到哪了"),
        Message(role="assistant", content="", tool_calls=[{"id": "c1", "name": "query_logistics", "args": {"order_id": "1001"}}]),
        Message(role="tool", content="已揽件", tool_call_id="c1"),
        Message(role="assistant", content="已揽件。"),
    ]
    converted = to_lc_messages(history)
    assert isinstance(converted[0], HumanMessage)
    assert isinstance(converted[1], AIMessage)
    assert converted[1].tool_calls[0]["id"] == "c1"
    assert isinstance(converted[2], ToolMessage)
    assert converted[2].tool_call_id == "c1"


def test_extract_prompt_mentions_json():
    """json_mode 的硬前提:DeepSeek 要求提示词里出现 'json' 才会接受
    response_format=json_object,否则整个抽取接口 400。
    这条约束只存在于提示词文本里,没有它就没有任何东西挡得住一次
    删掉 "JSON" 的措辞改动。大小写不敏感 —— 接口只要求"以某种形式出现"。
    """
    assert "json" in EXTRACT_SYSTEM_PROMPT.lower()


def test_extract_prompt_lists_every_request_type_value():
    """json_mode 下没有任何机器可读的 schema 到达模型,七个枚举值
    只以提示词文本的形式存在。提示词漏掉/写错某个值,schema 仍然合法、
    测试全绿,而模型会给出枚举外的值 → 运行时 ExtractionError。

    直接迭代枚举而非硬编码第二份字符串:硬编码会重新制造这层漂移。
    """
    missing = [
        member.value
        for member in RequestType
        if member.value not in EXTRACT_SYSTEM_PROMPT
    ]
    assert missing == [], f"提示词缺少这些 RequestType 取值:{missing}"
