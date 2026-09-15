from langchain.messages import AIMessage, HumanMessage, SystemMessage

from app.prompts import (
    build_extract_messages,
    build_messages,
    render_system_prompt,
)
from app.schemas import Message


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
