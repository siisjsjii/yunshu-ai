"""三个固定话术出口:闲聊/兜底零模型调用,投诉额外发 choices 帧。"""

import pytest

from app.agent.nodes import (
    CHOICE_HANDOFF,
    CHOICE_TICKET,
    CHITCHAT_REPLY,
    COMPLAINT_REPLY,
    FALLBACK_REPLY,
    make_chitchat_reply_node,
    make_complaint_reply_node,
    make_fallback_reply_node,
)


@pytest.mark.anyio
async def test_chitchat_emits_token_frame_so_the_bubble_is_not_blank():
    """**必须发 token 帧** —— 前端靠累积 token 画气泡。

    只写 `state["reply"]` 的话:后端 state 里有话、前端气泡是**空的**,
    而所有断言 `out["reply"]` 的单测全绿。验收 4 会直接失败。
    """
    frames = []
    node = make_chitchat_reply_node(emit=frames.append)
    out = await node({"user_input": "你好"})
    assert out["reply"] == CHITCHAT_REPLY
    assert out["choices"] == []          # 闲聊不给按钮
    assert out["trace"] == ["chitchat_reply"]
    assert frames == [{"frame": "token", "text": CHITCHAT_REPLY}]


@pytest.mark.anyio
async def test_fallback_emits_token_frame():
    frames = []
    node = make_fallback_reply_node(emit=frames.append)
    out = await node({"user_input": "帮我写诗", "intent": "其他"})
    assert out["reply"] == FALLBACK_REPLY
    assert out["choices"] == []
    assert out["trace"] == ["fallback_reply"]
    assert frames == [{"frame": "token", "text": FALLBACK_REPLY}]


@pytest.mark.anyio
async def test_complaint_emits_token_then_choices_frame():
    frames = []
    node = make_complaint_reply_node(emit=frames.append)
    out = await node({"user_input": "我要投诉"})

    assert out["reply"] == COMPLAINT_REPLY
    assert out["choices"] == ["handoff", "ticket"]
    assert frames == [
        {"frame": "token", "text": COMPLAINT_REPLY},
        {"frame": "choices", "options": [CHOICE_HANDOFF, CHOICE_TICKET]},
    ]
    # 两个选项是**两件事**,键必须不同且都在
    assert {o["key"] for o in frames[1]["options"]} == {"handoff", "ticket"}


def test_fixed_copy_factories_take_no_model_at_all():
    """静态保证:这三个工厂的签名里根本没有 model 参数。

    比「实现里记得别调模型」强 —— 参数不存在 = 结构上不可能调。
    """
    import inspect

    for factory in (make_chitchat_reply_node, make_fallback_reply_node,
                    make_complaint_reply_node):
        assert "model" not in inspect.signature(factory).parameters
