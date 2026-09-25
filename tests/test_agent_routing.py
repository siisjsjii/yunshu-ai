"""分流规则是确定性骨架的核心,必须表驱动全覆盖 —— 含越界与缺字段。"""

import pytest

from app.agent.routing import (
    BUSINESS,
    CHITCHAT,
    COMPLAINT,
    FALLBACK,
    HANDOFF,
    INTENT_LABELS,
    INTENT_TO_ROUTE,
    KNOWLEDGE,
    OTHER,
    REFUND,
    route_by_intent,
)

CASES = [
    ("商品咨询", KNOWLEDGE),
    # ch06:退款退货 / 售后从「知识 / 业务」改走**退款子流程**(spec §3.2)。
    # 两行都必须改 —— 只改一行的话,另一条意图会静默留在旧出口上:
    # 售后走 BUSINESS 时用户拿到的是「Agent 调工具回答」,永远弹不出订单卡片。
    ("退款退货", REFUND),
    ("物流", BUSINESS),
    ("订单", BUSINESS),
    ("售后", REFUND),
    ("投诉", COMPLAINT),
    ("闲聊", CHITCHAT),
    # ch06 起「其他」是**显式标签**(spec §3.2):它本来就落 `.get()` 的默认值,
    # 进表是为了让 `INTENT_LABELS` 带得上它 —— 提示词的标签表与这张表同源。
    (OTHER, FALLBACK),
    # ch10-A:「转人工」是**第九类**。它由主力 Agent 调模拟接口完成,
    # 所以路由值是 HANDOFF、目标节点是 agent —— **不开新出口**。
    ("转人工", HANDOFF),
]


@pytest.mark.parametrize("intent,expected", CASES)
def test_every_intent_maps_to_its_outlet(intent, expected):
    assert route_by_intent({"intent": intent}) == expected


def test_every_intent_is_covered():
    """九类一个不漏 —— 少一类会静默落进兜底,而兜底不调模型,问题就永远答不上。"""
    assert set(INTENT_TO_ROUTE) == {c[0] for c in CASES}
    assert INTENT_LABELS == tuple(INTENT_TO_ROUTE)


@pytest.mark.parametrize("bad", ["", "投诉 ", "COMPLAINT", "退款退货 "])
def test_unknown_or_malformed_intent_falls_back(bad):
    """越界/空串/带空格一律兜底 —— 不是 schema 校验,是**走向**的兜底。"""
    assert route_by_intent({"intent": bad}) == FALLBACK


@pytest.mark.parametrize("state", [{}, {"intent": None}])
def test_missing_intent_falls_back(state):
    assert route_by_intent(state) == FALLBACK


def test_make_emitter_outside_graph_degrades_to_collector():
    """图外调用必须退化到 collector,不能抛 RuntimeError(见 spec §12 订正)。

    (旧名 `..._is_a_noop_collector` 名实不符:走的就是那条**会调用** collector
    的分支 —— 本项目已抓到过一次「名字说回灌、函数体在断言 raises」。)
    """
    from app.agent.emit import make_emitter

    got = []
    emit = make_emitter(got.append)
    emit({"frame": "token", "text": "x"})
    assert got == [{"frame": "token", "text": "x"}]


def test_outlets_are_still_exactly_five():
    """出口数**锁死在 5**。ch10-A 加第九类意图时**没有**开新出口,这条是那个决定的守卫。

    加出口是个需要被看见的决定(同 `policy.WRITE_TOOLS` 那条精确相等断言的
    先例):改这里就必须改这条测试,改的时候你会被迫想一遍「真的需要第六个出口吗」。
    """
    from app.agent.graph import _OUTLETS

    assert len(_OUTLETS) == 5


def test_handoff_route_value_is_not_reused():
    """`HANDOFF` 必须是**新的**路由值,不能借用 BUSINESS。

    借用的后果:`log_turn` 的 trace 帧与 `route_by_intent` 的返回值里,
    「转人工」与「物流/订单」长得一模一样 —— 事后想统计「有多少轮真的走了转人工」
    时,这个数**永远取不出来**,而没有任何东西报错。
    """
    assert HANDOFF != BUSINESS
    assert HANDOFF not in (KNOWLEDGE, COMPLAINT, CHITCHAT, FALLBACK, REFUND)
