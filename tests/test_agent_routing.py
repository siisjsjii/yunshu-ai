"""分流规则是确定性骨架的核心,必须表驱动全覆盖 —— 含越界与缺字段。"""

import pytest

from app.agent.routing import (
    BUSINESS,
    CHITCHAT,
    COMPLAINT,
    FALLBACK,
    INTENT_LABELS,
    INTENT_TO_ROUTE,
    KNOWLEDGE,
    route_by_intent,
)

CASES = [
    ("商品咨询", KNOWLEDGE),
    ("退款退货", KNOWLEDGE),
    ("物流", BUSINESS),
    ("订单", BUSINESS),
    ("售后", BUSINESS),
    ("投诉", COMPLAINT),
    ("闲聊", CHITCHAT),
]


@pytest.mark.parametrize("intent,expected", CASES)
def test_seven_intents_map_to_their_outlets(intent, expected):
    assert route_by_intent({"intent": intent}) == expected


def test_all_seven_intents_are_covered():
    """七类一个不漏 —— 少一类会静默落进兜底,而兜底不调模型,问题就永远答不上。"""
    assert set(INTENT_TO_ROUTE) == {c[0] for c in CASES}
    assert INTENT_LABELS == tuple(INTENT_TO_ROUTE)


@pytest.mark.parametrize("bad", ["其他", "", "投诉 ", "COMPLAINT", "退款退货 "])
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
