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

    借用的后果:`route_by_intent` 的返回值 —— 也就是 `graph.py` 那张条件边字典的
    **键** —— 不再能区分「转人工」与「物流/订单」,两桶被焊死在同一个出口上。
    将来要给转人工一条自己的路(换节点、单独统计、单独报表)时**无处可挂**,
    而不会有任何东西报错。

    ⚠️ **这里原先写的理由是错的**(审查 M2):原文说借用了 BUSINESS 之后
    `log_turn` 的 trace 帧会把两者渲染成一个样 —— **不会**。trace 记的是**标签**
    (`app/agent/nodes.py` 那句 `f"classify_intent:{intent}"`),
    `classify_intent:转人工` 与 `classify_intent:物流` **本来就分得开**。
    一个错理由比没有理由更坏:它会引后来人去改 trace(那儿没毛病),
    而真正该守的那一层(路由值)反而没人看 —— **别再把 trace 当这条的理由**。
    """
    assert HANDOFF != BUSINESS
    assert HANDOFF not in (KNOWLEDGE, COMPLAINT, CHITCHAT, FALLBACK, REFUND)


def test_route_values_are_pairwise_distinct():
    """同类不变量,一次覆盖**所有**两两组合(上面那条只管 HANDOFF 参与的那些)。

    按本仓那条元教训:不变量要放在**唯一的地方**,别靠每个调用方自觉 ——
    这里那个地方就是这几个常量本身。把其中任意两个写成同一个值
    (最可能就是「复用某个已有的路由值」,ch10-A 明确否决过的那个决定)
    都在这里红,而上面那条只看得见 HANDOFF 参与的两两组合。

    ⚠️ **断的是「路由值**常量**两两不同」,不是「`INTENT_TO_ROUTE` 的值两两不同」**
    —— 后者**是假的**:九类意图**故意**共用出口(物流/订单共用 BUSINESS,
    退款退货/售后共用 REFUND)。把这两件事混成一条断言,它会红得毫无道理。
    """
    route_values = [KNOWLEDGE, BUSINESS, COMPLAINT, CHITCHAT, FALLBACK, REFUND, HANDOFF]
    assert len(set(route_values)) == len(route_values), (
        f"路由值常量里有重复:{route_values} —— 复用会让两桶在路由层再也分不开"
    )
