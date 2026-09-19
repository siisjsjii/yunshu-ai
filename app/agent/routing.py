"""意图 → 出口的映射。

**纯函数、无 IO、无模型调用** —— 这是「确定性骨架」的核心:模型只能决定
**意图标签**,不能决定**走向**。分流规则写死在代码里,故可以表驱动单测
穷举七类 + 越界 + 缺字段。
"""

KNOWLEDGE = "knowledge"
BUSINESS = "business"
COMPLAINT = "complaint"
CHITCHAT = "chitchat"
FALLBACK = "fallback"

#: 意图识别失败或输出越界时统一落这个标签,再由本表送进兜底出口。
OTHER = "其他"

#: 七类意图 → 四个出口。
#: 商品咨询 / 退款退货 → 知识(强制预检索;退款退货的 Agent 仍可自调订单工具);
#: 物流 / 订单 / 售后 → 业务数据(直接进 Agent 调工具,无检索证据故不过置信度闸);
#: 投诉、闲聊各有专属出口。
INTENT_TO_ROUTE: dict[str, str] = {
    "商品咨询": KNOWLEDGE,
    "退款退货": KNOWLEDGE,
    "物流": BUSINESS,
    "订单": BUSINESS,
    "售后": BUSINESS,
    "投诉": COMPLAINT,
    "闲聊": CHITCHAT,
}

#: 意图识别 Prompt 里允许输出的标签(与 INTENT_TO_ROUTE 同源,避免两处各写一份)。
INTENT_LABELS: tuple[str, ...] = tuple(INTENT_TO_ROUTE)


def route_by_intent(state) -> str:
    """七类之一 → 四出口;其余(解析失败 / 越界 / 缺字段)一律兜底。

    这里**不做**任何清洗(不 strip、不大小写归一):清洗会让「投诉 」这种
    近乎正确的输入静默走进投诉出口,而它更可能是一次真正的分类失败。
    宁可让它进兜底,兜底话术会请用户再说一遍。
    """
    return INTENT_TO_ROUTE.get(state.get("intent") or "", FALLBACK)
