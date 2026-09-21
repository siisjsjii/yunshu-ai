"""意图 → 出口的映射。

**纯函数、无 IO、无模型调用** —— 这是「确定性骨架」的核心:模型只能决定
**意图标签**,不能决定**走向**。分流规则写死在代码里,故可以表驱动单测
穷举八类 + 越界 + 缺字段。
"""

KNOWLEDGE = "knowledge"
BUSINESS = "business"
COMPLAINT = "complaint"
CHITCHAT = "chitchat"
FALLBACK = "fallback"
#: ch06:退款退货 / 售后走**确定性子流程**(见 `app/agent/refund_nodes.py`)。
REFUND = "refund"

#: 意图识别失败或输出越界时统一落这个标签,再由本表送进兜底出口。
OTHER = "其他"

#: 八类意图 → 五出口。
#: 商品咨询 → 知识(强制预检索 + 置信度闸);
#: 物流 / 订单 → 业务数据(直接进 Agent 调工具,无检索证据故不过置信度闸);
#: **退款退货 / 售后 → 退款子流程**(ch06:取这一单 → 查条款 → 判一次 → 给入口
#: 或解释;缺订单号时在子流程里 `interrupt` 弹卡片);
#: 投诉、闲聊各有专属出口;其他 → 兜底。
INTENT_TO_ROUTE: dict[str, str] = {
    "商品咨询": KNOWLEDGE,
    "退款退货": REFUND,
    "物流": BUSINESS,
    "订单": BUSINESS,
    "售后": REFUND,
    "投诉": COMPLAINT,
    "闲聊": CHITCHAT,
    # 显式写进来才让 `INTENT_LABELS` 带得上它 —— 提示词的标签表与这张表同源,
    # 少一行,模型就永远学不到「其他」这个词(行为上本就落 `.get()` 的默认值)。
    OTHER: FALLBACK,
}

#: 意图识别 Prompt 里允许输出的标签(与 INTENT_TO_ROUTE 同源,避免两处各写一份)。
INTENT_LABELS: tuple[str, ...] = tuple(INTENT_TO_ROUTE)


def route_by_intent(state) -> str:
    """八类之一 → 四出口;其余(解析失败 / 越界 / 缺字段)一律兜底。

    这里**不做**任何清洗(不 strip、不大小写归一):清洗会让「投诉 」这种
    近乎正确的输入静默走进投诉出口,而它更可能是一次真正的分类失败。
    宁可让它进兜底,兜底话术会请用户再说一遍。
    """
    return INTENT_TO_ROUTE.get(state.get("intent") or "", FALLBACK)
