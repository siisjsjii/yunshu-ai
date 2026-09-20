"""退款原因固定类目 —— **单一来源**。

前后端共用:`POST /api/refund` 用它做校验,`refund_offer` 帧用它下发选项。
**前端不得硬编码这份列表**(选项从帧里来),否则两处会漂移。
"""

#: 用户提交退款单时从这几项里自选。**不追问原因**,这是产品的明确选择。
REFUND_REASON_CATEGORIES: tuple[str, ...] = (
    "商品质量问题",
    "不想要了",
    "发错货",
    "少发/漏发",
    "与描述不符",
)


def is_valid_category(value: str) -> bool:
    """**精确匹配,不做 strip/大小写归一**。

    与 `app/agent/routing.py` 的 `route_by_intent` 同一理由:清洗会让
    「商品质量问题 」这种近乎正确的输入静默通过,而它更可能是一次真实的
    传参错误。宁可 422。
    """
    return value in REFUND_REASON_CATEGORIES
