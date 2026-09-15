class ToolNotFound(Exception):
    """业务性未找到(订单/商品不存在)。

    可恢复:executor 会把它转成 ok=False 的 ToolOutcome,回灌给模型,
    由模型用自然语言兜住。流不会中断。
    """


class ToolInfrastructureError(Exception):
    """基础设施故障(数据库连不上等)。

    不可恢复:executor 向上抛,API 层推 error 帧并终止流 —— 不能让
    "数据库挂了"被伪装成"你的订单号查不到"。
    """
