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


class TransientToolError(Exception):
    """**暂时性**故障(网络抖动、连接被拒、传输中断)。

    与 `ToolInfrastructureError` 的区别是**要不要再试一次**:
    它属于「过一会儿可能就好了」,而「数据库连接串写错了」不是。
    执行器在**重试规则内**重试它(ch08 起该规则由 `spec.kind` 推出:
    只读可重试、写操作结构性不重试);**重试用尽之后仍上抛
    `ToolInfrastructureError`** —— 三次都失败的故障不叫暂时性了,
    这时推 502 比推一句「工具暂时不可用」诚实。

    **谁抛它**:MCP 客户端(T7)—— 那是本仓唯一一条会「抖」的链路,
    内置工具走的是本机 MySQL 与内存 mock,不会凭空断连。
    """
