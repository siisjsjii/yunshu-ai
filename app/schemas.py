from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """会话历史中的一条消息。纯数据,不依赖 LangChain。

    role 含 "tool" 是因为模型可能要求调用工具 —— 那一轮的历史由
    assistant(带 tool_calls)+ tool(带 tool_call_id)两条构成。

    content 默认为空串:模型只申请调用工具、还没产出文字时,
    assistant 的 content 就是空的(上游确实会返回 text == "")。
    """

    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    """对话请求。

    `session_id` 的上限是 32,不是 ch01 的 128:`session_id` 在本章成了
    `conversations.id`(**varchar(32) 主键**),系统自己生成的 id 恒为
    uuid4().hex 的 32 位。留在 128 的话,33–128 字符的 id 会一路走到
    INSERT 才抛 DataError,按错误分类算**不可恢复 → 502** —— 一个参数
    问题被报成服务端故障。收窄后它以 422 被拒,语义诚实。

    `user_id` 的上限 128 与 `conversations.user` 的列宽一致,同理 ——
    否则宽度不一致会以同一个 DataError 形态复现。
    """

    session_id: str | None = Field(default=None, min_length=1, max_length=32)
    message: str = Field(min_length=1)
    user_id: str | None = Field(default=None, min_length=1, max_length=128)


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)


class RequestType(str, Enum):
    """诉求类型。用枚举收口,便于下游统计路由与评估集计算准确率。"""

    REFUND = "退货退款"
    EXCHANGE = "换货"
    LOGISTICS = "物流异常"
    INVOICE = "发票问题"
    PRODUCT = "商品咨询"
    COMPLAINT = "投诉"
    OTHER = "其他"


class ExtractResult(BaseModel):
    """从用户售后描述中抽取的结构化信息。"""

    order_id: str | None = Field(
        default=None,
        description=(
            "订单号。仅当用户明确给出时填写。"
            "无法确定时必须为 null,禁止编造。"
        ),
    )
    request_type: RequestType = Field(
        description="诉求类型,从给定枚举中选择最贴近的一项。",
    )
    expected_solution: str = Field(
        description=(
            "用户期望的解决方案,用一句话概括。"
            "用户未明说时,依据诉求类型给出最合理的一种。"
        ),
    )
