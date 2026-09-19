import re
from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class Message(BaseModel):
    """会话历史中的一条消息。纯数据,不依赖 LangChain。

    role 含 "tool" 是因为模型可能要求调用工具 —— 那一轮的历史由
    assistant(带 tool_calls)+ tool(带 tool_call_id)两条构成。

    content 默认为空串:模型只申请调用工具、还没产出文字时,
    assistant 的 content 就是空的(上游确实会返回 text == "")。

    `role="tool"` 必须带 `tool_call_id`,由下面的 validator 强制 —— 见其
    docstring 里那段实测记录。
    """

    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _tool_message_needs_its_call_id(self) -> Self:
        """`role="tool"` 而没有 `tool_call_id` 一律拒绝。

        这条不变量此前只是"写路径碰巧对":唯一的生产写入点
        (app/services/chat.py 的 `Message(role="tool", ...)`)总是带上
        `tool_call.id`,所以线上没炸过。但类型本身允许构造出这个形态,
        而它转成 ToolMessage 时 `tool_call_id` 退化成空串,发到上游**真的
        是畸形的** —— 实测 `convert_to_openai_messages(ToolMessage(
        content="x", tool_call_id=""))` 得到
        `{'role': 'tool', 'tool_call_id': '', 'content': 'x'}`,
        上游回一个无从解释的 400,而历史已经这样写进了库(会话被毒化,
        之后每次读历史都复现)。放在类型上,这条不变量就不再依赖调用方自觉。

        空串同样拒绝:app/prompts.py 的 `message.tool_call_id or ""`
        正是把 None 退化成空串的那一步,`not` 一并盖住两者。
        """
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError('role="tool" 的消息必须带非空 tool_call_id')
        return self


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


class TicketRequest(BaseModel):
    """建工单请求(ch05「建工单」按钮)。

    `session_id` 的上限 32 与 `ChatRequest` 一致,理由见那处的 docstring:
    它是 `conversations.id` 的 varchar(32) 主键,放宽会以 DataError 形态复现
    并被错误分类判成不可恢复 → 502。
    """

    session_id: str = Field(min_length=1, max_length=32)


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


class MinedQaItem(BaseModel):
    """从历史对话里挖出的一条问答对(ch03 挖知识管线)。"""

    question: str = Field(description="用户在对话中提出的问法,尽量用用户原话。")
    answer: str = Field(description="客服在对话中给出的答案,必须来自对话原文。")
    category: str = Field(description="这条知识所属的分类,如 退换货 / 物流异常。")


class MinedQaBatch(BaseModel):
    """一次抽取的返回体。

    **外层必须是对象、数组放在 `items` 里**:json_mode 下模型被要求输出一个
    JSON 对象,根直接给数组时解析行为不稳定(而且要跨供应商)。多包一层
    没有代价,却把返回形状钉死了。
    """

    items: list[MinedQaItem] = Field(
        default_factory=list, description="本批对话中挖出的问答对,没有就给空数组。"
    )


class UploadDocumentRequest(BaseModel):
    """上传知识文档请求(ch04 管理台)。文件名只收安全 Markdown;type 限 ch03 三类。"""

    filename: str
    type: Literal["policy", "faq", "manual"]
    content: str

    @field_validator("filename")
    @classmethod
    def _safe_md_name(cls, v: str) -> str:
        # 只收 \w 与 - 组成的 .md,拒绝路径分隔符与 `..`(防写越出 knowledge/ 目录)。
        if not re.fullmatch(r"[\w\-]+\.md", v):
            raise ValueError("文件名须为 [字母数字_-] 组成的 .md,不含路径分隔符")
        return v


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
