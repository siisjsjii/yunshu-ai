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
    #: MySQL `messages.id`。**本章起由 `load_history` 填充**。
    #:
    #: 为什么必须是它:两个锚点(`summary_upto_msg_id` / `layer1_from_msg_id`)
    #: 存的就是 MySQL 的主键,而分层是拿消息**逐条比对这两个 id** 做的 ——
    #: `Message` 没有 id 的话,分层根本无从下手。
    #:
    #: 顺带解掉另一个坑:`prompts.to_lc_messages` 用它做 LangChain 消息的 `id`。
    #: `add_messages` 是 **append-only**,无 id 的消息会被当场赋一个全新 uuid
    #: ⇒ 重新播种同一批消息会被**再追加一遍**,而每一轮的回复看起来都正常。
    #: 稳定 id 让重播种变成幂等。
    #:
    #: 默认 None:手工构造的消息(如 `log_turn` 里那两条)在落库前没有 id。
    id: int | None = None
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

    ---- ch10 跟进(认证,2026-09-27)----

    `user_id` **已删除**:身份从 token 来(`app/auth.py` 的 `require_user`),
    客户端再也不能自称是谁。「上限 128 与 `conversations.user` 的列宽一致」
    那条事实随字段一起挪进了 `app/db/models.py` 的 `User` docstring
    (那边的 `username` 同样是 128)。

    ⚠️ **本模型没有 `extra="forbid"`** ⇒ 请求体里多带一个 `user_id` 会被
    pydantic **静默忽略**、请求照旧 200(`tests/test_schemas.py` 里那条用例
    记着这件事)。这不是漏洞,但**别以为它还会 422**。

    ---- ch06:`resume` ----

    `message` 从「必填」变成「与 `resume` 二选一」,因为**续跑请求体里没有
    新消息**(spec §5.1):

    ```jsonc
    { "session_id": "…", "message": "…" }              // 开一轮
    { "session_id": "…", "resume": {"order_no": "1002"} } // 从挂起点续跑
    ```

    两条校验都放在**模型层**,不放在端点里:端点拿到的是校验过的对象,而
    422 必须在**流开始之前**返回(见 `app/api/chat.py` 的预算校验注释)。

    `resume` 的形状**只做 `dict` 这一层**:里面那个订单号由
    `refund_nodes._picked_order_no` 认(裸串 / `{"order_no": …}` / 带该属性的
    对象三种都收),它才是「什么算一个有效载荷」的权威。在这里再写一遍形状校验
    = 同一条规则两处实现,漂移的表现是「图收得下、端点却 422」。
    """

    session_id: str | None = Field(default=None, min_length=1, max_length=32)
    message: str | None = Field(default=None, min_length=1)
    #: 从挂起点续跑(点订单卡片)。给订单号即 resume;不给则开新一轮(见 spec F4)。
    resume: dict | None = None

    @model_validator(mode="after")
    def _message_and_resume_are_exclusive(self) -> Self:
        """`message` 与 `resume` **恰好给一个**。

        - **两个都不给**:没有任何东西可跑。必须在**入参**上拒 —— 判据与
          「流开始前的两条 400」同一条:一旦 yield 过首帧就再也改不了状态码,
          请求语义错只能变成一个 200 的 error 帧。
          (早先这里写的是「不放过去的话 `message=None` 会走到 `prepare_turn`
          在 tiktoken 里炸成 500」—— **那句已不成立**:T10 之后 `prepare_turn`
          不收历史、且对 `user_input is None` 有守卫,那条路不可能再抛。
          校验本身仍然对,只是理由要换成上面这条。)
        - **两个都给**:是自相矛盾的请求,而 `resume` 会**静默吞掉** `message`
          —— 用户那句原话既没被回答、也没落库,事后什么都查不到。
          (spec F4 的「挂起时改发普通新消息」是**只给 message** 那条路,与这里
          不冲突;前端也从不两个一起发。)
        """
        if self.message is None and self.resume is None:
            raise ValueError("必须给 message(开新一轮)或 resume(从挂起点续跑)")
        if self.message is not None and self.resume is not None:
            raise ValueError("message 与 resume 互斥,不能同时给")
        return self


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


class RefundRequestIn(BaseModel):
    """退款单提交入参。字段名与前端表单一一对应。

    **与 ORM 的 `app.db.models.RefundRequest` 同名不同物**,别混:这个是 HTTP 入参,
    那个是表行。三个字段的上限都对齐各自的列宽(`conversations.id` varchar(32) /
    `refund_requests.order_no` varchar(32) / `reason_category` varchar(64)),
    理由与 `TicketRequest.session_id` 那处一样 —— 放宽了会一路走到 INSERT 才抛
    DataError,一个参数问题被报成 500/502,语义不诚实。

    `reason_category` 这里**只做形状校验**,不做类目校验:`is_valid_category` 是
    端点的职责(单一来源在 `app.refund.categories`,prompt / 帧 / 表单同读它)。
    """

    session_id: str = Field(min_length=1, max_length=32)
    order_no: str = Field(min_length=4, max_length=32)
    reason_category: str = Field(min_length=1, max_length=64)


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


class LoginRequest(BaseModel):
    """登录请求。

    两个字段都**只做长度**校验:密码**不设** min_length —— 密码策略是产品决定,
    而这里多一条校验会让「旧账号的短密码登录被 422 拒掉」,报错还指向参数形状。
    """

    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    """登录响应。`expires_at` 是 ISO 串(前端拿它显示"什么时候要重新登录")。"""

    token: str
    username: str
    role: str
    expires_at: str
