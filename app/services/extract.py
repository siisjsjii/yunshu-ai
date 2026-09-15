from langchain_core.exceptions import OutputParserException

from app.prompts import build_extract_messages
from app.schemas import ExtractResult


class ExtractionError(Exception):
    """模型输出无法解析为 ExtractResult(设计文档 §6 的 422)。"""


async def extract_structured(*, model, text: str) -> ExtractResult:
    """从售后描述中抽取结构化字段。

    不做自动重试 —— 重试次数应由评估数据决定,不凭感觉设定。

    只把"模型输出不符合 schema"归为 ExtractionError(→ 422)。上游故障
    (401 密钥错、超时、限流)原样向上抛,由 API 层转成 502 —— 否则服务端
    故障会被报成"你的输入不匹配 schema",把锅甩给用户的文本。

    实测(`json_mode` 链路 = prompt | llm.bind(response_format) |
    PydanticOutputParser):模型返回的 JSON 字段不合法时抛出的是
    `langchain_core.exceptions.OutputParserException`(ValueError 的子类,
    但**不是** pydantic.ValidationError)。这条结论不是照类名猜的,见
    测试里直接构造该类型的用例。
    """
    chain = model.with_structured_output(ExtractResult, method="json_mode")
    try:
        return await chain.ainvoke(build_extract_messages(text))
    except OutputParserException as exc:
        raise ExtractionError(f"模型输出不符合 schema:{exc}") from exc
