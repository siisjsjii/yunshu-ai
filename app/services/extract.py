from app.prompts import build_extract_messages
from app.schemas import ExtractResult


class ExtractionError(Exception):
    """模型输出无法解析为 ExtractResult。"""


async def extract_structured(*, model, text: str) -> ExtractResult:
    """从售后描述中抽取结构化字段。

    不做自动重试 —— 重试次数应由评估数据决定,不凭感觉设定。
    失败直接抛错,由 API 层转成 422。
    """
    chain = model.with_structured_output(ExtractResult, method="json_mode")
    try:
        return await chain.ainvoke(build_extract_messages(text))
    except Exception as exc:
        raise ExtractionError(f"模型输出不符合 schema:{exc}") from exc
