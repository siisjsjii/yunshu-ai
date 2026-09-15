import logging

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.llm import create_extract_model
from app.sanitize import redact_api_key
from app.schemas import ExtractRequest, ExtractResult
from app.services.extract import ExtractionError, extract_structured

logger = logging.getLogger(__name__)

router = APIRouter()


def get_extract_model(settings: Settings = Depends(get_settings)):
    return create_extract_model(settings)


@router.post("/api/extract", response_model=ExtractResult)
async def extract(
    request: ExtractRequest,
    settings: Settings = Depends(get_settings),
    model=Depends(get_extract_model),
) -> ExtractResult:
    try:
        return await extract_structured(model=model, text=request.text)
    except ExtractionError as exc:
        # 422 只表示"抽取 schema 不符"(设计文档 §6)。
        raise HTTPException(
            status_code=422,
            detail=redact_api_key(str(exc), settings.openai_api_key),
        ) from exc
    except Exception as exc:
        # 上游故障(401/403、超时、限流……):是服务端的问题,不能报 422。
        # 设计文档 §6 未定义该场景,这里取 502 Bad Gateway。
        # detail 用固定文案,不拼接 str(exc) —— openai SDK 的异常文本是
        # "Error code: {status} - {上游响应体}",原样回显会泄漏响应体
        # (其中可能含密钥)。原始异常只进日志。
        logger.exception("抽取上游调用失败")
        raise HTTPException(
            status_code=502,
            detail=redact_api_key("抽取服务暂时不可用", settings.openai_api_key),
        ) from exc
