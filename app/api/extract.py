from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.llm import create_extract_model
from app.schemas import ExtractRequest, ExtractResult
from app.services.extract import ExtractionError, extract_structured

router = APIRouter()


def get_extract_model(settings: Settings = Depends(get_settings)):
    return create_extract_model(settings)


@router.post("/api/extract", response_model=ExtractResult)
async def extract(
    request: ExtractRequest,
    model=Depends(get_extract_model),
) -> ExtractResult:
    try:
        return await extract_structured(model=model, text=request.text)
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
