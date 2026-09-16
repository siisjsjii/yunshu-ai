from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router

app = FastAPI(title="电商智能客服 ch02")
app.include_router(chat_router)
app.include_router(extract_router)

# 静态页必须**最后**挂:mount("/") 会接管根路径,先挂会抢走 /api/*。
_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
