from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router

app = FastAPI(title="电商智能客服 ch01")
app.include_router(chat_router)
app.include_router(extract_router)
