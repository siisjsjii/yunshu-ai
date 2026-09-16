import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.config import get_settings

logger = logging.getLogger(__name__)


def _warm_embedding_in_background() -> None:
    """后台线程把 BGE-M3 权重读进内存。

    为什么必须预热:首次加载要十几秒,而工具执行有 10 秒超时
    (`tool_timeout_seconds`)—— 冷进程的第一个 query_faq 必然超时,用户看到
    的是「数据服务暂时不可用」(实测:acceptance.sh 验收 1 就是这个形态)。

    为什么放后台线程:阻塞启动会让每次 `uvicorn` 都干等十几秒。

    为什么 pytest 下跳过:**单测不加载 2.2GB 模型是本章硬规矩**(见
    CLAUDE.md / spec §8.1)。TestClient 会跑 lifespan,不挡住的话跑一次
    接口测试就会把权重load 进来。
    """
    if "pytest" in sys.modules:
        return

    def _load() -> None:
        try:
            from app.retrieval.embedder import get_embedder

            settings = get_settings()
            get_embedder(
                settings.embedding_model_path,
                settings.embedding_max_length,
                settings.embedding_batch_size,
            ).warmup()
            logger.info("BGE-M3 预热完成")
        except Exception:  # noqa: BLE001
            # 预热失败不该拖垮服务:真去用它的时候会再抛一次,那才是该报错的位置。
            logger.exception("BGE-M3 预热失败,首次检索会退化为现场加载")

    threading.Thread(target=_load, name="warmup-embedding", daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warm_embedding_in_background()
    yield


app = FastAPI(title="电商智能客服 ch02", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(extract_router)

# 静态页必须**最后**挂:mount("/") 会接管根路径,先挂会抢走 /api/*。
_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
