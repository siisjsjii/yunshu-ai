import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.api.extract import router as extract_router
from app.api.feedback import router as feedback_router
from app.api.kb import router as kb_router
from app.api.refund import router as refund_router
from app.config import get_settings
from app.logging_setup import setup_logging
from app.memory import budget
from app.prompts import render_system_prompt

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


def _startup_budget_self_check() -> None:
    """启动时自检:历史预算连**一轮**都装不下就报警。

    `fits_one_round` 在本章**只有这一个消费者**(请求路径上判的是别的口径)——
    它盯的是一种很难自己暴露出来的配置故障:窗口被固定开销与单轮峰值吃光,
    于是每一轮都带着**空历史**往下走,用户拿到一个没有任何上下文的回答,
    而没有任何东西报错。

    **报警但不拒绝启动**(spec §7.2):需求原文是「报警」,而写成拒绝启动会把
    一个纯算术问题变成服务起不来。两者只差一行,是一句话可翻的选择。

    **pytest 下跳过**:与 `_warm_embedding_in_background` 同款。`TestClient`
    会跑 lifespan,而单测不该因为一个算术配置去打 error 日志(那会让
    「测试输出干净」这条断掉,而它指向的还是配置而不是被测代码)。
    """
    if "pytest" in sys.modules:
        return

    settings = get_settings()
    b = budget.derive(
        settings=settings, system_prompt=render_system_prompt(settings.brand_name)
    )
    if not b.fits_one_round:
        logger.error(
            "上下文预算不足:历史预算 %s < 每轮稳态 %s —— 连一轮都装不下。"
            "查 model_context_window / max_output_tokens / max_agent_steps 等项。",
            b.history_budget, settings.per_round_steady,
        )
    else:
        logger.info(
            "上下文预算:窗口 %s − 固定开销 %s − 单轮峰值 %s = 历史 %s(层1 %s / 层2 %s)",
            b.window, b.fixed_overhead, b.peak,
            b.history_budget, b.layer1_budget, b.layer2_budget,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 日志落盘**在最前**:自检与预热打的每一条日志都该进 log/app.log,
    # 顺序反了的话最先那几条(恰恰是启动期最有用的)只留在控制台。
    #
    # **pytest 下不落盘**。理由有两条,第二条是承重的:
    # ① `TestClient` 会跑 lifespan,而它在单测里到处都是 —— 每跑一次套件就往
    #    仓库根写一个真实文件(实测一次全量套件 108 KB),而「单测不写文件系统」
    #    是本章 Global Constraints 的一条;spec §10.3 也把「日志落盘本身」
    #    明确划给**验收脚本**覆盖(单测的替身盖不住真实文件句柄的编码行为)。
    # ② 验收 4b 判的是 `grep log/app.log`。单测里那些端点用例**同样会打
    #    `history_ctx`** —— 不挡住的话,同一个文件里躺着上一次单测的产物,
    #    验收的 grep 会因为**旧行**而恒真。这正是本仓反复栽的那类假绿。
    if "pytest" not in sys.modules:
        setup_logging()
    _startup_budget_self_check()
    _warm_embedding_in_background()
    yield


app = FastAPI(title="电商智能客服 ch02", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(extract_router)
app.include_router(kb_router)
app.include_router(refund_router)
# ch07 的两个只读端点(会话侧栏)。**与其余 router 同在 mount("/") 之前** ——
# 顺序反了的话静态目录会把 /api/* 抢走,表现为"新端点 404 而服务照常起"。
app.include_router(conversations_router)
# ch09 用户反馈落池(飞轮入口 ③)。同样**必须在 `mount("/")` 之前** ——
# 不注册时它不会 404 而是 **405**:静态目录的 catch-all 只放行 GET/HEAD。
app.include_router(feedback_router)

# 静态页必须**最后**挂:mount("/") 会接管根路径,先挂会抢走 /api/*。
_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
