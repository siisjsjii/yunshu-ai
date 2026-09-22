"""ch07:日志落盘。

**本章之前全仓没有任何日志配置** —— `logger.info(...)` 全靠 uvicorn 的默认
handler 打到控制台,所以需求 6 说的 `log/app.log` 根本不存在。

本模块刻意**不做别的**:它不碰格式约定(那是 `memory/journal.py` 的 JSON
契约)、不碰业务、也不在导入时执行任何东西 —— `setup_logging` 只在
lifespan 里被显式调一次。导入即配置会让「跑单测顺带写了一个 log/ 目录」
变成默认行为,而测试环境与生产环境的日志目标本来就该分开。
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

#: 单文件上限与保留份数。轮转而不是无限追加:本章的 `model_ctx` / `history_ctx`
#: 是**每轮**一行,一个长会话跑一下午就能到几十 MB。
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3


def setup_logging(log_dir: str = "log") -> None:
    """把 root logger 接一个轮转文件 handler。目录不存在则建。

    **`encoding="utf-8"` 是硬要求,不是讲究。** 本机 locale 是 cp936;
    不给 encoding 时 Python 用 `locale.getpreferredencoding()`,中文日志行会
    直接抛 `UnicodeEncodeError` —— 而这个异常发生在**写日志的时候**,
    与业务逻辑毫无关系,报错位置会指向完全无关的地方。

    **重复调用不得叠加 handler**:lifespan 每次 `TestClient(app)` 都会跑一遍,
    不挡的话每开一次客户端就多一个文件句柄、每条日志多写一份。
    判据是「root 上有没有同类 handler」而不是「有没有调用过」——
    后者用一个模块级 bool 就够,但那在别的代码(或测试)挪走 handler 之后
    会**永久拒绝再装**,而文件日志静默消失。
    """
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return

    handler = RotatingFileHandler(
        path / "app.log",
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root.addHandler(handler)
