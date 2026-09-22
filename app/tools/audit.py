"""工具调用审计 —— **唯一写口**。

⚠️ **本文件此刻是占位版**:只打一条 debug 日志,不落库。
真正的实现(新表 `tool_audit_logs` + 独立 session + 失败只 `logger.error`)
在 **T5**。占位先行的理由:执行器(T4)必须在**固定几处**调它,而那几处的
**签名**正是 T5 要对着写的接口 —— 参数名在这一版就定死,别在 T5 里改。

**调用方**(`app/tools/executor.py`)传的参数名是固定的:
`conversation_id` / `tool_call_id` / `tool_name` / `source` / `args` /
`result_summary` / `status` / `retry_count` / `duration_ms`,
外加校验失败时的 `error_detail`。改任何一个都要连带改执行器。
"""

import logging

logger = logging.getLogger(__name__)


async def record_audit(**kwargs) -> None:
    """占位:真正的落库在 T5。"""
    logger.debug("audit %s", kwargs)
