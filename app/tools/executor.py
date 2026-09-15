"""工具执行:超时、重试、错误分类。"""

import asyncio
import logging
from dataclasses import dataclass

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.tools.errors import ToolInfrastructureError, ToolNotFound

logger = logging.getLogger(__name__)

# 重试用**白名单**:只有幂等的查询类工具可重试。
# create_ticket 是写操作,超时后重试会建出两张工单 —— 而"超时"恰恰意味着
# 我们不知道第一次到底成没成。默认不重试、显式声明可重试,比反过来安全。
RETRYABLE_TOOLS = frozenset(
    {"query_order", "query_product", "query_logistics", "query_faq"}
)

# tool_result 事件里给前端展示的摘要上限(spec §5.2)。
SUMMARY_MAX_CHARS = 200


@dataclass(frozen=True)
class ToolOutcome:
    tool_call_id: str
    name: str
    ok: bool
    content: str    # 完整内容,回灌给模型
    summary: str    # 截断后的展示用摘要


def _summarize(text: str) -> str:
    text = text.strip()
    if len(text) <= SUMMARY_MAX_CHARS:
        return text
    return text[:SUMMARY_MAX_CHARS] + "…"


async def execute_tool(*, tool_call: dict, registry: dict, settings) -> ToolOutcome:
    """执行一次工具调用。

    可恢复的失败返回 ok=False 的 ToolOutcome(调用方回灌给模型);
    基础设施故障抛 ToolInfrastructureError(调用方推 error 帧终止流)。
    """
    name = tool_call.get("name", "")
    tool_call_id = tool_call.get("id", "")

    tool = registry.get(name)
    if tool is None:
        message = (
            f"工具 {name} 不存在。可用工具:{', '.join(sorted(registry))}"
        )
        return ToolOutcome(tool_call_id, name, False, message, _summarize(message))

    attempts = 1 + (
        settings.tool_retry_attempts if name in RETRYABLE_TOOLS else 0
    )
    last_message = ""

    for attempt in range(attempts):
        try:
            message = await asyncio.wait_for(
                tool.ainvoke(tool_call), timeout=settings.tool_timeout_seconds
            )
            return ToolOutcome(
                tool_call_id, name, True, message.content, _summarize(message.content)
            )
        except TimeoutError:
            last_message = (
                f"工具 {name} 执行超时(超过 {settings.tool_timeout_seconds} 秒)"
            )
            logger.warning("工具 %s 超时,第 %d/%d 次尝试", name, attempt + 1, attempts)
            if attempt + 1 < attempts:
                await asyncio.sleep(settings.tool_retry_delay_seconds)
        except ValidationError as exc:
            # 参数不合 schema。重试无意义 —— 同一个工具调用重放一次还是同样的参数。
            last_message = f"工具 {name} 的参数不合法:{exc}"
            logger.warning("工具 %s 参数校验失败:%s", name, exc)
            break
        except ToolNotFound as exc:
            # 业务性未找到是决定性结果:重放同一个 tool_call 送的是同样的参数,
            # 只会同样落空。而且这是本章最常见的落空路径(FAQ 查不到),重试
            # 白搭一次 DB 往返加 tool_retry_delay_seconds 的等待 —— 可恢复路径
            # 本该是最便宜的那条。
            last_message = str(exc)
            break
        except SQLAlchemyError as exc:
            logger.exception("工具 %s 命中数据库故障", name)
            raise ToolInfrastructureError("数据服务暂时不可用") from exc
        except Exception as exc:
            logger.exception("工具 %s 抛出未预期异常", name)
            raise ToolInfrastructureError("工具执行失败") from exc

    return ToolOutcome(tool_call_id, name, False, last_message, _summarize(last_message))
