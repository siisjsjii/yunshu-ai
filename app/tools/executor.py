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

#: 失败**种类**。`ok=False` 只说明「没成功」,而「没成功」有四种来源,
#: 对调用方的含义完全不同 —— 尤其是**能不能把这句话说给用户听**:
#: 只有 `NOT_FOUND` 是「工具明确说没有这个东西」,其余三种都是**服务端或接线**
#: 的问题,把它们说成「你要的东西不存在」就是拿服务端故障指责用户输入。
#:
#: 谁在用:`app/agent/refund_nodes.py` 的取数节点据此分三种话说
#: (超时 → 不指责用户的「稍后再试」;注册表未命中 → 上抛,绝不产出面向用户的
#: 「查无此单」;业务性未找到 → 如实转述)。**Agent 那一轮不看它**:那条路上
#: 四种失败都是回灌给模型的原料。
ERROR_NOT_FOUND = "not_found"        # ToolNotFound:业务性未找到(可恢复)
ERROR_TIMEOUT = "timeout"            # 超时(重试白名单也可能已用尽)
ERROR_INVALID_ARGS = "invalid_args"  # 参数不合 schema(重放同样的参数只会同样失败)
ERROR_TOOL_MISSING = "tool_missing"  # 注册表里没有这个工具名 = 接线 bug


@dataclass(frozen=True)
class ToolOutcome:
    tool_call_id: str
    name: str
    ok: bool
    content: str    # 完整内容,回灌给模型
    summary: str    # 截断后的展示用摘要
    #: 成功时 `None`;失败时取上面那四个常量之一。**只加字段不改语义**:
    #: 既有调用方(`app/agent/nodes.py`、`app/api/chat.py`)一个字都不用动。
    error_kind: str | None = None


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
        return ToolOutcome(
            tool_call_id, name, False, message, _summarize(message),
            ERROR_TOOL_MISSING,
        )

    attempts = 1 + (
        settings.tool_retry_attempts if name in RETRYABLE_TOOLS else 0
    )
    last_message = ""
    # 重试循环只会以「某一种失败」收尾,退出时用它标记 `error_kind`。
    # 三种失败各有自己的分支,`last_kind` 就在那里赋值 —— **不要按文案反推种类**
    # (文案是给人看的,迟早有人改它)。
    #
    # 初值 `None` 只在「循环既没返回也没进任何 except 就走完了」时留下 ——
    # 那条路径不存在;万一将来被改出来,调用方会落进「非业务性失败」那一支
    # (上抛 502),而不是被误当成某一种具体失败。
    last_kind: str | None = None

    for attempt in range(attempts):
        try:
            message = await asyncio.wait_for(
                tool.ainvoke(tool_call), timeout=settings.tool_timeout_seconds
            )
            return ToolOutcome(
                tool_call_id, name, True, message.content, _summarize(message.content)
            )
        except TimeoutError:
            last_kind = ERROR_TIMEOUT
            last_message = (
                f"工具 {name} 执行超时(超过 {settings.tool_timeout_seconds} 秒)"
            )
            logger.warning("工具 %s 超时,第 %d/%d 次尝试", name, attempt + 1, attempts)
            if attempt + 1 < attempts:
                await asyncio.sleep(settings.tool_retry_delay_seconds)
        except ValidationError as exc:
            # 参数不合 schema。重试无意义 —— 同一个工具调用重放一次还是同样的参数。
            last_kind = ERROR_INVALID_ARGS
            last_message = f"工具 {name} 的参数不合法:{exc}"
            logger.warning("工具 %s 参数校验失败:%s", name, exc)
            break
        except ToolNotFound as exc:
            # 业务性未找到是决定性结果:重放同一个 tool_call 送的是同样的参数,
            # 只会同样落空。而且这是本章最常见的落空路径(FAQ 查不到),重试
            # 白搭一次 DB 往返加 tool_retry_delay_seconds 的等待 —— 可恢复路径
            # 本该是最便宜的那条。
            last_kind = ERROR_NOT_FOUND
            last_message = str(exc)
            break
        except SQLAlchemyError as exc:
            logger.exception("工具 %s 命中数据库故障", name)
            raise ToolInfrastructureError("数据服务暂时不可用") from exc
        except Exception as exc:
            logger.exception("工具 %s 抛出未预期异常", name)
            raise ToolInfrastructureError("工具执行失败") from exc

    return ToolOutcome(
        tool_call_id, name, False, last_message, _summarize(last_message), last_kind
    )
