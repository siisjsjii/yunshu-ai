"""工具执行:权限闸 → 参数校验 → 执行(超时/重试)→ 结果格式化 → 审计。

**顺序是有讲究的**,不是随手排的(spec §6):
1. 查注册表       —— 接线 bug,不审计、不上抛,回灌给模型
2. 权限闸         —— 写操作没确认就**不执行**;`pending` 不审计,`denied` 审计
3. JSON Schema 校验 —— 拦下之后**回灌给模型**,不抛异常
4. 执行           —— 超时 / 重试(只读可重试,**写操作结构性不重试**)
5. 结果格式化
6. 审计落库       —— 独立 session,失败只 `logger.error`,**不影响返回值**
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.tools.audit import record_audit
from app.tools.errors import (
    ToolInfrastructureError,
    ToolNotFound,
    TransientToolError,
)
from app.tools.spec import WRITE, validate_args

logger = logging.getLogger(__name__)

#: tool_result 事件里给前端展示的摘要上限。
SUMMARY_MAX_CHARS = 200

# ---- 失败种类 ----------------------------------------------------------
#
# `ok=False` 只说明「没成功」,而「没成功」有六种来源,对调用方的含义完全不同
# —— 尤其是**能不能把这句话说给用户听**:只有 `NOT_FOUND` 是「工具明确说
# 没有这个东西」,其余都是**服务端或接线**的问题,把它们说成「你要的东西
# 不存在」就是拿服务端故障指责用户输入。
#
# 谁在用:`app/agent/refund_nodes.py` 的取数节点据此分三种话说
# (超时 → 不指责用户的「稍后再试」;注册表未命中 → 上抛,绝不产出面向用户的
# 「查无此单」;业务性未找到 → 如实转述)。**Agent 那一轮不看它**:那条路上
# 六种失败都是回灌给模型的原料。
ERROR_NOT_FOUND = "not_found"                  # ToolNotFound:业务性未找到(可恢复)
ERROR_TIMEOUT = "timeout"                      # 超时(重试已用尽)
ERROR_INVALID_ARGS = "invalid_args"            # 参数不合 schema
ERROR_TOOL_MISSING = "tool_missing"            # 注册表里没有 = 接线 bug
ERROR_PERMISSION_DENIED = "permission_denied"  # 写操作被用户取消
ERROR_CONFIRMATION_REQUIRED = "confirmation_required"  # 写操作待确认

# ---- 写调用的三态决议(spec §5.3)----------------------------------------
#
# **布尔不够用**:「没问过」与「问过、用户说不」是两件不同的事。
# 用 `approved=False` 一个值表达两者的话,取消路径会**再拿到一次
# `confirmation_required`**,于是取消永远不会被记成「权限拒绝」。
PENDING = "pending"
APPROVED = "approved"
DENIED = "denied"


@dataclass(frozen=True)
class ToolOutcome:
    tool_call_id: str
    name: str
    ok: bool
    content: str    # 完整内容,回灌给模型
    summary: str    # 截断后的展示用摘要
    #: 成功时 `None`;失败时取上面那六个常量之一。**只加字段不改语义**:
    #: 既有调用方(`app/agent/nodes.py`、`app/api/chat.py`)一个字都不用动。
    error_kind: str | None = None
    #: 写操作待确认时,把**入参**带出去给前端渲染预览卡片。
    preview: dict | None = None
    #: **真实发生过的**重试次数(= 实际尝试数 − 1),**不是配置值**。
    retry_count: int = 0
    #: 这条登记项来自哪里(`"builtin"` / `"mcp:logistics"` …),审计要记。
    source: str = ""


def _summarize(text: str) -> str:
    text = text.strip()
    if len(text) <= SUMMARY_MAX_CHARS:
        return text
    return text[:SUMMARY_MAX_CHARS] + "…"


def render_tool_result(spec, raw) -> str:
    """把工具的返回统一成**一个字符串**。

    - 内置工具返回的已经是手挑过字段的 JSON 字符串 ⇒ 原样透传
      (顺带保证中文不转义 —— 本仓全仓已用 `ensure_ascii=False`)。
    - MCP 工具返回的是 MCP 的内容块列表 ⇒ 压成一行紧凑 JSON,
      **不把整个响应体塞进上下文**。

    "只挑回答用得上的字段"与"内部枚举码翻人话"这两条**不在这里**:
    它们落在**工具自身**(内置那份就只有工具知道自己的枚举怎么翻)。
    """
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(raw)


async def execute_tool(
    *,
    tool_call: dict,
    registry: dict,
    settings,
    conversation_id: str = "",
    write_decision: str = PENDING,
) -> ToolOutcome:
    """执行一次工具调用。

    可恢复的失败返回 `ok=False` 的 `ToolOutcome`(调用方回灌给模型);
    基础设施故障抛 `ToolInfrastructureError`(调用方推 error 帧终止流)。

    `registry` 是 `name → ToolSpec`(不是 `BaseTool`)—— 权限(`spec.kind`)
    与校验(`spec.input_schema`)只有登记项上才有。
    """
    name = tool_call.get("name", "")
    tool_call_id = tool_call.get("id", "")
    args = tool_call.get("args") or {}

    spec = registry.get(name)
    if spec is None:
        message = f"工具 {name} 不存在。可用工具:{', '.join(sorted(registry))}"
        # **不审计**:接线 bug 不是一次调用。混进流水会让验收 5/6 的
        # 「最近这几条」里混进与本次调用无关的行。
        return ToolOutcome(
            tool_call_id, name, False, message, _summarize(message),
            ERROR_TOOL_MISSING,
        )

    # ---- 闸 1:权限(只对写操作)------------------------------------
    if spec.kind == WRITE:
        if write_decision == PENDING:
            # 不执行、**不审计** —— 拦截那一刻没有任何人拒绝任何事。
            message = f"工具 {name} 是写操作,需要用户确认后才能执行。"
            return ToolOutcome(
                tool_call_id, name, False, message, _summarize(message),
                ERROR_CONFIRMATION_REQUIRED, preview=dict(args), source=spec.source,
            )
        if write_decision == DENIED:
            message = f"用户取消了 {name} 的调用,未执行。"
            await record_audit(
                conversation_id=conversation_id, tool_call_id=tool_call_id,
                tool_name=name, source=spec.source, args=args,
                result_summary=_summarize(message),
                status=ERROR_PERMISSION_DENIED, retry_count=0, duration_ms=0,
            )
            return ToolOutcome(
                tool_call_id, name, False, message, _summarize(message),
                ERROR_PERMISSION_DENIED, source=spec.source,
            )

    # ---- 闸 2:参数校验(**在 ainvoke 之前**)-------------------------
    problems = validate_args(spec, args)
    if problems:
        message = f"工具 {name} 的参数校验未通过:" + ";".join(problems)
        await record_audit(
            conversation_id=conversation_id, tool_call_id=tool_call_id,
            tool_name=name, source=spec.source, args=args,
            result_summary=_summarize(message),
            status=ERROR_INVALID_ARGS, error_detail=";".join(problems),
            retry_count=0, duration_ms=0,
        )
        return ToolOutcome(
            tool_call_id, name, False, message, _summarize(message),
            ERROR_INVALID_ARGS, source=spec.source,
        )

    # ---- 执行 ------------------------------------------------------
    # 重试次数由 **`kind` 推导**,不再是一张写死的白名单:
    # 新注册的只读工具自动可重试、写工具自动不可重试。
    # 写操作**永不重试**是结构保证 —— 超时未必没执行,重复执行比失败更糟。
    attempts = 1 + (settings.tool_retry_attempts if spec.kind != WRITE else 0)
    started = time.monotonic()
    last_message = ""
    last_kind: str | None = None
    # **真实发生过的**重试次数,不是配置值(spec §7.4)。写成配置值的话,
    # 一个第一次就成功的查询会被审计成「重试了 2 次」,而没有任何断言会红。
    retries = 0

    for attempt in range(attempts):
        if attempt:
            retries += 1
        try:
            message = await asyncio.wait_for(
                spec.tool.ainvoke(tool_call), timeout=settings.tool_timeout_seconds
            )
            content = render_tool_result(spec, message.content)
            await record_audit(
                conversation_id=conversation_id, tool_call_id=tool_call_id,
                tool_name=name, source=spec.source, args=args,
                result_summary=_summarize(content), status="success",
                retry_count=retries,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            return ToolOutcome(
                tool_call_id, name, True, content, _summarize(content),
                retry_count=retries, source=spec.source,
            )
        except TimeoutError:
            last_kind = ERROR_TIMEOUT
            last_message = (
                f"工具 {name} 执行超时(超过 {settings.tool_timeout_seconds} 秒)"
            )
            logger.warning("工具 %s 超时,第 %d/%d 次尝试", name, attempt + 1, attempts)
            if attempt + 1 < attempts:
                await asyncio.sleep(settings.tool_retry_delay_seconds)
        except TransientToolError as exc:
            # **暂时性**故障(网络抖动)—— 本章新增的那一类可重试故障(spec §6.3)。
            # 重试用尽后仍失败 ⇒ 三次都不成,不叫暂时性了,按基础设施故障上抛。
            logger.warning(
                "工具 %s 暂时性故障,第 %d/%d 次尝试:%s", name, attempt + 1, attempts, exc
            )
            if attempt + 1 < attempts:
                await asyncio.sleep(settings.tool_retry_delay_seconds)
            else:
                raise ToolInfrastructureError("工具暂时不可用") from exc
        except ValidationError as exc:
            # 第二道(第一道是上面的 `validate_args`)。走到这里说明工具的
            # `args_schema` 比它的 `input_schema` 更严 —— 那是我们的 bug。
            last_kind = ERROR_INVALID_ARGS
            last_message = f"工具 {name} 的参数不合法:{exc}"
            logger.warning("工具 %s 参数校验失败:%s", name, exc)
            break
        except ToolNotFound as exc:
            # 业务性未找到是**决定性**结果:重放同一个 tool_call 送的是同样的
            # 参数,只会同样落空。而且这是最常见的落空路径,重试白搭一次 DB
            # 往返加等待 —— 可恢复路径本该是最便宜的那条。
            last_kind = ERROR_NOT_FOUND
            last_message = str(exc)
            break
        except SQLAlchemyError as exc:
            logger.exception("工具 %s 命中数据库故障", name)
            raise ToolInfrastructureError("数据服务暂时不可用") from exc
        except Exception as exc:
            logger.exception("工具 %s 抛出未预期异常", name)
            raise ToolInfrastructureError("工具执行失败") from exc

    status = {
        ERROR_TIMEOUT: "timeout",
        ERROR_NOT_FOUND: "failed",
        ERROR_INVALID_ARGS: "invalid_args",
    }.get(last_kind or "", "failed")
    await record_audit(
        conversation_id=conversation_id, tool_call_id=tool_call_id,
        tool_name=name, source=spec.source, args=args,
        result_summary=_summarize(last_message), status=status,
        retry_count=retries,
        duration_ms=int((time.monotonic() - started) * 1000),
    )
    return ToolOutcome(
        tool_call_id, name, False, last_message, _summarize(last_message),
        last_kind, retry_count=retries, source=spec.source,
    )
