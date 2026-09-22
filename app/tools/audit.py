"""审计写入 —— 本章**唯一**的写口。

**两条硬约束**(要求 5 明写):

1. **写审计失败不许反过来拦工具执行。** 所以这里一律 `except` + 日志,
   绝不向上抛 —— 用 raise 的话,「库抖了一下」会变成「工具调用失败」,
   方向正好反了。
2. **自己开一个 session。** 不能复用工具那个:工具刚把 session 弄进
   待回滚状态时,拿它写审计会把两件事绑在一起(审计成了工具失败的陪葬)。

**为什么没有第二个调用点**:不变量要放在唯一写口上,不靠每个调用方自觉
(本仓的元教训之一)。执行器在固定的两处调它 —— 校验拦下、执行结束。
"""

import json
import logging

from app.db.base import get_sessionmaker
from app.db.models import ToolAuditLog

logger = logging.getLogger(__name__)

#: **与 `db/ch08.sql` 的列宽逐列对齐**(逐个抄过去的,改 DDL 时这里要跟着改)。
#: 超长必须**截断而不是抛** —— 一个被模型撑爆的摘要字段不该让整条审计行丢掉
#: (`create_ticket` 当年就栽过同一件事,它把 `ticket_type` 夹到 64)。
#: MySQL 严格模式下超长是 `DataError`,而 `record_audit` 又把异常吞掉 ⇒
#: **整行审计静默消失**,方向与「只记不拦」正好相反。
_CALL_ID_MAX = 128          # tool_call_id    VARCHAR(128)
_NAME_MAX = 64              # tool_name       VARCHAR(64)
_SOURCE_MAX = 64            # source          VARCHAR(64)
_SUMMARY_MAX = 500          # result_summary  VARCHAR(500)
_DETAIL_MAX = 500           # error_detail    VARCHAR(500)
_STATUS_MAX = 32            # status          VARCHAR(32)

#: `status` 那一行值得单独说一句:它是全表**唯一**一个「值**不受本模块控制**
#: 却按列宽硬存」的串 —— 其余几列都是模型的自由文本,唯独它来自执行器的状态常量。
#: 今天实测会写进去的取值共 8 个,最长 21 字符(`permission_denied`),离 32
#: 还有余量 ⇒ **这条现在触发不到**,它是防「将来加一个长状态名」的哨兵。
#: 仍要守的理由与上面一致:漏夹 ⇒ `DataError` ⇒ 被 `except` 吞掉 ⇒ 整行消失,
#: 而那时它记的已经是一次**不可逆的写操作**。
#: (`conversation_id` 的 32 不在这里 —— 它是直接切片的,见 `conversation_id[:32]`。)


def _clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _dump_args(args) -> str:
    """入参 → 存进表的 JSON 文本。

    `default=str` 是**必须的**:模型给的东西不受我们控制,
    一个不可序列化的值不该让整条审计行丢掉。
    """
    try:
        return json.dumps(args, ensure_ascii=False, default=str)
    except Exception:                                   # noqa: BLE001
        return repr(args)


async def record_audit(
    *,
    conversation_id: str,
    tool_call_id: str,
    tool_name: str,
    source: str,
    args,
    result_summary: str,
    status: str,
    error_detail: str = "",
    retry_count: int = 0,
    duration_ms: int = 0,
) -> None:
    """落一行。**永不抛。**"""
    try:
        async with get_sessionmaker()() as session:
            session.add(
                ToolAuditLog(
                    conversation_id=conversation_id[:32],
                    tool_call_id=_clip(tool_call_id, _CALL_ID_MAX),
                    tool_name=_clip(tool_name, _NAME_MAX),
                    source=_clip(source, _SOURCE_MAX),
                    args=_dump_args(args),
                    result_summary=_clip(result_summary, _SUMMARY_MAX),
                    status=_clip(status, _STATUS_MAX),
                    error_detail=_clip(error_detail, _DETAIL_MAX),
                    retry_count=retry_count,
                    duration_ms=duration_ms,
                )
            )
            await session.commit()
    except Exception:                                   # noqa: BLE001
        # 宽到 `Exception` 是**刻意的**:这条路径上任何失败都只该留下痕迹,
        # 不该影响工具执行。`BaseException`(CancelledError)不在内 —— 取消
        # 要照常向上传播。
        logger.exception(
            "写审计失败(不影响工具执行):tool=%s status=%s", tool_name, status
        )
