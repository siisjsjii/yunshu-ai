"""审计**真的落库了** —— 替身那层看不见这个。

读回时用**新 session**:SQLAlchemy 的身份映射持弱引用,同 session 重读
是否打到库取决于还有没有东西引用着那个 ORM 对象 —— 那会变成
「靠 refcount 走运」的断言(本仓记过)。

⚠️ **与 task-5 brief 的偏离(一条,已记账)**:brief 给的两条用例**没有清理**,
于是它们只在**第一次**跑是绿的 —— 每条用例都往 `tool_audit_logs` 里追加一行
探针(T4 起审计就是**只追加**的表),第二次跑时 `.one()` 撞
`MultipleResultsFound`。实测:连跑两轮,第二轮红。
按本仓在 `refund_requests` / `conversation_summaries` 上已有的做法补一个
按会话 id 清理的 `_cleanup_audit_rows()`,断言一字未动。
"""

import pytest
from sqlalchemy import select, text

from app.db.base import get_sessionmaker
from app.db.models import ToolAuditLog
from app.tools.audit import record_audit

pytestmark = pytest.mark.db

SCRATCH_CONV = "ch08-audit-db"


async def _cleanup_audit_rows() -> None:
    """按会话 id 清掉探针行。

    **必须 commit** —— `async with session` 退出时是 rollback,不 commit 的
    DELETE 会原地作废,探针行留库;下一条测试里的 `.one()` 于是撞
    MultipleResultsFound。(本仓在 refund_requests 上踩过同一件事,
    同样的注释也留在 `tests/test_db_models.py` 里。)
    """
    async with get_sessionmaker()() as session:
        await session.execute(
            text("DELETE FROM tool_audit_logs WHERE conversation_id = :c"),
            {"c": SCRATCH_CONV},
        )
        await session.commit()


@pytest.mark.anyio
async def test_row_is_really_persisted():
    await _cleanup_audit_rows()          # 上一轮跑挂了也不污染本轮
    await record_audit(
        conversation_id=SCRATCH_CONV, tool_call_id="call_db_1",
        tool_name="query_order", source="builtin",
        args={"order_id": "1002"}, result_summary="ok", status="success",
        retry_count=0, duration_ms=7,
    )
    async with get_sessionmaker()() as session:      # ← 新 session
        row = (
            await session.execute(
                select(ToolAuditLog)
                .where(ToolAuditLog.conversation_id == SCRATCH_CONV)
                .order_by(ToolAuditLog.id.desc())
                .limit(1)
            )
        ).scalars().one()
    assert row.tool_name == "query_order"
    assert row.status == "success"
    assert row.duration_ms == 7
    assert "1002" in row.args
    await _cleanup_audit_rows()


@pytest.mark.anyio
async def test_retry_count_is_what_was_passed_not_a_default():
    """**判别力所在**:一个把列写死成 0 的实现会在这条变红。

    (执行器那侧另有一条「首次成功 ⇒ 0」,两条合起来才锁住 §7.4。)
    """
    # 没有这一句,下面的 `.one()` 在**第二次**跑时必挂(只追加的表)。
    await _cleanup_audit_rows()
    await record_audit(
        conversation_id=SCRATCH_CONV, tool_call_id="call_db_2",
        tool_name="query_order", source="builtin", args={},
        result_summary="超时", status="timeout", retry_count=2, duration_ms=30000,
    )
    async with get_sessionmaker()() as session:
        row = (
            await session.execute(
                select(ToolAuditLog)
                .where(ToolAuditLog.tool_call_id == "call_db_2")
            )
        ).scalars().one()
    assert row.retry_count == 2
    assert row.duration_ms == 30000
    await _cleanup_audit_rows()
