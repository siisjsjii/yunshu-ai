"""审计写入的**形状**:参数怎么排版、超长怎么截、失败怎么吞。

⚠️ **不连库**。真落库那层在 `tests/test_tools_audit_db.py`。
分两层是刻意的:只留替身那层,「替身替被测对象完成了语义」——
端点/执行器删掉真写入照样绿(本仓记过的第 (g) 类假绿)。
"""

import json
import logging

import pytest

from app.tools import audit


class _FakeSession:
    def __init__(self, sink, *, boom=False):
        self.sink = sink
        self.boom = boom

    def add(self, row):
        if self.boom:
            raise RuntimeError("模拟库挂了")
        self.sink.append(row)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def commit(self):
        if self.boom:
            raise RuntimeError("模拟提交失败")


def _patch(monkeypatch, sink, *, boom=False):
    monkeypatch.setattr(
        audit, "get_sessionmaker",
        lambda: (lambda: _FakeSession(sink, boom=boom)),
    )


@pytest.mark.anyio
async def test_writes_one_row_with_the_right_columns(monkeypatch):
    rows: list = []
    _patch(monkeypatch, rows)
    await audit.record_audit(
        conversation_id="c1", tool_call_id="call_1", tool_name="query_order",
        source="builtin", args={"order_id": "1002"}, result_summary="ok",
        status="success", retry_count=0, duration_ms=12,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.conversation_id == "c1"
    assert row.tool_name == "query_order"
    assert row.source == "builtin"
    assert row.status == "success"
    assert row.retry_count == 0
    assert row.duration_ms == 12


@pytest.mark.anyio
async def test_args_are_stored_as_unescaped_json(monkeypatch):
    """中文不转义 —— 验收 5/6 是要**人眼**读这些行的。"""
    rows: list = []
    _patch(monkeypatch, rows)
    await audit.record_audit(
        conversation_id="c1", tool_call_id="c1", tool_name="create_ticket",
        source="builtin", args={"description": "耳机坏了"}, result_summary="",
        status="permission_denied",
    )
    assert "耳机坏了" in rows[0].args
    assert "\\u" not in rows[0].args
    assert json.loads(rows[0].args) == {"description": "耳机坏了"}


@pytest.mark.anyio
async def test_unserializable_args_do_not_raise(monkeypatch):
    """模型给的东西不受我们控制,`json.dumps` 失败**不能**反过来拦工具执行。"""
    rows: list = []
    _patch(monkeypatch, rows)
    await audit.record_audit(
        conversation_id="c1", tool_call_id="c1", tool_name="t", source="builtin",
        args={"weird": object()}, result_summary="", status="success",
    )
    assert len(rows) == 1


@pytest.mark.anyio
async def test_overlong_fields_are_truncated_not_raised(monkeypatch):
    """列宽是 `String(500)` —— MySQL 严格模式下超长会 `DataError`。

    `create_ticket` 当年就栽过同一件事(它把 `ticket_type` 夹到 64 而不是抛错):
    一个被模型撑爆的摘要字段不该让**整条审计行**丢掉。
    """
    rows: list = []
    _patch(monkeypatch, rows)
    await audit.record_audit(
        conversation_id="c1", tool_call_id="c1", tool_name="t", source="builtin",
        args={}, result_summary="啊" * 2000, status="failed",
        error_detail="唉" * 2000,
    )
    assert len(rows[0].result_summary) <= 500
    assert len(rows[0].error_detail) <= 500


@pytest.mark.anyio
async def test_overlong_status_is_also_truncated(monkeypatch):
    """`status` 与上面三列的**不同**在于它是本模块自己的常量,不是模型的自由文本。

    所以这是一条**防回归的哨兵**,而不是「现在有一条路径是坏的」的证据 ——
    执行器里 `record_audit` 的 4 个调用点能传进来的状态值共 5 个
    (`success` / `failed` / `timeout` / `invalid_args` / `permission_denied`),
    最长 17 字符(`permission_denied`),离列宽 32 还有余量,
    **这条用例今天在真实链路上触发不到**。
    (`confirmation_required` 21 字符**不会到达 `record_audit`** —— 待确认路径在
    写审计之前就返回了,别把它算进来。)

    仍然要守,因为漏夹的后果与上面三列一样重:`DataError` → 被 `except` 吞掉
    → **整行审计静默消失**,而那时它记的是一次**不可逆的写操作**。
    判别力是实测的:`status` 改回不夹,本用例红(43 字符 > 32)。
    """
    rows: list = []
    _patch(monkeypatch, rows)
    await audit.record_audit(
        conversation_id="c1", tool_call_id="c1", tool_name="t", source="builtin",
        args={}, result_summary="", status="x" * 43,
    )
    assert len(rows[0].status) <= 32


@pytest.mark.anyio
async def test_write_failure_is_swallowed(monkeypatch, caplog):
    """**写审计失败不许反过来拦工具执行**(要求 5 明写)。

    所以这里是 `except` + 日志,**不是** raise。用 `raise` 的实现会让
    「库抖了一下」变成「工具调用失败」,方向正好反了。
    """
    _patch(monkeypatch, [], boom=True)
    await audit.record_audit(
        conversation_id="c1", tool_call_id="c1", tool_name="t", source="builtin",
        args={}, result_summary="", status="success",
    )   # 不抛即通过


@pytest.mark.anyio
async def test_commit_failure_is_also_swallowed(monkeypatch, caplog):
    """⚠️ **补的一条**(上面那条走的是 `session.add` 抛,走不到 `commit`)。

    「永不抛」这个保证必须在**两条**失败路径上各自成立:`add` 之前就炸,
    与 `add` 成功、`commit` 才炸(连接断 / 约束冲突 / 库挂了),
    是**两段不同的代码**。只测前者的话,把 `await session.commit()` 挪到
    `try` 外面(或 `except` 之后)照样绿 —— 而那正是最可能真实发生的一种失败。

    `assert len(added) == 1` 是**非空真**的证明:它说明这条路真的走到了 commit,
    而不是在 `add` 那一步就提前炸掉(那样这条会退化成上面那条的复读)。
    """
    added: list = []

    class _CommitBoom(_FakeSession):
        async def commit(self):
            raise RuntimeError("模拟提交失败")

    monkeypatch.setattr(
        audit, "get_sessionmaker", lambda: (lambda: _CommitBoom(added))
    )
    with caplog.at_level(logging.ERROR, logger="app.tools.audit"):
        await audit.record_audit(
            conversation_id="c1", tool_call_id="c1", tool_name="t", source="builtin",
            args={}, result_summary="", status="success",
        )   # 不抛即通过
    assert len(added) == 1
    # 「吞掉」不等于「咽下去不说」:运维要能看见。断的是**真的进了日志**
    # (不是 `pass`),且带上了定位所需的那两个字段 —— 一条无声的吞掉
    # 会让审计在某天开始静默丢失而没有任何人知道。
    assert any(
        "tool=t" in r.getMessage() and r.name == "app.tools.audit"
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
