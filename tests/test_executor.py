"""执行器测试。全部用替身工具,不联网、不碰 DB。

⚠️ **ch08 T4 起注册表装的是 `ToolSpec`,不是 `BaseTool`** —— 执行器要用登记项
上的 `kind`(推权限与重试)与 `input_schema`(校验前置)。这里**不给执行器**
加「旧形状也认」的兼容分支:那会让「注册表里装的到底是什么」有两个答案,
而其中一个是错的。本文件的 `_reg()` 就是新形状的唯一造法。
"""

import asyncio

import pytest
from langchain.tools import tool
from sqlalchemy.exc import OperationalError

from app.config import Settings
from app.tools.errors import ToolInfrastructureError, ToolNotFound
from app.tools.executor import (
    ERROR_INVALID_ARGS,
    ERROR_NOT_FOUND,
    ERROR_TIMEOUT,
    ERROR_TOOL_MISSING,
    APPROVED,
    execute_tool,
)
from app.tools.registry import _spec_from_tool

REQUIRED = {
    "openai_base_url": "https://example.invalid/v1",
    "openai_api_key": "sk-test",
    "openai_model": "test-model",
    "database_url": "mysql+asyncmy://u:p@h:3306/db",
}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **REQUIRED, **overrides)


def _tc(name: str, args: dict) -> dict:
    return {"name": name, "args": args, "id": "call_1", "type": "tool_call"}


def _reg(*tools) -> dict:
    """`BaseTool` 列表 → `name → ToolSpec`(注册表的新形状)。

    用 `registry._spec_from_tool` 而不是手搭 `ToolSpec`:**它就是生产那个
    转换函数**。手搭的话,测试里的 `input_schema` 与真注册表给的会各是各的
    (比如漏掉 `required`),于是「校验前置」在单测里过得去、真机上不成立。
    """
    return {t.name: _spec_from_tool(t, source="builtin") for t in tools}


class _CountingTool:
    """计次壳,只透传 ainvoke,用来数"执行器尝试了几次"。

    计次必须在 ainvoke 这一层,不能数工具体调用:langchain 的参数校验由
    pydantic validate_arguments 包在工具体外层(StructuredTool.from_function
    → create_schema_from_function),参数不合法时工具体根本不进入 ——
    按工具体计数恒为 0,重试与否就区分不出来了。

    ⚠️ 它**同时**是校验前置的探针:装到 `ToolSpec.tool` 上之后,`ainvoke`
    只有在两道闸都放行之后才会被调到。
    """

    def __init__(self, inner, counter: dict):
        self._inner = inner
        self._counter = counter

    async def ainvoke(self, tool_call):
        self._counter["n"] += 1
        return await self._inner.ainvoke(tool_call)


def _spec_with_counter(inner, counter: dict):
    """把计次壳塞进登记项的 `tool` 槽位(而不是塞进注册表)。

    `_spec_from_tool` 要的是真 `BaseTool`(`args_schema` 那一套),
    所以先照真工具建 spec,再替换 `tool` 字段 —— `ToolSpec` 是 frozen 的,
    用 `dataclasses.replace`。
    """
    import dataclasses

    return dataclasses.replace(
        _spec_from_tool(inner, source="builtin"), tool=_CountingTool(inner, counter)
    )


@pytest.mark.anyio
async def test_unknown_tool_is_recoverable_not_fatal():
    outcome = await execute_tool(
        tool_call=_tc("nope", {}), registry={}, settings=_settings()
    )
    assert outcome.ok is False
    assert "nope" in outcome.content
    assert "不存在" in outcome.content


@pytest.mark.anyio
async def test_tool_not_found_is_recoverable():
    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        raise ToolNotFound("未找到订单 9999")

    outcome = await execute_tool(
        tool_call=_tc("query_order", {"order_id": "9999"}),
        registry=_reg(query_order),
        settings=_settings(),
    )
    assert outcome.ok is False
    assert "未找到订单" in outcome.content


@pytest.mark.anyio
async def test_validation_error_is_recoverable():
    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        return "ok"

    outcome = await execute_tool(
        tool_call=_tc("query_order", {}),   # 缺必填参数
        registry=_reg(query_order),
        settings=_settings(),
    )
    assert outcome.ok is False
    assert "参数" in outcome.content


@pytest.mark.anyio
async def test_timeout_is_recoverable_and_retries_a_read_tool():
    """只读工具超时可重试 —— ch08 起由 `spec.kind` 推出,不再是白名单。"""
    calls = {"n": 0}

    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        calls["n"] += 1
        await asyncio.sleep(5)
        return "never"

    outcome = await execute_tool(
        tool_call=_tc("query_order", {"order_id": "1001"}),
        registry=_reg(query_order),
        settings=_settings(tool_timeout_seconds=0.05, tool_retry_attempts=1,
                           tool_retry_delay_seconds=0.01),
    )
    assert outcome.ok is False
    assert "超时" in outcome.content
    assert calls["n"] == 2      # 首次 + 1 次重试


@pytest.mark.anyio
async def test_write_tool_is_never_retried_on_timeout():
    """`create_ticket` 超时后必须**恰好调用 1 次**(写操作结构性不重试)。

    `write_decision=APPROVED` 是必须的:默认的 `pending` 会在**权限闸**就返回
    `confirmation_required`,工具一次都不会跑 —— 那样这条用例断的
    「恰好 1 次」会因为「0 次」而红,红的原因却与重试规则无关。
    """
    calls = {"n": 0}

    @tool
    async def create_ticket(description: str, ticket_type: str) -> str:
        """替身。"""
        calls["n"] += 1
        await asyncio.sleep(5)
        return "never"

    outcome = await execute_tool(
        tool_call=_tc("create_ticket", {"description": "换货", "ticket_type": "换货"}),
        registry=_reg(create_ticket),
        settings=_settings(tool_timeout_seconds=0.05, tool_retry_attempts=1),
        write_decision=APPROVED,
    )
    assert outcome.ok is False
    assert calls["n"] == 1


@pytest.mark.anyio
async def test_invalid_args_are_rejected_before_ainvoke_and_not_retried():
    """参数错误重试无意义 —— 单轮下模型也没有第二次改参数的机会。

    ⚠️ **计数在 `ainvoke` 边界上**(CLAUDE.md 的硬约束):写在工具体里的话,
    校验失败时它恒为 0,「校验前置」与「工具跑了才报错」就区分不开 ——
    而这两件事的差别正是本节的全部内容。`n == 0` 是**校验前置**的判别式;
    `retry_count == 0` 是**不重试**的判别式。两条都要断,缺一条就少一件事。
    """
    calls = {"n": 0}

    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        return "ok"

    outcome = await execute_tool(
        tool_call=_tc("query_order", {}),
        registry={"query_order": _spec_with_counter(query_order, calls)},
        settings=_settings(tool_retry_attempts=3),
    )
    assert calls["n"] == 0      # 校验闸就拦下了,`ainvoke` 根本没被走到
    assert outcome.error_kind == ERROR_INVALID_ARGS
    assert outcome.retry_count == 0


@pytest.mark.anyio
async def test_tool_not_found_does_not_retry():
    """业务性未找到是决定性结果 —— 重放同样的参数只会同样落空。

    FAQ 查不到是本章最常见的落空路径,重试白搭一次 DB 往返加
    tool_retry_delay_seconds 的等待,可恢复路径本该是最便宜的那条。
    """
    calls = {"n": 0}

    @tool
    async def query_faq(keyword: str) -> str:
        """替身。"""
        raise ToolNotFound("没有匹配的条目")

    outcome = await execute_tool(
        tool_call=_tc("query_faq", {"keyword": "邮费"}),
        registry={"query_faq": _spec_with_counter(query_faq, calls)},
        settings=_settings(tool_retry_attempts=3, tool_retry_delay_seconds=0.01),
    )
    assert outcome.ok is False
    assert "没有匹配的条目" in outcome.content
    assert calls["n"] == 1      # 确定性落空,不重放


@pytest.mark.anyio
async def test_database_error_is_fatal():
    """DB 故障不能被伪装成"你的订单号查不到"。"""
    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        raise OperationalError("SELECT 1", {}, Exception("连接断开"))

    with pytest.raises(ToolInfrastructureError):
        await execute_tool(
            tool_call=_tc("query_order", {"order_id": "1001"}),
            registry=_reg(query_order),
            settings=_settings(),
        )


@pytest.mark.anyio
async def test_unexpected_exception_is_fatal():
    """未预期异常按 spec §6.7 判不可恢复。"""
    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        raise RuntimeError("没预料到的坏事")

    with pytest.raises(ToolInfrastructureError):
        await execute_tool(
            tool_call=_tc("query_order", {"order_id": "1001"}),
            registry=_reg(query_order),
            settings=_settings(),
        )


@pytest.mark.anyio
async def test_failure_kind_tells_the_caller_whom_to_blame():
    """`ok=False` 只说明「没成功」;不同来源对调用方是**不同的事**。

    取数节点(`app/agent/refund_nodes.py`)按这个字段分三种话说:业务性未找到 →
    如实转述给用户;超时 → 「稍后再试」(**不**指责用户报的号码);
    其余(工具名不在注册表 / 参数不合 schema)→ 上抛,绝不产出面向用户的
    「查无此单」。所以下面几个值必须**分别**钉住 —— 少一个,调用方就会把
    服务端或接线的问题说成「你要的东西不存在」。成功时它必须是 `None`。
    (ch08 新增的两个权限类种类由 `tests/test_executor_gate.py` 钉。)
    """
    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        return "ok"

    @tool
    async def not_found_tool(order_id: str) -> str:
        """替身。"""
        raise ToolNotFound("没有匹配的条目")

    @tool
    async def slow_tool(order_id: str) -> str:
        """替身。"""
        await asyncio.sleep(5)
        return "never"

    settings = _settings()

    ok = await execute_tool(
        tool_call=_tc("query_order", {"order_id": "1001"}),
        registry=_reg(query_order), settings=settings,
    )
    assert (ok.ok, ok.error_kind) == (True, None)

    missing = await execute_tool(
        tool_call=_tc("nope", {}), registry={}, settings=settings
    )
    assert (missing.ok, missing.error_kind) == (False, ERROR_TOOL_MISSING)

    invalid = await execute_tool(
        tool_call=_tc("query_order", {}),          # 缺 order_id
        registry=_reg(query_order), settings=settings,
    )
    assert (invalid.ok, invalid.error_kind) == (False, ERROR_INVALID_ARGS)

    not_found = await execute_tool(
        tool_call=_tc("not_found_tool", {"order_id": "1001"}),
        registry=_reg(not_found_tool), settings=settings,
    )
    assert (not_found.ok, not_found.error_kind) == (False, ERROR_NOT_FOUND)

    timed_out = await execute_tool(
        tool_call=_tc("slow_tool", {"order_id": "1001"}),
        registry=_reg(slow_tool),
        settings=_settings(tool_timeout_seconds=0.05, tool_retry_attempts=0),
    )
    assert (timed_out.ok, timed_out.error_kind) == (False, ERROR_TIMEOUT)


@pytest.mark.anyio
async def test_summary_is_truncated_to_200_chars():
    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        return "中" * 500

    outcome = await execute_tool(
        tool_call=_tc("query_order", {"order_id": "1001"}),
        registry=_reg(query_order),
        settings=_settings(),
    )
    assert outcome.ok is True
    assert len(outcome.summary) == 201      # 200 字符 + 省略号
    assert len(outcome.content) == 500      # 回灌给模型的仍是完整内容
