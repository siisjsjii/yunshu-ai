"""执行器测试。全部用替身工具,不联网、不碰 DB。"""

import asyncio

import pytest
from langchain.tools import tool
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError

from app.config import Settings
from app.tools.errors import ToolInfrastructureError, ToolNotFound
from app.tools.executor import RETRYABLE_TOOLS, execute_tool

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


def test_retry_whitelist_excludes_create_ticket():
    """create_ticket 是写操作,重试会建出两张工单 —— 必须在白名单之外。"""
    assert "create_ticket" not in RETRYABLE_TOOLS
    assert RETRYABLE_TOOLS == {
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
    }


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
        registry={"query_order": query_order},
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
        registry={"query_order": query_order},
        settings=_settings(),
    )
    assert outcome.ok is False
    assert "ValidationError" in outcome.content or "参数" in outcome.content


@pytest.mark.anyio
async def test_timeout_is_recoverable_and_retries_whitelisted_tool():
    calls = {"n": 0}

    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        calls["n"] += 1
        await asyncio.sleep(5)
        return "never"

    outcome = await execute_tool(
        tool_call=_tc("query_order", {"order_id": "1001"}),
        registry={"query_order": query_order},
        settings=_settings(tool_timeout_seconds=0.05, tool_retry_attempts=1,
                           tool_retry_delay_seconds=0.01),
    )
    assert outcome.ok is False
    assert "超时" in outcome.content
    assert calls["n"] == 2      # 首次 + 1 次重试


@pytest.mark.anyio
async def test_non_whitelisted_tool_is_never_retried():
    """create_ticket 超时后必须**恰好调用 1 次**。"""
    calls = {"n": 0}

    @tool
    async def create_ticket(description: str, ticket_type: str) -> str:
        """替身。"""
        calls["n"] += 1
        await asyncio.sleep(5)
        return "never"

    outcome = await execute_tool(
        tool_call=_tc("create_ticket", {"description": "换货", "ticket_type": "换货"}),
        registry={"create_ticket": create_ticket},
        settings=_settings(tool_timeout_seconds=0.05, tool_retry_attempts=1),
    )
    assert outcome.ok is False
    assert calls["n"] == 1


@pytest.mark.anyio
async def test_validation_error_does_not_retry():
    """参数错误重试无意义 —— 单轮下模型也没有第二次改参数的机会。

    计次在 ainvoke 这一层,不在工具体里:langchain 的参数校验由
    pydantic validate_arguments 包在工具体外层,参数不合法时工具体
    根本不会进入 —— 按工具体计数永远是 0,"重试没重试"就区分不出来。
    """
    calls = {"n": 0}

    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        return "ok"

    class _CountingTool:
        """计次壳,只透传 ainvoke。"""

        def __init__(self, inner):
            self._inner = inner

        async def ainvoke(self, tool_call):
            calls["n"] += 1
            return await self._inner.ainvoke(tool_call)

    await execute_tool(
        tool_call=_tc("query_order", {}),
        registry={"query_order": _CountingTool(query_order)},
        settings=_settings(tool_retry_attempts=3),
    )
    assert calls["n"] == 1      # 首次即校验失败,不再重放


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
            registry={"query_order": query_order},
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
            registry={"query_order": query_order},
            settings=_settings(),
        )


@pytest.mark.anyio
async def test_summary_is_truncated_to_200_chars():
    @tool
    async def query_order(order_id: str) -> str:
        """替身。"""
        return "中" * 500

    outcome = await execute_tool(
        tool_call=_tc("query_order", {"order_id": "1001"}),
        registry={"query_order": query_order},
        settings=_settings(),
    )
    assert outcome.ok is True
    assert len(outcome.summary) == 201      # 200 字符 + 省略号
    assert len(outcome.content) == 500      # 回灌给模型的仍是完整内容
