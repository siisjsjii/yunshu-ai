"""MCP 客户端:**每请求现问现拿**两个业务 Server 的工具,单个挂了就降级。

**为什么不缓存**(spec §8.4):本地 `list_tools` 是毫秒级,而缓存会引入
「我刚加的工具为什么没生效」这类**只能靠猜**的故障。验收 3 要的正是现问现拿。

**三个会静默变坏的细节**(spec §2.3,逐字核对过 adapters 0.3.2 的签名):

1. `convert_mcp_tool_to_langchain_tool` 传 **`connection=` 而不是 `session=`**
   —— 传 session 的话,那个 session 一关,造出来的工具就废了。
2. `handle_tool_errors` **必须显式关**。默认 `True` 会把 MCP 的调用故障
   **包成一条正常的工具返回**,于是在执行器眼里「物流服务连不上」是**成功** ——
   直接违反本仓那条「基础设施故障绝不伪装成查不到」。
3. 注册表里的 `input_schema` 用**原始的 `inputSchema`**,不走 adapters 的
   pydantic 转换 —— 转换会削平 `minimum` / `enum` 这类约束。
"""

import logging
from datetime import timedelta

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool

from app.tools.policy import kind_of
from app.tools.spec import ToolSpec

logger = logging.getLogger(__name__)


def _connections(settings) -> dict:
    timeout = timedelta(seconds=settings.mcp_discovery_timeout_seconds)
    return {
        "logistics": {
            "transport": "streamable_http",
            "url": settings.mcp_logistics_url,
            "timeout": timeout,
        },
        "aftersales": {
            "transport": "streamable_http",
            "url": settings.mcp_aftersales_url,
            "timeout": timeout,
        },
    }


def _to_spec(*, server_name: str, tool: BaseTool, schema: dict) -> ToolSpec:
    return ToolSpec(
        name=tool.name,
        description=(tool.description or "").strip(),
        input_schema=schema,
        # 未声明的 MCP 工具一律**只读**(app/tools/policy.py,spec §5.2)——
        # 写权限只认我们本地的表,外部 Server 改自己的用途声明拿不到。
        kind=kind_of(tool.name),
        source=f"mcp:{server_name}",
        tool=tool,
    )


async def discover_mcp_specs(*, settings) -> list[ToolSpec]:
    """连上两个业务 Server,**现问现拿**它们的工具清单。

    单个 Server 连不上 ⇒ **跳过它、打一条响亮的 warn、其余照常**(spec §8.5)。
    两个都挂 ⇒ 返回空列表,聊天仍可用(只剩内置工具)。

    **为什么不选「上抛 502」**:工具清单是**能力**,不是**结果**。一个可选插件
    挂掉不该让整个客服不可用;而且缺失是**可见的** —— 模型看不到那个工具,
    会在回复里如实说没有,不会把「服务挂了」伪装成「你查的东西不存在」。
    """
    connections = _connections(settings)
    client = MultiServerMCPClient(connections)
    specs: list[ToolSpec] = []
    for name in sorted(connections):
        try:
            async with client.session(name) as session:
                listed = await session.list_tools()
                for mcp_tool in listed.tools:
                    lc_tool = convert_mcp_tool_to_langchain_tool(
                        None,
                        mcp_tool,
                        connection=connections[name],
                        server_name=name,
                        handle_tool_errors=False,   # 见模块 docstring 第 2 条
                    )
                    specs.append(
                        _to_spec(
                            server_name=name,
                            tool=lc_tool,
                            schema=mcp_tool.inputSchema,
                        )
                    )
        except Exception:                                # noqa: BLE001
            logger.warning(
                "mcp discovery failed server=%s,已跳过该 Server(其余照常)", name,
                exc_info=True,
            )
            continue
    return specs
