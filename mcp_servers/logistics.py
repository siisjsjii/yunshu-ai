"""物流 MCP Server。

**`query_logistics` 在本章从内置下线,由本 Server 接管**(spec §8.3)。
名字**保留不变** —— 评估集与提示词里的工具名口径不跟着漂。

启动:`.venv/Scripts/python.exe -m mcp_servers.logistics`
"""

import json

from mcp.server import FastMCP

from app.tools.mock_data import logistics_record, require_order_no

#: ⚠️ `FastMCP` 的传输参数是**直接关键字参数**,不是 `FastMCP(..., settings=Settings(...))`
#: —— 1.30.0 的 `__init__` 逐字核对过。同级那个也叫 `Settings` 的 pydantic 模型
#: 有若干**无默认值**的字段,照猜会踩进去。
#:
#: `stateless_http=True` 是**刻意的**:客户端每请求建连接,有状态模式会让
#: session 堆在 Server 侧(`max_sessions` 迟早成为一处没人会想到的故障点)。
#: `json_response=True`:这个 Server 只服务工具调用,不需要 SSE 流式响应。
mcp = FastMCP(
    "物流服务",
    host="127.0.0.1",
    port=8101,
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
async def query_logistics(order_id: str) -> str:
    """查询订单的物流状态、当前位置与轨迹。用户问"到哪了""发货没"时使用。"""
    return json.dumps(
        logistics_record(require_order_no(order_id)), ensure_ascii=False
    )


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
