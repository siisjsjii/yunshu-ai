"""售后 MCP Server:查在保、查退货进度。

⚠️ **已知偏离(spec §8.6)**:本 Server「不接真实系统、不建表」,两个工具
返回的都是**伪随机 mock**。它与 ch06 的 `refund_requests` 表**无语义关联**
—— 用户刚在退款表单里提交的那条,来这里查进度会得到**另一套随机结果**。
演示时**不要**拿它当真实进度用。

要让它接真实数据,得让 Server 连 MySQL,那直接违反上面那条选型 —— 留作
将来单独一章的事。

启动:`.venv/Scripts/python.exe -m mcp_servers.aftersales`
"""

import json

from mcp.server import FastMCP

from app.tools.mock_data import order_record, require_order_no, rng

mcp = FastMCP(
    "售后服务",
    host="127.0.0.1",
    port=8102,
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
)

_WARRANTY_STATES = ["保修中", "已过保", "延保中"]
_RETURN_STAGES = ["已受理", "待寄回", "已寄回", "质检中", "退款中", "已完成"]


@mcp.tool()
async def query_warranty(order_id: str) -> str:
    """查询某订单商品的保修状态与到期时间。用户问"还在保修吗""过保没"时使用。"""
    order_no = require_order_no(order_id)
    # 经 `order_record` 取商品 —— **不要另起一条随机流**去抽商品名,
    # 那正是 ch02 记过的自相矛盾源头(同一个订单在两处显示不同商品)。
    order = order_record(order_no)
    r = rng("warranty", order_no)
    state = r.choice(_WARRANTY_STATES)
    return json.dumps(
        {
            "order_id": order_no,
            "product": order["product"],
            "warranty_state": state,
            "expires_on": f"2027-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}",
        },
        ensure_ascii=False,
    )


@mcp.tool()
async def query_return_progress(order_id: str) -> str:
    """查询退货申请的处理进度。用户问"我的退货到哪一步了""退款什么时候到"时使用。

    ⚠️ 返回的是**伪随机 mock**,与 ch06 的 `refund_requests` 无语义关联(见模块 docstring)。
    """
    order_no = require_order_no(order_id)
    r = rng("return", order_no)
    stage = r.choice(_RETURN_STAGES)
    return json.dumps(
        {
            "order_id": order_no,
            "stage": stage,
            "updated_at": f"2026-{r.randint(1, 9):02d}-{r.randint(10, 28):02d}",
        },
        ensure_ascii=False,
    )


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
