"""订单 / 商品两个「假装有上游系统」的只读工具。

数据全部来自 `app.tools.mock_data`(**唯一真相源**,与两个 MCP Server 共用)。

⚠️ **`query_logistics` 已下线内置**(ch08 T7),由物流 MCP Server
(`mcp_servers/logistics.py`,独立进程)接管。**名字没变** —— 评估集与提示词
里的工具名口径不跟着漂。它**只剩一个提供者**:重名的表现是「其中一个静默
胜出」,而谁胜出取决于排序,没人查得出来(`registry._dedupe` 会抛,
但那条是兜底,不是设计)。

⚠️ `mock_data.logistics_record` **留着** —— 物流 Server 与既有测试还要用它。
"""

import json

from langchain.tools import tool

from app.tools.errors import ToolNotFound
from app.tools.mock_data import (
    PRODUCT_SPECS,
    order_record,
    require_order_no,
    rng,
)


@tool
async def query_order(order_id: str) -> str:
    """查询订单详情:状态、商品、金额、下单时间。仅当用户给出订单号时使用。"""
    return json.dumps(order_record(require_order_no(order_id)), ensure_ascii=False)


@tool
async def query_product(keyword: str) -> str:
    """按关键词查询商品信息:名称、价格、库存、规格。用户问商品价格、有没有货时使用。"""
    cleaned = keyword.strip()
    if not cleaned:
        raise ToolNotFound("请提供商品名称或关键词")
    r = rng("product", cleaned)
    # 只抽一次。抽两次的话 name 里的规格与 spec 字段相互独立,四次里只有一次
    # 对得上 —— 工具会把自相矛盾的数据喂给模型,而本章验收全靠模型如实转述
    # 工具结果,喂矛盾数据等于从源头破坏它。
    spec = r.choice(PRODUCT_SPECS)
    return json.dumps(
        {
            "keyword": cleaned,
            "name": f"{cleaned}({spec})",
            "price": f"{r.randint(29, 1299)}.{r.randint(0, 99):02d}",
            "stock": r.randint(0, 200),
            "spec": spec,
        },
        ensure_ascii=False,
    )


def build(*, session, conversation_id, retriever):
    return [query_order, query_product]
