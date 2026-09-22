"""订单 / 商品 / 物流三个「假装有上游系统」的只读工具。

数据全部来自 `app.tools.mock_data`(**唯一真相源**,与两个 MCP Server 共用)。

⚠️ `query_logistics` **暂时**还在这里 —— T7 会把它删掉,由物流 MCP Server 接管。
"""

import json

from langchain.tools import tool

from app.tools.errors import ToolNotFound
from app.tools.mock_data import (
    PRODUCT_SPECS,
    logistics_record,
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


@tool
async def query_logistics(order_id: str) -> str:
    """查询订单的物流状态、当前位置与轨迹。用户问"到哪了""发货没"时使用。"""
    return json.dumps(
        logistics_record(require_order_no(order_id)), ensure_ascii=False
    )


def build(*, session, conversation_id, retriever):
    return [query_order, query_product, query_logistics]
