"""五个业务工具。

三个「假装有上游系统」的工具(query_order / query_product / query_logistics)
在本模块内用确定性伪随机生成数据 —— 不接真实接口、不建表。同一入参
永远得到同样结果,故验收可以写会失败的断言。

另两个工具需要数据库会话,故用工厂函数**每请求构造**,见 make_query_faq /
make_create_ticket。
"""

import hashlib
import json
import random

from langchain.tools import tool

from app.tools.errors import ToolNotFound


def _rng(*parts: str) -> random.Random:
    """由入参派生稳定种子。

    **绝不能用内置 hash()** —— 它对 str 每进程随机化(PYTHONHASHSEED),
    会让"同一订单号永远返回同样数据"在进程重启后失效,而同进程内的
    测试完全测不出来。sha256 跨进程、跨平台稳定。
    """
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _require_order_no(order_id: str) -> str:
    """订单号须为 4-32 位数字。不符合视为查无此单,而不是编一个结果。"""
    cleaned = order_id.strip()
    if not cleaned.isdigit() or not (4 <= len(cleaned) <= 32):
        raise ToolNotFound(f"未找到订单 {order_id},请核对订单号后重试")
    return cleaned


_ORDER_STATUS = ["待付款", "已付款", "已发货", "已完成", "已取消"]
_PRODUCT_NAMES = ["无线耳机", "运动鞋", "双肩包", "保温杯", "机械键盘"]


@tool
async def query_order(order_id: str) -> str:
    """查询订单详情:状态、商品、金额、下单时间。仅当用户给出订单号时使用。"""
    order_no = _require_order_no(order_id)
    r = _rng("order", order_no)
    return json.dumps(
        {
            "order_id": order_no,
            "status": r.choice(_ORDER_STATUS),
            "product": r.choice(_PRODUCT_NAMES),
            "amount": f"{r.randint(49, 999)}.{r.randint(0, 99):02d}",
            "created_at": (
                f"2026-{r.randint(1, 9):02d}-{r.randint(10, 28):02d} "
                f"{r.randint(9, 21):02d}:{r.randint(0, 59):02d}"
            ),
        },
        ensure_ascii=False,
    )


_PRODUCT_SPECS = ["标准版", "Pro 版", "家用款", "经典款"]


@tool
async def query_product(keyword: str) -> str:
    """按关键词查询商品信息:名称、价格、库存、规格。用户问商品价格、有没有货时使用。"""
    cleaned = keyword.strip()
    if not cleaned:
        raise ToolNotFound("请提供商品名称或关键词")
    r = _rng("product", cleaned)
    return json.dumps(
        {
            "keyword": cleaned,
            "name": f"{cleaned}({r.choice(_PRODUCT_SPECS)})",
            "price": f"{r.randint(29, 1299)}.{r.randint(0, 99):02d}",
            "stock": r.randint(0, 200),
            "spec": r.choice(_PRODUCT_SPECS),
        },
        ensure_ascii=False,
    )


_LOGISTICS_STATUS = ["已揽件", "运输中", "派送中", "已签收"]
_CITIES = ["广州分拨中心", "上海分拨中心", "北京分拨中心", "成都分拨中心"]


@tool
async def query_logistics(order_id: str) -> str:
    """查询订单的物流状态、当前位置与轨迹。用户问"到哪了""发货没"时使用。"""
    order_no = _require_order_no(order_id)
    r = _rng("logistics", order_no)
    status = r.choice(_LOGISTICS_STATUS)
    city = r.choice(_CITIES)
    day = r.randint(1, 15)
    return json.dumps(
        {
            "order_id": order_no,
            "status": status,
            "location": city,
            "traces": [
                {
                    "time": f"2026-09-{day:02d} {r.randint(9, 21):02d}:{r.randint(0, 59):02d}",
                    "desc": f"{city} 已发出",
                },
                {"time": "当前", "desc": f"当前状态:{status}"},
            ],
        },
        ensure_ascii=False,
    )
