"""订单 / 商品 / 物流的 **mock 唯一真相源**。

**谁在用**:内置工具(`app/tools/builtin/`)与两个业务 MCP Server
(`mcp_servers/`)。后两者是**独立进程** —— 这正是本模块存在的理由:
数据源不共用的话,同一个订单号在内置 `query_order` 与 MCP `query_logistics`
之间会**说两套话**(订单说「已发货」、物流说「待付款」),演示时一眼穿帮。

**不接真实系统、不建表** —— 全部由入参确定性派生,同一入参永远得到同样结果,
所以验收可以写**会失败的**断言。
"""

import hashlib
import random
from datetime import datetime, timedelta

from app.tools.errors import ToolNotFound


def rng(*parts: str) -> random.Random:
    """由入参派生稳定种子。

    **绝不能用内置 hash()** —— 它对 str 每进程随机化(PYTHONHASHSEED),
    会让"同一订单号永远返回同样数据"在进程重启后失效,而同进程内的
    测试完全测不出来。sha256 跨进程、跨平台稳定。
    """
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


#: 回显给模型的入参最多截这么长 —— 模型给的输入不受我们控制,原样回灌
#: 等于让它自己决定往上下文里塞多少 token。
ECHO_LIMIT = 32


def require_order_no(order_id: str) -> str:
    """订单号须为 4-32 位 ASCII 数字。不符合视为查无此单,而不是编一个结果。

    必须是 `isascii() and isdigit()` 两个条件:单独一个 `isdigit()` 是
    Unicode 感知的,`"١٢٣٤".isdigit()`(阿拉伯-印度数字)与 `"²²²²".isdigit()`
    (上标)都为 True —— 这类输入会**通过**校验并拿到一张凭空编造的订单,
    而不是 ToolNotFound。
    """
    cleaned = order_id.strip()
    if not (cleaned.isascii() and cleaned.isdigit()) or not (4 <= len(cleaned) <= 32):
        raise ToolNotFound(f"未找到订单 {cleaned[:ECHO_LIMIT]},请核对订单号后重试")
    return cleaned


ORDER_STATUS = ["待付款", "已付款", "已发货", "已完成", "已取消"]
PRODUCT_NAMES = ["无线耳机", "运动鞋", "双肩包", "保温杯", "机械键盘"]
PRODUCT_SPECS = ["标准版", "Pro 版", "家用款", "经典款"]
CITIES = ["广州分拨中心", "上海分拨中心", "北京分拨中心", "成都分拨中心"]

#: 订单状态 → 该状态下**可能**出现的物流状态。
#:
#: 这是本模块唯一的「状态耦合」定义:物流状态不是自己抽的,而是从订单状态
#: 派生出的候选里抽。反向的那半同样重要 ——「待付款 / 已付款 / 已取消」不在
#: 表里,没发货的单子就是**没有**物流记录,查物流应当查不到,而不是编一条出来。
LOGISTICS_BY_STATUS = {
    "已发货": ("已揽件", "运输中", "派送中"),
    "已完成": ("已签收",),
}


def order_record(order_no: str) -> dict:
    """订单的唯一真相源 —— `query_order` 与 `query_logistics` 都必须经它取值。

    两个工具各自 `rng(不同前缀, 同一订单号)` 是本模块最容易犯的错:那是
    **两条相互独立**的随机流,于是同一个订单可以同时是「已取消」和「已签收」。
    实测 1000 个订单里 807 个状态矛盾、2000 个里 217 个轨迹早于下单时间。
    共用同一条记录之后,这类矛盾在结构上不可能出现。

    **抽取顺序不可改动** —— 改动会改变每个订单号的具体取值。
    """
    r = rng("order", order_no)
    return {
        "order_id": order_no,
        "status": r.choice(ORDER_STATUS),
        "product": r.choice(PRODUCT_NAMES),
        "amount": f"{r.randint(49, 999)}.{r.randint(0, 99):02d}",
        "created_at": (
            f"2026-{r.randint(1, 9):02d}-{r.randint(10, 28):02d} "
            f"{r.randint(9, 21):02d}:{r.randint(0, 59):02d}"
        ),
    }


def logistics_record(order_no: str) -> dict:
    """物流轨迹。**必须经 `order_record` 取状态**(见它的 docstring)。"""
    order = order_record(order_no)
    candidates = LOGISTICS_BY_STATUS.get(order["status"])
    if candidates is None:
        # 未发货的单子**没有**物流记录 —— 这是"查无此物",不是上游故障,
        # 所以走 ToolNotFound(可恢复),不是 ToolInfrastructureError。
        raise ToolNotFound(
            f"订单 {order_no} 当前状态是「{order['status']}」,尚未发货、没有物流记录,"
            f"请如实告知用户,不要自行编造物流信息"
        )

    r = rng("logistics", order_no)
    status = r.choice(candidates)
    city = r.choice(CITIES)
    # 轨迹时间必须**从下单时间往后推**。另起一条随机流去抽 2026-09-xx 会得到
    # 早于下单的「已发出」时间 —— 那是与状态矛盾同一类的自相矛盾,实测 2000 个
    # 订单里 217 个中招。
    shipped = datetime.strptime(order["created_at"], "%Y-%m-%d %H:%M") + timedelta(
        days=r.randint(1, 3), hours=r.randint(1, 20)
    )
    # 末条轨迹必须带**真实时间戳**并描述当前状态,不能写成 {"time": "当前"}:
    # 那样整条轨迹无法排序,模型读到的是"最后一次扫描停在『已发出』",于是
    # status 为「已签收」时它会当场指出"两者信息不太一致"并追问用户是否收到货
    # —— 验收 4 的真实回复就是这么写的,演示看起来像坏了。
    latest = shipped + timedelta(days=r.randint(1, 4), hours=r.randint(1, 12))
    fmt = "%Y-%m-%d %H:%M"
    return {
        "order_id": order_no,
        "status": status,
        "location": city,
        "traces": [
            {"time": shipped.strftime(fmt), "desc": f"{city} 已发出"},
            {"time": latest.strftime(fmt), "desc": f"{city} {status}"},
        ],
    }
