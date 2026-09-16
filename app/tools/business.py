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
from datetime import datetime, timedelta

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


#: 回显给模型的入参最多截这么长 —— 模型给的输入不受我们控制,原样回灌
#: 等于让它自己决定往上下文里塞多少 token。
_ECHO_LIMIT = 32


def _require_order_no(order_id: str) -> str:
    """订单号须为 4-32 位 ASCII 数字。不符合视为查无此单,而不是编一个结果。

    必须是 `isascii() and isdigit()` 两个条件:单独一个 `isdigit()` 是
    Unicode 感知的,`"١٢٣٤".isdigit()`(阿拉伯-印度数字)与 `"²²²²".isdigit()`
    (上标)都为 True —— 这类输入会**通过**校验并拿到一张凭空编造的订单,
    而不是 ToolNotFound。
    """
    cleaned = order_id.strip()
    if not (cleaned.isascii() and cleaned.isdigit()) or not (4 <= len(cleaned) <= 32):
        raise ToolNotFound(f"未找到订单 {cleaned[:_ECHO_LIMIT]},请核对订单号后重试")
    return cleaned


_ORDER_STATUS = ["待付款", "已付款", "已发货", "已完成", "已取消"]
_PRODUCT_NAMES = ["无线耳机", "运动鞋", "双肩包", "保温杯", "机械键盘"]

#: 订单状态 → 该状态下**可能**出现的物流状态。
#:
#: 这是本模块唯一的「状态耦合」定义:物流状态不是自己抽的,而是从订单状态
#: 派生出的候选里抽。反向的那半同样重要 ——「待付款 / 已付款 / 已取消」不在
#: 表里,没发货的单子就是**没有**物流记录,查物流应当查不到,而不是编一条出来。
_LOGISTICS_BY_STATUS = {
    "已发货": ("已揽件", "运输中", "派送中"),
    "已完成": ("已签收",),
}


def _order_record(order_no: str) -> dict:
    """订单的唯一真相源 —— query_order 与 query_logistics 都必须经它取值。

    两个工具各自 `_rng(不同前缀, 同一订单号)` 是本模块最容易犯的错:那是
    **两条相互独立**的随机流,于是同一个订单可以同时是「已取消」和「已签收」。
    实测 1000 个订单里 807 个状态矛盾、2000 个里 217 个轨迹早于下单时间。
    共用同一条记录之后,这类矛盾在结构上不可能出现。

    **抽取顺序不可改动** —— 改动会改变每个订单号的具体取值。
    """
    r = _rng("order", order_no)
    return {
        "order_id": order_no,
        "status": r.choice(_ORDER_STATUS),
        "product": r.choice(_PRODUCT_NAMES),
        "amount": f"{r.randint(49, 999)}.{r.randint(0, 99):02d}",
        "created_at": (
            f"2026-{r.randint(1, 9):02d}-{r.randint(10, 28):02d} "
            f"{r.randint(9, 21):02d}:{r.randint(0, 59):02d}"
        ),
    }


@tool
async def query_order(order_id: str) -> str:
    """查询订单详情:状态、商品、金额、下单时间。仅当用户给出订单号时使用。"""
    return json.dumps(_order_record(_require_order_no(order_id)), ensure_ascii=False)


_PRODUCT_SPECS = ["标准版", "Pro 版", "家用款", "经典款"]


@tool
async def query_product(keyword: str) -> str:
    """按关键词查询商品信息:名称、价格、库存、规格。用户问商品价格、有没有货时使用。"""
    cleaned = keyword.strip()
    if not cleaned:
        raise ToolNotFound("请提供商品名称或关键词")
    r = _rng("product", cleaned)
    # 只抽一次。抽两次的话 name 里的规格与 spec 字段相互独立,四次里只有一次
    # 对得上 —— 工具会把自相矛盾的数据喂给模型,而本章验收全靠模型如实转述
    # 工具结果,喂矛盾数据等于从源头破坏它。
    spec = r.choice(_PRODUCT_SPECS)
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


_CITIES = ["广州分拨中心", "上海分拨中心", "北京分拨中心", "成都分拨中心"]


@tool
async def query_logistics(order_id: str) -> str:
    """查询订单的物流状态、当前位置与轨迹。用户问"到哪了""发货没"时使用。"""
    order_no = _require_order_no(order_id)
    order = _order_record(order_no)
    candidates = _LOGISTICS_BY_STATUS.get(order["status"])
    if candidates is None:
        # 未发货的单子**没有**物流记录 —— 这是"查无此物",不是上游故障,
        # 所以走 ToolNotFound(可恢复),不是 ToolInfrastructureError。
        raise ToolNotFound(
            f"订单 {order_no} 当前状态是「{order['status']}」,尚未发货、没有物流记录,"
            f"请如实告知用户,不要自行编造物流信息"
        )

    r = _rng("logistics", order_no)
    status = r.choice(candidates)
    city = r.choice(_CITIES)
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
    return json.dumps(
        {
            "order_id": order_no,
            "status": status,
            "location": city,
            "traces": [
                {"time": shipped.strftime(fmt), "desc": f"{city} 已发出"},
                {"time": latest.strftime(fmt), "desc": f"{city} {status}"},
            ],
        },
        ensure_ascii=False,
    )


# ---- 以下两个工具需要数据库会话,故每请求构造 ----
#
# 为什么用闭包工厂而不是 InjectedToolArg:实测发现后者虽然能把参数从
# 发给模型的 schema 里隐藏(tool_call_schema 确实不含它),但调用时该值
# **必须另行注入**,而注入机制在 langchain-core 1.6.3 上没有现成文档,
# 直接调用会抛 ValidationError。闭包让参数**根本不在签名里**,模型既看
# 不见也传不错,且不依赖任何注入机制。

FAQ_LIMIT = 3


def make_query_faq(session, retriever):
    """构造 FAQ 查询工具。会话与检索器绑在闭包里,模型看不到。

    **ch03 换的是内部实现**:关键词查 `faq` 表 → 向量语义检索
    `knowledge_chunks`。对模型的入参出参契约一字未动(spec §5.1):
    入参仍是 `keyword: str`,出参仍是
    `{"keyword", "count", "items": [{"question", "answer", "category"}]}`。

    `session` 参数已不再被本工具使用(原文回查由 retriever 承担),保留是为了
    与 `make_create_ticket` 的工厂形态一致、给后续可能的分页/过滤留位置。
    """

    @tool
    async def query_faq(keyword: str) -> str:
        """查询常见问题库:退货政策、发票、物流规则等。用户问政策或规则类问题时使用。"""
        cleaned = keyword.strip()
        if not cleaned:
            raise ToolNotFound("请提供要查询的关键词")

        # 基础设施故障(向量库/嵌入)必须原样抛上去走 502,**不能**落进下面的
        # ToolNotFound —— 那会把「检索服务挂了」伪装成「这条知识没收录」。
        chunks = await retriever.search(cleaned)

        if not chunks:
            # 全部命中都被相似度阈值滤掉 = 库里的确没有相关内容。回显截断:
            # 这段文本会回灌进模型上下文(可恢复路径),而关键词是模型给的,
            # 长度不受我们控制。理由与 _require_order_no 那处一致。
            raise ToolNotFound(
                f"常见问题库里没有与「{cleaned[:_ECHO_LIMIT]}」相关的内容,"
                f"请如实告知用户暂未收录,不要自行编造答案"
            )
        return json.dumps(
            {
                "keyword": cleaned,
                "count": len(chunks),
                "items": [
                    {"question": c.question, "answer": c.answer, "category": c.category}
                    for c in chunks
                ],
            },
            ensure_ascii=False,
        )

    return query_faq


def make_create_ticket(session, conversation_id: str):
    """构造建工单工具。conversation_id 绑在闭包里,模型看不到。

    非幂等写操作 —— executor 的重试白名单不含它,超时也绝不重试,
    否则会建出两张工单。
    """
    import secrets
    from datetime import datetime

    from sqlalchemy import select

    from app.db.models import Conversation, Ticket

    @tool
    async def create_ticket(description: str, ticket_type: str) -> str:
        """创建人工工单转交人工处理。用户明确要求人工介入、投诉或需人工核实时使用。"""
        cleaned = description.strip()
        if not cleaned:
            raise ToolNotFound("请描述需要人工处理的问题")

        ticket_no = f"T-{datetime.now():%Y%m%d%H%M%S}-{secrets.token_hex(2).upper()}"
        session.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=cleaned,
                # 夹到列宽(String(64))而不是抛错:这是**写**路径,目的是把
                # 用户的问题留下来。超长在 MySQL 严格模式下抛 DataError,
                # T6 归类为不可恢复 → 502 且整单丢失 —— 一个被模型撑爆的
                # 标签字段不该毁掉 description 里真正的问题描述。
                ticket_type=ticket_type.strip()[:64] or "其他",
                status="open",
            )
        )
        # 建单即转人工 —— 否则 conversations.status 是死列。
        conversation = (
            await session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
        ).scalars().one_or_none()
        if conversation is not None:
            conversation.status = "pending_human"
        await session.commit()

        return json.dumps(
            {"ticket_no": ticket_no, "conversation_id": conversation_id, "status": "open"},
            ensure_ascii=False,
        )

    return create_ticket
