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


# ---- 以下两个工具需要数据库会话,故每请求构造 ----
#
# 为什么用闭包工厂而不是 InjectedToolArg:实测发现后者虽然能把参数从
# 发给模型的 schema 里隐藏(tool_call_schema 确实不含它),但调用时该值
# **必须另行注入**,而注入机制在 langchain-core 1.6.3 上没有现成文档,
# 直接调用会抛 ValidationError。闭包让参数**根本不在签名里**,模型既看
# 不见也传不错,且不依赖任何注入机制。

FAQ_LIMIT = 3


def make_query_faq(session):
    """构造 FAQ 查询工具。会话绑在闭包里,模型看不到。"""

    @tool
    async def query_faq(keyword: str) -> str:
        """查询常见问题库:退货政策、发票、物流规则等。用户问政策或规则类问题时使用。"""
        from sqlalchemy import or_, select

        from app.db.models import Faq

        cleaned = keyword.strip()
        if not cleaned:
            raise ToolNotFound("请提供要查询的关键词")

        # 关键词里的 % 与 _ 必须按字面匹配,否则 LIKE 会把它们当通配符:
        # "%" 能命中表里任意一行,于是这条查询**永远查得到**,返回 ok=true
        # 加三条与用户问题无关的答案,模型会照着它们自信作答 —— 漏召回这条
        # 防线(见下面的 ToolNotFound)就被从另一头绕过了。关键词由模型从
        # 用户原话里摘("100% 纯棉"这类),% 与 _ 会原样传进来。
        #
        # 反斜杠必须**第一个**替换:放后面会把它自己刚加进去的转义符再翻一倍。
        # 不用 contains(autoescape=True):它在 MySQL 上的渲染没实测过,而显式
        # 写法的语义毫无歧义。
        escaped = (
            cleaned.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        pattern = f"%{escaped}%"
        rows = (
            (
                await session.execute(
                    select(Faq)
                    .where(
                        or_(
                            Faq.question.like(pattern, escape="\\"),
                            Faq.answer.like(pattern, escape="\\"),
                        )
                    )
                    .limit(FAQ_LIMIT)
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            # 回显同样截断:这段文本会回灌进模型上下文(可恢复路径),而关键词
            # 是模型给的。理由与 _require_order_no 那处一致。
            raise ToolNotFound(
                f"常见问题库里没有与「{cleaned[:_ECHO_LIMIT]}」相关的内容,"
                f"请如实告知用户暂未收录,不要自行编造答案"
            )
        return json.dumps(
            {
                "keyword": cleaned,
                "count": len(rows),
                "items": [
                    {"question": r.question, "answer": r.answer, "category": r.category}
                    for r in rows
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
