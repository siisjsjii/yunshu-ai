"""五个业务工具。

三个「假装有上游系统」的工具(query_order / query_product / query_logistics)
的数据由 `app/tools/mock_data.py` 用确定性伪随机生成 —— 不接真实接口、不建表。
同一入参永远得到同样结果,故验收可以写会失败的断言。

**数据源为什么在别的模块**:`query_logistics` 后续要搬进**独立进程**的
MCP Server,而订单的唯一真相源必须三处共用 —— 各带一套随机数的话,
同一个订单号会「订单说已发货、物流说待付款」。详见 `app/tools/mock_data.py`。

另两个工具需要数据库会话,故用工厂函数**每请求构造**,见 make_query_faq /
make_create_ticket。
"""

import json
from datetime import datetime, timedelta

from langchain.tools import tool

from app.tools.errors import ToolNotFound
from app.tools.mock_data import (
    CITIES,
    ECHO_LIMIT,
    LOGISTICS_BY_STATUS,
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


@tool
async def query_logistics(order_id: str) -> str:
    """查询订单的物流状态、当前位置与轨迹。用户问"到哪了""发货没"时使用。"""
    order_no = require_order_no(order_id)
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
            # 长度不受我们控制。理由与 mock_data.require_order_no 那处一致。
            raise ToolNotFound(
                f"常见问题库里没有与「{cleaned[:ECHO_LIMIT]}」相关的内容,"
                f"请如实告知用户暂未收录,不要自行编造答案"
            )
        return json.dumps(
            {
                "keyword": cleaned,
                "count": len(chunks),
                "items": [
                    {
                        "question": c.question,
                        "answer": c.answer,
                        "category": c.category,
                        # ch04 增补(引用定位用,spec §13):chunk_id 映射原文、
                        # section_path 展示章节路径。模型侧旧三字段不变。
                        "chunk_id": c.chunk_id,
                        "section_path": c.section_path,
                    }
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
