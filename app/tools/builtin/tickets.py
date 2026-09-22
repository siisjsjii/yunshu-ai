"""建工单工具 —— 需要**每请求**的会话与会话号,故用闭包工厂。

**为什么用闭包工厂而不是 `InjectedToolArg`**:实测发现后者虽然能把参数从
发给模型的 schema 里隐藏(tool_call_schema 确实不含它),但调用时该值
**必须另行注入**,而注入机制在 langchain-core 1.6.3 上没有现成文档,
直接调用会抛 ValidationError。闭包让参数**根本不在签名里**,模型既看
不见也传不错,且不依赖任何注入机制。
"""

import json

from langchain.tools import tool

from app.tools.errors import ToolNotFound


def make_create_ticket(session, conversation_id: str):
    """构造建工单工具。conversation_id 绑在闭包里,模型看不到。

    非幂等写操作 —— **本章起这不靠白名单了**:`app/tools/policy.py` 把它声明成写操作,
    执行器由 `kind == "write"` **结构性地**推出「永不重试」——
    新注册的写工具自动继承这条,不用回来改执行器。
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


def build(*, session, conversation_id, retriever):
    return [make_create_ticket(session, conversation_id)]
