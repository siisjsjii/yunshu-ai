"""工具注册表。

因 create_ticket / query_faq 需要每请求构造(见 business.py 的说明),
注册表不是纯模块级常量 —— 每个请求用 build_tools 组装自己的工具集,
再由 registry_for 建名字到工具的映射。
"""

from langchain_core.tools import BaseTool

from app.tools.business import (
    make_create_ticket,
    make_query_faq,
    query_logistics,
    query_order,
    query_product,
)


def build_tools(*, session, conversation_id: str) -> list[BaseTool]:
    """组装本请求可用的五个工具。"""
    return [
        query_order,
        query_product,
        query_logistics,
        make_query_faq(session),
        make_create_ticket(session, conversation_id),
    ]


def registry_for(tools: list[BaseTool]) -> dict[str, BaseTool]:
    """建名字到工具的映射,供 executor 按模型给的名字查找。"""
    return {tool.name: tool for tool in tools}
