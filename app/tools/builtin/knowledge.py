"""知识库检索工具 —— 需要**每请求**的检索器,故用闭包工厂。

`session` 与 `retriever` 绑在闭包里,模型既看不见也传不错
(为什么不用 `InjectedToolArg`,见 `builtin/tickets.py` 的同款说明)。

⚠️ 本模块**不能** import `app.tools.registry`:注册表要 import 本包才能发现工具,
反向再引一次就成环(registry → builtin → registry)。检索器由 `registry.py`
造好后**喂进来**(`build(..., retriever=...)`)。
"""

import json

from langchain.tools import tool

from app.tools.errors import ToolNotFound
from app.tools.mock_data import ECHO_LIMIT

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


def build(*, session, conversation_id, retriever):
    return [make_query_faq(session, retriever)]
