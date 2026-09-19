"""工具注册表。

因 create_ticket / query_faq 需要每请求构造(见 business.py 的说明),
注册表不是纯模块级常量 —— 每个请求用 build_tools 组装自己的工具集,
再由 registry_for 建名字到工具的映射。
"""

from langchain_core.tools import BaseTool

from app.config import get_settings
from app.retrieval.embedder import get_embedder
from app.retrieval.milvus import get_vector_store
from app.retrieval.reranker import get_reranker
from app.retrieval.search import KnowledgeRetriever
from app.tools.business import (
    make_create_ticket,
    make_query_faq,
    query_logistics,
    query_order,
    query_product,
)


def build_retriever(session) -> KnowledgeRetriever:
    """按配置组装检索器。

    构造**不连 Milvus、不加载 BGE-M3**(两者都是懒加载),所以拿在请求
    路径上建它是安全的;真正的连接/加载发生在第一次 `search`。
    """
    settings = get_settings()
    return KnowledgeRetriever(
        session,
        get_vector_store(settings.milvus_uri, settings.milvus_collection),
        get_embedder(
            settings.embedding_model_path,
            settings.embedding_max_length,
            settings.embedding_batch_size,
        ),
        get_reranker(settings.reranker_model_path),
        top_k=settings.retrieval_top_k,
        score_threshold=settings.retrieval_score_threshold,
    )


def build_tools(*, session, conversation_id: str) -> list[BaseTool]:
    """组装本请求可用的五个工具。"""
    return [
        query_order,
        query_product,
        query_logistics,
        make_query_faq(session, build_retriever(session)),
        make_create_ticket(session, conversation_id),
    ]


def registry_for(tools: list[BaseTool]) -> dict[str, BaseTool]:
    """建名字到工具的映射,供 executor 按模型给的名字查找。"""
    return {tool.name: tool for tool in tools}
