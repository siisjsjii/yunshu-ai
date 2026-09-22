"""工具注册表。

因 `query_faq` / `create_ticket` 需要每请求构造(见 `builtin/knowledge.py`
与 `builtin/tickets.py` 的说明),注册表不是纯模块级常量 ——
每个请求 `build_registry` 组装自己那份 `name → ToolSpec`。

**`ToolSpec` 三样齐备**:名 / 用途描述 / **原始 JSON Schema**(spec §3.1)。
"""

from langchain_core.tools import BaseTool

from app.config import get_settings
from app.retrieval.embedder import get_embedder
from app.retrieval.milvus import get_vector_store
from app.retrieval.reranker import get_reranker
from app.retrieval.search import KnowledgeRetriever
from app.tools import builtin
from app.tools.policy import kind_of
from app.tools.spec import ToolSpec


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
        top_k=settings.rerank_top_k,
        score_threshold=settings.retrieval_score_threshold,
    )


def _spec_from_tool(tool: BaseTool, *, source: str) -> ToolSpec:
    """`BaseTool` → 一条登记项。

    `kind` 由 `app/tools/policy.py` 给,**不由工具自己声明** ——
    一处真相源:外部 MCP 工具的用途声明是对方写的、不可信,
    「写」只认我们本地的表。
    """
    schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
    return ToolSpec(
        name=tool.name,
        description=(tool.description or "").strip(),
        input_schema=schema,
        kind=kind_of(tool.name),
        source=source,
        tool=tool,
    )


def _dedupe(specs: list[ToolSpec]) -> dict[str, ToolSpec]:
    """建表,并在**重名时响亮地失败**。

    静默去重会让「其中一个胜出」,而谁胜出取决于排序 ——
    表现是「这个工具偶尔返回另一种数据」,没人查得出来。
    """
    out: dict[str, ToolSpec] = {}
    for spec in specs:
        if spec.name in out:
            raise ValueError(
                f"工具重名:{spec.name} 同时来自 "
                f"{out[spec.name].source} 与 {spec.source}"
            )
        out[spec.name] = spec
    return out


def build_registry(*, session, conversation_id, settings=None) -> dict[str, ToolSpec]:
    """组装本请求的注册表:`name → ToolSpec`。

    `retriever` 在这里造好再喂进 `discover` —— 让 `builtin/knowledge.py`
    自己 import `registry` 会成环(registry → builtin → registry)。
    """
    retriever = build_retriever(session)
    specs = [
        _spec_from_tool(tool, source="builtin")
        for tool in builtin.discover(
            session=session, conversation_id=conversation_id, retriever=retriever
        )
    ]
    return _dedupe(specs)


def build_tools(*, session, conversation_id, settings=None) -> list[BaseTool]:
    """注册表的**投影**:只要绑给模型的那份工具列表。

    给评估脚本(`evals/run_tool_selection_eval.py`)用 —— 它不需要 schema
    也不需要权限,只要能把工具绑到模型上。
    """
    return [
        spec.tool
        for spec in build_registry(
            session=session, conversation_id=conversation_id, settings=settings
        ).values()
    ]


def registry_for(tools: list[BaseTool]) -> dict[str, BaseTool]:
    """建名字到工具的映射。**保留**(既有测试与调用方在用)。"""
    return {tool.name: tool for tool in tools}
