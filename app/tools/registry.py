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


def build_retriever(session, settings) -> KnowledgeRetriever:
    """按配置组装检索器。

    构造**不连 Milvus、不加载 BGE-M3**(两者都是懒加载),所以拿在请求
    路径上建它是安全的;真正的连接/加载发生在第一次 `search`。

    ⚠️ `settings` 是**必传参数,这里不再读模块级 `get_settings()`**
    (ch08 T7 收口)。原先它写死读全局,而 `build_registry` 收了一个
    `settings` 参数却**从不往下传** —— 于是「调用方传了一份、实际生效的是
    另一份」,`app/api/chat.py` 那行 `settings=settings` 看着像接上了、
    其实是个没有读者的装饰。本项目对「看起来接上、其实没接」的形状有明令。
    唯一还读全局的地方收在 `build_registry` 的 `settings is None` 分支
    (那是「调用方明确表示不关心配置」的入口)。
    """
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


def build_registry(
    *, session, conversation_id, settings=None, extra=None
) -> dict[str, ToolSpec]:
    """组装本请求的注册表:`name → ToolSpec`。

    `extra` 是 **MCP 那条路**拿回来的规格(由 `app/mcp/client.py` 的
    `discover_mcp_specs` 产出,那是异步的,**由调用方 await 之后喂进来**)。
    注册表本身是**纯组装**,不碰网络 —— 这样它保持可同步单测。

    `retriever` 在这里造好再喂进 `discover` —— 让 `builtin/knowledge.py`
    自己 import `registry` 会成环(registry → builtin → registry)。

    ⚠️ **不加 `lru_cache` 或任何装饰器**:本函数每请求组装(`query_faq` /
    `create_ticket` 是每请求闭包),缓存会让上一个会话的工具凭据被下一个
    会话用上。

    `settings is None` 是「调用方明确表示不关心配置」(纯组装类单测),
    只有在那个分支上才回落到模块级 `get_settings()` —— 生产的两条调用
    (`app/api/chat.py` 的两个端点)一律显式传,不留「传一份、读另一份」的缝。
    """
    settings = settings or get_settings()
    retriever = build_retriever(session, settings)
    specs = [
        _spec_from_tool(tool, source="builtin")
        for tool in builtin.discover(
            session=session, conversation_id=conversation_id, retriever=retriever
        )
    ]
    # 顺序稳定 = 工具定义块逐字节相同 = 前缀缓存命中(spec §3.4):
    # 内置在前,MCP 按 (server, name) 排。
    for spec in sorted(extra or [], key=lambda s: (s.source, s.name)):
        specs.append(spec)
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
