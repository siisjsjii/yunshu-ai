"""工具注册表。

因 `query_faq` / `create_ticket` 需要每请求构造(见 `builtin/knowledge.py`
与 `builtin/tickets.py` 的说明),注册表不是纯模块级常量 ——
每个请求 `build_registry` 组装自己那份 `name → ToolSpec`。

**`ToolSpec` 三样齐备**:名 / 用途描述 / **原始 JSON Schema**(spec §3.1)。
"""

import logging

from langchain_core.tools import BaseTool

from app.config import get_settings
from app.retrieval.embedder import get_embedder
from app.retrieval.milvus import get_vector_store
from app.retrieval.reranker import get_reranker
from app.retrieval.search import KnowledgeRetriever
from app.tools import builtin
from app.tools.policy import kind_of
from app.tools.spec import ToolSpec

logger = logging.getLogger(__name__)


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
    """建表,并处理重名 —— **两条规则,刻意不同**。

    - **内置 vs 内置 重名 ⇒ 响亮地抛。** 那是**我们自己的**接线 bug;
      静默去重会让「其中一个胜出」而谁胜出取决于排序 —— 表现是
      「这个工具偶尔返回另一种数据」,没人查得出来。
    - **任何涉及 MCP 的重名 ⇒ 丢掉外部那一个 + 一条响亮的 warn,内置留下。**

    ⚠️ 第二条是本章后期定的:`specs` 里现在**混进了外部来源的清单**,
    而外部的**名字**和它们的**用途声明**一样不可信 ——
    **外部 Server 只要起一个叫 `query_order` 的工具,上抛就会把每一个聊天请求
    打成 500**,而外部还能让我们的内置工具消失,方向完全错了。
    """

    def _warn(keep: ToolSpec, drop: ToolSpec) -> None:
        logger.warning(
            "工具重名,已丢弃外部来源的那一个:name=%s 保留=%s 丢弃=%s",
            keep.name, keep.source, drop.source,
        )

    out: dict[str, ToolSpec] = {}
    for spec in specs:
        existing = out.get(spec.name)
        if existing is None:
            out[spec.name] = spec
            continue
        if existing.source == "builtin" and spec.source == "builtin":
            raise ValueError(f"内置工具重名:{spec.name} —— 我们自己的接线 bug")
        # **按 `source` 判胜负,不按顺序** —— 顺序是 `build_registry` 的实现细节,
        # 而这条规则要的是「内置永远赢」。
        if existing.source == "builtin":
            _warn(existing, spec)
            continue
        if spec.source == "builtin":
            _warn(spec, existing)
            out[spec.name] = spec
            continue
        _warn(existing, spec)          # 两边都是外部的:先到先得
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

    ⚠️ **ch08 T7 起生产路径与评估脚本都不再走它**,本函数现在只剩
    `tests/test_registry.py` 在测(「投影 = 注册表」这条同源不变量)。
    原先的调用方 `evals/run_tool_selection_eval.py` 已改成与
    `app/api/chat.py` 同款的两步(`await discover_mcp_specs` → `build_registry`)
    —— 它需要 **MCP 那一半**,而本函数只投影**内置那一半**:
    `query_logistics` 下线内置之后,继续用它会让评估集里那 3 条物流用例
    **在结构上不可能通过**。

    保留它的理由:它是一条有判别力的不变量的载体(见上);删它属计划外的
    清理,**留给控制者裁定**(现在没有生产调用方了)。
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
