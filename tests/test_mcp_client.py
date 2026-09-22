"""MCP 客户端:**每请求现问现拿** + **单 Server 降级** + **原始 schema 保真**。

⚠️ 不联网:整个 `MultiServerMCPClient` 被替身换掉。
"""

import pytest

from app.mcp import client as mcp_client
from app.tools.registry import build_registry


class _FakeMCPTool:
    def __init__(self, name, schema):
        self.name = name
        self.description = f"{name} 的用途"
        self.inputSchema = schema


class _FakeListed:
    def __init__(self, tools):
        self.tools = tools


class _FakeSession:
    def __init__(self, tools):
        self._tools = tools

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list_tools(self):
        return _FakeListed(self._tools)


class _FakeClient:
    """`connections` → 每台服务器给什么工具。`raises` 里的服务器连接必失败。"""

    def __init__(self, connections, per_server, raises=()):
        self.connections = connections
        self._per_server = per_server
        self._raises = set(raises)

    def session(self, name, **_kw):
        if name in self._raises:
            raise ConnectionError(f"{name} 连不上")
        return _FakeSession(self._per_server[name])


@pytest.fixture
def patch_client(monkeypatch):
    def _install(per_server, raises=()):
        monkeypatch.setattr(
            mcp_client, "MultiServerMCPClient",
            lambda connections: _FakeClient(connections, per_server, raises),
        )
        monkeypatch.setattr(
            mcp_client, "convert_mcp_tool_to_langchain_tool",
            lambda session, tool, **kw: _FakeLC(tool.name),
        )
    return _install


class _FakeLC:
    def __init__(self, name):
        self.name = name
        self.description = f"{name} 的用途"
        self.args_schema = None


class _Settings:
    mcp_logistics_url = "http://127.0.0.1:8101/mcp"
    mcp_aftersales_url = "http://127.0.0.1:8102/mcp"
    mcp_discovery_timeout_seconds = 5.0


_SCHEMA = {
    "type": "object",
    "properties": {"order_id": {"type": "string", "minLength": 4}},
    "required": ["order_id"],
}


@pytest.mark.anyio
async def test_discovers_tools_from_both_servers(patch_client):
    patch_client(
        {
            "logistics": [_FakeMCPTool("query_logistics", _SCHEMA)],
            "aftersales": [
                _FakeMCPTool("query_warranty", _SCHEMA),
                _FakeMCPTool("query_return_progress", _SCHEMA),
            ],
        }
    )
    specs = await mcp_client.discover_mcp_specs(settings=_Settings())
    assert {s.name for s in specs} == {
        "query_logistics", "query_warranty", "query_return_progress"
    }


@pytest.mark.anyio
async def test_source_records_which_server(patch_client):
    """审计要记「来源是内置还是哪个 MCP Server」(要求 5)—— 就是这里给的。"""
    patch_client(
        {"logistics": [_FakeMCPTool("query_logistics", _SCHEMA)], "aftersales": []}
    )
    specs = await mcp_client.discover_mcp_specs(settings=_Settings())
    assert specs[0].source == "mcp:logistics"


@pytest.mark.anyio
async def test_raw_schema_survives_untouched(patch_client):
    """**本章最容易静默失效的一条**(spec §3.3)。

    走 adapters 的 pydantic 转换会把 `minLength` 这类约束削平,于是
    「统一按 JSON Schema 校验」退化成「只查必填和类型」,闸看起来在工作、
    实际漏掉一半。断言的是**原始约束还在**,不是「有个 schema 键」。
    """
    patch_client(
        {"logistics": [_FakeMCPTool("query_logistics", _SCHEMA)], "aftersales": []}
    )
    specs = await mcp_client.discover_mcp_specs(settings=_Settings())
    assert specs[0].input_schema == _SCHEMA
    assert specs[0].input_schema["properties"]["order_id"]["minLength"] == 4


@pytest.mark.anyio
async def test_one_dead_server_does_not_kill_the_other(patch_client):
    """降级(spec §8.5,用户 2026-09-22 拍板)。

    ⚠️ 断言的是**另一个 Server 的工具还在**,不是「没抛异常」——
    一个把所有 Server 都丢掉的实现同样「没抛异常」。
    """
    patch_client(
        {"logistics": [_FakeMCPTool("query_logistics", _SCHEMA)], "aftersales": []},
        raises=("logistics",),
    )
    specs = await mcp_client.discover_mcp_specs(settings=_Settings())
    assert "query_logistics" not in {s.name for s in specs}
    # aftersales 这边本来就没工具 —— 换一个真有工具的来证
    patch_client(
        {
            "logistics": [_FakeMCPTool("query_logistics", _SCHEMA)],
            "aftersales": [_FakeMCPTool("query_warranty", _SCHEMA)],
        },
        raises=("logistics",),
    )
    specs = await mcp_client.discover_mcp_specs(settings=_Settings())
    assert {s.name for s in specs} == {"query_warranty"}


@pytest.mark.anyio
async def test_both_dead_yields_empty_not_an_exception(patch_client):
    patch_client(
        {"logistics": [], "aftersales": []}, raises=("logistics", "aftersales")
    )
    assert await mcp_client.discover_mcp_specs(settings=_Settings()) == []


def test_registry_merges_builtin_and_mcp():
    """`build_registry` 是**纯组装**:MCP 那半由调用方 await 之后喂进来。

    (所以它保持同步、可同步单测 —— 网络 I/O 全在 `app/mcp/client.py` 里。)
    """
    from app.tools.spec import ToolSpec

    extra = [
        ToolSpec(
            name="query_warranty", description="查在保",
            input_schema=_SCHEMA, kind="read", source="mcp:aftersales", tool=None,
        )
    ]
    reg = build_registry(
        session=None, conversation_id="c1", settings=None, extra=extra
    )
    assert "query_warranty" in reg
    assert reg["query_warranty"].source == "mcp:aftersales"
    assert "query_order" in reg          # 内置还在


def test_mcp_specs_come_after_builtin():
    """顺序稳定 ⇒ 工具定义块逐字节相同 ⇒ 前缀缓存命中(spec §3.4)。"""
    from app.tools.spec import ToolSpec

    extra = [
        ToolSpec(name="zz_mcp", description="d", input_schema={},
                 kind="read", source="mcp:zz", tool=None),
    ]
    reg = build_registry(session=None, conversation_id="c1", settings=None, extra=extra)
    names = list(reg)
    assert names.index("zz_mcp") == len(names) - 1
