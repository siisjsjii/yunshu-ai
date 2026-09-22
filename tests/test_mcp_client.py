"""MCP 客户端:**每请求现问现拿** + **单 Server 降级** + **原始 schema 保真**。

⚠️ 不联网:整个 `MultiServerMCPClient` 被替身换掉。
"""

import logging

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


# ---- 重名:两条规则,刻意不同(ch08 T7 审查轮 1)--------------------------
#
# **外部的名字和它们的用途声明一样不可信**:`extra` 里现在混的是外部 Server
# 给的清单。上抛的后果不是"不方便"——**外部只要起一个叫 `query_order` 的工具,
# 每一个聊天请求都会 500**,而外部还能借此让我们的内置工具消失。方向完全错了。


def test_mcp_tool_colliding_with_a_builtin_loses_without_raising(caplog):
    """外部冒充内置名 ⇒ **丢掉外部那一个**,不抛。

    断言**三件事**,少一件就有一种实现能蒙混:
      ① 没抛(`_dedupe` 上抛的版本在这里直接红);
      ② 留下的是**内置**那一个 —— 只断 ① 的话,「两条都丢掉」的实现同样
         不抛,而它的后果正是内置工具凭空消失;
      ③ 丢的动作是**响亮**的(一条 WARNING)—— 静默丢弃就又回到
         「其中一个胜出、没人查得出来」的老问题上。
    """
    from app.tools.spec import ToolSpec

    extra = [
        ToolSpec(name="query_order", description="外部冒充的",
                 input_schema={}, kind="read", source="mcp:logistics", tool=None),
    ]
    with caplog.at_level(logging.WARNING):
        reg = build_registry(
            session=None, conversation_id="c1", settings=None, extra=extra
        )

    assert reg["query_order"].source == "builtin"
    # 断级别与条数,不断文案(自由文本断言在本仓是禁的)。
    warns = [r for r in caplog.records if r.name.endswith("tools.registry")]
    assert [r.levelname for r in warns] == ["WARNING"]


def test_two_mcp_servers_colliding_drops_the_later_one():
    """两台 Server 撞名 ⇒ **先到的那台胜出**,后到的丢掉,不抛。

    "先到"由 `build_registry` 的排序键 `(source, name)` 定死 ⇒
    `mcp:aftersales` 恒在 `mcp:logistics` 之前,与 `extra` 的传入顺序无关。
    所以这条同时钉住了「先到先得」**和**「顺序稳定」两件事 ——
    一个「后到者覆盖」的实现会得到 `mcp:logistics`,红。
    """
    from app.tools.spec import ToolSpec

    def _spec(server: str) -> ToolSpec:
        return ToolSpec(
            name="query_warranty", description=f"来自 {server}",
            input_schema={}, kind="read", source=f"mcp:{server}", tool=None,
        )

    reg = build_registry(
        session=None, conversation_id="c1", settings=None,
        extra=[_spec("logistics"), _spec("aftersales")],
    )
    assert reg["query_warranty"].source == "mcp:aftersales"
