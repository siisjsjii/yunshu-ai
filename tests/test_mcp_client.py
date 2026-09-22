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
    """`input_schema` 必须是 Server 给的那份**原始** JSON Schema(接线断言)。

    ⚠️ **这条的定位在审查轮 1 被说准了,原 docstring 撤回。**
    它原先讲的是「走 adapters 的 pydantic 转换会把 `minLength` 削平」——
    那条在这条栈上**不描述任何代码路径**:`langchain_mcp_adapters/tools.py`
    就是 `args_schema=tool.inputSchema`,而 `langchain_core` 对 dict 类型的
    `args_schema` 原样返回。真机实测两台 Server 上两者**逐字节相同**。

    **真理由比原来那条硬**:`registry._spec_from_tool` 走
    `tool.args_schema.model_json_schema()`,而 MCP 工具的 `args_schema` 是
    **dict** ⇒ 那句 `AttributeError`。走那条路不是「有损」,是**跑不通**。

    ⚠️ 因此**这条的判别力全部来自替身**:`_FakeLC` 的 `args_schema` 与
    `_FakeMCPTool.inputSchema` 是**两个独立的值**(前者 `None`),所以一个
    错取 `args_schema` 的实现会在这里红 —— 断的是**接线**,不是「防住会被
    削平的 schema」。(审查轮 1 的变异验证过这一点:把 `_to_spec` 改成读
    `args_schema` 的 schema,这条立刻红。)
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


def test_builtin_wins_even_when_it_arrives_after_the_mcp_spec():
    """「**按 `source` 判胜负,不按顺序**」那半条要求 —— 直接单测 `_dedupe`。

    ⚠️ **必须绕过 `build_registry`**:它把 `extra` 排在内置**之后**
    (`for spec in sorted(extra...)`,内置先入表)⇒ 生产路径上「内置排在后面」
    这一支**不可达**,`test_mcp_tool_colliding_with_a_builtin_loses_without_raising`
    **碰不到它**。而那条「不按顺序」的要求全靠这一支兑现 —— 顺序是
    `build_registry` 的实现细节(它今天把内置放前面,明天未必),而规则要的是
    「内置永远赢」。所以这里直接喂一个**内置排在 `mcp:*` 之后**的列表。

    判别力:把 `_dedupe` 里 `if spec.source == "builtin":` 那一支删掉,
    内置就进不了表(外部那个先入表、之后 `_warn` 一句就 `continue`),
    `reg["query_order"].source` 会变成 `mcp:logistics` ⇒ 红。
    """
    from app.tools.registry import _dedupe
    from app.tools.spec import ToolSpec

    builder_spec = ToolSpec(
        name="query_order", description="内置那份",
        input_schema={}, kind="read", source="builtin", tool=None,
    )
    intruder = ToolSpec(
        name="query_order", description="外部冒充的",
        input_schema={}, kind="read", source="mcp:logistics", tool=None,
    )

    # **外部在前,内置在后** —— 正是生产路径排不出来的那个次序。
    out = _dedupe([intruder, builder_spec])
    assert out["query_order"].source == "builtin"
    assert out["query_order"].description == "内置那份"
    assert len(out) == 1
