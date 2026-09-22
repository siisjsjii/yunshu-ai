"""注册表测试。用替身工具,不碰 DB。"""

from langchain.tools import tool

from app.tools.registry import build_registry, build_tools, registry_for


@tool
async def fake_a(x: str) -> str:
    """替身 A。"""
    return x


@tool
async def fake_b(x: str) -> str:
    """替身 B。"""
    return x


def test_registry_for_maps_name_to_tool():
    reg = registry_for([fake_a, fake_b])
    assert set(reg) == {"fake_a", "fake_b"}
    assert reg["fake_a"] is fake_a


def test_build_tools_includes_all_builtin_names():
    """内置四个的名字必须齐全 —— 少一个,模型就永远调不到它。

    ⚠️ **不再是五个**:ch08 T7 把 `query_logistics` 从内置下线,由物流 MCP
    Server 接管。`build_tools` 只投影**内置那一半** —— MCP 那半是异步发现出来的
    (`app/mcp/client.py`),不经过这个函数(它的调用方是评估脚本,那里没有
    Server 可连)。所以这里断言**精确相等**而不是「至少包含」。
    """
    tools = build_tools(session=None, conversation_id="s1")
    assert {t.name for t in tools} == {
        "query_order",
        "query_product",
        "query_faq",
        "create_ticket",
    }


def test_build_tools_is_the_registry_projection():
    """`build_tools` 就是注册表的**投影**,两者必须同源。

    绑给模型的那批(投影)与能执行的那批(注册表)各取一次的话,模型会
    「看得到却执行不到」—— 退化成一条 ok=false 的可恢复失败,事件序列
    长得一模一样,只是永远查不出东西。

    ch08 起注册表产出的是 `ToolSpec`、`build_tools` 只是它的投影:
    名字集合必须**逐字相等**(投影漏一个 = 模型调不到;多一个 = 模型调了
    却没有执行体)。
    """
    tools = build_tools(session=None, conversation_id="s1")
    reg = build_registry(session=None, conversation_id="s1", settings=None)
    assert set(reg) == set(registry_for(tools))
