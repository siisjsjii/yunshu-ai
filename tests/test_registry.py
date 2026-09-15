"""注册表测试。用替身工具,不碰 DB。"""

from langchain.tools import tool

from app.tools.registry import build_tools, registry_for


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


def test_build_tools_includes_all_five_names():
    """五个工具的名字必须齐全 —— 少一个,模型就永远调不到它。"""
    tools = build_tools(session=None, conversation_id="s1")
    assert {t.name for t in tools} == {
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    }
