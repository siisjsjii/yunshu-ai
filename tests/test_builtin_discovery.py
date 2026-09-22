"""内置工具的**自动发现** + 注册表的顺序与去重。

验收 1 是「新写一个简单工具,只做注册动作、不动核心代码,Agent 就能用上」——
本文件是它的单测版:**动态造一个 builtin 模块,断言它自己进了表**。
"""

import sys
import textwrap
from pathlib import Path

import pytest

from app.tools import builtin
from app.tools.policy import WRITE, kind_of
from app.tools.registry import build_registry


class _FakeRetriever:
    async def search(self, keyword):  # pragma: no cover - 本文件用不到
        return []


def _registry(session=None):
    return build_registry(
        session=session, conversation_id="c1", settings=None,
    )


def test_five_builtin_tools_are_registered():
    reg = _registry()
    assert set(reg) == {
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    }


def test_query_logistics_is_still_builtin_before_t7():
    """⚠️ **T7 会把这条测试删掉** —— 那时 `query_logistics` 已搬进物流 MCP Server。

    留它的理由:本任务结束时它必须还在内置里,否则 T3 与 T7 之间
    「物流查询」会有一段**谁都提供不了**的空窗,而验收 2 的题面在 T7 之前
    就已经被人手动跑过。
    """
    assert _registry()["query_logistics"].source == "builtin"


def test_every_spec_carries_the_three_things():
    """名 / 用途描述 / JSON Schema —— 注册中心的要求就是这三样齐备。"""
    for name, spec in _registry().items():
        assert spec.name == name
        assert spec.description.strip()
        assert isinstance(spec.input_schema, dict) and spec.input_schema


def test_input_schema_declares_the_parameters():
    """**判别力所在**:空 schema 的实现会让这条变红。"""
    props = _registry()["query_order"].input_schema["properties"]
    assert "order_id" in props


def test_write_kind_comes_from_the_policy_table():
    """`kind` 由**策略表**给,不由模块自己声明 —— 一处真相源。"""
    reg = _registry()
    assert reg["create_ticket"].kind == WRITE
    assert reg["query_order"].kind != WRITE
    for name, spec in reg.items():
        assert spec.kind == kind_of(name)


def test_order_is_stable_across_two_builds():
    """顺序稳定 = 工具定义块逐字节相同 = 前缀缓存命中(spec §3.4)。"""
    assert list(_registry()) == list(_registry())


def test_new_module_is_picked_up_without_touching_core_code(tmp_path, monkeypatch):
    """**验收 1 的单测版。**

    往 `app/tools/builtin/` 里丢一个模块(不碰任何既有文件),再 build 一次,
    新工具必须已经在表里。
    """
    pkg_dir = Path(builtin.__file__).parent
    new_module = pkg_dir / "zz_scratch_probe.py"
    new_module.write_text(
        textwrap.dedent(
            '''
            """临时探测模块 —— 本测试自己不碰任何核心代码。"""

            from langchain.tools import tool


            @tool
            async def echo_probe(text: str) -> str:
                """把入参原样回显。测试用。"""
                return text


            def build(*, session, conversation_id, retriever):
                return [echo_probe]
            '''
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "modules",
        {k: v for k, v in sys.modules.items() if "zz_scratch_probe" not in k},
    )
    try:
        reg = _registry()
        assert "echo_probe" in reg, "新增 builtin 模块没有被自动发现"
        assert reg["echo_probe"].description.strip()
    finally:
        new_module.unlink()
        sys.modules.pop("app.tools.builtin.zz_scratch_probe", None)


def test_duplicate_tool_name_raises_loudly(monkeypatch):
    """重名**必须响亮地失败**。

    静默的去重会让「其中一个胜出」,而两个实现谁胜出取决于排序 ——
    表现是「工具偶尔返回另一种数据」,没人查得出来。

    ⚠️ 与 task-3-brief 的一字之差:`match` 是 `query_order` 而非 `query_product`。
    `orders.build` 返回 `[query_order, query_product, query_logistics]`,
    `[0]` 就是 **query_order** —— brief 里那个匹配串在正确实现下**不可能通过**
    (它写的时候大概以为首个是商品)。断言的本意(报错**点名**重名工具)不变,
    所以改的是匹配串,不是「去重该不该抛」。
    """
    import app.tools.builtin.orders as orders

    original = orders.build

    def duplicated(*, session, conversation_id, retriever):
        return original(
            session=session, conversation_id=conversation_id, retriever=retriever
        ) + [original(
            session=session, conversation_id=conversation_id, retriever=retriever
        )[0]]

    monkeypatch.setattr(orders, "build", duplicated)
    with pytest.raises(ValueError, match="query_order"):
        _registry()
