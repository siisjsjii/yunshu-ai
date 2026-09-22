"""内置工具的**自动发现** + 注册表的顺序与去重。

验收 1 是「新写一个简单工具,只做注册动作、不动核心代码,Agent 就能用上」——
本文件是它的单测版:**动态造一个 builtin 模块,断言它自己进了表**。
"""

import pkgutil
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


def test_four_builtin_tools_are_registered():
    """**四个**,不再是五个。

    ch08 T7 把 `query_logistics` 从内置下线(物流 MCP Server 接管,名字不变),
    所以它**不该**出现在这张表里。这里断言的是**精确相等**:多一个少一个
    都要显式改这一行。少一个 → 模型永远调不到;多一个 → 模型调了却没有执行体。

    配套的那条 `test_query_logistics_is_still_builtin_before_t7`(T3 写、
    T7 删)**已按计划删除** —— 它的存在意义就是守「T3 与 T7 之间物流还有
    提供者」那段空窗,窗口关了就作废。
    """
    reg = _registry()
    assert set(reg) == {
        "query_order",
        "query_product",
        "query_faq",
        "create_ticket",
    }
    # 反面:它**只**换了提供者,不是消失了 —— 物流那半在 `app/mcp/client.py`。
    assert "query_logistics" not in reg


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


def test_order_is_stable_and_sorted(monkeypatch):
    """顺序必须**可复现**,与枚举顺序无关 —— 前缀缓存要的正是后者。

    ⚠️ 初稿只断言「两次构建相同」;把 `discover()` 里那句 `found.sort(...)`
    删掉它照样绿(实测),因为 `pkgutil.iter_modules` 在一个进程内本就给稳定
    顺序 —— 那条断言测的是「同进程内两次一样」,而那件事**同义反复**。

    ⚠️ 复审给的第二版是 `assert names == sorted(names)` —— **它在正确实现下
    不可能通过**,实测(**ch08 T7 之前的表,那时 `query_logistics` 还在内置**):
        实际 ['query_faq', 'query_logistics', 'query_order', 'query_product', 'create_ticket']
        sorted() ['create_ticket', 'query_faq', 'query_logistics', 'query_order', 'query_product']
    因为排序键是 `(模块名, 工具名)`(brief 逐字规定),而 `create_ticket` 来自
    `tickets` 模块 —— 全局按工具名的字母序根本不是这条实现的口径。
    (结论与那五个名字无关:T7 拿掉 `query_logistics` 之后 `create_ticket`
     仍然排在末尾,`names == sorted(names)` 照样不成立。)

    所以这里断言**那句 sort 真正提供的东西**:枚举顺序被搅乱时,注册表顺序不变。
    它比「写死一张五元素清单」结实 —— 后者在本章的核心场景(新增/搬迁工具)里
    每次都要跟着改,而验收 1 恰恰是「新增一个文件、别的都不动」。
    """

    def enumerate_in_order(order):
        """把 `pkgutil.iter_modules` 换成按指定次序吐模块的版本。"""
        real = pkgutil.iter_modules

        def fake(path=None, prefix=""):
            infos = list(real(path, prefix))
            return iter([infos[i] for i in order(len(infos))])

        return fake

    baseline = list(_registry())
    assert baseline, "注册表不该是空的 —— 空表会让下面两条断言都恒真"

    # 反向枚举(以及只调换首尾)都必须得到**同一个**顺序。
    for order in (lambda n: range(n - 1, -1, -1), lambda n: [n - 1, *range(0, n - 1)]):
        monkeypatch.setattr(pkgutil, "iter_modules", enumerate_in_order(order))
        assert list(_registry()) == baseline, (
            f"注册表顺序跟着枚举顺序变了 —— `discover()` 里的 `found.sort(...)` "
            f"多半没了(枚举被搅乱后得到 {list(_registry())},基准是 {baseline})"
        )


def test_new_module_is_picked_up_without_touching_core_code(tmp_path, monkeypatch):
    """**验收 1 的单测版。**

    往 `app/tools/builtin/` 里丢一个模块(不碰任何既有文件),再 build 一次,
    新工具必须已经在表里。
    """
    pkg_dir = Path(builtin.__file__).parent
    new_module = pkg_dir / "zz_scratch_probe.py"
    # 精确删掉这一个键,不整体替换 `sys.modules`:替换整张映射的话,测试期间
    # 新导入的模块会被注册进一个 monkeypatch 卸载时就丢掉的字典里。
    monkeypatch.delitem(
        sys.modules, "app.tools.builtin.zz_scratch_probe", raising=False
    )
    # ⚠️ **写文件也必须在 `try` 里**:它一旦落在外面,测试被中止(Ctrl-C /
    # `delitem` 抛错 / 超时)就会把 `zz_scratch_probe.py` 留在包目录里 ——
    # 此后 `discover()` 会把它装进**每一个请求**,模型凭空多出一个
    # `echo_probe` 工具,而 `test_four_builtin_tools_are_registered` 变红;
    # 一次 `git add -A` 还会把它带进提交。
    try:
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
        reg = _registry()
        assert "echo_probe" in reg, "新增 builtin 模块没有被自动发现"
        assert reg["echo_probe"].description.strip()
    finally:
        # `missing_ok=True`:写文件本身就可能失败,清理**不能**在这里再抛一次
        # FileNotFoundError 把真正的失败盖掉。
        new_module.unlink(missing_ok=True)
        sys.modules.pop("app.tools.builtin.zz_scratch_probe", None)


def test_duplicate_tool_name_raises_loudly(monkeypatch):
    """重名**必须响亮地失败**。

    静默的去重会让「其中一个胜出」,而两个实现谁胜出取决于排序 ——
    表现是「工具偶尔返回另一种数据」,没人查得出来。

    ⚠️ 与 task-3-brief 的一字之差:`match` 是 `query_order` 而非 `query_product`。
    `orders.build` 返回 `[query_order, query_product]`(T7 之后),
    `[0]` 仍是 **query_order** —— brief 里那个匹配串在正确实现下**不可能通过**
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
