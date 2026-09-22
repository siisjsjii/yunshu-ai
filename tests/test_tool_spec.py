"""`ToolSpec` 与统一校验器 —— 本章唯一的纯函数层。

⚠️ 本文件的断言**不许**写成「抛了异常就算过」:校验闸的价值在于
**回灌给模型的文案点名到字段**,一个只会返回空列表的实现必须让这里变红。
"""

import pytest

from app.tools.policy import WRITE_TOOLS, kind_of
from app.tools.spec import READ, WRITE, ToolSpec, validate_args


class _FakeTool:
    """只当占位 —— `validate_args` 不碰 `tool`,不必是 BaseTool。"""


def _spec(schema: dict, *, name: str = "demo", kind: str = READ) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="测试用",
        input_schema=schema,
        kind=kind,
        source="builtin",
        tool=_FakeTool(),
    )


SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "quantity": {"type": "integer", "minimum": 1},
        "channel": {"type": "string", "enum": ["web", "app"]},
    },
    "required": ["order_id"],
}


def test_valid_args_pass():
    assert validate_args(_spec(SCHEMA), {"order_id": "1002"}) == []


def test_missing_required_names_the_field():
    """断言的是**字段名出现在文案里**,不是「列表非空」。"""
    problems = validate_args(_spec(SCHEMA), {})
    assert problems
    assert any("order_id" in p for p in problems)


def test_wrong_type_is_reported():
    """⚠️ 初稿只写了 `assert problems` —— **对 `type` 分支零判别力**
    (实现者的常量探针证实:把 `_readable` 换成常量它照样绿)。
    与本文件头一条 docstring 的说法自相矛盾,已订正为字段级断言。
    """
    problems = validate_args(_spec(SCHEMA), {"order_id": 1002})
    assert problems
    assert any("order_id" in p for p in problems)


def test_below_minimum_is_reported():
    """**这条是判别力最强的一条**:只看必填与类型的实现会在这里变红。

    spec §3.3 明确要求 MCP 的原始 `inputSchema` **不经过 pydantic 转换** ——
    正是因为转换会把 `minimum` / `enum` 这类约束丢掉,闸就形同虚设。
    """
    problems = validate_args(_spec(SCHEMA), {"order_id": "1002", "quantity": 0})
    assert problems
    assert any("quantity" in p for p in problems)


def test_value_outside_enum_is_reported():
    problems = validate_args(
        _spec(SCHEMA), {"order_id": "1002", "channel": "fax"}
    )
    assert problems
    assert any("channel" in p for p in problems)


def test_problems_are_human_readable_not_pydantic_dumps():
    """文案是**给模型看**的:不许出现 pydantic 的堆栈式原文。"""
    problems = validate_args(_spec(SCHEMA), {})
    joined = " ".join(problems)
    for noise in ("Traceback", "pydantic", "validation error", "1 validation"):
        assert noise not in joined


def test_empty_schema_accepts_anything():
    """空 schema 是合法 JSON Schema(等价于「什么参数都行」),不是「校验失败」。"""
    assert validate_args(_spec({}), {"whatever": 1}) == []


def test_unexpected_extra_field_is_allowed():
    """**刻意不禁止**多余字段。

    JSON Schema 的默认语义就是「额外的键不校验」;加了
    `additionalProperties: false` 的话,模型多传一个它自己编的字段
    就会被拦下 —— 而那是个**无害**的行为,拦它只会白白浪费一轮对话。
    """
    assert validate_args(_spec(SCHEMA), {"order_id": "1", "extra": "x"}) == []


# ---- 权限策略 ----------------------------------------------------------


def test_declared_write_tool_is_write():
    assert kind_of("create_ticket") == WRITE


@pytest.mark.parametrize(
    "name", ["query_order", "query_product", "query_logistics", "query_faq"]
)
def test_declared_read_tools_are_read(name):
    assert kind_of(name) == READ


def test_undeclared_tool_defaults_to_read():
    """spec §5.2(**用户 2026-09-22 拍板**):未知 MCP 工具默认**只读**。

    这是验收 3 成立的前提 —— 在 Server 侧加工具、只重启该 Server,
    客服系统这边代码不动、服务不重启就能用上。默认拒绝的话,
    新工具还要回来加一行声明,直接与需求 1 / 验收 3 冲突。

    同时它也是安全的:**写只认我们本地的表**,外部 Server 无法靠改自己的
    用途声明拿到写权限。
    """
    assert kind_of("some_tool_we_never_declared") == READ
    assert kind_of("mcp__logistics__anything") == READ


def test_write_tools_is_exactly_create_ticket():
    """这份集合是**写操作的全集**,不是「已知写操作的一部分」。

    断言写成精确相等而不是 `in`:将来往表里加东西必须**显式改这条测试**,
    而 `assert "create_ticket" in WRITE_TOOLS` 对新增的写工具完全无感。
    """
    assert WRITE_TOOLS == frozenset({"create_ticket"})
