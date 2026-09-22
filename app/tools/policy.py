"""工具权限的**本地**声明表 —— 能不能调只认这份表。

**刻意不看 MCP Server 的用途声明**:那是对方自己写的,不可信。
**也刻意不让模型临场判断**:模型只负责「调不调」,不负责「能不能调」。

未声明的工具名一律按**只读**放行(spec §5.2,用户 2026-09-22 拍板):

- 它是验收 3 成立的前提 —— 在 Server 侧加工具、只重启该 Server,
  客服系统这边代码不动、服务不重启就能用上;
- 同时它是安全的 —— **写只认这份表**,外部 Server 无法靠改自己的
  用途声明拿到写权限。
"""

from app.tools.spec import READ, WRITE

#: 写操作的全集。**今天只有一条。**
#:
#: ⚠️ 测试 `test_write_tools_is_exactly_create_ticket` 断言的是**精确相等**,
#: 加一条就必须去改那条测试 —— 这是刻意的:**多一个写工具是个需要被看见的决定**。
WRITE_TOOLS = frozenset({"create_ticket"})


def kind_of(tool_name: str) -> str:
    """工具名 → `"read"` / `"write"`。**未声明 = 只读**(见模块 docstring)。"""
    return WRITE if tool_name in WRITE_TOOLS else READ
