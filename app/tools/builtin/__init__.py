"""内置工具包 —— **包内自动发现**。

**新写一个工具 = 在本包内新增一个模块,核心代码零改动**(验收 1)。

每个模块导出同一个接口:

    def build(*, session, conversation_id, retriever) -> list[BaseTool]

静态工具(不需要会话的)忽略后三个参数即可 —— 接口统一比省那几个字值钱:
`discover()` 不必分辨模块「是工厂还是常量」。

⚠️ 内置工具是 `import` 进来的,**新增内置工具需要重启客服服务**。
MCP 工具那条路是**每请求现问现拿**,不需要重启(验收 3)。两句不矛盾,是两条通道。
"""

import importlib
import pkgutil

from langchain_core.tools import BaseTool


def discover(*, session, conversation_id, retriever) -> list[BaseTool]:
    """走遍本包的所有子模块,收集它们导出的工具。

    顺序按 `(模块名, 工具名)` 排序 —— **稳定**是硬要求:工具定义块每轮都要
    逐字节相同,否则前缀缓存整段作废(spec §3.4)。
    """
    found: list[tuple[str, BaseTool]] = []
    for info in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{info.name}")
        for tool in module.build(
            session=session, conversation_id=conversation_id, retriever=retriever
        ):
            found.append((info.name, tool))
    found.sort(key=lambda pair: (pair[0], pair[1].name))
    return [tool for _, tool in found]
