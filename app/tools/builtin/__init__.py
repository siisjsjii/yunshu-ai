"""内置工具包 —— **包内自动发现**。

**新写一个工具 = 在本包内新增一个模块,核心代码零改动**(验收 1)。

每个模块导出同一个接口:

    def build(*, session, conversation_id, retriever) -> list[BaseTool]

静态工具(不需要会话的)忽略后三个参数即可 —— 接口统一比省那几个字值钱:
`discover()` 不必分辨模块「是工厂还是常量」。

⚠️ **新增 vs 修改,两条不同的规矩**(2026-09-23 实测订正,原先只写了「要重启」):

- **新增**一个模块(**新文件名**)**当场生效,不需要重启**。本函数每请求都
  `pkgutil.iter_modules` 重扫一遍包目录,而 FileFinder 的目录缓存按 mtime 失效
  ⇒ 新文件名**立刻**被列出来,再 `importlib.import_module` 导入即可。
  实测:客服服务**没重启**,把新模块写进本包,下一次请求模型就调到了它。
- **修改**一个**已有**模块**必须重启**。`importlib.import_module` 对已导入的名字
  直接返回 `sys.modules` 里的**缓存项**,磁盘上的改动不会重读。

**所以「内置 vs MCP」的差别不在热重载能力**(新增这一半两边都能即插即用),
**而在工具从哪来**:

- **内置**:`app/tools/builtin/` 里**我们的**代码,进程内 `import`,写完就在;
  「重载」靠的是「每请求重扫目录」这个副作用,不是设计出来的热重载;
- **MCP**(`app/mcp/client.py`):别的进程里的工具,每请求经 MCP 协议**现问现拿**
  —— 那边**改**代码只要重启**它自己**,客服服务一概不动(验收 3)。

一句话:**两者的差别不在「能不能热更新」,而在工具活在谁的进程里** ——
内置是我们**本进程**里 import 进来的代码(新增即插即用,改**已有**模块要重启我们),
MCP 是**别的进程**里的工具(改那边只要重启**它自己**,客服服务一概不动)。

⚠️ 这里一度写成「内置的数量在**部署**时定死,MCP 的数量在**运行时**由对方决定」——
那句**与上面第一条 bullet 直接打架**,而且它是本章**已被实测证伪**的那个说法的复述:
`discover()` 每请求重扫目录,内置**新增**一个文件当场生效。别再写回去。
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
