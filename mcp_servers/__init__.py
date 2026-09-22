"""两个业务 MCP Server —— **各自独立进程**。

启动(各自一个终端):

    .venv/Scripts/python.exe -m mcp_servers.logistics     # 127.0.0.1:8101/mcp
    .venv/Scripts/python.exe -m mcp_servers.aftersales    # 127.0.0.1:8102/mcp

**不接真实系统、不建表** —— 数据全部来自 `app.tools.mock_data`
(与内置工具**共用**,详见该模块的 docstring 与 spec §8.2)。
"""
