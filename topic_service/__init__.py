"""主题分类**旁路推理服务** —— 独立进程,不在主链路上。

启动:

    .venv/Scripts/python.exe -m topic_service --model models/topic-clf --port 8103

`8101` / `8102` 已经是两个 MCP Server 的,所以这里用 **8103**。

**照 `mcp_servers/` 的先例单独立在仓库根、不进 `app/`**:它是旁路进程,
`app/agent/` 与 `app/api/chat.py` 对它的 import 数为 **0** ⇒ **实时对话一个字节都不经过
分类器**(那是 spec §3.3 ① 那条不变量;守着它的源码扫描测试是 **Task 14** 的活,今天还没有)。
它是「池子 → 分布页」那条离线链路上的一环(由 `scripts/classify_topics.py` 调,
那是 **Task 13** 的活)。

⚠️ 它**只消费 `app.topic`**(纯数据 / 纯函数那一侧:`load_artifacts` 与类目表),
不 import `app/` 的请求侧任何东西。
"""
