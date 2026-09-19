"""建工单端点:按钮点击是 HTTP 请求,够不到模型工具 —— 故需要这个入口。

**这里刻意不用真库**(否则是本仓第一个「TestClient + 真实 engine」的组合):
`TestClient` 在它自己的 portal 事件循环里跑请求,会把 `get_engine()` 那个
**lru_cache 单例**绑到那个循环上;退出 `with` 后 portal 循环关闭,同进程里
后面所有走 `get_sessionmaker()` 的 db 测试(文件名排在 `test_api_ticket` 之后)
都会拿到跨循环的连接 —— 这是 ch04 记过账的故障形态。端点级测试在本仓
**一律替换 `get_session`**(`tests/test_api_chat.py` 的 `client_factory` 即此),
这里沿用同一条缝。

真库的「写进去了吗」由 `tests/test_tools_db.py::test_create_ticket_writes_row`
独占(它用新 session 回查 `select(Ticket)`)—— 那条钉的是工具,
这条钉的是**端点真的把工具调起来了**。
"""

from fastapi.testclient import TestClient

from app.db.models import Ticket
from app.db.session import get_session
from app.main import app

# 顶层 import(`tests/` 没有 `__init__.py`,pytest 默认的 prepend 导入模式会把
# `tests/` 放进 sys.path)—— **不要**写成 `tests.test_api_chat`:那会让同一份
# 替身以两个不同的模块名被加载两遍,`FakeSession` 变成两个类,
# `isinstance` 判断会莫名其妙地为假。已实测 `import test_api_chat` 可用。
from test_api_chat import FakeSession

SID = "00000000000000000000000000000001"


class _TicketSession(FakeSession):
    """在既有端点替身上补 `Ticket` 支持 —— 建工单端点写的正是它。

    继承而不是另写一份:`execute`/`commit`/`Conversation` 的行为必须与
    对话端点测试**同一套**,否则两个端点的替身会各自漂移。
    """

    def __init__(self):
        super().__init__()
        self.tickets: list[Ticket] = []

    def add(self, obj):
        if isinstance(obj, Ticket):
            self.tickets.append(obj)
        else:
            super().add(obj)


def _client(session):
    async def _override():
        yield session

    app.dependency_overrides[get_session] = _override
    return TestClient(app)


def test_build_ticket_invokes_the_tool_and_returns_its_payload():
    """按钮 → HTTP → **真的**调到 `create_ticket` 并把它的回参原样返回。

    **必须断言落库动作,不能只断言响应体**:把 `execute_tool(...)` 那段换成
    `return {"ticket_no": "T-假", "status": "open"}`,响应断言**全绿** —— 一个
    假 `ticket_no` 与真的一模一样(都是 `T-` 开头)。`session.tickets` 是唯一
    能区分「调了工具」和「编了一个」的东西。
    """
    session = _TicketSession()
    client = _client(session)
    try:
        resp = client.post("/api/ticket", json={"session_id": SID})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["ticket_no"].startswith("T-")
    assert body["status"] == "open"
    # ↓ 判别性断言:工具的写动作真的发生了,且号与响应里的一模一样
    assert [t.ticket_no for t in session.tickets] == [body["ticket_no"]]
    assert session.tickets[0].conversation_id == SID
    assert session.commits >= 1


def test_session_id_length_is_bounded_like_chat_request():
    """上限必须与 conversations.id 的 varchar(32) 对齐 —— 否则 DataError 会被判成 502。

    **同样要替换 `get_session`**,不要图省事写裸 `TestClient(app)`。FastAPI 在
    **422 之前就会进入 yield 依赖** —— 已实测:body 校验失败时依赖的 `enter`/`exit`
    都跑了。所以裸 `TestClient` 照样会把 `get_engine()` 的 lru_cache 单例建在
    portal 循环上,正是文件头警告的那个组合(第一版计划这里就是裸的,已订正)。
    """
    session = _TicketSession()
    client = _client(session)
    try:
        resp = client.post("/api/ticket", json={"session_id": "x" * 33})
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 422
    # 请求**根本没进端点**(校验先于函数体):长 session_id 不该建出工单
    assert session.tickets == []
