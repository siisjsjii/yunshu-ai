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

修复轮 1 追加:基础设施故障必须走 502(I1)、失败退出路径必须放锁(I2a)、
409 分支要有自己的断言(I2b),以及「建单即转人工」这条只有真工具会做的
副作用(M6)。
"""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain.tools import tool
from sqlalchemy.exc import OperationalError

from app.api import chat as chat_api
from app.config import Settings
from app.db.models import Conversation, Ticket
from app.db.session import get_session
from app.main import app
from app.memory.store import SessionStore
from app.tools.registry import _spec_from_tool

# 顶层 import(`tests/` 没有 `__init__.py`,pytest 默认的 prepend 导入模式会把
# `tests/` 放进 sys.path)—— **不要**写成 `tests.test_api_chat`:那会让同一份
# 替身以两个不同的模块名被加载两遍,`FakeSession` 变成两个类,
# `isinstance` 判断会莫名其妙地为假。已实测 `import test_api_chat` 可用。
from test_api_chat import DEFAULT_LOGIN_USER, FakeSession

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


_REQUIRED_SETTINGS = dict(
    openai_base_url="https://example.invalid/v1",
    openai_api_key="sk-test",
    openai_model="test-model",
    database_url="mysql+aiomysql://u:p@localhost/db",
)


def _settings(**overrides):
    """测试用 Settings:`_env_file=None` 是仓规(不让本机 .env 决定测试通过与否)。"""
    return Settings(_env_file=None, **{**_REQUIRED_SETTINGS, **overrides})


def _client(session, *, store=None, **settings_overrides):
    """端点级测试客户端:三条依赖缝都替换掉(会话 / 会话存储 / 配置)。

    ⚠️ **这条注记在 ch08 T7 被推翻了,原文作废**。它原先写着:

        「替换 `get_settings` 并不能让这个文件摆脱仓库根的 `.env` ——
         `build_registry` 内部的 `build_retriever` 是**硬连线**调模块级
         `get_settings()`,不走 `Depends`,所以 DI 缝够不到它。
         实测(移走 `.env` 后跑)本文件红 **1** 条,只剩主用例。」

    那是**真的**:`build_registry` 收了一个 `settings` 参数却从不往下传,
    `build_retriever` 于是越过 DI 缝读全局配置。T7 把这个「看起来接上、
    其实没接」的参数**接通了**(`build_retriever(session, settings)`,
    必传),这条覆盖的用途(控制等锁超时与密钥占位值)现在**真的**落在
    `get_settings` 的 DI 缝上。

    **实测(2026-09-22,T7 落地后,移走 `.env` 再跑)**:
        5 passed in 2.79s
    不再有红。同一轮里仍然依赖 `.env` 的只剩**显式传 `settings=None`**
    的那几条(「调用方明确表示不关心配置」的入口),如 `tests/test_registry.py`
    与 `tests/test_mcp_client.py` 里的纯组装用例。

    (顺带记一条与本改动无关的观测:移走 `.env` 后带 `@pytest.mark.db` 的
     文件会**挂住**而不是快速报错 —— 它们经 `app.db.base.get_sessionmaker`
     读真实 `.env` 的 `DATABASE_URL`,连不上时卡满前向超时。
     `tests/test_api_refund.py` 实测 `timeout 60` 用尽。)
    """
    store = store if store is not None else SessionStore(ttl_seconds=60, max_sessions=100)

    async def _session_override():
        yield session

    app.dependency_overrides[get_session] = _session_override
    app.dependency_overrides[chat_api.get_store] = lambda: store
    app.dependency_overrides[chat_api.get_settings] = lambda: _settings(**settings_overrides)
    client = TestClient(app)
    client.store = store          # 便于断言锁对象
    return client


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
    # ↓ 只有**真工具**会「建单即转人工」。端点若把建单逻辑内联重写一遍,
    #   状态会停在 ensure_conversation 建的 "active" —— 这条正好抓它。
    #   (原来这里断的是 `session.commits >= 1`,它近乎恒真:ensure_conversation
    #   自己就会 commit,把工具那行 `await session.commit()` 删掉它照样绿 ——
    #   所以换成上面这条只有真工具会满足的断言。)
    assert session.conversations[SID].status == "pending_human"


def test_a_foreign_session_is_refused_and_no_ticket_lands():
    """**不能往别人的会话里建工单**(认证,2026-09-27)。

    `ensure_conversation` 对**已存在**的行忽略传入的 `user_id` ⇒ 不查这一下的话,
    知道别人 32 位会话 id 的人能往那条会话上写一张工单。

    **三条**:404(与读端点同一个出口)、`tickets` 一行都没建、第二次仍是 404
    (不是 409 —— 锁必须放掉,否则那条会话永久钉死)。

    ⚠️ 会话是**预置**的、`user` 是别人的:`ensure_conversation` 只负责新建,
    不换一个 owner 出来的话这条用例根本走不到那道判据上。
    """
    session = _TicketSession()
    session.conversations[SID] = Conversation(
        id=SID, user="someone-else", status="active",
        summary_upto_msg_id=0, layer1_from_msg_id=0,
    )
    client = _client(session)
    try:
        first = client.post("/api/ticket", json={"session_id": SID})
        second = client.post("/api/ticket", json={"session_id": SID})
    finally:
        app.dependency_overrides.clear()

    assert first.status_code == 404, first.text
    assert first.json()["detail"] == "会话不存在"
    assert session.tickets == [], "往别人的会话里建出了工单"
    # 第二次仍是 404 而不是 409:失败退出路径放了锁(本仓记过账:漏放 = 永久 409)
    assert second.status_code == 404, second.text
    assert client.store.lock_for(SID).locked() is False


def test_own_existing_session_still_creates_a_ticket():
    """**正面对照**:会话已存在、且是自己的 ⇒ 照常建单。

    只有「别人的 ⇒ 404」的话,一个**一律 404** 的实现(判据写反、或拿错比较对象)
    照样满足它。这条同时钉住「已存在的会话不会被那道检查误伤」——
    `ensure_conversation` 走的是"已存在就返回"那一支,返回值的 `user` 必须被
    拿来与登录用户比**相等**,不是比不等。
    """
    session = _TicketSession()
    session.conversations[SID] = Conversation(
        id=SID, user=DEFAULT_LOGIN_USER, status="active",
        summary_upto_msg_id=0, layer1_from_msg_id=0,
    )
    client = _client(session)
    try:
        resp = client.post("/api/ticket", json={"session_id": SID})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200, resp.text
    assert [t.conversation_id for t in session.tickets] == [SID]


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


def _failing_create_ticket():
    """建工单工具的替身:写库时 DB 不可用。

    走的是**真实**的故障分类路径:executor 捕 `SQLAlchemyError` 后抛
    `ToolInfrastructureError`(app/tools/executor.py:91),端点必须把它翻成 502。
    """

    @tool
    async def create_ticket(description: str, ticket_type: str) -> str:
        """替身:写库时 DB 不可用。"""
        raise OperationalError("INSERT INTO tickets", {}, Exception("连接断开"))

    return create_ticket


def _reg(*tools) -> dict:
    """`BaseTool` 列表 → `name → ToolSpec`(ch08 T4 起注册表的形状)。

    用生产的 `registry._spec_from_tool` 转换,不手搭 `ToolSpec`:手搭的话
    `kind` / `input_schema` 就是测试自己编的,而**执行器的权限闸与校验闸
    读的正是这两个字段** —— 那样「点按钮 → APPROVED 就真建单」这条会退化成
    「测试自己造了一个能过闸的 spec」。
    """
    return {t.name: _spec_from_tool(t, source="builtin") for t in tools}


def _slow_create_ticket():
    """建工单工具的替身:慢工具。本端点没有模型,「占住锁」只能靠它。"""

    @tool
    async def create_ticket(description: str, ticket_type: str) -> str:
        """替身:慢工具,持锁 0.4s 后正常返回。"""
        await asyncio.sleep(0.4)
        return json.dumps(
            {"ticket_no": "T-slow", "conversation_id": SID, "status": "open"},
            ensure_ascii=False,
        )

    return create_ticket


def test_infrastructure_failure_returns_502_not_500(monkeypatch):
    """DB 故障必须是 502 + 固定文案,**不是** FastAPI 默认的 500。

    这是端点里**唯一真会发生的故障路径**:`if not outcome.ok` 那条 502 分支今天
    不可达(registry 必有 create_ticket、args 是硬编码合法值、`ToolNotFound` 要求
    description 为空 —— 三者都不可能),而 MySQL 瞬时不可用走的是
    `executor` → `ToolInfrastructureError` → 端点没有 `except` 时是 **500**
    「Internal Server Error」。500 会把「服务端出问题」说成「你的请求有问题」,
    与本仓的错误语义边界相悖(CLAUDE.md:基础设施故障一律 502)。

    **顺序**:这个 `except` 子句先于本用例落盘。反过来写的话,`ToolInfrastructureError`
    会从 `client.post(...)` **直接抛出**(`TestClient` 默认 `raise_server_exceptions=True`,
    未处理异常被重新抛出,而不是变成 500 响应),用例以 **error** 收场,而不是红在
    `500 != 502` 这条断言上 —— 那种红不告诉你是测试写错了还是实现错了,
    这正是「先把实现落盘、再写用例」的理由。
    """
    session = _TicketSession()
    client = _client(session)
    monkeypatch.setattr(
        chat_api,
        "build_registry",
        lambda *, session, conversation_id, settings=None: _reg(_failing_create_ticket()),
    )
    try:
        resp = client.post("/api/ticket", json={"session_id": SID})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 502
    # 出站的是 executor 那句固定文案,不是原始 SQLAlchemy 异常文本
    # (等价于「不回显密钥」:原始文本里带什么都不会流出去)。
    assert resp.json()["detail"] == "数据服务暂时不可用"


def test_failed_request_releases_lock_so_the_session_stays_usable(monkeypatch):
    """失败退出路径也必须放锁 —— 漏放的后果是**该会话永久 409**。

    持锁的锁既不被 TTL 也不被 LRU 回收,而且 ticket 端点与聊天端点**共用同一个
    进程级 `_store`**,泄漏会连带毒掉聊天。审查者变异实测:把端点里
    `finally: lock.release()` 换成 `finally: pass`,本文件**依然 2 passed** ——
    所以这条断言不是锦上添花,它是这个文件里唯一能看见锁的东西。

    等锁超时调成 0.15s:真漏了放锁,第二次请求会在 0.15s 内变红,而不是用默认
    的 60s 把测试挂死。
    """
    session = _TicketSession()
    client = _client(session, session_lock_timeout_seconds=0.15)
    monkeypatch.setattr(
        chat_api,
        "build_registry",
        lambda *, session, conversation_id, settings=None: _reg(_failing_create_ticket()),
    )
    try:
        first = client.post("/api/ticket", json={"session_id": SID})
        second = client.post("/api/ticket", json={"session_id": SID})
    finally:
        app.dependency_overrides.clear()

    assert first.status_code == 502
    # 第二次仍是 502 —— 关键是它**不是 409**:锁被放掉了,这个会话还能继续用。
    assert second.status_code == 502
    # 顺带断一次锁对象本身,便于定位(两条都在断"会话仍可用"这件事)。
    assert client.store.lock_for(SID).locked() is False


@pytest.mark.anyio
async def test_concurrent_same_session_second_request_times_out_with_409(monkeypatch):
    """同 session 并发:一个拿到锁走完,另一个等锁超时 → 恰好一个 409。

    这段 409 与对话端点 `app/api/chat.py` 的那段**行为等价**(同状态码、同 detail
    字符串),而那段在 `tests/test_api_chat.py` 有具名测试。**"它是复制过来的、
    那边测过了"正是本项目复盘时吃过亏的论证方式** —— 复制品要有自己的断言。

    照抄那边的并发写法(`httpx.ASGITransport` + `asyncio.gather`):TestClient 是
    同步的,两个线程里跑同一事件循环会带来额外的调度不确定性。
    """
    session = _TicketSession()
    client = _client(session, session_lock_timeout_seconds=0.15)
    monkeypatch.setattr(
        chat_api,
        "build_registry",
        lambda *, session, conversation_id, settings=None: _reg(_slow_create_ticket()),
    )
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as c:
            first, second = await asyncio.gather(
                c.post("/api/ticket", json={"session_id": SID}),
                c.post("/api/ticket", json={"session_id": SID}),
            )
    finally:
        app.dependency_overrides.clear()

    assert sorted([first.status_code, second.status_code]) == [200, 409]
    loser = first if first.status_code == 409 else second
    assert "正在处理另一条消息" in loser.text
