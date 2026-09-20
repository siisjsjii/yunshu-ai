"""POST /api/refund 的端点测试。需要 MySQL。

为什么这个文件打**真库**(`@pytest.mark.db`),而 `tests/test_api_ticket.py` 用替身
session:本文件要钉的核心是「行真的落进了 refund_requests」。替身 session 只能证明
"端点是按我以为的形状调用了 add",证明不了落库 —— 而"没落库"恰恰是这个端点最可能
的坏法(校验写在写库之后、写错了 conversation_id、commit 漏了)。

**`client_factory` 是本文件自己的 fixture。** 任务书写它在 `tests/conftest.py`,
实际不是:那份 conftest 只有 `anyio_backend`;`tests/test_api_chat.py` 里那个同名
fixture 是**模块局部**的(`tests/` 没有 `__init__.py`,别的模块取不到它),而且它
返回 `(client, model)` 并把 `get_session` 换成替身,与本文件「打真库」的前提直接
冲突。故按任务书给出的调用形态(`client = client_factory()`、`await client.post(...)`)
在此重建一个。
"""

import asyncio

import httpx
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError

from app.api import chat as chat_api
from app.config import Settings, get_settings
from app.db.base import get_sessionmaker
from app.db.models import RefundRequest
from app.main import app
from app.memory.store import SessionStore
from app.refund.categories import REFUND_REASON_CATEGORIES

pytestmark = pytest.mark.db

SCRATCH = "refundtest0000000000000000000000"
ORDER_NO = "20240915"

_REQUIRED_SETTINGS = dict(
    openai_base_url="https://example.invalid/v1",
    openai_api_key="sk-test",
    openai_model="test-model",
    database_url="mysql+aiomysql://u:p@localhost/db",
)


@pytest.fixture(autouse=True)
async def _cleanup():
    """每个用例后清掉本文件的探针行。

    比任务书多删一行 `conversations`:端点的 `ensure_conversation` 会为 SCRATCH 建
    一条会话壳,不删就会在真库里一条条攒下来。`refund_requests.conversation_id` 目前
    没有外键,先删它只是为了在将来加上外键之后顺序依然正确。
    """
    yield
    async with get_sessionmaker()() as s:
        await s.execute(
            text("DELETE FROM refund_requests WHERE conversation_id = :c"), {"c": SCRATCH}
        )
        await s.execute(
            text("DELETE FROM conversations WHERE id = :c"), {"c": SCRATCH}
        )
        await s.commit()


@pytest.fixture
def client_factory():
    """造端点级客户端。**替掉配置与会话锁,不替 `get_session`** —— 本文件打真库。

    为什么不用 `TestClient`:`tests/test_api_ticket.py` 文件头记过账 —— TestClient
    自建 portal 事件循环,会把 `get_engine()` 那个 lru_cache 单例绑到**它的**循环上,
    污染同进程里排在后面的 db 测试。`httpx.ASGITransport` 则把请求跑在**测试自己的**
    事件循环里(MCP/CLAUDE.md 记的 409 并发用例也是这么写的),与紧随其后回查用的是
    同一个循环,不存在跨循环连接。顺带:ASGITransport **不跑 lifespan**,所以
    BGE-M3 预热线程不会被拉起。

    `get_settings` 与会话锁被替换掉,是因为端点在**真库**上跑的同时还需要两个可控量:
    等锁超时(漏放锁时要在 0.15s 内变红,而不是挂满默认的 60s)与 `openai_api_key`
    占位值(脱敏路径的输入)。这两个替换够不到 `get_session` —— 它走模块级
    `app.db.base.get_sessionmaker`,读仓库根的真实 `.env`,这正是本文件要的。
    """

    def make(*, store=None, **settings_overrides):
        the_store = (
            store if store is not None else SessionStore(ttl_seconds=60, max_sessions=100)
        )
        app.dependency_overrides[chat_api.get_store] = lambda: the_store
        app.dependency_overrides[get_settings] = lambda: Settings(
            _env_file=None,
            **{
                **_REQUIRED_SETTINGS,
                "session_lock_timeout_seconds": 0.15,
                **settings_overrides,
            },
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        client.store = the_store
        return client

    yield make
    app.dependency_overrides.clear()


def _body(**overrides):
    return {
        "session_id": SCRATCH,
        "order_no": ORDER_NO,
        "reason_category": REFUND_REASON_CATEGORIES[0],
        **overrides,
    }


async def _rows_for_scratch() -> list[RefundRequest]:
    """新 session 回查 —— 身份映射持弱引用,复用写会话读回会退化成「靠 refcount 走运」。"""
    async with get_sessionmaker()() as s:
        return (
            (
                await s.execute(
                    select(RefundRequest).where(RefundRequest.conversation_id == SCRATCH)
                )
            )
            .scalars()
            .all()
        )


@pytest.mark.anyio
async def test_creates_row_with_valid_category(client_factory):
    client = client_factory()
    r = await client.post("/api/refund", json=_body())
    assert r.status_code == 200
    body = r.json()
    assert body["order_no"] == ORDER_NO
    assert body["status"] == "pending"

    async with get_sessionmaker()() as s:
        row = (
            await s.execute(
                select(RefundRequest).where(RefundRequest.conversation_id == SCRATCH)
            )
        ).scalars().one()
        assert row.reason_category == REFUND_REASON_CATEGORIES[0]
        # 上面 `.one()` 已经证明"有行";下面把响应体**逐字段**钉回库行 ——
        # 只回显请求、库里却写常量(order_no 写死 / conversation_id 用错 id /
        # status 硬编码 "pending")的实现能骗过纯响应断言,骗不过这一组。
        # `status` 尤其要两边都断:响应里硬编码的 "pending" 恰好等于 ORM `default`
        # 与 DDL `server_default` 两层的值,只看响应是看不出来的。
        assert row.order_no == body["order_no"]
        assert row.status == body["status"]
        assert row.conversation_id == body["conversation_id"] == SCRATCH
        assert row.created_at.isoformat() == body["created_at"]


@pytest.mark.anyio
async def test_rejects_category_outside_the_closed_set(client_factory):
    """不在固定类目里 → 422。且**不得落库**。"""
    client = client_factory()
    r = await client.post("/api/refund", json=_body(reason_category="随便写的原因"))
    assert r.status_code == 422
    assert await _rows_for_scratch() == []


async def _db_is_down(*a, **k):
    """写库时数据库连不上。**裸 SQLAlchemy 错误** —— 见下面用例的 docstring。"""
    raise OperationalError("INSERT INTO refund_requests", {}, Exception("连接断开"))


@pytest.mark.anyio
async def test_infrastructure_failure_returns_502_not_500(client_factory, monkeypatch):
    """数据库故障必须是 502 + **固定文案**,不是 FastAPI 默认的 500。

    **注入的是裸 `OperationalError`,不是 `ToolInfrastructureError`** —— 这一条是
    本用例的要害,写错整条就废了。`ToolInfrastructureError` 是
    `app/tools/executor.py:89-91` 那张分类表的**产物**,而本端点手写写库、不经过
    executor。注入一个"已经翻译好"的错误,等于**跳过被验的那一步**:在"真故障返回
    500、`except ToolInfrastructureError` 是死代码"的实现下,它照样绿 —— 这正是本
    仓定义的假绿,也是 ch05 spec §8.2 记账的那条(`tests/test_api_ticket.py:151-212`
    就是这个改法的样板)。

    断言固定文案而不只是状态码:502 也可能是别的分支给的,文案才钉住"这是本仓定的
    那条基础设施答复"。原始异常文本**不得**出站(`OperationalError` 的 `str()` 里
    带着 SQL 语句)。
    """
    import app.api.refund as mod

    monkeypatch.setattr(mod, "_persist", _db_is_down)
    client = client_factory()
    r = await client.post("/api/refund", json=_body())
    assert r.status_code == 502
    assert r.json()["detail"] == "数据服务暂时不可用"
    assert "INSERT INTO refund_requests" not in r.text


@pytest.mark.anyio
async def test_refund_endpoint_uses_the_chat_session_lock(client_factory):
    """退款端点必须与对话端点**共用同一把会话锁**。

    **行为断言**,不是比较函数对象 —— 它不关心端点是 `from app.api.chat import
    get_store` 还是 `chat_api.get_store`,只要求"那把锁来自 `app.api.chat` 的注册表":
    这里先把 `client_factory` 塞进去的那把锁持住,端点若从**别处**取锁,它会一路
    走完返回 200。

    为什么这条必须存在:两份注册表意味着同一会话有**两把不同的锁**,退款与对话可以
    同时进临界区 —— 端点赖以串行化的前提静默失效,而这个失效在本文件其余每一条用例
    里都**看不出来**(它们替换的正是 chat 那一个;端点自带注册表时,替换根本够不到它,
    "锁放掉了"之类的断言会恒真)。
    """
    client = client_factory()
    lock = client.store.lock_for(SCRATCH)
    await lock.acquire()
    try:
        r = await client.post("/api/refund", json=_body())
    finally:
        lock.release()
    assert r.status_code == 409
    assert await _rows_for_scratch() == []


@pytest.mark.anyio
async def test_lock_is_released_on_every_exit_path(client_factory, monkeypatch):
    """成功与失败两条非流式退出路径都必须放锁 —— 漏放的后果是**该会话永久 409**。

    持锁的锁既不被 TTL 也不被 LRU 回收,而且退款端点与对话端点共用同一个注册表,
    泄漏会连带毒掉聊天。两条路径分开断,是因为它们对应两种不同的写错法:把
    `finally: lock.release()` 删掉,两条都红;把它挪进某个 `except` 里(只给异常
    路径放锁),失败那条绿而成功这条红。

    失败路径注入的仍是**裸 `OperationalError`**(复用 `_db_is_down`),理由同
    `test_infrastructure_failure_returns_502_not_500`:注入已翻译的错误会绕开被验
    的那一步。

    等锁超时是 0.15s(`client_factory` 的默认),真漏了放锁,第二次请求会在 0.15s 内
    变红,而不是用默认的 60s 把测试挂死。
    """
    import app.api.refund as mod

    client = client_factory()

    # 成功路径
    ok_first = await client.post("/api/refund", json=_body())
    ok_second = await client.post("/api/refund", json=_body())
    assert [ok_first.status_code, ok_second.status_code] == [200, 200]

    # 失败路径
    monkeypatch.setattr(mod, "_persist", _db_is_down)
    bad_first = await client.post("/api/refund", json=_body())
    bad_second = await client.post("/api/refund", json=_body())
    assert bad_first.status_code == 502
    # 第二次仍是 502 —— 关键是它**不是 409**:锁被放掉了,这个会话还能继续用。
    assert bad_second.status_code == 502

    # 顺带断一次锁对象本身,便于定位(上面两条都在断"会话仍可用"这件事)。
    assert client.store.lock_for(SCRATCH).locked() is False


@pytest.mark.anyio
async def test_cancelled_request_releases_lock(client_factory, monkeypatch):
    """**取消**也必须放锁 —— `CancelledError` 是 `BaseException`,不是 `Exception`。

    这条路径在本端点上确实是结构安全的(持锁之后的整段都在一个 `try/finally` 里,
    且 `acquire()` 成功到进 `try` 之间**没有 await**),但本仓的规矩是「每一条退出
    路径都要有**具名测试**」—— 而"结构上必然安全"正是复盘时吃过亏的那种论证:
    把 `finally` 换成一个只捕 `Exception` 的 `except` 分支,这段代码**读起来依然对**
    (谁会把 `CancelledError` 当异常呢),只有这条用例会红。

    做法:把 `_persist` 卡在一个**永不 set 的 Event** 上 —— 请求因此停在"已持锁"的
    状态里 —— 确认锁真被持住之后 cancel 掉那个任务,再断锁已释放。
    `locked() is True` 那一步是必要的:没有它,若 `_persist` 压根没被调到,
    后面 `locked() is False` 会因为"锁从来没被拿过"而恒真。
    """
    import app.api.refund as mod

    entered = asyncio.Event()
    never = asyncio.Event()

    async def hang(*a, **k):
        entered.set()
        await never.wait()
        raise AssertionError("取消之后不该继续执行")

    monkeypatch.setattr(mod, "_persist", hang)
    client = client_factory()

    task = asyncio.create_task(client.post("/api/refund", json=_body()))
    await entered.wait()
    # 前置条件:此刻端点确实**持着**那把锁(否则下面的 False 是恒真的)。
    assert client.store.lock_for(SCRATCH).locked() is True

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.store.lock_for(SCRATCH).locked() is False
