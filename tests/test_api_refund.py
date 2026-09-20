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

import httpx
import pytest
from sqlalchemy import select, text

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
        # 上面 `.one()` 已经证明"有行";这里再钉住**写进去的是请求里的那单** ——
        # 只回显请求、库里却写常量(order_no 写死 / conversation_id 用错 id)的实现
        # 能骗过 response 那两行断言,骗不过这一条。
        assert row.order_no == body["order_no"]


@pytest.mark.anyio
async def test_rejects_category_outside_the_closed_set(client_factory):
    """不在固定类目里 → 422。且**不得落库**。"""
    client = client_factory()
    r = await client.post("/api/refund", json=_body(reason_category="随便写的原因"))
    assert r.status_code == 422
    assert await _rows_for_scratch() == []


@pytest.mark.anyio
async def test_infra_failure_is_502_not_500(client_factory, monkeypatch):
    """基础设施故障一律 502 + 固定文案,不是 FastAPI 默认的 500。"""
    import app.api.refund as mod

    async def boom(*a, **k):
        from app.tools.errors import ToolInfrastructureError

        raise ToolInfrastructureError("数据库暂时不可用")

    monkeypatch.setattr(mod, "_persist", boom)
    client = client_factory()
    r = await client.post("/api/refund", json=_body())
    assert r.status_code == 502
    assert await _rows_for_scratch() == []


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
    `finally: lock.release()` 删掉,成功那条红;把它挪进 `except ToolInfrastructureError`
    里(只给异常路径放锁),失败那条绿而成功这条红。

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
    async def boom(*a, **k):
        from app.tools.errors import ToolInfrastructureError

        raise ToolInfrastructureError("数据库暂时不可用")

    monkeypatch.setattr(mod, "_persist", boom)
    bad_first = await client.post("/api/refund", json=_body())
    bad_second = await client.post("/api/refund", json=_body())
    assert bad_first.status_code == 502
    # 第二次仍是 502 —— 关键是它**不是 409**:锁被放掉了,这个会话还能继续用。
    assert bad_second.status_code == 502

    # 顺带断一次锁对象本身,便于定位(上面两条都在断"会话仍可用"这件事)。
    assert client.store.lock_for(SCRATCH).locked() is False
