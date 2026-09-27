"""登录端点的行为面。**要 MySQL**(它要读 users 表)。

⚠️ 这一份**刻意请求** `_no_default_login`(Task 4 加的 autouse 开关;计划裁定 R9
把它从 Task 5 挪到了 T4)—— 它测的就是「没登录时会怎样」,
而那个全局装置默认替每条用例装一个已登录用户。
⚠️ **本仓没有 `session` fixture**:种账号用**直接 engine**,照
`tests/test_api_conversations_db.py` 既有形状写。

## ⚠️ 传输层**不用 `TestClient`**(T3 实现轮实测订正 brief 原稿)

brief 原稿写的是 `TestClient(app)`。本机 2026-09-27 实测三种形状:

| 形状 | 读数 |
|---|---|
| `TestClient` 打一枪、**不种库** | **200** —— 端点本身没问题 |
| `TestClient` + 先 `await` 种库(= brief 原稿) | **RuntimeError: got Future … attached to a different loop** |
| `httpx.ASGITransport` + 先 `await` 种库 | **200** |

根因:`TestClient` **每次请求起一个新的 portal 事件循环**
(`starlette/testclient.py` 的 `_portal_factory`,只有把它当 context manager 用时
才复用同一个),而 `_seed_now` 跑在 **pytest 自己的循环**上 —— 它在 `get_engine()`
那个 **lru_cache 单例**里留下一条绑在 pytest 循环上的池化连接,portal 循环下一步
`pre_ping` 它,当场炸。**换成 `with TestClient(app)` 也救不回来**(实测同红):
毒在种库那一步,不在 portal 的生命周期里。

这正是本仓两份文件头记过的**同一个** trap(`tests/test_api_ticket.py`:
「否则是本仓第一个『TestClient + 真实 engine』的组合」;`tests/test_api_feedback.py`:
「`TestClient` 自建 portal 事件循环,会把 `get_engine()` 那个 lru_cache 单例绑到
**它的**循环上」)。⇒ 走它们那条缝:`ASGITransport` 把请求跑在**测试自己的**循环里,
与紧随其后的回查同一个循环;它**也不跑 lifespan**(BGE-M3 预热线程不会被拉起)。
**六条断言逐字未动**,改的只有「谁来发这个请求」。
"""

import httpx
import pytest

import app.api.auth as auth_api
from app.auth import ADMIN, create_token
from app.main import app
from app.db.base import get_engine, get_sessionmaker
from scripts.seed_users import seed

pytestmark = pytest.mark.db

GOOD = {"username": "cinfly", "password": "123456"}


@pytest.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def _seed_now() -> None:
    async with get_sessionmaker()() as session:
        await seed(session)


@pytest.mark.anyio
async def test_login_returns_a_token_that_me_accepts(client, _no_default_login):
    await _seed_now()
    r = await client.post("/api/auth/login", json=GOOD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"token", "username", "role", "expires_at"}, body
    assert body["username"] == "cinfly" and body["role"] == ADMIN

    me = await client.get("/api/auth/me",
                          headers={"Authorization": f"Bearer {body['token']}"})
    assert me.status_code == 200, me.text
    assert me.json() == {"username": "cinfly", "role": ADMIN}
    await get_engine().dispose()


@pytest.mark.anyio
async def test_wrong_password_and_unknown_user_are_indistinguishable(client, _no_default_login):
    """两种失败必须**逐字相同** —— 否则响应文案成了「用户名枚举」的接口。"""
    await _seed_now()
    a = await client.post("/api/auth/login", json={"username": "cinfly", "password": "nope"})
    b = await client.post("/api/auth/login", json={"username": "nobody", "password": "nope"})
    assert a.status_code == b.status_code == 401
    assert a.json() == b.json(), f"{a.json()} != {b.json()}"
    await get_engine().dispose()


@pytest.mark.anyio
async def test_login_body_is_validated(client, _no_default_login):
    """缺 password ⇒ 422(pydantic 在碰库**之前**就拒了)。"""
    assert (await client.post("/api/auth/login",
                              json={"username": "x"})).status_code == 422


@pytest.mark.anyio
async def test_me_without_token_is_401_and_not_403(client, _no_default_login):
    r = await client.get("/api/auth/me")
    assert r.status_code == 401, f"缺 header 必须 401(不是 403),实际 {r.status_code}"


@pytest.mark.anyio
async def test_me_with_a_garbage_token_is_401(client, _no_default_login):
    """坏 token 与缺 header 必须是**同一个** 401(前端只认这一条去弹登录)。"""
    a = await client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
    b = await client.get("/api/auth/me")
    assert a.status_code == b.status_code == 401
    assert a.json() == b.json()


@pytest.mark.anyio
async def test_expired_token_is_401(client, _no_default_login):
    """**过期**必须被拒 —— 而且红的理由得是「过期」,不是「签名不匹配」。

    ⚠️ 所以密钥要**与 App 用的那一份相同**,只把有效期调成负数:
    `get_settings()` 是 App 依赖的同一个对象(lru_cache),`model_copy` 换掉一个字段。
    自己另造一份 `Settings(...)` 的话密钥很可能不同 ⇒ 这条**照样绿**,
    而它测的东西变成了「签名不匹配」—— 那是 `require_user` 的另一条分支。
    """
    from app.config import get_settings
    expired_settings = get_settings().model_copy(update={"jwt_expire_minutes": -1})
    expired = create_token(username="cinfly", role=ADMIN, settings=expired_settings)
    r = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401
    await get_engine().dispose()


@pytest.mark.anyio
async def test_both_failure_paths_still_run_verify_password(monkeypatch, client, _no_default_login):
    """**查无此人也要跑一次密码校验** —— 防的是**时序侧信道**。

    ⚠️ 这条**必须存在**:没有它的话,把「用户不存在」改成提前 `return`
    (省掉那次 scrypt)的实现**六条测试全绿**,而那正是要防的
    —— 两次失败的**耗时差一个数量级**,响应时间就成了枚举用户名的信道。
    ⚠️ 断的是**被调用过**,不是返回值(两种实现的返回值逐字相同 ⇒ 断返回值零判别力)。
    ⚠️ `monkeypatch` 打在 `app.api.auth.verify_password` 上:`login` 里是
    `from app.auth import verify_password` 之后按**局部名**调用 ⇒ 换命名空间的绑定即生效。
    """
    calls = []
    real = auth_api.verify_password
    monkeypatch.setattr(auth_api, "verify_password",
                        lambda pw, stored: (calls.append(stored), real(pw, stored))[1])
    await _seed_now()
    await client.post("/api/auth/login", json={"username": "cinfly", "password": "nope"})
    await client.post("/api/auth/login", json={"username": "nobody", "password": "nope"})
    assert len(calls) == 2, (
        f"两条失败路径都该跑一次 verify_password(实际 {len(calls)} 次)"
        " —— 少了的那次就是时序侧信道")
    assert calls[1] == "", "查无此人时传给 verify_password 的应当是空串(坏串回 False)"
    await get_engine().dispose()
