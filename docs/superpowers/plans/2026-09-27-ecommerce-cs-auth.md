# 认证(JWT 登录)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给云枢客服加上 JWT 登录,让会话侧栏与历史回载**只回当前登录用户自己的**会话,并且没带/坏了 token 的请求一律 401。

**Architecture:** 密钥与算法收在一个新模块 `app/auth.py`(全仓**唯一**的鉴权边界,写法对齐 `app/observability.py`)。九个既有 router 各改**一行** `APIRouter(dependencies=[...])` 挂上守卫,需要用户身份的端点再从依赖里拿。前端两份静态页共用一个新 `app/static/auth.js`,401 ⇒ 弹登录浮层 ⇒ 重放原请求。

**Tech Stack:** FastAPI 0.141.1 / PyJWT 2.14.0(HS256)/ `hashlib.scrypt`(stdlib,零新依赖)/ pytest。

**Spec:** `docs/superpowers/specs/2026-09-27-ecommerce-cs-auth-design.md`(读它,尤其是 §3 那条被推翻的既定非目标、§6.4 权限矩阵、§12.1 的实测订正)

## Global Constraints

- Python 一律用**显式** `.venv/Scripts/python.exe`(裸 `python` 今天恰好对,那是环境碰巧)。
- **不加 `-q`**:`pytest.ini` 的 `addopts` 里已经有一个,叠加成 `-qq` 会**不打印 `N passed`**。
- 只跑不需要库的:`-m "not db"`;标了 `@pytest.mark.db` 的**要 MySQL 起着**。
- **不新增第三方依赖**,除了显式声明 `PyJWT==2.14.0`(它今天只是 `mcp` 的传递依赖)。
- 所有**出站**错误文本过 `app/sanitize.py:redact_api_key`(本仓既有规矩,不新开例外)。
- 注释与文案**一律中文**,风格照抄邻近代码(为什么这么做、不这么做会怎样)。
- 每个任务**独立跑测试 + 独立提交**;提交信息用中文,前缀照本章既有格式。
- ⚠️ **Windows 平台陷阱**:含中文的请求体**不走 curl 的 argv**(走 stdin heredoc);
  跨工具传文件路径用 `$TEMP`(bash 的 `/tmp` 是 MSYS 的,Python 的 `/tmp` 是 `D:\tmp`)。
- ⚠️ **改完代码要重启服务**:静态页没有缓存头 ⇒ 前端改动还要**硬刷新(Ctrl+F5)**。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `app/auth.py` | **新建** | 唯一的鉴权边界:密码哈希 / 签发解码 / 两个依赖 / `AuthenticatedUser` / `AuthError` |
| `db/auth.sql` | **新建** | `users` 表(**刻意不幂等**,与 `db/ch08.sql` 等同规矩) |
| `scripts/seed_users.py` | **新建** | 幂等预置两个账号 |
| `app/api/auth.py` | **新建** | `POST /api/auth/login`、`GET /api/auth/me` |
| `app/static/auth.js` | **新建** | 前端唯一的 401 出口(token 存取 / 登录浮层 / `authFetch`) |
| `app/config.py` | 改 | `jwt_secret` / `jwt_expire_minutes` |
| `app/schemas.py` | 改 | `LoginRequest` / `TokenResponse`;**删** `ChatRequest.user_id` |
| `app/main.py` | 改 | 挂 `auth_router`(`mount("/")` **之前**) |
| `app/services/history.py` | 改 | 新增 `get_owned_conversation`(读路径的归属检查) |
| `app/api/conversations.py` | 改 | 列表按 token 过滤;明细做归属检查;删 `DEMO_USER` |
| `app/api/chat.py` / `refund.py` | 改 | `user_id` 来自 token,不再来自请求体 |
| `app/api/{chat,conversations,extract,feedback,refund}.py` | 改 | `APIRouter(dependencies=[Depends(require_user)])` |
| `app/api/{kb,review,topics,traces}.py` | 改 | `APIRouter(dependencies=[Depends(require_admin)])` |
| `app/static/index.html` / `admin.html` | 改 | 换成 `authFetch` + 引入 `auth.js` |
| `tests/conftest.py` | 改 | autouse 的「默认已登录」override |
| `tests/test_auth.py` / `test_api_auth.py` / `test_auth_wiring.py` | **新建** | 内核 / 端点 / 接线 |
| `scripts/acceptance*.sh`(7 个) | 改 | 各加三行:login + 遮蔽 `curl` |
| `requirements.txt` / `.env.example` / `CLAUDE.md` / `dev-notes/ch10.md` / spec §12 | 改 | 记账 |

---

## Task 1: 鉴权内核 `app/auth.py`(纯函数 + 两个依赖)

**Files:**
- Create: `app/auth.py`
- Modify: `app/config.py`(在 `retrieval_timeout_seconds` 之后追加)
- Modify: `requirements.txt`
- Test: `tests/test_auth.py`

**Interfaces:**
- Consumes: `app.config.Settings`(`jwt_secret` / `jwt_expire_minutes`)
- Produces(后面每个任务都用这些**确切名字**):
  - `class AuthError(Exception)` —— 一切鉴权失败的**唯一**异常类型
  - `@dataclass(frozen=True) class AuthenticatedUser: username: str; role: str`
  - `hash_password(password: str) -> str`
  - `verify_password(password: str, stored: str) -> bool`
  - `create_token(*, username: str, role: str, settings: Settings) -> str`
  - `decode_token(token: str, settings: Settings) -> AuthenticatedUser`(失败抛 `AuthError`)
  - `require_user(creds, settings) -> AuthenticatedUser`(FastAPI 依赖)
  - `require_admin(user) -> AuthenticatedUser`(FastAPI 依赖)
  - `ADMIN = "admin"` / `USER = "user"`(两个 role 字面量)

- [ ] **Step 1: 先加两个配置项**

在 `app/config.py` 里 `retrieval_timeout_seconds` 那一行**之后**追加:

```python
    # 认证(2026-09-27 用户点名要求;见 docs/superpowers/specs/2026-09-27-ecommerce-cs-auth-design.md)。
    #
    # `jwt_secret`:HS256 的签名密钥。
    #   ⚠️ **代码里不给真值** —— 空串 ⇒ `app/auth.py` 在**首次用到时**随机生成一个
    #   并打一条 WARNING(代价:重启后所有旧 token 失效)。这样「clone 下来不配也能跑」,
    #   而**密钥不进源码**。本机 `.env` 与入库的 `.env.example` 里都是 `itcinfly`
    #   (用户 2026-09-27 拍板的值)—— ⚠️ **`.env.example` 是入库的**,
    #   所以那个值是**公开的**,只能本机演示用。
    jwt_secret: str = ""
    #
    # `jwt_expire_minutes`:token 有效期。720 = 12 小时,一次登录够一个工作日。
    #   没有刷新机制(spec §2 明确不做)⇒ 到点就重新登录。
    #   一句话回退:`.env` 里调这个数。
    jwt_expire_minutes: int = Field(default=720, gt=0)
```

并**显式**在 `requirements.txt` 加一行(位置紧随 `httpx` 之后):

```
PyJWT==2.14.0
```

> ⚠️ 加它的理由写进提交信息:`pip show PyJWT` 显示 `Required by: mcp` ——
> 它今天**只是传递依赖**,mcp 哪天不带它了,登录会**在 import 处突然崩**。

- [ ] **Step 2: 写失败测试 `tests/test_auth.py`**

```python
"""鉴权内核:`app/auth.py` 的纯函数部分 + 两个 FastAPI 依赖。

**不打 `pytest.mark.db`** —— 这一份一个库都不碰(用户名与角色就在 token 里,
`require_user` **不查库**)。查库的只有登录端点(`tests/test_api_auth.py`)。

## 为什么每条都要断「它到底拒了没有」

这一份的性质是**拒绝**:一个「永远返回 True」的 `verify_password`、或一个
「任何 token 都解出 admin」的 `decode_token`,在**别的所有测试里**都不会红
(它们走 conftest 那个默认已登录的替身)。⇒ 这里每一条都必须有一个
**必须被拒**的输入,而且那条输入要真的构造得出来。
"""

import time

import jwt
import pytest

from app.auth import (
    ADMIN, USER, AuthError, AuthenticatedUser, create_token, decode_token,
    hash_password, require_admin, require_user, verify_password,
)
from app.config import Settings

_REQUIRED = dict(
    openai_base_url="https://example.invalid/v1",
    openai_api_key="sk-auth-test-KEY",
    openai_model="test-model",
    database_url="mysql+asyncmy://u:p@h:3306/db",
)

#: ⚠️ 非默认的密钥 —— 用默认值的话「密钥真的被用上了吗」区分不出来
#: (随机密钥与这个值下的行为长得一样)。
SECRET = "unit-test-secret"


def _settings(**over) -> Settings:
    return Settings(_env_file=None, jwt_secret=SECRET, **{**_REQUIRED, **over})


def test_hash_is_salted_so_two_hashes_of_the_same_password_differ():
    a, b = hash_password("123456"), hash_password("123456")
    assert a != b, "两次哈希相同 ⇒ 盐没生效(固定盐的库一次泄露全泄露)"
    assert a.startswith("scrypt$"), f"自描述串的形状变了:{a[:20]}"


def test_verify_accepts_the_right_password_and_rejects_a_wrong_one():
    stored = hash_password("123456")
    assert verify_password("123456", stored) is True
    assert verify_password("1234567", stored) is False, "错密码必须拒"
    assert verify_password("", stored) is False


def test_verify_rejects_a_malformed_stored_string_instead_of_raising():
    """坏串(手工改库 / 换过算法)必须**回 False**,不能抛。

    抛的话它会从端点里冒成 500 —— 而「这条记录的哈希读不懂」对用户是
    「密码不对」这一件事,不是服务端故障。
    """
    for bad in ("", "明文密码", "scrypt$16384$8$1$notbase64$xxx", "bcrypt$aa$bb"):
        assert verify_password("123456", bad) is False, f"{bad!r} 应当回 False"


def test_token_round_trip_carries_username_and_role():
    tok = create_token(username="cinfly", role=ADMIN, settings=_settings())
    u = decode_token(tok, _settings())
    assert u == AuthenticatedUser(username="cinfly", role=ADMIN)


def test_expired_token_is_rejected():
    """有效期靠 `exp`,由 PyJWT 自己校验 —— 这条钉住「真的设了 exp」。

    `jwt_expire_minutes` 传**负数**造一个出生即过期的 token(比 sleep 快且不骰子)。

    ⚠️ **必须用 `model_copy` 换那一个字段,不能 `_settings(jwt_expire_minutes=-1)`**:
    配置项带 `gt=0`(下界是**故意的** —— 写错要在启动时炸,不能等运行时变成
    「token 永不过期」这种静默故障),构造时传 −1 会被 pydantic 拒 ⇒
    那条用例会死在 `ValidationError` 上,**而它根本没验到「过期」**。
    (计划初稿就是这么写的 —— 实现者实测抓到并用 `model_copy` 修好;
    这里同步改掉,免得下游照抄。)
    """
    expired = _settings().model_copy(update={"jwt_expire_minutes": -1})
    tok = create_token(username="cinfly", role=ADMIN, settings=expired)
    with pytest.raises(AuthError):
        decode_token(tok, _settings())


def test_signature_tampering_is_rejected():
    """换一个密钥签的 token 必须被拒(签名真的在校验)。"""
    other = Settings(_env_file=None, jwt_secret="a-completely-different-secret",
                     **_REQUIRED)
    tok = create_token(username="cinfly", role=ADMIN, settings=other)
    with pytest.raises(AuthError):
        decode_token(tok, _settings())


def test_alg_none_token_is_rejected():
    """`alg: none` 的经典攻击:不签名的 token 必须被拒。

    这条守的是 `jwt.decode(..., algorithms=["HS256"])` 里那个**显式**列表 ——
    去掉它 PyJWT 会抛(它自己的防混淆机制),写错成 `algorithms=None` 才是真漏洞。
    """
    forged = jwt.encode({"sub": "cinfly", "role": ADMIN, "exp": int(time.time()) + 3600},
                        key="", algorithm="none")
    with pytest.raises(AuthError):
        decode_token(forged, _settings())


def test_token_without_sub_is_rejected():
    """**缺 `sub`** 的 token 不许当成匿名用户放过去。

    ⚠️ **两条 claim 必须分开测**(计划初稿把 `sub` 与 `role` **一起省掉** ——
    那样**删掉任何一条守卫它都照样绿**,本仓「假绿形态」里最经典的一种)。
    这里只**少给 `sub`**,`role` 给对的。
    """
    bare = jwt.encode({"role": ADMIN, "exp": int(time.time()) + 3600},
                      SECRET, algorithm="HS256")
    with pytest.raises(AuthError):
        decode_token(bare, _settings())


def test_token_with_an_unknown_role_is_rejected():
    """**角色不认识**的 token 也不许放过去(只少给 `role` 那一半)。"""
    bare = jwt.encode({"sub": "cinfly", "role": "root", "exp": int(time.time()) + 3600},
                      SECRET, algorithm="HS256")
    with pytest.raises(AuthError):
        decode_token(bare, _settings())


def test_empty_secret_generates_a_random_one_and_still_round_trips():
    """`.env` 没配时:同一个进程内签发/校验仍然自洽(重启后失效是**设计**)。

    ⚠️ `jwt_secret=""` **必须显式传**:`_env_file=None` 只关掉 `.env` 这个**文件**,
    pydantic-settings **照读 `os.environ`** —— 跑测试的 shell 里一旦 export 过
    `JWT_SECRET`,这条就会拿到那个值,而它断的正是「空配置」那条路。
    """
    s = Settings(_env_file=None, jwt_secret="", **_REQUIRED)
    assert s.jwt_secret == ""
    u = decode_token(create_token(username="demo-user", role=ADMIN, settings=s), s)
    assert u.username == "demo-user"


def test_require_admin_rejects_a_plain_user():
    with pytest.raises(Exception) as ei:
        require_admin(AuthenticatedUser(username="bob", role=USER))
    assert getattr(ei.value, "status_code", None) == 403, (
        "非 admin 打工作台必须是 403(不是 401 —— 前端只对 401 弹登录)")


def test_require_user_has_no_header_and_raises_401():
    """**缺 header 必须是 401**,且带 `WWW-Authenticate`。

    ⚠️ 这条是 Task 5 那条结构性测试的**行为**对应物:前者断「依赖挂上了」,
    这条断「挂上之后行为对」。
    """
    with pytest.raises(Exception) as ei:
        require_user(None, _settings())
    assert getattr(ei.value, "status_code", None) == 401, "缺 header 必须 401"
    assert ei.value.headers.get("WWW-Authenticate") == "Bearer"


def test_require_user_rejects_a_garbage_token_with_401_not_500():
    from fastapi.security import HTTPAuthorizationCredentials
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="not-a-jwt")
    with pytest.raises(Exception) as ei:
        require_user(creds, _settings())
    assert getattr(ei.value, "status_code", None) == 401
```

- [ ] **Step 3: 跑测试,确认全红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_auth.py -p no:cacheprovider`
Expected: **collection error**(`ModuleNotFoundError: No module named 'app.auth'`)—— 那是"没实现"的红。

- [ ] **Step 4: 实现 `app/auth.py`**

```python
"""认证(2026-09-27):**全仓唯一**的鉴权边界。

写法对齐 `app/observability.py`(本项目「一章/一个功能的边界只留一个文件」的既有形状):
别的模块只认下面这几个名字,换方案(换库、换成 OAuth)改的是**这一个文件**。

    AuthError / AuthenticatedUser / USER / ADMIN
    hash_password / verify_password / create_token / decode_token
    require_user / require_admin

## 用户名与角色**放在 token 里**,`require_user` 不查库

token 的 claims 是 `sub`(用户名)/ `role` / `iat` / `exp`。⇒ 每个受保护的请求
**零次数据库往返**,鉴权只剩一次 HMAC 计算。

代价如实记账:**改了某个账号的 role,旧 token 到过期前仍带着旧 role**
(本功能没有刷新、没有吊销、没有登出黑名单 —— spec §2 明确不做)。
对本项目这完全够用:`users` 表只在登录那一刻被读一次。

## 密钥从哪来

`settings.jwt_secret` 为空 ⇒ 首次用到时**随机生成一个**(进程级)+ 一条 WARNING。
那样「clone 下来不配 .env 也能跑」,而**密钥不进源码**。本机 `.env` 里是
用户拍板的 `itcinfly` —— ⚠️ 那个值同时写在**入库的** `.env.example` 里,
所以它是**公开值**,只能本机演示用。

## 为什么 401 自己抛、不用 `HTTPBearer` 的 auto_error

本机实测(fastapi 0.141.1):`HTTPBearer()` 的 auto_error 给的是
**401 + `{"detail":"Not authenticated"}`**,不是旧版的 403。所以理由不是状态码,
而是两条:① 文案要中文(`未认证`,与全仓其余错误一致);② **「缺 header」与
「token 坏了/过期了」必须走同一个出口** —— `HTTPBearer` 那个 scheme **从不校验
token 内容**(实测垃圾串照样放行),内容校验无论如何都得在这里做,
让两件事分居两处就会漂移。
"""

import base64
import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: 两个角色的字面量。`users.role` 列里存的就是这两个值。
USER = "user"
ADMIN = "admin"

#: scrypt 参数。`n=2**14 / r=8` ⇒ 单次约 16MB 内存、本机几毫秒 ——
#: 对「登录」这种低频动作是合适的量级(它不是每次请求都跑)。
_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_DKLEN = 32


class AuthError(Exception):
    """鉴权失败的**唯一**异常类型。

    调用方只认它 —— 于是「PyJWT 抛了七八种异常」这件事被关在**这一个文件**里,
    端点层不用去 import `jwt` 的任何异常类。
    """


@dataclass(frozen=True)
class AuthenticatedUser:
    """当前登录用户。**不返回 ORM 对象**:依赖不该把一个活着的 `AsyncSession`
    绑到一个跨函数的返回值上,而 `frozen` 让「端点悄悄改了自己的身份」不可能。"""

    username: str
    role: str


# --------------------------------------------------------------------------
# 密码
# --------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """随机盐 + scrypt,**自描述串** `scrypt$n$r$p$<b64盐>$<b64哈希>`。

    自描述的意思:参数跟着这条记录走 ⇒ 将来调大 `n` 时**旧密码照样验得过**
    (解析串里那两个数,而不是拿当前常量去算)。
    """
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_DKLEN)
    return "$".join([
        "scrypt", str(_SCRYPT_N), str(_SCRYPT_R), str(_SCRYPT_P),
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    ])


def verify_password(password: str, stored: str) -> bool:
    """校验密码。**任何异常都回 False** —— 见 `tests/test_auth.py` 里那条用例。

    常数时间比较(`hmac.compare_digest`):`==` 会在第一个不同字节处短路,
    理论上能从响应时间里逐字节试出哈希。
    """
    try:
        scheme, n, r, p, b64_salt, b64_dk = stored.split("$")
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode("utf-8"),
                            salt=base64.b64decode(b64_salt),
                            n=int(n), r=int(r), p=int(p), dklen=len(base64.b64decode(b64_dk)))
    except Exception:
        return False
    return hmac.compare_digest(dk, base64.b64decode(b64_dk))


# --------------------------------------------------------------------------
# token
# --------------------------------------------------------------------------

def _secret(settings: Settings) -> str:
    """签名密钥。配置为空时**生一个进程级的随机密钥**并**只警告一次**。"""
    if settings.jwt_secret:
        return settings.jwt_secret
    global _EPHEMERAL_SECRET
    if _EPHEMERAL_SECRET is None:
        _EPHEMERAL_SECRET = secrets.token_urlsafe(32)
        logger.warning(
            "JWT_SECRET 没配 ⇒ 本次进程用一个**随机密钥**。"
            "服务重启后所有已发出的 token 立即失效(用户会看到 401 并重新登录)。"
            "要固定下来就在 .env 里设 JWT_SECRET。"
        )
    return _EPHEMERAL_SECRET


#: 空配置下的进程级随机密钥。**只在这里赋值**(`_secret`)。
_EPHEMERAL_SECRET: str | None = None


def create_token(*, username: str, role: str, settings: Settings) -> str:
    """签一个 HS256 token。`exp` 由 `jwt_expire_minutes` 算。"""
    now = datetime.now(tz=timezone.utc)
    return jwt.encode(
        {
            "sub": username,
            "role": role,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=settings.jwt_expire_minutes)).timestamp()),
        },
        _secret(settings),
        algorithm="HS256",
    )


def decode_token(token: str, settings: Settings) -> AuthenticatedUser:
    """解 token;**任何问题一律 `AuthError`**(过期 / 签名错 / 缺 claim / 形状不对)。

    ⚠️ `algorithms=["HS256"]` **必须显式给**:那是 PyJWT 防 alg 混淆的机制
    (不给它会抛,给了 `None` 才是漏洞)。所以 `alg: none` 的伪造 token 在这里被拒。
    """
    try:
        payload = jwt.decode(token, _secret(settings), algorithms=["HS256"])
    except jwt.InvalidTokenError as exc:      # 过期/签名错/形状不对,**共同基类**
        raise AuthError("token 无效") from exc
    username, role = payload.get("sub"), payload.get("role")
    if not isinstance(username, str) or not username:
        raise AuthError("token 缺少 sub")
    if role not in (USER, ADMIN):
        raise AuthError("token 的角色不认识")
    return AuthenticatedUser(username=username, role=role)


# --------------------------------------------------------------------------
# FastAPI 依赖
# --------------------------------------------------------------------------

#: `auto_error=False` ⇒ 缺 header 时**回 None 而不是自己抛** —— 我们要自己抛,
#: 好让「缺 header」与「token 坏了」共用同一个 401 出口(见模块 docstring)。
_bearer = HTTPBearer(auto_error=False, description="Bearer <登录拿到的 token>")


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401, detail="未认证", headers={"WWW-Authenticate": "Bearer"}
    )


def require_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AuthenticatedUser:
    """**任何登录用户**。没带 / 坏了 / 过期 ⇒ 401。"""
    if creds is None or not creds.credentials:
        raise _unauthorized()
    try:
        return decode_token(creds.credentials, settings)
    except AuthError as exc:
        raise _unauthorized() from exc


def require_admin(
    user: Annotated[AuthenticatedUser, Depends(require_user)],
) -> AuthenticatedUser:
    """**工作台**用:`role == admin`。是合法用户但角色不够 ⇒ **403**。

    401 与 403 对前端是两件事:401 ⇒ 去登录;403 ⇒ 登录了但没权限(不弹登录)。
    """
    if user.role != ADMIN:
        raise HTTPException(status_code=403, detail="权限不足")
    return user
```

- [ ] **Step 5: 跑测试,确认全绿**

Run: `.venv/Scripts/python.exe -m pytest tests/test_auth.py -p no:cacheprovider`
Expected: **13 passed**。

⚠️ 这个数**改过两次**,过程记在这里免得后来人对不上账:
初稿写「13」是**错的**(那时 Step 2 里只有 12 个 `def test_…`);
实现者第一轮实测 **12** 并报了上来;随后 `test_token_without_sub_or_role_is_rejected`
**拆成两条**(见那条的 ⚠️)才真的是 **13**。
⇒ **以实际为准**,把真实数字写进报告;**对不上时别改测试去凑数**,先数一遍。

⚠️ **输出里有 8 条 `InsecureKeyLengthWarning`(PyJWT ≥2.14)是预期的、不要消掉**:
RFC 7518 要求 HS256 密钥 ≥32 字节,而用户拍板的演示密钥 `itcinfly` 是 8 字节。
**这条警告在真实运行时每个请求都有**(不是测试专属)。**不 filter、不 lengthen** ——
filter 掉等于把一条安全提醒静音,而换密钥违背用户点名的选型。
代价与回退写在 spec §9:真部署时 `.env` 里换成 32 字节随机值即可。

- [ ] **Step 6: 跑一条真机(`alg:none` 那条依赖 PyJWT 的行为)**

Run: `.venv/Scripts/python.exe -c "import jwt; print(jwt.encode({'sub':'x','role':'admin'}, key='', algorithm='none'))"`
Expected: 打出一个 `xxx.yyy.` 形状的串(空签名)。若 PyJWT 直接抛,把该用例改成
`pytest.raises(AuthError)` 包一层手工构造的串,并在注释里写明**实测日期与库版本**。

- [ ] **Step 7: 提交**

```bash
git add app/auth.py app/config.py requirements.txt tests/test_auth.py
git commit -m "认证 T1:鉴权内核 app/auth.py(唯一边界)+ 配置两项 + 13 条内核测试"
```

---

## Task 2: `users` 表 + 种子账号

**Files:**
- Create: `db/auth.sql`, `scripts/seed_users.py`
- Test: `tests/test_seed_users_db.py`(新,**带 `pytest.mark.db`**)

**Interfaces:**
- Consumes: Task 1 的 `hash_password`
- Produces: `scripts/seed_users.py` 的 `seed(session) -> int`(返回写入条数);
  两个账号 `cinfly` / `demo-user`,密码都 `123456`,`role` 都 `admin`

- [ ] **Step 1: 写 `db/auth.sql`**

```sql
-- 认证(2026-09-27):login 用的账号表。
--
-- ⚠️ **刻意不幂等**(与 db/ch03/04/06/07/08/ch09 同规矩):重复执行会在
--    CREATE TABLE 上响亮地报 1050。那是有意的 —— 静默跳过会让
--    「表已存在但形状不对」永远补不上。
--
-- ⚠️ **ORM 侧有同名模型**(`app/db/models.py` 的 `User`,本任务 Step 4 加的)⇒
--    `init_db.py` 的 create_all **会把这张表建出来**(它建「不存在的表」)——
--    ⇒ **正常路径只需要 `init_db.py`,不需要跑这一份**;在表已存在时跑它会在
--    那条 CREATE 上响亮地报 **1050**。**那是刻意的,不是脏库** ——
--    与 `db/ch10.sql` 的 `topic_classifications`、`db/ch08.sql` 的 `tool_audit_logs`
--    是**同一个**已知取舍(本仓既有的三处同族;写法与理由见 CLAUDE.md 的建库一段)。
--    想让**这份 DDL 成为形状的权威**就先 `DROP TABLE users;` 再跑一遍,
--    然后用 `SHOW CREATE TABLE users\G` 核对。
--
-- ⚠️ 两条路径的形状差异**只有文本**(纯注释与列序):列定义两边逐字一致。
--    这份文件的价值是「形状肉眼可读 + 有 COMMENT」,不是「建表的那一步」。
--
-- ⚠️ **不给 `conversations.user` 加外键**(spec §5.1 的取舍):那一列已有
--    581 行值、宽度 varchar(128),加 FK 要一条迁移,而收益只是「写错的 user
--    被数据库拦住」—— 写入方只有一个端点,且值来自 token。如实记账。
--
-- 账号本身**不在这份文件里**:scrypt 的盐是随机的,写进 SQL 就得把某一轮的盐
-- 焊死在文件里,且改密码要人来重算。见 `scripts/seed_users.py`(幂等、可重跑)。

-- ⚠️ `SET NAMES utf8mb4;` **不能少** —— `db/` 下**其余 8 份 DDL 全都带它**,
--    而本机 locale 是 cp936:不走它的话,从 cp936 客户端执行时下面那条
--    `COMMENT='登录账号…'` 里的中文会被**按 cp936 重编码**(本仓 cp936 家族的第 N 次)。
SET NAMES utf8mb4;

CREATE TABLE users (
  id            BIGINT       NOT NULL AUTO_INCREMENT,
  username      VARCHAR(128) NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  role          VARCHAR(16)  NOT NULL,
  created_at    DATETIME     NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uk_users_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
  COMMENT='登录账号(认证功能,2026-09-27)';
```

- [ ] **Step 2: 确认 MySQL 在跑(⚠️ **本步不建表**)**

⚠️ **建表要等 Step 4 之后** —— `create_all` 得先在 `app/db/models.py` 里看到
`User` 模型才知道有这张表。这里只确认库是活的:

Run: `docker ps --format '{{.Names}}' | grep mysql`
Expected: 打出 `mysql`。

> **建表 = Step 5 末尾跑一次 `init_db.py` + `SHOW CREATE TABLE users\G` 核对。**
> ⚠️ **不要**跑 `db/auth.sql`:表已存在时它会在 CREATE 上报 **1050** ——
> **那是刻意的、不是脏库**(与 `db/ch08.sql` / `db/ch10.sql` 同一个已知取舍,
> 理由写在那份 SQL 的头部注释里)。

- [ ] **Step 3: 写失败测试 `tests/test_seed_users_db.py`**

```python
"""种子账号(db)。**要 MySQL 起着**。

断三件事:两个账号真的能建出来 / **重跑不追加行** /
建出来的哈希**真的能验过 123456**(否则「账号建了但登不上」)。

⚠️ **本仓没有 `session` fixture** —— db 测试的既有形状是**直接
`get_sessionmaker()()`**,照 `tests/test_api_conversations_db.py` 那份写。
⚠️ **这两个账号是「生产数据」,不是探针**:所以**不删它们**
(删了本机就登不上了,而 re-run 一次 `seed_users.py` 才是正确的恢复动作)。
断言也因此**不写 `len(rows) == 2`**(将来多一个账号就假红),改成「两次 seed 之间
总数不变」。
"""

import pytest
from sqlalchemy import select

from app.auth import ADMIN, verify_password
from app.db.base import get_engine, get_sessionmaker
from app.db.models import User
from scripts.seed_users import ACCOUNTS, seed

pytestmark = pytest.mark.db

NAMES = [u for u, _, _ in ACCOUNTS]


async def _seed_now() -> int:
    async with get_sessionmaker()() as session:
        return await seed(session)


async def _rows() -> dict[str, User]:
    """**新 session** 读:身份映射里的旧对象会让断言变成「靠 refcount 走运」。"""
    async with get_sessionmaker()() as session:
        return {
            r.username: r
            for r in (await session.execute(select(User))).scalars().all()
        }


@pytest.mark.anyio
async def test_seed_creates_both_accounts_and_does_not_append_on_rerun():
    try:
        await _seed_now()
        first = await _rows()
        assert set(NAMES) <= set(first), f"两个账号都该在,实际 {sorted(first)}"

        await _seed_now()                       # 再跑一遍
        second = await _rows()
        assert len(second) == len(first), (
            f"重跑让行数从 {len(first)} 变成 {len(second)} ⇒ 不是幂等"
        )
        assert all(second[u].password_hash != first[u].password_hash for u in NAMES), (
            "两次哈希相同 ⇒ 种子脚本没重算盐(覆盖没真的发生)"
        )
    finally:
        await get_engine().dispose()            # 本仓 db 文件的既有收尾方式


@pytest.mark.anyio
async def test_seeded_passwords_verify_and_roles_are_admin():
    try:
        await _seed_now()
        rows = await _rows()
        for username, password, role in ACCOUNTS:
            assert verify_password(password, rows[username].password_hash), (
                f"{username} 的密码验不过 ⇒ 账号建了但登不上"
            )
            assert rows[username].role == role == ADMIN
    finally:
        await get_engine().dispose()
```

> ⚠️ **不要**为了「收尾」再写一条 `test_engine_disposes_cleanly`(计划初稿里有,
> **已删**):那条**不断言任何东西** —— 正是复审 rubric 要拦的「空测试」。
> 收尾照 `tests/test_api_conversations_db.py` 的既有形状:每个用例 `finally` 里 dispose。

- [ ] **Step 4: 加 ORM 模型 `User`(`app/db/models.py`)**

在文件末尾(照该文件既有风格,逐列带 `comment=`):

```python
class User(Base):
    """登录账号(认证功能,2026-09-27)。

    ⚠️ **`init_db.py` 的 create_all 会建出这张表**(因为本模型存在)——
    所以**正常路径只需要 `init_db.py`**,不需要跑 `db/auth.sql`(表已在时跑它会
    1050,**那是刻意的**,见那份 DDL 的头部注释)。两份文件**列定义逐字一致**。
    列宽与 `conversations.user` 同为 128 —— 两边不一致的话,账号名能写进 token
    却写不进会话,报错指向 DataError。
    """

    __tablename__ = "users"
    # ⚠️ **唯一键要具名**,不能用 `unique=True`:后者生成的键名是 `username`,
    # 而 `db/auth.sql` 里写的是 `uk_users_username` ⇒ 两条建表路径会**多出一处
    # 名字差异**(本仓 ch08 的 `idx_conv` vs `ix_tool_audit_logs_conversation_id`
    # 就是同一族)。具名之后差异只剩「表 COMMENT」一处,与既有那几份同款。
    __table_args__ = (UniqueConstraint("username", name="uk_users_username"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(128), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False,
                                                 default=datetime.now)
```

> 若 `app/db/models.py` 里 `BigInteger` / `DateTime` 还没 import,补进那一行 import。

- [ ] **Step 5: 实现 `scripts/seed_users.py`**

```python
"""预置两个演示账号(**幂等**,可重跑)。

用户 2026-09-27 拍板:两个账号都 `admin`(既能聊天也能看工作台),
密码都是 `123456`,**不做注册功能**。

    .venv/Scripts/python.exe scripts/seed_users.py

⚠️ **明文密码写在这个文件里是刻意的**:它是用户点名的演示凭据,而
`.env.example`(入库)里已经印着同一组账号密码 —— 藏在一个脚本里毫无意义。
`hash_password` 每次生成**新的随机盐** ⇒ 重跑 = 覆盖密码哈希,不是追加。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select                      # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession    # noqa: E402

from app.auth import ADMIN, hash_password          # noqa: E402
from app.db.base import get_engine, get_sessionmaker  # noqa: E402
from app.db.models import User                     # noqa: E402

#: (用户名, 密码, 角色)。**顺序即输出顺序**,测试按它断。
ACCOUNTS = [
    ("cinfly", "123456", ADMIN),
    ("demo-user", "123456", ADMIN),
]


async def seed(session: AsyncSession) -> int:
    """把 `ACCOUNTS` 写进去(按 username upsert),返回处理条数。"""
    for username, password, role in ACCOUNTS:
        row = (await session.execute(
            select(User).where(User.username == username))).scalar_one_or_none()
        if row is None:
            session.add(User(username=username, role=role,
                             password_hash=hash_password(password)))
        else:
            row.role = role
            row.password_hash = hash_password(password)   # 重算 ⇒ 每次盐都不同
    await session.commit()
    return len(ACCOUNTS)


async def main() -> None:
    factory = get_sessionmaker()
    async with factory() as session:
        n = await seed(session)
    print(f"已写入 {n} 个账号:" + "、".join(u for u, _, _ in ACCOUNTS))
    await get_engine().dispose()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 6: 建表 + 跑测试**

```bash
.venv/Scripts/python.exe scripts/init_db.py      # create_all:把 missing 的 users 建出来
.venv/Scripts/python.exe -m pytest tests/test_seed_users_db.py -p no:cacheprovider
```
Expected: 打出「2 passed」(**要 MySQL**;真实数字以实际为准)。
再用客户端核形状:`SHOW CREATE TABLE users\G` ⇒ 应出现 `uk_users_username` 唯一键
与上面五列。

- [ ] **Step 7: 真跑一次种子脚本并核对库**

Run:
```bash
.venv/Scripts/python.exe scripts/seed_users.py
.venv/Scripts/python.exe -c "import asyncio;from sqlalchemy import text;from app.db.base import get_engine;
async def m():
 e=get_engine()
 async with e.connect() as c:
  print((await c.execute(text('SELECT username,role,LEFT(password_hash,7) FROM users'))).all())
 await e.dispose()
asyncio.run(m())"
```
Expected: 两行,`role` 都是 `admin`,`password_hash` 前缀是 `scrypt$`。

- [ ] **Step 8: 提交**

```bash
git add db/auth.sql scripts/seed_users.py app/db/models.py tests/test_seed_users_db.py
git commit -m "认证 T2:users 表(db/auth.sql)+ 幂等种子脚本 + db 测试"
```

---

## Task 3: 登录端点

**Files:**
- Create: `app/api/auth.py`
- Modify: `app/schemas.py`(追加两个模型)、`app/main.py`(挂路由)
- Test: `tests/test_api_auth.py`

**Interfaces:**
- Consumes: Task 1 的 `verify_password` / `create_token` / `require_user` / `ADMIN`;Task 2 的 `User`
- Produces: `POST /api/auth/login`(**公开**)、`GET /api/auth/me`;
  响应体 `{"token": str, "username": str, "role": str, "expires_at": str}`
  —— 前端 Task 6 按这四个键读

- [ ] **Step 1: 写失败测试 `tests/test_api_auth.py`**

```python
"""登录端点的行为面。**要 MySQL**(它要读 users 表)。

⚠️ 这一份**刻意请求** `_no_default_login`(Task 5 定义的 autouse 开关)——
它测的就是「没登录时会怎样」,而那个全局装置默认替每条用例装一个已登录用户。
⚠️ **本仓没有 `session` fixture**:种账号用**直接 engine**,照
`tests/test_api_conversations_db.py` 既有形状写。
"""

import pytest
from fastapi.testclient import TestClient

from app.auth import ADMIN, create_token
from app.config import Settings
from app.main import app
from app.db.base import get_engine, get_sessionmaker
from scripts.seed_users import seed

pytestmark = pytest.mark.db

GOOD = {"username": "cinfly", "password": "123456"}


@pytest.fixture
def client():
    return TestClient(app)


async def _seed_now() -> None:
    async with get_sessionmaker()() as session:
        await seed(session)


@pytest.mark.anyio
async def test_login_returns_a_token_that_me_accepts(client, _no_default_login):
    await _seed_now()
    r = client.post("/api/auth/login", json=GOOD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"token", "username", "role", "expires_at"}, body
    assert body["username"] == "cinfly" and body["role"] == ADMIN

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {body['token']}"})
    assert me.status_code == 200, me.text
    assert me.json() == {"username": "cinfly", "role": ADMIN}
    await get_engine().dispose()


@pytest.mark.anyio
async def test_wrong_password_and_unknown_user_are_indistinguishable(client, _no_default_login):
    """两种失败必须**逐字相同** —— 否则响应文案成了「用户名枚举」的接口。"""
    await _seed_now()
    a = client.post("/api/auth/login", json={"username": "cinfly", "password": "nope"})
    b = client.post("/api/auth/login", json={"username": "nobody", "password": "nope"})
    assert a.status_code == b.status_code == 401
    assert a.json() == b.json(), f"{a.json()} != {b.json()}"
    await get_engine().dispose()


def test_login_body_is_validated(client, _no_default_login):
    """缺 password ⇒ 422(pydantic 在碰库**之前**就拒了)。"""
    assert client.post("/api/auth/login", json={"username": "x"}).status_code == 422


def test_me_without_token_is_401_and_not_403(client, _no_default_login):
    r = client.get("/api/auth/me")
    assert r.status_code == 401, f"缺 header 必须 401(不是 403),实际 {r.status_code}"


def test_me_with_a_garbage_token_is_401(client, _no_default_login):
    """坏 token 与缺 header 必须是**同一个** 401(前端只认这一条去弹登录)。"""
    a = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
    b = client.get("/api/auth/me")
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
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401
    await get_engine().dispose()
```

- [ ] **Step 2: 跑测试,确认全红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_auth.py -p no:cacheprovider`
Expected: FAIL —— `404`(路由还没挂)与 fixture `_no_default_login` 不存在。

- [ ] **Step 3: `app/schemas.py` 追加两个模型**

```python
class LoginRequest(BaseModel):
    """登录请求。

    两个字段都**只做长度**校验:密码**不设** min_length —— 密码策略是产品决定,
    而这里多一条校验会让「旧账号的短密码登录被 422 拒掉」,报错还指向参数形状。
    """

    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    """登录响应。`expires_at` 是 ISO 串(前端拿它显示"什么时候要重新登录")。"""

    token: str
    username: str
    role: str
    expires_at: str
```

- [ ] **Step 4: 实现 `app/api/auth.py`**

```python
"""登录与「我是谁」两个端点(认证功能,2026-09-27)。

`/api/auth/login` 是**唯一**不需要 token 的端点 —— 它自己就是拿 token 的地方。
`/api/auth/me` 挂在 `require_user` 上:它存在的价值是**前端启动时验一下手里的
token 还有效没**(有效就跳过登录浮层)。
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import AuthenticatedUser, create_token, require_user, verify_password
from app.config import Settings, get_settings
from app.db.models import User
from app.db.session import get_session
from app.schemas import LoginRequest, TokenResponse

router = APIRouter(tags=["auth"])

#: 登录失败的**唯一**文案。两种情况(查无此人 / 密码不对)共用它 ——
#: 分开写等于把「这个用户名存不存在」做成一个可枚举的接口。
_BAD_CREDENTIALS = "用户名或密码不正确"


@router.post("/api/auth/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    row = (await session.execute(
        select(User).where(User.username == body.username)
    )).scalar_one_or_none()

    # ⚠️ **查无此人时也要跑一次 `verify_password`** —— 提前 return 会让
    # 「用户不存在」比「密码错」快一个数量级,响应时间就成了枚举用户名的侧信道。
    # 用 `verify_password(body.password, row.password_hash if row else "")` 即可
    # (坏串回 False,见 tests/test_auth.py 那条)。
    if not verify_password(body.password, row.password_hash if row else ""):
        raise HTTPException(status_code=401, detail=_BAD_CREDENTIALS)

    token = create_token(username=row.username, role=row.role, settings=settings)
    expires = datetime.now(tz=timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    return TokenResponse(token=token, username=row.username, role=row.role,
                         expires_at=expires.isoformat())


@router.get("/api/auth/me")
async def me(user: Annotated[AuthenticatedUser, Depends(require_user)]):
    """回当前用户。**只看 token、不查库** —— 与 `require_user` 同一条路径。"""
    return {"username": user.username, "role": user.role}
```

- [ ] **Step 5: `app/main.py` 挂路由**

在 `app.include_router(chat_router)` **之前**插入两行(顺序只影响 `/docs` 里的分组):

```python
# 认证(2026-09-27):登录是**公开**端点,它自己不带守卫 —— 守卫挂在它的
# 内部路径上(`/api/auth/me` 走 require_user)。同样**必须在 `mount("/")`
# 之前**(`tests/test_topics_boundary.py` 有一条按注册顺序断的用例)。
app.include_router(auth_router)
```

并在文件顶部的 import 区加上 `from app.api.auth import router as auth_router`。

- [ ] **Step 6: 跑测试**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_auth.py -p no:cacheprovider`
Expected: **6 passed**(需要 MySQL)。⚠️ 计划初稿在这里写过「5」—— **那是错的**
(Step 1 里一共 **6** 个 `def test_…`;我后来加了「坏 token」那条却没改这个数)。
**以实际为准**并把真实数字写进报告。

⚠️ **`_no_default_login` 由本任务定义**(裁定 R1):Step 1 的测试**请求**它,
不定义的话整个文件在 collection 阶段就 error。在 `tests/conftest.py` 里加:

```python
@pytest.fixture
def _no_default_login():
    """**关掉**「默认已登录」装置的开关。

    只有认证自己的测试请求它 —— 那几条测的就是「没登录时会怎样」,
    被默认装置一盖,断言的「401」会变成 200 而**红得莫名其妙**。

    ⚠️ 这里只是**占个名字**:真正的开关在 Task 5 加的 autouse 装置里,
    它靠 `"_no_default_login" in request.fixturenames` 认这个名字。
    ⇒ **本任务不许改它的形状**(名字即契约),Task 5 才加那个装置。
    """
    return None
```

- [ ] **Step 7: 真机冒烟(服务重启 + curl)**

```bash
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000 &   # 起服务
sleep 8
curl -s -X POST http://127.0.0.1:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  --data-binary '{"username":"cinfly","password":"123456"}'
```
Expected: 含 `"token"` 的 JSON。再拿它打 `/api/auth/me` ⇒ 200。
⚠️ **含中文的 body 不走 argv**(本仓规矩);这里的 body 是纯 ASCII,安全。

- [ ] **Step 8: 提交**

```bash
git add app/api/auth.py app/schemas.py app/main.py tests/test_api_auth.py tests/conftest.py
git commit -m "认证 T3:POST /api/auth/login + GET /api/auth/me + 5 条端点测试"
```

---

## Task 4: 身份来自 token + 会话归属

**Files:**
- Modify: `app/schemas.py`(删 `ChatRequest.user_id`)、`app/api/chat.py`(`/api/ticket` 与 `/api/chat/stream`)、
  `app/api/refund.py`、`app/api/conversations.py`、`app/services/history.py`
- Test: `tests/test_api_conversations_db.py`(改 helper + 补一条)、`tests/test_api_chat.py`(三条 `user_id` 用例改写)

**Interfaces:**
- Consumes: Task 1 的 `AuthenticatedUser` / `require_user`
- Produces:
  - `app/services/history.py:get_owned_conversation(*, session, conversation_id, user_id) -> Conversation | None`
  - ⚠️ **两个既有函数签名要变**(它们的调用方不止端点):
    - `list_conversations(*, user: AuthenticatedUser, session)` —— 原来是 `(session)`
    - `list_messages(conversation_id, *, user: AuthenticatedUser, session)` —— 原来是 `(conversation_id, session)`

> ⚠️ **先读这两个文件再动手**(本仓那条「引用一个文件就得当场核」):
> `tests/test_api_conversations_db.py` 的 `_list_items()` / `_messages()` 是
> **直接调这两个函数**的(不是经 HTTP),签名一变它们**全红**;
> `tests/test_api_chat.py` 有**三条**关于 `user_id` 的用例。

- [ ] **Step 1: 先写要红的测试(归属)**

在 `tests/test_api_conversations_db.py` 里追加(**照该文件既有的 module 级 helper 风格**,
不要新造 fixture —— 那份文件用的是 `_conv` / `_msg` / `_insert` / `_cleanup` 这套):

```python
@pytest.mark.anyio
async def test_a_foreign_conversation_is_404_on_messages_not_403():
    """**别人的会话明细必须像不存在一样**(404)。

    ⚠️ 回 403 会把「这个 id 存在」告诉对方 —— 那就成了一个可枚举的接口。
    这条与 `test_list_excludes_other_users_on_a_real_table`(**列表**那一半)
    是一对:列表管「看不见」,这条管「点不进去」。
    """
    await _cleanup()
    try:
        await _insert([
            _conv(PROBE_FOREIGN, "someone-else", T_FOREIGN),
            _msg(PROBE_FOREIGN, "user", "别人的一句话"),
        ])
        mine = AuthenticatedUser(username="demo-user", role=ADMIN)
        with pytest.raises(HTTPException) as ei:
            await list_messages(conversation_id=PROBE_FOREIGN, session=_sess(), user=mine)
        assert ei.value.status_code == 404, (
            f"别人的会话必须 404(不是 {ei.value.status_code} —— 403 等于承认它存在)")
    finally:
        await _cleanup()
        await get_engine().dispose()
```

> `_sess()` 是个一行的 async helper(照 `_list_items` 里那句
> `async with get_sessionmaker()() as session` 的形状),或者直接把那句话抄进来。

- [ ] **Step 2: 跑它,确认红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_api_conversations_db.py -p no:cacheprovider`
Expected: FAIL(TypeError —— `list_messages` 还不接受 `user=`,且它今天**根本不查归属**)。

- [ ] **Step 3: 删 `ChatRequest.user_id` 并把那段 docstring 挪走**

`app/schemas.py`:删掉 `user_id: str | None = Field(...)` 那一行,把 docstring 里
那段「`user_id` 的上限 128 与 `conversations.user` 的列宽一致」**改写成**:

```
    ---- ch10 跟进(认证,2026-09-27)----

    `user_id` **已删除**:身份从 token 来(`app/auth.py` 的 `require_user`),
    客户端再也不能自称是谁。「上限 128 与 `conversations.user` 的列宽一致」
    那条事实随字段一起挪进了 `app/db/models.py` 的 `User` docstring
    (那边的 `username` 同样是 128)。
```

⚠️ **`ChatRequest` 没有 `extra="forbid"`**(实测:`grep -n 'extra=' app/schemas.py` 为空)
⇒ 删字段之后请求体里**多传**的 `user_id` 会被 pydantic **静默忽略**,请求照旧 200。
**别把它当 bug**(那是 pydantic 的默认行为),但也**别以为它还会 422** ——
Step 7 的三条既有用例正是死在这一点上。

- [ ] **Step 4: 改三个写路径的 `user_id` 来源**

`app/api/chat.py` 两处(`/api/chat/stream` 与 `/api/ticket`)、`app/api/refund.py` 一处,
把 `user_id = request.user_id or "demo-user"` 与 `user_id="demo-user"` 换成**依赖注入**:

```python
    user: Annotated[AuthenticatedUser, Depends(require_user)],
    ...
    user_id = user.username
```

顶部补 `from app.auth import AuthenticatedUser, require_user`。⚠️ `/api/chat/stream`
是手工构造 `EventSourceResponse` 的普通 `async def` —— 依赖在**函数签名**上,
所以 401 会在流开始前返回(与那条「预算校验必须在流开始前」同一条规矩)。

- [ ] **Step 5: 加 `get_owned_conversation`,改两个读端点的签名**

`app/services/history.py` 追加:

```python
async def get_owned_conversation(*, session, conversation_id: str,
                                 user_id: str) -> Conversation | None:
    """取会话,**且必须是这个用户的**;不属于他 ⇒ 回 `None`。

    ⚠️ **「别人的会话」与「不存在的会话」必须是同一个答案**(端点都翻成 404):
    回 403 等于承认「这个 id 存在」,那就给了枚举的口子。它与
    `ensure_conversation` 那条「已存在时忽略传入的 user_id」的分工是:
    那边保的是**归属不被改写**,这边保的是**读不到别人的**。
    """
    return (await session.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.user == user_id)
    )).scalars().one_or_none()
```

`app/api/conversations.py`:
- `list_conversations(*, user: Annotated[AuthenticatedUser, Depends(require_user)],
  session: Annotated[AsyncSession, Depends(get_session)])`
  —— `.where(Conversation.user == DEMO_USER)` 改成 `== user.username`;
- `list_messages(conversation_id, *, user=Depends(require_user), session=Depends(get_session))`
  —— 打开函数体后**先**做归属,不通过就 404(文案照抄现有那条「会话不存在」):
  ```python
  conv = await get_owned_conversation(session=session, conversation_id=conversation_id,
                                      user_id=user.username)
  if conv is None:
      raise HTTPException(status_code=404, detail="会话不存在")
  ```
- **删掉 `DEMO_USER` 常量**及其那段「与会话端点 `request.user_id or "demo-user"`
  同一个字面量」的注释(本仓那条「别让本次改动把别处的注释变假」)。

- [ ] **Step 6: 跟改 `tests/test_api_conversations_db.py`**

- `_list_items()` → 收一个 `user` 参数(默认造 `AuthenticatedUser("demo-user", ADMIN)`);
- `_messages(conversation_id)` 同理;
- `test_list_excludes_other_users_on_a_real_table` **保留**(它的断言仍然成立),
  但改成显式传 user,并把 docstring 里「`WHERE user = 'demo-user'`」改成
  「`WHERE user = <当前登录用户>`」—— 那句现在**不再准确**(字面量没了)。

- [ ] **Step 7: 改写 `tests/test_api_chat.py` 的三条 `user_id` 用例**

| 原用例 | 怎么处理 |
|---|---|
| `test_user_id_wider_than_the_column_is_rejected_as_422`(`:1120`) | **删除**。字段没了 ⇒ 多传的键被静默忽略,那条 422 **不可能再出现**;留着它就会红,而红的原因不是缺陷。**删掉的理由写进提交信息。** |
| `test_chat_stream_accepts_optional_user_id`(`:1130`) | **改写成它的反面**:请求体里带 `user_id: "alice"`,断言落到 `conversations.user` 的是 **token 里的那个用户名**(`tests-default-user`),**不是** `alice`。名字改成 `test_token_user_wins_over_a_smuggled_user_id`。 |
| `test_user_id_defaults_to_demo_user`(`:1145` 附近) | **删除**(「缺省 demo-user」这件事没有了),理由同上。 |

改后的那条**断言方向是反的**,这正是它的价值:它钉的是「客户端**不能**自称是谁」。

- [ ] **Step 7b: 跟改 `tests/test_api_conversations.py`(⚠️ **11 处会同时断**,一处修完)**

那个文件是**替身**测试,**经 HTTP** 打端点 —— 实测有 **11 处**:
`c.get("/api/conversations")` **4 处**(`:276` / `:304` / `:321` / `:351`)
与 `c.get(f"/api/conversations/{CONV_A}/messages")` **7 处**
(`:386` / `:410` / `:445` / `:479` / `:509` / `:536` / `:566`)。

**它们为什么会断**:
- 列表那 4 处:端点改成按**当前登录用户**过滤,而默认装置给的是
  `tests-default-user` ⇒ 探针行(`user="demo-user"`)**一条都回不来**;
- 明细那 7 处:新增的归属检查 ⇒ `CONV_A` 不属于 `tests-default-user` ⇒ **404**。

**一处修完**:在 `conv_client` 那个装置的 `make()` 里显式覆盖
(探针行本来就是 `demo-user`,于是 11 处**一个字都不用动**):

```python
        app.dependency_overrides[get_session] = _override
        # 认证(2026-09-27):本文件的探针会话都属于 `demo-user`,而端点现在
        # **按当前登录用户**过滤/校验归属 ⇒ 这里显式让「当前登录用户」就是
        # `demo-user`。**刻意不依赖 conftest 那个全局默认装置** ——
        # 默认给的是 `tests-default-user`,与本文件的探针不是一个用户,
        # 而那种不一致会以「列表空 / 明细 404」的形式红,读起来像端点坏了。
        app.dependency_overrides[require_user] = lambda: AuthenticatedUser(
            username="demo-user", role=ADMIN)
        return TestClient(app)
```

⚠️ `conv_client` 的收尾是 `app.dependency_overrides.clear()` —— 它会把 conftest 那个
全局装置也一并清掉,而**卸载顺序是「后装的先拆」**(本装置比 autouse 后装 ⇒ 先拆),
autouse 那边随后 `pop` 两个已经不存在的键(no-op)⇒ **安全**,不用改。

⚠️ **名字与 docstring 要跟着改**:`test_list_filters_by_demo_user` 这个名字与它
docstring 里那句「(无认证,前端从来不传 `user_id`)」**当场变假** ——
改成「按**当前登录用户**过滤」,并把 `demo-user` 的来源说清楚(是**这个装置**指定的,
不是常量)。本仓那条「别让本次改动把别处的注释变假」。

- [ ] **Step 8: ⚠️ 写路径的归属:**如实记账,不在本任务做**

`ensure_conversation` 今天对**已存在**的行**忽略**传入的 user_id ⇒ 知道别人 32 位
会话 id 的人,能往那个会话里**写**一条消息(读不到,写完也读不到)。

**不做的理由**(写进 spec §12.2):改它要动 `ensure_conversation` 的返回契约
(端点要能分辨「不存在」与「别人的」并各自翻 404),牵动 3 个调用点;
而它需要先知道一个**不可猜**的 uuid4 hex。**用户要求的是「按用户查会话历史」(读路径)**。
⇒ 本任务只收严**读**路径,写路径作为**已知缺口**记账。

- [ ] **Step 8b: ⚠️ 在 `tests/conftest.py` 加「默认已登录」装置(**本任务必须做**)**

⚠️ **为什么它在 T4 而不是 T5**(计划初稿把它排在 T5,**那会让 T4 的树是红的**):
本任务给 `POST /api/chat/stream` / `/api/ticket` / `/api/refund` 与两个读端点的
**签名**加上了 `user: Annotated[AuthenticatedUser, Depends(require_user)]` ——
**依赖写在签名里就已经生效**,不必等 T5 挂 router 级守卫。
⇒ 没有下面这个装置的话,**每一个经 HTTP 打这些端点的既有用例都会 401**,
而 T4 的实现者会卡在「一片红,但红的原因不是我的改动」。

> ⚠️ `_no_default_login` **Task 3 已经定义过了**(`return None` 就够,那个开关只认
> **名字在不在**)⇒ **本步只加下面这一个**,不要再写一遍那个 —— 同名的 fixture
> 写两遍会让 pytest 在收集阶段直接报错。

```python
@pytest.fixture(autouse=True)
def _default_login(request):
    """**每条用例**默认带一个已登录的 admin。

    ## 为什么要有它

    认证一挂,`app/api/` 下 25 个操作全部要 token ⇒ 16 个既有测试文件里
    **约 79 处**端点调用会集体变红,而它们**测的都不是认证**。
    在**一处**注入默认身份,那 79 处一个字都不用改。

    ## 它与「假绿」的关系(必须配套 `tests/test_auth_wiring.py`)

    它让「这个端点受不受保护」在测试里**恒真** —— 少挂一个守卫照样全绿。
    那件事由 `tests/test_auth_wiring.py` **不经过本装置**地钉住。
    **两条必须同时在**,少一条就是本仓编目过的形态 ⑦。

    ## 为什么是 admin 而不是 user

    默认给 `user` 的话,工作台那 18 个端点在既有测试里会**集体 403**;
    而 admin 是**超集**(能打用户面也能打工作台)—— 与 spec §6.4 的语义一致。

    ## ⚠️ 开关为什么用 `request.fixturenames` 而**不是参数**

    写成 `def _default_login(_no_default_login)` 的话,那个名字**永远**在
    `fixturenames` 里 ⇒ 默认登录**永远**被跳过 ⇒ 那 79 处调用**集体 401**。
    (实测过:临时 conftest + 两条用例,只有这一版对。)
    """
    if "_no_default_login" in request.fixturenames:
        yield None
        return

    from app.auth import ADMIN, AuthenticatedUser, require_admin, require_user
    from app.main import app as fastapi_app

    fake = AuthenticatedUser(username="tests-default-user", role=ADMIN)
    fastapi_app.dependency_overrides[require_user] = lambda: fake
    fastapi_app.dependency_overrides[require_admin] = lambda: fake
    yield
    fastapi_app.dependency_overrides.pop(require_user, None)
    fastapi_app.dependency_overrides.pop(require_admin, None)
```

- [ ] **Step 8c: 跑一次**面广一点**的回归,确认「一片红」没发生**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_api_chat.py tests/test_api_refund.py tests/test_api_conversations.py tests/test_api_conversations_db.py tests/test_chat_service.py tests/test_history.py -p no:cacheprovider
```
Expected: 全绿。**红了先看是不是「请求体还带 `user_id`」或「探针用户名不是 `demo-user`」**
—— 那两类是 Step 7 / Step 7b 已点名要改的;其余的红要当**真缺陷**查。

- [ ] **Step 9: 跑相关测试**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_api_conversations.py tests/test_api_conversations_db.py tests/test_api_chat.py tests/test_api_refund.py tests/test_history.py -p no:cacheprovider
```
Expected: 全绿(含新加那条 404 用例;`test_history.py` 直接调 `ensure_conversation`,
本任务没改它的签名 ⇒ 应当**一条都不用动**)。

- [ ] **Step 10: 提交**

```bash
git add app/schemas.py app/api/chat.py app/api/refund.py app/api/conversations.py app/services/history.py tests/
git commit -m "认证 T4:身份来自 token(删 user_id,3 条旧用例改写)+ 归属收严读路径 + 写路径缺口记账"
```

---

## Task 5: 九个 router 挂守卫 + 防假绿的结构性测试

**Files:**
- Modify: 9 个 `app/api/*.py` 的 router 声明行;`tests/conftest.py`
- Create: `tests/test_auth_wiring.py`

**Interfaces:**
- Consumes: Task 1 的 `require_user` / `require_admin`
- Produces: 每个受保护端点的请求都需要 `Authorization`;测试默认已登录

- [ ] **Step 1: 先写结构性测试(它会立刻红)**

```python
"""接线:**每一个**端点都挂了守卫吗。

## 为什么非有这一条不可

`tests/conftest.py` 那个 autouse 的「默认已登录」装置让**整套测试**都不带 token
也能打端点 —— 那是为了让 79 处既有调用不用改。但它的副作用是:
**一个忘了挂 `require_user` 的 router 在整套测试里全绿**
(本仓编目过的形态 ⑦「替身替被测对象完成了语义」)。

⇒ 这条**刻意不经过那个装置**:它不请求任何 fixture,直接看
`app.routes` 里每条路由身上挂了什么。

## 白名单

只有 `POST /api/auth/login` —— 它是**拿 token 的地方**,自己不能要 token。
"""

from fastapi.routing import APIRoute

from app.auth import require_admin, require_user
from app.main import app

#: 唯一允许不带守卫的端点。**加一条都要在这里写明理由。**
PUBLIC = {("POST", "/api/auth/login")}

# ⚠️ 这里**不要**再留一个「只要求 require_user」的集合:plan 初稿写过 `USER_ONLY`,
# 而它是**死代码**(下面那条用例自己就把 `me` 点名了)。定义一个没人读的常量
# 正是复审要拦的那类东西。


def _guard_names(route: APIRoute) -> set[str]:
    names = set()
    for dep in route.dependencies:                # APIRouter(dependencies=[...])
        call = getattr(dep, "dependency", None)
        if call is not None:
            names.add(call.__name__)
    # 端点签名里那一个也要算(有些端点只把它写在参数里)
    for sub in getattr(route.dependant, "dependencies", []):
        names.add(sub.call.__name__)
    return names


def _iter_api_routes():
    """遍历**所有**真实端点(**递归** —— 理由见下面那条护栏的 docstring)。

    ⚠️ **fastapi 0.141.1 实测**(2026-09-27):`include_router` 往 `app.routes` 里放的是
    一个 `_IncludedRouter` **包装对象**,**不是** `APIRoute`。所以
    `for r in app.routes: if isinstance(r, APIRoute)` **一条都遍历不到**。
    真正的路由挂在包装的 `.original_router.routes` 上(再往下一层还是包装就继续递归)。
    """
    def walk(routes):
        for r in routes:
            if isinstance(r, APIRoute):
                yield r
            else:
                inner = getattr(r, "original_router", None)
                if inner is not None:
                    yield from walk(inner.routes)

    for r in walk(app.routes):
        if r.path.startswith("/api/"):
            yield r


def test_the_scan_actually_finds_routes():
    """⚠️ **上面那条扫描器的护栏 —— 没有它,这一整个文件是一条空绿。**

    `assert not unguarded` 在**空列表**上恒真:一个「一条都没遍历到」的扫描器
    (比如 fastapi 换了个包装类型、或者写成了非递归)会**安静地通过**,
    而它一个端点都没检查过 —— 这正是本文件要防的那种假绿,只不过换到了它自己身上。

    27 = spec §6.4 的操作数(25 既有 + login + me)。**只多不少** ——
    将来加端点时这个数字不会假红(留了余量),但「遍历塌成 0」一定会。
    """
    found = [(sorted(r.methods)[0], r.path) for r in _iter_api_routes()]
    assert len(found) >= 27, (
        f"只扫到 {len(found)} 条端点 —— 扫描器塌了(不是端点少了):{found}"
    )


def test_every_api_route_is_guarded_or_whitelisted():
    unguarded = []
    for r in _iter_api_routes():
        key = (sorted(r.methods)[0], r.path)
        if key in PUBLIC:
            continue
        if not (_guard_names(r) & {"require_user", "require_admin"}):
            unguarded.append(key)
    assert not unguarded, (
        f"这些端点**没有**挂守卫:{unguarded}\n"
        f"(忘了挂的 router 在整套测试里是绿的 —— 那条 autouse 装置把它们盖住了)"
    )


def test_workbench_routes_require_admin():
    """工作台的四个 router 必须是 `require_admin` —— 挂成 `require_user` 的话
    任何登录用户都能核准知识,而**所有测试照样绿**(默认装置给的是 admin)。"""
    workbench = ("/api/kb/", "/api/review/", "/api/topics/", "/api/traces/")
    wrong = []
    for r in _iter_api_routes():
        if not r.path.startswith(workbench):
            continue
        if "require_admin" not in _guard_names(r):
            wrong.append((sorted(r.methods)[0], r.path, _guard_names(r)))
    assert not wrong, f"这些工作台端点不是 require_admin:{wrong}"


def test_login_is_public_and_me_is_user_only():
    by_key = {(sorted(r.methods)[0], r.path): r for r in _iter_api_routes()}
    assert _guard_names(by_key[("POST", "/api/auth/login")]) == set(), \
        "登录端点不能要 token(否则永远登不上)"
    assert "require_user" in _guard_names(by_key[("GET", "/api/auth/me")])
    assert "require_admin" not in _guard_names(by_key[("GET", "/api/auth/me")])
```

- [ ] **Step 2: 跑它,确认红**

Run: `.venv/Scripts/python.exe -m pytest tests/test_auth_wiring.py -p no:cacheprovider`
Expected: FAIL,并**逐条列出**未受保护的端点(把那份清单原样抄进提交信息 —— 它是"改动面"的证据)。

- [ ] **Step 3: 改九个 router 声明行**

五个用户面 router(`chat` / `conversations` / `extract` / `feedback` / `refund`):

```python
router = APIRouter(dependencies=[Depends(require_user)])
```

四个工作台 router(`kb` / `review` / `topics` / `traces`):

```python
router = APIRouter(dependencies=[Depends(require_admin)])
```

并在每个文件顶部补 import(`from fastapi import APIRouter, Depends, HTTPException`
—— 多数文件里 `Depends` 已经有了)与 `from app.auth import require_user`(或 `require_admin`)。

> **为什么用 router 级**:9 行 vs 25 处,少 16 个「调用点自己记得做」的机会;
> 而且它让上面那条结构性测试成为可能(本仓「不变量放在唯一写口上」)。

- [ ] **Step 4: 核对 `tests/conftest.py` 的默认登录装置**已经在**了(⚠️ **不要重复加**)**

⚠️ **它由 Task 4 落地**(那才是第一个让端点需要 token 的任务 —— 依赖写在**端点签名**里
就已经生效,不必等本任务的 router 级守卫;理由与代码见 T4 的 Step 8b)。
本步**一个字都不改**,只核对:

```bash
grep -n "_default_login\|_no_default_login" tests/conftest.py
```
Expected: 两个 fixture 都在;`_default_login` 是 **autouse**,
且用 `"_no_default_login" in request.fixturenames` 判断
(⚠️ **不是**把它写成参数 —— 见 T4 Step 8b 里那条说明)。

⚠️ 若它**不在**(T4 漏了):**就地补上**(代码抄 T4 的 Step 8b),
并把「T4 漏了」写进你的报告 —— 那是**跨任务的缺口**,不是本任务的实现问题。

- [ ] **Step 5: 跑两个方向都验一遍**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_auth_wiring.py tests/test_api_auth.py tests/test_auth.py -p no:cacheprovider
```
Expected: 全绿。**再手工验一次「装置真的关了」**:把 `test_me_without_token_is_401_and_not_403`
临时改成不带 `_no_default_login`,它必须**变成 200 而红** —— 红了说明装置真的生效、且那条用例真的在测「没登录」。
(验完改回去。)

- [ ] **Step 6: 全量回归**

Run: `.venv/Scripts/python.exe -m pytest -p no:cacheprovider`
Expected: 打印出 `N passed`(**不许**加 `-q`)。此时预计有一批红 —— **逐条看**,它们要么是
「请求体里还带着 `user_id`」(删掉即可),要么是**真的漏挂了守卫**(那条路由要修)。修完再跑。

- [ ] **Step 7: 提交**

```bash
git add app/api/ tests/conftest.py tests/test_auth_wiring.py
git commit -m "认证 T5:9 个 router 挂守卫(用户面 require_user / 工作台 require_admin)+ 结构性测试兜住假绿"
```

---

## Task 6: 前端(纯 UI,按项目规矩走 Vibe Coding)

**Files:**
- Create: `app/static/auth.js`
- Modify: `app/static/index.html`(8 处 `fetch(`)、`app/static/admin.html`(1 处 `req()`)

**Interfaces:**
- Consumes: Task 3 的 `POST /api/auth/login`(四个键)、`GET /api/auth/me`
- Produces: 全局 `authFetch(path, opts)`、`showLogin()`、`getToken()`、`clearToken()`、
  `authHeader()` —— 两个页面都用这一套

- [ ] **Step 1: 写 `app/static/auth.js`**

```javascript
/* 认证(2026-09-27):**唯一**一处实现「401 怎么办」的地方。
 *
 * 两个页面(聊天页 / 工作台)共用它 —— 401 的处置在两页上必须是同一件事,
 * 各写一遍迟早漂移,而漂移的表现是「一页弹登录、另一页白屏」。
 *
 * ⚠️ token 存 localStorage(**与现有的 mewhelp.session_id 同一个地方**)。
 *    代价如实记账:XSS 能读走它。本仓没有 httpOnly cookie + CSRF 那一套,
 *    而需求原话是「后续请求就携带 jwt」(见 spec §7.4)。
 */
(() => {
  "use strict";

  const TOKEN_KEY = "mewhelp.jwt";

  function getToken() {
    try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
  }
  function setToken(t) {
    try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); } catch { /* 隐私模式 */ }
  }
  function clearToken() { setToken(null); }

  function authHeader() {
    const t = getToken();
    return t ? { Authorization: `Bearer ${t}` } : {};
  }

  // ── 登录浮层 ──────────────────────────────────────────────
  let overlay = null;

  function buildOverlay() {
    const box = document.createElement("div");
    box.id = "auth-overlay";
    box.innerHTML = `
      <div class="auth-card">
        <h2>登录 · 云枢客服</h2>
        <label>用户名<input id="auth-user" type="text" autocomplete="username"></label>
        <label>密码<input id="auth-pass" type="password" autocomplete="current-password"></label>
        <div class="auth-err" id="auth-err"></div>
        <button id="auth-go" type="button">登录</button>
        <div class="auth-hint">
          演示账号:<code>cinfly / 123456</code>、<code>demo-user / 123456</code>
        </div>
      </div>`;
    document.body.appendChild(box);
    return box;
  }

  /** 弹登录浮层。`onDone` 在**登录成功**后调用一次(调用方拿它重放请求)。 */
  function showLogin(onDone) {
    if (!overlay) overlay = buildOverlay();
    overlay.style.display = "flex";
    const err = overlay.querySelector("#auth-err");
    err.textContent = "";
    const go = overlay.querySelector("#auth-go");

    async function submit() {
      const username = overlay.querySelector("#auth-user").value.trim();
      const password = overlay.querySelector("#auth-pass").value;
      if (!username || !password) { err.textContent = "用户名与密码都要填"; return; }
      go.disabled = true;
      err.textContent = "登录中…";
      try {
        const r = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password }),
        });
        const body = await r.json().catch(() => null);
        if (!r.ok) {
          // ⚠️ 原样显示服务端的文案(它是 401「用户名或密码不正确」)——
          //    前端**不许**自己编一句,那会与服务端漂移。
          err.textContent = body && body.detail ? body.detail : `HTTP ${r.status}`;
          return;
        }
        setToken(body.token);
        overlay.style.display = "none";
        overlay.querySelector("#auth-pass").value = "";
        if (onDone) await onDone();
      } catch (e) {
        err.textContent = `连不上服务(${e.message})`;
      } finally {
        go.disabled = false;
      }
    }

    go.onclick = submit;
    overlay.querySelector("#auth-pass").onkeydown = (e) => { if (e.key === "Enter") submit(); };
    overlay.querySelector("#auth-user").focus();
  }

  /**
   * 带 token 的 fetch。**401 ⇒ 清 token、弹登录、成功后重放那一次的调用**。
   *
   * ⚠️ **只重放一次**(`_retried`):token 坏了而服务端照回 401 时不重放,
   *    否则会变成「登录 → 401 → 弹登录」的死循环。
   * ⚠️ **流式请求不自动重放**(`opts.stream === true`):那要重发一条用户消息,
   *    比让他再点一次「发送」更糟(见 spec §7.3)。那种情况下只弹登录、不回放。
   */
  async function authFetch(path, opts = {}) {
    const init = { ...opts, headers: { ...(opts.headers || {}), ...authHeader() } };
    delete init.stream;
    let resp = await fetch(path, init);
    if (resp.status !== 401) return resp;

    clearToken();
    const retry = new Promise((resolve) => {
      showLogin(() => resolve(true));
    });
    if (opts.stream) return resp;          // 流式:不重放,把这次 401 原样交回调用方
    const ok = await retry;
    if (!ok) return resp;
    return fetch(path, { ...opts, headers: { ...(opts.headers || {}), ...authHeader() } });
  }

  /** 顶栏那个「当前用户 / 退出」。两个页面各自调用,`el` 是容器。 */
  async function mountUserBadge(el) {
    const r = await fetch("/api/auth/me", { headers: authHeader() });
    if (!r.ok) { el.textContent = "未登录"; return; }
    const me = await r.json();
    el.replaceChildren();
    const name = document.createElement("span");
    name.textContent = `${me.username}(${me.role})`;
    const out = document.createElement("a");
    out.href = "#"; out.textContent = "退出";
    out.onclick = (e) => { e.preventDefault(); clearToken(); location.reload(); };
    el.append(name, document.createTextNode(" · "), out);
  }

  window.authFetch = authFetch;
  window.showLogin = showLogin;
  window.getToken = getToken;
  window.setToken = setToken;
  window.clearToken = clearToken;
  window.mountUserBadge = mountUserBadge;
  //: 启动时:手里没 token 就直接弹(两页共用同一条路径)。
  window.authBoot = (onReady) => {
    if (getToken()) { onReady(); return; }
    showLogin(onReady);
  };
})();
```

- [ ] **Step 2: 加浮层的样式(两个页面各贴一段同样的 CSS)**

```css
  /* ── 登录浮层(认证)── */
  #auth-overlay { display: none; position: fixed; inset: 0; z-index: 50;
    background: rgba(28,43,57,.45); align-items: center; justify-content: center; }
  .auth-card { background: #fff; border: 3px solid var(--ink); border-radius: 4px;
    padding: 20px 22px; width: 320px; box-shadow: 6px 6px 0 rgba(28,43,57,.18); }
  .auth-card h2 { margin: 0 0 12px; font-size: 16px; }
  .auth-card label { display: block; font-size: 13px; color: #5b7186; margin-bottom: 8px; }
  .auth-card input { width: 100%; margin-top: 4px; }
  .auth-card .auth-err { color: var(--warn); font-size: 12.5px; min-height: 18px; }
  .auth-card .auth-hint { font-size: 11.5px; color: #9aa7b4; margin-top: 8px; line-height: 1.6; }
```

- [ ] **Step 3: 改 `index.html`**

1. `<head>` 之外、`<script>` **之前**加 `<script src="/static/auth.js"></script>`
   (⚠️ 必须**先**加载 —— 下面那把 IIFE 依赖 `window.authFetch`)。
2. 8 处 `fetch(` 改成 `authFetch(`:`:544`(feedback)、`:691`(chat/stream **加 `stream: true`**)、
   `:787` / `:859`(kb/documents)、`:999`(refund)、`:1069`(ticket)、
   `:1201`(conversations)、`:1296`(messages)。
3. 顶栏加一个容器 `<span id="who"></span>`,并在启动处:
   ```javascript
   mountUserBadge(document.getElementById("who"));
   ```
4. **把启动包进 `authBoot`**:原来那条 `boot().catch(...)` 改成
   ```javascript
   authBoot(() => { mountUserBadge(document.getElementById("who")); boot().catch((e) => console.warn("初始化失败(聊天不受影响):", e)); });
   ```
   ⇒ 没 token 时**先弹登录**,登录成功才 boot(否则侧栏会先打一串 401)。

- [ ] **Step 4: 改 `admin.html`**

1. 同样先引 `<script src="/static/auth.js"></script>`。
2. 唯一一处 `fetch(`(`:441` 的 `req()`)改成 `authFetch(path, opts)` —— **改这一处就全覆盖**。
3. 顶栏加 `#who` 容器 + 同样的 `authBoot` 包裹(启动处换成
   `authBoot(() => { mountUserBadge(...); switchTab("首页"); })`)。

- [ ] **Step 5: 语法与接线自检**

```bash
node --check <(sed -n '/^(() => {/,/^})();$/p' app/static/auth.js)   # 或把 auth.js 直接 node --check
.venv/Scripts/python.exe -m pytest tests/test_static_wiring.py -p no:cacheprovider
```
Expected: 语法过;接线测试仍绿(新加的 `#who` 与 `auth.js` 不影响它)。

- [ ] **Step 6: 真机手验(必须做,而且要**硬刷新**)**

1. 重启服务,浏览器开 `http://localhost:8000/` ⇒ **应当弹登录浮层**。
2. 用 `cinfly / 123456` 登录 ⇒ 进入聊天页;**侧栏应当是空的**(cinfly 没有历史,见 spec §4 的连带事实)。
3. 退出,改用 `demo-user / 123456` ⇒ **侧栏应当有 581 条**,点任意一条**能看到历史**。
4. 开工作台 `http://localhost:8000/admin.html` ⇒ 首页五张卡正常。
5. 手工把 `localStorage["mewhelp.jwt"]` 改成一个乱串,刷新 ⇒ **应当弹登录**(401 路径)。

- [ ] **Step 7: 提交**

```bash
git add app/static/auth.js app/static/index.html app/static/admin.html
git commit -m "认证 T6:前端 auth.js(唯一 401 出口)+ 两页接入(纯 UI,Vibe Coding)"
```

---

## Task 7: 七个验收脚本

**Files:**
- Modify: `scripts/acceptance.sh`、`acceptance_ch05.sh`、`acceptance_ch06.sh`、`acceptance_ch07.sh`、
  `acceptance_ch08.sh`、`acceptance_ch09.sh`、`acceptance_ch10.sh`

- [ ] **Step 1: 给每个脚本加三行(位置:服务就绪**之后**、第一处 API 调用**之前**)**

```bash
# ---- 认证(ch10 跟进)----
# 三行接入,让下面**一百处**裸 curl **一个字都不用改**:
#   ① 登录一次拿 token(cinfly 是 admin ⇒ 一个 token 覆盖用户面 + 工作台两侧);
#   ② 用**同名函数遮蔽 curl** —— 后续每一处 curl 自动带上 Authorization 头。
# ⚠️ 函数只在**当前 shell** 生效:脚本里若有 `( ... )` 子 shell 或 `bash -c`,
#    那些调用不受影响,要就地补 -H(逐脚本核一遍)。
TOKEN=$("$PYTHON" - <<'PYEOF'
import json, urllib.request
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/auth/login",
    data=json.dumps({"username": "cinfly", "password": "123456"}).encode(),
    headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=10))["token"])
PYEOF
)
curl() { command curl -H "Authorization: Bearer $TOKEN" "$@"; }
# ↑ 用 python 的 urllib 而不是 curl 去登录:此刻 curl 还没被遮蔽,但**登录本身
#   也不该走 argv 传中文**(本仓规矩);而且这样不依赖 jq。
```

> ⚠️ `ch08` / `ch09` 自己起服务:这三行要放在**它们等服务就绪之后**的位置
> (照各自既有的「等端口」写法)。
> ⚠️ 端口变量各脚本不同(`PORT` / `PORT_A` / 裸 `8000`)—— 用**该脚本自己的**那个。

- [ ] **Step 2: 每个脚本局部重跑一遍**

逐个跑并**把转录留档**:
```bash
bash scripts/acceptance_ch07.sh 2>&1 | tee .superpowers/auth_t7_ch07.txt | tail -20
```
⚠️ 跑之前**清残留进程**(8000/8001/8101/8102/8103)—— 本仓记过的那类假红。

- [ ] **Step 3: 对「还是红」的逐条判断**

三种可能,判据要分清:
1. **脚本自己没带 token**(函数遮蔽没生效 / 在子 shell 里)⇒ 修脚本;
2. **代码真的坏了** ⇒ 修代码;
3. **与认证无关的既有红**(`acceptance.sh` 本来就红,CLAUDE.md 记着)⇒ 如实记账,不动它。

- [ ] **Step 4: 提交**

```bash
git add scripts/acceptance*.sh .superpowers/auth_t7_*.txt
git commit -m "认证 T7:7 个验收脚本各加三行(遮蔽 curl),100 处调用一字未改"
```

---

## Task 8: 文档与记账

**Files:**
- Modify: `CLAUDE.md`、`.env.example`、`dev-notes/ch10.md`、spec 的 §12

- [ ] **Step 1: 改 `CLAUDE.md` 那句「全程不做:认证」**

原文:`**全程不做**:多轮 Agent Loop、认证。`
改成:

```
**全程不做**:多轮 Agent Loop。
(**认证**原本也在此列,由**用户 2026-09-27 要求**补上 —— 见
`docs/superpowers/specs/2026-09-27-ecommerce-cs-auth-design.md` 与
`dev-notes/ch10.md`。)
```

- [ ] **Step 2: `CLAUDE.md` 的架构段与高频命令**

- `app/` 树里加两行:`app/auth.py`(**唯一**鉴权边界)与 `app/api/auth.py`(登录);
- `db/` 清单里加 **`db/auth.sql`**(并写明它**必须跑**(ORM 里没有这张表)+ 它**不幂等**);
- 高频命令加:
```bash
.venv/Scripts/python.exe scripts/seed_users.py                    # 预置账号(幂等)
# ↑ 登录页的演示账号从这儿来;改密码 / 加账号都改这个脚本的 ACCOUNTS
```

- [ ] **Step 3: `.env.example` 加两行**

```
# ---- 认证(2026-09-27)----
# ⚠️ 下面这个值是**入库的**(本文件在版本控制里)⇒ 它是**公开值**,只能本机演示用。
# 真正部署必须换成随机密钥(`python -c "import secrets;print(secrets.token_urlsafe(32))"`)。
# 不配的话服务也能跑:会用一个**每次启动随机生成**的密钥,代价是重启后旧 token 全失效。
JWT_SECRET=itcinfly
JWT_EXPIRE_MINUTES=720
```

- [ ] **Step 4: `dev-notes/ch10.md` 补一段(阶段 13)**

四样必须写全(本仓规矩):
- **用户关键原话**:spec §3 那段逐字 + 「jwtsecret用itcinfly」+「两个账号都是超级用户」;
- **关键产出**:七个任务的提交、`db/auth.sql`、两个账号、25 个端点的权限矩阵;
- **被纠偏的**:spec §12.1 那条(FastAPI 缺 header 的实际状态码 —— 我先把**旧行为**
  读成当前事实,靠实测订正)、以及「用户点名 `itcinfly` 与两个超级用户」与我原设计
  (随机密钥 / 一 admin 一 user)**不一致**,按用户的来;
- **翻车与返工**:Task 5 全量回归里那批红的分类;Task 7 里哪些是脚本问题、哪些是真缺陷。

- [ ] **Step 5: spec §12 追加实现订正**

实现期间任何与文档不一致的地方(含 Task 4 Step 6 那条**写路径缺口**的最终裁定)。

- [ ] **Step 6: 全量回归 + 真机走一遍**

```bash
.venv/Scripts/python.exe -m pytest -p no:cacheprovider     # 打印 N passed
```
再按 Task 6 Step 6 那五条手验一次(硬刷新)。

- [ ] **Step 7: 提交**

```bash
git add CLAUDE.md .env.example dev-notes/ch10.md docs/superpowers/specs/2026-09-27-ecommerce-cs-auth-design.md
git commit -m "认证 T8:CLAUDE.md(推翻『不做认证』)+ .env.example + dev-notes 阶段 13 + spec 订正"
```

---

## Self-Review

**1. Spec coverage(逐节对)**

| spec 节 | 落点 |
|---|---|
| §1 目标 1(按用户隔离) | Task 4(列表 + 明细归属) |
| §1 目标 2(JWT + 401) | Task 1(内核)/ Task 3(端点)/ Task 5(挂守卫) |
| §1 目标 3(前端弹未认证) | Task 6 |
| §1 目标 4(删 `user_id`) | Task 4 Step 3-4 |
| §3 推翻「不做认证」的记账 | Task 8 Step 1 |
| §4 两个账号 / 密钥 / 都 admin | Task 2(`ACCOUNTS`)/ Task 1(`jwt_secret`) |
| §5.1 `users` 表 / 不加 FK | Task 2 Step 1(理由在 SQL 注释里) |
| §5.2 幂等种子 | Task 2 Step 5-6 |
| §6.1 唯一边界 + 401 自己抛 | Task 1(含 `AuthError` 收口) |
| §6.2 两个端点 | Task 3 |
| §6.3 router 级依赖 | Task 5 Step 3 |
| §6.4 权限矩阵 27 条 | Task 5 Step 1(结构性测试**逐条**遍历) |
| §6.5 删 user_id + 连带注释 | Task 4 Step 3 与 Step 5(删 `DEMO_USER` 注释) |
| §7 前端(含 §7.3 不重放流式 / §7.4 localStorage) | Task 6(`authFetch` 的 `stream` 分支 + 注释) |
| §8.1 autouse 装置 | Task 5 Step 4 |
| §8.2 结构性测试 | Task 5 Step 1 |
| §8.3 两个新测试文件 | Task 1 / Task 3 |
| §8.4 真库上的互不可见 | Task 4 Step 1 |
| §9 配置 + requirements | Task 1 Step 1 |
| §10 验收脚本三行接入 | Task 7 |
| §12.1 FastAPI 订正 | **已在 spec 里**(写计划时订正) |
| §2 非目标(注册/刷新/…) | 无任务 —— **故意的**,YAGNI |

**2. Placeholder scan**:无 TBD / 「加适当的错误处理」/「类似 Task N」。
两处**刻意**留成描述而非代码,都写明了理由与判据:Task 4 Step 1 里那个 `_sess()`
helper(该文件用的是 module 级 helper 风格,不新造 fixture —— 写死会在实现时与它打架)、
Task 7 Step 3 的「逐条判断三种可能」——两处都给了**必须断什么**。

> ⚠️ **写计划时抓到的两处「计划与实际不符」**(已订正,记在这里免得再犯):
> ① 我原以为 `tests/test_api_chat.py` 只是「构造带 `user_id` 的请求体」,
> 实际那里有**三条专门的用例**(含一条断 422 的),而 `ChatRequest` **没有
> `extra="forbid"`** ⇒ 删字段后那条 422 会变成 200 —— 「删掉那个字段」远远不够,
> 要按 Task 4 Step 7 那张表逐条改写;
> ② 我原以为 `tests/test_api_conversations_db.py` 用的是 `session`/`client` 装置,
> 实际它用 **module 级 helper 直接调端点函数**(`list_conversations(session=...)`)⇒
> 那两个函数的签名一改,它们的**调用方**就得跟着改。
> **两条都是「引用一个文件前先读它」那条规矩的又一次兑现。**

**3. Type consistency**:
- `AuthenticatedUser(username, role)` —— Task 1 定义,Task 3/4/5 一致使用;
- `get_owned_conversation(*, session, conversation_id, user_id) -> Conversation | None`
  —— Task 4 定义并使用;
- `AuthenticatedUser` 在 Task 5 的 conftest 装置里按**关键字**构造(`username=` / `role=`)✓;
- 依赖名全程 `require_user` / `require_admin`(**没有** `current_user` 残留 —— spec §6.1 那条订正已同步);
- 常量 `ADMIN` / `USER` 在 Task 1/2/3/5 里同名。
