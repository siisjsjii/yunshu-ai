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
