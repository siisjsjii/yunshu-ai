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
