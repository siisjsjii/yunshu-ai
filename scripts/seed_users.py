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
