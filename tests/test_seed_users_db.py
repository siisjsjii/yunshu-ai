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
