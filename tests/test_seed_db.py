"""种子数据测试。需要 MySQL。"""

import pytest
from sqlalchemy import func, select

from app.db.base import get_sessionmaker
from app.db.models import Faq
from scripts.seed_db import FAQ_ROWS, seed

pytestmark = pytest.mark.db


@pytest.mark.anyio
async def test_seed_is_idempotent_and_loads_faq():
    await seed()
    await seed()  # 再跑一次,不应产生重复
    async with get_sessionmaker()() as session:
        count = (await session.execute(select(func.count()).select_from(Faq))).scalar()
    assert count == len(FAQ_ROWS)


def test_seed_contains_no_shipping_fee_entry():
    """验收 3 依赖「邮费」查不到。种子数据里出现该词,验收 3 就会假通过。"""
    blob = " ".join(f"{r['question']} {r['answer']} {r['category']}" for r in FAQ_ROWS)
    assert "邮费" not in blob
    assert "运费" not in blob
