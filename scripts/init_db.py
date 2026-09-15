"""建表。用法:.venv/Scripts/python.exe scripts/init_db.py

幂等 —— create_all 只建不存在的表,重复跑不会破坏已有数据。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import models  # noqa: F401  导入以确保模型注册到 Base.metadata
from app.db.base import Base, get_engine


async def main() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    print("建表完成:", ", ".join(sorted(Base.metadata.tables)))


if __name__ == "__main__":
    asyncio.run(main())
