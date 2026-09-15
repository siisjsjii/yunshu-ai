from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_sessionmaker


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖。测试用 app.dependency_overrides 替换。"""
    async with get_sessionmaker()() as session:
        yield session
