from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    """全部 ORM 模型的基类。"""


@lru_cache
def get_engine() -> AsyncEngine:
    """进程内单例。pool_pre_ping 让空闲连接被 MySQL 掐断后能自愈。"""
    return create_async_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False:提交后仍可读属性,否则异步下访问会触发隐式 IO 报错。
    return async_sessionmaker(get_engine(), expire_on_commit=False)
