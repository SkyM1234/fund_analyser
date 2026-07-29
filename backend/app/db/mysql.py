"""MySQL 异步引擎与会话工厂（业务数据：用户、会话索引、审计日志）。"""
import logging

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        s = get_settings()
        _engine = create_async_engine(s.MYSQL_URI, pool_size=10, pool_pre_ping=True, echo=False)
        logger.info(f"✓ MySQL 引擎已创建: {s.MYSQL_HOST}:{s.MYSQL_PORT}/{s.MYSQL_DATABASE}")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def get_db() -> AsyncSession:
    """FastAPI 依赖：每请求一个 AsyncSession，用完自动关闭。"""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def close_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("✓ MySQL 引擎已关闭")
