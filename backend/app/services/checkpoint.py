"""PostgreSQL Checkpoint 初始化与获取（异步版本）。"""
import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg_pool import AsyncConnectionPool

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# 显式允许自定义类型，消除反序列化警告
_serializer = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("app.services.router", "RouteResult"),
        ("app.agent.state_reducers", "_Sentinel"),
        ("app.agent.state_reducers", "NewPlan"),
    ]
)

_checkpointer: AsyncPostgresSaver | None = None
_pool: AsyncConnectionPool | None = None


async def close_checkpointer() -> None:
    """关闭全局连接池，应在应用关闭时调用。"""
    global _checkpointer, _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        _checkpointer = None
        logger.info("✓ Checkpoint 连接池已关闭")


async def get_checkpointer() -> AsyncPostgresSaver:
    """获取全局 AsyncPostgresSaver 单例（懒加载 + 自动建表）。"""
    global _checkpointer, _pool
    if _checkpointer is None:
        s = get_settings()
        logger.info(f"初始化 AsyncPostgresSaver: {s.POSTGRES_URI.split('@')[-1]}")

        # 创建异步连接池（用 async with 或显式 open）
        _pool = AsyncConnectionPool(
            conninfo=s.POSTGRES_URI,
            min_size=s.PG_POOL_MIN_SIZE,
            max_size=s.PG_POOL_MAX_SIZE,
            timeout=s.PG_POOL_TIMEOUT,
            max_idle=s.PG_POOL_MAX_IDLE,
            max_lifetime=s.PG_POOL_MAX_LIFETIME,
            num_workers=s.PG_POOL_NUM_WORKERS,
            kwargs={"autocommit": True, "prepare_threshold": None},
            open=False,  # 不在构造时自动打开
        )
        await _pool.open()

        # 创建 checkpointer，使用自定义序列化器
        _checkpointer = AsyncPostgresSaver(_pool, serde=_serializer)
        _checkpointer.supports_pipeline = False

        # 自动创建 checkpoints 表（异步调用）
        await _checkpointer.setup()
        logger.info("✓ Checkpoint 表已就绪")
    return _checkpointer
