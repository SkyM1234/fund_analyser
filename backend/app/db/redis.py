"""Redis 异步客户端（缓存 / 限流 / 分布式锁 / refresh token 状态）。"""
import logging

from redis.asyncio import Redis

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_client: Redis | None = None


def get_redis_client() -> Redis:
    global _client
    if _client is None:
        s = get_settings()
        _client = Redis.from_url(s.REDIS_URL, decode_responses=True)
        logger.info(f"✓ Redis 客户端已创建: {s.REDIS_URL}")
    return _client


async def close_redis_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("✓ Redis 客户端已关闭")
