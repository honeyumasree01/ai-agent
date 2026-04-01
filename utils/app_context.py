"""Process-wide async resources (DB pool, Redis) set at FastAPI lifespan."""

import asyncpg
import redis.asyncio as redis

from utils.settings import get_settings

_pool: asyncpg.Pool | None = None
_redis: redis.Redis | None = None


def db_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialized")
    return _pool


def redis_client() -> redis.Redis:
    if _redis is None:
        raise RuntimeError("Redis client not initialized")
    return _redis


def _normalize_dsn(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def init_resources() -> None:
    global _pool, _redis
    s = get_settings()
    _pool = await asyncpg.create_pool(_normalize_dsn(s.database_url), min_size=1, max_size=10)
    _redis = redis.from_url(s.redis_url, decode_responses=True)


async def shutdown_resources() -> None:
    global _pool, _redis
    if _pool is not None:
        await _pool.close()
        _pool = None
    if _redis is not None:
        await _redis.aclose()
        _redis = None
