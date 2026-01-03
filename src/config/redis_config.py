import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from typing import Optional

from src.config.settings import settings

# Global Redis connection pool
_redis_pool: Optional[ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None


async def get_redis_pool() -> ConnectionPool:
    """Get or create Redis connection pool."""
    global _redis_pool
    
    if _redis_pool is None:
        _redis_pool = ConnectionPool.from_url(
            str(settings.redis_url),
            max_connections=settings.redis_max_connections,
            decode_responses=True,
            socket_keepalive=True,
            socket_timeout=5,
            retry_on_timeout=True,
        )
    
    return _redis_pool


async def get_redis_client() -> redis.Redis:
    """Get Redis client instance."""
    global _redis_client
    
    if _redis_client is None:
        pool = await get_redis_pool()
        _redis_client = redis.Redis(
            connection_pool=pool,
            socket_keepalive=True,
            health_check_interval=30,
        )
    
    return _redis_client


async def close_redis_connections():
    """Close Redis connections (call on shutdown)."""
    global _redis_client, _redis_pool
    
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
    
    if _redis_pool:
        await _redis_pool.disconnect()
        _redis_pool = None