"""
Health check endpoints.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.database import get_db
from src.config.logging_config import get_logger

router = APIRouter()
logger = get_logger("api.health")


@router.get("/")
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "service": "afie"}


@router.get("/database")
async def database_health(db: AsyncSession = Depends(get_db)):
    """Database health check."""
    from sqlalchemy import text
    
    try:
        result = await db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}


@router.get("/detailed")
async def detailed_health():
    """Detailed health check with all components."""
    from src.config.redis_config import get_redis_client
    import time
    
    checks = {}
    
    # Database check
    from src.config.database import engine
    try:
        async with engine.begin() as conn:
            start = time.time()
            await conn.execute(text("SELECT 1"))
            latency = (time.time() - start) * 1000
            checks["database"] = {
                "status": "healthy",
                "latency_ms": round(latency, 2)
            }
    except Exception as e:
        checks["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Redis check
    try:
        redis = await get_redis_client()
        start = time.time()
        await redis.ping()
        latency = (time.time() - start) * 1000
        checks["redis"] = {
            "status": "healthy",
            "latency_ms": round(latency, 2)
        }
    except Exception as e:
        checks["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Overall status
    all_healthy = all(c["status"] == "healthy" for c in checks.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": time.time(),
        "checks": checks
    }