"""
COMPLETE DATABASE CONFIGURATION
Production-ready async database setup with proper connection pooling and error handling.
"""
import asyncio
import logging
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    async_sessionmaker, 
    create_async_engine,
    AsyncEngine
)
from sqlalchemy.pool import AsyncAdaptedQueuePool
from sqlalchemy import event, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from src.config.settings import settings
from src.config.logging_config import get_logger

logger = get_logger("database.base")


def _get_async_db_url() -> str:
    """Get async database URL with proper driver."""
    db_url = str(settings.database_url)
    
    # Ensure we're using asyncpg driver
    if not db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
    
    return db_url


def _setup_engine_events(engine: AsyncEngine):
    """Setup SQLAlchemy engine events for monitoring and debugging."""
    
    @event.listens_for(engine.sync_engine, "connect")
    def connect(dbapi_connection, connection_record):
        """Log when a new database connection is created."""
        logger.debug("New database connection established")
    
    @event.listens_for(engine.sync_engine, "checkout")
    def checkout(dbapi_connection, connection_record, connection_proxy):
        """Log when a connection is checked out from the pool."""
        logger.debug("Database connection checked out from pool")
    
    @event.listens_for(engine.sync_engine, "checkin")
    def checkin(dbapi_connection, connection_record):
        """Log when a connection is returned to the pool."""
        logger.debug("Database connection returned to pool")
    
    @event.listens_for(engine.sync_engine, "handle_error")
    def handle_error(exception_context):
        """Log database errors."""
        logger.error(
            f"Database error: {exception_context.original_exception}",
            exc_info=exception_context.original_exception
        )


# Create async engine with connection pooling
engine: Optional[AsyncEngine] = None

try:
    engine = create_async_engine(
        _get_async_db_url(),
        echo=settings.database_echo,
        echo_pool=settings.database_echo,
        poolclass=AsyncAdaptedQueuePool,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,  # Recycle connections after 1 hour
        pool_timeout=30,  # Wait up to 30 seconds for connection
        connect_args={
            "command_timeout": 60,
            "server_settings": {
                "application_name": "afie_service",
                "jit": "off",  # Disable JIT for better performance
            }
        },
        future=True,
    )
    
    # Setup engine events
    _setup_engine_events(engine)
    
    logger.info(f"Database engine created with pool size: {settings.database_pool_size}")
    
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    if settings.environment == "production":
        raise
    # In development, we might continue without database
    engine = None


# Create async session factory
if engine:
    AsyncSessionLocal = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    # Also create async_session alias for compatibility
    async_session = AsyncSessionLocal
else:
    AsyncSessionLocal = None
    async_session = None
    logger.warning("Database engine not available. Database operations will fail.")


# Create declarative base
Base = declarative_base()


# Database dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function to get database session.
    
    Usage in FastAPI endpoints:
    ```python
    @app.get("/items")
    async def read_items(db: AsyncSession = Depends(get_db)):
        # Use db session
        result = await db.execute(text("SELECT 1"))
        return {"result": result.scalar()}
    ```
    
    Yields:
        AsyncSession: SQLAlchemy async session
    
    Raises:
        HTTPException: If database is not available
    """
    if not AsyncSessionLocal:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable"
        )
    
    session: Optional[AsyncSession] = None
    try:
        session = AsyncSessionLocal()
        logger.debug("Database session created")
        
        # Test connection
        await session.execute(text("SELECT 1"))
        
        yield session
        await session.commit()
        logger.debug("Database session committed")
        
    except OperationalError as e:
        logger.error(f"Database operational error: {e}")
        if session:
            await session.rollback()
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Database connection failed"
        )
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        if session:
            await session.rollback()
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in database session: {e}")
        if session:
            await session.rollback()
        raise
        
    finally:
        if session:
            await session.close()
            logger.debug("Database session closed")


# Simple context manager for async sessions
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Simple async session context manager."""
    if not async_session:
        raise RuntimeError("Database session factory not available")
    
    session = async_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


# Utility functions for database operations
async def create_tables():
    """Create all database tables if they don't exist."""
    if not engine:
        logger.warning("Cannot create tables: database engine not available")
        return
    
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
            
            # Log created tables
            result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result.fetchall()]
            logger.info(f"Created tables: {', '.join(tables)}")
            
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


async def drop_tables():
    """Drop all database tables (for testing only)."""
    if not engine:
        logger.warning("Cannot drop tables: database engine not available")
        return
    
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        raise


async def check_database_connection() -> bool:
    """Check if database connection is working."""
    if not engine:
        return False
    
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


async def get_database_stats() -> dict:
    """Get database connection pool statistics."""
    if not engine:
        return {"status": "not_available"}
    
    try:
        pool = engine.pool
        
        stats = {
            "status": "connected",
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "checkedin": pool.checkedin(),
            "connections": pool.status(),
        }
        
        # Get database version
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT version()"))
            stats["version"] = result.scalar()
            
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {"status": "error", "error": str(e)}


# Export public interface
__all__ = [
    "Base",
    "engine",
    "AsyncSessionLocal",
    "async_session",  # Added this export
    "get_db",
    "get_async_session",  # Added this function
    "create_tables",
    "drop_tables",
    "check_database_connection",
    "get_database_stats",
]