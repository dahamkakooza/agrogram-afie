from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from src.config.settings import settings

# Create database engine with connection pooling
engine = create_engine(
    str(settings.database_url),
    poolclass=QueuePool,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_pre_ping=True,  # Verify connections before using
    echo=settings.database_echo,
    echo_pool=True if settings.database_echo else False,
    connect_args={
        "connect_timeout": 10,
        "application_name": "afie_service",
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)

# Create base class for models
Base = declarative_base()


def get_db():
    """
    Dependency function to get database session.
    Use with FastAPI Depends() for automatic session management.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()