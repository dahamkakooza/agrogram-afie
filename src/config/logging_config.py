import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config.settings import settings

# Create logs directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configure root logger
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            LOG_DIR / "afie.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        ),
    ]
)

# Create logger instances
api_logger = logging.getLogger("afie.api")
core_logger = logging.getLogger("afie.core")
data_logger = logging.getLogger("afie.data")
ml_logger = logging.getLogger("afie.ml")
db_logger = logging.getLogger("afie.database")
cache_logger = logging.getLogger("afie.cache")


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger with consistent configuration.
    
    Args:
        name: Logger name (e.g., "afie.api.routes")
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"afie.{name}")