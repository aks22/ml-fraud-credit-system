"""
src/utils/logger.py
-------------------
Centralised logger using loguru.
Import `logger` from this module in every other module.

Usage:
    from src.utils.logger import logger
    logger.info("Training started")
    logger.error("Something failed: {err}", err=str(e))
"""

import sys
from loguru import logger
from config.settings import LOG_LEVEL, OUTPUTS_DIR

# Remove default handler
logger.remove()

# Console handler — coloured output for development
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)

# File handler — rotating JSON logs for production audit trail
logger.add(
    OUTPUTS_DIR / "app.log",
    level="INFO",
    rotation="10 MB",    # New file after 10 MB
    retention="30 days", # Keep logs for 30 days
    compression="zip",   # Compress old files
    serialize=True,      # JSON format for log aggregation tools
)

__all__ = ["logger"]
