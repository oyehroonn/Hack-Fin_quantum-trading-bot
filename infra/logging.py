"""Structured logging with correlation IDs."""

import contextvars
import sys
import uuid
from typing import Optional

from loguru import logger

# Context variable for correlation ID
correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id",
    default=None,
)


def get_correlation_id() -> str:
    """Get or create a correlation ID."""
    cid = correlation_id.get()
    if cid is None:
        cid = str(uuid.uuid4())
        correlation_id.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID."""
    correlation_id.set(cid)


def clear_correlation_id() -> None:
    """Clear the correlation ID."""
    correlation_id.set(None)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    correlation_id_enabled: bool = True,
) -> None:
    """Setup structured logging with correlation IDs."""
    # Remove default handler
    logger.remove()

    # Format with correlation ID
    if correlation_id_enabled:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<yellow>correlation_id={extra[correlation_id]}</yellow> | "
            "<level>{message}</level>"
        )
    else:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Filter to add correlation ID dynamically
    def add_correlation_id(record: dict) -> None:
        """Add correlation ID to log record."""
        if correlation_id_enabled:
            record["extra"]["correlation_id"] = get_correlation_id()

    # Add console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True,
        enqueue=True,
        filter=add_correlation_id,
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation="10 MB",
            retention="7 days",
            enqueue=True,
            filter=add_correlation_id,
        )


# Initialize with default settings
setup_logging()
