"""Centralized logging configuration using loguru.

This module provides a pre-configured logger for the DiFD project.
"""

import sys

from loguru import logger

# Remove default handler
logger.remove()

# Add custom handler with pretty formatting
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)


def configure_logging(level: str = "INFO", verbose: bool = False) -> None:
    """Configure logging level.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        verbose: If True, show full module path in logs.
    """
    logger.remove()

    if verbose:
        fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    else:
        fmt = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"

    logger.add(
        sys.stderr,
        format=fmt,
        level=level.upper(),
        colorize=True,
    )


__all__ = ["logger", "configure_logging"]
