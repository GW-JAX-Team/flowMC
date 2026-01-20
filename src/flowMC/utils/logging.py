"""Centralized logging utilities for flowMC.

This module provides utilities to configure logging for flowMC components
in a consistent and isolated manner.
"""

import logging
from typing import Optional


def enable_verbose_logging(
    logger: logging.Logger,
    level: int = logging.DEBUG,
    format_string: Optional[str] = None,
) -> None:
    """Enable verbose logging for a specific logger.

    This function configures a logger to output debug messages in an isolated way,
    without affecting other loggers in the application. It:
    - Sets the logger's level to the specified level (default: DEBUG)
    - Disables propagation to parent loggers to avoid interference
    - Adds a StreamHandler if one doesn't already exist

    Args:
        logger: The logger instance to configure.
        level: The logging level to set (default: logging.DEBUG).
        format_string: Custom format string for log messages. If None, uses
            a default format: '%(name)s - %(levelname)s - %(message)s'

    Example:
        >>> import logging
        >>> from flowMC.utils.logging import enable_verbose_logging
        >>>
        >>> logger = logging.getLogger(__name__)
        >>> enable_verbose_logging(logger)
        >>> logger.debug("This will now be printed")
    """
    if format_string is None:
        format_string = "%(name)s - %(levelname)s - %(message)s"

    # Set the logger's level
    logger.setLevel(level)

    # Disable propagation to parent loggers so we don't get filtered
    # by root logger's level or duplicate messages
    logger.propagate = False

    # Check if we already have a StreamHandler to avoid duplicates
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) for h in logger.handlers
    )

    if not has_stream_handler:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(handler)


def disable_verbose_logging(logger: logging.Logger) -> None:
    """Disable verbose logging and restore default behavior.

    This function resets a logger to its default state:
    - Resets the logger's level to NOTSET (inherits from parent)
    - Re-enables propagation to parent loggers
    - Removes any handlers that were added

    Args:
        logger: The logger instance to reset.
    """
    logger.setLevel(logging.NOTSET)
    logger.propagate = True

    # Remove all handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
