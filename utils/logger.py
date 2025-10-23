"""
Centralized logging utilities for cruijff_kit.

This module provides a consistent interface for logging across all cruijff_kit
scripts, with automatic timestamping and configurable output destinations.

Example usage:
    from cruijff_kit.utils.logger import setup_logger

    logger = setup_logger(__name__)
    logger.info("Starting processing")
    logger.warning("Potential issue detected")
    logger.error("Operation failed")

    # With file output for SLURM jobs
    logger = setup_logger(__name__, log_file="output.log")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger for cruijff_kit scripts.

    Args:
        name: Logger name (typically __name__ from calling module)
        level: Logging level (default: logging.INFO)
        log_file: Optional path to log file. If provided, logs will be written
                  to both console and file.
        console: Whether to log to console (default: True)
        format_string: Custom format string. If None, uses default format with
                       timestamp and level.

    Returns:
        Configured logging.Logger instance

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Processing started")
        [2025-10-23 12:34:56] INFO: Processing started

        >>> logger = setup_logger(__name__, log_file="run.log")
        >>> logger.info("Writing to both console and file")
    """
    logger = logging.getLogger(name)

    # Only configure if logger hasn't been set up yet
    # (prevents duplicate handlers if setup_logger called multiple times)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Default format: [timestamp] LEVEL: message
    if format_string is None:
        format_string = '[%(asctime)s] %(levelname)s: %(message)s'

    formatter = logging.Formatter(
        format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler with auto-flush
    if console:
        # Create a custom handler that flushes after each message
        # This is important for real-time output in SLURM logs and training loops
        class FlushingStreamHandler(logging.StreamHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()

        console_handler = FlushingStreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # File logs include logger name for more context
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoids duplicate messages)
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.

    This is a convenience wrapper around logging.getLogger() for consistency
    with setup_logger().

    Args:
        name: Logger name

    Returns:
        Logger instance (may not be configured if setup_logger wasn't called)

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Using existing logger")
    """
    return logging.getLogger(name)
