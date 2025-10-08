#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/tools/logger.py

Purpose:
    Lightweight logging utilities with tqdm-friendly stream output.
    Provides a default logger factory and a recursive config pretty-printer.

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Uses a global cache to avoid attaching duplicate handlers to the same logger.
    - No behavior-changing edits were made intentionally.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Standard library =====
import inspect
import logging
import sys
from typing import Any, Mapping, Sequence

# ===== Third-party =====
from tqdm import tqdm


# =============================================================================
# Log levels / formatter
# =============================================================================
log_name2level = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

formatter = logging.Formatter(
    "[%(asctime)s][%(levelname)s] %(message)s", "%y%m%d %H:%M:%S"
)


# =============================================================================
# Tqdm-aware handler
# =============================================================================
class TqdmHandler(logging.Handler):
    """Logging handler that writes via tqdm to keep progress bars intact."""

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stdout)
            self.flush()
        except Exception:
            self.handleError(record)


# Cache of created loggers to prevent duplicate handlers
default_loggers: dict[str, logging.Logger] = {}


# =============================================================================
# Factory for a default logger
# =============================================================================
def default_logger(
    filename: str | None = None,
    stream_level: str = "info",
    file_level: str = "debug",
    logger_name: str | None = None,
) -> logging.Logger:
    """Create (or fetch) a default logger with tqdm-friendly stream output.

    Args:
        filename: Optional path for a file handler. If None, file output is disabled.
        stream_level: Log level name for the stream handler ("info", "debug", ...).
        file_level: Log level name for the file handler.
        logger_name: Optional logger name. If None, derive from caller module.

    Returns:
        A configured `logging.Logger` instance.
    """
    if logger_name is None:
        logger_name = inspect.getmodulename(inspect.stack()[0].filename)

    logger = logging.getLogger(logger_name)

    if logger_name not in default_loggers:
        default_loggers[logger_name] = logger

        # Stream (tqdm) handler
        stream_handler = TqdmHandler(level=log_name2level[stream_level])
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File handler (optional)
        if filename is not None:
            file_handler = logging.FileHandler(filename=filename)
            file_handler.setLevel(log_name2level[file_level])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    # Always set logger level to DEBUG; per-handler levels govern actual output.
    logger.setLevel(logging.DEBUG)
    return logger


# =============================================================================
# Pretty-print config
# =============================================================================
def log_config(
    logger: logging.Logger,
    config: Mapping[str, Any] | Sequence[Any],
    level: int = logging.INFO,
    prefix: str = "",
) -> None:
    """Recursively log a nested config (dict/list) with indentation.

    Args:
        logger: Logger to write messages to.
        config: Dictionary or list-like structure to log.
        level: Logging level (e.g., logging.INFO).
        prefix: Indentation prefix used internally during recursion.
    """
    if isinstance(config, list):
        config = {f"[{i}]": value for i, value in enumerate(config)}
    for key, value in config.items():  # type: ignore[union-attr]
        if isinstance(value, (list, dict)):
            logger.log(level, prefix + f"{key}:")
            log_config(logger, value, level, prefix=prefix + "  ")
        else:
            logger.log(level, prefix + f"{key}: {value}")
