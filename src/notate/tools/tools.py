#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/tools/tools.py

Purpose:
    Lightweight utility functions used across modules:
      - `nullcontext`: no-op context manager
      - `prog`: simple progress marker printer
      - `check_leftargs`: warn about unused kwargs
      - `EMPTY`: identity function (lambda x: x)

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - No behavior-changing edits were made intentionally.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Standard library =====
from contextlib import contextmanager


# =============================================================================
# Context manager
# =============================================================================
@contextmanager
def nullcontext():
    """A context manager that does nothing (like contextlib.nullcontext)."""
    yield None


# =============================================================================
# Simple progress marker
# =============================================================================
def prog(marker: str = "*") -> None:
    """Print a one-character progress marker without newline (flush=True)."""
    print(marker, flush=True, end="")


# =============================================================================
# Argument check utility
# =============================================================================
def check_leftargs(self, logger, kwargs, show_content: bool = False) -> None:
    """Check for unused keyword arguments and optionally warn.

    Args:
        self: The calling object (for class name in message).
        logger: Logger instance (may be None).
        kwargs: Dictionary of unprocessed keyword arguments.
        show_content: If True, show full dict; otherwise, show only keys.
    """
    if len(kwargs) > 0 and logger is not None:
        unknown = kwargs if show_content else list(kwargs.keys())
        logger.warning(f"Unknown kwarg in {type(self).__name__}: {unknown}")


# =============================================================================
# Identity function
# =============================================================================
EMPTY = lambda x: x  # noqa: E731
