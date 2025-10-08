#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/core/__init__.py

Purpose:
    Public interface for the `notate.core` package.
    Re-exports the strict Model scaffold, function/initializer resolvers,
    and the explicit module registry.

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments.
    - Avoid wildcard imports to keep a clean namespace.
    - No behavior-changing edits were made intentionally.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# Re-export selected symbols from the core module.
from .core import (
    Model,
    function_config2func,
    init_config2func,
    module_type2class,
    PRINT_PROCESS,
)

__all__ = [
    # Main classes
    "Model",
    # Configuration utilities (resolvers)
    "function_config2func",
    "init_config2func",
    # Registries
    "module_type2class",
    # Utilities
    "PRINT_PROCESS",
]
