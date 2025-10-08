#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/__init__.py

Purpose:
    Top-level initializer for the `notate` package.
    Provides lightweight version resolution, lazy subpackage import,
    and an explicit function `enable_autoregistration()` to populate
    internal registries by side-effect imports.

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Lazy import minimizes import overhead for common workflows.
    - Registries (e.g., module_type2class) are filled when calling
      `notate.enable_autoregistration()`.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Standard library =====
from importlib import import_module, metadata

# =============================================================================
# Package metadata
# =============================================================================
try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["__version__", "Model", "enable_autoregistration"]


# =============================================================================
# Lazy import logic
# =============================================================================
def __getattr__(name: str):
    """Lazily import core symbols and subpackages.

    Attributes handled:
        - Model: imported from `notate.core` on first access.
        - Subpackages: dynamically loaded modules
          ('core', 'modules', 'training', 'data', 'tools', 'downstream').
    """
    if name == "Model":
        from .core import Model as _Model

        return _Model

    if name in {"core", "modules", "training", "data", "tools", "downstream"}:
        return import_module(f"{__name__}.{name}")

    raise AttributeError(name)


# =============================================================================
# Registry population helper
# =============================================================================
def enable_autoregistration(*, modules=True, training=False, data=False, tools=False, downstream=False):
    if modules:
        import_module(f"{__name__}.modules")
    if training:
        import_module(f"{__name__}.training")
    if data:
        import_module(f"{__name__}.data")
    if tools:
        import_module(f"{__name__}.tools")
    if downstream:
        import_module(f"{__name__}.downstream")