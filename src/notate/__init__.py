#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/__init__.py

Purpose:
    Top-level initializer for the `notate` package.
    - Lightweight version resolution
    - Lazy import of core symbols and subpackages
    - Autoregistration utility that recursively imports `notate.modules.*`
      so that module registries are populated via side effects.

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Behavior preserved from the provided implementation.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Standard library =====
import importlib
import pkgutil
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
          {'core', 'modules', 'training', 'data', 'tools', 'downstream'}.
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
def enable_autoregistration() -> None:
    """Recursively import `notate.modules.*` to populate registries via side effects.

    Example:
        >>> import notate
        >>> notate.enable_autoregistration()  # ensures register_module calls run
    """
    pkg_name = f"{__name__}.modules"
    try:
        package = importlib.import_module(pkg_name)
    except ModuleNotFoundError:
        # Be permissive: if modules package does not exist, do nothing.
        return

    for _, modname, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(modname)
