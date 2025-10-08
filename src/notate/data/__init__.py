#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/data/__init__.py

Purpose:
    Public interface for the `notate.data` package.
    Re-exports dataset factories/loaders and accumulator utilities.

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments.
    - Avoid wildcard imports to keep a clean namespace.
    - No behavior-changing edits were made intentionally.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# Datasets / loaders
from .dataset import Dataset, get_dataset, get_dataloader

# Accumulators
from .accumulator import (
    NumpyAccumulator,
    ListAccumulator,
    accumulator_type2class,
    get_accumulator,
)

__all__ = [
    # Datasets / loaders
    "Dataset",
    "get_dataset",
    "get_dataloader",
    # Accumulators
    "NumpyAccumulator",
    "ListAccumulator",
    "accumulator_type2class",
    "get_accumulator",
]
