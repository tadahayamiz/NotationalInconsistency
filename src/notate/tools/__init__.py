#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/tools/__init__.py

Purpose:
    Aggregated import point for all utility submodules:
      - Logging utilities
      - Argument/configuration helpers
      - Path/file utilities
      - Small general-purpose tools
      - Alarm system for training hooks

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - No behavior-changing edits were made intentionally.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Import submodules =====
from .logger import *
from .args import *
from .path import *
from .tools import *
from .alarm import *

# =============================================================================
# Public exports
# =============================================================================
__all__ = [
    # Logger
    "default_logger",
    "log_config",
    # Args / Config
    "load_config2",
    "subs_vars",
    "clip_config",
    # Path utilities
    "make_result_dir",
    "timestamp",
    # Tool helpers
    "nullcontext",
    "prog",
    "check_leftargs",
    "EMPTY",
    # Alarms
    "BaseAlarm",
    "SilentAlarm",
    "CountAlarm",
    "ListAlarm",
    "ThresholdAlarm",
    "get_alarm",
]
