#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/training/__init__.py

Purpose:
    Aggregate and expose all training-related components:
      - Hooks (alarms, accumulators, aborts, notices)
      - Process graph runners (forward/function/iterate)
      - Optimizers & schedulers
      - Metrics

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - No behavior-changing edits were made intentionally.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Import training submodules =====
from .hooks import *
from .process import *
from .optimizer import *
from .metric import *

# =============================================================================
# Public exports
# =============================================================================
__all__ = [
    # Hooks
    "AlarmHook",
    "SaveAlarmHook",
    "AccumulateHook",
    "AbortHook",
    "StepAbortHook",
    "EpochAbortHook",
    "TimeAbortHook",
    "NoticeAlarmHook",
    "hook_type2class",
    "get_hook",
    # Processes
    "Process",
    "CallProcess",
    "ForwardProcess",
    "FunctionProcess",
    "IterateProcess",
    "process_type2class",
    "get_process",
    # Optimizers
    "RAdam",
    "LookaheadOptimizer",
    "optimizer_type2class",
    "scheduler_type2class",
    "get_optimizer",
    "get_scheduler",
    # Metrics
    "Metric",
    "BinaryMetric",
    "AUROCMetric",
    "AUPRMetric",
    "RMSEMetric",
    "MAEMetric",
    "R2Metric",
    "MeanMetric",
    "ValueMetric",
    "PerfectAccuracyMetric",
    "PartialAccuracyMetric",
    "metric_type2class",
    "get_metric",
]
