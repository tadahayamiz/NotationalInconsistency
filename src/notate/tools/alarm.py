#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/tools/alarm.py

Purpose:
    Alarm system for training hooks.
    Provides modular alarm types that trigger actions (e.g., validation,
    checkpointing) at specified steps, epochs, or thresholds.

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Imports minimized (pure standard library).
    - No behavior-changing edits were made intentionally.

Requires:
    Python >= 3.9
"""

from __future__ import annotations


# =============================================================================
# Base class
# =============================================================================
class BaseAlarm:
    """Base class for all alarm types."""

    def __init__(self, logger, target: str = "step", **kwargs):
        """Initialize base alarm.

        Args:
            logger: Logger instance.
            target: Key in batch dict to monitor (e.g., 'step', 'epoch').
        """
        self.logger = logger
        self.target = target

    def __call__(self, batch: dict) -> bool:
        """Check if alarm should trigger.

        Args:
            batch: Dict containing step/epoch information.

        Returns:
            True if alarm triggers, False otherwise.
        """
        raise NotImplementedError


# =============================================================================
# Alarm types
# =============================================================================
class SilentAlarm(BaseAlarm):
    """Alarm that never triggers (default)."""

    def __call__(self, batch: dict) -> bool:
        return False


class CountAlarm(BaseAlarm):
    """Alarm that triggers every N steps/epochs."""

    def __init__(
        self, logger, target: str = "step", step: int = 1000, start: int = 0, **kwargs
    ):
        """Initialize CountAlarm.

        Args:
            step: Interval between triggers.
            start: Step/epoch to begin counting from.
        """
        super().__init__(logger, target, **kwargs)
        self.step = step
        self.start = start

    def __call__(self, batch: dict) -> bool:
        if self.target not in batch:
            return False
        current = batch[self.target]
        if current < self.start:
            return False
        return (current - self.start) % self.step == 0


class ListAlarm(BaseAlarm):
    """Alarm that triggers at specific steps/epochs."""

    def __init__(self, logger, target: str = "step", list: list | None = None, **kwargs):
        """Initialize ListAlarm.

        Args:
            list: List of step/epoch values to trigger at.
        """
        super().__init__(logger, target, **kwargs)
        self.trigger_list = set(list or [])

    def __call__(self, batch: dict) -> bool:
        if self.target not in batch:
            return False
        return batch[self.target] in self.trigger_list


class ThresholdAlarm(BaseAlarm):
    """Alarm that triggers when monitored value exceeds threshold."""

    def __init__(
        self, logger, target: str = "step", threshold: float = float("inf"), **kwargs
    ):
        """Initialize ThresholdAlarm.

        Args:
            threshold: Value above which the alarm triggers.
        """
        super().__init__(logger, target, **kwargs)
        self.threshold = threshold

    def __call__(self, batch: dict) -> bool:
        if self.target not in batch:
            return False
        return batch[self.target] >= self.threshold


# =============================================================================
# Factory and registry
# =============================================================================
alarm_type2class = {
    "silent": SilentAlarm,
    "count": CountAlarm,
    "list": ListAlarm,
    "threshold": ThresholdAlarm,
}


def get_alarm(logger, type: str = "silent", **kwargs) -> BaseAlarm:
    """Factory to create an alarm instance by type name.

    Args:
        logger: Logger instance.
        type: Alarm type ('silent', 'count', 'list', 'threshold').
        **kwargs: Additional parameters for the selected alarm.

    Returns:
        An initialized alarm instance.

    Examples:
        >>> alarm = get_alarm(logger, type="count", target="step", step=1000)
        >>> if alarm({"step": 2000}):
        ...     print("Triggered!")
    """
    if type not in alarm_type2class:
        logger.warning(f"Unknown alarm type '{type}', using 'silent' instead.")
        type = "silent"
    return alarm_type2class[type](logger=logger, **kwargs)
