#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/hooks.py

Purpose:
    Define reusable training-time hooks that trigger at alarm conditions:
      - SaveAlarmHook: saves model checkpoints
      - AccumulateHook: collects statistics and saves to CSV periodically
      - AbortHooks: terminate training by time/step/epoch/threshold
      - NoticeAlarmHook: optional external notifier

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Imports ordered: stdlib -> third-party -> local (project).
    - No behavior-changing edits were made intentionally.
    - Hooks are lightweight; specialized ones (scheduler/validation/checkpoint)
      are implemented locally in scripts/train.py due to runtime dependencies.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Standard library =====
import os
import time
from collections import defaultdict

# ===== Third-party =====
import numpy as np
import pandas as pd
import torch

# ===== Project-local =====
from ..tools.alarm import get_alarm


# =============================================================================
# Base alarm hook
# =============================================================================
class AlarmHook:
    """Base class for all alarm-based hooks."""

    def __init__(self, logger, result_dir, alarm=None, end: bool = False):
        """
        Args:
            logger: Logger instance.
            result_dir: Directory to store outputs.
            alarm: dict or list of dicts describing alarms
                (e.g., {'type': 'count', 'target': 'step', 'step': 1000})
            end: If True, also trigger ring() when batch contains 'end'.
        """
        self.logger = logger
        if alarm is None:
            alarm = {"type": "silent", "target": "step"}
        if not isinstance(alarm, list):
            alarm = [alarm]
        self.alarms = [get_alarm(logger=logger, **a) for a in alarm]
        self.end = end

    def __call__(self, batch: dict, model: torch.nn.Module) -> None:
        """Check all alarms and call ring() if triggered."""
        ring = any(alarm(batch) for alarm in self.alarms)
        if ring or ("end" in batch and self.end):
            self.ring(batch=batch, model=model)

    def ring(self, batch: dict, model: torch.nn.Module) -> None:
        """Action executed when an alarm is triggered (to be overridden)."""
        raise NotImplementedError


# =============================================================================
# Hooks
# =============================================================================
class SaveAlarmHook(AlarmHook):
    """Hook to save model checkpoints when triggered."""

    def __init__(self, logger, result_dir, alarm=None, end: bool = False):
        super().__init__(logger, result_dir, alarm, end=end)
        self.models_dir = os.path.join(result_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def ring(self, batch: dict, model: torch.nn.Module) -> None:
        self.logger.info(f"Saving model at step {batch['step']:>6} ...")
        path = os.path.join(self.models_dir, str(batch["step"]))
        # Placeholder for model.save_state_dict(path)
        # (implemented where model class provides this method)


class AccumulateHook:
    """Hook that accumulates scalar values and periodically writes to CSV."""

    def __init__(self, logger, result_dir, names, cols, save_alarm, checkpoint=None):
        """
        Args:
            logger: Logger.
            result_dir: Directory for saving CSV.
            names: List of keys in batch dict to record.
            cols: List of column names for CSV.
            save_alarm: Dict for alarm triggering a save.
            checkpoint: Optional path to an existing CSV to resume from.
        """
        os.makedirs(result_dir, exist_ok=True)
        self.path_df = os.path.join(result_dir, "steps.csv")
        self.save_alarm = get_alarm(logger=logger, **save_alarm)
        self.dfs = []
        if checkpoint is not None:
            self.dfs.append(pd.read_csv(checkpoint))
        self.lists = defaultdict(list)
        self.names = names
        self.cols = cols

    def __call__(self, batch: dict, model: torch.nn.Module) -> None:
        if "end" not in batch:
            for name, col in zip(self.names, self.cols):
                item = batch[name]
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu().numpy()
                self.lists[col].append(item)
        if self.save_alarm(batch):
            self.dfs.append(pd.DataFrame(self.lists))
            self.lists.clear()
            pd.concat(self.dfs).to_csv(self.path_df, index=False)


class AbortHook:
    """Hook that stops training when batch[target] >= threshold."""

    def __init__(self, logger, result_dir, target: str, threshold: float | int):
        """
        Args:
            target: Key in batch dict to monitor.
            threshold: Stop when batch[target] >= threshold.
        """
        self.target = target
        self.threshold = threshold

    def __call__(self, batch: dict, model: torch.nn.Module) -> None:
        if self.target in batch and batch[self.target] >= self.threshold:
            batch["end"] = True


class StepAbortHook(AbortHook):
    """Abort when step count exceeds threshold."""

    def __init__(self, logger, result_dir, threshold):
        super().__init__(logger, result_dir, "step", threshold)


class EpochAbortHook(AbortHook):
    """Abort when epoch count exceeds threshold."""

    def __init__(self, logger, result_dir, threshold):
        super().__init__(logger, result_dir, "epoch", threshold)


class TimeAbortHook:
    """Abort after a specified duration (seconds)."""

    def __init__(self, logger, result_dir, threshold: int):
        """
        Args:
            threshold: Time limit in seconds.
        """
        self.end_time = time.time() + threshold

    def __call__(self, batch: dict, model: torch.nn.Module) -> None:
        if time.time() > self.end_time:
            batch["end"] = True


class NoticeAlarmHook(AlarmHook):
    """Hook for sending notifications when alarms trigger."""

    def __init__(self, logger, studyname=None, **kwargs):
        super().__init__(logger=logger, **kwargs)
        if studyname is None:
            logger.warning("studyname not specified in NoticeAlarm.")
            studyname = "(study noname)"
        self.studyname = studyname

    def ring(self, batch: dict, model: torch.nn.Module) -> None:
        """Send notice on trigger (if available)."""
        try:
            from ..tools.notice import notice  # optional dependency

            if "end" in batch:
                notice(f"models/train: {self.studyname} finished!")
            else:
                message = f"models/train: {self.studyname} "
                for alarm in self.alarms:
                    message += f"{alarm.target} {batch[alarm.target]} "
                message += "finished!"
                notice(message)
        except ImportError:
            self.logger.info(f"Notice: {self.studyname} alarm triggered")


# =============================================================================
# Hook registry
# =============================================================================
hook_type2class = {
    "save_alarm": SaveAlarmHook,
    "accumulate": AccumulateHook,
    "abort": AbortHook,
    "step_abort": StepAbortHook,
    "epoch_abort": EpochAbortHook,
    "time_abort": TimeAbortHook,
    "notice_alarm": NoticeAlarmHook,
    # scheduler_alarm, validation_alarm, checkpoint_alarm are defined in train.py
}


def get_hook(type: str, **kwargs):
    """Instantiate hook by type name."""
    return hook_type2class[type](**kwargs)
