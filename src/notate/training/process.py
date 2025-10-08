#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/process.py

Purpose:
    Config-driven process graph runner:
      - Resolve callable modules/functions for forward-style steps
      - Support iterated subgraphs (loop over length)
      - Provide a small op-resolver for training ops (decoupled from core)

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Behavior preserved. The ops resolver expects an importable module name
      in `PROCESS_OPS_MODULE` that defines allowed ops (e.g., 'forward', 'loss').
    - `PRINT_PROCESS` toggles light debugging of shapes during iterate.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Third-party =====
import numpy as np
import torch

# ===== Local op resolver (decoupled from core) =====
import importlib

PRINT_PROCESS = False
# Set to your ops implementation module (must be importable):
PROCESS_OPS_MODULE = "notate.training.process_ops"
ALLOWED_PROCESS_OPS = {
    "forward",
    "loss",
    "backward",
    "step",
    "metrics",
    "accumulate",
    "zero_grad",
    "clip_grad",
}

_OPS_MOD = None
try:
    _OPS_MOD = importlib.import_module(PROCESS_OPS_MODULE)
except Exception:
    _OPS_MOD = None


def function_config2func(cfg, logger=None):
    """Resolve a process op (callable, kwargs) from a compact config.

    Accepts:
        - str: op name (e.g., "forward")
        - dict: {"op": "...", ...} or {"type": "...", ...}

    Returns:
        (callable, kwargs_dict)
    """
    if isinstance(cfg, str):
        name = cfg.strip()
        kwargs = {}
    elif isinstance(cfg, dict):
        name = str(cfg.get("op") or cfg.get("type") or "").strip()
        kwargs = {k: v for k, v in cfg.items() if k not in ("op", "type")}
    else:
        raise SystemExit("[CONFIG ERROR] process op spec must be a string or mapping")
    if not name:
        raise SystemExit("[CONFIG ERROR] process op name is required")
    if name not in ALLOWED_PROCESS_OPS:
        raise SystemExit(
            f"[CONFIG ERROR] Unknown process op '{name}'. "
            f"Allowed: {sorted(ALLOWED_PROCESS_OPS)}"
        )
    if _OPS_MOD is None:
        raise SystemExit(
            f"[CONFIG ERROR] Process ops module '{PROCESS_OPS_MODULE}' could not be imported. "
            "Define your ops there or set PROCESS_OPS_MODULE correctly."
        )
    try:
        fn = getattr(_OPS_MOD, name)
    except AttributeError:
        raise SystemExit(
            f"[CONFIG ERROR] Process op '{name}' is not defined in '{PROCESS_OPS_MODULE}'."
        )
    if logger:
        logger.debug(
            "[process-op] %s(%s)", name, ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        )
    return fn, kwargs


# =============================================================================
# Process base
# =============================================================================
class Process:
    """Abstract process node."""

    def __call__(self, model, batch):
        raise NotImplementedError


class CallProcess(Process):
    """Process that calls a callable with inputs from `batch` and writes outputs."""

    def __init__(self, input, output=None, **kwargs):
        self.input = input
        self.output = output if output is not None else input
        self.kwargs = kwargs

    def __call__(self, model, batch):
        callable_ = self.get_callable(model)

        if self.input is None:
            output = callable_(**self.kwargs)
        elif isinstance(self.input, str):
            output = callable_(batch[self.input], **self.kwargs)
        elif isinstance(self.input, list):
            output = callable_(*[batch[i] for i in self.input], **self.kwargs)
        elif isinstance(self.input, dict):
            output = callable_(**{name: batch[i] for name, i in self.input.items()}, **self.kwargs)
        else:
            raise ValueError(f"Unsupported type of input: {type(self.input).__name__}")

        if isinstance(self.output, str):
            batch[self.output] = output
        elif isinstance(self.output, list):
            for oname, o in zip(self.output, output):
                batch[oname] = o
        else:
            raise ValueError(f"Unsupported type of output: {type(self.output).__name__}")

    def get_callable(self, model):
        """Return the callable invoked by this process (override in subclass)."""
        raise NotImplementedError


class ForwardProcess(CallProcess):
    """Call a model submodule retrieved by name (dict-like access on Model)."""

    def __init__(self, module, input, output=None, **kwargs):
        """
        Args:
            module: Name of submodule registered in the Model.
            input: str | list[str] | dict[str, str] from batch.
            output: str | list[str] | None for batch output keys.
            **kwargs: Extra kwargs passed to the submodule call.
        """
        super().__init__(input, output, **kwargs)
        self.module = module

    def get_callable(self, model):
        return model[self.module]


class FunctionProcess(CallProcess):
    """Call a generic function resolved from a compact config."""

    def __init__(self, function, input, output=None, **kwargs):
        """
        Args:
            function: Dict for function_config2func (e.g., {"op": "loss", "reduction": "sum"}).
            input: str | list[str] | dict[str, str] from batch.
            output: str | list[str] | None for batch output keys.
            **kwargs: Extra kwargs passed to the function call.
        """
        super().__init__(input, output, **kwargs)
        self.function, self.func_kwargs = function_config2func(function)

    def get_callable(self, model):
        # Merge static func_kwargs with call-time kwargs (call-time wins).
        def _wrapped(*args, **kwargs):
            merged = dict(self.func_kwargs)
            merged.update(kwargs)
            return self.function(*args, **merged)

        return _wrapped


class IterateProcess(Process):
    """Looping process that executes a list of processes for each step."""

    def __init__(self, length, processes, i_name: str = "iterate_i"):
        """
        Args:
            length: int or name of length in batch.
            processes: List of process configs to execute each iteration.
            i_name: Name for iteration index in batch.
        """
        self.length = length
        self.processes = [get_process(**process) for process in processes]
        self.i_name = i_name

    def __call__(self, model, batch):
        if isinstance(self.length, int):
            length = self.length
        else:
            length = batch[self.length]

        for i in range(length):
            batch[self.i_name] = i

            for j, process in enumerate(self.processes):
                if PRINT_PROCESS:
                    # Show parameters
                    print(f"---process {j}---")
                    for key, value in batch.items():
                        if isinstance(value, (torch.Tensor, np.ndarray)):
                            print(f"  {key}: {list(value.shape)}")
                        else:
                            print(f"  {key}: {type(value).__name__}")
                process(model, batch)


# =============================================================================
# Factory
# =============================================================================
process_type2class = {
    "forward": ForwardProcess,
    "function": FunctionProcess,
    "iterate": IterateProcess,
}


def get_process(type: str = "forward", **kwargs) -> Process:
    """Construct a Process instance by type name."""
    return process_type2class[type](**kwargs)
