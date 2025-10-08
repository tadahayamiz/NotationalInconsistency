#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/core/core.py

Purpose:
    Core scaffolding and registry for the notate framework.
    Provides strict module registration, construction from config,
    and the base Model class without a monolithic forward.
    Legacy compatibility functions (function_config2func, init_config2func)
    are included for backward support.

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Imports ordered: stdlib -> third-party -> local (project).
    - No behavior-changing edits were made intentionally.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Standard library =====
import importlib
import inspect
import logging
import math
from functools import partial
from typing import Any, Dict, Optional, Type

# ===== Third-party =====
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Registry (explicit only)
# =============================================================================
module_type2class: Dict[str, Type[nn.Module]] = {}


def register_module(type_name: str, cls: Type[nn.Module]) -> None:
    """Register a module class to a string type name."""
    if not isinstance(type_name, str) or not type_name:
        raise ValueError(
            "[CONFIG ERROR] register_module: type_name must be a non-empty string"
        )
    if not issubclass(cls, nn.Module):
        raise TypeError(
            "[CONFIG ERROR] register_module: cls must be a subclass of torch.nn.Module"
        )
    prev = module_type2class.get(type_name)
    if prev is not None and prev is not cls:
        raise ValueError(
            f"[CONFIG ERROR] Module type '{type_name}' already registered with "
            f"{prev.__name__}, got conflicting class {cls.__name__}"
        )
    module_type2class[type_name] = cls


def resolve_module_class(type_name: str) -> Type[nn.Module]:
    """Resolve a module class from its registered type name."""
    cls = module_type2class.get(type_name)
    if cls is None:
        sample = list(module_type2class.keys())[:20]
        raise KeyError(
            f"[CONFIG ERROR] Unknown module.type '{type_name}'. "
            f"Registered={len(module_type2class)}; sample={sample}"
        )
    return cls


def _filter_kwargs_strict(cls: Type[nn.Module], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to match the target class __init__ signature in strict mode."""
    sig = inspect.signature(cls.__init__)
    valid = {
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.name != "self"
    }
    unknown = set(kwargs.keys()) - valid
    if unknown:
        raise TypeError(
            f"[CONFIG ERROR] Unknown kwargs for {cls.__name__}: {sorted(unknown)}"
        )
    return kwargs


def get_module(logger: Optional[logging.Logger], type: str, **kwargs) -> nn.Module:
    """Instantiate a module by type name with strict kwargs checking."""
    cls = resolve_module_class(type)
    use_kwargs = _filter_kwargs_strict(cls, dict(kwargs))
    if logger:
        logger.debug(
            "Instantiate %s(%s)",
            cls.__name__,
            ", ".join(f"{k}={v!r}" for k, v in use_kwargs.items()),
        )
    return cls(**use_kwargs)


def build_module_from_config(
    name: str, mcfg: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> nn.Module:
    """Build a single module from its config entry."""
    if not isinstance(mcfg, dict):
        raise TypeError(f"[CONFIG ERROR] model.modules.{name} must be a mapping")
    if "type" not in mcfg:
        raise KeyError(f"[CONFIG ERROR] model.modules.{name}.type is required")
    mtype = mcfg["type"]
    if not isinstance(mtype, str) or not mtype:
        raise TypeError(
            f"[CONFIG ERROR] model.modules.{name}.type must be a non-empty string"
        )
    kwargs = {k: v for k, v in mcfg.items() if k != "type"}
    mod = get_module(logger=logger, type=mtype, **kwargs)
    setattr(mod, "__module_name__", name)
    return mod


# =============================================================================
# Model scaffold (strict; no monolithic forward)
# =============================================================================
class Model(nn.Module):
    """Strict model scaffold with explicit module registry."""

    def __init__(
        self, config: Dict[str, Any], logger: Optional[logging.Logger] = None, **_ignored
    ):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        if not isinstance(config, dict):
            raise TypeError("[CONFIG ERROR] Model(config=...) must be a dict-like structure")

        modules_cfg = config.get("modules")
        if not isinstance(modules_cfg, dict) or not modules_cfg:
            raise ValueError("[CONFIG ERROR] model.modules must be a non-empty mapping")

        built: Dict[str, nn.Module] = {}
        for name, mcfg in modules_cfg.items():
            if not isinstance(name, str) or not name:
                raise ValueError("[CONFIG ERROR] module name must be a non-empty string")
            built[name] = build_module_from_config(name, mcfg, logger=self.logger)

        self.components = nn.ModuleDict(built)
        self.use_modules = list(config.get("use_modules", self.components.keys()))
        for n in self.use_modules:
            if n not in self.components:
                raise KeyError(
                    f"[CONFIG ERROR] model.use_modules contains unknown module name '{n}'"
                )

        self.init_cfg = dict(config.get("init", {}))
        self._apply_init_policy()
        self.raw_config = config

        self.logger.debug("[Model] Built modules: %s", list(self.components.keys()))
        self.logger.debug("[Model] use_modules: %s", list(self.use_modules))

    def _apply_init_policy(self) -> None:
        """Apply a simple global initialization policy if specified."""
        itype = str(self.init_cfg.get("type", "default")).lower()
        if itype in ("", "default", "none"):
            return
        elif itype == "xavier_uniform":
            for m in self.modules():
                if hasattr(m, "weight"):
                    try:
                        nn.init.xavier_uniform_(m.weight)  # type: ignore[arg-type]
                    except Exception:
                        pass
                if hasattr(m, "bias") and getattr(m, "bias") is not None:
                    try:
                        nn.init.zeros_(m.bias)  # type: ignore[arg-type]
                    except Exception:
                        pass
        elif itype == "kaiming_normal":
            for m in self.modules():
                if hasattr(m, "weight"):
                    try:
                        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")  # type: ignore[arg-type]
                    except Exception:
                        pass
                if hasattr(m, "bias") and getattr(m, "bias") is not None:
                    try:
                        nn.init.zeros_(m.bias)  # type: ignore[arg-type]
                    except Exception:
                        pass
        else:
            raise ValueError(f"[CONFIG ERROR] Unknown init.type '{self.init_cfg.get('type')}'")

    def get(self, name: str) -> nn.Module:
        """Get a component by name."""
        try:
            return self.components[name]
        except KeyError:
            raise KeyError(f"[CONFIG ERROR] Unknown component '{name}'")

    def forward(self, *args, **kwargs):
        """Intentionally undefined in strict mode."""
        raise NotImplementedError(
            "Model.forward is intentionally undefined in strict mode. "
            "Use process graphs (get_process) to orchestrate component calls."
        )

    def __getitem__(self, key: str):
        """Allow dict-like access to components and attributes."""
        if hasattr(self, "components") and key in self.components:
            return self.components[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Model has no submodule/attr '{key}'")

    def save_state_dict(self, out_prefix: str) -> None:
        """Save model state dict to <out_prefix>.pt."""
        from pathlib import Path

        p = Path(f"{out_prefix}.pt")
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p)


# =============================================================================
# Legacy API compatibility
# =============================================================================
try:
    function_config2func  # type: ignore[name-defined]
except NameError:
    PRINT_PROCESS = False

    # --- legacy activation and init functions ---
    def NewGELU(input):
        return 0.5 * input * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))
            )
        )

    def sigmoid(input):
        return 1 / (1 + math.e ** (-input))

    init_type2func = {
        "glorot_uniform": nn.init.xavier_uniform_,
        "glorot_normal": nn.init.xavier_normal_,
        "he_uniform": nn.init.kaiming_uniform_,
        "he_normal": nn.init.kaiming_normal_,
        "zero": nn.init.zeros_,
        "zeros": nn.init.zeros_,
        "one": nn.init.ones_,
        "ones": nn.init.ones_,
        "normal": nn.init.normal_,
        "none": lambda input: None,
    }

    def init_config2func(type="none", factor=None, **kwargs):
        """Return an initializer callable constructed from a config-like spec."""
        if factor is not None:
            def init(input: nn.Parameter):
                init_config2func(type, **kwargs)(input)
                input.data = input.data * factor
            return init

        if isinstance(type, dict):
            return init_config2func(**type)

        if isinstance(type, (int, float)):
            return lambda input: nn.init.constant_(input, float(type))
        elif type in init_type2func:
            return lambda input: init_type2func[type](input, **kwargs)
        else:
            raise ValueError(f"Unsupported type of init function: {type}")

    function_name2func = {
        "relu": F.relu,
        "gelu": F.gelu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "newgelu": NewGELU,
        "none": lambda input: input,
        "exp": torch.exp,
        "log": torch.log,
        "sum": torch.sum,
        "mean": torch.mean,
        "log_softmax": F.log_softmax,
        "softplus": F.softplus,
        "transpose": torch.transpose,
        "argmax": torch.argmax,
    }

    def function_config2func(config):
        """Legacy function resolver (string or dict -> callable)."""
        if isinstance(config, str):
            return function_name2func[config]
        else:
            return partial(function_name2func[config.pop("type")], **config)

    if "module_type2class" not in globals():
        module_type2class = {}  # type: ignore[assignment]


# =============================================================================
# Extended resolver (modern)
# =============================================================================
def _resolve_symbol(path_or_callable):
    """Resolve a callable or a dotted path like 'torch.nn.Embedding'."""
    if callable(path_or_callable):
        return path_or_callable
    if not isinstance(path_or_callable, str):
        raise TypeError(f"Unsupported symbol spec: {type(path_or_callable)}")
    if "." in path_or_callable:
        modname, attr = path_or_callable.rsplit(".", 1)
        mod = importlib.import_module(modname)
        return getattr(mod, attr)
    table = {
        "relu": F.relu,
        "gelu": F.gelu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "exp": torch.exp,
        "log": torch.log,
        "sum": torch.sum,
        "mean": torch.mean,
        "log_softmax": F.log_softmax,
        "softplus": F.softplus,
        "transpose": torch.transpose,
        "argmax": torch.argmax,
        "Embedding": nn.Embedding,
        "Linear": nn.Linear,
        "LayerNorm": nn.LayerNorm,
        "Dropout": nn.Dropout,
    }
    if path_or_callable in table:
        return table[path_or_callable]
    raise ImportError(f"Cannot resolve symbol '{path_or_callable}'")


def function_config2func(cfg, variables=None, logger=None):
    """Build a callable (constructor or function) from a flexible config."""
    if cfg is None:
        return None
    if callable(cfg):
        return cfg
    if isinstance(cfg, str):
        return _resolve_symbol(cfg)
    if isinstance(cfg, dict):
        if "type" not in cfg:
            raise ValueError("function_config2func requires 'type' in dict config")
        base = function_config2func(cfg["type"], variables=variables, logger=logger)
        kwargs = {k: v for k, v in cfg.items() if k != "type"}
        if inspect.isclass(base):
            return lambda *a, **kw: base(*a, **{**kwargs, **kw})
        return lambda *a, **kw: base(*a, **{**kwargs, **kw})
    if isinstance(cfg, (list, tuple)):
        funcs = [
            function_config2func(x, variables=variables, logger=logger)
            for x in cfg
            if x is not None
        ]

        def chained(x, *args, **kwargs):
            y = x
            for fn in funcs:
                y = fn(y, *args, **kwargs) if callable(fn) else y
            return y

        return chained
    raise TypeError(f"Unsupported function_config type: {type(cfg)}")