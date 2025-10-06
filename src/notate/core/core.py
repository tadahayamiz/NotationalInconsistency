# -*- coding: utf-8 -*-
"""
notate.core.core
- Central registry for module types
- Robust resolver for module classes
- Factory to build modules from config
- Model container that injects module registry for cross-module access
- Utility: function_config2func (used by training.process) and PRINT_PROCESS flag
"""

from __future__ import annotations
import importlib
import inspect
from typing import Any, Dict, Mapping, MutableMapping, Optional, Type, Callable, Iterable, Sequence

# ------------------------------------------------------------------------------
# Global flags (training.process expects this)
# ------------------------------------------------------------------------------
PRINT_PROCESS: bool = False  # training.process が参照するログトグル


# ------------------------------------------------------------------------------
# Central registry (shared with notate.modules.__init__)
# ------------------------------------------------------------------------------
module_type2class: Dict[str, Type] = {}


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _maybe_str_format(val: Any, variables: Optional[Mapping[str, Any]] = None) -> Any:
    """Mild variable substitution helper (keeps unknowns)."""
    if variables is None:
        return val
    if isinstance(val, str):
        try:
            return val.format(**variables)
        except Exception:
            return val
    if isinstance(val, list):
        return [_maybe_str_format(v, variables) for v in val]
    if isinstance(val, dict):
        return {k: _maybe_str_format(v, variables) for k, v in val.items()}
    return val


def _is_fully_qualified(name: str) -> bool:
    return isinstance(name, str) and "." in name and not name.endswith(".")


def _resolve_fully_qualified(name: str) -> Optional[Type]:
    """
    Resolve 'package.subpackage.ClassName' safely.
    """
    try:
        mod_name, cls_name = name.rsplit(".", 1)
    except ValueError:
        return None
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return None
    return getattr(mod, cls_name, None)


# ------------------------------------------------------------------------------
# Function resolver (training.process expects `function_config2func`)
# ------------------------------------------------------------------------------
def function_config2func(fcfg: Mapping[str, Any]) -> Callable:
    """
    Convert a function config dict into a Python callable.
    Minimal but practical coverage for typical tensor ops used in loops.

    Expected format:
      fcfg = {"type": "<name>", ...extra kwargs...}

    The returned callable is typically used as: out = fn(x, **kwargs_from_fcfg)
    NOTE: Some functions (like 'cat', 'stack') expect input to be a list/sequence.
    """
    if not isinstance(fcfg, Mapping) or "type" not in fcfg:
        raise ValueError(f"function_config2func: invalid config {fcfg}")

    ftype = fcfg.get("type")
    # ---- Built-ins ----
    if ftype == "identity":
        def _fn(x, **kw): return x
        return _fn

    if ftype == "transpose":
        # supports keys: dim0, ...1  (to be compatible with your YAML)
        def _fn(x, dim0=0, **kw):
            dim1 = kw.get("...1", 1)
            return x.transpose(dim0, dim1)
        return _fn

    if ftype == "permute":
        def _fn(x, *dims, **kw):
            if not dims:
                dims = tuple(kw.get("dims", []))
            return x.permute(*dims)
        return _fn

    if ftype == "reshape":
        def _fn(x, *shape, **kw):
            if not shape:
                shape = tuple(kw.get("shape", []))
            return x.reshape(*shape)
        return _fn

    if ftype == "unsqueeze":
        def _fn(x, dim=0, **kw):
            return x.unsqueeze(dim)
        return _fn

    if ftype == "squeeze":
        def _fn(x, dim=None, **kw):
            return x.squeeze() if dim is None else x.squeeze(dim)
        return _fn

    if ftype == "cat":
        def _fn(xs, dim=0, **kw):
            import torch
            return torch.cat(xs, dim=dim)
        return _fn

    if ftype == "stack":
        def _fn(xs, dim=0, **kw):
            import torch
            return torch.stack(xs, dim=dim)
        return _fn

    if ftype == "getitem":
        def _fn(x, key=None, **kw):
            return x[key]
        return _fn

    if ftype == "len":
        def _fn(x, **kw):
            return len(x)
        return _fn

    if ftype == "to":
        def _fn(x, device=None, dtype=None, non_blocking=False, **kw):
            if device is None and dtype is None:
                return x
            return x.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return _fn

    if ftype == "detach":
        def _fn(x, **kw):
            return x.detach()
        return _fn

    # reductions
    if ftype == "mean":
        def _fn(x, dim=None, keepdim=False, **kw):
            return x.mean(dim=dim, keepdim=keepdim) if dim is not None else x.mean()
        return _fn

    if ftype == "sum":
        def _fn(x, dim=None, keepdim=False, **kw):
            return x.sum(dim=dim, keepdim=keepdim) if dim is not None else x.sum()
        return _fn

    if ftype == "max":
        def _fn(x, dim=None, keepdim=False, **kw):
            import torch
            if dim is None:
                return x.max()
            return torch.max(x, dim=dim, keepdim=keepdim).values
        return _fn

    if ftype == "min":
        def _fn(x, dim=None, keepdim=False, **kw):
            import torch
            if dim is None:
                return x.min()
            return torch.min(x, dim=dim, keepdim=keepdim).values
        return _fn

    # basic arithmetic
    if ftype == "add":
        def _fn(x, y, **kw):
            return x + y
        return _fn

    if ftype == "sub":
        def _fn(x, y, **kw):
            return x - y
        return _fn

    if ftype == "mul":
        def _fn(x, y, **kw):
            return x * y
        return _fn

    if ftype == "div":
        def _fn(x, y, **kw):
            return x / y
        return _fn

    # fallback: allow fully-qualified function path "pkg.mod:func" or "pkg.mod.func"
    if isinstance(ftype, str):
        # colon form
        if ":" in ftype:
            mod_name, fn_name = ftype.split(":", 1)
        elif "." in ftype:
            # last dot separates attribute
            mod_name, fn_name = ftype.rsplit(".", 1)
        else:
            mod_name, fn_name = None, None

        if mod_name and fn_name:
            try:
                mod = importlib.import_module(mod_name)
                fn = getattr(mod, fn_name, None)
                if callable(fn):
                    return fn
            except Exception:
                pass

    raise KeyError(f"function_config2func: unknown function type '{ftype}' in {fcfg}")


# ------------------------------------------------------------------------------
# Initialization function resolver
# ------------------------------------------------------------------------------
def init_config2func(icfg: Mapping[str, Any]) -> Callable:
    """
    Convert an initialization config dict into a Python callable.
    
    Expected format:
      icfg = {"type": "<init_type>", ...extra kwargs...}
    
    Returns a function that can initialize a parameter tensor.
    """
    if not isinstance(icfg, Mapping) or "type" not in icfg:
        raise ValueError(f"init_config2func: invalid config {icfg}")
    
    itype = icfg.get("type")
    
    # Common PyTorch initializations
    if itype == "normal":
        def _fn(param):
            import torch.nn as nn
            mean = icfg.get("mean", 0.0)
            std = icfg.get("std", 1.0)
            nn.init.normal_(param, mean=mean, std=std)
        return _fn
    
    if itype == "uniform":
        def _fn(param):
            import torch.nn as nn
            a = icfg.get("a", 0.0)
            b = icfg.get("b", 1.0)
            nn.init.uniform_(param, a=a, b=b)
        return _fn
    
    if itype == "xavier_uniform":
        def _fn(param):
            import torch.nn as nn
            gain = icfg.get("gain", 1.0)
            nn.init.xavier_uniform_(param, gain=gain)
        return _fn
    
    if itype == "xavier_normal":
        def _fn(param):
            import torch.nn as nn
            gain = icfg.get("gain", 1.0)
            nn.init.xavier_normal_(param, gain=gain)
        return _fn
    
    if itype == "kaiming_uniform":
        def _fn(param):
            import torch.nn as nn
            a = icfg.get("a", 0)
            mode = icfg.get("mode", "fan_in")
            nonlinearity = icfg.get("nonlinearity", "leaky_relu")
            nn.init.kaiming_uniform_(param, a=a, mode=mode, nonlinearity=nonlinearity)
        return _fn
    
    if itype == "kaiming_normal":
        def _fn(param):
            import torch.nn as nn
            a = icfg.get("a", 0)
            mode = icfg.get("mode", "fan_in")
            nonlinearity = icfg.get("nonlinearity", "leaky_relu")
            nn.init.kaiming_normal_(param, a=a, mode=mode, nonlinearity=nonlinearity)
        return _fn
    
    if itype == "constant":
        def _fn(param):
            import torch.nn as nn
            val = icfg.get("val", 0.0)
            nn.init.constant_(param, val)
        return _fn
    
    if itype == "zeros":
        def _fn(param):
            import torch.nn as nn
            nn.init.zeros_(param)
        return _fn
    
    if itype == "ones":
        def _fn(param):
            import torch.nn as nn
            nn.init.ones_(param)
        return _fn
    
    if itype == "eye":
        def _fn(param):
            import torch.nn as nn
            nn.init.eye_(param)
        return _fn
    
    # Default: do nothing (identity)
    if itype == "identity" or itype is None:
        def _fn(param):
            pass
        return _fn
    
    raise KeyError(f"init_config2func: unknown initialization type '{itype}' in {icfg}")


# ------------------------------------------------------------------------------
# Class resolver (ROBUST)
# ------------------------------------------------------------------------------
def resolve_module_class(type_name: str) -> Type:
    """
    Resolve a module class by type name with robust fallbacks.

    Order:
      1) module_type2class (already registered)
      2) import notate.modules (its __init__ should register types)
      3) getattr(notate.modules, type_name)
      4) import notate.modules.<sub> for sub in ["pipeline","sequence","vae","tunnel","poolers"]
         then check registry/getattr
      5) fully-qualified resolution if type_name looks like 'pkg.mod.Class'
    """
    if not type_name or not isinstance(type_name, str):
        raise ImportError(f"Invalid module type: {type_name}")

    # 1) already in registry
    cls = module_type2class.get(type_name)
    if cls is not None:
        return cls

    # 5) fully qualified path support (early try if provided)
    if _is_fully_qualified(type_name):
        fq = _resolve_fully_qualified(type_name)
        if fq is not None:
            return fq

    # 2) import main package (triggers __init__ registrations)
    pkg = None
    try:
        pkg = importlib.import_module("notate.modules")
    except Exception:
        pkg = None

    # 2') check registry again
    cls = module_type2class.get(type_name)
    if cls is not None:
        return cls

    # 3) getattr from package
    if pkg is not None and hasattr(pkg, type_name):
        return getattr(pkg, type_name)

    # 4) try submodules
    for sub in ("pipeline", "sequence", "vae", "tunnel", "poolers"):
        try:
            m = importlib.import_module(f"notate.modules.{sub}")
        except Exception:
            continue
        # registry first
        cls = module_type2class.get(type_name)
        if cls is not None:
            return cls
        # then getattr
        if hasattr(m, type_name):
            return getattr(m, type_name)

    # Last: try fully-qualified again (in case submodule imports added path)
    if _is_fully_qualified(type_name):
        fq = _resolve_fully_qualified(type_name)
        if fq is not None:
            return fq

    raise ImportError(f"Unable to resolve module class for type='{type_name}'")


# ------------------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------------------
def build_module_from_config(
    name: str,
    mcfg: Mapping[str, Any],
    logger: Optional[Any] = None,
    variables: Optional[Mapping[str, Any]] = None,
) -> Any:
    """
    Instantiate a module from config dict (expects 'type' key).
    - Pass 'logger' if the constructor supports it.
    - Perform mild string formatting with variables.
    """
    if not isinstance(mcfg, Mapping):
        raise TypeError(f"module config for '{name}' must be a mapping, got {type(mcfg)}")

    if "type" not in mcfg:
        raise KeyError(f"module '{name}' requires 'type' in config")

    mtype = mcfg["type"]
    cls = resolve_module_class(str(mtype))

    # prepare kwargs (exclude 'type')
    kwargs = {k: v for k, v in mcfg.items() if k != "type"}
    if variables:
        kwargs = _maybe_str_format(kwargs, variables)

    # Inject logger only if accepted
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        sig = None

    if logger is not None and sig is not None and "logger" in sig.parameters:
        kwargs.setdefault("logger", logger)

    # Instantiate
    try:
        instance = cls(**kwargs)
    except TypeError as e:
        # Provide more context
        raise TypeError(f"Failed to instantiate '{name}' as {cls.__name__} with kwargs={kwargs}: {e}") from e

    # Attach a back-reference name if it helps debugging
    try:
        setattr(instance, "_module_name", name)
    except Exception:
        pass

    return instance


# ------------------------------------------------------------------------------
# Model container
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Thin container that builds modules from config and injects a registry so that
    modules can access siblings (e.g., PipelineModule resolving named modules).
    """

    def __init__(
        self,
        modules: Optional[Mapping[str, Mapping[str, Any]]] = None,
        logger: Optional[Any] = None,
        variables: Optional[Mapping[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.logger = logger
        self.variables = dict(variables) if variables else {}
        self._built_modules: Dict[str, nn.Module] = {}

        # Allow extra kwargs for compatibility; ignore them
        _ = args, kwargs

        # Build all modules
        if modules:
            for name, mcfg in modules.items():
                mod = build_module_from_config(name, mcfg, logger=self.logger, variables=self.variables)
                self._built_modules[name] = mod
                # also register as attribute for convenience (nn.Module handles submodules)
                try:
                    self.add_module(name, mod)
                except Exception:
                    # fallback: setattr (but prefer add_module)
                    setattr(self, name, mod)

        # ---- inject registry to every module (recommended) ----
        # Prefer attach_registry(...) if provided, otherwise set _module_registry
        for _name, _mod in self._built_modules.items():
            try:
                registry = self._built_modules
                if hasattr(_mod, "attach_registry") and callable(_mod.attach_registry):
                    _mod.attach_registry(registry)
                else:
                    setattr(_mod, "_module_registry", registry)
            except Exception:
                # Do not block initialization if a module rejects registry
                pass

    # Optional helpers
    def get(self, name: str) -> nn.Module:
        return self._built_modules[name]

    @property
    def modules_dict(self) -> Dict[str, nn.Module]:
        return self._built_modules.copy()
