# -*- coding: utf-8 -*-
"""
notate.core.core
- Central registry for module types
- Robust resolver for module classes
- Factory to build modules from config
- Model container that injects module registry for cross-module access
"""

from __future__ import annotations
import importlib
import inspect
from typing import Any, Dict, Mapping, MutableMapping, Optional, Type

# ---- central registry (shared with notate.modules.__init__) ----
module_type2class: Dict[str, Type] = {}


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
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
    return "." in name and not name.endswith(".")


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


# ----------------------------------------------------------------------
# Class resolver (ROBUST)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Model container
# ----------------------------------------------------------------------
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
