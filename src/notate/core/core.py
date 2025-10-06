"""
Core model components and utilities.

- Weight/init/function registries for config-driven setup
- Auto-registration & lazy resolution of torch.nn.Module classes
- Simple custom module: Affine
- Model: module factory and (de)serialization helpers
"""
import os
import math
import logging
from collections import OrderedDict
from functools import partial
from inspect import signature

import importlib
import inspect
from typing import Any, Dict, Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Debug option
# ---------------------------------------------------------------------
PRINT_PROCESS = False


# ---------------------------------------------------------------------
# Activations / functions
# ---------------------------------------------------------------------
def gelu(x):  # noqa: D401
    """GELU."""
    return F.gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


activation_name2func = {
    "relu": F.relu,
    "gelu": gelu,
    "swish": swish,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "softplus": F.softplus,
    "identity": lambda x: x,
}

# You can register more functions here as needed.
function_name2func: Dict[str, Callable] = {
    "softmax": F.softmax,
    "log_softmax": F.log_softmax,
    "relu": F.relu,
}


def function_config2func(config):
    """Return callable from (str|dict). If dict, expects {'type': name, ...kwargs}."""
    if isinstance(config, str):
        return function_name2func[config]
    cfg = dict(config)
    ftype = cfg.pop("type")
    return partial(function_name2func[ftype], **cfg)


# ---------------------------------------------------------------------
# Initializers
# ---------------------------------------------------------------------
init_type2func = {
    "glorot_uniform": nn.init.xavier_uniform_,
    "glorot_normal": nn.init.xavier_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "normal_": nn.init.normal_,
    "uniform_": nn.init.uniform_,
}


def apply_init_(module: nn.Module, init_cfg: dict, *, logger: logging.Logger = None):
    """Apply param initializer config to module in-place."""
    if not init_cfg:
        return
    init_cfg = dict(init_cfg)
    itype = init_cfg.pop("type", None)
    if not itype:
        return
    if itype not in init_type2func:
        raise KeyError(f"Unknown init type: {itype}")
    fn = init_type2func[itype]
    for name, p in module.named_parameters():
        if p.requires_grad:
            fn(p, **init_cfg)
            if logger:
                logger.debug(f"[init] {module.__class__.__name__}.{name} <- {itype}({init_cfg})")


# ---------------------------------------------------------------------
# Auto registration / import helpers
# ---------------------------------------------------------------------
def _camel_to_snake(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _lazy_import(module_path: str, class_name: str):
    """Import module_path and get class_name, with fallback to name variants."""
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        raise ImportError(f"Failed to import {module_path}") from e

    # direct
    if hasattr(mod, class_name):
        return getattr(mod, class_name)

    # try CamelCase / snake_case variations
    variants = {class_name, class_name.capitalize(), _camel_to_snake(class_name)}
    for attr in variants:
        if hasattr(mod, attr):
            return getattr(mod, attr)
    # search all classes
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if name.lower() == class_name.lower():
            return obj
    raise ImportError(f"Class '{class_name}' not found in {module_path}")


def resolve_module_class(type_name: str):
    """
    Resolve a class object from known namespaces or dotted path.

    - If dotted: import directly.
    - Else: try local 'modules' package and common name variants.
    """
    try:
        import importlib
        importlib.import_module("notate.modules")
    except Exception:
        pass

    if "." in type_name:
        module_path, cls_name = type_name.rsplit(".", 1)
        return _lazy_import(module_path, cls_name)

    # Try local namespace 'modules'
    try:
        return _lazy_import("modules", type_name)
    except Exception:
        pass

    # Try torch.nn
    try:
        return _lazy_import("torch.nn", type_name)
    except Exception:
        pass

    raise ImportError(f"Unable to resolve module class for type='{type_name}'")


def _filter_kwargs_for_class(cls, kwargs: dict, logger: logging.Logger = None, name: str = "") -> dict:
    """Filter kwargs to only those accepted by cls.__init__."""
    sig = signature(cls.__init__)
    valid = set(sig.parameters.keys())
    valid.discard("self")
    uargs = {k: v for k, v in kwargs.items() if k in valid}
    if logger:
        unknown = sorted(set(kwargs.keys()) - set(uargs.keys()))
        if unknown:
            logger.warning(f"[{name}] Unknown kwargs for {cls.__name__}: {unknown}")
    return uargs


def build_module_from_config(name: str, cfg: dict, logger: logging.Logger = None) -> nn.Module:
    """Build a single module from config entry: {'type': ..., ...kwargs}."""
    cfg = dict(cfg)
    mtype = cfg.pop("type")
    cls = resolve_module_class(mtype)
    kwargs = _filter_kwargs_for_class(cls, cfg, logger=logger, name=name)
    module = cls(**kwargs)
    return module


# ---------------------------------------------------------------------
# Simple custom module
# ---------------------------------------------------------------------
class Affine(nn.Module):
    def __init__(self, weight=1.0, bias=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(weight)))
        self.bias = nn.Parameter(torch.tensor(float(bias)))

    def forward(self, input):
        return input * self.weight + self.bias


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class Model(nn.ModuleDict):
    """
    Manage multiple submodules built from config.
    Provides simple forward-processing pipeline, load/save helpers.
    """
    def __init__(
        self,
        logger: logging.Logger,
        modules: dict,
        use_modules: list = None,
        omit_modules: list = None,
        seed: int = None,
        init: dict = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.logger = logger or logging.getLogger(__name__)

        # select subset
        use_modules = list(use_modules) if use_modules else None
        omit_modules = set(omit_modules) if omit_modules else set()

        built = OrderedDict()
        for name, mcfg in modules.items():
            if use_modules is not None and name not in use_modules:
                continue
            if name in omit_modules:
                continue
            module = build_module_from_config(name, mcfg, logger=self.logger)
            built[name] = module

        super().__init__(built)

        # Apply weight init if specified
        if init:
            for name, mod in self.items():
                apply_init_(mod, init, logger=self.logger)

        # (Optional) print module summary
        if PRINT_PROCESS:
            self.logger.debug("[Model] modules: %s", list(self.keys()))

    # -------------------------------
    # (De)serialization helpers
    # -------------------------------
    def load_state_dicts(self, path: str, strict: bool = False):
        """Load state dicts from a directory of {name}.pth"""
        for name, module in self.items():
            p = os.path.join(path, f"{name}.pth")
            if not os.path.exists(p):
                self.logger.warning("Missing state file for %s: %s", name, p)
                continue
            sd = torch.load(p, map_location="cpu")
            result = module.load_state_dict(sd, strict=strict)
            if result.missing_keys:
                self.logger.warning("Missing keys in %s: %s", name, result.missing_keys)
            if result.unexpected_keys:
                self.logger.warning("Unexpected keys in %s: %s", name, result.unexpected_keys)

    def save_state_dicts(self, path: str):
        os.makedirs(path, exist_ok=True)
        for name, module in self.items():
            torch.save(module.state_dict(), os.path.join(path, f"{name}.pth"))

    # (任意で追加) 互換API
    def save_state_dict(self, path: str):
        self.save_state_dicts(path)


# === Added: PipelineModule (config-driven loop executor) ===
import torch
import torch.nn as nn

class PipelineModule(nn.Module):
    """
    Execute config-defined train_loop/val_loop.
    This module acts as a single entry point for ForwardProcess (process.type=forward).
    """
    def __init__(self, train_loop=None, val_loop=None, loss_names=None, logger=None):
        super().__init__()
        self.train_loop = train_loop or []
        self.val_loop   = val_loop or []
        self.loss_names = loss_names or []
        self.logger     = logger
        self._module_registry = None
        self._function_registry = {}

    def attach_registry(self, registry: dict):
        self._module_registry = registry

    def forward(self, input=None, mode="train", **batch):
        if input is None and batch:
            ctx = dict(batch)
        elif isinstance(input, dict):
            ctx = dict(input)
        else:
            ctx = {}
        loop = self.train_loop if mode == "train" else self.val_loop
        for step in loop:
            stype = step.get("type", "module")
            if stype == "function":
                self._run_function_step(ctx, step); continue
            if stype == "iterate":
                self._run_iterate_step(ctx, step); continue
            self._run_module_step(ctx, step)
        losses = {name: ctx[name] for name in self.loss_names if name in ctx}
        return losses if losses else ctx

    # helpers
    def _get_module(self, name: str):
        if not self._module_registry:
            raise RuntimeError("PipelineModule: module registry not attached.")
        if name not in self._module_registry:
            raise KeyError(f"PipelineModule: unknown module '{name}'")
        return self._module_registry[name]

    def _resolve_input(self, ctx, spec):
        if spec is None: return None
        if isinstance(spec, list):
            return [self._resolve_input(ctx, s) for s in spec]
        if isinstance(spec, dict):
            return {k: self._resolve_input(ctx, v) for k, v in spec.items()}
        return ctx[spec] if isinstance(spec, str) else spec

    def _as_kwargs(self, ctx, src):
        if src is None: return {}
        if isinstance(src, dict):
            return {k: self._resolve_input(ctx, v) for k, v in src.items()}
        if isinstance(src, list):
            return {"args": [self._resolve_input(ctx, v) for v in src]}
        return {"input": self._resolve_input(ctx, src)}

    def _write_output(self, ctx, dst, out):
        if dst is None: return
        if isinstance(dst, list):
            if isinstance(out, (list, tuple)) and len(dst) == len(out):
                for k, v in zip(dst, out): ctx[k] = v
            else:
                ctx[dst[0]] = out
        elif isinstance(dst, str):
            ctx[dst] = out
        else:
            raise RuntimeError(f"PipelineModule: invalid output spec: {dst}")

    def _run_module_step(self, ctx, step):
        name = step.get("module")
        if not name: raise RuntimeError(f"PipelineModule: module name missing: {step}")
        mod  = self._get_module(name)
        kwargs = self._as_kwargs(ctx, step.get("input"))
        out = mod(**kwargs, mode=step["mode"]) if "mode" in step else mod(**kwargs)
        self._write_output(ctx, step.get("output"), out)

    def _run_function_step(self, ctx, step):
        spec = step.get("function", {})
        ftype = spec.get("type")
        fn = self._resolve_function(ftype, spec)
        x  = self._resolve_input(ctx, step.get("input"))
        fn_kwargs = {k: v for k, v in spec.items() if k != "type"}
        out = fn(x, **fn_kwargs)
        self._write_output(ctx, step.get("output"), out)

    def _run_iterate_step(self, ctx, step):
        length = self._resolve_input(ctx, step["length"])
        procs  = step["processes"]
        L = int(length)
        for i in range(L):
            ctx["iterate_i"] = i
            for sub in procs:
                if sub.get("type", "module") == "function":
                    self._run_function_step(ctx, sub)
                else:
                    self._run_module_step(ctx, sub)
        ctx.pop("iterate_i", None)

    def _resolve_function(self, ftype, spec):
        if ftype in self._function_registry: return self._function_registry[ftype]
        if ftype == "transpose":
            def _fn(x, dim0=0, **kw):
                dim1 = kw.get("...1", 1)
                return x.transpose(dim0, dim1)
            return _fn
        raise KeyError(f"PipelineModule: unknown function '{ftype}'")


# === Added: registry injection into Model.__init__ ===
try:
    _orig_Model___init__ = Model.__init__
    def _patched_Model___init__(self, *args, **kwargs):
        _orig_Model___init__(self, *args, **kwargs)
        # attach registry (name->module) to all submodules
        try:
            # Inherit nn.ModuleDict interface: self.items() yields (name, module)
            registry = {name: mod for name, mod in self.items()}
            for _name, _mod in registry.items():
                try:
                    setattr(_mod, "_module_registry", registry)
                except Exception:
                    pass
                if hasattr(_mod, "attach_registry") and callable(_mod.attach_registry):
                    try:
                        _mod.attach_registry(registry)
                    except Exception:
                        pass
        except Exception:
            pass
    Model.__init__ = _patched_Model___init__
except Exception:
    # If Model is not defined, do nothing
    pass
