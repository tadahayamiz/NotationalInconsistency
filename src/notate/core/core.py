"""
Core model components and utilities.

- Weight/init/function registries for config-driven setup
- Auto-registration of torch.nn.Module classes under notate.modules.*
- Simple custom module: Affine
- Model: module factory, (de)serialization helpers
"""
import os
import math
import logging
from collections import OrderedDict
from functools import partial
from inspect import signature

import importlib
import inspect
import re

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
def NewGELU(x: torch.Tensor) -> torch.Tensor:
    """New GELU activation."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

function_name2func = {
    # activations
    "relu": F.relu,
    "gelu": F.gelu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "newgelu": NewGELU,
    "softplus": F.softplus,
    "log_softmax": F.log_softmax,
    "none": lambda x: x,
    # utilities
    "exp": torch.exp,
    "log": torch.log,
    "sum": torch.sum,
    "mean": torch.mean,
    "transpose": torch.transpose,
    "argmax": torch.argmax,
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
    "he_uniform": nn.init.kaiming_uniform_,
    "he_normal": nn.init.kaiming_normal_,
    "zero": nn.init.zeros_,
    "zeros": nn.init.zeros_,
    "one": nn.init.ones_,
    "ones": nn.init.ones_,
    "normal": nn.init.normal_,
    "none": lambda _: None,
}

def init_config2func(type="none", factor=None, **kwargs):
    """
    Make an initializer from (str|number|dict).
    If factor is set, multiply parameters by factor after init.
    """
    if factor is not None:
        def init_fn(param: nn.Parameter):
            init_config2func(type, **kwargs)(param)
            param.data = param.data * factor
        return init_fn

    if isinstance(type, dict):
        return init_config2func(**type)

    if isinstance(type, (int, float)):
        return lambda p: nn.init.constant_(p, float(type))
    if type in init_type2func:
        return lambda p: init_type2func[type](p, **kwargs)
    raise ValueError(f"Unsupported init type: {type}")


# ---------------------------------------------------------------------
# Module registry (legacy + auto)
# ---------------------------------------------------------------------
module_type2class = {}

class Affine(nn.Module):
    """output = input * weight + bias"""
    def __init__(self, weight=1.0, bias=0.0):
        super().__init__()
        self.weight = weight
        self.bias = bias
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias

# keep legacy registrations
module_type2class["Affine"] = Affine

def _to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

def _collect_module_classes():
    """
    Collect torch.nn.Module subclasses from notate.modules.* and
    register both 'CamelCase' and 'snake_case' keys.
    """
    candidates = [
        "notate.modules.poolers",
        "notate.modules.tunnel",
        "notate.modules.vae",
        "notate.modules.sequence",
        # downstream depends on optuna; skip here
    ]
    reg = {}
    for modname in candidates:
        try:
            m = importlib.import_module(modname)
        except Exception:
            continue
        for cls_name, cls in inspect.getmembers(m, inspect.isclass):
            try:
                if issubclass(cls, nn.Module):
                    reg[cls_name] = cls
                    reg[_to_snake(cls_name)] = cls
            except Exception:
                pass
    return reg

# merge auto-registrations
module_type2class.update(_collect_module_classes())

logging.getLogger(__name__).info(
    "[core] module_type2class registered: %d modules", len(module_type2class)
)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
def get_module(logger: logging.Logger, type: str, **kwargs) -> nn.Module:
    """
    Create module instance from type and configuration.
    Unknown kwargs are warned and ignored.
    """
    if type not in module_type2class:
        raise KeyError(f"Module type '{type}' not found "
                       f"(registered={len(module_type2class)}).")
    cls = module_type2class[type]
    valid = set(signature(cls.__init__).parameters.keys())
    uargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in valid}
    if kwargs:
        logger.warning("Unknown kwarg in %s: %s", cls.__name__, kwargs)
    return cls(**uargs)


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
            torch.cuda.manual_seed(seed)

        if use_modules is not None and omit_modules is not None:
            raise ValueError("Specify either use_modules or omit_modules, not both.")

        # build modules
        mods = OrderedDict()
        for name, cfg in modules.items():
            if use_modules is not None and name not in use_modules:
                continue
            if omit_modules is not None and name in omit_modules:
                continue
            logger.debug("Building module: %s", name)
            mods[name] = get_module(logger=logger, **cfg)

        super().__init__(modules=mods)
        self.logger = logger

        # optional init
        if init:
            self._apply_initialization(init)

    def _apply_initialization(self, init_config: dict):
        """Apply parameter initialization based on config (pattern match)."""
        for name, tensor in self.state_dict().items():
            for pattern, cfg in init_config.items():
                if pattern in name:
                    init_config2func(cfg)(tensor)

    def forward(self, batch: dict, processes: list):
        """
        Run a sequence of 'process' callables on (self, batch).
        Each process is assumed to have signature process(model, batch).
        """
        for i, proc in enumerate(processes):
            if PRINT_PROCESS:
                print(f"-----process {i}-----")
                for k, v in batch.items():
                    if isinstance(v, (torch.Tensor, np.ndarray)):
                        print(f"  {k}: {list(v.shape)}")
                    else:
                        print(f"  {k}: {type(v).__name__}")
            proc(self, batch)
        return batch

    # ------------------ I/O helpers ------------------
    def load(self, path: str, replace: dict = None, strict: bool = True):
        replace = replace or {}
        if os.path.isfile(path):
            self._load_from_file(path, replace, strict)
        elif os.path.isdir(path):
            self._load_from_directory(path, replace, strict)
        else:
            raise FileNotFoundError(path)

    def _load_from_file(self, path: str, replace: dict, strict: bool):
        device = next(self.parameters()).device
        state = torch.load(path, map_location=device)

        # key replacements
        if replace:
            for old, new in list(replace.items()):
                for k in list(state.keys()):
                    if k.startswith(old):
                        state[new + k[len(old):]] = state[k]
                        del state[k]

        result = self.load_state_dict(state, strict=strict)
        self._report_loading_issues(result)

    def _load_from_directory(self, path: str, replace: dict, strict: bool):
        device = next(self.parameters()).device
        inv = {v: k for k, v in (replace or {}).items()}
        for mname, module in self.items():
            oname = inv.get(mname, mname)
            mpath = os.path.join(path, f"{oname}.pth")
            if os.path.exists(mpath):
                result = module.load_state_dict(torch.load(mpath, map_location=device), strict=strict)
                self._report_loading_issues(result, module_name=mname)
            elif strict:
                raise ValueError(f"State dict file of {mname} does not exist.")
            else:
                self.logger.warning("State dict file of %s does not exist.", mname)

    def _report_loading_issues(self, result, module_name: str = None):
        prefix = f"in {module_name} " if module_name else ""
        if result.missing_keys:
            self.logger.warning("Missing keys %s:", prefix)
            for k in result.missing_keys:
                self.logger.warning("  %s", k)
        if result.unexpected_keys:
            self.logger.warning("Unexpected keys %s:", prefix)
            for k in result.unexpected_keys:
                self.logger.warning("  %s", k)

    def save_state_dict(self, path: str):
        os.makedirs(path, exist_ok=True)
        for name, module in self.items():
            torch.save(module.state_dict(), os.path.join(path, f"{name}.pth"))
