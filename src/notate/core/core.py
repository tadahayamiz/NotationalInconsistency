# src/notate/core/core.py
# STRICT: no dynamic import, no fallback.
# - Unknown module.type -> hard error
# - Unknown kwargs -> hard error
# - Explicit registry only (register_module / resolve_module_class)
# - Model builds submodules from config["modules"] and optional config["use_modules"]

from __future__ import annotations
from typing import Dict, Type, Any, Optional
import logging
import inspect
import torch.nn as nn

# -----------------------------------------------------------------------------
# Registry (explicit only)
# -----------------------------------------------------------------------------
module_type2class: Dict[str, Type[nn.Module]] = {}

def register_module(type_name: str, cls: Type[nn.Module]) -> None:
    if not isinstance(type_name, str) or not type_name:
        raise ValueError("[CONFIG ERROR] register_module: type_name must be a non-empty string")
    if not issubclass(cls, nn.Module):
        raise TypeError("[CONFIG ERROR] register_module: cls must be a subclass of torch.nn.Module")
    prev = module_type2class.get(type_name)
    if prev is not None and prev is not cls:
        raise ValueError(
            f"[CONFIG ERROR] Module type '{type_name}' already registered with {prev.__name__}, "
            f"got conflicting class {cls.__name__}"
        )
    module_type2class[type_name] = cls

def resolve_module_class(type_name: str) -> Type[nn.Module]:
    cls = module_type2class.get(type_name)
    if cls is None:
        sample = list(module_type2class.keys())[:20]
        raise KeyError(
            f"[CONFIG ERROR] Unknown module.type '{type_name}'. "
            f"Registered={len(module_type2class)}; sample={sample}"
        )
    return cls

def _filter_kwargs_strict(cls: Type[nn.Module], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    valid = {p.name for p in sig.parameters.values()
             if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.name != "self"}
    unknown = set(kwargs.keys()) - valid
    if unknown:
        raise TypeError(f"[CONFIG ERROR] Unknown kwargs for {cls.__name__}: {sorted(unknown)}")
    return kwargs

def get_module(logger: Optional[logging.Logger], type: str, **kwargs) -> nn.Module:
    cls = resolve_module_class(type)
    use_kwargs = _filter_kwargs_strict(cls, dict(kwargs))
    if logger:
        logger.debug("Instantiate %s(%s)", cls.__name__, ", ".join(f"{k}={v!r}" for k, v in use_kwargs.items()))
    return cls(**use_kwargs)

def build_module_from_config(name: str, mcfg: Dict[str, Any], logger: Optional[logging.Logger] = None) -> nn.Module:
    if not isinstance(mcfg, dict):
        raise TypeError(f"[CONFIG ERROR] model.modules.{name} must be a mapping")
    if "type" not in mcfg:
        raise KeyError(f"[CONFIG ERROR] model.modules.{name}.type is required")
    mtype = mcfg["type"]
    if not isinstance(mtype, str) or not mtype:
        raise TypeError(f"[CONFIG ERROR] model.modules.{name}.type must be a non-empty string")
    kwargs = {k: v for k, v in mcfg.items() if k != "type"}
    mod = get_module(logger=logger, type=mtype, **kwargs)
    setattr(mod, "__module_name__", name)
    return mod

# -----------------------------------------------------------------------------
# Model scaffold (strict; no monolithic forward)
# -----------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None, **_ignored):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        if not isinstance(config, dict):
            raise TypeError("[CONFIG ERROR] Model(config=...) must be a dict-like structure")

        modules_cfg = config.get("modules")
        if not isinstance(modules_cfg, dict) or not modules_cfg:
            raise ValueError("[CONFIG ERROR] model.modules must be a non-empty mapping")

        built = {}
        for name, mcfg in modules_cfg.items():
            if not isinstance(name, str) or not name:
                raise ValueError("[CONFIG ERROR] module name must be a non-empty string")
            built[name] = build_module_from_config(name, mcfg, logger=self.logger)

        self.components = nn.ModuleDict(built)
        self.use_modules = list(config.get("use_modules", self.components.keys()))
        for n in self.use_modules:
            if n not in self.components:
                raise KeyError(f"[CONFIG ERROR] model.use_modules contains unknown module name '{n}'")

        self.init_cfg = dict(config.get("init", {}))
        self._apply_init_policy()
        self.raw_config = config

        self.logger.debug("[Model] Built modules: %s", list(self.components.keys()))
        self.logger.debug("[Model] use_modules: %s", list(self.use_modules))

    def _apply_init_policy(self) -> None:
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
        try:
            return self.components[name]
        except KeyError:
            raise KeyError(f"[CONFIG ERROR] Unknown component '{name}'")

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Model.forward is intentionally undefined in strict mode. "
            "Use process graphs (get_process) to orchestrate component calls."
        )
