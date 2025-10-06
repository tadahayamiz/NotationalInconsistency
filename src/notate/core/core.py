# src/notate/core/core.py
# Strict mode: no lazy (dynamic) resolution, no silent fallbacks.
# - Unknown module "type" -> hard error
# - Unknown kwargs for a module -> hard error
# - Keeps a simple, explicit registry that other files can populate via `register_module`
# - Model builds submodules from config["modules"] (dict) and optional config["use_modules"] (list)
# - No variable expansion here (handled at scripts/train.py)

from __future__ import annotations
from typing import Dict, Type, Any, Optional
import logging
import inspect
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

module_type2class: Dict[str, Type[nn.Module]] = {}

def register_module(type_name: str, cls: Type[nn.Module]) -> None:
    """
    Explicitly register a module class under a string key.
    Re-registering with a different class raises, identical is a no-op.
    """
    if not isinstance(type_name, str) or not type_name:
        raise ValueError("[CONFIG ERROR] register_module: type_name must be a non-empty string")
    if not issubclass(cls, nn.Module):
        raise TypeError("[CONFIG ERROR] register_module: cls must be a subclass of torch.nn.Module")
    prev = module_type2class.get(type_name)
    if prev is not None and prev is not cls:
        raise ValueError(f"[CONFIG ERROR] Module type '{type_name}' already registered with {prev.__name__}, "
                         f"got conflicting class {cls.__name__}")
    module_type2class[type_name] = cls


def resolve_module_class(type_name: str) -> Type[nn.Module]:
    """
    Strict resolver: only uses the explicit registry.
    No dynamic import, no guessing.
    """
    cls = module_type2class.get(type_name)
    if cls is None:
        sample = list(module_type2class.keys())[:20]
        raise KeyError(
            f"[CONFIG ERROR] Unknown module.type '{type_name}'. "
            f"Registered={len(module_type2class)}; sample={sample}"
        )
    return cls


def _filter_kwargs_strict(cls: Type[nn.Module], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate kwargs against __init__ signature of the class.
    - Any key not present in the signature -> hard error
    - We do not auto-fill defaults or coerce types here; ctor will enforce requireds.
    """
    sig = inspect.signature(cls.__init__)
    valid_names = {p.name for p in sig.parameters.values()
                   if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.name != "self"}
    unknown = set(kwargs.keys()) - valid_names
    if unknown:
        raise TypeError(f"[CONFIG ERROR] Unknown kwargs for {cls.__name__}: {sorted(unknown)}")
    return kwargs


def get_module(logger: Optional[logging.Logger], type: str, **kwargs) -> nn.Module:
    """
    Create a module from a registered type and kwargs. Strict:
    - No lazy resolution
    - Unknown kwargs -> error (not warning)
    """
    cls = resolve_module_class(type)
    use_kwargs = _filter_kwargs_strict(cls, dict(kwargs))
    if logger:
        logger.debug("Instantiate %s(%s)", cls.__name__, ", ".join(f"{k}={v!r}" for k, v in use_kwargs.items()))
    return cls(**use_kwargs)


def build_module_from_config(name: str, mcfg: Dict[str, Any], logger: Optional[logging.Logger] = None) -> nn.Module:
    """
    Build a module from a config mapping:
      { "type": "<RegisteredType>", ...other kwargs... }
    """
    if not isinstance(mcfg, dict):
        raise TypeError(f"[CONFIG ERROR] model.modules.{name} must be a mapping")
    if "type" not in mcfg:
        raise KeyError(f"[CONFIG ERROR] model.modules.{name}.type is required")

    mtype = mcfg["type"]
    if not isinstance(mtype, str) or not mtype:
        raise TypeError(f"[CONFIG ERROR] model.modules.{name}.type must be a non-empty string")

    # strict: don't mutate original dict
    kwargs = {k: v for k, v in mcfg.items() if k != "type"}
    mod = get_module(logger=logger, type=mtype, **kwargs)
    # (optional) attach name for debugging
    setattr(mod, "__module_name__", name)
    return mod

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class Model(nn.Module):
    """
    Minimal strict model scaffold that:
      - builds submodules defined in config["modules"] (dict name -> spec)
      - optionally orders / exposes a subset via config["use_modules"] (list of names)
      - provides a ModuleDict `components` (all built modules)
    Notes:
      * This class does not implement a specific forward pass.
        The training/validation processes (get_process graphs) are expected to
        orchestrate calls to the components explicitly.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None, **_ignored):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        if not isinstance(config, dict):
            raise TypeError("[CONFIG ERROR] Model(config=...) must be a dict-like structure")

        modules_cfg = config.get("modules")
        if not isinstance(modules_cfg, dict) or not modules_cfg:
            raise ValueError("[CONFIG ERROR] model.modules must be a non-empty mapping")

        # Build every declared module strictly
        built = {}
        for name, mcfg in modules_cfg.items():
            if not isinstance(name, str) or not name:
                raise ValueError("[CONFIG ERROR] module name must be a non-empty string")
            mod = build_module_from_config(name, mcfg, logger=self.logger)
            built[name] = mod

        # Store as ModuleDict (keeps registration with nn.Module)
        self.components = nn.ModuleDict(built)

        # Optional: an ordered list of module names used by downstream process graphs
        self.use_modules = list(config.get("use_modules", self.components.keys()))
        # Basic validation: names in use_modules must exist
        for n in self.use_modules:
            if n not in self.components:
                raise KeyError(f"[CONFIG ERROR] model.use_modules contains unknown module name '{n}'")

        # Optional: initialization policy, seeds or other meta
        self.init_cfg = dict(config.get("init", {}))
        self._apply_init_policy()

        # Store raw config if needed by hooks or savers
        self.raw_config = config

        self.logger.debug("[Model] Built modules: %s", list(self.components.keys()))
        self.logger.debug("[Model] use_modules: %s", list(self.use_modules))

    # -------------------------------------------------------------------------
    # Initialization helpers
    # -------------------------------------------------------------------------
    def _apply_init_policy(self) -> None:
        """
        Apply a basic initialization policy if requested.
        Extend here only with explicit, deterministic behaviors.
        """
        itype = str(self.init_cfg.get("type", "default")).lower()
        if itype in ("", "default", "none"):
            return
        elif itype == "xavier_uniform":
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
        elif itype == "kaiming_normal":
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
        else:
            raise ValueError(f"[CONFIG ERROR] Unknown init.type '{self.init_cfg.get('type')}'")

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------
    def get(self, name: str) -> nn.Module:
        """Return a built component by name (strict)."""
        try:
            return self.components[name]
        except KeyError:
            raise KeyError(f"[CONFIG ERROR] Unknown component '{name}'")

    def forward(self, *args, **kwargs):
        """
        This scaffold intentionally avoids defining a monolithic forward.
        The pipeline is expected to call specific components in configured order.
        """
        raise NotImplementedError(
            "Model.forward is intentionally undefined in strict mode. "
            "Use process graphs (get_process) to orchestrate component calls."
        )

# -----------------------------------------------------------------------------
# Convenience: one-shot registration import hook (optional)
# -----------------------------------------------------------------------------
# If you prefer to populate the registry here explicitly, you can import
# concrete classes and call `register_module("TypeName", Class)` below.
# For example:
#
# from notate.modules.sequence import TransformerEncoder, TransformerDecoder, GreedyDecoder
# from notate.modules.poolers import Pooler
# from notate.modules.tunnel import GaussianSampler
# from notate.modules.vae import Linear, LinearClassifier, PositionalEmbedding, TeacherForcer, SequenceMasker
#
# register_module("TransformerEncoder", TransformerEncoder)
# register_module("TransformerDecoder", TransformerDecoder)
# register_module("GreedyDecoder", GreedyDecoder)
# register_module("Pooler", Pooler)
# register_module("GaussianSampler", GaussianSampler)
# register_module("Linear", Linear)
# register_module("LinearClassifier", LinearClassifier)
# register_module("PositionalEmbedding", PositionalEmbedding)
# register_module("TeacherForcer", TeacherForcer)
# register_module("SequenceMasker", SequenceMasker)
#
# …etc.
#
# Alternatively, each module file can self-register on import to keep concerns local.
