# src/notate/__init__.py
from importlib import metadata, import_module
import importlib, pkgutil

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["__version__", "Model", "enable_autoregistration"]

def __getattr__(name):
    # 軽量 import は維持
    if name == "Model":
        from .core import Model as _Model
        return _Model
    if name in {"core", "modules", "training", "data", "tools", "downstream"}:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(name)

def enable_autoregistration():
    """
    modules パッケージ以下を再帰的に import して、
    各モジュールの @register(...) の副作用でレジストリを埋める
    """
    pkg_name = f"{__name__}.modules"
    try:
        package = importlib.import_module(pkg_name)
    except ModuleNotFoundError:
        return  # modules が無くても落とさない

    for _, modname, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(modname)
