import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev" # fallback if package is not installed

# Import all modules to ensure registries are populated
from . import core
from . import modules
from . import training
from . import data
from . import tools
from . import downstream

# Import main public API
from .core import Model

__all__ = [
    'Model',
    '__version__',
]