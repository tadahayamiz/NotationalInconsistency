import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev" # fallback if package is not installed

# Import main components
from .core import *
from .training import *
from .data import *
from .modules import *
from .tools import *

# For backward compatibility, expose Model at top level
from .core import Model

__all__ = [
    # Core
    'Model', 'module_type2class',
    
    # Data
    'get_dataloader', 'get_dataset', 'get_accumulator',
    
    # Training  
    'get_optimizer', 'get_scheduler', 'get_metric', 'get_process', 'get_hook',
    
    # Tools
    'default_logger', 'load_config2', 'make_result_dir',
]