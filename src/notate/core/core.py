"""
Core model components and utilities.

This module provides the main Model class and supporting utilities for
building, configuring, and managing neural network models. It includes:

- Function and initialization registries for configuration-based setup
- Custom modules (Affine transformation, etc.)
- Model class with module management and serialization
- Configuration-to-function conversion utilities

Classes
-------
Model : Main model class managing multiple modules
Affine : Simple affine transformation module

Functions
---------
init_config2func : Convert initialization config to function
function_config2func : Convert function config to callable
get_module : Create module from configuration
"""
import os
import math
import logging
from collections import OrderedDict
from inspect import signature
from functools import partial
from addict import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Option for debug
PRINT_PROCESS = False 

# functions
def NewGELU(input):
    """New GELU activation function implementation."""
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) \
        * (input + 0.044715 * torch.pow(input, 3.0))))

# Weight initialization functions registry
init_type2func = {
    'glorot_uniform': nn.init.xavier_uniform_,
    'glorot_normal': nn.init.xavier_normal_,
    'he_uniform': nn.init.kaiming_uniform_,
    'he_normal': nn.init.kaiming_normal_,
    'zero': nn.init.zeros_,
    'zeros': nn.init.zeros_,
    'one': nn.init.ones_,
    'ones': nn.init.ones_,
    'normal': nn.init.normal_,
    'none': lambda input: None,
}


def init_config2func(type='none', factor=None, **kwargs):
    """
    Create weight initialization function from configuration.
    
    Parameters
    ----------
    type : int, float, str or dict
        Initialization type or configuration dict
    factor : float, optional
        Scaling factor to apply after initialization
    **kwargs
        Additional parameters for the initialization function
        
    Returns
    -------
    function
        Initialization function to apply to parameters
    """
    
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



# Activation and utility functions registry
function_name2func = {
    # Activation functions
    'relu': F.relu,
    'gelu': F.gelu,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'newgelu': NewGELU,
    'softplus': F.softplus,
    'log_softmax': F.log_softmax,
    'none': lambda input: input,
    
    # Utility functions
    'exp': torch.exp,
    'log': torch.log,
    'sum': torch.sum,
    'mean': torch.mean,
    'transpose': torch.transpose,
    'argmax': torch.argmax
}

def function_config2func(config):
    """
    Create function from configuration.
    
    Parameters
    ----------
    config : str or dict
        Function name or configuration dict with 'type' key
        
    Returns
    -------
    function
        The requested function with any additional parameters applied
    """
    if isinstance(config, str):
        return function_name2func[config]
    else:
        func_type = config.pop('type')
        return partial(function_name2func[func_type], **config)

# Module registry and custom modules
module_type2class = {}

class Affine(nn.Module):
    """Simple affine transformation: output = input * weight + bias"""
    def __init__(self, weight=1.0, bias=0.0):
        super().__init__()
        self.weight = weight
        self.bias = bias
    
    def forward(self, input):
        return input * self.weight + self.bias

# Register custom modules
for cls in [Affine]:
    module_type2class[cls.__name__] = cls

def get_module(logger, type, **kwargs):
    """
    Create module instance from type and configuration.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger for warnings
    type : str
        Module type name
    **kwargs
        Module configuration parameters
        
    Returns
    -------
    nn.Module
        Initialized module instance
    """
    cls = module_type2class[type]
    args = set(signature(cls.__init__).parameters.keys())
    uargs = {}
    for key in list(kwargs.keys()):
        if key in args:
            uargs[key] = kwargs.pop(key)
    if len(kwargs) > 0:
        logger.warning(f"Unknown kwarg in {cls.__name__}: {kwargs}")
    return cls(**uargs)

class Model(nn.ModuleDict):
    """
    Main model class that manages multiple modules and their execution.
    
    This class builds a collection of modules from configuration and provides
    forward pass processing through a series of configurable processes.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger for debugging and warnings
    modules : dict
        Dictionary of module configurations
    use_modules : list, optional
        List of module names to include (mutually exclusive with omit_modules)
    omit_modules : list, optional
        List of module names to exclude (mutually exclusive with use_modules)
    seed : int, optional
        Random seed for reproducible module initialization
    init : dict, optional
        Parameter initialization configuration
    """
    def __init__(self, logger: logging.Logger, modules: dict, use_modules: list = None,
                 omit_modules: list = None, seed: int = None, init: dict = None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        if (use_modules is not None) and (omit_modules is not None):
            raise ValueError("Please specify either use_modules or omit_modules, not both")
        
        # Build modules based on configuration
        mods = OrderedDict()
        for mod_name, mod_config in modules.items():
            if (use_modules is not None and mod_name not in use_modules) or \
               (omit_modules is not None and mod_name in omit_modules):
                continue
            logger.debug(f"Building {mod_name}...")
            mods[mod_name] = get_module(logger=logger, **mod_config)
        
        logger.debug("Building finished.")
        super().__init__(modules=mods)
        self.logger = logger

        # Apply parameter initialization
        if init:
            self._apply_initialization(init)

    def _apply_initialization(self, init_config: dict):
        """Apply parameter initialization based on configuration."""
        for name, param in self.state_dict().items():
            for pattern, config in init_config.items():
                if pattern in name:
                    init_config2func(config)(param)

    def forward(self, batch: dict, processes: list):
        """
        Execute forward pass through a series of processes.
        
        Parameters
        ----------
        batch : dict
            Input batch containing data and metadata
        processes : list
            List of process objects to execute sequentially
            
        Returns
        -------
        dict
            Updated batch after processing
        """
        for i, process in enumerate(processes):
            if PRINT_PROCESS:
                print(f"-----process {i}-----")
                for key, value in batch.items():
                    if isinstance(value, (torch.Tensor, np.ndarray)):
                        print(f"  {key}: {list(value.shape)}")
                    else:
                        print(f"  {key}: {type(value).__name__}")
            process(self, batch)
        return batch
    
    def load(self, path: str, replace: dict = None, strict: bool = True):
        """
        Load model state from file or directory.
        
        Parameters
        ----------
        path : str
            Path to state dict file (.pth) or directory containing module files
        replace : dict, optional
            Dictionary for key name replacement during loading
        strict : bool, default True
            Whether to strictly enforce state dict key matching
        """
        replace = replace or {}
        
        if os.path.isfile(path):
            self._load_from_file(path, replace, strict)
        elif os.path.isdir(path):
            self._load_from_directory(path, replace, strict)
        else:
            if os.path.exists(path):
                raise ValueError(f"Invalid file: {path}")
            else:
                raise FileNotFoundError(f"No such file or directory: {path}")
    
    def _load_from_file(self, path: str, replace: dict, strict: bool):
        """Load state from a single file."""
        device = list(self.parameters())[0].device
        state_dict = torch.load(path, map_location=device)
        
        # Apply key replacements
        for key, replacement in replace.items():
            for sdict_key, sdict_value in list(state_dict.items()):
                if sdict_key.startswith(key):
                    new_key = replacement + sdict_key[len(key):]
                    state_dict[new_key] = sdict_value
                    del state_dict[sdict_key]
        
        # Load and report issues
        result = self.load_state_dict(state_dict, strict=strict)
        self._report_loading_issues(result)
    
    def _load_from_directory(self, path: str, replace: dict, strict: bool):
        """Load state from directory containing module files."""
        replace_inverse = {value: key for key, value in replace.items()}
        device = list(self.parameters())[0].device
        
        for mname, module in self.items():
            original_name = replace_inverse.get(mname, mname)
            mpath = f"{path}/{original_name}.pth"
            
            if os.path.exists(mpath):
                result = module.load_state_dict(
                    torch.load(mpath, map_location=device), strict=strict
                )
                self._report_loading_issues(result, module_name=mname)
            elif strict:
                raise ValueError(f"State dict file of {mname} does not exist.")
            else:
                self.logger.warning(f"State dict file of {mname} does not exist.")
    
    def _report_loading_issues(self, result, module_name: str = None):
        """Report missing and unexpected keys during loading."""
        prefix = f"in {module_name} " if module_name else ""
        
        if len(result.missing_keys) > 0:
            self.logger.warning(f"Missing keys {prefix}:")
            for key in result.missing_keys:
                self.logger.warning(f"  {key}")
        
        if len(result.unexpected_keys) > 0:
            self.logger.warning(f"Unexpected keys {prefix}:")
            for key in result.unexpected_keys:
                self.logger.warning(f"  {key}")    
            
    def save_state_dict(self, path: str):
        """
        Save model state to directory.
        
        Each module's state dict is saved as a separate .pth file
        in the specified directory.
        
        Parameters
        ----------
        path : str
            Directory path to save module state dicts
        """
        os.makedirs(path, exist_ok=True)
        for key, module in self.items():
            torch.save(module.state_dict(), os.path.join(path, f"{key}.pth"))