"""
For compatibility only. use tools.models2 instead.

"""
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.models2.models import check_leftargs, NewGELU, sigmoid, \
    function_config2func, Affine, BatchSecondBatchNorm
from tools.models2.utils import EMPTY

def init_config2func(layer_config):
    if type(layer_config) == str:
        name = layer_config
    elif type(layer_config) in {int, float}:
        name = layer_config
    elif layer_config == {}:
        name = 'none'
    else:
        name = layer_config.name
    if type(name) in {int, float}:
        return lambda x: nn.init.constant_(x, float(name))
    if name == 'glorot_uniform':
        return nn.init.xavier_uniform_
    elif name == 'glorot_normal':
        return nn.init.xavier_normal_
    elif name == 'he_uniform':
        return nn.init.kaiming_uniform_
    elif name == 'he_normal':
        return nn.init.kaiming_normal_
    elif name == 'uniform':
        return lambda x: nn.init.uniform_(x, layer_config.a, layer_config.b)
    elif name == 'normal':
        return lambda x: nn.init.normal_(x, layer_config.mean, layer_config.std)
    elif name in ['zero', 'zeros']:
        return nn.init.zeros_
    elif name in ['one', 'ones']:
        return nn.init.ones_
    elif name == 'none':
        return lambda x: None
    else:
        raise ValueError(f"Unsupported types of init function: {layer_config}")


tunnel_name2maxconfig = {
    'norm': {'args'}, 'layernorm': {'args', 'init'}, 'ln': {'args'},
    'batchnorm': {'args'}, 'bn': {'args'},
    'batchsecond_batchnorm': {'args'}, 'bsbn': {'args'},
    'linear': {'size', 'init', 'args'},
    'function': {'type', 'weight', 'bias'},
    'dropout': {'args'},
    'affine': {'weight', 'bias'},
    'laffine': {'init'}
}

tunnel_name2minconfig = {name: set() for name in tunnel_name2maxconfig.keys()}
tunnel_name2minconfig.update({'linear': {'size'}, 'function': {'type'},
    'affine': {'weight', 'bias'}})
class Tunnel(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.layers = []
        modules = []
        for layer_config in config:
            unknown_args = set(layer_config.keys()) - {'name'} - tunnel_name2maxconfig[layer_config.name]
            if len(unknown_args) > 0:
                raise ValueError(f"Unknown config argument in tunnel ({layer_config.name}): {unknown_args}")
            undefined_args = tunnel_name2minconfig[layer_config.name] - set(layer_config.keys())
            if len(undefined_args) > 0:
                raise ValueError(f"config arguments {undefined_args} is not defined.")
            if layer_config.name in ['norm', 'layernorm', 'ln']:
                layer = nn.LayerNorm(input_size, **layer_config.args)
                init_config2func(layer_config.init.weight)(layer.weight)
                init_config2func(layer_config.init.bias)(layer.bias)
                modules.append(layer)
            elif layer_config.name in ['batchnorm', 'bn']:
                layer = nn.BatchNorm1d(input_size, **layer_config.args)
                init_config2func(layer_config.init.weight)(layer.weight)
                init_config2func(layer_config.init.bias)(layer.bias)
                modules.append(layer)
            elif layer_config.name in ['batchsecond_batchnorm', 'bsbn']:
                layer = BatchSecondBatchNorm(input_size, args=layer_config.args)
                modules.append(layer)
            elif layer_config.name == "linear":
                layer = nn.Linear(input_size, layer_config.size, **layer_config.args)
                init_config2func(layer_config.init.weight)(layer.weight)
                init_config2func(layer_config.init.bias)(layer.bias)
                modules.append(layer)
                input_size = layer_config.size
            elif layer_config.name == "laffine":
                layer = Affine(layer_config.init.weight, layer_config.init.bias, input_size)
                modules.append(layer)
            elif layer_config.name == "affine": # for compatibility
                layer = Affine(layer_config.weight, layer_config.bias, input_size)
                modules.append(layer)
            elif layer_config.name == "function":
                layer = function_config2func(layer_config)
            elif layer_config.name == "dropout":
                layer = nn.Dropout(**layer_config.args)
                modules.append(layer)
            else:
                raise ValueError(f"Unsupported layer_config: {layer_config.name}")
            self.layers.append(layer)
        self.output_size = input_size
        self.modules_ = nn.ModuleList(modules)
    def forward(self, input):
        next_input = input
        for layer in self.layers:
            next_input = layer(next_input)
        return next_input

# modules
class TunnelModule(nn.Module):
    def __init__(self, logger, config, sizes):
        super().__init__()
        self.input = config.input
        self.output = config.output
        self.tunnel = Tunnel(config.tunnel, sizes[self.input])
        sizes[self.output] = self.tunnel.output_size
    def forward(self, batch):
        batch[self.output] = self.tunnel(batch[self.input])
        return batch
