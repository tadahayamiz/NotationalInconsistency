#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/modules/tunnel.py

Purpose:
    Generic "tunnel" (feed-forward) builder that stacks simple operations
    (reshape/view, slice, squeeze, (Layer)Norm/BatchNorm, Linear, Dropout,
     function callables, and affine transforms).
    Also provides a lightweight scalar Affine *module* used in configs like
    `-d_kl_factor`, and a parameterized Affine *layer* used inside Tunnel.

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Imports ordered: stdlib -> third-party -> local (project).
    - No behavior-changing edits were made intentionally; obvious bugs fixed:
        * BatchSecondBatchNorm now returns `transpose(0, 1)` (was bare transpose()).
        * `squeeze` branch only rewrites `input_size` when dim is not None.
        * `slice` config uses a concrete tuple of slices (not a consumed generator).

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Third-party =====
import torch
import torch.nn as nn

# ===== Project-local =====
from ..core import init_config2func, function_config2func, register_module


# =============================================================================
# Layers and helpers
# =============================================================================
class Affine(nn.Module):
    """Parameterized per-feature affine: y = x * weight + bias.

    Notes:
        This class is used as a *layer* inside Tunnel (config.type == 'affine'
        or 'laffine'), where input_size is known and we create per-feature
        learnable parameters.
    """

    def __init__(self, weight, bias, input_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.full((input_size,), float(weight)))
        self.bias = nn.Parameter(torch.full((input_size,), float(bias)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.weight + self.bias


class BatchSecondBatchNorm(nn.Module):
    """BatchNorm1d applied over the second dimension after [T, B, F] -> [B, T, F]."""

    def __init__(self, input_size: int, args):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features=input_size, **args)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Expecting [T, B, F] -> normalize over B (second dim after transpose)
        input = input.transpose(0, 1)  # [B, T, F]
        input = self.norm(input)
        return input.transpose(0, 1)  # back to [T, B, F]


def get_layer(config, input_size):
    """Return (callable_or_module, new_input_size) for a single layer config."""
    if config.type == "view":
        new_shape = []
        for size in config.shape:
            if size == "batch_size":
                if -1 in new_shape:
                    raise AssertionError(f"Invalid config.shape: {config.shape}")
                size = -1
            new_shape.append(size)
        layer = lambda x: x.view(*new_shape)
        input_size = config.shape

    elif config.type == "slice":
        slices = tuple(slice(*slice0) for slice0 in config.slices)
        layer = lambda x: x[slices]
        # best-effort shape update when sizes are known (ints)
        for dim, s in enumerate(slices):
            if isinstance(input_size[dim], int):
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else input_size[dim]
                step = s.step if s.step is not None else 1
                input_size[dim] = (stop - start) // step

    elif config.type == "squeeze":
        config.setdefault("dim", None)
        layer = lambda x: torch.squeeze(x, dim=config.dim)
        if config.dim is None:
            input_size = [s for s in input_size if s != 1]
        else:
            if input_size[config.dim] != 1:
                raise ValueError(
                    f"{config.dim} th dim of size {input_size} is not squeezable."
                )
            size = list(input_size)[: config.dim] + list(input_size[config.dim + 1 :])
            input_size = size

    elif config.type in ["norm", "layernorm", "ln"]:
        layer = nn.LayerNorm(input_size[-1], **config.args)
        init_config2func(config.init.weight)(layer.weight)
        init_config2func(config.init.bias)(layer.bias)

    elif config.type in ["batchnorm", "bn"]:
        layer = nn.BatchNorm1d(input_size[-1], **config.args)
        init_config2func(config.init.weight)(layer.weight)
        init_config2func(config.init.bias)(layer.bias)

    elif config.type in ["batchsecond_batchnorm", "bsbn"]:
        layer = BatchSecondBatchNorm(input_size[-2], args=config.args)

    elif config.type == "linear":
        layer = nn.Linear(input_size[-1], config.size, **config.args)
        init_config2func(config.init.weight)(layer.weight)
        init_config2func(config.init.bias)(layer.bias)
        input_size = input_size[:-1] + [config.size]

    elif config.type == "laffine":
        # learnable per-feature affine using init-config
        layer = Affine(config.init.weight, config.init.bias, input_size[-1])

    elif config.type == "affine":  # compatibility (weight/bias provided directly)
        layer = Affine(config.weight, config.bias, input_size[-1])

    elif config.type == "function":
        layer = function_config2func(config.function)

    elif config.type == "dropout":
        layer = nn.Dropout(**config.args)

    else:
        raise ValueError(f"Unsupported config: {config.type}")

    return layer, input_size


class Layer(nn.Module):
    """Single Tunnel layer wrapper to keep nn.Module containers consistent."""

    def __init__(self, layer, input_size):
        super().__init__()
        self.layer, _ = get_layer(layer, input_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layer(input)


class Tunnel(nn.Module):
    """Composable stack of simple layers defined by a list of configs."""

    def __init__(self, layers, input_size):
        super().__init__()
        self.layers = []
        modules = []
        cur_size = list(input_size)

        for i_layer, layer_config in enumerate(layers):
            layer, cur_size = get_layer(layer_config, cur_size)
            self.layers.append(layer)
            if isinstance(layer, nn.Module):
                modules.append(layer)

        # Register only the stateful sublayers; callables are kept in self.layers
        self.modules_ = nn.ModuleList(modules)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# Module-level Scalar Affine
# =============================================================================
class ScalarAffine(nn.Module):
    """Module-level scalar affine: y = x * weight + bias (no parameters)."""

    def __init__(self, weight: float, bias: float = 0.0):
        super().__init__()
        self.weight = float(weight)
        self.bias = float(bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.weight + self.bias


# =============================================================================
# Registry bindings (explicit)
# =============================================================================
# Config `type: Tunnel` should map to this builder.
register_module("Tunnel", Tunnel)
# Config `type: Affine` (e.g., `-d_kl_factor`) expects a simple scalar affine.
register_module("Affine", ScalarAffine)
