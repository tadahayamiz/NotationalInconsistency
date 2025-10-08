#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/modules/__init__.py

Purpose:
    Unified entry point for neural network modules in the `notate` package.
    This file imports all submodules (sequence, vae, tunnel, poolers),
    registers them in the global `module_type2class` registry (shared with core),
    and exposes them through __all__.

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Includes a lightweight non-trainable scalar Affine module for safe
      scaling (used in configs like `-d_kl_factor`).
    - No behavior-changing edits were made intentionally.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Import concrete module groups =====
from .sequence import *
from .vae import *
from .tunnel import *
from .poolers import *

# ===== Central registry shared with core =====
from ..core.core import module_type2class

# ===== Register common PyTorch losses (convenience) =====
import torch
import torch.nn as nn

for cls in [nn.MSELoss, nn.BCEWithLogitsLoss]:
    module_type2class[cls.__name__] = cls

# ===== Register sequence / transformer modules =====
for cls in [
    TeacherForcer,
    MaskMaker,
    SelfAttentionLayer,
    PositionalEmbedding,
    TransformerEncoder,
    TransformerDecoder,
    AttentionDecoder,
    TransformerLMDecoder,
    GreedyDecoder,
    CrossEntropyLoss,
    BCELoss,
    MLP,
]:
    module_type2class[cls.__name__] = cls

# ===== Register VAE modules =====
for cls in [VAE, MinusD_KLLoss, Random]:
    module_type2class[cls.__name__] = cls

# ===== Register tunnel modules =====
for cls in [Layer, Tunnel]:
    module_type2class[cls.__name__] = cls

# ===== Register pooler modules =====
for cls in [
    MeanPooler,
    StartPooler,
    MaxPooler,
    MeanStartMaxPooler,
    MeanStartEndMaxPooler,
    MeanStdStartEndMaxMinPooler,
    NoAffinePooler,
    NemotoPooler,
    GraphPooler,
]:
    module_type2class[cls.__name__] = cls

# =============================================================================
# Minimal non-trainable scalar Affine
# =============================================================================
class Affine(nn.Module):
    """Scalar affine transform (non-trainable).

    Notes:
        y = x * weight + bias
        - weight, bias are plain floats (not Parameters)
        - safely broadcastable across any tensor shape
        - used for fixed scaling (e.g., -d_kl_factor)
    """

    def __init__(self, weight: float = 1.0, bias: float = 0.0):
        super().__init__()
        self.weight = float(weight)
        self.bias = float(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


# Explicitly register scalar Affine
module_type2class["Affine"] = Affine

# =============================================================================
# Public exports
# =============================================================================
__all__ = [
    # Sequence / Transformer modules
    "TeacherForcer",
    "MaskMaker",
    "SelfAttentionLayer",
    "PositionalEmbedding",
    "TransformerEncoder",
    "TransformerDecoder",
    "AttentionDecoder",
    "TransformerLMDecoder",
    "GreedyDecoder",
    "CrossEntropyLoss",
    "BCELoss",
    "MLP",
    # VAE modules
    "VAE",
    "MinusD_KLLoss",
    "Random",
    # Tunnel modules
    "Layer",
    "Tunnel",
    # Pooler modules
    "MeanPooler",
    "StartPooler",
    "MaxPooler",
    "MeanStartMaxPooler",
    "MeanStartEndMaxPooler",
    "MeanStdStartEndMaxMinPooler",
    "NoAffinePooler",
    "NemotoPooler",
    "GraphPooler",
    # Scalar Affine
    "Affine",
]
