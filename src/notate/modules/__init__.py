#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/modules/__init__.py

Purpose:
    Unified registry and import point for neural network modules used in `notate`:
      - Transformer-based components
      - VAE modules
      - Poolers
      - Tunnels
      - Minimal constant Affine scaling (for factors like -d_kl_factor)

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Explicitly registers all modules into the shared `module_type2class` registry
      imported from `notate.core.core`.
    - No behavior-changing edits were made intentionally.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Import module groups =====
from .sequence import *
from .vae import *
from .tunnel import *
from .poolers import *

# ===== Shared registry =====
from ..core.core import module_type2class

# ===== Register common torch losses =====
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
# Minimal constant Affine (for scalar/tensor-safe scaling)
# =============================================================================
class Affine(nn.Module):
    """Constant scalar affine transform.

    Notes:
        y = x * weight + bias
        - weight, bias are floats (non-trainable)
        - Broadcasts safely for arbitrary-shaped tensors
        - Used for fixed scaling (e.g., -d_kl_factor)
    """

    def __init__(self, weight: float = 1.0, bias: float = 0.0):
        super().__init__()
        self.weight = float(weight)
        self.bias = float(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


# Register the scalar Affine
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
    # Constant Affine
    "Affine",
]
