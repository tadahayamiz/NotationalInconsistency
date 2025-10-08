#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/modules/vae.py

Purpose:
    Variational Autoencoder (latent sampler) and its KL loss.
    Includes a simple Random module for generating vectors correlated to input.

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Imports ordered: stdlib -> third-party -> local (project).
    - No behavior-changing edits were made intentionally.
      (The Random module retains its original device behavior: returns on "cuda:0".)

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Third-party =====
import numpy as np
import torch
import torch.nn as nn

# ===== Project-local =====
from ..core import register_module


class VAE(nn.Module):
    """Latent sampler for VAE with optional evaluation (non-noisy) mode."""

    name = "vae"

    def __init__(self, var_coef: float = 1.0, eval_vae: bool = False):
        """Initialize VAE sampler.

        Args:
            var_coef: Scales the sampling noise (std) during training.
            eval_vae: If True, always behave as evaluation (use mu only).
        """
        super().__init__()
        self.var_coef = var_coef
        self.eval_vae = eval_vae

        # Add _device_param for device-safe buffer loading in state_dict
        self._device_param = nn.Parameter(torch.zeros((0,)))

        def hook(
            model,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            key = prefix + "_device_param"
            if key not in state_dict:
                state_dict[key] = model._device_param

        self._register_load_state_dict_pre_hook(hook, with_module=True)

    @property
    def device(self):
        """Current device derived from the internal parameter."""
        return self._device_param.device

    def forward(
        self,
        mode: str = "train",
        mu: torch.Tensor | None = None,
        var: torch.Tensor | None = None,
        latent_size: int | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Sample or generate latent vectors.

        Args:
            mode: "train", "eval", or "generate".
            mu: Mean tensor [B, D] when sampling.
            var: Variance tensor [B, D] when sampling.
            latent_size: Used only in "generate" mode.
            batch_size: Used only in "generate" mode.

        Returns:
            Latent tensor [B, D].
        """
        if mode == "generate":
            return torch.randn(size=(batch_size, latent_size), device=self.device)
        # Sample if training or eval_vae=True, otherwise return mu (deterministic)
        if mode == "train" or self.eval_vae:
            return mu + torch.randn(*mu.shape, device=mu.device) * torch.sqrt(var) * self.var_coef
        return mu


class MinusD_KLLoss(nn.Module):
    """KL divergence term for a diagonal Gaussian prior (negative D_KL)."""

    def __init__(self):
        super().__init__()

    def forward(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        # 0.5 * sum(mu^2 + var - log var - 1) with '-1' expanded to total elements
        return 0.5 * (torch.sum(mu**2) + torch.sum(var) - torch.sum(torch.log(var)) - var.numel())


class Random(nn.Module):
    """Generate vectors with a specified correlation to each input row.

    Notes:
        This module uses NumPy to sample, then returns a CUDA tensor on "cuda:0"
        (original behavior preserved). It assumes `mu` is on CUDA and that
        "cuda:0" exists in the environment.
    """

    def __init__(self):
        super().__init__()

    # --- helpers (NumPy space) ---
    def generate_random_with_variance(self, size, variance):
        """Generate N(0, variance) random numbers with given size (NumPy)."""
        mean = 0.0
        std_dev = 1.0
        rnd = np.random.normal(mean, std_dev, size)
        scaling = np.sqrt(variance)
        return rnd * scaling

    def generate_correlated_vectors(self, reference_row, random_numbers, correlation):
        """Create a vector with the given correlation to reference_row (NumPy)."""
        x = reference_row
        y = random_numbers
        z = correlation * x + (1 - correlation**2) ** 0.5 * y
        return z

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """Return a tensor whose rows are correlated to mu's rows (corr=1.0)."""
        target_correlation = 1.0  # preserved from original

        new_data = []
        for tensor_row in mu:
            row = tensor_row.cpu().numpy()
            size = len(row)
            variance = np.var(row)
            rnd = self.generate_random_with_variance(size, variance)
            new_vec = self.generate_correlated_vectors(row, rnd, target_correlation)
            new_data.append(new_vec)

        new_data = np.array(new_data)
        tensor = torch.from_numpy(new_data).float().to(mu.device)  # original device behavior
        return tensor


# =============================================================================
# Registry bindings (explicit)
# =============================================================================
register_module("VAE", VAE)
register_module("MinusD_KLLoss", MinusD_KLLoss)
register_module("Random", Random)
