#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/modules/poolers.py

Purpose:
    Pooling modules for sequence and graph representations. These modules
    aggregate per-token (or per-node) embeddings into fixed-size vectors using
    mean/max/start-token statistics, with careful handling of padding masks.

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Imports ordered: stdlib -> third-party -> local (project).
    - No behavior-changing edits were made intentionally.
    - Mask conventions:
        * For sequence poolers, `padding_mask` is shaped [T, B] (bool) where
          True indicates PAD positions (to be ignored).
        * Some classes assume an additional `end_mask` marking END tokens.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Third-party =====
import torch
import torch.nn as nn

# ===== Project-local =====
from ..core import module_type2class, register_module


# =============================================================================
# Sequence poolers
# =============================================================================
class MeanPooler(nn.Module):
    """Mean over valid (non-pad) positions."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Compute masked mean.

        Args:
            input: [T, B, F] float tensor.
            padding_mask: [T, B] bool tensor. True = PAD (ignored).

        Returns:
            [B, F] masked mean.
        """
        valid = ~padding_mask.unsqueeze(-1)  # [T, B, 1]
        return torch.sum(input * valid, dim=0) / torch.sum(valid, dim=0)


class StartPooler(nn.Module):
    """Return the first token embedding."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Select the first time step.

        Args:
            input: [T, B, F] float tensor.

        Returns:
            [B, F] first-step slice.
        """
        return input[0]


class MaxPooler(nn.Module):
    """Max over valid (non-pad) positions."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Compute masked max.

        Args:
            input: [T, B, F].
            padding_mask: [T, B] (True = PAD).

        Returns:
            [B, F] masked max.
        """
        masked = input.masked_fill(padding_mask.unsqueeze(-1), -torch.inf)
        return torch.max(masked, dim=0)[0]


class MeanStartMaxPooler(nn.Module):
    """Concatenate mean, start-token, and max features."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Compute [mean ; start ; max] with masking.

        Args:
            input: [T, B, F].
            padding_mask: [T, B] (True = PAD).

        Returns:
            [B, 3F] concatenated features.
        """
        valid = ~padding_mask.unsqueeze(-1)
        masked_max = input.masked_fill(~valid, -torch.inf)
        return torch.cat(
            [
                torch.sum(input * valid, dim=0) / torch.sum(valid, dim=0),
                input[0],
                torch.max(masked_max, dim=0)[0],
            ],
            dim=-1,
        )


class MeanStartEndMaxPooler(nn.Module):
    """Concatenate mean, start-token, end-token (as sum), and max features."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, input: torch.Tensor, padding_mask: torch.Tensor, end_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute [mean ; start ; end ; max] with masks.

        Args:
            input: [T, B, F].
            padding_mask: [T, B] (True = PAD) -> ignored in mean/max.
            end_mask: [T, B] (True = END) -> used to pick END embedding via sum.

        Returns:
            [B, 4F] concatenated features.
        """
        pad = padding_mask.unsqueeze(-1)
        masked_max = input.masked_fill(pad, -torch.inf)
        valid = ~pad
        end_mask = end_mask.unsqueeze(-1)
        return torch.cat(
            [
                torch.sum(input * valid, dim=0) / torch.sum(valid, dim=0),
                input[0],
                torch.sum(input * end_mask, dim=0),
                torch.max(masked_max, dim=0)[0],
            ],
            dim=-1,
        )


class MeanStdStartEndMaxMinPooler(nn.Module):
    """Concatenate mean, std, start, end, max, min features."""

    def forward(
        self, input: torch.Tensor, padding_mask: torch.Tensor, end_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute [mean ; std ; start ; end ; max ; min] with masks.

        Args:
            input: [T, B, F].
            padding_mask: [T, B] (True = PAD).
            end_mask: [T, B] (True = END).

        Returns:
            [B, 6F] concatenated features.
        """
        pad = padding_mask.unsqueeze(-1)
        masked_max = input.masked_fill(pad, -torch.inf)
        masked_min = input.masked_fill(pad, torch.inf)
        valid = ~pad
        end_mask = end_mask.unsqueeze(-1)

        mean = torch.sum(input * valid, dim=0) / torch.sum(valid, dim=0)
        std = (
            torch.sum(((input - mean.unsqueeze(0)) ** 2) * valid, dim=0)
            / torch.sum(valid, dim=0)
        )
        return torch.cat(
            [
                mean,
                std,
                input[0],
                torch.sum(input * end_mask, dim=0),
                torch.max(masked_max, dim=0)[0],
                torch.min(masked_min, dim=0)[0],
            ],
            dim=-1,
        )


class NoAffinePooler(nn.Module):
    """Mean/Start/Max features with LayerNorm (elementwise_affine=False)."""

    def __init__(self, input_size) -> None:
        """Initialize with expected input size for LayerNorm setup.

        Args:
            input_size: Shape-like; last dim is feature size (F).
        """
        super().__init__()
        self.mean_norm = nn.LayerNorm(input_size[-1], elementwise_affine=False)
        self.start_norm = nn.LayerNorm(input_size[-1], elementwise_affine=False)
        self.max_norm = nn.LayerNorm(input_size[-1], elementwise_affine=False)
        self.output_size = list(input_size[1:])
        self.output_size[-1] = self.output_size[-1] * 3

    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Compute normalized [mean ; start ; max] with log-mask trick for max.

        Args:
            input: [T, B, F].
            padding_mask: [T, B] (True = PAD). Internally inverted for valid mask.

        Returns:
            [B, 3F] concatenated features after LayerNorm.
        """
        valid = ~padding_mask  # [T, B]
        masked_for_max = input + torch.log(valid).unsqueeze(-1)  # log(0) -> -inf
        mean_feat = torch.sum(input * valid.unsqueeze(-1), dim=0) / torch.sum(
            valid, dim=0
        ).unsqueeze(-1)
        start_feat = input[0]
        max_feat = torch.max(masked_for_max, dim=0)[0]
        return torch.cat(
            [
                self.mean_norm(mean_feat),
                self.start_norm(start_feat),
                self.max_norm(max_feat),
            ],
            dim=-1,
        )


class NemotoPooler(nn.Module):
    """Concatenate [max ; mean ; first] without mask handling."""

    def __init__(self, input_size) -> None:
        super().__init__()
        self.output_size = list(input_size[1:])
        self.output_size[-1] = self.output_size[-1] * 3

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Aggregate without masks.

        Args:
            input: [T, B, F].

        Returns:
            [B, 3F] concatenated [max, mean, first].
        """
        mx = torch.max(input, dim=0)[0]
        ave = torch.mean(input, dim=0)
        first = input[0]
        return torch.cat([mx, ave, first], dim=1)


# =============================================================================
# Graph pooler
# =============================================================================
class GraphPooler(nn.Module):
    """Graph-level pooling from node/edge features with padding mask.

    WARNING:
        `padding_mask` convention differs from sequence poolers:
        padding_mask is [B, N] where True indicates PAD nodes.
    """

    def __init__(self, node_size: int, edge_size: int) -> None:
        super().__init__()
        self.node_norm = nn.LayerNorm(node_size, elementwise_affine=False)
        self.edge_norm = nn.LayerNorm(edge_size, elementwise_affine=False)

    def forward(
        self, node: torch.Tensor, edge: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool node and edge features with masks.

        Args:
            node: [N, B, Fn] float tensor (node features over time/position).
            edge: [B, N, N, Fe] float tensor (pairwise edge features).
            padding_mask: [B, N] bool, True = PAD nodes.

        Returns:
            [B, Fn+Fe] concatenated normalized node/edge features.
        """
        n_node, _, _ = node.shape

        # Node pooling: mask and average across N (time/position) dimension.
        node_pad = padding_mask.T.unsqueeze(-1)  # [N, B, 1]
        node_sum = torch.sum(torch.masked_fill(node, node_pad, 0), dim=0)  # [B, Fn]
        node_den = n_node - torch.sum(node_pad, dim=0)  # [B, 1]
        node_feat = node_sum / node_den

        # Edge pooling: mask any pair with a PAD endpoint.
        edge_pad = (padding_mask.unsqueeze(-1) | padding_mask.unsqueeze(-2)).unsqueeze(
            -1
        )  # [B, N, N, 1]
        edge_sum = torch.sum(torch.masked_fill(edge, edge_pad, 0), dim=(1, 2))  # [B, Fe]
        edge_den = n_node**2 - torch.sum(edge_pad, dim=(1, 2))  # [B, 1, 1]
        edge_feat = edge_sum / edge_den

        return torch.cat([self.node_norm(node_feat), self.edge_norm(edge_feat)], dim=-1)


# =============================================================================
# Registration (explicit)
# =============================================================================
# Use the strict registry to map config.type -> class.
register_module("MeanPooler", MeanPooler)
register_module("StartPooler", StartPooler)
register_module("MaxPooler", MaxPooler)
register_module("MeanStartMaxPooler", MeanStartMaxPooler)
register_module("MeanStartEndMaxPooler", MeanStartEndMaxPooler)
register_module("MeanStdStartEndMaxMinPooler", MeanStdStartEndMaxMinPooler)
register_module("NoAffinePooler", NoAffinePooler)
register_module("NemotoPooler", NemotoPooler)
register_module("GraphPooler", GraphPooler)
