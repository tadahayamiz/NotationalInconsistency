#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/data/accumulator.py

Purpose:
    Implement accumulators that collect batch-wise outputs into numpy arrays
    or Python lists across training/validation iterations.

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Imports ordered: stdlib -> third-party -> local (project).
    - No behavior-changing edits were made intentionally.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Standard library =====
import os
import pickle

# ===== Third-party =====
import numpy as np

# ===== Project-local =====
from ..tools.tools import check_leftargs, EMPTY


# =============================================================================
# NumpyAccumulator
# =============================================================================
class NumpyAccumulator:
    """Accumulate tensors or arrays along a batch dimension into a single ndarray."""

    def __init__(self, logger, input, batch_dim=0, org_type="torch.tensor", **kwargs):
        """Initialize the NumpyAccumulator.

        Args:
            logger: Logger object (not used directly, only for API parity).
            input: Key name in batch dict whose value to accumulate.
            batch_dim: Batch dimension along which to concatenate.
            org_type: One of {"torch.tensor", "np.array"} describing the input type.
            **kwargs: Ignored additional arguments (checked by check_leftargs).
        """
        check_leftargs(self, logger, kwargs)
        self.input = input
        self.org_type = org_type
        if org_type in {"tensor", "torch", "torch.tensor"}:
            self.converter = lambda x: x.cpu().numpy()
        elif org_type in {
            "np.array",
            "np.ndarray",
            "numpy",
            "numpy.array",
            "numpy.ndarray",
        }:
            self.converter = EMPTY
        else:
            raise ValueError(
                f"Unsupported type of config.org_type: {org_type} in NumpyAccumulator"
            )
        self.batch_dim = batch_dim

    def init(self) -> None:
        """Reset the accumulator."""
        self.accums = []

    def accumulate(self, indices=None):
        """Concatenate accumulated arrays along batch_dim, with padding if needed."""
        if self.accums[0].ndim == 3:
            mid_values = [arr.shape[1] for arr in self.accums]
            max_mid_value = max(mid_values)
            # Pad arrays to equalize the middle dimension
            for i in range(len(self.accums)):
                current_mid_value = self.accums[i].shape[1]
                if current_mid_value < max_mid_value:
                    pad_top = (max_mid_value - current_mid_value) // 2
                    pad_bottom = max_mid_value - current_mid_value - pad_top
                    self.accums[i] = np.pad(
                        self.accums[i],
                        ((0, 0), (pad_top, pad_bottom), (0, 0)),
                        mode="constant",
                    )

        max_shape = np.array([a.shape for a in self.accums]).max(axis=0)
        padded_accums = []

        if self.org_type == "torch.tensor":
            # Pad last dimension only to unify array shapes
            for arr in self.accums:
                pad_width = [(0, 0) for _ in range(arr.ndim)]
                pad_width[-1] = (0, max_shape[-1] - arr.shape[-1])
                padded_arr = np.pad(arr, pad_width, mode="constant", constant_values=0)
                padded_accums.append(padded_arr)

        if len(padded_accums) > 0:
            accums = np.concatenate(padded_accums, axis=self.batch_dim)
        else:
            accums = np.concatenate(self.accums, axis=self.batch_dim)

        if indices is not None:
            accums = accums[indices]
        return accums

    def save(self, path_without_ext: str, indices=None) -> None:
        """Save the accumulated array to a .npy file."""
        path = path_without_ext + ".npy"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.accumulate(indices=indices))

    def __call__(self, batch) -> None:
        """Append a converted batch value."""
        self.accums.append(self.converter(batch[self.input]))


# =============================================================================
# ListAccumulator
# =============================================================================
class ListAccumulator:
    """Accumulate batch elements into a Python list."""

    def __init__(self, logger, input, org_type="torch.tensor", batch_dim=None, **kwargs):
        """Initialize the ListAccumulator.

        Args:
            logger: Logger object (not used directly).
            input: Key name in batch dict whose value to accumulate.
            org_type: "list", "torch.tensor", or "np.array".
            batch_dim: Batch dimension if applicable.
            **kwargs: Ignored, checked by check_leftargs.
        """
        check_leftargs(self, logger, kwargs)
        self.input = input

        if org_type == "list":
            assert batch_dim is None, "batch_dim cannot be defined when org_type is list"
            self.converter = EMPTY
        else:
            if batch_dim is None:
                batch_dim = 0
            if org_type in {"tensor", "torch.tensor"}:
                if batch_dim == 0:
                    self.converter = lambda x: list(x.cpu().numpy())
                else:
                    self.converter = lambda x: list(x.transpose(batch_dim, 0).cpu().numpy())
            elif org_type in {
                "np.array",
                "np.ndarray",
                "numpy",
                "numpy.array",
                "numpy.ndarray",
            }:
                if batch_dim == 0:
                    self.converter = lambda x: list(x)
                else:
                    self.converter = lambda x: list(x.swapaxes(0, batch_dim))

    def init(self) -> None:
        """Reset the accumulator."""
        self.accums = []

    def accumulate(self, indices=None):
        """Return the accumulated list (optionally reordered by indices)."""
        if indices is not None:
            accums = np.array(self.accums, dtype=object)
            return accums[indices].tolist()
        return self.accums

    def save(self, path_without_ext: str, indices=None) -> None:
        """Save the accumulated list as a pickle file."""
        path = path_without_ext + ".pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.accumulate(indices=indices), f)

    def __call__(self, batch) -> None:
        """Append converted values from batch to the accumulator."""
        self.accums += self.converter(batch[self.input])


# =============================================================================
# Factory
# =============================================================================
accumulator_type2class = {
    "numpy": NumpyAccumulator,
    "list": ListAccumulator,
}


def get_accumulator(type: str, **kwargs):
    """Return an accumulator instance by type name."""
    return accumulator_type2class[type](**kwargs)
