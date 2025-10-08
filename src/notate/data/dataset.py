#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/data/dataset.py

Purpose:
    Data loading utilities with two loader variants (normal and bucketed) and a
    small set of dataset classes (string/array-like/sparse/generator).
    Designed to interoperate with config-driven pipelines.

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Imports ordered: stdlib -> third-party -> local (project).
    - No behavior-changing edits were made intentionally.
    - Known oddity kept for compatibility:
        NdarrayDataset: `array = array(split)` line preserved (likely a bug)
        to avoid changing runtime behavior.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Standard library =====
import gc
import inspect
import itertools
import os
import pickle
import sys

# ===== Third-party =====
import numpy as np
import pandas as pd
import torch
import yaml
from addict import Dict

# ===== Project-local =====
from ..tools.tools import check_leftargs


# =============================================================================
# DataLoader base
# =============================================================================
class DataLoader:
    """Base dataloader that iterates over config-defined datasets."""

    def __init__(self, logger, datasets, seed, device, checkpoint=None, **kwargs):
        """Initialize the dataloader.

        Args:
            logger: Logger instance.
            datasets: A dict or list of dicts with keys:
                - dfs:    {df_name: dict for pandas.read_csv kwargs}
                - datasets: {name: dict for get_dataset kwargs}
            seed: RNG seed (int).
            device: Target torch device.
            checkpoint: Optional path to a previous checkpoint directory.
            **kwargs: Ignored (validated via check_leftargs).
        """
        check_leftargs(self, logger, kwargs)
        if not isinstance(datasets, list):
            datasets = [datasets]
        self.dset_configss = datasets

        # Normalize pandas read_csv argument name.
        for dset_config in self.dset_configss:
            for df_config in dset_config.dfs.values():
                if "path" in df_config:
                    df_config["filepath_or_buffer"] = df_config.pop("path")

        self.n_dset = len(datasets)
        self.i_current_idx = 0
        self.i_dset = 0
        self.epoch = self.step = 0
        self.current_idxs = None
        self.rstate = np.random.RandomState(seed=seed)
        self.logger = logger
        self.device = device
        self.cur_dsets = None

        # Load checkpoint if provided.
        if checkpoint is not None:
            with open(f"{checkpoint}/config.yaml") as f:
                config = Dict(yaml.load(f, yaml.Loader))
            self.i_dset = config.i_dset
            self.i_current_idx = config.i_current_idx
            self.epoch = config.epoch
            self.step = config.step
            with open(f"{checkpoint}/current_idxs.pkl", "rb") as f:
                self.current_idxs = pickle.load(f)
            with open(f"{checkpoint}/rstate.pkl", "rb") as f:
                self.rstate.set_state(pickle.load(f))

        self.load_datasets()

    def load_datasets(self) -> None:
        """(Re)load current dataset group and construct dataset objects."""
        del self.cur_dsets
        gc.collect()
        dfs = {}

        for df_name, df_config in self.dset_configss[self.i_dset].dfs.items():
            self.logger.info(f"Loading {df_config.filepath_or_buffer} ...")
            dfs[df_name] = pd.read_csv(**df_config)

        self.cur_dsets = {
            name: get_dataset(logger=self.logger, name=name, dfs=dfs, **dset_config)
            for name, dset_config in self.dset_configss[self.i_dset].datasets.items()
        }
        del dfs
        self.i_cur_dsets = self.i_dset

    def get_batch(self, batch=None):
        """Return one batch dict constructed from the current dataset group."""
        if self.i_cur_dsets != self.i_dset:
            self.load_datasets()
        if self.current_idxs is None:
            self.current_idxs = self.get_idxs(self.cur_dsets)

        idx = self.current_idxs[self.i_current_idx].astype(int)
        if batch is None:
            batch = {}
        batch["idx"] = idx

        for dset in self.cur_dsets.values():
            dset.make_batch(batch, idx, self.device)

        batch["batch_size"] = len(batch["idx"])
        self.i_current_idx += 1
        self.step += 1

        if self.i_current_idx == len(self.current_idxs):
            self.i_current_idx = 0
            self.current_idxs = None
            self.i_dset = (self.i_dset + 1) % self.n_dset
            if self.i_dset == 0:
                self.epoch += 1
        return batch

    def __iter__(self):
        """Yield batches until one full epoch is completed."""
        self.epoch = self.i_dset = self.i_current_idx = 0
        # while self.epoch == 0 -> one pass
        while self.epoch == 0:
            yield self.get_batch()

    def get_idxs(self, dsets):
        """Return index chunks for batching (implemented in subclasses)."""
        raise NotImplementedError

    def checkpoint(self, path_checkpoint: str) -> None:
        """Save dataloader state under a directory."""
        os.makedirs(path_checkpoint)
        config = {
            "i_dset": self.i_dset,
            "i_current_idx": self.i_current_idx,
            "epoch": self.epoch,
            "step": self.step,
        }
        with open(f"{path_checkpoint}/config.yaml", "w") as f:
            yaml.dump(config, f)
        with open(f"{path_checkpoint}/rstate.pkl", "wb") as f:
            pickle.dump(self.rstate.get_state(), f)
        with open(f"{path_checkpoint}/current_idxs.pkl", "wb") as f:
            pickle.dump(self.current_idxs, f)


# =============================================================================
# NormalDataLoader
# =============================================================================
class NormalDataLoader(DataLoader):
    """Simple dataloader that shuffles indices and splits by fixed batch size."""

    def __init__(self, logger, device, datasets, seed, batch_size, checkpoint=None, **kwargs):
        super().__init__(
            logger=logger,
            datasets=datasets,
            seed=seed,
            device=device,
            checkpoint=checkpoint,
            **kwargs,
        )
        self.batch_size = batch_size
        if not isinstance(datasets, list):
            datasets = [datasets]
        self.dset_name0 = list(datasets[0].datasets.keys())[0]

    def get_idxs(self, dsets):
        dset_size = len(dsets[self.dset_name0])
        idxs = np.arange(dset_size, dtype=int)
        self.rstate.shuffle(idxs)
        idxs = np.split(idxs, range(0, dset_size, self.batch_size))
        return idxs


# =============================================================================
# BucketDataLoader
# =============================================================================
class BucketDataLoader(DataLoader):
    """Length-bucketed dataloader supporting dynamic batch sizes."""

    def __init__(
        self,
        logger,
        device,
        datasets,
        seed,
        bucket_dset,
        checkpoint=None,
        bin_linspace=None,
        bins=None,
        add_lower_margin=True,
        add_upper_margin=True,
        batch_size=None,
        num_tokens=None,
        num_tokens_dim=1,
        **kwargs,
    ):
        """Initialize bucketed dataloader.

        Args:
            bucket_dset: Dataset name to base bucketing on.
            bin_linspace: Optional (start, stop, num) for np.linspace bins.
            bins: Explicit bin edges; bucket[i] => bins[i] <= len < bins[i+1]
            add_lower_margin: If True, ensures a 0 lower bound.
            add_upper_margin: If True, ensures an inf upper bound.
            batch_size: Fixed batch size per bucket (int or list); XOR num_tokens.
            num_tokens: If set, derive batch sizes by length-based token budget.
            num_tokens_dim: Power for token budget (length ** num_tokens_dim).
        """
        super().__init__(
            logger=logger,
            datasets=datasets,
            seed=seed,
            device=device,
            checkpoint=checkpoint,
            **kwargs,
        )
        # Validate args (XOR checks kept as-is).
        if (bin_linspace is None) == (bins is None):
            raise ValueError(
                f"Either bin_linspace({bin_linspace}) XOR bins({bins}) must be specified"
            )
        if (batch_size is None) == (num_tokens is None):
            raise ValueError(
                f"Either batch_size({batch_size}) XOR num_tokens({num_tokens}) must be specified."
            )

        self.buckets = [None] * len(self.dset_configss)
        self.bucket_dset = bucket_dset

        # Compute bucket bins
        if bin_linspace is not None:
            bins = list(np.linspace(*bin_linspace))
        if add_lower_margin and (len(bins) == 0 or bins[0] > 0):
            bins.insert(0, 0)
        if add_upper_margin and (len(bins) == 0 or bins[-1] < float("inf")):
            bins.append(float("inf"))
        self.bins = bins
        self.n_bucket = len(self.bins) - 1

        # Compute per-bucket batch sizes
        self.num_tokens = num_tokens
        self.num_tokens_dim = num_tokens_dim
        if batch_size is not None:
            if isinstance(batch_size, list):
                self.batch_sizes = batch_size
            else:
                self.batch_sizes = [batch_size] * (len(self.bins) - 1)
        else:
            self.batch_sizes = [
                int(num_tokens // (np.ceil(sup_len) - 1) ** num_tokens_dim)
                for sup_len in self.bins[1:]
            ]

    def get_idxs(self, dsets):
        lengths = dsets[self.bucket_dset].lengths
        ibs = np.digitize(lengths, self.bins) - 1
        batch_sizes = self.batch_sizes

        if self.num_tokens is not None and self.bins[-1] == float("inf"):
            batch_sizes[-1] = int(
                self.num_tokens // torch.max(lengths).item() ** self.num_tokens_dim
            )

        idxs = []
        for ib, batch_size in enumerate(self.batch_sizes):
            bucket_idxs = np.where(ibs == ib)[0]
            if len(bucket_idxs) == 0:
                continue
            self.rstate.shuffle(bucket_idxs)
            idxs += [
                bucket_idxs[i : i + batch_size]
                for i in range(0, len(bucket_idxs), batch_size)
            ]
        idxs = np.array(idxs, dtype=object)
        self.rstate.shuffle(idxs)
        return idxs


# =============================================================================
# Dataloader factory
# =============================================================================
dataloader_type2class = {
    "normal": NormalDataLoader,
    "bucket": BucketDataLoader,
}


def get_dataloader(type, **kwargs):
    """Construct a dataloader by type name."""
    return dataloader_type2class[type](**kwargs)


# =============================================================================
# Dataset base and helpers
# =============================================================================
class Dataset:
    """Abstract dataset interface for make_batch/len and split helpers."""

    def __init__(self, logger, name, dfs, **kwargs):
        check_leftargs(self, logger, kwargs)
        self.name = name

    def make_batch(self, batch, idx, device):
        """Populate `batch` with a tensor/array under self.name for given indices."""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def calc_split(self, dfs, df, col, idx=None):
        """Return boolean mask or equality match mask configured by split spec."""
        split: np.ndarray = dfs[df][col].values
        if idx is not None:
            if isinstance(idx, list):
                idx = set(idx)
                split = (
                    np.frompyfunc(lambda x: x in idx, nin=1, nout=1)(split).astype(bool)
                )
            else:
                split = split == idx
        else:
            split = split.astype(bool)
        return split


torch_name2dtype = {
    "int": torch.int,
    "long": torch.long,
    "float": torch.float,
    "bool": torch.bool,
}
numpy_name2dtype = {
    "int": int,
    "float": float,
    "bool": bool,
}


# =============================================================================
# StringDataset
# =============================================================================
class StringDataset(Dataset):
    """Variable-length sequence dataset (e.g., token IDs) with padding."""

    def __init__(
        self,
        logger,
        name,
        dfs,
        padding_value,
        list=None,
        path_list=None,
        len_name=None,
        shape=[],
        dtype="long",
        dim=1,
        split=None,
        **kwargs,
    ):
        """Initialize string-like dataset.

        Args:
            padding_value: Pad token (int).
            list: In-memory list of sequences.
            path_list: Path to pickle containing the list.
                       Either `list` or `path_list` must be provided.
            len_name: Name for the token-length entry in batch dict.
            shape: Additional trailing shape for each example.
            dtype: torch dtype name (see `torch_name2dtype`).
            dim: Dimension count for the variable-length axis (usually 1).
            split: Optional dict used by `calc_split`.
        """
        super().__init__(logger, name, dfs, **kwargs)
        if (list is None) == (path_list is None):
            raise ValueError(
                f"Either list({list}) XOR path_list({path_list}) has to be specified."
            )

        self.len_name = len_name or f"{self.name}_len"

        # Load list
        if list is not None:
            self.str_list = list
        else:
            logger.info(f"Loading {path_list} ...")
            with open(path_list, "rb") as f:
                self.str_list = pickle.load(f)

        # Optional split
        if split:
            split = self.calc_split(dfs, **split)
            self.str_list = list(itertools.compress(self.str_list, split))

        self.lengths = torch.tensor(
            [len(string) for string in self.str_list], dtype=torch.long
        )
        self.shape = tuple(shape)
        self.dtype = torch_name2dtype[dtype]
        self.dim = dim
        logger.info(f"Max length of {name}: {torch.max(self.lengths)}")

        self.padding_value = padding_value

    def make_batch(self, batch, idx, device):
        n = len(idx)
        batch_lengths = self.lengths[idx].to(device)
        batch[self.len_name] = batch_lengths
        batch_strings = torch.full(
            (n,) + (torch.max(batch_lengths),) * self.dim + self.shape,
            fill_value=self.padding_value,
            dtype=self.dtype,
        )
        for i, idx_ in enumerate(idx):
            batch_strings[(i,) + (slice(batch_lengths[i]),) * self.dim] = torch.tensor(
                self.str_list[idx_], dtype=self.dtype
            )
        batch[self.name] = batch_strings.to(device)

    def __len__(self):
        return len(self.str_list)


# =============================================================================
# Array-like datasets
# =============================================================================
class ArrayDataset(Dataset):
    """Array-backed dataset in either torch or numpy representation."""

    def __init__(self, logger, name, dfs, dtype, atype="torch", **kwargs):
        super().__init__(logger, name, dfs, **kwargs)
        self.type = atype
        if self.type in ["numpy", "np"]:
            self.type = "numpy"

        # Check dtype
        if self.type == "torch":
            self.dtype = torch_name2dtype[dtype]
        elif self.type == "numpy":
            self.dtype = numpy_name2dtype[dtype]
        else:
            raise ValueError(f"Unsupported atype: {atype}")

        self.array = None

    def make_batch(self, batch, idx, device=None):
        item = self.array[idx]
        if device is not None:
            item = item.to(device)
        batch[self.name] = item

    def __len__(self):
        return len(self.array)


class NdarrayDataset(ArrayDataset):
    """Dataset loaded from a .npy/.npz/.pt file into torch or numpy array."""

    def __init__(
        self,
        logger,
        name,
        dfs,
        dtype,
        path,
        cols=None,
        atype="torch",
        split=None,
        **kwargs,
    ):
        super().__init__(logger, name, dfs, dtype, atype, **kwargs)
        ext_path = os.path.splitext(path)[-1][1:]
        if ext_path in ["npy", "npz"]:
            array = np.load(path)
            if self.type == "torch":
                array = torch.tensor(array)
        elif ext_path in ["pt"]:
            array = torch.load(path)
            if self.type == "numpy":
                array = array.numpy()
        else:
            raise ValueError(f"Unsupported type of ndarray: {path}")

        if cols is not None:
            array = array[:, cols]

        if split:
            split = self.calc_split(dfs, **split)
            # NOTE: The following line is unusual but preserved to avoid
            # behavior changes. It likely should be `array = array[split]`.
            array = array(split)

        self.array = array
        self.size = ["batch_size"] + list(self.array.shape[1:])


class SeriesDataset(ArrayDataset):
    """1-D series dataset from a DataFrame column."""

    def __init__(
        self, logger, name, dfs, df, dtype, col, atype="torch", split=None, **kwargs
    ):
        super().__init__(logger, name, dfs, dtype, atype, **kwargs)
        array = dfs[df][col].values
        if split:
            split = self.calc_split(dfs, **split)
            array = array[split]
        if self.type == "torch":
            self.array = torch.tensor(array, dtype=self.dtype)
        elif self.type == "numpy":
            self.array = array.astype(self.dtype)


class DataFrameDataset(ArrayDataset):
    """Multi-column dataset from a DataFrame."""

    def __init__(
        self, logger, name, dfs, df, dtype, cols=None, atype="torch", split=None, **kwargs
    ):
        super().__init__(logger, name, dfs, dtype, atype, **kwargs)
        if cols is None:
            cols = dfs[df].columns
        array = dfs[df][cols].values
        if split:
            split = self.calc_split(dfs, **split)
            array = array[split]
        if self.type == "torch":
            self.array = torch.tensor(array, dtype=self.dtype)
        elif self.type == "numpy":
            self.array = array.astype(self.dtype)


# =============================================================================
# SparseSquareDataset
# =============================================================================
class SparseSquareDataset(Dataset):
    """Sparse square matrix dataset assembled into dense at batch time."""

    def __init__(
        self,
        logger,
        name,
        dfs,
        padding_value,
        path_length,
        path_index,
        path_value,
        len_name=None,
        dtype=None,
        split=None,
        **kwargs,
    ):
        super().__init__(logger, name, dfs, **kwargs)
        """
        Parameters to write
        -------------------
        padding_value (int): Pad token.
        path_length (str): Path to .npy or .pkl of list/array lengths per item.
        path_index (str): Path to .pkl of list of index arrays [n_edge, 2].
        path_value (str): Path to .pkl of list of values arrays [n_edge].
        """
        self.len_name = len_name or f"{self.name}_len"
        if split:
            raise NotImplementedError(
                "Splitting for SparseSquareDataset is not defined."
            )

        ext = os.path.splitext(path_length)[1]
        if ext == ".npy":
            self.lengths = torch.tensor(np.load(path_length), dtype=torch.long)
        elif ext == ".pkl":
            with open(path_length, "rb") as f:
                self.lengths = torch.tensor(pickle.load(f), dtype=torch.long)
        else:
            raise ValueError(f"Unsupported type of path_length: {path_length}")

        with open(path_index, "rb") as f:
            self.indices = pickle.load(f)
        with open(path_value, "rb") as f:
            self.values = pickle.load(f)

        self.padding_value = padding_value
        if dtype is not None:
            self.dtype = torch_name2dtype[dtype]
        else:
            self.dtype = None

    def make_batch(self, batch, idx, device):
        batch_size = len(idx)
        lengths = self.lengths[idx].to(device)
        batch[self.len_name] = lengths
        max_len = torch.max(lengths)
        ibatches = []
        indices = []
        values = []
        for i, idx_ in enumerate(idx):
            index = torch.tensor(self.indices[idx_], dtype=torch.int, device=device)  # [n_edge, 2]
            indices.append(index)
            ibatches.append(
                torch.full(
                    (index.shape[0],), fill_value=i, dtype=torch.int, device=device
                )
            )
            values.append(torch.tensor(self.values[idx_], dtype=self.dtype, device=device))
        ibatches = torch.cat(ibatches, dim=0)
        indices = torch.cat(indices, dim=0).T  # [2, n_edges]
        indices = torch.cat([ibatches.unsqueeze(0), indices], dim=0)
        values = torch.cat(values, dim=0)
        data = torch.sparse_coo_tensor(indices, values, size=(batch_size, max_len, max_len)).to_dense()
        batch[self.name] = data

    def __len__(self):
        return len(self.lengths)


# =============================================================================
# GenerateDataset
# =============================================================================
class GenerateDataset(Dataset):
    """Synthetic feature generator (for quick debugging)."""

    def __init__(self, logger, name, dfs, feature_size, data_size, **kwargs):
        super().__init__(logger, name, dfs, **kwargs)
        self.feature_size = feature_size
        self.data_size = data_size

    def make_batch(self, batch, idx, device):
        batch[self.name] = torch.randn(len(idx), self.feature_size, device=device)

    def __len__(self):
        return self.data_size


# =============================================================================
# Dataset factory
# =============================================================================
dataset_type2class = {
    "string": StringDataset,
    "ndarray": NdarrayDataset,
    "series": SeriesDataset,
    "dataframe": DataFrameDataset,
    "sparse_square": SparseSquareDataset,
    "generate": GenerateDataset,
}

# from .datasets.moleculedataset import MoleculeGraphDataset
# dataset_type2class['molecule_graph'] = MoleculeGraphDataset


def get_dataset(type, **kwargs):
    """Construct a dataset instance by type name."""
    return dataset_type2class[type](**kwargs)
