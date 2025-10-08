#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/tools/path.py

Purpose:
    Path, I/O, and result-directory helpers:
      - Safe directory clear/merge utilities
      - File finding with optional prefix
      - CSV/TSV readers and generic table/array loaders
      - Result directory creation policy
      - Short timestamp helper

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - No behavior-changing edits were made intentionally.
    - `make_result_dir` keeps backward compatibility with the legacy `result_dir` arg.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Standard library =====
import os
import pickle
import re
import shutil
import subprocess  # (kept for compatibility; not used here)
import sys
from datetime import datetime
from glob import glob

# ===== Third-party =====
import pandas as pd


# =============================================================================
# Basic path helpers
# =============================================================================
def get_root_path() -> str:
    """Return project root path inferred from this file's path (3 levels up)."""
    dirnames = re.split("/", os.path.abspath(__file__))
    root_path = ""
    for dirname in dirnames[1 : len(dirnames) - 3]:
        root_path = root_path + "/" + dirname
    return root_path


def cleardir(dirname: str, exist_ok: bool | None = None) -> None:
    """Remove a directory tree and recreate an empty directory."""
    _cleardir(dirname)
    os.makedirs(dirname)


def _cleardir(dirname: str) -> None:
    """Recursively remove all files/dirs under dirname, then remove dirname."""
    for path in glob(os.path.join(dirname, "*")):
        if os.path.isdir(path):
            _cleardir(path)
        else:
            os.remove(path)
    if os.path.exists(dirname):
        os.rmdir(dirname)


def find_file_s(file_s, prefix: str):
    """Resolve one or multiple file paths with an optional prefix fallback.

    Accepts a str or list of str. For each item, try:
      1) path as-is,
      2) prefix + file,
      3) prefix + "/" + file.

    Raises:
        FileNotFoundError if none exist for a given entry.
    """
    if isinstance(file_s, str):
        return find_file_s([file_s], prefix)[0]
    founds = []
    for file in file_s:
        if os.path.exists(file):
            founds.append(file)
        elif os.path.exists(prefix + file):
            founds.append(prefix + file)
        elif os.path.exists(prefix + "/" + file):
            founds.append(prefix + "/" + file)
        else:
            raise FileNotFoundError(
                f'Neither "{file}", "{prefix}{file}" nor "{prefix}/{file}" was found.'
            )
    return founds


def make_pardir(path: str) -> None:
    """Create parent directory (if any) for a file path."""
    path_dir = os.path.dirname(path)
    if len(path_dir) > 0:
        os.makedirs(path_dir, exist_ok=True)


def find_file(file: str, prefix: str) -> str | None:
    """Return first existing path among {file, prefix+file, prefix+'/'+file}."""
    if os.path.exists(file):
        return file
    if os.path.exists(prefix + file):
        return prefix + file
    if os.path.exists(prefix + "/" + file):
        return prefix + "/" + file
    return None


# =============================================================================
# Readers
# =============================================================================
def read_csv_tsv(file: str, **kwargs) -> pd.DataFrame:
    """Read CSV/TSV/TXT into a DataFrame (keeps empty strings)."""
    ext = file.split(".")[-1]
    if ext == "csv":
        df = pd.read_csv(file, keep_default_na=False, **kwargs)
    elif ext in ["tsv", "txt"]:
        df = pd.read_csv(file, sep="\t", keep_default_na=False, **kwargs)
    else:
        raise ValueError(f"Unsupported type of file extension: {file}")
    return df


def read_table2(path: str, sep: str | None = None, args: dict = {}, **kwargs) -> pd.DataFrame:
    """General table reader with optional explicit separator.

    Args:
        path: File path.
        sep: Separator; if None, inferred from extension (csv/tsv/txt).
        args: Additional kwargs forwarded to pandas.read_csv.
        **kwargs: Unused (kept for backward compatibility; warns to stderr).

    Returns:
        pandas.DataFrame
    """
    if len(kwargs) > 0:
        print(f"[WARNING] Unknown kwargs: {kwargs.keys()}", file=sys.stderr)
    if sep is None:
        ext = path.split(".")[-1]
        if ext in ["txt", "tsv"]:
            sep = "\t"
        elif ext == "csv":
            sep = ","
        else:
            raise ValueError("Unknown format of table. Please specify config.sep")
    return pd.read_csv(path, sep=sep, **args)


def read_table(config):
    """Deprecated shim for read_table2."""
    print("Usage of read_table is deprecated. use read_table2 (from args) instead.")
    return read_table2(**config)


def read_array(config, df_cache: dict | None = None):
    """Read array-like data from a config pointing to CSV/TSV/TXT/PKL.

    For CSV/TSV:
        - Optionally use a `df_cache` dict to avoid repeated reads.
        - Requires `config.col` to extract a column as a NumPy array.

    For PKL:
        - Loads pickled Python object.

    Returns:
        Numpy array or object (depending on the source).
    """
    if isinstance(config, dict):
        config = config.copy()
        col = config.pop("col")
        ext = config.path.split(".")[-1]
        if ext in ["csv", "tsv", "txt"]:
            if df_cache is not None:
                assert isinstance(df_cache, dict)
                if config.path not in df_cache:
                    df_cache[config.path] = read_table2(**config)
                df = df_cache[config.path]
            else:
                df = read_table2(**config)
            array = df[col].values
        elif ext == "pkl":
            with open(config.path, "rb") as f:
                array = pickle.load(f)
        else:
            raise ValueError(f"Unsupported type of path in read_array: {config.path}")
        return array
    else:
        return config


# =============================================================================
# Result directory
# =============================================================================
def make_result_dir(result_dir: str | None = None, dirname: str | None = None, duplicate: str | None = None) -> str | None:
    """Create a result directory according to a duplicate policy.

    Compatibility:
        Either `result_dir` XOR `dirname` must be specified. If `result_dir`
        is provided, it is treated as `dirname` with a deprecation message.

    Args:
        result_dir: Legacy argument (deprecated).
        dirname: Target directory path.
        duplicate: One of {'error', 'ask', 'overwrite', 'merge'}.

    Returns:
        The directory path (str), or None if user declined in 'ask' mode.
    """
    if (result_dir is None) == (dirname is None):
        raise ValueError(
            f"Please specify either result_dir({result_dir}) XOR dirname({dirname})"
        )
    if result_dir is not None:
        print(
            "from make_result_dir: usage of 'result_dir' is deprecated. Use 'dirname' instead."
        )
        dirname = result_dir

    assert dirname is not None  # for type checkers

    if os.path.exists(dirname):
        if duplicate == "error":
            raise FileExistsError(f"'{dirname}' already exists.")
        elif duplicate == "ask":
            answer = None
            while answer not in ["y", "n"]:
                answer = input(f"'{dirname}' already exists. Will you overwrite this study? (y/n)")
            if answer == "n":
                return None
        elif duplicate in {"overwrite", "merge"}:
            pass
        else:
            raise ValueError(f"Unsupported config.result_dir.duplicate: {duplicate}")

    if duplicate == "merge":
        os.makedirs(dirname, exist_ok=True)
    else:
        cleardir(dirname)
    return dirname


# =============================================================================
# Misc
# =============================================================================
def timestamp() -> str:
    """Return YYMMDD-style timestamp string."""
    dt_now = datetime.now()
    return f"{dt_now.year % 100:02}{dt_now.month:02}{dt_now.day:02}"
