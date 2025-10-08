#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/notate/tools/args.py

Purpose:
    Argument & configuration utilities:
      - Parse CLI and CSV-saved args
      - Layer multiple YAML configs with overwrite warnings
      - Substitute variables and fill arg placeholders
      - Clip config dicts to callable signatures

Notes:
    - Style unified to PEP8 / Black (88 cols) with English comments/docstrings.
    - Behavior preserved; only safe fixes:
        * Added missing `ast` import used in args_from_df().
    - Uses `addict.Dict` for convenient dotted access.

Requires:
    Python >= 3.9
"""

from __future__ import annotations

# ===== Standard library =====
import argparse
import ast
import inspect
import os
import sys
from typing import Callable

# ===== Third-party =====
import pandas as pd
import yaml
from addict import Dict


# =============================================================================
# Basic CLI helpers
# =============================================================================
def default_parser() -> argparse.ArgumentParser:
    """Return a minimal default parser (kept for backward compatibility)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--notice", action="store_true")
    return parser


def save_args(args: argparse.Namespace, file: str) -> None:
    """Persist argparse.Namespace to a CSV file with name/type/value columns."""
    arg_names, arg_types, arg_values = [], [], []
    for arg_name in dir(args):
        if arg_name.startswith("_"):
            continue
        arg = getattr(args, arg_name)
        arg_names.append(arg_name)
        arg_types.append(type(arg).__name__)
        arg_values.append(arg)
    arg_df = pd.DataFrame({"name": arg_names, "type": arg_types, "value": arg_values})
    arg_df.to_csv(file, index=False)


def args_from_df(file: str) -> argparse.Namespace:
    """Load arguments from a CSV produced by save_args()."""
    arg_df = pd.read_csv(file, keep_default_na=False)
    args = argparse.Namespace()
    for _, row in arg_df.iterrows():
        typ = row["type"]
        name = row["name"]
        val = row["value"]

        if str(typ).lower() == "nonetype":
            arg_value = None
        elif typ == "bool":
            arg_value = val == "True"
        elif typ == "list":
            arg_value = list(ast.literal_eval(val))
        else:
            # Fall back to constructing the type by name
            arg_value = eval(f"{typ}(val)")
        setattr(args, name, arg_value)
    return args


# =============================================================================
# Config composition & variable substitution
# =============================================================================
def update_with_check(origin, after, path, modifieds):
    """Recursively merge `after` into `origin` with overwrite warnings."""
    if isinstance(origin, dict) and isinstance(after, dict):
        for key, value in after.items():
            origin[key], modifieds = update_with_check(
                origin[key] if key in origin else {},
                value,
                path + (key,),
                modifieds,
            )
        return origin, modifieds
    else:
        if origin != after:
            if origin != {}:
                for mpath in modifieds:
                    if path[: len(mpath)] == mpath or mpath[: len(path)] == path:
                        print(
                            f"WARNING: {'.'.join(path)} was overwritten for multiple times.",
                            file=sys.stderr,
                        )
                modifieds.append(path)
        return after, modifieds


def search_args(config):
    """Find CLI-like tokens in config (strings starting with '--')."""
    args = []
    types = []
    if isinstance(config, dict):
        for child in config.values():
            a, t = search_args(child)
            args += a
            types += t
    elif isinstance(config, list):
        for child in config:
            a, t = search_args(child)
            args += a
            types += t
    elif type(config) == str:
        if len(config) > 2 and config[:2] == "--":
            configs = config.split(":")
            args = [configs[0]]
            types = [str] if len(configs) == 1 else [eval(configs[1])]
    return args, types


def gather_args(config):
    """Collect argument specs embedded in a config (using 'argname' blocks)."""
    args = []
    if isinstance(config, dict):
        if "argname" in config:
            arg_args = {}
            for key, value in config.items():
                if key == "argname":
                    continue
                elif key == "type":
                    arg_args["type"] = eval(config.type)
                else:
                    arg_args[key] = value
            args = [(config.argname, arg_args)]
        else:
            for child in config.values():
                args += gather_args(child)
    elif isinstance(config, list):
        for child in config:
            args += gather_args(child)
    return args


def fill_args(config, args):
    """Replace 'argname' placeholders in config with actual CLI values."""
    if isinstance(config, dict):
        if "argname" in config:
            return args[config.argname]
        else:
            for label, child in list(config.items()):
                config[label] = fill_args(child, args)
            return config
    elif isinstance(config, list):
        return [fill_args(child, args) for child in config]
    else:
        return config


def subs_vars(config, vars):
    """Substitute variables in a config using a mapping `vars`."""
    if isinstance(config, str):
        if config in vars:
            return vars[config]
        for key, value in vars.items():
            config = config.replace(key, str(value))
        return config
    elif isinstance(config, dict):
        return Dict({label: subs_vars(child, vars) for label, child in config.items()})
    elif isinstance(config, list):
        return [subs_vars(child, vars) for child in config]
    else:
        return config


def delete_args(config):
    """Delete keys whose value is exactly '$delete' (recursive)."""
    if isinstance(config, dict):
        new_config = Dict()
        for key, value in config.items():
            if value == "$delete":
                continue
            elif isinstance(value, (dict, list)):
                value = delete_args(value)
            new_config[key] = value
        return new_config
    elif isinstance(config, list):
        return [delete_args(child) for child in config]
    else:
        return config


def load_config2(config_dir, default_configs):
    """Load config using argv and optional defaults (compat wrapper)."""
    return load_config3(sys.argv[1:], config_dir, default_configs)


# =============================================================================
# Main config loader
# =============================================================================
def load_config3(argv, config_dir, default_configs):
    """Parse CLI, read YAMLs, merge with overwrite warnings, and substitute vars."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="*", default=[])
    args = parser.parse_known_args(argv)[0]

    config = None
    modifieds = []
    for file in default_configs + args.config:
        if os.path.exists(file):
            with open(file, "rb") as f:
                aconfig = Dict(yaml.load(f, yaml.Loader))
        else:
            with open(os.path.join(config_dir, file) + ".yaml", "rb") as f:
                aconfig = Dict(yaml.load(f, yaml.Loader))

        if config is None:
            config = aconfig
        else:
            config, modifieds = update_with_check(config, aconfig, tuple(), modifieds)

    config = Dict(config)

    # Define additional CLI args from config blocks
    arg_specs = gather_args(config)
    for arg_name, arg_args in arg_specs:
        parser.add_argument(f"--{arg_name}", **arg_args)

    # Parse full argv and fill into config
    parsed = vars(parser.parse_args(argv))
    config = fill_args(config, parsed)

    # Collect top-level *variable* blocks and substitute
    variables = {}
    for top_label in config.keys():
        if "variable" in top_label:
            variables.update(config[top_label])
    config = subs_vars(config, variables)

    # Delete keys explicitly marked for removal
    config = delete_args(config)
    return Dict(config)


# =============================================================================
# Misc
# =============================================================================
def clip_config(config: dict, func: Callable) -> dict:
    """Clip a config dict to the signature of `func` (drop unknown keys)."""
    sig = inspect.signature(func)
    fkeys = sig.parameters.keys()
    # Remove unknown keys
    for ckey in list(config.keys()):
        if ckey not in fkeys:
            del config[ckey]
    # Fill defaults for missing keys
    for fkey, fparam in sig.parameters.items():
        if fkey not in config and fparam.default != inspect.Parameter.empty:
            config[fkey] = fparam.default
    return config
