# scripts/train.py
# Strict-mode ready: no implicit variable expansion, no fallbacks.
# - Forbids unknown keys in top-level and major blocks
# - Requires end_token for GreedyDecoder
# - Allows only $TIMESTAMP (in result_dir.dirname)
# - Resolves relative paths based on the directory of this config.yaml
#
# NOTE:
#   This file keeps the original training pipeline structure (get_dataloader /
#   get_optimizer / get_scheduler / get_metric / get_process / Model, etc.).
#   Only the config-loading/validation and path handling around the entrypoint
#   were hardened. If any type/kwargs are unknown at runtime, core/core.py
#   will now raise (strict mode).

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import Any

import yaml
import torch
import random
import numpy as np
from addict import Dict as ADict

from notate.tools.path import make_result_dir, timestamp
from notate.tools.logger import default_logger
from notate.tools.tools import nullcontext

from notate.data import get_dataloader, get_accumulator, NumpyAccumulator
from notate.training import (
    get_metric, get_optimizer, get_scheduler, get_process,
    AlarmHook, hook_type2class, get_hook
)
from notate.core import Model


# ============================================================
# Strict validators (stdlib only)
# ============================================================

ALLOWED_TOP = {
    "result_dir", "device", "seed", "logging",
    "data", "model", "optimizer", "scheduler",
    "metrics", "accumulators", "pre_hooks", "post_hooks",
    "training"
}

ALLOWED_RESULT_DIR = {"dirname", "duplicate"}
ALLOWED_LOGGING = {"stream_level", "file_level"}

ALLOWED_DATA_TOP = {"train", "valid"}
ALLOWED_DATA_SPLIT = {"dataset", "loader"}
ALLOWED_LOADER = {"batch_size", "shuffle", "num_workers", "pin_memory", "drop_last"}

# dataset側は柔軟にしつつ、主要フィールドは型を確認（余計なキーは core 側で弾く）
DATASET_PATH_KEYS = {"path", "path_list", "vocab_json"}

ALLOWED_OPTIMIZER = {"type", "lr", "betas", "eps", "weight_decay", "momentum"}
ALLOWED_SCHEDULER = {"type", "warmup", "t_max", "eta_min", "step_size", "gamma"}

ALLOWED_ACCUMULATOR = {"name", "type", "fields", "save_dir"}

ALLOWED_HOOK = {"type", "target", "every", "args", "kwargs"}

ALLOWED_TRAINING = {
    "epochs", "steps_per_epoch", "grad_accum", "save_every",
    "verbose", "regularize_loss",
    "train_loop", "val_loop"
}

def _forbid_extra_keys(node: dict, allowed: set, path=""):
    if not isinstance(node, dict):
        raise SystemExit(f"[CONFIG ERROR] '{path[:-1]}' must be a mapping")
    extra = set(node.keys()) - allowed
    if extra:
        k = sorted(extra)[0]
        raise SystemExit(f"[CONFIG ERROR] unexpected key '{path}{k}' (extra keys forbidden)")

def _ensure_type(val: Any, expected, path: str):
    if not isinstance(val, expected):
        exp = expected.__name__ if hasattr(expected, "__name__") else str(expected)
        got = type(val).__name__
        raise SystemExit(f"[CONFIG ERROR] '{path}' must be of type {exp} (got {got})")

def _validate_result_dir(cfg: ADict):
    if "result_dir" not in cfg:
        raise SystemExit("[CONFIG ERROR] result_dir is required")
    _forbid_extra_keys(dict(cfg.result_dir), ALLOWED_RESULT_DIR, "result_dir.")
    if "dirname" not in cfg.result_dir or not cfg.result_dir.dirname:
        raise SystemExit("[CONFIG ERROR] result_dir.dirname is required")
    dirname = str(cfg.result_dir.dirname)
    if "$" in dirname and "$TIMESTAMP" not in dirname:
        raise SystemExit("[CONFIG ERROR] Only $TIMESTAMP is allowed in result_dir.dirname")

def _validate_device_seed_logging(cfg: ADict):
    dev = str(cfg.get("device", "cuda")).lower()
    if dev not in {"cuda", "cpu"}:
        raise SystemExit("[CONFIG ERROR] device must be 'cuda' or 'cpu'")
    if "seed" in cfg and not isinstance(cfg.seed, int):
        raise SystemExit("[CONFIG ERROR] seed must be int")
    if "logging" in cfg:
        _forbid_extra_keys(dict(cfg.logging), ALLOWED_LOGGING, "logging.")
        if "stream_level" in cfg.logging and not isinstance(cfg.logging.stream_level, str):
            raise SystemExit("[CONFIG ERROR] logging.stream_level must be str")
        if "file_level" in cfg.logging and not isinstance(cfg.logging.file_level, str):
            raise SystemExit("[CONFIG ERROR] logging.file_level must be str")

def _validate_data(cfg: ADict):
    if "data" not in cfg:
        raise SystemExit("[CONFIG ERROR] data block is required")
    _forbid_extra_keys(dict(cfg.data), ALLOWED_DATA_TOP, "data.")

    for split in ("train", "valid"):
        if split not in cfg.data:
            raise SystemExit(f"[CONFIG ERROR] data.{split} is required")
        node = cfg.data[split]
        _forbid_extra_keys(dict(node), ALLOWED_DATA_SPLIT, f"data.{split}.")
        if "dataset" not in node: raise SystemExit(f"[CONFIG ERROR] data.{split}.dataset is required")
        if "loader"  not in node: raise SystemExit(f"[CONFIG ERROR] data.{split}.loader is required")
        # loader keys check
        _forbid_extra_keys(dict(node.loader), ALLOWED_LOADER, f"data.{split}.loader.")
        if "batch_size" in node.loader and not isinstance(node.loader.batch_size, int):
            raise SystemExit(f"[CONFIG ERROR] data.{split}.loader.batch_size must be int")
        # dataset minimal presence (type only checked at core/factory)
        if "type" not in node.dataset:
            raise SystemExit(f"[CONFIG ERROR] data.{split}.dataset.type is required")

def _validate_optimizer_scheduler(cfg: ADict):
    if "optimizer" not in cfg or "type" not in cfg.optimizer:
        raise SystemExit("[CONFIG ERROR] optimizer.type is required")
    _forbid_extra_keys(dict(cfg.optimizer), ALLOWED_OPTIMIZER, "optimizer.")

    if "scheduler" in cfg and cfg.scheduler is not None:
        _forbid_extra_keys(dict(cfg.scheduler), ALLOWED_SCHEDULER, "scheduler.")
        if "type" not in cfg.scheduler:
            raise SystemExit("[CONFIG ERROR] scheduler.type is required when scheduler is provided")
        st = str(cfg.scheduler.type).lower()
        if st == "warmup":
            if "warmup" not in cfg.scheduler or cfg.scheduler.warmup is None:
                raise SystemExit("[CONFIG ERROR] scheduler.warmup is required when scheduler.type == 'warmup'")

def _validate_metrics(cfg: ADict):
    if "metrics" in cfg and cfg.metrics is not None:
        if not isinstance(cfg.metrics, list):
            raise SystemExit("[CONFIG ERROR] metrics must be a list")
        # metric名の妥当性は get_metric 側で最終チェック

def _validate_accumulators(cfg: ADict):
    if "accumulators" in cfg and cfg.accumulators is not None:
        if not isinstance(cfg.accumulators, list):
            raise SystemExit("[CONFIG ERROR] accumulators must be a list")
        for i, acc in enumerate(cfg.accumulators):
            _forbid_extra_keys(dict(acc), ALLOWED_ACCUMULATOR, f"accumulators[{i}].")
            for k in ("name", "type", "fields", "save_dir"):
                if k not in acc:
                    raise SystemExit(f"[CONFIG ERROR] accumulators[{i}].{k} is required")

def _validate_hooks(cfg: ADict):
    for key in ("pre_hooks", "post_hooks"):
        if key in cfg and cfg[key] is not None:
            if not isinstance(cfg[key], list):
                raise SystemExit(f"[CONFIG ERROR] {key} must be a list")
            for i, hk in enumerate(cfg[key]):
                _forbid_extra_keys(dict(hk), ALLOWED_HOOK, f"{key}[{i}].")
                if "type" not in hk:
                    raise SystemExit(f"[CONFIG ERROR] {key}[{i}].type is required")

def _validate_training(cfg: ADict):
    if "training" not in cfg:
        raise SystemExit("[CONFIG ERROR] training block is required")
    _forbid_extra_keys(dict(cfg.training), ALLOWED_TRAINING, "training.")
    for k in ("epochs", "steps_per_epoch", "save_every"):
        if k not in cfg.training:
            raise SystemExit(f"[CONFIG ERROR] training.{k} is required")
    if not isinstance(cfg.training.epochs, int) or cfg.training.epochs <= 0:
        raise SystemExit("[CONFIG ERROR] training.epochs must be positive int")
    if not isinstance(cfg.training.steps_per_epoch, int) or cfg.training.steps_per_epoch <= 0:
        raise SystemExit("[CONFIG ERROR] training.steps_per_epoch must be positive int")

def _validate_model_greedy_end_token(cfg: ADict):
    # 明示的に GreedyDecoder の end_token を確認
    if "model" not in cfg or "modules" not in cfg.model:
        # modules 構成自体の整合は core / Model 側で検証
        return
    modules = cfg.model.modules
    for name, mod in modules.items():
        try:
            mtype = str(mod.get("type", "")).lower()
        except Exception:
            continue
        if mtype == "greedydecoder".lower():
            if "end_token" not in mod:
                raise SystemExit(f"[CONFIG ERROR] model.modules.{name}.end_token is required for GreedyDecoder")

def _validate_no_variables_block(cfg: ADict):
    if "variables" in cfg:
        raise SystemExit("[CONFIG ERROR] 'variables' block is not allowed (strict mode)")

def _validate_cfg(cfg: ADict):
    _forbid_extra_keys(dict(cfg), ALLOWED_TOP)
    _validate_no_variables_block(cfg)
    _validate_result_dir(cfg)
    _validate_device_seed_logging(cfg)
    _validate_data(cfg)
    _validate_optimizer_scheduler(cfg)
    _validate_metrics(cfg)
    _validate_accumulators(cfg)
    _validate_hooks(cfg)
    _validate_training(cfg)
    _validate_model_greedy_end_token(cfg)


# ============================================================
# Path handling
# ============================================================

def _abspath_inplace_for_known_paths(cfg: ADict, cfg_dir: Path) -> None:
    """
    Resolve relative paths based on the directory of config.yaml.
    Only for known path fields (dataset path/path_list/vocab_json, accumulators.save_dir).
    """
    def _to_abs(p: str) -> str:
        # Keep as-is if already absolute; otherwise resolve from cfg_dir
        q = Path(p)
        return str(q if q.is_absolute() else (cfg_dir / q).resolve())

    # data.train/valid.dataset
    for split in ("train", "valid"):
        if "data" in cfg and split in cfg.data and "dataset" in cfg.data[split]:
            dset = cfg.data[split].dataset
            for key in DATASET_PATH_KEYS:
                if key in dset and dset[key]:
                    if key == "path_list":
                        if not isinstance(dset.path_list, list):
                            raise SystemExit(f"[CONFIG ERROR] data.{split}.dataset.path_list must be a list")
                        dset.path_list = [ _to_abs(x) for x in dset.path_list ]
                    else:
                        dset[key] = _to_abs(str(dset[key]))

    # accumulators[*].save_dir
    if "accumulators" in cfg and cfg.accumulators:
        for i, acc in enumerate(cfg.accumulators):
            if "save_dir" in acc and acc.save_dir:
                acc.save_dir = _to_abs(str(acc.save_dir))

    # result_dir.dirname ($TIMESTAMP はこの後で置換)
    if "result_dir" in cfg and "dirname" in cfg.result_dir:
        dn = str(cfg.result_dir.dirname)
        if "$TIMESTAMP" not in dn:
            cfg.result_dir.dirname = _to_abs(dn)
        # $TIMESTAMP を含む場合は make_result_dir 前に置換してから絶対化する


# ============================================================
# Utilities
# ============================================================

def replace_epoch_placeholders(obj, epoch: int):
    """
    Recursively replace '{epoch}' placeholders ONLY within data.* blocks.
    This is called by training loop where needed.
    """
    if isinstance(obj, dict):
        return {k: replace_epoch_placeholders(v, epoch) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_epoch_placeholders(v, epoch) for v in obj]
    elif isinstance(obj, str):
        return obj.replace("{epoch}", str(epoch))
    else:
        return obj

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise SystemExit(f"[CONFIG ERROR] config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = ADict(raw)

    # Strict validation (no fallbacks)
    _validate_cfg(cfg)

    # Resolve relative paths against the config directory
    _abspath_inplace_for_known_paths(cfg, cfg_dir=cfg_path.parent)

    # Prepare result directory (allow $TIMESTAMP here only)
    ts = timestamp()
    rd = str(cfg.result_dir.dirname)
    if "$TIMESTAMP" in rd:
        rd = rd.replace("$TIMESTAMP", ts)
        # After replacement, absolutize if still relative
        rdp = Path(rd)
        if not rdp.is_absolute():
            rd = str((cfg_path.parent / rdp).resolve())
    duplicate = cfg.result_dir.get("duplicate", "error")
    result_dir = make_result_dir(dirname=rd, duplicate=duplicate)

    # -------------------------
    # Logger (strict: stdlib only, no project-specific wrapper)
    # -------------------------
    import logging, os
    os.makedirs(result_dir, exist_ok=True)

    logger = logging.getLogger("notate")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    stream_level = str(cfg.get("logging", {}).get("stream_level", "info")).upper()
    file_level   = str(cfg.get("logging", {}).get("file_level", "debug")).upper()

    sh = logging.StreamHandler()
    sh.setLevel(getattr(logging, stream_level, logging.INFO))
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    log_path = os.path.join(result_dir, "train.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(getattr(logging, file_level, logging.DEBUG))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(fh)

    logger.info("[result_dir] %s", result_dir)
    logger.info("[log_file]  %s", log_path)


    # -------------------------
    # Build dataloaders
    # -------------------------
    def _split_type_kwargs(d: ADict):
        d = ADict(d)  # shallow copy-ish
        if "type" not in d:
            raise SystemExit("[CONFIG ERROR] data.<split>.type is required ('normal' or 'bucket')")
        t = d.type
        rest = {k: v for k, v in d.items() if k != "type"}
        return t, rest

    t_train, kw_train = _split_type_kwargs(cfg.data.train)
    train_loader = get_dataloader(t_train, logger=logger, **kw_train)

    valid_loader = None
    if "valid" in cfg.data and cfg.data.valid:
        t_valid, kw_valid = _split_type_kwargs(cfg.data.valid)
        valid_loader = get_dataloader(t_valid, logger=logger, **kw_valid)

    # -------------------------
    # Build model
    # -------------------------
    # NOTE: Model のコンストラクタ引数は既存実装に従います。
    #       よくある形: Model(config=cfg.model, logger=logger)
    model = Model(config=cfg.model, logger=logger)  # ←既存の署名に合わせてください

    # -------------------------
    # Optimizer / Scheduler
    # -------------------------
    optimizer = get_optimizer(cfg.optimizer, model=model, logger=logger)
    scheduler = get_scheduler(cfg.get("scheduler", None), optimizer=optimizer, logger=logger)

    # -------------------------
    # Metrics
    # -------------------------
    metrics = []
    if "metrics" in cfg and cfg.metrics:
        for mcfg in cfg.metrics:
            metrics.append(get_metric(mcfg, logger=logger))

    # -------------------------
    # Accumulators
    # -------------------------
    accumulators = []
    if "accumulators" in cfg and cfg.accumulators:
        for acfg in cfg.accumulators:
            accumulators.append(get_accumulator(acfg, logger=logger))
    if not accumulators:
        accumulators = [NumpyAccumulator(name="default", fields=["loss"])]

    # -------------------------
    # Hooks
    # -------------------------
    pre_hooks = []
    if "pre_hooks" in cfg and cfg.pre_hooks:
        for hcfg in cfg.pre_hooks:
            pre_hooks.append(get_hook(hcfg, logger=logger, result_dir=result_dir))
    post_hooks = []
    if "post_hooks" in cfg and cfg.post_hooks:
        for hcfg in cfg.post_hooks:
            post_hooks.append(get_hook(hcfg, logger=logger, result_dir=result_dir))

    # -------------------------
    # Training process
    # -------------------------
    # get_process は既存の「逐次実行グラフ」を組み立てるファクトリです。
    train_proc = get_process(cfg.training.get("train_loop", []), logger=logger)
    val_proc   = get_process(cfg.training.get("val_loop", []), logger=logger)

    epochs = int(cfg.training.epochs)
    steps_per_epoch = int(cfg.training.steps_per_epoch)
    grad_accum = int(cfg.training.get("grad_accum", 1))
    save_every = int(cfg.training.get("save_every", 1))
    verbose = bool(cfg.training.get("verbose", True))

    # -------------------------
    # Main training loop
    # -------------------------
    scaler_cm = nullcontext()  # AMP 等を後で挿すならここ（現状は使わない）
    with scaler_cm:
        global_step = 0
        for epoch in range(1, epochs + 1):
            logger.info(f"[Epoch {epoch}/{epochs}] start")
            model.train()
            # epoch埋め込み: data.* 配下での {epoch} を必要に応じて適用
            # （必要なら train_loader を再構築する場合のみ使用）
            # ex) epochごとにファイルを切り替える運用がある場合:
            # cfg.data.train = ADict(replace_epoch_placeholders(cfg.data.train, epoch))

            # ---- train steps
            step = 0
            for batch in train_loader:
                step += 1
                global_step += 1

                # forward/backward/step are defined in train_proc graph
                train_proc.run(model=model,
                               batch=batch,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               device=device,
                               accumulators=accumulators,
                               metrics=metrics,
                               logger=logger)

                if step >= steps_per_epoch:
                    break

            # ---- validation
            if valid_loader is not None and val_proc is not None:
                model.eval()
                with torch.no_grad():
                    for batch in valid_loader:
                        val_proc.run(model=model,
                                     batch=batch,
                                     device=device,
                                     accumulators=accumulators,
                                     metrics=metrics,
                                     logger=logger)

            # ---- hooks & save
            for h in pre_hooks:
                if hasattr(h, "on_epoch_end"):
                    h.on_epoch_end(epoch)
            if (epoch % save_every) == 0:
                # Save minimal checkpoint; actual saver hook may do richer I/O
                ckpt_path = Path(result_dir) / f"checkpoint_epoch{epoch}.pt"
                torch.save({"model": model.state_dict(),
                            "optimizer": optimizer.state_dict() if optimizer else None,
                            "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
                            "epoch": epoch}, ckpt_path)
                logger.info(f"[checkpoint] saved: {ckpt_path}")
            for h in post_hooks:
                if hasattr(h, "on_epoch_end"):
                    h.on_epoch_end(epoch)

            logger.info(f"[Epoch {epoch}/{epochs}] done")

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
