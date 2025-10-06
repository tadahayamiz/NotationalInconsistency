# train.py (normalized, fixed)
import os, sys, argparse, yaml, random
from typing import Any, Dict, List, Union

import torch
from addict import Dict as ADict

from notate.tools.path import make_result_dir, timestamp
from notate.tools.logger import default_logger
from notate.tools.args import subs_vars
from notate.tools.tools import nullcontext

from notate.data import get_dataloader, get_accumulator, NumpyAccumulator  # noqa: F401 (accumulators not used in this minimal loop)
from notate.training import (
    get_metric, get_optimizer, get_scheduler, get_process,
    hook_type2class, get_hook, AlarmHook  # noqa: F401 (hook_type2class/AlarmHook kept for compatibility)
)
from notate.core import Model


# -------------------------
# utilities
# -------------------------
def replace_epoch_placeholders(obj, epoch: int):
    """Recursively replace '{epoch}' placeholders in any str fields."""
    if isinstance(obj, dict):
        return {k: replace_epoch_placeholders(v, epoch) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_epoch_placeholders(v, epoch) for v in obj]
    if isinstance(obj, str):
        return obj.replace("{epoch}", str(epoch))
    return obj

def build_logger(result_dir: str, cfg: ADict):
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, "log.txt")
    # default_logger は小文字レベルのみ許容するため lower() で渡す
    stream_level = str(cfg.logging.get("stream_level", "info")).lower() if "logging" in cfg else "info"
    file_level   = str(cfg.logging.get("file_level",   "debug")).lower() if "logging" in cfg else "debug"
    logger = default_logger(
        filename=log_path,
        stream_level=stream_level,
        file_level=file_level,
        logger_name=None,
    )
    logger.info(f"[result_dir] {result_dir}")
    return logger

def build_hooks(logger, result_dir: str, hook_cfg_list: List[dict]):
    hooks = []
    if not hook_cfg_list:
        return hooks
    for hk in hook_cfg_list:
        hk = dict(hk)  # shallow copy
        htype = hk.pop("type")
        hook = get_hook(type=htype, **hk)
        hooks.append(hook)
        logger.info(f"[hook] registered: {htype} ({hk})")
    return hooks

def set_seed(seed: Union[int, None]):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)  # 'cuda:0' など。未指定なら自動
    args = parser.parse_args()

    # --- load config (yaml -> ADict)
    with open(args.config, "r") as f:
        cfg = ADict(yaml.safe_load(f))

    # --- variables expansion ($TIMESTAMP 等)
    vars_dict = {"$TIMESTAMP": timestamp()}
    cfg = ADict(subs_vars(cfg, vars_dict))

    # --- result_dir: dirname 方式に統一
    rcfg = cfg.get("result_dir", {})
    dirname = rcfg.get("dirname", None)
    duplicate = rcfg.get("duplicate", None)  # 'error' | 'ask' | 'overwrite' | 'merge'
    if not dirname:
        raise ValueError("config.result_dir.dirname is required")
    result_dir = make_result_dir(dirname=dirname, duplicate=duplicate)

    # --- device
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- logger
    logger = build_logger(result_dir, cfg)
    logger.info(f"[device] {device}")
    logger.info(f"[timestamp] {vars_dict['$TIMESTAMP']}")

    # --- seed
    set_seed(cfg.get("seed", None))

    # --- model
    mcfg = cfg.get("model", {})
    modules = mcfg.get("modules", {})
    model = Model(
        logger=logger,
        modules=modules,
        use_modules=mcfg.get("use_modules", None),
        omit_modules=mcfg.get("omit_modules", None),
        seed=mcfg.get("seed", None),
        init=mcfg.get("init", None),
    ).to(device)

    # --- optimizer
    ocfg = cfg.get("optimizer", {})
    opt_type = ocfg.get("type", None)
    if not opt_type:
        raise ValueError("config.optimizer.type is required")
    optimizer = get_optimizer(type=opt_type, params=model.parameters(), **{k: v for k, v in ocfg.items() if k != "type"})

    # --- scheduler（任意）
    scfg = cfg.get("scheduler", None)
    scheduler = None
    if scfg and "type" in scfg:
        stype = scfg["type"]
        skw = {k: v for k, v in scfg.items() if k != "type"}
        scheduler = get_scheduler(optimizer=optimizer, type=stype, **skw)
        logger.info(f"[scheduler] type={stype}, kwargs={skw}")

    # --- metrics（複数定義対応）
    metrics = None
    mcfg_multi = cfg.get("metrics") or cfg.get("metric")  # 後方互換
    if isinstance(mcfg_multi, dict):
        metrics = {
            name: get_metric(type=spec.get("type"), **{k: v for k, v in spec.items() if k != "type"})
            for name, spec in mcfg_multi.items()
        }
        logger.info(f"[metrics] {list(metrics.keys())}")
    elif isinstance(mcfg_multi, list):
        metrics = {
            f"{i}:{(spec.get('type') if isinstance(spec, dict) else 'metric')}":
            get_metric(type=(spec.get("type") if isinstance(spec, dict) else spec), **({k: v for k, v in spec.items() if k != "type"} if isinstance(spec, dict) else {}))
            for i, spec in enumerate(mcfg_multi)
        }
        logger.info(f"[metrics] {list(metrics.keys())}")
    elif mcfg_multi:
        # 単数定義（後方互換）
        spec = mcfg_multi
        metrics = {"metric": get_metric(type=spec.get("type"), **{k: v for k, v in spec.items() if k != "type"})}
        logger.info(f"[metrics] single -> {list(metrics.keys())}")

    # --- hooks（pre/post）
    pre_hooks = build_hooks(logger, result_dir, cfg.get("pre_hooks", []))
    post_hooks = build_hooks(logger, result_dir, cfg.get("post_hooks", []))

    # --- training loop params
    tcfg = cfg.get("training", {})
    epochs = int(tcfg.get("epochs", 1))
    steps_per_epoch = int(tcfg.get("steps_per_epoch", 10))
    grad_accum = int(tcfg.get("grad_accum", 1))
    ckpt_dir = os.path.join(result_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_every = int(tcfg.get("save_every", max(1, epochs)))  # default: 最終のみ

    # --- process（実装依存）
    # 典型例: proc(batch, model, optimizer, device, logger, metrics=metrics) -> dict(loss=..., ...)
    pcfg = tcfg.get("process", {"type": "forward"})
    process = get_process(type=pcfg.get("type", "forward"), **{k: v for k, v in pcfg.items() if k != "type"})

    # --- epoch loop
    for epoch in range(1, epochs + 1):
        logger.info(f"=== [epoch {epoch}/{epochs}] ===")

        # dataloader を epoch ごとに再構築（{epoch} 対応）
        # train
        train_dcfg = replace_epoch_placeholders(cfg.data.train, epoch) if "data" in cfg and "train" in cfg.data else None
        if not train_dcfg:
            raise ValueError("config.data.train is required")
        train_loader = get_dataloader(**train_dcfg, device=device, logger=logger)

        # val（任意）
        val_loader = None
        if "vals" in cfg.data and cfg.data.vals:
            vcfg = replace_epoch_placeholders(cfg.data.vals[0], epoch)
            val_loader = get_dataloader(**vcfg, device=device, logger=logger)

        # pre_hooks
        for h in pre_hooks:
            try:
                h()
            except Exception as e:
                logger.warning(f"[pre_hook] {h} raised: {e}")

        model.train()
        optimizer.zero_grad(set_to_none=True)

        step = 0
        for batch in train_loader:
            step += 1
            # --- ここが実装依存ポイント：processの返り値に loss が含まれる前提 ---
            out = process(batch=batch, model=model, optimizer=optimizer, device=device, logger=logger, metrics=metrics)
            loss = out["loss"] if isinstance(out, dict) and "loss" in out else out
            if not torch.is_tensor(loss):
                loss = torch.as_tensor(loss, dtype=torch.float32, device=device)

            (loss / grad_accum).backward()

            if step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

            if step >= steps_per_epoch:
                break

        # post_hooks
        for h in post_hooks:
            try:
                h()
            except Exception as e:
                logger.warning(f"[post_hook] {h} raised: {e}")

        # checkpoint
        if (epoch % save_every) == 0 or (epoch == epochs):
            ckpt_path = os.path.join(ckpt_dir, f"epoch{epoch:04d}.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)
            logger.info(f"[checkpoint] saved: {ckpt_path}")

        # validation（任意）
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vsteps = min(len(val_loader), tcfg.get("val_steps", steps_per_epoch))
                vcnt = 0
                for vbatch in val_loader:
                    vcnt += 1
                    _ = process(batch=vbatch, model=model, optimizer=None, device=device, logger=logger, metrics=metrics)
                    if vcnt >= vsteps:
                        break

    logger.info("[done]")


if __name__ == "__main__":
    main()
