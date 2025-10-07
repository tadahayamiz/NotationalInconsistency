#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import pickle
import time
import yaml
from addict import Dict
import numpy as np
import pandas as pd
import torch
import random
import shutil
from tqdm import tqdm
import gc

# ==== import from notate.* (editable install friendly) ====
from notate.tools.path import make_result_dir, timestamp
from notate.tools.logger import default_logger
from notate.tools.args import load_config2, subs_vars
from notate.tools.tools import nullcontext

from notate.data import get_dataloader, get_accumulator, NumpyAccumulator
from notate.training import (
    get_metric, get_optimizer, get_scheduler, get_process,
    AlarmHook, hook_type2class, get_hook
)
from notate.core import Model


# ====== runner-side utility ======
def _seed_everything(seed: int):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _dget(dct: Dict, path: str, default=None):
    """dot-path getter for addict.Dict"""
    cur = dct
    for p in path.split('.'):
        if p not in cur:
            return default
        cur = cur[p]
    return cur


def _dset(dct: Dict, path: str, value):
    cur = dct
    parts = path.split('.')
    for p in parts[:-1]:
        if p not in cur:
            cur[p] = Dict()
        cur = cur[p]
    cur[parts[-1]] = value


def _ensure_list(x):
    if x is None:
        return []
    return x if isinstance(x, (list, tuple)) else [x]


def _file_exists(path):
    try:
        return os.path.exists(path) and os.path.isfile(path)
    except Exception:
        return False


def _load_pickle_list(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, (list, tuple)):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj  # leave as-is; caller validates


def _save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


# ====== Preflight (inline; train/val 両方に適用可) ======
def run_preflight_and_rewrite_paths(cfg: Dict, result_dir: str, logger):
    """
    Preflight:
      - token 済み pickle（input/target）を長さでフィルタ
      - split ごと（train / 各 val）に drop 率を判定し、閾値超過なら中止
      - {result_dir}/preflight/<split>/ に _clean.pkl を保存
      - config の path_list を “clean 側” に書き換え

    Config 追加キー:
      preflight:
        enable: true
        max_length: 198
        max_drop_pct: 5.0
        check_files: true
        report_dir: "./qc"
        apply_to_train: true
        apply_to_vals:  false
    """
    pf = cfg.get('preflight', None)
    if not pf or not pf.get('enable', False):
        logger.info("[preflight] disabled")
        return cfg  # unchanged

    max_len       = int(pf.get('max_length', 198))
    max_drop_pct  = float(pf.get('max_drop_pct', 5.0))
    check_files   = bool(pf.get('check_files', True))
    apply_train   = bool(pf.get('apply_to_train', True))
    apply_vals    = bool(pf.get('apply_to_vals', False))

    report_dir = pf.get('report_dir', os.path.join(result_dir, "qc"))
    if not os.path.isabs(report_dir):
        report_dir = os.path.join(result_dir, os.path.normpath(report_dir))
    os.makedirs(report_dir, exist_ok=True)

    def _apply_to_split(base_key: str, split_label: str):
        in_paths = _dget(cfg, f"{base_key}.input.path_list")
        tgt_paths = _dget(cfg, f"{base_key}.target.path_list")
        if in_paths is None or tgt_paths is None:
            logger.warning(f"[preflight] {split_label}: path_list not found. Skipped.")
            return None

        in_paths = _ensure_list(in_paths)
        tgt_paths = _ensure_list(tgt_paths)

        if check_files:
            for p in in_paths + tgt_paths:
                if not _file_exists(p):
                    raise SystemExit(f"[preflight] {split_label}: missing data file: {p}")

        if len(in_paths) != len(tgt_paths):
            logger.warning(f"[preflight] {split_label}: #input pickles != #target pickles. Proceed per index until min length.")
        pair_n = min(len(in_paths), len(tgt_paths))

        clean_dir = os.path.join(result_dir, "preflight", split_label)
        os.makedirs(clean_dir, exist_ok=True)

        cleaned_in_paths, cleaned_tgt_paths = [], []
        total_before, total_after, total_dropped = 0, 0, 0

        for i in range(pair_n):
            src_in = in_paths[i]
            src_tg = tgt_paths[i]
            try:
                arr_in = _load_pickle_list(src_in)
                arr_tg = _load_pickle_list(src_tg)
            except Exception as e:
                raise SystemExit(f"[preflight] {split_label}: failed to load {src_in} or {src_tg}: {e}")

            L = min(len(arr_in), len(arr_tg))
            keep_idx = []
            for k in range(L):
                try:
                    li = len(arr_in[k])
                    lt = len(arr_tg[k])
                except Exception:
                    continue
                ok = (li <= max_len) and (lt <= max_len) and (li > 0) and (lt > 0)
                if ok:
                    keep_idx.append(k)

            total_before += L
            total_after  += len(keep_idx)
            total_dropped += (L - len(keep_idx))

            base_in = os.path.basename(src_in)
            base_tg = os.path.basename(src_tg)
            dst_in = os.path.join(clean_dir, base_in.replace(".pkl", "_clean.pkl"))
            dst_tg = os.path.join(clean_dir, base_tg.replace(".pkl", "_clean.pkl"))
            _save_pickle([arr_in[k] for k in keep_idx], dst_in)
            _save_pickle([arr_tg[k] for k in keep_idx], dst_tg)
            cleaned_in_paths.append(dst_in)
            cleaned_tgt_paths.append(dst_tg)

            logger.info(f"[preflight] {split_label}: {base_in}/{base_tg}: {L} -> {len(keep_idx)} kept (max_len={max_len})")

        drop_pct = (100.0 * total_dropped / max(1, total_before))
        summary = {
            "split": split_label,
            "total_before": int(total_before),
            "total_after":  int(total_after),
            "total_dropped": int(total_dropped),
            "drop_pct":     float(drop_pct),
            "max_len":      int(max_len),
            "allowed_max_drop_pct": float(max_drop_pct),
        }
        _save_pickle(summary, os.path.join(report_dir, f"preflight_summary_{split_label}.pkl"))
        with open(os.path.join(report_dir, f"preflight_summary_{split_label}.txt"), "w", encoding="utf-8") as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")

        if total_after == 0:
            raise SystemExit(f"[preflight] {split_label}: all samples dropped (0 remain).")
        if drop_pct > max_drop_pct:
            raise SystemExit(f"[preflight] {split_label}: drop {drop_pct:.2f}% > allowed {max_drop_pct:.2f}%")

        _dset(cfg, f"{base_key}.input.path_list",
              cleaned_in_paths if len(cleaned_in_paths) > 1 else cleaned_in_paths[0])
        _dset(cfg, f"{base_key}.target.path_list",
              cleaned_tgt_paths if len(cleaned_tgt_paths) > 1 else cleaned_tgt_paths[0])

        logger.info(f"[preflight] {split_label}: kept={total_after} / {total_before} (drop={drop_pct:.2f}%)")
        return summary

    all_summaries = []
    if apply_train:
        base_key = "training.data.train.datasets.datasets"
        s = _apply_to_split(base_key, "train")
        if s is not None:
            all_summaries.append(s)

    if apply_vals:
        vals = _dget(cfg, "training.data.vals", None)
        if isinstance(vals, Dict) and len(vals) > 0:
            for name in list(vals.keys()):
                base_key = f"training.data.vals.{name}.datasets.datasets"
                s = _apply_to_split(base_key, f"val_{name}")
                if s is not None:
                    all_summaries.append(s)
        else:
            logger.warning("[preflight] apply_to_vals=True だが validation 設定が見つかりませんでした。")

    if len(all_summaries) > 0:
        total_before = sum(s["total_before"] for s in all_summaries)
        total_after  = sum(s["total_after"]  for s in all_summaries)
        total_dropped = sum(s["total_dropped"] for s in all_summaries)
        global_summary = {
            "splits": [s["split"] for s in all_summaries],
            "total_before": int(total_before),
            "total_after":  int(total_after),
            "total_dropped": int(total_dropped),
            "drop_pct": float(100.0 * total_dropped / max(1, total_before)),
            "max_len": int(max_len),
            "allowed_max_drop_pct_per_split": float(max_drop_pct),
        }
        _save_pickle(global_summary, os.path.join(report_dir, "preflight_summary_all.pkl"))
        with open(os.path.join(report_dir, "preflight_summary_all.txt"), "w", encoding="utf-8") as f:
            for k, v in global_summary.items():
                f.write(f"{k}: {v}\n")

    return cfg


# --- add: strict validator (voc.py contract: <pad>=0, <start>=1, <end>=2) ---
def strict_validate_special_tokens(cfg):
    st = cfg.model.get("special_tokens", {})
    def need(k, v):
        if int(st.get(k, -999)) != v:
            raise RuntimeError(f"[strict] model.special_tokens.{k} must be {v} (voc.py contract)")
    need("padding_idx", 0)
    need("start_idx",   1)
    need("end_idx",     2)

    # embeddings must match padding_idx=0
    emb = cfg.model["modules"]["enc_embedding"]["embedding"]
    if int(emb.get("padding_idx", -1)) != 0:
        raise RuntimeError("[strict] enc_embedding.embedding.padding_idx must be 0")
    demb = cfg.model["modules"]["dec_embedding"]["embedding"]
    if int(demb.get("padding_idx", -1)) != 0:
        raise RuntimeError("[strict] dec_embedding.embedding.padding_idx must be 0")

    # GreedyDecoder 側も end_token=2 を要求（config で明示済みのはず）
    decsup = cfg.model["modules"]["dec_supporter"]
    if int(decsup.get("end_token", -1)) != 2:
        raise RuntimeError("[strict] dec_supporter.end_token must be 2")


# ====== random state I/O (kept) ======
def save_rstate(dirname):
    os.makedirs(dirname, exist_ok=True)
    with open(f"{dirname}/random.pkl", 'wb') as f:
        pickle.dump(random.getstate(), f)
    with open(f"{dirname}/numpy.pkl", 'wb') as f:
        pickle.dump(np.random.get_state(), f)
    torch.save(torch.get_rng_state(), f"{dirname}/torch.pt")
    torch.save(torch.cuda.get_rng_state_all(), f"{dirname}/cuda.pt")


def set_rstate(config):
    if 'random' in config:
        with open(config.random, 'rb') as f:
            random.setstate(pickle.load(f))
    if 'numpy' in config:
        with open(config.numpy, 'rb') as f:
            np.random.set_state(pickle.load(f))
    if 'torch' in config:
        torch.set_rng_state(torch.load(config.torch))
    if 'cuda' in config:
        torch.cuda.set_rng_state_all(torch.load(config.cuda))


# ====== hooks (normalize flattened 'target/step' into alarm=...) ======
def _normalize_alarm_kwargs(kwargs: dict):
    """
    Accept both styles:
      A) alarm=dict(type=..., target=..., step=..., start=..., list=...)
      B) flattened kwargs: target=..., step=..., start=..., list=..., every=...
    Return: (alarm_dict_or_None, remaining_kwargs)
    """
    kw = dict(kwargs)  # shallow copy
    alarm = kw.pop('alarm', None)

    flat_keys = ('target', 'step', 'start', 'list', 'every', 'interval')
    has_flat = any(k in kw for k in flat_keys)

    if alarm is None and has_flat:
        alarm = {}
        if 'list' in kw:
            alarm['type'] = 'list'
            alarm['list'] = kw.pop('list')
        else:
            alarm['type'] = 'count'
            if 'every' in kw:
                alarm['step'] = kw.pop('every')
            if 'interval' in kw:
                alarm['step'] = kw.pop('interval')
            if 'step' in kw:
                alarm['step'] = kw.pop('step')
        if 'target' in kw:
            alarm['target'] = kw.pop('target')
        if 'start' in kw:
            alarm['start'] = kw.pop('start')
    return alarm, kw


class NoticeAlarmHook(AlarmHook):
    def __init__(self, logger=None, studyname=None, **kwargs):
        alarm, kwargs = _normalize_alarm_kwargs(kwargs)
        super().__init__(logger=logger, alarm=alarm, **kwargs)
        if studyname is None:
            self.logger.warning("studyname not specified in NoticeAlarm.")
            studyname = "(study noname)"
        self.studyname = studyname

    def ring(self, batch, model):
        print("ring")


hook_type2class['notice_alarm'] = NoticeAlarmHook


def main(config, args=None):
    # strict: voc.py の契約と一致しているか検証
    strict_validate_special_tokens(config)

    # substitute variables (kept)
    config = subs_vars(config, {"$TIMESTAMP": timestamp()})
    trconfig = config.training

    # result dir & logger
    result_dir = make_result_dir(**trconfig.result_dir)
    logger = default_logger(result_dir + "/log.txt",
                            trconfig.verbose.loglevel.stream,
                            trconfig.verbose.loglevel.file)
    with open(result_dir + "/config.yaml", mode='w', encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, sort_keys=False, allow_unicode=True)
    if args is not None:
        logger.warning(f"options: {' '.join(args)}")

    # runner-side seed
    _seed_everything(int(trconfig.get('runner_seed', trconfig.get('model_seed', 0))))

    # --- runner-side preflight (train/vals) ---
    try:
        config = run_preflight_and_rewrite_paths(config, result_dir, logger)
    except SystemExit as e:
        logger.error(str(e))
        raise

    # helper for epoch-wise dataloader rewrite (strict: ファイル存在必須)
    def update_dataloader_for_epoch(epoch, config):
        new_ran = f"./data/Pubchem_chunk_pro_{epoch}_ran.pkl"
        new_can = f"./data/Pubchem_chunk_pro_{epoch}_can.pkl"
        if not (_file_exists(new_ran) and _file_exists(new_can)):
            raise SystemExit(f"[strict] epoch {epoch}: missing data files: {new_ran} / {new_can}")
        updated_config = config.copy()
        _dset(updated_config, "training.data.train.datasets.datasets.input.path_list", new_ran)
        _dset(updated_config, "training.data.train.datasets.datasets.target.path_list", new_can)
        return get_dataloader(logger=logger, device=DEVICE, **updated_config.training.data.train)

    # environment
    DEVICE = torch.device('cuda', index=trconfig.gpuid or 0) \
        if torch.cuda.is_available() else torch.device('cpu')
    logger.warning(f"DEVICE: {DEVICE}")
    if trconfig.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    if trconfig.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True
        torch.backends.cudnn.benchmark = False

    # prepare data
    dl_train = get_dataloader(logger=logger, device=DEVICE, **trconfig.data.train)
    dls_val = {name: get_dataloader(logger=logger, device=DEVICE, **dl_val_config)
               for name, dl_val_config in trconfig.data.vals.items()}

    # --------------------------
    # prepare model (strict config check + sanitize)
    # --------------------------
    if 'model_seed' in trconfig:
        random.seed(trconfig.model_seed)
        np.random.seed(trconfig.model_seed)
        torch.manual_seed(trconfig.model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(trconfig.model_seed)

    if not isinstance(config.get('model', None), Dict):
        raise SystemExit("[CONFIG ERROR] top-level `model` section is missing in your config.")

    mods = config.model.get('modules', None)
    if not isinstance(mods, (dict, Dict)) or len(mods) == 0:
        dump_path = os.path.join(result_dir, "model_section_dump.yaml")
        try:
            with open(dump_path, "w", encoding="utf-8") as f:
                yd = config.model.to_dict() if hasattr(config.model, "to_dict") else dict(config.model)
                yaml.dump(yd, f, sort_keys=False, allow_unicode=True)
        except Exception:
            pass
        raise SystemExit("[CONFIG ERROR] `model.modules` must be a non-empty mapping. "
                         f"Dumped model section to: {dump_path}")

    mod_keys = list(mods.keys())
    use = config.model.get('use_modules', None)
    omit = config.model.get('omit_modules', None)
    if use is not None and not isinstance(use, (list, tuple)):
        use = [use]
    if omit is not None and not isinstance(omit, (list, tuple)):
        omit = [omit]

    if use is not None:
        unknown_use = [m for m in use if m not in mod_keys]
        if len(unknown_use) > 0:
            logger.warning(f"[config] unknown names in model.use_modules -> dropped: {unknown_use}")
            use = [m for m in use if m in mod_keys]
            config.model.use_modules = use
        if len(use) == 0:
            logger.warning("[config] model.use_modules became empty after filtering; falling back to ALL modules.")
            try:
                config.model.pop('use_modules')
            except Exception:
                try:
                    del config.model['use_modules']
                except Exception:
                    config.model.use_modules = None
            use = None

    if omit is not None:
        unknown_omit = [m for m in omit if m not in mod_keys]
        if len(unknown_omit) > 0:
            logger.warning(f"[config] unknown names in model.omit_modules -> dropped: {unknown_omit}")
            omit = [m for m in omit if m in mod_keys]
            config.model.omit_modules = omit

    selected = []
    for k in mod_keys:
        if use is not None and k not in use:
            continue
        if omit is not None and k in omit:
            continue
        selected.append(k)

    if len(selected) == 0:
        dump_path = os.path.join(result_dir, "model_modules_filter_dump.yaml")
        with open(dump_path, "w", encoding="utf-8") as f:
            yaml.dump({
                "all_module_keys": mod_keys,
                "use_modules": use,
                "omit_modules": omit
            }, f, sort_keys=False, allow_unicode=True)
        raise SystemExit("[CONFIG ERROR] No modules remain after applying use_modules/omit_modules. "
                         f"Check names. Dumped filter info to: {dump_path}")

    logger.info(f"[model] modules: {len(mod_keys)} defined; selected -> {len(selected)} : "
                f"{selected[:8]}{'...' if len(selected)>8 else ''}")

    # ---- kwargs validation for modules (optional but helpful) ----
    import inspect
    from notate.core.core import module_type2class

    def validate_model_module_kwargs(model_cfg, result_dir, logger, policy="strict"):
        errs = []
        lines = []
        for name, mcfg in model_cfg.get("modules", {}).items():
            mtype = mcfg.get("type")
            if mtype not in module_type2class:
                errs.append(f"{name}: unknown module type '{mtype}'. Registered: {sorted(list(module_type2class.keys()))[:20]} ...")
                continue
            cls = module_type2class[mtype]
            sig = inspect.signature(cls.__init__)
            allowed = [p for p in sig.parameters.keys() if p != "self"]
            provided = sorted([k for k in mcfg.keys() if k != "type"])
            unknown = sorted(set(provided) - set(allowed))
            missing = sorted([p for p, pr in sig.parameters.items()
                              if p != "self" and pr.default is inspect._empty and p not in provided])

            if unknown:
                msg = f"{name} (type={mtype}): unknown kwargs -> {unknown} ; allowed={allowed}"
                if policy == "drop":
                    for u in unknown:
                        del mcfg[u]
                    logger.warning("[kwargs:drop] " + msg)
                else:
                    errs.append(msg)
            if missing:
                errs.append(f"{name} (type={mtype}): missing required -> {missing} ; allowed={allowed}")

            lines.append(f"{name}: type={mtype}\n  allowed={allowed}\n  provided={provided}\n  unknown={unknown}\n  missing={missing}\n")

        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "kwargs_validation.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        if errs:
            raise SystemExit("[CONFIG ERROR] Module kwargs mismatch. See kwargs_validation.txt for details.\n"
                             + "\n".join(errs[:3]))

    model_cfg = config.model
    validate_model_module_kwargs(model_cfg, result_dir, logger, policy="strict")

    if not isinstance(model_cfg.get('modules', None), dict) or len(model_cfg['modules']) == 0:
        raise SystemExit("[CONFIG ERROR] (Option A) `model_cfg['modules']` is empty or not a mapping.")

    # build model
    model = Model(config=model_cfg, logger=logger, **model_cfg)

    if getattr(trconfig, 'init_weight', None):
        model.load(**trconfig.init_weight)
    model.to(DEVICE)
    optimizer = get_optimizer(params=model.parameters(), **trconfig.optimizer)
    if getattr(trconfig, 'optimizer_init_weight', None):
        optimizer.load_state_dict(torch.load(trconfig.optimizer_init_weight))

    train_processes = [get_process(**process) for process in trconfig.train_loop]

    # ====== Alarm hooks (normalized kwargs) ======
    class SchedulerAlarmHook(AlarmHook):
        def __init__(self, scheduler, logger=None, **kwargs):
            alarm, kwargs = _normalize_alarm_kwargs(kwargs)
            super().__init__(logger=logger, alarm=alarm, **kwargs)
            # warmup 等で last_epoch を持つ実装には、現ステップに同期
            scheduler.setdefault('last_epoch', dl_train.step - 1)
            self.scheduler = get_scheduler(optimizer, **scheduler)

        def ring(self, batch, model):
            # ★ そのstepで optimizer.step() を実行した場合のみ進める
            if batch.get('opt_stepped', False):
                self.scheduler.step()

    hook_type2class['scheduler_alarm'] = SchedulerAlarmHook

    class ValidationAlarmHook(AlarmHook):
        def __init__(self, logger=None, **kwargs):
            alarm, kwargs = _normalize_alarm_kwargs(kwargs)
            super().__init__(logger=logger, alarm=alarm, **kwargs)
            self.eval_steps = []

        def ring(self, batch, model):
            step = batch['step']
            if step in self.eval_steps:
                return
            self.logger.info(f"Validating step{step:7} ...")
            model.eval()
            for x in metrics + accumulators + [idx_accumulator]:
                x.init()
            with torch.no_grad():
                for key, dl in dls_val.items():
                    for metric in metrics:
                        metric.set_val_name(key)
                    for batch0 in dl:
                        # ---- run processes (strict Model.forward) ----
                        _run_processes(model, batch0, val_processes)
                        for x in metrics + accumulators + [idx_accumulator]:
                            x(batch0)
                        del batch0
                        torch.cuda.empty_cache()

            scores = {}
            for metric in metrics:
                scores = metric.calc(scores)
            batch.update(scores)
            for score_lb, score in scores.items():
                self.logger.info(f"  {score_lb:20}: {score:.3f}")
                scores_df.loc[step, score_lb] = score
            scores_df.to_csv(result_dir + "/val_score.csv", index_label="Step")

            idx = idx_accumulator.accumulate()
            idx = np.argsort(idx)
            for accumulator in accumulators:
                accumulator.save(f"{result_dir}/accumulates/{accumulator.input}/{step}", indices=idx)

            self.eval_steps.append(step)
            model.train()

    hook_type2class['validation_alarm'] = ValidationAlarmHook

    class CheckpointAlarm(AlarmHook):
        def __init__(self, logger=None, **kwargs):
            alarm, kwargs = _normalize_alarm_kwargs(kwargs)
            super().__init__(logger=logger, alarm=alarm, **kwargs)
            os.makedirs(f"{result_dir}/checkpoints", exist_ok=True)
            self.checkpoint_steps = []

        def ring(self, batch, model: Model):
            if batch['step'] in self.checkpoint_steps:
                return
            checkpoint_dir = f"{result_dir}/checkpoints/{batch['step']}"
            self.logger.info(f"Making checkpoint at step {batch['step']:6>}...")
            if len(self.checkpoint_steps) > 0:
                shutil.rmtree(f"{result_dir}/checkpoints/{self.checkpoint_steps[-1]}/")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_state_dict(f"{result_dir}/models/{batch['step']}")
            torch.save(optimizer.state_dict(), f"{checkpoint_dir}/optimizer.pth")
            dl_train.checkpoint(f"{checkpoint_dir}/dataloader_train")
            scores_df.to_csv(checkpoint_dir + "/val_score.csv", index_label="Step")
            save_rstate(f"{checkpoint_dir}/rstate")
            self.checkpoint_steps.append(batch['step'])

    hook_type2class['checkpoint_alarm'] = CheckpointAlarm

    # Prepare abortion
    abort_step = trconfig.abortion.step or float('inf')
    abort_epoch = trconfig.abortion.epoch or float('inf')
    abort_time = trconfig.abortion.time or float('inf')

    # Prepare metrics
    accumulators = [get_accumulator(logger=logger, **acc_config) for acc_config in trconfig.accumulators]
    idx_accumulator = NumpyAccumulator(logger, input='idx', org_type='numpy')
    metrics = [get_metric(logger=logger, name=name, **met_config) for name, met_config in trconfig.metrics.items()]

    scores_df = pd.DataFrame(columns=[], dtype=float)
    if getattr(trconfig.stocks, 'score_df', ""):
        try:
            scores_df = pd.read_csv(trconfig.stocks.score_df, index_col="Step")
        except Exception:
            logger.warning(f"Failed to load score_df: {trconfig.stocks.score_df}. Start from empty.")
            scores_df = pd.DataFrame(columns=[], dtype=float)

    val_processes = [get_process(**process) for process in trconfig.val_loop]
    if trconfig.val_loop_add_train:
        val_processes = train_processes + val_processes

    pre_hooks = [get_hook(logger=logger, result_dir=result_dir, **hconfig) for hconfig in trconfig.pre_hooks.values()]
    post_hooks = [get_hook(logger=logger, result_dir=result_dir, **hconfig) for hconfig in trconfig.post_hooks.values()]

    # load random state
    set_rstate(getattr(trconfig, 'rstate', Dict()))

    # ====== helper: run processes (strict Model.forward) ======
    def _run_processes(model_obj: Model, batch_dict: dict, processes):
        """Execute process graph explicitly, since Model.forward is strict."""
        for proc in processes:
            out = proc(model_obj, batch_dict)
            # processes typically mutate batch in-place; accept optional dict returns
            if isinstance(out, dict) and out is not batch_dict:
                batch_dict.update(out)
        return batch_dict

    # regularize_loss のデフォルト（config に無い場合に備える）
    _reg = getattr(trconfig, 'regularize_loss', Dict())
    _reg.normalize = bool(getattr(_reg, 'normalize', False))
    _reg.clip_grad_norm = float(getattr(_reg, 'clip_grad_norm', 0.0) or 0.0)
    _reg.clip_grad_value = float(getattr(_reg, 'clip_grad_value', 0.0) or 0.0)

    # training
    training_start = time.time()
    logger.info("Training started.")
    print("Start")
    with (tqdm(total=None, initial=dl_train.step) if trconfig.verbose.show_tqdm else nullcontext()) as pbar:
        now = time.time()
        optimizer.zero_grad()
        epoch = 0
        steps_per_epoch = int(getattr(trconfig, "steps_per_epoch", 30000))

        while True:
            batch = {'step': dl_train.step, 'epoch': dl_train.epoch}
            for hook in pre_hooks:
                hook(batch, model)

            if dl_train.step >= abort_step \
               or now - training_start >= abort_time \
               or dl_train.epoch >= abort_epoch:
                logger.warning("Use of abort_step, abort_time, abort_epoch is deprecated. Use AbortHook instead.")
                batch['end'] = True
            if 'end' in batch:
                break

            # training step
            batch = dl_train.get_batch(batch)
            start = time.time()
            # ---- run processes (strict Model.forward) ----
            _run_processes(model, batch, train_processes)

            loss = sum(batch[loss_name] for loss_name in trconfig.loss_names)
            if _reg.normalize:
                # 注意: 正規化の意味が曖昧なので、zero division を避けるだけの安全策
                denom = float(loss.detach().abs().item()) or 1.0
                loss = loss / denom
            try:
                loss.backward()
            except Exception as e:
                os.makedirs(f"{result_dir}/error/batch", exist_ok=True)
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        torch.save(value, f"{result_dir}/error/batch/{key}.pt")
                    else:
                        with open(f"{result_dir}/error/batch/{key}.pkl", 'wb') as f:
                            pickle.dump(value, f)
                model.save_state_dict(f"{result_dir}/error/model")
                raise e

            # epoch-wise dataloader update (strict file check)
            if dl_train.step % steps_per_epoch == 0:
                epoch += 1
                del dl_train
                gc.collect()
                dl_train = update_dataloader_for_epoch(epoch, config)
                dl_train.step = epoch * steps_per_epoch

            # grad stats (kept)
            grad_max = grad_min = 0
            grad_mean = 0
            grad_numel = 0
            for p in model.parameters():
                if p.grad is None:
                    continue
                grad_mean += torch.sum(p.grad ** 2)
                grad_numel += p.grad.numel()
                grad_max = max(grad_max, p.grad.max().item())
                grad_min = min(grad_min, p.grad.min().item())
            batch['grad_mean'] = grad_mean / grad_numel if grad_numel > 0 else 0.
            batch['grad_max'] = grad_max
            batch['grad_min'] = grad_min

            if _reg.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=_reg.clip_grad_norm,
                    error_if_nonfinite=True
                )
            if _reg.clip_grad_value:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(),
                    clip_value=_reg.clip_grad_value
                )

            # optimizer -> scheduler の順序を保証
            if dl_train.step % trconfig.schedule.opt_freq == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                batch['opt_stepped'] = True
            else:
                batch['opt_stepped'] = False
            batch['time'] = time.time() - start

            for hook in post_hooks:
                hook(batch, model)
            del batch, loss
            torch.cuda.empty_cache()
            if pbar is not None:
                pbar.update(1)

    for hook in pre_hooks + post_hooks:
        hook(batch, model)


if __name__ == '__main__':
    # 重要: 既定の 'config' を重ねない。CLI 指定の --config のみを採用
    config = load_config2("", default_configs=[])
    main(config, sys.argv)
