import sys, os
"""Runner-side hardening (workers untouched)
 - Import from `notate.*` (no external `tools` package needed)
 - Preflight (inline) filters token pickles by length and rewrites paths
 - Keeps original training loop behavior as much as possible
"""

import pickle
import time
import yaml
from addict import Dict
import numpy as np
import pandas as pd
import torch
import random
import shutil
from collections import defaultdict
from tqdm import tqdm
import gc

# ==== import from notate.* (Colab / editable install friendly) ====
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
        max_drop_pct: 5.0          # %（各 split に個別適用）
        check_files: true
        report_dir: "./qc"         # 相対は result_dir 配下に解決
        apply_to_train: true       # train に適用
        apply_to_vals:  false      # validation にも適用（true で有効）
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

    # resolve report dir
    report_dir = pf.get('report_dir', os.path.join(result_dir, "qc"))
    if not os.path.isabs(report_dir):
        report_dir = os.path.join(result_dir, os.path.normpath(report_dir))
    os.makedirs(report_dir, exist_ok=True)

    # ---- 内部 helper: 1 split を preflight して config を書き換え ----
    def _apply_to_split(base_key: str, split_label: str):
        """
        base_key 例:
          - train: 'training.data.train.datasets.datasets'
          - valX : 'training.data.vals.<name>.datasets.datasets'
        """
        in_paths = _dget(cfg, f"{base_key}.input.path_list")
        tgt_paths = _dget(cfg, f"{base_key}.target.path_list")
        if in_paths is None or tgt_paths is None:
            logger.warning(f"[preflight] {split_label}: path_list not found. Skipped.")
            return None  # nothing to do

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
                    # malformed item -> drop
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

        # split 単位の drop 判定
        drop_pct = (100.0 * total_dropped / max(1, total_before))
        # split summary
        summary = {
            "split": split_label,
            "total_before": int(total_before),
            "total_after":  int(total_after),
            "total_dropped": int(total_dropped),
            "drop_pct":     float(drop_pct),
            "max_len":      int(max_len),
            "allowed_max_drop_pct": float(max_drop_pct),
        }
        # 保存（split 名付き）
        _save_pickle(summary, os.path.join(report_dir, f"preflight_summary_{split_label}.pkl"))
        with open(os.path.join(report_dir, f"preflight_summary_{split_label}.txt"), "w", encoding="utf-8") as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")

        if total_after == 0:
            raise SystemExit(f"[preflight] {split_label}: all samples dropped (0 remain).")
        if drop_pct > max_drop_pct:
            raise SystemExit(f"[preflight] {split_label}: drop {drop_pct:.2f}% > allowed {max_drop_pct:.2f}%")

        # config の書き換え
        _dset(cfg, f"{base_key}.input.path_list",
              cleaned_in_paths if len(cleaned_in_paths) > 1 else cleaned_in_paths[0])
        _dset(cfg, f"{base_key}.target.path_list",
              cleaned_tgt_paths if len(cleaned_tgt_paths) > 1 else cleaned_tgt_paths[0])

        logger.info(f"[preflight] {split_label}: kept={total_after} / {total_before} (drop={drop_pct:.2f}%)")
        return summary

    # ---- 適用先 split を組み立てて実行 ----
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

    # ---- 全体サマリ（任意） ----
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


# ====== hooks (kept, minimal change) ======
class NoticeAlarmHook(AlarmHook):
    def __init__(self, logger, studyname=None, **kwargs):
        super().__init__(logger=logger, **kwargs)
        if studyname is None:
            logger.warning("studyname not specified in NoticeAlarm.")
            studyname =  "(study noname)"
        self.studyname = studyname
    def ring(self, batch, model):
        print("ring")

hook_type2class['notice_alarm'] = NoticeAlarmHook


def main(config, args=None):
    # substitute variables (kept)
    config = subs_vars(config, {"$TIMESTAMP": timestamp()})
    trconfig = config.training

    # result dir & logger
    result_dir = make_result_dir(**trconfig.result_dir)
    logger = default_logger(result_dir+"/log.txt", trconfig.verbose.loglevel.stream, trconfig.verbose.loglevel.file)
    with open(result_dir+"/config.yaml", mode='w') as f:
        yaml.dump(config.to_dict(), f, sort_keys=False)
    if args is not None:
        logger.warning(f"options: {' '.join(args)}")

    # runner-side seed
    _seed_everything(int(trconfig.get('runner_seed', trconfig.get('model_seed', 0))))

    # --- runner-side preflight (new) ---
    try:
        config = run_preflight_and_rewrite_paths(config, result_dir, logger)
    except SystemExit as e:
        logger.error(str(e))
        raise

    # helper for epoch-wise dataloader rewrite (kept; path pattern update)
    def update_dataloader_for_epoch(epoch, config):
        # NOTE: original behavior for PubChem chunk rotation (kept)
        new_path = f"./data/Pubchem_chunk_pro_{epoch}_ran.pkl"
        new_path2 = f"./data/Pubchem_chunk_pro_{epoch}_can.pkl"
        updated_config = config.copy()
        try:
            updated_config['training']['data']['train']['datasets']['datasets']['input']['path_list'] = new_path
            updated_config['training']['data']['train']['datasets']['datasets']['target']['path_list'] = new_path2
        except Exception:
            pass
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

    # prepare model
    if 'model_seed' in trconfig:
        random.seed(trconfig.model_seed)
        np.random.seed(trconfig.model_seed)
        torch.manual_seed(trconfig.model_seed)
        torch.cuda.manual_seed(trconfig.model_seed)
    model = Model(config=config, logger=logger, **config.model)
    if trconfig.init_weight:
        model.load(**trconfig.init_weight)
    model.to(DEVICE)
    optimizer = get_optimizer(params=model.parameters(), **trconfig.optimizer)
    if trconfig.optimizer_init_weight:
        optimizer.load_state_dict(torch.load(trconfig.optimizer_init_weight))

    train_processes = [get_process(**process) for process in trconfig.train_loop]

    class SchedulerAlarmHook(AlarmHook):
        def __init__(self, scheduler, **kwargs):
            super().__init__(**kwargs)
            scheduler.setdefault('last_epoch', dl_train.step - 1)
            self.scheduler = get_scheduler(optimizer, **scheduler)
        def ring(self, batch, model):
            self.scheduler.step()
    hook_type2class['scheduler_alarm'] = SchedulerAlarmHook

    # Prepare abortion
    abort_step = trconfig.abortion.step or float('inf')
    abort_epoch = trconfig.abortion.epoch or float('inf')
    abort_time = trconfig.abortion.time or float('inf')

    # Prepare metrics
    accumulators = [ get_accumulator(logger=logger, **acc_config) for acc_config in trconfig.accumulators ]
    idx_accumulator = NumpyAccumulator(logger, input='idx', org_type='numpy')
    metrics = [ get_metric(logger=logger, name=name, **met_config) for name, met_config in trconfig.metrics.items() ]
    scores_df = pd.DataFrame(columns=[], dtype=float)
    if trconfig.stocks.score_df:
        try:
            scores_df = pd.read_csv(trconfig.stocks.score_df, index_col="Step")
        except Exception:
            logger.warning(f"Failed to load score_df: {trconfig.stocks.score_df}. Start from empty.")
            scores_df = pd.DataFrame(columns=[], dtype=float)
    val_processes = [get_process(**process) for process in trconfig.val_loop]
    if trconfig.val_loop_add_train:
        val_processes = train_processes + val_processes

    class ValidationAlarmHook(AlarmHook):
        eval_steps = []
        def ring(self, batch, model):
            step = batch['step']
            if step in self.eval_steps: return
            self.logger.info(f"Validating step{step:7} ...")
            model.eval()
            for x in metrics+accumulators+[idx_accumulator]: x.init()
            with torch.no_grad():
                for key, dl in dls_val.items():
                    for metric in metrics:
                        metric.set_val_name(key)
                    for batch0 in dl:
                        batch0 = model(batch0, processes=val_processes)
                        for x in metrics+accumulators+[idx_accumulator]:
                            x(batch0)
                        del batch0
                        torch.cuda.empty_cache()

            scores = {}
            for metric in metrics: scores = metric.calc(scores)
            batch.update(scores)
            for score_lb, score in scores.items():
                self.logger.info(f"  {score_lb:20}: {score:.3f}")
                scores_df.loc[step, score_lb] = score
            scores_df.to_csv(result_dir+"/val_score.csv", index_label="Step")

            idx = idx_accumulator.accumulate()
            idx = np.argsort(idx)
            for accumulator in accumulators:
                accumulator.save(f"{result_dir}/accumulates/{accumulator.input}/{step}", indices=idx)

            self.eval_steps.append(step)
            model.train()
    hook_type2class['validation_alarm'] = ValidationAlarmHook

    class CheckpointAlarm(AlarmHook):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            os.makedirs(f"{result_dir}/checkpoints", exist_ok=True)
            self.checkpoint_steps = []
        def ring(self, batch, model: Model):
            if batch['step'] in self.checkpoint_steps: return
            checkpoint_dir = f"{result_dir}/checkpoints/{batch['step']}"
            logger.info(f"Making checkpoint at step {batch['step']:6>}...")
            if len(self.checkpoint_steps) > 0:
                shutil.rmtree(f"{result_dir}/checkpoints/{self.checkpoint_steps[-1]}/")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_state_dict(f"{result_dir}/models/{batch['step']}")
            torch.save(optimizer.state_dict(), f"{checkpoint_dir}/optimizer.pth")
            dl_train.checkpoint(f"{checkpoint_dir}/dataloader_train")
            scores_df.to_csv(checkpoint_dir+"/val_score.csv", index_label="Step")
            save_rstate(f"{checkpoint_dir}/rstate")
            self.checkpoint_steps.append(batch['step'])
    hook_type2class['checkpoint_alarm'] = CheckpointAlarm

    pre_hooks = [get_hook(logger=logger, result_dir=result_dir, **hconfig) for hconfig in trconfig.pre_hooks.values()]
    post_hooks = [get_hook(logger=logger, result_dir=result_dir, **hconfig) for hconfig in trconfig.post_hooks.values()]

    # load random state
    set_rstate(trconfig.rstate)

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
                logger.warning(f"Use of abort_step, abort_time, abort_epoch is deprecated. Use AbortHook instead.")
                batch['end'] = True
            if 'end' in batch:
                break

            # training step
            batch = dl_train.get_batch(batch)
            start = time.time()
            batch = model(batch, processes=train_processes)
            loss = sum(batch[loss_name] for loss_name in trconfig.loss_names)
            if trconfig.regularize_loss.normalize:
                loss = loss / loss.detach()
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

            # epoch-wise dataloader update (kept behavior)
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
                if p.grad is None: continue
                grad_mean += torch.sum(p.grad**2)
                grad_numel += p.grad.numel()
                grad_max = max(grad_max, p.grad.max().item())
                grad_min = min(grad_min, p.grad.min().item())
            batch['grad_mean'] = grad_mean / grad_numel if grad_numel > 0 else 0.
            batch['grad_max'] = grad_max
            batch['grad_min'] = grad_min

            if trconfig.regularize_loss.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                    max_norm=trconfig.regularize_loss.clip_grad_norm, error_if_nonfinite=True)
            if trconfig.regularize_loss.clip_grad_value:
                torch.nn.utils.clip_grad_value_(model.parameters(),
                    clip_value=trconfig.regularize_loss.clip_grad_value)

            if dl_train.step % trconfig.schedule.opt_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
            batch['time'] = time.time() - start

            for hook in post_hooks:
                hook(batch, model)
            del batch, loss
            torch.cuda.empty_cache()
            if pbar is not None: pbar.update(1)

    for hook in pre_hooks+post_hooks:
        hook(batch, model)


if __name__ == '__main__':
    # Let notate.tools.args handle --config config.yaml etc.
    config = load_config2("", default_configs=['config'])
    main(config, sys.argv)
