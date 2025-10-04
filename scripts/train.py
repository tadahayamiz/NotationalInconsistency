# scripts/train.py
import sys, os, argparse, gc, time, random, pickle, shutil, yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from addict import Dict

from notate.tools.path import make_result_dir, timestamp
from notate.tools.logger import default_logger
from notate.tools.args import subs_vars
from notate.tools.tools import nullcontext

from notate.data import get_dataloader, get_accumulator, NumpyAccumulator
from notate.training import (
    get_metric, get_optimizer, get_scheduler, get_process,
    AlarmHook, hook_type2class, get_hook
)
from notate.core import Model


def _deepcopy_as_dict(d):
    try:
        return d.to_dict()
    except AttributeError:
        return yaml.safe_load(yaml.dump(d))


def save_rstate(dirname):
    os.makedirs(dirname, exist_ok=True)
    with open(f"{dirname}/random.pkl", 'wb') as f:
        pickle.dump(random.getstate(), f)
    with open(f"{dirname}/numpy.pkl", 'wb') as f:
        pickle.dump(np.random.get_state(), f)
    torch.save(torch.get_rng_state(), f"{dirname}/torch.pt")
    if torch.cuda.is_available():
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
    if 'cuda' in config and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch.load(config.cuda))


def _format_epoch_path(path_like, epoch):
    if isinstance(path_like, str) and "{epoch}" in path_like:
        return path_like.format(epoch=epoch)
    return path_like


def main(cfg: Dict, argv=None):
    # ----- variable expansion -----
    # Merge TIMESTAMP and user-defined variables (if provided)
    repl = {"$TIMESTAMP": timestamp()}
    if "variables" in cfg:
        # allow both str and numbers
        for k, v in cfg["variables"].items():
            repl[k] = v
    cfg = subs_vars(cfg, repl)
    cfg = Dict(cfg)  # ensure attribute access after substitution
    tr = cfg.training

    # ----- result dir & logger -----
    result_dir = make_result_dir(**tr.result_dir)
    logger = default_logger(result_dir + "/log.txt",
                            tr.verbose.loglevel.stream,
                            tr.verbose.loglevel.file)
    with open(result_dir + "/config.yaml", "w") as f:
        yaml.dump(_deepcopy_as_dict(cfg), f, sort_keys=False)
    if argv is not None:
        logger.warning(f"options: {' '.join(argv)}")

    # ----- device / determinism -----
    DEVICE = torch.device('cuda', index=tr.gpuid or 0) if torch.cuda.is_available() else torch.device('cpu')
    logger.warning(f"DEVICE: {DEVICE}")
    if tr.get('detect_anomaly', False):
        torch.autograd.set_detect_anomaly(True)
    if tr.get('deterministic', True):
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True
        torch.backends.cudnn.benchmark = False

    # helper: epoch-wise dataloader refresh using {epoch} placeholder
    def update_dataloader_for_epoch(epoch, base_cfg: Dict):
        updated = Dict(_deepcopy_as_dict(base_cfg))
        try:
            ds = updated.training.data.train.datasets.datasets
            for k in ["input", "target"]:
                if "path_list" in ds[k]:
                    ds[k]["path_list"] = _format_epoch_path(ds[k]["path_list"], epoch)
        except Exception:
            pass
        return get_dataloader(logger=logger, device=DEVICE, **updated.training.data.train)

    # ----- data -----
    dl_train = get_dataloader(logger=logger, device=DEVICE, **tr.data.train)
    dls_val = {name: get_dataloader(logger=logger, device=DEVICE, **val_cfg)
               for name, val_cfg in tr.data.vals.items()}

    # ----- model / optimizer / processes -----
    if 'model_seed' in tr:
        random.seed(tr.model_seed); np.random.seed(tr.model_seed)
        torch.manual_seed(tr.model_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(tr.model_seed)

    model = Model(logger=logger, **cfg.model)
    if tr.init_weight:
        model.load(**tr.init_weight)
    model.to(DEVICE)

    optimizer = get_optimizer(params=model.parameters(), **tr.optimizer)
    if tr.get('optimizer_init_weight'):
        optimizer.load_state_dict(torch.load(tr.optimizer_init_weight))
    train_processes = [get_process(**p) for p in tr.train_loop]

    # legacy aborts (prefer AbortHook)
    abort_step  = tr.get('abortion', {}).get('step',  float('inf'))
    abort_epoch = tr.get('abortion', {}).get('epoch', float('inf'))
    abort_time  = tr.get('abortion', {}).get('time',  float('inf'))

    # metrics & accumulators
    accumulators = [get_accumulator(logger=logger, **a) for a in tr.accumulators]
    idx_acc = NumpyAccumulator(logger, input='idx', org_type='numpy')
    metrics = [get_metric(logger=logger, name=name, **m) for name, m in tr.metrics.items()]
    stocks = tr.get('stocks', {})
    scores_df = pd.read_csv(stocks.get('score_df'), index_col="Step") if stocks.get('score_df') else pd.DataFrame(columns=[], dtype=float)

    val_processes = [get_process(**p) for p in tr.val_loop]
    if tr.get('val_loop_add_train', False):
        val_processes = train_processes + val_processes

    # ----- local hooks that need runtime refs -----
    class ValidationAlarmHook(AlarmHook):
        eval_steps = []
        def ring(self, batch, model):
            step = batch['step']
            if step in self.eval_steps: return
            self.logger.info(f"Validating step{step:7} ...")
            model.eval()
            for x in metrics + accumulators + [idx_acc]: x.init()
            with torch.no_grad():
                for key, dl in dls_val.items():
                    for metric in metrics: metric.set_val_name(key)
                    for b0 in dl:
                        b0 = model(b0, processes=val_processes)
                        for x in metrics + accumulators + [idx_acc]: x(b0)
                        del b0; torch.cuda.empty_cache()
            scores = {}
            for metric in metrics: scores = metric.calc(scores)
            batch.update(scores)
            for k, v in scores.items():
                self.logger.info(f"  {k:20}: {v:.3f}")
                scores_df.loc[step, k] = v
            scores_df.to_csv(result_dir + "/val_score.csv", index_label="Step")
            idx = idx_acc.accumulate(); idx = np.argsort(idx)
            for acc in accumulators:
                acc.save(f"{result_dir}/accumulates/{acc.input}/{step}", indices=idx)
            self.eval_steps.append(step)
            model.train()

    class CheckpointAlarmHook(AlarmHook):
        def __init__(self, **kw):
            super().__init__(**kw)
            os.makedirs(f"{result_dir}/checkpoints", exist_ok=True)
            self.ck_steps = []
        def ring(self, batch, model: Model):
            if batch['step'] in self.ck_steps: return
            ckdir = f"{result_dir}/checkpoints/{batch['step']}"
            self.logger.info(f"Making checkpoint at step {batch['step']:6>}...")
            if self.ck_steps:
                shutil.rmtree(f"{result_dir}/checkpoints/{self.ck_steps[-1]}/", ignore_errors=True)
            os.makedirs(ckdir, exist_ok=True)
            model.save_state_dict(f"{result_dir}/models/{batch['step']}")
            torch.save(optimizer.state_dict(), f"{ckdir}/optimizer.pth")
            dl_train.checkpoint(f"{ckdir}/dataloader_train")
            scores_df.to_csv(ckdir + "/val_score.csv", index_label="Step")
            save_rstate(f"{ckdir}/rstate")
            self.ck_steps.append(batch['step'])

    class SchedulerAlarmHook(AlarmHook):
        def __init__(self, scheduler, **kw):
            super().__init__(**kw)
            scheduler.setdefault('last_epoch', dl_train.step - 1)
            self.scheduler = get_scheduler(optimizer, **scheduler)
        def ring(self, batch, model):
            self.scheduler.step()

    # register local hooks
    hook_type2class['validation_alarm'] = ValidationAlarmHook
    hook_type2class['checkpoint_alarm'] = CheckpointAlarmHook
    hook_type2class['scheduler_alarm'] = SchedulerAlarmHook
    pre_hooks  = [get_hook(logger=logger, result_dir=result_dir, **h) for h in tr.get('pre_hooks', {}).values()]
    post_hooks = [get_hook(logger=logger, result_dir=result_dir, **h) for h in tr.get('post_hooks', {}).values()]

    # rng restore
    if hasattr(tr, "rstate"):
        set_rstate(tr.rstate)

    # ----- training loop -----
    training_start = time.time()
    logger.info("Training started.")
    print("Start")
    steps_per_epoch = 30000  # move to config if needed

    with (tqdm(total=None, initial=dl_train.step) if tr.verbose.show_tqdm else nullcontext()) as pbar:
        now = time.time()
        optimizer.zero_grad()
        epoch = 0
        while True:
            batch = {'step': dl_train.step, 'epoch': dl_train.epoch}
            for h in pre_hooks: h(batch, model)

            if dl_train.step >= abort_step or now - training_start >= abort_time or dl_train.epoch >= abort_epoch:
                logger.warning("Use of abort_step/time/epoch is deprecated. Prefer AbortHook.")
                batch['end'] = True
            if 'end' in batch: break

            # one step
            b = dl_train.get_batch(batch)
            start = time.time()
            b = model(b, processes=train_processes)
            loss = sum(b[name] for name in tr.loss_names)
            if tr.regularize_loss.normalize: loss = loss / loss.detach()
            try:
                loss.backward()
            except Exception as e:
                os.makedirs(f"{result_dir}/error/batch", exist_ok=True)
                for k, v in b.items():
                    if isinstance(v, torch.Tensor):
                        torch.save(v, f"{result_dir}/error/batch/{k}.pt")
                    else:
                        with open(f"{result_dir}/error/batch/{k}.pkl", 'wb') as f:
                            pickle.dump(v, f)
                model.save_state_dict(f"{result_dir}/error/model")
                raise e

            # optional epoch roll
            if dl_train.step > 0 and dl_train.step % steps_per_epoch == 0:
                epoch += 1
                del dl_train; gc.collect()
                dl_train = update_dataloader_for_epoch(epoch, cfg)
                dl_train.step = epoch * steps_per_epoch

            # grad stats
            gm = 0; gn = 0; gmax = 0; gmin = 0
            for p in model.parameters():
                if p.grad is None: continue
                gm += torch.sum(p.grad ** 2); gn += p.grad.numel()
                gmax = max(gmax, p.grad.max().item()); gmin = min(gmin, p.grad.min().item())
            b['grad_mean'] = gm / gn if gn > 0 else 0.; b['grad_max'] = gmax; b['grad_min'] = gmin

            if tr.regularize_loss.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=tr.regularize_loss.clip_grad_norm, error_if_nonfinite=True)
            if tr.regularize_loss.clip_grad_value:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=tr.regularize_loss.clip_grad_value)

            if dl_train.step % tr.schedule.opt_freq == 0:
                optimizer.step(); optimizer.zero_grad()
            b['time'] = time.time() - start

            for h in post_hooks: h(b, model)
            del b, loss
            torch.cuda.empty_cache()
            if pbar is not None: pbar.update(1)

    for h in pre_hooks + post_hooks: h(batch, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training entrypoint")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML configuration file (relative or absolute).")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_raw = yaml.safe_load(f)

    # Allow plain dict; convert to Dict later in main after substitution
    main(cfg_raw, sys.argv)
