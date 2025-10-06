# src/notate/modules/pipeline.py
import torch
import torch.nn as nn

class PipelineModule(nn.Module):
    """
    設定ファイルの train_loop / val_loop を解釈して実行する“入口”モジュール。
    ForwardProcess からは本モジュールを1回呼べばよい。
    他のモジュールは名前で解決して呼び出す（_module_registry 経由）。
    """
    def __init__(self, train_loop=None, val_loop=None, loss_names=None, logger=None):
        super().__init__()
        self.train_loop = train_loop or []
        self.val_loop   = val_loop or []
        self.loss_names = loss_names or []
        self.logger     = logger
        self._module_registry = None      # {name -> nn.Module}
        self._function_registry = {}      # 追加したい関数があれば差し込み

    # Model 側からレジストリを注入できるように
    def attach_registry(self, registry: dict):
        self._module_registry = registry

    # ForwardProcess から呼ばれる想定
    # input=dict で来る実装・**batch で来る実装の両方を許容
    def forward(self, input=None, mode="train", **batch):
        if input is None and batch:
            ctx = dict(batch)
        elif isinstance(input, dict):
            ctx = dict(input)
        else:
            ctx = {}

        loop = self.train_loop if mode == "train" else self.val_loop

        for step in loop:
            stype = step.get("type", "module")
            if stype == "function":
                self._run_function_step(ctx, step); continue
            if stype == "iterate":
                self._run_iterate_step(ctx, step);  continue
            self._run_module_step(ctx, step)  # module

        losses = {name: ctx[name] for name in self.loss_names if name in ctx}
        return losses if losses else ctx

    # ---------- helpers ----------
    def _get_module(self, name: str):
        if not self._module_registry:
            raise RuntimeError("PipelineModule: module registry is not attached.")
        if name not in self._module_registry:
            raise KeyError(f"PipelineModule: unknown module '{name}'")
        return self._module_registry[name]

    def _resolve_input(self, ctx, spec):
        if spec is None: return None
        if isinstance(spec, list): return [self._resolve_input(ctx, s) for s in spec]
        if isinstance(spec, dict): return {k: self._resolve_input(ctx, v) for k, v in spec.items()}
        return ctx[spec] if isinstance(spec, str) else spec

    def _as_kwargs(self, ctx, src):
        if src is None: return {}
        if isinstance(src, dict): return {k: self._resolve_input(ctx, v) for k, v in src.items()}
        if isinstance(src, list): return {"args": [self._resolve_input(ctx, v) for v in src]}
        return {"input": self._resolve_input(ctx, src)}

    def _write_output(self, ctx, dst, out):
        if dst is None: return
        if isinstance(dst, list):
            if isinstance(out, (list, tuple)) and len(dst) == len(out):
                for k, v in zip(dst, out): ctx[k] = v
            else:
                ctx[dst[0]] = out
        elif isinstance(dst, str):
            ctx[dst] = out
        else:
            raise RuntimeError(f"PipelineModule: invalid output spec: {dst}")

    def _run_module_step(self, ctx, step):
        name = step.get("module")
        if not name: raise RuntimeError(f"PipelineModule: module name missing in step: {step}")
        mod  = self._get_module(name)
        kwargs = self._as_kwargs(ctx, step.get("input"))
        if "mode" in step:
            out = mod(**kwargs, mode=step["mode"])
        else:
            out = mod(**kwargs)
        self._write_output(ctx, step.get("output"), out)

    def _run_function_step(self, ctx, step):
        spec = step.get("function", {})
        ftype = spec.get("type")
        fn = self._resolve_function(ftype, spec)
        x  = self._resolve_input(ctx, step.get("input"))
        fn_kwargs = {k: v for k, v in spec.items() if k != "type"}
        out = fn(x, **fn_kwargs)
        self._write_output(ctx, step.get("output"), out)

    def _run_iterate_step(self, ctx, step):
        length = self._resolve_input(ctx, step["length"])
        procs  = step["processes"]
        for i in range(int(length)):
            ctx["iterate_i"] = i
            for sub in procs:
                if sub.get("type", "module") == "function":
                    self._run_function_step(ctx, sub)
                else:
                    self._run_module_step(ctx, sub)
        ctx.pop("iterate_i", None)

    # ---------- minimal built-ins ----------
    def _resolve_function(self, ftype, spec):
        if ftype in self._function_registry:
            return self._function_registry[ftype]
        if ftype == "transpose":
            def _fn(x, dim0=0, **kw):
                dim1 = kw.get("...1", 1)
                return x.transpose(dim0, dim1)
            return _fn
        raise KeyError(f"PipelineModule: unknown function '{ftype}'")
