# All Need
import torch
import numpy as np

# Strict: define process-op resolver locally to avoid coupling to core.
import importlib
PRINT_PROCESS = False
PROCESS_OPS_MODULE = "notate.training.process_ops"  # 既存のops実装モジュール名に合わせてください
ALLOWED_PROCESS_OPS = {
    "forward", "loss", "backward", "step", "metrics", "accumulate",
    "zero_grad", "clip_grad"
}

_OPS_MOD = None
try:
    _OPS_MOD = importlib.import_module(PROCESS_OPS_MODULE)
except Exception:
    _OPS_MOD = None

def function_config2func(cfg, logger=None):
    # cfg: "opname" or {"op": "...", ...} or {"type": "...", ...}
    if isinstance(cfg, str):
        name = cfg.strip()
        kwargs = {}
    elif isinstance(cfg, dict):
        name = str(cfg.get("op") or cfg.get("type") or "").strip()
        kwargs = {k: v for k, v in cfg.items() if k not in ("op", "type")}
    else:
        raise SystemExit("[CONFIG ERROR] process op spec must be a string or mapping")
    if not name:
        raise SystemExit("[CONFIG ERROR] process op name is required")
    if name not in ALLOWED_PROCESS_OPS:
        raise SystemExit(f"[CONFIG ERROR] Unknown process op '{name}'. Allowed: {sorted(ALLOWED_PROCESS_OPS)}")
    if _OPS_MOD is None:
        raise SystemExit(
            f"[CONFIG ERROR] Process ops module '{PROCESS_OPS_MODULE}' could not be imported. "
            "Define your ops there or set PROCESS_OPS_MODULE correctly."
        )
    try:
        fn = getattr(_OPS_MOD, name)
    except AttributeError:
        raise SystemExit(
            f"[CONFIG ERROR] Process op '{name}' is not defined in '{PROCESS_OPS_MODULE}'."
        )
    if logger:
        logger.debug("[process-op] %s(%s)", name, ", ".join(f"{k}={v!r}" for k, v in kwargs.items()))
    return fn, kwargs


class Process:
    def __init__(self):
        pass
    def __call__(self, model, batch):
        raise NotImplementedError

class CallProcess(Process):
    def __init__(self, input, output=None, **kwargs):
        self.input = input
        self.output = output
        if self.output is None:
            self.output = self.input
        self.kwargs = kwargs

    def __call__(self, model, batch):
        callable_ = self.get_callable(model)
        if self.input is None:
            output = callable_(**self.kwargs)
        elif isinstance(self.input, str):
            output = callable_(batch[self.input], **self.kwargs)
        elif isinstance(self.input, list):
            output = callable_(*[batch[i] for i in self.input], **self.kwargs)
        elif isinstance(self.input, dict):
            output = callable_(**{name: batch[i] for name, i in self.input.items()}, **self.kwargs)
        else:
            raise ValueError(f'Unsupported type of input: {self.input}')
        if isinstance(self.output, str):
            batch[self.output] = output
        elif isinstance(self.output, list):
            for oname, o in zip(self.output, output):
                batch[oname] = o
        else:
            raise ValueError(f'Unsupported type of output: {self.output}')
        
    def get_callable(self, model):
        raise NotImplementedError
    
class ForwardProcess(CallProcess):
    def __init__(self, module, input, output=None, **kwargs):
        """
        Parameters
        ----------
        module: str
            Name of module.
        input: str, list[str] or dict[str, str]
            Name of input(s) in the batch to the module.
        output: str, list[str], or None
            Name of output(s) in the batch from the module.
            If None, input is used as output (inplace process)
        kwargs: dict
            他のパラメータはモジュールに直接渡される。    
        """
        super().__init__(input, output, **kwargs)
        self.module = module

    def get_callable(self, model):
        return  model[self.module]
    
class FunctionProcess(CallProcess):
    def __init__(self, function, input, output=None, **kwargs):
        """
        Parameters
        ----------
        function: dict
            Input for function_config2func
        input: str, list[str] or dict[str, str]
            Name of input(s) in the batch to the module.
        output: str, list[str], or None
            Name of output(s) in the batch from the module.
            If None, input is used as output (inplace process) 
        kwargs: dict
            他のパラメータはモジュールに直接渡される。    
        """
        super().__init__(input, output, **kwargs)
        self.function = function_config2func(function)

    def get_callable(self, model):
        return self.function

class IterateProcess(Process):
    def __init__(self, length, processes, i_name='iterate_i'):
        """
        Parameters
        ----------
        length: int | str
            Length of iteration | name of it in batch.
        processes: list[dict]
            Parameters for processes to iterate
        i_name: str
            Name of index of iteration in batch        
        """
        self.length = length
        self.processes = [get_process(**process) for process in processes]
        self.i_name = i_name

    def __call__(self, model, batch):
        if isinstance(self.length, int): length = self.length
        else: length = batch[self.length]
        for i in range(length):
            batch[self.i_name] = i
            for i, process in enumerate(self.processes):

                if PRINT_PROCESS:
                    # Show parameters
                    print(f"---process {i}---")
                    for key, value in batch.items():
                        if isinstance(value, (torch.Tensor, np.ndarray)):
                            print(f"  {key}: {list(value.shape)}")
                        else:
                            print(f"  {key}: {type(value).__name__}")
                

                process(model, batch)

process_type2class = {
    'forward': ForwardProcess,
    'function': FunctionProcess,
    'iterate': IterateProcess
}

def get_process(type='forward', **kwargs):
    return process_type2class[type](**kwargs)
