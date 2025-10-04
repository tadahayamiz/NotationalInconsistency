# 何もloadしなくてよいもののみ
from contextlib import contextmanager

@contextmanager
def nullcontext():
    yield None

def prog(marker='*'):
    print(marker, flush=True, end='')

# Merged from models/utils.py
def check_leftargs(self, logger, kwargs, show_content=False):
    """Check for unused keyword arguments and warn if any exist."""
    if len(kwargs) > 0 and logger is not None:
        logger.warning(f"Unknown kwarg in {type(self).__name__}: {kwargs if show_content else list(kwargs.keys())}")

EMPTY = lambda x: x  # Identity function