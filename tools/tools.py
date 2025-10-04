# 何もloadしなくてよいもののみ
from contextlib import contextmanager

@contextmanager
def nullcontext():
    yield None

def prog(marker='*'):
    print(marker, flush=True, end='')