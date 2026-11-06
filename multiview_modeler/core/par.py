"""par.py - run independent per-view work on several CPU cores (threads: numpy / scipy release the GIL in the heavy parts)"""
import os
from concurrent.futures import ThreadPoolExecutor

N = max(1, min(16, (os.cpu_count() or 2)))


def pmap(fn, items):
    items = list(items)
    if N == 1 or len(items) < 2:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(N, len(items))) as ex:
        return list(ex.map(fn, items))
