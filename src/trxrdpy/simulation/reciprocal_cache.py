"""Bounded, foreground-first preparation of nearby reciprocal configurations."""
import atexit
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import math
from threading import Lock
import weakref

import numpy as np

_CACHES = weakref.WeakSet()


def _bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (bytes, str)):
        return len(value)
    if isinstance(value, dict):
        return sum(_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_bytes(v) for v in value)
    return 64


def neighbors(index, sizes, radius=2):
    strides = [math.prod(sizes[i+1:]) for i in range(len(sizes))]
    for distance in range(1, radius+1):
        for size, stride in zip(sizes, strides):
            position = (index // stride) % size
            for direction in (-1, 1):
                if 0 <= position + direction*distance < size:
                    yield index + direction*distance*stride


class NeighborhoodCache:
    def __init__(self, compute, sizes, *, parallel=False, workers=None,
                 max_entries=48, max_bytes=64*1024*1024):
        self.compute, self.sizes = compute, tuple(sizes)
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.values = OrderedDict()
        self.bytes = 0
        self.lock = Lock()
        self.generation = 0
        self.closed = False
        self.pending = []
        count = max(1, min(4, (workers or 4)-1)) if parallel else 1
        self.pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix='reciprocal-prefetch')
        _CACHES.add(self)

    def _put(self, index, value):
        size = _bytes(value)
        if size > self.max_bytes:
            return
        previous = self.values.pop(index, None)
        if previous is not None:
            self.bytes -= _bytes(previous)
        self.values[index] = value
        self.bytes += size
        while len(self.values)>self.max_entries or self.bytes>self.max_bytes:
            _, old = self.values.popitem(last=False)
            self.bytes -= _bytes(old)

    def get(self, index):
        if not 0 <= index < math.prod(self.sizes):
            raise ValueError('Configuration index is outside the saved scan.')
        with self.lock:
            if self.closed:
                raise RuntimeError('Reciprocal cache is closed.')
            self.generation += 1
            generation = self.generation
            for future in self.pending:
                future.cancel()
            self.pending.clear()
            value = self.values.get(index)
            if value is not None:
                self.values.move_to_end(index)
        # Never queue a foreground selection behind speculative work.
        if value is None:
            value = self.compute(index)
        with self.lock:
            if not self.closed and generation == self.generation:
                self._put(index, value)
                for nearby in neighbors(index, self.sizes):
                    if nearby not in self.values:
                        self.pending.append(self.pool.submit(self._prepare, nearby, generation))
        return value

    def _prepare(self, index, generation):
        with self.lock:
            if self.closed or generation != self.generation:
                return
        try:
            value = self.compute(index)
        except Exception:
            # A speculative failure must not affect the current view. Foreground
            # selection will retry and report the error through its normal path.
            return
        with self.lock:
            if not self.closed and generation == self.generation:
                self._put(index, value)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.generation += 1
            for future in self.pending:
                future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        with self.lock:
            self.values.clear()
            self.bytes = 0
            self.pending.clear()
        self.compute = None


def close_caches():
    for cache in list(_CACHES):
        cache.close()


atexit.register(close_caches)
