from concurrent.futures import wait
from threading import Event

import numpy as np
import pytest

from trxrdpy.simulation.reciprocal_cache import NeighborhoodCache, close_caches, neighbors


@pytest.fixture(autouse=True)
def cleanup():
    yield
    close_caches()


def test_neighbors_do_not_wrap_or_enumerate_product():
    assert list(neighbors(0, (3, 4))) == [4, 1, 8, 2]
    assert list(neighbors(11, (3, 4))) == [7, 10, 3, 9]
    assert len(list(neighbors(10**40-1, (100000,)*8))) == 16


def test_prefetch_reuses_results_and_bounds_memory():
    called = []
    def compute(index):
        called.append(index)
        return bytes([index])*8
    cache = NeighborhoodCache(compute, (20,), max_entries=3, max_bytes=24)
    assert cache.get(0) == bytes(8)
    wait(cache.pending, timeout=5)
    assert set(called) == {0, 1, 2}
    cache.get(1)
    wait(cache.pending, timeout=5)
    assert called.count(1) == 1
    assert len(cache.values) <= 3 and cache.bytes <= 24
    cache.close()
    assert cache.bytes == 0 and not cache.values
    with pytest.raises(RuntimeError, match='closed'):
        cache.get(0)


def test_foreground_bypasses_pending_work_and_discards_stale_results():
    started, release = Event(), Event()
    def compute(index):
        if index == 1:
            started.set()
            assert release.wait(5)
        return bytes([index])
    cache = NeighborhoodCache(compute, (20,))
    try:
        cache.get(0)
        assert started.wait(5)
        assert cache.get(10) == bytes([10])
        release.set()
        wait(cache.pending, timeout=5)
        assert 1 not in cache.values
        assert 2 not in cache.values
        assert 10 in cache.values
    finally:
        release.set()
        cache.close()


def test_speculative_errors_are_reported_only_when_selected():
    def compute(index):
        if index == 1:
            raise ValueError('failed configuration')
        return b'ok'
    cache = NeighborhoodCache(compute, (3,), parallel=True, workers=3)
    assert cache.get(0) == b'ok'
    wait(cache.pending, timeout=5)
    with pytest.raises(ValueError, match='failed configuration'):
        cache.get(1)
    assert cache.pool._max_workers == 2


def test_oversized_results_are_not_retained():
    cache = NeighborhoodCache(lambda _: np.zeros(100), (1,), max_bytes=16)
    assert cache.get(0).shape == (100,)
    assert cache.bytes == 0 and not cache.values
