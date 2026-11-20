"""Opt-in, ordered process execution for independent simulation scan rows."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import importlib
import inspect
import multiprocessing
import os

import numpy as np
from tqdm import tqdm

_OPTIONS = ContextVar('simulation_execution', default=(False, None))
_CANCEL = ContextVar('simulation_cancel', default=None)


def execution_settings():
    return _OPTIONS.get()


class SimulationCancelled(Exception):
    pass


def check_cancelled():
    event = _CANCEL.get()
    if event is not None and event.is_set():
        raise SimulationCancelled('Simulation stopped safely.')


@contextmanager
def cancellation(event):
    token = _CANCEL.set(event)
    try:
        check_cancelled()
        yield
    finally:
        _CANCEL.reset(token)


def _init_worker(cancel_event=None):
    _CANCEL.set(cancel_event)
    # Workers calculate only; avoid nested BLAS pools competing for the same CPUs.
    os.environ['MPLBACKEND'] = 'Agg'
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    global _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)


@contextmanager
def execution_options(parallel=False, workers=None):
    if workers is not None and (int(workers) != workers or workers < 1):
        raise ValueError('Workers must be a positive integer.')
    token = _OPTIONS.set((bool(parallel), workers))
    try:
        yield
    finally:
        _OPTIONS.reset(token)


def _run(task):
    check_cancelled()
    module, name, kwargs = task
    return getattr(importlib.import_module(module), name)(**kwargs, parallel=False)


def simulation_execution(function):
    """Expose execution options without changing the numerical implementation."""
    @wraps(function)
    def wrapped(*args, parallel=None, workers=None, **kwargs):
        enabled, defaults = _OPTIONS.get()
        with execution_options(enabled if parallel is None else parallel,
                               defaults if workers is None else workers):
            return function(*args, **kwargs)
    return wrapped


def parallel_scan(range_name=None, *, ranges_dict=False, detector_grid=False, targets=False, energy_mapping=False):
    """Partition only independent rows; merge in serial order without GUI workers."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, parallel=None, workers=None, **kwargs):
            enabled, default_workers = _OPTIONS.get()
            enabled = enabled if parallel is None else parallel
            if not enabled:
                return function(*args, **kwargs)
            bound = inspect.signature(function).bind(*args, **kwargs)
            bound.apply_defaults()
            values = dict(bound.arguments)
            if (energy_mapping and 'energy' in (values['param1_name'], values['param2_name'])
                    and not values.get('require_detector_hit') and 0 <= values['e_bandwidth'] < 200):
                with execution_options(True, workers if workers is not None else default_workers):
                    return function(*args, **kwargs)
            if values.get('detector_obj') is not None:
                from copy import copy
                values['detector_obj'] = copy(values['detector_obj'])
                values['detector_obj'].lab_grid = None
            if targets:
                # Motor IK uses the previous solution as a seed; preserve its continuity.
                if values.get('motor_chain') is not None:
                    print('Motor-chain inverse targeting retains serial continuation seeds.')
                    return function(*args, **kwargs)
                count = len(values['q_targets'])
                partitions = []
                for index in range(count):
                    part = dict(values)
                    for key in ('q_targets', 'q_target_pixels', 'q_target_errors_px', 'eta_deg_kept'):
                        part[key] = values[key][index:index + 1]
                    partitions.append(part)
            else:
                ranges = values[range_name]
                key = next(iter(ranges)) if ranges_dict else None
                interval = ranges[key] if ranges_dict else ranges
                start, stop, step = map(float, interval)
                if not np.isfinite([start, stop, step]).all() or step == 0 or (stop-start)*step < 0:
                    raise ValueError('Invalid scan range.')
                points = np.arange(start, stop + step, step)
                partitions = []
                for point in points:
                    part = dict(values)
                    if detector_grid:
                        part['_roty_values'] = [point]
                    else:
                        part['_parallel_row'] = (key, point)
                    partitions.append(part)
            if len(partitions) < 2:
                return function(*args, **kwargs)
            workers = workers if workers is not None else (default_workers or min(4, os.cpu_count() or 1))
            if int(workers) != workers or workers < 1:
                raise ValueError('Workers must be a positive integer.')
            tasks = [(function.__module__, function.__name__, part) for part in partitions]
            print(f'Simulation scan: {len(tasks)} row tasks; {min(int(workers), len(tasks))} processes.')
            with ProcessPoolExecutor(max_workers=min(int(workers), len(tasks)),
                                     mp_context=multiprocessing.get_context('spawn'),
                                     initializer=_init_worker, initargs=(_CANCEL.get(),)) as pool:
                results = [None] * len(tasks)
                pending = {}
                next_row = 0
                with tqdm(total=len(tasks), desc='Simulation rows') as progress:
                    try:
                        while next_row < len(tasks) or pending:
                            check_cancelled()
                            while next_row < len(tasks) and len(pending) < int(workers) * 2:
                                pending[pool.submit(_run, tasks[next_row])] = next_row
                                next_row += 1
                            done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                            for future in done:
                                results[pending.pop(future)] = future.result()
                                progress.update(1)
                    finally:
                        for future in pending:
                            future.cancel()
            if targets:
                from dataclasses import fields
                results = [result for result in results if result is not None]
                if not results:
                    return None
                merged = {}
                for field in fields(results[0]):
                    first = getattr(results[0], field.name)
                    merged[field.name] = (np.concatenate([getattr(result, field.name) for result in results])
                                          if isinstance(first, np.ndarray) else first)
                return type(results[0])(**merged)
            merged = {}
            for result in results:
                for key, rows in result.items():
                    merged.setdefault(key, []).extend(rows)
            return merged
        return wrapped
    return decorate
