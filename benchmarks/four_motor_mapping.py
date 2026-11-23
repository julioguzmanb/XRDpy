"""Profile four-motor mapping work and compare process splitting strategies.

Run with ``OPENBLAS_NUM_THREADS=1 python benchmarks/four_motor_mapping.py``.
The small preset has 9x9 plotted points and 3x3 extra slices. ``--preset large``
uses 25x25 plotted points and 5x5 slices. Timings are machine-specific.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from trxrdpy.simulation import sample, single_crystal, utils
from trxrdpy.simulation.geometry import MotorChain
from trxrdpy.simulation.parallel import _init_worker


HKLS = [(-1, 1, 0), (-1, 1, 1), (-2, 1, 1), (-3, 0, 0), (-2, -1, 1)]


def motor_chain():
    return MotorChain.from_list([
        {'name': 'omega', 'axis': 'z', 'frame': 'lab'},
        {'name': 'chi', 'axis': 'y', 'frame': 'local'},
        {'name': 'phi', 'axis': 'x', 'frame': 'local'},
        {'name': 'eta', 'axis': 'z', 'frame': 'local'},
    ], name='four_motor_benchmark')


def ranges_for(preset):
    plotted, extra = (9, 3) if preset == 'small' else (25, 5)
    def interval(count, step):
        half = count // 2 * step
        return (-half, half, step)
    return {'omega': interval(plotted, 5), 'chi': interval(plotted, 5),
            'phi': interval(extra, 10), 'eta': interval(extra, 10)}


def core_options(chain, ranges):
    return dict(hkls=HKLS, initial_crystal_orientation=4 * np.eye(3),
                wavelength=utils.energy_to_wavelength(15000), e_bandwidth=50,
                motor_chain=chain, scan_ranges=ranges)


def slice_options(chain, ranges):
    return dict(motor1_name='omega', motor2_name='chi',
                motor1_range=ranges['omega'], motor2_range=ranges['chi'],
                sam_space_group=221, sam_a=4, sam_b=4, sam_c=4,
                sam_alpha=90, sam_beta=90, sam_gamma=90,
                energy=15000, e_bandwidth=50, hkls_names=HKLS,
                motor_chain=chain,
                extra_scan_ranges={'phi': ranges['phi'], 'eta': ranges['eta']})


def count(result):
    if 'slices' in result:
        return sum(sum(len(points) for points in item['points'].values())
                   for item in result['slices'].values())
    return sum(len(points) for points in result.values())


def solution_set(result):
    """Compare the actual motor combinations, including extra slice angles."""
    solutions = set()
    if 'slices' in result:
        for index, extra_values in enumerate(itertools.product(*result['extra_axes'].values())):
            for hkl, pairs in result['slices'][str(index)]['points'].items():
                for omega, chi in pairs:
                    solutions.add((hkl, float(omega), float(chi),
                                   *(float(value) for value in extra_values)))
    else:
        for hkl, rows in result.items():
            for angles in rows:
                solutions.add((hkl, *(float(angles[name]) for name in
                                      ('omega', 'chi', 'phi', 'eta'))))
    return solutions


def run_piece(task):
    options, fixed_angles = task
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return sample.find_Bragg_orientations_with_chain(
            **options, fixed_angles=fixed_angles, parallel=False)


def pooled_slices(chain, ranges, workers, chunk_rows=None):
    values = {name: np.arange(start, stop + step, step)
              for name, (start, stop, step) in ranges.items()}
    omega_chunks = [values['omega']]
    if chunk_rows is not None:
        omega_chunks = [values['omega'][start:start + chunk_rows]
                        for start in range(0, len(values['omega']), chunk_rows)]
    tasks = []
    for phi, eta in itertools.product(values['phi'], values['eta']):
        for omega in omega_chunks:
            piece_ranges = {'omega': (omega[0], omega[-1], ranges['omega'][2]),
                            'chi': ranges['chi']}
            tasks.append((core_options(chain, piece_ranges), {'phi': phi, 'eta': eta}))
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                             initializer=_init_worker) as pool:
        pieces = list(pool.map(run_piece, tasks))
    return sum(count(piece) for piece in pieces)


def timed(label, action):
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = action()
    elapsed = time.perf_counter() - start
    accepted = result if isinstance(result, int) else count(result)
    print(f'{label:27} {elapsed:8.3f} s  accepted={accepted}', flush=True)
    return accepted, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=('small', 'large'), default='small')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--chunk-rows', type=int, default=5)
    parser.add_argument('--include-current-parallel', action='store_true',
                        help='Run the current pool-per-slice path (very slow on small slices).')
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_rows < 1:
        parser.error('workers and chunk-rows must be positive')
    chain = motor_chain()
    ranges = ranges_for(args.preset)
    n = {name: len(np.arange(start, stop + step, step))
         for name, (start, stop, step) in ranges.items()}
    print(f'grid={n}, combinations={np.prod(list(n.values()))}, workers={args.workers}', flush=True)
    sliced = slice_options(chain, ranges)
    core = core_options(chain, ranges)
    expected, reference = timed('current slices, serial', lambda: single_crystal.scan_two_motors_for_Bragg_condition(
        **sliced, parallel=False))
    if args.include_current_parallel:
        actual, result = timed('current slices, processes', lambda: single_crystal.scan_two_motors_for_Bragg_condition(
            **sliced, parallel=True, workers=args.workers))
        assert actual == expected
        assert solution_set(result) == solution_set(reference)
    for label, action in (
        ('one 4D scan, serial', lambda: sample.find_Bragg_orientations_with_chain(
            **core, parallel=False)),
        ('one 4D scan, processes', lambda: sample.find_Bragg_orientations_with_chain(
            **core, parallel=True, workers=args.workers)),
        ('one pool, one job/slice', lambda: pooled_slices(chain, ranges, args.workers)),
        (f'one pool, {args.chunk_rows} rows/job', lambda: pooled_slices(
            chain, ranges, args.workers, chunk_rows=args.chunk_rows)),
    ):
        actual, result = timed(label, action)
        assert actual == expected
        if not isinstance(result, int):
            assert solution_set(result) == solution_set(reference)


if __name__ == '__main__':
    main()
