"""Energy/rotation mappings without repeating geometry at every energy."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextvars import copy_context
import numpy as np
from tqdm import tqdm

from . import sample, utils
from .parallel import check_cancelled, execution_settings


def energy_rotation_points(initial, hkls, names, grids, order, scan_angles, bandwidth):
    energy_column = names.index('energy')
    angle_column = 1-energy_column
    energies, angles = grids[energy_column], grids[angle_column]
    if not np.isfinite(energies).all() or np.any(energies <= 0):
        raise ValueError('Scanned energies must be positive and finite.')
    permutation = np.argsort(energies)
    sorted_energy = energies[permutation]
    wavelengths = utils.energy_to_wavelength(sorted_energy)*1e10
    keys = [f'{hkl}' for hkl in hkls]
    enabled, requested = execution_settings()
    workers = int(requested or 4) if enabled else 1
    if workers < 1:
        raise ValueError('Workers must be positive.')
    chunks = [range(start, min(start+32, len(angles))) for start in range(0, len(angles), 32)]
    print(f'Energy/rotation mapping: {len(angles):,} unique orientations, '
          f'{len(energies):,} sampled energies; {min(workers, len(chunks))} shared-memory worker(s).', flush=True)

    def compute(indices):
        result = [[] for _ in hkls]
        for j in indices:
            check_cancelled()
            rotation = dict(scan_angles or {})
            name = names[angle_column]
            rotation[name] = rotation.get(name, 0)+angles[j]
            direct = utils.apply_rotation(initial, *(rotation.get('rot'+axis, 0) for axis in 'xyz'), order)
            q = sample.calculate_q_hkl(hkls, sample.cal_reciprocal_lattice(direct))
            norm2 = np.linalg.norm(q, axis=1)**2
            for h, vector in enumerate(q):
                if vector[0] >= 0:
                    continue
                critical = -(utils.energy_to_wavelength(1)*1e10/(4*np.pi))*norm2[h]/vector[0]
                lo = max(0, np.searchsorted(sorted_energy, critical/(1+bandwidth/200), side='left')-1)
                hi = min(len(energies), np.searchsorted(sorted_energy, critical/(1-bandwidth/200), side='right')+1)
                # Recheck neighboring boundary samples using the original arithmetic.
                ratio = -(wavelengths[lo:hi]/(4*np.pi))*norm2[h]/vector[0]
                selected = permutation[lo:hi][(ratio >= 1-bandwidth/200) & (ratio <= 1+bandwidth/200)]
                if len(selected):
                    pair = np.column_stack((selected, np.full(len(selected), j)))
                    result[h].append(pair if energy_column == 0 else pair[:, ::-1])
        return [np.concatenate(rows) if rows else np.empty((0, 2), dtype=int) for rows in result]

    blocks = []
    with tqdm(total=len(angles), desc='Unique mapping orientations') as progress:
        if workers == 1:
            for chunk in chunks:
                check_cancelled()
                blocks.append(compute(chunk))
                progress.update(len(chunk))
        else:
            with ThreadPoolExecutor(max_workers=min(workers, max(1, len(chunks)))) as pool:
                pending, next_chunk = {}, 0
                try:
                    while pending or next_chunk < len(chunks):
                        check_cancelled()
                        while next_chunk < len(chunks) and len(pending) < workers*2:
                            chunk = chunks[next_chunk]
                            pending[pool.submit(copy_context().run, compute, chunk)] = len(chunk)
                            next_chunk += 1
                        done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                        for future in done:
                            blocks.append(future.result())
                            progress.update(pending.pop(future))
                finally:
                    for future in pending:
                        future.cancel()
    check_cancelled()
    points = {}
    for h, key in enumerate(keys):
        check_cancelled()
        indices = np.concatenate([block[h] for block in blocks]) if blocks else np.empty((0, 2), dtype=int)
        indices = indices[np.lexsort((indices[:, 1], indices[:, 0]))]
        points[key] = list(zip(grids[0][indices[:, 0]], grids[1][indices[:, 1]]))
    print(f'Energy/rotation mapping complete: {sum(map(len, points.values())):,} accepted reflection points.', flush=True)
    return points
