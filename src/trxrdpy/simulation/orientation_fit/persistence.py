"""Versioned, atomic project and orientation-fit result files."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .observations import OrientationExperiment
from .refine import FittedCandidate, OrientationFitResult


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write(path, payload, *, overwrite=False):
    path = Path(path).expanduser()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.orientation-',
                                         suffix='.tmp', encoding='utf-8', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write('\n')
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_experiment(path, experiment: OrientationExperiment, *, overwrite=False):
    """Save an editable project, including one that has no peaks yet."""
    return _write(path, {'format': 'trxrdpy-orientation-project', 'version': 1,
                         'experiment': _jsonable(experiment.to_dict())}, overwrite=overwrite)


def load_experiment(path) -> OrientationExperiment:
    """Load an editable pose-and-peak project."""
    with open(path, encoding='utf-8') as stream:
        payload = json.load(stream)
    if payload.get('format') != 'trxrdpy-orientation-project' or payload.get('version') != 1:
        raise ValueError('Not a supported orientation project file.')
    return OrientationExperiment.from_dict(payload['experiment'])


def save_fit_result(path, result: OrientationFitResult, *, overwrite=False):
    """Save the fitted candidates with the exact project snapshot used."""
    return _write(path, {'format': 'trxrdpy-orientation-fit', 'version': 1,
                         'result': _jsonable(asdict(result))}, overwrite=overwrite)


def load_fit_result(path) -> OrientationFitResult:
    """Load numerical candidates and their source observations."""
    with open(path, encoding='utf-8') as stream:
        payload = json.load(stream)
    if payload.get('format') != 'trxrdpy-orientation-fit' or payload.get('version') != 1:
        raise ValueError('Not a supported orientation fit result file.')
    data = payload['result']
    candidates = []
    for item in data['candidates']:
        candidates.append(FittedCandidate(
            rotation=np.asarray(item['rotation'], dtype=float),
            reference_matrix=np.asarray(item['reference_matrix'], dtype=float),
            assignments={key: tuple(value) for key, value in item['assignments'].items()},
            weighted_error=float(item['weighted_error']), rms_pixel=float(item['rms_pixel']),
            rank=int(item['rank']), angular_sigma_deg=item['angular_sigma_deg'],
            peak_diagnostics=item['peak_diagnostics'],
            pose_matrices={key: np.asarray(value, dtype=float) for key, value in item['pose_matrices'].items()},
        ))
    return OrientationFitResult(candidates, data['project'], data['message'])
