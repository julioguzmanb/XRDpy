"""Portable numeric simulation results for replotting without recomputation."""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np


@dataclass
class SimulationResult:
    kind: str
    data: dict
    metadata: dict


def targeting_result(result, metadata):
    metadata = {**metadata, 'title': getattr(result, 'plot_title', 'Fixed-energy targeting')}
    data = {name: getattr(result, name) for name in (
        'target_hkl', 'target_pixel', 'pixel_tolerance_px', 'q0', 'q0_norm', 'wavelength_m',
        'eta_deg', 'q_cone', 'all_pixels', 'all_ray_points', 'on_detector_mask',
        'accepted_mask', 'pixel_error_px', 'best_idx', 'beam_center_pixel_inferred')}
    data['solutions'] = asdict(result.solutions) if result.solutions is not None else None
    return SimulationResult('targeting', data, metadata)


def _write(group, data):
    for index, (key, value) in enumerate(data.items()):
        node = group.create_group(str(index))
        node.attrs['key'] = key
        if isinstance(value, dict):
            node.attrs['kind'] = 'dict'
            _write(node, value)
        elif value is None or isinstance(value, str) or (isinstance(value, tuple) and all(isinstance(x, str) for x in value)):
            node.attrs['kind'] = 'json'
            node.attrs['value'] = json.dumps(value)
        else:
            node.attrs['kind'] = 'array'
            array = np.asarray(value)
            node.create_dataset('value', data=array, compression='gzip' if array.ndim and array.size else None)


def _read(group):
    result = {}
    for name in sorted(group, key=int):
        node = group[name]
        kind = node.attrs['kind']
        if kind == 'dict':
            value = _read(node)
        elif kind == 'json':
            value = json.loads(node.attrs['value'])
        elif kind == 'array':
            value = node['value'][()]
        else:
            raise ValueError('Unsupported result data type.')
        result[node.attrs['key']] = value
    return result


def save_result(path, result, *, overwrite=False):
    if result.metadata.get('display_only'):
        raise ValueError('Save the original complete mapping, not its detector-filtered plotting view.')
    path = Path(path).expanduser()
    if not path.suffix:
        path = path.with_suffix('.h5')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.simulation-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
        with h5py.File(temporary, 'w') as handle:
            handle.attrs['format'] = 'trxrdpy-simulation-result'
            handle.attrs['version'] = 1
            handle.attrs['kind'] = result.kind
            handle.attrs['metadata'] = json.dumps(result.metadata)
            _write(handle.create_group('data'), result.data)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_result(path):
    with h5py.File(path, 'r') as handle:
        if handle.attrs.get('format') != 'trxrdpy-simulation-result' or handle.attrs.get('version') != 1:
            raise ValueError('Not a supported XRDpy simulation result file.')
        kind = handle.attrs['kind']
        if kind not in ('targeting', 'parameter_mapping', 'detector_mapping', 'reciprocal'):
            raise ValueError('Unsupported simulation result kind.')
        return SimulationResult(kind, _read(handle['data']), json.loads(handle.attrs['metadata']))


def plot_result(result, *, detector_plot=True, plot_2d=True, plot_3d=False, display_mode='points',
                require_detector_hit=False, detector_obj=None, reciprocal_options=None, mapping_axes=None,
                mapping_backend='matplotlib', mapping_show=True, mapping_parallel=False, mapping_workers=None):
    from . import plot
    if result.kind == 'parameter_mapping' and mapping_backend == 'plotly':
        from .mapping_plotly import plot_mapping
        return plot_mapping(result, display_mode, require_detector_hit, detector_obj,
                            show=mapping_show, parallel=mapping_parallel, workers=mapping_workers)
    if mapping_backend not in ('matplotlib', 'plotly'):
        raise ValueError('Unknown mapping renderer.')
    if result.data.get('format') == 'mapping-slices-v1':
        from .mapping_slices import plot_slices
        return plot_slices(result, display_mode, require_detector_hit, detector_obj)
    if result.kind == 'reciprocal':
        if 'scan' in result.data:
            from .reciprocal_scan import plot_scan
            return plot_scan(**{**result.data, **(reciprocal_options or {})})
        return plot.plot_reciprocal(**{**result.data, **(reciprocal_options or {})})
    if result.kind == 'parameter_mapping' and result.metadata.get('require_detector_hit') and not result.metadata.get('points_are_complete', False):
        import warnings
        warnings.warn('This older mapping was filtered during computation. Removed Bragg points cannot '
                      'be recovered for other detector poses; recompute once to save a complete mapping.',
                      UserWarning, stacklevel=2)
    if require_detector_hit and result.kind == 'parameter_mapping':
        from .mapping_filter import filtered_mapping
        result = filtered_mapping(result, detector_obj)
    data, meta = result.data, result.metadata
    def draw(points, names, steps=None):
        if mapping_axes is not None:
            return plot.plot_parameter_mapping(points, *names, display_mode=display_mode,
                                               grid_steps=steps, existing_axes=mapping_axes, show=False)
        if display_mode == 'points':
            return plot.plot_parameter_mapping(points, *names)
        return plot.plot_parameter_mapping(points, *names, display_mode=display_mode, grid_steps=steps)
    if result.kind in ('parameter_mapping', 'detector_mapping'):
        points = data['points']
        if result.kind == 'detector_mapping' and meta.get('motor_names'):
            names = meta['motor_names']
            motor_steps = meta.get('motor_steps', {})
            if not motor_steps:
                import ast
                try:
                    ranges = ast.literal_eval(meta['gui_state']['single']['geometry_detector_scan_ranges'])
                    motor_steps = {name: abs(value[2]) for name, value in ranges.items()}
                except (KeyError, ValueError, SyntaxError, TypeError, AttributeError):
                    pass
            from itertools import combinations
            if len(names) == 1:
                values = {key: list(zip(np.arange(len(rows[names[0]])), rows[names[0]]))
                          for key, rows in points.items()}
                return draw(values, ['configuration', names[0]])
            for first, second in combinations(names, 2):
                values = {key: list(zip(rows[first], rows[second])) for key, rows in points.items()}
                steps = motor_steps
                draw(values, [first, second], [steps[first], steps[second]] if first in steps and second in steps else None)
            return
        else:
            names = meta.get('axes', ['roty', 'rotz'])
        steps = meta.get('grid_steps')
        if steps is None:
            # Older HDF5 exports stored the scan definition in the GUI snapshot.
            import ast
            saved = meta.get('gui_state', {}).get('single', {})
            keys = ('param1_range', 'param2_range') if result.kind == 'parameter_mapping' else ('angle_range', 'angle_range')
            if names == [saved.get('geometry_motor1_name'), saved.get('geometry_motor2_name')]:
                keys = ('geometry_motor1_range', 'geometry_motor2_range')
            try:
                steps = [abs(float(ast.literal_eval(saved[key])[2])) for key in keys]
            except (KeyError, ValueError, TypeError, SyntaxError):
                pass
        return draw(points, names, steps)
    if result.kind != 'targeting':
        raise ValueError('Unsupported simulation result kind.')
    title = result.metadata.get('title', 'Fixed-energy targeting')
    if detector_plot:
        plot.plot_fixed_energy_detector_hits(data['all_pixels'], data['on_detector_mask'],
            data['accepted_mask'], data['target_pixel'], data['pixel_tolerance_px'], title=title)
    if data['solutions'] is not None:
        solutions = SimpleNamespace(**data['solutions'])
        if plot_2d:
            plot.plot_fixed_energy_motor_projections(solutions, title=title)
        if plot_3d:
            plot.plot_fixed_energy_motor_family_3d(solutions, title=title)
