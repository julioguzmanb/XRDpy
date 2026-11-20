"""Non-destructive detector acceptance for stored Bragg-condition mappings."""
import json
from dataclasses import asdict

import numpy as np
from scipy.spatial.transform import Rotation

from . import sample, utils
from .geometry import MotorChain
from .parallel import check_cancelled


def mapping_context(values):
    lattice = sample.LatticeStructure(
        space_group=values['sam_space_group'],
        **{key: values.get('sam_' + key) for key in ('a', 'b', 'c', 'alpha', 'beta', 'gamma')},
        initial_crystal_orientation=values.get('sam_initial_crystal_orientation'),
        rotation_order=values.get('sam_rotation_order', 'xyz'),
    )
    chain = values.get('motor_chain')
    if values.get('geometry') is not None:
        chain = values['geometry'].sample
    if chain is None:
        lattice.apply_rotation(**{key: values.get('sam_' + key, 0) for key in ('rotx', 'roty', 'rotz')})
    motors = None
    if chain is not None:
        motors = [{key: value for key, value in asdict(motor).items() if key != 'metadata'} for motor in chain.motors]
        for motor in motors:
            motor['axis'] = motor['axis'].tolist()
            motor['origin'] = motor['origin'].tolist()
    return dict(initial_orientation=lattice.crystal_orientation.tolist(),
                rotation_order=values.get('sam_rotation_order', 'xyz'),
                energy=values.get('energy', 10000), motors=motors,
                fixed_angles=values.get('fixed_angles') or {},
                scan_angles=values.get('scan_angles') or {})


def _legacy_context(metadata):
    """Recover the original sample definition, never current GUI sample inputs."""
    from . import diffractometers
    saved = metadata.get('gui_state', {}).get('single', {})
    try:
        values = {'sam_space_group': int(float(saved['space_group'])), 'energy': float(saved['energy'])}
        for key in ('a', 'b', 'c', 'alpha', 'beta', 'gamma', 'sam_rotx', 'sam_roty', 'sam_rotz'):
            values[key if key.startswith('sam_') else 'sam_' + key] = float(saved[key])
        if saved.get('use_custom_orientation'):
            values['sam_initial_crystal_orientation'] = np.asarray(saved['orientation_matrix'], dtype=float)
        mode = saved.get('geometry_mode', 'Legacy Euler')
        if mode == 'Custom geometry':
            values['motor_chain'] = MotorChain.from_dict(json.loads(saved['custom_sample_chain'] or '[]'))
        elif mode == 'Predefined diffractometer':
            values['geometry'] = diffractometers.make_diffractometer(
                saved['geometry_kind'], **json.loads(saved['geometry_kwargs'] or '{}'))
        values['fixed_angles'] = json.loads(saved.get('geometry_sample_angles') or '{}')
        return mapping_context(values)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError('This saved mapping lacks the sample/energy metadata needed for detector filtering. '
                         'It can still be plotted without the filter.') from exc


def mapping_rays(result):
    """Calculate outgoing directions only for accepted Bragg points, in batches."""
    cached = getattr(result, '_mapping_ray_cache', None)
    if cached is not None:
        return cached
    context = result.metadata.get('mapping_context') or _legacy_context(result.metadata)
    axes = result.metadata.get('axes')
    if axes is None:
        saved = result.metadata.get('gui_state', {}).get('single', {})
        names = ('param1', 'param2') if context['motors'] is None else ('geometry_motor1_name', 'geometry_motor2_name')
        axes = [saved.get(key) for key in names]
    if len(axes) != 2 or any(name is None for name in axes):
        raise ValueError('Saved mapping scan axes are unavailable.')
    reciprocal = sample.cal_reciprocal_lattice(np.asarray(context['initial_orientation'], dtype=float))
    chain = MotorChain.from_list(context['motors']) if context['motors'] is not None else None
    rays = {}
    for key, values in result.data['points'].items():
        points = np.asarray(values, dtype=float).reshape(-1, 2)
        hkl = np.fromstring(key.strip('[]()').replace(',', ' '), sep=' ')
        if hkl.shape != (3,):
            raise ValueError('Cannot decode Miller indices in saved mapping: ' + key)
        q0 = sample.calculate_q_hkl(hkl.reshape(1, 3), reciprocal)[0]
        outgoing = np.empty((len(points), 3))
        for start in range(0, len(points), 65536):
            check_cancelled()
            rows = points[start:start + 65536]
            energy = np.full(len(rows), float(context['energy']))
            if chain is None:
                angles = np.zeros((len(rows), 3))
                order = context['rotation_order']
                for name, value in context.get('scan_angles', {}).items():
                    angles[:, order.index(name[-1])] = value
                for column, name in enumerate(axes):
                    if name == 'energy':
                        energy = rows[:, column]
                    else:
                        angles[:, order.index(name[-1])] = rows[:, column]
                q = Rotation.from_euler(order, angles, degrees=True).apply(q0)
            else:
                q = np.empty((len(rows), 3))
                for index, row in enumerate(rows):
                    check_cancelled()
                    angles = {**context['fixed_angles'], **dict(zip(axes, row))}
                    q[index] = q0 @ chain.as_transform(angles=angles)[:3, :3].T
            q[:, 0] += 2 * np.pi / (utils.energy_to_wavelength(energy) * 1e10)
            outgoing[start:start + len(rows)] = q
        rays[key] = outgoing
    result._mapping_ray_cache = rays
    return rays


def filtered_mapping(result, detector_obj):
    """Return a plotting copy, preserving every source point and all metadata."""
    from .experiment import detector_hit_mask
    from .results import SimulationResult
    if detector_obj is None:
        raise ValueError('detector_obj is required for the detector-hit display filter.')
    points = {}
    for key, rays in mapping_rays(result).items():
        mask = np.zeros(len(rays), dtype=bool)
        for start in range(0, len(rays), 65536):
            check_cancelled()
            # Reuse the existing forward-ray projection with an arbitrary unit k_i.
            mask[start:start + 65536] = detector_hit_mask(rays[start:start + 65536] - [1, 0, 0],
                                                         2 * np.pi * 1e-10, detector_obj)
        points[key] = np.asarray(result.data['points'][key], dtype=float).reshape(-1, 2)[mask]
    return SimulationResult(result.kind, {**result.data, 'points': points},
                            {**result.metadata, 'display_only': True})
