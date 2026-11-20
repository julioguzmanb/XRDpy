"""Compact reciprocal scans: one lattice, sampled axes, and reusable geometry."""
from copy import deepcopy
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
import numpy as np

from . import utils
from .geometry import MotorChain
from .parallel import check_cancelled


def chain_json(chain):
    if chain is None:
        return None
    motors = []
    for motor in chain.motors:
        values = asdict(motor)
        values.pop('metadata', None)
        values['axis'] = values['axis'].tolist()
        values['origin'] = values['origin'].tolist()
        motors.append(values)
    return json.dumps(motors)


def build(axes, lattice_options, detector_obj, detector_angles):
    from .single_crystal import _build_single_crystal_lattice
    geometry = lattice_options.get('geometry')
    options = dict(lattice_options)
    options.update(geometry=None, sample_angles=None, sam_rotx=0, sam_roty=0, sam_rotz=0)
    check_cancelled()
    print('Preparing reciprocal lattice (once, not per configuration)...', flush=True)
    lattice = _build_single_crystal_lattice(**options)
    norms = np.linalg.norm(lattice.q_hkls, axis=1)
    keep = (norms > 0) & (norms <= options['qmax'])
    detector_state = None
    if detector_obj is not None:
        detector_state = {name: getattr(detector_obj, name) for name in
            ('pxsize_h', 'pxsize_v', 'num_pixels_h', 'num_pixels_v', 'dist', 'poni1', 'poni2',
             'rotx', 'roty', 'rotz', 'rotation_order')}
        detector_state['transform'] = detector_obj.get_transform()
    count = math.prod(len(values) for values in axes.values())
    print(f'Reciprocal scan: {count:,} selectable configurations; storing one lattice of {keep.sum():,} reflections.')
    return dict(format='compact-v1', axes=axes, Q0=lattice.q_hkls[keep], hkls=lattice.allowed_hkls[keep],
                energy=options['energy'], bandwidth=options['e_bandwidth'], detector_state=detector_state,
                sample_euler=np.array([lattice_options.get('sam_rot'+axis, 0) for axis in 'xyz']),
                sample_order=options.get('sam_rotation_order', 'xyz'),
                sample_chain=chain_json(geometry.sample) if geometry else None,
                detector_chain=chain_json(geometry.detector) if geometry else None,
                sample_angles=dict(lattice_options.get('sample_angles') or {}),
                detector_angles=dict(detector_angles or {}))


def transform_points(points, rotation, parallel=False, workers=None):
    workers = 4 if workers is None else workers
    if int(workers) != workers or not 1 <= workers <= 1024:
        raise ValueError('Workers must be an integer between 1 and 1024.')
    check_cancelled()
    if not parallel or len(points) < 4096 or workers == 1:
        return points @ rotation.T
    count = min(int(workers), max(1, int(np.ceil(len(points)/4096))))
    # NumPy releases the GIL; shared-memory threads avoid process copies of the lattice.
    with ThreadPoolExecutor(max_workers=count) as pool:
        result = list(pool.map(lambda block: block @ rotation.T, np.array_split(points, count)))
    check_cancelled()
    return np.concatenate(result)


def configuration(scan, index, detector_obj=None, parallel=False, workers=None):
    from .detector import Detector
    sizes = tuple(len(values) for values in scan['axes'].values())
    if isinstance(index, bool) or int(index) != index or not 0 <= index < math.prod(sizes):
        raise ValueError('Configuration index is outside the saved scan.')
    indices = []
    remainder = int(index)
    for size in reversed(sizes):
        remainder, offset = divmod(remainder, size)
        indices.insert(0, offset)
    config = {name: values[i] for (name, values), i in zip(scan['axes'].items(), indices)}
    if scan['sample_chain'] is not None:
        angles = {**scan['sample_angles'], **{name.split(':', 1)[1]: value for name, value in config.items() if name.startswith('sample:')}}
        rotation = MotorChain.from_list(json.loads(scan['sample_chain'])).as_transform(angles)[:3, :3]
    else:
        angles = [config.get('sam_rot'+axis, value) for axis, value in zip('xyz', scan['sample_euler'])]
        rotation = utils.apply_rotation(np.eye(3), *angles, scan['sample_order']).T
    q = transform_points(scan['Q0'], rotation, parallel, workers)
    saved = scan['detector_state']
    det = deepcopy(detector_obj)
    if det is None and saved is not None:
        det = Detector(detector_type='manual', **{k: v for k, v in saved.items() if k != 'transform'})
        det.set_transform(saved['transform'])
    if det is not None and saved is not None:
        transform = saved['transform']
        if any(name.startswith(('det_', 'detector:')) for name in config):
            if scan['detector_chain'] is not None:
                angles = {**scan['detector_angles'], **{name.split(':', 1)[1]: value for name, value in config.items() if name.startswith('detector:')}}
                transform = MotorChain.from_list(json.loads(scan['detector_chain'])).as_transform(angles)
            else:
                angles = [config.get('det_rot'+axis, saved['rot'+axis]) for axis in 'xyz']
                transform = utils.make_transform(rotation_matrix=utils.apply_rotation(np.eye(3), *angles, saved['rotation_order']).T)
        det.set_transform(det.get_transform() @ np.linalg.solve(saved['transform'], transform))
    energy = config.get('energy', scan['energy'])
    bandwidth = config.get('e_bandwidth', scan['bandwidth'])
    wavelength = utils.energy_to_wavelength(energy)
    from scipy.spatial.transform import Rotation
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        sample_angles = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)
        detector_angles = Rotation.from_matrix(det.get_transform()[:3,:3]).as_euler('xyz', degrees=True) if det is not None else None
    title = ('Reciprocal space and Ewald construction\n'
             f'Energy: {energy:g} eV; wavelength: {wavelength*1e10:.4f} \u00c5; full bandwidth: {bandwidth:g}%\n'
             'Sample [deg, equivalent xyz]: ' + ', '.join(f'{value:.2f}' for value in sample_angles))
    if detector_angles is not None:
        title += '\nDetector [deg, equivalent xyz]: ' + ', '.join(f'{value:.2f}' for value in detector_angles)
    from .reciprocal_scan import parameter_label
    labels = [f'{parameter_label(name)} = {value:g}' for name, value in config.items()]
    title += '\n' + '\n'.join('; '.join(labels[i:i+4]) for i in range(0, len(labels), 4))
    return dict(q=q, hkls=scan['hkls'], wavelength=wavelength,
                bandwidth=bandwidth, title=title, detector_obj=det)


def ranges(scan):
    from .reciprocal import _ranges
    energy = max(scan['axes'].get('energy', [scan['energy']]))
    bandwidth = max(scan['axes'].get('e_bandwidth', [scan['bandwidth']]))
    radius = 2*np.pi / (utils.energy_to_wavelength(energy)*1e10) * (1+bandwidth/200)
    qmax = np.linalg.norm(scan['Q0'], axis=1).max(initial=0)
    return _ranges(np.array([[-qmax]*3, [qmax]*3]), radius)
