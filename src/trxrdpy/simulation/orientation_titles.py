"""Describe the controls and conventions behind plotted orientations."""
import warnings

from scipy.spatial.transform import Rotation

from . import utils


def _angle_values(values, digits):
    if isinstance(values, dict):
        values = [values['rotx'], values['roty'], values['rotz']]
    return ', '.join(
        f'{name}={float(value):.{digits}f}'
        for name, value in zip(('rotx', 'roty', 'rotz'), values))


def equivalent_xyz(rotation, digits):
    """Return fixed-laboratory-axis xyz angles for an active rotation."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        angles = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)
    return _angle_values(angles, digits)


def sample_title(pose, initial, current, digits=2):
    """Describe a sample pose without replacing entered angles by a conversion."""
    if pose is not None and pose[0] == 'euler':
        _, order, angles = pose
        return (f'Sample rotations [°, order {order}, relative to initial]:\n'
                + _angle_values(angles, digits))

    relative = utils.relative_orientation(initial, current)
    if pose is not None and pose[0] == 'motors':
        _, source, angles = pose
        values = ', '.join(f'{name}={float(value):.{digits}f}'
                           for name, value in angles.items()) or 'none'
        return (f'Sample convention: {source} (motor chain order)\n'
                f'Sample motors [°]: {values}\n'
                'Equivalent sample Euler [°, fixed lab xyz, relative to initial]:\n'
                + equivalent_xyz(relative, digits))

    source = pose[1] if pose is not None and pose[0] == 'transform' else 'orientation matrix'
    return (f'Sample convention: {source}\n'
            'Equivalent sample Euler [°, fixed lab xyz, relative to initial]:\n'
            + equivalent_xyz(relative, digits))


def detector_title(pose, detector, digits=2):
    """Describe the active detector pose, including custom motor transforms."""
    if pose is not None and pose[0] == 'euler':
        _, order, angles = pose
        return (f'Detector rotations [°, order {order}]: '
                + _angle_values(angles, digits))

    if pose is not None and pose[0] == 'motors':
        _, source, angles = pose
        values = ', '.join(f'{name}={float(value):.{digits}f}'
                           for name, value in angles.items()) or 'none'
        return (f'Detector convention: {source} (motor chain order)\n'
                f'Detector motors [°]: {values}\n'
                'Equivalent detector Euler [°, fixed lab xyz]:\n'
                + equivalent_xyz(detector.get_transform()[:3, :3], digits))

    if ((pose is not None and pose[0] == 'transform') or
            detector.custom_transform is not None):
        source = pose[1] if pose is not None and pose[0] == 'transform' else 'custom transform'
        return (f'Detector convention: {source}\n'
                'Equivalent detector Euler [°, fixed lab xyz]:\n'
                + equivalent_xyz(detector.get_transform()[:3, :3], digits))

    return (f'Detector rotations [°, order {detector.rotation_order}]: '
            + _angle_values([detector.rotx, detector.roty, detector.rotz], digits))
