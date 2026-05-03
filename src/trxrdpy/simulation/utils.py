import numpy as np
from scipy.spatial.transform import Rotation as R

from dataclasses import dataclass, field
from typing import List, Optional, Union



def energy_to_wavelength(energy):
    """
    Convert energy (in electron volts) to wavelength (in meters).

    Parameters:
        energy (float): The energy in electron volts (eV).

    Returns:
        float: The corresponding wavelength in meters (m).
    """
    h = 4.135667696e-15  # Planck's constant in eV·s
    c = 299_792_458      # Speed of light in m/s
    return h * c / energy


def wavelength_to_energy(wavelength):
    """
    Convert wavelength (in meters) to energy (in electron volts).

    Parameters:
        wavelength (float): The wavelength in meters (m).

    Returns:
        float: The corresponding energy in electron volts (eV).
    """
    h = 4.135667696e-15  # Planck's constant in eV·s
    c = 299_792_458      # Speed of light in m/s
    return h * c / wavelength


def apply_rotation(initial_matrix, rotation1, rotation2, rotation3, rotation_order="xyz"):
    """
    Apply rotations about the x, y, and z axes (in degrees) to a set of 3D coordinates
    using simple matrix multiplication.

    Parameters:
        initial_matrix (numpy.ndarray): Array of shape (..., 3) representing 3D coordinates.
        rotation1 (float): Rotation angle in degrees around the x-axis.
        rotation2 (float): Rotation angle in degrees around the y-axis.
        rotation3 (float): Rotation angle in degrees around the z-axis.
        rotation_order (str): Order in which to apply rotations (e.g. "xyz").

    Returns:
        numpy.ndarray: Rotated positions as an array of the same shape as the input.
    """
    # Define rotation matrices about each axis (angles in degrees converted to radians)
    def rot_x(theta):
        rad = np.deg2rad(theta)
        return np.array([
            [1, 0, 0],
            [0, np.cos(rad), -np.sin(rad)],
            [0, np.sin(rad),  np.cos(rad)]
        ])

    def rot_y(theta):
        rad = np.deg2rad(theta)
        return np.array([
            [ np.cos(rad), 0, np.sin(rad)],
            [ 0, 1, 0],
            [-np.sin(rad), 0, np.cos(rad)]
        ])

    def rot_z(theta):
        rad = np.deg2rad(theta)
        return np.array([
            [np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad),  np.cos(rad), 0],
            [0, 0, 1]
        ])

    # Map each axis to its rotation angle and corresponding rotation matrix function.
    axis_to_angle = {'x': rotation1, 'y': rotation2, 'z': rotation3}
    axis_to_func  = {'x': rot_x,     'y': rot_y,     'z': rot_z}

    # Start with the identity matrix as the overall rotation matrix.
    overall_R = np.eye(3)
    # Apply rotations in the specified order.
    # Pre-multiplication ensures the first rotation in the string is applied first.
    for axis in rotation_order:
        overall_R = axis_to_func[axis](axis_to_angle[axis]) @ overall_R

    # Since the positions are represented as row vectors,
    # we multiply them by the transpose of the overall rotation matrix.
    rotated = np.dot(initial_matrix, overall_R.T)
    return rotated


def get_indices(main_array, sub_array):
    """
    Get the indices of elements in `main_array` that match elements in `sub_array`.

    Parameters:
        main_array (numpy.ndarray): Main array of shape (N, M).
        sub_array (numpy.ndarray): Sub-array of shape (K, M) to find in the main array.

    Returns:
        numpy.ndarray: Indices of matching elements in `main_array`.
    """
    matches = (main_array[:, None, :] == sub_array[None, :, :]).all(axis=2)
    return np.nonzero(matches.any(axis=1))[0]


def q_to_two_theta(q, wavelength):
    """
    Convert a scattering vector to a two-theta angle.

    Parameters:
        q (float): Scattering vector magnitude in reciprocal angstroms (Å⁻¹).
        wavelength (float): Wavelength of the incident radiation in angstroms (Å).

    Returns:
        float: The corresponding two-theta angle in degrees.
    """
    return np.degrees(2 * np.arcsin(q * wavelength / (4 * np.pi)))


def d_to_two_theta(d, wavelength):
    """
    Convert a d-spacing to a two-theta angle.

    Parameters:
        d (float): Lattice plane spacing in angstroms (Å).
        wavelength (float): Wavelength of the incident radiation in angstroms (Å).

    Returns:
        float: The corresponding two-theta angle in degrees.
    """
    return np.degrees(2 * np.arcsin(wavelength / (2 * d)))





def _as_xyz_array(array, name="array"):
    """Validate an array with last dimension 3."""
    arr = np.asarray(array, dtype=float)
    if arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3). Got {arr.shape}.")
    return arr


def _as_vector3(vector, name="vector"):
    """Validate and return a 3-component vector."""
    vec = np.asarray(vector, dtype=float)
    if vec.shape != (3,):
        raise ValueError(f"{name} must have shape (3,). Got {vec.shape}.")
    return vec


def normalize_vector(vector):
    """
    Return a normalized copy of a 3-vector.

    Parameters:
        vector (array-like): Input vector of shape (3,).

    Returns:
        numpy.ndarray: Unit vector.
    """
    vec = _as_vector3(vector, name="vector")
    norm = np.linalg.norm(vec)
    if np.isclose(norm, 0.0):
        raise ValueError("Cannot normalize a zero vector.")
    return vec / norm


def skew_symmetric(vector):
    """
    Build the skew-symmetric matrix [v]_x associated with a 3-vector.

    Parameters:
        vector (array-like): Input vector of shape (3,).

    Returns:
        numpy.ndarray: Skew-symmetric matrix of shape (3, 3).
    """
    vx, vy, vz = _as_vector3(vector, name="vector")
    return np.array([
        [0.0, -vz,  vy],
        [vz,  0.0, -vx],
        [-vy, vx,  0.0],
    ])


def is_rotation_matrix(matrix, atol=1e-8):
    """
    Check whether a matrix is a proper 3D rotation matrix.

    Parameters:
        matrix (array-like): Candidate matrix of shape (3, 3).
        atol (float): Absolute tolerance.

    Returns:
        bool: True if orthogonal with determinant +1.
    """
    mat = np.asarray(matrix, dtype=float)
    if mat.shape != (3, 3):
        return False
    should_be_identity = mat.T @ mat
    return np.allclose(should_be_identity, np.eye(3), atol=atol) and np.isclose(np.linalg.det(mat), 1.0, atol=atol)


def apply_rotation_matrix(initial_matrix, rotation_matrix):
    """
    Apply a 3x3 rotation matrix to row-vector coordinates.

    Parameters:
        initial_matrix (numpy.ndarray): Array of shape (..., 3) representing 3D coordinates.
        rotation_matrix (array-like): Rotation matrix of shape (3, 3).

    Returns:
        numpy.ndarray: Rotated positions with the same shape as the input.
    """
    points = _as_xyz_array(initial_matrix, name="initial_matrix")
    matrix = np.asarray(rotation_matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation_matrix must have shape (3, 3). Got {matrix.shape}.")
    return np.dot(points, matrix.T)


def axis_angle_to_matrix(axis, angle, degrees=True):
    """
    Build a 3x3 rotation matrix from an arbitrary axis and angle using Rodrigues' formula.

    Parameters:
        axis (array-like): Rotation axis of shape (3,).
        angle (float): Rotation angle.
        degrees (bool): If True, interpret angle in degrees.

    Returns:
        numpy.ndarray: Rotation matrix of shape (3, 3).
    """
    unit_axis = normalize_vector(axis)
    theta = np.deg2rad(angle) if degrees else angle
    K = skew_symmetric(unit_axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def axis_angle_to_scipy(axis, angle, degrees=True):
    """
    Build a scipy Rotation object from an arbitrary axis and angle.

    Parameters:
        axis (array-like): Rotation axis of shape (3,).
        angle (float): Rotation angle.
        degrees (bool): If True, interpret angle in degrees.

    Returns:
        scipy.spatial.transform.Rotation: Rotation object.
    """
    unit_axis = normalize_vector(axis)
    theta = np.deg2rad(angle) if degrees else angle
    return R.from_rotvec(unit_axis * theta)


def compose_rotation_matrices(*rotation_matrices):
    """
    Compose rotation matrices with the same convention as apply_rotation:
    the first matrix passed is applied first.

    Parameters:
        *rotation_matrices: Sequence of 3x3 rotation matrices.

    Returns:
        numpy.ndarray: Composed rotation matrix of shape (3, 3).
    """
    overall = np.eye(3)
    for i, matrix in enumerate(rotation_matrices):
        mat = np.asarray(matrix, dtype=float)
        if mat.shape != (3, 3):
            raise ValueError(f"rotation_matrices[{i}] must have shape (3, 3). Got {mat.shape}.")
        overall = mat @ overall
    return overall


def invert_rotation_matrix(rotation_matrix):
    """
    Invert a proper rotation matrix.

    Parameters:
        rotation_matrix (array-like): Rotation matrix of shape (3, 3).

    Returns:
        numpy.ndarray: Inverse rotation matrix.
    """
    matrix = np.asarray(rotation_matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation_matrix must have shape (3, 3). Got {matrix.shape}.")
    return matrix.T


def euler_to_matrix(rotation1, rotation2, rotation3, rotation_order="xyz"):
    """
    Build a 3x3 rotation matrix that matches the current apply_rotation convention.

    Parameters:
        rotation1 (float): Angle associated with x.
        rotation2 (float): Angle associated with y.
        rotation3 (float): Angle associated with z.
        rotation_order (str): Order in which rotations are applied.

    Returns:
        numpy.ndarray: Rotation matrix of shape (3, 3).
    """
    axis_to_angle = {"x": rotation1, "y": rotation2, "z": rotation3}
    axis_to_vector = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }

    matrices = []
    for axis in rotation_order:
        if axis not in axis_to_angle:
            raise ValueError(f"Unsupported axis '{axis}' in rotation_order='{rotation_order}'.")
        matrices.append(axis_angle_to_matrix(axis_to_vector[axis], axis_to_angle[axis], degrees=True))

    return compose_rotation_matrices(*matrices)


def matrix_to_euler(rotation_matrix, rotation_order="xyz", degrees=True):
    """
    Convert a 3x3 rotation matrix to Euler angles using scipy.

    Parameters:
        rotation_matrix (array-like): Rotation matrix of shape (3, 3).
        rotation_order (str): Euler order string.
        degrees (bool): If True, return angles in degrees.

    Returns:
        numpy.ndarray: Euler angles in the requested order.
    """
    matrix = np.asarray(rotation_matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation_matrix must have shape (3, 3). Got {matrix.shape}.")
    return R.from_matrix(matrix).as_euler(rotation_order, degrees=degrees)


def rotate_about_axis(initial_matrix, axis, angle, origin=None, degrees=True):
    """
    Rotate points about an arbitrary axis, optionally passing through a given origin.

    Parameters:
        initial_matrix (numpy.ndarray): Array of shape (..., 3) representing 3D coordinates.
        axis (array-like): Rotation axis of shape (3,).
        angle (float): Rotation angle.
        origin (array-like or None): A point of shape (3,) through which the axis passes.
        degrees (bool): If True, interpret angle in degrees.

    Returns:
        numpy.ndarray: Rotated positions with the same shape as the input.
    """
    points = _as_xyz_array(initial_matrix, name="initial_matrix")
    origin_vec = np.zeros(3) if origin is None else _as_vector3(origin, name="origin")
    rotation_matrix = axis_angle_to_matrix(axis, angle, degrees=degrees)
    shifted = points - origin_vec
    rotated = apply_rotation_matrix(shifted, rotation_matrix)
    return rotated + origin_vec


def make_transform(rotation_matrix=None, translation=None):
    """
    Build a 4x4 homogeneous transform.

    Parameters:
        rotation_matrix (array-like or None): Rotation matrix of shape (3, 3).
        translation (array-like or None): Translation vector of shape (3,).

    Returns:
        numpy.ndarray: Homogeneous transform of shape (4, 4).
    """
    transform = np.eye(4)

    if rotation_matrix is not None:
        matrix = np.asarray(rotation_matrix, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError(f"rotation_matrix must have shape (3, 3). Got {matrix.shape}.")
        transform[:3, :3] = matrix

    if translation is not None:
        transform[:3, 3] = _as_vector3(translation, name="translation")

    return transform


def axis_angle_to_transform(axis, angle, origin=None, degrees=True):
    """
    Build a 4x4 homogeneous transform for a rotation about an arbitrary axis
    passing through a given origin.

    Parameters:
        axis (array-like): Rotation axis of shape (3,).
        angle (float): Rotation angle.
        origin (array-like or None): A point of shape (3,) through which the axis passes.
        degrees (bool): If True, interpret angle in degrees.

    Returns:
        numpy.ndarray: Homogeneous transform of shape (4, 4).
    """
    rotation_matrix = axis_angle_to_matrix(axis, angle, degrees=degrees)
    origin_vec = np.zeros(3) if origin is None else _as_vector3(origin, name="origin")

    to_origin = make_transform(translation=-origin_vec)
    rotate = make_transform(rotation_matrix=rotation_matrix)
    back = make_transform(translation=origin_vec)

    return back @ rotate @ to_origin


def compose_transforms(*transforms):
    """
    Compose homogeneous transforms so that the first transform passed is applied first.

    Parameters:
        *transforms: Sequence of 4x4 homogeneous transforms.

    Returns:
        numpy.ndarray: Composed homogeneous transform of shape (4, 4).
    """
    overall = np.eye(4)
    for i, transform in enumerate(transforms):
        T = np.asarray(transform, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"transforms[{i}] must have shape (4, 4). Got {T.shape}.")
        overall = T @ overall
    return overall


def apply_transform(initial_matrix, transform):
    """
    Apply a 4x4 homogeneous transform to row-vector coordinates.

    Parameters:
        initial_matrix (numpy.ndarray): Array of shape (..., 3) representing 3D coordinates.
        transform (array-like): Homogeneous transform of shape (4, 4).

    Returns:
        numpy.ndarray: Transformed positions with the same shape as the input.
    """
    points = _as_xyz_array(initial_matrix, name="initial_matrix")
    T = np.asarray(transform, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4). Got {T.shape}.")

    original_shape = points.shape
    flat_points = points.reshape(-1, 3)
    homogeneous_points = np.hstack([flat_points, np.ones((flat_points.shape[0], 1))])
    transformed = homogeneous_points @ T.T
    return transformed[:, :3].reshape(original_shape)


def invert_transform(transform):
    """
    Invert a rigid 4x4 homogeneous transform.

    Parameters:
        transform (array-like): Homogeneous transform of shape (4, 4).

    Returns:
        numpy.ndarray: Inverse homogeneous transform.
    """
    T = np.asarray(transform, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4). Got {T.shape}.")

    rotation = T[:3, :3]
    translation = T[:3, 3]

    inv_rotation = invert_rotation_matrix(rotation)
    inv_translation = -inv_rotation @ translation

    return make_transform(rotation_matrix=inv_rotation, translation=inv_translation)


@dataclass
class AxisRotation:
    """
    Lightweight description of a physical rotation axis.
    """
    axis: np.ndarray
    angle: float
    origin: Optional[np.ndarray] = None
    name: Optional[str] = None
    degrees: bool = True

    def as_matrix(self):
        """Return the 3x3 rotation matrix for this axis rotation."""
        return axis_angle_to_matrix(self.axis, self.angle, degrees=self.degrees)

    def as_transform(self):
        """Return the 4x4 homogeneous transform for this axis rotation."""
        return axis_angle_to_transform(self.axis, self.angle, origin=self.origin, degrees=self.degrees)

    def apply(self, initial_matrix):
        """Apply this rotation to coordinates."""
        return rotate_about_axis(initial_matrix, self.axis, self.angle, origin=self.origin, degrees=self.degrees)

    def inverse(self):
        """Return the inverse axis rotation."""
        return AxisRotation(
            axis=np.asarray(self.axis, dtype=float).copy(),
            angle=-self.angle,
            origin=None if self.origin is None else np.asarray(self.origin, dtype=float).copy(),
            name=self.name,
            degrees=self.degrees,
        )


@dataclass
class RotationChain:
    """
    Ordered collection of axis rotations or raw transforms/matrices.

    Notes:
        - The first element in the chain is applied first.
        - Elements can be AxisRotation, 3x3 rotation matrices, or 4x4 transforms.
    """
    elements: List[Union[AxisRotation, np.ndarray]] = field(default_factory=list)

    def append(self, element):
        """Append an element to the chain."""
        self.elements.append(element)

    def extend(self, elements):
        """Extend the chain with multiple elements."""
        self.elements.extend(elements)

    def as_transform(self):
        """Return the full 4x4 homogeneous transform of the chain."""
        transforms = []
        for i, element in enumerate(self.elements):
            if isinstance(element, AxisRotation):
                transforms.append(element.as_transform())
            else:
                arr = np.asarray(element, dtype=float)
                if arr.shape == (3, 3):
                    transforms.append(make_transform(rotation_matrix=arr))
                elif arr.shape == (4, 4):
                    transforms.append(arr)
                else:
                    raise ValueError(
                        f"elements[{i}] must be AxisRotation, a 3x3 matrix, or a 4x4 transform. Got shape {arr.shape}."
                    )
        return compose_transforms(*transforms)

    def apply(self, initial_matrix):
        """Apply the full chain to coordinates."""
        return apply_transform(initial_matrix, self.as_transform())

    def inverse(self):
        """Return the inverse chain."""
        inverse_elements = []
        for element in reversed(self.elements):
            if isinstance(element, AxisRotation):
                inverse_elements.append(element.inverse())
            else:
                arr = np.asarray(element, dtype=float)
                if arr.shape == (3, 3):
                    inverse_elements.append(invert_rotation_matrix(arr))
                elif arr.shape == (4, 4):
                    inverse_elements.append(invert_transform(arr))
                else:
                    raise ValueError(f"Cannot invert element with shape {arr.shape}.")
        return RotationChain(inverse_elements)

