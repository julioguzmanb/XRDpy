import numpy as np
from scipy.spatial.transform import Rotation as R


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
    Apply a precomputed rotation to a set of positions, ensuring that inputs are
    always interpreted as rotations around the x-, y-, and z-axes respectively,
    in any order specified.

    Parameters:
        initial_matrix (numpy.ndarray): Array of shape (..., 3) representing 3D coordinates.
        rotation1 (float): Rotation angle around the x-axis in degrees.
        rotation2 (float): Rotation angle around the y-axis in degrees.
        rotation3 (float): Rotation angle around the z-axis in degrees.
        rotation_order (str): Order of rotations as a string (default is "xyz").

    Returns:
        numpy.ndarray: Rotated positions as an array of the same shape as the input.
    """
    # Create a mapping of axis to rotation angle
    axis_to_angle = {'x': rotation1, 'y': rotation2, 'z': rotation3}
    
    # Create the angle list in the order specified
    ordered_angles = [axis_to_angle[axis] for axis in rotation_order]
    
    # Create the rotation object with the ordered angles
    rotation_matrix = R.from_euler(rotation_order, ordered_angles, degrees=True)
    
    # Apply the rotation to the initial matrix
    return rotation_matrix.apply(initial_matrix)

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
