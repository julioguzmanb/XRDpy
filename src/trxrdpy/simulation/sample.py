import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar

import xrayutilities.materials as xu

from . import utils
from .plot import plot_crystal
from .plot import plot_rotation_mapping
from .cif import Cif

from itertools import product
from typing import Optional, Dict

from .geometry import MotorChain, DiffractometerGeometry


def _coerce_sample_transform(rotation_object, angles=None):
    """
    Resolve a sample-side transform from one of:
      - MotorChain
      - DiffractometerGeometry
      - utils.AxisRotation
      - utils.RotationChain
      - 3x3 rotation matrix
      - 4x4 homogeneous transform
    """
    if isinstance(rotation_object, DiffractometerGeometry):
        return rotation_object.sample_transform(angles=angles)

    if isinstance(rotation_object, MotorChain):
        return rotation_object.as_transform(angles=angles)

    if isinstance(rotation_object, utils.AxisRotation):
        return rotation_object.as_transform()

    if isinstance(rotation_object, utils.RotationChain):
        return rotation_object.as_transform()

    arr = np.asarray(rotation_object, dtype=float)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 3):
        return utils.make_transform(rotation_matrix=arr)

    raise TypeError(
        "rotation_object must be one of: MotorChain, DiffractometerGeometry, "
        "AxisRotation, RotationChain, 3x3 rotation matrix, or 4x4 transform."
    )


def apply_orientation_transform(orientation, rotation_object, angles=None):
    """
    Apply a rotation-like object to a crystal orientation matrix.

    Notes
    -----
    Only the rotational part is used. Any translation in a 4x4 transform is ignored.
    """
    orientation = np.asarray(orientation, dtype=float)
    if orientation.shape != (3, 3):
        raise ValueError(f"orientation must have shape (3, 3). Got {orientation.shape}.")

    transform = _coerce_sample_transform(rotation_object, angles=angles)
    rotation_matrix = np.asarray(transform, dtype=float)[:3, :3]
    return utils.apply_rotation_matrix(orientation, rotation_matrix)


def _coerce_q_norms(q_hkls):
    """
    Accept either:
      - q magnitudes of shape (N,)
      - q vectors of shape (N, 3)
    and always return magnitudes of shape (N,).
    """
    q_hkls = np.asarray(q_hkls, dtype=float)

    if q_hkls.ndim == 1:
        return q_hkls

    if q_hkls.ndim == 2 and q_hkls.shape[1] == 3:
        return np.linalg.norm(q_hkls, axis=1)

    raise ValueError(
        f"q_hkls must have shape (N,) or (N, 3). Got {q_hkls.shape}."
    )


def _inclusive_angle_values(angle_range):
    """
    Convert (start, stop, step) into an inclusive np.arange-like array.
    """
    if len(angle_range) != 3:
        raise ValueError("angle_range must be a 3-tuple: (start, stop, step).")

    start, stop, step = angle_range
    if step == 0:
        raise ValueError("angle_range step cannot be zero.")

    return np.arange(start, stop + step, step, dtype=float)


class LatticeStructure:
    # Space group lookup logic (integrated from CrystalSystemLookup)
    sgrp_sym = {
        range(1, 3): ('triclinic', 6),
        range(3, 16): ('monoclinic', 4),
        range(16, 75): ('orthorhombic', 3),
        range(75, 143): ('tetragonal', 2),
        range(143, 168): ('trigonal', 2),
        range(168, 195): ('hexagonal', 2),
        range(195, 231): ('cubic', 1)
    }

    sgrp_params = {
        'cubic':       (('a',), ('a', 'a', 'a', 90, 90, 90)),
        'hexagonal':   (('a', 'c'), ('a', 'a', 'c', 90, 90, 120)),
        'trigonal:R':  (('a', 'alpha'), ('a', 'a', 'a', 'alpha', 'alpha', 'alpha')),
        'trigonal:H':  (('a', 'c'), ('a', 'a', 'c', 90, 90, 120)),
        'tetragonal':  (('a', 'c'), ('a', 'a', 'c', 90, 90, 90)),
        'orthorhombic':(('a', 'b', 'c'), ('a', 'b', 'c', 90, 90, 90)),
        'monoclinic':  (('a', 'b', 'c', 'beta'), ('a', 'b', 'c', 90, 'beta', 90)),
        'triclinic':   (('a', 'b', 'c', 'alpha', 'beta', 'gamma'), ('a', 'b', 'c', 'alpha', 'beta', 'gamma'))
    }

    # Pre-flatten the ranges for efficient lookup
    flattened_sgrp_sym = {
        sg_num: (system, nparams)
        for rng, (system, nparams) in sgrp_sym.items()
        for sg_num in rng
    }

    @classmethod
    def get_lattice_params(cls, space_group_number):
        """Get the parameter names and default values for a given space group."""
        if space_group_number not in cls.flattened_sgrp_sym:
            raise ValueError(f"Invalid space group number: {space_group_number}")

        crystal_system, _ = cls.flattened_sgrp_sym[space_group_number]

        if crystal_system in cls.sgrp_params:
            return cls.sgrp_params[crystal_system]
        else:
            variants = [k for k in cls.sgrp_params if k.startswith(crystal_system)]
            if len(variants) == 1:
                return cls.sgrp_params[variants[0]]
            elif len(variants) > 1:
                return cls.sgrp_params[sorted(variants)[0]]
            else:
                raise ValueError(f"No parameter entry found for {crystal_system}")

    @classmethod
    def get_phase(cls, space_group_number):
        """
        Return the crystal phase (e.g., triclinic, cubic) corresponding to the space group.
        """
        if space_group_number not in cls.flattened_sgrp_sym:
            raise ValueError(f"Invalid space group number: {space_group_number}")
        return cls.flattened_sgrp_sym[space_group_number][0]

    @staticmethod
    def calculate_initial_orientation(a, b, c, alpha, beta, gamma):
        """
        Calculate the initial crystal orientation matrix based on lattice parameters.

        Parameters:
            a, b, c (float): Lattice parameters (lengths).
            alpha, beta, gamma (float): Angles between lattice vectors (in degrees).

        Returns:
            numpy.ndarray: The orientation matrix as a 3x3 array.
        """
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)

        v1 = np.array([a, 0, 0])
        v2 = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0])
        v3_x = c * np.cos(beta_rad)
        v3_y = c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
        term = 1 - np.cos(beta_rad)**2 - ((np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad))**2
        term = np.maximum(term, 0)
        v3_z = c * np.sqrt(term)

        v3 = np.array([v3_x, v3_y, v3_z])

        return np.array([v1, v2, v3])

    def __init__(
            self,
            space_group=None,
            a=None, b=None, c=None,
            alpha=None, beta=None, gamma=None,
            initial_crystal_orientation=None,
            rotation_order="xyz",
            atom_positions=None,
            cif_file_path=None
        ):
        """
        Initialize a lattice structure with its parameters and initial crystal orientation.
        """
        self.space_group = space_group
        self.phase = None
        self.atom_positions = atom_positions

        if cif_file_path is not None:
            try:
                cif = Cif(cif_file_path)
                if cif.space_group == 0:
                    if space_group is None:
                        print("Space Group Assumed to be 1")
                        self.space_group = 1
                    else:
                        self.space_group = space_group
                else:
                    self.space_group = cif.space_group

                self.a = a = cif.a
                self.b = b = cif.b
                self.c = c = cif.c
                self.alpha = alpha = cif.alpha
                self.beta = beta = cif.beta
                self.gamma = gamma = cif.gamma
                self.atom_positions = cif.atom_positions
                self.cif_data = cif.data

            except:
                ImportError

        if self.space_group is not None:
            param_names, default_values = self.get_lattice_params(self.space_group)
            self.phase = self.get_phase(self.space_group)
            defaults = dict(zip(param_names, default_values))

            self.a = a if a is not None else defaults.get('a')
            self.b = b if b is not None else defaults.get('b', self.a)
            self.c = c if c is not None else defaults.get('c', self.a)
            self.alpha = alpha if alpha is not None else defaults.get('alpha', 90)
            self.beta = beta if beta is not None else defaults.get('beta', 90)
            self.gamma = gamma if gamma is not None else defaults.get('gamma', 90)

            lattice_args = [getattr(self, name) for name in param_names]
            self.xu_lattice = xu.SGLattice(self.space_group, *lattice_args)
        else:
            self.a = a
            self.b = b
            self.c = c
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.xu_lattice = None

        if initial_crystal_orientation is None:
            self.initial_crystal_orientation = self.calculate_initial_orientation(
                self.a, self.b, self.c, self.alpha, self.beta, self.gamma
            )
        else:
            self.initial_crystal_orientation = initial_crystal_orientation

        self.crystal_orientation = self.initial_crystal_orientation
        self.rotation_order = rotation_order

        self.reciprocal_lattice = None
        self.allowed_hkls = None
        self.wavelength = None
        self.q_hkls = None
        self.d_hkls = None
        self.two_theta_hkls = None
        self.hkls_in_Bragg_condition = None
        self.q_hkls_in_Bragg_condition = None
        self.diffraction_cones = None
        self.kf_hkls = None
        self.rotations_for_Bragg_condition = None

    def apply_rotation(self, rotx=0, roty=0, rotz=0, mode="absolute"):
        """
        Apply rotation to the crystal orientation matrix.
        """
        if mode not in ["relative", "absolute"]:
            raise ValueError("Mode must be 'relative' or 'absolute'.")

        base_orientation = (
            self.initial_crystal_orientation.copy()
            if mode == "absolute"
            else self.crystal_orientation.copy()
        )

        self.crystal_orientation = utils.apply_rotation(
            base_orientation, rotx, roty, rotz, self.rotation_order
        )

        if self.reciprocal_lattice is not None:
            self.calculate_reciprocal_lattice()

        if self.q_hkls is not None:
            self.calculate_q_hkls()

        if self.hkls_in_Bragg_condition is not None:
            self.check_Bragg_condition(self.wavelength, self.e_bandwidth)

    def apply_transform(self, transform, mode="absolute"):
        """
        Apply a generic transform/rotation object to the crystal orientation.

        Accepted inputs
        ---------------
        - 3x3 rotation matrix
        - 4x4 homogeneous transform
        - utils.AxisRotation
        - utils.RotationChain
        - MotorChain
        - DiffractometerGeometry (sample chain is used)
        """
        if mode not in ["relative", "absolute"]:
            raise ValueError("Mode must be 'relative' or 'absolute'.")

        base_orientation = (
            self.initial_crystal_orientation.copy()
            if mode == "absolute"
            else self.crystal_orientation.copy()
        )

        self.crystal_orientation = apply_orientation_transform(base_orientation, transform)

        if self.reciprocal_lattice is not None:
            self.calculate_reciprocal_lattice()

        if self.q_hkls is not None:
            self.calculate_q_hkls()

        if self.hkls_in_Bragg_condition is not None:
            self.check_Bragg_condition(self.wavelength, self.e_bandwidth)

    def apply_motor_chain(self, motor_chain, angles=None, mode="absolute"):
        """
        Apply a MotorChain to the crystal orientation.
        """
        transform = _coerce_sample_transform(motor_chain, angles=angles)
        self.apply_transform(transform, mode=mode)

    def apply_diffractometer(self, geometry, angles=None, mode="absolute"):
        """
        Apply the sample arm of a DiffractometerGeometry to the crystal orientation.
        """
        transform = _coerce_sample_transform(geometry, angles=angles)
        self.apply_transform(transform, mode=mode)

    def calculate_reciprocal_lattice(self):
        """Calculate and store the reciprocal lattice vectors."""
        self.reciprocal_lattice = cal_reciprocal_lattice(self.crystal_orientation)

    def create_possible_reflections(self, qmax):
        """Get allowed reflections for a given maximum Q-value in Å^-1."""
        if not isinstance(qmax, (int, float)) or qmax <= 0:
            raise ValueError("qmax must be a positive number.")
        if self.xu_lattice is None:
            raise RuntimeError("SGLattice is not initialized.")
        reflections = self.xu_lattice.get_allowed_hkl(qmax)

        self.allowed_hkls = np.array(list(reflections))

    def check_if_hkl_allowed(self, hkl):
        print(self.xu_lattice.hkl_allowed(hkl))

    def get_equivalent_reflections(self, hkl):
        hkls_list = []

        for i in self.xu_lattice.equivalent_hkls(hkl):
            hkls_list.append(list(i))

        return hkls_list

    def calculate_q_hkls(self):
        """
        Calculate the Q vectors for the allowed Miller indices and reciprocal lattice.
        """
        if self.reciprocal_lattice is None:
            self.reciprocal_lattice = cal_reciprocal_lattice(self.crystal_orientation)

        if self.allowed_hkls is None:
            raise AttributeError("Compute the allowed HKLs using create_possible_reflections.")

        self.q_hkls = calculate_q_hkl(self.allowed_hkls, self.reciprocal_lattice)

    def calculate_d_hkls(self):
        """
        Calculate the d-spacing values for the allowed Miller indices.
        """
        if self.reciprocal_lattice is None:
            self.reciprocal_lattice = cal_reciprocal_lattice(self.crystal_orientation)

        if self.allowed_hkls is None:
            raise AttributeError("Compute the allowed HKLs using create_possible_reflections.")

        self.d_hkls = calculate_dspacing(self.allowed_hkls, self.reciprocal_lattice)

    def calculate_two_theta_hkls(self, wavelength):
        """
        Calculate the two-theta angles for the allowed Miller indices.
        """
        if self.reciprocal_lattice is None:
            self.reciprocal_lattice = cal_reciprocal_lattice(self.crystal_orientation)

        if self.allowed_hkls is None:
            raise AttributeError("Compute the allowed HKLs using create_possible_reflections.")

        self.two_theta_hkls = calculate_two_theta(
            self.allowed_hkls,
            self.reciprocal_lattice,
            wavelength
        )
        self.wavelength = wavelength

    def check_Bragg_condition(self, wavelength, e_bandwidth):
        """
        Check if the allowed Miller indices satisfy the Bragg condition.
        """
        if self.reciprocal_lattice is None:
            self.reciprocal_lattice = cal_reciprocal_lattice(self.crystal_orientation)

        if self.allowed_hkls is None:
            raise AttributeError("Compute the allowed HKLs using create_possible_reflections.")

        if self.q_hkls is None:
            self.q_hkls = calculate_q_hkl(self.allowed_hkls, self.reciprocal_lattice)

        in_Bragg_condition = check_Bragg_condition(self.q_hkls, wavelength, e_bandwidth)

        self.q_hkls_in_Bragg_condition = self.q_hkls[in_Bragg_condition]
        indices = utils.get_indices(self.q_hkls, self.q_hkls_in_Bragg_condition)
        self.hkls_in_Bragg_condition = self.allowed_hkls[indices]
        self.wavelength = wavelength
        self.e_bandwidth = e_bandwidth

    def plot_crystal(self, xlims=(-14, 14), ylims=(-14, 14), zlims=(-14, 14), axis_labels=None):
        """
        Plot the current crystal orientation in the laboratory frame.
        """
        if self.crystal_orientation is None:
            raise ValueError("Crystal orientation is not defined.")

        plot_crystal(
            self.crystal_orientation,
            xlims=xlims,
            ylims=ylims,
            zlims=zlims,
            axis_labels=axis_labels
        )

    def create_diffraction_cones(self, q_hkls, coeff=1, num_points=100):
        """
        Create 3D cone surfaces for the given q_hkls arrays.
        """
        if self.wavelength is None:
            raise AttributeError(
                "Wavelength is not defined. Please define it before creating diffraction cones."
            )

        self.diffraction_cones = create_diffraction_cones(
            wavelength=self.wavelength,
            q_hkls=q_hkls,
            coeff=coeff,
            num_points=num_points
        )

    def create_kf_hkls(self, q_hkls, num_points=100):
        """
        Compute kf vectors for the given q_hkls arrays.
        """
        if self.wavelength is None:
            raise AttributeError(
                "Wavelength is not defined. Please define it before creating diffraction cones."
            )

        self.kf_hkls = compute_kf_vectors(
            q_hkls,
            self.wavelength,
            num_points=num_points
        )

    def create_kf_hkls_about_axis(
        self,
        q_hkls,
        axis=(1.0, 0.0, 0.0),
        num_points=100,
        origin=None,
        start_angle=0.0,
        stop_angle=360.0,
        endpoint=False,
    ):
        """
        Generalized version of the current kf generation by rotating around an arbitrary axis.
        """
        if self.wavelength is None:
            raise AttributeError(
                "Wavelength is not defined. Please define it before creating kf vectors."
            )

        self.kf_hkls = compute_kf_vectors_about_axis(
            q_hkls=q_hkls,
            wavelength=self.wavelength,
            axis=axis,
            num_points=num_points,
            origin=origin,
            start_angle=start_angle,
            stop_angle=stop_angle,
            endpoint=endpoint,
        )

    def find_Bragg_orientations(self, hkls, angle_range=(-180, 180, 5)):
        """
        Find sample rotations that put each of the specified hkls in Bragg condition.
        """
        if (self.wavelength is None) or (self.e_bandwidth is None):
            raise AttributeError(
                "wavelength and e_bandwidth need to be different than None"
            )

        self.rotations_for_Bragg_condition = find_Bragg_orientations(
            hkls,
            self.crystal_orientation,
            self.wavelength,
            self.e_bandwidth,
            angle_range=angle_range,
            rotation_order=self.rotation_order
        )

    def find_Bragg_orientations_with_chain(
        self,
        hkls,
        motor_chain,
        scan_ranges,
        fixed_angles=None,
    ):
        """
        Generalized Bragg-orientation search using a MotorChain.
        """
        if (self.wavelength is None) or (self.e_bandwidth is None):
            raise AttributeError("wavelength and e_bandwidth need to be different than None")

        self.rotations_for_Bragg_condition = find_Bragg_orientations_with_chain(
            hkls=hkls,
            initial_crystal_orientation=self.crystal_orientation,
            wavelength=self.wavelength,
            e_bandwidth=self.e_bandwidth,
            motor_chain=motor_chain,
            scan_ranges=scan_ranges,
            fixed_angles=fixed_angles,
        )

    def find_Bragg_orientations_with_geometry(
        self,
        hkls,
        geometry,
        scan_ranges,
        fixed_sample_angles=None,
    ):
        """
        Generalized Bragg-orientation search using the sample arm of a DiffractometerGeometry.
        """
        if (self.wavelength is None) or (self.e_bandwidth is None):
            raise AttributeError("wavelength and e_bandwidth need to be different than None")

        self.rotations_for_Bragg_condition = find_Bragg_orientations_with_geometry(
            hkls=hkls,
            initial_crystal_orientation=self.crystal_orientation,
            wavelength=self.wavelength,
            e_bandwidth=self.e_bandwidth,
            geometry=geometry,
            scan_ranges=scan_ranges,
            fixed_sample_angles=fixed_sample_angles,
        )

    def plot_rotation_mapping(self):
        """
        Plot (roty, rotz) combinations from self.rotations_for_Bragg_condition
        that satisfy the Bragg condition.
        """
        if self.rotations_for_Bragg_condition is None:
            raise AttributeError(
                "rotations_for_Bragg_condition need to be computed first. "
                "Do find_Bragg_orientations(hkls, angle_range=(-180, 180, 5))"
            )

        title = (
            f"roty and rotz for (h k l) in Bragg condition\n"
            f"Energy [keV]: {utils.wavelength_to_energy(self.wavelength)*1e-3}, "
            f"Bandwidth [%]: ±{self.e_bandwidth/2}\n"
            f"Wavelength [Å]: {self.wavelength*1e10:.2}"
        )
        plot_rotation_mapping(self.rotations_for_Bragg_condition, title=title, s=20)

    def simulate_1d_pattern(
        self,
        energy,
        qmax,
        cif_file_path=None,
        atom_positions=False,
        include_multiplicity=False,
        include_lorentz_polarization=True,
        x_axis="q",
        step=0.01,
        fwhm=0.05,
        convolve=True,
        normalize=True,
    ):
        peaks = powder_peaks_1d(
            lattice=self,
            energy=energy,
            qmax=qmax,
            cif_file_path=cif_file_path,
            atom_positions=atom_positions,
            include_multiplicity=include_multiplicity,
            include_lorentz_polarization=include_lorentz_polarization,
            normalize=normalize,
        )

        if convolve:
            x, I = convolve_peaks_1d(
                peaks, x_axis=x_axis, step=step, fwhm=fwhm, normalize=normalize
            )
        else:
            x = np.array([])
            I = np.array([])

        out = {"x": x, "I": I, "peaks": peaks}
        self.pattern_1d = out
        return out


def compute_kf_vectors_about_axis(
    q_hkls,
    wavelength,
    axis=(1.0, 0.0, 0.0),
    num_points=30,
    origin=None,
    start_angle=0.0,
    stop_angle=360.0,
    endpoint=False,
):
    """
    Generalized kf-vector generation by rotating around an arbitrary axis.

    Parameters
    ----------
    q_hkls : array-like
        Either q magnitudes with shape (N,) or q vectors with shape (N, 3).
    wavelength : float
        X-ray wavelength in meters.
    axis : array-like, shape (3,)
        Rotation axis.
    num_points : int
        Number of rotation angles.
    origin : array-like or None
        A point through which the axis passes.
    start_angle, stop_angle : float
        Angular sweep in degrees.
    endpoint : bool
        Passed to np.linspace.
    """
    q_norms = _coerce_q_norms(q_hkls)

    radius, height = q_to_radius_and_height(wavelength, q_norms, coeff=1)
    kf_vectors = np.vstack((height, np.zeros_like(height), radius)).T
    kf_vectors *= (2 * np.pi / (wavelength * 1e10))

    angles = np.linspace(start_angle, stop_angle, num_points, endpoint=endpoint)
    all_rotated_kf_vectors = np.empty((len(kf_vectors), len(angles), 3), dtype=float)

    for i, angle in enumerate(angles):
        all_rotated_kf_vectors[:, i, :] = utils.rotate_about_axis(
            kf_vectors,
            axis=axis,
            angle=angle,
            origin=origin,
            degrees=True,
        )

    return all_rotated_kf_vectors


def find_Bragg_orientations_with_chain(
    hkls,
    initial_crystal_orientation,
    wavelength,
    e_bandwidth,
    motor_chain,
    scan_ranges,
    fixed_angles=None,
):
    """
    Generalized Bragg-orientation search over an arbitrary MotorChain.

    Parameters
    ----------
    hkls : array-like
        Miller indices.
    initial_crystal_orientation : numpy.ndarray
        3x3 orientation matrix.
    wavelength : float
        Wavelength in meters.
    e_bandwidth : float
        Energy bandwidth in percent.
    motor_chain : MotorChain
        Chain to scan.
    scan_ranges : dict
        Mapping motor_name -> (start, stop, step).
    fixed_angles : dict or None
        Fixed angles merged with motor defaults.

    Returns
    -------
    dict
        keys: stringified hkls
        values: list of dicts with the motor-angle combinations that satisfy Bragg.
    """
    if not isinstance(motor_chain, MotorChain):
        raise TypeError("motor_chain must be an instance of MotorChain.")

    hkls = np.asarray(hkls, dtype=int)
    initial_crystal_orientation = np.asarray(initial_crystal_orientation, dtype=float)

    if initial_crystal_orientation.shape != (3, 3):
        raise ValueError(
            "initial_crystal_orientation must have shape (3, 3)."
        )

    fixed_angles = {} if fixed_angles is None else dict(fixed_angles)
    scan_ranges = dict(scan_ranges)

    valid_orientations = {f"{hkl}": [] for hkl in hkls}

    motor_names = list(scan_ranges.keys())
    if len(motor_names) == 0:
        raise ValueError("scan_ranges cannot be empty.")

    angle_values = {
        name: _inclusive_angle_values(scan_ranges[name])
        for name in motor_names
    }

    total_steps = int(np.prod([len(vals) for vals in angle_values.values()]))
    progress_bar = tqdm(total=total_steps, desc="Calculating valid orientations")

    for combo in product(*[angle_values[name] for name in motor_names]):
        current_angles = dict(fixed_angles)
        current_angles.update(dict(zip(motor_names, combo)))

        transform = motor_chain.as_transform(angles=current_angles)
        rotated_matrix = apply_orientation_transform(initial_crystal_orientation, transform)

        reciprocal_lattice = cal_reciprocal_lattice(rotated_matrix)
        q_hkls = calculate_q_hkl(hkls, reciprocal_lattice)
        in_bragg = check_Bragg_condition(q_hkls, wavelength, e_bandwidth)

        combo_dict = {name: angle for name, angle in zip(motor_names, combo)}

        for hkl_val, ok in zip(hkls, in_bragg):
            if ok:
                valid_orientations[f"{hkl_val}"].append(combo_dict.copy())

        progress_bar.update(1)

    progress_bar.close()
    return valid_orientations


def find_Bragg_orientations_with_geometry(
    hkls,
    initial_crystal_orientation,
    wavelength,
    e_bandwidth,
    geometry,
    scan_ranges,
    fixed_sample_angles=None,
):
    """
    Convenience wrapper using the sample chain of a DiffractometerGeometry.
    """
    if not isinstance(geometry, DiffractometerGeometry):
        raise TypeError("geometry must be an instance of DiffractometerGeometry.")

    return find_Bragg_orientations_with_chain(
        hkls=hkls,
        initial_crystal_orientation=initial_crystal_orientation,
        wavelength=wavelength,
        e_bandwidth=e_bandwidth,
        motor_chain=geometry.sample,
        scan_ranges=scan_ranges,
        fixed_angles=fixed_sample_angles,
    )


def cal_reciprocal_lattice(lattice):
    """
    Calculate the reciprocal lattice vectors from the direct lattice vectors.

    Parameters:
        lattice (numpy.ndarray): A 3x3 matrix representing the direct lattice vectors.

    Returns:
        numpy.ndarray: A 3x3 matrix representing the reciprocal lattice vectors.
    """
    return np.linalg.inv(lattice).T * 2 * np.pi


def calculate_q_hkl(hkl, reciprocal_lattice):
    """
    Calculate the Q vector for given Miller indices and reciprocal lattice.

    Parameters:
        hkl (numpy.ndarray): An array of Miller indices (h, k, l).
        reciprocal_lattice (numpy.ndarray): A 3x3 matrix representing the reciprocal lattice vectors.

    Returns:
        numpy.ndarray: The Q vector corresponding to the given Miller indices.
    """
    return np.dot(hkl, reciprocal_lattice)


def calculate_dspacing(hkl, reciprocal_lattice):
    """
    Calculate the d-spacing for given Miller indices and reciprocal lattice.

    Parameters:
        hkl (numpy.ndarray): An array of Miller indices (h, k, l).
        reciprocal_lattice (numpy.ndarray): A 3x3 matrix representing the reciprocal lattice vectors.

    Returns:
        numpy.ndarray: The d-spacing values for the given Miller indices.
    """
    q_hkl = np.linalg.norm(calculate_q_hkl(hkl, reciprocal_lattice), axis=1)
    return 2 * np.pi / q_hkl


def calculate_two_theta(hkl, reciprocal_lattice, wavelength):
    """
    Calculate the two-theta angles for given Miller indices, reciprocal lattice, and wavelength.

    Parameters:
        hkl (numpy.ndarray): An array of Miller indices (h, k, l).
        reciprocal_lattice (numpy.ndarray): A 3x3 matrix representing the reciprocal lattice vectors.
        wavelength (float): The wavelength of the incident X-ray in meters.

    Returns:
        numpy.ndarray: The two-theta angles in degrees for the given Miller indices.
    """
    wavelength_angstrom = wavelength * 1e10  # Convert to Angstroms
    q_hkl = np.linalg.norm(calculate_q_hkl(hkl, reciprocal_lattice), axis=1)
    return np.rad2deg(2 * np.arcsin(wavelength_angstrom * q_hkl / (4 * np.pi)))


def check_Bragg_condition(q_hkls, wavelength, E_bandwidth):
    """
    Check if given Q vectors satisfy the Bragg condition for a given wavelength and energy bandwidth.

    Parameters:
        q_hkls (numpy.ndarray): Array of Q vectors.
        wavelength (float): Wavelength in meters.
        E_bandwidth (float): Energy bandwidth as a percentage.

    Returns:
        numpy.ndarray: Boolean array indicating which Q vectors satisfy the Bragg condition.
    """
    wavelength_angstrom = wavelength * 1e10  # Convert to Angstroms
    norm_squared = np.linalg.norm(q_hkls, axis=1) ** 2
    q_hkl0 = q_hkls[:, 0]

    result = np.zeros_like(q_hkl0, dtype=float)
    non_zero_indices = q_hkl0 != 0
    result[non_zero_indices] = (
        -(wavelength_angstrom / (4 * np.pi))
        * norm_squared[non_zero_indices]
        / q_hkl0[non_zero_indices]
    )

    min_i = 1 - E_bandwidth / 200
    max_f = 1 + E_bandwidth / 200

    return (result >= min_i) & (result <= max_f)


def q_to_radius_and_height(wavelength, q_hkls, coeff):
    """
    Convert q_hkl norms to diffraction cone radii.
    
    Parameters:
        q_hkls (numpy.ndarray): Norms of q_hkl values.
        wavelength (float): X-ray wavelength in meters.
        coeff (float): Scaling coefficient for the ring size.

    Returns:
        tuple: (r, h) arrays corresponding to cone radii and heights.
    """
    wavelength = wavelength * 1e10  # Convert to Angstroms
    arg = q_hkls * wavelength / (4 * np.pi)
    r = coeff * np.sin(2 * np.arcsin(arg))
    h = coeff * np.cos(2 * np.arcsin(arg))
    return r, h


def create_diffraction_cones(wavelength, q_hkls, coeff, num_points=100):
    """
    Create proper 3D cone surfaces for given q_hkl values.
    
    Parameters:
        q_hkls (numpy.ndarray): Array of q_hkl vectors (shape: (N, 3)).
        wavelength (float): Wavelength of the incident X-ray in meters.
        coeff (float): Scaling coefficient for the cone size.
        num_points (int): Number of points to define each cone's base.

    Returns:
        list: A list of arrays representing the cone surfaces for each q_hkl.
    """
    q_hkls = np.array(q_hkls)
    radii, heights = q_to_radius_and_height(wavelength, q_hkls, coeff)
    cones = []

    for radius, height in zip(radii, heights):
        # Generate cone surface using cylindrical coordinates
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = np.linspace(0, height, num_points)

        theta_grid, x_grid = np.meshgrid(theta, x)

        # Calculate y, z, x coordinates for the cone surface (x is along the height direction)
        r_grid = (radius / height) * x_grid
        y = r_grid * np.cos(theta_grid)
        z = r_grid * np.sin(theta_grid)
        cones.append((x_grid, y, z))

    return cones


def compute_kf_vectors(q_hkls, wavelength, num_points=30):
    """
    Compute initial kf vectors from q_hkls norms.

    Parameters:
        q_hkls (numpy.ndarray): Array of q_hkl norms.
        wavelength (float): X-ray wavelength in meters.
        num_points (int): Number of rotation angles around the beam direction.

    Returns:
        numpy.ndarray: Array of initial kf vectors rotated around the beam axis.
    """
    # Convert the q_hkls to radius and height
    radius, height = q_to_radius_and_height(wavelength, q_hkls, coeff=1)

    # Calculate the kf vectors
    kf_vectors = np.vstack((height, np.zeros_like(height), radius)).T \
        * (2 * np.pi / (wavelength * 1e10))

    theta = 360 / num_points
    rotation_angles = np.linspace(0, 360 - theta, num_points)
    all_rotated_kf_vectors = np.empty((len(kf_vectors), num_points, 3))

    for i in range(num_points):
        all_rotated_kf_vectors[:, i, :] = utils.apply_rotation(
            kf_vectors, rotation_angles[i], 0, 0
        )

    return all_rotated_kf_vectors


def find_Bragg_orientations(hkls, initial_crystal_orientation, wavelength, e_bandwidth,
                            angle_range=(-180, 180, 5), rotation_order="xyz"):
    """
    Find rotations that put each [h, k, l] from a list of hkls into the Bragg condition.

    Parameters:
        hkls (list of lists): List of Miller indices [h, k, l].
        initial_crystal_orientation (numpy.ndarray): Initial 3x3 orientation matrix.
        wavelength (float): Wavelength of the incident beam in meters.
        e_bandwidth (float): Energy bandwidth in percentage.
        angle_range (tuple): Range and step for angles (start, stop, step) in degrees.

    Returns:
        dict: Dictionary where keys are hkl tuples and values are lists of (roty, rotz) tuples.
    """
    angle_range = list(angle_range)
    angle_range[1] = angle_range[1] + angle_range[2]
    
    hkls = np.array(hkls)
    valid_orientations = {}

    for hkl in hkls:
        valid_orientations[f"{hkl}"] = []

    total_steps = len(np.arange(*angle_range)) * len(np.arange(*angle_range))
    progress_bar = tqdm(total=total_steps, desc="Calculating valid orientations")

    for roty in np.arange(*angle_range):
        for rotz in np.arange(*angle_range):
            rotated_matrix = utils.apply_rotation(
                initial_crystal_orientation, 0, roty, rotz, rotation_order
            )
            reciprocal_lattice = cal_reciprocal_lattice(rotated_matrix)

            q_hkls = calculate_q_hkl(hkls, reciprocal_lattice)
            dummies = check_Bragg_condition(q_hkls, wavelength, e_bandwidth)

            for hkl_val, dummy in zip(hkls, dummies):
                if dummy:
                    valid_orientations[f"{hkl_val}"].append((roty, rotz))

            progress_bar.update(1)

    progress_bar.close()
    return valid_orientations


def fractional_to_cartesian(lattice, fractional_coords):
    return np.dot(fractional_coords, lattice)


def extract_element_symbol(atom_label):

    # Initialize an empty string for the element symbol
    symbol = ''
    
    # Iterate through each character in the atom label
    for char in atom_label:
        if char.isalpha():
            symbol += char
        else:
            break 
    return symbol.capitalize()


def _build_xu_crystal_from_lattice(lattice, cif_file_path=None, atom_positions=False):
    """
    Build an xrayutilities Crystal.
    Semantics:
      - atom_positions=False: prefer xu.Crystal.fromCIF(cif_file_path) if possible
      - atom_positions=True: build using lattice.space_group + lattice_args + lattice.atom_positions
    """

    if (not atom_positions) and (cif_file_path is not None) and hasattr(xu.Crystal, "fromCIF"):
        return xu.Crystal.fromCIF(cif_file_path)

    if lattice.atom_positions is None:
        raise AttributeError("No atom_positions available to build xu.Crystal")

    atoms = list(lattice.atom_positions.keys())
    atom_symbols = [extract_element_symbol(a) for a in atoms]
    frac = np.array([lattice.atom_positions[a] for a in atoms], dtype=float)

    param_names, _ = lattice.get_lattice_params(lattice.space_group)
    lattice_args = [getattr(lattice, name) for name in param_names]

    sg_lat = xu.SGLattice(lattice.space_group, *lattice_args, atoms=atom_symbols, pos=frac)
    return xu.Crystal("sample", sg_lat)


def powder_peaks_1d(
    lattice,
    energy,
    qmax,
    cif_file_path=None,
    atom_positions=False,
    include_multiplicity=False,
    include_lorentz_polarization=True,
    normalize=True,
):
    """
    Return a peak list with q, d, 2θ, and relative intensities for a powder-like pattern.
    No convolution, no plotting.
    """
    from . import utils

    # Ensure hkls exist
    if lattice.allowed_hkls is None:
        lattice.create_possible_reflections(qmax)

    hkls = np.array(lattice.allowed_hkls, dtype=int)
    if hkls.size == 0:
        raise RuntimeError("No allowed hkls produced (check qmax / space group / lattice params).")

    # Beam
    wavelength_m = utils.energy_to_wavelength(energy)
    wavelength_A = wavelength_m * 1e10

    # Build xu crystal (for Q + structure factor)
    crystal = _build_xu_crystal_from_lattice(lattice, cif_file_path=cif_file_path, atom_positions=atom_positions)

    # Q vectors and |Q|
    qvecs = np.array([crystal.Q(hkl) for hkl in hkls], dtype=float)  # Å^-1 vectors
    q = np.linalg.norm(qvecs, axis=1)

    # Keep within qmax
    mask = (q > 0) & (q <= qmax * (1 + 1e-12))
    hkls = hkls[mask]
    qvecs = qvecs[mask]
    q = q[mask]

    d = (2 * np.pi) / q  # Å
    arg = np.clip(wavelength_A * q / (4 * np.pi), 0.0, 1.0)
    two_theta = np.degrees(2 * np.arcsin(arg))

    # Structure factors → intensity
    F = np.array([crystal.StructureFactor(qv, energy) for qv in qvecs], dtype=complex)
    I = np.abs(F) ** 2

    # Optional multiplicity (OFF by default since in your best match it was off)
    multiplicity = np.ones_like(I, dtype=int)
    if include_multiplicity and getattr(lattice, "xu_lattice", None) is not None:
        mult = []
        for hkl in hkls:
            eq = list(lattice.xu_lattice.equivalent_hkls(tuple(int(x) for x in hkl)))
            mult.append(max(1, len(eq)))
        multiplicity = np.array(mult, dtype=int)
        I = I * multiplicity

    # Optional Lorentz–polarization
    if include_lorentz_polarization:
        tt = np.radians(two_theta)
        th = 0.5 * tt
        denom = (np.sin(th) ** 2) * np.cos(th)
        denom = np.where(denom <= 1e-12, np.nan, denom)
        lp = (1.0 + np.cos(tt) ** 2) / denom
        lp = np.nan_to_num(lp, nan=0.0, posinf=0.0, neginf=0.0)
        I = I * lp

    if normalize and I.size and np.max(I) > 0:
        I = I / np.max(I)

    # Pack peaks
    peak_dtype = [
        ("h", "i4"), ("k", "i4"), ("l", "i4"),
        ("q", "f8"), ("d", "f8"), ("two_theta", "f8"),
        ("I", "f8"), ("multiplicity", "i4"),
    ]
    peaks = np.empty(len(hkls), dtype=peak_dtype)
    peaks["h"], peaks["k"], peaks["l"] = hkls[:, 0], hkls[:, 1], hkls[:, 2]
    peaks["q"], peaks["d"], peaks["two_theta"] = q, d, two_theta
    peaks["I"], peaks["multiplicity"] = I, multiplicity

    return peaks


def convolve_peaks_1d(peaks, x_axis="q", step=0.01, fwhm=0.05, x_max=None, normalize=True):
    """
    Turn peak list into a continuous 1D curve using Gaussian peak profile.
    """
    if x_axis not in {"q", "two_theta"}:
        raise ValueError("x_axis must be 'q' or 'two_theta'.")

    if step <= 0 or fwhm <= 0:
        raise ValueError("step and fwhm must be > 0.")

    x0 = peaks["q"] if x_axis == "q" else peaks["two_theta"]
    I0 = peaks["I"]

    # Sort by position
    order = np.argsort(x0)
    x0 = x0[order]
    I0 = I0[order]

    if x_max is None:
        x_max = float(np.max(x0)) if len(x0) else 0.0

    x = np.arange(0.0, x_max + step, step, dtype=float)
    y = np.zeros_like(x)

    if len(x0) == 0:
        return x, y

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half_window = int(np.ceil((6.0 * sigma) / step))

    for xc, amp in zip(x0, I0):
        if amp <= 0:
            continue
        center_idx = int(np.round(xc / step))
        lo = max(0, center_idx - half_window)
        hi = min(len(x), center_idx + half_window + 1)
        xs = x[lo:hi]
        y[lo:hi] += amp * np.exp(-0.5 * ((xs - xc) / sigma) ** 2)

    if normalize and y.size and np.max(y) > 0:
        y = y / np.max(y)

    return x, y
