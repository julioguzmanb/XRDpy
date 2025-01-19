import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar

import xrayutilities.materials as xu

from . import utils
from .plot import plot_crystal
from .plot import plot_rotation_mapping
from .cif import Cif


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

        # Identify the crystal system
        crystal_system, _ = cls.flattened_sgrp_sym[space_group_number]

        # Fetch parameters
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
        # Phase is equivalent to the crystal system in `flattened_sgrp_sym`
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
        v3_z = c * np.sqrt(1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 + 
                           2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad))
        v3 = np.array([v3_x, v3_y, v3_z])

        # Combine into a 3x3 matrix (lattice vectors as rows)
        return np.array([v1, v2, v3])


    def __init__(
            self, 
            space_group=None,
            a=None, b=None, c=None, 
            alpha=None, beta=None, gamma=None, 
            initial_crystal_orientation=None,
            rotation_order="xyz",
            cif_file_path=None
        ):
        """
        Initialize a lattice structure with its parameters and initial crystal orientation.

        Parameters:
            space_group (int): The space group number for the lattice.
            a, b, c (float): Lattice vector lengths.
            alpha, beta, gamma (float): Angles between lattice vectors in degrees.
            initial_crystal_orientation (numpy.ndarray): Initial crystal orientation matrix.
        """
        self.space_group = space_group
        self.phase = None  # Initialize phase attribute

        if cif_file_path is not None:
            try:
                cif=Cif(cif_file_path)
                if cif.space_group == 0:
                    if space_group is None:
                        print("Space Group Assumed to be 1")
                        self.space_group=1
                    else:
                        self.space_group=space_group
                else:
                    self.space_group=cif.space_group

                self.a = a = cif.a
                self.b = b = cif.b
                self.c = c = cif.c
                self.alpha = alpha = cif.alpha
                self.beta = beta = cif.beta
                self.gamma = gamma = cif.gamma
                self.atom_positions=cif.atom_positions

            except: ImportError

        if self.space_group is not None:
            param_names, default_values = self.get_lattice_params(self.space_group)
            self.phase = self.get_phase(self.space_group)
            defaults = dict(zip(param_names, default_values))

            # Assign lattice parameters with fallback to defaults
            self.a = a if a is not None else defaults.get('a')
            self.b = b if b is not None else defaults.get('b', self.a)
            self.c = c if c is not None else defaults.get('c', self.a)
            self.alpha = alpha if alpha is not None else defaults.get('alpha', 90)
            self.beta = beta if beta is not None else defaults.get('beta', 90)
            self.gamma = gamma if gamma is not None else defaults.get('gamma', 90)

            # Prepare arguments for SGLattice
            lattice_args = [getattr(self, name) for name in param_names]

            # Initialize SGLattice object
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
        self.rotation_order=rotation_order

        # Optional attributes for reciprocal lattice and reflections
        self.reciprocal_lattice = None
        self.allowed_hkls = None
        self.wavelength=None
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

        Parameters:
            rotx (float): Rotation angle around the x-axis in degrees.
            roty (float): Rotation angle around the y-axis in degrees.
            rotz (float): Rotation angle around the z-axis in degrees.
            rotation_order (str): Order of rotation, e.g., "xyz".
            mode (str): "absolute" applies rotation to the initial orientation,
                        "relative" applies rotation to the current orientation.

        Raises:
            ValueError: If mode is not "relative" or "absolute".
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

    def calculate_reciprocal_lattice(self):
        """Calculate and store the reciprocal lattice vectors."""
        self.reciprocal_lattice = cal_reciprocal_lattice(self.crystal_orientation)


    def create_possible_reflections(self, qmax):
        """Get allowed reflections for a given maximum Q-value in Å^-1."""
        if not isinstance(qmax, (int, float)) or qmax <= 0:
            raise ValueError("qmax must be a positive number.")
        if self.xu_lattice is None:
            raise RuntimeError("SGLattice is not initialized.")
        reflections=self.xu_lattice.get_allowed_hkl(qmax)

        self.allowed_hkls=np.array(list(reflections))

    
    def check_if_hkl_allowed(self, hkl):
        print(self.xu_lattice.hkl_allowed(hkl))
    
    def get_equivalent_reflections(self, hkl):
        print(self.xu_lattice.equivalent_hkls(hkl))
    
    
    

    def calculate_q_hkls(self):
        """
        Calculate the Q vectors for the allowed Miller indices and reciprocal lattice.

        Raises:
            AttributeError: If allowed HKLs are not computed.
        """
        if self.reciprocal_lattice is None:
            self.reciprocal_lattice = cal_reciprocal_lattice(self.crystal_orientation)

        if self.allowed_hkls is None:
            raise AttributeError("Compute the allowed HKLs using create_possible_reflections.")

        self.q_hkls = calculate_q_hkl(self.allowed_hkls, self.reciprocal_lattice)

    def calculate_d_hkls(self):
        """
        Calculate the d-spacing values for the allowed Miller indices.

        Raises:
            AttributeError: If allowed HKLs are not computed.
        """
        if self.reciprocal_lattice is None:
            self.reciprocal_lattice = cal_reciprocal_lattice(self.crystal_orientation)

        if self.allowed_hkls is None:
            raise AttributeError("Compute the allowed HKLs using create_possible_reflections.")

        self.d_hkls = calculate_dspacing(self.allowed_hkls, self.reciprocal_lattice)

    def calculate_two_theta_hkls(self, wavelength):
        """
        Calculate the two-theta angles for the allowed Miller indices.

        Parameters:
            wavelength (float): The wavelength of the incident X-ray in meters.

        Raises:
            AttributeError: If allowed HKLs are not computed.
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

        Parameters:
            wavelength (float): The wavelength in meters.
            e_bandwidth (float): Energy bandwidth as a percentage.
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

        Parameters:
            xlims (tuple): Limits for the x-axis.
            ylims (tuple): Limits for the y-axis.
            zlims (tuple): Limits for the z-axis.

        Raises:
            ValueError: If the crystal orientation is not defined.
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

        Parameters:
            q_hkls (numpy.ndarray): Q vectors for the reflections of interest.
            coeff (float): Scaling coefficient for the cone size.
            num_points (int): Number of points defining the cone's perimeter.

        Raises:
            AttributeError: If wavelength is not defined.
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

        Parameters:
            q_hkls (numpy.ndarray): Q vectors for the reflections of interest.
            num_points (int): Number of rotation increments around the beam direction.

        Raises:
            AttributeError: If wavelength is not defined.
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

    def find_Bragg_orientations(self, hkls, angle_range=(-180, 180, 5)):
        """
        Find sample rotations that put each of the specified hkls in Bragg condition.

        Parameters:
            hkls (array-like): Array of Miller indices to check.
            angle_range (tuple): (start, stop, step) for roty, rotz angle sweeps.

        Raises:
            AttributeError: If wavelength or e_bandwidth is None.
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
            angle_range=angle_range
        )

    def plot_rotation_mapping(self):
        """
        Plot (roty, rotz) combinations from self.rotations_for_Bragg_condition 
        that satisfy the Bragg condition.
        
        Raises:
            AttributeError: If self.rotations_for_Bragg_condition is None.
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
        -(wavelength_angstrom / (2 * np.pi))
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





def allowed_reflections(phase, hkl):
    """
    Determine if a given set of Miller indices (hkl) is allowed for a specific crystal phase.

    Parameters:
        phase (str): The crystal phase, either "Hexagonal" or "Monoclinic".
        hkl (tuple of int): A tuple containing the Miller indices (h, k, l).

    Returns:
        bool: True if the reflection is allowed for the specified phase, False otherwise.
    """
    h, k, l = hkl

    if h == k == l == 0:
        return False  # Exclude the [0, 0, 0] reflection

    if phase == "Hexagonal":
        # Rules for space group 167
        if (h, k, l) in [(-2, 0, 4), (2, 0, 4)]:
            return True
        if ((-h + k + l) % 3 == 0) and h and k and l:
            return True
        if h == 0 and l % 2 == 0 and (k + l) % 3 == 0 and k and l:
            return True
        if k == 0 and l % 2 == 0 and (h - l) % 3 == 0 and h and l:
            return True
        if l == 0 and (h - k) % 3 == 0 and h and k:
            return True
        if h == k != 0 and l % 3 == 0:
            return True
        if k == 0 and l == 0 and h % 3 == 0 and h:
            return True
        if h == 0 and l == 0 and k % 3 == 0 and k:
            return True
        if h == k == 0 and l % 6 == 0 and l:
            return True

    elif phase == "Monoclinic":
        # Rules for Monoclinic phase
        conditions = [
            ((h + k) % 2 == 0 and h and k and l),
            (h == 0 and k % 2 == 0 and k and l),
            (k == 0 and h % 2 == 0 and l % 2 == 0 and h and l),
            (l == 0 and (h + k) % 2 == 0 and h and k),
            (k == l == 0 and h % 2 == 0),
            (h == l == 0 and k % 2 == 0),
            (h == k == 0 and l % 2 == 0),
            (h, k, l) in [
                (2, 2, 1), (-2, 2, -1), (2, -2, 1), (-2, -2, -1),
                (0, 1, 3), (0, 1, -3), (0, -1, 3), (0, -1, -3),
                (4, 1, -1), (4, 1, 1), (4, -1, -1), (4, -1, 1),
            ]
        ]
        if any(conditions):
            return True

    return False


def create_possible_reflections(phase, smallest_number, largest_number):
    """
    Generate all possible reflections within a specified range for a given crystal phase.

    Parameters:
        phase (str): The crystal phase, e.g., "Hexagonal" or "Monoclinic".
        smallest_number (int): The smallest Miller index to consider.
        largest_number (int): The largest Miller index to consider.

    Returns:
        numpy.ndarray: An array of allowed (h, k, l) reflections.
    """
    rango_hkl = np.arange(smallest_number, largest_number + 1)
    h, k, l = np.meshgrid(rango_hkl, rango_hkl, rango_hkl)

    combinaciones_hkl = np.column_stack((h.ravel(), k.ravel(), l.ravel()))
    allowed_mask = np.apply_along_axis(
        lambda x: allowed_reflections(phase, tuple(x)),
        axis=1,
        arr=combinaciones_hkl
    )

    return combinaciones_hkl[allowed_mask]
