from .. import utils
from .. import sample
from .. import detector
from .. import experiment
from tqdm import tqdm

import numpy as np



def simulate_3d(
        det_type="manual", 
        det_pxsize_h=50e-6, det_pxsize_v=50e-6, 
        det_ntum_pixels_h=2000, det_num_pixels_v=2000, det_binning=(1,1),
        det_dist=0.5, det_poni1=0, det_poni2=0, 
        det_rotx=0, det_roty=0, det_rotz=0, 
        det_rotation_order="xyz",
        energy=10e3, e_bandwidth=1.5,
        sam_space_group=167, 
        sam_a=None, sam_b=None, sam_c=None, sam_alpha=None, sam_beta=None, sam_gamma=None,
        sam_initial_crystal_orientation=None,
        sam_rotx=0, sam_roty=0, sam_rotz=0,
        sam_rotation_order="xyz",
        qmax=10
):

    try:
        det = detector.Detector(
            detector_type=det_type,
            pxsize_h=det_pxsize_h, pxsize_v=det_pxsize_v,
            num_pixels_h=det_ntum_pixels_h, num_pixels_v=det_num_pixels_v,
            dist=det_dist, poni1=det_poni1, poni2=det_poni2,
            rotx=det_rotx, roty=det_roty, rotz=det_rotz,
            rotation_order=det_rotation_order,
            binning=det_binning
        )
        det.calculate_lab_grid()
    except:
        raise ImportError

        
    lattice = sample.LatticeStructure(
        space_group=sam_space_group,
        a=sam_a, b=sam_b, c=sam_c, alpha=sam_alpha, beta=sam_beta, gamma=sam_gamma,
        initial_crystal_orientation=sam_initial_crystal_orientation,
        rotation_order=sam_rotation_order
        )
        
    lattice.apply_rotation(rotx=sam_rotx, roty=sam_roty, rotz=sam_rotz)
    lattice.calculate_reciprocal_lattice()
    
    lattice.create_possible_reflections(qmax)
    lattice.calculate_q_hkls()
    wavelength = utils.energy_to_wavelength(energy)
    lattice.check_Bragg_condition(wavelength, e_bandwidth)

    exp = experiment.Experiment(det, lattice, energy = energy, e_bandwidth = e_bandwidth)
    exp.summary()
    exp.calculate_diffraction_direction(qmax)
    exp.plot_3d_single_xstal_exp()

    return exp

    
def simulate_2d(
        det_type="manual", 
        det_pxsize_h=50e-6, det_pxsize_v=50e-6, 
        det_ntum_pixels_h=2000, det_num_pixels_v=2000, det_binning=(1,1),
        det_dist=0.5, det_poni1=0, det_poni2=0, 
        det_rotx=0, det_roty=0, det_rotz=0, 
        det_rotation_order="xyz",
        energy=10e3, e_bandwidth=1.5,
        sam_space_group=167, 
        sam_a=None, sam_b=None, sam_c=None, sam_alpha=None, sam_beta=None, sam_gamma=None,
        sam_initial_crystal_orientation=None,
        sam_rotx=0, sam_roty=0, sam_rotz=0,
        sam_rotation_order="xyz",
        qmax=10, 

):

    try:
        det = detector.Detector(
            detector_type=det_type,
            pxsize_h=det_pxsize_h, pxsize_v=det_pxsize_v,
            num_pixels_h=det_ntum_pixels_h, num_pixels_v=det_num_pixels_v,
            dist=det_dist, poni1=det_poni1, poni2=det_poni2,
            rotx=det_rotx, roty=det_roty, rotz=det_rotz,
            rotation_order=det_rotation_order,
            binning=det_binning
        )
        det.calculate_lab_grid()
    except:
        raise ImportError


    lattice = sample.LatticeStructure(
        space_group=sam_space_group,
        a=sam_a, b=sam_b, c=sam_c, alpha=sam_alpha, beta=sam_beta, gamma=sam_gamma,
        initial_crystal_orientation=sam_initial_crystal_orientation,
        rotation_order=sam_rotation_order
        )
        
    lattice.apply_rotation(rotx=sam_rotx, roty=sam_roty, rotz=sam_rotz)
    lattice.calculate_reciprocal_lattice()
    
    lattice.create_possible_reflections(qmax)
    lattice.calculate_q_hkls()
    wavelength = utils.energy_to_wavelength(energy)
    lattice.check_Bragg_condition(wavelength, e_bandwidth)

    exp = experiment.Experiment(det, lattice, energy = energy, e_bandwidth = e_bandwidth)
    exp.summary()
    exp.calculate_diffraction_direction(qmax)
    exp.calculate_pixel_positions()
    exp.plot_2d_single_xstal_exp()

    return exp


def sample_rotations_for_Bragg_condition(
        sam_space_group, 
        sam_a=None, sam_b=None, sam_c=None, sam_alpha=None, sam_beta=None, sam_gamma=None,
        sam_initial_crystal_orientation=None,
        sam_rotx=0, sam_roty=0, sam_rotz=0,
        sam_rotation_order="xyz",
        angle_range=(-180, 180, 1),
        energy=10e3, e_bandwidth=1.5,
        q_hkls=None, d_hkls=None,
        hkls_names=None
):

    if d_hkls is not None:
        d_hkls = np.array(d_hkls)
        q_hkls = 2 * np.pi / d_hkls

    elif q_hkls is not None:
        q_hkls = np.array(q_hkls)
    
    if ((q_hkls is not None) and (hkls_names is not None)) and (len(q_hkls) != len(hkls_names)):
        raise ValueError
    
    
    lattice = sample.LatticeStructure(
        space_group=sam_space_group,
        a=sam_a, b=sam_b, c=sam_c, alpha=sam_alpha, beta=sam_beta, gamma=sam_gamma,
        initial_crystal_orientation=sam_initial_crystal_orientation,
        rotation_order=sam_rotation_order
        )
    
    lattice.apply_rotation(rotx=sam_rotx, roty=sam_roty, rotz=sam_rotz)
   
    lattice.wavelength=utils.energy_to_wavelength(energy)
    lattice.e_bandwidth=e_bandwidth

    lattice.find_Bragg_orientations(hkls_names, angle_range=angle_range)
    lattice.plot_rotation_mapping()

    return lattice


def detector_rotations_collecting_Braggs(
        det_type="manual", 
        det_pxsize_h=50e-6, det_pxsize_v=50e-6, 
        det_ntum_pixels_h=2000, det_num_pixels_v=2000, det_binning=(1,1),
        det_dist=0.5, det_poni1=0, det_poni2=0, 
        det_rotation_order="xyz",
        angle_range=(-90, 90, 10),
        energy=10e3, e_bandwidth=1.5,
        sam_space_group=167, 
        sam_a=None, sam_b=None, sam_c=None, sam_alpha=None, sam_beta=None, sam_gamma=None,
        sam_initial_crystal_orientation=None,
        sam_rotx=0, sam_roty=0, sam_rotz=0,
        sam_rotation_order="xyz",
        qmax=10, 
        hkls=None
):

    try:
        det = detector.Detector(
            detector_type=det_type,
            pxsize_h=det_pxsize_h, pxsize_v=det_pxsize_v,
            num_pixels_h=det_ntum_pixels_h, num_pixels_v=det_num_pixels_v,
            dist=det_dist, poni1=det_poni1, poni2=det_poni2,
            rotation_order=det_rotation_order,
            binning=det_binning

        )
        det.calculate_lab_grid()
    except:
        raise ImportError
    
        
    lattice = sample.LatticeStructure(
        space_group=sam_space_group,
        a=sam_a, b=sam_b, c=sam_c, alpha=sam_alpha, beta=sam_beta, gamma=sam_gamma,
        initial_crystal_orientation=sam_initial_crystal_orientation,
        rotation_order=sam_rotation_order
        )
        
    lattice.apply_rotation(rotx=sam_rotx, roty=sam_roty, rotz=sam_rotz)

    lattice.calculate_reciprocal_lattice()


    if hkls is None:
        lattice.create_possible_reflections(qmax)
        
    else:
        lattice.allowed_hkls = np.array(list(hkls))
    
    lattice.calculate_q_hkls()
    wavelength = utils.energy_to_wavelength(energy)
    lattice.check_Bragg_condition(wavelength, e_bandwidth)
    
    exp = experiment.Experiment(det, lattice, energy = energy, e_bandwidth = e_bandwidth)
    exp.summary()
    exp.find_detector_rotations(hkls, angle_range=angle_range)
    exp.plot_rotation_mapping()

    return exp


def scan_two_parameters_for_Bragg_condition(
        param1_name,
        param2_name,
        param1_range,
        param2_range,
        sam_space_group,
        sam_a=None, sam_b=None, sam_c=None,
        sam_alpha=None, sam_beta=None, sam_gamma=None,
        sam_initial_crystal_orientation=None,
        sam_rotx=0, sam_roty=0, sam_rotz=0,
        sam_rotation_order="xyz",
        energy=10e3,
        e_bandwidth=1.5,
        hkls_names=None,
        hkl_equivalent = False
):
    """
    Scan two parameters (chosen among 'rotx', 'roty', 'rotz', 'energy') and
    build a mapping of (param1, param2) pairs that put given hkls in Bragg condition.

    Parameters
    ----------
    param1_name : str
        Name of the first parameter: one of {"rotx", "roty", "rotz", "energy"}.
    param2_name : str
        Name of the second parameter: one of {"rotx", "roty", "rotz", "energy"}.
        Must be different from param1_name.
    param1_range : tuple
        (start, stop, step) for param1 values. Inclusive on the stop side.
    param2_range : tuple
        (start, stop, step) for param2 values. Inclusive on the stop side.
    sam_space_group : int
        Space group number for the sample lattice.
    sam_a, sam_b, sam_c : float
        Lattice constants in Å.
    sam_alpha, sam_beta, sam_gamma : float
        Lattice angles in degrees.
    sam_initial_crystal_orientation : np.ndarray, optional
        3x3 initial orientation matrix. If None, it is built from lattice parameters.
    sam_rotx, sam_roty, sam_rotz : float
        Initial sample rotations (deg) applied *before* scanning param1/param2.
    sam_rotation_order : str
        Rotation order used in utils.apply_rotation.
    energy : float
        Base X-ray energy in eV (used if neither param1 nor param2 is 'energy').
    e_bandwidth : float
        Energy bandwidth in percent.
    hkls_names : array-like, shape (N, 3)
        Miller indices to test for Bragg condition.
    hkl_equivalent : bool
        If True, include equivalent reflections for each hkl in hkls_names.

    Returns
    -------
    dict
        Dictionary mapping each hkl (as string "[h k l]") to a list of
        (param1_value, param2_value) tuples where that reflection is in Bragg condition.

    Notes
    -----
    - If param1_name='roty' and param2_name='rotz', with param1_range and
      param2_range given in degrees, this reproduces the logic of
      `sample_rotations_for_Bragg_condition` (up to the initial sam_rotx/sam_roty/sam_rotz).
    - The scan is performed by:
        1) Building a baseline crystal orientation (including sam_rotx/y/z),
        2) For each (p1, p2), applying additional rotations on the selected axes,
        3) Computing Q_hkl and checking Bragg condition at the corresponding energy.
    """

    valid_param_names = {"rotx", "roty", "rotz", "energy"}

    if param1_name not in valid_param_names:
        raise ValueError(f"param1_name must be one of {valid_param_names}, got {param1_name!r}")
    if param2_name not in valid_param_names:
        raise ValueError(f"param2_name must be one of {valid_param_names}, got {param2_name!r}")
    if param1_name == param2_name:
        raise ValueError("param1_name and param2_name must be different.")

    if hkls_names is None:
        raise ValueError("hkls_names must be provided and not None.")

    hkls = np.array(hkls_names, dtype=float)
    if hkls.ndim != 2 or hkls.shape[1] != 3:
        raise ValueError("hkls_names must be an array-like of shape (N, 3).")

    # Build the lattice
    lattice = sample.LatticeStructure(
        space_group=sam_space_group,
        a=sam_a, b=sam_b, c=sam_c,
        alpha=sam_alpha, beta=sam_beta, gamma=sam_gamma,
        initial_crystal_orientation=sam_initial_crystal_orientation,
        rotation_order=sam_rotation_order
    )

    if hkl_equivalent:
        total_hkls = []
        seen = set()

        for hkl in hkls:
            # ensure plain python list or 1D array
            equivs = lattice.get_equivalent_reflections(hkl)

            for eq in equivs:
                t = tuple(int(x) for x in eq)  # hashable
                if t not in seen:
                    seen.add(t)
                    total_hkls.append([t[0], t[1], t[2]])

        if total_hkls:
            hkls = np.array(total_hkls, dtype=float)
        else:
            hkls = np.array(hkls, dtype=float)
    else:
        pass



    # Apply initial sample rotation (baseline orientation for the scan)
    lattice.apply_rotation(rotx=sam_rotx, roty=sam_roty, rotz=sam_rotz)
    initial_orientation = lattice.crystal_orientation.copy()

    # Prepare parameter grids (inclusive range, like in other functions)
    p1_range = list(param1_range)
    p1_range[1] = p1_range[1] + p1_range[2]
    p2_range = list(param2_range)
    p2_range[1] = p2_range[1] + p2_range[2]

    p1_values = np.arange(*p1_range)
    p2_values = np.arange(*p2_range)

    # Prepare output container
    valid_points = {f"{hkl}": [] for hkl in hkls}

    # Total steps for optional progress bar
    total_steps = len(p1_values) * len(p2_values)

    # If you don't want tqdm, just replace the tqdm(...) loop with a simple for-loop.
    with tqdm(total=total_steps, desc="Scanning parameters for Bragg condition") as progress_bar:
        for p1 in p1_values:
            for p2 in p2_values:
                # Determine current energy
                if param1_name == "energy":
                    current_energy = p1
                elif param2_name == "energy":
                    current_energy = p2
                else:
                    current_energy = energy

                current_wavelength = utils.energy_to_wavelength(current_energy)

                # Determine additional rotation increments from param1/param2
                # (on top of the baseline orientation we already built)
                extra_rotx = 0.0
                extra_roty = 0.0
                extra_rotz = 0.0

                if param1_name == "rotx":
                    extra_rotx += p1
                elif param1_name == "roty":
                    extra_roty += p1
                elif param1_name == "rotz":
                    extra_rotz += p1

                if param2_name == "rotx":
                    extra_rotx += p2
                elif param2_name == "roty":
                    extra_roty += p2
                elif param2_name == "rotz":
                    extra_rotz += p2

                # Apply extra rotations to the baseline orientation
                rotated_matrix = utils.apply_rotation(
                    initial_orientation,
                    extra_rotx,
                    extra_roty,
                    extra_rotz,
                    sam_rotation_order
                )

                # Reciprocal lattice and Q_hkl
                reciprocal_lattice = sample.cal_reciprocal_lattice(rotated_matrix)
                q_hkls = sample.calculate_q_hkl(hkls, reciprocal_lattice)

                # Bragg condition check
                in_bragg = sample.check_Bragg_condition(q_hkls, current_wavelength, e_bandwidth)

                # Store (p1, p2) pairs where Bragg condition is satisfied
                for hkl_val, is_in in zip(hkls, in_bragg):
                    if is_in:
                        valid_points[f"{hkl_val}"].append((p1, p2))

                progress_bar.update(1)

    return valid_points


