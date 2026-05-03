from . import utils
from . import sample
from . import detector
from . import experiment
from . import plot

from .geometry import MotorChain, DiffractometerGeometry

import numpy as np

from tqdm import tqdm

import warnings
from dataclasses import dataclass

from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

def _normalize_hkl_triplets(hkls, name="hkls"):
    arr = np.asarray(hkls, dtype=int)

    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError(f"{name} must be a Miller triplet [h, k, l] or an array of shape (N, 3).")
        arr = arr.reshape(1, 3)
    elif arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be a Miller triplet [h, k, l] or an array of shape (N, 3).")

    return arr


def _merge_extra_hkls_into_lattice(lattice, extra_hkls=None):
    allowed_hkls = np.asarray(lattice.allowed_hkls, dtype=int)
    if allowed_hkls.ndim == 1:
        allowed_hkls = allowed_hkls.reshape(1, 3)

    if extra_hkls is None:
        lattice.allowed_hkls = allowed_hkls
        return

    extra_hkls = _normalize_hkl_triplets(extra_hkls, name="extra_hkls")
    combined_hkls = np.vstack((allowed_hkls, extra_hkls))

    _, unique_idx = np.unique(combined_hkls, axis=0, return_index=True)
    lattice.allowed_hkls = combined_hkls[np.sort(unique_idx)]


def _resolve_geometry_pair(geometry=None, sample_geometry=None, detector_geometry=None):
    if geometry is not None:
        if sample_geometry is None:
            sample_geometry = geometry
        if detector_geometry is None:
            detector_geometry = geometry
    return sample_geometry, detector_geometry


def _build_single_crystal_detector(
    det_type="manual",
    det_pxsize_h=50e-6,
    det_pxsize_v=50e-6,
    det_ntum_pixels_h=2000,
    det_num_pixels_v=2000,
    det_binning=(1, 1),
    det_dist=0.5,
    det_poni1=0,
    det_poni2=0,
    det_rotx=0,
    det_roty=0,
    det_rotz=0,
    det_rotation_order="xyz",
    geometry=None,
    detector_motor_chain=None,
    detector_angles=None,
    detector_transform=None,
):
    det = detector.Detector(
        detector_type=det_type,
        pxsize_h=det_pxsize_h,
        pxsize_v=det_pxsize_v,
        num_pixels_h=det_ntum_pixels_h,
        num_pixels_v=det_num_pixels_v,
        dist=det_dist,
        poni1=det_poni1,
        poni2=det_poni2,
        rotx=det_rotx,
        roty=det_roty,
        rotz=det_rotz,
        rotation_order=det_rotation_order,
        binning=det_binning,
    )

    if geometry is not None:
        det.set_diffractometer(geometry, angles=detector_angles)
    elif detector_motor_chain is not None:
        det.set_motor_chain(detector_motor_chain, angles=detector_angles)
    elif detector_transform is not None:
        det.set_transform(detector_transform)
    else:
        det.clear_transform()

    det.calculate_lab_grid()
    return det


def _build_single_crystal_lattice(
    sam_space_group=167,
    sam_a=None,
    sam_b=None,
    sam_c=None,
    sam_alpha=None,
    sam_beta=None,
    sam_gamma=None,
    sam_initial_crystal_orientation=None,
    sam_rotx=0,
    sam_roty=0,
    sam_rotz=0,
    sam_rotation_order="xyz",
    energy=10e3,
    e_bandwidth=1.5,
    qmax=10,
    extra_hkls=None,
    geometry=None,
    sample_motor_chain=None,
    sample_angles=None,
    sample_transform=None,
):
    lattice = sample.LatticeStructure(
        space_group=sam_space_group,
        a=sam_a,
        b=sam_b,
        c=sam_c,
        alpha=sam_alpha,
        beta=sam_beta,
        gamma=sam_gamma,
        initial_crystal_orientation=sam_initial_crystal_orientation,
        rotation_order=sam_rotation_order,
    )

    if geometry is not None:
        lattice.apply_diffractometer(geometry, angles=sample_angles, mode="absolute")
    elif sample_motor_chain is not None:
        lattice.apply_motor_chain(sample_motor_chain, angles=sample_angles, mode="absolute")
    elif sample_transform is not None:
        lattice.apply_transform(sample_transform, mode="absolute")
    else:
        lattice.apply_rotation(rotx=sam_rotx, roty=sam_roty, rotz=sam_rotz, mode="absolute")

    lattice.calculate_reciprocal_lattice()
    lattice.create_possible_reflections(qmax)
    _merge_extra_hkls_into_lattice(lattice, extra_hkls=extra_hkls)

    lattice.calculate_q_hkls()
    wavelength = utils.energy_to_wavelength(energy)
    lattice.check_Bragg_condition(wavelength, e_bandwidth)

    return lattice


def _expand_equivalent_hkls(lattice, hkls_names):
    hkls = _normalize_hkl_triplets(hkls_names, name="hkls_names")
    total_hkls = []
    seen = set()

    for hkl in hkls:
        equivs = lattice.get_equivalent_reflections(hkl)
        for eq in equivs:
            t = tuple(int(x) for x in eq)
            if t not in seen:
                seen.add(t)
                total_hkls.append([t[0], t[1], t[2]])

    if total_hkls:
        return np.asarray(total_hkls, dtype=int)

    return hkls


def simulate_3d_with_geometry(
    det_type="manual",
    det_pxsize_h=50e-6,
    det_pxsize_v=50e-6,
    det_ntum_pixels_h=2000,
    det_num_pixels_v=2000,
    det_binning=(1, 1),
    det_dist=0.5,
    det_poni1=0,
    det_poni2=0,
    det_rotx=0,
    det_roty=0,
    det_rotz=0,
    det_rotation_order="xyz",
    energy=10e3,
    e_bandwidth=1.5,
    sam_space_group=167,
    sam_a=None,
    sam_b=None,
    sam_c=None,
    sam_alpha=None,
    sam_beta=None,
    sam_gamma=None,
    sam_initial_crystal_orientation=None,
    sam_rotx=0,
    sam_roty=0,
    sam_rotz=0,
    sam_rotation_order="xyz",
    qmax=10,
    extra_hkls=None,
    geometry=None,
    sample_geometry=None,
    detector_geometry=None,
    sample_motor_chain=None,
    detector_motor_chain=None,
    sample_angles=None,
    detector_angles=None,
    sample_transform=None,
    detector_transform=None,
):
    sample_geometry, detector_geometry = _resolve_geometry_pair(
        geometry=geometry,
        sample_geometry=sample_geometry,
        detector_geometry=detector_geometry,
    )

    det = _build_single_crystal_detector(
        det_type=det_type,
        det_pxsize_h=det_pxsize_h,
        det_pxsize_v=det_pxsize_v,
        det_ntum_pixels_h=det_ntum_pixels_h,
        det_num_pixels_v=det_num_pixels_v,
        det_binning=det_binning,
        det_dist=det_dist,
        det_poni1=det_poni1,
        det_poni2=det_poni2,
        det_rotx=det_rotx,
        det_roty=det_roty,
        det_rotz=det_rotz,
        det_rotation_order=det_rotation_order,
        geometry=detector_geometry,
        detector_motor_chain=detector_motor_chain,
        detector_angles=detector_angles,
        detector_transform=detector_transform,
    )

    lattice = _build_single_crystal_lattice(
        sam_space_group=sam_space_group,
        sam_a=sam_a,
        sam_b=sam_b,
        sam_c=sam_c,
        sam_alpha=sam_alpha,
        sam_beta=sam_beta,
        sam_gamma=sam_gamma,
        sam_initial_crystal_orientation=sam_initial_crystal_orientation,
        sam_rotx=sam_rotx,
        sam_roty=sam_roty,
        sam_rotz=sam_rotz,
        sam_rotation_order=sam_rotation_order,
        energy=energy,
        e_bandwidth=e_bandwidth,
        qmax=qmax,
        extra_hkls=extra_hkls,
        geometry=sample_geometry,
        sample_motor_chain=sample_motor_chain,
        sample_angles=sample_angles,
        sample_transform=sample_transform,
    )

    exp = experiment.Experiment(det, lattice, energy=energy, e_bandwidth=e_bandwidth)
    exp.summary()
    exp.calculate_diffraction_direction(qmax)
    exp.plot_3d_single_xstal_exp()

    return exp


def simulate_2d_with_geometry(
    det_type="manual",
    det_pxsize_h=50e-6,
    det_pxsize_v=50e-6,
    det_ntum_pixels_h=2000,
    det_num_pixels_v=2000,
    det_binning=(1, 1),
    det_dist=0.5,
    det_poni1=0,
    det_poni2=0,
    det_rotx=0,
    det_roty=0,
    det_rotz=0,
    det_rotation_order="xyz",
    energy=10e3,
    e_bandwidth=1.5,
    sam_space_group=167,
    sam_a=None,
    sam_b=None,
    sam_c=None,
    sam_alpha=None,
    sam_beta=None,
    sam_gamma=None,
    sam_initial_crystal_orientation=None,
    sam_rotx=0,
    sam_roty=0,
    sam_rotz=0,
    sam_rotation_order="xyz",
    qmax=10,
    extra_hkls=None,
    geometry=None,
    sample_geometry=None,
    detector_geometry=None,
    sample_motor_chain=None,
    detector_motor_chain=None,
    sample_angles=None,
    detector_angles=None,
    sample_transform=None,
    detector_transform=None,
):
    sample_geometry, detector_geometry = _resolve_geometry_pair(
        geometry=geometry,
        sample_geometry=sample_geometry,
        detector_geometry=detector_geometry,
    )

    det = _build_single_crystal_detector(
        det_type=det_type,
        det_pxsize_h=det_pxsize_h,
        det_pxsize_v=det_pxsize_v,
        det_ntum_pixels_h=det_ntum_pixels_h,
        det_num_pixels_v=det_num_pixels_v,
        det_binning=det_binning,
        det_dist=det_dist,
        det_poni1=det_poni1,
        det_poni2=det_poni2,
        det_rotx=det_rotx,
        det_roty=det_roty,
        det_rotz=det_rotz,
        det_rotation_order=det_rotation_order,
        geometry=detector_geometry,
        detector_motor_chain=detector_motor_chain,
        detector_angles=detector_angles,
        detector_transform=detector_transform,
    )

    lattice = _build_single_crystal_lattice(
        sam_space_group=sam_space_group,
        sam_a=sam_a,
        sam_b=sam_b,
        sam_c=sam_c,
        sam_alpha=sam_alpha,
        sam_beta=sam_beta,
        sam_gamma=sam_gamma,
        sam_initial_crystal_orientation=sam_initial_crystal_orientation,
        sam_rotx=sam_rotx,
        sam_roty=sam_roty,
        sam_rotz=sam_rotz,
        sam_rotation_order=sam_rotation_order,
        energy=energy,
        e_bandwidth=e_bandwidth,
        qmax=qmax,
        extra_hkls=extra_hkls,
        geometry=sample_geometry,
        sample_motor_chain=sample_motor_chain,
        sample_angles=sample_angles,
        sample_transform=sample_transform,
    )

    exp = experiment.Experiment(det, lattice, energy=energy, e_bandwidth=e_bandwidth)
    exp.summary()
    exp.calculate_diffraction_direction(qmax)
    exp.calculate_pixel_positions()
    exp.plot_2d_single_xstal_exp()

    return exp


def scan_two_motors_for_Bragg_condition(
    motor1_name,
    motor2_name,
    motor1_range,
    motor2_range,
    sam_space_group,
    sam_a=None,
    sam_b=None,
    sam_c=None,
    sam_alpha=None,
    sam_beta=None,
    sam_gamma=None,
    sam_initial_crystal_orientation=None,
    sam_rotation_order="xyz",
    energy=10e3,
    e_bandwidth=1.5,
    hkls_names=None,
    hkl_equivalent=False,
    motor_chain=None,
    geometry=None,
    fixed_angles=None,
):
    if motor1_name == motor2_name:
        raise ValueError("motor1_name and motor2_name must be different.")

    if hkls_names is None:
        raise ValueError("hkls_names must be provided.")

    if geometry is not None:
        motor_chain = geometry.sample

    if not isinstance(motor_chain, MotorChain):
        raise TypeError("motor_chain must be provided as a MotorChain, or via geometry.sample.")

    lattice = sample.LatticeStructure(
        space_group=sam_space_group,
        a=sam_a,
        b=sam_b,
        c=sam_c,
        alpha=sam_alpha,
        beta=sam_beta,
        gamma=sam_gamma,
        initial_crystal_orientation=sam_initial_crystal_orientation,
        rotation_order=sam_rotation_order,
    )

    if hkl_equivalent:
        hkls = _expand_equivalent_hkls(lattice, hkls_names)
    else:
        hkls = _normalize_hkl_triplets(hkls_names, name="hkls_names")

    raw = sample.find_Bragg_orientations_with_chain(
        hkls=hkls,
        initial_crystal_orientation=lattice.crystal_orientation,
        wavelength=utils.energy_to_wavelength(energy),
        e_bandwidth=e_bandwidth,
        motor_chain=motor_chain,
        scan_ranges={
            motor1_name: motor1_range,
            motor2_name: motor2_range,
        },
        fixed_angles=fixed_angles,
    )

    out = {}
    for key, combos in raw.items():
        out[key] = [
            (combo[motor1_name], combo[motor2_name])
            for combo in combos
        ]

    return out


def detector_rotations_collecting_Braggs_with_chain(
    det_type="manual",
    det_pxsize_h=50e-6,
    det_pxsize_v=50e-6,
    det_ntum_pixels_h=2000,
    det_num_pixels_v=2000,
    det_binning=(1, 1),
    det_dist=0.5,
    det_poni1=0,
    det_poni2=0,
    energy=10e3,
    e_bandwidth=1.5,
    sam_space_group=167,
    sam_a=None,
    sam_b=None,
    sam_c=None,
    sam_alpha=None,
    sam_beta=None,
    sam_gamma=None,
    sam_initial_crystal_orientation=None,
    sam_rotx=0,
    sam_roty=0,
    sam_rotz=0,
    sam_rotation_order="xyz",
    qmax=10,
    hkls=None,
    detector_motor_chain=None,
    scan_ranges=None,
    fixed_detector_angles=None,
):
    if not isinstance(detector_motor_chain, MotorChain):
        raise TypeError("detector_motor_chain must be a MotorChain.")
    if scan_ranges is None:
        raise ValueError("scan_ranges must be provided.")

    det = detector.Detector(
        detector_type=det_type,
        pxsize_h=det_pxsize_h,
        pxsize_v=det_pxsize_v,
        num_pixels_h=det_ntum_pixels_h,
        num_pixels_v=det_num_pixels_v,
        dist=det_dist,
        poni1=det_poni1,
        poni2=det_poni2,
        binning=det_binning,
    )
    det.calculate_lab_grid()

    lattice = sample.LatticeStructure(
        space_group=sam_space_group,
        a=sam_a,
        b=sam_b,
        c=sam_c,
        alpha=sam_alpha,
        beta=sam_beta,
        gamma=sam_gamma,
        initial_crystal_orientation=sam_initial_crystal_orientation,
        rotation_order=sam_rotation_order,
    )

    lattice.apply_rotation(rotx=sam_rotx, roty=sam_roty, rotz=sam_rotz)
    lattice.calculate_reciprocal_lattice()

    if hkls is None:
        lattice.create_possible_reflections(qmax)
    else:
        lattice.allowed_hkls = _normalize_hkl_triplets(hkls, name="hkls")

    lattice.calculate_q_hkls()
    wavelength = utils.energy_to_wavelength(energy)
    lattice.check_Bragg_condition(wavelength, e_bandwidth)

    exp = experiment.Experiment(det, lattice, energy=energy, e_bandwidth=e_bandwidth)
    exp.summary()
    exp.find_detector_rotations_with_chain(
        hkls=hkls,
        motor_chain=detector_motor_chain,
        scan_ranges=scan_ranges,
        fixed_angles=fixed_detector_angles,
    )

    return exp


def detector_rotations_collecting_Braggs_with_geometry(
    det_type="manual",
    det_pxsize_h=50e-6,
    det_pxsize_v=50e-6,
    det_ntum_pixels_h=2000,
    det_num_pixels_v=2000,
    det_binning=(1, 1),
    det_dist=0.5,
    det_poni1=0,
    det_poni2=0,
    energy=10e3,
    e_bandwidth=1.5,
    sam_space_group=167,
    sam_a=None,
    sam_b=None,
    sam_c=None,
    sam_alpha=None,
    sam_beta=None,
    sam_gamma=None,
    sam_initial_crystal_orientation=None,
    sam_rotx=0,
    sam_roty=0,
    sam_rotz=0,
    sam_rotation_order="xyz",
    qmax=10,
    hkls=None,
    geometry=None,
    scan_ranges=None,
    fixed_detector_angles=None,
):
    if not isinstance(geometry, DiffractometerGeometry):
        raise TypeError("geometry must be a DiffractometerGeometry.")
    if scan_ranges is None:
        raise ValueError("scan_ranges must be provided.")

    det = detector.Detector(
        detector_type=det_type,
        pxsize_h=det_pxsize_h,
        pxsize_v=det_pxsize_v,
        num_pixels_h=det_ntum_pixels_h,
        num_pixels_v=det_num_pixels_v,
        dist=det_dist,
        poni1=det_poni1,
        poni2=det_poni2,
        binning=det_binning,
    )
    det.calculate_lab_grid()

    lattice = sample.LatticeStructure(
        space_group=sam_space_group,
        a=sam_a,
        b=sam_b,
        c=sam_c,
        alpha=sam_alpha,
        beta=sam_beta,
        gamma=sam_gamma,
        initial_crystal_orientation=sam_initial_crystal_orientation,
        rotation_order=sam_rotation_order,
    )

    lattice.apply_rotation(rotx=sam_rotx, roty=sam_roty, rotz=sam_rotz)
    lattice.calculate_reciprocal_lattice()

    if hkls is None:
        lattice.create_possible_reflections(qmax)
    else:
        lattice.allowed_hkls = _normalize_hkl_triplets(hkls, name="hkls")

    lattice.calculate_q_hkls()
    wavelength = utils.energy_to_wavelength(energy)
    lattice.check_Bragg_condition(wavelength, e_bandwidth)

    exp = experiment.Experiment(det, lattice, energy=energy, e_bandwidth=e_bandwidth)
    exp.summary()
    exp.find_detector_rotations_with_geometry(
        hkls=hkls,
        geometry=geometry,
        scan_ranges=scan_ranges,
        fixed_detector_angles=fixed_detector_angles,
    )

    return exp


def _coerce_sample_rotation_matrix(
    sam_rotx=0.0,
    sam_roty=0.0,
    sam_rotz=0.0,
    sam_rotation_order="xyz",
    geometry=None,
    sample_geometry=None,
    sample_motor_chain=None,
    sample_angles=None,
    sample_transform=None,
):
    """
    Return the rotational part of the active sample transform as a 3x3 matrix.

    Notes
    -----
    - If a geometry or motor chain is provided, the resulting matrix is the
      total sample-arm rotation in the lab frame.
    - If nothing geometry-related is provided, this falls back to the legacy
      Euler description.
    """
    sample_geometry, _ = _resolve_geometry_pair(
        geometry=geometry,
        sample_geometry=sample_geometry,
        detector_geometry=None,
    )

    if sample_geometry is not None:
        return np.asarray(sample_geometry.sample_transform(angles=sample_angles), dtype=float)[:3, :3]

    if sample_motor_chain is not None:
        return np.asarray(sample_motor_chain.as_transform(angles=sample_angles), dtype=float)[:3, :3]

    if sample_transform is not None:
        if isinstance(sample_transform, np.ndarray):
            arr = np.asarray(sample_transform, dtype=float)
            if arr.shape == (3, 3):
                return arr
            if arr.shape == (4, 4):
                return arr[:3, :3]
            raise ValueError(f"sample_transform must have shape (3, 3) or (4, 4). Got {arr.shape}.")
        # fallback through sample helper
        return np.asarray(sample._coerce_sample_transform(sample_transform), dtype=float)[:3, :3]

    return R.from_euler(
        sam_rotation_order,
        [sam_rotx, sam_roty, sam_rotz],
        degrees=True,
    ).as_matrix()


def _extract_active_sample_chain(
    geometry=None,
    sample_geometry=None,
    sample_motor_chain=None,
):
    sample_geometry, _ = _resolve_geometry_pair(
        geometry=geometry,
        sample_geometry=sample_geometry,
        detector_geometry=None,
    )

    if sample_geometry is not None:
        return sample_geometry.sample

    if sample_motor_chain is not None:
        return sample_motor_chain

    return None


def _motor_chain_initial_vector(motor_chain, seed_angles=None):
    angle_map = motor_chain.default_angles()
    if seed_angles is not None:
        unknown = set(seed_angles) - set(motor_chain.motor_names)
        if unknown:
            raise KeyError(f"Unknown sample motors: {sorted(unknown)}")
        angle_map.update(seed_angles)

    return np.array([angle_map[name] for name in motor_chain.motor_names], dtype=float)


def _motor_chain_rotation_residual(params_deg, motor_chain, motor_names, target_rotation_matrix):
    angle_map = {name: float(val) for name, val in zip(motor_names, params_deg)}
    current_transform = motor_chain.as_transform(angles=angle_map)
    current_rotation = np.asarray(current_transform, dtype=float)[:3, :3]
    return (current_rotation - target_rotation_matrix).ravel()


def _solve_motor_chain_to_rotation(
    motor_chain,
    target_rotation_matrix,
    seed_angles=None,
    previous_solution=None,
    max_nfev=4000,
    residual_tol=1e-7,
):
    """
    Solve inverse kinematics for a sample motor chain.

    Returns
    -------
    solution : np.ndarray or None
        Wrapped motor angles in degrees if an actual solution is found.
        None if only approximate fits were found.
    motor_names : tuple[str, ...]
        Names of the motors in order.
    residual : float
        Frobenius-norm residual of the fitted rotation matrix.
    """
    motor_names = tuple(motor_chain.motor_names)
    base = _motor_chain_initial_vector(motor_chain, seed_angles=seed_angles)

    seeds = []
    if previous_solution is not None:
        prev = np.asarray(previous_solution, dtype=float)
        if prev.shape == base.shape:
            seeds.append(prev.copy())

    seeds.append(base.copy())

    for shift in (90.0, -90.0, 180.0, -180.0):
        for i in range(len(base)):
            trial = base.copy()
            trial[i] += shift
            seeds.append(trial)

    best = None
    best_cost = np.inf

    for x0 in seeds:
        res = least_squares(
            _motor_chain_rotation_residual,
            x0=x0,
            args=(motor_chain, motor_names, target_rotation_matrix),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=max_nfev,
        )
        if res.cost < best_cost:
            best = res
            best_cost = res.cost

    if best is None:
        return None, motor_names, np.inf

    solution = np.asarray(best.x, dtype=float)

    residual = np.linalg.norm(
        _motor_chain_rotation_residual(
            solution,
            motor_chain,
            motor_names,
            target_rotation_matrix,
        )
    )

    if (not np.isfinite(residual)) or (residual > residual_tol):
        return None, motor_names, residual

    solution = _wrapped_deg(solution)
    return solution, motor_names, residual


def _build_detector_explicit_poni(
    detector_type,
    dist_m,
    poni1_m,
    poni2_m,
    det_rot_deg,
    rotation_order,
    binning,
    pxsize_h,
    pxsize_v,
    num_pixels_h,
    num_pixels_v,
):
    rotx, roty, rotz = det_rot_deg

    det0 = detector.Detector(
        detector_type=detector_type,
        pxsize_h=pxsize_h,
        pxsize_v=pxsize_v,
        num_pixels_h=num_pixels_h,
        num_pixels_v=num_pixels_v,
        dist=dist_m,
        poni1=0.0,
        poni2=0.0,
        rotx=rotx,
        roty=roty,
        rotz=rotz,
        rotation_order=rotation_order,
        binning=binning,
    )

    center_poni1, center_poni2 = _geometric_center_poni(det0)
    if poni1_m is None:
        poni1_m = center_poni1
    if poni2_m is None:
        poni2_m = center_poni2

    det = detector.Detector(
        detector_type=detector_type,
        pxsize_h=pxsize_h,
        pxsize_v=pxsize_v,
        num_pixels_h=num_pixels_h,
        num_pixels_v=num_pixels_v,
        dist=dist_m,
        poni1=poni1_m,
        poni2=poni2_m,
        rotx=rotx,
        roty=roty,
        rotz=rotz,
        rotation_order=rotation_order,
        binning=binning,
    )
    det.calculate_lab_grid()

    return det, _infer_beam_center_pixel_from_poni(det)


def _q_vectors_to_detector_pixels(q_vectors, det, wavelength_m, detector_transform=None):
    """
    Map q-vectors to detector pixels using the active detector transform.

    Notes
    -----
    This works for both:
    - legacy Euler detector rotations
    - geometry/custom detector transforms
    """
    active_transform = det.get_transform() if detector_transform is None else np.asarray(detector_transform, dtype=float)

    ray_points, _ = experiment.calculate_diffraction_direction_with_transform(
        q_vectors,
        wavelength_m,
        detector_dist=det.dist,
        poni1=det.poni1,
        poni2=det.poni2,
        detector_transform=active_transform,
    )

    pix = experiment.lab_to_pixel_coordinates_with_transform(
        ray_points,
        detector_dist=det.dist,
        pxsize_h=det.pxsize_h,
        pxsize_v=det.pxsize_v,
        poni1=det.poni1,
        poni2=det.poni2,
        detector_transform=active_transform,
    )
    return np.asarray(pix, dtype=float), np.asarray(ray_points, dtype=float)

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
        qmax=10,
        extra_hkls=None
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

    if extra_hkls is not None:
        extra_hkls = np.asarray(extra_hkls, dtype=int)

        if extra_hkls.ndim == 1:
            if extra_hkls.size != 3:
                raise ValueError("extra_hkls must be a Miller triplet [h, k, l] or an array-like of shape (N, 3).")
            extra_hkls = extra_hkls.reshape(1, 3)
        elif extra_hkls.ndim != 2 or extra_hkls.shape[1] != 3:
            raise ValueError("extra_hkls must be a Miller triplet [h, k, l] or an array-like of shape (N, 3).")

        allowed_hkls = np.asarray(lattice.allowed_hkls, dtype=int)
        if allowed_hkls.ndim == 1:
            allowed_hkls = allowed_hkls.reshape(1, 3)

        combined_hkls = np.vstack((allowed_hkls, extra_hkls))

        # remove exact duplicates while preserving first appearance order
        _, unique_idx = np.unique(combined_hkls, axis=0, return_index=True)
        lattice.allowed_hkls = combined_hkls[np.sort(unique_idx)]
    else:
        lattice.allowed_hkls = np.asarray(lattice.allowed_hkls, dtype=int)

    lattice.calculate_q_hkls()
    wavelength = utils.energy_to_wavelength(energy)
    lattice.check_Bragg_condition(wavelength, e_bandwidth)

    exp = experiment.Experiment(det, lattice, energy=energy, e_bandwidth=e_bandwidth)
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
        extra_hkls=None
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

    if extra_hkls is not None:
        extra_hkls = np.asarray(extra_hkls, dtype=int)

        if extra_hkls.ndim == 1:
            if extra_hkls.size != 3:
                raise ValueError("extra_hkls must be a Miller triplet [h, k, l] or an array-like of shape (N, 3).")
            extra_hkls = extra_hkls.reshape(1, 3)
        elif extra_hkls.ndim != 2 or extra_hkls.shape[1] != 3:
            raise ValueError("extra_hkls must be a Miller triplet [h, k, l] or an array-like of shape (N, 3).")

        allowed_hkls = np.asarray(lattice.allowed_hkls, dtype=int)
        if allowed_hkls.ndim == 1:
            allowed_hkls = allowed_hkls.reshape(1, 3)

        combined_hkls = np.vstack((allowed_hkls, extra_hkls))

        # remove exact duplicates while preserving first appearance order
        _, unique_idx = np.unique(combined_hkls, axis=0, return_index=True)
        lattice.allowed_hkls = combined_hkls[np.sort(unique_idx)]
    else:
        lattice.allowed_hkls = np.asarray(lattice.allowed_hkls, dtype=int)

    lattice.calculate_q_hkls()
    wavelength = utils.energy_to_wavelength(energy)
    lattice.check_Bragg_condition(wavelength, e_bandwidth)

    exp = experiment.Experiment(det, lattice, energy=energy, e_bandwidth=e_bandwidth)
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

@dataclass
class FixedEnergyAcceptedSolutions:
    eta_deg: np.ndarray
    phi_deg: np.ndarray
    rotx_deg: np.ndarray
    roty_deg: np.ndarray
    rotz_deg: np.ndarray
    predicted_pixels: np.ndarray
    pixel_error_px: np.ndarray

    solution_mode: str = "euler"            # "euler" or "motor_chain"
    motor_names: tuple = ()
    motor_angles_deg: np.ndarray = None     # shape (Nsolutions, Nmotors)

@dataclass
class FixedEnergyTargetingResult:
    target_hkl: np.ndarray
    target_pixel: tuple
    pixel_tolerance_px: float
    q0: np.ndarray
    q0_norm: float
    wavelength_m: float
    eta_deg: np.ndarray
    q_cone: np.ndarray
    all_pixels: np.ndarray
    all_ray_points: np.ndarray
    on_detector_mask: np.ndarray
    accepted_mask: np.ndarray
    pixel_error_px: np.ndarray
    best_idx: int
    beam_center_pixel_inferred: tuple
    detector_obj: object
    solutions: object = None


def _wrapped_deg(angles_deg):
    return (np.asarray(angles_deg, dtype=float) + 180.0) % 360.0 - 180.0


def _normalize(vec, eps=1e-15):
    vec = np.asarray(vec, dtype=float)
    n = np.linalg.norm(vec)
    if n < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / n


def _rotation_aligning_a_to_b(a, b):
    a_u = _normalize(a)
    b_u = _normalize(b)

    cross = np.cross(a_u, b_u)
    s = np.linalg.norm(cross)
    c = np.clip(np.dot(a_u, b_u), -1.0, 1.0)

    if s < 1e-14:
        if c > 0:
            return np.eye(3)

        trial = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(a_u, trial)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])

        axis = _normalize(np.cross(a_u, trial))
        return R.from_rotvec(np.pi * axis).as_matrix()

    vx = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _infer_beam_center_pixel_from_poni(det):
    beam_h = det.poni1 / det.pxsize_h - 0.5
    beam_v = det.poni2 / det.pxsize_v - 0.5
    return float(beam_h), float(beam_v)


def _geometric_center_poni(det0):
    beam_h = det0.num_pixels_h / 2.0 - 0.5
    beam_v = det0.num_pixels_v / 2.0 - 0.5
    poni1 = (beam_h + 0.5) * det0.pxsize_h
    poni2 = (beam_v + 0.5) * det0.pxsize_v
    return float(poni1), float(poni2)

def _q_for_hkl_initial(lattice, hkl):
    lattice.calculate_reciprocal_lattice()
    q = sample.calculate_q_hkl(np.asarray(hkl, dtype=int), lattice.reciprocal_lattice)
    return np.asarray(q, dtype=float).reshape(3)


def _fixed_energy_q_cone(q_norm, wavelength_m, eta_deg):
    k_mag = 2.0 * np.pi / (wavelength_m * 1e10)  # Å^-1
    if q_norm > 2.0 * k_mag + 1e-12:
        raise RuntimeError(
            "This reflection cannot satisfy elastic Bragg diffraction at the fixed energy.\n"
            f"|q_hkl| = {q_norm:.9f} Å^-1\n"
            f"2k     = {2.0 * k_mag:.9f} Å^-1"
        )

    qx = -(q_norm ** 2) / (2.0 * k_mag)
    q_perp_sq = max(q_norm ** 2 - qx ** 2, 0.0)
    q_perp = np.sqrt(q_perp_sq)

    eta_rad = np.deg2rad(eta_deg)
    qy = q_perp * np.cos(eta_rad)
    qz = q_perp * np.sin(eta_rad)
    return np.column_stack([np.full_like(qy, qx), qy, qz])

def _solve_orientations_for_q_targets(
    q0,
    q_targets,
    q_target_pixels,
    q_target_errors_px,
    rotation_order,
    phi_samples,
    display_wrapped_angles,
    eta_deg_kept,
    baseline_rotation_matrix=None,
    motor_chain=None,
    motor_seed_angles=None,
):
    phi_grid_deg = np.linspace(0.0, 360.0, int(phi_samples), endpoint=True)

    eta_out = []
    phi_out = []
    rx_out = []
    ry_out = []
    rz_out = []
    pix_out = []
    err_out = []

    motor_rows = []

    if baseline_rotation_matrix is None:
        baseline_rotation_matrix = np.eye(3)
    else:
        baseline_rotation_matrix = np.asarray(baseline_rotation_matrix, dtype=float)

    use_motor_chain = motor_chain is not None
    previous_solution = None
    motor_names_out = ()

    for eta_one, q_target, pix_one, err_one in zip(
        eta_deg_kept, q_targets, q_target_pixels, q_target_errors_px
    ):
        r_align = _rotation_aligning_a_to_b(q0, q_target)
        axis = _normalize(q_target)

        for phi in phi_grid_deg:
            r_phi = R.from_rotvec(np.deg2rad(phi) * axis).as_matrix()
            r_incremental = r_phi @ r_align
            r_total = r_incremental @ baseline_rotation_matrix

            if use_motor_chain:
                motor_solution, motor_names, residual = _solve_motor_chain_to_rotation(
                    motor_chain=motor_chain,
                    target_rotation_matrix=r_total,
                    seed_angles=motor_seed_angles,
                    previous_solution=previous_solution,
                )

                if motor_solution is None:
                    continue

                previous_solution = motor_solution
                motor_names_out = motor_names
                motor_rows.append(motor_solution.copy())

                # Compatibility fields for legacy plotting code
                eul = np.full(3, np.nan, dtype=float)
                ncopy = min(3, len(motor_solution))
                eul[:ncopy] = motor_solution[:ncopy]

            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    eul = R.from_matrix(r_total).as_euler(rotation_order, degrees=True)

                if display_wrapped_angles:
                    eul = _wrapped_deg(eul)

            eta_out.append(float(eta_one))
            phi_out.append(float(phi))
            rx_out.append(float(eul[0]))
            ry_out.append(float(eul[1]))
            rz_out.append(float(eul[2]))
            pix_out.append(np.asarray(pix_one, dtype=float))
            err_out.append(float(err_one))

    if len(pix_out) == 0:
        return None

    motor_angles_deg = None
    if use_motor_chain:
        motor_angles_deg = np.asarray(motor_rows, dtype=float)

    return FixedEnergyAcceptedSolutions(
        eta_deg=np.asarray(eta_out, dtype=float),
        phi_deg=np.asarray(phi_out, dtype=float),
        rotx_deg=np.asarray(rx_out, dtype=float),
        roty_deg=np.asarray(ry_out, dtype=float),
        rotz_deg=np.asarray(rz_out, dtype=float),
        predicted_pixels=np.asarray(pix_out, dtype=float),
        pixel_error_px=np.asarray(err_out, dtype=float),
        solution_mode="motor_chain" if use_motor_chain else "euler",
        motor_names=tuple(motor_names_out),
        motor_angles_deg=motor_angles_deg,
    )

def target_hkl_near_pixel_fixed_energy(
    det_type="manual",
    det_pxsize_h=50e-6,
    det_pxsize_v=50e-6,
    det_ntum_pixels_h=2000,
    det_num_pixels_v=2000,
    det_binning=(1, 1),
    det_dist=0.5,
    det_poni1=0.0,
    det_poni2=0.0,
    det_rotx=0.0,
    det_roty=0.0,
    det_rotz=0.0,
    det_rotation_order="xyz",
    energy=10e3,
    sam_space_group=167,
    sam_a=None,
    sam_b=None,
    sam_c=None,
    sam_alpha=None,
    sam_beta=None,
    sam_gamma=None,
    sam_initial_crystal_orientation=None,
    sam_rotx=0.0,
    sam_roty=0.0,
    sam_rotz=0.0,
    sam_rotation_order="xyz",
    target_hkl=None,
    target_pixel=(900.0, 900.0),
    pixel_tolerance_px=20.0,
    eta_samples=1441,
    phi_samples=361,
    display_wrapped_angles=True,
    do_detector_plot=True,
    do_2d_plot=True,
    do_3d_plot=False,
    phi_colormap="hsv",
    scatter_size=8,
    geometry=None,
    sample_geometry=None,
    detector_geometry=None,
    sample_motor_chain=None,
    detector_motor_chain=None,
    sample_angles=None,
    detector_angles=None,
    sample_transform=None,
    detector_transform=None,
):
    """
    Fixed-energy inverse single-crystal targeting.

    At fixed X-ray energy, build the elastic diffraction cone for one selected
    reflection and keep the cone points whose detector intersections fall close
    to a target detector pixel. For every accepted cone point, expand the
    one-parameter family of sample orientations obtained by spinning around the
    accepted q target.

    Notes
    -----
    The input sample rotations sam_rotx/sam_roty/sam_rotz are treated as a
    baseline orientation. The returned motor angles are the total absolute
    sample rotations with respect to the initial crystal orientation.
    """
    if target_hkl is None:
        raise ValueError("target_hkl must be provided as a Miller triplet [h, k, l].")

    target_hkl = np.asarray(target_hkl, dtype=int)
    if target_hkl.ndim == 2 and target_hkl.shape == (1, 3):
        target_hkl = target_hkl.reshape(3)
    if target_hkl.shape != (3,):
        raise ValueError("target_hkl must be a Miller triplet [h, k, l].")

    target_pixel = tuple(float(x) for x in target_pixel)
    if len(target_pixel) != 2:
        raise ValueError("target_pixel must be a pair (d_h, d_v).")

    sample_geometry, detector_geometry = _resolve_geometry_pair(
        geometry=geometry,
        sample_geometry=sample_geometry,
        detector_geometry=detector_geometry,
    )

    active_sample_chain = _extract_active_sample_chain(
        geometry=geometry,
        sample_geometry=sample_geometry,
        sample_motor_chain=sample_motor_chain,
    )

    det = _build_single_crystal_detector(
        det_type=det_type,
        det_pxsize_h=det_pxsize_h,
        det_pxsize_v=det_pxsize_v,
        det_ntum_pixels_h=det_ntum_pixels_h,
        det_num_pixels_v=det_num_pixels_v,
        det_binning=det_binning,
        det_dist=det_dist,
        det_poni1=det_poni1,
        det_poni2=det_poni2,
        det_rotx=det_rotx,
        det_roty=det_roty,
        det_rotz=det_rotz,
        det_rotation_order=det_rotation_order,
        geometry=detector_geometry,
        detector_motor_chain=detector_motor_chain,
        detector_angles=detector_angles,
        detector_transform=detector_transform,
    )

    beam_center_pixel_inferred = _infer_beam_center_pixel_from_poni(det)

    lattice0 = _build_single_crystal_lattice(
        sam_space_group=sam_space_group,
        sam_a=sam_a,
        sam_b=sam_b,
        sam_c=sam_c,
        sam_alpha=sam_alpha,
        sam_beta=sam_beta,
        sam_gamma=sam_gamma,
        sam_initial_crystal_orientation=sam_initial_crystal_orientation,
        sam_rotation_order=sam_rotation_order,
    )

    q_initial = _q_for_hkl_initial(lattice0, target_hkl)

    baseline_rotation_matrix = _coerce_sample_rotation_matrix(
        sam_rotx=sam_rotx,
        sam_roty=sam_roty,
        sam_rotz=sam_rotz,
        sam_rotation_order=sam_rotation_order,
        geometry=sample_geometry,
        sample_geometry=sample_geometry,
        sample_motor_chain=sample_motor_chain,
        sample_angles=sample_angles,
        sample_transform=sample_transform,
    )

    q0 = q_initial @ baseline_rotation_matrix.T
    q0_norm = float(np.linalg.norm(q0))

    wavelength_m = utils.energy_to_wavelength(energy)
    eta_deg = np.linspace(0.0, 360.0, int(eta_samples), endpoint=True)
    q_cone = _fixed_energy_q_cone(q0_norm, wavelength_m, eta_deg)

    all_pixels, all_ray_points = _q_vectors_to_detector_pixels(
        q_cone,
        det,
        wavelength_m,
        detector_transform=det.get_transform(),
    )

    pixel_error_px = np.linalg.norm(
        all_pixels - np.asarray(target_pixel, dtype=float),
        axis=1,
    )

    finite_mask = np.isfinite(all_pixels).all(axis=1) & np.isfinite(pixel_error_px)
    if not np.any(finite_mask):
        raise RuntimeError("No finite detector intersections were produced for the fixed-energy cone.")

    on_detector_mask = (
        finite_mask
        & (all_pixels[:, 0] >= 0.0)
        & (all_pixels[:, 0] < det.num_pixels_h)
        & (all_pixels[:, 1] >= 0.0)
        & (all_pixels[:, 1] < det.num_pixels_v)
    )

    accepted_mask = on_detector_mask & (pixel_error_px <= float(pixel_tolerance_px))
    best_idx = int(np.argmin(np.where(finite_mask, pixel_error_px, np.inf)))

    solutions = None
    if np.any(accepted_mask):
        q_kept = q_cone[accepted_mask]
        pixels_kept = all_pixels[accepted_mask]
        errors_kept = pixel_error_px[accepted_mask]
        eta_kept = eta_deg[accepted_mask]

        solutions = _solve_orientations_for_q_targets(
            q0=q0,
            q_targets=q_kept,
            q_target_pixels=pixels_kept,
            q_target_errors_px=errors_kept,
            rotation_order=sam_rotation_order,
            phi_samples=phi_samples,
            display_wrapped_angles=display_wrapped_angles,
            eta_deg_kept=eta_kept,
            baseline_rotation_matrix=baseline_rotation_matrix,
            motor_chain=active_sample_chain,
            motor_seed_angles=sample_angles,
        )

    result = FixedEnergyTargetingResult(
        target_hkl=np.asarray(target_hkl, dtype=int),
        target_pixel=target_pixel,
        pixel_tolerance_px=float(pixel_tolerance_px),
        q0=q0,
        q0_norm=q0_norm,
        wavelength_m=float(wavelength_m),
        eta_deg=np.asarray(eta_deg, dtype=float),
        q_cone=np.asarray(q_cone, dtype=float),
        all_pixels=np.asarray(all_pixels, dtype=float),
        all_ray_points=np.asarray(all_ray_points, dtype=float),
        on_detector_mask=np.asarray(on_detector_mask, dtype=bool),
        accepted_mask=np.asarray(accepted_mask, dtype=bool),
        pixel_error_px=np.asarray(pixel_error_px, dtype=float),
        best_idx=best_idx,
        beam_center_pixel_inferred=beam_center_pixel_inferred,
        detector_obj=det,
        solutions=solutions,
    )

    wavelength_A = wavelength_m * 1e10
    hkl_str = f"[{int(target_hkl[0])}, {int(target_hkl[1])}, {int(target_hkl[2])}]"
    target_pixel_str = f"({target_pixel[0]:.1f}, {target_pixel[1]:.1f})"

    common_title = (
        f"Detector: {det.detector_type}\n"
        f"offsets [m]:dist = {det.dist:.2e}, poni1 = {det.poni1:.1e}, poni2 = {det.poni2:.1e}\n"
        f"det. rotations [°]: Rotx={det.rotx:.2f}, Roty={det.roty:.2f}, Rotz={det.rotz:.2f}\n"
        f"E = {energy:.1f} eV, $\\lambda$ = {wavelength_A:.2f} Å\n"
        f"hkl = {hkl_str.replace('[', '(').replace(']', ')')}\n"
        f"target pixel = {target_pixel_str}, pixel tolerance = {pixel_tolerance_px:.2f} px"
    )

    detector_title = common_title
    motor_title = common_title
    motor_title_3d = common_title

    if do_detector_plot:
        plot.plot_fixed_energy_detector_hits(
            result.all_pixels,
            result.on_detector_mask,
            result.accepted_mask,
            result.target_pixel,
            result.pixel_tolerance_px,
            title=detector_title,
        )

    if solutions is not None and do_2d_plot:
        plot.plot_fixed_energy_motor_projections(
            solutions,
            phi_colormap=phi_colormap,
            scatter_size=scatter_size,
            title=motor_title,
        )

    if solutions is not None and do_3d_plot:
        plot.plot_fixed_energy_motor_family_3d(
            solutions,
            phi_colormap=phi_colormap,
            scatter_size=scatter_size,
            title=motor_title_3d,
        )

    return result
