"""Weighted multi-pose orientation refinement with explicit rank diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .. import sample, utils
from ..parallel import check_cancelled
from .candidates import OrientationCandidate, generate_candidates
from .forward_model import build_detector, observed_scattering_vector, predict_bragg_pixel, sample_rotation_at_pose
from .observations import OrientationExperiment, resolve_pose, validate_experiment


@dataclass
class FittedCandidate:
    """A refined orientation with assignments and per-measurement diagnostics."""
    rotation: np.ndarray
    reference_matrix: np.ndarray
    assignments: dict[str, tuple[int, int, int]]
    weighted_error: float
    rms_pixel: float
    rank: int
    angular_sigma_deg: list[float] | None
    peak_diagnostics: list[dict]
    pose_matrices: dict[str, np.ndarray]


@dataclass
class OrientationFitResult:
    """Ranked solutions and the exact observation project used to produce them."""
    candidates: list[FittedCandidate]
    project: dict
    message: str


def _prepare(experiment, assignments):
    lattice = sample.LatticeStructure(**experiment.lattice)
    reciprocal = sample.cal_reciprocal_lattice(lattice.initial_crystal_orientation)
    rows = []
    for peak in experiment.peaks:
        if not peak.enabled:
            continue
        pose = resolve_pose(experiment, peak.pose_id)
        detector = build_detector(experiment, pose)
        q_lab = observed_scattering_vector(experiment, pose, peak, detector)
        hkl = np.asarray(assignments[peak.id], dtype=float)
        q_crystal = hkl @ reciprocal
        motor = sample_rotation_at_pose(experiment, pose)
        k = 2 * np.pi / (utils.energy_to_wavelength(pose.energy_eV) * 1e10)
        pixel_error = k * np.hypot(
            np.hypot(peak.sigma_h, experiment.detector_sigma_px) * detector.pxsize_h,
            np.hypot(peak.sigma_v, experiment.detector_sigma_px) * detector.pxsize_v,
        ) / detector.dist
        energy_error = np.linalg.norm(q_lab) * experiment.energy_sigma_eV / pose.energy_eV
        motor_error = np.linalg.norm(q_crystal) * np.deg2rad(experiment.motor_sigma_deg)
        lattice_error = np.linalg.norm(q_crystal) * experiment.lattice_sigma_fraction
        sigma = max(float(np.hypot(np.hypot(pixel_error, energy_error),
                                   np.hypot(motor_error, lattice_error))), 1e-5)
        rows.append((peak, pose, detector, q_lab, q_crystal, motor, sigma))
    return lattice, rows


def _residuals(rotvec, rows):
    check_cancelled()
    rotation = Rotation.from_rotvec(rotvec).as_matrix()
    return np.concatenate([(q_crystal @ rotation.T @ motor.T - q_lab) / sigma
                           for _, _, _, q_lab, q_crystal, motor, sigma in rows])


def refine_candidate(experiment: OrientationExperiment, candidate: OrientationCandidate) -> FittedCandidate:
    """Fit one discrete assignment and quantify its supported rotation axes."""
    validate_experiment(experiment)
    lattice, rows = _prepare(experiment, candidate.assignments)
    start = Rotation.from_matrix(candidate.rotation).as_rotvec()
    fit = least_squares(_residuals, start, args=(rows,), loss='soft_l1', max_nfev=300)
    rotation = Rotation.from_rotvec(fit.x).as_matrix()
    data_residual = _residuals(fit.x, rows)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    rank = int(np.sum(singular > max(singular[0] * 1e-5, 1e-8))) if len(singular) else 0
    angular_sigma = None
    if rank == 3 and len(data_residual) > 3:
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac)
        covariance *= max(1.0, float(data_residual @ data_residual) / (len(data_residual) - 3))
        angular_sigma = np.rad2deg(np.sqrt(np.clip(np.diag(covariance), 0, np.inf))).tolist()
    direct = utils.apply_rotation_matrix(lattice.initial_crystal_orientation, rotation)
    pose_matrices = {pose.id: utils.apply_rotation_matrix(
        direct, sample_rotation_at_pose(experiment, resolve_pose(experiment, pose.id)))
        for pose in experiment.poses}
    peak_diagnostics = []
    pixel_errors = []
    for peak, pose, detector, q_lab, q_crystal, motor, sigma in rows:
        predicted = predict_bragg_pixel(experiment, pose, rotation, candidate.assignments[peak.id], detector)
        error_px = float(np.hypot(predicted.pixel_h - peak.pixel_h, predicted.pixel_v - peak.pixel_v))
        pixel_errors.append(error_px)
        q_error = float(np.linalg.norm(predicted.q_lab - q_lab))
        peak_diagnostics.append(dict(
            peak_id=peak.id, pose_id=pose.id, hkl=list(candidate.assignments[peak.id]),
            observed_pixel=[peak.pixel_h, peak.pixel_v],
            predicted_pixel=[predicted.pixel_h, predicted.pixel_v],
            pixel_error=error_px, q_error=q_error,
            normalized_error=q_error / sigma, ewald_error=predicted.ewald_error,
            on_detector=predicted.on_detector,
        ))
    return FittedCandidate(rotation, direct, dict(candidate.assignments),
                           float(data_residual @ data_residual),
                           float(np.sqrt(np.mean(np.square(pixel_errors)))), rank,
                           angular_sigma, peak_diagnostics, pose_matrices)


def fit_orientation(experiment: OrientationExperiment, initial_guess=None, *,
                    max_candidates=12) -> OrientationFitResult:
    """Fit ranked orientation alternatives from all enabled measured peaks."""
    validate_experiment(experiment)
    seeds = generate_candidates(experiment, initial_guess, max_candidates=max_candidates * 3)
    fitted = []
    for index, seed in enumerate(seeds, 1):
        check_cancelled()
        print(f'Orientation candidate {index}/{len(seeds)}', flush=True)
        fitted.append(refine_candidate(experiment, seed))
    fitted.sort(key=lambda item: (item.weighted_error, item.rms_pixel))
    distinct = []
    for candidate in fitted:
        if any(np.linalg.norm(Rotation.from_matrix(candidate.rotation @ previous.rotation.T).as_rotvec())
               < np.deg2rad(0.2) and candidate.assignments == previous.assignments for previous in distinct):
            continue
        distinct.append(candidate)
        if len(distinct) >= max_candidates:
            break
    if not distinct:
        raise ValueError('No viable orientation candidates were found.')
    message = ('Orientation is underdetermined by these observations; add a nonparallel reflection.'
               if distinct[0].rank < 3 else
               'Candidate orientations are ranked by weighted reciprocal-vector residual.')
    return OrientationFitResult(distinct, experiment.to_dict(), message)
