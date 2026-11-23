"""Observed and predicted Bragg geometry in XRDpy's coordinate conventions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import experiment as beam_geometry, sample, utils
from ..detector import Detector
from ..geometry import MotorChain
from .observations import BraggObservation, OrientationExperiment, ResolvedPose, resolve_pose


@dataclass(frozen=True)
class PredictedPeak:
    pixel_h: float
    pixel_v: float
    q_lab: np.ndarray
    ewald_error: float
    on_detector: bool


def build_detector(experiment: OrientationExperiment, pose: ResolvedPose) -> Detector:
    """Recreate the effective detector from the project's saved calibration."""
    setup = next(item for item in experiment.detectors if item.id == pose.detector_setup_id)
    detector = Detector(**setup.parameters)
    if setup.motor_chain is not None:
        detector.set_motor_chain(MotorChain.from_dict(setup.motor_chain), angles=pose.detector_angles)
    return detector


def observed_scattering_vector(experiment: OrientationExperiment, pose: ResolvedPose,
                               observation: BraggObservation, detector=None) -> np.ndarray:
    """Convert a measured horizontal/vertical pixel and energy into lab q, Å⁻¹."""
    detector = detector or build_detector(experiment, pose)
    if not (0 <= observation.pixel_h < detector.num_pixels_h and
            0 <= observation.pixel_v < detector.num_pixels_v):
        raise ValueError(f'Peak {observation.id}: pixel is outside detector '
                         f'{pose.detector_setup_id}. Check the selected pose or pixel coordinates.')
    point_native = np.array([[
        detector.dist,
        detector.poni2 - (observation.pixel_h + 0.5) * detector.pxsize_h,
        -detector.poni1 + (observation.pixel_v + 0.5) * detector.pxsize_v,
    ]], dtype=float)
    point_lab = utils.apply_transform(point_native, detector.get_transform())[0]
    distance = np.linalg.norm(point_lab)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError(f'Peak {observation.id}: detector geometry gives an invalid ray.')
    outgoing = point_lab / distance
    k = 2 * np.pi / (utils.energy_to_wavelength(pose.energy_eV) * 1e10)
    return k * (outgoing - np.array([1.0, 0.0, 0.0]))


def reference_direct_lattice(experiment: OrientationExperiment, orientation: np.ndarray) -> np.ndarray:
    """Apply a proper rotation to the cell's canonical direct-lattice vectors."""
    lattice = sample.LatticeStructure(**experiment.lattice)
    rotation = np.asarray(orientation, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError('Orientation must be a finite 3×3 rotation matrix.')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError('Orientation must be a proper rotation matrix.')
    return utils.apply_rotation_matrix(lattice.initial_crystal_orientation, rotation)


def zero_motor_direct_lattice(experiment: OrientationExperiment,
                              reference_matrix: np.ndarray) -> np.ndarray:
    """Undo reference-pose sample motors to recover the mounting orientation."""
    reference_id = experiment.reference_pose_id or experiment.poses[0].id
    reference = resolve_pose(experiment, reference_id)
    reference_rotation = MotorChain.from_dict(experiment.sample_chain).as_transform(
        angles=reference.motor_angles)[:3, :3]
    # Row-vector convention: reference = mounting @ reference_rotation.T.
    return np.asarray(reference_matrix, dtype=float) @ reference_rotation


def sample_rotation_at_pose(experiment: OrientationExperiment, pose: ResolvedPose) -> np.ndarray:
    """Rotation from the declared reference pose to this pose, in lab axes."""
    chain = MotorChain.from_dict(experiment.sample_chain)
    reference_id = experiment.reference_pose_id or experiment.poses[0].id
    reference = resolve_pose(experiment, reference_id)
    current = chain.as_transform(angles=pose.motor_angles)[:3, :3]
    initial = chain.as_transform(angles=reference.motor_angles)[:3, :3]
    return current @ initial.T


def orientation_at_pose(experiment: OrientationExperiment, pose: ResolvedPose,
                        orientation: np.ndarray) -> np.ndarray:
    """Return direct-lattice row vectors after the recorded sample motor chain."""
    return sample.apply_orientation_transform(reference_direct_lattice(experiment, orientation),
                                              sample_rotation_at_pose(experiment, pose))


def predict_bragg_pixel(experiment: OrientationExperiment, pose: ResolvedPose,
                        orientation: np.ndarray, hkl, detector=None) -> PredictedPeak:
    """Predict pixel and Ewald mismatch using the current detector projection."""
    detector = detector or build_detector(experiment, pose)
    direct = orientation_at_pose(experiment, pose, orientation)
    q = sample.calculate_q_hkl(np.asarray(hkl, dtype=float), sample.cal_reciprocal_lattice(direct))
    wavelength = utils.energy_to_wavelength(pose.energy_eV)
    k = 2 * np.pi / (wavelength * 1e10)
    outgoing = q + np.array([k, 0.0, 0.0])
    ewald_error = float(np.linalg.norm(outgoing) - k)
    points, signs = beam_geometry.calculate_diffraction_direction_with_transform(
        q[None, :], wavelength, detector.dist, detector.poni1, detector.poni2,
        detector.get_transform(),
    )
    pixels = beam_geometry.lab_to_pixel_coordinates_with_transform(
        points, detector.dist, detector.pxsize_h, detector.pxsize_v,
        detector.poni1, detector.poni2, detector.get_transform(),
    )
    h, v = map(float, pixels[0])
    inside = bool(signs[0] > 0 and 0 <= h < detector.num_pixels_h and 0 <= v < detector.num_pixels_v)
    return PredictedPeak(h, v, np.asarray(q), ewald_error, inside)
