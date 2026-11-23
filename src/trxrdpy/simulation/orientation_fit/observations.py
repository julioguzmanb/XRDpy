"""Experimental poses and peaks for single-crystal orientation fitting."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any

import numpy as np

from ..geometry import MotorChain


@dataclass
class DetectorSetup:
    """Reusable detector calibration and default detector motor readings."""
    id: str
    parameters: dict[str, Any]
    motor_angles: dict[str, float] = field(default_factory=dict)
    motor_chain: dict[str, Any] | None = None
    source: str = ''


@dataclass
class ExperimentalPose:
    """One condition set; sample values are absolute readings or parent deltas."""
    id: str
    parent_id: str | None = None
    motor_mode: str = 'absolute'
    motor_values: dict[str, float] = field(default_factory=dict)
    energy_eV: float | None = None
    detector_setup_id: str | None = None
    detector_angles: dict[str, float] = field(default_factory=dict)


@dataclass
class BraggObservation:
    """One observed pixel with exact, family, or unknown HKL identity."""
    id: str
    pose_id: str
    pixel_h: float
    pixel_v: float
    hkl: tuple[int, int, int] | None = None
    assignment_mode: str = 'exact'
    sigma_h: float = 1.0
    sigma_v: float = 1.0
    enabled: bool = True


@dataclass
class OrientationExperiment:
    """Editable project containing setup, linked poses, and their peaks."""
    lattice: dict[str, Any]
    sample_chain: dict[str, Any]
    detectors: list[DetectorSetup]
    poses: list[ExperimentalPose]
    peaks: list[BraggObservation]
    default_energy_eV: float
    reference_pose_id: str | None = None
    initial_guess: list[list[float]] | None = None
    energy_sigma_eV: float = 0.0
    motor_sigma_deg: float = 0.0
    lattice_sigma_fraction: float = 0.0
    detector_sigma_px: float = 0.0
    default_peak_sigma_h: float = 1.0
    default_peak_sigma_v: float = 1.0
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrientationExperiment:
        if data.get('version') != 1:
            raise ValueError('Unsupported orientation experiment version.')
        return cls(
            lattice=dict(data['lattice']), sample_chain=dict(data['sample_chain']),
            detectors=[DetectorSetup(**item) for item in data['detectors']],
            poses=[ExperimentalPose(**item) for item in data['poses']],
            peaks=[BraggObservation(**item) for item in data['peaks']],
            default_energy_eV=float(data['default_energy_eV']),
            reference_pose_id=data.get('reference_pose_id'),
            initial_guess=data.get('initial_guess'),
            energy_sigma_eV=float(data.get('energy_sigma_eV', 0)),
            motor_sigma_deg=float(data.get('motor_sigma_deg', 0)),
            lattice_sigma_fraction=float(data.get('lattice_sigma_fraction', 0)),
            detector_sigma_px=float(data.get('detector_sigma_px', 0)),
            default_peak_sigma_h=float(data.get('default_peak_sigma_h', 1)),
            default_peak_sigma_v=float(data.get('default_peak_sigma_v', 1)),
        )


@dataclass(frozen=True)
class ResolvedPose:
    """Effective experimental conditions used by the forward model."""
    id: str
    motor_angles: dict[str, float]
    energy_eV: float
    detector_setup_id: str
    detector_angles: dict[str, float]


def _finite(value, label, *, positive=False, nonnegative=False):
    value = float(value)
    if not isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f'{label} must be finite and {"positive" if positive else "nonnegative" if nonnegative else "valid"}.')
    return value


def resolve_pose(experiment: OrientationExperiment, pose_id: str) -> ResolvedPose:
    """Resolve inherited values and relative motor changes without Euler shortcuts."""
    chain = MotorChain.from_dict(experiment.sample_chain)
    by_id = {pose.id: pose for pose in experiment.poses}
    detectors = {item.id: item for item in experiment.detectors}
    if pose_id not in by_id:
        raise KeyError(f'Unknown pose {pose_id!r}.')
    resolved = {}
    visiting = set()

    def visit(key):
        if key in resolved:
            return resolved[key]
        if key in visiting:
            raise ValueError('Pose parent references contain a cycle.')
        if key not in by_id:
            raise ValueError(f'Unknown parent pose {key!r}.')
        visiting.add(key)
        pose = by_id[key]
        parent = visit(pose.parent_id) if pose.parent_id else None
        if pose.motor_mode not in ('absolute', 'relative'):
            raise ValueError(f'Pose {key}: motor mode must be absolute or relative.')
        if pose.motor_mode == 'relative' and parent is None:
            raise ValueError(f'Pose {key}: relative motors require a parent.')
        unknown = set(pose.motor_values) - set(chain.motor_names)
        if unknown:
            raise ValueError(f'Pose {key}: unknown sample motors {sorted(unknown)}.')
        motors = dict(parent.motor_angles) if parent else chain.default_angles()
        if pose.motor_mode == 'absolute':
            motors.update(pose.motor_values)
        else:
            for name, delta in pose.motor_values.items():
                motors[name] += float(delta)
        motors = {name: _finite(value, f'Pose {key} motor {name}') for name, value in motors.items()}
        energy = _finite(pose.energy_eV if pose.energy_eV is not None else
                         parent.energy_eV if parent else experiment.default_energy_eV,
                         f'Pose {key} energy', positive=True)
        detector_id = pose.detector_setup_id or (parent.detector_setup_id if parent else experiment.detectors[0].id)
        if detector_id not in detectors:
            raise ValueError(f'Pose {key}: unknown detector setup {detector_id!r}.')
        angles = {**detectors[detector_id].motor_angles,
                  **(parent.detector_angles if parent and detector_id == parent.detector_setup_id else {}),
                  **pose.detector_angles}
        detector_chain = detectors[detector_id].motor_chain
        allowed_detector_motors = (set(MotorChain.from_dict(detector_chain).motor_names)
                                   if detector_chain is not None else set())
        unknown_detector = set(angles) - allowed_detector_motors
        if unknown_detector:
            raise ValueError(f'Pose {key}: unknown detector motors {sorted(unknown_detector)}.')
        angles = {name: _finite(value, f'Pose {key} detector motor {name}') for name, value in angles.items()}
        visiting.remove(key)
        resolved[key] = ResolvedPose(key, motors, energy, detector_id, angles)
        return resolved[key]

    return visit(pose_id)


def validate_experiment(experiment: OrientationExperiment) -> None:
    """Reject invalid references, measurements, motor states, and uncertainties."""
    if not experiment.detectors or not experiment.poses:
        raise ValueError('At least one detector setup and one pose are required.')
    for label, ids in (('detector', [x.id for x in experiment.detectors]),
                       ('pose', [x.id for x in experiment.poses]),
                       ('peak', [x.id for x in experiment.peaks])):
        if any(not x for x in ids) or len(ids) != len(set(ids)):
            raise ValueError(f'{label} IDs must be nonempty and unique.')
    _finite(experiment.default_energy_eV, 'Default energy', positive=True)
    _finite(experiment.energy_sigma_eV, 'Energy uncertainty', nonnegative=True)
    _finite(experiment.motor_sigma_deg, 'Motor uncertainty', nonnegative=True)
    _finite(experiment.lattice_sigma_fraction, 'Relative lattice uncertainty', nonnegative=True)
    _finite(experiment.detector_sigma_px, 'Detector calibration uncertainty', nonnegative=True)
    _finite(experiment.default_peak_sigma_h, 'Default peak horizontal uncertainty', positive=True)
    _finite(experiment.default_peak_sigma_v, 'Default peak vertical uncertainty', positive=True)
    if not MotorChain.from_dict(experiment.sample_chain).motors:
        raise ValueError('The sample goniometer must contain a motor.')
    for pose in experiment.poses:
        resolve_pose(experiment, pose.id)
    pose_ids = {pose.id for pose in experiment.poses}
    if experiment.reference_pose_id is not None and experiment.reference_pose_id not in pose_ids:
        raise ValueError('Unknown reference pose.')
    for peak in experiment.peaks:
        if peak.pose_id not in pose_ids:
            raise ValueError(f'Peak {peak.id}: unknown pose.')
        if peak.assignment_mode not in ('exact', 'family', 'unknown'):
            raise ValueError(f'Peak {peak.id}: invalid HKL assignment mode.')
        if peak.assignment_mode != 'unknown' and peak.hkl is None:
            raise ValueError(f'Peak {peak.id}: HKL is required.')
        if peak.hkl is not None and (len(peak.hkl) != 3 or not all(float(x).is_integer() for x in peak.hkl)
                                     or not any(peak.hkl)):
            raise ValueError(f'Peak {peak.id}: HKL must be a nonzero integer triplet.')
        _finite(peak.pixel_h, f'Peak {peak.id} horizontal pixel')
        _finite(peak.pixel_v, f'Peak {peak.id} vertical pixel')
        _finite(peak.sigma_h, f'Peak {peak.id} horizontal uncertainty', positive=True)
        _finite(peak.sigma_v, f'Peak {peak.id} vertical uncertainty', positive=True)
    if not any(peak.enabled for peak in experiment.peaks):
        raise ValueError('At least one enabled peak is required.')
    if experiment.initial_guess is not None:
        matrix = np.asarray(experiment.initial_guess, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10:
            raise ValueError('Initial matrix must be a nonsingular 3×3 array.')
