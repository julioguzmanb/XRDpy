"""Normalize all supported goniometers for orientation fitting."""
from __future__ import annotations

from dataclasses import dataclass

from ..diffractometers import make_euler_geometry, motor_chain_to_dict
from ..geometry import DiffractometerGeometry, MotorChain
from .observations import DetectorSetup, ExperimentalPose, OrientationExperiment


@dataclass(frozen=True)
class OrientationSetup:
    """Numerical starting conditions captured from the Single Crystal controls."""
    lattice: dict
    detector: DetectorSetup
    sample_chain: MotorChain
    energy_eV: float
    initial_guess: list[list[float]] | None = None
    sample_angles: dict[str, float] | None = None


def normalize_goniometer(geometry=None, *, euler_order='xyz') -> MotorChain:
    """Return a sample MotorChain for Eulerian, predefined, or custom modes."""
    if geometry is None:
        return make_euler_geometry(sample_rotation_order=euler_order).sample
    if isinstance(geometry, DiffractometerGeometry):
        return geometry.sample
    if isinstance(geometry, MotorChain):
        return geometry
    raise TypeError('Geometry must be a DiffractometerGeometry, MotorChain, or None.')


def snapshot_setup(crystal, detector, goniometer, beam, *, initial_guess=None,
                   sample_angles=None) -> OrientationSetup:
    """Copy the effective setup so later GUI edits cannot change a running fit."""
    if not isinstance(detector, DetectorSetup):
        raise TypeError('detector must be a DetectorSetup.')
    chain = normalize_goniometer(goniometer)
    return OrientationSetup(dict(crystal), detector, chain, float(beam['energy_eV']),
                            initial_guess, dict(sample_angles or chain.default_angles()))


def new_experiment(setup: OrientationSetup) -> OrientationExperiment:
    """Start an editable project with one reference pose and no peaks."""
    return OrientationExperiment(
        lattice=dict(setup.lattice), sample_chain=motor_chain_to_dict(setup.sample_chain),
        detectors=[setup.detector],
        poses=[ExperimentalPose('pose-1', motor_values=dict(setup.sample_angles or {}),
                                energy_eV=setup.energy_eV, detector_setup_id=setup.detector.id)],
        peaks=[], default_energy_eV=setup.energy_eV, reference_pose_id='pose-1',
        initial_guess=setup.initial_guess,
    )
