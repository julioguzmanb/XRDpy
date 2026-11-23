"""Fit single-crystal orientation from measured Bragg pixels and motor poses.

The experiment model, diffraction geometry, candidate search, refinement, and
file persistence are separate so fitting can also be used without Qt.
"""

from .observations import (BraggObservation, DetectorSetup, ExperimentalPose,
                           OrientationExperiment, resolve_pose, validate_experiment)
from .refine import FittedCandidate, OrientationFitResult, fit_orientation
from .setup import OrientationSetup, new_experiment, snapshot_setup

__all__ = [
    'BraggObservation', 'DetectorSetup', 'ExperimentalPose', 'OrientationExperiment',
    'resolve_pose', 'validate_experiment', 'FittedCandidate', 'OrientationFitResult',
    'fit_orientation', 'OrientationSetup', 'new_experiment', 'snapshot_setup',
]
