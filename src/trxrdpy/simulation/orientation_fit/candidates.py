"""Discrete reflection assignments and orientation seeds from measured peaks."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation

from .. import sample
from ..parallel import check_cancelled
from .forward_model import build_detector, observed_scattering_vector, sample_rotation_at_pose
from .observations import OrientationExperiment, resolve_pose, validate_experiment


@dataclass
class OrientationCandidate:
    """One initial reference rotation with an HKL choice for each enabled peak."""
    rotation: np.ndarray
    assignments: dict[str, tuple[int, int, int]]
    seed_score: float


def _cell(experiment):
    lattice = sample.LatticeStructure(**experiment.lattice)
    reciprocal = sample.cal_reciprocal_lattice(lattice.initial_crystal_orientation)
    return lattice, reciprocal


def _observations(experiment):
    rows = []
    for peak in experiment.peaks:
        if not peak.enabled:
            continue
        pose = resolve_pose(experiment, peak.pose_id)
        q_lab = observed_scattering_vector(experiment, pose, peak, build_detector(experiment, pose))
        motor_rotation = sample_rotation_at_pose(experiment, pose)
        rows.append((peak, q_lab @ motor_rotation))
    return rows


def _possible_indices(experiment, peak, q_ref, lattice, reciprocal, max_unknown=80):
    if peak.assignment_mode == 'exact':
        return [tuple(map(int, peak.hkl))]
    if peak.assignment_mode == 'family':
        return sorted({tuple(map(int, hkl)) for hkl in lattice.get_equivalent_reflections(peak.hkl)})
    q_size = np.linalg.norm(q_ref)
    lattice.create_possible_reflections(float(q_size * 1.12 + 0.05))
    options = [tuple(map(int, hkl)) for hkl in lattice.allowed_hkls if any(hkl)]
    options.sort(key=lambda hkl: abs(np.linalg.norm(np.asarray(hkl) @ reciprocal) - q_size))
    tolerance = max(0.05, q_size * 0.08)
    matching = [hkl for hkl in options if
                abs(np.linalg.norm(np.asarray(hkl) @ reciprocal) - q_size) <= tolerance]
    if len(matching) > max_unknown:
        raise ValueError(f'Peak {peak.id}: {len(matching)} HKLs match its q magnitude. '
                         'Enter a likely family or exact HKL to constrain the search.')
    return matching


def _initial_guess_rotation(experiment, lattice):
    if experiment.initial_guess is None:
        return None
    trial = np.linalg.solve(lattice.initial_crystal_orientation, np.asarray(experiment.initial_guess, dtype=float)).T
    u, _, vt = np.linalg.svd(trial)
    fixed = np.eye(3)
    fixed[-1, -1] = np.linalg.det(u @ vt)
    return u @ fixed @ vt


def generate_candidates(experiment: OrientationExperiment, initial_guess=None, *,
                        max_candidates=24) -> list[OrientationCandidate]:
    """Build plausible rotations from independent peaks and HKL alternatives.

    A single independent reflection leaves a free spin angle; the returned
    seeds sample that family and the diagnostics report its rank deficiency.
    """
    validate_experiment(experiment)
    lattice, reciprocal = _cell(experiment)
    rows = _observations(experiment)
    possible = [_possible_indices(experiment, peak, q, lattice, reciprocal) for peak, q in rows]
    if any(not choices for choices in possible):
        raise ValueError('No allowed HKLs match at least one unknown peak; check cell, energy, or pixel coordinates.')
    seeds = []
    prior = _initial_guess_rotation(experiment, lattice)
    if initial_guess is not None:
        prior = Rotation.from_rotvec(np.asarray(initial_guess, dtype=float)).as_matrix()
    if prior is not None:
        seeds.append(prior)

    # Choose the most independent measured pair with a manageable assignment count.
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            q1, q2 = rows[i][1], rows[j][1]
            sine = np.linalg.norm(np.cross(q1, q2)) / (np.linalg.norm(q1) * np.linalg.norm(q2))
            if sine > 0.05:
                pairs.append((len(possible[i]) * len(possible[j]) / sine, i, j))
    if pairs:
        _, i, j = min(pairs)
        observed = np.array([rows[i][1], rows[j][1]])
        for h1, h2 in product(possible[i], possible[j]):
            check_cancelled()
            predicted = np.array([np.asarray(h1) @ reciprocal, np.asarray(h2) @ reciprocal])
            cosine_observed = np.dot(*observed) / np.prod(np.linalg.norm(observed, axis=1))
            cosine_predicted = np.dot(*predicted) / np.prod(np.linalg.norm(predicted, axis=1))
            if abs(cosine_observed - cosine_predicted) > 0.12:
                continue
            rotation, _ = Rotation.align_vectors(observed, predicted)
            seeds.append(rotation.as_matrix())
    else:
        # For one independent vector, build representative spins without
        # pretending they exhaust the continuous family of possible rotations.
        peak, q_ref = rows[0]
        for hkl in possible[0][:12]:
            source = np.asarray(hkl) @ reciprocal
            aligned, _ = Rotation.align_vectors([q_ref], [source])
            axis = q_ref / np.linalg.norm(q_ref)
            for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                seeds.append((Rotation.from_rotvec(axis * angle) * aligned).as_matrix())
    if not seeds:
        raise ValueError('No orientation seeds satisfy the measured reciprocal-vector geometry.')

    ranked = []
    for rotation in seeds:
        check_cancelled()
        assignments = {}
        score = 0.0
        for (peak, q_ref), choices in zip(rows, possible):
            vectors = np.asarray(choices) @ reciprocal @ rotation.T
            differences = np.linalg.norm(vectors - q_ref, axis=1)
            chosen = int(np.argmin(differences))
            assignments[peak.id] = choices[chosen]
            score += float(differences[chosen] ** 2)
        ranked.append(OrientationCandidate(rotation, assignments, score))
    ranked.sort(key=lambda candidate: candidate.seed_score)
    unique = []
    for candidate in ranked:
        if any(np.linalg.norm(Rotation.from_matrix(candidate.rotation @ old.rotation.T).as_rotvec()) < np.deg2rad(0.2)
               and candidate.assignments == old.assignments for old in unique):
            continue
        unique.append(candidate)
        if len(unique) >= max_candidates:
            break
    return unique
