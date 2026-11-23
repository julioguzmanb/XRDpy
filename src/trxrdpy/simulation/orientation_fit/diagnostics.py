"""Human-readable diagnostics for orientation candidates."""
from __future__ import annotations

from .refine import FittedCandidate


def assess_observability(candidate: FittedCandidate) -> str:
    """Describe whether the measurements constrain all three rotation axes."""
    if candidate.rank == 3:
        return 'All three reference-orientation rotation directions are locally constrained.'
    if candidate.rank == 2:
        return 'One continuous rotation remains unresolved; add a nonparallel reflection.'
    return 'The reference orientation is underdetermined; add independent reflections.'


def summarize_candidate(candidate: FittedCandidate) -> str:
    """Summarize fit quality, assignments, and peak-by-peak differences."""
    lines = [assess_observability(candidate),
             f'Weighted squared residual: {candidate.weighted_error:.3g}',
             f'RMS pixel difference: {candidate.rms_pixel:.3f} px']
    if candidate.angular_sigma_deg is not None:
        lines.append('Approximate local rotation uncertainty [°]: ' +
                     ', '.join(f'{value:.3g}' for value in candidate.angular_sigma_deg))
    for row in candidate.peak_diagnostics:
        hkl = tuple(row['hkl'])
        lines.append(f"{row['peak_id']} at {row['pose_id']}: {hkl}, "
                     f"pixel difference {row['pixel_error']:.2f} px, "
                     f"normalized q difference {row['normalized_error']:.2f}, "
                     f"Ewald mismatch {row['ewald_error']:.3g} Å⁻¹, "
                     f"on detector: {'yes' if row['on_detector'] else 'no'}")
    return '\n'.join(lines)
