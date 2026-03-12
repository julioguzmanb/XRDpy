"""
SPring-8 / SACLA azimuthal-integration API for trxrdpy.

This module exposes the shared 2D-image azimuthal-integration workflow under
the SACLA facility namespace.

Implementation note
-------------------
The actual implementation lives in :mod:`trxrdpy.analysis._shared_2d.azimint`.
This is appropriate for SACLA because, once 2D images are produced, the XY
generation and 1D pattern comparison workflow is the same as for FemtoMAX.

Scope
-----
- compute/cache XY files from 2D images
- compare delay-series 1D patterns to a reference
- compare fluence-series 1D patterns to a reference

Path handling
-------------
All entry points require either:
  - ``paths=AnalysisPaths(...)``
or:
  - ``path_root=...`` and ``analysis_subdir=...``

Notes
-----
- This module does not implement SACLA raw-data reduction.
- SACLA-specific reduction remains in :mod:`trxrdpy.analysis.Spring8_SACLA.datared`.
- This wrapper exists so users can call:
    ``trxrdpy.analysis.Spring8_SACLA.azimint...``
  while the implementation stays shared.
"""

from .._shared_2d.azimint import (
    integrate_dark_1d,
    integrate_delay_1d,
    plot_1D_abs_and_diffs_delay,
    integrate_fluence_1d,
    plot_1D_abs_and_diffs_fluence,
)

__all__ = [
    "integrate_dark_1d",
    "integrate_delay_1d",
    "plot_1D_abs_and_diffs_delay",
    "integrate_fluence_1d",
    "plot_1D_abs_and_diffs_fluence",
]