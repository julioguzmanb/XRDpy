"""
User-facing FemtoMAX azimuthal-integration API.

This module is a thin facility-facing wrapper around
:mod:`trxrdpy.analysis._shared_2d.azimint`.

Purpose
-------
FemtoMAX and SACLA currently share the same 2D-image-based azimuthal-integration
workflow after data reduction. To avoid duplicating code, the implementation
lives in the shared 2D layer, while this module preserves the beamline-specific
public import path:

    trxrdpy.analysis.MaxIV_FemtoMAX.azimint

Path handling
-------------
All entry points require either:
  - ``paths=AnalysisPaths(...)``
or:
  - ``path_root=...`` and ``analysis_subdir=...``

Notes
-----
- ``integrate_*`` functions compute/cache XY files.
- ``plot_*`` functions can reuse existing XY files or compute them on demand.
- ``poni_path`` / ``mask_edf_path`` are only required when XY files must be computed.
- This wrapper exists mainly for API stability and beamline clarity.
"""

from __future__ import annotations

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

