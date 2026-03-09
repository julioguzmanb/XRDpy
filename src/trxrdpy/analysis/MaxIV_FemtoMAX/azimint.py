"""
User-facing FemtoMAX azimuthal-integration API.

This module is a thin facility-facing wrapper around
:mod:`XRDpy.analysis._shared_2d.azimint`.

Purpose
-------
FemtoMAX and SACLA currently share the same 2D-image-based azimuthal-integration
workflow after data reduction. To avoid duplicating code, the implementation
lives in the shared 2D layer, while this module preserves the beamline-specific
public import path:

    XRDpy.analysis.MaxIV_FemtoMAX.azimint

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

# ============================================================
# Example usage
# ============================================================
"""
from pathlib import Path
import numpy as np
import XRDpy as XRD
from XRDpy.analysis.common.paths import AnalysisPaths

# ------------------------------------------------------------------
# Common path configuration (choose ONE style)
# ------------------------------------------------------------------
# Style A: plain strings
path_root = "/Users/julioguzman/Desktop/LSF2025/FemtoMAX2025"
analysis_subdir = "analysis"

# Style B: AnalysisPaths
# paths = AnalysisPaths(path_root=Path(path_root), analysis_subdir="analysis")


# ============================================================
# Produce / update a dark scan XY cache
# ============================================================
sample_name = "DET55"
temperature_K = 77
dark_tag = 167246

poni_path = f"{path_root}/calibration/{sample_name}_{dark_tag}.poni"
mask_edf_path = f"{path_root}/calibration/{sample_name}_{dark_tag}_mask.edf"

XRD.analysis.MaxIV_FemtoMAX.azimint.integrate_dark_1d(
    sample_name=sample_name,
    temperature_K=temperature_K,
    poni_path=poni_path,
    mask_edf_path=mask_edf_path,
    dark_tag=dark_tag,
    azimuthal_edges=np.arange(-90, 90 + 20, 45),
    include_full=True,
    overwrite_xy=True,
    path_root=path_root,
    analysis_subdir=analysis_subdir,
)


# ============================================================
# Experiment 1. Delay scan.
# Pure V2O3, large grains, ≈ 60 nm thick.
# ============================================================
sample_name = "DET55"
temperature_K = 77
excitation_wl_nm = 1500
fluence_mJ_cm2 = 15
time_window_fs = 1000

poni_path = f"{path_root}/calibration/DET55_167246.poni"
mask_edf_path = f"{path_root}/calibration/DET55_167246_mask.edf"

delays_to_compute = "all"
overwrite_xy = True
ref_type, ref_value = "dark", [167246, 167285]
azim_window = (-90, 90)

# Compute + cache XY files for all available delay points
# XRD.analysis.MaxIV_FemtoMAX.azimint.integrate_delay_1d(
#     sample_name=sample_name,
#     temperature_K=temperature_K,
#     excitation_wl_nm=excitation_wl_nm,
#     fluence_mJ_cm2=fluence_mJ_cm2,
#     time_window_fs=time_window_fs,
#     delays_fs=delays_to_compute,
#     poni_path=poni_path,
#     mask_edf_path=mask_edf_path,
#     azimuthal_edges=np.arange(-90 + 15, 90 + 15, 30),
#     include_full=True,
#     full_range=(-90, 90),
#     npt=1000,
#     normalize=True,
#     q_norm_range=(2.65, 2.75),
#     overwrite_xy=overwrite_xy,
#     path_root=path_root,
#     analysis_subdir=analysis_subdir,
# )

# Compare + save plot
q_ref, I_ref, fig, axes = XRD.analysis.MaxIV_FemtoMAX.azimint.plot_1D_abs_and_diffs_delay(
    sample_name=sample_name,
    temperature_K=temperature_K,
    excitation_wl_nm=excitation_wl_nm,
    fluence_mJ_cm2=fluence_mJ_cm2,
    time_window_fs=time_window_fs,
    delays_fs=delays_to_compute,
    ref_type=ref_type,
    ref_value=ref_value,
    poni_path=poni_path,
    mask_edf_path=mask_edf_path,
    azim_window=azim_window,
    xlim=(1.5, 4.5),
    save_plots=True,
    save_format="png",
    save_dpi=400,
    save_overwrite=True,
    from_2D_imgs=False,
    path_root=path_root,
    analysis_subdir=analysis_subdir,
)


# ============================================================
# Experiment 2. Fluence scan at fixed delay.
# Pure V2O3, large grains, ≈ 60 nm thick.
# ============================================================
sample_name = "DET55"
temperature_K = 77
excitation_wl_nm = 1500
delay_fs = -1000
time_window_fs = 500

poni_path = f"{path_root}/calibration/DET55_167246.poni"
mask_edf_path = f"{path_root}/calibration/DET55_167285_mask.edf"

fluences_to_compute = "all"
overwrite_xy = True
ref_type = "dark"
ref_value = [167246, 167285]
azim_window = (-90, 90)

# Compute + cache XY files for all available fluences
# XRD.analysis.MaxIV_FemtoMAX.azimint.integrate_fluence_1d(
#     sample_name=sample_name,
#     temperature_K=temperature_K,
#     excitation_wl_nm=excitation_wl_nm,
#     delay_fs=delay_fs,
#     time_window_fs=time_window_fs,
#     fluences_mJ_cm2=fluences_to_compute,
#     poni_path=poni_path,
#     mask_edf_path=mask_edf_path,
#     azimuthal_edges=np.arange(-90 + 15, 90 + 15, 30),
#     include_full=True,
#     full_range=(-90, 90),
#     npt=1000,
#     normalize=True,
#     q_norm_range=(2.65, 2.75),
#     overwrite_xy=overwrite_xy,
#     path_root=path_root,
#     analysis_subdir=analysis_subdir,
# )

fig, axes = XRD.analysis.MaxIV_FemtoMAX.azimint.plot_1D_abs_and_diffs_fluence(
    sample_name=sample_name,
    temperature_K=temperature_K,
    excitation_wl_nm=excitation_wl_nm,
    delay_fs=delay_fs,
    time_window_fs=time_window_fs,
    fluences_mJ_cm2=fluences_to_compute,
    ref_type=ref_type,
    ref_value=ref_value,
    poni_path=poni_path,
    mask_edf_path=mask_edf_path,
    azim_window=azim_window,
    xlim=(1.5, 4.5),
    save_plots=True,
    save_format="png",
    save_dpi=400,
    save_overwrite=True,
    path_root=path_root,
    analysis_subdir=analysis_subdir,
)
"""

