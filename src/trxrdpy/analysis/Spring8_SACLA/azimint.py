"""User-facing SACLA azimuthal-integration API.

Representative 2D images remain the default source. SACLA metadata-selected
detector tags can alternatively be integrated into a separate single-shot 1D
cache and aggregated into the same final ``xy_files`` products.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .._shared_2d.azimint import (
    integrate_dark_1d as _integrate_dark_1d_from_2d,
    integrate_delay_1d as _integrate_delay_1d_from_2d,
    integrate_fluence_1d as _integrate_fluence_1d_from_2d,
    plot_1D_abs_and_diffs_delay,
    plot_1D_abs_and_diffs_fluence,
)
from ..common.paths import AnalysisPaths
from .single_shot_azimint import (
    DEFAULT_DETECTOR_ID,
    aggregate_dark_1d,
    aggregate_delay_1d,
    aggregate_fluence_1d,
    integrate_single_shot_1d as _integrate_single_shot_1d,
)


def _normalize_source(source: str) -> str:
    """Return the canonical final-pattern input mode."""
    value = str(source).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "2d": "representative_2d",
        "representative": "representative_2d",
        "representative_2d_image": "representative_2d",
        "representative_2d_images": "representative_2d",
        "single_shot": "single_shot_1d",
        "single_shots": "single_shot_1d",
        "single_shot_1d_pattern": "single_shot_1d",
        "single_shot_1d_patterns": "single_shot_1d",
    }
    value = aliases.get(value, value)
    if value not in {"representative_2d", "single_shot_1d"}:
        raise ValueError("source must be 'representative_2d' or 'single_shot_1d'.")
    return value


def _resolve_paths(
    *,
    paths: Optional[AnalysisPaths],
    path_root: Optional[Union[str, Path]],
    analysis_subdir: Optional[Union[str, Path]],
    raw_subdir: Optional[Union[str, Path]] = None,
) -> AnalysisPaths:
    """Resolve the path object used by cache and background-run lookups."""
    if paths is not None:
        return paths
    if path_root is None or analysis_subdir is None:
        raise ValueError(
            "Provide either paths=AnalysisPaths(...), or both path_root=... "
            "and analysis_subdir=...."
        )
    return AnalysisPaths(
        path_root=Path(path_root),
        raw_subdir=str(raw_subdir or ""),
        analysis_subdir=str(analysis_subdir),
    )


def integrate_single_shot_1d(
    *,
    metadata_h5_path: Union[str, Path],
    poni_path: Union[str, Path],
    mask_edf_path: Optional[Union[str, Path]],
    azimuthal_edges: Sequence[float] = np.arange(-90, 90 + 20, 45),
    include_full: bool = True,
    full_range: Tuple[float, float] = (-90.0, 90.0),
    npt: int = 1000,
    overwrite: bool = False,
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    beamline: Optional[int] = None,
    detector_id: str = DEFAULT_DETECTOR_ID,
    background=None,
    background_path: Optional[Union[str, Path]] = None,
    threshold_counts: float = 40.0,
    intensity_col: Optional[str] = None,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
    facility_client=None,
    chunk_id: int = 1,
    n_chunks: int = 1,
):
    """Integrate one scheduler-friendly partition of selected SACLA tags."""
    resolved_paths = _resolve_paths(
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
    )
    return _integrate_single_shot_1d(
        metadata_h5_path=metadata_h5_path,
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azimuthal_edges=azimuthal_edges,
        include_full=include_full,
        full_range=full_range,
        npt=npt,
        overwrite=overwrite,
        azim_offset_deg=azim_offset_deg,
        polarization_factor=polarization_factor,
        beamline=beamline,
        detector_id=detector_id,
        background=background,
        background_path=background_path,
        threshold_counts=threshold_counts,
        intensity_col=intensity_col,
        paths=resolved_paths,
        facility_client=facility_client,
        chunk_id=chunk_id,
        n_chunks=n_chunks,
    )


def _single_shot_settings(
    *,
    beamline,
    detector_id,
    background,
    background_path,
    threshold_counts,
    intensity_col,
):
    """Return SACLA preprocessing keywords used only by the shot-cache route."""
    return {
        "beamline": beamline,
        "detector_id": detector_id,
        "background": background,
        "background_path": background_path,
        "threshold_counts": threshold_counts,
        "intensity_col": intensity_col,
    }


def integrate_dark_1d(
    *,
    sample_name: str,
    temperature_K: int,
    poni_path: str,
    mask_edf_path: Optional[str],
    dark_tag=None,
    azimuthal_edges: np.ndarray = np.arange(-90, 90 + 20, 45),
    include_full: bool = True,
    full_range: Tuple[float, float] = (-90, 90),
    npt: int = 2000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    overwrite_xy: bool = False,
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
    source: str = "representative_2d",
    metadata_h5_path: Optional[Union[str, Path]] = None,
    beamline: Optional[int] = None,
    detector_id: str = DEFAULT_DETECTOR_ID,
    background=None,
    background_path: Optional[Union[str, Path]] = None,
    threshold_counts: float = 40.0,
    intensity_col: Optional[str] = None,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """Create final dark XY files from a representative image or shot cache."""
    common = dict(
        sample_name=sample_name,
        temperature_K=temperature_K,
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        dark_tag=dark_tag,
        azimuthal_edges=azimuthal_edges,
        include_full=include_full,
        full_range=full_range,
        npt=npt,
        normalize=normalize,
        q_norm_range=q_norm_range,
        overwrite_xy=overwrite_xy,
        azim_offset_deg=azim_offset_deg,
        polarization_factor=polarization_factor,
        use_parallel=use_parallel,
        max_workers=max_workers,
    )
    if _normalize_source(source) == "representative_2d":
        return _integrate_dark_1d_from_2d(
            **common,
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )
    if metadata_h5_path is None:
        raise ValueError("metadata_h5_path is required for source='single_shot_1d'.")
    resolved_paths = _resolve_paths(
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )
    return aggregate_dark_1d(
        **common,
        metadata_h5_path=metadata_h5_path,
        paths=resolved_paths,
        **_single_shot_settings(
            beamline=beamline,
            detector_id=detector_id,
            background=background,
            background_path=background_path,
            threshold_counts=threshold_counts,
            intensity_col=intensity_col,
        ),
    )


def integrate_delay_1d(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    fluence_mJ_cm2: float,
    time_window_fs: int,
    delays_fs,
    poni_path: str,
    mask_edf_path: Optional[str],
    azimuthal_edges: np.ndarray = np.arange(-90, 90 + 20, 45),
    include_full: bool = True,
    full_range: Tuple[float, float] = (-90, 90),
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    overwrite_xy: bool = False,
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
    source: str = "representative_2d",
    metadata_h5_path: Optional[Union[str, Path]] = None,
    beamline: Optional[int] = None,
    detector_id: str = DEFAULT_DETECTOR_ID,
    background=None,
    background_path: Optional[Union[str, Path]] = None,
    threshold_counts: float = 40.0,
    intensity_col: Optional[str] = None,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """Create final delay XY files from representative images or shot cache."""
    common = dict(
        sample_name=sample_name,
        temperature_K=temperature_K,
        excitation_wl_nm=excitation_wl_nm,
        fluence_mJ_cm2=fluence_mJ_cm2,
        time_window_fs=time_window_fs,
        delays_fs=delays_fs,
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azimuthal_edges=azimuthal_edges,
        include_full=include_full,
        full_range=full_range,
        npt=npt,
        normalize=normalize,
        q_norm_range=q_norm_range,
        overwrite_xy=overwrite_xy,
        azim_offset_deg=azim_offset_deg,
        polarization_factor=polarization_factor,
        use_parallel=use_parallel,
        max_workers=max_workers,
    )
    if _normalize_source(source) == "representative_2d":
        return _integrate_delay_1d_from_2d(
            **common,
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )
    if metadata_h5_path is None:
        raise ValueError("metadata_h5_path is required for source='single_shot_1d'.")
    resolved_paths = _resolve_paths(
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )
    return aggregate_delay_1d(
        **common,
        metadata_h5_path=metadata_h5_path,
        paths=resolved_paths,
        **_single_shot_settings(
            beamline=beamline,
            detector_id=detector_id,
            background=background,
            background_path=background_path,
            threshold_counts=threshold_counts,
            intensity_col=intensity_col,
        ),
    )


def integrate_fluence_1d(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    delay_fs: int,
    time_window_fs: int,
    fluences_mJ_cm2,
    poni_path: str,
    mask_edf_path: Optional[str],
    azimuthal_edges: np.ndarray = np.arange(-90, 90 + 20, 45),
    include_full: bool = True,
    full_range: Tuple[float, float] = (-90, 90),
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    overwrite_xy: bool = False,
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
    source: str = "representative_2d",
    metadata_h5_path: Optional[Union[str, Path]] = None,
    beamline: Optional[int] = None,
    detector_id: str = DEFAULT_DETECTOR_ID,
    background=None,
    background_path: Optional[Union[str, Path]] = None,
    threshold_counts: float = 40.0,
    intensity_col: Optional[str] = None,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """Create final fluence XY files from representative images or shot cache."""
    common = dict(
        sample_name=sample_name,
        temperature_K=temperature_K,
        excitation_wl_nm=excitation_wl_nm,
        delay_fs=delay_fs,
        time_window_fs=time_window_fs,
        fluences_mJ_cm2=fluences_mJ_cm2,
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azimuthal_edges=azimuthal_edges,
        include_full=include_full,
        full_range=full_range,
        npt=npt,
        normalize=normalize,
        q_norm_range=q_norm_range,
        overwrite_xy=overwrite_xy,
        azim_offset_deg=azim_offset_deg,
        polarization_factor=polarization_factor,
        use_parallel=use_parallel,
        max_workers=max_workers,
    )
    if _normalize_source(source) == "representative_2d":
        return _integrate_fluence_1d_from_2d(
            **common,
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )
    if metadata_h5_path is None:
        raise ValueError("metadata_h5_path is required for source='single_shot_1d'.")
    resolved_paths = _resolve_paths(
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )
    return aggregate_fluence_1d(
        **common,
        metadata_h5_path=metadata_h5_path,
        paths=resolved_paths,
        **_single_shot_settings(
            beamline=beamline,
            detector_id=detector_id,
            background=background,
            background_path=background_path,
            threshold_counts=threshold_counts,
            intensity_col=intensity_col,
        ),
    )


__all__ = [
    "integrate_dark_1d",
    "integrate_delay_1d",
    "integrate_fluence_1d",
    "integrate_single_shot_1d",
    "plot_1D_abs_and_diffs_delay",
    "plot_1D_abs_and_diffs_fluence",
]
