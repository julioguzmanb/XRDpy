from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .common import calibration_utils, general_utils, plot_utils
from .common.paths import AnalysisPaths

ScanSpec = calibration_utils.ScanSpec

FIGURES_SUBDIR_DEFAULT = "figures/calibration/"
DEFAULT_AZIMUTHAL_RANGES = tuple(np.arange(-90 + 15, 90 + 15, 30))


def _make_context(
    *,
    sample_name: str,
    temperature_K: int,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> calibration_utils.CalibrationContext:
    return calibration_utils.CalibrationContext(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )


def _azim_window_int(
    azim_window: Tuple[Union[int, float], Union[int, float]]
) -> Tuple[int, int]:
    return (int(round(float(azim_window[0]))), int(round(float(azim_window[1]))))


def _full_range_int(
    full_range: Tuple[Union[int, float], Union[int, float]]
) -> Tuple[int, int]:
    return (int(round(float(full_range[0]))), int(round(float(full_range[1]))))


def _windows_from_edges(
    azimuthal_ranges: Sequence[Union[int, float]],
    *,
    include_full: bool,
    full_range: Tuple[Union[int, float], Union[int, float]],
) -> List[Tuple[int, int]]:
    edges = np.asarray(azimuthal_ranges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("azimuthal_ranges must be a 1D sequence with at least two entries.")

    windows: List[Tuple[int, int]] = []
    if bool(include_full):
        windows.append(_full_range_int(full_range))

    for j in range(edges.size - 1):
        windows.append((int(round(float(edges[j]))), int(round(float(edges[j + 1])))))

    return windows


def _is_int_like(x) -> bool:
    return isinstance(x, (int, np.integer))


def _normalize_scan_specs(x, *, mode: str) -> List[ScanSpec]:
    if _is_int_like(x) or isinstance(x, str):
        return [x]

    try:
        xs = list(x)
    except Exception:
        return [x]

    if len(xs) == 0:
        return []

    if all(_is_int_like(v) for v in xs):
        if mode == "together":
            return [[int(v) for v in xs]]
        if mode == "separate":
            return [int(v) for v in xs]
        raise ValueError("int_list_mode must be 'together' or 'separate'.")

    specs: List[ScanSpec] = []
    for v in xs:
        if _is_int_like(v) or isinstance(v, str):
            specs.append(v)
        else:
            vv = list(v)
            if len(vv) == 0 or (not all(_is_int_like(t) for t in vv)):
                raise ValueError("Invalid scans format.")
            specs.append([int(t) for t in vv])

    return specs


def _spec_label(spec: ScanSpec) -> str:
    try:
        return str(general_utils.scan_tag(spec))
    except Exception:
        return str(spec)


def compute_xy_files(
    sample_name,
    scan,
    temperature_K,
    *,
    azimuthal_ranges: Sequence[Union[int, float]] = DEFAULT_AZIMUTHAL_RANGES,
    include_full: bool = False,
    full_range: Tuple[Union[int, float], Union[int, float]] = (-90, 90),
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    overwrite_xy: bool = False,
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    azim_offset_deg: float = -90.0,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    ctx = _make_context(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )
    return ctx.compute_xy_files(
        scan,
        azimuthal_ranges=np.asarray(azimuthal_ranges, dtype=float),
        include_full=bool(include_full),
        full_range=tuple(float(v) for v in full_range),
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=tuple(float(v) for v in q_norm_range),
        overwrite_xy=bool(overwrite_xy),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azim_offset_deg=float(azim_offset_deg),
    )


def do_peak_fitting(
    sample_name,
    scan,
    temperature_K,
    *,
    q_fit_range: Tuple[float, float] = (2.4, 2.65),
    azimuthal_ranges: Sequence[Union[int, float]] = DEFAULT_AZIMUTHAL_RANGES,
    include_full: bool = False,
    full_range: Tuple[Union[int, float], Union[int, float]] = (-90, 90),
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    eta: float = 0.3,
    fit_method: str = "leastsq",
    force_refit: bool = True,
    out_csv_name: str = "peak_fits.csv",
    overwrite_xy: bool = False,
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    azim_offset_deg: float = -90.0,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    ctx = _make_context(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )
    return ctx.do_peak_fitting(
        scan,
        q_fit_range=tuple(float(v) for v in q_fit_range),
        azimuthal_ranges=np.asarray(azimuthal_ranges, dtype=float),
        include_full=bool(include_full),
        full_range=tuple(float(v) for v in full_range),
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=tuple(float(v) for v in q_norm_range),
        eta=float(eta),
        fit_method=str(fit_method),
        force_refit=bool(force_refit),
        out_csv_name=str(out_csv_name),
        overwrite_xy=bool(overwrite_xy),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azim_offset_deg=float(azim_offset_deg),
    )


def plot_caked_1D_patterns(
    sample_name,
    scan,
    temperature_K,
    *,
    azimuthal_ranges: Sequence[Union[int, float]] = DEFAULT_AZIMUTHAL_RANGES,
    include_full: bool = False,
    full_range: Tuple[Union[int, float], Union[int, float]] = (-90, 90),
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    overwrite_xy: bool = False,
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    azim_offset_deg: float = -90.0,
    xlim=None,
    ylim=None,
    figure_title: Optional[str] = None,
    save: bool = False,
    figures_subdir: str = FIGURES_SUBDIR_DEFAULT,
    save_format: str = "png",
    save_dpi: int = 400,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    ctx = _make_context(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    ctx.compute_xy_files(
        scan,
        azimuthal_ranges=np.asarray(azimuthal_ranges, dtype=float),
        include_full=bool(include_full),
        full_range=tuple(float(v) for v in full_range),
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=tuple(float(v) for v in q_norm_range),
        overwrite_xy=bool(overwrite_xy),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azim_offset_deg=float(azim_offset_deg),
    )

    windows = _windows_from_edges(
        azimuthal_ranges,
        include_full=bool(include_full),
        full_range=full_range,
    )

    patterns = []
    for w in windows:
        q, intensity, _ = ctx.load_xy(
            scan,
            azim_window=w,
            npt=int(npt),
            normalize=bool(normalize),
            q_norm_range=tuple(float(v) for v in q_norm_range),
            compute_if_missing=True,
            overwrite_xy=bool(overwrite_xy),
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            azim_offset_deg=float(azim_offset_deg),
        )
        patterns.append((general_utils.azim_range_str(w), q, intensity))

    tag = _spec_label(scan)
    base_dir = ctx.analysis_dir(scan)
    save_name = f"{sample_name}_{temperature_K}K_caked_1D_patterns_{tag}"

    save_kw = calibration_utils._save_kwargs(
        save=bool(save),
        base_dir=base_dir,
        figures_subdir=str(figures_subdir),
        save_name=str(save_name),
        save_format=str(save_format),
        save_dpi=int(save_dpi),
    )

    p = plot_utils.Pattern1DPlotter()
    p.plot_caked_patterns(
        patterns,
        title=figure_title,
        xlim=xlim,
        ylim=ylim,
        figsize=(6, 6),
        legend_ncol=1,
        **save_kw,
    )
    return patterns


def plot_property_vs_azimuth(
    sample_name,
    scan,
    temperature_K,
    *,
    _property: str = "pv_center",
    figure_title: Optional[str] = None,
    only_success: bool = True,
    out_csv_name: str = "peak_fits.csv",
    ylim=None,
    save: bool = False,
    figures_subdir: str = FIGURES_SUBDIR_DEFAULT,
    save_format: str = "png",
    save_dpi: int = 400,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    ctx = _make_context(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    csv_path = str(ctx.peak_fits_csv_path(scan, out_csv_name=str(out_csv_name)))
    tag = _spec_label(scan)
    base_dir = ctx.analysis_dir(scan)

    if figure_title is None:
        figure_title = f"{sample_name}, {temperature_K}K, {tag}\n{_property} vs azimuth".replace(
            "_", ": "
        )

    save_name = f"{sample_name}_{temperature_K}K_{str(_property)}_vs_azimuth_{tag}"

    save_kw = calibration_utils._save_kwargs(
        save=bool(save),
        base_dir=base_dir,
        figures_subdir=str(figures_subdir),
        save_name=str(save_name),
        save_format=str(save_format),
        save_dpi=int(save_dpi),
    )

    p = plot_utils.FitCSVPlotter()
    return p.plot_property_vs_azimuth(
        csv_path,
        y=str(_property),
        only_success=bool(only_success),
        title=figure_title,
        ylim=ylim,
        **save_kw,
    )


def plot_1D_plus_fit(
    sample_name,
    scan,
    temperature_K,
    *,
    azim_window: Tuple[Union[int, float], Union[int, float]] = (-45, 0),
    out_csv_name: str = "peak_fits.csv",
    fit_oversample: int = 10,
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    overwrite_xy: bool = False,
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    azim_offset_deg: float = -90.0,
    figure_title: Optional[str] = None,
    save: bool = False,
    figures_subdir: str = FIGURES_SUBDIR_DEFAULT,
    save_format: str = "png",
    save_dpi: int = 400,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    ctx = _make_context(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    azim_window_i = _azim_window_int(azim_window)

    q, intensity, _ = ctx.load_xy(
        scan,
        azim_window=azim_window_i,
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=tuple(float(v) for v in q_norm_range),
        compute_if_missing=True,
        overwrite_xy=bool(overwrite_xy),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azim_offset_deg=float(azim_offset_deg),
    )

    csv_path = str(ctx.peak_fits_csv_path(scan, out_csv_name=str(out_csv_name)))
    azim_str = general_utils.azim_range_str(azim_window_i)

    tag = _spec_label(scan)
    base_dir = ctx.analysis_dir(scan)
    save_name = f"{sample_name}_{temperature_K}K_fit_overlay_{azim_str}_{tag}"

    save_kw = calibration_utils._save_kwargs(
        save=bool(save),
        base_dir=base_dir,
        figures_subdir=str(figures_subdir),
        save_name=str(save_name),
        save_format=str(save_format),
        save_dpi=int(save_dpi),
    )

    p = plot_utils.FitCSVPlotter()
    return p.plot_fit_overlay(
        q=q,
        I=intensity,
        csv_path=csv_path,
        azim_range_str=str(azim_str),
        title=figure_title,
        fit_oversample=int(fit_oversample),
        **save_kw,
    )


def compare_1D_patterns(
    *,
    sample_name,
    scans,
    temperature_K,
    scan_ref,
    azim_window: Tuple[Union[int, float], Union[int, float]] = (-90, 90),
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    overwrite_xy: bool = False,
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    azim_offset_deg: float = -90.0,
    xlim: Tuple[float, float] = (1.5, 4.5),
    ylim_top=None,
    ylim_diff=None,
    figure_title: Optional[str] = None,
    int_list_mode: str = "together",
    save: bool = False,
    figures_subdir: str = FIGURES_SUBDIR_DEFAULT,
    save_format: str = "png",
    save_dpi: int = 400,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    ctx = _make_context(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    specs = _normalize_scan_specs(scans, mode=str(int_list_mode))
    ref_specs = _normalize_scan_specs(scan_ref, mode="together")
    if len(ref_specs) != 1:
        raise ValueError("scan_ref must resolve to exactly one dataset.")
    ref_spec = ref_specs[0]

    azim_window_i = _azim_window_int(azim_window)

    q_ref, I_ref, _ = ctx.load_xy(
        ref_spec,
        azim_window=azim_window_i,
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=tuple(float(v) for v in q_norm_range),
        compute_if_missing=True,
        overwrite_xy=bool(overwrite_xy),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azim_offset_deg=float(azim_offset_deg),
    )

    ref_label = _spec_label(ref_spec)

    patterns = []
    for s in specs:
        if s == ref_spec:
            continue
        q, intensity, _ = ctx.load_xy(
            s,
            azim_window=azim_window_i,
            npt=int(npt),
            normalize=bool(normalize),
            q_norm_range=tuple(float(v) for v in q_norm_range),
            compute_if_missing=True,
            overwrite_xy=bool(overwrite_xy),
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            azim_offset_deg=float(azim_offset_deg),
        )
        patterns.append((_spec_label(s), q, intensity))

    azim_str = general_utils.azim_range_str(azim_window_i)
    if figure_title is None:
        figure_title = f"{sample_name} {temperature_K}K (dark) azim={azim_str}"

    base_dir = ctx.analysis_dir(ref_spec)
    save_name = f"{sample_name}_{temperature_K}K_compare_ref_{ref_label}_{azim_str}"

    save_kw = calibration_utils._save_kwargs(
        save=bool(save),
        base_dir=base_dir,
        figures_subdir=str(figures_subdir),
        save_name=str(save_name),
        save_format=str(save_format),
        save_dpi=int(save_dpi),
    )

    p = plot_utils.Pattern1DPlotter()
    p.compare_to_reference(
        q_ref=q_ref,
        I_ref=I_ref,
        ref_label=ref_label,
        patterns=patterns,
        title=figure_title,
        xlim=xlim,
        ylim_top=ylim_top,
        ylim_diff=ylim_diff,
        legend_title="Dataset",
        legend_loc="upper left",
        legend_outside=True,
        **save_kw,
    )
    return q_ref, I_ref, patterns


__all__ = [
    "ScanSpec",
    "FIGURES_SUBDIR_DEFAULT",
    "DEFAULT_AZIMUTHAL_RANGES",
    "compute_xy_files",
    "do_peak_fitting",
    "plot_caked_1D_patterns",
    "plot_property_vs_azimuth",
    "plot_1D_plus_fit",
    "compare_1D_patterns",
]

