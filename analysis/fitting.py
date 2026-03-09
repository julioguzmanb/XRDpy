
# fitting.py
"""
User-facing peak fitting API.

Scope:
- Delay scans and fluence scans.
- Runs peak fitting across series (optionally including a reference).
- Exports CSV.
- Plots fitted overlays and evolution traces.

Path handling:
- Callers must provide either:
    * paths=AnalysisPaths(...)
  or
    * path_root=... and analysis_subdir=...
- Multi-experiment helpers also support per-experiment path configuration via
  exp["paths"] or exp["path_root"] + exp["analysis_subdir"].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import azimint_utils, fitting_utils, plot_utils
from .common.paths import AnalysisPaths

plt.ion()

def _resolve_exp_path_kwargs(
    exp: Dict[str, Any],
    *,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """
    Resolve path configuration for multi-experiment helpers.

    Priority:
      1) exp["paths"]
      2) exp["path_root"] + exp["analysis_subdir"]
      3) function-level paths
      4) function-level path_root + analysis_subdir
    """
    exp_paths = exp.get("paths", None)
    if exp_paths is not None:
        return {"paths": exp_paths}

    exp_root = exp.get("path_root", None)
    exp_sub = exp.get("analysis_subdir", None)
    if exp_root is not None and exp_sub is not None:
        return {
            "path_root": exp_root,
            "analysis_subdir": exp_sub,
        }

    if paths is not None:
        return {"paths": paths}

    if path_root is not None and analysis_subdir is not None:
        return {
            "path_root": path_root,
            "analysis_subdir": analysis_subdir,
        }

    raise ValueError(
        "No path configuration available. Provide either:\n"
        "  - paths=AnalysisPaths(...), or\n"
        "  - path_root=... and analysis_subdir=..., or\n"
        "  - per-experiment exp['paths'], or\n"
        "  - per-experiment exp['path_root'] and exp['analysis_subdir']."
    )


def _resolve_delay_csv_path(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    fluence_mJ_cm2: float,
    time_window_fs: int,
    out_csv_name: str = "peak_fits_delay.csv",
    csv_path: Optional[Union[str, Path]] = None,
    phi_mode: Optional[str] = None,
    phi_reduce: str = "sum",
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> str:
    if csv_path is not None and str(csv_path).strip() != "":
        return str(csv_path)

    try:
        return fitting_utils.resolve_delay_fitting_csv_path(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            fluence_mJ_cm2=float(fluence_mJ_cm2),
            time_window_fs=int(time_window_fs),
            out_csv_name=str(out_csv_name),
            phi_mode=phi_mode,
            phi_reduce=str(phi_reduce),
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )
    except FileNotFoundError:
        fallback_name = str(out_csv_name)
        if phi_mode is not None:
            fallback_name = fitting_utils._tagged_out_csv_name(
                str(out_csv_name),
                phi_mode=str(phi_mode),
                phi_reduce=str(phi_reduce),
            )

        return fitting_utils._default_fitting_csv_path(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            fluence_mJ_cm2=float(fluence_mJ_cm2),
            time_window_fs=int(time_window_fs),
            out_csv_name=str(fallback_name),
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )


def _resolve_fluence_csv_path(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    delay_fs: int,
    time_window_fs: int,
    out_csv_name: str = "peak_fits_fluence.csv",
    csv_path: Optional[Union[str, Path]] = None,
    phi_mode: Optional[str] = None,
    phi_reduce: str = "sum",
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> str:
    if csv_path is not None and str(csv_path).strip() != "":
        return str(csv_path)

    try:
        return fitting_utils.resolve_fluence_fitting_csv_path(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            delay_fs=int(delay_fs),
            time_window_fs=int(time_window_fs),
            out_csv_name=str(out_csv_name),
            phi_mode=str(phi_mode) if phi_mode is not None else "separate_phi",
            phi_reduce=str(phi_reduce),
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )
    except FileNotFoundError:
        fallback_name = str(out_csv_name)
        if phi_mode is not None:
            fallback_name = fitting_utils._tagged_out_csv_name(
                str(out_csv_name),
                phi_mode=str(phi_mode),
                phi_reduce=str(phi_reduce),
            )

        return fitting_utils._default_fluence_fitting_csv_path(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            delay_fs=int(delay_fs),
            time_window_fs=int(time_window_fs),
            out_csv_name=str(fallback_name),
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )


def run_delay_peak_fitting(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    fluence_mJ_cm2: float,
    time_window_fs: int,
    delays_fs: Union[int, Sequence[int], str],
    peak_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    poni_path: Optional[str] = None,
    mask_edf_path: Optional[str] = None,
    azim_windows: Optional[Sequence[Tuple[float, float]]] = None,
    azim_offset_deg: float = -90.0,
    npt: int = 1000,
    normalize_xy: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    default_eta: float = 0.3,
    fit_method: str = "leastsq",
    phi_mode: str = "separate_phi",
    phi_reduce: str = "sum",
    ref_type: Optional[str] = None,
    ref_value: Optional[Union[int, str, Sequence[int]]] = None,
    include_reference_in_output: bool = True,
    out_csv_path: Optional[str] = None,
    out_csv_name: str = "peak_fits_delay.csv",
    show_fit_figures: bool = False,
    save_fit_figures: bool = False,
    fit_figures_dir: Optional[str] = None,
    fit_figures_format: str = "png",
    fit_figures_dpi: int = 300,
    fit_figures_overwrite: bool = True,
    close_figures_after_save: bool = True,
    plot_only_success: bool = True,
    fit_oversample: int = 10,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    if peak_specs is None:
        peak_specs = PEAK_SPECS
    if not isinstance(peak_specs, dict) or len(peak_specs) == 0:
        raise ValueError("Provide a non-empty peak_specs dict (or set PEAK_SPECS at top of fitting.py).")

    phi_mode = str(phi_mode).strip()
    phi_reduce = str(phi_reduce).strip()
    if phi_mode not in ("separate_phi", "phi_avg"):
        raise ValueError(f"phi_mode must be 'separate_phi' or 'phi_avg', got: {phi_mode}")
    if phi_reduce not in ("sum", "mean"):
        raise ValueError(f"phi_reduce must be 'sum' or 'mean', got: {phi_reduce}")

    azim_windows = fitting_utils._normalize_azim_windows(azim_windows)

    fitter = fitting_utils.DelayPeakFitter(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        excitation_wl_nm=float(excitation_wl_nm),
        fluence_mJ_cm2=float(fluence_mJ_cm2),
        time_window_fs=int(time_window_fs),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize_xy=bool(normalize_xy),
        q_norm_range=tuple(q_norm_range),
        azim_offset_deg=float(azim_offset_deg),
        default_eta=float(default_eta),
        fit_method=str(fit_method),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    df = fitter.fit_delay_series(
        delays_fs=delays_fs,
        peak_specs=dict(peak_specs),
        azim_windows=azim_windows,
        compute_if_missing=bool(compute_if_missing),
        overwrite_xy=bool(overwrite_xy),
        ref_type=ref_type,
        ref_value=ref_value,
        include_reference_in_output=bool(include_reference_in_output),
        phi_mode=str(phi_mode),
        phi_reduce=str(phi_reduce),
        show_fit_figures=bool(show_fit_figures),
        save_fit_figures=bool(save_fit_figures),
        fit_figures_dir=fit_figures_dir,
        fit_figures_format=str(fit_figures_format),
        fit_figures_dpi=int(fit_figures_dpi),
        fit_figures_overwrite=bool(fit_figures_overwrite),
        close_figures_after_save=bool(close_figures_after_save),
        plot_only_success=bool(plot_only_success),
        fit_oversample=int(fit_oversample),
    )

    tagged_name = fitting_utils._tagged_out_csv_name(
        str(out_csv_name),
        phi_mode=str(phi_mode),
        phi_reduce=str(phi_reduce),
    )

    if out_csv_path is None:
        out_csv_path = str(fitter.default_csv_path(out_csv_name=tagged_name))
    else:
        out_csv_path = str(out_csv_path)
        if out_csv_path.lower().endswith(".csv"):
            out_dir = os.path.dirname(out_csv_path)
            out_csv_path = os.path.join(out_dir, tagged_name)

    csv_path = fitter.save_csv(df, path=out_csv_path)
    print("fitting csv:", csv_path)
    return df, csv_path


def plot_fit_overlay_from_csv(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    fluence_mJ_cm2: float,
    time_window_fs: int,
    poni_path: Optional[str] = None,
    mask_edf_path: Optional[str] = None,
    peak: str,
    delay_fs: Optional[int] = None,
    is_reference: bool = False,
    ref_type: Optional[str] = None,
    ref_value: Optional[Union[int, str, Sequence[int]]] = None,
    group: Optional[Union[str, float, int, Tuple[float, float]]] = None,
    phi_mode: Optional[str] = None,
    phi_reduce: str = "sum",
    out_csv_name: str = "peak_fits_delay.csv",
    csv_path: Optional[str] = None,
    ensure_csv: bool = True,
    delays_fs: Union[int, Sequence[int], str] = "all",
    peak_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    azim_windows: Optional[Sequence[Tuple[float, float]]] = None,
    azim_offset_deg: float = -90.0,
    npt: int = 1000,
    normalize_xy: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    default_eta: float = 0.3,
    fit_method: str = "leastsq",
    show: bool = True,
    save: bool = True,
    fit_figures_dir: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 300,
    save_overwrite: bool = True,
    close_after: bool = False,
    fit_oversample: int = 10,
    only_success: bool = True,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """
    Plot a single "1D + fitted overlay" after fitting, using the CSV as source of truth.

    Guarantees:
      - no re-fitting
      - uses pv_sigma stored in CSV
      - legend values (r2, height/intensity, fwhm) are recycled from CSV
    """
    csv_path = _resolve_delay_csv_path(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        excitation_wl_nm=float(excitation_wl_nm),
        fluence_mJ_cm2=float(fluence_mJ_cm2),
        time_window_fs=int(time_window_fs),
        out_csv_name=str(out_csv_name),
        csv_path=csv_path,
        phi_mode=phi_mode,
        phi_reduce=str(phi_reduce),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    if (not os.path.exists(str(csv_path))) and bool(ensure_csv):
        if peak_specs is None:
            peak_specs = PEAK_SPECS
        if azim_windows is None:
            raise ValueError("CSV missing: provide azim_windows so fitting can be run to create it.")

        _df, created_csv = run_delay_peak_fitting(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            fluence_mJ_cm2=float(fluence_mJ_cm2),
            time_window_fs=int(time_window_fs),
            delays_fs=delays_fs,
            peak_specs=dict(peak_specs),
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            azim_windows=azim_windows,
            azim_offset_deg=float(azim_offset_deg),
            npt=int(npt),
            normalize_xy=bool(normalize_xy),
            q_norm_range=tuple(q_norm_range),
            compute_if_missing=bool(compute_if_missing),
            overwrite_xy=bool(overwrite_xy),
            default_eta=float(default_eta),
            fit_method=str(fit_method),
            phi_mode=str(phi_mode) if phi_mode is not None else "separate_phi",
            phi_reduce=str(phi_reduce),
            ref_type=ref_type,
            ref_value=ref_value,
            include_reference_in_output=True,
            out_csv_path=None,
            out_csv_name=str(out_csv_name),
            show_fit_figures=False,
            save_fit_figures=False,
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )
        csv_path = str(created_csv)

    if not os.path.exists(str(csv_path)):
        raise FileNotFoundError(str(csv_path))

    df = pd.read_csv(str(csv_path))

    if phi_mode is None and "phi_mode" in df.columns and df["phi_mode"].notna().any():
        try:
            phi_mode = str(df["phi_mode"].dropna().iloc[0])
        except Exception:
            phi_mode = "separate_phi"
    if phi_mode is None:
        phi_mode = "separate_phi"
    phi_mode = str(phi_mode).strip()

    cols = fitting_utils.DEFAULT_COLS

    dsel = df[df[cols.peak_col].astype(str) == str(peak)].copy()

    if bool(only_success) and (cols.success_col in dsel.columns):
        dsel = dsel[dsel[cols.success_col].astype(bool)]

    if bool(is_reference):
        dsel = dsel[dsel[cols.is_ref_col].astype(bool)]
    else:
        if delay_fs is None:
            raise ValueError("Provide delay_fs when is_reference=False.")
        dsel = dsel[~dsel[cols.is_ref_col].astype(bool)]
        dsel = dsel[np.isfinite(dsel[cols.delay_fs_col].astype(float).values)]
        dsel = dsel[dsel[cols.delay_fs_col].astype(int) == int(delay_fs)]

    group_by = "phi_label" if (phi_mode == "phi_avg" and "phi_label" in dsel.columns) else cols.azim_str_col

    if group is not None:
        if phi_mode == "phi_avg" and group_by == "phi_label":
            group_key = fitting_utils._coerce_group_to_phi_label(group)
            dsel = dsel[dsel[group_by].astype(str) == str(group_key)]
        else:
            if isinstance(group, tuple) and len(group) == 2 and ("phi0" in dsel.columns) and ("phi1" in dsel.columns):
                g0, g1 = float(group[0]), float(group[1])
                dsel = dsel[(dsel["phi0"].astype(float) == g0) & (dsel["phi1"].astype(float) == g1)]
            else:
                dsel = dsel[dsel[group_by].astype(str) == str(group)]

    if len(dsel) == 0:
        raise ValueError("No matching row found in CSV for the requested selection.")

    if cols.azim_center_col in dsel.columns:
        dsel = dsel.sort_values([cols.azim_center_col], na_position="last")

    row = dsel.iloc[0].to_dict()

    if phi_mode == "phi_avg" and "phi_reduce" in dsel.columns and dsel["phi_reduce"].notna().any():
        try:
            pr0 = str(dsel["phi_reduce"].dropna().iloc[0]).strip()
            if pr0 in ("sum", "mean"):
                phi_reduce = pr0
        except Exception:
            pass
    phi_reduce = str(phi_reduce).strip()

    fitter = fitting_utils.DelayPeakFitter(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        excitation_wl_nm=float(excitation_wl_nm),
        fluence_mJ_cm2=float(fluence_mJ_cm2),
        time_window_fs=int(time_window_fs),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize_xy=bool(normalize_xy),
        q_norm_range=tuple(q_norm_range),
        azim_offset_deg=float(azim_offset_deg),
        default_eta=float(default_eta),
        fit_method=str(fit_method),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    if bool(is_reference):
        if ref_type is None or ref_value is None:
            raise ValueError("For is_reference=True, provide ref_type and ref_value.")
        rt = str(ref_type).strip().lower()
        if rt == "delay":
            dref = int(row.get(cols.delay_fs_col))
            dataset = fitter._delay_dataset(int(dref))
            delay_fs_val = int(dref)
            series_type = "delay"
        elif rt == "dark":
            dataset = fitter._dark_dataset(ref_value)
            delay_fs_val = None
            series_type = "dark"
        else:
            raise ValueError("ref_type must be 'delay' or 'dark'.")
    else:
        dataset = fitter._delay_dataset(int(delay_fs))
        delay_fs_val = int(delay_fs)
        series_type = "delay"

    phi0 = float(row.get("phi0", np.nan))
    phi1 = float(row.get("phi1", np.nan))
    if not (np.isfinite(phi0) and np.isfinite(phi1)):
        azs = str(row.get(cols.azim_str_col, ""))
        phi0, phi1 = fitting_utils._extract_phi_range_from_string(azs)

    azim_str, q, I = fitter.get_xy_for_phi_mode(
        dataset,
        phi0=phi0,
        phi1=phi1,
        phi_mode=str(phi_mode),
        phi_reduce=str(phi_reduce),
        compute_if_missing=bool(compute_if_missing),
        overwrite_xy=bool(overwrite_xy),
    )

    q0 = float(row.get(cols.q0_col, np.nan))
    q1 = float(row.get(cols.q1_col, np.nan))
    bg_c0 = float(row.get(cols.bg_c0_col, np.nan))
    bg_c1 = float(row.get(cols.bg_c1_col, np.nan))
    pv_center = float(row.get(cols.pos_col, np.nan))
    pv_amplitude = float(row.get(cols.area_col, np.nan))
    eta = float(row.get(cols.eta_col, default_eta))

    pv_sigma = float(row.get(getattr(cols, "sigma_col", "hkl_sigma"), np.nan))
    if not np.isfinite(pv_sigma):
        raise RuntimeError(
            "CSV does not contain a valid hkl_sigma (pv_sigma). "
            "Regenerate the CSV with a fitter version that writes hkl_sigma."
        )

    payload = fitter.build_overlay_payload_from_params(
        q=q,
        I=I,
        q0=q0,
        q1=q1,
        bg_c0=bg_c0,
        bg_c1=bg_c1,
        pv_center=pv_center,
        pv_sigma=pv_sigma,
        pv_amplitude=pv_amplitude,
        eta=eta,
        fit_oversample=int(fit_oversample),
        r2_hint=float(row.get(cols.r2_col, np.nan)),
        pv_height_hint=float(row.get(cols.i_col, np.nan)),
        pv_fwhm_hint=float(row.get(cols.fwhm_col, np.nan)),
    )

    if not bool(payload.get("success", False)):
        raise RuntimeError("Could not reconstruct overlay payload from CSV parameters.")

    phi_label = str(row.get("phi_label", fitting_utils._coerce_group_to_phi_label((phi0, phi1))))

    delay_part = "dark reference" if (series_type == "dark") else f"delay={delay_fs_val} fs"
    deg_or_full = "°" if phi_label != "Full" else ""
    az_part = (
        f"|$\\Phi$|={phi_label}{deg_or_full} (mode={phi_mode})"
        if phi_mode == "phi_avg"
        else f"$\\Phi$=({phi0:g}, {phi1:g}){deg_or_full}"
    )

    title = (
        f"{sample_name}, {int(temperature_K)} K\n"
        f"ex. wl={float(excitation_wl_nm):g} nm, flu={float(fluence_mJ_cm2):g} mJ/cm$^2$\n"
        f"tw={int(time_window_fs)} fs, {delay_part}\n"
        f"{az_part}\n"
        f"hkl=({str(peak)}), q=({float(payload.get('q0', np.nan)):.3f}, {float(payload.get('q1', np.nan)):.3f}) Å$^{{-1}}$."
    )

    save_dir_path, save_name = fitter.overlay_save_dir_and_name(
        phi_mode=str(phi_mode),
        phi_reduce=str(phi_reduce),
        phi_label=str(phi_label),
        azim_str=str(azim_str),
        peak_name=str(peak),
        is_reference=bool(is_reference),
        delay_fs_val=delay_fs_val,
        fit_figures_dir=fit_figures_dir,
    )

    style = getattr(plot_utils, "DEFAULT_STYLE", None)
    plotter = plot_utils.PeakFitOverlayPlotter(style=style)

    fig, ax, saved_path = plotter.plot_from_payload(
        payload,
        title=title,
        show=bool(show),
        save=bool(save),
        save_dir=save_dir_path,
        save_name=str(save_name),
        save_format=str(save_format),
        save_dpi=int(save_dpi),
        save_overwrite=bool(save_overwrite),
        close_after=bool(close_after),
    )

    return dict(
        csv_path=str(csv_path),
        selected_row=row,
        azim_str=str(azim_str),
        save_dir=str(save_dir_path),
        save_name=str(save_name),
        saved_path=saved_path,
        fig=fig,
        ax=ax,
    )


def plot_time_evolution(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    fluence_mJ_cm2: float,
    time_window_fs: int,
    peak: str = "110",
    _property: str = "hkl_pos",
    out_csv_name: str = "peak_fits_delay.csv",
    unit: str = "ps",
    group_by: str = "azim_range_str",
    groups: Optional[Sequence[Union[str, float, int, Tuple[float, float]]]] = None,
    only_success: bool = True,
    include_reference: bool = True,
    title: Optional[str] = None,
    as_lines: bool = False,
    delay_offset: float = 0.0,
    show_baseline_sigma: bool = False,
    baseline_sigma: float = 1.0,
    baseline_alpha: float = 0.18,
    baseline_mode: str = "errorbar",
    baseline_estimator: str = "std",
    baseline_ddof: int = 1,
    csv_path: Optional[str] = None,
    phi_mode: Optional[str] = None,
    phi_reduce: str = "sum",
    save: bool = False,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    save_fmt: str = "png",
    save_dpi: int = 300,
    save_tight: bool = True,
    close_after_save: bool = False,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    csv_path = _resolve_delay_csv_path(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        excitation_wl_nm=float(excitation_wl_nm),
        fluence_mJ_cm2=float(fluence_mJ_cm2),
        time_window_fs=int(time_window_fs),
        out_csv_name=str(out_csv_name),
        csv_path=csv_path,
        phi_mode=phi_mode,
        phi_reduce=str(phi_reduce),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    if not os.path.exists(str(csv_path)):
        raise FileNotFoundError(str(csv_path))

    df = pd.read_csv(str(csv_path))

    if phi_mode is None and "phi_mode" in df.columns and df["phi_mode"].notna().any():
        try:
            phi_mode = str(df["phi_mode"].dropna().iloc[0])
        except Exception:
            phi_mode = None

    if title is None:
        title = (
            f"{sample_name}. {temperature_K}K.\n"
            f"ex. wl={excitation_wl_nm}nm. flu={fluence_mJ_cm2} mJ/cm$^2$.\n"
            f"tw={time_window_fs}fs"
        )

    legend_title = None
    if str(phi_mode) == "phi_avg":
        if group_by == "azim_range_str":
            if "phi_label" in df.columns:
                group_by = "phi_label"
            elif "phi_center_abs" in df.columns:
                group_by = "phi_center_abs"

        if "phi_halfwidth_deg" in df.columns and df["phi_halfwidth_deg"].notna().any():
            try:
                hw = float(df["phi_halfwidth_deg"].dropna().iloc[0])
                if hw == 90 and df["phi_halfwidth_deg"].dropna().size > 1:
                    hw = float(df["phi_halfwidth_deg"].dropna().iloc[1])
                hw_s = str(int(round(hw))) if abs(hw - round(hw)) < 1e-9 else f"{hw:g}"
                legend_title = f"|$\\Phi$| ± {hw_s} [°]"
            except Exception:
                legend_title = "|$\\Phi$| window [°]"
        else:
            legend_title = "|$\\Phi$| window [°]"

        if groups is not None and group_by == "phi_label":
            groups = [fitting_utils._coerce_group_to_phi_label(g) for g in list(groups)]

    plotter = plot_utils.FitTimeEvolutionPlotter()
    out = plotter.plot(
        df,
        peak=str(peak),
        y=str(_property),
        unit=str(unit),
        group_by=str(group_by),
        groups=groups,
        only_success=bool(only_success),
        include_reference=bool(include_reference),
        title=title,
        as_lines=bool(as_lines),
        delay_offset=float(delay_offset),
        show_baseline_sigma=bool(show_baseline_sigma),
        baseline_sigma=float(baseline_sigma),
        baseline_alpha=float(baseline_alpha),
        baseline_mode=str(baseline_mode),
        baseline_estimator=str(baseline_estimator),
        baseline_ddof=int(baseline_ddof),
        legend_title=legend_title,
    )

    saved_path = None
    if bool(save):
        if hasattr(out, "savefig"):
            fig = out
        elif isinstance(out, (tuple, list)) and len(out) > 0 and hasattr(out[0], "savefig"):
            fig = out[0]
        else:
            fig = plt.gcf()

        if save_dir is None:
            default_csv = fitting_utils._default_fitting_csv_path(
                sample_name=str(sample_name),
                temperature_K=int(temperature_K),
                excitation_wl_nm=float(excitation_wl_nm),
                fluence_mJ_cm2=float(fluence_mJ_cm2),
                time_window_fs=int(time_window_fs),
                out_csv_name=str(out_csv_name),
                paths=paths,
                path_root=path_root,
                analysis_subdir=analysis_subdir,
            )
            save_dir_path = Path(default_csv).parent.parent / "figures" / "fitting"
        else:
            save_dir_path = Path(str(save_dir))

        save_dir_path.mkdir(parents=True, exist_ok=True)

        if save_name is None or str(save_name).strip() == "":
            wl_tok = f"{int(excitation_wl_nm)}"
            flu_tok = f"{float(fluence_mJ_cm2)}".replace(".", "p")
            mode_tok = (str(phi_mode).strip() if phi_mode is not None else "").replace(" ", "")
            if mode_tok == "":
                mode_tok = "auto"
            if mode_tok == "phi_avg":
                mode_tok = f"phiavg_{str(phi_reduce).strip()}"
            elif mode_tok == "separate_phi":
                mode_tok = "sepphi"

            grp_tok = str(group_by).strip()

            name = (
                f"{sample_name}_{int(temperature_K)}K_"
                f"{wl_tok}nm_{flu_tok}mJ_{int(time_window_fs)}fs_"
                f"peak{str(peak)}_{str(_property)}_{grp_tok}_{str(unit)}_mode_{mode_tok}"
            )
            save_name = fitting_utils._sanitize_path_token(name)

        fmt = str(save_fmt).lstrip(".").lower()
        out_path = save_dir_path / f"{save_name}.{fmt}"

        if bool(save_tight):
            fig.savefig(str(out_path), dpi=int(save_dpi), bbox_inches="tight")
        else:
            fig.savefig(str(out_path), dpi=int(save_dpi))

        saved_path = str(out_path)
        print("saved figure:", saved_path)

        if bool(close_after_save):
            try:
                plt.close(fig)
            except Exception:
                pass

    return (out, saved_path) if saved_path is not None else out


def plot_time_evolution_multi(
    *,
    experiments: Sequence[Dict[str, object]],
    peak: str = "110",
    _property: str = "hkl_pos",
    out_csv_name: str = "peak_fits_delay.csv",
    unit: str = "ps",
    phi_mode: Optional[str] = None,
    phi_reduce: str = "sum",
    phi_window: Optional[Union[str, float, int, Tuple[float, float]]] = None,
    only_success: bool = True,
    include_reference: bool = True,
    title: Optional[str] = None,
    as_lines: bool = False,
    delay_offset: Optional[float] = None,
    show_baseline_sigma: bool = True,
    baseline_sigma: float = 1.0,
    baseline_alpha: float = 0.18,
    baseline_mode: str = "errorbar",
    baseline_estimator: str = "std",
    baseline_ddof: int = 1,
    norm_min_max: bool = False,
    delay_for_norm_max: Optional[float] = None,
    cmap: Optional[str] = None,
    save: bool = False,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    save_fmt: str = "png",
    save_dpi: int = 300,
    save_overwrite: bool = True,
    close_after_save: bool = False,
    show: bool = True,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    if not isinstance(experiments, (list, tuple)) or len(experiments) == 0:
        raise ValueError("experiments must be a non-empty list of experiment dicts.")

    u = str(unit).strip().lower()
    if u not in ("fs", "ps"):
        raise ValueError("unit must be 'fs' or 'ps'.")

    pr = str(phi_reduce).strip()
    if pr not in ("sum", "mean"):
        raise ValueError(f"phi_reduce must be 'sum' or 'mean', got: {phi_reduce}")

    pm_req = None if phi_mode is None else str(phi_mode).strip()
    if pm_req is not None and pm_req not in ("separate_phi", "phi_avg"):
        raise ValueError(f"phi_mode must be None, 'separate_phi', or 'phi_avg' (got {phi_mode!r}).")

    plotter_paths = paths
    if plotter_paths is None and path_root is not None and analysis_subdir is not None:
        plotter_paths = AnalysisPaths(
            path_root=Path(path_root),
            analysis_subdir=str(analysis_subdir),
        )

    plotter = plot_utils.FitTimeEvolutionMultiPlotter(
        style=getattr(plot_utils, "DEFAULT_STYLE", None),
        paths=plotter_paths,
    )
    cols = fitting_utils.DEFAULT_COLS

    series_list = []
    inferred_phi_modes = []

    def _is_merged_item(item: dict) -> bool:
        if not isinstance(item, dict):
            return False
        for k in ("merge", "experiments", "parts", "series"):
            v = item.get(k, None)
            if isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(x, dict) for x in v):
                return True
        return False

    def _extract_members_and_meta(item: dict):
        if _is_merged_item(item):
            members = None
            for k in ("merge", "experiments", "parts", "series"):
                v = item.get(k, None)
                if isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(x, dict) for x in v):
                    members = list(v)
                    break
            meta = dict(item)
            for k in ("merge", "experiments", "parts", "series"):
                meta.pop(k, None)
            return members, meta, True
        return [item], {}, False

    def _merged_negative_delay_baseline(x_vals, y_vals):
        x_vals = np.asarray(x_vals, float)
        y_vals = np.asarray(y_vals, float)

        m = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals < 0.0)
        if not np.any(m):
            return np.nan, np.nan

        yneg = y_vals[m]
        if yneg.size == 0:
            return np.nan, np.nan

        y0 = float(np.mean(yneg))

        est = str(baseline_estimator).strip().lower()
        if yneg.size >= 2:
            resid = yneg - y0
            if est == "mad":
                med = float(np.median(resid))
                mad = float(np.median(np.abs(resid - med)))
                sig = float(1.4826 * mad)
            else:
                sig = float(np.std(resid, ddof=int(baseline_ddof)))
        else:
            sig = 0.0

        if np.isfinite(sig):
            sig = float(baseline_sigma) * float(sig)

        return float(y0), float(sig)

    def _member_default_label(exp: dict) -> str:
        lab = str(exp.get("label", "")).strip()
        if lab != "":
            return lab
        return plotter.default_label_from_experiment(exp)

    def _merged_time_window_text(members: Sequence[dict]) -> str:
        vals = []
        for mem in list(members):
            try:
                vals.append(int(float(mem.get("time_window_fs"))))
            except Exception:
                pass

        vals = sorted(set(vals))
        if len(vals) == 0:
            return ""
        if len(vals) == 1:
            return str(vals[0])
        return "[" + ", ".join(str(v) for v in vals) + "]"

    def _append_time_window_to_label(base: str, tw_txt: str) -> str:
        base = str(base).strip()
        tw_txt = str(tw_txt).strip()

        if tw_txt == "":
            return base
        if base == "":
            return tw_txt
        if base == tw_txt or base.endswith(f", {tw_txt}"):
            return base
        return f"{base}, {tw_txt}"

    def _merged_base_label(members: Sequence[dict], series_meta: dict) -> str:
        outer_label = str(series_meta.get("label", "")).strip()
        if outer_label != "":
            return outer_label

        if len(members) == 0:
            return "merged"

        keys = ("sample_name", "temperature_K", "excitation_wl_nm", "fluence_mJ_cm2")
        same_core = True
        ref0 = members[0]
        for mem in members[1:]:
            for k in keys:
                try:
                    if str(mem.get(k)) != str(ref0.get(k)):
                        same_core = False
                        break
                except Exception:
                    same_core = False
                    break
            if not same_core:
                break

        if same_core:
            sn = str(ref0.get("sample_name", "")).strip()
            try:
                tK = int(float(ref0.get("temperature_K", 0)))
            except Exception:
                tK = ref0.get("temperature_K", "")
            try:
                wl = int(float(ref0.get("excitation_wl_nm", 0)))
            except Exception:
                wl = ref0.get("excitation_wl_nm", "")
            try:
                flu = float(ref0.get("fluence_mJ_cm2", 0))
                flu_s = f"{flu:g}"
            except Exception:
                flu_s = str(ref0.get("fluence_mJ_cm2", ""))

            return f"{sn}, {tK}, {wl}, {flu_s}"

        return " + ".join([str(_member_default_label(mem)) for mem in list(members)])

    def _merged_label(members: Sequence[dict], series_meta: dict) -> str:
        base = _merged_base_label(members, series_meta)
        tw_txt = _merged_time_window_text(members)
        return _append_time_window_to_label(base, tw_txt)

    def _resolve_member_fragment(exp: dict, *, offset_override_in_unit=None) -> dict:
        if not isinstance(exp, dict):
            raise ValueError("Each experiment/member must be a dict.")

        sample_name = str(exp.get("sample_name"))
        temperature_K = int(exp.get("temperature_K"))
        excitation_wl_nm = float(exp.get("excitation_wl_nm"))
        fluence_mJ_cm2 = float(exp.get("fluence_mJ_cm2"))
        time_window_fs = int(exp.get("time_window_fs"))

        csv_path_local = exp.get("csv_path", None)
        if csv_path_local is not None and str(csv_path_local).strip() != "":
            csv_path_local = str(csv_path_local)
        else:
            csv_path_local = fitting_utils.resolve_delay_fitting_csv_path(
                sample_name=sample_name,
                temperature_K=temperature_K,
                excitation_wl_nm=excitation_wl_nm,
                fluence_mJ_cm2=fluence_mJ_cm2,
                time_window_fs=time_window_fs,
                out_csv_name=str(out_csv_name),
                phi_mode=pm_req,
                phi_reduce=str(pr),
                **_resolve_exp_path_kwargs(
                    exp,
                    paths=paths,
                    path_root=path_root,
                    analysis_subdir=analysis_subdir,
                ),
            )

        if not os.path.exists(str(csv_path_local)):
            raise FileNotFoundError(str(csv_path_local))

        df = pd.read_csv(str(csv_path_local))

        pm_here = pm_req
        if pm_here is None:
            if "phi_mode" in df.columns and df["phi_mode"].notna().any():
                try:
                    pm_here = str(df["phi_mode"].dropna().iloc[0]).strip()
                except Exception:
                    pm_here = None
            if pm_here is None:
                low = str(csv_path_local).lower()
                pm_here = "phi_avg" if ("phiavg" in low or "phi_avg" in low) else "separate_phi"

        pm_here = str(pm_here).strip()
        if pm_here not in ("separate_phi", "phi_avg"):
            raise ValueError(f"Invalid phi_mode inferred for experiment: {pm_here}")

        dpk = df[df[cols.peak_col].astype(str) == str(peak)].copy()
        if len(dpk) == 0:
            raise ValueError(f"No rows found for peak={peak!r} in {csv_path_local}")

        group_by_local = "phi_label" if (pm_here == "phi_avg" and "phi_label" in dpk.columns) else cols.azim_str_col

        if phi_window is None:
            if pm_here == "phi_avg" and group_by_local == "phi_label":
                phi_key = "Full"
            else:
                if group_by_local not in dpk.columns or dpk[group_by_local].dropna().empty:
                    raise ValueError(f"Could not infer a default group in {csv_path_local}. Provide phi_window=...")
                phi_key = str(dpk[group_by_local].dropna().iloc[0])
        else:
            if pm_here == "phi_avg" and group_by_local == "phi_label":
                phi_key = fitting_utils._coerce_group_to_phi_label(phi_window)
            else:
                phi_key = str(phi_window)

        dgrp = dpk[dpk[group_by_local].astype(str) == str(phi_key)].copy()
        if len(dgrp) == 0:
            raise ValueError(f"No rows with {group_by_local}={phi_key!r} in {csv_path_local}")

        dly_fs_all = pd.to_numeric(dgrp[cols.delay_fs_col], errors="coerce").values.astype(float)
        y_all = pd.to_numeric(dgrp[str(_property)], errors="coerce").values.astype(float)

        off = plotter.delay_offset_in_unit(exp, u, global_override=offset_override_in_unit)

        if u == "ps":
            x_all = dly_fs_all * 1e-3 + off
            xerr_half = 0.5 * float(time_window_fs) * 1e-3
        else:
            x_all = dly_fs_all + off
            xerr_half = 0.5 * float(time_window_fs)

        dplot = dgrp.copy()
        if bool(only_success) and (cols.success_col in dplot.columns):
            dplot = dplot[dplot[cols.success_col].astype(bool)]
        if not bool(include_reference) and (cols.is_ref_col in dplot.columns):
            dplot = dplot[~dplot[cols.is_ref_col].astype(bool)]

        dly_fs_plot = pd.to_numeric(dplot[cols.delay_fs_col], errors="coerce").values.astype(float)
        y_plot = pd.to_numeric(dplot[str(_property)], errors="coerce").values.astype(float)

        if u == "ps":
            x_plot = dly_fs_plot * 1e-3 + off
        else:
            x_plot = dly_fs_plot + off

        x_plot = np.asarray(x_plot, float)
        y_plot = np.asarray(y_plot, float)
        m_plot = np.isfinite(x_plot) & np.isfinite(y_plot)
        x_plot = x_plot[m_plot]
        y_plot = y_plot[m_plot]
        xerr_plot = np.full_like(x_plot, float(xerr_half), dtype=float)

        return dict(
            experiment=exp,
            csv_path=str(csv_path_local),
            phi_mode=str(pm_here),
            group_by=str(group_by_local),
            group=str(phi_key),
            x=np.asarray(x_plot, float),
            y=np.asarray(y_plot, float),
            xerr=np.asarray(xerr_plot, float),
            x_all=np.asarray(x_all, float),
            y_all=np.asarray(y_all, float),
            dgrp=dgrp.copy(),
            label_single=_member_default_label(exp),
        )

    for item in list(experiments):
        if not isinstance(item, dict):
            raise ValueError("Each item in experiments must be a dict.")

        members, series_meta, is_merged = _extract_members_and_meta(item)

        if isinstance(series_meta, dict) and any(k in series_meta for k in ("delay_offset_ps", "delay_offset_fs", "delay_offset")):
            outer_offset_override = plotter.delay_offset_in_unit(series_meta, u, global_override=delay_offset)
        else:
            outer_offset_override = delay_offset

        fragments = [
            _resolve_member_fragment(mem, offset_override_in_unit=outer_offset_override)
            for mem in list(members)
        ]

        pm_list = [str(fr["phi_mode"]) for fr in fragments]
        pm0 = pm_list[0]
        if any(str(pm) != str(pm0) for pm in pm_list):
            raise ValueError("Merged series contains experiments with mixed phi_mode.")
        inferred_phi_modes.append(pm0)

        group_by_eff = str(fragments[0]["group_by"])
        group_eff = str(fragments[0]["group"])
        if any(str(fr["group_by"]) != group_by_eff for fr in fragments):
            raise ValueError("Merged series contains experiments resolved to different group_by values.")
        if any(str(fr["group"]) != group_eff for fr in fragments):
            raise ValueError("Merged series contains experiments resolved to different phi/group selection.")

        x_cat = np.concatenate([np.asarray(fr["x"], float) for fr in fragments]) if len(fragments) else np.array([], float)
        y_cat = np.concatenate([np.asarray(fr["y"], float) for fr in fragments]) if len(fragments) else np.array([], float)
        xerr_cat = np.concatenate([np.asarray(fr["xerr"], float) for fr in fragments]) if len(fragments) else np.array([], float)

        m_cat = np.isfinite(x_cat) & np.isfinite(y_cat)
        x_cat = x_cat[m_cat]
        y_cat = y_cat[m_cat]
        if xerr_cat.size == m_cat.size:
            xerr_cat = xerr_cat[m_cat]
        else:
            xerr_cat = np.full_like(x_cat, np.nan, dtype=float)

        if x_cat.size == 0:
            continue

        order = np.argsort(x_cat)
        x_cat = x_cat[order]
        y_cat = y_cat[order]
        xerr_cat = xerr_cat[order]

        x_all_cat = np.concatenate([np.asarray(fr["x_all"], float) for fr in fragments]) if len(fragments) else np.array([], float)
        y_all_cat = np.concatenate([np.asarray(fr["y_all"], float) for fr in fragments]) if len(fragments) else np.array([], float)

        if is_merged:
            lab = _merged_label(list(members), series_meta if isinstance(series_meta, dict) else {})
        else:
            lab = _append_time_window_to_label(
                str(fragments[0]["label_single"]),
                _merged_time_window_text(list(members)),
            )

        baseline_y0 = np.nan
        baseline_sig = np.nan

        if bool(show_baseline_sigma):
            if is_merged:
                baseline_y0, baseline_sig = _merged_negative_delay_baseline(x_all_cat, y_all_cat)
            else:
                fr0 = fragments[0]
                exp0 = fr0["experiment"]
                ref_type = str(exp0.get("ref_type", "dark"))
                ref_value = exp0.get("ref_value", None)

                baseline_y0, baseline_sig = plotter.baseline_from_reference(
                    df_sel=fr0["dgrp"],
                    x=np.asarray(fr0["x_all"], float),
                    y=np.asarray(fr0["y_all"], float),
                    prop=str(_property),
                    cols=cols,
                    ref_type=ref_type,
                    ref_value=ref_value,
                    sigma_scale=float(baseline_sigma),
                    estimator=str(baseline_estimator),
                    ddof=int(baseline_ddof),
                )

        norm_target = delay_for_norm_max
        if isinstance(series_meta, dict) and (series_meta.get("delay_for_norm_max", None) is not None):
            try:
                norm_target = float(series_meta.get("delay_for_norm_max"))
            except Exception:
                pass
        elif (not is_merged) and isinstance(fragments[0]["experiment"], dict):
            try:
                v0 = fragments[0]["experiment"].get("delay_for_norm_max", None)
                if v0 is not None:
                    norm_target = float(v0)
            except Exception:
                pass

        if bool(norm_min_max):
            mask = np.isfinite(x_cat) & np.isfinite(y_cat)
            if np.any(mask):
                if norm_target is None:
                    ymax = y_cat[mask][np.argmax(y_cat[mask])]
                else:
                    try:
                        ymax = y_cat[mask][np.argmin(np.abs(x_cat[mask] - float(norm_target)))]
                    except Exception:
                        ymax = y_cat[mask][np.argmax(y_cat[mask])]

                y0_orig = float(baseline_y0)
                if np.isfinite(y0_orig) and np.isfinite(ymax) and (ymax != y0_orig):
                    y_cat = (y_cat - y0_orig) / (ymax - y0_orig)

                    if bool(show_baseline_sigma):
                        y_all_norm = (y_all_cat - y0_orig) / (ymax - y0_orig)

                        if is_merged:
                            baseline_y0, baseline_sig = _merged_negative_delay_baseline(x_all_cat, y_all_norm)
                        else:
                            fr0 = fragments[0]
                            exp0 = fr0["experiment"]
                            ref_type = str(exp0.get("ref_type", "dark"))
                            ref_value = exp0.get("ref_value", None)

                            baseline_y0, baseline_sig = plotter.baseline_from_reference(
                                df_sel=fr0["dgrp"],
                                x=np.asarray(fr0["x_all"], float),
                                y=np.asarray(y_all_norm, float),
                                prop=str(_property),
                                cols=cols,
                                ref_type=ref_type,
                                ref_value=ref_value,
                                sigma_scale=float(baseline_sigma),
                                estimator=str(baseline_estimator),
                                ddof=int(baseline_ddof),
                            )

        s = dict(
            x=np.asarray(x_cat, float),
            y=np.asarray(y_cat, float),
            xerr=np.asarray(xerr_cat, float),
            label=str(lab),
            _group_by=str(group_by_eff),
            _group=str(group_eff),
            _is_merged=bool(is_merged),
            _n_members=int(len(fragments)),
        )

        if bool(show_baseline_sigma):
            s["baseline_y0"] = float(baseline_y0)
            s["baseline_sig"] = float(baseline_sig)

        series_list.append(s)

    if len(series_list) == 0:
        raise ValueError("No series to plot after filtering.")

    if pm_req is None:
        pm0 = inferred_phi_modes[0]
        if any(str(p) != str(pm0) for p in inferred_phi_modes):
            raise ValueError(
                "Experiments have mixed phi_mode. "
                "Pass phi_mode=... explicitly (and ensure all CSVs match)."
            )
        pm_eff = pm0
    else:
        pm_eff = str(pm_req)

    if title is None:
        gb = str(series_list[0].get("_group_by", "")) if len(series_list) else ""
        gk = str(series_list[0].get("_group", "")) if len(series_list) else ""
        title = plotter.title_default(peak=str(peak), prop=str(_property), group_by=gb, group_key=gk)

    if bool(save) and (save_name is None or str(save_name).strip() == ""):
        gb = str(series_list[0].get("_group_by", "")) if len(series_list) else ""
        gk = str(series_list[0].get("_group", "")) if len(series_list) else ""
        save_name = plotter.default_save_name(
            peak=str(peak),
            prop=str(_property),
            group_by=gb,
            group_key=gk,
            phi_mode=str(pm_eff),
            phi_reduce=str(pr),
            n_series=len(series_list),
            unit=str(u),
        )

    xlabel = "Delay [ps]" if u == "ps" else "Delay [fs]"

    if bool(save) and (save_dir is None or str(save_dir).strip() == ""):
        if paths is not None:
            save_dir = str(Path(paths.analysis_root) / "general_figures")
        elif path_root is not None and analysis_subdir is not None:
            save_dir = str(Path(path_root) / Path(analysis_subdir) / "general_figures")

    fig, ax, saved_path = plotter.plot(
        series_list,
        title=str(title),
        xlabel=str(xlabel),
        ylabel=plotter.ylabel_for_property(str(_property), str(peak)),
        legend_title=plotter.legend_title_default(),
        as_lines=bool(as_lines),
        show_baseline_sigma=bool(show_baseline_sigma),
        baseline_mode=str(baseline_mode),
        baseline_alpha=float(baseline_alpha),
        cmap=(str(cmap) if cmap is not None else None),
        show=bool(show),
        save=bool(save),
        save_dir=(str(save_dir) if save_dir is not None else None),
        save_name=(str(save_name) if save_name is not None else None),
        save_format=str(save_fmt),
        save_dpi=int(save_dpi),
        save_overwrite=bool(save_overwrite),
        close_after=bool(close_after_save),
        legend_outside=True,
    )

    return dict(
        fig=fig,
        ax=ax,
        saved_path=saved_path,
        series=series_list,
        phi_mode=str(pm_eff),
        phi_reduce=str(pr),
        unit=str(u),
        peak=str(peak),
        y=str(_property),
    )


def run_fluence_peak_fitting(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    delay_fs: int,
    time_window_fs: int,
    fluences_mJ_cm2: Union[float, int, Sequence[Union[float, int]], str],
    peak_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    poni_path: Optional[str] = None,
    mask_edf_path: Optional[str] = None,
    azim_windows: Optional[Sequence[Tuple[float, float]]] = None,
    azim_offset_deg: float = -90.0,
    npt: int = 1000,
    normalize_xy: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    default_eta: float = 0.3,
    fit_method: str = "leastsq",
    phi_mode: str = "separate_phi",
    phi_reduce: str = "sum",
    ref_type: Optional[str] = None,
    ref_value: Optional[Union[float, int, str, Sequence[int]]] = None,
    include_reference_in_output: bool = True,
    out_csv_path: Optional[str] = None,
    out_csv_name: str = "peak_fits_fluence.csv",
    show_fit_figures: bool = False,
    save_fit_figures: bool = False,
    fit_figures_dir: Optional[str] = None,
    fit_figures_format: str = "png",
    fit_figures_dpi: int = 300,
    fit_figures_overwrite: bool = True,
    close_figures_after_save: bool = True,
    plot_only_success: bool = True,
    fit_oversample: int = 10,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    if peak_specs is None:
        peak_specs = PEAK_SPECS
    if not isinstance(peak_specs, dict) or len(peak_specs) == 0:
        raise ValueError("Provide a non-empty peak_specs dict (or set PEAK_SPECS at top of fitting.py).")

    phi_mode = str(phi_mode).strip()
    phi_reduce = str(phi_reduce).strip()
    if phi_mode not in ("separate_phi", "phi_avg"):
        raise ValueError(f"phi_mode must be 'separate_phi' or 'phi_avg', got: {phi_mode}")
    if phi_reduce not in ("sum", "mean"):
        raise ValueError(f"phi_reduce must be 'sum' or 'mean', got: {phi_reduce}")

    azim_windows = fitting_utils._normalize_azim_windows(azim_windows)

    fitter = fitting_utils.FluencePeakFitter(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        excitation_wl_nm=float(excitation_wl_nm),
        delay_fs=int(delay_fs),
        time_window_fs=int(time_window_fs),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize_xy=bool(normalize_xy),
        q_norm_range=tuple(q_norm_range),
        azim_offset_deg=float(azim_offset_deg),
        default_eta=float(default_eta),
        fit_method=str(fit_method),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    df = fitter.fit_fluence_series(
        fluences_mJ_cm2=fluences_mJ_cm2,
        peak_specs=dict(peak_specs),
        azim_windows=azim_windows,
        compute_if_missing=bool(compute_if_missing),
        overwrite_xy=bool(overwrite_xy),
        ref_type=ref_type,
        ref_value=ref_value,
        include_reference_in_output=bool(include_reference_in_output),
        phi_mode=str(phi_mode),
        phi_reduce=str(phi_reduce),
        show_fit_figures=bool(show_fit_figures),
        save_fit_figures=bool(save_fit_figures),
        fit_figures_dir=fit_figures_dir,
        fit_figures_format=str(fit_figures_format),
        fit_figures_dpi=int(fit_figures_dpi),
        fit_figures_overwrite=bool(fit_figures_overwrite),
        close_figures_after_save=bool(close_figures_after_save),
        plot_only_success=bool(plot_only_success),
        fit_oversample=int(fit_oversample),
    )

    tagged_name = fitting_utils._tagged_out_csv_name(
        str(out_csv_name),
        phi_mode=str(phi_mode),
        phi_reduce=str(phi_reduce),
    )

    if out_csv_path is None:
        out_csv_path = str(fitter.default_csv_path(out_csv_name=tagged_name))
    else:
        out_csv_path = str(out_csv_path)
        if out_csv_path.lower().endswith(".csv"):
            out_dir = os.path.dirname(out_csv_path)
            out_csv_path = os.path.join(out_dir, tagged_name)

    csv_path = fitter.save_csv(df, path=out_csv_path)
    print("fitting csv:", csv_path)
    return df, csv_path


def plot_fit_overlay_from_csv_fluence(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    delay_fs: int,
    time_window_fs: int,
    poni_path: Optional[str] = None,
    mask_edf_path: Optional[str] = None,
    peak: str,
    fluence_mJ_cm2: Optional[float] = None,
    is_reference: bool = False,
    ref_type: Optional[str] = None,
    ref_value: Optional[Union[float, int, str, Sequence[int]]] = None,
    group: Optional[Union[str, float, int, Tuple[float, float]]] = None,
    phi_mode: Optional[str] = None,
    phi_reduce: str = "sum",
    out_csv_name: str = "peak_fits_fluence.csv",
    csv_path: Optional[str] = None,
    ensure_csv: bool = True,
    fluences_mJ_cm2: Union[float, int, Sequence[Union[float, int]], str] = "all",
    peak_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    azim_windows: Optional[Sequence[Tuple[float, float]]] = None,
    azim_offset_deg: float = -90.0,
    npt: int = 1000,
    normalize_xy: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    default_eta: float = 0.3,
    fit_method: str = "leastsq",
    show: bool = True,
    save: bool = True,
    fit_figures_dir: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 300,
    save_overwrite: bool = True,
    close_after: bool = False,
    fit_oversample: int = 10,
    only_success: bool = True,
    fluence_tol: float = 1e-9,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """
    Plot a single "1D + fitted overlay" after fitting, using the CSV as source of truth.
    Guarantees:
      - no re-fitting
      - uses pv_sigma stored in CSV
      - legend values (r2, height/intensity, fwhm) are recycled from CSV
    """
    csv_path = _resolve_fluence_csv_path(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        excitation_wl_nm=float(excitation_wl_nm),
        delay_fs=int(delay_fs),
        time_window_fs=int(time_window_fs),
        out_csv_name=str(out_csv_name),
        csv_path=csv_path,
        phi_mode=phi_mode,
        phi_reduce=str(phi_reduce),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    if (not os.path.exists(str(csv_path))) and bool(ensure_csv):
        if peak_specs is None:
            peak_specs = PEAK_SPECS
        if azim_windows is None:
            raise ValueError("CSV missing: provide azim_windows so fitting can be run to create it.")

        _df, created_csv = run_fluence_peak_fitting(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            delay_fs=int(delay_fs),
            time_window_fs=int(time_window_fs),
            fluences_mJ_cm2=fluences_mJ_cm2,
            peak_specs=dict(peak_specs),
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            azim_windows=azim_windows,
            azim_offset_deg=float(azim_offset_deg),
            npt=int(npt),
            normalize_xy=bool(normalize_xy),
            q_norm_range=tuple(q_norm_range),
            compute_if_missing=bool(compute_if_missing),
            overwrite_xy=bool(overwrite_xy),
            default_eta=float(default_eta),
            fit_method=str(fit_method),
            phi_mode=str(phi_mode) if phi_mode is not None else "separate_phi",
            phi_reduce=str(phi_reduce),
            ref_type=ref_type,
            ref_value=ref_value,
            include_reference_in_output=True,
            out_csv_path=None,
            out_csv_name=str(out_csv_name),
            show_fit_figures=False,
            save_fit_figures=False,
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )
        csv_path = str(created_csv)

    if not os.path.exists(str(csv_path)):
        raise FileNotFoundError(str(csv_path))

    df = pd.read_csv(str(csv_path))

    if phi_mode is None and "phi_mode" in df.columns and df["phi_mode"].notna().any():
        try:
            phi_mode = str(df["phi_mode"].dropna().iloc[0])
        except Exception:
            phi_mode = "separate_phi"
    if phi_mode is None:
        phi_mode = "separate_phi"
    phi_mode = str(phi_mode).strip()

    cols = fitting_utils.DEFAULT_COLS

    dsel = df[df[cols.peak_col].astype(str) == str(peak)].copy()

    if bool(only_success) and (cols.success_col in dsel.columns):
        dsel = dsel[dsel[cols.success_col].astype(bool)]

    if bool(is_reference):
        dsel = dsel[dsel[cols.is_ref_col].astype(bool)]
    else:
        if fluence_mJ_cm2 is None:
            raise ValueError("Provide fluence_mJ_cm2 when is_reference=False.")
        dsel = dsel[~dsel[cols.is_ref_col].astype(bool)]
        if "fluence_mJ_cm2" not in dsel.columns:
            raise ValueError("CSV does not contain 'fluence_mJ_cm2' column required for fluence selection.")
        fcol = pd.to_numeric(dsel["fluence_mJ_cm2"], errors="coerce").values
        dsel = dsel[np.isfinite(fcol) & (np.abs(fcol - float(fluence_mJ_cm2)) <= float(fluence_tol))]

    group_by = "phi_label" if (phi_mode == "phi_avg" and "phi_label" in dsel.columns) else cols.azim_str_col

    if group is not None:
        if phi_mode == "phi_avg" and group_by == "phi_label":
            group_key = fitting_utils._coerce_group_to_phi_label(group)
            dsel = dsel[dsel[group_by].astype(str) == str(group_key)]
        else:
            if isinstance(group, tuple) and len(group) == 2 and ("phi0" in dsel.columns) and ("phi1" in dsel.columns):
                g0, g1 = float(group[0]), float(group[1])
                dsel = dsel[(dsel["phi0"].astype(float) == g0) & (dsel["phi1"].astype(float) == g1)]
            else:
                dsel = dsel[dsel[group_by].astype(str) == str(group)]

    if len(dsel) == 0:
        raise ValueError("No matching row found in CSV for the requested selection.")

    if cols.azim_center_col in dsel.columns:
        dsel = dsel.sort_values([cols.azim_center_col], na_position="last")

    row = dsel.iloc[0].to_dict()

    if phi_mode == "phi_avg" and "phi_reduce" in dsel.columns and dsel["phi_reduce"].notna().any():
        try:
            pr0 = str(dsel["phi_reduce"].dropna().iloc[0]).strip()
            if pr0 in ("sum", "mean"):
                phi_reduce = pr0
        except Exception:
            pass
    phi_reduce = str(phi_reduce).strip()

    fitter = fitting_utils.FluencePeakFitter(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        excitation_wl_nm=float(excitation_wl_nm),
        delay_fs=int(delay_fs),
        time_window_fs=int(time_window_fs),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize_xy=bool(normalize_xy),
        q_norm_range=tuple(q_norm_range),
        azim_offset_deg=float(azim_offset_deg),
        default_eta=float(default_eta),
        fit_method=str(fit_method),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    series_type = str(row.get(cols.series_type_col, "fluence")).strip().lower()

    if bool(is_reference):
        if ref_type is None or ref_value is None:
            raise ValueError("For is_reference=True, provide ref_type and ref_value.")
        rt = str(ref_type).strip().lower()

        if rt == "fluence":
            frow = float(row.get("fluence_mJ_cm2", np.nan))
            fref = frow if np.isfinite(frow) else float(ref_value)
            dataset = fitter._fluence_dataset(float(fref))
            fluence_val = float(fref)
            series_type = "fluence"
        elif rt == "dark":
            dataset = fitter._dark_dataset(ref_value)
            fluence_val = None
            series_type = "dark"
        else:
            raise ValueError("ref_type must be 'fluence' or 'dark'.")
    else:
        dataset = fitter._fluence_dataset(float(fluence_mJ_cm2))
        fluence_val = float(fluence_mJ_cm2)
        series_type = "fluence"

    phi0 = float(row.get("phi0", np.nan))
    phi1 = float(row.get("phi1", np.nan))
    if not (np.isfinite(phi0) and np.isfinite(phi1)):
        azs = str(row.get(cols.azim_str_col, ""))
        phi0, phi1 = fitting_utils._extract_phi_range_from_string(azs)

    azim_str, q, I = fitter.get_xy_for_phi_mode(
        dataset,
        phi0=phi0,
        phi1=phi1,
        phi_mode=str(phi_mode),
        phi_reduce=str(phi_reduce),
        compute_if_missing=bool(compute_if_missing),
        overwrite_xy=bool(overwrite_xy),
    )

    q0 = float(row.get(cols.q0_col, np.nan))
    q1 = float(row.get(cols.q1_col, np.nan))
    bg_c0 = float(row.get(cols.bg_c0_col, np.nan))
    bg_c1 = float(row.get(cols.bg_c1_col, np.nan))
    pv_center = float(row.get(cols.pos_col, np.nan))
    pv_amplitude = float(row.get(cols.area_col, np.nan))
    eta = float(row.get(cols.eta_col, default_eta))

    pv_sigma = float(row.get(getattr(cols, "sigma_col", "hkl_sigma"), np.nan))
    if not np.isfinite(pv_sigma):
        raise RuntimeError(
            "CSV does not contain a valid hkl_sigma (pv_sigma). "
            "Regenerate the CSV with a fitter version that writes hkl_sigma."
        )

    payload = fitter.build_overlay_payload_from_params(
        q=q,
        I=I,
        q0=q0,
        q1=q1,
        bg_c0=bg_c0,
        bg_c1=bg_c1,
        pv_center=pv_center,
        pv_sigma=pv_sigma,
        pv_amplitude=pv_amplitude,
        eta=eta,
        fit_oversample=int(fit_oversample),
        r2_hint=float(row.get(cols.r2_col, np.nan)),
        pv_height_hint=float(row.get(cols.i_col, np.nan)),
        pv_fwhm_hint=float(row.get(cols.fwhm_col, np.nan)),
    )

    if not bool(payload.get("success", False)):
        raise RuntimeError("Could not reconstruct overlay payload from CSV parameters.")

    phi_label = str(row.get("phi_label", fitting_utils._coerce_group_to_phi_label((phi0, phi1))))

    delay_part = f"delay={int(delay_fs)} fs"
    flu_part = "dark reference" if (series_type == "dark") else f"fluence={float(fluence_val):g} mJ/cm$^2$"

    deg_or_full = "°" if phi_label != "Full" else ""
    az_part = (
        f"|$\\Phi$|={phi_label}{deg_or_full} (mode={phi_mode})"
        if phi_mode == "phi_avg"
        else f"$\\Phi$=({phi0:g}, {phi1:g}){deg_or_full}"
    )

    title = (
        f"{sample_name}, {int(temperature_K)} K\n"
        f"ex. wl={float(excitation_wl_nm):g} nm, {delay_part}\n"
        f"tw={int(time_window_fs)} fs, {flu_part}\n"
        f"{az_part}\n"
        f"hkl=({str(peak)}), q=({float(payload.get('q0', np.nan)):.3f}, {float(payload.get('q1', np.nan)):.3f}) Å$^{{-1}}$."
    )

    save_dir_path, save_name = fitter.overlay_save_dir_and_name_fluence(
        phi_mode=str(phi_mode),
        phi_reduce=str(phi_reduce),
        phi_label=str(phi_label),
        azim_str=str(azim_str),
        peak_name=str(peak),
        is_reference=bool(is_reference),
        series_type=str(series_type),
        fluence_mJ_cm2_val=fluence_val,
        fit_figures_dir=fit_figures_dir,
    )

    style = getattr(plot_utils, "DEFAULT_STYLE", None)
    plotter = plot_utils.PeakFitOverlayPlotter(style=style)

    fig, ax, saved_path = plotter.plot_from_payload(
        payload,
        title=title,
        show=bool(show),
        save=bool(save),
        save_dir=save_dir_path,
        save_name=str(save_name),
        save_format=str(save_format),
        save_dpi=int(save_dpi),
        save_overwrite=bool(save_overwrite),
        close_after=bool(close_after),
    )

    return dict(
        csv_path=str(csv_path),
        selected_row=row,
        azim_str=str(azim_str),
        save_dir=str(save_dir_path),
        save_name=str(save_name),
        saved_path=saved_path,
        fig=fig,
        ax=ax,
    )


def plot_fluence_evolution(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    delay_fs: int,
    time_window_fs: int,
    peak: str,
    _property: str,
    unit: str = "mJ/cm$^2$",
    out_csv_name: str = "peak_fits_fluence.csv",
    groups=None,
    phi_mode: str = "separate_phi",
    phi_reduce: str = "sum",
    as_lines: bool = False,
    fluence_offset: float = 0.0,
    show_baseline_sigma: bool = False,
    baseline_sigma: float = 1.0,
    baseline_alpha: float = 0.18,
    baseline_mode: str = "band",
    baseline_estimator: str = "std",
    baseline_ddof: int = 1,
    title: str | None = None,
    save: bool = False,
    save_dir: str | None = None,
    save_name: str | None = None,
    save_fmt: str = "png",
    save_dpi: int = 300,
    save_tight: bool = True,
    close_after_save: bool = False,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    pm = str(phi_mode).strip()
    pr = str(phi_reduce).strip()

    if pm not in ("separate_phi", "phi_avg"):
        raise ValueError(f"phi_mode must be 'separate_phi' or 'phi_avg', got: {pm}")
    if pr not in ("sum", "mean"):
        raise ValueError(f"phi_reduce must be 'sum' or 'mean', got: {pr}")

    csv_path = fitting_utils.resolve_fluence_fitting_csv_path(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        excitation_wl_nm=float(excitation_wl_nm),
        delay_fs=int(delay_fs),
        time_window_fs=int(time_window_fs),
        out_csv_name=str(out_csv_name),
        phi_mode=str(pm),
        phi_reduce=str(pr),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    df = pd.read_csv(csv_path)
    if df is None or len(df) == 0:
        raise ValueError(f"Empty CSV: {csv_path}")

    group_by = "phi_label" if pm == "phi_avg" else "azim_range_str"

    if title is None:
        title = (
            f"{sample_name}, {int(temperature_K)} K\n"
            f"ex. wl={float(excitation_wl_nm):g} nm, delay={int(delay_fs)} fs\n"
            f"tw={int(time_window_fs)} fs\n"
            f"peak={peak}, y={_property}"
        )

    if save and (save_dir is None):
        save_dir = fitting_utils._default_fluence_fitting_figures_dir(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            delay_fs=int(delay_fs),
            time_window_fs=int(time_window_fs),
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )

    if save and (save_name is None):
        wl_tok = int(float(excitation_wl_nm))
        tw_tok = int(time_window_fs)
        dly_tok = int(delay_fs)

        def _tok(x: str) -> str:
            return str(x).replace(".", "p")

        mode_tag = "phiavg_" + pr if pm == "phi_avg" else "sepphi"
        save_name = (
            f"{sample_name}_"
            f"{int(temperature_K)}K_"
            f"{int(wl_tok)}nm_"
            f"delay{int(dly_tok)}fs_"
            f"{int(tw_tok)}fs_"
            f"peak{peak}_"
            f"{_property}_"
            f"{group_by}_"
            f"{_tok(unit)}_"
            f"mode_{mode_tag}"
        )

    plotter = plot_utils.FitFluenceEvolutionPlotter(style=getattr(plot_utils, "DEFAULT_STYLE", None))

    fig, ax = plotter.plot(
        df,
        peak=str(peak),
        y=str(_property),
        fluence_unit=str(unit),
        group_by=str(group_by),
        groups=groups,
        only_success=True,
        include_reference=True,
        title=str(title) if title is not None else None,
        as_lines=bool(as_lines),
        fluence_offset=float(fluence_offset),
        show_baseline_sigma=bool(show_baseline_sigma),
        baseline_sigma=float(baseline_sigma),
        baseline_alpha=float(baseline_alpha),
        baseline_mode=str(baseline_mode),
        baseline_estimator=str(baseline_estimator),
        baseline_ddof=int(baseline_ddof),
        save=bool(save),
        save_dir=(Path(save_dir) if save_dir is not None else None),
        save_name=str(save_name) if save_name is not None else None,
        save_format=str(save_fmt),
        save_dpi=int(save_dpi),
        save_overwrite=True,
    )

    if bool(save_tight):
        try:
            fig.tight_layout()
        except Exception:
            pass

    if bool(close_after_save) and bool(save):
        try:
            plt.close(fig)
        except Exception:
            pass

    return fig, ax, str(csv_path)


def plot_fluence_evolution_multi(
    *,
    experiments,
    peak: str,
    prop: str,
    group_by: str = "azim_range_str",
    group: str = "Full",
    fluence_unit: str = "mJ/cm$^2$",
    x_col: str = "fluence_mJ_cm2",
    only_success: bool = True,
    include_reference: bool = True,
    title: str = None,
    legend_title: str = None,
    as_lines: bool = False,
    show_baseline_sigma: bool = False,
    baseline_mode: str = "errorbar",
    baseline_estimator: str = "std",
    baseline_ddof: int = 1,
    baseline_sigma_scale: float = 1.0,
    show: bool = True,
    legend_outside: bool = True,
    save: bool = False,
    save_dir: str = None,
    save_name: str = None,
    save_format: str = "png",
    save_dpi: int = 300,
    save_overwrite: bool = True,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """
    Multi-experiment fluence evolution plot.

    - include_reference is honored for plotted points.
    - baseline_from_reference is called with aligned df/x/y arrays.
    - explicit exp['fit_csv_path'] / exp['csv_path'] still wins if provided.
    """
    def _infer_fit_csv_path(exp: dict) -> str:
        for k in ("fit_csv_path", "csv_path", "peak_fits_csv", "fit_csv"):
            p = exp.get(k, None)
            if p is not None and str(p).strip() != "":
                return str(p)

        return fitting_utils.resolve_fluence_fitting_csv_path(
            sample_name=str(exp.get("sample_name")),
            temperature_K=int(exp.get("temperature_K")),
            excitation_wl_nm=float(exp.get("excitation_wl_nm")),
            delay_fs=int(exp.get("delay_fs")),
            time_window_fs=int(exp.get("time_window_fs")),
            out_csv_name="peak_fits_fluence.csv",
            phi_mode=(str(exp.get("phi_mode")).strip() if exp.get("phi_mode", None) is not None else "separate_phi"),
            phi_reduce=str(exp.get("phi_reduce", "sum")),
            **_resolve_exp_path_kwargs(
                exp,
                paths=paths,
                path_root=path_root,
                analysis_subdir=analysis_subdir,
            ),
        )

    series_list = []
    phi_mode_seen = ""
    phi_reduce_seen = ""
    gkey_for_title = None

    plotter_paths = paths
    if plotter_paths is None and path_root is not None and analysis_subdir is not None:
        plotter_paths = AnalysisPaths(
            path_root=Path(path_root),
            analysis_subdir=str(analysis_subdir),
        )

    plotter = plot_utils.FitFluenceEvolutionMultiPlotter(
        style=getattr(plot_utils, "DEFAULT_STYLE", None),
        paths=plotter_paths,
    )

    for exp in list(experiments):
        csv_path = _infer_fit_csv_path(exp)
        if not os.path.exists(str(csv_path)):
            raise FileNotFoundError(str(csv_path))

        df = pd.read_csv(str(csv_path))

        if "peak" not in df.columns:
            raise KeyError(f"CSV missing column 'peak': {csv_path}")
        if str(group_by) not in df.columns:
            raise KeyError(f"CSV missing column '{group_by}': {csv_path}")
        if str(x_col) not in df.columns:
            raise KeyError(f"CSV missing x column '{x_col}': {csv_path}")
        if str(prop) not in df.columns:
            raise KeyError(f"CSV missing property column '{prop}': {csv_path}")

        if phi_mode_seen == "" and "phi_mode" in df.columns:
            try:
                phi_mode_seen = str(df["phi_mode"].dropna().iloc[0])
            except Exception:
                phi_mode_seen = ""
        if phi_reduce_seen == "" and "phi_reduce" in df.columns:
            try:
                phi_reduce_seen = str(df["phi_reduce"].dropna().iloc[0])
            except Exception:
                phi_reduce_seen = ""

        dpk = df[df["peak"].astype(str) == str(peak)].copy()
        if len(dpk) == 0:
            continue

        gkey = plotter.resolve_group_key(dpk, str(group_by), group)
        if gkey_for_title is None:
            gkey_for_title = gkey

        dgrp_all = dpk[dpk[str(group_by)].astype(str) == str(gkey)].copy()
        if len(dgrp_all) == 0:
            continue

        x_all = pd.to_numeric(dgrp_all[str(x_col)], errors="coerce").values.astype(float)
        y_all = pd.to_numeric(dgrp_all[str(prop)], errors="coerce").values.astype(float)

        m_y = np.isfinite(y_all)
        dgrp_all = dgrp_all.loc[m_y].copy()
        x_all = x_all[m_y]
        y_all = y_all[m_y]

        if len(dgrp_all) == 0:
            continue

        dplot = dgrp_all.copy()

        if bool(only_success) and ("success" in dplot.columns):
            dplot = dplot[dplot["success"].astype(bool)]

        if (not bool(include_reference)) and ("is_reference" in dplot.columns):
            dplot = dplot[~dplot["is_reference"].astype(bool)]

        x_plot = pd.to_numeric(dplot[str(x_col)], errors="coerce").values.astype(float)
        y_plot = pd.to_numeric(dplot[str(prop)], errors="coerce").values.astype(float)
        m_xy = np.isfinite(x_plot) & np.isfinite(y_plot)

        x_plot = x_plot[m_xy]
        y_plot = y_plot[m_xy]

        if x_plot.size == 0:
            continue

        lab = str(exp.get("label", "")).strip()
        if lab == "":
            lab = plotter.default_label_from_experiment(exp)

        y0, sig = 0.0, 0.0
        if bool(show_baseline_sigma):
            from types import SimpleNamespace

            cols = SimpleNamespace(delay_fs_col=str(x_col), is_ref_col="is_reference")

            y0, sig = plot_utils.FitTimeEvolutionMultiPlotter.baseline_from_reference(
                df_sel=dgrp_all,
                x=x_all,
                y=y_all,
                prop=str(prop),
                cols=cols,
                ref_type=str(exp.get("ref_type", "dark")),
                ref_value=exp.get("ref_value", None),
                sigma_scale=float(baseline_sigma_scale),
                estimator=str(baseline_estimator),
                ddof=int(baseline_ddof),
            )

        series_list.append(
            dict(
                x=x_plot,
                y=y_plot,
                label=lab,
                baseline_y0=float(y0),
                baseline_sig=float(sig),
            )
        )

    if len(series_list) == 0:
        raise ValueError(f"No series to plot after filtering. (peak={peak}, {group_by}={group}, prop={prop})")

    if title is None:
        title = plotter.title_default(
            peak=str(peak),
            prop=str(prop),
            group_by=str(group_by),
            group_key=str(gkey_for_title if gkey_for_title is not None else group),
        )

    if save_name is None or str(save_name).strip() == "":
        save_name = plotter.default_save_name(
            peak=str(peak),
            prop=str(prop),
            group_by=str(group_by),
            group_key=str(gkey_for_title if gkey_for_title is not None else group),
            phi_mode=str(phi_mode_seen),
            phi_reduce=str(phi_reduce_seen),
            n_series=int(len(series_list)),
            unit=str(fluence_unit),
        )

    ylabel = plotter.ylabel_for_property(prop=str(prop), peak=str(peak))

    if bool(save) and (save_dir is None or str(save_dir).strip() == ""):
        if paths is not None:
            save_dir = str(Path(paths.analysis_root) / "general_figures")
        elif path_root is not None and analysis_subdir is not None:
            save_dir = str(Path(path_root) / Path(analysis_subdir) / "general_figures")

    fig, ax, saved_path = plotter.plot(
        series_list,
        title=title,
        fluence_unit=str(fluence_unit),
        ylabel=ylabel,
        legend_title=legend_title,
        as_lines=bool(as_lines),
        show_baseline_sigma=bool(show_baseline_sigma),
        baseline_mode=str(baseline_mode),
        show=bool(show),
        legend_outside=bool(legend_outside),
        save=bool(save),
        save_dir=save_dir,
        save_name=str(save_name),
        save_format=str(save_format),
        save_dpi=int(save_dpi),
        save_overwrite=bool(save_overwrite),
    )

    return fig, ax, saved_path

