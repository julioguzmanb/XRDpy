"""
User-facing differential analysis API.

Provides:
- plot_differential_integrals: time traces of integrated differential signal
  (peak vs background) for delay scans
- plot_differential_fft: FFT of the selected time trace, with background overlay
- plot_differential_integrals_multi: multi-experiment delay comparison
- plot_differential_fft_multi: multi-experiment delay FFT comparison
- plot_differential_integrals_fluence: fluence-scan differential integrals
- plot_differential_integrals_fluence_multi: multi-experiment fluence comparison

Path configuration
------------------
All single-experiment entry points require either:
  - paths=AnalysisPaths(...)
or:
  - path_root=... and analysis_subdir=...

All multi-experiment entry points support either:
  - per-experiment path config inside each experiment dict
  - function-level paths=...
  - function-level path_root=... and analysis_subdir=...

Saving
------
If save=True and save_dir=None, figures are saved under the experiment analysis folder:
  .../<analysis>/<sample>/temperature_.../excitation_wl_.../(delay|fluence)/.../figures/diff_analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

#from . import azimint
from .common import differential_analysis_utils as da_utils
from .common import plot_utils
from .common.paths import AnalysisPaths

plt.ion()


PEAK_SPECS: Dict[str, Dict[str, Any]] = {
    "012": {"q_range": (1.6438, 1.8), "bg_side": "right"},
    "104": {"q_range": (2.21, 2.40), "bg_side": "left"},
    "110": {"q_range": (2.45, 2.6), "bg_side": "right"},
    "116": {"q_range": (3.58, 3.82), "bg_side": "right"},
    "300": {"q_range": (4.30, 4.46), "bg_side": "left"},
}


def plot_differential_integrals(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    fluence_mJ_cm2: float,
    time_window_fs: int,
    delays_fs: Union[int, Sequence[int], str],
    ref_type: str,
    ref_value: Union[int, str, Sequence[int]],
    poni_path: Optional[str] = None,
    mask_edf_path: Optional[str] = None,
    azim_window: Tuple[float, float] = (-90.0, 90.0),
    azim_offset_deg: float = -90.0,
    peak: str = "110",
    peak_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    bg_mode: Optional[str] = None,
    npt: int = 1000,
    normalize_xy: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    unit: str = "ps",
    delay_offset: float = 0.0,
    show_errorbars: bool = True,
    errorbar_scale: float = 1.0,
    title: Optional[str] = None,
    plot_abs_and_diffs: bool = False,
    save: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 400,
    save_overwrite: bool = False,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """
    Compute and plot integrated differential signals (ΔI and |ΔI|) for:
      - a peak integration window ("peak")
      - a matched background window ("background")

    Error bars (if show_errorbars=True)
    ----------------------------------
    The error bars are per-delay and derived from the background response at the same delay.

    Definitions:
      - signed panel: yerr(delay) = | int_delta(background, delay) |
      - abs panel:    yerr(delay) =   int_abs_delta(background, delay)

    The plotted error bars are symmetric (±yerr). You can scale them by errorbar_scale.
    """
    pk = da_utils.get_peak_spec(
        peak,
        peak_specs=peak_specs,
        bg_mode=bg_mode,
        default_peak_specs=PEAK_SPECS,
    )

    if plot_abs_and_diffs:
        vlines_peak = pk.q_range
        q_delta = vlines_peak[-1] - vlines_peak[0]
        if pk.bg_mode == "before":
            vlines_bckg = (vlines_peak[0] - q_delta, vlines_peak[0])
        else:
            vlines_bckg = (vlines_peak[-1], vlines_peak[-1] + q_delta)

        # azimint.plot_1D_abs_and_diffs_delay(
        #     sample_name=sample_name,
        #     temperature_K=temperature_K,
        #     excitation_wl_nm=excitation_wl_nm,
        #     fluence_mJ_cm2=fluence_mJ_cm2,
        #     time_window_fs=time_window_fs,
        #     delays_fs="all",
        #     ref_type=ref_type,
        #     ref_value=ref_value,
        #     poni_path=poni_path,
        #     mask_edf_path=mask_edf_path,
        #     azim_window=azim_window,
        #     xlim=(1.5, 4.5),
        #     vlines_peak=vlines_peak,
        #     vlines_bckg=vlines_bckg,
        #     save_plots=False,
        #     save_format=save_format,
        #     save_dpi=save_dpi,
        #     save_overwrite=save_overwrite,
        #     from_2D_imgs=False,
        #     paths=paths,
        #     path_root=path_root,
        #     analysis_subdir=analysis_subdir,
        # )

    analyzer = da_utils.DelayDifferentialAnalyzer(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        excitation_wl_nm=float(excitation_wl_nm),
        fluence_mJ_cm2=float(fluence_mJ_cm2),
        time_window_fs=int(time_window_fs),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize_xy=bool(normalize_xy),
        q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
        azim_offset_deg=float(azim_offset_deg),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    df = analyzer.compute_delay_integrals(
        delays_fs=delays_fs,
        azim_window=(float(azim_window[0]), float(azim_window[1])),
        peak_spec=pk,
        ref_type=str(ref_type),
        ref_value=ref_value,
        compute_if_missing=bool(compute_if_missing),
        overwrite_xy=bool(overwrite_xy),
        include_reference_in_output=False,
        from_2D_imgs=False,
    )

    if "delay_ps" not in df.columns:
        df["delay_ps"] = df["delay_fs"].astype(float) * 1e-3

    if title is None:
        title = (
            f"{sample_name}, {temperature_K}K\n"
            f"$\\lambda$ = {excitation_wl_nm} nm, flu = {fluence_mJ_cm2} mJ/cm$^{{2}}$\n"
            f"hkl = ({pk.name}), q = ({pk.q_range[0]:.2f}, {pk.q_range[1]:.2f}) Å$^{{-1}}$\n"
            f"azim_win = ({azim_window[0]}, {azim_window[1]}) °, tw = {time_window_fs} fs\n"
            f"bg_mode: {pk.bg_mode}."
        )

    if save and (save_dir is None):
        save_dir = da_utils.default_save_dir_delay_experiment(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            fluence_mJ_cm2=float(fluence_mJ_cm2),
            time_window_fs=int(time_window_fs),
            figures_subdir=Path("figures") / "diff_analysis",
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )

    if save and (save_name is None):
        save_name = da_utils.default_save_name_integrals(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            fluence_mJ_cm2=float(fluence_mJ_cm2),
            time_window_fs=int(time_window_fs),
            peak_name=str(pk.name),
            q_range=(float(pk.q_range[0]), float(pk.q_range[1])),
            azim_window=(float(azim_window[0]), float(azim_window[1])),
            ref_type=str(ref_type),
        )

    colors = {"peak": "blue", "background": "gray"}

    plotter = plot_utils.DifferentialTimeTracePlotter(
        style=getattr(plot_utils, "DEFAULT_STYLE", None)
    )
    fig, _axes = plotter.plot(
        df,
        title=title,
        unit=str(unit),
        delay_offset=float(delay_offset),
        group_by="region",
        groups=["peak", "background"],
        colors=colors,
        show_errorbars=bool(show_errorbars),
        errorbars_from_group="background",
        errorbars_for_groups=("peak",),
        errorbar_scale=float(errorbar_scale),
        legend_outside=True,
        save=bool(save),
        save_dir=save_dir,
        save_name=save_name,
        save_format=str(save_format),
        save_dpi=int(save_dpi),
        save_overwrite=bool(save_overwrite),
    )

    return df, fig


def plot_differential_fft(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    fluence_mJ_cm2: float,
    time_window_fs: int,
    delays_fs: Union[int, Sequence[int], str],
    delay_offset: float = 0.0,
    ref_type: str,
    ref_value: Union[int, str, Sequence[int]],
    poni_path: Optional[str] = None,
    mask_edf_path: Optional[str] = None,
    azim_window: Tuple[float, float] = (-90, 90),
    azim_offset_deg: float = -90,
    peak: str = "110",
    peak_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    bg_mode: Optional[str] = None,
    region: str = "peak",
    kind: str = "diff",
    time_window_select_ps: Optional[Tuple[float, float]] = None,
    poly_order: int = 1,
    freq_unit: str = "cm^-1",
    xlim_freq: Optional[Tuple[float, float]] = (-1, 8),
    ylim_freq: Optional[Tuple[float, float]] = None,
    ylim_time: Optional[Tuple[float, float]] = None,
    npt: int = 1000,
    normalize_xy: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    title: Optional[str] = None,
    save: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 400,
    save_overwrite: bool = False,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    pk = da_utils.get_peak_spec(
        peak,
        peak_specs=peak_specs,
        bg_mode=bg_mode,
        default_peak_specs=PEAK_SPECS,
    )

    analyzer = da_utils.DelayDifferentialAnalyzer(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        excitation_wl_nm=float(excitation_wl_nm),
        fluence_mJ_cm2=float(fluence_mJ_cm2),
        time_window_fs=int(time_window_fs),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize_xy=bool(normalize_xy),
        q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
        azim_offset_deg=float(azim_offset_deg),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    df = analyzer.compute_delay_integrals(
        delays_fs=delays_fs,
        azim_window=(float(azim_window[0]), float(azim_window[1])),
        peak_spec=pk,
        ref_type=str(ref_type),
        ref_value=ref_value,
        compute_if_missing=bool(compute_if_missing),
        overwrite_xy=bool(overwrite_xy),
        include_reference_in_output=False,
        from_2D_imgs=False,
    )

    time_window_select_ps_eff: Optional[Tuple[float, float]]
    if time_window_select_ps is not None:
        time_window_select_ps_eff = (
            float(time_window_select_ps[0]) - float(delay_offset),
            float(time_window_select_ps[-1]) - float(delay_offset),
        )
    else:
        time_window_select_ps_eff = None

    if "delay_ps" not in df.columns:
        df["delay_ps"] = df["delay_fs"].astype(float) * 1e-3

    t_peak, y_peak = da_utils.select_series_for_fft(
        df,
        region="peak",
        kind=kind,
        time_window_select_ps=time_window_select_ps_eff,
    )
    t_bg, y_bg = da_utils.select_series_for_fft(
        df,
        region="background",
        kind=kind,
        time_window_select_ps=time_window_select_ps_eff,
    )

    fft_peak = analyzer.compute_fft(
        time_ps=t_peak,
        signal=y_peak,
        poly_order=int(poly_order),
        resample_uniform=False,
        dt_ps=None,
        freq_unit=str(freq_unit),
    )

    fft_bg = analyzer.compute_fft(
        time_ps=t_bg,
        signal=y_bg,
        poly_order=int(poly_order),
        resample_uniform=False,
        dt_ps=None,
        freq_unit=str(freq_unit),
    )

    region = str(region).strip().lower()
    if region == "bg":
        region = "background"
    if region not in ("peak", "background"):
        raise ValueError("region must be 'peak' or 'background'.")

    if title is None:
        win = ""
        if time_window_select_ps is not None:
            lo = min(time_window_select_ps)
            hi = max(time_window_select_ps)
            win = f"\nwindow={lo:g} to {hi:g} ps"

        title = (
            f"{sample_name}, {temperature_K}K\n"
            f"$\\lambda$ = {excitation_wl_nm} nm, flu = {fluence_mJ_cm2} mJ/cm$^{{2}}$\n"
            f"hkl = ({pk.name}), q = ({pk.q_range[0]:.2f}, {pk.q_range[1]:.2f}) Å$^{{-1}}$\n"
            f"azim_win = ({azim_window[0]}, {azim_window[1]}) °, tw = {time_window_fs} fs\n"
            f"bg_mode: {pk.bg_mode}. poly={poly_order}{win}"
        )

    if save and (save_dir is None):
        save_dir = da_utils.default_save_dir_delay_experiment(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            fluence_mJ_cm2=float(fluence_mJ_cm2),
            time_window_fs=int(time_window_fs),
            figures_subdir=Path("figures") / "diff_analysis",
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )

    if save and (save_name is None):
        save_name = da_utils.default_save_name_fft(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            fluence_mJ_cm2=float(fluence_mJ_cm2),
            time_window_fs=int(time_window_fs),
            peak_name=str(pk.name),
            azim_window=(float(azim_window[0]), float(azim_window[1])),
            region=str(region),
            kind=str(kind),
            poly_order=int(poly_order),
            time_window_select_ps=time_window_select_ps,
            ref_type=str(ref_type),
        )

    plotter = plot_utils.DifferentialFFTPlotter(
        style=getattr(plot_utils, "DEFAULT_STYLE", None)
    )

    if region == "peak":
        label_main = "FFT\n(peak)"
        label_bg = "FFT\n(background)"
        fft_main = fft_peak
        fft_bg_for_plot = fft_bg
    else:
        label_main = "FFT\n(background)"
        label_bg = "FFT\n(peak)"
        fft_main = fft_bg
        fft_bg_for_plot = fft_peak

    fig, _axes = plotter.plot(
        fft_main,
        fft_bg=fft_bg_for_plot,
        title=title,
        freq_unit=str(freq_unit),
        xlim_freq=xlim_freq,
        ylim_freq=ylim_freq,
        ylim_time=ylim_time,
        delay_offset=delay_offset,
        show_baseline=True,
        label_main=label_main,
        label_bg=label_bg,
        save=bool(save),
        save_dir=save_dir,
        save_name=save_name,
        save_format=str(save_format),
        save_dpi=int(save_dpi),
        save_overwrite=bool(save_overwrite),
    )

    return df, (fft_peak, fft_bg), fig


def plot_differential_integrals_multi(
    *,
    experiments: Sequence[Dict[str, object]],
    delays_fs: Union[int, Sequence[int], str],
    poni_path: Optional[str] = None,
    mask_edf_path: Optional[str] = None,
    azim_window: Tuple[float, float] = (-90.0, 90.0),
    peak: str = "110",
    peak_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    bg_mode: Optional[str] = None,
    npt: int = 1000,
    normalize_xy: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    azim_offset_deg: float = -90.0,
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    unit: str = "ps",
    show_errorbars: bool = True,
    errorbar_scale: float = 1.0,
    as_lines: bool = False,
    title: Optional[str] = None,
    save: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 400,
    save_overwrite: bool = True,
    show: bool = True,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """
    Multi-experiment version of plot_differential_integrals (DELAY scans).

    - One peak trace per experiment (no background curves).
    - Error bars are per-delay and derived from the background response
      of the same experiment (same rule as single-experiment version).
    - Legend entries follow the fitting multi style:
        * 'label' from experiment dict if non-empty
        * otherwise FitTimeEvolutionMultiPlotter.default_label_from_experiment(exp)
    - Supports unit='ps' or 'fs' and per-experiment delay_offset_ps in experiments.
    """
    if peak_specs is None:
        peak_specs = PEAK_SPECS

    pk = da_utils.get_peak_spec(
        peak,
        peak_specs=peak_specs,
        bg_mode=bg_mode,
        default_peak_specs=PEAK_SPECS,
    )

    series = da_utils.build_multi_delay_integral_series(
        experiments=experiments,
        delays_fs=delays_fs,
        pk_spec=pk,
        azim_window=(float(azim_window[0]), float(azim_window[1])),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize_xy=bool(normalize_xy),
        q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
        azim_offset_deg=float(azim_offset_deg),
        compute_if_missing=bool(compute_if_missing),
        overwrite_xy=bool(overwrite_xy),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    if not series:
        raise ValueError(
            "No series to plot in plot_differential_integrals_multi (check experiments / data)."
        )

    fit_multi_plotter = plot_utils.FitTimeEvolutionMultiPlotter(
        style=getattr(plot_utils, "DEFAULT_STYLE", None)
    )

    for s in series:
        exp = s["experiment"]
        lbl = str(exp.get("label", "")).strip()
        if lbl == "":
            lbl = fit_multi_plotter.default_label_from_experiment(exp)
        s["label"] = lbl

    if title is None:
        title = f"hkl = ({peak}), azim=({azim_window[0]}, {azim_window[1]})°"

    plotter = plot_utils.DifferentialTimeTraceMultiPlotter(
        style=getattr(plot_utils, "DEFAULT_STYLE", None)
    )

    fig, axes, saved_path = plotter.plot(
        series,
        unit=str(unit),
        as_lines=bool(as_lines),
        show_errorbars=bool(show_errorbars),
        errorbar_scale=float(errorbar_scale),
        title=title,
        legend_title=plotter.legend_title_default(),
        show=bool(show),
        save=bool(save),
        save_dir=save_dir,
        save_name=save_name,
        save_format=str(save_format),
        save_dpi=int(save_dpi),
        save_overwrite=bool(save_overwrite),
        legend_outside=True,
    )

    return {
        "fig": fig,
        "axes": axes,
        "saved_path": saved_path,
        "series": series,
    }


def plot_differential_fft_multi(
    *,
    experiments: Sequence[Dict[str, object]],
    delays_fs: Union[int, Sequence[int], str],
    poni_path: Optional[str] = None,
    mask_edf_path: Optional[str] = None,
    azim_window: Tuple[float, float] = (-90.0, 90.0),
    peak: str = "110",
    peak_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    bg_mode: Optional[str] = None,
    kind: str = "diff",
    time_window_select_ps: Optional[Tuple[float, float]] = None,
    poly_order: int = 1,
    freq_unit: str = "cm^-1",
    npt: int = 1000,
    normalize_xy: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    azim_offset_deg: float = -90.0,
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    xlim_freq: Optional[Tuple[float, float]] = None,
    ylim_freq: Optional[Tuple[float, float]] = None,
    ylim_time: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    save: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 400,
    save_overwrite: bool = True,
    show: bool = True,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """
    Multi-experiment version of plot_differential_fft (DELAY scans).

    - Upper panel: detrended time traces for PEAK only (one per experiment).
    - Lower panel: FFT amplitude for PEAK (solid) and BACKGROUND (same color,
      alpha ~0.7) for each experiment.
    - Legend entries follow the fitting multi style (auto labels if label="").
    - Clickable legend toggles both time trace and FFT pair per experiment.
    """
    if peak_specs is None:
        peak_specs = PEAK_SPECS

    pk = da_utils.get_peak_spec(
        peak,
        peak_specs=peak_specs,
        bg_mode=bg_mode,
        default_peak_specs=PEAK_SPECS,
    )

    series = da_utils.build_multi_delay_fft_series(
        experiments=experiments,
        delays_fs=delays_fs,
        pk_spec=pk,
        azim_window=(float(azim_window[0]), float(azim_window[1])),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        kind=str(kind),
        time_window_select_ps=time_window_select_ps,
        poly_order=int(poly_order),
        freq_unit=str(freq_unit),
        npt=int(npt),
        normalize_xy=bool(normalize_xy),
        q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
        azim_offset_deg=float(azim_offset_deg),
        compute_if_missing=bool(compute_if_missing),
        overwrite_xy=bool(overwrite_xy),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    if not series:
        raise ValueError(
            "No series to plot in plot_differential_fft_multi (check experiments / data)."
        )

    fit_multi_plotter = plot_utils.FitTimeEvolutionMultiPlotter(
        style=getattr(plot_utils, "DEFAULT_STYLE", None)
    )
    for s in series:
        exp = s["experiment"]
        lbl = str(exp.get("label", "")).strip()
        if lbl == "":
            lbl = fit_multi_plotter.default_label_from_experiment(exp)
        s["label"] = lbl

    if title is None:
        win = ""
        if time_window_select_ps is not None:
            lo = min(time_window_select_ps)
            hi = max(time_window_select_ps)
            win = f", window={lo:g} to {hi:g} ps"

        title = f"hkl = ({peak}), azim=({azim_window[0]}, {azim_window[1]})°\npoly={poly_order}{win}"

    plotter = plot_utils.DifferentialFFTMultiPlotter(
        style=getattr(plot_utils, "DEFAULT_STYLE", None)
    )

    fig, axes, saved_path = plotter.plot(
        series,
        freq_unit=str(freq_unit),
        xlim_freq=xlim_freq,
        ylim_freq=ylim_freq,
        ylim_time=ylim_time,
        title=title,
        legend_title=plotter.legend_title_default(),
        show=bool(show),
        save=bool(save),
        save_dir=save_dir,
        save_name=save_name,
        save_format=str(save_format),
        save_dpi=int(save_dpi),
        save_overwrite=bool(save_overwrite),
        legend_outside=True,
    )

    return {
        "fig": fig,
        "axes": axes,
        "saved_path": saved_path,
        "series": series,
    }


def plot_differential_integrals_fluence(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    delay_fs: int,
    time_window_fs: int,
    fluences_mJ_cm2: Union[float, int, Sequence[Union[float, int]], str],
    ref_type: str,
    ref_value: Union[int, float, str, Sequence[int]],
    poni_path: Optional[str] = None,
    mask_edf_path: Optional[str] = None,
    azim_window: Tuple[float, float] = (-90.0, 90.0),
    azim_offset_deg: float = -90.0,
    peak: str = "110",
    peak_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    bg_mode: Optional[str] = None,
    npt: int = 1000,
    normalize_xy: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    fluence_unit: str = "mJ/cm$^2$",
    fluence_offset: float = 0.0,
    show_errorbars: bool = True,
    errorbar_scale: float = 1.0,
    title: Optional[str] = None,
    plot_abs_and_diffs: bool = False,
    save: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 400,
    save_overwrite: bool = False,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """
    Fluence-scan version of plot_differential_integrals().

    Computes and plots integrated differential signals (ΔI and |ΔI|) for:
      - peak window ("peak")
      - matched background window ("background")

    Error bars (if show_errorbars=True) are derived from the background at the same fluence:
      - signed panel: yerr(flu) = | int_delta(background, flu) |
      - abs panel:    yerr(flu) =   int_abs_delta(background, flu)
    """
    pk = da_utils.get_peak_spec(
        peak,
        peak_specs=peak_specs,
        bg_mode=bg_mode,
        default_peak_specs=PEAK_SPECS,
    )

    # if plot_abs_and_diffs:
    #     vlines_peak = pk.q_range
    #     q_delta = vlines_peak[-1] - vlines_peak[0]
    #     if pk.bg_mode == "before":
    #         vlines_bckg = (vlines_peak[0] - q_delta, vlines_peak[0])
    #     else:
    #         vlines_bckg = (vlines_peak[-1], vlines_peak[-1] + q_delta)

    #     azimint.plot_1D_abs_and_diffs_fluence(
    #         sample_name=sample_name,
    #         temperature_K=temperature_K,
    #         excitation_wl_nm=excitation_wl_nm,
    #         fluences_mJ_cm2="all",
    #         delay_fs=delay_fs,
    #         time_window_fs=time_window_fs,
    #         ref_type=ref_type,
    #         ref_value=ref_value,
    #         poni_path=poni_path,
    #         mask_edf_path=mask_edf_path,
    #         azim_window=azim_window,
    #         vlines_peak=vlines_peak,
    #         vlines_bckg=vlines_bckg,
    #         save_plots=False,
    #         save_format=save_format,
    #         save_dpi=save_dpi,
    #         save_overwrite=save_overwrite,
    #         paths=paths,
    #         path_root=path_root,
    #         analysis_subdir=analysis_subdir,
    #     )

    analyzer = da_utils.FluenceDifferentialAnalyzer(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        excitation_wl_nm=float(excitation_wl_nm),
        delay_fs=int(delay_fs),
        time_window_fs=time_window_fs,
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize_xy=bool(normalize_xy),
        q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
        azim_offset_deg=float(azim_offset_deg),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    df = analyzer.compute_fluence_integrals(
        fluences_mJ_cm2=fluences_mJ_cm2,
        azim_window=(float(azim_window[0]), float(azim_window[1])),
        peak_spec=pk,
        ref_type=str(ref_type),
        ref_value=ref_value,
        compute_if_missing=bool(compute_if_missing),
        overwrite_xy=bool(overwrite_xy),
        include_reference_in_output=False,
    )

    if title is None:
        title = (
            f"{sample_name}, {temperature_K}K\n"
            f"$\\lambda$ = {excitation_wl_nm} nm, delay = {int(delay_fs)} fs\n"
            f"hkl = ({pk.name}), q = ({pk.q_range[0]:.2f}, {pk.q_range[1]:.2f}) Å$^{{-1}}$\n"
            f"azim_win = ({azim_window[0]}, {azim_window[1]}) °, tw = {time_window_fs} fs\n"
            f"bg_mode: {pk.bg_mode}. ref={str(ref_type)}"
        )

    if save and (save_dir is None):
        save_dir = da_utils.default_save_dir_fluence_experiment(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            delay_fs=int(delay_fs),
            time_window_fs=int(time_window_fs),
            figures_subdir=Path("figures") / "diff_analysis",
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )

    if save and (save_name is None):
        save_name = da_utils.default_save_name_integrals_fluence(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            delay_fs=int(delay_fs),
            time_window_fs=int(time_window_fs),
            peak_name=str(pk.name),
            q_range=(float(pk.q_range[0]), float(pk.q_range[1])),
            azim_window=(float(azim_window[0]), float(azim_window[1])),
            ref_type=str(ref_type),
        )

    colors = {"peak": "blue", "background": "gray"}

    plotter = plot_utils.DifferentialFluenceTracePlotter(
        style=getattr(plot_utils, "DEFAULT_STYLE", None)
    )
    fig, _axes = plotter.plot(
        df,
        title=title,
        fluence_unit=str(fluence_unit),
        fluence_offset=float(fluence_offset),
        group_by="region",
        groups=["peak", "background"],
        colors=colors,
        show_errorbars=bool(show_errorbars),
        errorbars_from_group="background",
        errorbars_for_groups=("peak",),
        errorbar_scale=float(errorbar_scale),
        legend_outside=True,
        save=bool(save),
        save_dir=save_dir,
        save_name=save_name,
        save_format=str(save_format),
        save_dpi=int(save_dpi),
        save_overwrite=bool(save_overwrite),
    )

    return df, fig


def plot_differential_integrals_fluence_multi(
    *,
    experiments: Sequence[Dict[str, object]],
    fluences_mJ_cm2: Union[float, int, Sequence[Union[float, int]], str],
    poni_path: Optional[str] = None,
    mask_edf_path: Optional[str] = None,
    azim_window: Tuple[float, float] = (-90.0, 90.0),
    peak: str = "110",
    peak_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    bg_mode: Optional[str] = None,
    npt: int = 1000,
    normalize_xy: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    azim_offset_deg: float = -90.0,
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    fluence_unit: str = "mJ/cm$^2$",
    show_errorbars: bool = True,
    errorbar_scale: float = 1.0,
    as_lines: bool = False,
    title: Optional[str] = None,
    save: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 400,
    save_overwrite: bool = True,
    show: bool = True,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """
    Multi-experiment version of plot_differential_integrals_fluence (FLUENCE scans).

    - One peak trace per experiment (no background curves).
    - Error bars are derived from the background response of the same experiment:
        signed: yerr = | int_delta(background) |
        abs:    yerr =   int_abs_delta(background)
    - Legend entries:
        * 'label' from experiment dict if non-empty
        * otherwise auto from experiment fields (sample/T/wl/delay/tw/ref)
    - Supports per-experiment 'fluence_offset' for x-shifting if needed.
    """
    if peak_specs is None:
        peak_specs = PEAK_SPECS

    pk = da_utils.get_peak_spec(
        peak,
        peak_specs=peak_specs,
        bg_mode=bg_mode,
        default_peak_specs=PEAK_SPECS,
    )

    series = da_utils.build_multi_fluence_integral_series(
        experiments=experiments,
        fluences_mJ_cm2=fluences_mJ_cm2,
        pk_spec=pk,
        azim_window=(float(azim_window[0]), float(azim_window[1])),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize_xy=bool(normalize_xy),
        q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
        azim_offset_deg=float(azim_offset_deg),
        compute_if_missing=bool(compute_if_missing),
        overwrite_xy=bool(overwrite_xy),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    if not series:
        raise ValueError(
            "No series to plot in plot_differential_integrals_fluence_multi (check experiments / data)."
        )

    for s in series:
        exp = s["experiment"]
        lbl = str(exp.get("label", "")).strip()
        if lbl == "":
            sample_name = str(exp.get("sample_name", ""))
            temperature_K = exp.get("temperature_K", "")
            excitation_wl_nm = exp.get("excitation_wl_nm", "")
            delay_fs = exp.get("delay_fs", "")
            time_window_fs = exp.get("time_window_fs", "")
            lbl = f"{sample_name}, {temperature_K}, {excitation_wl_nm}, {delay_fs}, {time_window_fs}"
        s["label"] = lbl

    if title is None:
        title = f"hkl = ({peak}), azim=({azim_window[0]}, {azim_window[1]})°"

    plotter = plot_utils.DifferentialFluenceTraceMultiPlotter(
        style=getattr(plot_utils, "DEFAULT_STYLE", None)
    )

    fig, axes, saved_path = plotter.plot(
        series,
        fluence_unit=str(fluence_unit),
        as_lines=bool(as_lines),
        show_errorbars=bool(show_errorbars),
        errorbar_scale=float(errorbar_scale),
        title=title,
        legend_title=plotter.legend_title_default(),
        show=bool(show),
        save=bool(save),
        save_dir=save_dir,
        save_name=save_name,
        save_format=str(save_format),
        save_dpi=int(save_dpi),
        save_overwrite=bool(save_overwrite),
        legend_outside=True,
    )

    return {
        "fig": fig,
        "axes": axes,
        "saved_path": saved_path,
        "series": series,
    }


# ============================================================
# Example usage
# ============================================================
"""
path_root = "/Users/julioguzman/Desktop/LSF2025/FemtoMAX2025"
analysis_subdir = "analysis"

# ============================================================
# Experiment 1. Delay. Pure V2O3, large grains, ≈ 60 nm thick.
# ============================================================
sample_name = "DET55"
temperature_K = 77
excitation_wl_nm = 1500
fluence_mJ_cm2 = 15
time_window_fs = 1000

delays_fs = "all"
ref_type = "dark"
ref_value = [167246, 167285]
delay_offset = 13.2
peak = "104"
poni_path = "/Users/julioguzman/Desktop/LSF2025/FemtoMAX2025/calibration/DET55_167307.poni"
mask_edf_path = "/Users/julioguzman/Desktop/LSF2025/FemtoMAX2025/calibration/DET55_167285_mask.edf"

df_int, fig_int = XRD.analysis.differential_analysis.plot_differential_integrals(
    sample_name=sample_name,
    temperature_K=temperature_K,
    excitation_wl_nm=excitation_wl_nm,
    fluence_mJ_cm2=fluence_mJ_cm2,
    time_window_fs=time_window_fs,
    delays_fs=delays_fs,
    ref_type=ref_type,
    ref_value=ref_value,
    peak=peak,
    azim_window=(-90, 90),
    unit="ps",
    delay_offset=delay_offset,
    plot_abs_and_diffs=True,
    poni_path=poni_path,
    mask_edf_path=mask_edf_path,
    save=True,
    save_overwrite=True,
    path_root=path_root,
    analysis_subdir=analysis_subdir,
)

df_all, (fft_peak, fft_bg), fig_fft = XRD.analysis.differential_analysis.plot_differential_fft(
    sample_name=sample_name,
    temperature_K=temperature_K,
    excitation_wl_nm=excitation_wl_nm,
    fluence_mJ_cm2=fluence_mJ_cm2,
    time_window_fs=time_window_fs,
    delays_fs=delays_fs,
    delay_offset=delay_offset,
    ref_type=ref_type,
    ref_value=ref_value,
    peak=peak,
    region="peak",
    kind="diff",
    time_window_select_ps=(-1, 200),
    poly_order=2,
    xlim_freq=(-1, 8),
    save=True,
    save_overwrite=True,
    path_root=path_root,
    analysis_subdir=analysis_subdir,
    poni_path=poni_path,
    mask_edf_path=mask_edf_path,
)

# ============================================================
# Experiment 2. Fluence. Short delay. Pure V2O3, large grains, ≈ 60 nm thick.
# ============================================================
df, fig = plot_differential_integrals_fluence(
    sample_name="DET55",
    temperature_K=77,
    excitation_wl_nm=1500,
    delay_fs=-1000,
    time_window_fs=500,
    fluences_mJ_cm2="all",
    ref_type="dark",
    ref_value=[167285],
    peak="110",
    azim_window=(-90, 90),
    fluence_offset=0.0,
    plot_abs_and_diffs=True,
    save=True,
    save_overwrite=False,
    path_root=path_root,
    analysis_subdir=analysis_subdir,
)
"""
