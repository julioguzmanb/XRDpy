"""
Shared user-facing azimuthal-integration API for beamlines with the common 2D-image workflow.

This module provides the reusable azimuthal-integration layer for facilities whose
post-reduction data model is based on homogenized 2D images and shared XY generation.
It is used as the implementation backend for beamline-specific wrappers such as
:mod:`trxrdpy.analysis.MaxIV_FemtoMAX.azimint` and
:mod:`trxrdpy.analysis.Spring8_SACLA.azimint`.

Implementation
--------------
This module is a thin user-facing layer over :mod:`trxrdpy.analysis.common.azimint_utils`
and related utilities in :mod:`trxrdpy.analysis.common`.

Goals
-----
- No dependency on a local ``config.py``.
- Shared path handling with the rest of ``trxrdpy.analysis``.
- Avoid code duplication across beamlines that use the same 2D-image/XY workflow.
- Keep beamline-specific public APIs thin while delegating reusable logic to ``common``.

Path handling
-------------
All entry points require either:
  - ``paths=AnalysisPaths(...)``
or:
  - ``path_root=...`` and ``analysis_subdir=...``

Notes
-----
- ``integrate_*`` functions compute/cache XY files.
- ``plot_*`` functions can either reuse existing XY files or compute them on demand.
- ``poni_path`` / ``mask_edf_path`` are only required when XY files must be computed.
- This module is intended for beamlines sharing the common 2D-image pipeline; beamlines
  with a different post-reduction structure may require their own azimuthal-integration API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from ..common import azimint_utils, general_utils, plot_utils
from ..common.paths import AnalysisPaths

plt.ion()


def _resolve_path_config(
    *,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, object], Dict[str, str]]:
    """
    Return two dictionaries:
      - dataset kwargs for classes in common.azimint_utils
      - legacy kwargs for helper functions that still accept path_root/analysis_subdir
    """
    if paths is not None:
        dataset_kwargs: Dict[str, object] = {"paths": paths}
        legacy_kwargs = {
            "path_root": str(paths.path_root),
            "analysis_subdir": str(paths.analysis_subdir),
        }
        return dataset_kwargs, legacy_kwargs

    if path_root is not None and analysis_subdir is not None:
        dataset_kwargs = {
            "path_root": Path(path_root),
            "analysis_subdir": Path(analysis_subdir),
        }
        legacy_kwargs = {
            "path_root": str(path_root),
            "analysis_subdir": str(analysis_subdir),
        }
        return dataset_kwargs, legacy_kwargs

    raise ValueError(
        "Provide either paths=AnalysisPaths(...), or both "
        "path_root=... and analysis_subdir=...."
    )


def integrate_dark_1d(
    *,
    sample_name: str,
    temperature_K: int,
    poni_path: str,
    mask_edf_path: str,
    dark_tag: Optional[Union[str, int, Sequence[int]]] = None,
    azimuthal_edges: np.ndarray = np.arange(-90, 90 + 20, 45),
    include_full: bool = True,
    full_range: Tuple[float, float] = (-90, 90),
    npt: int = 2000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    overwrite_xy: bool = False,
    azim_offset_deg: float = -90.0,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """
    Compute and cache XY files for a dark dataset.

    Parameters
    ----------
    dark_tag
        Can be:
          - ``None`` (auto-resolve if only one dark dataset exists)
          - ``"scan_167246"`` / ``"scans_167246-167285"``
          - ``167246``
          - ``[167246, 167285]``
    """
    dataset_kwargs, _ = _resolve_path_config(
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    integrator = azimint_utils.AzimIntegrator(
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
        azim_offset_deg=float(azim_offset_deg),
    )

    resolved_tag = None
    if dark_tag is not None:
        resolved_tag = azimint_utils.dark_tag_from_scan_spec(dark_tag)

    ds = azimint_utils.DarkDataset(
        sample_name=sample_name,
        temperature_K=temperature_K,
        dark_tag=resolved_tag,
        **dataset_kwargs,
    )

    integrator.integrate_and_cache_xy(
        ds,
        azimuthal_edges=np.asarray(azimuthal_edges, float),
        include_full=bool(include_full),
        full_range=(float(full_range[0]), float(full_range[1])),
        overwrite_xy=bool(overwrite_xy),
    )

    return integrator, ds


def integrate_delay_1d(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    fluence_mJ_cm2: float,
    time_window_fs: int,
    delays_fs: Union[int, Sequence[int], str],
    poni_path: str,
    mask_edf_path: str,
    azimuthal_edges: np.ndarray = np.arange(-90, 90 + 20, 45),
    include_full: bool = True,
    full_range: Tuple[float, float] = (-90, 90),
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    overwrite_xy: bool = False,
    azim_offset_deg: float = -90.0,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """
    Compute and cache XY files for one or many delay points.

    Returns
    -------
    integrator, datasets
        ``integrator`` is an ``AzimIntegrator``.
        ``datasets`` is a list of ``DelayDataset`` objects.
    """
    dataset_kwargs, legacy_kwargs = _resolve_path_config(
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    delays_list = azimint_utils.normalize_delays_fs(
        delays_fs,
        sample_name=sample_name,
        temperature_K=temperature_K,
        excitation_wl_nm=excitation_wl_nm,
        fluence_mJ_cm2=fluence_mJ_cm2,
        time_window_fs=time_window_fs,
        **legacy_kwargs,
    )

    integrator = azimint_utils.AzimIntegrator(
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
        azim_offset_deg=float(azim_offset_deg),
    )

    datasets = []
    for d in delays_list:
        ds = azimint_utils.DelayDataset(
            sample_name,
            temperature_K,
            excitation_wl_nm,
            fluence_mJ_cm2,
            time_window_fs,
            int(d),
            **dataset_kwargs,
        )
        integrator.integrate_and_cache_xy(
            ds,
            azimuthal_edges=np.asarray(azimuthal_edges, float),
            include_full=bool(include_full),
            full_range=(float(full_range[0]), float(full_range[1])),
            overwrite_xy=bool(overwrite_xy),
        )
        datasets.append(ds)

    return integrator, datasets


def plot_1D_abs_and_diffs_delay(
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
    azim_window: Tuple[float, float] = (-90, 90),
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    xlim: Tuple[float, float] = (1.5, 4.5),
    ylim_top=None,
    ylim_diff=None,
    vlines_peak: Optional[Tuple[float, float]] = None,
    vlines_bckg: Optional[Tuple[float, float]] = None,
    fs_or_ps: str = "ps",
    digits: int = 2,
    title: Optional[str] = None,
    azim_offset_deg: float = -90.0,
    save_plots: bool = False,
    out_name: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 400,
    save_overwrite: bool = True,
    save_base_dir: Optional[Union[str, Path]] = None,
    from_2D_imgs: bool = True,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """
    Compare multiple delay 1D patterns to a reference.

    Reference modes
    ---------------
    - ``ref_type="delay"``, ``ref_value=<delay_fs>``
    - ``ref_type="dark"``,  ``ref_value=<scan int | scans list | dark_tag str>``
    """
    dataset_kwargs, legacy_kwargs = _resolve_path_config(
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    integrator = azimint_utils.AzimIntegrator(
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
        azim_offset_deg=float(azim_offset_deg),
    )

    delays_list = azimint_utils.normalize_delays_fs(
        delays_fs,
        sample_name=sample_name,
        temperature_K=temperature_K,
        excitation_wl_nm=excitation_wl_nm,
        fluence_mJ_cm2=fluence_mJ_cm2,
        time_window_fs=time_window_fs,
        from_2D_imgs=bool(from_2D_imgs),
        **legacy_kwargs,
    )

    if len(delays_list) == 0:
        raise FileNotFoundError("No delays found to compare (empty delays_list).")

    ref_type_n = str(ref_type).strip().lower()
    if ref_type_n == "delay":
        ref_delay_fs = int(ref_value)
        ds_ref = azimint_utils.DelayDataset(
            sample_name,
            temperature_K,
            excitation_wl_nm,
            fluence_mJ_cm2,
            time_window_fs,
            ref_delay_fs,
            **dataset_kwargs,
        )

        ref_label_val = azimint_utils.delay_label_value(
            ref_delay_fs,
            fs_or_ps=fs_or_ps,
            digits=digits,
        )
        ref_label = f"ref: {ref_label_val}"
        ref_tag_for_file = f"delay_{ref_delay_fs}fs"

    elif ref_type_n == "dark":
        dark_tag = azimint_utils.dark_tag_from_scan_spec(ref_value)
        ds_ref = azimint_utils.DarkDataset(
            sample_name,
            temperature_K,
            dark_tag=dark_tag,
            **dataset_kwargs,
        )
        ref_label = f"ref: dark\n{azimint_utils.pretty_dark_tag(dark_tag)}"
        ref_tag_for_file = f"dark_{dark_tag}"

    else:
        raise ValueError("ref_type must be 'delay' or 'dark'.")

    _, q_ref, I_ref = integrator.get_xy_for_window(
        ds_ref,
        azim_window,
        compute_if_missing=bool(compute_if_missing),
        overwrite_xy=bool(overwrite_xy),
    )

    patterns = []
    for d in delays_list:
        if ref_type_n == "delay" and int(d) == int(ref_value):
            continue

        ds = azimint_utils.DelayDataset(
            sample_name,
            temperature_K,
            excitation_wl_nm,
            fluence_mJ_cm2,
            time_window_fs,
            int(d),
            **dataset_kwargs,
        )
        _, q, I = integrator.get_xy_for_window(
            ds,
            azim_window,
            compute_if_missing=bool(compute_if_missing),
            overwrite_xy=bool(overwrite_xy),
        )
        dd_val = azimint_utils.delay_label_value(int(d), fs_or_ps=fs_or_ps, digits=digits)
        patterns.append((f"{dd_val}", q, I))

    if title is None:
        title = (
            f"{sample_name}. {temperature_K}K.\n"
            f"ex. wl={excitation_wl_nm}nm. flu={fluence_mJ_cm2} mJ/cm$^2$.\n"
            f"tw={time_window_fs}fs. azim=({azim_window[0]},{azim_window[1]})"
        )

    save_kwargs = dict(save=False)
    if save_plots:
        if save_base_dir is not None:
            base_dir = Path(save_base_dir)
        else:
            ds_out = azimint_utils.DelayDataset(
                sample_name,
                temperature_K,
                excitation_wl_nm,
                fluence_mJ_cm2,
                time_window_fs,
                int(delays_list[0]),
                **dataset_kwargs,
            )
            base_dir = ds_out.analysis_dir()

        if out_name is None:
            azs = general_utils.azim_range_str((azim_window[0], azim_window[1]))
            out_name = f"compare_{azs}_to_{ref_tag_for_file}"

        save_kwargs = plot_utils.build_save_kwargs(
            save=True,
            base_dir=base_dir,
            figures_subdir="figures/1D_patterns",
            save_name=out_name,
            save_format=save_format,
            save_dpi=int(save_dpi),
            overwrite=bool(save_overwrite),
        )

    p = plot_utils.Pattern1DPlotter()
    fig, axes = p.compare_to_reference(
        q_ref=q_ref,
        I_ref=I_ref,
        ref_label=ref_label,
        patterns=patterns,
        title=title,
        xlim=xlim,
        ylim_top=ylim_top,
        ylim_diff=ylim_diff,
        vlines_peak=vlines_peak,
        vlines_bckg=vlines_bckg,
        legend_title=f"Delay [{fs_or_ps}]",
        legend_loc="upper left",
        legend_outside=True,
        **save_kwargs,
    )

    return q_ref, I_ref, fig, axes


def integrate_fluence_1d(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    delay_fs: int,
    time_window_fs: int,
    fluences_mJ_cm2: Union[float, Sequence[float], str],
    poni_path: str,
    mask_edf_path: str,
    azimuthal_edges: np.ndarray = np.arange(-90, 90 + 20, 45),
    include_full: bool = True,
    full_range: Tuple[float, float] = (-90, 90),
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    overwrite_xy: bool = False,
    azim_offset_deg: float = -90.0,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """
    Compute and cache XY files for one or many fluence points at fixed delay.

    Returns
    -------
    integrator, datasets
        ``integrator`` is an ``AzimIntegrator``.
        ``datasets`` is a list of ``FluenceDataset`` objects.
    """
    dataset_kwargs, legacy_kwargs = _resolve_path_config(
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    fl_list = azimint_utils.normalize_fluences_mJ_cm2(
        fluences_mJ_cm2,
        sample_name=sample_name,
        temperature_K=temperature_K,
        excitation_wl_nm=excitation_wl_nm,
        delay_fs=delay_fs,
        time_window_fs=time_window_fs,
        from_2D_imgs=True,
        **legacy_kwargs,
    )

    integrator = azimint_utils.AzimIntegrator(
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
        azim_offset_deg=float(azim_offset_deg),
    )

    datasets = []
    for f in fl_list:
        ds = azimint_utils.FluenceDataset(
            sample_name,
            temperature_K,
            excitation_wl_nm,
            float(f),
            time_window_fs,
            int(delay_fs),
            **dataset_kwargs,
        )
        integrator.integrate_and_cache_xy(
            ds,
            azimuthal_edges=np.asarray(azimuthal_edges, float),
            include_full=bool(include_full),
            full_range=(float(full_range[0]), float(full_range[1])),
            overwrite_xy=bool(overwrite_xy),
        )
        datasets.append(ds)

    return integrator, datasets


def plot_1D_abs_and_diffs_fluence(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    delay_fs: int,
    time_window_fs: int,
    fluences_mJ_cm2: Union[float, Sequence[float], str],
    ref_type: str = "dark",
    ref_value: Optional[Union[float, int, str, Sequence[int]]] = None,
    poni_path: Optional[str] = None,
    mask_edf_path: Optional[str] = None,
    azim_window: Tuple[float, float] = (-90, 90),
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    xlim: Tuple[float, float] = (1.5, 4.5),
    ylim_top=None,
    ylim_diff=None,
    vlines_peak: Optional[Tuple[float, float]] = None,
    vlines_bckg: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    azim_offset_deg: float = -90.0,
    save_plots: bool = False,
    out_name: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 400,
    save_overwrite: bool = True,
    save_base_dir: Optional[Union[str, Path]] = None,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """
    Compare multiple fluence 1D patterns (sorted low->high) to a reference.

    Reference modes
    ---------------
    - ``ref_type="fluence"``, ``ref_value=<fluence_mJ_cm2>``
    - ``ref_type="dark"``,    ``ref_value=<scan int | scans list | dark_tag str | None>``
      If ``ref_value`` is ``None``, ``DarkDataset`` auto-resolves and there must be a unique dark dataset.
    """
    dataset_kwargs, legacy_kwargs = _resolve_path_config(
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    integrator = azimint_utils.AzimIntegrator(
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
        azim_offset_deg=float(azim_offset_deg),
    )

    fl_list = azimint_utils.normalize_fluences_mJ_cm2(
        fluences_mJ_cm2,
        sample_name=sample_name,
        temperature_K=temperature_K,
        excitation_wl_nm=excitation_wl_nm,
        delay_fs=delay_fs,
        time_window_fs=time_window_fs,
        **legacy_kwargs,
    )
    if len(fl_list) == 0:
        raise FileNotFoundError("No fluences found to compare (empty fl_list).")

    ref_type_n = str(ref_type).strip().lower()
    if ref_type_n not in ("fluence", "dark"):
        raise ValueError("ref_type must be 'fluence' or 'dark'.")

    if ref_type_n == "fluence":
        if ref_value is None:
            raise ValueError("ref_type='fluence' requires ref_value=<fluence_mJ_cm2>.")
        ref_f = float(ref_value)
        ds_ref = azimint_utils.FluenceDataset(
            sample_name,
            temperature_K,
            excitation_wl_nm,
            ref_f,
            time_window_fs,
            int(delay_fs),
            **dataset_kwargs,
        )
        ref_label = f"ref: {ref_f:g} mJ/cm$^2$"
        ref_tag_for_file = f"fluence_{general_utils.fluence_tag_file(ref_f)}mJ"

    else:
        resolved_tag = None
        if ref_value is not None:
            resolved_tag = azimint_utils.dark_tag_from_scan_spec(ref_value)

        ds_ref = azimint_utils.DarkDataset(
            sample_name,
            temperature_K,
            dark_tag=resolved_tag,
            **dataset_kwargs,
        )
        ref_label = f"ref: dark\n{azimint_utils.pretty_dark_tag(ds_ref.dark_tag)}"
        ref_tag_for_file = f"dark_{ds_ref.dark_tag}"

    _, q_ref, I_ref = integrator.get_xy_for_window(
        ds_ref,
        azim_window,
        compute_if_missing=bool(compute_if_missing),
        overwrite_xy=bool(overwrite_xy),
    )

    patterns = []
    for f in sorted(fl_list):
        if ref_type_n == "fluence" and abs(float(f) - float(ref_value)) < 1e-12:
            continue

        ds = azimint_utils.FluenceDataset(
            sample_name,
            temperature_K,
            excitation_wl_nm,
            float(f),
            time_window_fs,
            int(delay_fs),
            **dataset_kwargs,
        )
        _, q, I = integrator.get_xy_for_window(
            ds,
            azim_window,
            compute_if_missing=bool(compute_if_missing),
            overwrite_xy=bool(overwrite_xy),
        )
        patterns.append((f"{float(f):g}", q, I))

    if title is None:
        title = (
            f"{sample_name}. {temperature_K}K.\n"
            f"ex. wl={excitation_wl_nm}nm. delay={int(delay_fs)}fs. tw={time_window_fs}fs.\n"
            f"azim=({azim_window[0]},{azim_window[1]})"
        )

    save_kwargs = dict(save=False)
    if save_plots:
        if save_base_dir is not None:
            base_dir = Path(save_base_dir)
        else:
            ds_out = azimint_utils.FluenceDataset(
                sample_name,
                temperature_K,
                excitation_wl_nm,
                float(sorted(fl_list)[0]),
                time_window_fs,
                int(delay_fs),
                **dataset_kwargs,
            )
            base_dir = ds_out.analysis_dir()

        if out_name is None:
            azs = general_utils.azim_range_str((azim_window[0], azim_window[1]))
            out_name = f"compare_fluence_{azs}_to_{ref_tag_for_file}"

        save_kwargs = plot_utils.build_save_kwargs(
            save=True,
            base_dir=base_dir,
            figures_subdir="figures/1D_patterns",
            save_name=out_name,
            save_format=save_format,
            save_dpi=int(save_dpi),
            overwrite=bool(save_overwrite),
        )

    p = plot_utils.Pattern1DPlotter()
    fig, axes = p.compare_to_reference(
        q_ref=q_ref,
        I_ref=I_ref,
        ref_label=ref_label,
        patterns=patterns,
        title=title,
        xlim=xlim,
        ylim_top=ylim_top,
        ylim_diff=ylim_diff,
        vlines_peak=vlines_peak,
        vlines_bckg=vlines_bckg,
        legend_title="Fluence [mJ/cm$^2$]",
        legend_loc="upper left",
        legend_outside=True,
        **save_kwargs,
    )

    return fig, axes


__all__ = [
    "integrate_dark_1d",
    "integrate_delay_1d",
    "plot_1D_abs_and_diffs_delay",
    "integrate_fluence_1d",
    "plot_1D_abs_and_diffs_fluence",
]

