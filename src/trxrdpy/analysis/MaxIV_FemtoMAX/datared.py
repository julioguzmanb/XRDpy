# datared.py
"""
User-facing FemtoMAX data-reduction wrappers.

This module stays intentionally simple:
- instantiate datared_utils.Experiment
- build metadata HDF5 files
- export averaged 2D detector images
- plot ping / delay distributions

No config.py dependency:
provide either
  - paths=AnalysisPaths(...)
or
  - path_root=..., raw_subdir=..., analysis_subdir=...

Typical import:
    from trxrdpy.analysis.common.paths import AnalysisPaths
    from trxrdpy.analysis.MaxIV_FemtoMAX import datared
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Union, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..common.paths import AnalysisPaths
from . import datared_utils

plt.ion()


def default_ping_reference_path() -> Path:
    """Return the packaged FemtoMAX scan-to-ping timing-reference table path."""
    return datared_utils.default_ping_reference_path()


def load_ping_reference_table(
    path: Optional[Union[str, Path]] = None,
) -> datared_utils.PingReferenceTable:
    """Load and validate a FemtoMAX ping-reference CSV table.

    The packaged table is used when no path is supplied. Validation rejects
    overlapping ranges, missing columns, and malformed ping values.

    Parameters
    ----------
    path : path-like, optional
        Custom reference CSV; ``None`` selects the packaged table.

    Returns
    -------
    PingReferenceTable
        Validated scan-range lookup table for FemtoMAX timing references.
    """
    return datared_utils.load_ping_reference_table(path)


def _make_experiment(
    scans,
    *,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
    ping_reference_path: Optional[Union[str, Path]] = None,
    ref_provider: Optional[Callable[[int], Tuple[float, float]]] = None,
) -> datared_utils.Experiment:
    """Construct a FemtoMAX ``Experiment`` from normalized wrapper arguments."""
    return datared_utils.Experiment(
        scans=scans,
        ref_provider=ref_provider,
        ping_reference_path=ping_reference_path,
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
    )


def _metadata_path_for(
    exp: datared_utils.Experiment,
    meta: datared_utils.ExperimentMeta,
    *,
    scans,
) -> str:
    """Build the standardized FemtoMAX metadata path for an experiment."""
    return exp.metadata_h5_path(meta, scans=scans, paths=exp.paths)


def _normalize_fluence_selected_delays(
    selected_delays: Union[int, Sequence[int], str],
) -> list[int]:
    """Fluence exports are organized one metadata/output set per fixed delay.
    Therefore selected_delays must be explicit here.
    """
    if isinstance(selected_delays, str):
        if selected_delays.strip().lower() == "auto":
            raise ValueError(
                "For scan_type='fluence', selected_delays must be explicit "
                "(e.g. [-1000] or [-1000, 5000]), not 'auto'."
            )
        return [int(selected_delays)]

    if isinstance(selected_delays, (int, np.integer)):
        return [int(selected_delays)]

    return [int(x) for x in list(selected_delays)]


def plot_pings_distribution(
    scans,
    *,
    mode: str = "overlay",
    delay_source: str = "avg",
    unit: str = "fs",
    view: str = "scatter",
    bins: int = 250,
    hist_range: Optional[Tuple[float, float]] = None,
    density: bool = False,
    show_median: bool = True,
    require_both: bool = True,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
    ping_reference_path: Optional[Union[str, Path]] = None,
    ref_provider: Optional[Callable[[int], Tuple[float, float]]] = None,
):
    """Plot corrected timing-tool ping distributions for selected scans.

    Reference ping values are resolved from the configured table and used to
    center the corrected delay distributions.

    Parameters
    ----------
    scans
        Scan number or collection of scan numbers to inspect.
    mode, delay_source, unit, view
        Scan overlay/grouping, timing channel, display unit, and plot type.
    bins, hist_range, density, show_median
        Histogram resolution, range, normalization, and median marker controls.
    require_both : bool
        Require both timing-tool channels to be valid for a frame.
    paths, path_root, raw_subdir, analysis_subdir
        Modern or legacy FemtoMAX path configuration.
    ping_reference_path, ref_provider
        CSV-based or callable timing-reference source.

    Returns
    -------
    Experiment
        Configured experiment used to read and plot the distributions.
    """
    exp = _make_experiment(
        scans=scans,
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
        ping_reference_path=ping_reference_path,
        ref_provider=ref_provider,
    )
    exp.plot_delay_distribution(
        mode=mode,
        delay_source=delay_source,
        unit=unit,
        view=view,
        bins=bins,
        hist_range=hist_range,
        density=bool(density),
        show_median=bool(show_median),
        require_both=bool(require_both),
    )
    return exp


def create_h5_files(
    scans,
    sample_name,
    temperature_K,
    excitation_wl_nm,
    fluence_mJ_cm2,
    time_window_fs,
    *,
    scan_type: str = "delay",
    selected_delays: Union[str, Sequence[int]] = "auto",
    delay_source: str = "avg",
    require_both: bool = True,
    nb_shot_threshold: Optional[int] = None,
    overwrite: bool = True,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
    ping_reference_path: Optional[Union[str, Path]] = None,
    ref_provider: Optional[Callable[[int], Tuple[float, float]]] = None,
):
    """Create standardized FemtoMAX metadata HDF5 files for data reduction.

    Raw timing, frame validity, delay bins, and experiment metadata are
    consolidated for subsequent serial or parallel detector-image averaging.

    Parameters
    ----------
    scans, sample_name, temperature_K
        Source scans and sample identity.
    excitation_wl_nm, fluence_mJ_cm2, time_window_fs
        Pump and temporal metadata; dark scans may leave pump values unused.
    scan_type : {"dark", "delay", "fluence"}
        Reduction layout to prepare.
    selected_delays, delay_source, require_both, nb_shot_threshold
        Delay-bin selection, timing channel, validity, and minimum-shot filters.
    overwrite : bool
        Replace metadata files that already exist.
    paths, path_root, raw_subdir, analysis_subdir
        Modern or legacy FemtoMAX path configuration.
    ping_reference_path, ref_provider
        CSV-based or callable timing-reference source.

    Returns
    -------
    Experiment
        Configured experiment whose metadata files were created.

    Raises
    ------
    ValueError
        If ``scan_type`` is unsupported or fluence delays are not explicit.
    """
    exp = _make_experiment(
        scans=scans,
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
        ping_reference_path=ping_reference_path,
        ref_provider=ref_provider,
    )
    st = str(scan_type).strip().lower()

    if st == "dark":
        meta = datared_utils.ExperimentMeta(
            sample_name=sample_name,
            temperature_K=temperature_K,
            excitation_wl_nm=None,
            fluence_mJ_cm2=None,
            scan_type="dark",
            time_window_fs=None,
            delay=None,
        )
        out_path = exp.build_metadata_h5(meta=meta, overwrite=overwrite)
        print("metadata h5:", out_path)
        return exp

    if st == "delay":
        meta = datared_utils.ExperimentMeta(
            sample_name=sample_name,
            temperature_K=temperature_K,
            excitation_wl_nm=excitation_wl_nm,
            fluence_mJ_cm2=fluence_mJ_cm2,
            scan_type="delay",
            time_window_fs=time_window_fs,
            delay=None,
        )
        out_path = exp.build_metadata_h5(
            meta=meta,
            selected_delays=selected_delays,
            delay_source=delay_source,
            require_both=require_both,
            nb_shot_threshold=nb_shot_threshold,
            overwrite=overwrite,
        )
        print("metadata h5:", out_path)
        return exp

    if st == "fluence":
        delays_list = _normalize_fluence_selected_delays(selected_delays)

        for d in delays_list:
            meta = datared_utils.ExperimentMeta(
                sample_name=sample_name,
                temperature_K=temperature_K,
                excitation_wl_nm=excitation_wl_nm,
                fluence_mJ_cm2=fluence_mJ_cm2,   # must be list aligned with scans
                scan_type="fluence",
                time_window_fs=time_window_fs,
                delay=int(d),
            )
            out_path = exp.build_metadata_h5(
                meta=meta,
                selected_delays=[int(d)],
                delay_source=delay_source,
                require_both=require_both,
                nb_shot_threshold=nb_shot_threshold,
                overwrite=overwrite,
            )
            print("metadata h5:", out_path)

        return exp

    raise ValueError("scan_type must be 'delay', 'fluence', or 'dark'.")


def generate_2D_imgs(
    scans,
    sample_name,
    temperature_K,
    excitation_wl_nm,
    fluence_mJ_cm2,
    time_window_fs,
    *,
    scan_type: str = "delay",
    selected_delays: Union[str, Sequence[int]] = "auto",
    delay_source: str = "avg",
    require_both: bool = True,
    nb_shot_threshold: Optional[int] = None,
    overwrite: bool = False,
    # export config
    batch_size: int = 1000,
    use_parallel: bool = True,
    max_workers: int = 4,
    chunk_size: int = 2,
    start_method: str = "spawn",
    # paths
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
    ping_reference_path: Optional[Union[str, Path]] = None,
    ref_provider: Optional[Callable[[int], Tuple[float, float]]] = None,
):
    """Average FemtoMAX detector frames and export standardized 2D images.

    The selected dark, delay, or fluence workflow reads prepared metadata,
    rejects invalid frames, and writes NumPy arrays under the shared analysis
    directory layout.

    Parameters
    ----------
    scans, sample_name, temperature_K
        Source scans and sample identity.
    excitation_wl_nm, fluence_mJ_cm2, time_window_fs
        Pump and temporal metadata used in output paths.
    scan_type : {"dark", "delay", "fluence"}
        Reduction layout to export.
    selected_delays, delay_source, require_both, nb_shot_threshold
        Delay-bin selection, timing channel, validity, and minimum-shot filters.
    overwrite : bool
        Replace metadata and image outputs that already exist.
    batch_size : int
        Maximum frame count accumulated per read batch.
    use_parallel, max_workers, chunk_size, start_method
        Multiprocessing enablement and worker scheduling controls.
    paths, path_root, raw_subdir, analysis_subdir
        Modern or legacy FemtoMAX path configuration.
    ping_reference_path, ref_provider
        CSV-based or callable timing-reference source.

    Returns
    -------
    tuple
        Configured :class:`Experiment` and export result mapping.

    Raises
    ------
    ValueError
        If ``scan_type`` is unsupported or fluence delays are not explicit.
    """
    exp = _make_experiment(
        scans=scans,
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
        ping_reference_path=ping_reference_path,
        ref_provider=ref_provider,
    )
    st = str(scan_type).strip().lower()

    # ----------------------------
    # DARK
    # ----------------------------
    if st == "dark":
        meta = datared_utils.ExperimentMeta(
            sample_name=sample_name,
            temperature_K=temperature_K,
            excitation_wl_nm=None,
            fluence_mJ_cm2=None,
            scan_type="dark",
            time_window_fs=None,
            delay=None,
        )

        metadata_path = _metadata_path_for(exp, meta, scans=scans)

        if overwrite or (not os.path.exists(metadata_path)):
            exp.build_metadata_h5(meta=meta, overwrite=overwrite)

        if use_parallel:
            res = exp.export_delay_2d_images_parallel(
                meta=meta,
                batch_size=batch_size,
                overwrite=overwrite,
                max_workers=max_workers,
                chunk_size=chunk_size,
                start_method=start_method,
                rdcc_nbytes=0,
                rdcc_nslots=0,
            )
        else:
            res = exp.export_delay_2d_images(
                meta=meta,
                batch_size=batch_size,
                overwrite=overwrite,
            )

        print(res)
        return exp, res

    # ----------------------------
    # DELAY
    # ----------------------------
    if st == "delay":
        meta = datared_utils.ExperimentMeta(
            sample_name=sample_name,
            temperature_K=temperature_K,
            excitation_wl_nm=excitation_wl_nm,
            fluence_mJ_cm2=fluence_mJ_cm2,
            scan_type="delay",
            time_window_fs=time_window_fs,
            delay=None,
        )

        metadata_path = _metadata_path_for(exp, meta, scans=scans)

        if overwrite or (not os.path.exists(metadata_path)):
            exp.build_metadata_h5(
                meta=meta,
                selected_delays=selected_delays,
                delay_source=delay_source,
                require_both=require_both,
                nb_shot_threshold=nb_shot_threshold,
                overwrite=overwrite,
            )
        else:
            exp.validate_metadata_ping_references(metadata_path)

        if use_parallel:
            res = exp.export_delay_2d_images_parallel(
                meta=meta,
                batch_size=batch_size,
                overwrite=overwrite,
                max_workers=max_workers,
                chunk_size=chunk_size,
                start_method=start_method,
                rdcc_nbytes=0,
                rdcc_nslots=0,
            )
        else:
            res = exp.export_delay_2d_images(
                meta=meta,
                batch_size=batch_size,
                overwrite=overwrite,
            )

        print(res)
        return exp, res

    # ----------------------------
    # FLUENCE (one metadata + output folder per selected delay)
    # ----------------------------
    if st == "fluence":
        delays_list = _normalize_fluence_selected_delays(selected_delays)
        all_res = {}

        for d in delays_list:
            meta = datared_utils.ExperimentMeta(
                sample_name=sample_name,
                temperature_K=temperature_K,
                excitation_wl_nm=excitation_wl_nm,
                fluence_mJ_cm2=fluence_mJ_cm2,  # sequence aligned with scans
                scan_type="fluence",
                time_window_fs=time_window_fs,
                delay=int(d),
            )

            metadata_path = _metadata_path_for(exp, meta, scans=scans)

            if overwrite or (not os.path.exists(metadata_path)):
                exp.build_metadata_h5(
                    meta=meta,
                    selected_delays=[int(d)],
                    delay_source=delay_source,
                    require_both=require_both,
                    nb_shot_threshold=nb_shot_threshold,
                    overwrite=overwrite,
                )
            else:
                exp.validate_metadata_ping_references(metadata_path)

            if use_parallel:
                res = exp.export_delay_2d_images_parallel(
                    meta=meta,
                    batch_size=batch_size,
                    overwrite=overwrite,
                    max_workers=max_workers,
                    chunk_size=chunk_size,
                    start_method=start_method,
                    rdcc_nbytes=0,
                    rdcc_nslots=0,
                )
            else:
                res = exp.export_delay_2d_images(
                    meta=meta,
                    batch_size=batch_size,
                    overwrite=overwrite,
                )

            all_res[int(d)] = res

        print(all_res)
        return exp, all_res

    raise ValueError("scan_type must be 'delay', 'fluence', or 'dark'.")
