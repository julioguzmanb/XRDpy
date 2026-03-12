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


def _make_experiment(
    scans,
    *,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
    ref_provider: Callable[[int], Tuple[float, float]] = datared_utils.ref_pings,
) -> datared_utils.Experiment:
    return datared_utils.Experiment(
        scans=scans,
        ref_provider=ref_provider,
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
    return exp.metadata_h5_path(meta, scans=scans, paths=exp.paths)


def _normalize_fluence_selected_delays(
    selected_delays: Union[int, Sequence[int], str],
) -> list[int]:
    """
    Fluence exports are organized one metadata/output set per fixed delay.
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
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
    ref_provider: Callable[[int], Tuple[float, float]] = datared_utils.ref_pings,
):
    exp = _make_experiment(
        scans=scans,
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
        ref_provider=ref_provider,
    )
    exp.plot_delay_distribution(
        mode=mode,
        delay_source=delay_source,
        unit=unit,
        view=view,
        bins=bins,
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
    ref_provider: Callable[[int], Tuple[float, float]] = datared_utils.ref_pings,
):
    exp = _make_experiment(
        scans=scans,
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
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
    ref_provider: Callable[[int], Tuple[float, float]] = datared_utils.ref_pings,
):
    exp = _make_experiment(
        scans=scans,
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
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


