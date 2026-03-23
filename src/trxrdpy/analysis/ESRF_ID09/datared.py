#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from pathlib import Path
from typing import List, Optional, Union

from tqdm.auto import tqdm

import txs

try:
    from ..common.paths import AnalysisPaths
    from ..common import general_utils
    from ..common import azimint_utils as common_azimint_utils
    from .azimint import delay_token_to_fs
except Exception:
    from trxrdpy.analysis.common.paths import AnalysisPaths
    from trxrdpy.analysis.common import general_utils
    from trxrdpy.analysis.common import azimint_utils as common_azimint_utils
    from trxrdpy.analysis.ESRF_ID09.azimint import delay_token_to_fs


DarkDataset = common_azimint_utils.DarkDataset
DelayDataset = common_azimint_utils.DelayDataset


def _resolve_paths(
    *,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Union[str, Path] = "RAW_DATA",
    analysis_subdir: Union[str, Path] = "analysis",
) -> AnalysisPaths:
    if paths is not None:
        return paths

    if path_root is None:
        raise ValueError(
            "Provide either paths=AnalysisPaths(...), or path_root=... "
            "(optionally with raw_subdir=... and analysis_subdir=...)."
        )

    return AnalysisPaths(
        path_root=Path(path_root),
        raw_subdir=str(raw_subdir),
        analysis_subdir=str(analysis_subdir),
    )


def _effective_raw_sample_name(
    sample_name: str,
    raw_sample_name: Optional[str] = None,
) -> str:
    raw_name = sample_name if raw_sample_name is None else raw_sample_name
    raw_name = str(raw_name)

    if len(raw_name.strip()) == 0:
        raise ValueError("raw_sample_name must not be empty.")

    return raw_name


def _open_id09_scan(
    *,
    sample_name: str,
    dataset: int,
    scan_nb: int,
    raw_sample_name: Optional[str] = None,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Union[str, Path] = "RAW_DATA",
    analysis_subdir: Union[str, Path] = "analysis",
):
    pths = _resolve_paths(
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
    )

    raw_name = _effective_raw_sample_name(
        sample_name=sample_name,
        raw_sample_name=raw_sample_name,
    )

    dataset_dir = (
        Path(pths.raw_root)
        / raw_name
        / f"{raw_name}_{int(dataset):04d}"
    )
    raw_h5_path = dataset_dir / f"{raw_name}_{int(dataset):04d}.h5"

    if not raw_h5_path.exists():
        raise FileNotFoundError(str(raw_h5_path))

    dset = txs.BlissDataset(str(raw_h5_path))
    scan_key = f"{int(scan_nb)}.1"

    try:
        scan = dset[scan_key]
    except Exception as exc:
        raise KeyError(
            f"Could not open scan '{scan_key}' in file: {raw_h5_path}"
        ) from exc

    return pths, raw_h5_path, scan


def _delay_tokens_str(scan, *, digits: int = 1) -> np.ndarray:
    delays = scan.metadata["delay"]
    return np.asarray(txs.utils.t2str(delays, digits=digits), dtype=str)


def _normalize_delay_token(delay) -> str:
    if isinstance(delay, bytes):
        return delay.decode("utf-8")
    if isinstance(delay, str):
        return delay

    try:
        return str(txs.utils.t2str(np.asarray([delay]), digits=1)[0])
    except Exception:
        return str(delay)


def _unique_delay_tokens(scan) -> List[str]:
    tokens = _delay_tokens_str(scan)
    out = []
    seen = set()

    for tok in tokens:
        tok = str(tok)
        if tok not in seen:
            out.append(tok)
            seen.add(tok)

    return out


def _normalize_delay_selection(scan, delays="all") -> List[str]:
    if isinstance(delays, str):
        if delays.lower() == "all":
            return _unique_delay_tokens(scan)
        return [str(delays)]

    if isinstance(delays, (list, tuple, np.ndarray)):
        return [_normalize_delay_token(d) for d in list(delays)]

    return [_normalize_delay_token(delays)]


def _mean_selected_frames(
    scan,
    mask: np.ndarray,
    *,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> np.ndarray:
    indices = np.flatnonzero(mask)

    if indices.size == 0:
        raise ValueError("No frames selected for averaging.")

    if not show_progress:
        imgs = np.asarray(scan)[indices, :, :]
        if imgs.size == 0:
            raise ValueError("Selected frame stack is empty.")
        return np.mean(imgs, axis=0)

    sum_img = None
    n_used = 0

    iterator = tqdm(
        indices,
        desc=progress_desc or "Averaging frames",
        leave=False,
        unit="frame",
    )

    for idx in iterator:
        img = np.asarray(scan[idx], dtype=np.float64)

        if sum_img is None:
            sum_img = np.zeros_like(img, dtype=np.float64)

        sum_img += img
        n_used += 1

    if sum_img is None or n_used == 0:
        raise ValueError("No valid frames were accumulated.")

    return sum_img / float(n_used)


def get_2D_img(
    sample_name,
    dataset,
    scan_nb,
    delay,
    *,
    raw_sample_name: Optional[str] = None,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Union[str, Path] = "RAW_DATA",
    analysis_subdir: Union[str, Path] = "analysis",
    show_progress: bool = False,
):
    _, _, scan = _open_id09_scan(
        sample_name=sample_name,
        raw_sample_name=raw_sample_name,
        dataset=dataset,
        scan_nb=scan_nb,
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
    )

    delay_tokens = _delay_tokens_str(scan)
    target_delay = _normalize_delay_token(delay)
    mask = delay_tokens == target_delay

    if not np.any(mask):
        available = _unique_delay_tokens(scan)
        raise KeyError(
            f"Delay '{target_delay}' not found in scan {scan_nb}. "
            f"Available delays: {available}"
        )

    final_img = _mean_selected_frames(
        scan,
        mask,
        show_progress=show_progress,
        progress_desc=f"Frames for {target_delay}",
    )
    return np.asarray(final_img)


def create_dark_from_ref_delay(
    sample_name,
    dataset,
    scan_nb,
    delay_ref,
    temperature_K,
    *,
    raw_sample_name: Optional[str] = None,
    overwrite: bool = True,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Union[str, Path] = "RAW_DATA",
    analysis_subdir: Union[str, Path] = "analysis",
    show_progress: bool = True,
):
    pths = _resolve_paths(
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
    )

    final_img = get_2D_img(
        sample_name=sample_name,
        raw_sample_name=raw_sample_name,
        dataset=dataset,
        scan_nb=scan_nb,
        delay=delay_ref,
        paths=pths,
        show_progress=show_progress,
    )

    dark_ds = DarkDataset(
        sample_name=sample_name,
        temperature_K=temperature_K,
        dark_tag=general_utils.scan_tag(int(scan_nb)),
        paths=pths,
    )

    file_path = dark_ds.img_path()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists() and (not bool(overwrite)):
        raise FileExistsError(f"File exists: {file_path}")

    np.save(str(file_path), final_img)
    return str(file_path)


def create_final_2D_images(
    sample_name,
    dataset,
    scan_nb,
    temperature_K,
    excitation_wl_nm,
    fluence_mJ_cm2,
    time_window_fs,
    delays="all",
    *,
    raw_sample_name: Optional[str] = None,
    overwrite: bool = True,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Union[str, Path] = "RAW_DATA",
    analysis_subdir: Union[str, Path] = "analysis",
    show_progress: bool = True,
    show_frame_progress: bool = False,
):
    pths, _, scan = _open_id09_scan(
        sample_name=sample_name,
        raw_sample_name=raw_sample_name,
        dataset=dataset,
        scan_nb=scan_nb,
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
    )

    selected_delays = _normalize_delay_selection(scan, delays=delays)
    saved_paths = []

    delay_iterator = selected_delays
    if show_progress:
        delay_iterator = tqdm(
            selected_delays,
            desc=f"Creating delay images ({sample_name}, scan {scan_nb})",
            unit="delay",
        )

    for delay in delay_iterator:
        final_img = get_2D_img(
            sample_name=sample_name,
            raw_sample_name=raw_sample_name,
            dataset=dataset,
            scan_nb=scan_nb,
            delay=delay,
            paths=pths,
            show_progress=show_frame_progress,
        )

        delay_fs = int(delay_token_to_fs(delay))

        delay_ds = DelayDataset(
            sample_name=sample_name,
            temperature_K=temperature_K,
            excitation_wl_nm=excitation_wl_nm,
            fluence_mJ_cm2=fluence_mJ_cm2,
            time_window_fs=int(time_window_fs),
            delay_fs=delay_fs,
            paths=pths,
        )

        file_path = delay_ds.img_path()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists() and (not bool(overwrite)):
            raise FileExistsError(f"File exists: {file_path}")

        np.save(str(file_path), final_img)
        saved_paths.append(str(file_path))

    return saved_paths