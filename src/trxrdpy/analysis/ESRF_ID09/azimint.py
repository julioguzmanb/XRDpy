# azimint.py
"""
ESRF ID09 user-facing azimuthal-integration API.

Purpose
-------
This module converts ID09 delay-scan data into the same standardized XY layout
used by the rest of the trxrdpy analysis pipeline.

Scope
-----
- beamline-specific raw access + reduction for ESRF ID09
- writing standardized XY files
- plotting cached 1D patterns vs a delay reference

Notes
-----
- No ``config.py`` is used here.
- Paths are resolved through ``AnalysisPaths`` (preferred) or
  ``path_root=...`` + subdir arguments.
- Once XY files exist, the shared analysis code in ``trxrdpy.analysis.common``
  can reuse them exactly like for the other facilities.

Naming convention
-----------------
- ``sample_name`` is the analysis/output identifier used for:
    - standardized XY cache naming
    - downstream analysis compatibility
    - calibration/mask auto-discovery
- ``raw_sample_name`` is optional and used ONLY to locate the raw ESRF ID09 HDF5
  dataset on disk. If omitted, it defaults to ``sample_name``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import fabio
import numpy as np
import pyFAI
import matplotlib.pyplot as plt

from ..common import general_utils, plot_utils
from ..common import azimint_utils as common_azimint_utils
from ..common.paths import AnalysisPaths

try:
    import txs  # type: ignore
except Exception:  # pragma: no cover - optional external dependency
    txs = None

plt.ion()

DelayDataset = common_azimint_utils.DelayDataset

CalibrationResolver = Callable[..., Tuple[str, str]]

__all__ = [
    "ESRFScanSource",
    "DelayDataset",
    "get_raw_images",
    "default_calibration_paths",
    "resolve_calibration",
    "delay_token_to_fs",
    "delay_label_value",
    "available_delay_points_fs",
    "integrate_delay_1d",
    "plot_1D_abs_and_diffs_delay",
]


# -----------------------------------------------------------------------------
# Generic path helpers
# -----------------------------------------------------------------------------

def _resolve_paths(
    *,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Union[str, Path] = "",
    analysis_subdir: Union[str, Path] = "analysis",
) -> AnalysisPaths:
    """
    Build or validate an AnalysisPaths object.

    ``path_root`` is the experiment root containing e.g.:
      - <path_root>/<raw_subdir>/...
      - <path_root>/<analysis_subdir>/...
      - <path_root>/calibration/...
    """
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


def _calibration_root(
    *,
    paths: AnalysisPaths,
    calibration_subdir: Union[str, Path] = "calibration",
) -> Path:
    return Path(paths.path_root) / Path(calibration_subdir)


def _effective_raw_sample_name(
    sample_name: str,
    raw_sample_name: Optional[str] = None,
) -> str:
    raw_name = sample_name if raw_sample_name is None else raw_sample_name
    raw_name = str(raw_name)

    if len(raw_name.strip()) == 0:
        raise ValueError("raw_sample_name must not be empty.")

    return raw_name


def _load_ai_with_compat(poni_path: Union[str, Path]):
    """
    Load a pyFAI azimuthal integrator from a PONI file.

    Prefer the shared compatibility loader. Fall back to txs when available.
    """
    try:
        ai, _used, _changes = common_azimint_utils.load_poni_with_compat(
            poni_path,
            verbose=False,
        )
        return ai
    except Exception:
        if txs is None:
            raise
        return txs.utils.load_ai(poni_path)


def _require_txs() -> None:
    if txs is None:
        raise ImportError(
            "The optional dependency 'txs' is required for ESRF ID09 reduction "
            "(integrate_delay_1d / raw->XY generation), but it could not be imported."
        )


# -----------------------------------------------------------------------------
# Source access (beamline-specific)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ESRFScanSource:
    sample_name: str
    dataset: int
    scan_nb: int
    paths: AnalysisPaths
    raw_sample_name: Optional[str] = None

    @property
    def effective_raw_sample_name(self) -> str:
        return _effective_raw_sample_name(
            sample_name=self.sample_name,
            raw_sample_name=self.raw_sample_name,
        )

    @property
    def dataset_dir(self) -> Path:
        raw_name = self.effective_raw_sample_name
        return (
            Path(self.paths.raw_root)
            / raw_name
            / f"{raw_name}_{int(self.dataset):04d}"
        )

    @property
    def raw_h5_path(self) -> Path:
        raw_name = self.effective_raw_sample_name
        return self.dataset_dir / f"{raw_name}_{int(self.dataset):04d}.h5"

    @property
    def scan_path(self) -> str:
        return str(self.dataset_dir / f"scan{int(self.scan_nb):04d}")

    def read_raw_images(self) -> np.ndarray:
        path = self.raw_h5_path
        if not path.exists():
            raise FileNotFoundError(str(path))

        with h5py.File(path, "r") as f:
            dset = f[f"{int(self.scan_nb)}.1"]["measurement"]["rayonix"]
            return np.asarray(dset)


def get_raw_images(
    *,
    sample_name: str,
    dataset: int,
    scan_nb: int,
    raw_sample_name: Optional[str] = None,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Union[str, Path] = "",
    analysis_subdir: Union[str, Path] = "analysis",
) -> np.ndarray:
    src = ESRFScanSource(
        sample_name=str(sample_name),
        raw_sample_name=raw_sample_name,
        dataset=int(dataset),
        scan_nb=int(scan_nb),
        paths=_resolve_paths(
            paths=paths,
            path_root=path_root,
            raw_subdir=raw_subdir,
            analysis_subdir=analysis_subdir,
        ),
    )
    return src.read_raw_images()


# -----------------------------------------------------------------------------
# Calibration helpers
# -----------------------------------------------------------------------------

def default_calibration_paths(
    sample_name: str,
    *,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    calibration_subdir: Union[str, Path] = "calibration",
) -> Tuple[str, str]:
    """
    Try to auto-resolve calibration files from ``<path_root>/calibration``.

    Strategy:
      - PONI: newest file containing sample_name with extension .poni
      - mask: newest file containing sample_name and 'mask' with extension .edf

    If your project needs stricter rules, pass ``calibration_resolver=...``.
    """
    pths = _resolve_paths(paths=paths, path_root=path_root)
    cal_root = _calibration_root(paths=pths, calibration_subdir=calibration_subdir)

    poni_candidates = general_utils.find_candidates(
        cal_root,
        must_contain=[str(sample_name)],
        exts=[".poni"],
        newest_first=True,
    )
    mask_candidates = general_utils.find_candidates(
        cal_root,
        must_contain=[str(sample_name), "mask"],
        exts=[".edf"],
        newest_first=True,
    )

    if len(poni_candidates) == 0:
        raise FileNotFoundError(
            f"No .poni calibration file found for sample '{sample_name}' under: {cal_root}"
        )
    if len(mask_candidates) == 0:
        raise FileNotFoundError(
            f"No mask .edf file found for sample '{sample_name}' under: {cal_root}"
        )

    return str(poni_candidates[0]), str(mask_candidates[0])


def resolve_calibration(
    *,
    sample_name: str,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    calibration_subdir: Union[str, Path] = "calibration",
    calibration_resolver: Optional[CalibrationResolver] = None,
) -> Tuple[str, str]:
    """
    Resolve calibration file paths.

    Priority:
      1) explicit ``poni_path`` and ``mask_edf_path``
      2) auto-resolution via ``calibration_resolver`` or ``default_calibration_paths``
    """
    if poni_path is not None and mask_edf_path is not None:
        return str(poni_path), str(mask_edf_path)

    resolver = default_calibration_paths if calibration_resolver is None else calibration_resolver
    auto_poni, auto_mask = resolver(
        str(sample_name),
        paths=paths,
        path_root=path_root,
        calibration_subdir=calibration_subdir,
    )

    return (
        str(auto_poni if poni_path is None else poni_path),
        str(auto_mask if mask_edf_path is None else mask_edf_path),
    )


# -----------------------------------------------------------------------------
# Delay-label helpers
# -----------------------------------------------------------------------------

_DELAY_UNIT_TO_FS = {
    "fs": 1.0,
    "ps": 1.0e3,
    "ns": 1.0e6,
    "us": 1.0e9,
    "µs": 1.0e9,
    "ms": 1.0e12,
    "s": 1.0e15,
}


def delay_token_to_fs(token: Union[str, bytes, int, float]) -> int:
    if isinstance(token, (int, float, np.integer, np.floating)):
        return int(np.rint(float(token)))

    s = str(general_utils.decode_if_bytes(token)).strip().replace("μ", "µ")

    if re.match(r"^[+-]?0+(?:\.0+)?$", s):
        return 0

    m = re.match(
        r"^\s*([+-]?\d+(?:\.\d+)?)\s*(fs|ps|ns|us|µs|ms|s)\s*$",
        s,
        flags=re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"Cannot parse delay label '{s}' into fs.")

    value = float(m.group(1))
    unit = m.group(2).lower()
    return int(np.rint(value * _DELAY_UNIT_TO_FS[unit]))


def delay_label_value(
    delay_fs: Union[int, float],
    *,
    fs_or_ps: str = "ps",
    digits: int = 2,
) -> Union[int, float]:
    v = float(delay_fs)
    if str(fs_or_ps).lower() == "ps":
        return round(v * 1e-3, digits)
    return int(np.rint(v))


def _normalize_delay_selection(
    delays_fs: Union[int, Sequence[int], str],
    *,
    available_delays_fs: Sequence[int],
) -> List[int]:
    available = sorted(set(int(x) for x in available_delays_fs))

    if isinstance(delays_fs, str):
        if delays_fs.lower() != "all":
            raise ValueError("delays_fs string must be 'all' or an int/list of ints.")
        return available

    if isinstance(delays_fs, int):
        out = [int(delays_fs)]
    else:
        out = [int(x) for x in list(delays_fs)]

    missing = [d for d in out if d not in available]
    if missing:
        raise KeyError(f"Requested delays not found in scan: {missing}. Available: {available}")
    return out


def _delay_index_map(delay_tokens: Sequence[Union[str, bytes]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for i, tok in enumerate(delay_tokens):
        fs = delay_token_to_fs(tok)
        if fs in out:
            raise ValueError(f"Duplicate delay label after fs conversion: {tok!r} -> {fs} fs")
        out[fs] = int(i)
    return out


# -----------------------------------------------------------------------------
# Small internal helpers
# -----------------------------------------------------------------------------

def _standard_windows_from_edges(
    azimuthal_edges: Sequence[Union[int, float]],
    *,
    include_full: bool = True,
    full_range: Tuple[float, float] = (-90.0, 90.0),
) -> List[Tuple[float, float]]:
    wins = general_utils.windows_from_edges(
        azimuthal_edges,
        include_full=include_full,
        full_range=full_range,
        full_first=True,
        make_int=False,
    )
    return [(float(a), float(b)) for a, b in wins]


def _esrf_integration_window(
    standard_window: Tuple[float, float],
    *,
    azim_offset_deg: float = -90.0,
) -> Tuple[float, float]:
    return (
        float(standard_window[0]) + float(azim_offset_deg),
        float(standard_window[1]) + float(azim_offset_deg),
    )


def _extract_abs_trace(
    abs_av: np.ndarray,
    *,
    delay_index: int,
    q_size: int,
    n_delays: int,
) -> np.ndarray:
    arr = np.asarray(abs_av, float)
    if arr.ndim != 2:
        raise ValueError(f"Expected abs_av to be 2D, got shape {arr.shape}")

    if arr.shape == (q_size, n_delays):
        return np.asarray(arr[:, int(delay_index)], float)

    if arr.shape == (n_delays, q_size):
        return np.asarray(arr[int(delay_index), :], float)

    raise ValueError(
        f"Could not interpret abs_av shape {arr.shape}. "
        f"Expected ({q_size}, {n_delays}) or ({n_delays}, {q_size})."
    )


def _discover_cached_delay_points_fs(
    *,
    sample_name: str,
    temperature_K: Union[int, float],
    excitation_wl_nm: Union[int, float],
    fluence_mJ_cm2: Union[int, float],
    time_window_fs: int,
    paths: AnalysisPaths,
) -> List[int]:
    tmp = DelayDataset(
        sample_name=sample_name,
        temperature_K=temperature_K,
        excitation_wl_nm=excitation_wl_nm,
        fluence_mJ_cm2=fluence_mJ_cm2,
        time_window_fs=time_window_fs,
        delay_fs=0,
        paths=paths,
    )

    folder = tmp.xy_folder()
    if not folder.is_dir():
        raise FileNotFoundError(f"XY folder not found: {folder}")

    wl_tag = general_utils.wl_tag_nm(excitation_wl_nm)
    flu_tag = general_utils.fluence_tag_file(fluence_mJ_cm2)
    T = general_utils.to_int(temperature_K)
    tw = int(time_window_fs)

    patt = re.compile(
        rf"^{re.escape(sample_name)}_{T}K_{re.escape(wl_tag)}nm_{re.escape(flu_tag)}mJ_{tw}fs_(-?\d+)fs_.*\.xy$"
    )

    delays: List[int] = []
    for p in folder.iterdir():
        if p.suffix.lower() != ".xy":
            continue
        m = patt.match(p.name)
        if m:
            delays.append(int(m.group(1)))

    delays = sorted(set(delays))
    if not delays:
        raise FileNotFoundError(f"No cached XY patterns found in: {folder}")
    return delays


def available_delay_points_fs(
    *,
    sample_name: str,
    temperature_K: Union[int, float],
    excitation_wl_nm: Union[int, float],
    fluence_mJ_cm2: Union[int, float],
    time_window_fs: int,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Union[str, Path] = "",
    analysis_subdir: Union[str, Path] = "analysis",
) -> List[int]:
    pths = _resolve_paths(
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
    )
    return _discover_cached_delay_points_fs(
        sample_name=sample_name,
        temperature_K=temperature_K,
        excitation_wl_nm=excitation_wl_nm,
        fluence_mJ_cm2=fluence_mJ_cm2,
        time_window_fs=time_window_fs,
        paths=pths,
    )


def _ensure_xy_exists(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    fluence_mJ_cm2: float,
    time_window_fs: int,
    delay_fs: int,
    azim_window: Tuple[float, float],
    paths: AnalysisPaths,
) -> Path:
    ds = DelayDataset(
        sample_name=sample_name,
        temperature_K=temperature_K,
        excitation_wl_nm=excitation_wl_nm,
        fluence_mJ_cm2=fluence_mJ_cm2,
        time_window_fs=time_window_fs,
        delay_fs=int(delay_fs),
        paths=paths,
    )
    azim_str = general_utils.azim_range_str(azim_window)
    xy_path = ds.xy_path(azim_str)
    if not xy_path.exists():
        raise FileNotFoundError(
            f"Missing XY file:\n  {xy_path}\n"
            "Create the XY files first with integrate_delay_1d(...) "
            "using this same azimuthal window."
        )
    return xy_path


def _load_cached_xy(
    dataset_obj: DelayDataset,
    *,
    azim_window: Tuple[float, float],
    wavelength_m: float,
) -> Tuple[str, np.ndarray, np.ndarray]:
    azim_str = general_utils.azim_range_str(azim_window)
    xy_path = dataset_obj.xy_path(azim_str)
    if not xy_path.exists():
        raise FileNotFoundError(str(xy_path))
    two_theta, I = general_utils.load_xy(xy_path)
    q = general_utils.two_theta_to_q(two_theta, wavelength_m)
    return azim_str, np.asarray(q, float), np.asarray(I, float)


# -----------------------------------------------------------------------------
# Beamline-specific reduction -> standardized XY writer
# -----------------------------------------------------------------------------

def _reduce_delay_scan(
    *,
    sample_name: str,
    dataset: int,
    scan_nb: int,
    standard_azim_window: Tuple[float, float],
    paths: AnalysisPaths,
    raw_sample_name: Optional[str] = None,
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    calibration_resolver: Optional[CalibrationResolver] = None,
    calibration_subdir: Union[str, Path] = "calibration",
    npt: int = 600,
    force: bool = False,
    ref_delay: Union[str, int, float] = "-5ns",
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    azim_offset_deg: float = -90.0,
):
    _require_txs()

    src = ESRFScanSource(
        sample_name=str(sample_name),
        raw_sample_name=raw_sample_name,
        dataset=int(dataset),
        scan_nb=int(scan_nb),
        paths=paths,
    )
    if not src.raw_h5_path.exists():
        raise FileNotFoundError(str(src.raw_h5_path))

    poni_path_res, mask_path_res = resolve_calibration(
        sample_name=sample_name,
        paths=paths,
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        calibration_subdir=calibration_subdir,
        calibration_resolver=calibration_resolver,
    )

    mask = fabio.open(str(mask_path_res)).data
    ai = _load_ai_with_compat(poni_path_res)

    beamline_window = _esrf_integration_window(
        standard_azim_window,
        azim_offset_deg=azim_offset_deg,
    )

    azav = txs.azav.integrate1d_dataset(
        src.scan_path,
        ai,
        mask=mask,
        npt=int(npt),
        force=bool(force),
        azimuthal_range=beamline_window,
    )

    data = txs.datared.datared(
        azav,
        ref_delay=str(ref_delay),
        norm=tuple(q_norm_range),
    )
    return data, ai


def _write_delay_xy_bundle(
    *,
    data: dict,
    ai,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    fluence_mJ_cm2: float,
    time_window_fs: int,
    selected_delays_fs: Sequence[int],
    azim_window: Tuple[float, float],
    paths: AnalysisPaths,
    overwrite_xy: bool = False,
) -> Dict[int, str]:
    q = np.asarray(data["q"], float)
    delay_tokens = [general_utils.decode_if_bytes(x) for x in list(data["t"])]
    abs_av = np.asarray(data["diff_plus_ref_av"], float)

    delay_to_idx = _delay_index_map(delay_tokens)
    two_theta = general_utils.q_to_two_theta(q, ai.wavelength)

    azim_str = general_utils.azim_range_str(azim_window)
    saved: Dict[int, str] = {}

    for d_fs in selected_delays_fs:
        if int(d_fs) not in delay_to_idx:
            raise KeyError(
                f"Delay {d_fs} fs not present in reduced scan. "
                f"Available: {sorted(delay_to_idx)}"
            )

        idx = delay_to_idx[int(d_fs)]
        I = _extract_abs_trace(
            abs_av,
            delay_index=idx,
            q_size=q.size,
            n_delays=len(delay_tokens),
        )

        ds = DelayDataset(
            sample_name=sample_name,
            temperature_K=temperature_K,
            excitation_wl_nm=excitation_wl_nm,
            fluence_mJ_cm2=fluence_mJ_cm2,
            time_window_fs=time_window_fs,
            delay_fs=int(d_fs),
            paths=paths,
        )

        xy_path = ds.xy_path(azim_str)
        xy_path.parent.mkdir(parents=True, exist_ok=True)

        if xy_path.exists() and (not overwrite_xy):
            saved[int(d_fs)] = str(xy_path)
            continue

        general_utils.save_xy(xy_path, two_theta, I)
        saved[int(d_fs)] = str(xy_path)

    return saved


# -----------------------------------------------------------------------------
# User-facing functional API
# -----------------------------------------------------------------------------

def integrate_delay_1d(
    *,
    sample_name: str,
    dataset: int,
    scan_nb: int,
    temperature_K: int,
    excitation_wl_nm: float,
    fluence_mJ_cm2: float,
    time_window_fs: int,
    delays_fs: Union[int, Sequence[int], str] = "all",
    raw_sample_name: Optional[str] = None,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Union[str, Path] = "",
    analysis_subdir: Union[str, Path] = "analysis",
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    calibration_subdir: Union[str, Path] = "calibration",
    calibration_resolver: Optional[CalibrationResolver] = None,
    azimuthal_edges: Sequence[Union[int, float]] = tuple(np.arange(-90, 90 + 20, 45)),
    include_full: bool = True,
    full_range: Tuple[float, float] = (-90.0, 90.0),
    npt: int = 600,
    force: bool = False,
    ref_delay: Union[str, int, float] = "-5ns",
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    overwrite_xy: bool = False,
    azim_offset_deg: float = -90.0,
):
    """
    ESRF-style delay workflow:
      sample_name + dataset + scan_nb -> txs reduction -> standardized XY files

    ``sample_name`` is used for output/cache naming.
    ``raw_sample_name`` is used only to locate the raw HDF5 dataset. If omitted,
    raw access falls back to ``sample_name``.

    Output XY files follow the SAME folder structure and filename conventions
    as the shared trxrdpy analysis pipeline.

    Returns
    -------
    source : ESRFScanSource
    datasets : list[DelayDataset]
    saved_paths : dict[int, dict[str, str]]
        Mapping: delay_fs -> {azim_range_str -> xy_path}
    """
    pths = _resolve_paths(
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
    )

    windows = _standard_windows_from_edges(
        azimuthal_edges,
        include_full=include_full,
        full_range=full_range,
    )

    saved_paths: Dict[int, Dict[str, str]] = {}
    datasets_out: Dict[int, DelayDataset] = {}

    for std_window in windows:
        data, ai = _reduce_delay_scan(
            sample_name=str(sample_name),
            raw_sample_name=raw_sample_name,
            dataset=int(dataset),
            scan_nb=int(scan_nb),
            standard_azim_window=std_window,
            paths=pths,
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            calibration_subdir=calibration_subdir,
            calibration_resolver=calibration_resolver,
            npt=npt,
            force=force,
            ref_delay=ref_delay,
            q_norm_range=q_norm_range,
            azim_offset_deg=azim_offset_deg,
        )

        available_fs = sorted(
            _delay_index_map(
                [general_utils.decode_if_bytes(x) for x in list(data["t"])]
            ).keys()
        )
        selected_fs = _normalize_delay_selection(
            delays_fs,
            available_delays_fs=available_fs,
        )

        saved_here = _write_delay_xy_bundle(
            data=data,
            ai=ai,
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            fluence_mJ_cm2=float(fluence_mJ_cm2),
            time_window_fs=int(time_window_fs),
            selected_delays_fs=selected_fs,
            azim_window=std_window,
            paths=pths,
            overwrite_xy=overwrite_xy,
        )

        azim_str = general_utils.azim_range_str(std_window)
        for d_fs, p in saved_here.items():
            saved_paths.setdefault(int(d_fs), {})[azim_str] = str(p)
            if int(d_fs) not in datasets_out:
                datasets_out[int(d_fs)] = DelayDataset(
                    sample_name=str(sample_name),
                    temperature_K=int(temperature_K),
                    excitation_wl_nm=float(excitation_wl_nm),
                    fluence_mJ_cm2=float(fluence_mJ_cm2),
                    time_window_fs=int(time_window_fs),
                    delay_fs=int(d_fs),
                    paths=pths,
                )

    source = ESRFScanSource(
        sample_name=str(sample_name),
        raw_sample_name=raw_sample_name,
        dataset=int(dataset),
        scan_nb=int(scan_nb),
        paths=pths,
    )

    datasets = [datasets_out[k] for k in sorted(datasets_out)]
    return source, datasets, saved_paths


def plot_1D_abs_and_diffs_delay(
    *,
    sample_name: str,
    dataset: int,
    scan_nb: int,
    temperature_K: int,
    excitation_wl_nm: float,
    fluence_mJ_cm2: float,
    time_window_fs: int,
    delays_fs: Union[int, Sequence[int], str] = "all",
    ref_type: str = "delay",
    ref_value: Optional[Union[int, float, str]] = None,
    raw_sample_name: Optional[str] = None,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Union[str, Path] = "",
    analysis_subdir: Union[str, Path] = "analysis",
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    calibration_subdir: Union[str, Path] = "calibration",
    calibration_resolver: Optional[CalibrationResolver] = None,
    azim_window: Tuple[float, float] = (-90.0, 90.0),
    npt: int = 600,
    force: bool = False,
    ref_delay: Union[str, int, float] = "-5ns",
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
):
    """
    Plot delay-resolved 1D patterns against a delay reference.

    Behavior
    --------
    - Reads cached XY files from the standardized analysis folder.
    - If requested files are missing and ``compute_if_missing=True``, it first
      generates them via ``integrate_delay_1d(...)`` for the requested
      azimuth window.

    Notes
    -----
    This ESRF ID09 implementation currently supports ``ref_type='delay'`` only.

    ``raw_sample_name`` is only used if missing XY files need to be generated
    from the raw HDF5 data.
    """
    pths = _resolve_paths(
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
    )

    ref_type_n = str(ref_type).strip().lower()
    if ref_type_n != "delay":
        raise NotImplementedError(
            "This ESRF ID09 delay workflow currently supports ref_type='delay' only."
        )

    ref_delay_fs = delay_token_to_fs(ref_delay if ref_value is None else ref_value)

    if isinstance(delays_fs, str) and delays_fs.lower() == "all":
        try:
            delays_list = _discover_cached_delay_points_fs(
                sample_name=str(sample_name),
                temperature_K=int(temperature_K),
                excitation_wl_nm=float(excitation_wl_nm),
                fluence_mJ_cm2=float(fluence_mJ_cm2),
                time_window_fs=int(time_window_fs),
                paths=pths,
            )
        except FileNotFoundError:
            delays_list = []
    elif isinstance(delays_fs, int):
        delays_list = [int(delays_fs)]
    else:
        delays_list = [int(x) for x in list(delays_fs)]

    if ref_delay_fs not in delays_list:
        delays_list = sorted(set(delays_list + [int(ref_delay_fs)]))

    missing_delays: List[int] = []
    for d_fs in delays_list:
        try:
            _ensure_xy_exists(
                sample_name=str(sample_name),
                temperature_K=int(temperature_K),
                excitation_wl_nm=float(excitation_wl_nm),
                fluence_mJ_cm2=float(fluence_mJ_cm2),
                time_window_fs=int(time_window_fs),
                delay_fs=int(d_fs),
                azim_window=azim_window,
                paths=pths,
            )
        except FileNotFoundError:
            missing_delays.append(int(d_fs))

    if missing_delays:
        if not compute_if_missing:
            raise FileNotFoundError(
                "Missing cached XY files for delays: "
                f"{missing_delays}. Set compute_if_missing=True to generate them."
            )

        delays_to_compute: Union[str, List[int]]
        if isinstance(delays_fs, str) and delays_fs.lower() == "all":
            delays_to_compute = "all"
        else:
            delays_to_compute = sorted(set(missing_delays + [int(ref_delay_fs)]))

        integrate_delay_1d(
            sample_name=str(sample_name),
            raw_sample_name=raw_sample_name,
            dataset=int(dataset),
            scan_nb=int(scan_nb),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            fluence_mJ_cm2=float(fluence_mJ_cm2),
            time_window_fs=int(time_window_fs),
            delays_fs=delays_to_compute,
            paths=pths,
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            calibration_subdir=calibration_subdir,
            calibration_resolver=calibration_resolver,
            azimuthal_edges=[float(azim_window[0]), float(azim_window[1])],
            include_full=False,
            full_range=azim_window,
            npt=npt,
            force=force,
            ref_delay=ref_delay,
            q_norm_range=q_norm_range,
            overwrite_xy=overwrite_xy,
            azim_offset_deg=azim_offset_deg,
        )

        if isinstance(delays_fs, str) and delays_fs.lower() == "all":
            delays_list = _discover_cached_delay_points_fs(
                sample_name=str(sample_name),
                temperature_K=int(temperature_K),
                excitation_wl_nm=float(excitation_wl_nm),
                fluence_mJ_cm2=float(fluence_mJ_cm2),
                time_window_fs=int(time_window_fs),
                paths=pths,
            )
            if ref_delay_fs not in delays_list:
                delays_list = sorted(set(delays_list + [int(ref_delay_fs)]))

    poni_path_res, _mask_path_res = resolve_calibration(
        sample_name=str(sample_name),
        paths=pths,
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        calibration_subdir=calibration_subdir,
        calibration_resolver=calibration_resolver,
    )
    ai = _load_ai_with_compat(poni_path_res)

    ds_ref = DelayDataset(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        excitation_wl_nm=float(excitation_wl_nm),
        fluence_mJ_cm2=float(fluence_mJ_cm2),
        time_window_fs=int(time_window_fs),
        delay_fs=int(ref_delay_fs),
        paths=pths,
    )
    _, q_ref, I_ref = _load_cached_xy(
        ds_ref,
        azim_window=azim_window,
        wavelength_m=ai.wavelength,
    )

    patterns = []
    for d_fs in sorted(set(int(x) for x in delays_list)):
        if int(d_fs) == int(ref_delay_fs):
            continue

        ds = DelayDataset(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            fluence_mJ_cm2=float(fluence_mJ_cm2),
            time_window_fs=int(time_window_fs),
            delay_fs=int(d_fs),
            paths=pths,
        )
        _, q, I = _load_cached_xy(
            ds,
            azim_window=azim_window,
            wavelength_m=ai.wavelength,
        )
        lab = f"{delay_label_value(d_fs, fs_or_ps=fs_or_ps, digits=digits)}"
        patterns.append((lab, q, I))

    if title is None:
        title = (
            f"{sample_name}. {temperature_K}K.\n"
            f"ex. wl={excitation_wl_nm}nm. flu={fluence_mJ_cm2} mJ/cm$^2$.\n"
            f"tw={time_window_fs}fs. azim=({azim_window[0]},{azim_window[1]})"
        )

    save_kwargs = dict(save=False)
    if save_plots:
        base_dir = (
            Path(save_base_dir)
            if save_base_dir is not None
            else DelayDataset(
                sample_name=str(sample_name),
                temperature_K=int(temperature_K),
                excitation_wl_nm=float(excitation_wl_nm),
                fluence_mJ_cm2=float(fluence_mJ_cm2),
                time_window_fs=int(time_window_fs),
                delay_fs=int(ref_delay_fs),
                paths=pths,
            ).analysis_dir()
        )

        if out_name is None:
            azs = general_utils.azim_range_str(azim_window)
            out_name = f"compare_{azs}_to_delay_{int(ref_delay_fs)}fs"

        save_kwargs = plot_utils.build_save_kwargs(
            save=True,
            base_dir=base_dir,
            figures_subdir="figures/1D_patterns",
            save_name=out_name,
            save_format=save_format,
            save_dpi=save_dpi,
            overwrite=save_overwrite,
        )

    ref_label = f"ref: {delay_label_value(ref_delay_fs, fs_or_ps=fs_or_ps, digits=digits)}"

    fig, axes = plot_utils.Pattern1DPlotter().compare_to_reference(
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

    return fig, axes

