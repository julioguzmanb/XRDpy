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
    "create_fluence_scan_from_delay_scans",
    "plot_1D_abs_and_diffs_delay",
    "plot_1D_abs_and_diffs_fluence",
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
    """Build or validate an AnalysisPaths object.

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
    """Return the directory searched for ID09 PONI and detector-mask files."""
    return Path(paths.path_root) / Path(calibration_subdir)


def _effective_raw_sample_name(
    sample_name: str,
    raw_sample_name: Optional[str] = None,
) -> str:
    """Select and validate the sample name used in the ID09 raw-data tree."""
    raw_name = sample_name if raw_sample_name is None else raw_sample_name
    raw_name = str(raw_name)

    if len(raw_name.strip()) == 0:
        raise ValueError("raw_sample_name must not be empty.")

    return raw_name

    
def _load_ai_with_compat(poni_path: Union[str, Path]):
    """Load a PONI file through the shared legacy-format compatibility layer."""
    ai = txs.utils.load_ai(poni_path)
    return ai


def _require_txs() -> None:
    """Raise an informative import error when the ID09 ``pytxs`` backend is unavailable."""
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
    """Locate raw HDF5 files and BLISS scan groups for an ID09 dataset.

    The source resolves the effective raw sample name, proposal/dataset folder,
    HDF5 filename, and internal scan path. ``read_raw_images`` returns the full
    detector stack and raises a filesystem or HDF5 error when the configured
    scan cannot be found.

    Attributes
    ----------
    sample_name : str
        Analysis-facing sample identifier.
    dataset : int
        ID09 dataset number used in proposal folder and HDF5 names.
    scan_nb : int
        BLISS scan number within the dataset.
    paths : AnalysisPaths
        Raw-data and processed-analysis root configuration.
    raw_sample_name : str or None
        Optional beamline-facing sample name when it differs from
        ``sample_name``.
    """
    sample_name: str
    dataset: int
    scan_nb: int
    paths: AnalysisPaths
    raw_sample_name: Optional[str] = None

    @property
    def effective_raw_sample_name(self) -> str:
        """Return the raw-data sample name after applying any override."""
        return _effective_raw_sample_name(
            sample_name=self.sample_name,
            raw_sample_name=self.raw_sample_name,
        )

    @property
    def dataset_dir(self) -> Path:
        """Return the ID09 raw directory for this numbered dataset."""
        raw_name = self.effective_raw_sample_name
        return (
            Path(self.paths.raw_root)
            / raw_name
            / f"{raw_name}_{int(self.dataset):04d}"
        )

    @property
    def raw_h5_path(self) -> Path:
        """Return the complete filesystem path of the BLISS dataset HDF5 file."""
        raw_name = self.effective_raw_sample_name
        return self.dataset_dir / f"{raw_name}_{int(self.dataset):04d}.h5"

    @property
    def scan_path(self) -> str:
        """Return the conventional raw scan-directory path for this scan number."""
        return str(self.dataset_dir / f"scan{int(self.scan_nb):04d}")

    def read_raw_images(self) -> np.ndarray:
        """Load the complete Rayonix frame stack for the configured BLISS scan."""
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
    """Load raw detector frames from one ID09 scan.

    The scan source resolves facility paths and returns the detector stack with
    its original frame and pixel dimensions.

    Parameters
    ----------
    sample_name : str
        Analysis-facing sample identifier.
    dataset, scan_nb : int
        Numbered ID09 dataset and BLISS scan within that dataset.
    raw_sample_name : str, optional
        Beamline-facing sample name when the raw directory uses another name.
    paths, path_root, raw_subdir, analysis_subdir
        Modern or legacy experiment-path configuration.

    Returns
    -------
    numpy.ndarray
        Detector stack with shape ``(frames, detector_y, detector_x)``.
    """
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
    """Try to auto-resolve calibration files from ``<path_root>/calibration``.

    The newest ``.poni`` containing ``sample_name`` and newest ``.edf`` also
    containing ``mask`` are selected.

    If your project needs stricter rules, pass ``calibration_resolver=...``.

    Parameters
    ----------
    sample_name : str
        Text required in both calibration filenames.
    paths, path_root
        Modern or legacy experiment-root configuration.
    calibration_subdir : path-like
        Calibration directory relative to the experiment root.

    Returns
    -------
    tuple of str
        Resolved PONI path followed by the detector-mask EDF path.

    Raises
    ------
    FileNotFoundError
        If no matching PONI file or mask file exists.
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
    """Resolve calibration file paths.

    Explicit paths take priority. Any missing path is filled by
    ``calibration_resolver`` or :func:`default_calibration_paths`.

    Parameters
    ----------
    sample_name : str
        Sample identifier used during automatic discovery.
    paths, path_root
        Modern or legacy experiment-root configuration.
    poni_path, mask_edf_path : path-like, optional
        Explicit geometry and detector-mask files.
    calibration_subdir : path-like
        Calibration directory relative to the experiment root.
    calibration_resolver : callable, optional
        Custom resolver returning ``(poni_path, mask_path)``.

    Returns
    -------
    tuple of str
        Fully resolved PONI and mask paths.
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
    """Convert an ID09 delay token, such as ``10ps`` or ``-2ns``, to fs.

    Parameters
    ----------
    token : Union[str, bytes, int, float]
        ID09 delay token or numeric delay value to convert.

    Returns
    -------
    int
        Delay rounded to an integer number of femtoseconds.

    Raises
    ------
    ValueError
        If a selector, range, mode, unit, or metadata value is invalid.
    """
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
    """Convert an ID09 delay token to a rounded value in a display unit.

    Parameters
    ----------
    delay_fs : Union[int, float]
        Pump-probe delay in femtoseconds.
    fs_or_ps : str
        Supported display unit used for the converted delay label.
    digits : int
        Number of decimal places used for ordinary display-unit rounding.

    Returns
    -------
    Union[int, float]
        Rounded delay value in the selected display unit.
    """
    return common_azimint_utils.delay_label_value(
        delay_fs,
        fs_or_ps=fs_or_ps,
        digits=digits,
    )


def _normalize_delay_selection(
    delays_fs: Union[int, Sequence[int], str],
    *,
    available_delays_fs: Sequence[int],
) -> List[int]:
    """Resolve numeric or token delay selectors against the scan's delay map."""
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
    """Map each canonical delay token to its detector-frame indices."""
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
    """Build adjacent package-coordinate azimuth windows from ordered edges."""
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
    """Translate a package-coordinate azimuth window to ID09 integration angles."""
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
    """Extract and flatten the absolute-delay trace from a pytxs dataset."""
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
    """Discover femtosecond delays encoded in standardized XY cache filenames."""
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
    """Return unique sorted delays available in an ID09 scan, expressed in fs.

    Parameters
    ----------
    sample_name : str
        Sample identifier used in the standardized analysis directory layout.
    temperature_K : Union[int, float]
        Sample temperature in kelvin.
    excitation_wl_nm : Union[int, float]
        Pump wavelength in nanometres.
    fluence_mJ_cm2 : Union[int, float]
        Pump fluence in mJ/cm².
    time_window_fs : int
        Width of the delay bin or acquisition window in femtoseconds.
    paths : Optional[AnalysisPaths]
        Resolved ``AnalysisPaths`` configuration. It takes precedence over legacy path arguments.
    path_root : Optional[Union[str, Path]]
        Root directory containing raw and analysis data trees.
    raw_subdir : Union[str, Path]
        Raw-data path relative to ``path_root``.
    analysis_subdir : Union[str, Path]
        Analysis-directory path relative to ``path_root``.

    Returns
    -------
    List[int]
        Unique sorted delay points in femtoseconds.
    """
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
    """Return a cached ID09 XY path, reducing the source scan when permitted."""
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
    normalize: bool = False,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Load one cached ID09 pattern and convert its radial axis to q."""
    azim_str = general_utils.azim_range_str(azim_window)
    xy_path = dataset_obj.xy_path(azim_str)
    if not xy_path.exists():
        raise FileNotFoundError(str(xy_path))
    two_theta, I = general_utils.load_xy(xy_path)
    q = general_utils.two_theta_to_q(two_theta, wavelength_m)
    if normalize:
        I = general_utils.normalize_y_by_mean_in_xrange(q, I, q_norm_range)
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
    polarization_factor: Optional[float] = None,
):
    """Run pytxs azimuthal reduction for selected delay tokens and windows."""
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
    polarization_factor = common_azimint_utils.normalize_polarization_factor(
        polarization_factor
    )

    azav = txs.azav.integrate1d_dataset(
        src.scan_path,
        ai,
        mask=mask,
        npt=int(npt),
        force=bool(force),
        azimuthal_range=beamline_window,
        polarization_factor=polarization_factor,
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
    """Write pytxs results using standardized delay and azimuthal filenames."""
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
    polarization_factor: Optional[float] = None,
):
    """Reduce one ID09 delay scan and write standardized azimuthal XY files.

    ``sample_name`` is used for output/cache naming.
    ``raw_sample_name`` is used only to locate the raw HDF5 dataset. If omitted,
    raw access falls back to ``sample_name``.

    Output XY files follow the SAME folder structure and filename conventions
    as the shared trxrdpy analysis pipeline.

    Parameters
    ----------
    sample_name, dataset, scan_nb
        Analysis identifier and raw ID09 dataset/scan coordinates.
    temperature_K, excitation_wl_nm, fluence_mJ_cm2, time_window_fs
        Experimental metadata encoded in standardized output paths.
    delays_fs : int, sequence of int, or "all"
        Delay points to export, expressed in femtoseconds.
    raw_sample_name : str, optional
        Raw directory sample name when it differs from ``sample_name``.
    paths, path_root, raw_subdir, analysis_subdir
        Modern or legacy experiment-path configuration.
    poni_path, mask_edf_path, calibration_subdir, calibration_resolver
        Explicit or automatically resolved pyFAI calibration settings.
    azimuthal_edges, include_full, full_range
        Package-coordinate sector boundaries and optional full sector.
    npt : int
        Number of radial integration bins.
    force : bool
        Force the underlying ``pytxs`` reduction to be recomputed.
    ref_delay : str or number
        Delay token used as the reduction reference.
    q_norm_range : tuple of float
        q interval used by the reduction normalization.
    overwrite_xy : bool
        Replace standardized XY files that already exist.
    azim_offset_deg : float
        Offset from package azimuth coordinates to ID09 coordinates.
    polarization_factor : float, optional
        pyFAI polarization correction; ``None`` disables it.

    Returns
    -------
    tuple
        ``(source, datasets, saved_paths)`` where ``source`` describes the raw
        scan, ``datasets`` contains one :class:`DelayDataset` per selected
        delay, and ``saved_paths`` maps each delay and azimuth label to its XY
        filename.
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
            polarization_factor=polarization_factor,
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
    normalize: bool = True,
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    xlim: Tuple[float, float] = (1.5, 4.5),
    ylim_top=None,
    ylim_diff=None,
    vlines_peak: Optional[Tuple[float, float]] = None,
    vlines_bckg: Optional[Tuple[float, float]] = None,
    fs_or_ps: str = "ps",
    digits: int = 2,
    delay_offset_fs: float = 0.0,
    fluence_scale: float = 1.0,
    fluence_offset: float = 0.0,
    fluence_unit: str = "mJ/cm$^2$",
    max_curves: Optional[int] = None,
    title: Optional[str] = None,
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    save_plots: bool = False,
    out_name: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 400,
    save_overwrite: bool = True,
    save_base_dir: Optional[Union[str, Path]] = None,
):
    """Plot delay-resolved 1D patterns against a delay reference.

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

    Parameters
    ----------
    sample_name, dataset, scan_nb
        Analysis identifier and raw ID09 dataset/scan coordinates.
    temperature_K, excitation_wl_nm, fluence_mJ_cm2, time_window_fs
        Experimental metadata identifying the cached patterns.
    delays_fs : int, sequence of int, or "all"
        Delay patterns to plot in femtoseconds.
    ref_type, ref_value, ref_delay
        Reference selection. ID09 currently accepts only a delay reference.
    raw_sample_name, paths, path_root, raw_subdir, analysis_subdir
        Raw naming and experiment-path configuration.
    poni_path, mask_edf_path, calibration_subdir, calibration_resolver
        Calibration settings used when files must be generated or q converted.
    azim_window, npt, force, q_norm_range, normalize
        Azimuthal integration, radial binning, reduction, and normalization
        settings.
    compute_if_missing, overwrite_xy
        Control on-demand integration and replacement of cached XY files.
    xlim, ylim_top, ylim_diff, vlines_peak, vlines_bckg
        Axis limits and optional highlighted q intervals.
    fs_or_ps, digits, delay_offset_fs, fluence_scale, fluence_offset,
    fluence_unit, max_curves, title
        Delay-label unit, rounding precision, display-only delay offset,
        display-only fluence title scale/offset, maximum displayed curves
        including reference, and optional title.
    azim_offset_deg, polarization_factor
        ID09 azimuth conversion and pyFAI polarization correction.
    save_plots, out_name, save_format, save_dpi, save_overwrite, save_base_dir
        Figure-output controls.

    Returns
    -------
    tuple
        Matplotlib figure and the two axes for absolute and difference traces.

    Raises
    ------
    NotImplementedError
        If ``ref_type`` is not ``"delay"``.
    FileNotFoundError
        If required XY files are missing and generation is disabled or fails.
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
            polarization_factor=polarization_factor,
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
        normalize=bool(normalize),
        q_norm_range=q_norm_range,
    )

    non_ref_delays = [
        int(x)
        for x in sorted(set(int(x) for x in delays_list))
        if int(x) != int(ref_delay_fs)
    ]
    max_non_ref = None if max_curves is None else max(int(max_curves) - 1, 0)
    non_ref_delays = common_azimint_utils.evenly_spaced_subset(
        non_ref_delays,
        max_non_ref,
    )

    patterns = []
    for d_fs in non_ref_delays:
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
            normalize=bool(normalize),
            q_norm_range=q_norm_range,
        )
        lab = f"{delay_label_value(d_fs + float(delay_offset_fs), fs_or_ps=fs_or_ps, digits=digits)}"
        patterns.append((lab, q, I))

    if title is None:
        display_fluence = float(fluence_mJ_cm2) * float(fluence_scale) + float(fluence_offset)
        title = (
            f"{sample_name}. {temperature_K}K.\n"
            f"ex. wl={excitation_wl_nm}nm. flu={display_fluence:g} {fluence_unit}.\n"
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

    ref_label = f"ref: {delay_label_value(ref_delay_fs + float(delay_offset_fs), fs_or_ps=fs_or_ps, digits=digits)}"

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
        legend_title=f"Delay [{general_utils.time_unit_label(fs_or_ps)}]",
        legend_loc="upper left",
        legend_outside=True,
        **save_kwargs,
    )

    return fig, axes


def create_fluence_scan_from_delay_scans(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    delay_fs: int,
    time_window_fs: int,
    fluences_mJ_cm2: Union[float, int, Sequence[Union[float, int]], str] = "all",
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Union[str, Path] = "",
    analysis_subdir: Union[str, Path] = "analysis",
    azimuthal_edges: Sequence[Union[int, float]] = tuple(np.arange(-90, 90 + 20, 45)),
    include_full: bool = True,
    full_range: Tuple[float, float] = (-90.0, 90.0),
    copy_2d_image: bool = False,
    overwrite: bool = False,
):
    """Create a synthetic fluence-scan cache from already processed delay scans.

    This function does NOT perform azimuthal integration. It assumes the ID09
    delay-scan XY cache already exists, and reorganizes the selected delay point
    across several fluences into the shared fluence-scan folder layout.

    Parameters
    ----------
    delay_fs
        Fixed delay to extract from each already-processed delay scan.
    fluences_mJ_cm2
        One fluence, many fluences, or "all" to auto-discover available
        fluence folders in the delay-scan tree.
    copy_2d_image
        If True, also copy the corresponding 2D image when it exists.
        Disabled by default because ID09 XY creation does not require it.

    Returns
    -------
    datasets : list[FluenceDataset]
    copied_paths : dict[float, dict[str, str]]
        Mapping: fluence_mJ_cm2 -> {azim_range_str -> xy_path, "2D_image" -> img_path}
        The "2D_image" entry is included only when ``copy_2d_image=True``.
    """
    pths = _resolve_paths(
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
    )

    if isinstance(fluences_mJ_cm2, str):
        if fluences_mJ_cm2.lower() != "all":
            raise ValueError(
                "fluences_mJ_cm2 string must be 'all' or a float/list of floats."
            )

        base = (
            Path(pths.analysis_root)
            / str(sample_name)
            / f"temperature_{general_utils.to_int(temperature_K)}K"
        )
        wl_tag = general_utils.wl_tag_nm(excitation_wl_nm)

        delay_roots = [
            base / f"excitation_wl_{wl_tag}nm" / "delay",
            base / f"excitation_wl_{excitation_wl_nm}nm" / "delay",
        ]

        delay_root = None
        for cand in delay_roots:
            if cand.is_dir():
                delay_root = cand
                break
        if delay_root is None:
            delay_root = delay_roots[0]

        if not delay_root.is_dir():
            raise FileNotFoundError(
                "Delay-scan base folder not found:\n"
                f"  {delay_root}\n"
                "Create/process the ID09 delay scans first."
            )

        patt = re.compile(r"^fluence_([0-9]+(?:p[0-9]+)?)mJ$")
        fl_list: List[float] = []

        for child in sorted(delay_root.iterdir()):
            if not child.is_dir():
                continue
            m = patt.match(child.name)
            if not m:
                continue
            try:
                fl_list.append(float(str(m.group(1)).replace("p", ".")))
            except Exception:
                continue

        fl_list = sorted(set(fl_list))
        if len(fl_list) == 0:
            raise FileNotFoundError(
                "No fluence folders found in:\n"
                f"  {delay_root}\n"
            )

    elif isinstance(fluences_mJ_cm2, (int, float)):
        fl_list = [float(fluences_mJ_cm2)]
    else:
        fl_list = sorted(set(float(x) for x in list(fluences_mJ_cm2)))

    windows = _standard_windows_from_edges(
        azimuthal_edges,
        include_full=include_full,
        full_range=full_range,
    )

    copied_paths: Dict[float, Dict[str, str]] = {}
    datasets_out: Dict[float, common_azimint_utils.FluenceDataset] = {}

    for flu in fl_list:
        src_ds = DelayDataset(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            fluence_mJ_cm2=float(flu),
            time_window_fs=int(time_window_fs),
            delay_fs=int(delay_fs),
            paths=pths,
        )

        dst_ds = common_azimint_utils.FluenceDataset(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            fluence_mJ_cm2=float(flu),
            time_window_fs=int(time_window_fs),
            delay_fs=int(delay_fs),
            paths=pths,
        )

        for win in windows:
            azim_str = general_utils.azim_range_str(win)

            src_xy = _ensure_xy_exists(
                sample_name=str(sample_name),
                temperature_K=int(temperature_K),
                excitation_wl_nm=float(excitation_wl_nm),
                fluence_mJ_cm2=float(flu),
                time_window_fs=int(time_window_fs),
                delay_fs=int(delay_fs),
                azim_window=win,
                paths=pths,
            )

            dst_xy = dst_ds.xy_path(azim_str)
            if overwrite or (not dst_xy.exists()):
                dst_xy.parent.mkdir(parents=True, exist_ok=True)
                dst_xy.write_bytes(src_xy.read_bytes())

            copied_paths.setdefault(float(flu), {})[azim_str] = str(dst_xy)

        if copy_2d_image:
            src_img = src_ds.img_path()
            if not src_img.exists():
                raise FileNotFoundError(
                    "Requested copy_2d_image=True, but source 2D image is missing:\n"
                    f"  {src_img}"
                )

            dst_img = dst_ds.img_path()
            if overwrite or (not dst_img.exists()):
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                dst_img.write_bytes(src_img.read_bytes())

            copied_paths.setdefault(float(flu), {})["2D_image"] = str(dst_img)

        datasets_out[float(flu)] = dst_ds

    datasets = [datasets_out[k] for k in sorted(datasets_out)]
    return datasets, copied_paths


def plot_1D_abs_and_diffs_fluence(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    delay_fs: int,
    time_window_fs: int,
    fluences_mJ_cm2: Union[float, Sequence[float], str] = "all",
    ref_type: str = "fluence",
    ref_value: Optional[Union[float, int, str, Sequence[int]]] = None,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Union[str, Path] = "",
    analysis_subdir: Union[str, Path] = "analysis",
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    calibration_subdir: Union[str, Path] = "calibration",
    calibration_resolver: Optional[CalibrationResolver] = None,
    azim_window: Tuple[float, float] = (-90.0, 90.0),
    polarization_factor: Optional[float] = None,
    compute_if_missing: bool = True,
    copy_2d_image_if_missing: bool = False,
    overwrite_xy: bool = False,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    xlim: Tuple[float, float] = (1.5, 4.5),
    ylim_top=None,
    ylim_diff=None,
    vlines_peak: Optional[Tuple[float, float]] = None,
    vlines_bckg: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    fluence_scale: float = 1.0,
    fluence_offset: float = 0.0,
    delay_offset_fs: float = 0.0,
    fs_or_ps: str = "fs",
    digits: int = 2,
    fluence_digits: Optional[int] = None,
    fluence_unit: str = "mJ/cm$^2$",
    max_curves: Optional[int] = None,
    save_plots: bool = False,
    out_name: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 400,
    save_overwrite: bool = True,
    save_base_dir: Optional[Union[str, Path]] = None,
):
    """Plot fluence-resolved 1D patterns at fixed delay, using a synthetic fluence
    scan cache created from already processed ID09 delay scans.

    ``polarization_factor`` is accepted for API consistency. It must be applied
    while integrating each source delay scan; copying cached XY files into a
    synthetic fluence scan cannot change their polarization correction.

    Behavior
    --------
    - Reads cached XY files from the standardized fluence-scan folder.
    - If requested files are missing and ``compute_if_missing=True``, it first
      creates the synthetic fluence scan via
      ``create_fluence_scan_from_delay_scans(...)``.
    - Optionally normalizes loaded 1D curves in-memory before computing
      differentials and plotting.

    Notes
    -----
    Supported references:
      - ``ref_type='fluence'`` with ``ref_value=<fluence>``
      - ``ref_type='dark'`` with optional ``ref_value=<dark spec>``

    Parameters
    ----------
    sample_name, temperature_K, excitation_wl_nm, delay_fs, time_window_fs
        Experimental identity and fixed-delay metadata locating the patterns.
    fluences_mJ_cm2 : float, sequence of float, or "all"
        Fluence points to compare in mJ/cm².
    ref_type, ref_value
        Fluence or dark reference selection.
    paths, path_root, raw_subdir, analysis_subdir
        Modern or legacy experiment-path configuration.
    poni_path, mask_edf_path, calibration_subdir, calibration_resolver
        Calibration settings used for radial-axis conversion and discovery.
    azim_window, polarization_factor
        Cached sector to load and correction provenance expected for its XY data.
    compute_if_missing, copy_2d_image_if_missing, overwrite_xy
        Synthetic fluence-cache creation and replacement controls.
    normalize, q_norm_range
        Optional in-memory intensity normalization.
    xlim, ylim_top, ylim_diff, vlines_peak, vlines_bckg, title
        Axis limits, highlighted q regions, and figure title.
    fluence_scale, fluence_offset, delay_offset_fs, fs_or_ps, digits,
    fluence_digits, fluence_unit, max_curves
        Display-only fluence/delay corrections and curve-count limiting.
        ``digits`` controls the fixed-delay label; ``fluence_digits`` controls
        fluence labels in the legend.
    save_plots, out_name, save_format, save_dpi, save_overwrite, save_base_dir
        Figure-output controls.

    Returns
    -------
    tuple
        Matplotlib figure and the two axes for absolute and difference traces.

    Raises
    ------
    ValueError
        If the reference type or required reference value is invalid.
    FileNotFoundError
        If source or synthetic fluence data cannot be located.
    """
    def _fluence_display_label(value: float) -> str:
        ndig = digits if fluence_digits is None else fluence_digits
        try:
            rounded = round(float(value), int(ndig))
        except Exception:
            return f"{float(value):g}"
        return f"{rounded:g}"

    pths = _resolve_paths(
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
    )

    ref_type_n = str(ref_type).strip().lower()
    if ref_type_n not in ("fluence", "dark"):
        raise ValueError("ref_type must be 'fluence' or 'dark'.")

    fl_list: List[float]
    if isinstance(fluences_mJ_cm2, str):
        if fluences_mJ_cm2.lower() != "all":
            raise ValueError(
                "fluences_mJ_cm2 string must be 'all' or a float/list of floats."
            )
        fl_list = []
    elif isinstance(fluences_mJ_cm2, (int, float)):
        fl_list = [float(fluences_mJ_cm2)]
    else:
        fl_list = sorted(set(float(x) for x in list(fluences_mJ_cm2)))

    if ref_type_n == "fluence":
        if ref_value is None:
            raise ValueError("ref_type='fluence' requires ref_value=...")
        ref_f = float(ref_value)
        if ref_f not in fl_list:
            fl_list = sorted(set(fl_list + [ref_f]))

    if compute_if_missing:
        datasets_created, _copied = create_fluence_scan_from_delay_scans(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            delay_fs=int(delay_fs),
            time_window_fs=int(time_window_fs),
            fluences_mJ_cm2=("all" if isinstance(fluences_mJ_cm2, str) else fl_list),
            paths=pths,
            copy_2d_image=bool(copy_2d_image_if_missing),
            overwrite=bool(overwrite_xy),
            azimuthal_edges=[float(azim_window[0]), float(azim_window[1])],
            include_full=False,
            full_range=azim_window,
        )
        fl_list = sorted(set(float(ds.fluence_mJ_cm2) for ds in datasets_created))

    if isinstance(fluences_mJ_cm2, str) and fluences_mJ_cm2.lower() == "all" and (not compute_if_missing):
        tmp_ds = common_azimint_utils.FluenceDataset(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            fluence_mJ_cm2=0.0,
            time_window_fs=int(time_window_fs),
            delay_fs=int(delay_fs),
            paths=pths,
        )
        folder = tmp_ds.xy_folder()
        if not folder.is_dir():
            raise FileNotFoundError(
                "Synthetic fluence-scan XY folder not found:\n"
                f"  {folder}\n"
                "Set compute_if_missing=True to create it from the delay scans."
            )

        wl_tag = general_utils.wl_tag_nm(excitation_wl_nm)
        T = general_utils.to_int(temperature_K)
        tw = int(time_window_fs)
        azim_str = general_utils.azim_range_str(azim_window)

        patt = re.compile(
            rf"^{re.escape(sample_name)}_{T}K_{re.escape(wl_tag)}nm_"
            rf"([0-9]+(?:p[0-9]+)?)mJ_{tw}fs_{int(delay_fs)}fs_"
            rf"{re.escape(azim_str)}\.xy$"
        )

        fl_list = []
        for p in folder.iterdir():
            if p.suffix.lower() != ".xy":
                continue
            m = patt.match(p.name)
            if not m:
                continue
            fl_list.append(float(str(m.group(1)).replace("p", ".")))

        fl_list = sorted(set(fl_list))
        if len(fl_list) == 0:
            raise FileNotFoundError(
                "No synthetic fluence-scan XY files found in:\n"
                f"  {folder}\n"
                "Set compute_if_missing=True to create them from the delay scans."
            )

        if ref_type_n == "fluence" and ref_f not in fl_list:
            fl_list = sorted(set(fl_list + [ref_f]))

    if len(fl_list) == 0:
        raise FileNotFoundError("No fluences found to compare (empty fl_list).")

    if not compute_if_missing:
        for f in fl_list:
            ds_chk = common_azimint_utils.FluenceDataset(
                sample_name=str(sample_name),
                temperature_K=int(temperature_K),
                excitation_wl_nm=float(excitation_wl_nm),
                fluence_mJ_cm2=float(f),
                time_window_fs=int(time_window_fs),
                delay_fs=int(delay_fs),
                paths=pths,
            )
            xy_path = ds_chk.xy_path(general_utils.azim_range_str(azim_window))
            if not xy_path.exists():
                raise FileNotFoundError(
                    f"Missing synthetic fluence-scan XY file:\n  {xy_path}\n"
                    "Set compute_if_missing=True to create it from the delay scans."
                )

    poni_path_res, _mask_path_res = resolve_calibration(
        sample_name=str(sample_name),
        paths=pths,
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        calibration_subdir=calibration_subdir,
        calibration_resolver=calibration_resolver,
    )
    ai = _load_ai_with_compat(poni_path_res)

    if ref_type_n == "fluence":
        ds_ref = common_azimint_utils.FluenceDataset(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            fluence_mJ_cm2=float(ref_f),
            time_window_fs=int(time_window_fs),
            delay_fs=int(delay_fs),
            paths=pths,
        )
        ref_f_display = float(ref_f) * float(fluence_scale) + float(fluence_offset)
        ref_label = f"ref: {_fluence_display_label(ref_f_display)} {fluence_unit}"
        ref_tag_for_file = f"fluence_{general_utils.fluence_tag_file(ref_f)}mJ"
    else:
        resolved_tag = None
        if ref_value is not None:
            resolved_tag = common_azimint_utils.dark_tag_from_scan_spec(ref_value)

        ds_ref = common_azimint_utils.DarkDataset(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            dark_tag=resolved_tag,
            paths=pths,
        )
        ref_label = f"ref: dark\n{common_azimint_utils.pretty_dark_tag(ds_ref.dark_tag)}"
        ref_tag_for_file = f"dark_{ds_ref.dark_tag}"

    _, q_ref, I_ref = _load_cached_xy(
        ds_ref,
        azim_window=azim_window,
        wavelength_m=ai.wavelength,
        normalize=bool(normalize),
        q_norm_range=q_norm_range,
    )

    non_ref_fluences = [
        float(f)
        for f in sorted(fl_list)
        if not (ref_type_n == "fluence" and abs(float(f) - float(ref_f)) < 1e-12)
    ]
    max_non_ref = None if max_curves is None else max(int(max_curves) - 1, 0)
    non_ref_fluences = common_azimint_utils.evenly_spaced_subset(
        non_ref_fluences,
        max_non_ref,
    )

    patterns = []
    for f in non_ref_fluences:
        if ref_type_n == "fluence" and abs(float(f) - float(ref_f)) < 1e-12:
            continue

        ds = common_azimint_utils.FluenceDataset(
            sample_name=str(sample_name),
            temperature_K=int(temperature_K),
            excitation_wl_nm=float(excitation_wl_nm),
            fluence_mJ_cm2=float(f),
            time_window_fs=int(time_window_fs),
            delay_fs=int(delay_fs),
            paths=pths,
        )
        _, q, I = _load_cached_xy(
            ds,
            azim_window=azim_window,
            wavelength_m=ai.wavelength,
            normalize=bool(normalize),
            q_norm_range=q_norm_range,
        )
        f_display = float(f) * float(fluence_scale) + float(fluence_offset)
        patterns.append((_fluence_display_label(f_display), q, I))

    if title is None:
        display_delay_fs = int(delay_fs) + float(delay_offset_fs)
        display_delay = delay_label_value(
            display_delay_fs,
            fs_or_ps=fs_or_ps,
            digits=digits,
        )
        display_unit = general_utils.time_unit_label(fs_or_ps)
        title = (
            f"{sample_name}. {temperature_K}K.\n"
            f"ex. wl={excitation_wl_nm}nm. delay={display_delay:g} {display_unit}.\n"
            f"tw={time_window_fs}fs. azim=({azim_window[0]},{azim_window[1]})"
        )

    save_kwargs = dict(save=False)
    if save_plots:
        base_dir = (
            Path(save_base_dir)
            if save_base_dir is not None
            else common_azimint_utils.FluenceDataset(
                sample_name=str(sample_name),
                temperature_K=int(temperature_K),
                excitation_wl_nm=float(excitation_wl_nm),
                fluence_mJ_cm2=float(sorted(fl_list)[0]),
                time_window_fs=int(time_window_fs),
                delay_fs=int(delay_fs),
                paths=pths,
            ).analysis_dir()
        )

        if out_name is None:
            azs = general_utils.azim_range_str(azim_window)
            out_name = f"compare_fluence_{azs}_to_{ref_tag_for_file}"

        save_kwargs = plot_utils.build_save_kwargs(
            save=True,
            base_dir=base_dir,
            figures_subdir="figures/1D_patterns",
            save_name=out_name,
            save_format=save_format,
            save_dpi=save_dpi,
            overwrite=save_overwrite,
        )

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
        legend_title=f"Fluence [{fluence_unit}]",
        legend_loc="upper left",
        legend_outside=True,
        **save_kwargs,
    )

    return fig, axes
