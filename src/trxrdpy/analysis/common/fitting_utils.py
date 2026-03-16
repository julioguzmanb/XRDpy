# fitting_utils.py
"""
Peak fitting utilities for delay-series (and reusable for calibration later).

Design goals:
- OOP engine: fitting logic + dataset/XY retrieval.
- Minimal user-facing "experiment intent" here (peak dictionaries belong in fitting.py).
- Reuse general helpers from general_utils.py when available.
- Plotting is handled via plot_utils (NOT here).

Model:
- Linear background (degree-1 polynomial) + Pseudo-Voigt peak
- Pseudo-Voigt fraction is fixed by default to 0.3
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union



import numpy as np
import pandas as pd

from . import general_utils  
from . import azimint_utils
from . import plot_utils
from .paths import AnalysisPaths

from tqdm.auto import tqdm as _tqdm  


PeakSpecDict = Dict[str, Any]
AzimWindow = Tuple[float, float]


def _default_fitting_csv_path(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    fluence_mJ_cm2: float,
    time_window_fs: int,
    out_csv_name: str = "peak_fits_delay.csv",
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,          # legacy fallback
    analysis_subdir: Optional[Union[str, Path]] = None,    # legacy fallback
) -> str:
    ds = azimint_utils.DelayDataset(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        excitation_wl_nm=float(excitation_wl_nm),
        fluence_mJ_cm2=float(fluence_mJ_cm2),
        time_window_fs=int(time_window_fs),
        delay_fs=0,
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )
    return str(ds.analysis_dir() / "fitting" / str(out_csv_name))


def _default_fluence_fitting_csv_path(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    delay_fs: int,
    time_window_fs: int,
    out_csv_name: str = "peak_fits_fluence.csv",
    fluence_for_paths_mJ_cm2: float = 1.0,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,          # legacy fallback
    analysis_subdir: Optional[Union[str, Path]] = None,    # legacy fallback
) -> str:
    """
    Default CSV path for fluence scans.

    Important detail: FluenceDataset.analysis_dir() does NOT depend on fluence,
    so fluence_for_paths_mJ_cm2 is just a harmless placeholder used to build the dataset.
    """
    ds = azimint_utils.FluenceDataset(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        excitation_wl_nm=float(excitation_wl_nm),
        fluence_mJ_cm2=float(fluence_for_paths_mJ_cm2),
        time_window_fs=int(time_window_fs),
        delay_fs=int(delay_fs),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )
    return str(ds.analysis_dir() / "fitting" / str(out_csv_name))


def _tagged_out_csv_name(out_csv_name: str, *, phi_mode: str, phi_reduce: str) -> str:
    name = str(out_csv_name)
    if "_phiavg_" in name:
        return name

    pm = str(phi_mode).strip()
    pr = str(phi_reduce).strip()

    if pm != "phi_avg":
        return name

    root, ext = os.path.splitext(name)
    if ext.lower() != ".csv":
        ext = ext or ""
    return f"{root}_phiavg_{pr}{ext}"


def _candidate_csv_names(out_csv_name: str) -> Sequence[str]:
    name = str(out_csv_name)
    root, ext = os.path.splitext(name)
    ext = ext if ext else ".csv"
    return [
        name,
        f"{root}_phiavg_sum{ext}",
        f"{root}_phiavg_mean{ext}",
        f"{root}_sepphi{ext}",
    ]


def _coerce_group_to_phi_label(g: Any) -> str:
    if isinstance(g, tuple) and len(g) == 2:
        phi0, phi1 = float(g[0]), float(g[1])
        a, b = (min(phi0, phi1), max(phi0, phi1))
        if abs(a + 90.0) < 1e-9 and abs(b - 90.0) < 1e-9:
            return "Full"
        c = _phi_center_abs_for_sym(phi0, phi1)
        if abs(c - round(c)) < 1e-9:
            return str(int(round(c)))
        return f"{c:g}"

    if isinstance(g, (int, float, np.integer, np.floating)):
        x = float(g)
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:g}"

    return str(g)


def _extract_phi_range_from_string(s: str) -> Tuple[float, float]:
    txt = str(s)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", txt)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    raise ValueError(f"Could not parse (phi0, phi1) from string: {s!r}")


def _default_fluence_fitting_figures_dir(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    delay_fs: int,
    time_window_fs: int,
    fluence_for_paths_mJ_cm2: float = 1.0,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,          # legacy fallback
    analysis_subdir: Optional[Union[str, Path]] = None,    # legacy fallback
) -> str:
    """
    Default figures dir for fluence scans:
      <analysis_dir>/figures/fitting

    Recycles FluenceDataset.analysis_dir() exactly like
    _default_fluence_fitting_csv_path.
    """
    ds = azimint_utils.FluenceDataset(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        excitation_wl_nm=float(excitation_wl_nm),
        fluence_mJ_cm2=float(fluence_for_paths_mJ_cm2),  # placeholder
        time_window_fs=int(time_window_fs),
        delay_fs=int(delay_fs),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )
    return str(ds.analysis_dir() / "figures" / "fitting")


def _make_single_peak_model():
    from lmfit.models import PolynomialModel, PseudoVoigtModel

    bg = PolynomialModel(degree=1, prefix="bg_")
    pv = PseudoVoigtModel(prefix="pv_")
    return bg + pv


def _compute_r2(y: np.ndarray, yfit: np.ndarray) -> float:
    fn = getattr(general_utils, "compute_r2", None)
    if not callable(fn):
        raise AttributeError("general_utils.compute_r2 is required but was not found.")
    return float(fn(np.asarray(y, float), np.asarray(yfit, float)))


def _pv_fwhm_from_result(result) -> float:
    try:
        p = result.params.get("pv_fwhm", None)
        if p is not None:
            v = float(p.value)
            if np.isfinite(v):
                return float(v)
    except Exception:
        pass

    try:
        sigma = float(result.params["pv_sigma"].value)
        return float(2.354820045 * sigma)
    except Exception:
        return float("nan")


def _eval_components(model, params, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    yfit = np.asarray(model.eval(params, x=x), float)
    comps = model.eval_components(params=params, x=x)
    bg = np.asarray(comps.get("bg_", np.zeros_like(x)), float)
    pv = np.asarray(comps.get("pv_", np.zeros_like(x)), float)
    return yfit, bg, pv


def _phi_width(phi0: float, phi1: float) -> float:
    return abs(float(phi1) - float(phi0))


def _is_crossing_zero(phi0: float, phi1: float) -> bool:
    a, b = (float(phi0), float(phi1))
    lo, hi = (a, b) if a <= b else (b, a)
    return lo <= 0.0 <= hi


def _phi_center_abs_for_sym(phi0: float, phi1: float) -> float:
    if _is_crossing_zero(phi0, phi1):
        return 0.0
    return 0.5 * (abs(float(phi0)) + abs(float(phi1)))


def _canonical_sym_key(phi0: float, phi1: float, *, tol: float = 1e-9):
    w = _phi_width(phi0, phi1)
    if _is_crossing_zero(phi0, phi1):
        return ("cross0", round(w / tol) * tol)
    a0 = abs(float(phi0))
    a1 = abs(float(phi1))
    lo = min(a0, a1)
    hi = max(a0, a1)
    return (round(lo / tol) * tol, round(hi / tol) * tol)


def _format_phi_label(phi0: float, phi1: float, phi_mode: str) -> str:
    phi0 = float(phi0)
    phi1 = float(phi1)
    pm = str(phi_mode).strip()

    def _fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:g}"

    a, b = (phi0, phi1) if phi0 <= phi1 else (phi1, phi0)

    if pm == "phi_avg":
        if abs(a + 90.0) < 1e-9 and abs(b - 90.0) < 1e-9:
            return "Full"
        c = _phi_center_abs_for_sym(phi0, phi1)
        return _fmt(c)

    return f"({_fmt(phi0)}, {_fmt(phi1)})"


def merge_phi_symmetric_patterns(pattern_entries: Sequence[Dict[str, Any]], *, reduce: str = "sum") -> List[Dict[str, Any]]:
    red = str(reduce).strip().lower()
    if red not in ("sum", "mean"):
        raise ValueError(f"reduce must be 'sum' or 'mean', got: {reduce}")

    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for e in pattern_entries:
        phi0, phi1 = float(e["phi0"]), float(e["phi1"])
        k = _canonical_sym_key(phi0, phi1)
        groups.setdefault(k, []).append(e)

    merged: List[Dict[str, Any]] = []
    for k, items in groups.items():
        if len(items) == 1:
            merged.append(items[0])
            continue

        base = items[0]
        x = np.asarray(base["x"], float)
        y = np.asarray(base["y"], float).copy()

        for it in items[1:]:
            xi = np.asarray(it["x"], float)
            yi = np.asarray(it["y"], float)
            if (len(xi) != len(x)) or (np.max(np.abs(xi - x)) > 1e-12):
                raise ValueError("Cannot merge phi patterns: x-grids differ.")
            y = y + yi

        if red == "mean":
            y = y / float(len(items))

        if isinstance(k[0], str) and k[0] == "cross0":
            phi0_m, phi1_m = float(base["phi0"]), float(base["phi1"])
        else:
            phi0_m, phi1_m = float(k[0]), float(k[1])

        out = dict(base)
        out["phi0"] = phi0_m
        out["phi1"] = phi1_m
        out["x"] = x
        out["y"] = y
        merged.append(out)

    merged.sort(key=lambda e: _phi_center_abs_for_sym(float(e["phi0"]), float(e["phi1"])))
    return merged


def infer_common_phi_half_aperture(df_or_rows: Union[pd.DataFrame, Sequence[Dict[str, Any]]]) -> Optional[float]:
    try:
        phiw = np.array(
            [abs(float(a) - float(b)) for a, b in zip(df_or_rows["phi0"], df_or_rows["phi1"])],  # type: ignore[index]
            dtype=float,
        )
    except Exception:
        phiw = np.array([abs(float(r["phi1"]) - float(r["phi0"])) for r in df_or_rows], dtype=float)  # type: ignore[arg-type]

    if phiw.size == 0:
        return None
    w0 = float(phiw[0])
    if float(np.max(np.abs(phiw - w0))) < 1e-9:
        return 0.5 * w0
    return None


def _sanitize_path_token(s: str) -> str:
    fn = getattr(general_utils, "sanitize_tag", None)
    if callable(fn):
        out = str(fn(s))
        return out if out else "x"

    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return s if s else "x"


def _normalize_azim_windows(
    azim_windows: Optional[Sequence[Tuple[float, float]]] = None,
) -> Sequence[Tuple[float, float]]:
    if azim_windows is None or len(list(azim_windows)) == 0:
        return [(-90.0, 90.0)]
    return [(float(a), float(b)) for (a, b) in list(azim_windows)]


def make_delay_fit_overlay_stem(
    *,
    sample_name: str,
    temperature_K: Union[int, float],
    excitation_wl_nm: Union[int, float],
    fluence_mJ_cm2: Union[int, float],
    time_window_fs: int,
    is_reference: bool,
    delay_fs: Optional[int],
    azim_str: str,
) -> str:
    sample_name = str(sample_name)
    temperature_K = general_utils.to_int(temperature_K)
    wl_tok = int(float(excitation_wl_nm))
    flu_tok = str(float(fluence_mJ_cm2)).replace(".", "p")
    tw_tok = int(time_window_fs)

    delay_tok = "ref" if bool(is_reference) else f"{int(delay_fs)}fs"
    az_tok = _sanitize_path_token(str(azim_str))

    return (
        f"{sample_name}_"
        f"{int(temperature_K)}K_"
        f"{int(wl_tok)}nm_"
        f"{flu_tok}mJ_"
        f"{int(tw_tok)}fs_"
        f"{delay_tok}_"
        f"{az_tok}.xy"
    )


def resolve_fluence_fitting_csv_path(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    delay_fs: int,
    time_window_fs: int,
    out_csv_name: str = "peak_fits_fluence.csv",
    phi_mode: str = "separate_phi",
    phi_reduce: str = "sum",
    fluence_for_paths_mJ_cm2: float = 1.0,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Robust resolver:
      - builds the default path using _default_fluence_fitting_csv_path(...)
      - tries candidate variants via _candidate_csv_names(...)
      - if phi_mode='phi_avg', also tries the tagged name via _tagged_out_csv_name(...)

    Returns the first existing path; raises FileNotFoundError if none found.
    """
    base_path = Path(
        _default_fluence_fitting_csv_path(
            sample_name=sample_name,
            temperature_K=temperature_K,
            excitation_wl_nm=excitation_wl_nm,
            delay_fs=delay_fs,
            time_window_fs=time_window_fs,
            out_csv_name=out_csv_name,
            fluence_for_paths_mJ_cm2=float(fluence_for_paths_mJ_cm2),
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )
    )
    base_dir = base_path.parent

    names = list(_candidate_csv_names(out_csv_name))

    pm = str(phi_mode).strip()
    pr = str(phi_reduce).strip()
    if pm == "phi_avg":
        names.insert(0, _tagged_out_csv_name(out_csv_name, phi_mode=pm, phi_reduce=pr))

    # de-dup while preserving order
    seen = set()
    names_unique = []
    for n in names:
        if n not in seen:
            names_unique.append(n)
            seen.add(n)

    # try each candidate
    for name in names_unique:
        p = base_dir / str(name)
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "Could not find fluence fitting CSV. Tried:\n"
        + "\n".join([str(base_dir / n) for n in names_unique])
    )


def resolve_delay_fitting_csv_path(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    fluence_mJ_cm2: float,
    time_window_fs: int,
    out_csv_name: str = "peak_fits_delay.csv",
    phi_mode: Optional[str] = "separate_phi",
    phi_reduce: str = "sum",
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Robust resolver for DELAY fitting CSV (mirrors resolve_fluence_fitting_csv_path).

    - If phi_mode is "phi_avg": also tries phiavg-tagged variants.
    - If phi_mode is "separate_phi": tries untagged/common candidates.
    - If phi_mode is None: tries BOTH (phi_avg first, then separate_phi).

    Returns the first existing path; raises FileNotFoundError if none found.
    """
    base_path = Path(
        _default_fitting_csv_path(
            sample_name=sample_name,
            temperature_K=temperature_K,
            excitation_wl_nm=excitation_wl_nm,
            fluence_mJ_cm2=fluence_mJ_cm2,
            time_window_fs=time_window_fs,
            out_csv_name=out_csv_name,
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )
    )
    base_dir = base_path.parent

    pr = str(phi_reduce).strip()

    def _names_for_mode(pm: str):
        names = list(_candidate_csv_names(out_csv_name))
        pm_s = str(pm).strip()
        if pm_s == "phi_avg":
            names.insert(
                0,
                _tagged_out_csv_name(
                    out_csv_name,
                    phi_mode=pm_s,
                    phi_reduce=pr,
                ),
            )

        seen = set()
        out = []
        for n in names:
            if n not in seen:
                out.append(n)
                seen.add(n)
        return out

    tried = []

    pm_in = None if phi_mode is None else str(phi_mode).strip()

    if pm_in is None:
        modes = ["phi_avg", "separate_phi"]
    else:
        modes = [pm_in]

    for pm in modes:
        names_unique = _names_for_mode(pm)
        for name in names_unique:
            p = base_dir / str(name)
            tried.append(str(p))
            if p.exists():
                return str(p)

    raise FileNotFoundError(
        "Could not find delay fitting CSV. Tried:\n" + "\n".join(tried)
    )


@dataclass(frozen=True)
class FitColumns:
    peak_col: str = "peak"
    delay_fs_col: str = "delay_fs"
    series_type_col: str = "series_type"
    is_ref_col: str = "is_reference"

    azim_str_col: str = "azim_range_str"
    azim_center_col: str = "azim_center"

    success_col: str = "success"
    r2_col: str = "r2"

    pos_col: str = "hkl_pos"
    i_col: str = "hkl_i"
    fwhm_col: str = "hkl_fwhm"
    sigma_col: str = "hkl_sigma"
    area_col: str = "hkl_area"

    q0_col: str = "q_fit0"
    q1_col: str = "q_fit1"
    bg_c0_col: str = "bg_c0"
    bg_c1_col: str = "bg_c1"

    eta_col: str = "eta"


DEFAULT_COLS = FitColumns()


class DelayPeakFitter:
    def __init__(
        self,
        *,
        sample_name: str,
        temperature_K: Union[int, float],
        excitation_wl_nm: Union[int, float],
        fluence_mJ_cm2: Union[int, float],
        time_window_fs: int,
        poni_path: Optional[str] = None,
        mask_edf_path: Optional[str] = None,
        npt: int = 1000,
        normalize_xy: bool = True,
        q_norm_range: Tuple[float, float] = (2.65, 2.75),
        azim_offset_deg: float = -90.0,
        default_eta: float = 0.3,
        fit_method: str = "leastsq",
        cols: FitColumns = DEFAULT_COLS,
        paths: Optional["AnalysisPaths"] = None,
        path_root: Optional[Union[str, Path]] = None,
        analysis_subdir: Optional[Union[str, Path]] = None,
    ):
        self.sample_name = str(sample_name)
        self.temperature_K = general_utils.to_int(temperature_K)
        self.excitation_wl_nm = float(excitation_wl_nm)
        self.fluence_mJ_cm2 = float(fluence_mJ_cm2)
        self.time_window_fs = int(time_window_fs)

        self.default_eta = float(default_eta)
        self.fit_method = str(fit_method)
        self.cols = cols

        self.paths = paths
        self.path_root = None if path_root is None else Path(path_root)
        self.analysis_subdir = None if analysis_subdir is None else Path(analysis_subdir)

        if self.paths is not None:
            self.analysis_root = Path(self.paths.analysis_root)
            self._dataset_kwargs = {"paths": self.paths}

        elif self.path_root is not None and self.analysis_subdir is not None:
            self.analysis_root = self.path_root / self.analysis_subdir
            self._dataset_kwargs = {
                "path_root": self.path_root,
                "analysis_subdir": self.analysis_subdir,
            }

        else:
            raise ValueError(
                "Provide either paths=AnalysisPaths(...), or both "
                "path_root=... and analysis_subdir=...."
            )

        self.integrator = azimint_utils.AzimIntegrator(
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            npt=int(npt),
            normalize=bool(normalize_xy),
            q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
            azim_offset_deg=float(azim_offset_deg),
        )

    def _dataset_path_kwargs(self) -> Dict[str, object]:
        return dict(self._dataset_kwargs)

    def _legacy_path_kwargs(self) -> Dict[str, str]:
        if self.path_root is not None and self.analysis_subdir is not None:
            return {
                "path_root": str(self.path_root),
                "analysis_subdir": str(self.analysis_subdir),
            }

        if self.paths is not None:
            pr = getattr(self.paths, "path_root", None)
            ad = getattr(self.paths, "analysis_subdir", None)

            if pr is not None and ad is not None:
                return {
                    "path_root": str(pr),
                    "analysis_subdir": str(ad),
                }

            ar = Path(self.analysis_root)
            return {
                "path_root": str(ar.parent),
                "analysis_subdir": str(ar.name),
            }

        ar = Path(self.analysis_root)
        return {
            "path_root": str(ar.parent),
            "analysis_subdir": str(ar.name),
        }

    def analysis_dir(self) -> Path:
        ds = azimint_utils.DelayDataset(
            self.sample_name,
            self.temperature_K,
            self.excitation_wl_nm,
            self.fluence_mJ_cm2,
            self.time_window_fs,
            delay_fs=0,
            **self._dataset_path_kwargs(),
        )
        return ds.analysis_dir()

    def fitting_dir(self) -> Path:
        p = self.analysis_dir() / "fitting"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def default_csv_path(self, *, out_csv_name: str = "peak_fits_delay.csv") -> Path:
        return self.fitting_dir() / str(out_csv_name)

    def fitting_figures_dir(self) -> Path:
        p = self.analysis_dir() / "figures" / "fitting"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _delay_dataset(self, delay_fs: int) -> azimint_utils.DelayDataset:
        return azimint_utils.DelayDataset(
            self.sample_name,
            self.temperature_K,
            self.excitation_wl_nm,
            self.fluence_mJ_cm2,
            self.time_window_fs,
            int(delay_fs),
            **self._dataset_path_kwargs(),
        )

    def _dark_dataset(self, ref_value: Union[int, str, Sequence[int]]) -> azimint_utils.DarkDataset:
        dark_tag = azimint_utils.dark_tag_from_scan_spec(ref_value)
        return azimint_utils.DarkDataset(
            self.sample_name,
            self.temperature_K,
            dark_tag=dark_tag,
            **self._dataset_path_kwargs(),
        )

    def get_xy(
        self,
        dataset: Union[azimint_utils.DelayDataset, azimint_utils.DarkDataset],
        *,
        azim_window: AzimWindow = (-90.0, 90.0),
        compute_if_missing: bool = True,
        overwrite_xy: bool = False,
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        azim_str, q, I = self.integrator.get_xy_for_window(
            dataset,
            azim_window,
            compute_if_missing=bool(compute_if_missing),
            overwrite_xy=bool(overwrite_xy),
        )
        return str(azim_str), np.asarray(q, float), np.asarray(I, float)

    def get_xy_for_phi_mode(
        self,
        dataset: Union[azimint_utils.DelayDataset, azimint_utils.DarkDataset],
        *,
        phi0: float,
        phi1: float,
        phi_mode: str = "separate_phi",
        phi_reduce: str = "sum",
        compute_if_missing: bool = True,
        overwrite_xy: bool = False,
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        pm = str(phi_mode).strip()
        pr = str(phi_reduce).strip()
        if pm not in ("separate_phi", "phi_avg"):
            raise ValueError(f"phi_mode must be 'separate_phi' or 'phi_avg', got: {pm}")
        if pr not in ("sum", "mean"):
            raise ValueError(f"phi_reduce must be 'sum' or 'mean', got: {pr}")

        a = float(phi0)
        b = float(phi1)
        lo, hi = (a, b) if a <= b else (b, a)

        if pm == "separate_phi":
            return self.get_xy(
                dataset,
                azim_window=(lo, hi),
                compute_if_missing=bool(compute_if_missing),
                overwrite_xy=bool(overwrite_xy),
            )

        if abs(lo + 90.0) < 1e-9 and abs(hi - 90.0) < 1e-9:
            return self.get_xy(
                dataset,
                azim_window=(-90.0, 90.0),
                compute_if_missing=bool(compute_if_missing),
                overwrite_xy=bool(overwrite_xy),
            )

        if _is_crossing_zero(lo, hi):
            return self.get_xy(
                dataset,
                azim_window=(lo, hi),
                compute_if_missing=bool(compute_if_missing),
                overwrite_xy=bool(overwrite_xy),
            )

        lo_abs = min(abs(lo), abs(hi))
        hi_abs = max(abs(lo), abs(hi))
        win_pos = (float(lo_abs), float(hi_abs))
        win_neg = (-float(hi_abs), -float(lo_abs))

        _, q_pos, I_pos = self.get_xy(
            dataset,
            azim_window=win_pos,
            compute_if_missing=bool(compute_if_missing),
            overwrite_xy=bool(overwrite_xy),
        )
        _, q_neg, I_neg = self.get_xy(
            dataset,
            azim_window=win_neg,
            compute_if_missing=bool(compute_if_missing),
            overwrite_xy=bool(overwrite_xy),
        )

        if (len(q_pos) != len(q_neg)) or (np.max(np.abs(q_pos - q_neg)) > 1e-12):
            raise ValueError("Cannot phi-avg merge: q-grids differ between symmetric windows.")

        I_out = (I_pos + I_neg) if pr == "sum" else 0.5 * (I_pos + I_neg)
        azim_str_out = general_utils.azim_range_str(win_pos)
        return str(azim_str_out), np.asarray(q_pos, float), np.asarray(I_out, float)

    def _mode_folder_name(self, *, phi_mode: str, phi_reduce: str) -> str:
        pm = str(phi_mode).strip()
        pr = str(phi_reduce).strip()
        return f"phi_avg_{pr}" if pm == "phi_avg" else "separate_phi"

    def _phi_bin_folder_name(self, *, phi_mode: str, phi_label: str, azim_str: str) -> str:
        return f"phi_{_sanitize_path_token(phi_label)}" if str(phi_mode).strip() == "phi_avg" else f"az_{_sanitize_path_token(azim_str)}"

    def _default_overlay_save_dir(
        self,
        *,
        phi_mode: str,
        phi_reduce: str,
        phi_label: str,
        azim_str: str,
        peak_name: str,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        base = Path(str(save_dir)) if save_dir is not None else self.fitting_figures_dir()
        mode_dir = base / self._mode_folder_name(phi_mode=phi_mode, phi_reduce=phi_reduce)
        phi_dir = mode_dir / self._phi_bin_folder_name(phi_mode=phi_mode, phi_label=phi_label, azim_str=azim_str)
        peak_dir = phi_dir / f"peak_{_sanitize_path_token(peak_name)}"
        peak_dir.mkdir(parents=True, exist_ok=True)
        return peak_dir

    def overlay_save_dir_and_name(
        self,
        *,
        phi_mode: str,
        phi_reduce: str,
        phi_label: str,
        azim_str: str,
        peak_name: str,
        is_reference: bool,
        delay_fs_val: Optional[int],
        fit_figures_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, str]:
        out_dir = self._default_overlay_save_dir(
            phi_mode=str(phi_mode),
            phi_reduce=str(phi_reduce),
            phi_label=str(phi_label),
            azim_str=str(azim_str),
            peak_name=str(peak_name),
            save_dir=fit_figures_dir,
        )
        save_name = (
            f"{self.sample_name}_"
            f"{int(self.temperature_K)}K_"
            f"{int(self.excitation_wl_nm)}nm_"
            f"{str(float(self.fluence_mJ_cm2)).replace('.','p')}mJ_"
            f"{int(self.time_window_fs)}fs_"
            f"{'ref' if bool(is_reference) else f'{int(delay_fs_val)}fs'}_"
            f"{_sanitize_path_token(str(azim_str))}"
        )
        return out_dir, str(save_name)

    def _plot_and_maybe_save_overlay(
        self,
        *,
        payload: Dict[str, Any],
        title: str,
        save_dir: Union[str, Path],
        save_name: str,
        show_fig: bool,
        save_fig: bool,
        save_fmt: str = "png",
        save_dpi: int = 300,
        save_overwrite: bool = True,
        close_after: bool = True,
    ):

        style = getattr(plot_utils, "DEFAULT_STYLE", None)
        p = plot_utils.PeakFitOverlayPlotter(style=style)
        return p.plot_from_payload(
            payload,
            title=str(title),
            show=bool(show_fig),
            save=bool(save_fig),
            save_dir=Path(save_dir),
            save_name=str(save_name),
            save_format=str(save_fmt),
            save_dpi=int(save_dpi),
            save_overwrite=bool(save_overwrite),
            close_after=bool(close_after),
        )

    @staticmethod
    def _sigma_guess_from_spec(peak_spec: PeakSpecDict) -> float:
        if peak_spec.get("sigma_guess", None) is not None:
            return float(peak_spec["sigma_guess"])
        if peak_spec.get("fwhm_guess", None) is not None:
            return float(peak_spec["fwhm_guess"]) / 2.354820045
        return 0.01

    def fit_one_peak(
        self,
        *,
        q: np.ndarray,
        I: np.ndarray,
        peak_name: str,
        peak_spec: PeakSpecDict,
        return_payload: bool = False,
        fit_oversample: int = 10,
    ) -> Dict[str, object]:
        q = np.asarray(q, float)
        I = np.asarray(I, float)

        q_fit_range = peak_spec.get("q_fit_range", None)
        if q_fit_range is None:
            raise ValueError(f"peak_spec for '{peak_name}' must provide q_fit_range=(q0,q1).")

        q0, q1 = float(q_fit_range[0]), float(q_fit_range[1])
        lo, hi = (q0, q1) if q0 <= q1 else (q1, q0)
        m = (q >= lo) & (q <= hi)

        row: Dict[str, object] = {
            self.cols.peak_col: str(peak_name),
            self.cols.q0_col: float(lo),
            self.cols.q1_col: float(hi),
            self.cols.success_col: False,
            self.cols.r2_col: np.nan,
            self.cols.pos_col: np.nan,
            self.cols.i_col: np.nan,
            self.cols.fwhm_col: np.nan,
            self.cols.sigma_col: np.nan,
            self.cols.area_col: np.nan,
            self.cols.bg_c0_col: np.nan,
            self.cols.bg_c1_col: np.nan,
            self.cols.eta_col: np.nan,
        }

        if not np.any(m):
            if return_payload:
                row["_fit_payload"] = {"success": False, "q": q, "q0": lo, "q1": hi}
            return row

        qfit = q[m]
        Ifit = I[m]
        if qfit.size < 5:
            if return_payload:
                row["_fit_payload"] = {"success": False, "q": q, "q0": lo, "q1": hi, "qfit": qfit, "I": I, "Ifit": Ifit}
            return row

        bg_slope_auto = (Ifit[-1] - Ifit[0]) / (qfit[-1] - qfit[0] + 1e-12)
        bg_c0_guess = float(peak_spec.get("bg_c0_guess", np.median(Ifit)))
        bg_c1_guess = float(peak_spec.get("bg_c1_guess", bg_slope_auto))

        center_guess = peak_spec.get("hkl_pos_guess", None)
        if center_guess is None:
            center_guess = float(qfit[int(np.argmax(Ifit))])
        else:
            center_guess = float(center_guess)
            center_guess = min(max(center_guess, lo), hi)

        sigma_guess = float(self._sigma_guess_from_spec(peak_spec))

        amp_guess = peak_spec.get("amplitude_guess", None)
        if amp_guess is None:
            baseline = float(np.median(Ifit))
            amp_guess = float(np.trapezoid(np.maximum(Ifit - baseline, 0.0), qfit))
        amp_guess = max(float(amp_guess), 1e-12)

        eta = float(peak_spec.get("eta", self.default_eta))

        model = _make_single_peak_model()
        params = model.make_params()
        params["bg_c0"].set(value=bg_c0_guess)
        params["bg_c1"].set(value=bg_c1_guess)
        params["pv_center"].set(value=center_guess, min=lo, max=hi)
        params["pv_sigma"].set(value=sigma_guess, min=1e-6, max=0.2)
        params["pv_amplitude"].set(value=amp_guess, min=0.0)
        params["pv_fraction"].set(value=eta, vary=False, min=0.0, max=1.0)

        try:
            result = model.fit(Ifit, params, x=qfit, method=str(self.fit_method))
        except Exception:
            result = None

        if (result is None) or (not getattr(result, "success", False)):
            row[self.cols.eta_col] = float(eta)
            if return_payload:
                row["_fit_payload"] = {"success": False, "q": q, "q0": lo, "q1": hi, "qfit": qfit, "I": I, "Ifit": Ifit, "eta": eta}
            return row

        yfit, _bg, _pv = _eval_components(model, result.params, qfit)
        r2 = _compute_r2(Ifit, yfit)

        pv_center = float(result.params["pv_center"].value)
        pv_sigma = float(result.params["pv_sigma"].value)
        pv_amp = float(result.params["pv_amplitude"].value)
        fwhm = _pv_fwhm_from_result(result)

        _, _, pv_c = _eval_components(model, result.params, np.array([pv_center], float))
        peak_height = float(pv_c[0])

        row[self.cols.success_col] = True
        row[self.cols.r2_col] = float(r2)
        row[self.cols.pos_col] = float(pv_center)
        row[self.cols.i_col] = float(peak_height)
        row[self.cols.fwhm_col] = float(fwhm)
        row[self.cols.sigma_col] = float(pv_sigma)
        row[self.cols.area_col] = float(pv_amp)
        row[self.cols.bg_c0_col] = float(result.params["bg_c0"].value)
        row[self.cols.bg_c1_col] = float(result.params["bg_c1"].value)
        row[self.cols.eta_col] = float(eta)

        if return_payload:
            osamp = max(int(fit_oversample), 1)
            n_dense = max(int(len(qfit) * osamp), int(len(qfit) + 1))
            q_dense = np.linspace(float(qfit[0]), float(qfit[-1]), n_dense)
            y_dense, bg_dense, pv_dense = _eval_components(model, result.params, q_dense)

            row["_fit_payload"] = dict(
                success=True,
                q=q,
                I=I,
                q0=lo,
                q1=hi,
                qfit=np.asarray(qfit, float),
                Ifit=np.asarray(Ifit, float),
                q_dense=np.asarray(q_dense, float),
                y_dense=np.asarray(y_dense, float),
                bg_dense=np.asarray(bg_dense, float),
                pv_dense=np.asarray(pv_dense, float),
                r2=float(r2),
                eta=float(eta),
                pv_center=float(pv_center),
                pv_sigma=float(pv_sigma),
                pv_height=float(peak_height),
                pv_fwhm=float(fwhm),
            )

        return row

    def _make_model_and_params_from_values(
        self,
        *,
        bg_c0: float,
        bg_c1: float,
        pv_center: float,
        pv_sigma: float,
        pv_amplitude: float,
        eta: float,
        q0: float,
        q1: float,
    ):
        model = _make_single_peak_model()
        params = model.make_params()

        params["bg_c0"].set(value=float(bg_c0))
        params["bg_c1"].set(value=float(bg_c1))

        lo = min(float(q0), float(q1))
        hi = max(float(q0), float(q1))

        c = float(pv_center)
        if np.isfinite(c):
            c = min(max(c, lo), hi)
        else:
            c = 0.5 * (lo + hi)
        params["pv_center"].set(value=float(c), min=lo, max=hi)

        s = float(pv_sigma)
        if (not np.isfinite(s)) or (s <= 0):
            s = 0.01
        params["pv_sigma"].set(value=float(s), min=1e-6, max=0.2)

        a = float(pv_amplitude)
        if (not np.isfinite(a)) or (a < 0):
            a = 0.0
        params["pv_amplitude"].set(value=float(a), min=0.0)

        e = float(eta)
        if not np.isfinite(e):
            e = float(self.default_eta)
        e = min(max(e, 0.0), 1.0)
        params["pv_fraction"].set(value=float(e), vary=False, min=0.0, max=1.0)

        return model, params

    def build_overlay_payload_from_params(
        self,
        *,
        q: np.ndarray,
        I: np.ndarray,
        q0: float,
        q1: float,
        bg_c0: float,
        bg_c1: float,
        pv_center: float,
        pv_sigma: float,
        pv_amplitude: float,
        eta: float,
        fit_oversample: int = 10,
        r2_hint: Optional[float] = None,
        pv_height_hint: Optional[float] = None,
        pv_fwhm_hint: Optional[float] = None,
    ) -> Dict[str, Any]:
        q = np.asarray(q, float)
        I = np.asarray(I, float)

        lo = min(float(q0), float(q1))
        hi = max(float(q0), float(q1))

        m = (q >= lo) & (q <= hi)
        if not np.any(m):
            return dict(success=False, q=q, I=I, q0=float(q0), q1=float(q1))

        qfit = q[m]
        Ifit = I[m]
        if qfit.size < 5:
            return dict(success=False, q=q, I=I, q0=float(q0), q1=float(q1), qfit=qfit, Ifit=Ifit)

        model, params = self._make_model_and_params_from_values(
            bg_c0=float(bg_c0),
            bg_c1=float(bg_c1),
            pv_center=float(pv_center),
            pv_sigma=float(pv_sigma),
            pv_amplitude=float(pv_amplitude),
            eta=float(eta),
            q0=float(q0),
            q1=float(q1),
        )

        yfit, _bg, _pv = _eval_components(model, params, qfit)
        r2_calc = _compute_r2(Ifit, yfit)

        c = float(params["pv_center"].value)
        _, _, pv_c = _eval_components(model, params, np.array([c], float))
        pv_height_calc = float(pv_c[0])

        osamp = max(int(fit_oversample), 1)
        n_dense = max(int(len(qfit) * osamp), int(len(qfit) + 1))
        q_dense = np.linspace(float(qfit[0]), float(qfit[-1]), n_dense)
        y_dense, bg_dense, pv_dense = _eval_components(model, params, q_dense)

        out = dict(
            success=True,
            q=q,
            I=I,
            q0=float(q0),
            q1=float(q1),
            qfit=np.asarray(qfit, float),
            Ifit=np.asarray(Ifit, float),
            q_dense=np.asarray(q_dense, float),
            y_dense=np.asarray(y_dense, float),
            bg_dense=np.asarray(bg_dense, float),
            pv_dense=np.asarray(pv_dense, float),
            eta=float(params["pv_fraction"].value),
            pv_center=float(c),
            pv_sigma=float(params["pv_sigma"].value),
            r2=float(r2_calc),
            pv_height=float(pv_height_calc),
            pv_fwhm=float(2.354820045 * float(params["pv_sigma"].value)),
        )

        if r2_hint is not None and np.isfinite(float(r2_hint)):
            out["r2"] = float(r2_hint)
        if pv_height_hint is not None and np.isfinite(float(pv_height_hint)):
            out["pv_height"] = float(pv_height_hint)
        if pv_fwhm_hint is not None and np.isfinite(float(pv_fwhm_hint)):
            out["pv_fwhm"] = float(pv_fwhm_hint)

        return out

    def fit_delay_series(
        self,
        *,
        delays_fs: Union[int, Sequence[int], str],
        peak_specs: Dict[str, PeakSpecDict],
        azim_windows: Optional[Sequence[AzimWindow]] = None,
        compute_if_missing: bool = True,
        overwrite_xy: bool = False,
        ref_type: Optional[str] = None,
        ref_value: Optional[Union[int, str, Sequence[int]]] = None,
        include_reference_in_output: bool = True,
        phi_mode: str = "separate_phi",
        phi_reduce: str = "sum",
        show_fit_figures: bool = False,
        save_fit_figures: bool = False,
        fit_figures_dir: Optional[Union[str, Path]] = None,
        fit_figures_format: str = "png",
        fit_figures_dpi: int = 300,
        fit_figures_overwrite: bool = True,
        close_figures_after_save: bool = True,
        plot_only_success: bool = True,
        fit_oversample: int = 10,
    ) -> pd.DataFrame:
        azim_windows = _normalize_azim_windows(azim_windows)
        azim_windows = [(float(a), float(b)) for (a, b) in list(azim_windows)]

        pm = str(phi_mode).strip()
        pr = str(phi_reduce).strip()
        if pm not in ("separate_phi", "phi_avg"):
            raise ValueError(f"phi_mode must be 'separate_phi' or 'phi_avg', got: {phi_mode}")
        if pr not in ("sum", "mean"):
            raise ValueError(f"phi_reduce must be 'sum' or 'mean', got: {pr}")

        delays_list = azimint_utils.normalize_delays_fs(
            delays_fs,
            sample_name=self.sample_name,
            temperature_K=self.temperature_K,
            excitation_wl_nm=self.excitation_wl_nm,
            fluence_mJ_cm2=self.fluence_mJ_cm2,
            time_window_fs=self.time_window_fs,
            from_2D_imgs=False,
            **self._legacy_path_kwargs(),
        )

        rows: List[Dict[str, object]] = []

        def _patterns_for_dataset(dataset) -> List[Dict[str, object]]:
            entries: List[Dict[str, object]] = []
            for azw in azim_windows:
                azim_str, q, I = self.get_xy(
                    dataset,
                    azim_window=azw,
                    compute_if_missing=compute_if_missing,
                    overwrite_xy=overwrite_xy,
                )

                entries.append(
                    dict(
                        phi0=float(azw[0]),
                        phi1=float(azw[1]),
                        x=np.asarray(q, float),
                        y=np.asarray(I, float),
                        azim_str=str(azim_str),
                        source=str(azim_str),
                    )
                )

            return entries
        
        def _append_rows_for_dataset(*, dataset, series_type: str, delay_fs_val: Optional[int], is_ref: bool):
            pattern_entries = _patterns_for_dataset(dataset)
            patterns = merge_phi_symmetric_patterns(pattern_entries, reduce=pr) if pm == "phi_avg" else pattern_entries

            for e in patterns:
                phi0 = float(e["phi0"])
                phi1 = float(e["phi1"])
                q = np.asarray(e["x"], float)
                I = np.asarray(e["y"], float)

                azim_str = general_utils.azim_range_str((phi0, phi1)) if pm == "phi_avg" else str(e.get("azim_str", ""))
                az_center = general_utils.azim_center((phi0, phi1))

                phi_center_abs = float(_phi_center_abs_for_sym(phi0, phi1))
                phi_label = str(_format_phi_label(phi0, phi1, pm))
                phi_halfwidth = 0.5 * abs(phi1 - phi0)

                for peak_name, peak_spec in peak_specs.items():
                    want_payload = bool(show_fit_figures or save_fit_figures)

                    r = self.fit_one_peak(
                        q=q,
                        I=I,
                        peak_name=str(peak_name),
                        peak_spec=dict(peak_spec),
                        return_payload=want_payload,
                        fit_oversample=int(fit_oversample),
                    )
                    payload = r.pop("_fit_payload", None) if "_fit_payload" in r else None

                    r[self.cols.series_type_col] = str(series_type)
                    r[self.cols.is_ref_col] = bool(is_ref)
                    r[self.cols.delay_fs_col] = (np.nan if delay_fs_val is None else int(delay_fs_val))
                    r[self.cols.azim_str_col] = str(azim_str)
                    r[self.cols.azim_center_col] = float(az_center)

                    r["phi_mode"] = str(pm)
                    r["phi_reduce"] = (str(pr) if pm == "phi_avg" else "")
                    r["phi0"] = float(phi0)
                    r["phi1"] = float(phi1)
                    r["phi_center_abs"] = float(phi_center_abs)
                    r["phi_label"] = str(phi_label)
                    r["phi_halfwidth_deg"] = float(phi_halfwidth)

                    r["sample_name"] = self.sample_name
                    r["temperature_K"] = int(self.temperature_K)
                    r["excitation_wl_nm"] = float(self.excitation_wl_nm)
                    r["fluence_mJ_cm2"] = float(self.fluence_mJ_cm2)
                    r["time_window_fs"] = int(self.time_window_fs)

                    if (show_fit_figures or save_fit_figures) and payload is not None:
                        ok = bool(payload.get("success", False))
                        if (not bool(plot_only_success)) or ok:
                            delay_part = "dark reference" if (series_type == "dark") else f"delay={delay_fs_val} fs"
                            az_part = (
                                f"|$\\Phi$|={phi_label}° (mode={pm})"
                                if pm == "phi_avg"
                                else f"$\\Phi$=({phi0:g},{phi1:g})°"
                            )

                            title = (
                                f"{self.sample_name}, {int(self.temperature_K)} K\n"
                                f"ex. wl={float(self.excitation_wl_nm):g} nm, flu={float(self.fluence_mJ_cm2):g} mJ/cm$^2$\n"
                                f"tw={int(self.time_window_fs)} fs, {delay_part}\n"
                                f"{az_part}\n"
                                f"hkl=({str(peak_name)}), q=({float(payload.get('q0', np.nan)):.3f},{float(payload.get('q1', np.nan)):.3f}) Å$^{{-1}}$."
                            )

                            out_dir, save_name = self.overlay_save_dir_and_name(
                                phi_mode=str(pm),
                                phi_reduce=str(pr),
                                phi_label=str(phi_label),
                                azim_str=str(azim_str),
                                peak_name=str(peak_name),
                                is_reference=bool(is_ref),
                                delay_fs_val=delay_fs_val,
                                fit_figures_dir=fit_figures_dir,
                            )

                            self._plot_and_maybe_save_overlay(
                                payload=payload,
                                title=title,
                                save_dir=out_dir,
                                save_name=save_name,
                                show_fig=bool(show_fit_figures),
                                save_fig=bool(save_fit_figures),
                                save_fmt=str(fit_figures_format),
                                save_dpi=int(fit_figures_dpi),
                                save_overwrite=bool(fit_figures_overwrite),
                                close_after=bool(close_figures_after_save),
                            )

                    rows.append(r)

        tasks: List[Tuple[Union[azimint_utils.DelayDataset, azimint_utils.DarkDataset], str, Optional[int], bool]] = []

        if include_reference_in_output and (ref_type is not None):
            rt = str(ref_type).strip().lower()
            if rt not in ("delay", "dark"):
                raise ValueError("ref_type must be None, 'delay', or 'dark'.")
            if ref_value is None:
                raise ValueError("If ref_type is provided, ref_value must be provided too.")

            if rt == "delay":
                ds_ref = self._delay_dataset(int(ref_value))  # type: ignore[arg-type]
                tasks.append((ds_ref, "delay", int(ref_value), True))
            else:
                ds_ref = self._dark_dataset(ref_value)
                tasks.append((ds_ref, "dark", None, True))

        for d in delays_list:
            ds = self._delay_dataset(int(d))
            tasks.append((ds, "delay", int(d), False))

        iterator = tasks
        if _tqdm is not None:
            iterator = _tqdm(tasks, desc="run_delay_peak_fitting", unit="scan")

        for dataset, series_type, delay_fs_val, is_ref in iterator:  # type: ignore[assignment]
            if _tqdm is not None and hasattr(iterator, "set_postfix"):
                try:
                    iterator.set_postfix(  # type: ignore[attr-defined]
                        {
                            "series": str(series_type),
                            "delay_fs": ("ref" if delay_fs_val is None else int(delay_fs_val)),
                        }
                    )
                except Exception:
                    pass

            _append_rows_for_dataset(
                dataset=dataset,
                series_type=str(series_type),
                delay_fs_val=delay_fs_val,
                is_ref=bool(is_ref),
            )

        df = pd.DataFrame(rows)
        if self.cols.delay_fs_col in df.columns:
            df = df.sort_values(
                [self.cols.peak_col, self.cols.azim_center_col, self.cols.delay_fs_col],
                na_position="last",
            )
        return df

    def save_csv(self, df: pd.DataFrame, *, path: Union[str, Path]) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(p), index=False)
        return str(p)


class FluencePeakFitter(DelayPeakFitter):
    """
    Peak fitting engine for FLUENCE scans at fixed delay_fs + time_window_fs.

    Design:
      - Recycles DelayPeakFitter fitting logic (fit_one_peak, overlay payload build, phi-merge logic).
      - Only replaces:
          * analysis_dir root resolution
          * dataset factory for scan points (FluenceDataset)
          * scan loop orchestration (fit_fluence_series)
          * overlay save-name construction (includes fluence token)
    """

    def __init__(
        self,
        *,
        sample_name: str,
        temperature_K: Union[int, float],
        excitation_wl_nm: Union[int, float],
        delay_fs: int,
        time_window_fs: int,
        poni_path: Optional[str] = None,
        mask_edf_path: Optional[str] = None,
        npt: int = 1000,
        normalize_xy: bool = True,
        q_norm_range: Tuple[float, float] = (2.65, 2.75),
        azim_offset_deg: float = -90.0,
        default_eta: float = 0.3,
        fit_method: str = "leastsq",
        cols: FitColumns = DEFAULT_COLS,
        fluence_for_paths_mJ_cm2: float = 1.0,
        paths: Optional["AnalysisPaths"] = None,
        path_root: Optional[Union[str, Path]] = None,
        analysis_subdir: Optional[Union[str, Path]] = None,
    ):
        self.delay_fs_fixed = int(delay_fs)
        self.fluence_for_paths_mJ_cm2 = float(fluence_for_paths_mJ_cm2)

        super().__init__(
            sample_name=str(sample_name),
            temperature_K=temperature_K,
            excitation_wl_nm=excitation_wl_nm,
            fluence_mJ_cm2=float(self.fluence_for_paths_mJ_cm2),
            time_window_fs=int(time_window_fs),
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            npt=int(npt),
            normalize_xy=bool(normalize_xy),
            q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
            azim_offset_deg=float(azim_offset_deg),
            default_eta=float(default_eta),
            fit_method=str(fit_method),
            cols=cols,
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
        )

    def analysis_dir(self) -> Path:
        ds = azimint_utils.FluenceDataset(
            self.sample_name,
            self.temperature_K,
            self.excitation_wl_nm,
            fluence_mJ_cm2=float(self.fluence_for_paths_mJ_cm2),
            time_window_fs=int(self.time_window_fs),
            delay_fs=int(self.delay_fs_fixed),
            **self._dataset_path_kwargs(),
        )
        return ds.analysis_dir()

    def _fluence_dataset(self, fluence_mJ_cm2: Union[int, float]) -> azimint_utils.FluenceDataset:
        return azimint_utils.FluenceDataset(
            self.sample_name,
            self.temperature_K,
            self.excitation_wl_nm,
            float(fluence_mJ_cm2),
            self.time_window_fs,
            int(self.delay_fs_fixed),
            **self._dataset_path_kwargs(),
        )

    def overlay_save_dir_and_name_fluence(
        self,
        *,
        phi_mode: str,
        phi_reduce: str,
        phi_label: str,
        azim_str: str,
        peak_name: str,
        is_reference: bool,
        series_type: str,
        fluence_mJ_cm2_val: Optional[float],
        fit_figures_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, str]:
        out_dir = self._default_overlay_save_dir(
            phi_mode=str(phi_mode),
            phi_reduce=str(phi_reduce),
            phi_label=str(phi_label),
            azim_str=str(azim_str),
            peak_name=str(peak_name),
            save_dir=fit_figures_dir,
        )

        wl_tok = int(float(self.excitation_wl_nm))
        tw_tok = int(self.time_window_fs)
        dly_tok = int(self.delay_fs_fixed)

        if bool(is_reference) and str(series_type).strip().lower() == "dark":
            flu_tok = "darkref"
        else:
            if fluence_mJ_cm2_val is None or (not np.isfinite(float(fluence_mJ_cm2_val))):
                flu_tok = "x"
            else:
                flu_tok = str(float(fluence_mJ_cm2_val)).replace(".", "p")

        save_name = (
            f"{self.sample_name}_"
            f"{int(self.temperature_K)}K_"
            f"{int(wl_tok)}nm_"
            f"delay{int(dly_tok)}fs_"
            f"{flu_tok}mJ_"
            f"{int(tw_tok)}fs_"
            f"{'ref' if bool(is_reference) else 'pt'}_"
            f"{_sanitize_path_token(str(azim_str))}"
        )
        return out_dir, str(save_name)

    def fit_fluence_series(
        self,
        *,
        fluences_mJ_cm2: Union[float, int, Sequence[Union[float, int]], str],
        peak_specs: Dict[str, PeakSpecDict],
        azim_windows: Optional[Sequence[AzimWindow]] = None,
        compute_if_missing: bool = True,
        overwrite_xy: bool = False,
        ref_type: Optional[str] = None,
        ref_value: Optional[Union[float, int, str, Sequence[int]]] = None,
        include_reference_in_output: bool = True,
        phi_mode: str = "separate_phi",
        phi_reduce: str = "sum",
        show_fit_figures: bool = False,
        save_fit_figures: bool = False,
        fit_figures_dir: Optional[Union[str, Path]] = None,
        fit_figures_format: str = "png",
        fit_figures_dpi: int = 300,
        fit_figures_overwrite: bool = True,
        close_figures_after_save: bool = True,
        plot_only_success: bool = True,
        fit_oversample: int = 10,
    ) -> pd.DataFrame:
        azim_windows = _normalize_azim_windows(azim_windows)
        azim_windows = [(float(a), float(b)) for (a, b) in list(azim_windows)]

        pm = str(phi_mode).strip()
        pr = str(phi_reduce).strip()
        if pm not in ("separate_phi", "phi_avg"):
            raise ValueError(f"phi_mode must be 'separate_phi' or 'phi_avg', got: {phi_mode}")
        if pr not in ("sum", "mean"):
            raise ValueError(f"phi_reduce must be 'sum' or 'mean', got: {pr}")

        fl_list = azimint_utils.normalize_fluences_mJ_cm2(
            fluences_mJ_cm2,
            sample_name=self.sample_name,
            temperature_K=self.temperature_K,
            excitation_wl_nm=self.excitation_wl_nm,
            delay_fs=int(self.delay_fs_fixed),
            time_window_fs=int(self.time_window_fs),
            from_2D_imgs=False,
            **self._legacy_path_kwargs(),
        )

        rows: List[Dict[str, object]] = []

        def _patterns_for_dataset(dataset) -> List[Dict[str, object]]:
            entries: List[Dict[str, object]] = []
            for azw in azim_windows:
                azim_str, q, I = self.get_xy(
                    dataset,
                    azim_window=azw,
                    compute_if_missing=bool(compute_if_missing),
                    overwrite_xy=bool(overwrite_xy),
                )
                entries.append(
                    dict(
                        phi0=float(azw[0]),
                        phi1=float(azw[1]),
                        x=np.asarray(q, float),
                        y=np.asarray(I, float),
                        azim_str=str(azim_str),
                        source=str(azim_str),
                    )
                )
            return entries

        def _append_rows_for_dataset(
            *,
            dataset,
            series_type: str,
            fluence_val: Optional[float],
            is_ref: bool,
        ):
            pattern_entries = _patterns_for_dataset(dataset)
            patterns = merge_phi_symmetric_patterns(pattern_entries, reduce=pr) if pm == "phi_avg" else pattern_entries

            for e in patterns:
                phi0 = float(e["phi0"])
                phi1 = float(e["phi1"])
                q = np.asarray(e["x"], float)
                I = np.asarray(e["y"], float)

                azim_str = general_utils.azim_range_str((phi0, phi1)) if pm == "phi_avg" else str(e.get("azim_str", ""))
                az_center = general_utils.azim_center((phi0, phi1))

                phi_center_abs = float(_phi_center_abs_for_sym(phi0, phi1))
                phi_label = str(_format_phi_label(phi0, phi1, pm))
                phi_halfwidth = 0.5 * abs(phi1 - phi0)

                for peak_name, peak_spec in peak_specs.items():
                    want_payload = bool(show_fit_figures or save_fit_figures)

                    r = self.fit_one_peak(
                        q=q,
                        I=I,
                        peak_name=str(peak_name),
                        peak_spec=dict(peak_spec),
                        return_payload=want_payload,
                        fit_oversample=int(fit_oversample),
                    )
                    payload = r.pop("_fit_payload", None) if "_fit_payload" in r else None

                    r[self.cols.series_type_col] = str(series_type)
                    r[self.cols.is_ref_col] = bool(is_ref)
                    r[self.cols.delay_fs_col] = int(self.delay_fs_fixed)

                    r["fluence_mJ_cm2"] = (np.nan if fluence_val is None else float(fluence_val))
                    r["delay_fs_fixed"] = int(self.delay_fs_fixed)
                    r["scan_axis"] = "fluence"

                    r[self.cols.azim_str_col] = str(azim_str)
                    r[self.cols.azim_center_col] = float(az_center)

                    r["phi_mode"] = str(pm)
                    r["phi_reduce"] = (str(pr) if pm == "phi_avg" else "")
                    r["phi0"] = float(phi0)
                    r["phi1"] = float(phi1)
                    r["phi_center_abs"] = float(phi_center_abs)
                    r["phi_label"] = str(phi_label)
                    r["phi_halfwidth_deg"] = float(phi_halfwidth)

                    r["sample_name"] = self.sample_name
                    r["temperature_K"] = int(self.temperature_K)
                    r["excitation_wl_nm"] = float(self.excitation_wl_nm)
                    r["time_window_fs"] = int(self.time_window_fs)

                    if (show_fit_figures or save_fit_figures) and payload is not None:
                        ok = bool(payload.get("success", False))
                        if (not bool(plot_only_success)) or ok:
                            delay_part = f"delay={int(self.delay_fs_fixed)} fs"
                            flu_part = "dark reference" if (str(series_type).lower() == "dark") else f"fluence={float(fluence_val):g} mJ/cm$^2$"

                            az_part = (
                                f"|$\\Phi$|={phi_label}° (mode={pm})"
                                if pm == "phi_avg"
                                else f"$\\Phi$=({phi0:g},{phi1:g})°"
                            )

                            title = (
                                f"{self.sample_name}, {int(self.temperature_K)} K\n"
                                f"ex. wl={float(self.excitation_wl_nm):g} nm, {delay_part}\n"
                                f"tw={int(self.time_window_fs)} fs, {flu_part}\n"
                                f"{az_part}\n"
                                f"hkl=({str(peak_name)}), q=({float(payload.get('q0', np.nan)):.3f},{float(payload.get('q1', np.nan)):.3f}) Å$^{{-1}}$."
                            )

                            out_dir, save_name = self.overlay_save_dir_and_name_fluence(
                                phi_mode=str(pm),
                                phi_reduce=str(pr),
                                phi_label=str(phi_label),
                                azim_str=str(azim_str),
                                peak_name=str(peak_name),
                                is_reference=bool(is_ref),
                                series_type=str(series_type),
                                fluence_mJ_cm2_val=(None if fluence_val is None else float(fluence_val)),
                                fit_figures_dir=fit_figures_dir,
                            )

                            self._plot_and_maybe_save_overlay(
                                payload=payload,
                                title=title,
                                save_dir=out_dir,
                                save_name=save_name,
                                show_fig=bool(show_fit_figures),
                                save_fig=bool(save_fit_figures),
                                save_fmt=str(fit_figures_format),
                                save_dpi=int(fit_figures_dpi),
                                save_overwrite=bool(fit_figures_overwrite),
                                close_after=bool(close_figures_after_save),
                            )

                    rows.append(r)

        tasks: List[Tuple[Union[azimint_utils.FluenceDataset, azimint_utils.DarkDataset], str, Optional[float], bool]] = []

        if include_reference_in_output and (ref_type is not None):
            rt = str(ref_type).strip().lower()
            if rt not in ("fluence", "dark"):
                raise ValueError("ref_type must be None, 'fluence', or 'dark'.")
            if ref_value is None:
                raise ValueError("If ref_type is provided, ref_value must be provided too.")

            if rt == "fluence":
                rf = float(ref_value)  # type: ignore[arg-type]
                ds_ref = self._fluence_dataset(rf)
                tasks.append((ds_ref, "fluence", float(rf), True))
            else:
                ds_ref = self._dark_dataset(ref_value)  # type: ignore[arg-type]
                tasks.append((ds_ref, "dark", None, True))

        for f in fl_list:
            ds = self._fluence_dataset(float(f))
            tasks.append((ds, "fluence", float(f), False))

        iterator = tasks
        if _tqdm is not None:
            iterator = _tqdm(tasks, desc="run_fluence_peak_fitting", unit="scan")

        for dataset, series_type, fluence_val, is_ref in iterator:  # type: ignore[assignment]
            if _tqdm is not None and hasattr(iterator, "set_postfix"):
                try:
                    iterator.set_postfix(  # type: ignore[attr-defined]
                        {
                            "series": str(series_type),
                            "fluence": ("ref" if fluence_val is None else float(fluence_val)),
                        }
                    )
                except Exception:
                    pass

            _append_rows_for_dataset(
                dataset=dataset,
                series_type=str(series_type),
                fluence_val=fluence_val,
                is_ref=bool(is_ref),
            )

        df = pd.DataFrame(rows)

        if (self.cols.peak_col in df.columns) and (self.cols.azim_center_col in df.columns) and ("fluence_mJ_cm2" in df.columns):
            flu = pd.to_numeric(df["fluence_mJ_cm2"], errors="coerce")
            df = df.assign(_flu_sort=flu)
            df = df.sort_values(
                [self.cols.peak_col, self.cols.azim_center_col, "_flu_sort"],
                na_position="last",
            ).drop(columns=["_flu_sort"], errors="ignore")

        return df


