# general_utils.py
"""
General small utilities shared across scripts.

Design goals:
- Small + generic helpers only.
- No beamline-specific paths, no calibration-specific logic.
- Stable APIs: keep function names/signatures used across the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, TypeVar, Union
import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import pandas as pd


T = TypeVar("T")


# ----------------------------
# Small helpers
# ----------------------------

def as_list(x):
    """None -> [], list/tuple -> list(x), otherwise -> [x]."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def to_int(v) -> int:
    """Robust conversion to int (supports float/np scalar)."""
    return int(np.round(float(v)))


def decode_if_bytes(x: Any) -> Any:
    """Decode bytes/np.bytes_ to utf-8 string, otherwise passthrough."""
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    return x


def is_sequence_of_ints(x: Any) -> bool:
    """True if x is a non-empty iterable of ints/np.integer (but not str/bytes)."""
    if isinstance(x, (str, bytes)):
        return False
    try:
        xs = list(x)  # type: ignore
    except Exception:
        return False
    if len(xs) == 0:
        return False
    return all(isinstance(v, (int, np.integer)) for v in xs)


def first_scan_id(scan_spec: Any) -> Optional[int]:
    """
    If scan_spec is:
      - int -> that int
      - str -> None
      - sequence-of-ints -> min element
      - otherwise -> None
    """
    if isinstance(scan_spec, int):
        return int(scan_spec)
    if isinstance(scan_spec, str):
        return None
    if is_sequence_of_ints(scan_spec):
        xs = sorted(int(v) for v in list(scan_spec))
        return xs[0] if xs else None
    return None


def scan_tag(scans: Union[int, Sequence[int]]) -> str:
    """
    Folder-style scan tag:
      - int -> "scan_167246"
      - [167246,167285] -> "scans_167246-167285"
    """
    if isinstance(scans, int):
        return f"scan_{int(scans)}"
    ss = [int(s) for s in list(scans)]
    if len(ss) == 0:
        return "scans_unknown"
    ss = sorted(ss)
    if len(ss) == 1:
        return f"scan_{ss[0]}"
    return f"scans_{ss[0]}-{ss[-1]}"


def scan_tag_file(scans: Union[int, Sequence[int]]) -> str:
    """
    File-style scan tag (no underscore after 'scan(s)'):
      - int -> "scan167246"
      - [167246,167285] -> "scans167246-167285"
    """
    if isinstance(scans, int):
        return f"scan{int(scans)}"
    ss = [int(s) for s in list(scans)]
    if len(ss) == 0:
        return "scansunknown"
    ss = sorted(ss)
    if len(ss) == 1:
        return f"scan{ss[0]}"
    return f"scans{ss[0]}-{ss[-1]}"


def azim_range_str(azim_window: Tuple[int, int]) -> str:
    """Canonical string key: 'a0_a1'."""
    a0, a1 = int(azim_window[0]), int(azim_window[1])
    return f"{a0}_{a1}"


def azim_center(azim_window: Tuple[int, int]) -> float:
    """Center angle (deg) for an azimuth window."""
    a0, a1 = float(azim_window[0]), float(azim_window[1])
    return 0.5 * (a0 + a1)


def chunk_list(xs: Sequence[T], chunk_size: int) -> List[List[T]]:
    """Split a sequence into consecutive chunks of size chunk_size."""
    xs = list(xs)
    if chunk_size <= 0:
        chunk_size = 1
    return [xs[i : i + chunk_size] for i in range(0, len(xs), chunk_size)]


def wl_tag_nm(excitation_wl_nm: Union[int, float, str]) -> str:
    """
    Canonical wavelength tag used in filenames/folders.
      - 1500 or 1500.0 -> "1500"
      - 1500.5 -> "1500p5"
      - fallback -> str(x)
    """
    try:
        v = float(excitation_wl_nm)
        if v.is_integer():
            return str(int(v))
        return str(v).replace(".", "p")
    except Exception:
        return str(excitation_wl_nm)


# ----------------------------
# Window builders
# ----------------------------

def windows_from_edges(
    edges: Sequence[Union[int, float]],
    *,
    include_full: bool = False,
    full_range: Tuple[Union[int, float], Union[int, float]] = (-90, 90),
    full_first: bool = True,
    make_int: bool = False,
) -> List[Tuple[Union[int, float], Union[int, float]]]:
    """
    Convert a list of edges into consecutive windows.

    - If make_int=True: edges are rounded to int (via to_int), then sorted+unique.
    - Otherwise edges are converted to float, preserving order.
    - include_full adds (full_range[0], full_range[1]) either first or last.

    Returns: [(e0,e1), (e1,e2), ...] (+ optional full_range)
    """
    ee = list(edges)

    if make_int:
        ee2 = [to_int(x) for x in ee]
        ee = sorted(set(ee2))
    else:
        ee = [float(x) for x in ee]

    if len(ee) < 2:
        return [(full_range[0], full_range[1])] if include_full else []

    wins: List[Tuple[Union[int, float], Union[int, float]]] = []
    if include_full and full_first:
        wins.append((full_range[0], full_range[1]))

    wins.extend([(ee[i], ee[i + 1]) for i in range(len(ee) - 1)])

    if include_full and (not full_first):
        wins.append((full_range[0], full_range[1]))

    return wins


def windows_from_ranges(
    azimuthal_ranges: Sequence[Union[int, float]],
    *,
    include_full: bool = False,
    full_range: Tuple[int, int] = (-90, 90),
) -> List[Tuple[int, int]]:
    """
    Backwards-compatible wrapper: rounds to int, sorts+unique, then windows.
    """
    wins = windows_from_edges(
        azimuthal_ranges,
        include_full=include_full,
        full_range=(int(full_range[0]), int(full_range[1])),
        full_first=False,
        make_int=True,
    )
    return [(int(a), int(b)) for a, b in wins]


# ----------------------------
# Fluence tags
# ----------------------------

def fluence_tag_file(fluence_mJ_cm2: Union[int, float]) -> str:
    """
    Canonical fluence tag for filenames (WITHOUT 'mJ' suffix):
      15   -> "15p0"
      15.2 -> "15p2"
    """
    return str(float(fluence_mJ_cm2)).replace(".", "p")


def fluence_tag_folder(fluence_mJ_cm2: Union[int, float]) -> str:
    """Canonical fluence tag for folder names (WITH 'mJ' suffix)."""
    return fluence_tag_file(fluence_mJ_cm2) + "mJ"


# ----------------------------
# XY file I/O
# ----------------------------

def save_xy(path: Union[str, Path], x, y, *, delimiter: str = " ") -> None:
    """Save 1D XY to text (two columns). Complements load_xy()."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack((np.asarray(x), np.asarray(y)))
    np.savetxt(str(p), arr, delimiter=delimiter)


def find_candidates(
    root: Union[str, Path],
    *,
    must_contain: Sequence[str] = (),
    exts: Sequence[str] = (),
    exclude_names: Sequence[str] = (),
    newest_first: bool = True,
) -> List[Path]:
    rootp = Path(root)
    if not rootp.exists():
        return []

    exts_l = tuple(e.lower() for e in exts) if exts else ()
    tokens = [t.lower() for t in must_contain if t is not None and str(t) != ""]
    exclude = {n.lower() for n in exclude_names}

    out: List[Path] = []
    for p in rootp.rglob("*"):
        if not p.is_file():
            continue
        if p.name.lower() in exclude:
            continue
        if exts_l and p.suffix.lower() not in exts_l:
            continue
        name = p.name.lower()
        if all(t in name for t in tokens):
            out.append(p)

    out.sort(key=lambda x: x.stat().st_mtime, reverse=bool(newest_first))
    return out


def load_xy(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load 1D XY from:
      - .xy / .txt / .dat (2 columns; header allowed)
      - .npy (Nx2)
      - .npz (keys like q/I, two_theta/I, x/y, or Nx2 array inside)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    suf = p.suffix.lower()

    if suf == ".npz":
        z = np.load(str(p), allow_pickle=True)
        keys = set(z.files)

        # Prefer common explicit key pairs first (safe enhancement)
        key_pairs = [
            ("q", "I"),
            ("q", "i"),
            ("two_theta", "I"),
            ("two_theta", "i"),
            ("x", "y"),
        ]
        for kx, ky in key_pairs:
            if (kx in keys) and (ky in keys):
                return np.asarray(z[kx], float), np.asarray(z[ky], float)

        # Fallback: any Nx2-like array stored under any key
        for k in z.files:
            arr = np.asarray(z[k])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return np.asarray(arr[:, 0], float), np.asarray(arr[:, 1], float)

        raise KeyError(f"Unrecognized .npz structure in {p.name}. Keys: {sorted(keys)}")

    if suf in (".xy", ".txt", ".dat"):
        try:
            arr = np.loadtxt(str(p))
        except Exception:
            try:
                arr = np.loadtxt(str(p), skiprows=1)
            except Exception:
                arr = np.genfromtxt(str(p))
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Expected >=2 columns in {p.name}")
        return np.asarray(arr[:, 0], float), np.asarray(arr[:, 1], float)

    if suf == ".npy":
        arr = np.asarray(np.load(str(p), allow_pickle=True))
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return np.asarray(arr[:, 0], float), np.asarray(arr[:, 1], float)
        raise ValueError(f"Unrecognized .npy structure in {p.name} (expected Nx2)")

    raise ValueError(f"Unsupported file extension for XY: {p.suffix}")


# ----------------------------
# Numeric utilities
# ----------------------------

def normalize_y_by_mean_in_xrange(
    x: np.ndarray,
    y: np.ndarray,
    x_range: Tuple[float, float],
) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x0, x1 = float(x_range[0]), float(x_range[1])
    m = (x >= x0) & (x <= x1)
    if not np.any(m):
        return y
    denom = float(np.nanmean(y[m]))
    if (not np.isfinite(denom)) or denom == 0.0:
        return y
    return y / denom


def resample_linear(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    x1: float,
    npt: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    npt = int(npt)
    if npt < 2:
        npt = 2
    xg = np.linspace(float(x0), float(x1), npt)
    yg = np.interp(xg, x, y)
    return xg, yg


def compute_r2(y: np.ndarray, yfit: np.ndarray) -> float:
    y = np.asarray(y, float)
    yfit = np.asarray(yfit, float)
    if y.size < 2:
        return float("nan")
    ss_res = float(np.nansum((y - yfit) ** 2))
    ybar = float(np.nanmean(y))
    ss_tot = float(np.nansum((y - ybar) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def fwhm_from_curve(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 5:
        return float("nan")
    ymax = float(np.nanmax(y))
    if not np.isfinite(ymax) or ymax <= 0:
        return float("nan")
    half = 0.5 * ymax
    above = np.where(y >= half)[0]
    if above.size < 2:
        return float("nan")
    i0, i1 = int(above[0]), int(above[-1])
    return float(abs(x[i1] - x[i0]))


def trapz_over_xmask(y2d, x, x_mask, *, axis=1, fill_value=0.0):
    """
    Integrate y2d over x for entries where x_mask is True, using np.trapezoid.

    Typical use:
      y2d: (n_rows, n_x)
      x:   (n_x,)
      axis=1 integrates each row over x.

    Returns:
      1D array of length y2d.shape[0] (for axis=1), or length y2d.shape[1] (for axis=0).
    """
    y2d = np.asarray(y2d, float)
    x = np.asarray(x, float)
    x_mask = np.asarray(x_mask, bool)

    if y2d.ndim != 2:
        raise ValueError("trapz_over_xmask expects y2d to be 2D.")
    if x.ndim != 1 or x_mask.ndim != 1 or x.size != x_mask.size:
        raise ValueError("x and x_mask must be 1D arrays with the same length.")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")

    if not np.any(x_mask):
        if axis == 1:
            return np.full(y2d.shape[0], float(fill_value), dtype=float)
        return np.full(y2d.shape[1], float(fill_value), dtype=float)

    if axis == 1:
        return np.trapezoid(y2d[:, x_mask], x[x_mask], axis=1)
    return np.trapezoid(y2d[x_mask, :], x[x_mask], axis=0)


# ----------------------------
# Fraction / folding utilities
# ----------------------------

def fraction_profile_centered(
    x,
    y,
    *,
    window_width: float,
    x_range=(-90.0, 90.0),
    clip_negative: bool = True,
    renormalize: bool = True,
    center_name: str = "x_center",
    lo_name: str = "x_lo",
    hi_name: str = "x_hi",
    integral_name: str = "integral",
    fraction_name: str = "fraction",
):
    """
    Bin (x,y) into centered windows of width window_width across x_range and compute:
      - integral in each window (trapz)
      - fraction = integral / total_integral (over x_range)

    Returns a pandas DataFrame with columns:
      [center_name, lo_name, hi_name, integral_name, fraction_name]
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    x0, x1 = float(x_range[0]), float(x_range[1])
    m = np.isfinite(x) & np.isfinite(y) & (x >= x0) & (x <= x1)
    x = x[m]
    y = y[m]

    if clip_negative:
        y = np.maximum(y, 0.0)

    if x.size < 2:
        return pd.DataFrame(columns=[center_name, lo_name, hi_name, integral_name, fraction_name])

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    total = float(np.trapezoid(y, x))
    if (not np.isfinite(total)) or total <= 0:
        total = np.nan

    w = float(window_width)
    half = 0.5 * w

    n = int(round((x1 - x0) / w))
    centers = [x0 + k * w for k in range(n + 1)]

    rows = []
    for k, c in enumerate(centers):
        lo = c - half
        hi = c + half
        lo_eff = max(lo, x0)
        hi_eff = min(hi, x1)

        if k < len(centers) - 1:
            mm = (x >= lo_eff) & (x < hi_eff)
        else:
            mm = (x >= lo_eff) & (x <= hi_eff)

        integ = float(np.trapezoid(y[mm], x[mm])) if np.count_nonzero(mm) >= 2 else 0.0
        frac = float(integ / total) if np.isfinite(total) else np.nan
        rows.append((float(c), float(lo_eff), float(hi_eff), float(integ), float(frac)))

    df = pd.DataFrame(rows, columns=[center_name, lo_name, hi_name, integral_name, fraction_name])

    if renormalize and len(df):
        s = float(np.nansum(df[fraction_name].values))
        if np.isfinite(s) and s > 0:
            df[fraction_name] = df[fraction_name] / s

    return df


def fold_symmetric_to_abs_center(
    df,
    *,
    center_col: str,
    frac_col: str,
    abs_col: str = "abs_x",
    mass_col: str = "fraction_mass",
    per_side_col: str = "fraction_per_side",
    n_col: str = "n",
    max_abs: float = 90.0,
    decimals: int = 6,
    renormalize_mass: bool = True,
):
    """
    Fold a symmetric distribution around 0 by grouping bins with the same |center|.

    Output columns:
      abs_col, mass_col, per_side_col, n_col
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[abs_col, mass_col, per_side_col, n_col])

    d = df.copy()
    d = d[np.isfinite(d[center_col]) & np.isfinite(d[frac_col])].copy()

    d[abs_col] = np.abs(d[center_col].astype(float))
    d = d[(d[abs_col] >= 0) & (d[abs_col] <= float(max_abs) + 1e-12)].copy()
    d[abs_col] = d[abs_col].round(int(decimals))

    g = (
        d.groupby(abs_col, as_index=False)[frac_col]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": mass_col, "count": n_col})
    )

    g[per_side_col] = g[mass_col] / g[n_col].astype(float)
    g = g.sort_values(abs_col).reset_index(drop=True)

    if renormalize_mass and len(g):
        s = float(np.nansum(g[mass_col].values))
        if np.isfinite(s) and s > 0:
            g[mass_col] = g[mass_col] / s
            g[per_side_col] = g[mass_col] / g[n_col].astype(float)

    return g[[abs_col, mass_col, per_side_col, n_col]]


# ----------------------------
# Physics conversions
# ----------------------------

def energy_to_wavelength(energy_ev: Union[float, int]) -> float:
    """Convert photon energy (eV) to wavelength (m): lambda = (h*c)/E."""
    e = float(energy_ev)
    if not np.isfinite(e) or e <= 0.0:
        return float("nan")
    h_ev_s = 4.135667696e-15
    c_m_s = 299_792_458.0
    return (h_ev_s * c_m_s) / e


def wavelength_to_energy(wavelength_m: Union[float, int]) -> float:
    """Convert wavelength (m) to photon energy (eV): E = (h*c)/lambda."""
    wl = float(wavelength_m)
    if not np.isfinite(wl) or wl <= 0.0:
        return float("nan")
    h_ev_s = 4.135667696e-15
    c_m_s = 299_792_458.0
    return (h_ev_s * c_m_s) / wl


def q_to_two_theta(
    q: Union[float, int, np.ndarray],
    wavelength: Union[float, int, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Convert scattering vector magnitude q (Å^-1) to 2theta (deg).

    NOTE: wavelength is expected in meters in this project (energy_to_wavelength outputs meters).
    Internally we convert to Å.
    """
    q = np.asarray(q, float)
    wl_a = np.asarray(wavelength, float) * 1e10  # m -> Å

    with np.errstate(invalid="ignore", divide="ignore"):
        arg = (q * wl_a) / (4.0 * np.pi)
        arg = np.clip(arg, -1.0, 1.0)
        two_theta = 2.0 * np.arcsin(arg)

    out = np.degrees(two_theta)
    return float(out) if out.shape == () else out


def two_theta_to_q(
    two_theta: Union[float, int, np.ndarray],
    wavelength: Union[float, int, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Convert 2theta (deg) to scattering vector magnitude q (Å^-1).

    NOTE: wavelength is expected in meters in this project (converted internally to Å).
    """
    tt = np.asarray(two_theta, float)
    wl_a = np.asarray(wavelength, float) * 1e10  # m -> Å

    with np.errstate(invalid="ignore", divide="ignore"):
        theta = 0.5 * np.radians(tt)
        q = (4.0 * np.pi / wl_a) * np.sin(theta)

    return float(q) if q.shape == () else q


# ----------------------------
# Integration / interpolation
# ----------------------------

def integrate_trapz_in_range(
    x: np.ndarray,
    y: np.ndarray,
    x_range: Tuple[float, float],
    *,
    require_min_points: int = 2,
) -> float:
    """Trapz-integrate y(x) over x_range. Returns 0.0 if too few points."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x0, x1 = float(x_range[0]), float(x_range[1])
    lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)

    m = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    if np.count_nonzero(m) < int(require_min_points):
        return 0.0
    xx = x[m]
    yy = y[m]
    idx = np.argsort(xx)
    xx = xx[idx]
    yy = yy[idx]
    return float(np.trapezoid(yy, xx))


def interp_1d_to_grid(
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_tgt: np.ndarray,
    *,
    left: float = np.nan,
    right: float = np.nan,
) -> np.ndarray:
    """Linear interpolation y_src(x_src) -> x_tgt. Out-of-bounds become left/right."""
    x_src = np.asarray(x_src, float)
    y_src = np.asarray(y_src, float)
    x_tgt = np.asarray(x_tgt, float)

    m = np.isfinite(x_src) & np.isfinite(y_src)
    if np.count_nonzero(m) < 2:
        return np.full_like(x_tgt, np.nan, dtype=float)

    xs = x_src[m]
    ys = y_src[m]
    idx = np.argsort(xs)
    xs = xs[idx]
    ys = ys[idx]

    return np.interp(x_tgt, xs, ys, left=left, right=right).astype(float)


# ----------------------------
# Polynomial baseline / detrend
# ----------------------------

def poly_baseline(
    x: np.ndarray,
    y: np.ndarray,
    order: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit polynomial baseline y ~ poly(x). Returns (coeffs, baseline(x))."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(m) < max(2, int(order) + 1):
        coeffs = np.full(int(order) + 1, np.nan, dtype=float)
        baseline = np.full_like(x, np.nan, dtype=float)
        return coeffs, baseline

    coeffs = np.polyfit(x[m], y[m], int(order))
    baseline = np.polyval(coeffs, x).astype(float)
    return np.asarray(coeffs, float), baseline


def detrend_poly(
    x: np.ndarray,
    y: np.ndarray,
    order: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_detrended, coeffs, baseline)."""
    coeffs, baseline = poly_baseline(x, y, order=order)
    y = np.asarray(y, float)
    y_detr = y - baseline
    return np.asarray(y_detr, float), np.asarray(coeffs, float), np.asarray(baseline, float)


# ----------------------------
# Uniform resampling helpers
# ----------------------------

def infer_uniform_step(
    x: np.ndarray,
    *,
    tol_rel: float = 1e-2,
    tol_abs: float = 0.0,
) -> Tuple[float, bool]:
    """
    Infer if x is (approximately) uniformly spaced.
    Returns (dt, is_uniform), where dt is median(diff(x)).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan"), False

    x = np.sort(x)
    dx = np.diff(x)
    dx = dx[np.isfinite(dx)]
    if dx.size == 0:
        return float("nan"), False

    dt = float(np.median(dx))
    if dt == 0:
        return dt, False

    err = float(np.max(np.abs(dx - dt)))
    ok = err <= max(float(tol_abs), float(tol_rel) * abs(dt))
    return dt, bool(ok)


def resample_to_uniform_grid(
    x: np.ndarray,
    y: np.ndarray,
    *,
    dt: Optional[float] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample (x,y) onto a uniform grid using linear interpolation.
    If dt is None, dt = median(diff(sorted(x))).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return np.asarray(x, float), np.asarray(y, float)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    if dt is None:
        dt0, _ = infer_uniform_step(x)
        dt = dt0

    dt = float(dt)
    if (not np.isfinite(dt)) or dt <= 0:
        return np.asarray(x, float), np.asarray(y, float)

    xmin = float(np.nanmin(x) if x_min is None else x_min)
    xmax = float(np.nanmax(x) if x_max is None else x_max)
    if xmax <= xmin:
        return np.asarray(x, float), np.asarray(y, float)

    n = int(np.floor((xmax - xmin) / dt)) + 1
    if n < 2:
        return np.asarray(x, float), np.asarray(y, float)

    xu = xmin + dt * np.arange(n, dtype=float)
    yu = interp_1d_to_grid(x, y, xu, left=np.nan, right=np.nan)

    mm = np.isfinite(yu)
    return np.asarray(xu[mm], float), np.asarray(yu[mm], float)


# ----------------------------
# FFT spectrum
# ----------------------------

def fft_spectrum(
    t_ps: np.ndarray,
    y: np.ndarray,
    *,
    detrend_order: int = 2,
    resample_uniform: bool = True,
    dt_ps: Optional[float] = None,
    freq_unit: str = "cm^-1",   # "hz" | "thz" | "cm^-1"
) -> dict:
    """
    Compute FFT spectrum of y(t).

    Steps:
      - (optional) resample to uniform grid
      - detrend with polynomial baseline of given order
      - FFT using numpy.fft.fft

    Returns dict with keys:
      t_ps, y_raw, baseline, y_detrended,
      dt_ps, freqs, fft_complex, freqs_pos, fft_pos
    """
    t_ps = np.asarray(t_ps, float)
    y = np.asarray(y, float)

    m = np.isfinite(t_ps) & np.isfinite(y)
    t_ps = t_ps[m]
    y = y[m]
    if t_ps.size < 2:
        return {
            "t_ps": t_ps,
            "y_raw": y,
            "baseline": np.full_like(y, np.nan),
            "y_detrended": np.full_like(y, np.nan),
            "dt_ps": np.nan,
            "freqs": np.array([], float),
            "fft_complex": np.array([], complex),
            "freqs_pos": np.array([], float),
            "fft_pos": np.array([], complex),
        }

    idx = np.argsort(t_ps)
    t_ps = t_ps[idx]
    y = y[idx]

    if resample_uniform:
        t_ps_u, y_u = resample_to_uniform_grid(t_ps, y, dt=dt_ps)
    else:
        t_ps_u, y_u = t_ps, y

    if t_ps_u.size < 2:
        return {
            "t_ps": t_ps_u,
            "y_raw": y_u,
            "baseline": np.full_like(y_u, np.nan),
            "y_detrended": np.full_like(y_u, np.nan),
            "dt_ps": np.nan,
            "freqs": np.array([], float),
            "fft_complex": np.array([], complex),
            "freqs_pos": np.array([], float),
            "fft_pos": np.array([], complex),
        }

    dt = float(np.mean(np.diff(t_ps_u)))  # ps
    if not np.isfinite(dt) or dt <= 0:
        dt = float("nan")

    y_detr, coeffs, baseline = detrend_poly(t_ps_u, y_u, order=int(detrend_order))

    dt_s = dt * 1e-12  # ps -> s
    fft_vals = np.fft.fft(y_detr)
    freqs_hz = np.fft.fftfreq(len(y_detr), d=dt_s)

    freq_unit_n = str(freq_unit).strip().lower()
    if freq_unit_n == "hz":
        freqs = freqs_hz
    elif freq_unit_n == "thz":
        freqs = freqs_hz * 1e-12
    elif freq_unit_n in ("cm^-1", "cm-1", "cm1"):
        c_cm_s = 2.99792458e10
        freqs = freqs_hz / c_cm_s
    else:
        raise ValueError("freq_unit must be 'hz', 'thz', or 'cm^-1'")

    pos = freqs >= 0
    freqs_pos = freqs[pos]
    fft_pos = fft_vals[pos]

    return {
        "t_ps": np.asarray(t_ps_u, float),
        "y_raw": np.asarray(y_u, float),
        "baseline": np.asarray(baseline, float),
        "y_detrended": np.asarray(y_detr, float),
        "dt_ps": float(dt),
        "freqs": np.asarray(freqs, float),
        "fft_complex": np.asarray(fft_vals, complex),
        "freqs_pos": np.asarray(freqs_pos, float),
        "fft_pos": np.asarray(fft_pos, complex),
        "poly_coeffs": np.asarray(coeffs, float),
    }


