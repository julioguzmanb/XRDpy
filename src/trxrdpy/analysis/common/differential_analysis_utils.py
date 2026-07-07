# differential_analysis_utils.py
"""
Utilities for differential 1D-pattern analyses on delay scans.

Recycles:
- azimint_utils.AzimIntegrator + DelayDataset/DarkDataset for loading/caching XY
- general_utils.integrate_trapz_in_range
- general_utils.fft_spectrum
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import general_utils
from . import azimint_utils
from .paths import AnalysisPaths


# import importlib
# importlib.reload(general_utils)


@dataclass(frozen=True)
class PeakSpec:
    """Define a q-space peak window and its background convention.

    Attributes
    ----------
    name : str
        Stable peak identifier written to result tables and figure labels.
    q_range : tuple of float
        Lower and upper q limits of the peak integration window in Å⁻¹.
    bg_mode : {"before", "after", "avg", "custom"}
        Location of the equal-width background window. ``"avg"`` averages
        windows immediately below and above the peak interval. ``"custom"``
        uses the explicit ``bg_range`` interval.
    bg_range : tuple of float, optional
        Explicit q limits for a custom background integration window.
    """

    name: str
    q_range: Tuple[float, float]
    bg_mode: str = "after"
    bg_range: Optional[Tuple[float, float]] = None


def _normalize_q_range(value: Sequence[Any], *, name: str) -> Tuple[float, float]:
    """Return sorted q limits and validate positive width."""
    try:
        q0, q1 = value
    except Exception:
        raise ValueError(f"{name} must be a two-value q range.")

    lo, hi = float(q0), float(q1)
    if hi < lo:
        lo, hi = hi, lo
    if hi <= lo:
        raise ValueError(f"{name} must have positive width.")
    return lo, hi


def resolve_bg_mode(spec: Dict[str, Any], bg_mode: Optional[str]) -> str:
    """Resolve the adjacent-background rule for a peak specification.

    An explicit mode takes precedence over the legacy ``bg_side`` entry.
    ``left`` and ``right`` are accepted as aliases for ``before`` and ``after``.
    """
    source = bg_mode
    if source is None and spec.get("bg_mode", None) is not None:
        source = spec.get("bg_mode")
    if source is None and spec.get("bg_side", None) is not None:
        source = spec.get("bg_side")

    if source is not None:
        m = str(source).strip().lower()
        if m in ("before", "after", "avg", "custom"):
            return m
        if m in ("left", "before_peak"):
            return "before"
        if m in ("right", "rigth", "after_peak"):
            return "after"
        if m in ("both", "average"):
            return "avg"
        if m in ("manual", "range", "explicit"):
            return "custom"
        raise ValueError(
            "bg_mode must be 'before'/'after'/'avg'/'custom' "
            "(or 'left'/'right')."
        )

    if spec.get("bg_range", None) is not None:
        return "custom"

    return "after"


def get_peak_spec(
    peak: str,
    *,
    peak_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    bg_mode: Optional[str] = None,
    # default_peak_specs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> PeakSpec:
    # specs = dict(default_peak_specs or {})
    """Validate and return one named differential-integration peak definition.

    Parameters
    ----------
    peak : str
        Peak label used to select a peak specification or fitting-result rows.
    peak_specs : Optional[Dict[str, Dict[str, Any]]]
        Mapping from peak labels to q ranges and optional background or fit settings.
    bg_mode : Optional[str]
        Background choice: ``"before"``, ``"after"``, ``"avg"``, or
        ``"custom"``.

    Returns
    -------
    PeakSpec
        Validated immutable peak specification.

    Raises
    ------
    ValueError
        If a selector, range, mode, unit, or metadata value is invalid.
    """
    specs = dict({})
    if peak_specs is not None:
        specs.update(dict(peak_specs))

    if str(peak) not in specs:
        raise ValueError(f"Unknown peak '{peak}'. Add it to PEAK_SPECS or pass peak_specs=...")

    s = dict(specs[str(peak)])
    if "q_range" not in s:
        raise ValueError(f"Peak '{peak}' spec must include q_range=(q0,q1).")

    q0, q1 = _normalize_q_range(s["q_range"], name=f"Peak '{peak}' q_range")
    bm = resolve_bg_mode(s, bg_mode)
    bg_range = None
    if s.get("bg_range", None) is not None:
        bg_range = _normalize_q_range(
            s["bg_range"],
            name=f"Peak '{peak}' bg_range",
        )
    if bm == "custom" and bg_range is None:
        raise ValueError(
            f"Peak '{peak}' uses bg_mode='custom' and must include bg_range=(q0,q1)."
        )
    return PeakSpec(
        name=str(peak),
        q_range=(float(q0), float(q1)),
        bg_mode=str(bm),
        bg_range=bg_range,
    )


def background_description(peak_spec: PeakSpec) -> str:
    """Return a compact background description for figure titles and logs."""
    mode = str(peak_spec.bg_mode).strip().lower()
    if mode == "custom" and peak_spec.bg_range is not None:
        q0, q1 = peak_spec.q_range
        b0, b1 = peak_spec.bg_range
        scale = abs(float(q1) - float(q0)) / abs(float(b1) - float(b0))
        return f"custom bg=({b0:.3g}, {b1:.3g}), scale={scale:g}"
    return mode


def select_series_for_fft(
    df: pd.DataFrame,
    *,
    region: str,
    kind: str,
    time_window_select_ps: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a sorted delay-domain signal from a differential result table.

    ``region`` selects the peak or background rows and ``kind`` selects signed
    or absolute integrated difference. An optional window clips delays in ps.
    """
    region = str(region).strip().lower()
    if region == "bg":
        region = "background"
    if region not in ("peak", "background"):
        raise ValueError("region must be 'peak' or 'background'.")

    kind = str(kind).strip().lower()
    if kind not in ("diff", "absdiff"):
        raise ValueError("kind must be 'diff' or 'absdiff'.")

    ycol = "int_delta" if kind == "diff" else "int_abs_delta"

    dsel = df[df["region"].astype(str) == region].copy()
    dsel = dsel[np.isfinite(dsel["delay_ps"].astype(float).values) & np.isfinite(dsel[ycol].astype(float).values)].copy()
    dsel = dsel.sort_values("delay_ps")

    t_ps = dsel["delay_ps"].astype(float).values
    y = dsel[ycol].astype(float).values

    
    if time_window_select_ps is not None:
        t0, t1 = float(time_window_select_ps[0]), float(time_window_select_ps[1])
        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        m = (t_ps >= lo) & (t_ps <= hi)
        t_ps = t_ps[m]
        y = y[m]
    
    return np.asarray(t_ps, float), np.asarray(y, float)


def default_save_dir_delay_experiment(
    *,
    sample_name: str,
    temperature_K: Union[int, float],
    excitation_wl_nm: Union[int, float],
    fluence_mJ_cm2: Union[int, float],
    time_window_fs: int,
    figures_subdir: Union[str, Path] = Path("figures") / "diff_analysis",
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
    paths: Optional[AnalysisPaths] = None,
) -> Path:
    """Return the figure directory for one delay-series experiment.

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
    figures_subdir : Union[str, Path]
        Relative figures directory below the experiment analysis directory.
    path_root : Optional[Union[str, Path]]
        Root directory containing raw and analysis data trees.
    analysis_subdir : Optional[Union[str, Path]]
        Analysis-directory path relative to ``path_root``.
    paths : Optional[AnalysisPaths]
        Resolved ``AnalysisPaths`` configuration. It takes precedence over legacy path arguments.

    Returns
    -------
    Path
        Resolved path, label, or filename derived from experiment metadata.

    Raises
    ------
    ValueError
        If a selector, range, mode, unit, or metadata value is invalid.

    Notes
    -----
    This operation may create or replace analysis artifacts according to its save and overwrite settings.
    """
    wl_tag = general_utils.wl_tag_nm(excitation_wl_nm)
    flu_tag = general_utils.fluence_tag_folder(fluence_mJ_cm2)

    if paths is not None:
        analysis_root = Path(paths.analysis_root)
    elif path_root is not None and analysis_subdir is not None:
        analysis_root = Path(path_root) / Path(analysis_subdir)
    else:
        raise ValueError(
            "Provide either paths=AnalysisPaths(...), or both "
            "path_root=... and analysis_subdir=...."
        )

    return (
        analysis_root
        / str(sample_name)
        / f"temperature_{general_utils.to_int(temperature_K)}K"
        / f"excitation_wl_{wl_tag}nm"
        / "delay"
        / f"fluence_{flu_tag}"
        / f"time_window_{int(time_window_fs)}fs"
        / Path(figures_subdir)
    )


def default_save_name_integrals(
    *,
    sample_name: str,
    temperature_K: Union[int, float],
    excitation_wl_nm: Union[int, float],
    fluence_mJ_cm2: Union[int, float],
    time_window_fs: int,
    peak_name: str,
    q_range: Tuple[float, float],
    azim_window: Tuple[float, float],
    ref_type: str,
) -> str:
    """Build a descriptive filename stem for delay-series integral plots.

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
    peak_name : str
        Human-readable peak identifier used in results and filenames.
    q_range : Tuple[float, float]
        Closed q interval in Å⁻¹ used for fitting or integration.
    azim_window : Tuple[float, float]
        Inclusive azimuthal integration limits in degrees.
    ref_type : str
        Reference source, normally ``"dark"``, ``"delay"``, or ``"fluence"``.

    Returns
    -------
    str
        Resolved path, label, or filename derived from experiment metadata.

    Notes
    -----
    This operation may create or replace analysis artifacts according to its save and overwrite settings.
    """
    az0, az1 = float(azim_window[0]), float(azim_window[1])
    return (
        f"diff_integrals_{sample_name}_T{general_utils.to_int(temperature_K)}K_"
        f"wl{general_utils.wl_tag_nm(excitation_wl_nm)}_"
        f"flu{general_utils.fluence_tag_file(fluence_mJ_cm2)}_"
        f"tw{int(time_window_fs)}fs_hkl{str(peak_name)}_"
        f"az{az0:g}_{az1:g}_ref{str(ref_type)}"
    )


def default_save_name_fft(
    *,
    sample_name: str,
    temperature_K: Union[int, float],
    excitation_wl_nm: Union[int, float],
    fluence_mJ_cm2: Union[int, float],
    time_window_fs: int,
    peak_name: str,
    azim_window: Tuple[float, float],
    region: str,
    kind: str,
    poly_order: int,
    time_window_select_ps: Optional[Tuple[float, float]],
    ref_type: str,
) -> str:
    """Build a descriptive filename stem for differential FFT plots.

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
    peak_name : str
        Human-readable peak identifier used in results and filenames.
    azim_window : Tuple[float, float]
        Inclusive azimuthal integration limits in degrees.
    region : str
        Differential region to use, either the peak or its matched background window.
    kind : str
        Use signed ``"diff"`` or integrated ``"absdiff"`` values.
    poly_order : int
        Polynomial order used for baseline removal before the FFT.
    time_window_select_ps : Optional[Tuple[float, float]]
        Optional inclusive picosecond interval retained before Fourier transformation.
    ref_type : str
        Reference source, normally ``"dark"``, ``"delay"``, or ``"fluence"``.

    Returns
    -------
    str
        Resolved path, label, or filename derived from experiment metadata.

    Notes
    -----
    This operation may create or replace analysis artifacts according to its save and overwrite settings.
    """
    az0, az1 = float(azim_window[0]), float(azim_window[1])

    wtag = ""
    if time_window_select_ps is not None:
        lo = min(time_window_select_ps)
        hi = max(time_window_select_ps)
        wtag = f"_win{lo:g}_{hi:g}ps"

    return (
        f"diff_fft_{sample_name}_T{general_utils.to_int(temperature_K)}K_"
        f"wl{general_utils.wl_tag_nm(excitation_wl_nm)}_"
        f"flu{general_utils.fluence_tag_file(fluence_mJ_cm2)}_"
        f"tw{int(time_window_fs)}fs_hkl{str(peak_name)}_"
        f"az{az0:g}_{az1:g}_{str(region)}_{str(kind)}_poly{int(poly_order)}{wtag}_ref{str(ref_type)}"
    )


def _resolve_exp_path_kwargs(
    exp: Dict[str, Any],
    *,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """Resolve path configuration for multi-experiment helpers.

    Priority:
      1) per-experiment `exp["paths"]`
      2) per-experiment `exp["path_root"]` + `exp["analysis_subdir"]`
      3) function-level `paths`
      4) function-level `path_root` + `analysis_subdir`
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
        "No path configuration available for experiment. Provide either:\n"
        "  - exp['paths'] = AnalysisPaths(...), or\n"
        "  - exp['path_root'] and exp['analysis_subdir'], or\n"
        "  - function-level paths=..., or\n"
        "  - function-level path_root=... and analysis_subdir=..."
    )


class DelayDifferentialAnalyzer:
    """Compute integrated differential signals over a peak window and a matching background window.

    Output columns (long-form):
      delay_fs, delay_ps, region, int_delta, int_abs_delta,
      q0, q1, bg0, bg1, bg_mode, bg_scale, peak_width, bg_width, peak_name,
      ref_type, ref_value, azim0, azim1

    ``region`` is either ``"peak"`` or ``"background"``.

    Attributes
    ----------
    sample_name : str
        Sample identifier used to resolve delay-series datasets.
    temperature_K : int
        Sample temperature in kelvin.
    excitation_wl_nm : int or float
        Pump-laser wavelength in nanometres.
    fluence_mJ_cm2 : float
        Fixed pump fluence for the delay series in mJ cm⁻².
    time_window_fs : int
        Temporal averaging-window width in femtoseconds.
    integrator : azimint_utils.AzimIntegrator
        Shared loader/integrator used to obtain cached or newly integrated XY data.
    analysis_root : pathlib.Path
        Root directory for processed experiment products.
    _dataset_kwargs : dict
        Normalized path arguments forwarded to dataset descriptors.
    """

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
        polarization_factor: Optional[float] = None,
        paths: Optional[AnalysisPaths] = None,
        path_root: Optional[Union[str, Path]] = None,
        analysis_subdir: Optional[Union[str, Path]] = None,
    ):
        """Bind one delay experiment and configure its shared azimuthal integrator."""
        self.poni_path = None if poni_path is None else str(poni_path)
        self.mask_edf_path = None if mask_edf_path is None else str(mask_edf_path)

        self.sample_name = str(sample_name)
        self.temperature_K = general_utils.to_int(temperature_K)
        self.excitation_wl_nm = excitation_wl_nm
        self.fluence_mJ_cm2 = float(fluence_mJ_cm2)
        self.time_window_fs = int(time_window_fs)

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
            polarization_factor=polarization_factor,
        )

    def _dataset_path_kwargs(self) -> Dict[str, object]:
        """Return normalized path arguments forwarded to differential dataset constructors."""
        return dict(self._dataset_kwargs)

    def _legacy_path_kwargs(self) -> Dict[str, str]:
        """Represent the active path configuration using legacy string arguments."""
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

    @staticmethod
    def _bg_ranges(
        q_range: Tuple[float, float],
        bg_mode: str,
        bg_range: Optional[Tuple[float, float]] = None,
    ) -> Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Construct background interval(s) for a peak range."""
        q0, q1 = float(q_range[0]), float(q_range[1])
        if q1 < q0:
            q0, q1 = q1, q0
        w = q1 - q0
        if w <= 0:
            raise ValueError("q_range must have positive width.")

        m = str(bg_mode).strip().lower()
        if m == "before":
            return (q0 - w, q0)
        if m == "after":
            return (q1, q1 + w)
        if m == "avg":
            return ((q0 - w, q0), (q1, q1 + w))
        if m == "custom":
            if bg_range is None:
                raise ValueError("bg_mode='custom' requires bg_range=(q0,q1).")
            return _normalize_q_range(bg_range, name="bg_range")
        raise ValueError("bg_mode must be 'before', 'after', 'avg', or 'custom'.")

    @staticmethod
    def _background_width(
        bg_ranges: Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]],
        bg_mode: str,
    ) -> float:
        """Return the effective background q width."""
        if str(bg_mode).strip().lower() == "avg":
            return float(abs(float(bg_ranges[0][1]) - float(bg_ranges[0][0])))  # type: ignore[index]
        return float(abs(float(bg_ranges[1]) - float(bg_ranges[0])))  # type: ignore[arg-type]

    @classmethod
    def _background_scale(
        cls,
        q_range: Tuple[float, float],
        bg_ranges: Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]],
        bg_mode: str,
    ) -> float:
        """Scale custom-background integrals to the peak q width."""
        if str(bg_mode).strip().lower() != "custom":
            return 1.0
        peak_width = abs(float(q_range[1]) - float(q_range[0]))
        bg_width = cls._background_width(bg_ranges, bg_mode)
        if bg_width <= 0:
            raise ValueError("bg_range must have positive width.")
        return float(peak_width / bg_width)

    @staticmethod
    def _interp_to_ref_grid(q_ref: np.ndarray, q: np.ndarray, I: np.ndarray) -> np.ndarray:
        """Interpolate a pattern onto the overlapping portion of a reference q grid."""
        q_ref = np.asarray(q_ref, float)
        q = np.asarray(q, float)
        I = np.asarray(I, float)

        m = np.isfinite(q) & np.isfinite(I)
        if np.count_nonzero(m) < 2:
            return np.full_like(q_ref, np.nan, dtype=float)

        qq = q[m]
        II = I[m]
        idx = np.argsort(qq)
        qq = qq[idx]
        II = II[idx]

        return np.interp(q_ref, qq, II, left=np.nan, right=np.nan).astype(float)

    @staticmethod
    def _integrals_from_diff(
        q_ref: np.ndarray,
        diff: np.ndarray,
        q_range: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Integrate signed and absolute differential intensity over q bounds."""
        int_delta = general_utils.integrate_trapz_in_range(q_ref, diff, q_range)
        int_abs = general_utils.integrate_trapz_in_range(q_ref, np.abs(diff), q_range)
        return float(int_delta), float(int_abs)

    def _load_reference_xy(
        self,
        *,
        azim_window: Tuple[float, float],
        ref_type: str,
        ref_value: Union[int, str, Sequence[int]],
        compute_if_missing: bool = True,
        overwrite_xy: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load or integrate the configured dark or delay reference pattern."""
        ref_type = str(ref_type).strip().lower()
        if ref_type not in ("dark", "delay"):
            raise ValueError("ref_type must be 'dark' or 'delay'.")

        if ref_type == "dark":
            dark_tag = azimint_utils.dark_tag_from_scan_spec(ref_value)
            ds = azimint_utils.DarkDataset(
                sample_name=self.sample_name,
                temperature_K=self.temperature_K,
                dark_tag=dark_tag,
                **self._dataset_path_kwargs(),
            )
            _, q_ref, I_ref = self.integrator.get_xy_for_window(
                ds,
                azimuthal_range=azim_window,
                compute_if_missing=bool(compute_if_missing),
                overwrite_xy=bool(overwrite_xy),
            )
            return np.asarray(q_ref, float), np.asarray(I_ref, float)

        if isinstance(ref_value, (list, tuple)):
            delay_fs = int(list(ref_value)[0]) if len(ref_value) else 0
        else:
            delay_fs = int(ref_value)

        ds = azimint_utils.DelayDataset(
            self.sample_name,
            self.temperature_K,
            self.excitation_wl_nm,
            self.fluence_mJ_cm2,
            self.time_window_fs,
            delay_fs=int(delay_fs),
            **self._dataset_path_kwargs(),
        )
        _, q_ref, I_ref = self.integrator.get_xy_for_window(
            ds,
            azimuthal_range=azim_window,
            compute_if_missing=bool(compute_if_missing),
            overwrite_xy=bool(overwrite_xy),
        )
        return np.asarray(q_ref, float), np.asarray(I_ref, float)

    def compute_delay_integrals(
        self,
        *,
        delays_fs: Union[int, Sequence[int], str],
        azim_window: Tuple[float, float],
        peak_spec: PeakSpec,
        ref_type: str,
        ref_value: Union[int, str, Sequence[int]],
        compute_if_missing: bool = True,
        overwrite_xy: bool = False,
        include_reference_in_output: bool = False,
        from_2D_imgs: bool = True
    ) -> pd.DataFrame:
        """Compute delay integrals.

        Parameters
        ----------
        delays_fs : Union[int, Sequence[int], str]
            Delay selector in femtoseconds; may be a scalar, sequence, or ``"all"``.
        azim_window : Tuple[float, float]
            Inclusive azimuthal integration limits in degrees.
        peak_spec : PeakSpec
            Validated peak name, q interval, and adjacent-background mode.
        ref_type : str
            Reference source, normally ``"dark"``, ``"delay"``, or ``"fluence"``.
        ref_value : Union[int, str, Sequence[int]]
            Reference scan specification, delay in fs, or fluence in mJ/cm² according to ``ref_type``.
        compute_if_missing : bool
            Whether missing XY patterns may be integrated from their 2D images.
        overwrite_xy : bool
            Whether existing XY cache files should be recomputed.
        include_reference_in_output : bool
            Whether to prepend a zero-valued ``reference`` row to the returned
            long-form table.
        from_2D_imgs : bool
            Whether available points are discovered from 2D images rather than XY caches.

        Returns
        -------
        pd.DataFrame
            Long-form table with peak and background integrals for each delay.
        """
        az0, az1 = float(azim_window[0]), float(azim_window[1])
        pk = peak_spec

        delays_list = azimint_utils.normalize_delays_fs(
            delays_fs,
            sample_name=self.sample_name,
            temperature_K=self.temperature_K,
            excitation_wl_nm=self.excitation_wl_nm,
            fluence_mJ_cm2=self.fluence_mJ_cm2,
            time_window_fs=self.time_window_fs,
            from_2D_imgs=from_2D_imgs,
            **self._legacy_path_kwargs(),
        )

        q_ref, I_ref = self._load_reference_xy(
            azim_window=(az0, az1),
            ref_type=ref_type,
            ref_value=ref_value,
            compute_if_missing=bool(compute_if_missing),
            overwrite_xy=bool(overwrite_xy),
        )

        q_range = (float(pk.q_range[0]), float(pk.q_range[1]))
        bg_mode = str(pk.bg_mode).strip().lower()
        bg_ranges = self._bg_ranges(q_range, bg_mode, pk.bg_range)
        peak_width = abs(float(q_range[1]) - float(q_range[0]))
        bg_width = self._background_width(bg_ranges, bg_mode)
        bg_scale = self._background_scale(q_range, bg_ranges, bg_mode)

        rows: List[Dict[str, Any]] = []

        if include_reference_in_output:
            rows.append(
                dict(
                    delay_fs=np.nan,
                    delay_ps=np.nan,
                    region="reference",
                    int_delta=0.0,
                    int_abs_delta=0.0,
                    q0=q_range[0],
                    q1=q_range[1],
                    bg0=np.nan,
                    bg1=np.nan,
                    bg_mode=bg_mode,
                    bg_scale=np.nan,
                    peak_width=peak_width,
                    bg_width=np.nan,
                    peak_name=pk.name,
                    ref_type=str(ref_type),
                    ref_value=str(ref_value),
                    azim0=az0,
                    azim1=az1,
                    time_window_fs=int(self.time_window_fs),
                )
            )

        for d in delays_list:
            ds = azimint_utils.DelayDataset(
                self.sample_name,
                self.temperature_K,
                self.excitation_wl_nm,
                self.fluence_mJ_cm2,
                self.time_window_fs,
                delay_fs=int(d),
                **self._dataset_path_kwargs(),
            )

            try:
                _, q, I = self.integrator.get_xy_for_window(
                    ds,
                    azimuthal_range=(az0, az1),
                    compute_if_missing=bool(compute_if_missing),
                    overwrite_xy=bool(overwrite_xy),
                )
            except Exception:
                continue

            I_on_ref = self._interp_to_ref_grid(q_ref, q, I)
            diff = I_on_ref - I_ref

            pk_int, pk_abs = self._integrals_from_diff(q_ref, diff, q_range)
            rows.append(
                dict(
                    delay_fs=int(d),
                    delay_ps=float(d) * 1e-3,
                    region="peak",
                    int_delta=float(pk_int),
                    int_abs_delta=float(pk_abs),
                    q0=q_range[0],
                    q1=q_range[1],
                    bg0=np.nan,
                    bg1=np.nan,
                    bg_mode=bg_mode,
                    bg_scale=np.nan,
                    peak_width=peak_width,
                    bg_width=np.nan,
                    peak_name=pk.name,
                    ref_type=str(ref_type),
                    ref_value=str(ref_value),
                    azim0=az0,
                    azim1=az1,
                    time_window_fs=int(self.time_window_fs),
                )
            )

            if bg_mode == "avg" and isinstance(bg_ranges, tuple) and len(bg_ranges) == 2 and isinstance(bg_ranges[0], tuple):
                (b0a, b1a), (b0b, b1b) = bg_ranges  # type: ignore[misc]
                b_int_a, b_abs_a = self._integrals_from_diff(q_ref, diff, (b0a, b1a))
                b_int_b, b_abs_b = self._integrals_from_diff(q_ref, diff, (b0b, b1b))
                b_int = 0.5 * (b_int_a + b_int_b)
                b_abs = 0.5 * (b_abs_a + b_abs_b)
                bg0, bg1 = np.nan, np.nan
            else:
                b0, b1 = bg_ranges  # type: ignore[misc]
                b_int, b_abs = self._integrals_from_diff(q_ref, diff, (b0, b1))
                bg0, bg1 = float(b0), float(b1)

            b_int = float(b_int) * float(bg_scale)
            b_abs = float(b_abs) * float(bg_scale)

            rows.append(
                dict(
                    delay_fs=int(d),
                    delay_ps=float(d) * 1e-3,
                    region="background",
                    int_delta=float(b_int),
                    int_abs_delta=float(b_abs),
                    q0=q_range[0],
                    q1=q_range[1],
                    bg0=bg0,
                    bg1=bg1,
                    bg_mode=bg_mode,
                    bg_scale=float(bg_scale),
                    peak_width=float(peak_width),
                    bg_width=float(bg_width),
                    peak_name=pk.name,
                    ref_type=str(ref_type),
                    ref_value=str(ref_value),
                    azim0=az0,
                    azim1=az1,
                    time_window_fs=int(self.time_window_fs),
                )
            )

        df = pd.DataFrame(rows)
        if "delay_fs" in df.columns:
            df = df.sort_values(["region", "delay_fs"], na_position="first").reset_index(drop=True)

        return df

    def compute_fft(
        self,
        *,
        time_ps: np.ndarray,
        signal: np.ndarray,
        poly_order: int = 1,
        resample_uniform: bool = False,
        dt_ps: Optional[float] = None,
        freq_unit: str = "cm^-1",
    ) -> dict:
        """Detrend and Fourier-transform a differential time trace.

        Parameters
        ----------
        time_ps : np.ndarray
            Sampling times in picoseconds corresponding to ``signal``.
        signal : np.ndarray
            Sampled delay-domain signal to detrend and transform.
        poly_order : int
            Polynomial order used for baseline removal before the FFT.
        resample_uniform : bool
            Whether to interpolate the time trace onto a uniform grid before the FFT.
        dt_ps : Optional[float]
            Optional uniform time step in picoseconds used for FFT resampling.
        freq_unit : str
            Frequency-axis unit, for example ``"THz"`` or ``"cm^-1"``.

        Returns
        -------
        dict
            FFT mapping containing the processed time trace, frequency axis, amplitude, and detrending metadata.
        """
        return general_utils.fft_spectrum(
            np.asarray(time_ps, float),
            np.asarray(signal, float),
            detrend_order=int(poly_order),
            resample_uniform=bool(resample_uniform),
            dt_ps=dt_ps,
            freq_unit=str(freq_unit),
        )


def default_save_dir_fluence_experiment(
    *,
    sample_name: str,
    temperature_K: Union[int, float],
    excitation_wl_nm: Union[int, float],
    delay_fs: int,
    time_window_fs: int,
    figures_subdir: Union[str, Path] = Path("figures") / "diff_analysis",
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> Path:
    """Return the figure directory for one fluence-series experiment.

    Parameters
    ----------
    sample_name : str
        Sample identifier used in the standardized analysis directory layout.
    temperature_K : Union[int, float]
        Sample temperature in kelvin.
    excitation_wl_nm : Union[int, float]
        Pump wavelength in nanometres.
    delay_fs : int
        Pump-probe delay in femtoseconds.
    time_window_fs : int
        Width of the delay bin or acquisition window in femtoseconds.
    figures_subdir : Union[str, Path]
        Relative figures directory below the experiment analysis directory.
    paths : Optional[AnalysisPaths]
        Resolved ``AnalysisPaths`` configuration. It takes precedence over legacy path arguments.
    path_root : Optional[Union[str, Path]]
        Root directory containing raw and analysis data trees.
    analysis_subdir : Optional[Union[str, Path]]
        Analysis-directory path relative to ``path_root``.

    Returns
    -------
    Path
        Resolved path, label, or filename derived from experiment metadata.

    Raises
    ------
    ValueError
        If a selector, range, mode, unit, or metadata value is invalid.

    Notes
    -----
    This operation may create or replace analysis artifacts according to its save and overwrite settings.
    """
    wl_tag = general_utils.wl_tag_nm(excitation_wl_nm)

    if paths is not None:
        analysis_root = Path(paths.analysis_root)
    elif path_root is not None and analysis_subdir is not None:
        analysis_root = Path(path_root) / Path(analysis_subdir)
    else:
        raise ValueError(
            "Provide either paths=AnalysisPaths(...), or both "
            "path_root=... and analysis_subdir=...."
        )

    return (
        analysis_root
        / str(sample_name)
        / f"temperature_{general_utils.to_int(temperature_K)}K"
        / f"excitation_wl_{wl_tag}nm"
        / "fluence"
        / f"delay_{int(delay_fs)}fs"
        / f"time_window_{int(time_window_fs)}fs"
        / Path(figures_subdir)
    )


def default_save_name_integrals_fluence(
    *,
    sample_name: str,
    temperature_K: Union[int, float],
    excitation_wl_nm: Union[int, float],
    delay_fs: int,
    time_window_fs: int,
    peak_name: str,
    q_range: Tuple[float, float],
    azim_window: Tuple[float, float],
    ref_type: str,
) -> str:
    """Build a descriptive filename stem for fluence-series integral plots.

    Parameters
    ----------
    sample_name : str
        Sample identifier used in the standardized analysis directory layout.
    temperature_K : Union[int, float]
        Sample temperature in kelvin.
    excitation_wl_nm : Union[int, float]
        Pump wavelength in nanometres.
    delay_fs : int
        Pump-probe delay in femtoseconds.
    time_window_fs : int
        Width of the delay bin or acquisition window in femtoseconds.
    peak_name : str
        Human-readable peak identifier used in results and filenames.
    q_range : Tuple[float, float]
        Closed q interval in Å⁻¹ used for fitting or integration.
    azim_window : Tuple[float, float]
        Inclusive azimuthal integration limits in degrees.
    ref_type : str
        Reference source, normally ``"dark"``, ``"delay"``, or ``"fluence"``.

    Returns
    -------
    str
        Resolved path, label, or filename derived from experiment metadata.

    Notes
    -----
    This operation may create or replace analysis artifacts according to its save and overwrite settings.
    """
    az0, az1 = float(azim_window[0]), float(azim_window[1])
    return (
        f"diff_integrals_fluence_{sample_name}_T{general_utils.to_int(temperature_K)}K_"
        f"wl{general_utils.wl_tag_nm(excitation_wl_nm)}_"
        f"dly{int(delay_fs)}fs_tw{int(time_window_fs)}fs_hkl{str(peak_name)}_"
        f"q{float(q_range[0]):g}_{float(q_range[1]):g}_"
        f"az{az0:g}_{az1:g}_ref{str(ref_type)}"
    )


class FluenceDifferentialAnalyzer:
    """Compute integrated differential signals over a peak window and a matching background window,
    but across FLUENCE points at a fixed delay.

    Output columns (long-form):
      fluence_mJ_cm2, region, int_delta, int_abs_delta,
      q0, q1, bg0, bg1, bg_mode, bg_scale, peak_width, bg_width, peak_name,
      ref_type, ref_value, azim0, azim1, delay_fs

    ``region`` is either ``"peak"`` or ``"background"``.

    Attributes
    ----------
    sample_name : str
        Sample identifier used to resolve fluence-series datasets.
    temperature_K : int
        Sample temperature in kelvin.
    excitation_wl_nm : int or float
        Pump-laser wavelength in nanometres.
    delay_fs : int
        Fixed pump–probe delay in femtoseconds.
    time_window_fs : int
        Temporal averaging-window width in femtoseconds.
    integrator : azimint_utils.AzimIntegrator
        Shared loader/integrator used to obtain cached or newly integrated XY data.
    analysis_root : pathlib.Path
        Root directory for processed experiment products.
    _dataset_kwargs : dict
        Normalized path arguments forwarded to dataset descriptors.
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
        polarization_factor: Optional[float] = None,
        paths: Optional[AnalysisPaths] = None,
        path_root: Optional[Union[str, Path]] = None,
        analysis_subdir: Optional[Union[str, Path]] = None,
    ):
        """Bind one fixed-delay fluence experiment and configure its integrator."""
        self.poni_path = None if poni_path is None else str(poni_path)
        self.mask_edf_path = None if mask_edf_path is None else str(mask_edf_path)

        self.sample_name = str(sample_name)
        self.temperature_K = general_utils.to_int(temperature_K)
        self.excitation_wl_nm = excitation_wl_nm
        self.delay_fs = int(delay_fs)
        self.time_window_fs = int(time_window_fs)

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
            polarization_factor=polarization_factor,
        )

    def _dataset_path_kwargs(self) -> Dict[str, object]:
        """Return normalized path arguments forwarded to differential dataset constructors."""
        return dict(self._dataset_kwargs)

    def _legacy_path_kwargs(self) -> Dict[str, str]:
        """Represent active paths using backward-compatible root and subdirectory strings."""
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

    @staticmethod
    def _bg_ranges(
        q_range: Tuple[float, float],
        bg_mode: str,
        bg_range: Optional[Tuple[float, float]] = None,
    ) -> Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Construct background interval(s) for a peak range."""
        q0, q1 = float(q_range[0]), float(q_range[1])
        if q1 < q0:
            q0, q1 = q1, q0
        w = q1 - q0
        if w <= 0:
            raise ValueError("q_range must have positive width.")

        m = str(bg_mode).strip().lower()
        if m == "before":
            return (q0 - w, q0)
        if m == "after":
            return (q1, q1 + w)
        if m == "avg":
            return ((q0 - w, q0), (q1, q1 + w))
        if m == "custom":
            if bg_range is None:
                raise ValueError("bg_mode='custom' requires bg_range=(q0,q1).")
            return _normalize_q_range(bg_range, name="bg_range")
        raise ValueError("bg_mode must be 'before', 'after', 'avg', or 'custom'.")

    @staticmethod
    def _background_width(
        bg_ranges: Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]],
        bg_mode: str,
    ) -> float:
        """Return the effective background q width."""
        if str(bg_mode).strip().lower() == "avg":
            return float(abs(float(bg_ranges[0][1]) - float(bg_ranges[0][0])))  # type: ignore[index]
        return float(abs(float(bg_ranges[1]) - float(bg_ranges[0])))  # type: ignore[arg-type]

    @classmethod
    def _background_scale(
        cls,
        q_range: Tuple[float, float],
        bg_ranges: Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]],
        bg_mode: str,
    ) -> float:
        """Scale custom-background integrals to the peak q width."""
        if str(bg_mode).strip().lower() != "custom":
            return 1.0
        peak_width = abs(float(q_range[1]) - float(q_range[0]))
        bg_width = cls._background_width(bg_ranges, bg_mode)
        if bg_width <= 0:
            raise ValueError("bg_range must have positive width.")
        return float(peak_width / bg_width)

    @staticmethod
    def _interp_to_ref_grid(q_ref: np.ndarray, q: np.ndarray, I: np.ndarray) -> np.ndarray:
        """Interpolate a pattern onto the overlapping portion of the reference q grid."""
        q_ref = np.asarray(q_ref, float)
        q = np.asarray(q, float)
        I = np.asarray(I, float)

        m = np.isfinite(q) & np.isfinite(I)
        if np.count_nonzero(m) < 2:
            return np.full_like(q_ref, np.nan, dtype=float)

        qq = q[m]
        II = I[m]
        idx = np.argsort(qq)
        qq = qq[idx]
        II = II[idx]

        return np.interp(q_ref, qq, II, left=np.nan, right=np.nan).astype(float)

    @staticmethod
    def _integrals_from_diff(
        q_ref: np.ndarray,
        diff: np.ndarray,
        q_range: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Integrate signed and absolute differential intensity inside the requested q interval."""
        int_delta = general_utils.integrate_trapz_in_range(q_ref, diff, q_range)
        int_abs = general_utils.integrate_trapz_in_range(q_ref, np.abs(diff), q_range)
        return float(int_delta), float(int_abs)

    def _load_reference_xy(
        self,
        *,
        azim_window: Tuple[float, float],
        ref_type: str,
        ref_value: Union[int, float, str, Sequence[int]],
        compute_if_missing: bool = True,
        overwrite_xy: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load or integrate the configured dark or fluence reference pattern."""
        ref_type = str(ref_type).strip().lower()
        if ref_type not in ("dark", "fluence"):
            raise ValueError("ref_type must be 'dark' or 'fluence' for fluence analysis.")

        if ref_type == "dark":
            dark_tag = azimint_utils.dark_tag_from_scan_spec(ref_value)  # type: ignore[arg-type]
            ds = azimint_utils.DarkDataset(
                sample_name=self.sample_name,
                temperature_K=self.temperature_K,
                dark_tag=dark_tag,
                **self._dataset_path_kwargs(),
            )
            _, q_ref, I_ref = self.integrator.get_xy_for_window(
                ds,
                azimuthal_range=azim_window,
                compute_if_missing=bool(compute_if_missing),
                overwrite_xy=bool(overwrite_xy),
            )
            return np.asarray(q_ref, float), np.asarray(I_ref, float)

        try:
            fref = float(ref_value)  # type: ignore[arg-type]
        except Exception:
            raise ValueError("For ref_type='fluence', ref_value must be a numeric fluence (mJ/cm^2).")

        ds = azimint_utils.FluenceDataset(
            self.sample_name,
            self.temperature_K,
            self.excitation_wl_nm,
            fluence_mJ_cm2=float(fref),
            time_window_fs=int(self.time_window_fs),
            delay_fs=int(self.delay_fs),
            **self._dataset_path_kwargs(),
        )
        _, q_ref, I_ref = self.integrator.get_xy_for_window(
            ds,
            azimuthal_range=azim_window,
            compute_if_missing=bool(compute_if_missing),
            overwrite_xy=bool(overwrite_xy),
        )
        return np.asarray(q_ref, float), np.asarray(I_ref, float)

    def compute_fluence_integrals(
        self,
        *,
        fluences_mJ_cm2: Union[float, int, Sequence[Union[float, int]], str],
        azim_window: Tuple[float, float],
        peak_spec: PeakSpec,
        ref_type: str,
        ref_value: Union[int, float, str, Sequence[int]],
        compute_if_missing: bool = True,
        overwrite_xy: bool = False,
        include_reference_in_output: bool = False,
    ) -> pd.DataFrame:
        """Compute fluence integrals.

        Parameters
        ----------
        fluences_mJ_cm2 : Union[float, int, Sequence[Union[float, int]], str]
            Compatibility selector. The current implementation discovers and
            processes every cached fluence point at the fixed delay.
        azim_window : Tuple[float, float]
            Inclusive azimuthal integration limits in degrees.
        peak_spec : PeakSpec
            Validated peak name, q interval, and adjacent-background mode.
        ref_type : str
            Reference source, normally ``"dark"``, ``"delay"``, or ``"fluence"``.
        ref_value : Union[int, float, str, Sequence[int]]
            Reference scan specification, delay in fs, or fluence in mJ/cm² according to ``ref_type``.
        compute_if_missing : bool
            Whether missing XY patterns may be integrated from their 2D images.
        overwrite_xy : bool
            Whether existing XY cache files should be recomputed.
        include_reference_in_output : bool
            Whether to prepend a zero-valued ``reference`` row to the returned
            long-form table.

        Returns
        -------
        pd.DataFrame
            Long-form table with peak and background integrals for each fluence.
        """
        az0, az1 = float(azim_window[0]), float(azim_window[1])
        pk = peak_spec

        flist = azimint_utils.available_fluence_points_mJ_cm2(
            sample_name=self.sample_name,
            temperature_K=self.temperature_K,
            excitation_wl_nm=self.excitation_wl_nm,
            delay_fs=int(self.delay_fs),
            time_window_fs=int(self.time_window_fs),
            from_2D_imgs=False,
            **self._legacy_path_kwargs(),
        )

        q_ref, I_ref = self._load_reference_xy(
            azim_window=(az0, az1),
            ref_type=str(ref_type),
            ref_value=ref_value,
            compute_if_missing=bool(compute_if_missing),
            overwrite_xy=bool(overwrite_xy),
        )

        q_range = (float(pk.q_range[0]), float(pk.q_range[1]))
        bg_mode = str(pk.bg_mode).strip().lower()
        bg_ranges = self._bg_ranges(q_range, bg_mode, pk.bg_range)
        peak_width = abs(float(q_range[1]) - float(q_range[0]))
        bg_width = self._background_width(bg_ranges, bg_mode)
        bg_scale = self._background_scale(q_range, bg_ranges, bg_mode)

        rows: List[Dict[str, Any]] = []

        if include_reference_in_output:
            rows.append(
                dict(
                    fluence_mJ_cm2=np.nan,
                    region="reference",
                    int_delta=0.0,
                    int_abs_delta=0.0,
                    q0=q_range[0],
                    q1=q_range[1],
                    bg0=np.nan,
                    bg1=np.nan,
                    bg_mode=bg_mode,
                    bg_scale=np.nan,
                    peak_width=peak_width,
                    bg_width=np.nan,
                    peak_name=pk.name,
                    ref_type=str(ref_type),
                    ref_value=str(ref_value),
                    azim0=az0,
                    azim1=az1,
                    delay_fs=int(self.delay_fs),
                )
            )

        for f in flist:
            ds = azimint_utils.FluenceDataset(
                self.sample_name,
                self.temperature_K,
                self.excitation_wl_nm,
                fluence_mJ_cm2=float(f),
                time_window_fs=int(self.time_window_fs),
                delay_fs=int(self.delay_fs),
                **self._dataset_path_kwargs(),
            )

            try:
                _, q, I = self.integrator.get_xy_for_window(
                    ds,
                    azimuthal_range=(az0, az1),
                    compute_if_missing=bool(compute_if_missing),
                    overwrite_xy=bool(overwrite_xy),
                )
            except Exception:
                continue

            I_on_ref = self._interp_to_ref_grid(q_ref, q, I)
            diff = I_on_ref - I_ref

            pk_int, pk_abs = self._integrals_from_diff(q_ref, diff, q_range)
            rows.append(
                dict(
                    fluence_mJ_cm2=float(f),
                    region="peak",
                    int_delta=float(pk_int),
                    int_abs_delta=float(pk_abs),
                    q0=q_range[0],
                    q1=q_range[1],
                    bg0=np.nan,
                    bg1=np.nan,
                    bg_mode=bg_mode,
                    bg_scale=np.nan,
                    peak_width=peak_width,
                    bg_width=np.nan,
                    peak_name=pk.name,
                    ref_type=str(ref_type),
                    ref_value=str(ref_value),
                    azim0=az0,
                    azim1=az1,
                    delay_fs=int(self.delay_fs),
                )
            )

            if bg_mode == "avg" and isinstance(bg_ranges, tuple) and len(bg_ranges) == 2 and isinstance(bg_ranges[0], tuple):
                (b0a, b1a), (b0b, b1b) = bg_ranges  # type: ignore[misc]
                b_int_a, b_abs_a = self._integrals_from_diff(q_ref, diff, (b0a, b1a))
                b_int_b, b_abs_b = self._integrals_from_diff(q_ref, diff, (b0b, b1b))
                b_int = 0.5 * (b_int_a + b_int_b)
                b_abs = 0.5 * (b_abs_a + b_abs_b)
                bg0, bg1 = np.nan, np.nan
            else:
                b0, b1 = bg_ranges  # type: ignore[misc]
                b_int, b_abs = self._integrals_from_diff(q_ref, diff, (b0, b1))
                bg0, bg1 = float(b0), float(b1)

            b_int = float(b_int) * float(bg_scale)
            b_abs = float(b_abs) * float(bg_scale)

            rows.append(
                dict(
                    fluence_mJ_cm2=float(f),
                    region="background",
                    int_delta=float(b_int),
                    int_abs_delta=float(b_abs),
                    q0=q_range[0],
                    q1=q_range[1],
                    bg0=bg0,
                    bg1=bg1,
                    bg_mode=bg_mode,
                    bg_scale=float(bg_scale),
                    peak_width=float(peak_width),
                    bg_width=float(bg_width),
                    peak_name=pk.name,
                    ref_type=str(ref_type),
                    ref_value=str(ref_value),
                    azim0=az0,
                    azim1=az1,
                    delay_fs=int(self.delay_fs),
                )
            )

        df = pd.DataFrame(rows)
        if "fluence_mJ_cm2" in df.columns:
            df = df.sort_values(["region", "fluence_mJ_cm2"], na_position="first").reset_index(drop=True)
        return df


def build_multi_delay_integral_series(
    experiments: Sequence[Dict[str, Any]],
    *,
    delays_fs: Union[int, Sequence[int], str],
    pk_spec: PeakSpec,
    azim_window: Tuple[float, float],
    poni_path: Optional[str] = None,
    mask_edf_path: Optional[str] = None,
    npt: int = 1000,
    normalize_xy: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """For each experiment dict in `experiments`, compute integrated differential
    signals (peak + background) using DelayDifferentialAnalyzer and return a
    list of per-experiment series dicts ready for plotting.

    Each returned dict has keys:
      - "experiment": original experiment dict
      - "delay_offset_ps": float
      - "time_ps": 1D np.ndarray (delay in ps, no offset)
      - "int_delta": 1D np.ndarray (peak ΔI)
      - "int_abs_delta": 1D np.ndarray (peak |ΔI|)
      - "err_delta": 1D np.ndarray (per-delay error for ΔI, from background)
      - "err_abs_delta": 1D np.ndarray (per-delay error for |ΔI|, from background)

    Parameters
    ----------
    experiments : sequence of dict
        Delay experiment descriptors containing metadata and reference fields.
    delays_fs, pk_spec, azim_window
        Delay selector, peak/background definition, and integration sector.
    poni_path, mask_edf_path, azim_offset_deg, polarization_factor
        Default geometry, mask, and detector corrections.
    npt, normalize_xy, q_norm_range, compute_if_missing, overwrite_xy
        Radial integration, normalization, and XY cache controls.
    paths, path_root, analysis_subdir
        Default paths for experiments without their own configuration.

    Returns
    -------
    list of dict
        Plot-ready delay series with signals and background-derived errors.
    """
    az0, az1 = float(azim_window[0]), float(azim_window[1])

    series_list: List[Dict[str, Any]] = []

    for exp in list(experiments):
        sample_name = str(exp.get("sample_name"))
        temperature_K = int(exp.get("temperature_K"))
        excitation_wl_nm = float(exp.get("excitation_wl_nm"))
        fluence_mJ_cm2 = float(exp.get("fluence_mJ_cm2"))
        time_window_fs = int(exp.get("time_window_fs"))

        ref_type = str(exp.get("ref_type", "dark"))
        ref_value = exp.get("ref_value", None)
        if ref_value is None:
            raise ValueError("Each experiment must provide ref_value (e.g. dark scans or delay).")

        delay_offset_ps = float(exp.get("delay_offset_ps", 0.0))

        analyzer = DelayDifferentialAnalyzer(
            sample_name=sample_name,
            temperature_K=temperature_K,
            excitation_wl_nm=excitation_wl_nm,
            fluence_mJ_cm2=fluence_mJ_cm2,
            time_window_fs=time_window_fs,
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            npt=int(npt),
            normalize_xy=bool(normalize_xy),
            q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
            azim_offset_deg=float(azim_offset_deg),
            polarization_factor=polarization_factor,
            **_resolve_exp_path_kwargs(
                exp,
                paths=paths,
                path_root=path_root,
                analysis_subdir=analysis_subdir,
            ),
        )

        df = analyzer.compute_delay_integrals(
            delays_fs=delays_fs,
            azim_window=(az0, az1),
            peak_spec=pk_spec,
            ref_type=ref_type,
            ref_value=ref_value,
            compute_if_missing=bool(compute_if_missing),
            overwrite_xy=bool(overwrite_xy),
            include_reference_in_output=False,
            from_2D_imgs=False
        )

        if "delay_ps" not in df.columns:
            df["delay_ps"] = df["delay_fs"].astype(float) * 1e-3

        d_peak = df[df["region"].astype(str) == "peak"].copy()
        d_bg = df[df["region"].astype(str) == "background"].copy()

        if d_peak.empty or d_bg.empty:
            continue

        merged = d_peak.merge(
            d_bg[["delay_fs", "int_delta", "int_abs_delta"]],
            on="delay_fs",
            how="inner",
            suffixes=("_peak", "_bg"),
        )

        if merged.empty:
            continue

        merged = merged[np.isfinite(merged["delay_ps"].astype(float).values)].copy()
        merged = merged.sort_values("delay_ps")

        t_ps = merged["delay_ps"].astype(float).values
        y_delta = merged["int_delta_peak"].astype(float).values
        y_abs = merged["int_abs_delta_peak"].astype(float).values

        err_delta = np.abs(merged["int_delta_bg"].astype(float).values)
        err_abs = merged["int_abs_delta_bg"].astype(float).values

        series_list.append(
            dict(
                experiment=exp,
                delay_offset_ps=float(delay_offset_ps),
                time_ps=np.asarray(t_ps, float),
                int_delta=np.asarray(y_delta, float),
                int_abs_delta=np.asarray(y_abs, float),
                err_delta=np.asarray(err_delta, float),
                err_abs_delta=np.asarray(err_abs, float),
            )
        )

    return series_list


def _fft_from_time_series(
    time_ps: np.ndarray,
    signal: np.ndarray,
    *,
    poly_order: int = 2,
    freq_unit: str = "cm^-1",
) -> Dict[str, np.ndarray]:
    """Simple FFT helper:

      - detrends signal with a polynomial of order `poly_order`
      - resamples onto a uniform grid (if needed)
      - computes one-sided FFT amplitude

    Returned dict keys:
      - "time_ps": time array used for detrending (original, sorted)
      - "signal_detrended": detrended signal (on original grid)
      - "freq": frequency axis (in freq_unit)
      - "amp": FFT magnitude
    """
    t = np.asarray(time_ps, float)
    y = np.asarray(signal, float)

    m = np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(m) < 3:
        return dict(time_ps=t, signal_detrended=np.zeros_like(t), freq=np.array([]), amp=np.array([]))

    t = t[m]
    y = y[m]

    # Sort by time
    idx = np.argsort(t)
    t = t[idx]
    y = y[idx]

    # Detrend with polynomial
    if poly_order is not None and poly_order >= 0:
        # Shift time to improve conditioning
        t0 = 0.5 * (t[0] + t[-1])
        tc = t - t0
        try:
            coeffs = np.polyfit(tc, y, int(poly_order))
            trend = np.polyval(coeffs, tc)
            y_det = y - trend
        except Exception:
            y_det = y - np.mean(y)
    else:
        y_det = y - np.mean(y)

    # Uniform resampling for FFT
    n = len(t)
    if n < 4:
        return dict(time_ps=t, signal_detrended=y_det, freq=np.array([]), amp=np.array([]))

    t_min, t_max = float(t[0]), float(t[-1])
    if t_max <= t_min:
        return dict(time_ps=t, signal_detrended=y_det, freq=np.array([]), amp=np.array([]))

    t_uniform = np.linspace(t_min, t_max, n)
    y_uniform = np.interp(t_uniform, t, y_det)

    # FFT
    dt_ps = (t_uniform[-1] - t_uniform[0]) / float(len(t_uniform) - 1)
    # convert ps -> s
    dt_s = dt_ps * 1e-12

    if dt_s <= 0:
        return dict(time_ps=t, signal_detrended=y_det, freq=np.array([]), amp=np.array([]))

    fft_vals = np.fft.rfft(y_uniform)
    freqs_hz = np.fft.rfftfreq(len(y_uniform), d=dt_s)

    # Convert to requested units
    fu = str(freq_unit).strip().lower()
    if fu in ("cm^-1", "cm^-1.", "cm^-1", "cm-1"):
        c = 2.99792458e8  # m/s
        freqs = freqs_hz / (c * 100.0)  # Hz -> cm^-1
    elif fu in ("thz", "tHz"):
        freqs = freqs_hz * 1e-12
    else:
        # Fallback: plain Hz
        freqs = freqs_hz

    # Simple magnitude normalization
    amp = 2.0 * np.abs(fft_vals) / float(len(y_uniform))

    return dict(
        time_ps=t,
        signal_detrended=y_det,
        freq=np.asarray(freqs, float),
        amp=np.asarray(amp, float),
    )


def build_multi_delay_fft_series(
    experiments: Sequence[Dict[str, Any]],
    *,
    delays_fs: Union[int, Sequence[int], str],
    pk_spec: PeakSpec,
    azim_window: Tuple[float, float],
    poni_path: str,
    mask_edf_path: str,
    kind: str = "diff",
    time_window_select_ps: Optional[Tuple[float, float]] = None,
    poly_order: int = 1,
    freq_unit: str = "cm^-1",
    npt: int = 1000,
    normalize_xy: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """For each experiment dict in `experiments`, compute the differential time
    trace (peak/background) and FFTs for DELAY scans.

    IMPORTANT:
      This implementation now uses DelayDifferentialAnalyzer.compute_fft,
      i.e. the same FFT engine as the single-experiment plot_differential_fft
      (general_utils.fft_spectrum), so the spectra are numerically consistent.

    Each returned dict has keys:
      - "experiment": original experiment dict
      - "delay_offset_ps": float
      - "time_ps": 1D np.ndarray (time axis for peak trace, same as in single FFT)
      - "signal_peak": 1D np.ndarray (detrended peak time trace)
      - "signal_bg": 1D np.ndarray (detrended background time trace)
      - "fft_peak": dict with "freq" and "amp" (from freqs_pos / fft_pos)
      - "fft_bg":   dict with "freq" and "amp" (from freqs_pos / fft_pos)

    Parameters
    ----------
    experiments : sequence of dict
        Delay experiment descriptors containing metadata and reference fields.
    delays_fs, pk_spec, azim_window
        Delay selector, peak/background definition, and integration sector.
    poni_path, mask_edf_path
        Default pyFAI geometry and detector mask.
    kind : {"diff", "absdiff"}
        Signed or absolute differential signal to transform.
    time_window_select_ps, poly_order, freq_unit
        Time clipping, detrending order, and FFT frequency unit.
    npt, normalize_xy, q_norm_range, azim_offset_deg, polarization_factor,
    compute_if_missing, overwrite_xy
        Integration, correction, normalization, and XY cache controls.
    paths, path_root, analysis_subdir
        Default paths for experiments without their own configuration.

    Returns
    -------
    list of dict
        Plot-ready detrended traces and peak/background FFT spectra.
    """
    region_kind = str(kind).strip().lower()
    if region_kind not in ("diff", "absdiff"):
        raise ValueError("kind must be 'diff' or 'absdiff'.")

    az0, az1 = float(azim_window[0]), float(azim_window[1])

    series_list: List[Dict[str, Any]] = []

    for exp in list(experiments):
        sample_name = str(exp.get("sample_name"))
        temperature_K = int(exp.get("temperature_K"))
        excitation_wl_nm = float(exp.get("excitation_wl_nm"))
        fluence_mJ_cm2 = float(exp.get("fluence_mJ_cm2"))
        time_window_fs = int(exp.get("time_window_fs"))

        delay_offset_ps = float(exp.get("delay_offset_ps", 0.0))

        ref_type = str(exp.get("ref_type", "dark"))
        ref_value = exp.get("ref_value", None)
        if ref_value is None:
            raise ValueError("Each experiment must provide ref_value (e.g. dark scans or delay).")

        analyzer = DelayDifferentialAnalyzer(
            sample_name=sample_name,
            temperature_K=temperature_K,
            excitation_wl_nm=excitation_wl_nm,
            fluence_mJ_cm2=fluence_mJ_cm2,
            time_window_fs=time_window_fs,
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            npt=int(npt),
            normalize_xy=bool(normalize_xy),
            q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
            azim_offset_deg=float(azim_offset_deg),
            polarization_factor=polarization_factor,
            **_resolve_exp_path_kwargs(
                exp,
                paths=paths,
                path_root=path_root,
                analysis_subdir=analysis_subdir,
            ),
        )

        df = analyzer.compute_delay_integrals(
            delays_fs=delays_fs,
            azim_window=(az0, az1),
            peak_spec=pk_spec,
            ref_type=ref_type,
            ref_value=ref_value,
            compute_if_missing=bool(compute_if_missing),
            overwrite_xy=bool(overwrite_xy),
            include_reference_in_output=False,
            from_2D_imgs=False
        )

        if "delay_ps" not in df.columns:
            df["delay_ps"] = df["delay_fs"].astype(float) * 1e-3

        if time_window_select_ps is not None:
            win_lo, win_hi = float(time_window_select_ps[0]), float(time_window_select_ps[1])
            win_lo_eff = win_lo - delay_offset_ps
            win_hi_eff = win_hi - delay_offset_ps
            win_eff = (win_lo_eff, win_hi_eff)
        else:
            win_eff = None

        t_peak, y_peak = select_series_for_fft(
            df, region="peak", kind=region_kind, time_window_select_ps=win_eff
        )
        t_bg, y_bg = select_series_for_fft(
            df, region="background", kind=region_kind, time_window_select_ps=win_eff
        )

        if t_peak.size == 0:
            continue

        fft_peak_full = analyzer.compute_fft(
            time_ps=t_peak,
            signal=y_peak,
            poly_order=int(poly_order),
            resample_uniform=False,
            dt_ps=None,
            freq_unit=str(freq_unit),
        )

        fft_bg_full = analyzer.compute_fft(
            time_ps=t_bg,
            signal=y_bg,
            poly_order=int(poly_order),
            resample_uniform=False,
            dt_ps=None,
            freq_unit=str(freq_unit),
        )

        freq_peak = np.asarray(fft_peak_full.get("freqs_pos", np.array([])), float)
        freq_bg = np.asarray(fft_bg_full.get("freqs_pos", np.array([])), float)

        amp_peak = np.abs(np.asarray(fft_peak_full.get("fft_pos", np.array([])), complex))
        amp_bg = np.abs(np.asarray(fft_bg_full.get("fft_pos", np.array([])), complex))

        t_fft = np.asarray(fft_peak_full.get("t_ps", t_peak), float)
        sig_peak = np.asarray(fft_peak_full.get("y_detrended", y_peak), float)
        sig_bg = np.asarray(fft_bg_full.get("y_detrended", y_bg), float)

        series_list.append(
            dict(
                experiment=exp,
                delay_offset_ps=float(delay_offset_ps),
                time_ps=t_fft,
                signal_peak=sig_peak,
                signal_bg=sig_bg,
                fft_peak=dict(freq=freq_peak, amp=amp_peak),
                fft_bg=dict(freq=freq_bg, amp=amp_bg),
            )
        )

    return series_list


def build_multi_fluence_integral_series(
    experiments: Sequence[Dict[str, Any]],
    *,
    fluences_mJ_cm2: Union[float, int, Sequence[Union[float, int]], str],
    pk_spec: PeakSpec,
    azim_window: Tuple[float, float],
    poni_path: Optional[str] = None,
    mask_edf_path: Optional[str] = None,
    npt: int = 1000,
    normalize_xy: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    compute_if_missing: bool = True,
    overwrite_xy: bool = False,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """For each experiment dict in `experiments`, compute integrated differential
    signals (peak + background) using FluenceDifferentialAnalyzer and return a
    list of per-experiment series dicts ready for plotting.

    Each returned dict has keys:
      - "experiment": original experiment dict
      - "fluence_scale": float (optional x-scale for plotting)
      - "fluence_offset": float (optional x-shift for plotting)
      - "delay_offset_fs": float (optional display offset for fixed-delay labels)
      - "fluence_mJ_cm2": 1D np.ndarray (x axis, no offset)
      - "int_delta": 1D np.ndarray (peak ΔI)
      - "int_abs_delta": 1D np.ndarray (peak |ΔI|)
      - "err_delta": 1D np.ndarray (per-fluence error for ΔI, from background)
      - "err_abs_delta": 1D np.ndarray (per-fluence error for |ΔI|, from background)

    Parameters
    ----------
    experiments : sequence of dict
        Fluence experiment descriptors containing metadata and reference fields.
    fluences_mJ_cm2, pk_spec, azim_window
        Fluence selector, peak/background definition, and integration sector.
    poni_path, mask_edf_path, azim_offset_deg, polarization_factor
        Default geometry, mask, and detector corrections.
    npt, normalize_xy, q_norm_range, compute_if_missing, overwrite_xy
        Radial integration, normalization, and XY cache controls.
    paths, path_root, analysis_subdir
        Default paths for experiments without their own configuration.

    Returns
    -------
    list of dict
        Plot-ready fluence series with signals and background-derived errors.
    """
    az0, az1 = float(azim_window[0]), float(azim_window[1])

    series_list: List[Dict[str, Any]] = []

    for exp in list(experiments):
        sample_name = str(exp.get("sample_name"))
        temperature_K = int(exp.get("temperature_K"))
        excitation_wl_nm = float(exp.get("excitation_wl_nm"))
        delay_fs = int(exp.get("delay_fs"))
        time_window_fs = int(exp.get("time_window_fs"))

        ref_type = str(exp.get("ref_type", "dark"))
        ref_value = exp.get("ref_value", None)
        if ref_value is None:
            raise ValueError("Each experiment must provide ref_value (e.g. dark scans or fluence).")

        fluence_scale = float(exp.get("fluence_scale", 1.0))
        fluence_offset = float(exp.get("fluence_offset", 0.0))
        delay_offset_fs = float(
            exp.get(
                "delay_offset_fs",
                float(exp.get("delay_offset_ps", 0.0)) * 1000.0,
            )
        )

        analyzer = FluenceDifferentialAnalyzer(
            sample_name=sample_name,
            temperature_K=temperature_K,
            excitation_wl_nm=excitation_wl_nm,
            delay_fs=int(delay_fs),
            time_window_fs=time_window_fs,
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            npt=int(npt),
            normalize_xy=bool(normalize_xy),
            q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
            azim_offset_deg=float(azim_offset_deg),
            polarization_factor=polarization_factor,
            **_resolve_exp_path_kwargs(
                exp,
                paths=paths,
                path_root=path_root,
                analysis_subdir=analysis_subdir,
            ),
        )

        df = analyzer.compute_fluence_integrals(
            fluences_mJ_cm2=fluences_mJ_cm2,
            azim_window=(az0, az1),
            peak_spec=pk_spec,
            ref_type=ref_type,
            ref_value=ref_value,
            compute_if_missing=bool(compute_if_missing),
            overwrite_xy=bool(overwrite_xy),
            include_reference_in_output=False,
        )

        if "fluence_mJ_cm2" not in df.columns:
            continue

        d_peak = df[df["region"].astype(str) == "peak"].copy()
        d_bg = df[df["region"].astype(str) == "background"].copy()
        if d_peak.empty or d_bg.empty:
            continue

        merged = d_peak.merge(
            d_bg[["fluence_mJ_cm2", "int_delta", "int_abs_delta"]],
            on="fluence_mJ_cm2",
            how="inner",
            suffixes=("_peak", "_bg"),
        )
        if merged.empty:
            continue

        merged = merged[np.isfinite(merged["fluence_mJ_cm2"].astype(float).values)].copy()
        merged = merged.sort_values("fluence_mJ_cm2")

        x_flu = merged["fluence_mJ_cm2"].astype(float).values
        y_delta = merged["int_delta_peak"].astype(float).values
        y_abs = merged["int_abs_delta_peak"].astype(float).values

        err_delta = np.abs(merged["int_delta_bg"].astype(float).values)
        err_abs = merged["int_abs_delta_bg"].astype(float).values

        series_list.append(
            dict(
                experiment=exp,
                fluence_scale=float(fluence_scale),
                fluence_offset=float(fluence_offset),
                delay_offset_fs=float(delay_offset_fs),
                fluence_mJ_cm2=np.asarray(x_flu, float),
                int_delta=np.asarray(y_delta, float),
                int_abs_delta=np.asarray(y_abs, float),
                err_delta=np.asarray(err_delta, float),
                err_abs_delta=np.asarray(err_abs, float),
            )
        )

    return series_list
