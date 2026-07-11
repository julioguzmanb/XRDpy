from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from .common import calibration_utils, general_utils, plot_utils
from .common.paths import AnalysisPaths

ScanSpec = calibration_utils.ScanSpec

FIGURES_SUBDIR_DEFAULT = "figures/calibration/"
DEFAULT_AZIMUTHAL_RANGES = tuple(np.arange(-90 + 15, 90 + 15, 30))


def _make_context(
    *,
    sample_name: str,
    temperature_K: int,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> calibration_utils.CalibrationContext:
    """Build a calibration context for the requested sample and temperature."""
    return calibration_utils.CalibrationContext(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )


def _azim_window_int(
    azim_window: Tuple[Union[int, float], Union[int, float]]
) -> Tuple[int, int]:
    """Round a two-angle window to the integer labels used by XY caches."""
    return (int(round(float(azim_window[0]))), int(round(float(azim_window[1]))))


def _full_range_int(
    full_range: Tuple[Union[int, float], Union[int, float]]
) -> Tuple[int, int]:
    """Round a full-range azimuth interval to canonical integer limits."""
    return (int(round(float(full_range[0]))), int(round(float(full_range[1]))))


def _windows_from_edges(
    azimuthal_ranges: Sequence[Union[int, float]],
    *,
    include_full: bool,
    full_range: Tuple[Union[int, float], Union[int, float]],
) -> List[Tuple[int, int]]:
    """Convert ordered azimuthal edges into adjacent integration windows.

    The optional full-range window is placed first so plots and fit tables use
    the same ordering as the integration cache.
    """
    edges = np.asarray(azimuthal_ranges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("azimuthal_ranges must be a 1D sequence with at least two entries.")

    windows: List[Tuple[int, int]] = []
    if bool(include_full):
        windows.append(_full_range_int(full_range))

    for j in range(edges.size - 1):
        windows.append((int(round(float(edges[j]))), int(round(float(edges[j + 1])))))

    return windows


def _is_int_like(x) -> bool:
    """Return whether a value is an integer or NumPy integer."""
    return isinstance(x, (int, np.integer))


def _normalize_scan_specs(x, *, mode: str) -> List[ScanSpec]:
    """Normalize scalar, grouped, and nested scan selectors.

    In ``"together"`` mode a flat integer list represents one combined dark
    dataset; in ``"separate"`` mode each integer becomes an independent
    dataset. Explicit nested lists always remain combined groups.
    """
    if _is_int_like(x) or isinstance(x, str):
        return [x]

    try:
        xs = list(x)
    except Exception:
        return [x]

    if len(xs) == 0:
        return []

    if all(_is_int_like(v) for v in xs):
        if mode == "together":
            return [[int(v) for v in xs]]
        if mode == "separate":
            return [int(v) for v in xs]
        raise ValueError("int_list_mode must be 'together' or 'separate'.")

    specs: List[ScanSpec] = []
    for v in xs:
        if _is_int_like(v) or isinstance(v, str):
            specs.append(v)
        else:
            vv = list(v)
            if len(vv) == 0 or (not all(_is_int_like(t) for t in vv)):
                raise ValueError("Invalid scans format.")
            specs.append([int(t) for t in vv])

    return specs


def _spec_label(spec: ScanSpec) -> str:
    """Return a stable scan-group label, falling back to the input string."""
    try:
        return str(general_utils.scan_tag(spec))
    except Exception:
        return str(spec)


def _band_from_center_width(
    *,
    center: float,
    width: float,
    name: str,
) -> Tuple[float, float]:
    """Return an increasing q band from a center and full width."""
    c = float(center)
    w = float(width)
    if not np.isfinite(c):
        raise ValueError(f"{name} center must be finite.")
    if (not np.isfinite(w)) or w <= 0.0:
        raise ValueError(f"{name} width must be positive and finite.")
    return (c - 0.5 * w, c + 0.5 * w)


def _validate_q_band(
    band: Tuple[Union[int, float], Union[int, float]],
    *,
    name: str,
) -> Tuple[float, float]:
    """Validate and normalize a q interval."""
    lo = float(band[0])
    hi = float(band[1])
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        raise ValueError(f"{name} must contain two increasing finite q values.")
    return lo, hi


def _integrate_cake_q_band(
    cake_intensity: np.ndarray,
    q: np.ndarray,
    q_band: Tuple[float, float],
    *,
    name: str,
) -> Tuple[np.ndarray, float, int]:
    """Integrate cake intensity over a q band for each azimuth bin."""
    cake = np.asarray(cake_intensity, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    q0, q1 = _validate_q_band(q_band, name=name)

    if cake.ndim != 2:
        raise ValueError("cake_intensity must be two-dimensional.")
    if q_arr.ndim != 1 or cake.shape[1] != q_arr.size:
        raise ValueError("q must be one-dimensional and match cake_intensity columns.")

    m = (q_arr >= q0) & (q_arr <= q1) & np.isfinite(q_arr)
    n_points = int(np.count_nonzero(m))
    if n_points < 1:
        raise ValueError(
            f"{name} q band {q0:g} to {q1:g} Å^-1 contains no q bins in the cake."
        )

    values = cake[:, m]
    if n_points == 1:
        finite_q = q_arr[np.isfinite(q_arr)]
        if finite_q.size > 1:
            dq = float(np.nanmedian(np.diff(finite_q)))
        else:
            dq = float(q1 - q0)
        profile = values[:, 0] * max(abs(dq), 1e-12)
    else:
        profile = np.trapezoid(values, q_arr[m], axis=1)

    return np.asarray(profile, dtype=float), float(q1 - q0), n_points


def _background_profiles_from_mode(
    cake_intensity: np.ndarray,
    q: np.ndarray,
    *,
    signal_band: Tuple[float, float],
    signal_width: float,
    bg_mode: str,
    bg_q_range: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Build raw and scaled background profiles from a selected background mode."""
    mode = str(bg_mode or "left").strip().lower()
    if mode in ("before", "low", "lower"):
        mode = "left"
    if mode in ("after", "high", "higher"):
        mode = "right"
    if mode in ("avg", "both"):
        mode = "average"

    q0, q1 = signal_band
    width = float(signal_width)

    def one_band(label: str, band: Tuple[float, float]) -> Tuple[np.ndarray, float, int, Tuple[float, float]]:
        profile, bg_width, n_points = _integrate_cake_q_band(
            cake_intensity,
            q,
            band,
            name=label,
        )
        return profile, bg_width, n_points, _validate_q_band(band, name=label)

    if mode == "left":
        bg_raw, bg_width, n_points, used_range = one_band(
            "left background",
            (q0 - width, q0),
        )
    elif mode == "right":
        bg_raw, bg_width, n_points, used_range = one_band(
            "right background",
            (q1, q1 + width),
        )
    elif mode == "manual":
        if bg_q_range is None:
            raise ValueError("bg_q_range is required when bg_mode='manual'.")
        bg_raw, bg_width, n_points, used_range = one_band(
            "manual background",
            bg_q_range,
        )
    elif mode == "average":
        left_raw, left_width, left_n, left_range = one_band(
            "left background",
            (q0 - width, q0),
        )
        right_raw, right_width, right_n, right_range = one_band(
            "right background",
            (q1, q1 + width),
        )
        left_scaled = left_raw * (width / left_width)
        right_scaled = right_raw * (width / right_width)
        bg_scaled = 0.5 * (left_scaled + right_scaled)
        bg_raw = 0.5 * (left_raw + right_raw)
        return bg_raw, bg_scaled, dict(
            bg_mode=mode,
            bg_q_range=(float(left_range[0]), float(left_range[1]), float(right_range[0]), float(right_range[1])),
            bg_width=float(width),
            bg_scale=float(1.0),
            bg_points=int(left_n + right_n),
        )
    else:
        raise ValueError("bg_mode must be one of 'left', 'right', 'average', or 'manual'.")

    scale = float(width / bg_width)
    return bg_raw, bg_raw * scale, dict(
        bg_mode=mode,
        bg_q_range=(float(used_range[0]), float(used_range[1])),
        bg_width=float(bg_width),
        bg_scale=scale,
        bg_points=int(n_points),
    )


def _normalize_center_halfwidth_windows(
    phi_windows: Optional[Sequence[Tuple[Union[int, float], Union[int, float]]]]
) -> List[Tuple[float, float]]:
    """Normalize phi windows encoded as ``(center, half_width)`` pairs."""
    if phi_windows is None:
        return []

    out: List[Tuple[float, float]] = []
    for item in phi_windows:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each phi window must be a (center, half_width) pair.")
        center = float(item[0])
        half_width = float(item[1])
        if (not np.isfinite(center)) or (not np.isfinite(half_width)) or half_width <= 0.0:
            raise ValueError("Phi window center must be finite and half_width must be positive.")
        out.append((center, half_width))
    return out


def _phi_window_ranges(
    *,
    center: float,
    half_width: float,
    mirror_mode: str,
) -> List[Tuple[str, float, float]]:
    """Expand a center/half-width selector into one or more phi ranges."""
    mode = str(mirror_mode or "none").strip().lower()
    if mode not in ("none", "separate", "together"):
        raise ValueError("mirror_mode must be 'none', 'separate', or 'together'.")

    c = float(center)
    hw = float(half_width)
    if mode == "none":
        return [(f"{c:g} +/- {hw:g}", c - hw, c + hw)]

    c_abs = abs(c)
    pos = (c_abs - hw, c_abs + hw)
    neg = (-c_abs - hw, -c_abs + hw)
    if np.isclose(c_abs, 0.0):
        return [(f"0 +/- {hw:g}", -hw, hw)]
    if mode == "separate":
        return [
            (f"+{c_abs:g} +/- {hw:g}", pos[0], pos[1]),
            (f"-{c_abs:g} +/- {hw:g}", neg[0], neg[1]),
        ]
    return [(f"+/-{c_abs:g} +/- {hw:g}", min(neg[0], pos[0]), max(neg[1], pos[1]))]


def _summarize_phi_windows(
    profile_df: pd.DataFrame,
    *,
    phi_windows: Optional[Sequence[Tuple[Union[int, float], Union[int, float]]]],
    mirror_mode: str,
) -> pd.DataFrame:
    """Quantify corrected intensity fractions inside selected phi windows."""
    selectors = _normalize_center_halfwidth_windows(phi_windows)
    if not selectors:
        return pd.DataFrame(
            columns=[
                "label",
                "center_deg",
                "half_width_deg",
                "mirror_mode",
                "phi0_deg",
                "phi1_deg",
                "phi_ranges",
                "n_points",
                "sample_intensity_sum",
                "fraction",
                "percent",
            ]
        )

    phi = profile_df["azimuth_deg"].to_numpy(dtype=float)
    corrected = profile_df["sample_intensity"].to_numpy(dtype=float)
    total = float(np.nansum(corrected))
    rows: List[Dict[str, object]] = []

    for center, half_width in selectors:
        ranges = _phi_window_ranges(
            center=center,
            half_width=half_width,
            mirror_mode=mirror_mode,
        )

        if str(mirror_mode).strip().lower() == "together" and len(ranges) == 1 and not np.isclose(center, 0.0):
            c_abs = abs(float(center))
            masks = [
                (phi >= (c_abs - half_width)) & (phi <= (c_abs + half_width)),
                (phi >= (-c_abs - half_width)) & (phi <= (-c_abs + half_width)),
            ]
            mask = masks[0] | masks[1]
            label, phi0, phi1 = ranges[0]
            value = float(np.nansum(corrected[mask]))
            frac = value / total if np.isfinite(total) and not np.isclose(total, 0.0) else np.nan
            rows.append(
                dict(
                    label=label,
                    center_deg=float(center),
                    half_width_deg=float(half_width),
                    mirror_mode=str(mirror_mode),
                    phi0_deg=float(phi0),
                    phi1_deg=float(phi1),
                    phi_ranges=str(
                        [
                            (float(-c_abs - half_width), float(-c_abs + half_width)),
                            (float(c_abs - half_width), float(c_abs + half_width)),
                        ]
                    ),
                    n_points=int(np.count_nonzero(mask)),
                    sample_intensity_sum=value,
                    fraction=float(frac),
                    percent=float(100.0 * frac) if np.isfinite(frac) else np.nan,
                )
            )
            continue

        for label, phi0, phi1 in ranges:
            lo, hi = min(phi0, phi1), max(phi0, phi1)
            mask = (phi >= lo) & (phi <= hi)
            value = float(np.nansum(corrected[mask]))
            frac = value / total if np.isfinite(total) and not np.isclose(total, 0.0) else np.nan
            rows.append(
                dict(
                    label=label,
                    center_deg=float(center),
                    half_width_deg=float(half_width),
                    mirror_mode=str(mirror_mode),
                    phi0_deg=float(lo),
                    phi1_deg=float(hi),
                    phi_ranges=str([(float(lo), float(hi))]),
                    n_points=int(np.count_nonzero(mask)),
                    sample_intensity_sum=value,
                    fraction=float(frac),
                    percent=float(100.0 * frac) if np.isfinite(frac) else np.nan,
                )
            )

    return pd.DataFrame(rows)


def compute_xy_files(
    sample_name,
    scan,
    temperature_K,
    *,
    azimuthal_ranges: Sequence[Union[int, float]] = DEFAULT_AZIMUTHAL_RANGES,
    include_full: bool = False,
    full_range: Tuple[Union[int, float], Union[int, float]] = (-90, 90),
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    overwrite_xy: bool = False,
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """Integrate a calibration image over full or segmented azimuthal windows.

    The resulting two-column XY files are stored in the calibration analysis
    directory. Existing files are reused unless ``overwrite_xy`` is true.
    ``scan`` accepts a scan number, scan-tag string, or combined scan sequence.

    Parameters
    ----------
    sample_name : str
        Sample identifier used in standardized dark/calibration paths.
    scan : int, str, or sequence of int
        Dark scan, existing dark tag, or combined scan group.
    temperature_K : int
        Sample temperature in kelvin.
    azimuthal_ranges : sequence of float
        Ordered azimuthal edges defining adjacent integration sectors.
    include_full : bool
        Also integrate ``full_range`` as an additional pattern.
    full_range : tuple of float
        Package-coordinate azimuthal limits for the optional full pattern.
    npt : int
        Number of radial q bins.
    normalize : bool
        Normalize each pattern over ``q_norm_range``.
    q_norm_range : tuple of float
        q interval in Å⁻¹ used for intensity normalization.
    overwrite_xy : bool
        Recompute existing XY cache files.
    poni_path, mask_edf_path : path-like or None
        Explicit pyFAI geometry and detector-mask files; defaults are discovered.
    azim_offset_deg : float
        Package-to-pyFAI azimuthal coordinate offset in degrees.
    polarization_factor : float or None
        Optional pyFAI polarization correction in ``[-1, 1]``.
    paths : AnalysisPaths or None
        Preferred path configuration.
    path_root, analysis_subdir : path-like or None
        Legacy path arguments used when ``paths`` is omitted.

    Returns
    -------
    dict
        Mapping from azimuthal tags to ``(q, intensity)`` arrays.
    """
    ctx = _make_context(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )
    return ctx.compute_xy_files(
        scan,
        azimuthal_ranges=np.asarray(azimuthal_ranges, dtype=float),
        include_full=bool(include_full),
        full_range=tuple(float(v) for v in full_range),
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=tuple(float(v) for v in q_norm_range),
        overwrite_xy=bool(overwrite_xy),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azim_offset_deg=float(azim_offset_deg),
        polarization_factor=polarization_factor,
    )


def plot_detector_and_cake(
    sample_name,
    scan,
    temperature_K,
    *,
    npt_rad: int = 1000,
    npt_azim: int = 360,
    radial_range: Optional[Tuple[float, float]] = None,
    azimuthal_range: Tuple[float, float] = (-90.0, 90.0),
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    use_mask: bool = True,
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    detector_clim: Optional[Tuple[float, float]] = None,
    cake_clim: Optional[Tuple[float, float]] = None,
    detector_log_scale: bool = False,
    cake_log_scale: bool = False,
    invert_detector_x: bool = False,
    invert_detector_y: bool = False,
    figure_title: Optional[str] = None,
    save: bool = False,
    figures_subdir: str = FIGURES_SUBDIR_DEFAULT,
    save_format: str = "png",
    save_dpi: int = 400,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """Plot a bare calibration detector image beside its pyFAI 2D cake.

    The detector panel uses pixel coordinates. The cake panel uses q in Å⁻¹
    and the package's display-coordinate azimuth convention. When
    ``normalize`` is true, each azimuthal row is normalized independently over
    ``q_norm_range``. ``use_mask`` controls whether the resolved EDF mask is
    passed to pyFAI. Detector x and y directions can be flipped independently.

    Parameters
    ----------
    sample_name, scan, temperature_K
        Calibration sample identifier, dark scan specification, and temperature.
    npt_rad, npt_azim : int
        Number of radial and azimuthal bins in the cake.
    radial_range : tuple of float or None
        Optional displayed q limits in Å⁻¹.
    azimuthal_range : tuple of float
        Package-coordinate azimuthal limits in degrees.
    normalize : bool
        Normalize each cake row over ``q_norm_range``.
    q_norm_range : tuple of float
        q interval used for row-wise normalization.
    use_mask : bool
        Apply the resolved EDF detector mask when true.
    poni_path, mask_edf_path : path-like or None
        Explicit pyFAI geometry and optional detector mask.
    azim_offset_deg, polarization_factor : float or None
        Azimuth-coordinate offset and optional polarization correction.
    detector_clim, cake_clim : tuple of float or None
        Independent color limits for the detector and cake panels.
    detector_log_scale, cake_log_scale : bool
        Use logarithmic color normalization for the corresponding panel.
    invert_detector_x, invert_detector_y : bool
        Flip only the rendered detector axes; integration arrays remain unchanged.
    figure_title : str or None
        Optional figure-level title.
    save : bool
        Save the combined figure when true.
    figures_subdir, save_format, save_dpi
        Output subdirectory, file format, and raster resolution.
    paths, path_root, analysis_subdir
        Modern or legacy path configuration.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, np.ndarray, dict]
        Figure, two plotting axes, and the detector/cake arrays and coordinates.
    """
    ctx = _make_context(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )
    detector_image, cake_intensity, q, azimuth = ctx.compute_2d_cake(
        scan,
        npt_rad=int(npt_rad),
        npt_azim=int(npt_azim),
        radial_range=(
            None
            if radial_range is None
            else tuple(float(v) for v in radial_range)
        ),
        azimuthal_range=tuple(float(v) for v in azimuthal_range),
        normalize=bool(normalize),
        q_norm_range=tuple(float(v) for v in q_norm_range),
        use_mask=bool(use_mask),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azim_offset_deg=float(azim_offset_deg),
        polarization_factor=polarization_factor,
    )

    tag = _spec_label(scan)
    base_dir = ctx.analysis_dir(scan)
    save_name = f"{sample_name}_{temperature_K}K_detector_and_2D_cake_{tag}"
    save_kw = calibration_utils._save_kwargs(
        save=bool(save),
        base_dir=base_dir,
        figures_subdir=str(figures_subdir),
        save_name=str(save_name),
        save_format=str(save_format),
        save_dpi=int(save_dpi),
    )

    if figure_title is None:
        figure_title = f"{sample_name}, {temperature_K}K, {tag}"

    fig, axes = plot_utils.DetectorCakePlotter().plot(
        detector_image,
        cake_intensity,
        q,
        azimuth,
        detector_clim=detector_clim,
        cake_clim=cake_clim,
        detector_log_scale=bool(detector_log_scale),
        cake_log_scale=bool(cake_log_scale),
        invert_detector_x=bool(invert_detector_x),
        invert_detector_y=bool(invert_detector_y),
        title=figure_title,
        **save_kw,
    )
    data = {
        "detector_image": detector_image,
        "cake_intensity": cake_intensity,
        "q": q,
        "azimuth": azimuth,
    }
    return fig, axes, data


def plot_cake_azimuthal_windows(
    sample_name,
    scan,
    temperature_K,
    *,
    azimuthal_edges: Sequence[Union[int, float]] = (-90, -45, 0, 45, 90),
    include_full: bool = True,
    full_range: Tuple[Union[int, float], Union[int, float]] = (-90, 90),
    npt_rad: int = 1000,
    npt_azim: int = 360,
    radial_range: Optional[Tuple[float, float]] = None,
    azimuthal_range: Tuple[float, float] = (-90.0, 90.0),
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    use_mask: bool = True,
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    cake_clim: Optional[Tuple[float, float]] = None,
    cake_log_scale: bool = False,
    cmap: str = "viridis",
    window_cmap: str = "tab20",
    window_alpha: float = 0.12,
    show_side_bar: bool = True,
    figure_title: Optional[str] = None,
    save: bool = False,
    figures_subdir: str = FIGURES_SUBDIR_DEFAULT,
    save_format: str = "png",
    save_dpi: int = 400,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """Plot a calibration cake with segmented azimuthal windows overlaid.

    This diagnostic uses the same detector geometry and pyFAI integration path
    as :func:`plot_detector_and_cake`, then overlays adjacent azimuthal sectors
    defined by ``azimuthal_edges``. The optional side bar labels each sector in
    the same order used by calibration integration and fitting workflows.

    Parameters
    ----------
    sample_name, scan, temperature_K
        Calibration sample identifier, dark scan specification, and
        temperature.
    azimuthal_edges : sequence of float
        Ordered package-coordinate azimuthal edges defining adjacent windows.
    include_full, full_range
        Whether the full-range reference window should be shown as black
        boundary lines.
    npt_rad, npt_azim, radial_range, azimuthal_range
        Cake binning and displayed q/azimuth limits.
    normalize, q_norm_range, use_mask
        Cake normalization and detector-mask controls.
    poni_path, mask_edf_path, azim_offset_deg, polarization_factor
        pyFAI geometry, mask, coordinate, and correction settings.
    cake_clim, cake_log_scale, cmap
        Cake color limits, logarithmic display, and colormap.
    window_cmap, window_alpha, show_side_bar
        Overlay colormap, transparency, and side-bar display controls.
    figure_title
        Optional title for the cake panel.
    save, figures_subdir, save_format, save_dpi
        Figure-output controls.
    paths, path_root, analysis_subdir
        Modern or legacy path configuration.

    Returns
    -------
    tuple
        Figure, axes, and a data dictionary containing the detector image,
        cake array, q grid, azimuth grid, normalized edges, and window list.
    """
    edges = np.asarray(azimuthal_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("azimuthal_edges must be a one-dimensional sequence with at least two entries.")
    if not np.all(np.isfinite(edges)):
        raise ValueError("azimuthal_edges must contain only finite values.")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("azimuthal_edges must be strictly increasing.")

    windows = [(float(edges[i]), float(edges[i + 1])) for i in range(edges.size - 1)]
    full_range_f = (float(full_range[0]), float(full_range[1]))

    ctx = _make_context(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )
    detector_image, cake_intensity, q, azimuth = ctx.compute_2d_cake(
        scan,
        npt_rad=int(npt_rad),
        npt_azim=int(npt_azim),
        radial_range=(
            None
            if radial_range is None
            else tuple(float(v) for v in radial_range)
        ),
        azimuthal_range=tuple(float(v) for v in azimuthal_range),
        normalize=bool(normalize),
        q_norm_range=tuple(float(v) for v in q_norm_range),
        use_mask=bool(use_mask),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azim_offset_deg=float(azim_offset_deg),
        polarization_factor=polarization_factor,
    )

    q_arr = np.asarray(q, dtype=float)
    az_arr = np.asarray(azimuth, dtype=float)
    cake = np.asarray(cake_intensity, dtype=float)
    cake_plot = np.where(np.isfinite(cake), cake, np.nan)

    if cake.ndim != 2:
        raise ValueError("cake_intensity must be two-dimensional.")
    if q_arr.ndim != 1 or cake.shape[1] != q_arr.size:
        raise ValueError("q must be one-dimensional and match cake columns.")
    if az_arr.ndim != 1 or cake.shape[0] != az_arr.size:
        raise ValueError("azimuth must be one-dimensional and match cake rows.")

    if show_side_bar:
        fig, (ax, ax_side) = plt.subplots(
            1,
            2,
            figsize=(9.4, 4.2),
            gridspec_kw={"width_ratios": [4.2, 1.25]},
            constrained_layout=True,
        )
        axes = np.asarray([ax, ax_side], dtype=object)
    else:
        fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
        ax_side = None
        axes = np.asarray([ax], dtype=object)

    norm = None
    if cake_clim is not None:
        vmin, vmax = float(cake_clim[0]), float(cake_clim[1])
    else:
        vmin = vmax = None
    if cake_log_scale:
        finite_positive = cake_plot[np.isfinite(cake_plot) & (cake_plot > 0.0)]
        if finite_positive.size == 0:
            raise ValueError("cake_log_scale=True requires positive cake values.")
        log_vmin = float(vmin) if vmin is not None and vmin > 0 else float(np.nanmin(finite_positive))
        log_vmax = float(vmax) if vmax is not None and vmax > 0 else float(np.nanmax(finite_positive))
        norm = mcolors.LogNorm(vmin=log_vmin, vmax=log_vmax)
        vmin = vmax = None

    im = ax.imshow(
        cake_plot,
        extent=(
            float(np.nanmin(q_arr)),
            float(np.nanmax(q_arr)),
            float(np.nanmin(az_arr)),
            float(np.nanmax(az_arr)),
        ),
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        cmap=str(cmap),
    )

    colors = plt.get_cmap(str(window_cmap), max(len(windows), 1))
    for i, (phi0, phi1) in enumerate(windows):
        color = colors(i)
        ax.axhspan(phi0, phi1, color=color, alpha=float(window_alpha))
        ax.axhline(phi0, color="white", linewidth=0.6, alpha=0.85)
        if ax_side is not None:
            ax_side.axhspan(phi0, phi1, color=color, alpha=0.45)
            ax_side.text(
                0.5,
                0.5 * (phi0 + phi1),
                f"{phi0:g} to {phi1:g}",
                ha="center",
                va="center",
                fontsize=8,
            )
    ax.axhline(float(edges[-1]), color="white", linewidth=0.6, alpha=0.85)

    if include_full:
        ax.axhline(full_range_f[0], color="black", linewidth=1.2)
        ax.axhline(full_range_f[1], color="black", linewidth=1.2)

    ax.set_xlabel(r"q [$\mathrm{\AA}^{-1}$]")
    ax.set_ylabel("Azimuth [deg]")
    ax.set_ylim(full_range_f if include_full else tuple(float(v) for v in azimuthal_range))
    ax.set_title(
        "Cake with segmented azimuthal windows"
        if figure_title is None
        else str(figure_title)
    )
    fig.colorbar(im, ax=ax, label="Intensity [a.u.]", pad=0.02)

    if ax_side is not None:
        ax_side.set_ylim(ax.get_ylim())
        ax_side.set_xlim(0, 1)
        ax_side.set_xticks([])
        ax_side.set_yticks(edges)
        for spine in ("top", "right", "bottom"):
            ax_side.spines[spine].set_visible(False)

    tag = _spec_label(scan)
    base_dir = ctx.analysis_dir(scan)
    save_name = f"{sample_name}_{temperature_K}K_cake_azimuthal_windows_{tag}"
    save_kw = calibration_utils._save_kwargs(
        save=bool(save),
        base_dir=base_dir,
        figures_subdir=str(figures_subdir),
        save_name=str(save_name),
        save_format=str(save_format),
        save_dpi=int(save_dpi),
    )
    if save_kw["save"]:
        plot_utils.save_figure(
            fig,
            save_dir=save_kw["save_dir"],
            save_name=save_kw["save_name"],
            fmt=save_kw["save_format"],
            dpi=save_kw["save_dpi"],
            overwrite=save_kw["save_overwrite"],
        )

    data = {
        "detector_image": detector_image,
        "cake_intensity": cake_intensity,
        "q": q,
        "azimuth": azimuth,
        "azimuthal_edges": edges,
        "windows": windows,
        "full_range": full_range_f,
    }
    return fig, axes, data


def analyze_cake_azimuthal_distribution(
    sample_name,
    scan,
    temperature_K,
    *,
    q_value: float,
    q_width: float,
    bg_mode: str = "left",
    bg_q_range: Optional[Tuple[float, float]] = None,
    phi_windows: Optional[Sequence[Tuple[Union[int, float], Union[int, float]]]] = None,
    mirror_mode: str = "none",
    npt_rad: int = 1000,
    npt_azim: int = 360,
    radial_range: Optional[Tuple[float, float]] = None,
    azimuthal_range: Tuple[float, float] = (-90.0, 90.0),
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    use_mask: bool = True,
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    profile_ylim: Optional[Tuple[float, float]] = None,
    fraction_ylim: Optional[Tuple[float, float]] = None,
    fraction_as_percent: bool = True,
    figure_title: Optional[str] = None,
    make_plots: bool = True,
    save: bool = False,
    figures_subdir: str = FIGURES_SUBDIR_DEFAULT,
    save_format: str = "png",
    save_dpi: int = 400,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    """Analyze azimuthal intensity distribution from a calibration 2D cake.

    A signal q band centered at ``q_value`` with full width ``q_width`` is
    integrated over q for every azimuthal bin. A background band is integrated
    in the same way, scaled to the signal-band width, and subtracted. The
    corrected profile is normalized to produce a fractional or percentage
    azimuthal distribution. Optional phi-window selectors quantify how much
    corrected intensity lies in user-defined angular sectors, with optional
    mirror treatment.

    Parameters
    ----------
    q_value, q_width
        Signal-band center and full width in Å⁻¹.
    bg_mode
        ``"left"``, ``"right"``, ``"average"``, or ``"manual"``.
    bg_q_range
        Manual background q range used when ``bg_mode="manual"``.
    phi_windows
        Optional sequence of ``(center_deg, half_width_deg)`` selectors.
    mirror_mode
        ``"none"``, ``"separate"``, or ``"together"`` for ``phi_windows``.
    fraction_as_percent
        Plot normalized corrected intensity as percent when true, fraction
        when false. Both columns are always returned in the profile table.
    make_plots
        Create/show Matplotlib figures when true. GUI background workers should
        pass false and create figures later on the Qt main thread.

    Returns
    -------
    dict
        ``profile_df``, ``summary_df``, figure handles, and cake metadata.
    """
    ctx = _make_context(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    detector_image, cake_intensity, q, azimuth = ctx.compute_2d_cake(
        scan,
        npt_rad=int(npt_rad),
        npt_azim=int(npt_azim),
        radial_range=(
            None
            if radial_range is None
            else tuple(float(v) for v in radial_range)
        ),
        azimuthal_range=tuple(float(v) for v in azimuthal_range),
        normalize=bool(normalize),
        q_norm_range=tuple(float(v) for v in q_norm_range),
        use_mask=bool(use_mask),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azim_offset_deg=float(azim_offset_deg),
        polarization_factor=polarization_factor,
    )

    signal_band = _band_from_center_width(
        center=float(q_value),
        width=float(q_width),
        name="signal q band",
    )
    full_profile, signal_width, signal_points = _integrate_cake_q_band(
        cake_intensity,
        q,
        signal_band,
        name="signal",
    )
    bg_raw, bg_scaled, bg_info = _background_profiles_from_mode(
        cake_intensity,
        q,
        signal_band=signal_band,
        signal_width=signal_width,
        bg_mode=bg_mode,
        bg_q_range=(
            None
            if bg_q_range is None
            else _validate_q_band(bg_q_range, name="bg_q_range")
        ),
    )

    corrected = np.asarray(full_profile, dtype=float) - np.asarray(bg_scaled, dtype=float)
    total = float(np.nansum(corrected))
    if np.isfinite(total) and not np.isclose(total, 0.0):
        fraction = corrected / total
    else:
        fraction = np.full_like(corrected, np.nan, dtype=float)

    profile_df = pd.DataFrame(
        dict(
            azimuth_deg=np.asarray(azimuth, dtype=float),
            full_intensity=np.asarray(full_profile, dtype=float),
            background_raw=np.asarray(bg_raw, dtype=float),
            background_scaled=np.asarray(bg_scaled, dtype=float),
            sample_intensity=np.asarray(corrected, dtype=float),
            fraction=np.asarray(fraction, dtype=float),
            percent=100.0 * np.asarray(fraction, dtype=float),
            q_signal0=float(signal_band[0]),
            q_signal1=float(signal_band[1]),
            q_signal_center=float(q_value),
            q_signal_width=float(signal_width),
            q_signal_points=int(signal_points),
            bg_mode=str(bg_info["bg_mode"]),
            bg_q_range=str(bg_info["bg_q_range"]),
            bg_width=float(bg_info["bg_width"]),
            bg_scale=float(bg_info["bg_scale"]),
            bg_points=int(bg_info["bg_points"]),
        )
    )
    summary_df = _summarize_phi_windows(
        profile_df,
        phi_windows=phi_windows,
        mirror_mode=mirror_mode,
    )

    tag = _spec_label(scan)
    base_dir = ctx.analysis_dir(scan)
    save_base = f"{sample_name}_{temperature_K}K_cake_phi_distribution_{tag}_q{float(q_value):.4g}"
    title = (
        str(figure_title)
        if figure_title is not None
        else f"{sample_name}, {temperature_K}K, {tag}\nq={float(q_value):g} +/- {0.5 * float(q_width):g} Å$^{{-1}}$"
    )

    fig_profile = ax_profile = fig_fraction = ax_fraction = None
    if bool(make_plots):
        plot_result = plot_cake_azimuthal_distribution_profiles(
            profile_df,
            summary_df=summary_df,
            title=title,
            profile_ylim=profile_ylim,
            fraction_ylim=fraction_ylim,
            fraction_as_percent=bool(fraction_as_percent),
            save=bool(save),
            base_dir=base_dir,
            figures_subdir=str(figures_subdir),
            save_name_prefix=save_base,
            save_format=str(save_format),
            save_dpi=int(save_dpi),
        )
        fig_profile = plot_result["fig_profile"]
        ax_profile = plot_result["ax_profile"]
        fig_fraction = plot_result["fig_fraction"]
        ax_fraction = plot_result["ax_fraction"]

    profile_csv_path = None
    summary_csv_path = None
    if bool(save):
        out_dir = plot_utils.figures_dir(base_dir, figures_subdir=str(figures_subdir))
        out_dir.mkdir(parents=True, exist_ok=True)
        profile_csv_path = out_dir / f"{plot_utils._sanitize_stem(save_base)}_profiles.csv"
        summary_csv_path = out_dir / f"{plot_utils._sanitize_stem(save_base)}_summary.csv"
        profile_df.to_csv(str(profile_csv_path), index=False)
        summary_df.to_csv(str(summary_csv_path), index=False)

    return dict(
        profile_df=profile_df,
        summary_df=summary_df,
        fig_profile=fig_profile,
        ax_profile=ax_profile,
        fig_fraction=fig_fraction,
        ax_fraction=ax_fraction,
        profile_csv_path=profile_csv_path,
        summary_csv_path=summary_csv_path,
        figure_title=title,
        base_dir=base_dir,
        save_base=save_base,
        detector_image=detector_image,
        cake_intensity=cake_intensity,
        q=q,
        azimuth=azimuth,
        signal_band=signal_band,
        background_info=bg_info,
    )


def _shade_phi_windows(ax, summary_df: pd.DataFrame) -> None:
    """Add selected phi-window spans to an axes."""
    if summary_df is None or summary_df.empty:
        return

    for i, row in summary_df.reset_index(drop=True).iterrows():
        label = str(row.get("label", "phi window"))
        alpha = 0.10 if i % 2 == 0 else 0.16
        ranges = None
        try:
            import ast
            ranges = ast.literal_eval(str(row.get("phi_ranges", "")))
        except Exception:
            ranges = None
        if not ranges:
            ranges = [(float(row["phi0_deg"]), float(row["phi1_deg"]))]
        for j, item in enumerate(ranges):
            ax.axvspan(
                float(item[0]),
                float(item[1]),
                color="tab:blue",
                alpha=alpha,
                label=label if i == 0 and j == 0 else None,
            )


def plot_cake_azimuthal_distribution_profiles(
    profile_df: pd.DataFrame,
    *,
    summary_df: Optional[pd.DataFrame] = None,
    title: Optional[str] = None,
    profile_ylim: Optional[Tuple[float, float]] = None,
    fraction_ylim: Optional[Tuple[float, float]] = None,
    fraction_as_percent: bool = True,
    save: bool = False,
    base_dir: Optional[Union[str, Path]] = None,
    figures_subdir: str = FIGURES_SUBDIR_DEFAULT,
    save_name_prefix: str = "cake_phi_distribution",
    save_format: str = "png",
    save_dpi: int = 400,
) -> Dict[str, object]:
    """Plot already-computed calibration cake azimuthal distribution tables.

    This is separated from :func:`analyze_cake_azimuthal_distribution` so GUI
    callers can run the pyFAI/cake computation in a worker thread, then create
    Matplotlib windows on the Qt main thread.
    """
    save_kw_profile = calibration_utils._save_kwargs(
        save=bool(save),
        base_dir=base_dir,
        figures_subdir=str(figures_subdir),
        save_name=f"{save_name_prefix}_profiles",
        save_format=str(save_format),
        save_dpi=int(save_dpi),
    )
    save_kw_fraction = calibration_utils._save_kwargs(
        save=bool(save),
        base_dir=base_dir,
        figures_subdir=str(figures_subdir),
        save_name=f"{save_name_prefix}_fraction",
        save_format=str(save_format),
        save_dpi=int(save_dpi),
    )

    fig_profile, ax_profile = _plot_cake_phi_profiles(
        profile_df,
        summary_df=summary_df,
        title=title,
        ylim=profile_ylim,
        **save_kw_profile,
    )
    fig_fraction, ax_fraction = _plot_cake_phi_fraction(
        profile_df,
        summary_df=summary_df,
        title=(
            None
            if title is None
            else f"{title}\nbackground-subtracted distribution"
        ),
        as_percent=bool(fraction_as_percent),
        ylim=fraction_ylim,
        **save_kw_fraction,
    )
    return dict(
        fig_profile=fig_profile,
        ax_profile=ax_profile,
        fig_fraction=fig_fraction,
        ax_fraction=ax_fraction,
    )


def _plot_cake_phi_profiles(
    profile_df: pd.DataFrame,
    *,
    summary_df: Optional[pd.DataFrame],
    title: Optional[str],
    ylim: Optional[Tuple[float, float]],
    save: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 400,
    save_overwrite: bool = False,
):
    """Plot full, scaled background, and corrected profiles versus azimuth."""
    plot_utils.DEFAULT_STYLE.apply()
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    phi = profile_df["azimuth_deg"].to_numpy(dtype=float)

    artists = []
    artists.append(ax.plot(phi, profile_df["full_intensity"], "-", linewidth=2, label="Full q band")[0])
    artists.append(ax.plot(phi, profile_df["background_scaled"], "-", linewidth=2, label="Scaled background")[0])
    artists.append(ax.plot(phi, profile_df["sample_intensity"], "-o", markersize=3, linewidth=1.5, color="black", label="Sample response")[0])
    _shade_phi_windows(ax, summary_df)

    ax.set_xlabel("Azimuth [deg]", fontsize=plot_utils.DEFAULT_STYLE.label_fontsize)
    ax.set_ylabel("Integrated intensity [a.u.]", fontsize=plot_utils.DEFAULT_STYLE.label_fontsize)
    if title is not None:
        ax.set_title(str(title), fontsize=plot_utils.DEFAULT_STYLE.title_fontsize)
    if ylim is not None:
        ax.set_ylim(tuple(float(v) for v in ylim))
    ax.grid(alpha=0.3)
    leg = ax.legend(framealpha=1.0)
    if leg is not None:
        plot_utils._make_legend_clickable(
            ax,
            legend=leg,
            label_to_artists={a.get_label(): [a] for a in artists},
        )
    fig.tight_layout()

    if save:
        if save_dir is None:
            raise ValueError("_plot_cake_phi_profiles(save=True) requires save_dir=...")
        plot_utils.save_figure(
            fig,
            save_dir=save_dir,
            save_name=save_name or "cake_phi_profiles",
            fmt=save_format,
            dpi=save_dpi,
            overwrite=save_overwrite,
        )
    plt.show()
    return fig, ax


def _plot_cake_phi_fraction(
    profile_df: pd.DataFrame,
    *,
    summary_df: Optional[pd.DataFrame],
    title: Optional[str],
    as_percent: bool,
    ylim: Optional[Tuple[float, float]],
    save: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 400,
    save_overwrite: bool = False,
):
    """Plot normalized corrected intensity versus azimuth."""
    plot_utils.DEFAULT_STYLE.apply()
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    phi = profile_df["azimuth_deg"].to_numpy(dtype=float)
    col = "percent" if bool(as_percent) else "fraction"
    label = "Corrected intensity [%]" if bool(as_percent) else "Corrected intensity fraction"
    y = profile_df[col].to_numpy(dtype=float)

    line = ax.plot(phi, y, "-o", markersize=3, linewidth=1.5, color="black", label=label)[0]
    _shade_phi_windows(ax, summary_df)
    if summary_df is not None and not summary_df.empty:
        text = "\n".join(
            f"{row.label}: {row.percent:.2f}%"
            for row in summary_df.itertuples(index=False)
            if np.isfinite(float(row.percent))
        )
        if text:
            ax.text(
                0.02,
                0.98,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
            )

    ax.set_xlabel("Azimuth [deg]", fontsize=plot_utils.DEFAULT_STYLE.label_fontsize)
    ax.set_ylabel(label, fontsize=plot_utils.DEFAULT_STYLE.label_fontsize)
    if title is not None:
        ax.set_title(str(title), fontsize=plot_utils.DEFAULT_STYLE.title_fontsize)
    if ylim is not None:
        ax.set_ylim(tuple(float(v) for v in ylim))
    ax.grid(alpha=0.3)
    leg = ax.legend(framealpha=1.0)
    if leg is not None:
        plot_utils._make_legend_clickable(ax, legend=leg, label_to_artists={label: [line]})
    fig.tight_layout()

    if save:
        if save_dir is None:
            raise ValueError("_plot_cake_phi_fraction(save=True) requires save_dir=...")
        plot_utils.save_figure(
            fig,
            save_dir=save_dir,
            save_name=save_name or "cake_phi_fraction",
            fmt=save_format,
            dpi=save_dpi,
            overwrite=save_overwrite,
        )
    plt.show()
    return fig, ax


def do_peak_fitting(
    sample_name,
    scan,
    temperature_K,
    *,
    q_fit_range: Tuple[float, float] = (2.4, 2.65),
    azimuthal_ranges: Sequence[Union[int, float]] = DEFAULT_AZIMUTHAL_RANGES,
    include_full: bool = False,
    full_range: Tuple[Union[int, float], Union[int, float]] = (-90, 90),
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    eta: float = 0.3,
    eta_vary: bool = False,
    fit_method: str = "leastsq",
    force_refit: bool = True,
    out_csv_name: str = "peak_fits.csv",
    overwrite_xy: bool = False,
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """Fit a linear background and pseudo-Voigt peak in each azimuthal window.

    Integration files are created as needed. Fit parameters and quality metrics
    are written to ``out_csv_name`` below the calibration directory and returned
    together with the resolved CSV path.

    Parameters
    ----------
    sample_name, scan, temperature_K
        Calibration sample identifier, dark scan specification, and temperature.
    q_fit_range : tuple of float
        q interval in Å⁻¹ fitted in every azimuthal sector.
    azimuthal_ranges, include_full, full_range
        Sector edges and optional full-range integration configuration.
    npt, normalize, q_norm_range
        Radial binning and optional XY intensity normalization settings.
    eta : float
        Initial pseudo-Voigt mixing fraction.
    eta_vary : bool
        Whether the pseudo-Voigt mixing fraction is refined.
    fit_method : str
        Optimization method forwarded to ``lmfit``.
    force_refit : bool
        Ignore reusable rows from an existing result table.
    out_csv_name : str
        Result filename below the dark analysis directory.
    overwrite_xy : bool
        Recompute cached integration patterns before fitting.
    poni_path, mask_edf_path, azim_offset_deg, polarization_factor
        pyFAI geometry, mask, coordinate, and correction settings.
    paths, path_root, analysis_subdir
        Modern or legacy path configuration.

    Returns
    -------
    pandas.DataFrame
        One schema-complete fit row per requested azimuthal window.
    """
    ctx = _make_context(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )
    return ctx.do_peak_fitting(
        scan,
        q_fit_range=tuple(float(v) for v in q_fit_range),
        azimuthal_ranges=np.asarray(azimuthal_ranges, dtype=float),
        include_full=bool(include_full),
        full_range=tuple(float(v) for v in full_range),
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=tuple(float(v) for v in q_norm_range),
        eta=float(eta),
        eta_vary=bool(eta_vary),
        fit_method=str(fit_method),
        force_refit=bool(force_refit),
        out_csv_name=str(out_csv_name),
        overwrite_xy=bool(overwrite_xy),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azim_offset_deg=float(azim_offset_deg),
        polarization_factor=polarization_factor,
    )


def plot_caked_1D_patterns(
    sample_name,
    scan,
    temperature_K,
    *,
    azimuthal_ranges: Sequence[Union[int, float]] = DEFAULT_AZIMUTHAL_RANGES,
    include_full: bool = False,
    full_range: Tuple[Union[int, float], Union[int, float]] = (-90, 90),
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    overwrite_xy: bool = False,
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    xlim=None,
    ylim=None,
    figure_title: Optional[str] = None,
    save: bool = False,
    figures_subdir: str = FIGURES_SUBDIR_DEFAULT,
    save_format: str = "png",
    save_dpi: int = 400,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """Plot calibration patterns integrated over consecutive azimuthal slices.

    Missing XY patterns are integrated before plotting. Saving uses the
    calibration figures directory unless an explicit output path is supplied.

    Parameters
    ----------
    sample_name, scan, temperature_K
        Calibration sample identifier, dark scan specification, and temperature.
    azimuthal_ranges, include_full, full_range
        Sector boundaries and optional additional full-range sector.
    npt, normalize, q_norm_range
        Radial bin count and optional intensity-normalization configuration.
    overwrite_xy : bool
        Recompute existing XY files instead of reusing them.
    poni_path, mask_edf_path, azim_offset_deg, polarization_factor
        pyFAI geometry, detector mask, azimuth convention, and polarization
        correction settings.
    xlim, ylim
        Optional q and intensity display limits.
    figure_title : str, optional
        Figure title; ``None`` leaves the plotter's default title behavior.
    save, figures_subdir, save_format, save_dpi
        Figure-output controls.
    paths, path_root, analysis_subdir
        Modern or legacy path configuration.

    Returns
    -------
    list of tuple
        ``(azimuth_label, q, intensity)`` for every plotted sector.
    """
    ctx = _make_context(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    ctx.compute_xy_files(
        scan,
        azimuthal_ranges=np.asarray(azimuthal_ranges, dtype=float),
        include_full=bool(include_full),
        full_range=tuple(float(v) for v in full_range),
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=tuple(float(v) for v in q_norm_range),
        overwrite_xy=bool(overwrite_xy),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azim_offset_deg=float(azim_offset_deg),
        polarization_factor=polarization_factor,
    )

    windows = _windows_from_edges(
        azimuthal_ranges,
        include_full=bool(include_full),
        full_range=full_range,
    )

    patterns = []
    for w in windows:
        q, intensity, _ = ctx.load_xy(
            scan,
            azim_window=w,
            npt=int(npt),
            normalize=bool(normalize),
            q_norm_range=tuple(float(v) for v in q_norm_range),
            compute_if_missing=True,
            overwrite_xy=bool(overwrite_xy),
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            azim_offset_deg=float(azim_offset_deg),
            polarization_factor=polarization_factor,
        )
        patterns.append((general_utils.azim_range_str(w), q, intensity))

    tag = _spec_label(scan)
    base_dir = ctx.analysis_dir(scan)
    save_name = f"{sample_name}_{temperature_K}K_caked_1D_patterns_{tag}"

    save_kw = calibration_utils._save_kwargs(
        save=bool(save),
        base_dir=base_dir,
        figures_subdir=str(figures_subdir),
        save_name=str(save_name),
        save_format=str(save_format),
        save_dpi=int(save_dpi),
    )

    p = plot_utils.Pattern1DPlotter()
    p.plot_caked_patterns(
        patterns,
        title=figure_title,
        xlim=xlim,
        ylim=ylim,
        figsize=(6, 6),
        legend_ncol=1,
        **save_kw,
    )
    return patterns


def plot_property_vs_azimuth(
    sample_name,
    scan,
    temperature_K,
    *,
    _property: str = "pv_center",
    peak_name: Optional[str] = "",
    figure_title: Optional[str] = None,
    only_success: bool = True,
    out_csv_name: str = "peak_fits.csv",
    ylim=None,
    save: bool = False,
    figures_subdir: str = FIGURES_SUBDIR_DEFAULT,
    save_format: str = "png",
    save_dpi: int = 400,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """Plot one fitted calibration property as a function of azimuth.

    Results are loaded from the calibration fit CSV. ``_property`` identifies
    a column produced by the peak-fitting workflow.

    Parameters
    ----------
    sample_name, scan, temperature_K
        Calibration sample identifier, dark scan specification, and temperature.
    _property : str
        Fit-table column to place on the vertical axis, such as ``pv_center``.
    peak_name : str, optional
        hkl label used for q-center plots. Blank labels the axis as q.
    figure_title : str, optional
        Custom title; when omitted, a title is generated from the dataset.
    only_success : bool
        Exclude rows whose fit-success flag is false.
    out_csv_name : str
        Fit result filename below the dark analysis directory.
    ylim
        Optional vertical-axis limits.
    save, figures_subdir, save_format, save_dpi
        Figure-output controls.
    paths, path_root, analysis_subdir
        Modern or legacy path configuration.

    Returns
    -------
    tuple
        Matplotlib figure and axes returned by :class:`FitCSVPlotter`.
    """
    ctx = _make_context(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    csv_path = str(ctx.peak_fits_csv_path(scan, out_csv_name=str(out_csv_name)))
    tag = _spec_label(scan)
    base_dir = ctx.analysis_dir(scan)

    if figure_title is None:
        figure_title = f"{sample_name}, {temperature_K}K, {tag}\n{_property} vs azimuth".replace(
            "_", ": "
        )

    save_name = f"{sample_name}_{temperature_K}K_{str(_property)}_vs_azimuth_{tag}"

    save_kw = calibration_utils._save_kwargs(
        save=bool(save),
        base_dir=base_dir,
        figures_subdir=str(figures_subdir),
        save_name=str(save_name),
        save_format=str(save_format),
        save_dpi=int(save_dpi),
    )

    p = plot_utils.FitCSVPlotter()
    return p.plot_property_vs_azimuth(
        csv_path,
        y=str(_property),
        peak_name=peak_name,
        only_success=bool(only_success),
        title=figure_title,
        ylim=ylim,
        **save_kw,
    )


def plot_1D_plus_fit(
    sample_name,
    scan,
    temperature_K,
    *,
    azim_window: Tuple[Union[int, float], Union[int, float]] = (-45, 0),
    out_csv_name: str = "peak_fits.csv",
    fit_oversample: int = 10,
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    overwrite_xy: bool = False,
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    figure_title: Optional[str] = None,
    save: bool = False,
    figures_subdir: str = FIGURES_SUBDIR_DEFAULT,
    save_format: str = "png",
    save_dpi: int = 400,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """Overlay a calibration pattern with its stored pseudo-Voigt fit.

    The requested scan and azimuthal window select both the XY file and the CSV
    row. The figure can be displayed, saved, or both.

    Parameters
    ----------
    sample_name, scan, temperature_K
        Calibration sample identifier, dark scan specification, and temperature.
    azim_window : tuple of float
        Lower and upper azimuth bounds in degrees.
    out_csv_name : str
        Fit result filename below the dark analysis directory.
    fit_oversample : int
        Multiplication factor used to draw a smooth fitted curve.
    npt, normalize, q_norm_range, overwrite_xy
        XY integration, normalization, and cache-reuse controls.
    poni_path, mask_edf_path, azim_offset_deg, polarization_factor
        pyFAI geometry, detector mask, azimuth convention, and polarization
        correction settings.
    figure_title : str, optional
        Custom title for the fit overlay.
    save, figures_subdir, save_format, save_dpi
        Figure-output controls.
    paths, path_root, analysis_subdir
        Modern or legacy path configuration.

    Returns
    -------
    tuple
        Matplotlib figure and axes returned by :class:`FitCSVPlotter`.
    """
    ctx = _make_context(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    azim_window_i = _azim_window_int(azim_window)

    q, intensity, _ = ctx.load_xy(
        scan,
        azim_window=azim_window_i,
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=tuple(float(v) for v in q_norm_range),
        compute_if_missing=True,
        overwrite_xy=bool(overwrite_xy),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azim_offset_deg=float(azim_offset_deg),
        polarization_factor=polarization_factor,
    )

    csv_path = str(ctx.peak_fits_csv_path(scan, out_csv_name=str(out_csv_name)))
    azim_str = general_utils.azim_range_str(azim_window_i)

    tag = _spec_label(scan)
    base_dir = ctx.analysis_dir(scan)
    save_name = f"{sample_name}_{temperature_K}K_fit_overlay_{azim_str}_{tag}"

    save_kw = calibration_utils._save_kwargs(
        save=bool(save),
        base_dir=base_dir,
        figures_subdir=str(figures_subdir),
        save_name=str(save_name),
        save_format=str(save_format),
        save_dpi=int(save_dpi),
    )

    p = plot_utils.FitCSVPlotter()
    return p.plot_fit_overlay(
        q=q,
        I=intensity,
        csv_path=csv_path,
        azim_range_str=str(azim_str),
        title=figure_title,
        fit_oversample=int(fit_oversample),
        **save_kw,
    )


def compare_1D_patterns(
    *,
    sample_name,
    scans,
    temperature_K,
    scan_ref,
    azim_window: Tuple[Union[int, float], Union[int, float]] = (-90, 90),
    npt: int = 1000,
    normalize: bool = True,
    q_norm_range: Tuple[float, float] = (2.65, 2.75),
    overwrite_xy: bool = False,
    poni_path: Optional[Union[str, Path]] = None,
    mask_edf_path: Optional[Union[str, Path]] = None,
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    xlim: Tuple[float, float] = (1.5, 4.5),
    ylim_top=None,
    ylim_diff=None,
    figure_title: Optional[str] = None,
    int_list_mode: str = "together",
    save: bool = False,
    figures_subdir: str = FIGURES_SUBDIR_DEFAULT,
    save_format: str = "png",
    save_dpi: int = 400,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
):
    """Compare integrated calibration patterns from one or more dark scans.

    Integer scan lists can represent one combined dataset or separate datasets,
    as selected by ``int_list_mode``. Patterns are calculated on demand and may
    be normalized over ``q_norm_range`` before plotting.

    Parameters
    ----------
    sample_name, temperature_K
        Calibration sample identifier and temperature.
    scans
        Scan specification or collection of specifications to compare.
    scan_ref
        Single scan specification used as the subtraction reference.
    azim_window : tuple of float
        Azimuthal integration bounds in degrees.
    npt, normalize, q_norm_range, overwrite_xy
        XY integration, normalization, and cache-reuse controls.
    poni_path, mask_edf_path, azim_offset_deg, polarization_factor
        pyFAI geometry, detector mask, azimuth convention, and polarization
        correction settings.
    xlim, ylim_top, ylim_diff
        Display limits for the pattern and difference panels.
    figure_title : str, optional
        Custom figure title.
    int_list_mode : {"together", "separate"}
        Interpret a flat integer list as one combined dataset or individual
        datasets.
    save, figures_subdir, save_format, save_dpi
        Figure-output controls.
    paths, path_root, analysis_subdir
        Modern or legacy path configuration.

    Returns
    -------
    tuple
        Reference q array, reference intensity array, and compared pattern
        records as ``(label, q, intensity)`` tuples.

    Raises
    ------
    ValueError
        If ``scan_ref`` resolves to more or fewer than one dataset.
    """
    ctx = _make_context(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    specs = _normalize_scan_specs(scans, mode=str(int_list_mode))
    ref_specs = _normalize_scan_specs(scan_ref, mode="together")
    if len(ref_specs) != 1:
        raise ValueError("scan_ref must resolve to exactly one dataset.")
    ref_spec = ref_specs[0]

    azim_window_i = _azim_window_int(azim_window)

    q_ref, I_ref, _ = ctx.load_xy(
        ref_spec,
        azim_window=azim_window_i,
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=tuple(float(v) for v in q_norm_range),
        compute_if_missing=True,
        overwrite_xy=bool(overwrite_xy),
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azim_offset_deg=float(azim_offset_deg),
        polarization_factor=polarization_factor,
    )

    ref_label = _spec_label(ref_spec)

    patterns = []
    for s in specs:
        if s == ref_spec:
            continue
        q, intensity, _ = ctx.load_xy(
            s,
            azim_window=azim_window_i,
            npt=int(npt),
            normalize=bool(normalize),
            q_norm_range=tuple(float(v) for v in q_norm_range),
            compute_if_missing=True,
            overwrite_xy=bool(overwrite_xy),
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            azim_offset_deg=float(azim_offset_deg),
            polarization_factor=polarization_factor,
        )
        patterns.append((_spec_label(s), q, intensity))

    azim_str = general_utils.azim_range_str(azim_window_i)
    if figure_title is None:
        figure_title = f"{sample_name} {temperature_K}K (dark) azim={azim_str}"

    base_dir = ctx.analysis_dir(ref_spec)
    save_name = f"{sample_name}_{temperature_K}K_compare_ref_{ref_label}_{azim_str}"

    save_kw = calibration_utils._save_kwargs(
        save=bool(save),
        base_dir=base_dir,
        figures_subdir=str(figures_subdir),
        save_name=str(save_name),
        save_format=str(save_format),
        save_dpi=int(save_dpi),
    )

    p = plot_utils.Pattern1DPlotter()
    p.compare_to_reference(
        q_ref=q_ref,
        I_ref=I_ref,
        ref_label=ref_label,
        patterns=patterns,
        title=figure_title,
        xlim=xlim,
        ylim_top=ylim_top,
        ylim_diff=ylim_diff,
        legend_title="Dataset",
        legend_loc="upper left",
        legend_outside=True,
        **save_kw,
    )
    return q_ref, I_ref, patterns


__all__ = [
    "ScanSpec",
    "FIGURES_SUBDIR_DEFAULT",
    "DEFAULT_AZIMUTHAL_RANGES",
    "compute_xy_files",
    "plot_detector_and_cake",
    "plot_cake_azimuthal_windows",
    "analyze_cake_azimuthal_distribution",
    "do_peak_fitting",
    "plot_caked_1D_patterns",
    "plot_property_vs_azimuth",
    "plot_1D_plus_fit",
    "compare_1D_patterns",
]
