from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

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
        Fixed pseudo-Voigt mixing fraction.
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
    "do_peak_fitting",
    "plot_caked_1D_patterns",
    "plot_property_vs_azimuth",
    "plot_1D_plus_fit",
    "compare_1D_patterns",
]
