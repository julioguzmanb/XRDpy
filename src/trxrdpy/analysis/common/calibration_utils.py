from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import fabio
import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
    
import pandas as pd

from . import azimint_utils, general_utils, plot_utils
from .paths import AnalysisPaths

ScanSpec = Union[int, Sequence[int], str]


def _resolve_dark_tag(scan_spec: ScanSpec) -> str:
    """Convert a scan specification to the standardized dark-directory tag."""
    if isinstance(scan_spec, str):
        tag = str(scan_spec).strip()
        if tag == "":
            raise ValueError("scan_spec string cannot be empty.")
        return tag
    return general_utils.scan_tag(scan_spec)


def _coerce_paths(
    *,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> AnalysisPaths:
    """Resolve modern or legacy path arguments to one ``AnalysisPaths`` object."""
    if paths is not None:
        return paths
    if path_root is None:
        raise ValueError(
            "Provide either paths=AnalysisPaths(...) or "
            "path_root=... together with analysis_subdir=...."
        )
    return AnalysisPaths(
        path_root=Path(path_root),
        analysis_subdir=str(analysis_subdir) if analysis_subdir is not None else "analysis",
    )


def _save_kwargs(
    *,
    save: bool,
    base_dir: Union[str, Path],
    figures_subdir: str,
    save_name: str,
    save_format: str,
    save_dpi: int,
) -> dict:
    """Build the common plotting save options for a calibration figure."""
    return plot_utils.build_save_kwargs(
        save=bool(save),
        base_dir=base_dir,
        figures_subdir=str(figures_subdir),
        save_name=str(save_name),
        save_format=str(save_format),
        save_dpi=int(save_dpi),
        overwrite=True,
    )


def _make_single_peak_model():
    """Create a linear-background plus pseudo-Voigt peak model using ``lmfit``."""
    from lmfit.models import PolynomialModel, PseudoVoigtModel

    bg = PolynomialModel(degree=1, prefix="bg_")
    pv = PseudoVoigtModel(prefix="pv_")
    return bg + pv


def _pv_fwhm_from_result(result) -> float:
    """Extract pseudo-Voigt FWHM, deriving it from sigma when unavailable."""
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
        return 2.354820045 * sigma
    except Exception:
        return float("nan")


def _failed_fit_row(
    *,
    azim_window: Tuple[float, float],
    azim_str: str,
    q_fit_range: Tuple[float, float],
    eta: float,
) -> Dict[str, object]:
    """Create a schema-complete result row for an unsuccessful peak fit."""
    return dict(
        success=False,
        azim_center=float(general_utils.azim_center(azim_window)),
        azim_range_str=str(azim_str),
        q_fit0=float(q_fit_range[0]),
        q_fit1=float(q_fit_range[1]),
        bg_c0=np.nan,
        bg_c1=np.nan,
        pv_center=np.nan,
        pv_sigma=np.nan,
        pv_amplitude=np.nan,
        pv_fraction=float(eta),
        pv_height=np.nan,
        pv_fwhm=np.nan,
        r2=np.nan,
    )


@dataclass(frozen=True)
class CalibrationContext:
    """Bind calibration paths and operations to one sample and temperature.

    The context locates dark images, calibration XY caches, PONI/mask defaults,
    and the peak-fit CSV. It can integrate missing azimuthal patterns and fit a
    single calibration peak consistently across windows. Explicit
    ``AnalysisPaths`` take precedence over legacy root/subdirectory arguments.

    Attributes
    ----------
    sample_name : str
        Sample identifier used to resolve calibration and dark-data products.
    temperature_K : int
        Calibration-image sample temperature in kelvin.
    paths : AnalysisPaths
        Normalized path configuration used by all context operations.
    path_root : str, pathlib.Path, or None
        Legacy experiment root retained for backward-compatible construction.
    analysis_subdir : str, pathlib.Path, or None
        Legacy processed-analysis directory name.
    """
    sample_name: str
    temperature_K: int
    paths: Optional[AnalysisPaths] = None
    path_root: Optional[Union[str, Path]] = None
    analysis_subdir: Optional[Union[str, Path]] = None

    def __post_init__(self) -> None:
        """Normalize metadata and replace legacy path arguments with ``AnalysisPaths``."""
        object.__setattr__(self, "sample_name", str(self.sample_name))
        object.__setattr__(self, "temperature_K", int(self.temperature_K))
        object.__setattr__(
            self,
            "paths",
            _coerce_paths(
                paths=self.paths,
                path_root=self.path_root,
                analysis_subdir=self.analysis_subdir,
            ),
        )

    @property
    def calibration_dir(self) -> Path:
        """Return the experiment-level directory containing PONI and mask files."""
        return Path(self.paths.root("calibration"))

    def dark_dataset(self, scan_spec: ScanSpec) -> azimint_utils.DarkDataset:
        """Construct the standardized dark dataset selected by ``scan_spec``."""
        return azimint_utils.DarkDataset(
            self.sample_name,
            int(self.temperature_K),
            dark_tag=_resolve_dark_tag(scan_spec),
            paths=self.paths,
        )

    def analysis_dir(self, scan_spec: ScanSpec) -> Path:
        """Return the processed dark-data analysis directory selected by ``scan_spec``."""
        return Path(self.dark_dataset(scan_spec).analysis_dir())

    def xy_dir(self, scan_spec: ScanSpec) -> Path:
        """Return the cached calibration XY-pattern directory selected by ``scan_spec``."""
        return Path(self.dark_dataset(scan_spec).xy_folder())

    def peak_fits_csv_path(
        self,
        scan_spec: ScanSpec,
        *,
        out_csv_name: str = "peak_fits.csv",
    ) -> Path:
        """Return peak fits CSV path.

        Parameters
        ----------
        scan_spec : ScanSpec
            Dark-scan identifier, combined scan sequence, or existing scan-tag string.
        out_csv_name : str
            Filename for the fitting-result CSV within the experiment fitting directory.

        Returns
        -------
        Path
            Calibration fitting CSV path below the dark dataset analysis directory.
        """
        return self.analysis_dir(scan_spec) / str(out_csv_name)

    def _default_poni_and_mask(
        self,
        scan_spec: ScanSpec,
        *,
        poni_path: Optional[Union[str, Path]] = None,
        mask_edf_path: Optional[Union[str, Path]] = None,
        require_mask: bool = True,
    ) -> Tuple[str, Optional[str]]:
        """Return the default PONI and, when requested, detector mask."""
        if poni_path is not None and (mask_edf_path is not None or not require_mask):
            return (
                str(poni_path),
                None if not require_mask else str(mask_edf_path),
            )

        first = general_utils.first_scan_id(scan_spec)
        if first is None:
            raise ValueError(
                "Cannot infer calibration files from scan_spec string. "
                "Provide poni_path explicitly and mask_edf_path when masking is enabled."
            )

        cal_dir = self.calibration_dir
        if poni_path is None:
            poni_path = cal_dir / f"{self.sample_name}_{first}.poni"
        if require_mask and mask_edf_path is None:
            mask_edf_path = cal_dir / f"{first}_mask.edf"

        poni_path = Path(str(poni_path))
        mask_path = None if not require_mask else Path(str(mask_edf_path))

        if not poni_path.exists():
            fallback_poni = cal_dir / "DET55_167246.poni"
            poni_path = fallback_poni
            if require_mask:
                mask_path = cal_dir / "167246_mask.edf"

        if require_mask and mask_path is not None and not mask_path.exists():
            fallback_mask = cal_dir / "167246_mask.edf"
            mask_path = fallback_mask

        if not poni_path.exists():
            raise FileNotFoundError(f"PONI not found: {poni_path}")
        if require_mask and (mask_path is None or not mask_path.exists()):
            raise FileNotFoundError(f"Mask EDF not found: {mask_path}")

        return str(poni_path), None if mask_path is None else str(mask_path)

    def compute_xy_files(
        self,
        scan_spec: ScanSpec,
        *,
        azimuthal_ranges: Sequence[Union[int, float]],
        include_full: bool = False,
        full_range: Tuple[float, float] = (-90.0, 90.0),
        npt: int = 1000,
        normalize: bool = True,
        q_norm_range: Tuple[float, float] = (2.65, 2.75),
        overwrite_xy: bool = False,
        poni_path: Optional[Union[str, Path]] = None,
        mask_edf_path: Optional[Union[str, Path]] = None,
        azim_offset_deg: float = -90.0,
        polarization_factor: Optional[float] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Integrate and cache calibration patterns for all requested windows.

        Parameters
        ----------
        scan_spec : ScanSpec
            Dark-scan identifier, combined scan sequence, or existing scan-tag string.
        azimuthal_ranges : Sequence[Union[int, float]]
            Ordered azimuthal edges in degrees used to construct adjacent windows.
        include_full : bool
            Whether to include an additional pattern integrated over ``full_range``.
        full_range : Tuple[float, float]
            Azimuthal limits in degrees for the optional full-range pattern.
        npt : int
            Number of radial points in each integrated pattern.
        normalize : bool
            Whether to normalize intensities by their mean in ``q_norm_range``.
        q_norm_range : Tuple[float, float]
            q interval in Å⁻¹ used to calculate the normalization mean.
        overwrite_xy : bool
            Whether existing XY cache files should be recomputed.
        poni_path : Optional[Union[str, Path]]
            pyFAI PONI calibration file. Automatic discovery is used where supported when omitted.
        mask_edf_path : Optional[Union[str, Path]]
            Optional detector-mask EDF file; masked pixels are excluded from integration.
        azim_offset_deg : float
            Angular offset in degrees applied before azimuthal integration.
        polarization_factor : Optional[float]
            pyFAI polarization correction in ``[-1, 1]``; ``None`` disables correction.

        Returns
        -------
        Dict[str, Tuple[np.ndarray, np.ndarray]]
            Mapping from azimuthal tags to integrated ``(q, intensity)`` arrays.
        """
        poni_path_s, mask_path_s = self._default_poni_and_mask(
            scan_spec,
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
        )

        ds = self.dark_dataset(scan_spec)
        integrator = azimint_utils.AzimIntegrator(
            poni_path=poni_path_s,
            mask_edf_path=mask_path_s,
            npt=int(npt),
            normalize=bool(normalize),
            q_norm_range=tuple(q_norm_range),
            azim_offset_deg=float(azim_offset_deg),
            polarization_factor=polarization_factor,
        )

        return integrator.integrate_and_cache_xy(
            ds,
            azimuthal_edges=np.asarray(azimuthal_ranges, dtype=float),
            include_full=bool(include_full),
            full_range=tuple(full_range),
            overwrite_xy=bool(overwrite_xy),
        )

    def compute_2d_cake(
        self,
        scan_spec: ScanSpec,
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load a calibration image and calculate its pyFAI 2D cake.

        The EDF mask is resolved and applied only when ``use_mask`` is true.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Bare detector image, cake intensity, q coordinates in Å⁻¹, and
            display-coordinate azimuths in degrees.
        """
        poni_path_s, mask_path_s = self._default_poni_and_mask(
            scan_spec,
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            require_mask=bool(use_mask),
        )
        dataset = self.dark_dataset(scan_spec)
        image = np.asarray(dataset.load_2d())

        integrator = azimint_utils.AzimIntegrator(
            poni_path=poni_path_s,
            mask_edf_path=mask_path_s,
            npt=int(npt_rad),
            normalize=bool(normalize),
            q_norm_range=tuple(float(v) for v in q_norm_range),
            azim_offset_deg=float(azim_offset_deg),
            polarization_factor=polarization_factor,
        )
        cake, q, azimuth = integrator.integrate2d(
            image,
            npt_rad=int(npt_rad),
            npt_azim=int(npt_azim),
            radial_range=(
                None
                if radial_range is None
                else tuple(float(v) for v in radial_range)
            ),
            azimuthal_range=tuple(float(v) for v in azimuthal_range),
        )
        return image, cake, q, azimuth

    def load_xy(
        self,
        scan_spec: ScanSpec,
        *,
        azim_window: Tuple[float, float],
        npt: int = 1000,
        normalize: bool = True,
        q_norm_range: Tuple[float, float] = (2.65, 2.75),
        compute_if_missing: bool = True,
        overwrite_xy: bool = False,
        poni_path: Optional[Union[str, Path]] = None,
        mask_edf_path: Optional[Union[str, Path]] = None,
        azim_offset_deg: float = -90.0,
        polarization_factor: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Path]:
        """Load one cached calibration pattern, computing it when requested.

        Parameters
        ----------
        scan_spec : ScanSpec
            Dark-scan identifier, combined scan sequence, or existing scan-tag string.
        azim_window : Tuple[float, float]
            Inclusive azimuthal integration limits in degrees.
        npt : int
            Number of radial points in each integrated pattern.
        normalize : bool
            Whether to normalize intensities by their mean in ``q_norm_range``.
        q_norm_range : Tuple[float, float]
            q interval in Å⁻¹ used to calculate the normalization mean.
        compute_if_missing : bool
            Whether missing XY patterns may be integrated from their 2D images.
        overwrite_xy : bool
            Whether existing XY cache files should be recomputed.
        poni_path : Optional[Union[str, Path]]
            pyFAI PONI calibration file. Automatic discovery is used where supported when omitted.
        mask_edf_path : Optional[Union[str, Path]]
            Optional detector-mask EDF file; masked pixels are excluded from integration.
        azim_offset_deg : float
            Angular offset in degrees applied before azimuthal integration.
        polarization_factor : Optional[float]
            pyFAI polarization correction in ``[-1, 1]``; ``None`` disables correction.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Path]
            q values in Å⁻¹, intensities, and the XY file path used.

        Raises
        ------
        FileNotFoundError
            If required raw data, calibration, metadata, or cached analysis files are missing.
        """
        azim_window_i = (
            int(general_utils.to_int(azim_window[0])),
            int(general_utils.to_int(azim_window[1])),
        )
        azim_str = general_utils.azim_range_str(azim_window_i)
        ds = self.dark_dataset(scan_spec)
        xy_path = Path(ds.xy_path(azim_str))

        poni_path_s, mask_path_s = self._default_poni_and_mask(
            scan_spec,
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
        )
        integrator = azimint_utils.AzimIntegrator(
            poni_path=poni_path_s,
            mask_edf_path=mask_path_s,
            npt=int(npt),
            normalize=bool(normalize),
            q_norm_range=tuple(q_norm_range),
            azim_offset_deg=float(azim_offset_deg),
            polarization_factor=polarization_factor,
        )

        if xy_path.exists() and (not overwrite_xy):
            two_theta, intensity = general_utils.load_xy(xy_path)
            q = general_utils.two_theta_to_q(two_theta, integrator._ai.wavelength)
            if normalize:
                intensity = general_utils.normalize_y_by_mean_in_xrange(
                    q,
                    intensity,
                    q_norm_range,
                )
            return np.asarray(q, float), np.asarray(intensity, float), xy_path

        if not compute_if_missing:
            raise FileNotFoundError(str(xy_path))

        _, q, intensity = integrator.get_xy_for_window(
            ds,
            azim_window_i,
            compute_if_missing=True,
            overwrite_xy=bool(overwrite_xy),
        )
        return np.asarray(q, float), np.asarray(intensity, float), xy_path

    def do_peak_fitting(
        self,
        scan_spec: ScanSpec,
        *,
        q_fit_range: Tuple[float, float] = (2.4, 2.65),
        azimuthal_ranges: Sequence[Union[int, float]] = tuple(np.arange(-90, 90 + 20, 45)),
        include_full: bool = False,
        full_range: Tuple[float, float] = (-90.0, 90.0),
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
    ) -> pd.DataFrame:
        """Fit the calibration peak independently in every azimuthal window.

        Parameters
        ----------
        scan_spec : ScanSpec
            Dark-scan identifier, combined scan sequence, or existing scan-tag string.
        q_fit_range : Tuple[float, float]
            q interval in Å⁻¹ included in the peak fit.
        azimuthal_ranges : Sequence[Union[int, float]]
            Ordered azimuthal edges in degrees used to construct adjacent windows.
        include_full : bool
            Whether to include an additional pattern integrated over ``full_range``.
        full_range : Tuple[float, float]
            Azimuthal limits in degrees for the optional full-range pattern.
        npt : int
            Number of radial points in each integrated pattern.
        normalize : bool
            Whether to normalize intensities by their mean in ``q_norm_range``.
        q_norm_range : Tuple[float, float]
            q interval in Å⁻¹ used to calculate the normalization mean.
        eta : float
            Pseudo-Voigt mixing fraction between Gaussian and Lorentzian components.
        fit_method : str
            lmfit optimization method passed to the model fit.
        force_refit : bool
            Whether to recompute fits when a result CSV already exists.
        out_csv_name : str
            Filename for the fitting-result CSV within the experiment fitting directory.
        overwrite_xy : bool
            Whether existing XY cache files should be recomputed.
        poni_path : Optional[Union[str, Path]]
            pyFAI PONI calibration file. Automatic discovery is used where supported when omitted.
        mask_edf_path : Optional[Union[str, Path]]
            Optional detector-mask EDF file; masked pixels are excluded from integration.
        azim_offset_deg : float
            Angular offset in degrees applied before azimuthal integration.
        polarization_factor : Optional[float]
            pyFAI polarization correction in ``[-1, 1]``; ``None`` disables correction.

        Returns
        -------
        pd.DataFrame
            Fit-result table with one row per azimuthal window.
        """
        self.compute_xy_files(
            scan_spec,
            azimuthal_ranges=azimuthal_ranges,
            include_full=bool(include_full),
            full_range=tuple(full_range),
            npt=int(npt),
            normalize=bool(normalize),
            q_norm_range=tuple(q_norm_range),
            overwrite_xy=bool(overwrite_xy),
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            azim_offset_deg=float(azim_offset_deg),
            polarization_factor=polarization_factor,
        )

        windows = general_utils.windows_from_ranges(
            azimuthal_ranges,
            include_full=bool(include_full),
            full_range=(
                int(general_utils.to_int(full_range[0])),
                int(general_utils.to_int(full_range[1])),
            ),
        )

        csv_path = self.peak_fits_csv_path(scan_spec, out_csv_name=str(out_csv_name))
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        existing_df: Optional[pd.DataFrame] = None
        if (not bool(force_refit)) and csv_path.exists():
            existing_df = pd.read_csv(str(csv_path))

        model = _make_single_peak_model()
        q_fit_range = (float(q_fit_range[0]), float(q_fit_range[1]))

        rows: List[Dict[str, object]] = []

        for w in windows:
            azim_str = general_utils.azim_range_str(w)

            if (
                (not bool(force_refit))
                and (existing_df is not None)
                and ("azim_range_str" in existing_df.columns)
            ):
                if np.any(existing_df["azim_range_str"].astype(str).values == azim_str):
                    continue

            q, intensity, _ = self.load_xy(
                scan_spec,
                azim_window=w,
                npt=int(npt),
                normalize=bool(normalize),
                q_norm_range=tuple(q_norm_range),
                compute_if_missing=True,
                overwrite_xy=bool(overwrite_xy),
                poni_path=poni_path,
                mask_edf_path=mask_edf_path,
                azim_offset_deg=float(azim_offset_deg),
                polarization_factor=polarization_factor,
            )

            m = (q >= q_fit_range[0]) & (q <= q_fit_range[1])
            if not np.any(m):
                rows.append(
                    _failed_fit_row(
                        azim_window=w,
                        azim_str=azim_str,
                        q_fit_range=q_fit_range,
                        eta=float(eta),
                    )
                )
                continue

            qfit = np.asarray(q[m], float)
            Ifit = np.asarray(intensity[m], float)
            if qfit.size < 5:
                rows.append(
                    _failed_fit_row(
                        azim_window=w,
                        azim_str=azim_str,
                        q_fit_range=q_fit_range,
                        eta=float(eta),
                    )
                )
                continue

            bg_slope = (Ifit[-1] - Ifit[0]) / (qfit[-1] - qfit[0] + 1e-12)
            bg_c0_guess = float(np.median(Ifit))
            bg_c1_guess = float(bg_slope)

            center_guess = float(qfit[int(np.argmax(Ifit))])
            amp_guess = float(np.trapezoid(np.maximum(Ifit - np.median(Ifit), 0.0), qfit))
            amp_guess = max(float(amp_guess), 1e-12)
            sigma_guess = 0.01

            params = model.make_params()
            params["bg_c0"].set(value=bg_c0_guess)
            params["bg_c1"].set(value=bg_c1_guess)
            params["pv_center"].set(value=center_guess, min=q_fit_range[0], max=q_fit_range[1])
            params["pv_sigma"].set(value=sigma_guess, min=1e-6, max=0.2)
            params["pv_amplitude"].set(value=amp_guess, min=0.0)
            params["pv_fraction"].set(value=float(eta), vary=False, min=0.0, max=1.0)

            try:
                result = model.fit(Ifit, params, x=qfit, method=str(fit_method))
            except Exception:
                result = None

            if (result is None) or (not getattr(result, "success", False)):
                rows.append(
                    _failed_fit_row(
                        azim_window=w,
                        azim_str=azim_str,
                        q_fit_range=q_fit_range,
                        eta=float(eta),
                    )
                )
                continue

            yfit = np.asarray(result.eval(x=qfit), float)
            r2 = float(general_utils.compute_r2(Ifit, yfit))

            pv_center = float(result.params["pv_center"].value)
            comps = model.eval_components(params=result.params, x=np.array([pv_center], float))
            pv_height = float(np.asarray(comps.get("pv_", [np.nan]), float)[0])

            pv_sigma = float(result.params["pv_sigma"].value)
            pv_fwhm = float(_pv_fwhm_from_result(result))

            rows.append(
                dict(
                    success=True,
                    azim_center=float(general_utils.azim_center(w)),
                    azim_range_str=str(azim_str),
                    q_fit0=float(q_fit_range[0]),
                    q_fit1=float(q_fit_range[1]),
                    bg_c0=float(result.params["bg_c0"].value),
                    bg_c1=float(result.params["bg_c1"].value),
                    pv_center=float(pv_center),
                    pv_sigma=float(pv_sigma),
                    pv_amplitude=float(result.params["pv_amplitude"].value),
                    pv_fraction=float(result.params["pv_fraction"].value),
                    pv_height=float(pv_height),
                    pv_fwhm=float(pv_fwhm),
                    r2=float(r2),
                )
            )

        df_new = pd.DataFrame(rows)

        if bool(force_refit) or (existing_df is None) or existing_df.empty:
            df_out = df_new
        else:
            if "azim_range_str" in existing_df.columns and "azim_range_str" in df_new.columns:
                old = existing_df.copy()
                old["azim_range_str"] = old["azim_range_str"].astype(str)
                new_keys = set(df_new["azim_range_str"].astype(str).tolist())
                old = old[~old["azim_range_str"].isin(new_keys)]
                df_out = pd.concat([old, df_new], ignore_index=True)
            else:
                df_out = df_new

        if "azim_center" in df_out.columns:
            df_out = df_out.sort_values("azim_center", na_position="last").reset_index(drop=True)

        tmp_path = csv_path.with_suffix(".tmp.csv")
        df_out.to_csv(str(tmp_path), index=False)
        os.replace(str(tmp_path), str(csv_path))

        return df_out


class MaskManager:
    """Create and update pyFAI-compatible detector masks.

    Masks are boolean arrays in detector-pixel coordinates. The manager can
    start from an existing EDF mask or an empty detector-shaped template, add
    negative-valued image pixels, and persist the result as EDF. Input images
    and masks must have identical shapes.

    Attributes
    ----------
    dataset : azimint_utils.DarkDataset
        Dark dataset providing the detector image used to update the mask.
    calibration_dir : pathlib.Path
        Default directory for input templates and generated EDF masks.
    """
    def __init__(
        self,
        dataset: azimint_utils.DarkDataset,
        *,
        calibration_dir: Union[str, Path],
    ):
        """Bind a dark detector dataset and normalize the mask output directory."""
        self.dataset = dataset
        self.calibration_dir = Path(calibration_dir)

    def update_with_negatives(
        self,
        mask_edf_path: Union[str, Path],
        *,
        out_mask_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ) -> Path:
        """Add every negative-valued detector pixel to the current mask.

        Parameters
        ----------
        mask_edf_path : Union[str, Path]
            Optional detector-mask EDF file; masked pixels are excluded from integration.
        out_mask_path : Optional[Union[str, Path]]
            Filesystem path for out mask.
        overwrite : bool
            Whether existing output artifacts may be replaced.

        Returns
        -------
        Path
            Path of the mask file after adding all negative detector pixels.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.

        Notes
        -----
        This operation may create or replace analysis artifacts according to its save and overwrite settings.
        """
        img = np.asarray(self.dataset.load_2d())
        mask_path = Path(mask_edf_path)

        mask_fabio = fabio.open(str(mask_path))
        mask = np.asarray(mask_fabio.data, dtype=np.uint8)

        if mask.shape != img.shape:
            raise ValueError(f"Mask shape {mask.shape} != img shape {img.shape}")

        mask = mask.copy()
        mask[img < 0] = 1
        mask_fabio.data = mask

        if out_mask_path is None:
            if not overwrite:
                raise ValueError(
                    "out_mask_path is None and overwrite=False. Refusing to overwrite."
                )
            out_path = mask_path
        else:
            out_path = Path(out_mask_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        mask_fabio.write(str(out_path))
        return out_path

    def create_from_empty_template(
        self,
        empty_mask_path: Union[str, Path],
        *,
        out_mask_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Create an empty mask matching a supplied detector image.

        Parameters
        ----------
        empty_mask_path : Union[str, Path]
            Filesystem path for empty mask.
        out_mask_path : Optional[Union[str, Path]]
            Filesystem path for out mask.

        Returns
        -------
        Path
            Path of the newly written empty detector mask.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        img = np.asarray(self.dataset.load_2d())
        empty_path = Path(empty_mask_path)

        mask_fabio = fabio.open(str(empty_path))
        mask = np.asarray(mask_fabio.data, dtype=np.uint8)

        if mask.shape != img.shape:
            raise ValueError(f"Mask shape {mask.shape} != img shape {img.shape}")

        mask = mask.copy()
        mask[img < 0] = 1
        mask_fabio.data = mask

        if out_mask_path is None:
            out_path = self.calibration_dir / f"{self.dataset.file_tag}_mask.edf"
        else:
            out_path = Path(out_mask_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        mask_fabio.write(str(out_path))
        return out_path


def create_mask(
    *,
    sample_name: str,
    scan_spec: ScanSpec,
    temperature_K: int,
    empty_mask_path: Optional[Union[str, Path]] = None,
    out_mask_path: Optional[Union[str, Path]] = None,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> Path:
    """Create or edit a detector mask and save it as EDF.

    Parameters
    ----------
    sample_name : str
        Sample identifier used in the standardized analysis directory layout.
    scan_spec : ScanSpec
        Dark-scan identifier, combined scan sequence, or existing scan-tag string.
    temperature_K : int
        Sample temperature in kelvin.
    empty_mask_path : Optional[Union[str, Path]]
        Filesystem path for empty mask.
    out_mask_path : Optional[Union[str, Path]]
        Filesystem path for out mask.
    paths : Optional[AnalysisPaths]
        Resolved ``AnalysisPaths`` configuration. It takes precedence over legacy path arguments.
    path_root : Optional[Union[str, Path]]
        Root directory containing raw and analysis data trees.
    analysis_subdir : Optional[Union[str, Path]]
        Analysis-directory path relative to ``path_root``.

    Returns
    -------
    Path
        Path of the EDF mask written by the interactive mask manager.
    """
    ctx = CalibrationContext(
        sample_name=str(sample_name),
        temperature_K=int(temperature_K),
        paths=paths,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )
    dataset = ctx.dark_dataset(scan_spec)
    mask_mgr = MaskManager(dataset, calibration_dir=ctx.calibration_dir)

    if empty_mask_path is None:
        empty_mask_path = ctx.calibration_dir / "empty_mask.edf"

    if out_mask_path is None:
        out_mask_path = ctx.calibration_dir / f"{sample_name}_{dataset.file_tag}_mask.edf"

    return mask_mgr.update_with_negatives(
        empty_mask_path,
        out_mask_path=out_mask_path,
        overwrite=False,
    )


__all__ = [
    "ScanSpec",
    "CalibrationContext",
    "MaskManager",
    "create_mask",
]
