from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import fabio
import numpy as np
import pandas as pd

from . import azimint_utils, general_utils, plot_utils
from .paths import AnalysisPaths

ScanSpec = Union[int, Sequence[int], str]


def _resolve_dark_tag(scan_spec: ScanSpec) -> str:
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
    from lmfit.models import PolynomialModel, PseudoVoigtModel

    bg = PolynomialModel(degree=1, prefix="bg_")
    pv = PseudoVoigtModel(prefix="pv_")
    return bg + pv


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
    sample_name: str
    temperature_K: int
    paths: Optional[AnalysisPaths] = None
    path_root: Optional[Union[str, Path]] = None
    analysis_subdir: Optional[Union[str, Path]] = None

    def __post_init__(self) -> None:
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
        return Path(self.paths.root("calibration"))

    def dark_dataset(self, scan_spec: ScanSpec) -> azimint_utils.DarkDataset:
        return azimint_utils.DarkDataset(
            self.sample_name,
            int(self.temperature_K),
            dark_tag=_resolve_dark_tag(scan_spec),
            paths=self.paths,
        )

    def analysis_dir(self, scan_spec: ScanSpec) -> Path:
        return Path(self.dark_dataset(scan_spec).analysis_dir())

    def xy_dir(self, scan_spec: ScanSpec) -> Path:
        return Path(self.dark_dataset(scan_spec).xy_folder())

    def peak_fits_csv_path(
        self,
        scan_spec: ScanSpec,
        *,
        out_csv_name: str = "peak_fits.csv",
    ) -> Path:
        return self.analysis_dir(scan_spec) / str(out_csv_name)

    def _default_poni_and_mask(
        self,
        scan_spec: ScanSpec,
        *,
        poni_path: Optional[Union[str, Path]] = None,
        mask_edf_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[str, str]:
        if poni_path is not None and mask_edf_path is not None:
            return str(poni_path), str(mask_edf_path)

        first = general_utils.first_scan_id(scan_spec)
        if first is None:
            raise ValueError(
                "Cannot infer default poni/mask from scan_spec string. "
                "Provide poni_path and mask_edf_path explicitly."
            )

        cal_dir = self.calibration_dir
        if poni_path is None:
            poni_path = cal_dir / f"{self.sample_name}_{first}.poni"
        if mask_edf_path is None:
            mask_edf_path = cal_dir / f"{first}_mask.edf"

        poni_path = Path(str(poni_path))
        mask_edf_path = Path(str(mask_edf_path))

        if not poni_path.exists():
            fallback_poni = cal_dir / "DET55_167246.poni"
            fallback_mask = cal_dir / "167246_mask.edf"
            poni_path = fallback_poni
            mask_edf_path = fallback_mask

        if not mask_edf_path.exists():
            fallback_mask = cal_dir / "167246_mask.edf"
            mask_edf_path = fallback_mask

        if not poni_path.exists():
            raise FileNotFoundError(f"PONI not found: {poni_path}")
        if not mask_edf_path.exists():
            raise FileNotFoundError(f"Mask EDF not found: {mask_edf_path}")

        return str(poni_path), str(mask_edf_path)

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
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
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
        )

        return integrator.integrate_and_cache_xy(
            ds,
            azimuthal_edges=np.asarray(azimuthal_ranges, dtype=float),
            include_full=bool(include_full),
            full_range=tuple(full_range),
            overwrite_xy=bool(overwrite_xy),
        )

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
    ) -> Tuple[np.ndarray, np.ndarray, Path]:
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
        )

        if xy_path.exists() and (not overwrite_xy):
            two_theta, intensity = general_utils.load_xy(xy_path)
            q = general_utils.two_theta_to_q(two_theta, integrator._ai.wavelength)
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
    ) -> pd.DataFrame:
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
            amp_guess = float(np.trapz(np.maximum(Ifit - np.median(Ifit), 0.0), qfit))
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
    def __init__(
        self,
        dataset: azimint_utils.DarkDataset,
        *,
        calibration_dir: Union[str, Path],
    ):
        self.dataset = dataset
        self.calibration_dir = Path(calibration_dir)

    def update_with_negatives(
        self,
        mask_edf_path: Union[str, Path],
        *,
        out_mask_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ) -> Path:
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