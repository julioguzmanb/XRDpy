# azimint_utils.py
"""
Azimuthal integration utilities:
- Dataset path resolution (delay/dark)
- PONI compatibility loading
- pyFAI integration and XY caching
- Delay discovery helpers

This module is utils-like: reusable logic with minimal user-facing glue.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import fabio
import pyFAI
from tqdm import tqdm

from . import general_utils  
from .paths import AnalysisPaths

_LOAD_XY = general_utils.load_xy
_SAVE_XY = general_utils.save_xy


def _patch_poni_text_minimal(text: str) -> Tuple[str, List[str]]:
    changes: List[str] = []
    lines = text.splitlines(True)
    out: List[str] = []

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith("poni_version"):
            m = re.match(r"^(poni_version\s*[:=]\s*)(\S+)\s*$", stripped, flags=re.IGNORECASE)
            if m:
                prefix, ver = m.group(1), m.group(2)
                if re.match(r"^\d+\.\d+$", ver):
                    new_ver = ver.split(".")[0]
                    new_stripped = prefix + new_ver
                    changes.append(f"poni_version {ver} -> {new_ver}")
                    out.append(re.sub(re.escape(stripped), new_stripped, line))
                    continue

        if stripped.startswith("Detector_config"):
            m = re.match(r"^(Detector_config\s*[:=]\s*)(\{.*\})\s*$", stripped)
            if m:
                prefix, cfg_txt = m.group(1), m.group(2)
                new_cfg_txt = cfg_txt
                did = False

                try:
                    cfg = json.loads(cfg_txt)
                    if isinstance(cfg, dict) and "orientation" in cfg:
                        cfg.pop("orientation", None)
                        new_cfg_txt = json.dumps(cfg)
                        did = True
                except Exception:
                    new_cfg_txt2 = re.sub(r',\s*"orientation"\s*:\s*[^,}]+', "", cfg_txt)
                    new_cfg_txt2 = re.sub(r'"orientation"\s*:\s*[^,}]+\s*,\s*', "", new_cfg_txt2)
                    if new_cfg_txt2 != cfg_txt:
                        new_cfg_txt = new_cfg_txt2
                        did = True

                if did:
                    changes.append("Detector_config: dropped 'orientation'")
                    newline = "\n" if line.endswith("\n") else ""
                    out.append(prefix + new_cfg_txt + newline)
                    continue

        out.append(line)

    return "".join(out), changes


def load_poni_with_compat(
    poni_path: Union[str, Path],
    *,
    write_patched_copy: bool = True,
    patched_suffix: str = ".pyfai021",
    overwrite_patched: bool = True,
    verbose: bool = False,
):
    """
    Try pyFAI.load(poni_path). If it fails, patch minimally and retry.

    Returns:
      (ai, used_poni_path, changes_list)
    """
    poni_path = Path(poni_path)

    try:
        ai = pyFAI.load(str(poni_path))
        return ai, str(poni_path), []
    except Exception as e1:
        orig_err = e1

    text = poni_path.read_text(encoding="utf-8", errors="replace")
    patched_text, changes = _patch_poni_text_minimal(text)

    if not changes:
        raise orig_err

    if write_patched_copy:
        patched_path = Path(str(poni_path) + patched_suffix)

        if patched_path.exists() and (not overwrite_patched):
            ai = pyFAI.load(str(patched_path))
            return ai, str(patched_path), ["(reused existing patched file)"]

        patched_path.write_text(patched_text, encoding="utf-8")

        if verbose:
            print(f"[load_poni_with_compat] original failed: {poni_path}")
            for c in changes:
                print(f"[load_poni_with_compat] patch: {c}")
            print(f"[load_poni_with_compat] wrote patched: {patched_path}")

        ai = pyFAI.load(str(patched_path))
        return ai, str(patched_path), changes

    backup = poni_path.with_suffix(poni_path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(str(poni_path), str(backup))
    poni_path.write_text(patched_text, encoding="utf-8")

    if verbose:
        print(f"[load_poni_with_compat] patched IN PLACE: {poni_path} (backup: {backup})")
        for c in changes:
            print(f"[load_poni_with_compat] patch: {c}")

    ai = pyFAI.load(str(poni_path))
    return ai, str(poni_path), changes


def _dataset_label(ds: Union[DelayDataset, DarkDataset, FluenceDataset]) -> str:
    if isinstance(ds, DelayDataset):
        return f"{ds.sample_name} {ds.temperature_K}K delay {ds.delay_fs}fs"
    if isinstance(ds, FluenceDataset):
        return (
            f"{ds.sample_name} {ds.temperature_K}K fluence {ds.fluence_mJ_cm2:g} mJ/cm2 "
            f"(delay {ds.delay_fs}fs)"
        )
    return f"{ds.sample_name} {ds.temperature_K}K dark {ds.dark_tag}"


class DelayDataset:
    """One delay point output by datared export (2D_images/*.npy), with XY output paths."""

    def __init__(
        self,
        sample_name: str,
        temperature_K: Union[int, float],
        excitation_wl_nm: Union[int, float],
        fluence_mJ_cm2: Union[int, float],
        time_window_fs: int,
        delay_fs: int,
        *,
        paths: Optional[AnalysisPaths] = None,
        path_root: Optional[Union[str, Path]] = None,          # legacy fallback
        analysis_subdir: Optional[Union[str, Path]] = None,    # legacy fallback
    ):
        if paths is not None:
            self.analysis_root = Path(paths.analysis_root)
        elif path_root is not None and analysis_subdir is not None:
            self.analysis_root = Path(path_root) / Path(analysis_subdir)
        else:
            raise ValueError(
                "Provide either paths=AnalysisPaths(...) or both "
                "path_root=... and analysis_subdir=..."
            )

        self.sample_name = str(sample_name)
        self.temperature_K = general_utils.to_int(temperature_K)
        self.excitation_wl_nm = excitation_wl_nm
        self.fluence_mJ_cm2 = float(fluence_mJ_cm2)
        self.time_window_fs = int(time_window_fs)
        self.delay_fs = int(delay_fs)

        self._wl_tag = general_utils.wl_tag_nm(self.excitation_wl_nm)
        self._flu_folder = general_utils.fluence_tag_folder(self.fluence_mJ_cm2)
        self._flu_file = general_utils.fluence_tag_file(self.fluence_mJ_cm2)

    def analysis_dir(self) -> Path:
        base = self.analysis_root / self.sample_name / f"temperature_{self.temperature_K}K"
        candidates = [
            base
            / f"excitation_wl_{self._wl_tag}nm"
            / "delay"
            / f"fluence_{self._flu_folder}"
            / f"time_window_{self.time_window_fs}fs",
            base
            / f"excitation_wl_{self.excitation_wl_nm}nm"
            / "delay"
            / f"fluence_{self._flu_folder}"
            / f"time_window_{self.time_window_fs}fs",
        ]
        for c in candidates:
            if c.is_dir():
                return c
        return candidates[0]

    def img_folder(self) -> Path:
        return self.analysis_dir() / "2D_images"

    def xy_folder(self) -> Path:
        return self.analysis_dir() / "xy_files"

    def ensure_dirs(self) -> None:
        self.xy_folder().mkdir(parents=True, exist_ok=True)

    def img_path(self) -> Path:
        name = (
            f"{self.sample_name}_{self.temperature_K}K_{self._wl_tag}nm_"
            f"{self._flu_file}mJ_{int(self.time_window_fs)}fs_{int(self.delay_fs)}fs.npy"
        )
        return self.img_folder() / name

    def load_2d(self) -> np.ndarray:
        p = self.img_path()
        if not p.exists():
            raise FileNotFoundError(str(p))
        return np.load(str(p))

    def xy_path(self, azim_str: str) -> Path:
        self.ensure_dirs()
        name = (
            f"{self.sample_name}_{self.temperature_K}K_{self._wl_tag}nm_"
            f"{self._flu_file}mJ_{int(self.time_window_fs)}fs_{int(self.delay_fs)}fs_{azim_str}.xy"
        )
        return self.xy_folder() / name


class DarkDataset:
    """One dark dataset produced by datared export, with XY output paths."""

    def __init__(
        self,
        sample_name: str,
        temperature_K: Union[int, float],
        *,
        dark_tag: Optional[str] = None,
        paths: Optional[AnalysisPaths] = None,
        path_root: Optional[Union[str, Path]] = None,          # legacy fallback
        analysis_subdir: Optional[Union[str, Path]] = None,    # legacy fallback
    ):
        if paths is not None:
            self.analysis_root = Path(paths.analysis_root)
        elif path_root is not None and analysis_subdir is not None:
            self.analysis_root = Path(path_root) / Path(analysis_subdir)
        else:
            raise ValueError(
                "Provide either paths=AnalysisPaths(...) or both "
                "path_root=... and analysis_subdir=..."
            )

        self.sample_name = str(sample_name)
        self.temperature_K = general_utils.to_int(temperature_K)
        self.dark_tag = self._resolve_dark_tag(dark_tag)

        if self.dark_tag.startswith("scan_"):
            n = self.dark_tag.split("_", 1)[1]
            self.file_tag = f"scan{n}"
        elif self.dark_tag.startswith("scans_"):
            ab = self.dark_tag.split("_", 1)[1]
            self.file_tag = f"scans{ab}"
        else:
            self.file_tag = self.dark_tag.replace("_", "")

    def _dark_base(self) -> Path:
        return (
            self.analysis_root
            / self.sample_name
            / f"temperature_{self.temperature_K}K"
            / "dark"
        )

    def _resolve_dark_tag(self, dark_tag: Optional[str]) -> str:
        base = self._dark_base()
        if dark_tag is not None:
            return str(dark_tag)

        if not base.is_dir():
            raise FileNotFoundError(f"Dark base folder not found: {base}")

        candidates: List[str] = []
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            if not (child.name.startswith("scan_") or child.name.startswith("scans_")):
                continue
            img_dir = child / "2D_images"
            if not img_dir.is_dir():
                continue
            ok = any(
                (p.suffix == ".npy" and p.name.startswith(f"{self.sample_name}_{self.temperature_K}K_dark_"))
                for p in img_dir.iterdir()
            )
            if ok:
                candidates.append(child.name)

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0:
            raise FileNotFoundError(f"No usable dark folders found under: {base}")

        msg = "Multiple dark datasets exist. Provide dark_tag=... explicitly. Options:\n"
        for c in candidates:
            msg += f"  - {c}\n"
        raise ValueError(msg)

    def analysis_dir(self) -> Path:
        return self._dark_base() / self.dark_tag

    def img_folder(self) -> Path:
        return self.analysis_dir() / "2D_images"

    def xy_folder(self) -> Path:
        return self.analysis_dir() / "xy_files"

    def ensure_dirs(self) -> None:
        self.xy_folder().mkdir(parents=True, exist_ok=True)

    def img_path(self) -> Path:
        name = f"{self.sample_name}_{self.temperature_K}K_dark_{self.file_tag}.npy"
        return self.img_folder() / name

    def load_2d(self) -> np.ndarray:
        p = self.img_path()
        if not p.exists():
            raise FileNotFoundError(str(p))
        return np.load(str(p))

    def xy_path(self, azim_str: str) -> Path:
        self.ensure_dirs()
        name = f"{self.sample_name}_{self.temperature_K}K_dark_{self.file_tag}_{azim_str}.xy"
        return self.xy_folder() / name


class FluenceDataset:
    """
    One fluence point output by datared export (2D_images/*.npy), with XY output paths.

    Folder layout expected:
      .../<sample>/temperature_<T>K/excitation_wl_<wl>nm/fluence/delay_<delay>fs/time_window_<tw>fs/2D_images
      Files:
        <sample>_<T>K_<wl>nm_<flu>mJ_<tw>fs_<delay>fs.npy
    """

    def __init__(
        self,
        sample_name: str,
        temperature_K: Union[int, float],
        excitation_wl_nm: Union[int, float],
        fluence_mJ_cm2: Union[int, float],
        time_window_fs: int,
        delay_fs: int,
        *,
        paths: Optional[AnalysisPaths] = None,
        path_root: Optional[Union[str, Path]] = None,          # legacy fallback
        analysis_subdir: Optional[Union[str, Path]] = None,    # legacy fallback
    ):
        if paths is not None:
            self.analysis_root = Path(paths.analysis_root)
        elif path_root is not None and analysis_subdir is not None:
            self.analysis_root = Path(path_root) / Path(analysis_subdir)
        else:
            raise ValueError(
                "Provide either paths=AnalysisPaths(...) or both "
                "path_root=... and analysis_subdir=..."
            )

        self.sample_name = str(sample_name)
        self.temperature_K = general_utils.to_int(temperature_K)
        self.excitation_wl_nm = excitation_wl_nm
        self.fluence_mJ_cm2 = float(fluence_mJ_cm2)
        self.time_window_fs = int(time_window_fs)
        self.delay_fs = int(delay_fs)

        self._wl_tag = general_utils.wl_tag_nm(self.excitation_wl_nm)
        self._flu_file = general_utils.fluence_tag_file(self.fluence_mJ_cm2)

    def analysis_dir(self) -> Path:
        base = self.analysis_root / self.sample_name / f"temperature_{self.temperature_K}K"
        candidates = [
            base
            / f"excitation_wl_{self._wl_tag}nm"
            / "fluence"
            / f"delay_{int(self.delay_fs)}fs"
            / f"time_window_{int(self.time_window_fs)}fs",
            base
            / f"excitation_wl_{self.excitation_wl_nm}nm"
            / "fluence"
            / f"delay_{int(self.delay_fs)}fs"
            / f"time_window_{int(self.time_window_fs)}fs",
        ]
        for c in candidates:
            if c.is_dir():
                return c
        return candidates[0]

    def img_folder(self) -> Path:
        return self.analysis_dir() / "2D_images"

    def xy_folder(self) -> Path:
        return self.analysis_dir() / "xy_files"

    def ensure_dirs(self) -> None:
        self.xy_folder().mkdir(parents=True, exist_ok=True)

    def img_path(self) -> Path:
        name = (
            f"{self.sample_name}_{self.temperature_K}K_{self._wl_tag}nm_"
            f"{self._flu_file}mJ_{int(self.time_window_fs)}fs_{int(self.delay_fs)}fs.npy"
        )
        return self.img_folder() / name

    def load_2d(self) -> np.ndarray:
        p = self.img_path()
        if not p.exists():
            raise FileNotFoundError(str(p))
        return np.load(str(p))

    def xy_path(self, azim_str: str) -> Path:
        self.ensure_dirs()
        name = (
            f"{self.sample_name}_{self.temperature_K}K_{self._wl_tag}nm_"
            f"{self._flu_file}mJ_{int(self.time_window_fs)}fs_{int(self.delay_fs)}fs_{azim_str}.xy"
        )
        return self.xy_folder() / name


class AzimIntegrator:
    def __init__(
        self,
        *,
        poni_path: Optional[Union[str, Path]] = None,
        mask_edf_path: Optional[Union[str, Path]] = None,
        npt: int = 1000,
        normalize: bool = True,
        q_norm_range: Tuple[float, float] = (2.65, 2.75),
        poni_verbose: bool = False,
        azim_offset_deg: float = -90.0,
        default_poni_path: Optional[Union[str, Path]] = None,
    ):
        self.poni_path = None if poni_path is None else str(poni_path)
        self.mask_edf_path = None if mask_edf_path is None else str(mask_edf_path)
        self.default_poni_path = None if default_poni_path is None else str(default_poni_path)
        self.poni_verbose = poni_verbose

        self.npt = int(npt)
        self.normalize = bool(normalize)
        self.q_norm_range = (float(q_norm_range[0]), float(q_norm_range[1]))
        self.azim_offset_deg = float(azim_offset_deg)

        self._ai = None
        self._poni_used: Optional[str] = None
        self._poni_patches: List[str] = []

        if self.poni_path is not None:
            self._ai, self._poni_used, self._poni_patches = load_poni_with_compat(
                self.poni_path,
                verbose=poni_verbose,
            )

        self._mask = None
        if self.mask_edf_path is not None:
            self._mask = fabio.open(self.mask_edf_path).data

    @staticmethod
    def build_windows(
        azimuthal_edges: np.ndarray,
        *,
        include_full: bool = True,
        full_range: Tuple[float, float] = (-180, 180),
    ) -> List[Tuple[float, float]]:
        edges = np.asarray(azimuthal_edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("azimuthal_edges must be a 1D array of at least 2 values.")
        wins = general_utils.windows_from_edges(
            edges.tolist(),
            include_full=bool(include_full),
            full_range=(float(full_range[0]), float(full_range[1])),
            full_first=True,
            make_int=False,
        )
        return [(float(a), float(b)) for a, b in wins]

    def integrate1d(self, img: np.ndarray, azimuthal_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        if self._ai is None:
            raise ValueError("AzimIntegrator has no PONI loaded (poni_path=None).")
        if self._mask is None:
            raise ValueError("AzimIntegrator has no mask loaded (mask_edf_path=None).")

        phi0, phi1 = float(azimuthal_range[0]), float(azimuthal_range[1])

        q, I = self._ai.integrate1d(
            img,
            npt=self.npt,
            mask=self._mask,
            azimuth_range=(phi0 + self.azim_offset_deg, phi1 + self.azim_offset_deg),
            unit="q_A^-1",
        )

        if self.normalize:
            q0, q1 = self.q_norm_range
            m = (q >= q0) & (q <= q1)
            denom = float(np.mean(I[m])) if np.any(m) else float(np.mean(I))
            if denom != 0:
                I = I / denom

        return np.asarray(q), np.asarray(I)

    def _ensure_ai_loaded(self) -> None:
        if self._ai is not None:
            return

        if self.default_poni_path is None:
            raise ValueError(
                "AzimIntegrator has no PONI loaded. Provide poni_path=... when creating "
                "the integrator, or provide default_poni_path=... for lazy loading."
            )

        self.poni_path = str(self.default_poni_path)
        self._ai, self._poni_used, self._poni_patches = load_poni_with_compat(
            self.poni_path,
            verbose=self.poni_verbose,
        )

    def get_xy_for_window(
        self,
        dataset: Union[DelayDataset, DarkDataset, FluenceDataset],
        azimuthal_range: Tuple[float, float],
        *,
        compute_if_missing: bool = True,
        overwrite_xy: bool = False,
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        self._ensure_ai_loaded()

        azim_str = general_utils.azim_range_str((azimuthal_range[0], azimuthal_range[1]))
        xy_path = dataset.xy_path(azim_str)

        if (not overwrite_xy) and xy_path.exists():
            two_theta, I = _LOAD_XY(xy_path)
            q = general_utils.two_theta_to_q(two_theta, self._ai.wavelength)
            return azim_str, q, I

        img = dataset.load_2d()
        q, I = self.integrate1d(img, azimuthal_range)

        two_theta = general_utils.q_to_two_theta(q, self._ai.wavelength)
        _SAVE_XY(xy_path, two_theta, I)
        return azim_str, q, I

    def integrate_and_cache_xy(
        self,
        dataset: Union[DelayDataset, DarkDataset, FluenceDataset],
        *,
        azimuthal_edges: np.ndarray,
        include_full: bool = True,
        full_range: Tuple[float, float] = (-180, 180),
        overwrite_xy: bool = False,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        self._ensure_ai_loaded()

        windows = self.build_windows(
            azimuthal_edges,
            include_full=include_full,
            full_range=full_range,
        )

        patterns: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        img: Optional[np.ndarray] = None

        for ar in tqdm(windows, desc=f"xy_files: {_dataset_label(dataset)}", leave=False):
            azim_str = general_utils.azim_range_str((ar[0], ar[1]))
            xy_path = dataset.xy_path(azim_str)

            if (not overwrite_xy) and xy_path.exists():
                two_theta, I = _LOAD_XY(xy_path)
                q = general_utils.two_theta_to_q(two_theta, self._ai.wavelength)
            else:
                if img is None:
                    img = dataset.load_2d()
                q, I = self.integrate1d(img, ar)
                two_theta = general_utils.q_to_two_theta(q, self._ai.wavelength)
                _SAVE_XY(xy_path, two_theta, I)

            patterns[azim_str] = (q, I)

        return patterns


@dataclass(frozen=True)
class RefSpec:
    ref_type: str  # "delay" or "dark"
    ref_value: Union[int, str, Sequence[int]]


def available_delay_points_fs(
    *,
    sample_name: str,
    temperature_K: Union[int, float],
    excitation_wl_nm: Union[int, float],
    fluence_mJ_cm2: Union[int, float],
    time_window_fs: int,
    path_root: Optional[str] = None,
    analysis_subdir: Optional[str] = None,
    from_2D_imgs: bool = True
) -> List[int]:
    tmp = DelayDataset(
        sample_name,
        temperature_K,
        excitation_wl_nm,
        fluence_mJ_cm2,
        time_window_fs,
        delay_fs=0,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )
    if from_2D_imgs:
        format_sufix="npy"
        type_dummy=""
        folder = tmp.img_folder()
    else:
        format_sufix="xy"
        type_dummy="_(-?\d+)_(-?\d+)"
        folder = tmp.xy_folder()

    wl_tag = general_utils.wl_tag_nm(excitation_wl_nm)
    flu_file = general_utils.fluence_tag_file(fluence_mJ_cm2)
    T = general_utils.to_int(temperature_K)
    tw = int(time_window_fs)

    patt = re.compile(
        rf"^{re.escape(sample_name)}_{T}K_{re.escape(wl_tag)}nm_{re.escape(flu_file)}mJ_{tw}fs_(-?\d+)fs{type_dummy}\.{format_sufix}$"
    )

    if not folder.is_dir():
        raise FileNotFoundError(
            "folder not found:\n"
            f"  {folder}\n"
            "Check datared export parameters (wl/fluence/tw) and folder naming."
        )

    delays: List[int] = []
    for p in folder.iterdir():
        if p.suffix != f".{format_sufix}":
            continue
        m = patt.match(p.name)
        if m:
            delays.append(int(m.group(1)))

    delays = sorted(set(delays))
    if len(delays) == 0:
        raise FileNotFoundError(
            "No delay found for this experiment in:\n"
            f"  {folder}\n"
        )
    return delays


def normalize_delays_fs(
    delays_fs: Union[int, Sequence[int], str],
    *,
    sample_name: str,
    temperature_K: Union[int, float],
    excitation_wl_nm: Union[int, float],
    fluence_mJ_cm2: Union[int, float],
    time_window_fs: int,
    path_root: Optional[str] = None,
    analysis_subdir: Optional[str] = None,
    from_2D_imgs: bool = True
) -> List[int]:
    if isinstance(delays_fs, str):
        if delays_fs.lower() != "all":
            raise ValueError("delays_fs string must be 'all' (or provide int/list).")
        return available_delay_points_fs(
            sample_name=sample_name,
            temperature_K=temperature_K,
            excitation_wl_nm=excitation_wl_nm,
            fluence_mJ_cm2=fluence_mJ_cm2,
            time_window_fs=time_window_fs,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
            from_2D_imgs=from_2D_imgs
        )

    if isinstance(delays_fs, int):
        return [int(delays_fs)]

    return [int(x) for x in list(delays_fs)]


def available_fluence_points_mJ_cm2(
    *,
    sample_name: str,
    temperature_K: Union[int, float],
    excitation_wl_nm: Union[int, float],
    delay_fs: int,
    time_window_fs: int,
    path_root: Optional[str] = None,
    analysis_subdir: Optional[str] = None,
    from_2D_imgs: bool = True
) -> List[float]:
    
    tmp = FluenceDataset(
        sample_name,
        temperature_K,
        excitation_wl_nm,
        fluence_mJ_cm2=1.0,
        time_window_fs=time_window_fs,
        delay_fs=delay_fs,
        path_root=path_root,
        analysis_subdir=analysis_subdir,
    )

    folder = tmp.img_folder()

    wl_tag = general_utils.wl_tag_nm(excitation_wl_nm)
    T = general_utils.to_int(temperature_K)
    tw = int(time_window_fs)
    dly = int(delay_fs)

    if from_2D_imgs:
        format_sufix="npy"
        type_dummy=""
        folder = tmp.img_folder()
    else:
        format_sufix="xy"
        type_dummy="_(-?\d+)_(-?\d+)"
        folder = tmp.xy_folder()

    # flu tag in filenames looks like "15p0mJ"
    patt = re.compile(
        rf"^{re.escape(sample_name)}_{T}K_{re.escape(wl_tag)}nm_([0-9]+(?:p[0-9]+)?)mJ_{tw}fs_{dly}fs{type_dummy}\.{format_sufix}$"
    )

    if not folder.is_dir():
        raise FileNotFoundError(
            "folder not found:\n"
            f"  {folder}\n"
            "Check datared export parameters (wl/delay/tw) and folder naming."
        )

    vals: List[float] = []
    for p in folder.iterdir():
        if p.suffix != f".{format_sufix}":
            continue
        m = patt.match(p.name)
        if not m:
            continue
        tag = str(m.group(1))
        try:
            vals.append(float(tag.replace("p", ".")))
        except Exception:
            continue

    vals = sorted(set(vals))
    if len(vals) == 0:
        raise FileNotFoundError(
            "No fluence found for this experiment in:\n"
            f"  {folder}\n"
        )
    return vals


def normalize_fluences_mJ_cm2(
    fluences_mJ_cm2: Union[float, int, Sequence[Union[float, int]], str],
    *,
    sample_name: str,
    temperature_K: Union[int, float],
    excitation_wl_nm: Union[int, float],
    delay_fs: int,
    time_window_fs: int,
    path_root: Optional[str] = None,
    analysis_subdir: Optional[str] = None,
    from_2D_imgs: bool = None,

) -> List[float]:
    if isinstance(fluences_mJ_cm2, str):
        if fluences_mJ_cm2.lower() != "all":
            raise ValueError("fluences_mJ_cm2 string must be 'all' (or provide float/list).")
        return available_fluence_points_mJ_cm2(
            sample_name=sample_name,
            temperature_K=temperature_K,
            excitation_wl_nm=excitation_wl_nm,
            delay_fs=delay_fs,
            time_window_fs=time_window_fs,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
            from_2D_imgs=from_2D_imgs
        )
    if isinstance(fluences_mJ_cm2, (int, float)):
        return [float(fluences_mJ_cm2)]

    out = [float(x) for x in list(fluences_mJ_cm2)]

    # always sort smallest -> largest (requirement)
    out = sorted(out)

    return out


def dark_tag_from_scan_spec(scan_spec: Union[int, Sequence[int], str]) -> str:
    """
    Convert a scan spec into the folder tag used by datared:
      167246 -> "scan_167246"
      [167246,167285] -> "scans_167246-167285"
      "scans_..." -> as-is
    """
    if isinstance(scan_spec, str):
        return scan_spec
    return general_utils.scan_tag(scan_spec)


def pretty_dark_tag(dark_tag: str) -> str:
    return str(dark_tag).replace("_", ":\n").replace("-", "\n")


def delay_label_value(delay_fs: Union[int, float], *, fs_or_ps: str = "ps", digits: int=2) -> Union[int, float]:
    v = float(delay_fs)
    if str(fs_or_ps).lower() == "ps":
        return round(v * 1e-3, digits)
    return int(round(v))


