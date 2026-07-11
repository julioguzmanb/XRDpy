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


def normalize_polarization_factor(
    polarization_factor: Optional[Union[int, float]],
) -> Optional[float]:
    """Validate pyFAI/txs polarization convention (-1 vertical, +1 horizontal).

    Parameters
    ----------
    polarization_factor : Optional[Union[int, float]]
        pyFAI polarization correction in ``[-1, 1]``; ``None`` disables correction.

    Returns
    -------
    Optional[float]
        Validated polarization factor, or ``None`` when correction is disabled.

    Raises
    ------
    ValueError
        If a selector, range, mode, unit, or metadata value is invalid.
    """
    if polarization_factor is None:
        return None

    value = float(polarization_factor)
    if not np.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(
            "polarization_factor must be None or a finite value between -1 and 1."
        )
    return value


def _patch_poni_text_minimal(text: str) -> Tuple[str, List[str]]:
    """Apply narrowly scoped compatibility edits to legacy PONI text.

    Only unsupported PONI-version syntax and detector ``orientation`` metadata
    are changed. The returned list records every applied edit for provenance.
    """
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
    """Load a PONI file, minimally patching legacy fields when required.

    Parameters
    ----------
    poni_path : path-like
        Source pyFAI geometry file.
    write_patched_copy : bool
        Persist compatible text beside the source rather than loading a
        temporary representation.
    patched_suffix : str
        Suffix appended to the original filename for the compatible copy.
    overwrite_patched : bool
        Replace a previously generated compatible copy.
    verbose : bool
        Print each compatibility change and selected file.

    Returns
    -------
    tuple
        Loaded azimuthal integrator, PONI path actually used, and descriptions
        of compatibility edits. The edit list is empty for a native load.

    Raises
    ------
    Exception
        Re-raises the original pyFAI load error if no safe compatibility edit
        applies.
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
    """Build a human-readable progress label from a dataset descriptor."""
    if isinstance(ds, DelayDataset):
        return f"{ds.sample_name} {ds.temperature_K}K delay {ds.delay_fs}fs"
    if isinstance(ds, FluenceDataset):
        return (
            f"{ds.sample_name} {ds.temperature_K}K fluence {ds.fluence_mJ_cm2:g} mJ/cm2 "
            f"(delay {ds.delay_fs}fs)"
        )
    return f"{ds.sample_name} {ds.temperature_K}K dark {ds.dark_tag}"


class DelayDataset:
    """Describe one standardized delay-series data point.

    Experiment metadata and ``delay_fs`` determine the averaged 2D-image path,
    XY-cache directory, and per-azimuth filename. The class only models paths;
    ``load_2d`` performs the filesystem read and raises ``FileNotFoundError``
    when reduction output is absent.

    Attributes
    ----------
    sample_name : str
        Sample identifier used in directory and file names.
    temperature_K : int
        Sample temperature in kelvin.
    excitation_wl_nm : int or float
        Pump-laser wavelength in nanometres.
    fluence_mJ_cm2 : float
        Pump fluence in mJ cm⁻².
    time_window_fs : int
        Width of the temporal averaging window in femtoseconds.
    delay_fs : int
        Pump–probe delay represented by this dataset, in femtoseconds.
    analysis_root : pathlib.Path
        Root of the standardized processed-analysis tree.
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
        """Store delay-series metadata and validate its path configuration."""
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
        """Return analysis dir.

        Returns
        -------
        Path
            Resolved path, label, or filename derived from experiment metadata.
        """
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
        """Return the standardized directory containing averaged detector-image arrays."""
        return self.analysis_dir() / "2D_images"

    def xy_folder(self) -> Path:
        """Return the standardized directory containing cached azimuthal XY integrations."""
        return self.analysis_dir() / "xy_files"

    def ensure_dirs(self) -> None:
        """Create the standardized XY-cache directory required by this dataset."""
        self.xy_folder().mkdir(parents=True, exist_ok=True)

    def img_path(self) -> Path:
        """Build the standardized ``.npy`` path for this delay point."""
        name = (
            f"{self.sample_name}_{self.temperature_K}K_{self._wl_tag}nm_"
            f"{self._flu_file}mJ_{int(self.time_window_fs)}fs_{int(self.delay_fs)}fs.npy"
        )
        return self.img_folder() / name

    def load_2d(self) -> np.ndarray:
        """Load the averaged detector image and validate that it is two-dimensional."""
        p = self.img_path()
        if not p.exists():
            raise FileNotFoundError(str(p))
        return np.load(str(p))

    def xy_path(self, azim_str: str) -> Path:
        """Build the cached XY filename for one azimuthal-window tag.

        The XY directory is created on demand before the path is returned.
        """
        self.ensure_dirs()
        name = (
            f"{self.sample_name}_{self.temperature_K}K_{self._wl_tag}nm_"
            f"{self._flu_file}mJ_{int(self.time_window_fs)}fs_{int(self.delay_fs)}fs_{azim_str}.xy"
        )
        return self.xy_folder() / name


class DarkDataset:
    """Describe one standardized dark/reference dataset.

    ``dark_tag`` identifies either a single scan or a combined scan group. The
    object derives the averaged image and azimuthal XY-cache paths shared by
    calibration, fitting, viewing, and differential workflows.

    Attributes
    ----------
    sample_name : str
        Sample identifier used in the standardized dark-data tree.
    temperature_K : int
        Sample temperature in kelvin.
    dark_tag : str
        Directory tag identifying a single scan or combined scan group.
    file_tag : str
        Filename-safe representation of ``dark_tag``.
    analysis_root : pathlib.Path
        Root directory containing processed analysis products.
    """

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
        """Store dark-scan metadata and validate its path configuration."""
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
        """Return the sample and temperature directory containing all dark datasets."""
        return (
            self.analysis_root
            / self.sample_name
            / f"temperature_{self.temperature_K}K"
            / "dark"
        )

    def _resolve_dark_tag(self, dark_tag: Optional[str]) -> str:
        """Use an explicit dark tag or discover the only usable dark dataset.

        Discovery accepts directories named ``scan_*`` or ``scans_*`` that
        contain a matching averaged detector image. Ambiguous discovery is
        rejected so callers cannot silently use the wrong reference.
        """
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
        """Return the processed directory for the selected dark dataset."""
        return self._dark_base() / self.dark_tag

    def img_folder(self) -> Path:
        """Return the directory containing averaged dark detector images."""
        return self.analysis_dir() / "2D_images"

    def xy_folder(self) -> Path:
        """Return the directory containing integrated dark XY patterns."""
        return self.analysis_dir() / "xy_files"

    def ensure_dirs(self) -> None:
        """Create the standardized XY-cache directory required by this dataset."""
        self.xy_folder().mkdir(parents=True, exist_ok=True)

    def img_path(self) -> Path:
        """Build the standardized averaged dark detector-image filename and path."""
        name = f"{self.sample_name}_{self.temperature_K}K_dark_{self.file_tag}.npy"
        return self.img_folder() / name

    def load_2d(self) -> np.ndarray:
        """Load the averaged detector image and validate that it is two-dimensional."""
        p = self.img_path()
        if not p.exists():
            raise FileNotFoundError(str(p))
        return np.load(str(p))

    def xy_path(self, azim_str: str) -> Path:
        """Build the cached dark-pattern path for an azimuthal-window tag."""
        self.ensure_dirs()
        name = f"{self.sample_name}_{self.temperature_K}K_dark_{self.file_tag}_{azim_str}.xy"
        return self.xy_folder() / name


class FluenceDataset:
    """One fluence point output by datared export (2D_images/*.npy), with XY output paths.

    Folder layout expected:
      .../<sample>/temperature_<T>K/excitation_wl_<wl>nm/fluence/delay_<delay>fs/time_window_<tw>fs/2D_images
      Files:
        <sample>_<T>K_<wl>nm_<flu>mJ_<tw>fs_<delay>fs.npy

    Attributes
    ----------
    sample_name : str
        Sample identifier used in directory and file names.
    temperature_K : int
        Sample temperature in kelvin.
    excitation_wl_nm : int or float
        Pump-laser wavelength in nanometres.
    fluence_mJ_cm2 : float
        Fluence represented by this dataset in mJ cm⁻².
    time_window_fs : int
        Averaging-window width in femtoseconds.
    delay_fs : int
        Fixed pump–probe delay in femtoseconds.
    analysis_root : pathlib.Path
        Root of the standardized processed-analysis tree.
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
        """Store fluence-series metadata and validate its path configuration."""
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
        """Return analysis dir.

        Returns
        -------
        Path
            Resolved path, label, or filename derived from experiment metadata.
        """
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
        """Return the standardized directory containing fluence-resolved detector images."""
        return self.analysis_dir() / "2D_images"

    def xy_folder(self) -> Path:
        """Return the standardized directory containing fluence-resolved integrated XY patterns."""
        return self.analysis_dir() / "xy_files"

    def ensure_dirs(self) -> None:
        """Create the standardized XY-cache directory required by this dataset."""
        self.xy_folder().mkdir(parents=True, exist_ok=True)

    def img_path(self) -> Path:
        """Build the standardized detector-image path for this fluence."""
        name = (
            f"{self.sample_name}_{self.temperature_K}K_{self._wl_tag}nm_"
            f"{self._flu_file}mJ_{int(self.time_window_fs)}fs_{int(self.delay_fs)}fs.npy"
        )
        return self.img_folder() / name

    def load_2d(self) -> np.ndarray:
        """Load the averaged detector image and validate that it is two-dimensional."""
        p = self.img_path()
        if not p.exists():
            raise FileNotFoundError(str(p))
        return np.load(str(p))

    def xy_path(self, azim_str: str) -> Path:
        """Build the cached XY path for an azimuthal-window tag."""
        self.ensure_dirs()
        name = (
            f"{self.sample_name}_{self.temperature_K}K_{self._wl_tag}nm_"
            f"{self._flu_file}mJ_{int(self.time_window_fs)}fs_{int(self.delay_fs)}fs_{azim_str}.xy"
        )
        return self.xy_folder() / name


class AzimIntegrator:
    """Integrate detector images into one- and two-dimensional patterns.

    The integrator loads a pyFAI PONI calibration and EDF mask, applies the
    configured azimuthal offset and optional polarization correction, and
    returns q in Å⁻¹. Cached XY files store two-theta and intensity; cache reads
    convert two-theta back to q using the PONI wavelength. Intensity
    normalization uses the finite mean inside ``q_norm_range``.

    Attributes
    ----------
    poni_path : str or None
        User-supplied pyFAI geometry file.
    mask_edf_path : str or None
        EDF detector mask; nonzero pixels are excluded by pyFAI.
    npt : int
        Default number of radial integration bins.
    normalize : bool
        Whether integrated patterns are normalized over ``q_norm_range``.
    q_norm_range : tuple of float
        q interval in Å⁻¹ used to compute the normalization mean.
    azim_offset_deg : float
        Offset mapping package azimuth coordinates to pyFAI coordinates.
    polarization_factor : float or None
        pyFAI polarization correction in the interval ``[-1, 1]``.
    default_poni_path : str or None
        Geometry file used for lazy initialization when ``poni_path`` is absent.
    _ai : object or None
        Loaded pyFAI azimuthal integrator.
    _mask : numpy.ndarray or None
        Boolean detector mask in the unmodified detector-array orientation.
    """
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
        polarization_factor: Optional[float] = None,
        default_poni_path: Optional[Union[str, Path]] = None,
    ):
        """Load calibration and mask data and configure integration corrections."""
        self.poni_path = None if poni_path is None else str(poni_path)
        self.mask_edf_path = None if mask_edf_path is None else str(mask_edf_path)
        self.default_poni_path = None if default_poni_path is None else str(default_poni_path)
        self.poni_verbose = poni_verbose

        self.npt = int(npt)
        self.normalize = bool(normalize)
        self.q_norm_range = (float(q_norm_range[0]), float(q_norm_range[1]))
        self.azim_offset_deg = float(azim_offset_deg)
        self.polarization_factor = normalize_polarization_factor(
            polarization_factor
        )

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
            self._mask = np.asarray(fabio.open(self.mask_edf_path).data) != 0

    @staticmethod
    def build_windows(
        azimuthal_edges: np.ndarray,
        *,
        include_full: bool = True,
        full_range: Tuple[float, float] = (-180, 180),
    ) -> List[Tuple[float, float]]:
        """Construct validated azimuthal integration windows.

        Parameters
        ----------
        azimuthal_edges : np.ndarray
            Ordered azimuthal edges in degrees.
        include_full : bool
            Whether to include an additional pattern integrated over ``full_range``.
        full_range : Tuple[float, float]
            Azimuthal limits in degrees for the optional full-range pattern.

        Returns
        -------
        List[Tuple[float, float]]
            Validated azimuthal windows in degrees, including ``full_range``
            when requested.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
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
        """Integrate one detector image over an azimuthal window.

        Parameters
        ----------
        img : np.ndarray
            Two-dimensional detector image.
        azimuthal_range : Tuple[float, float]
            Lower and upper azimuthal integration limits in degrees.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            q values in Å⁻¹ and the corresponding integrated intensities.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
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
            polarization_factor=self.polarization_factor,
            unit="q_A^-1",
        )

        if self.normalize:
            I = general_utils.normalize_y_by_mean_in_xrange(
                q,
                I,
                self.q_norm_range,
            )

        return np.asarray(q), np.asarray(I)

    def integrate2d(
        self,
        img: np.ndarray,
        *,
        npt_rad: Optional[int] = None,
        npt_azim: int = 360,
        radial_range: Optional[Tuple[float, float]] = None,
        azimuthal_range: Tuple[float, float] = (-90.0, 90.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cake one detector image onto radial and azimuthal coordinates.

        Parameters
        ----------
        img : np.ndarray
            Two-dimensional detector image.
        npt_rad : Optional[int]
            Number of radial q bins. The integrator's configured ``npt`` is
            used when omitted.
        npt_azim : int
            Number of azimuthal bins.
        radial_range : Optional[Tuple[float, float]]
            Optional q limits in Å⁻¹.
        azimuthal_range : Tuple[float, float]
            Display-coordinate azimuthal limits in degrees. The configured
            offset is applied only for the pyFAI call and removed again from
            the returned azimuthal coordinates.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Cake intensity with shape ``(npt_azim, npt_rad)``, q coordinates
            in Å⁻¹, and display-coordinate azimuths in degrees.
        """
        if self._ai is None:
            raise ValueError("AzimIntegrator has no PONI loaded (poni_path=None).")

        image = np.asarray(img)
        if image.ndim != 2:
            raise ValueError(f"img must be two-dimensional, got shape {image.shape}.")

        radial_bins = self.npt if npt_rad is None else int(npt_rad)
        azimuthal_bins = int(npt_azim)
        if radial_bins < 1:
            raise ValueError("npt_rad must be at least 1.")
        if azimuthal_bins < 1:
            raise ValueError("npt_azim must be at least 1.")

        phi0, phi1 = (float(azimuthal_range[0]), float(azimuthal_range[1]))
        if not np.isfinite(phi0) or not np.isfinite(phi1) or phi1 <= phi0:
            raise ValueError("azimuthal_range must contain two increasing finite values.")
        azimuthal_width = phi1 - phi0
        if azimuthal_width > 360.0 + 1e-9:
            raise ValueError("azimuthal_range cannot span more than 360 degrees.")

        radial_limits = None
        if radial_range is not None:
            radial_limits = (float(radial_range[0]), float(radial_range[1]))
            if (
                not np.isfinite(radial_limits[0])
                or not np.isfinite(radial_limits[1])
                or radial_limits[1] <= radial_limits[0]
            ):
                raise ValueError("radial_range must contain two increasing finite values.")

        def run_pyfai(
            bins: int,
            pyfai_range: Tuple[float, float],
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Run pyFAI integration for one interval that does not cross its angular boundary."""
            result = self._ai.integrate2d(
                image,
                npt_rad=radial_bins,
                npt_azim=int(bins),
                mask=self._mask,
                radial_range=radial_limits,
                azimuth_range=tuple(float(v) for v in pyfai_range),
                polarization_factor=self.polarization_factor,
                unit="q_A^-1",
            )

            if all(
                hasattr(result, name)
                for name in ("intensity", "radial", "azimuthal")
            ):
                part_intensity = np.asarray(result.intensity, dtype=float)
                part_q = np.asarray(result.radial, dtype=float)
                part_azimuth = np.asarray(result.azimuthal, dtype=float)
            else:
                part_intensity = np.asarray(result[0], dtype=float)
                part_q = np.asarray(result[1], dtype=float)
                part_azimuth = np.asarray(result[2], dtype=float)
            return part_intensity, part_q, part_azimuth

        is_full_circle = np.isclose(azimuthal_width, 360.0, rtol=0.0, atol=1e-9)
        if is_full_circle:
            intensity, q, pyfai_azimuth = run_pyfai(
                azimuthal_bins,
                (-180.0, 180.0),
            )
        else:
            internal_start = (
                (phi0 + self.azim_offset_deg + 180.0) % 360.0
            ) - 180.0
            internal_end = internal_start + azimuthal_width

            if internal_end <= 180.0 + 1e-9:
                intensity, q, pyfai_azimuth = run_pyfai(
                    azimuthal_bins,
                    (internal_start, min(internal_end, 180.0)),
                )
            else:
                if azimuthal_bins < 2:
                    raise ValueError(
                        "npt_azim must be at least 2 when the requested azimuthal "
                        "range crosses pyFAI's -180/180-degree boundary."
                    )
                first_width = 180.0 - internal_start
                first_bins = int(round(azimuthal_bins * first_width / azimuthal_width))
                first_bins = min(max(first_bins, 1), azimuthal_bins - 1)
                second_bins = azimuthal_bins - first_bins

                first_intensity, first_q, first_azimuth = run_pyfai(
                    first_bins,
                    (internal_start, 180.0),
                )
                second_intensity, second_q, second_azimuth = run_pyfai(
                    second_bins,
                    (-180.0, internal_end - 360.0),
                )
                if not np.allclose(first_q, second_q, equal_nan=True):
                    raise ValueError(
                        "pyFAI returned inconsistent radial coordinates across a "
                        "wrapped azimuthal integration."
                    )
                intensity = np.vstack([first_intensity, second_intensity])
                q = first_q
                pyfai_azimuth = np.concatenate([first_azimuth, second_azimuth])

        azimuth = (
            (pyfai_azimuth - self.azim_offset_deg - phi0) % 360.0
        ) + phi0
        order = np.argsort(azimuth)
        azimuth = azimuth[order]
        intensity = intensity[order]

        expected_shape = (azimuthal_bins, radial_bins)
        if intensity.shape != expected_shape:
            raise ValueError(
                "pyFAI returned an unexpected 2D integration shape: "
                f"{intensity.shape}, expected {expected_shape}."
            )
        if q.shape != (radial_bins,) or azimuth.shape != (azimuthal_bins,):
            raise ValueError(
                "pyFAI returned coordinate arrays inconsistent with the requested "
                f"binning: q={q.shape}, azimuth={azimuth.shape}."
            )

        if self.normalize:
            intensity = np.vstack(
                [
                    general_utils.normalize_y_by_mean_in_xrange(
                        q,
                        row,
                        self.q_norm_range,
                    )
                    for row in intensity
                ]
            )

        return intensity, q, azimuth

    def _ensure_ai_loaded(self) -> None:
        """Lazily load the configured fallback PONI geometry when necessary."""
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
        """Load or compute one azimuthally integrated XY pattern.

        Parameters
        ----------
        dataset : Union[DelayDataset, DarkDataset, FluenceDataset]
            Delay, fluence, or dark dataset that provides the 2D image and XY cache paths.
        azimuthal_range : Tuple[float, float]
            Lower and upper azimuthal integration limits in degrees.
        compute_if_missing : bool
            Compatibility flag retained by higher-level APIs. A missing cache is
            currently integrated regardless of this value.
        overwrite_xy : bool
            Whether existing XY cache files should be recomputed.

        Returns
        -------
        Tuple[str, np.ndarray, np.ndarray]
            Canonical azimuthal tag, q grid in Å⁻¹, and intensity array.
        """
        self._ensure_ai_loaded()

        azim_str = general_utils.azim_range_str((azimuthal_range[0], azimuthal_range[1]))
        xy_path = dataset.xy_path(azim_str)

        if (not overwrite_xy) and xy_path.exists():
            two_theta, I = _LOAD_XY(xy_path)
            q = general_utils.two_theta_to_q(two_theta, self._ai.wavelength)
            if self.normalize:
                I = general_utils.normalize_y_by_mean_in_xrange(
                    q,
                    I,
                    self.q_norm_range,
                )
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
        """Integrate and cache every requested azimuthal window.

        Parameters
        ----------
        dataset : Union[DelayDataset, DarkDataset, FluenceDataset]
            Delay, fluence, or dark dataset that provides the 2D image and XY cache paths.
        azimuthal_edges : np.ndarray
            Ordered azimuthal edges in degrees.
        include_full : bool
            Whether to include an additional pattern integrated over ``full_range``.
        full_range : Tuple[float, float]
            Azimuthal limits in degrees for the optional full-range pattern.
        overwrite_xy : bool
            Whether existing XY cache files should be recomputed.

        Returns
        -------
        Dict[str, Tuple[np.ndarray, np.ndarray]]
            Mapping from azimuthal tags to ``(q, intensity)`` arrays.

        Notes
        -----
        This operation may create or replace analysis artifacts according to its save and overwrite settings.
        """
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
                if self.normalize:
                    I = general_utils.normalize_y_by_mean_in_xrange(
                        q,
                        I,
                        self.q_norm_range,
                    )
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
    """Identify the dataset used as an analysis reference.

    ``ref_type`` selects a delay or dark reference. ``ref_value`` is therefore
    interpreted as a femtosecond delay, scan number, scan collection, or
    existing dark-tag string.

    Attributes
    ----------
    ref_type : str
        Reference family, currently ``"delay"`` or ``"dark"``.
    ref_value : int, str, or sequence of int
        Delay value, scan identifier, combined scans, or an existing dark tag.
    """
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
    """Discover delay points available for a standardized experiment dataset.

    The search can inspect averaged 2D ``.npy`` files or integrated ``.xy``
    files. Returned delays are unique, sorted, and expressed in femtoseconds.

    Parameters
    ----------
    sample_name, temperature_K, excitation_wl_nm, fluence_mJ_cm2,
    time_window_fs
        Experiment identity encoded in standardized paths and filenames.
    path_root, analysis_subdir
        Legacy analysis-path configuration.
    from_2D_imgs : bool
        Search image files; false searches integrated XY files.

    Returns
    -------
    list of int
        Unique sorted delay points in femtoseconds.

    Raises
    ------
    FileNotFoundError
        If the expected directory or matching files do not exist.
    """
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
        type_dummy=r"_(-?\d+)_(-?\d+)"
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
    """Normalize a delay selector to a list of integer femtosecond values.

    The special value ``"all"`` discovers points from the experiment folders;
    an integer becomes a one-element list and sequences preserve their order.

    Parameters
    ----------
    delays_fs : int, sequence of int, or "all"
        Explicit delay selection or discovery request.
    sample_name, temperature_K, excitation_wl_nm, fluence_mJ_cm2,
    time_window_fs
        Experiment identity used only for automatic discovery.
    path_root, analysis_subdir, from_2D_imgs
        Discovery path and source-file controls.

    Returns
    -------
    list of int
        Normalized delay values in femtoseconds.
    """
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
    
    """Discover excitation fluences available at one pump-probe delay.

    Values are decoded from standardized 2D-image or XY filenames and returned
    as unique, ascending values in mJ/cm².

    Parameters
    ----------
    sample_name, temperature_K, excitation_wl_nm, delay_fs, time_window_fs
        Experiment identity encoded in standardized paths and filenames.
    path_root, analysis_subdir
        Legacy analysis-path configuration.
    from_2D_imgs : bool
        Search image files; false searches integrated XY files.

    Returns
    -------
    list of float
        Unique sorted excitation fluences in mJ/cm².

    Raises
    ------
    FileNotFoundError
        If the expected directory or matching files do not exist.
    """
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
        type_dummy=r"_(-?\d+)_(-?\d+)"
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
    """Normalize a fluence selector to ascending values in mJ/cm².

    The special value ``"all"`` discovers points from the experiment folders.
    Scalar values become one-element lists.

    Parameters
    ----------
    fluences_mJ_cm2 : float, sequence of float, or "all"
        Explicit fluence selection or discovery request.
    sample_name, temperature_K, excitation_wl_nm, delay_fs, time_window_fs
        Experiment identity used only for automatic discovery.
    path_root, analysis_subdir, from_2D_imgs
        Discovery path and source-file controls.

    Returns
    -------
    list of float
        Normalized ascending fluence values in mJ/cm².
    """
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
    """Convert a scan spec into the folder tag used by datared:
    167246 -> "scan_167246"
    [167246,167285] -> "scans_167246-167285"
    "scans_..." -> as-is
    """
    if isinstance(scan_spec, str):
        return scan_spec
    return general_utils.scan_tag(scan_spec)


def pretty_dark_tag(dark_tag: str) -> str:
    """Format a dark-scan folder tag as a compact multiline plot label."""
    return str(dark_tag).replace("_", ":\n").replace("-", "\n")


def delay_label_value(delay_fs: Union[int, float], *, fs_or_ps: str = "ps", digits: int=2) -> Union[int, float]:
    """Convert a femtosecond delay to a rounded value suitable for plot labels.

    Very small nonzero values retain significant digits instead of being shown
    as zero in coarse display units.
    """
    unit = general_utils.normalize_time_unit(fs_or_ps)
    value = float(general_utils.time_values_from_fs(delay_fs, unit))
    if unit == "fs":
        return int(round(value))
    rounded = round(value, int(digits))
    if value != 0.0 and rounded == 0.0:
        # Avoid collapsing real ns/µs/ms/s values to a misleading ``0.0``
        # when the legacy decimal-place setting is too coarse.
        return float(f"{value:.{max(int(digits), 1)}g}")
    return rounded


def evenly_spaced_subset(values, max_count: Optional[int]):
    """Return up to ``max_count`` values sampled evenly over their current order."""
    items = list(values)
    if max_count is None:
        return items
    n = int(max_count)
    if n < 0:
        return items
    if n == 0:
        return []
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n)
    picked = []
    seen = set()
    for i in np.rint(idx).astype(int):
        ii = int(min(max(i, 0), len(items) - 1))
        if ii not in seen:
            picked.append(items[ii])
            seen.add(ii)
    cursor = 0
    while len(picked) < n and cursor < len(items):
        if cursor not in seen:
            picked.append(items[cursor])
            seen.add(cursor)
        cursor += 1
    return picked
