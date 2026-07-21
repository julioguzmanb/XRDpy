"""SACLA single-shot detector integration and final-pattern aggregation.

The metadata HDF5 file remains the authority for selected ``(run, tag)``
pairs. Raw facility access is isolated in :class:`SaclaFacilityClient`, which
uses the SACLA ``stpy`` and ``dbpy`` APIs directly. Importing this module does
not require either facility package, so cached single-shot patterns can be
aggregated on a normal analysis workstation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from tqdm import tqdm

from ..common import azimint_utils, general_utils
from ..common.paths import AnalysisPaths


SINGLE_SHOT_FOLDER = "single_shot_1D_patterns"
DEFAULT_DETECTOR_ID = "MPCCD-8N0-3-002"
DEFAULT_INTENSITY_COLUMN = "xfel_bl_3_st_2_pd_user_9_fitting_peak/voltage"
_SETTINGS_HEADER = "# xrdpy_single_shot_settings_sha256="


@dataclass(frozen=True)
class SingleShotRecord:
    """One SACLA detector-frame selection from the metadata HDF5 file."""

    scan_type: str
    run: int
    tag: int
    sample_name: str
    temperature_K: int
    excitation_wl_nm: Optional[float]
    fluence_mJ_cm2: Optional[float]
    time_window_fs: Optional[int]
    delay_fs: Optional[int]
    dark_file_tag: Optional[str]
    pulse_intensity: Optional[float]

    @property
    def group_key(self) -> Union[int, float, str]:
        """Return the delay, fluence, or dark value represented by this shot."""
        if self.scan_type == "delay":
            return int(self.delay_fs)  # type: ignore[arg-type]
        if self.scan_type == "fluence":
            return float(self.fluence_mJ_cm2)  # type: ignore[arg-type]
        return "dark"

    def group_folder(self) -> Optional[str]:
        """Return the optional delay/fluence cache directory component."""
        if self.scan_type == "delay":
            return "delay_{}fs".format(int(self.delay_fs))
        if self.scan_type == "fluence":
            tag = general_utils.fluence_tag_file(float(self.fluence_mJ_cm2))
            return "fluence_{}mJ".format(tag)
        return None

    def final_stem(self, azimuth_tag: str) -> str:
        """Return the canonical final-pattern stem before run/tag identity."""
        if self.scan_type == "dark":
            return "{}_{}K_dark_{}_{}".format(
                self.sample_name,
                self.temperature_K,
                self.dark_file_tag,
                azimuth_tag,
            )
        wl_tag = general_utils.wl_tag_nm(float(self.excitation_wl_nm))
        flu_tag = general_utils.fluence_tag_file(float(self.fluence_mJ_cm2))
        return "{}_{}K_{}nm_{}mJ_{}fs_{}fs_{}".format(
            self.sample_name,
            self.temperature_K,
            wl_tag,
            flu_tag,
            int(self.time_window_fs),
            int(self.delay_fs),
            azimuth_tag,
        )


class SaclaFacilityClient:
    """Direct, lazily imported adapter for SACLA detector and database APIs."""

    def __init__(
        self,
        *,
        beamline: int,
        detector_id: str = DEFAULT_DETECTOR_ID,
        stpy_module=None,
        dbpy_module=None,
    ):
        if stpy_module is None:
            try:
                import stpy as stpy_module
            except ImportError as exc:
                raise ImportError(
                    "SACLA single-shot production requires the facility stpy module. "
                    "Final aggregation from an existing cache does not."
                ) from exc
        self.beamline = int(beamline)
        self.detector_id = str(detector_id)
        self._stpy = stpy_module
        self._dbpy = dbpy_module
        self._readers: Dict[int, Tuple[object, object]] = {}

    def _reader_for_run(self, run: int):
        """Create and cache one typed ``stpy`` reader/buffer pair per run."""
        run = int(run)
        if run not in self._readers:
            reader = self._stpy.StorageReader(
                self.detector_id,
                self.beamline,
                (run,),
            )
            buffer = self._stpy.StorageBuffer(reader)
            self._readers[run] = (reader, buffer)
        return self._readers[run]

    def read_image(self, run: int, tag: int) -> np.ndarray:
        """Read one detector frame through direct ``stpy`` method calls."""
        reader, buffer = self._reader_for_run(run)
        reader.collect(buffer, int(tag))
        return np.asarray(buffer.read_det_data(0))

    def read_pulse_intensities(
        self,
        run: int,
        tags: Sequence[int],
        intensity_col: str,
    ) -> np.ndarray:
        """Read one run's selected pulse intensities in a single ``dbpy`` call."""
        if self._dbpy is None:
            try:
                import dbpy as dbpy_module
            except ImportError as exc:
                raise ImportError(
                    "Legacy SACLA metadata without stored pulse intensities requires "
                    "the facility dbpy module."
                ) from exc
            self._dbpy = dbpy_module
        hightag = self._dbpy.read_hightagnumber(self.beamline, int(run))
        values = self._dbpy.read_syncdatalist_float(
            str(intensity_col),
            hightag,
            tuple(int(tag) for tag in tags),
        )
        return np.asarray(values, dtype=float)


def _read_scalar(group: h5py.Group, name: str, default=None):
    """Read and decode one scalar dataset or attribute."""
    if name in group:
        value = group[name][()]
    elif name in group.attrs:
        value = group.attrs[name]
    else:
        return default
    value = general_utils.decode_if_bytes(value)
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return value


def _dark_file_tag(dark_tag: str) -> str:
    """Translate a SACLA dark directory tag to its canonical filename token."""
    if dark_tag.startswith("scan_"):
        return "scan{}".format(dark_tag.split("_", 1)[1])
    if dark_tag.startswith("scans_"):
        return "scans{}".format(dark_tag.split("_", 1)[1])
    return dark_tag.replace("_", "")


def _scan_group_records(
    *,
    scans_group: h5py.Group,
    scan_type: str,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: Optional[float],
    fluence_mJ_cm2: Optional[float],
    time_window_fs: Optional[int],
    delay_fs: Optional[int],
    dark_file_tag: Optional[str],
) -> List[SingleShotRecord]:
    """Read aligned tags and optional pulse intensities below a scans group."""
    records: List[SingleShotRecord] = []
    for run_name in sorted(
        (name for name in scans_group.keys() if str(name).isdigit()),
        key=int,
    ):
        run_group = scans_group[run_name]
        if "tags" not in run_group:
            continue
        tags = np.asarray(run_group["tags"], dtype=np.int64)
        if "pulse_intensity" in run_group:
            intensities = np.asarray(run_group["pulse_intensity"], dtype=float)
            if intensities.shape != tags.shape:
                raise ValueError(
                    "SACLA metadata tags and pulse_intensity arrays have different "
                    "shapes below {}.".format(run_group.name)
                )
        else:
            intensities = np.full(tags.shape, np.nan, dtype=float)

        seen = set()
        for tag, intensity in zip(tags.tolist(), intensities.tolist()):
            identity = int(tag)
            if identity in seen:
                continue
            seen.add(identity)
            records.append(
                SingleShotRecord(
                    scan_type=scan_type,
                    run=int(run_name),
                    tag=identity,
                    sample_name=sample_name,
                    temperature_K=temperature_K,
                    excitation_wl_nm=excitation_wl_nm,
                    fluence_mJ_cm2=fluence_mJ_cm2,
                    time_window_fs=time_window_fs,
                    delay_fs=delay_fs,
                    dark_file_tag=dark_file_tag,
                    pulse_intensity=(
                        float(intensity) if np.isfinite(float(intensity)) else None
                    ),
                )
            )
    return records


def _metadata_records(
    metadata_h5_path: Union[str, Path],
) -> Tuple[Dict[str, object], List[SingleShotRecord]]:
    """Read SACLA experiment identity and selected ``(run, tag)`` records."""
    metadata_path = Path(metadata_h5_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(str(metadata_path))
    analysis_dir = (
        metadata_path.parent.parent
        if metadata_path.parent.name == "metadata"
        else metadata_path.parent
    )

    records: List[SingleShotRecord] = []
    with h5py.File(metadata_path, "r") as handle:
        if "meta" not in handle:
            raise ValueError(
                "Invalid SACLA metadata HDF5 (missing /meta): {}".format(metadata_path)
            )
        meta_group = handle["meta"]
        scan_type = str(_read_scalar(meta_group, "scan_type", "")).strip().lower()
        if scan_type == "calibration":
            scan_type = "dark"
        if scan_type not in {"dark", "delay", "fluence"}:
            raise ValueError(
                "SACLA metadata scan_type must be 'dark', 'delay', or 'fluence'."
            )

        sample_name = str(_read_scalar(meta_group, "sample_name"))
        temperature_K = int(_read_scalar(meta_group, "temperature_K"))
        excitation = _read_scalar(meta_group, "excitation_wl_nm")
        time_window = _read_scalar(meta_group, "time_window_fs")
        beamline = int(_read_scalar(meta_group, "beamline", 3))
        intensity_col = str(
            _read_scalar(meta_group, "intensity_col", DEFAULT_INTENSITY_COLUMN)
        )
        metadata: Dict[str, object] = {
            "path": metadata_path,
            "analysis_dir": analysis_dir,
            "scan_type": scan_type,
            "sample_name": sample_name,
            "temperature_K": temperature_K,
            "excitation_wl_nm": None if excitation is None else float(excitation),
            "time_window_fs": None if time_window is None else int(time_window),
            "beamline": beamline,
            "intensity_col": intensity_col,
        }

        if scan_type == "dark":
            if "scans" not in handle:
                raise ValueError("Invalid SACLA dark metadata HDF5 (missing /scans).")
            dark_tag = analysis_dir.name
            metadata["dark_tag"] = dark_tag
            records.extend(
                _scan_group_records(
                    scans_group=handle["scans"],
                    scan_type="dark",
                    sample_name=sample_name,
                    temperature_K=temperature_K,
                    excitation_wl_nm=None,
                    fluence_mJ_cm2=None,
                    time_window_fs=None,
                    delay_fs=None,
                    dark_file_tag=_dark_file_tag(dark_tag),
                )
            )
        elif scan_type == "delay":
            if "delays" not in handle:
                raise ValueError("Invalid SACLA delay metadata HDF5 (missing /delays).")
            fluence = float(_read_scalar(meta_group, "fluence_mJ_cm2"))
            metadata["fluence_mJ_cm2"] = fluence
            for name in sorted(
                handle["delays"].keys(),
                key=lambda value: int(str(value).replace("fs", "")),
            ):
                group = handle["delays"][name]
                delay = int(group.attrs.get("delay_fs", str(name).replace("fs", "")))
                if "scans" not in group:
                    continue
                records.extend(
                    _scan_group_records(
                        scans_group=group["scans"],
                        scan_type="delay",
                        sample_name=sample_name,
                        temperature_K=temperature_K,
                        excitation_wl_nm=float(excitation),
                        fluence_mJ_cm2=fluence,
                        time_window_fs=int(time_window),
                        delay_fs=delay,
                        dark_file_tag=None,
                    )
                )
        else:
            if "fluences" not in handle:
                raise ValueError(
                    "Invalid SACLA fluence metadata HDF5 (missing /fluences)."
                )
            fixed_delay = int(_read_scalar(meta_group, "delay_fs"))
            metadata["delay_fs"] = fixed_delay
            for name in sorted(
                handle["fluences"].keys(),
                key=lambda value: float(str(value).replace("mJ", "").replace("p", ".")),
            ):
                group = handle["fluences"][name]
                fluence = float(group.attrs["fluence_mJ_cm2"])
                delay = int(group.attrs.get("delay_fs", fixed_delay))
                if "scans" not in group:
                    continue
                records.extend(
                    _scan_group_records(
                        scans_group=group["scans"],
                        scan_type="fluence",
                        sample_name=sample_name,
                        temperature_K=temperature_K,
                        excitation_wl_nm=float(excitation),
                        fluence_mJ_cm2=fluence,
                        time_window_fs=int(time_window),
                        delay_fs=delay,
                        dark_file_tag=None,
                    )
                )

    metadata["delays_fs"] = sorted(
        {int(record.delay_fs) for record in records if record.delay_fs is not None}
    )
    metadata["fluences_mJ_cm2"] = sorted(
        {
            float(record.fluence_mJ_cm2)
            for record in records
            if record.fluence_mJ_cm2 is not None
        }
    )
    return metadata, records


def _file_sha256(path: Optional[Union[str, Path]]) -> Optional[str]:
    """Return a stable content digest for a correction or geometry file."""
    if path is None or not str(path).strip():
        return None
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(str(source))
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_background_path(
    *,
    background: Optional[Union[int, str, Path]],
    background_path: Optional[Union[str, Path]],
    paths: Optional[AnalysisPaths],
) -> Optional[Path]:
    """Resolve either an explicit image path or a processed SACLA background run."""
    if background_path is not None and str(background_path).strip():
        resolved = Path(background_path).expanduser().resolve()
    elif background is None or not str(background).strip():
        return None
    else:
        if paths is None:
            raise ValueError(
                "paths=AnalysisPaths(...) is required when background is a SACLA run."
            )
        run = int(background)
        resolved = Path(paths.analysis_root) / str(run) / "{}.npy".format(run)
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return resolved


def _settings_signature(
    *,
    poni_sha256: str,
    mask_sha256: Optional[str],
    background_sha256: Optional[str],
    npt: int,
    azimuthal_range: Tuple[float, float],
    azim_offset_deg: float,
    polarization_factor: Optional[float],
    beamline: int,
    detector_id: str,
    threshold_counts: float,
    intensity_col: str,
    pulse_normalized: bool,
) -> str:
    """Fingerprint every setting that changes a SACLA single-shot pattern."""
    payload = {
        "schema": 2,
        "poni_sha256": poni_sha256,
        "mask_sha256": mask_sha256,
        "background_sha256": background_sha256,
        "npt": int(npt),
        "azimuthal_range_deg": [
            float(azimuthal_range[0]),
            float(azimuthal_range[1]),
        ],
        "azim_offset_deg": float(azim_offset_deg),
        "polarization_factor": (
            None if polarization_factor is None else float(polarization_factor)
        ),
        "beamline": int(beamline),
        "detector_id": str(detector_id),
        "threshold_counts": float(threshold_counts),
        "intensity_col": str(intensity_col),
        "pulse_normalized": bool(pulse_normalized),
        "single_shot_normalized": False,
        "x_coordinate": "two_theta_deg",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _single_shot_path(
    analysis_dir: Path,
    record: SingleShotRecord,
    azimuth_tag: str,
) -> Path:
    """Return the agreed azimuth/group/run hierarchy and SACLA tag filename."""
    folder = analysis_dir / SINGLE_SHOT_FOLDER / azimuth_tag
    group_folder = record.group_folder()
    if group_folder is not None:
        folder = folder / group_folder
    folder = folder / "scan_{}".format(record.run)
    filename = "{}_scan{}_tag{}.xy".format(
        record.final_stem(azimuth_tag),
        record.run,
        record.tag,
    )
    return folder / filename


def _save_single_shot_xy_atomic(
    path: Path,
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    settings_signature: str,
) -> None:
    """Atomically publish one fingerprinted single-shot XY file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            stream.write("{}{}\n".format(_SETTINGS_HEADER, settings_signature))
            np.savetxt(stream, np.column_stack((two_theta, intensity)))
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _save_final_xy_atomic(path: Path, x: np.ndarray, intensity: np.ndarray) -> None:
    """Atomically replace a final XY file during on-the-fly refresh."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name), suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        general_utils.save_xy(temporary_path, x, intensity)
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_single_shot_xy(path: Path) -> Tuple[str, np.ndarray, np.ndarray]:
    """Load one complete single-shot file and fingerprint with one open."""
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip()
        values = np.asarray(np.loadtxt(stream), dtype=float)
    if not header.startswith(_SETTINGS_HEADER):
        raise ValueError(
            "Single-shot pattern has no XRDpy settings fingerprint: {}".format(path)
        )
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(
            "Expected two XY columns in single-shot pattern: {}".format(path)
        )
    signature = header[len(_SETTINGS_HEADER):].strip()
    return signature, values[:, 0], values[:, 1]


def _fill_missing_intensities(
    records: Sequence[SingleShotRecord],
    *,
    client,
    intensity_col: str,
) -> List[SingleShotRecord]:
    """Batch-load pulse intensities only for legacy metadata that omits them."""
    if not records or records[0].scan_type == "dark":
        return list(records)
    missing_by_run: Dict[int, List[int]] = {}
    for record in records:
        if record.pulse_intensity is None:
            missing_by_run.setdefault(record.run, []).append(record.tag)
    if not missing_by_run:
        return list(records)

    resolved: Dict[Tuple[int, int], float] = {}
    for run, run_tags in sorted(missing_by_run.items()):
        unique_tags = sorted(set(int(tag) for tag in run_tags))
        values = np.asarray(
            client.read_pulse_intensities(run, unique_tags, intensity_col),
            dtype=float,
        )
        if values.size != len(unique_tags):
            raise ValueError(
                "dbpy returned {} pulse intensities for {} selected tags in run {}.".format(
                    values.size, len(unique_tags), run
                )
            )
        for tag, value in zip(unique_tags, values.tolist()):
            resolved[(run, tag)] = float(value)

    return [
        replace(
            record,
            pulse_intensity=resolved.get((record.run, record.tag)),
        )
        if record.pulse_intensity is None
        else record
        for record in records
    ]


def _integration_context(
    *,
    metadata: Dict[str, object],
    poni_path: Union[str, Path],
    mask_edf_path: Optional[Union[str, Path]],
    background: Optional[Union[int, str, Path]],
    background_path: Optional[Union[str, Path]],
    paths: Optional[AnalysisPaths],
    beamline: Optional[int],
    detector_id: str,
    threshold_counts: float,
    intensity_col: Optional[str],
) -> Dict[str, object]:
    """Resolve shared preprocessing values for production and aggregation."""
    resolved_background = _resolve_background_path(
        background=background,
        background_path=background_path,
        paths=paths,
    )
    return {
        "beamline": int(metadata["beamline"] if beamline is None else beamline),
        "detector_id": str(detector_id),
        "threshold_counts": float(threshold_counts),
        "intensity_col": str(
            metadata["intensity_col"] if intensity_col is None else intensity_col
        ),
        "background_path": resolved_background,
        "poni_sha256": _file_sha256(poni_path),
        "mask_sha256": _file_sha256(mask_edf_path),
        "background_sha256": _file_sha256(resolved_background),
    }


def integrate_single_shot_1d(
    *,
    metadata_h5_path: Union[str, Path],
    poni_path: Union[str, Path],
    mask_edf_path: Optional[Union[str, Path]],
    azimuthal_edges: Sequence[float],
    include_full: bool = True,
    full_range: Tuple[float, float] = (-90.0, 90.0),
    npt: int = 1000,
    overwrite: bool = False,
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    beamline: Optional[int] = None,
    detector_id: str = DEFAULT_DETECTOR_ID,
    background: Optional[Union[int, str, Path]] = None,
    background_path: Optional[Union[str, Path]] = None,
    threshold_counts: float = 40.0,
    intensity_col: Optional[str] = None,
    paths: Optional[AnalysisPaths] = None,
    facility_client=None,
    chunk_id: int = 1,
    n_chunks: int = 1,
) -> Dict[str, object]:
    """Integrate one deterministic array-job partition of selected SACLA tags."""
    metadata, records = _metadata_records(metadata_h5_path)
    if not records:
        raise ValueError("The SACLA metadata file contains no selected detector tags.")
    context = _integration_context(
        metadata=metadata,
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        background=background,
        background_path=background_path,
        paths=paths,
        beamline=beamline,
        detector_id=detector_id,
        threshold_counts=threshold_counts,
        intensity_col=intensity_col,
    )
    integrator = azimint_utils.AzimIntegrator(
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize=False,
        azim_offset_deg=float(azim_offset_deg),
        polarization_factor=polarization_factor,
    )
    windows = integrator.build_windows(
        np.asarray(azimuthal_edges, dtype=float),
        include_full=bool(include_full),
        full_range=(float(full_range[0]), float(full_range[1])),
    )
    window_info = []
    pulse_normalized = metadata["scan_type"] in {"delay", "fluence"}
    for window in windows:
        azimuth_tag = general_utils.azim_range_str(window)
        signature = _settings_signature(
            poni_sha256=str(context["poni_sha256"]),
            mask_sha256=context["mask_sha256"],
            background_sha256=context["background_sha256"],
            npt=int(npt),
            azimuthal_range=window,
            azim_offset_deg=float(azim_offset_deg),
            polarization_factor=integrator.polarization_factor,
            beamline=int(context["beamline"]),
            detector_id=str(context["detector_id"]),
            threshold_counts=float(context["threshold_counts"]),
            intensity_col=str(context["intensity_col"]),
            pulse_normalized=pulse_normalized,
        )
        window_info.append((window, azimuth_tag, signature))

    records_by_identity: Dict[Tuple[int, int], List[SingleShotRecord]] = {}
    for record in records:
        records_by_identity.setdefault((record.run, record.tag), []).append(record)

    chunk_id = int(chunk_id)
    n_chunks = int(n_chunks)
    if n_chunks < 1:
        raise ValueError("n_chunks must be at least 1.")
    if chunk_id < 1 or chunk_id > n_chunks:
        raise ValueError(
            "chunk_id must be between 1 and n_chunks (received {} of {}).".format(
                chunk_id,
                n_chunks,
            )
        )
    all_identities = sorted(records_by_identity)
    identity_indices = np.array_split(np.arange(len(all_identities)), n_chunks)
    selected_identities = [
        all_identities[int(index)] for index in identity_indices[chunk_id - 1]
    ]
    records_by_identity = {
        identity: records_by_identity[identity]
        for identity in selected_identities
    }

    report: Dict[str, object] = {
        "scan_type": metadata["scan_type"],
        "metadata_h5_path": str(metadata["path"]),
        "single_shot_root": str(
            Path(metadata["analysis_dir"]) / SINGLE_SHOT_FOLDER
        ),
        "selected_shots": len(records_by_identity),
        "selected_shots_total": len(all_identities),
        "chunk_id": chunk_id,
        "n_chunks": n_chunks,
        "written_patterns": 0,
        "existing_patterns": 0,
        "invalid_shots": 0,
    }
    pending_by_identity = {}
    for identity in sorted(records_by_identity):
        shot_records = records_by_identity[identity]
        unique_targets = {}
        for record in shot_records:
            for window, azimuth_tag, signature in window_info:
                output_path = _single_shot_path(
                    Path(metadata["analysis_dir"]), record, azimuth_tag
                )
                unique_targets[str(output_path)] = (
                    window,
                    output_path,
                    signature,
                )
        targets = list(unique_targets.values())
        pending = [
            item for item in targets if bool(overwrite) or not item[1].is_file()
        ]
        report["existing_patterns"] = int(report["existing_patterns"]) + (
            len(targets) - len(pending)
        )
        if pending:
            pending_by_identity[identity] = pending

    if not pending_by_identity:
        print(
            "SACLA single-shot integration: 0 written, {} existing, 0 invalid "
            "tags skipped.".format(report["existing_patterns"])
        )
        return report

    client = facility_client or SaclaFacilityClient(
        beamline=int(context["beamline"]),
        detector_id=str(context["detector_id"]),
    )
    pending_records = [
        record
        for identity in pending_by_identity
        for record in records_by_identity[identity]
    ]
    resolved_records = _fill_missing_intensities(
        pending_records,
        client=client,
        intensity_col=str(context["intensity_col"]),
    )
    resolved_by_identity: Dict[Tuple[int, int], List[SingleShotRecord]] = {}
    for record in resolved_records:
        resolved_by_identity.setdefault((record.run, record.tag), []).append(record)

    background_image = None
    if context["background_path"] is not None:
        background_image = np.asarray(np.load(context["background_path"]), dtype=float)
        if background_image.ndim == 3 and background_image.shape[0] == 1:
            background_image = background_image[0]
        if background_image.ndim != 2:
            raise ValueError("The SACLA background image must be two-dimensional.")
        if not np.isfinite(background_image).all():
            raise ValueError("The SACLA background image contains non-finite values.")

    first_read_error = None
    two_theta_by_signature: Dict[str, np.ndarray] = {}

    for identity in tqdm(
        sorted(pending_by_identity),
        desc="SACLA single-shot tags",
        unit="tag",
    ):
        run, tag = identity
        shot_records = resolved_by_identity[identity]
        pending = pending_by_identity[identity]

        try:
            image = np.asarray(client.read_image(run, tag))
        except Exception as exc:
            if first_read_error is None:
                first_read_error = exc
            report["invalid_shots"] = int(report["invalid_shots"]) + 1
            continue
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 2 or not image.size or np.isnan(image).any():
            report["invalid_shots"] = int(report["invalid_shots"]) + 1
            continue

        image = image.astype(float)
        if background_image is not None:
            if background_image.shape != image.shape:
                raise ValueError(
                    "SACLA background shape {} does not match detector frame shape {}.".format(
                        background_image.shape, image.shape
                    )
                )
            image = image - background_image
        image[image < float(context["threshold_counts"])] = 0.0
        if pulse_normalized:
            pulse_intensity = shot_records[0].pulse_intensity
            if (
                pulse_intensity is None
                or not np.isfinite(float(pulse_intensity))
                or float(pulse_intensity) == 0.0
            ):
                report["invalid_shots"] = int(report["invalid_shots"]) + 1
                continue
            image = image / float(pulse_intensity)
        if np.isnan(image).any():
            report["invalid_shots"] = int(report["invalid_shots"]) + 1
            continue

        pending_by_window: Dict[Tuple[Tuple[float, float], str], List[Path]] = {}
        for window, output_path, signature in pending:
            pending_by_window.setdefault((window, signature), []).append(output_path)
        for (window, signature), output_paths in pending_by_window.items():
            if hasattr(integrator, "_azimuth_engine_caches"):
                q, intensity = integrator.integrate1d(
                    image,
                    window,
                    cache_engine_by_azimuth=True,
                )
            else:
                q, intensity = integrator.integrate1d(image, window)
            two_theta = two_theta_by_signature.get(signature)
            if two_theta is None:
                two_theta = general_utils.q_to_two_theta(
                    q, integrator._ai.wavelength
                )
                two_theta_by_signature[signature] = two_theta
            for output_path in output_paths:
                _save_single_shot_xy_atomic(
                    output_path,
                    two_theta,
                    intensity,
                    settings_signature=signature,
                )
                report["written_patterns"] = int(report["written_patterns"]) + 1

    if (
        int(report["written_patterns"]) == 0
        and int(report["existing_patterns"]) == 0
        and first_read_error is not None
    ):
        raise RuntimeError(
            "SACLA detector access failed before any single-shot pattern was "
            "created: {}".format(first_read_error)
        ) from first_read_error

    print(
        "SACLA single-shot integration: {} written, {} existing, {} invalid "
        "tags skipped.".format(
            report["written_patterns"],
            report["existing_patterns"],
            report["invalid_shots"],
        )
    )
    return report


def _selected_values(selector, available: Sequence[Union[int, float]], *, name: str):
    """Normalize an all/scalar/sequence selector against metadata values."""
    if isinstance(selector, str) and selector.strip().lower() == "all":
        return list(available)
    if isinstance(selector, (int, float, np.integer, np.floating)):
        requested = [selector]
    else:
        requested = list(selector)
    if not requested:
        raise ValueError("No {} values were requested.".format(name))
    return requested


def _matching_records(
    records: Sequence[SingleShotRecord],
    group_value: Union[int, float, str],
) -> List[SingleShotRecord]:
    """Return metadata records belonging to one final pattern."""
    if group_value == "dark":
        return [record for record in records if record.scan_type == "dark"]
    if records and records[0].scan_type == "delay":
        return [
            record for record in records if int(record.group_key) == int(group_value)
        ]
    return [
        record
        for record in records
        if np.isclose(
            float(record.group_key), float(group_value), rtol=0.0, atol=1e-12
        )
    ]


def _make_integrator_and_windows(
    *,
    poni_path,
    mask_edf_path,
    azimuthal_edges,
    include_full,
    full_range,
    npt,
    normalize,
    q_norm_range,
    azim_offset_deg,
    polarization_factor,
):
    """Create the final-pattern integrator and canonical azimuth windows."""
    integrator = azimint_utils.AzimIntegrator(
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        npt=int(npt),
        normalize=bool(normalize),
        q_norm_range=(float(q_norm_range[0]), float(q_norm_range[1])),
        azim_offset_deg=float(azim_offset_deg),
        polarization_factor=polarization_factor,
    )
    windows = integrator.build_windows(
        np.asarray(azimuthal_edges, dtype=float),
        include_full=bool(include_full),
        full_range=(float(full_range[0]), float(full_range[1])),
    )
    return integrator, windows


def _aggregate_dataset(
    *,
    metadata: Dict[str, object],
    records: Sequence[SingleShotRecord],
    group_value: Union[int, float, str],
    dataset,
    integrator,
    windows,
    context: Dict[str, object],
    overwrite_xy: bool,
    read_workers: int = 1,
    show_progress: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Average all currently available compatible SACLA shot patterns."""
    metadata_dir = Path(metadata["analysis_dir"]).resolve()
    dataset_dir = Path(dataset.analysis_dir()).resolve()
    if dataset_dir != metadata_dir:
        raise ValueError(
            "The experiment fields do not identify the selected SACLA metadata "
            "directory. Expected {}, got {}.".format(metadata_dir, dataset_dir)
        )
    group_records = _matching_records(records, group_value)
    if not group_records:
        raise FileNotFoundError(
            "No metadata-selected SACLA tags match {} value {}.".format(
                metadata["scan_type"], group_value
            )
        )

    result: Dict[str, Dict[str, object]] = {}
    pulse_normalized = metadata["scan_type"] in {"delay", "fluence"}
    for window in windows:
        azimuth_tag = general_utils.azim_range_str(window)
        output_path = dataset.xy_path(azimuth_tag)
        if output_path.is_file() and not overwrite_xy:
            result[azimuth_tag] = {
                "path": str(output_path),
                "available_shots": 0,
                "expected_shots": len(group_records),
                "status": "existing",
            }
            continue
        expected_signature = _settings_signature(
            poni_sha256=str(context["poni_sha256"]),
            mask_sha256=context["mask_sha256"],
            background_sha256=context["background_sha256"],
            npt=integrator.npt,
            azimuthal_range=window,
            azim_offset_deg=integrator.azim_offset_deg,
            polarization_factor=integrator.polarization_factor,
            beamline=int(context["beamline"]),
            detector_id=str(context["detector_id"]),
            threshold_counts=float(context["threshold_counts"]),
            intensity_col=str(context["intensity_col"]),
            pulse_normalized=pulse_normalized,
        )
        available_paths = []
        seen_paths = set()
        for record in group_records:
            path = _single_shot_path(metadata_dir, record, azimuth_tag)
            if path.is_file() and path not in seen_paths:
                available_paths.append(path)
                seen_paths.add(path)
        if not available_paths:
            raise FileNotFoundError(
                "No completed SACLA single-shot patterns are available for {}, "
                "azimuth {}, under {}.".format(
                    group_value, azimuth_tag, metadata_dir
                )
            )

        reference_x = None
        summed_intensity = None
        finite_count = None
        read_workers = max(1, min(int(read_workers), len(available_paths)))

        def consume(loaded, path):
            nonlocal reference_x, summed_intensity, finite_count
            signature, two_theta, intensity = loaded
            if signature != expected_signature:
                raise ValueError(
                    "Single-shot integration settings do not match the requested "
                    "final pattern settings: {}".format(path)
                )
            if reference_x is None:
                reference_x = two_theta
            elif reference_x.shape != two_theta.shape or not np.allclose(
                reference_x,
                two_theta,
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(
                    "Incompatible radial grids in SACLA single-shot cache: {}".format(
                        path
                    )
                )
            if summed_intensity is None:
                summed_intensity = np.zeros(intensity.shape, dtype=float)
                finite_count = np.zeros(intensity.shape, dtype=np.int64)
            finite = np.isfinite(intensity)
            summed_intensity[finite] += intensity[finite]
            finite_count[finite] += 1

        if read_workers == 1:
            for path in tqdm(
                available_paths,
                desc=f"Aggregate SACLA {group_value} azimuth {azimuth_tag}",
                unit="tag",
                dynamic_ncols=True,
                disable=not show_progress,
            ):
                consume(_load_single_shot_xy(path), path)
        else:
            with ThreadPoolExecutor(max_workers=read_workers) as executor:
                loaded_values = executor.map(_load_single_shot_xy, available_paths)
                with tqdm(
                    total=len(available_paths),
                    desc=f"Aggregate SACLA {group_value} azimuth {azimuth_tag}",
                    unit="tag",
                    dynamic_ncols=True,
                    disable=not show_progress,
                ) as progress:
                    for path, loaded in zip(available_paths, loaded_values):
                        consume(loaded, path)
                        progress.update(1)

        mean_intensity = np.full(summed_intensity.shape, np.nan, dtype=float)
        np.divide(
            summed_intensity,
            finite_count,
            out=mean_intensity,
            where=finite_count > 0,
        )
        if integrator.normalize:
            q = general_utils.two_theta_to_q(reference_x, integrator._ai.wavelength)
            mean_intensity = general_utils.normalize_y_by_mean_in_xrange(
                q, mean_intensity, integrator.q_norm_range
            )
        _save_final_xy_atomic(output_path, reference_x, mean_intensity)
        result[azimuth_tag] = {
            "path": str(output_path),
            "available_shots": len(available_paths),
            "expected_shots": len(group_records),
            "status": "written",
        }
        print(
            "Final SACLA {} XY {}: {}/{} tags available.".format(
                metadata["scan_type"],
                output_path.name,
                len(available_paths),
                len(group_records),
            )
        )
    return result


def _aggregate_dataset_jobs(
    *,
    jobs,
    use_parallel,
    max_workers,
    aggregate_kwargs,
):
    """Aggregate independent SACLA groups with one bounded worker budget."""
    jobs = list(jobs)
    requested_workers = azimint_utils.resolve_parallel_worker_count(
        use_parallel=use_parallel,
        max_workers=max_workers,
    )
    group_workers = min(requested_workers, max(1, len(jobs)))
    read_workers = 1 if group_workers > 1 else requested_workers

    def aggregate_one(job):
        group_value, dataset = job
        return _aggregate_dataset(
            group_value=group_value,
            dataset=dataset,
            read_workers=read_workers,
            show_progress=group_workers == 1,
            **aggregate_kwargs,
        )

    if group_workers == 1:
        for job in jobs:
            aggregate_one(job)
        return

    print(
        f"Final SACLA {aggregate_kwargs['metadata']['scan_type']} aggregation "
        f"starting: {len(jobs)} group(s), {group_workers} worker(s).",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=group_workers) as executor:
        list(executor.map(aggregate_one, jobs))


def _aggregation_setup(
    *,
    metadata_h5_path,
    expected_scan_type,
    poni_path,
    mask_edf_path,
    azimuthal_edges,
    include_full,
    full_range,
    npt,
    normalize,
    q_norm_range,
    azim_offset_deg,
    polarization_factor,
    beamline,
    detector_id,
    background,
    background_path,
    threshold_counts,
    intensity_col,
    paths,
):
    """Build shared metadata, integrator, windows, and fingerprint context."""
    metadata, records = _metadata_records(metadata_h5_path)
    if metadata["scan_type"] != expected_scan_type:
        raise ValueError(
            "The selected metadata HDF5 is not a SACLA {} scan.".format(
                expected_scan_type
            )
        )
    integrator, windows = _make_integrator_and_windows(
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azimuthal_edges=azimuthal_edges,
        include_full=include_full,
        full_range=full_range,
        npt=npt,
        normalize=normalize,
        q_norm_range=q_norm_range,
        azim_offset_deg=azim_offset_deg,
        polarization_factor=polarization_factor,
    )
    context = _integration_context(
        metadata=metadata,
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        background=background,
        background_path=background_path,
        paths=paths,
        beamline=beamline,
        detector_id=detector_id,
        threshold_counts=threshold_counts,
        intensity_col=intensity_col,
    )
    return metadata, records, integrator, windows, context


def aggregate_delay_1d(
    *,
    sample_name,
    temperature_K,
    excitation_wl_nm,
    fluence_mJ_cm2,
    time_window_fs,
    delays_fs,
    metadata_h5_path,
    poni_path,
    mask_edf_path,
    azimuthal_edges,
    include_full=True,
    full_range=(-90.0, 90.0),
    npt=1000,
    normalize=True,
    q_norm_range=(2.65, 2.75),
    overwrite_xy=False,
    azim_offset_deg=-90.0,
    polarization_factor=None,
    beamline=None,
    detector_id=DEFAULT_DETECTOR_ID,
    background=None,
    background_path=None,
    threshold_counts=40.0,
    intensity_col=None,
    use_parallel=True,
    max_workers=None,
    paths: AnalysisPaths,
):
    """Build delay-series final XY files from available SACLA shot patterns."""
    metadata, records, integrator, windows, context = _aggregation_setup(
        metadata_h5_path=metadata_h5_path,
        expected_scan_type="delay",
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azimuthal_edges=azimuthal_edges,
        include_full=include_full,
        full_range=full_range,
        npt=npt,
        normalize=normalize,
        q_norm_range=q_norm_range,
        azim_offset_deg=azim_offset_deg,
        polarization_factor=polarization_factor,
        beamline=beamline,
        detector_id=detector_id,
        background=background,
        background_path=background_path,
        threshold_counts=threshold_counts,
        intensity_col=intensity_col,
        paths=paths,
    )
    available = sorted({int(record.delay_fs) for record in records})
    selected = [
        int(value) for value in _selected_values(delays_fs, available, name="delay")
    ]
    datasets = []
    jobs = []
    for delay in selected:
        dataset = azimint_utils.DelayDataset(
            sample_name,
            temperature_K,
            excitation_wl_nm,
            fluence_mJ_cm2,
            time_window_fs,
            delay,
            paths=paths,
        )
        datasets.append(dataset)
        jobs.append((delay, dataset))
    _aggregate_dataset_jobs(
        jobs=jobs,
        use_parallel=use_parallel,
        max_workers=max_workers,
        aggregate_kwargs={
            "metadata": metadata,
            "records": records,
            "integrator": integrator,
            "windows": windows,
            "context": context,
            "overwrite_xy": bool(overwrite_xy),
        },
    )
    return integrator, datasets


def aggregate_fluence_1d(
    *,
    sample_name,
    temperature_K,
    excitation_wl_nm,
    delay_fs,
    time_window_fs,
    fluences_mJ_cm2,
    metadata_h5_path,
    poni_path,
    mask_edf_path,
    azimuthal_edges,
    include_full=True,
    full_range=(-90.0, 90.0),
    npt=1000,
    normalize=True,
    q_norm_range=(2.65, 2.75),
    overwrite_xy=False,
    azim_offset_deg=-90.0,
    polarization_factor=None,
    beamline=None,
    detector_id=DEFAULT_DETECTOR_ID,
    background=None,
    background_path=None,
    threshold_counts=40.0,
    intensity_col=None,
    use_parallel=True,
    max_workers=None,
    paths: AnalysisPaths,
):
    """Build fluence-series final XY files from available SACLA shot patterns."""
    metadata, records, integrator, windows, context = _aggregation_setup(
        metadata_h5_path=metadata_h5_path,
        expected_scan_type="fluence",
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azimuthal_edges=azimuthal_edges,
        include_full=include_full,
        full_range=full_range,
        npt=npt,
        normalize=normalize,
        q_norm_range=q_norm_range,
        azim_offset_deg=azim_offset_deg,
        polarization_factor=polarization_factor,
        beamline=beamline,
        detector_id=detector_id,
        background=background,
        background_path=background_path,
        threshold_counts=threshold_counts,
        intensity_col=intensity_col,
        paths=paths,
    )
    available = sorted({float(record.fluence_mJ_cm2) for record in records})
    selected = [
        float(value)
        for value in _selected_values(
            fluences_mJ_cm2, available, name="fluence"
        )
    ]
    datasets = []
    jobs = []
    for fluence in selected:
        dataset = azimint_utils.FluenceDataset(
            sample_name,
            temperature_K,
            excitation_wl_nm,
            fluence,
            time_window_fs,
            delay_fs,
            paths=paths,
        )
        datasets.append(dataset)
        jobs.append((fluence, dataset))
    _aggregate_dataset_jobs(
        jobs=jobs,
        use_parallel=use_parallel,
        max_workers=max_workers,
        aggregate_kwargs={
            "metadata": metadata,
            "records": records,
            "integrator": integrator,
            "windows": windows,
            "context": context,
            "overwrite_xy": bool(overwrite_xy),
        },
    )
    return integrator, datasets


def aggregate_dark_1d(
    *,
    sample_name,
    temperature_K,
    dark_tag=None,
    metadata_h5_path,
    poni_path,
    mask_edf_path,
    azimuthal_edges,
    include_full=True,
    full_range=(-90.0, 90.0),
    npt=2000,
    normalize=True,
    q_norm_range=(2.65, 2.75),
    overwrite_xy=False,
    azim_offset_deg=-90.0,
    polarization_factor=None,
    beamline=None,
    detector_id=DEFAULT_DETECTOR_ID,
    background=None,
    background_path=None,
    threshold_counts=40.0,
    intensity_col=None,
    use_parallel=True,
    max_workers=None,
    paths: AnalysisPaths,
):
    """Build a final dark XY file from available SACLA shot patterns."""
    metadata, records, integrator, windows, context = _aggregation_setup(
        metadata_h5_path=metadata_h5_path,
        expected_scan_type="dark",
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        azimuthal_edges=azimuthal_edges,
        include_full=include_full,
        full_range=full_range,
        npt=npt,
        normalize=normalize,
        q_norm_range=q_norm_range,
        azim_offset_deg=azim_offset_deg,
        polarization_factor=polarization_factor,
        beamline=beamline,
        detector_id=detector_id,
        background=background,
        background_path=background_path,
        threshold_counts=threshold_counts,
        intensity_col=intensity_col,
        paths=paths,
    )
    resolved_tag = (
        str(metadata["dark_tag"])
        if dark_tag is None
        else azimint_utils.dark_tag_from_scan_spec(dark_tag)
    )
    dataset = azimint_utils.DarkDataset(
        sample_name=sample_name,
        temperature_K=temperature_K,
        dark_tag=resolved_tag,
        paths=paths,
    )
    _aggregate_dataset(
        metadata=metadata,
        records=records,
        group_value="dark",
        dataset=dataset,
        integrator=integrator,
        windows=windows,
        context=context,
        overwrite_xy=bool(overwrite_xy),
        read_workers=azimint_utils.resolve_parallel_worker_count(
            use_parallel=use_parallel,
            max_workers=max_workers,
        ),
    )
    return integrator, dataset


def main(argv=None):
    """Run one SACLA single-shot 1D PBS-array worker partition."""
    parser = argparse.ArgumentParser(
        description=(
            "Integrate one deterministic chunk of metadata-selected SACLA "
            "run/tag detector frames into the single-shot 1D cache."
        )
    )
    parser.add_argument("--metadata-h5", required=True)
    parser.add_argument("--poni", required=True)
    parser.add_argument("--mask", default="")
    parser.add_argument(
        "--azimuthal-edges",
        nargs="+",
        type=float,
        required=True,
        help="Ordered azimuthal edges in degrees.",
    )
    parser.add_argument("--no-include-full", dest="include_full", action="store_false")
    parser.set_defaults(include_full=True)
    parser.add_argument("--full-range", nargs=2, type=float, default=(-90.0, 90.0))
    parser.add_argument("--npt", type=int, default=1000)
    parser.add_argument("--azim-offset-deg", type=float, default=-90.0)
    parser.add_argument("--polarization-factor", type=float, default=None)
    parser.add_argument("--beamline", type=int, default=None)
    parser.add_argument("--detector-id", default=DEFAULT_DETECTOR_ID)
    parser.add_argument("--background", default="")
    parser.add_argument("--background-path", default="")
    parser.add_argument("--threshold-counts", type=float, default=40.0)
    parser.add_argument("--intensity-col", default=None)
    parser.add_argument("--path-root", default="")
    parser.add_argument("--raw-subdir", default="")
    parser.add_argument("--analysis-subdir", default="analysis")
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--n-chunks", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    background = None
    if str(args.background).strip():
        text = str(args.background).strip()
        try:
            background = int(text)
        except ValueError:
            background = text
    paths = None
    if str(args.path_root).strip():
        paths = AnalysisPaths(
            path_root=Path(args.path_root),
            raw_subdir=str(args.raw_subdir),
            analysis_subdir=str(args.analysis_subdir),
        )

    report = integrate_single_shot_1d(
        metadata_h5_path=args.metadata_h5,
        poni_path=args.poni,
        mask_edf_path=(str(args.mask).strip() or None),
        azimuthal_edges=args.azimuthal_edges,
        include_full=bool(args.include_full),
        full_range=(float(args.full_range[0]), float(args.full_range[1])),
        npt=int(args.npt),
        overwrite=bool(args.overwrite),
        azim_offset_deg=float(args.azim_offset_deg),
        polarization_factor=args.polarization_factor,
        beamline=args.beamline,
        detector_id=args.detector_id,
        background=background,
        background_path=(str(args.background_path).strip() or None),
        threshold_counts=float(args.threshold_counts),
        intensity_col=args.intensity_col,
        paths=paths,
        chunk_id=int(args.chunk),
        n_chunks=int(args.n_chunks),
    )
    print(json.dumps(report, sort_keys=True))
    return report


__all__ = [
    "DEFAULT_DETECTOR_ID",
    "DEFAULT_INTENSITY_COLUMN",
    "SaclaFacilityClient",
    "SingleShotRecord",
    "aggregate_dark_1d",
    "aggregate_delay_1d",
    "aggregate_fluence_1d",
    "integrate_single_shot_1d",
    "main",
]


if __name__ == "__main__":
    main()
