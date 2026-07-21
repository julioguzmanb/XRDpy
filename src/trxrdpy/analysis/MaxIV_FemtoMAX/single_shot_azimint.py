"""FemtoMAX single-shot detector-image integration and 1D aggregation.

The metadata HDF5 file remains the authority for shot selection.  Individual
detector frames are integrated into a separate cache and can then be averaged
into the same final ``xy_files`` paths used by the representative-2D route.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import queue
import sys
import tempfile
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from tqdm import tqdm

from ..common import azimint_utils, general_utils
from ..common.paths import AnalysisPaths
from .datared_utils import DEFAULT_SCAN_FILE_PATTERN, Experiment, ExperimentMeta


SINGLE_SHOT_FOLDER = "single_shot_1D_patterns"
_SETTINGS_HEADER = "# xrdpy_single_shot_settings_sha256="
_SETTINGS_JSON_HEADER = "# xrdpy_single_shot_settings_json="
_AGGREGATE_CACHE_FOLDER = ".aggregate_cache"
_WORKER_COMMON_PAYLOAD = None
_WORKER_INTEGRATOR = None
_WORKER_PROGRESS_QUEUE = None
_WORKER_CANCEL_EVENT = None
_WORKER_RAW_HANDLES = OrderedDict()
_WORKER_OUTPUT_DIRECTORIES = set()
_MAX_WORKER_RAW_CACHE_SIZE = 64


@dataclass(frozen=True)
class SingleShotRecord:
    """One detector-frame selection represented in FemtoMAX metadata."""

    scan_type: str
    scan: int
    shot: int
    sample_name: str
    temperature_K: int
    excitation_wl_nm: Optional[float]
    fluence_mJ_cm2: Optional[float]
    time_window_fs: Optional[int]
    delay_fs: Optional[int]
    dark_file_tag: Optional[str]

    @property
    def group_key(self) -> Union[int, float, str]:
        """Return the delay, fluence, or dark key used during aggregation."""
        if self.scan_type == "delay":
            return int(self.delay_fs)  # type: ignore[arg-type]
        if self.scan_type == "fluence":
            return float(self.fluence_mJ_cm2)  # type: ignore[arg-type]
        return "dark"

    def group_folder(self) -> Optional[str]:
        """Return the scan-variable folder below an azimuthal-window folder."""
        if self.scan_type == "delay":
            return f"delay_{int(self.delay_fs)}fs"
        if self.scan_type == "fluence":
            tag = general_utils.fluence_tag_file(float(self.fluence_mJ_cm2))
            return f"fluence_{tag}mJ"
        return None

    def final_stem(self, azimuth_tag: str) -> str:
        """Return the final XY stem before scan/shot identity is appended."""
        if self.scan_type == "dark":
            return (
                f"{self.sample_name}_{self.temperature_K}K_dark_"
                f"{self.dark_file_tag}_{azimuth_tag}"
            )

        wl_tag = general_utils.wl_tag_nm(float(self.excitation_wl_nm))
        flu_tag = general_utils.fluence_tag_file(float(self.fluence_mJ_cm2))
        return (
            f"{self.sample_name}_{self.temperature_K}K_{wl_tag}nm_"
            f"{flu_tag}mJ_{int(self.time_window_fs)}fs_"
            f"{int(self.delay_fs)}fs_{azimuth_tag}"
        )


def _read_scalar(group: h5py.Group, name: str, default=None):
    """Read a scalar metadata dataset or attribute and decode byte strings."""
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


def _metadata_records(
    metadata_h5_path: Union[str, Path],
) -> Tuple[Dict[str, object], List[SingleShotRecord]]:
    """Read experiment identity and selected frame indices from metadata HDF5."""
    metadata_path = Path(metadata_h5_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(str(metadata_path))

    records: List[SingleShotRecord] = []
    with h5py.File(metadata_path, "r") as handle:
        if "meta" not in handle:
            raise ValueError(f"Invalid FemtoMAX metadata HDF5 (missing /meta): {metadata_path}")
        meta_group = handle["meta"]
        scan_type = str(_read_scalar(meta_group, "scan_type", "")).strip().lower()
        if scan_type not in {"dark", "delay", "fluence"}:
            raise ValueError(
                "FemtoMAX metadata scan_type must be 'dark', 'delay', or 'fluence'."
            )

        sample_name = str(_read_scalar(meta_group, "sample_name"))
        temperature_K = int(_read_scalar(meta_group, "temperature_K"))
        excitation_wl_nm = _read_scalar(meta_group, "excitation_wl_nm")
        time_window_fs = _read_scalar(meta_group, "time_window_fs")

        metadata: Dict[str, object] = {
            "path": metadata_path,
            "analysis_dir": metadata_path.parent,
            "scan_type": scan_type,
            "sample_name": sample_name,
            "temperature_K": temperature_K,
            "excitation_wl_nm": (
                None if excitation_wl_nm is None else float(excitation_wl_nm)
            ),
            "time_window_fs": (
                None if time_window_fs is None else int(time_window_fs)
            ),
        }

        if scan_type == "dark":
            if "scans" not in handle:
                raise ValueError("Invalid FemtoMAX dark metadata HDF5 (missing /scans).")
            scans = sorted(int(name) for name in handle["scans"].keys())
            dark_file_tag = general_utils.scan_tag_file(scans)
            metadata["dark_tag"] = metadata_path.parent.name
            metadata["dark_file_tag"] = dark_file_tag
            for scan in scans:
                scan_group = handle["scans"][str(scan)]
                nshots = int(scan_group.attrs.get("nshots_expected", 0))
                for shot in range(max(0, nshots)):
                    records.append(
                        SingleShotRecord(
                            scan_type="dark",
                            scan=scan,
                            shot=shot,
                            sample_name=sample_name,
                            temperature_K=temperature_K,
                            excitation_wl_nm=None,
                            fluence_mJ_cm2=None,
                            time_window_fs=None,
                            delay_fs=None,
                            dark_file_tag=dark_file_tag,
                        )
                    )
            return metadata, records

        if "delays" not in handle:
            raise ValueError("Invalid FemtoMAX metadata HDF5 (missing /delays).")

        delay_fluence = _read_scalar(meta_group, "fluence_mJ_cm2")
        scan_to_fluence: Dict[int, float] = {}
        if scan_type == "fluence":
            if "scans" not in meta_group or "fluences_mJ_cm2" not in meta_group:
                raise ValueError(
                    "Invalid FemtoMAX fluence metadata HDF5: missing scan/fluence mapping."
                )
            scans = np.asarray(meta_group["scans"], dtype=np.int64)
            fluences = np.asarray(meta_group["fluences_mJ_cm2"], dtype=float)
            if scans.size != fluences.size:
                raise ValueError("FemtoMAX metadata scan and fluence arrays have different sizes.")
            scan_to_fluence = {
                int(scan): float(fluence)
                for scan, fluence in zip(scans.tolist(), fluences.tolist())
            }

        for delay_name in sorted(handle["delays"].keys()):
            delay_group = handle["delays"][delay_name]
            delay_fs = int(delay_group.attrs.get("delay_fs", str(delay_name).removesuffix("fs")))
            if "scans" not in delay_group:
                continue
            for scan_name in sorted(delay_group["scans"].keys(), key=int):
                scan = int(scan_name)
                scan_group = delay_group["scans"][scan_name]
                indices = np.asarray(scan_group["indices"], dtype=np.int64)
                if scan_type == "fluence":
                    fluence = scan_to_fluence.get(
                        scan,
                        float(scan_group.attrs.get("fluence_mJ_cm2", np.nan)),
                    )
                    if not np.isfinite(fluence):
                        raise ValueError(f"No finite fluence is recorded for scan {scan}.")
                else:
                    fluence = float(delay_fluence)

                for shot in np.unique(indices).tolist():
                    records.append(
                        SingleShotRecord(
                            scan_type=scan_type,
                            scan=scan,
                            shot=int(shot),
                            sample_name=sample_name,
                            temperature_K=temperature_K,
                            excitation_wl_nm=float(excitation_wl_nm),
                            fluence_mJ_cm2=float(fluence),
                            time_window_fs=int(time_window_fs),
                            delay_fs=delay_fs,
                            dark_file_tag=None,
                        )
                    )

        if scan_type == "delay":
            metadata["fluence_mJ_cm2"] = float(delay_fluence)
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


def resolve_metadata_h5_path(
    *,
    scan_type: str,
    sample_name: str,
    temperature_K: int,
    paths: AnalysisPaths,
    excitation_wl_nm: Optional[float] = None,
    fluence_mJ_cm2: Optional[float] = None,
    time_window_fs: Optional[int] = None,
    delay_fs: Optional[int] = None,
    scans: Optional[Sequence[int]] = None,
) -> Path:
    """Resolve an existing FemtoMAX metadata file from experiment fields."""
    normalized_scan_type = str(scan_type).strip().lower()
    if normalized_scan_type not in {"dark", "delay", "fluence"}:
        raise ValueError("scan_type must be 'dark', 'delay', or 'fluence'.")
    meta = ExperimentMeta(
        sample_name=str(sample_name).strip(),
        temperature_K=int(temperature_K),
        excitation_wl_nm=(
            None if normalized_scan_type == "dark" else float(excitation_wl_nm)
        ),
        fluence_mJ_cm2=(
            float(fluence_mJ_cm2)
            if normalized_scan_type == "delay"
            else None
        ),
        scan_type=normalized_scan_type,
        time_window_fs=(
            None if normalized_scan_type == "dark" else int(time_window_fs)
        ),
        delay=(int(delay_fs) if normalized_scan_type == "fluence" else None),
    )
    candidate = Path(
        Experiment.metadata_h5_path(meta, scans=scans, paths=paths)
    )
    if candidate.is_file():
        return candidate.resolve()

    if normalized_scan_type == "dark":
        search_root = (
            paths.analysis_root
            / meta.sample_name
            / f"temperature_{meta.temperature_K}K"
            / "dark"
        )
        candidates = sorted(search_root.rglob("*.h5")) if search_root.is_dir() else []
    else:
        analysis_dir = Path(Experiment.analysis_dir(meta, scans=scans, paths=paths))
        candidates = sorted(analysis_dir.glob("*.h5")) if analysis_dir.is_dir() else []

    matches = []
    for path in candidates:
        try:
            metadata, records = _metadata_records(path)
        except (OSError, ValueError, KeyError):
            continue
        if metadata["scan_type"] != normalized_scan_type:
            continue
        if str(metadata["sample_name"]) != meta.sample_name:
            continue
        if int(metadata["temperature_K"]) != meta.temperature_K:
            continue
        if normalized_scan_type != "dark":
            if not np.isclose(
                float(metadata["excitation_wl_nm"]),
                float(meta.excitation_wl_nm),
                rtol=0.0,
                atol=1e-12,
            ):
                continue
            if int(metadata["time_window_fs"]) != int(meta.time_window_fs):
                continue
        if normalized_scan_type == "delay" and not np.isclose(
            float(metadata["fluence_mJ_cm2"]),
            float(meta.fluence_mJ_cm2),
            rtol=0.0,
            atol=1e-12,
        ):
            continue
        if normalized_scan_type == "fluence" and delay_fs is not None:
            record_delays = {int(record.delay_fs) for record in records}
            if int(delay_fs) not in record_delays:
                continue
        matches.append(path.resolve())

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            "No FemtoMAX metadata HDF5 matches the current experiment fields. "
            f"Expected {candidate}. Create the metadata first or select it explicitly."
        )
    raise ValueError(
        "Multiple FemtoMAX metadata HDF5 files match the current experiment fields: "
        + ", ".join(str(path) for path in matches)
    )


def _file_sha256(path: Optional[Union[str, Path]]) -> Optional[str]:
    """Return a file-content digest used in the integration fingerprint."""
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


def _settings_payload(
    *,
    poni_path: Union[str, Path],
    mask_edf_path: Optional[Union[str, Path]],
    npt: int,
    azimuthal_range: Tuple[float, float],
    azim_offset_deg: float,
    polarization_factor: Optional[float],
    poni_sha256: Optional[str] = None,
    mask_sha256: Optional[str] = None,
) -> Dict[str, object]:
    """Return serializable settings that affect single-shot XY values."""
    return {
        "schema": 1,
        "poni_sha256": poni_sha256 or _file_sha256(poni_path),
        "mask_sha256": (
            mask_sha256
            if mask_sha256 is not None
            else _file_sha256(mask_edf_path)
        ),
        "npt": int(npt),
        "azimuthal_range_deg": [
            float(azimuthal_range[0]),
            float(azimuthal_range[1]),
        ],
        "azim_offset_deg": float(azim_offset_deg),
        "polarization_factor": (
            None if polarization_factor is None else float(polarization_factor)
        ),
        "single_shot_normalized": False,
        "x_coordinate": "two_theta_deg",
    }


def _settings_signature_from_payload(payload: Dict[str, object]) -> str:
    """Hash one normalized single-shot settings payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _settings_signature(**kwargs) -> str:
    """Build a stable fingerprint for settings that affect single-shot XY values."""
    return _settings_signature_from_payload(_settings_payload(**kwargs))


def _single_shot_path(
    analysis_dir: Path,
    record: SingleShotRecord,
    azimuth_tag: str,
) -> Path:
    """Return the canonical cache path for one integrated FemtoMAX frame."""
    folder = analysis_dir / SINGLE_SHOT_FOLDER / azimuth_tag
    group_folder = record.group_folder()
    if group_folder is not None:
        folder = folder / group_folder
    filename = (
        f"{record.final_stem(azimuth_tag)}_scan{record.scan}_shot{record.shot}.xy"
    )
    return folder / filename


def _legacy_single_shot_path(
    analysis_dir: Path,
    record: SingleShotRecord,
    azimuth_tag: str,
) -> Path:
    """Return the former scan-subfolder path retained for cache compatibility."""
    canonical = _single_shot_path(analysis_dir, record, azimuth_tag)
    return canonical.parent / f"scan_{record.scan}" / canonical.name


def _existing_single_shot_path(
    analysis_dir: Path,
    record: SingleShotRecord,
    azimuth_tag: str,
) -> Optional[Path]:
    """Return a completed flat or legacy nested shot file when one exists."""
    canonical = _single_shot_path(analysis_dir, record, azimuth_tag)
    if canonical.is_file():
        return canonical
    legacy = _legacy_single_shot_path(analysis_dir, record, azimuth_tag)
    return legacy if legacy.is_file() else None


def _save_single_shot_xy_atomic(
    path: Path,
    two_theta: np.ndarray,
    intensity: np.ndarray,
    *,
    settings_signature: str,
    settings_payload: Optional[Dict[str, object]] = None,
) -> None:
    """Write a complete single-shot pattern before atomically publishing its name."""
    global _WORKER_OUTPUT_DIRECTORIES
    directory_key = str(path.parent)
    if directory_key not in _WORKER_OUTPUT_DIRECTORIES:
        path.parent.mkdir(parents=True, exist_ok=True)
        _WORKER_OUTPUT_DIRECTORIES.add(directory_key)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(f"{_SETTINGS_HEADER}{settings_signature}\n")
            if settings_payload is not None:
                stream.write(
                    _SETTINGS_JSON_HEADER
                    + json.dumps(
                        settings_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            np.savetxt(
                stream,
                np.column_stack((two_theta, intensity)),
                fmt="%.12g",
            )
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _save_final_xy_atomic(path: Path, x: np.ndarray, intensity: np.ndarray) -> None:
    """Atomically replace a final XY file so readers never see partial text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        general_utils.save_xy(temporary_path, x, intensity)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_single_shot_xy(
    path: Path,
) -> Tuple[str, Optional[Dict[str, object]], np.ndarray, np.ndarray]:
    """Load one shot, its fingerprint, and optional explicit settings provenance."""
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip()
        second_line_position = stream.tell()
        second_line = stream.readline().strip()
        settings_payload = None
        if second_line.startswith(_SETTINGS_JSON_HEADER):
            try:
                settings_payload = json.loads(
                    second_line[len(_SETTINGS_JSON_HEADER):].strip()
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid single-shot settings provenance: {path}"
                ) from exc
        else:
            stream.seek(second_line_position)
        values = np.fromstring(stream.read(), dtype=float, sep=" ")
    if not header.startswith(_SETTINGS_HEADER):
        raise ValueError(f"Single-shot pattern has no XRDpy settings fingerprint: {path}")
    if values.size == 0 or values.size % 2:
        raise ValueError(f"Expected two XY columns in single-shot pattern: {path}")
    values = values.reshape(-1, 2)
    signature = header[len(_SETTINGS_HEADER):].strip()
    return signature, settings_payload, values[:, 0], values[:, 1]


def _read_single_shot_settings(
    path: Path,
) -> Tuple[str, Optional[Dict[str, object]]]:
    """Read only the fingerprint and optional settings JSON from one shot file."""
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip()
        second_line = stream.readline().strip()
    if not header.startswith(_SETTINGS_HEADER):
        raise ValueError(f"Single-shot pattern has no XRDpy settings fingerprint: {path}")
    payload = None
    if second_line.startswith(_SETTINGS_JSON_HEADER):
        try:
            payload = json.loads(second_line[len(_SETTINGS_JSON_HEADER):].strip())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid single-shot settings provenance: {path}"
            ) from exc
    return header[len(_SETTINGS_HEADER):].strip(), payload


def _settings_mismatch_message(
    *,
    path: Path,
    cached_signature: str,
    cached_payload: Optional[Dict[str, object]],
    expected_payload: Dict[str, object],
) -> str:
    """Explain a cache mismatch using explicit provenance when it is available."""
    expected_signature = _settings_signature_from_payload(expected_payload)
    lines = [
        "Single-shot integration settings do not match the requested final pattern.",
        f"File: {path}",
        f"Cached fingerprint: {cached_signature}",
        f"Requested fingerprint: {expected_signature}",
    ]
    if cached_payload is not None:
        differences = []
        for key in sorted(set(cached_payload) | set(expected_payload)):
            cached_value = cached_payload.get(key)
            expected_value = expected_payload.get(key)
            if cached_value != expected_value:
                differences.append(
                    f"  {key}: cached={cached_value!r}; requested={expected_value!r}"
                )
        if differences:
            lines.append("Differing settings:")
            lines.extend(differences)
    else:
        try:
            _signature, _payload, two_theta, _intensity = _load_single_shot_xy(path)
            lines.append(
                "This legacy shot file has no explicit settings provenance; "
                f"it contains {two_theta.size} q points while "
                f"{expected_payload['npt']} were requested."
            )
        except (OSError, ValueError):
            lines.append(
                "This legacy shot file has no explicit settings provenance."
            )
        lines.append(
            "The mismatch is otherwise in the PONI or mask contents, azimuth "
            "offset, polarization factor, q-point count, or azimuthal range."
        )
    lines.append(
        "Use identical Data Reduction and Azimuthal Integration settings, or "
        "reproduce the incompatible single-shot files with overwrite enabled."
    )
    return "\n".join(lines)


def _read_detector_frame_batch(detector, shot_indices: Sequence[int]) -> np.ndarray:
    """Read sorted detector indices with one HDF5 selection where possible."""
    indices = [int(index) for index in shot_indices]
    if not indices:
        return np.empty((0,), dtype=float)
    if len(indices) == 1:
        return np.asarray(detector[indices[0]])[None, ...]
    is_contiguous = all(
        current == previous + 1
        for previous, current in zip(indices[:-1], indices[1:])
    )
    if is_contiguous:
        return np.asarray(detector[indices[0] : indices[-1] + 1])
    return np.asarray(detector[np.asarray(indices, dtype=np.int64)])


def _close_worker_raw_handles() -> None:
    """Close raw HDF5 files retained by the current worker process."""
    global _WORKER_RAW_HANDLES
    for handle, _detector in _WORKER_RAW_HANDLES.values():
        try:
            handle.close()
        except Exception:
            pass
    _WORKER_RAW_HANDLES = OrderedDict()


def _clear_worker_state() -> None:
    """Release process-local integration state after serial or fallback execution."""
    global _WORKER_COMMON_PAYLOAD
    global _WORKER_INTEGRATOR
    global _WORKER_PROGRESS_QUEUE
    global _WORKER_CANCEL_EVENT
    global _WORKER_OUTPUT_DIRECTORIES
    _close_worker_raw_handles()
    _WORKER_COMMON_PAYLOAD = None
    _WORKER_INTEGRATOR = None
    _WORKER_PROGRESS_QUEUE = None
    _WORKER_CANCEL_EVENT = None
    _WORKER_OUTPUT_DIRECTORIES = set()


def _initialize_work_process(
    common_payload: Dict[str, object],
    progress_queue=None,
    cancel_event=None,
) -> None:
    """Initialize process-local pyFAI state once for all assigned work chunks."""
    global _WORKER_COMMON_PAYLOAD
    global _WORKER_INTEGRATOR
    global _WORKER_PROGRESS_QUEUE
    global _WORKER_CANCEL_EVENT
    global _WORKER_OUTPUT_DIRECTORIES

    _close_worker_raw_handles()
    _WORKER_COMMON_PAYLOAD = dict(common_payload)
    _WORKER_PROGRESS_QUEUE = progress_queue
    _WORKER_CANCEL_EVENT = cancel_event
    _WORKER_OUTPUT_DIRECTORIES = set()
    _WORKER_INTEGRATOR = azimint_utils.AzimIntegrator(
        poni_path=common_payload["poni_path"],
        mask_edf_path=common_payload["mask_edf_path"],
        npt=int(common_payload["npt"]),
        normalize=False,
        azim_offset_deg=float(common_payload["azim_offset_deg"]),
        polarization_factor=common_payload["polarization_factor"],
    )


def _worker_detector(payload: Dict[str, object]):
    """Return a detector dataset using a process-local HDF5 dataset cache."""
    global _WORKER_RAW_HANDLES
    raw_path = str(payload["raw_path"])
    cached = _WORKER_RAW_HANDLES.pop(raw_path, None)
    if cached is None:
        handle = h5py.File(raw_path, "r")
        detector = handle
        common = _WORKER_COMMON_PAYLOAD or payload
        for component in tuple(common["detector_h5_path"]):
            detector = detector[component]
        cached = (handle, detector)
    _WORKER_RAW_HANDLES[raw_path] = cached
    common = _WORKER_COMMON_PAYLOAD or payload
    cache_size = int(
        common.get("raw_handle_cache_size", _MAX_WORKER_RAW_CACHE_SIZE)
    )
    while len(_WORKER_RAW_HANDLES) > cache_size:
        _unused_path, (unused_handle, _unused_detector) = (
            _WORKER_RAW_HANDLES.popitem(last=False)
        )
        unused_handle.close()
    return cached[1]


def _worker_cancel_requested() -> bool:
    """Return whether the parent requested a graceful stop between HDF5 batches."""
    return bool(
        _WORKER_CANCEL_EVENT is not None
        and _WORKER_CANCEL_EVENT.is_set()
    )


def _emit_worker_progress(processed_shots: int, group_label: str) -> None:
    """Send one batch-level progress update to the parent process when configured."""
    if _WORKER_PROGRESS_QUEUE is not None:
        _WORKER_PROGRESS_QUEUE.put((int(processed_shots), str(group_label)))


def _integrate_work_chunk(payload: Dict[str, object]) -> Dict[str, int]:
    """Integrate one independent FemtoMAX scan/shot chunk in a worker process."""
    common = _WORKER_COMMON_PAYLOAD or payload
    integrator = _WORKER_INTEGRATOR
    owns_state = integrator is None
    if owns_state:
        _initialize_work_process(common)
        integrator = _WORKER_INTEGRATOR
    work_items = list(payload["work_items"])
    read_batch_size = int(common["read_batch_size"])
    written_patterns = 0
    invalid_shots = 0
    processed_shots = 0
    cancelled = False
    two_theta_by_signature: Dict[str, np.ndarray] = {}

    try:
        detector = _worker_detector(payload)
        frame_count = int(detector.shape[0])
        for batch_start in range(0, len(work_items), read_batch_size):
            if _worker_cancel_requested():
                cancelled = True
                break
            batch = work_items[batch_start : batch_start + read_batch_size]
            shot_indices = [int(item[0]) for item in batch]
            invalid_indices = [
                shot for shot in shot_indices if shot < 0 or shot >= frame_count
            ]
            if invalid_indices:
                raise IndexError(
                    "Shots {} are outside scan {} detector data with {} frames.".format(
                        invalid_indices,
                        int(payload["scan"]),
                        frame_count,
                    )
                )
            images = _read_detector_frame_batch(detector, shot_indices)
            if images.shape[0] != len(batch):
                raise ValueError(
                    "FemtoMAX detector returned {} frames for a requested batch of {}.".format(
                        images.shape[0],
                        len(batch),
                    )
                )

            for (_shot, pending_by_window), image in zip(batch, images):
                image = np.asarray(image)
                if image.ndim != 2 or (
                    image.dtype.kind in {"f", "c"}
                    and not np.isfinite(image).all()
                ):
                    invalid_shots += 1
                    continue

                for key, output_names in pending_by_window.items():
                    window, signature = key
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
                            q,
                            integrator._ai.wavelength,
                        )
                        two_theta_by_signature[signature] = two_theta
                    for output_name in output_names:
                        _save_single_shot_xy_atomic(
                            Path(output_name),
                            two_theta,
                            intensity,
                            settings_signature=signature,
                            settings_payload=common.get(
                                "settings_payloads",
                                {},
                            ).get(signature),
                        )
                        written_patterns += 1

            processed_shots += len(batch)
            _emit_worker_progress(
                len(batch),
                str(payload.get("group_label", "")),
            )
    finally:
        if owns_state:
            _clear_worker_state()

    return {
        "written_patterns": written_patterns,
        "invalid_shots": invalid_shots,
        "processed_shots": processed_shots,
        "cancelled": bool(cancelled),
        "worker_pid": os.getpid(),
    }


def _integrate_worker_lane(payloads: Sequence[Dict[str, object]]) -> Dict[str, int]:
    """Process one scan-affine lane while retaining its HDF5 handles and engines."""
    result = {
        "written_patterns": 0,
        "invalid_shots": 0,
        "processed_shots": 0,
        "cancelled": False,
        "worker_pid": os.getpid(),
    }
    for payload in payloads:
        if _worker_cancel_requested():
            result["cancelled"] = True
            break
        part = _integrate_work_chunk(payload)
        for key in ("written_patterns", "invalid_shots", "processed_shots"):
            result[key] += int(part[key])
        if part.get("cancelled"):
            result["cancelled"] = True
            break
    return result


def _build_scan_affinity_lanes(
    payloads: Sequence[Dict[str, object]],
    worker_count: int,
) -> List[List[Dict[str, object]]]:
    """Assign every raw scan to one balanced worker lane when enough scans exist."""
    raw_loads: Dict[str, int] = {}
    for payload in payloads:
        raw_path = str(payload["raw_path"])
        raw_loads[raw_path] = raw_loads.get(raw_path, 0) + len(
            payload["work_items"]
        )
    if len(raw_loads) < int(worker_count):
        return []

    lane_loads = [0] * int(worker_count)
    raw_to_lane = {}
    for raw_path, load in sorted(
        raw_loads.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        lane_index = min(
            range(int(worker_count)),
            key=lambda index: (lane_loads[index], index),
        )
        raw_to_lane[raw_path] = lane_index
        lane_loads[lane_index] += int(load)

    lanes: List[List[Dict[str, object]]] = [
        [] for _index in range(int(worker_count))
    ]
    for payload in payloads:
        lanes[raw_to_lane[str(payload["raw_path"])]].append(payload)
    return [lane for lane in lanes if lane]


def _spawn_context_is_usable() -> bool:
    """Return whether spawn workers can safely re-import the main module."""
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return False
    if getattr(main_module, "__spec__", None) is not None:
        return True
    main_file = str(getattr(main_module, "__file__", "") or "").strip()
    return bool(main_file and not main_file.startswith("<") and Path(main_file).is_file())


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
    read_batch_size: int = 16,
    work_chunk_size: int = 64,
    use_parallel: bool = True,
    max_workers: Optional[int] = None,
    start_method: str = "spawn",
    cancel_event=None,
    azim_offset_deg: float = -90.0,
    polarization_factor: Optional[float] = None,
    paths: AnalysisPaths,
    scan_file_pattern: str = DEFAULT_SCAN_FILE_PATTERN,
) -> Dict[str, object]:
    """Integrate every metadata-selected FemtoMAX detector frame separately.

    Single-shot files intentionally store unnormalized intensity.  Optional
    normalization is applied once, after averaging, when final XY files are
    built from this cache.
    """
    metadata, records = _metadata_records(metadata_h5_path)
    if not records:
        raise ValueError("The FemtoMAX metadata file contains no selected detector shots.")
    read_batch_size = int(read_batch_size)
    if read_batch_size < 1:
        raise ValueError("read_batch_size must be at least 1.")
    work_chunk_size = int(work_chunk_size)
    if work_chunk_size < read_batch_size:
        raise ValueError("work_chunk_size must be at least read_batch_size.")

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
    poni_sha256 = _file_sha256(poni_path)
    mask_sha256 = _file_sha256(mask_edf_path)
    window_info = []
    settings_payloads = {}
    for window in windows:
        tag = general_utils.azim_range_str(window)
        settings_payload = _settings_payload(
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            npt=int(npt),
            azimuthal_range=window,
            azim_offset_deg=float(azim_offset_deg),
            polarization_factor=integrator.polarization_factor,
            poni_sha256=poni_sha256,
            mask_sha256=mask_sha256,
        )
        signature = _settings_signature_from_payload(settings_payload)
        settings_payloads[signature] = settings_payload
        window_info.append((window, tag, signature))

    records_by_scan: Dict[int, Dict[int, List[SingleShotRecord]]] = {}
    for record in records:
        records_by_scan.setdefault(int(record.scan), {}).setdefault(
            int(record.shot), []
        ).append(record)

    cache_root = Path(metadata["analysis_dir"]) / SINGLE_SHOT_FOLDER

    report: Dict[str, object] = {
        "scan_type": metadata["scan_type"],
        "metadata_h5_path": str(metadata["path"]),
        "single_shot_root": str(Path(metadata["analysis_dir"]) / SINGLE_SHOT_FOLDER),
        "selected_shots": sum(len(shots) for shots in records_by_scan.values()),
        "written_patterns": 0,
        "existing_patterns": 0,
        "invalid_shots": 0,
        "read_batch_size": read_batch_size,
        "work_chunk_size": work_chunk_size,
        # Hidden atomic-write remnants are ignored by discovery. Recursively
        # scanning a million-file cache just to remove them is prohibitively
        # expensive and could interfere with another active reader/producer.
        "stale_temporary_files_removed": 0,
        "cancelled": False,
    }
    scan_group_work_items: Dict[
        int,
        Dict[Union[int, float, str], List[Tuple[int, dict]]],
    ] = {}
    pending_patterns = 0
    validated_existing_directories = set()
    existing_group_directories: Dict[str, bool] = {}
    for scan in sorted(records_by_scan):
        for shot, shot_records in sorted(records_by_scan[scan].items()):
            unique_targets = {}
            for record in shot_records:
                for window, tag, signature in window_info:
                    output_path = _single_shot_path(
                        Path(metadata["analysis_dir"]),
                        record,
                        tag,
                    )
                    parent_key = str(output_path.parent)
                    parent_exists = existing_group_directories.get(parent_key)
                    if parent_exists is None:
                        parent_exists = output_path.parent.is_dir()
                        existing_group_directories[parent_key] = parent_exists
                    unique_targets[str(output_path)] = (
                        window,
                        output_path,
                        signature,
                        (
                            _existing_single_shot_path(
                                Path(metadata["analysis_dir"]),
                                record,
                                tag,
                            )
                            if not overwrite and parent_exists
                            else None
                        ),
                    )
            targets = list(unique_targets.values())
            if not overwrite:
                for _window, _output_path, signature, existing_path in targets:
                    if existing_path is None:
                        continue
                    validation_key = (str(existing_path.parent), signature)
                    if validation_key in validated_existing_directories:
                        continue
                    cached_signature, cached_payload = _read_single_shot_settings(
                        existing_path
                    )
                    if cached_signature != signature:
                        expected_payload = settings_payloads[signature]
                        raise ValueError(
                            _settings_mismatch_message(
                                path=existing_path,
                                cached_signature=cached_signature,
                                cached_payload=cached_payload,
                                expected_payload=expected_payload,
                            )
                        )
                    validated_existing_directories.add(validation_key)
            pending = [
                item for item in targets if overwrite or item[3] is None
            ]
            report["existing_patterns"] = int(report["existing_patterns"]) + (
                len(targets) - len(pending)
            )
            if not pending:
                continue
            pending_patterns += len(pending)
            pending_by_window: Dict[
                Tuple[Tuple[float, float], str], List[Path]
            ] = {}
            for window, output_path, signature, _existing_path in pending:
                pending_by_window.setdefault((window, signature), []).append(
                    str(output_path)
                )
            group_keys = {record.group_key for record in shot_records}
            group_key = sorted(group_keys, key=lambda value: str(value))[0]
            scan_group_work_items.setdefault(int(scan), {}).setdefault(
                group_key,
                [],
            ).append((int(shot), pending_by_window))

    pending_shots = sum(
        len(items)
        for groups in scan_group_work_items.values()
        for items in groups.values()
    )
    resolved_workers = (
        max(1, min(4, os.cpu_count() or 2))
        if max_workers is None
        else int(max_workers)
    )
    if resolved_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    requested_start_method = str(start_method).strip().lower()
    if requested_start_method not in mp.get_all_start_methods():
        raise ValueError(
            "Unsupported multiprocessing start_method {!r}. Available methods: {}.".format(
                start_method,
                ", ".join(mp.get_all_start_methods()),
            )
        )

    parallel_enabled = bool(use_parallel) and resolved_workers > 1 and pending_shots > 1
    if parallel_enabled and requested_start_method == "spawn" and not _spawn_context_is_usable():
        print(
            "FemtoMAX single-shot multiprocessing disabled because the current "
            "interactive entry point cannot be safely imported by spawn workers."
        )
        parallel_enabled = False

    common_payload = {
        "poni_path": str(Path(poni_path).expanduser().resolve()),
        "mask_edf_path": (
            None
            if mask_edf_path is None or not str(mask_edf_path).strip()
            else str(Path(mask_edf_path).expanduser().resolve())
        ),
        "npt": int(npt),
        "azim_offset_deg": float(azim_offset_deg),
        "polarization_factor": integrator.polarization_factor,
        "read_batch_size": read_batch_size,
        "detector_h5_path": tuple(Experiment.PILATUS_H5_PATH),
        "raw_handle_cache_size": min(
            max(1, len(records_by_scan)),
            _MAX_WORKER_RAW_CACHE_SIZE,
        ),
        "settings_payloads": settings_payloads,
    }
    payloads_by_group: Dict[Union[int, float, str], deque] = {}
    effective_chunk_size = (
        work_chunk_size if parallel_enabled else read_batch_size
    )
    for scan, grouped_items in sorted(scan_group_work_items.items()):
        raw_path = Path(paths.raw_root) / str(scan_file_pattern).format(scan=scan)
        if not raw_path.is_file():
            raise FileNotFoundError(str(raw_path))
        for group_key, work_items in sorted(
            grouped_items.items(),
            key=lambda item: str(item[0]),
        ):
            group_label = (
                "dark"
                if group_key == "dark"
                else f"{metadata['scan_type']} {group_key}"
            )
            group_queue = payloads_by_group.setdefault(group_key, deque())
            for start in range(0, len(work_items), effective_chunk_size):
                group_queue.append(
                    {
                        "scan": scan,
                        "raw_path": str(raw_path),
                        "work_items": work_items[
                            start : start + effective_chunk_size
                        ],
                        "group_label": group_label,
                    }
                )

    payloads = []
    group_order = sorted(payloads_by_group, key=lambda value: str(value))
    while any(payloads_by_group.values()):
        for group_key in group_order:
            if payloads_by_group[group_key]:
                payloads.append(payloads_by_group[group_key].popleft())

    worker_count = (
        min(resolved_workers, len(payloads)) if parallel_enabled else 1
    )
    scan_affinity_lanes = (
        _build_scan_affinity_lanes(payloads, worker_count)
        if parallel_enabled
        else []
    )
    report.update(
        pending_shots=pending_shots,
        pending_patterns=int(pending_patterns),
        use_parallel=parallel_enabled,
        max_workers=worker_count,
        start_method=(requested_start_method if parallel_enabled else "serial"),
        task_count=len(payloads),
        worker_lane_count=(
            len(scan_affinity_lanes) if scan_affinity_lanes else len(payloads)
        ),
        scan_affinity=bool(scan_affinity_lanes),
        raw_handle_cache_size=int(common_payload["raw_handle_cache_size"]),
    )

    print(
        "FemtoMAX single-shot integration starting: "
        f"{pending_shots} pending shots producing {pending_patterns} patterns "
        f"in {len(payloads)} balanced chunks; "
        f"{report['max_workers']} worker(s); HDF5 batch {read_batch_size}; "
        f"worker task {effective_chunk_size}; up to "
        f"{report['raw_handle_cache_size']} retained raw scan(s) per worker"
        + ("; scan-affine scheduling enabled." if scan_affinity_lanes else "."),
        flush=True,
    )

    def add_result(part):
        report["written_patterns"] = int(report["written_patterns"]) + int(
            part["written_patterns"]
        )
        report["invalid_shots"] = int(report["invalid_shots"]) + int(
            part["invalid_shots"]
        )
        if part.get("cancelled"):
            report["cancelled"] = True

    if parallel_enabled and payloads:
        context = mp.get_context(requested_start_method)
        progress_queue = context.Queue()
        process_cancel_event = context.Event()
        pool = context.Pool(
            processes=worker_count,
            initializer=_initialize_work_process,
            initargs=(common_payload, progress_queue, process_cancel_event),
        )
        try:
            pool_function = (
                _integrate_worker_lane
                if scan_affinity_lanes
                else _integrate_work_chunk
            )
            pool_payloads = scan_affinity_lanes or payloads
            results = pool.imap_unordered(
                pool_function,
                pool_payloads,
                chunksize=1,
            )
            completed_pool_tasks = 0
            processed_from_results = 0
            with tqdm(
                total=pending_shots,
                desc="FemtoMAX single-shot integration (parallel)",
                unit="shot",
                dynamic_ncols=True,
            ) as progress:
                while completed_pool_tasks < len(pool_payloads):
                    if cancel_event is not None and cancel_event.is_set():
                        process_cancel_event.set()
                        report["cancelled"] = True
                    try:
                        part = results.next(timeout=0.1)
                    except mp.TimeoutError:
                        part = None
                    if part is not None:
                        add_result(part)
                        processed_from_results += int(part["processed_shots"])
                        completed_pool_tasks += 1
                    while True:
                        try:
                            processed_count, group_label = progress_queue.get_nowait()
                        except queue.Empty:
                            break
                        progress.update(int(processed_count))
                        if group_label:
                            progress.set_postfix_str(str(group_label), refresh=False)
                if progress.n < processed_from_results:
                    progress.update(processed_from_results - progress.n)
            pool.close()
            pool.join()
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        finally:
            progress_queue.close()
            progress_queue.join_thread()
    elif payloads:
        _initialize_work_process(common_payload, cancel_event=cancel_event)
        try:
            with tqdm(
                total=pending_shots,
                desc="FemtoMAX single-shot integration",
                unit="shot",
                dynamic_ncols=True,
            ) as progress:
                for payload in payloads:
                    if cancel_event is not None and cancel_event.is_set():
                        report["cancelled"] = True
                        break
                    part = _integrate_work_chunk(payload)
                    add_result(part)
                    progress.update(int(part["processed_shots"]))
                    progress.set_postfix_str(
                        str(payload.get("group_label", "")),
                        refresh=False,
                    )
                    if part.get("cancelled"):
                        break
        finally:
            _clear_worker_state()

    print(
        "FemtoMAX single-shot integration: "
        f"{report['written_patterns']} written, {report['existing_patterns']} existing, "
        f"{report['invalid_shots']} invalid shots skipped"
        + ("; stopped safely." if report["cancelled"] else ".")
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
        raise ValueError(f"No {name} values were requested.")
    return requested


def _build_available_shot_index(
    *,
    metadata: Dict[str, object],
    records: Sequence[SingleShotRecord],
    windows: Sequence[Tuple[float, float]],
) -> Tuple[
    Dict[Tuple[Union[int, float, str], str], List[Path]],
    Dict[Union[int, float, str], int],
]:
    """Index one atomic filesystem snapshot of completed flat and legacy files."""
    analysis_dir = Path(metadata["analysis_dir"])
    records_by_group: Dict[Union[int, float, str], List[SingleShotRecord]] = {}
    for record in records:
        records_by_group.setdefault(record.group_key, []).append(record)

    expected_counts = {
        group_key: len({(record.scan, record.shot) for record in group_records})
        for group_key, group_records in records_by_group.items()
    }
    index: Dict[Tuple[Union[int, float, str], str], List[Path]] = {}

    for window in windows:
        azimuth_tag = general_utils.azim_range_str(window)
        root = analysis_dir / SINGLE_SHOT_FOLDER / azimuth_tag
        if not root.is_dir():
            continue

        name_to_group: Dict[str, Union[int, float, str]] = {}
        canonical_parent_by_name: Dict[str, Path] = {}
        for group_key, group_records in records_by_group.items():
            for record in group_records:
                canonical = _single_shot_path(analysis_dir, record, azimuth_tag)
                name_to_group[canonical.name] = group_key
                canonical_parent_by_name[canonical.name] = canonical.parent

        selected_by_group: Dict[
            Union[int, float, str], Dict[str, Path]
        ] = {}
        for path in root.rglob("*.xy"):
            group_key = name_to_group.get(path.name)
            if group_key is None:
                continue
            selected = selected_by_group.setdefault(group_key, {})
            current = selected.get(path.name)
            if current is None or path.parent == canonical_parent_by_name[path.name]:
                selected[path.name] = path

        for group_key, selected in selected_by_group.items():
            index[(group_key, azimuth_tag)] = [
                selected[name] for name in sorted(selected)
            ]

    return index, expected_counts


def _available_group_values(
    *,
    index: Dict[Tuple[Union[int, float, str], str], List[Path]],
    windows: Sequence[Tuple[float, float]],
) -> List[Union[int, float, str]]:
    """Return group values having at least one completed file in every window."""
    azimuth_tags = [general_utils.azim_range_str(window) for window in windows]
    group_values = {group_key for group_key, _tag in index}
    return sorted(
        (
            group_key
            for group_key in group_values
            if all(index.get((group_key, tag)) for tag in azimuth_tags)
        ),
        key=lambda value: str(value),
    )


def _match_available_group(
    value: Union[int, float, str],
    available: Sequence[Union[int, float, str]],
) -> Optional[Union[int, float, str]]:
    """Match an explicit delay or fluence selector to one discovered group key."""
    if value == "dark":
        return "dark" if "dark" in available else None
    for candidate in available:
        if candidate == "dark":
            continue
        if np.isclose(float(candidate), float(value), rtol=0.0, atol=1e-12):
            return candidate
    return None


def _select_completed_groups(
    selector,
    *,
    metadata_values: Sequence[Union[int, float]],
    available_values: Sequence[Union[int, float, str]],
    name: str,
) -> List[Union[int, float, str]]:
    """Resolve requested values while skipping metadata groups not completed yet."""
    if isinstance(selector, str) and selector.strip().lower() == "all":
        selected = list(available_values)
    else:
        requested = _selected_values(selector, metadata_values, name=name)
        selected = []
        for value in requested:
            match = _match_available_group(value, available_values)
            if match is None:
                print(
                    f"Skipping {name} {value}: no completed compatible "
                    "single-shot files are available yet.",
                    flush=True,
                )
            elif match not in selected:
                selected.append(match)
    if not selected:
        raise FileNotFoundError(
            f"No completed single-shot {name} groups are currently available."
        )
    return selected


def _expected_payloads_for_windows(
    *,
    integrator: azimint_utils.AzimIntegrator,
    windows: Sequence[Tuple[float, float]],
    poni_path: Union[str, Path],
    mask_edf_path: Optional[Union[str, Path]],
    poni_sha256: str,
    mask_sha256: Optional[str],
) -> Dict[str, Dict[str, object]]:
    """Build requested cache provenance keyed by canonical azimuth tag."""
    return {
        general_utils.azim_range_str(window): _settings_payload(
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            npt=integrator.npt,
            azimuthal_range=window,
            azim_offset_deg=integrator.azim_offset_deg,
            polarization_factor=integrator.polarization_factor,
            poni_sha256=poni_sha256,
            mask_sha256=mask_sha256,
        )
        for window in windows
    }


def _validate_selected_group_signatures(
    *,
    selected_groups: Sequence[Union[int, float, str]],
    windows: Sequence[Tuple[float, float]],
    availability_index: Dict[
        Tuple[Union[int, float, str], str], List[Path]
    ],
    expected_payloads: Dict[str, Dict[str, object]],
) -> None:
    """Reject incompatible groups before announcing them as aggregatable."""
    for group_value in selected_groups:
        for window in windows:
            azimuth_tag = general_utils.azim_range_str(window)
            paths = availability_index.get((group_value, azimuth_tag), [])
            if not paths:
                continue
            cached_signature, cached_payload = _read_single_shot_settings(paths[0])
            expected_payload = expected_payloads[azimuth_tag]
            if cached_signature != _settings_signature_from_payload(expected_payload):
                raise ValueError(
                    _settings_mismatch_message(
                        path=paths[0],
                        cached_signature=cached_signature,
                        cached_payload=cached_payload,
                        expected_payload=expected_payload,
                    )
                )


def _aggregate_cache_path(
    metadata_dir: Path,
    *,
    azimuth_tag: str,
    group_value: Union[int, float, str],
) -> Path:
    """Return the hidden incremental accumulator path for one final pattern."""
    if group_value == "dark":
        group_tag = "dark"
    elif isinstance(group_value, (int, np.integer)):
        group_tag = f"delay_{int(group_value)}fs"
    else:
        group_tag = f"fluence_{general_utils.fluence_tag_file(float(group_value))}mJ"
    return (
        metadata_dir
        / SINGLE_SHOT_FOLDER
        / _AGGREGATE_CACHE_FOLDER
        / azimuth_tag
        / f"{group_tag}.npz"
    )


def _save_aggregate_cache_atomic(path: Path, **arrays) -> None:
    """Publish one incremental accumulator atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _accumulate_single_shot_paths(
    *,
    paths: Sequence[Path],
    expected_signature: str,
    expected_payload: Dict[str, object],
    reference_x: Optional[np.ndarray] = None,
    summed_intensity: Optional[np.ndarray] = None,
    finite_count: Optional[np.ndarray] = None,
    description: str,
    read_workers: int = 1,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and reduce completed XY files with bounded concurrent text parsing."""
    read_workers = max(1, min(int(read_workers), len(paths)))

    def consume(loaded, path):
        nonlocal reference_x, summed_intensity, finite_count
        signature, settings_payload, two_theta, intensity = loaded
        if signature != expected_signature:
            raise ValueError(
                _settings_mismatch_message(
                    path=path,
                    cached_signature=signature,
                    cached_payload=settings_payload,
                    expected_payload=expected_payload,
                )
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
            raise ValueError(f"Incompatible radial grids in single-shot cache: {path}")
        if summed_intensity is None:
            summed_intensity = np.zeros(intensity.shape, dtype=float)
            finite_count = np.zeros(intensity.shape, dtype=np.int64)
        finite = np.isfinite(intensity)
        summed_intensity[finite] += intensity[finite]
        finite_count[finite] += 1

    if read_workers == 1:
        with tqdm(
            paths,
            desc=description,
            unit="shot",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as progress:
            for path in progress:
                consume(_load_single_shot_xy(path), path)
        return reference_x, summed_intensity, finite_count

    with ThreadPoolExecutor(max_workers=read_workers) as executor:
        with tqdm(
            total=len(paths),
            desc=description,
            unit="shot",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as progress:
            for start in range(0, len(paths), 256):
                path_batch = list(paths[start : start + 256])
                loaded_values = executor.map(_load_single_shot_xy, path_batch)
                for path, loaded in zip(path_batch, loaded_values):
                    consume(loaded, path)
                    progress.update(1)

    return reference_x, summed_intensity, finite_count


def _aggregate_dataset(
    *,
    metadata: Dict[str, object],
    group_value: Union[int, float, str],
    dataset,
    integrator: azimint_utils.AzimIntegrator,
    windows: Sequence[Tuple[float, float]],
    poni_path: Union[str, Path],
    mask_edf_path: Optional[Union[str, Path]],
    overwrite_xy: bool,
    poni_sha256: str,
    mask_sha256: Optional[str],
    availability_index: Dict[
        Tuple[Union[int, float, str], str], List[Path]
    ],
    expected_shots: int,
    read_workers: int = 1,
    show_progress: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Average available compatible single-shot patterns into final XY files."""
    metadata_dir = Path(metadata["analysis_dir"]).resolve()
    dataset_dir = Path(dataset.analysis_dir()).resolve()
    if dataset_dir != metadata_dir:
        raise ValueError(
            "The experiment fields do not identify the selected FemtoMAX metadata "
            f"directory. Expected {metadata_dir}, got {dataset_dir}."
        )
    result: Dict[str, Dict[str, object]] = {}
    for window in windows:
        azimuth_tag = general_utils.azim_range_str(window)
        available_paths = list(
            availability_index.get((group_value, azimuth_tag), [])
        )
        if not available_paths:
            continue
        output_path = dataset.xy_path(azimuth_tag)
        if output_path.is_file() and not overwrite_xy:
            result[azimuth_tag] = {
                "path": str(output_path),
                "available_shots": len(available_paths),
                "expected_shots": int(expected_shots),
                "status": "existing",
            }
            continue

        expected_payload = _settings_payload(
            poni_path=poni_path,
            mask_edf_path=mask_edf_path,
            npt=integrator.npt,
            azimuthal_range=window,
            azim_offset_deg=integrator.azim_offset_deg,
            polarization_factor=integrator.polarization_factor,
            poni_sha256=poni_sha256,
            mask_sha256=mask_sha256,
        )
        expected_signature = _settings_signature_from_payload(expected_payload)
        reference_x: Optional[np.ndarray] = None
        summed_intensity: Optional[np.ndarray] = None
        finite_count: Optional[np.ndarray] = None

        file_stats = {}
        for path in available_paths:
            stat = path.stat()
            file_stats[path.name] = (int(stat.st_mtime_ns), int(stat.st_size))
        cache_path = _aggregate_cache_path(
            metadata_dir,
            azimuth_tag=azimuth_tag,
            group_value=group_value,
        )
        cached_names = set()
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cache:
                    cache_signature = str(cache["settings_signature"].item())
                    names = [str(value) for value in cache["file_names"].tolist()]
                    mtimes = [int(value) for value in cache["mtimes_ns"].tolist()]
                    sizes = [int(value) for value in cache["sizes"].tolist()]
                    cached_stats = dict(zip(names, zip(mtimes, sizes)))
                    cache_is_valid = (
                        cache_signature == expected_signature
                        and all(
                            file_stats.get(name) == stat
                            for name, stat in cached_stats.items()
                        )
                    )
                    if cache_is_valid:
                        reference_x = np.array(cache["x"], dtype=float, copy=True)
                        summed_intensity = np.array(
                            cache["summed_intensity"],
                            dtype=float,
                            copy=True,
                        )
                        finite_count = np.array(
                            cache["finite_count"],
                            dtype=np.int64,
                            copy=True,
                        )
                        cached_names = set(cached_stats)
            except (OSError, ValueError, KeyError):
                cached_names = set()
                reference_x = None
                summed_intensity = None
                finite_count = None

        new_paths = [path for path in available_paths if path.name not in cached_names]
        if new_paths:
            reference_x, summed_intensity, finite_count = (
                _accumulate_single_shot_paths(
                    paths=new_paths,
                    expected_signature=expected_signature,
                    expected_payload=expected_payload,
                    reference_x=reference_x,
                    summed_intensity=summed_intensity,
                    finite_count=finite_count,
                    description=(
                        f"Aggregate {metadata['scan_type']} {group_value} "
                        f"azimuth {azimuth_tag}"
                    ),
                    read_workers=read_workers,
                    show_progress=show_progress,
                )
            )

        if reference_x is None or summed_intensity is None or finite_count is None:
            raise FileNotFoundError(
                f"No readable completed single-shot patterns are available for "
                f"{group_value}, azimuth {azimuth_tag}."
            )

        ordered_names = [path.name for path in available_paths]
        _save_aggregate_cache_atomic(
            cache_path,
            settings_signature=np.asarray(expected_signature),
            file_names=np.asarray(ordered_names),
            mtimes_ns=np.asarray(
                [file_stats[name][0] for name in ordered_names],
                dtype=np.int64,
            ),
            sizes=np.asarray(
                [file_stats[name][1] for name in ordered_names],
                dtype=np.int64,
            ),
            x=reference_x,
            summed_intensity=summed_intensity,
            finite_count=finite_count,
        )

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
                q,
                mean_intensity,
                integrator.q_norm_range,
            )
        _save_final_xy_atomic(output_path, reference_x, mean_intensity)
        result[azimuth_tag] = {
            "path": str(output_path),
            "available_shots": len(available_paths),
            "expected_shots": int(expected_shots),
            "status": "written",
            "new_shots_read": len(new_paths),
            "cached_shots_reused": len(cached_names),
        }
        print(
            f"Final {metadata['scan_type']} XY {output_path.name}: "
            f"{len(available_paths)}/{expected_shots} shots available "
            f"({len(new_paths)} newly read, {len(cached_names)} cached).",
            flush=True,
        )
    return result


def _aggregate_dataset_jobs(
    *,
    jobs,
    use_parallel,
    max_workers,
    aggregate_kwargs,
):
    """Aggregate independent final groups without creating nested worker pools."""
    jobs = list(jobs)
    requested_workers = azimint_utils.resolve_parallel_worker_count(
        use_parallel=use_parallel,
        max_workers=max_workers,
    )
    group_workers = min(requested_workers, max(1, len(jobs)))
    read_workers = 1 if group_workers > 1 else requested_workers

    def aggregate_one(job):
        group_value, dataset, expected_shots = job
        return _aggregate_dataset(
            group_value=group_value,
            dataset=dataset,
            expected_shots=expected_shots,
            read_workers=read_workers,
            show_progress=group_workers == 1,
            **aggregate_kwargs,
        )

    if group_workers == 1:
        for job in jobs:
            aggregate_one(job)
        return

    print(
        f"Final {aggregate_kwargs['metadata']['scan_type']} aggregation starting: "
        f"{len(jobs)} group(s), {group_workers} worker(s).",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=group_workers) as executor:
        list(executor.map(aggregate_one, jobs))


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
    """Create the final-pattern integrator and canonical window list."""
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


def aggregate_delay_1d(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    fluence_mJ_cm2: float,
    time_window_fs: int,
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
    use_parallel=True,
    max_workers=None,
    paths: AnalysisPaths,
):
    """Build delay-series final XY files from available single-shot patterns."""
    metadata, records = _metadata_records(metadata_h5_path)
    if metadata["scan_type"] != "delay":
        raise ValueError("The selected metadata HDF5 is not a FemtoMAX delay scan.")
    metadata_values = sorted({int(record.delay_fs) for record in records})
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
    poni_sha256 = _file_sha256(poni_path)
    mask_sha256 = _file_sha256(mask_edf_path)
    expected_payloads = _expected_payloads_for_windows(
        integrator=integrator,
        windows=windows,
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        poni_sha256=poni_sha256,
        mask_sha256=mask_sha256,
    )
    availability_index, expected_counts = _build_available_shot_index(
        metadata=metadata,
        records=records,
        windows=windows,
    )
    available = _available_group_values(index=availability_index, windows=windows)
    selected = [
        int(value)
        for value in _select_completed_groups(
            delays_fs,
            metadata_values=metadata_values,
            available_values=available,
            name="delay",
        )
    ]
    _validate_selected_group_signatures(
        selected_groups=selected,
        windows=windows,
        availability_index=availability_index,
        expected_payloads=expected_payloads,
    )
    print(
        "FemtoMAX delay aggregation snapshot: "
        f"{len(selected)} completed delay group(s) available.",
        flush=True,
    )
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
        jobs.append((delay, dataset, expected_counts[delay]))
    _aggregate_dataset_jobs(
        jobs=jobs,
        use_parallel=use_parallel,
        max_workers=max_workers,
        aggregate_kwargs={
            "metadata": metadata,
            "integrator": integrator,
            "windows": windows,
            "poni_path": poni_path,
            "mask_edf_path": mask_edf_path,
            "overwrite_xy": bool(overwrite_xy),
            "poni_sha256": poni_sha256,
            "mask_sha256": mask_sha256,
            "availability_index": availability_index,
        },
    )
    return integrator, datasets


def aggregate_fluence_1d(
    *,
    sample_name: str,
    temperature_K: int,
    excitation_wl_nm: float,
    delay_fs: int,
    time_window_fs: int,
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
    use_parallel=True,
    max_workers=None,
    paths: AnalysisPaths,
):
    """Build fluence-series final XY files from available single-shot patterns."""
    metadata, records = _metadata_records(metadata_h5_path)
    if metadata["scan_type"] != "fluence":
        raise ValueError("The selected metadata HDF5 is not a FemtoMAX fluence scan.")
    metadata_values = sorted({float(record.fluence_mJ_cm2) for record in records})
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
    poni_sha256 = _file_sha256(poni_path)
    mask_sha256 = _file_sha256(mask_edf_path)
    expected_payloads = _expected_payloads_for_windows(
        integrator=integrator,
        windows=windows,
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        poni_sha256=poni_sha256,
        mask_sha256=mask_sha256,
    )
    availability_index, expected_counts = _build_available_shot_index(
        metadata=metadata,
        records=records,
        windows=windows,
    )
    available = _available_group_values(index=availability_index, windows=windows)
    selected = [
        float(value)
        for value in _select_completed_groups(
            fluences_mJ_cm2,
            metadata_values=metadata_values,
            available_values=available,
            name="fluence",
        )
    ]
    _validate_selected_group_signatures(
        selected_groups=selected,
        windows=windows,
        availability_index=availability_index,
        expected_payloads=expected_payloads,
    )
    print(
        "FemtoMAX fluence aggregation snapshot: "
        f"{len(selected)} completed fluence group(s) available.",
        flush=True,
    )
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
        jobs.append((fluence, dataset, expected_counts[fluence]))
    _aggregate_dataset_jobs(
        jobs=jobs,
        use_parallel=use_parallel,
        max_workers=max_workers,
        aggregate_kwargs={
            "metadata": metadata,
            "integrator": integrator,
            "windows": windows,
            "poni_path": poni_path,
            "mask_edf_path": mask_edf_path,
            "overwrite_xy": bool(overwrite_xy),
            "poni_sha256": poni_sha256,
            "mask_sha256": mask_sha256,
            "availability_index": availability_index,
        },
    )
    return integrator, datasets


def aggregate_dark_1d(
    *,
    sample_name: str,
    temperature_K: int,
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
    use_parallel=True,
    max_workers=None,
    paths: AnalysisPaths,
):
    """Build dark final XY files from available single-shot patterns."""
    metadata, records = _metadata_records(metadata_h5_path)
    if metadata["scan_type"] != "dark":
        raise ValueError("The selected metadata HDF5 is not a FemtoMAX dark scan.")
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
    poni_sha256 = _file_sha256(poni_path)
    mask_sha256 = _file_sha256(mask_edf_path)
    expected_payloads = _expected_payloads_for_windows(
        integrator=integrator,
        windows=windows,
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        poni_sha256=poni_sha256,
        mask_sha256=mask_sha256,
    )
    availability_index, expected_counts = _build_available_shot_index(
        metadata=metadata,
        records=records,
        windows=windows,
    )
    available = _available_group_values(index=availability_index, windows=windows)
    if "dark" not in available:
        raise FileNotFoundError(
            "No completed single-shot dark files are currently available."
        )
    _validate_selected_group_signatures(
        selected_groups=["dark"],
        windows=windows,
        availability_index=availability_index,
        expected_payloads=expected_payloads,
    )
    _aggregate_dataset(
        metadata=metadata,
        group_value="dark",
        dataset=dataset,
        integrator=integrator,
        windows=windows,
        poni_path=poni_path,
        mask_edf_path=mask_edf_path,
        overwrite_xy=bool(overwrite_xy),
        poni_sha256=poni_sha256,
        mask_sha256=mask_sha256,
        availability_index=availability_index,
        expected_shots=expected_counts["dark"],
        read_workers=azimint_utils.resolve_parallel_worker_count(
            use_parallel=use_parallel,
            max_workers=max_workers,
        ),
    )
    return integrator, dataset


__all__ = [
    "SINGLE_SHOT_FOLDER",
    "SingleShotRecord",
    "resolve_metadata_h5_path",
    "integrate_single_shot_1d",
    "aggregate_dark_1d",
    "aggregate_delay_1d",
    "aggregate_fluence_1d",
]
