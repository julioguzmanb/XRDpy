# datared_utils.py
"""
FemtoMAX-specific data reduction utilities.

Beamline-specific scope:
- read raw FemtoMAX scan files
- build metadata HDF5 for delay / dark / fluence workflows
- export averaged 2D detector images from selected shots

Path handling:
- no config.py dependency
- provide either:
    * paths=AnalysisPaths(...)
  or
    * path_root=... and analysis_subdir=...
- raw_subdir can be provided explicitly, or taken from AnalysisPaths.raw_subdir
"""

from __future__ import annotations

import os
import sys
import csv
import hashlib
import multiprocessing as mp
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union, List, Dict, Any

import h5py as h5
import numpy as np
from tqdm import tqdm

from ..common import plot_utils
from ..common import general_utils
from ..common.paths import AnalysisPaths


# Default raw data location (override as needed)
DEFAULT_RAW_SUBDIR = ""
DEFAULT_SCAN_FILE_PATTERN = "scan-{scan}.h5"

# HDF5 dataset paths for pings (beamline specific)
PING2_H5_PATH = ("entry", "measurement", "oscc_02_maui", "Ping_Ch2_value")
PING4_H5_PATH = ("entry", "measurement", "oscc_02_maui", "Ping_Ch4_value")


# ----------------------------
# Path helpers
# ----------------------------
def _coerce_paths(
    *,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    raw_subdir: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
) -> AnalysisPaths:
    """Normalize path configuration into an AnalysisPaths instance.

    Accepted inputs:
      - paths=AnalysisPaths(...)
      - path_root=... and analysis_subdir=... (raw_subdir optional)
    """
    if paths is not None:
        if raw_subdir is not None or analysis_subdir is not None:
            return AnalysisPaths(
                path_root=Path(paths.path_root),
                raw_subdir=str(raw_subdir) if raw_subdir is not None else str(paths.raw_subdir),
                analysis_subdir=str(analysis_subdir) if analysis_subdir is not None else str(paths.analysis_subdir),
                values=dict(paths.values),
            )
        return paths

    if path_root is None or analysis_subdir is None:
        raise ValueError(
            "Provide either paths=AnalysisPaths(...), or both "
            "path_root=... and analysis_subdir=...."
        )

    return AnalysisPaths(
        path_root=Path(path_root),
        raw_subdir=(str(raw_subdir) if raw_subdir is not None else str(DEFAULT_RAW_SUBDIR)),
        analysis_subdir=str(analysis_subdir),
    )


def _analysis_root(
    *,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
    raw_subdir: Optional[Union[str, Path]] = None,
) -> Path:
    """Return analysis root."""
    ap = _coerce_paths(
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
    )
    return Path(ap.analysis_root)


def _raw_root(
    *,
    paths: Optional[AnalysisPaths] = None,
    path_root: Optional[Union[str, Path]] = None,
    analysis_subdir: Optional[Union[str, Path]] = None,
    raw_subdir: Optional[Union[str, Path]] = None,
) -> Path:
    """Return raw root."""
    ap = _coerce_paths(
        paths=paths,
        path_root=path_root,
        raw_subdir=raw_subdir,
        analysis_subdir=analysis_subdir,
    )
    return Path(ap.raw_root)


# ----------------------------
# Formatting helpers (avoid 1500.0nm in paths / filenames)
# ----------------------------
def _wl_tag_nm_local(x: Union[int, float]) -> str:
    """Returns a string suitable for folder/file naming, avoiding trailing '.0' when integer-like.
    Uses general_utils.wl_tag_nm if available but falls back to robust formatting.
    """
    fn = getattr(general_utils, "wl_tag_nm", None)
    if callable(fn):
        s = str(fn(x))
        if s.endswith(".0"):
            try:
                f = float(s)
                if abs(f - round(f)) < 1e-9:
                    return str(int(round(f)))
            except Exception:
                pass
        return s

    v = float(x)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


DEFAULT_PING_REFERENCES_PATH = Path(__file__).with_name(
    "ping_references_default.csv"
)


@dataclass(frozen=True)
class PingReferenceRange:
    """Associate an inclusive FemtoMAX scan range with reference ping values.

    ``scan_start`` and ``scan_end`` are inclusive. ``ping2_ref`` and
    ``ping4_ref`` provide the timing-tool zero references used when correcting
    every scan in that range.
    """
    scan_start: int
    scan_end: int
    ping2_ref_s: float
    ping4_ref_s: float


@dataclass(frozen=True)
class PingReferenceTable:
    """Validate and query FemtoMAX timing-tool reference ranges.

    Ranges must be ordered, non-overlapping, and internally valid. Lookup maps a
    scan number to exactly one ping-2/ping-4 reference pair; uncovered scans are
    reported explicitly so stale metadata cannot be reused silently.
    """
    path: Path
    ranges: Tuple[PingReferenceRange, ...]
    sha256: str

    def __post_init__(self):
        """Validate and normalize the initialized fields."""
        object.__setattr__(
            self,
            "_starts",
            tuple(item.scan_start for item in self.ranges),
        )

    @property
    def scan_min(self) -> int:
        """Return scan min."""
        return self.ranges[0].scan_start

    @property
    def scan_max(self) -> int:
        """Return scan max."""
        return self.ranges[-1].scan_end

    def reference_for(self, scan: int) -> Tuple[float, float]:
        """Return reference for."""
        scan = int(scan)
        index = bisect_right(self._starts, scan) - 1
        if index >= 0:
            item = self.ranges[index]
            if scan <= item.scan_end:
                return item.ping2_ref_s, item.ping4_ref_s
        raise KeyError(
            f"No ping reference is configured for scan {scan} in {self.path}."
        )

    def missing_scans(self, scans: Sequence[int]) -> List[int]:
        """Return missing scans."""
        missing: List[int] = []
        for scan in scans:
            try:
                self.reference_for(int(scan))
            except KeyError:
                missing.append(int(scan))
        return missing

    def validate_scans(self, scans: Sequence[int]) -> None:
        """Verify that every requested scan is covered by exactly one reference range."""
        missing = self.missing_scans(scans)
        if missing:
            raise KeyError(
                "The ping-reference file does not cover scan(s) "
                f"{missing}: {self.path}"
            )


def default_ping_reference_path() -> Path:
    """Return the packaged FemtoMAX ping-reference CSV path."""
    return DEFAULT_PING_REFERENCES_PATH


def _parse_ping_reference_csv(path: Path) -> PingReferenceTable:
    """Parse ping reference CSV."""
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    content_lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not content_lines:
        raise ValueError(f"Ping-reference file is empty: {path}")

    reader = csv.DictReader(content_lines)
    required = ("scan_start", "scan_end", "ping2_ref_s", "ping4_ref_s")
    fields = tuple((name or "").strip() for name in (reader.fieldnames or ()))
    missing_fields = [name for name in required if name not in fields]
    if missing_fields:
        raise ValueError(
            f"Ping-reference file {path} is missing column(s): {missing_fields}. "
            f"Required columns are: {list(required)}"
        )
    reader.fieldnames = list(fields)

    ranges: List[PingReferenceRange] = []
    for row_number, row in enumerate(reader, start=2):
        try:
            scan_start = int(str(row["scan_start"]).strip())
            scan_end = int(str(row["scan_end"]).strip())
            ping2_ref_s = float(str(row["ping2_ref_s"]).strip())
            ping4_ref_s = float(str(row["ping4_ref_s"]).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid value in ping-reference file {path}, data row {row_number}."
            ) from exc

        if scan_start > scan_end:
            raise ValueError(
                f"Invalid scan range {scan_start}-{scan_end} in {path}: "
                "scan_start must be <= scan_end."
            )
        if not np.isfinite(ping2_ref_s) or not np.isfinite(ping4_ref_s):
            raise ValueError(
                f"Non-finite ping reference in {path}, data row {row_number}."
            )
        ranges.append(
            PingReferenceRange(
                scan_start=scan_start,
                scan_end=scan_end,
                ping2_ref_s=ping2_ref_s,
                ping4_ref_s=ping4_ref_s,
            )
        )

    if not ranges:
        raise ValueError(f"Ping-reference file has no data rows: {path}")

    ranges.sort(key=lambda item: (item.scan_start, item.scan_end))
    for previous, current in zip(ranges, ranges[1:]):
        if current.scan_start <= previous.scan_end:
            raise ValueError(
                "Overlapping ping-reference ranges in "
                f"{path}: {previous.scan_start}-{previous.scan_end} and "
                f"{current.scan_start}-{current.scan_end}."
            )

    return PingReferenceTable(
        path=path,
        ranges=tuple(ranges),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


@lru_cache(maxsize=32)
def _load_ping_reference_table_cached(
    path_text: str,
    modified_ns: int,
    size: int,
) -> PingReferenceTable:
    """Load ping reference table cached."""
    del modified_ns, size
    return _parse_ping_reference_csv(Path(path_text))


def load_ping_reference_table(
    path: Optional[Union[str, Path]] = None,
) -> PingReferenceTable:
    """Load and validate a ping-reference CSV, reloading it after file changes.

    Parameters
    ----------
    path : Optional[Union[str, Path]]
        Input filesystem path.

    Returns
    -------
    PingReferenceTable
        Validated, queryable ping-reference table.

    Raises
    ------
    FileNotFoundError
        If required raw data, calibration, metadata, or cached analysis files are missing.
    """
    resolved = Path(path or DEFAULT_PING_REFERENCES_PATH).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Ping-reference file not found: {resolved}")
    stat = resolved.stat()
    return _load_ping_reference_table_cached(
        str(resolved),
        int(stat.st_mtime_ns),
        int(stat.st_size),
    )


def ref_pings(
    scan: int,
    reference_path: Optional[Union[str, Path]] = None,
) -> Tuple[float, float]:
    """Return one scan using a validated CSV table (packaged default if omitted).

    Parameters
    ----------
    scan : int
        Facility scan identifier or scan collection accepted by the selected backend.
    reference_path : Optional[Union[str, Path]]
        Filesystem path for reference.

    Returns
    -------
    Tuple[float, float]
        Reference ping-2 and ping-4 values for the scan, in seconds.
    """
    return load_ping_reference_table(reference_path).reference_for(int(scan))


# Metadata (needed for saving/building)
@dataclass(frozen=True)
class ExperimentMeta:
    """Store normalized metadata for a FemtoMAX reduction run.

    The record combines sample conditions, pump settings, scan selection, delay
    window, and resolved data roots. ``Experiment`` uses it to construct the
    standardized metadata, 2D-image, and downstream analysis paths.
    """
    sample_name: str
    temperature_K: int
    excitation_wl_nm: Optional[Union[int, float]]                # can be None for dark
    fluence_mJ_cm2: Optional[Union[int, float, Sequence[float]]] # scalar for delay; sequence for fluence; None for dark
    scan_type: str = "delay"                                     # "delay" | "dark" | "fluence"
    time_window_fs: Optional[int] = 500                          # can be None for dark
    delay: Optional[Union[int, str]] = None                      # for scan_type="fluence" folder naming


# ----------------------------
# Multiprocessing helpers (MUST be top-level to be picklable)
# ----------------------------
def _h5_get_local(h5obj, path_tuple: Tuple[str, ...]):
    """Return HDF5 get local."""
    x = h5obj
    for k in path_tuple:
        x = x[k]
    return x


def _read_meta_value_local(meta_g: h5.Group, key: str, default=None):
    """Read meta value local."""
    if key in meta_g:
        v = meta_g[key][()]
        v = general_utils.decode_if_bytes(v)
        if isinstance(v, np.ndarray) and v.shape == ():
            return v.item()
        return v
    if key in meta_g.attrs:
        v = meta_g.attrs[key]
        v = general_utils.decode_if_bytes(v)
        if isinstance(v, np.ndarray) and v.shape == ():
            return v.item()
        return v
    return default


def _sum_frames_by_indices_local(
    dset,
    indices: np.ndarray,
    *,
    batch_size: int = 128,
) -> Tuple[Optional[np.ndarray], int]:
    """Sum selected frames, rejecting any frame that contains a NaN.
    Reads in contiguous batches for HDF5 efficiency.
    Returns (sum_image, n_good_frames).
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return None, 0

    n_frames = int(dset.shape[0])
    indices = indices[(indices >= 0) & (indices < n_frames)]
    if indices.size == 0:
        return None, 0

    indices = np.unique(indices)
    indices.sort()

    need_nan_check = (dset.dtype.kind in ("f", "c"))

    sum_img: Optional[np.ndarray] = None
    n_good = 0

    breaks = np.where(np.diff(indices) > 1)[0]
    run_starts = np.r_[0, breaks + 1]
    run_ends = np.r_[breaks + 1, indices.size]

    for rs, re in zip(run_starts, run_ends):
        start = int(indices[rs])
        stop = int(indices[re - 1]) + 1

        for i0 in range(start, stop, batch_size):
            i1 = min(i0 + batch_size, stop)
            frames = dset[i0:i1]
            if frames.size == 0:
                continue

            if sum_img is None:
                sum_img = np.zeros(frames.shape[1:], dtype=np.float64)

            p0 = np.searchsorted(indices, i0, side="left")
            p1 = np.searchsorted(indices, i1, side="left")
            if p1 <= p0:
                continue

            local = indices[p0:p1] - i0
            mask = np.zeros((i1 - i0,), dtype=bool)
            mask[local] = True

            selected = frames[mask]
            if selected.shape[0] == 0:
                continue

            if need_nan_check:
                bad = np.isnan(selected).any(axis=(1, 2))
                if np.any(bad):
                    selected = selected[~bad]
                    if selected.shape[0] == 0:
                        continue

            sum_img += selected.sum(axis=0, dtype=np.float64)
            n_good += int(selected.shape[0])

    return sum_img, n_good


def _export_delay_chunk_worker(payload: dict) -> Dict[int, Dict[str, Union[str, int]]]:
    """Worker processes a CHUNK of delays (delay scan_type)."""
    metadata_h5_path = payload["metadata_h5_path"]
    delays_chunk = payload["delays_chunk"]
    out_dir = payload["out_dir"]
    overwrite = payload["overwrite"]
    batch_size = payload["batch_size"]
    out_dtype_str = payload["out_dtype_str"]

    pilatus_h5_path = tuple(payload["pilatus_h5_path"])
    raw_root = str(payload["raw_root"])
    scan_file_pattern = payload["scan_file_pattern"]

    rdcc_nbytes = int(payload.get("rdcc_nbytes", 0))
    rdcc_nslots = int(payload.get("rdcc_nslots", 0))

    out_dtype = np.dtype(out_dtype_str)
    results: Dict[int, Dict[str, Union[str, int]]] = {}

    with h5.File(metadata_h5_path, "r") as f:
        if "meta" not in f or "delays" not in f:
            raise ValueError("Invalid metadata H5: missing /meta or /delays")

        meta_g = f["meta"]
        delays_root = f["delays"]

        sample_name = str(_read_meta_value_local(meta_g, "sample_name"))
        temperature_K = int(_read_meta_value_local(meta_g, "temperature_K"))
        excitation_wl_nm = float(_read_meta_value_local(meta_g, "excitation_wl_nm"))
        fluence_mJ_cm2 = float(_read_meta_value_local(meta_g, "fluence_mJ_cm2"))
        time_window_fs = int(_read_meta_value_local(meta_g, "time_window_fs"))

        wl_tag = _wl_tag_nm_local(excitation_wl_nm)
        flu_file = general_utils.fluence_tag_file(float(fluence_mJ_cm2))

        scan_to_idxbydelay: Dict[int, Dict[int, np.ndarray]] = {}

        for delay_fs in delays_chunk:
            delay_fs = int(delay_fs)
            gname = f"{delay_fs}fs"
            if gname not in delays_root:
                raise KeyError(f"Missing group /delays/{gname}")

            scans_g = delays_root[gname]["scans"]
            for scan_str in scans_g.keys():
                scan = int(scan_str)
                idx = np.array(scans_g[scan_str]["indices"], dtype=np.int64)
                if idx.size == 0:
                    continue
                scan_to_idxbydelay.setdefault(scan, {})[delay_fs] = idx

    total_sum: Dict[int, Optional[np.ndarray]] = {int(d): None for d in delays_chunk}
    total_n: Dict[int, int] = {int(d): 0 for d in delays_chunk}

    for scan, idx_by_delay in scan_to_idxbydelay.items():
        raw_path = os.path.join(raw_root, scan_file_pattern.format(scan=int(scan)))
        if not os.path.exists(raw_path):
            raise FileNotFoundError(raw_path)

        if rdcc_nbytes > 0 and rdcc_nslots > 0:
            rf = h5.File(raw_path, "r", rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots)
        else:
            rf = h5.File(raw_path, "r")

        try:
            dset = _h5_get_local(rf, pilatus_h5_path)
            for delay_fs, idx in idx_by_delay.items():
                scan_sum, n_good = _sum_frames_by_indices_local(dset, idx, batch_size=batch_size)
                if scan_sum is None or n_good == 0:
                    continue
                if total_sum[delay_fs] is None:
                    total_sum[delay_fs] = np.zeros_like(scan_sum, dtype=np.float64)
                total_sum[delay_fs] += scan_sum
                total_n[delay_fs] += int(n_good)
        finally:
            rf.close()

    for delay_fs in delays_chunk:
        delay_fs = int(delay_fs)
        s = total_sum.get(delay_fs, None)
        n = int(total_n.get(delay_fs, 0))

        if s is None or n == 0:
            results[delay_fs] = {"path": "", "n_images": 0}
            continue

        avg_img = (s / float(n)).astype(out_dtype, copy=False)

        out_name = (
            f"{sample_name}_{temperature_K}K_{wl_tag}nm_"
            f"{flu_file}mJ_{time_window_fs}fs_{delay_fs}fs.npy"
        )
        out_path = os.path.join(out_dir, out_name)

        if os.path.exists(out_path) and not overwrite:
            raise FileExistsError(f"File exists: {out_path} (set overwrite=True to replace).")

        np.save(out_path, avg_img)
        results[delay_fs] = {"path": out_path, "n_images": n}

    return results


def _export_dark_scan_chunk_worker(payload: dict) -> Dict[str, object]:
    """Worker for dark scan_type: compute partial sum over a chunk of scans."""
    scans_chunk = [int(s) for s in payload["scans_chunk"]]

    batch_size = int(payload["batch_size"])
    pilatus_h5_path = tuple(payload["pilatus_h5_path"])
    raw_root = str(payload["raw_root"])
    scan_file_pattern = str(payload.get("scan_file_pattern", DEFAULT_SCAN_FILE_PATTERN))

    rdcc_nbytes = int(payload.get("rdcc_nbytes", 0))
    rdcc_nslots = int(payload.get("rdcc_nslots", 0))

    nshots_map = payload.get("nshots_map", None)
    if nshots_map is None:
        metadata_h5_path = payload["metadata_h5_path"]
        nshots_map = {}
        with h5.File(metadata_h5_path, "r") as f:
            if "scans" not in f:
                raise ValueError("Invalid dark metadata H5: missing /scans")
            scans_root = f["scans"]
            for scan in scans_chunk:
                sname = str(int(scan))
                if sname in scans_root:
                    nshots_map[int(scan)] = int(scans_root[sname].attrs.get("nshots_expected", 0))

    total_sum: Optional[np.ndarray] = None
    total_n: int = 0

    for scan in scans_chunk:
        scan = int(scan)
        nshots = int(nshots_map.get(scan, 0))
        if nshots <= 0:
            continue

        raw_path = os.path.join(raw_root, scan_file_pattern.format(scan=scan))
        if not os.path.exists(raw_path):
            raise FileNotFoundError(raw_path)

        if rdcc_nbytes > 0 and rdcc_nslots > 0:
            rf = h5.File(raw_path, "r", rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots)
        else:
            rf = h5.File(raw_path, "r")

        try:
            dset = _h5_get_local(rf, pilatus_h5_path)
            idx = np.arange(nshots, dtype=np.int64)
            s, n = _sum_frames_by_indices_local(dset, idx, batch_size=batch_size)
        finally:
            rf.close()

        if s is None or n == 0:
            continue

        if total_sum is None:
            total_sum = np.zeros_like(s, dtype=np.float64)

        total_sum += s
        total_n += int(n)

    return {"sum": total_sum, "n": int(total_n)}


def _export_fluence_group_chunk_worker(payload: dict) -> Dict[str, Dict[str, Union[str, int, float]]]:
    """Worker for scan_type="fluence": for a SINGLE delay bin, export one image per fluence group,
    combining all scans that share that fluence.
    """
    metadata_h5_path = payload["metadata_h5_path"]
    delay_fs = int(payload["delay_fs"])
    flu_tags_chunk = [str(x) for x in payload["flu_tags_chunk"]]

    out_dir = payload["out_dir"]
    overwrite = bool(payload["overwrite"])
    batch_size = int(payload["batch_size"])
    out_dtype_str = str(payload["out_dtype_str"])

    pilatus_h5_path = tuple(payload["pilatus_h5_path"])
    raw_root = str(payload["raw_root"])
    scan_file_pattern = str(payload.get("scan_file_pattern", DEFAULT_SCAN_FILE_PATTERN))

    rdcc_nbytes = int(payload.get("rdcc_nbytes", 0))
    rdcc_nslots = int(payload.get("rdcc_nslots", 0))

    out_dtype = np.dtype(out_dtype_str)
    results: Dict[str, Dict[str, Union[str, int, float]]] = {}

    with h5.File(metadata_h5_path, "r") as f:
        if "meta" not in f or "delays" not in f:
            raise ValueError("Invalid fluence metadata H5: missing /meta or /delays")

        meta_g = f["meta"]
        delays_root = f["delays"]

        sample_name = str(_read_meta_value_local(meta_g, "sample_name"))
        temperature_K = int(_read_meta_value_local(meta_g, "temperature_K"))
        excitation_wl_nm = float(_read_meta_value_local(meta_g, "excitation_wl_nm"))
        time_window_fs = int(_read_meta_value_local(meta_g, "time_window_fs"))

        wl_tag = _wl_tag_nm_local(excitation_wl_nm)

        scans_arr = np.array(meta_g["scans"], dtype=np.int64) if "scans" in meta_g else np.array([], dtype=np.int64)
        flu_arr = np.array(meta_g["fluences_mJ_cm2"], dtype=float) if "fluences_mJ_cm2" in meta_g else np.array([], dtype=float)

        scan_to_fl: Dict[int, float] = {}
        scan_to_tag: Dict[int, str] = {}
        if scans_arr.size == flu_arr.size and scans_arr.size > 0:
            for s, fl in zip(scans_arr.tolist(), flu_arr.tolist()):
                s = int(s)
                fl = float(fl)
                scan_to_fl[s] = fl
                scan_to_tag[s] = general_utils.fluence_tag_file(fl)

        gname = f"{delay_fs}fs"
        if gname not in delays_root:
            raise KeyError(f"Missing group /delays/{gname}")
        scans_g = delays_root[gname]["scans"]

        tag_to_entries: Dict[str, List[Tuple[int, np.ndarray]]] = {t: [] for t in flu_tags_chunk}
        for scan_str in scans_g.keys():
            scan = int(scan_str)
            tag = scan_to_tag.get(scan, None)
            if tag is None or tag not in tag_to_entries:
                continue
            idx = np.array(scans_g[scan_str]["indices"], dtype=np.int64)
            if idx.size == 0:
                continue
            tag_to_entries[tag].append((scan, idx))

    for tag, entries in tag_to_entries.items():
        if not entries:
            results[tag] = {"path": "", "n_images": 0, "fluence_mJ_cm2": float("nan")}
            continue

        rep_scan = int(entries[0][0])
        rep_flu = float("nan")
        with h5.File(metadata_h5_path, "r") as f2:
            meta_g2 = f2["meta"]
            scans_arr2 = np.array(meta_g2["scans"], dtype=np.int64)
            flu_arr2 = np.array(meta_g2["fluences_mJ_cm2"], dtype=float)
            if scans_arr2.size == flu_arr2.size:
                m = {int(s): float(fl) for s, fl in zip(scans_arr2.tolist(), flu_arr2.tolist())}
                rep_flu = float(m.get(rep_scan, float("nan")))

        total_sum: Optional[np.ndarray] = None
        total_n = 0

        for scan, idx in entries:
            raw_path = os.path.join(raw_root, scan_file_pattern.format(scan=int(scan)))
            if not os.path.exists(raw_path):
                raise FileNotFoundError(raw_path)

            if rdcc_nbytes > 0 and rdcc_nslots > 0:
                rf = h5.File(raw_path, "r", rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots)
            else:
                rf = h5.File(raw_path, "r")

            try:
                dset = _h5_get_local(rf, pilatus_h5_path)
                ssum, n_good = _sum_frames_by_indices_local(dset, idx, batch_size=batch_size)
            finally:
                rf.close()

            if ssum is None or n_good == 0:
                continue

            if total_sum is None:
                total_sum = np.zeros_like(ssum, dtype=np.float64)

            total_sum += ssum
            total_n += int(n_good)

        if total_sum is None or total_n == 0:
            results[tag] = {"path": "", "n_images": 0, "fluence_mJ_cm2": float(rep_flu)}
            continue

        avg_img = (total_sum / float(total_n)).astype(out_dtype, copy=False)

        out_name = (
            f"{sample_name}_{temperature_K}K_{wl_tag}nm_"
            f"{tag}mJ_{time_window_fs}fs_{delay_fs}fs.npy"
        )
        out_path = os.path.join(out_dir, out_name)

        if os.path.exists(out_path) and not overwrite:
            raise FileExistsError(f"File exists: {out_path} (set overwrite=True to replace).")

        np.save(out_path, avg_img)
        results[tag] = {"path": out_path, "n_images": int(total_n), "fluence_mJ_cm2": float(rep_flu)}

    return results


# ----------------------------
# Main class
# ----------------------------
class Experiment:
    """Supports:
    - scan_type="delay"   -> /meta + /delays/<delay>/scans/<scan>/indices
    - scan_type="dark"    -> /meta + /scans/<scan>/...  (all shots)
    - scan_type="fluence" -> /meta + /delays/<delay>/scans/<scan>/indices (export one per fluence group)
    """
    PILATUS_H5_PATH = ("entry", "measurement", "pilatus", "data")

    def __init__(
        self,
        scans: Union[int, Sequence[int]],
        *,
        ref_provider: Optional[Callable[[int], Tuple[float, float]]] = None,
        ping_reference_path: Optional[Union[str, Path]] = None,
        paths: Optional[AnalysisPaths] = None,
        path_root: Optional[Union[str, Path]] = None,
        raw_subdir: Optional[Union[str, Path]] = None,
        analysis_subdir: Optional[Union[str, Path]] = None,
        scan_file_pattern: str = DEFAULT_SCAN_FILE_PATTERN,
        ping2_h5_path: Tuple[str, ...] = PING2_H5_PATH,
        ping4_h5_path: Tuple[str, ...] = PING4_H5_PATH,
    ):
        """Bind experiment metadata, raw scans, timing references, and output paths."""
        if isinstance(scans, int):
            self.scans = [int(scans)]
        else:
            self.scans = [int(s) for s in list(scans)]

        self.paths = _coerce_paths(
            paths=paths,
            path_root=path_root,
            raw_subdir=raw_subdir,
            analysis_subdir=analysis_subdir,
        )
        self.raw_root = Path(self.paths.raw_root)
        self.analysis_root = Path(self.paths.analysis_root)

        if ref_provider is not None and ping_reference_path is not None:
            raise ValueError(
                "Provide either ref_provider or ping_reference_path, not both."
            )

        self.ping_reference_table: Optional[PingReferenceTable] = None
        if ref_provider is None:
            self.ping_reference_table = load_ping_reference_table(
                ping_reference_path
            )
            self.ref_provider = self.ping_reference_table.reference_for
            self.ping_reference_path: Optional[Path] = self.ping_reference_table.path
        else:
            self.ref_provider = ref_provider
            self.ping_reference_path = None
        self.scan_file_pattern = str(scan_file_pattern)
        self.ping2_h5_path = ping2_h5_path
        self.ping4_h5_path = ping4_h5_path

    # ----------------------------
    # Paths
    # ----------------------------
    def _scan_file(self, scan: int) -> str:
        """Return scan file."""
        return str(self.raw_root / self.scan_file_pattern.format(scan=int(scan)))

    def validate_ping_references(
        self,
        scans: Optional[Sequence[int]] = None,
    ) -> None:
        """Validate ping references.

        Parameters
        ----------
        scans : Optional[Sequence[int]]
            Facility scan identifiers included in the operation.

        Raises
        ------
        KeyError
            If a required mapping or DataFrame column is absent.
        """
        scans_list = self.scans if scans is None else [int(scan) for scan in scans]
        if self.ping_reference_table is not None:
            self.ping_reference_table.validate_scans(scans_list)
            return

        missing: List[int] = []
        for scan in scans_list:
            try:
                self.ref_provider(int(scan))
            except (KeyError, ValueError):
                missing.append(int(scan))
        if missing:
            raise KeyError(
                f"The custom ping-reference provider does not cover scan(s): {missing}"
            )

    def _write_ping_reference_metadata(self, meta_group) -> None:
        """Write ping reference metadata."""
        self.validate_ping_references()
        scans = np.asarray(self.scans, dtype=np.int64)
        refs = np.asarray(
            [self.ref_provider(int(scan)) for scan in self.scans],
            dtype=float,
        )

        if self.ping_reference_table is None:
            source = "custom ref_provider callable"
            digest = ""
        else:
            source = str(self.ping_reference_table.path)
            digest = self.ping_reference_table.sha256

        meta_group.attrs["ping_reference_source"] = source
        meta_group.attrs["ping_reference_sha256"] = digest
        meta_group.attrs["ping_reference_units"] = "seconds"
        self._write_scalar_dataset(meta_group, "ping_reference_source", source)
        self._write_scalar_dataset(meta_group, "ping_reference_sha256", digest)
        self._write_scalar_dataset(meta_group, "ping_reference_units", "seconds")
        meta_group.create_dataset("ping_reference_scans", data=scans)
        meta_group.create_dataset("ping2_reference_s", data=refs[:, 0])
        meta_group.create_dataset("ping4_reference_s", data=refs[:, 1])

    def validate_metadata_ping_references(self, metadata_path: Union[str, Path]) -> None:
        """Reject reuse of cached metadata made with a different reference file.

        Parameters
        ----------
        metadata_path : Union[str, Path]
            Filesystem path for metadata.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        if self.ping_reference_table is None:
            return
        with h5.File(str(metadata_path), "r") as handle:
            meta_group = handle["meta"]
            stored = general_utils.decode_if_bytes(
                meta_group.attrs.get("ping_reference_sha256", "")
            )
        if not stored:
            raise ValueError(
                f"Metadata file {metadata_path} does not record its ping-reference "
                "configuration. Recreate it with overwrite=True."
            )
        if str(stored) != self.ping_reference_table.sha256:
            raise ValueError(
                f"Metadata file {metadata_path} was created with a different "
                "ping-reference table. Recreate it with overwrite=True."
            )

    @classmethod
    def analysis_dir(
        cls,
        meta: "ExperimentMeta",
        scans: Optional[Sequence[int]] = None,
        *,
        paths: Optional[AnalysisPaths] = None,
        path_root: Optional[Union[str, Path]] = None,
        analysis_subdir: Optional[Union[str, Path]] = None,
        raw_subdir: Optional[Union[str, Path]] = None,
    ) -> str:
        """delay:
          .../<ANALYSIS_SUBDIR>/sample/temperature_XXXK/excitation_wl_YYYnm/delay/fluence_.../time_window_...fs/
        fluence:
          .../<ANALYSIS_SUBDIR>/sample/temperature_XXXK/excitation_wl_YYYnm/fluence/delay_<delay>fs/time_window_...fs/
        dark:
          .../<ANALYSIS_SUBDIR>/sample/temperature_XXXK/dark/<scan_tag>/
        """
        analysis_root = _analysis_root(
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
            raw_subdir=raw_subdir,
        )

        st = str(meta.scan_type).strip().lower()

        if st == "dark":
            scan_tag = general_utils.scan_tag(scans if scans is not None else [])
            return str(
                analysis_root
                / meta.sample_name
                / f"temperature_{meta.temperature_K}K"
                / "dark"
                / scan_tag
            )

        if meta.excitation_wl_nm is None:
            raise ValueError("excitation_wl_nm must be provided for scan_type != 'dark'.")
        if meta.time_window_fs is None or int(meta.time_window_fs) <= 0:
            raise ValueError("time_window_fs must be provided and positive for scan_type != 'dark'.")

        wl_tag = _wl_tag_nm_local(meta.excitation_wl_nm)

        if st == "fluence":
            if meta.delay is None:
                raise ValueError("For scan_type='fluence', meta.delay must be set (e.g. -1000).")
            delay_fs = int(meta.delay)
            return str(
                analysis_root
                / meta.sample_name
                / f"temperature_{meta.temperature_K}K"
                / f"excitation_wl_{wl_tag}nm"
                / "fluence"
                / f"delay_{delay_fs}fs"
                / f"time_window_{int(meta.time_window_fs)}fs"
            )

        if meta.fluence_mJ_cm2 is None or isinstance(meta.fluence_mJ_cm2, (list, tuple, np.ndarray)):
            raise ValueError("For scan_type='delay', fluence_mJ_cm2 must be a scalar (mJ/cm^2).")

        flu_folder = general_utils.fluence_tag_folder(float(meta.fluence_mJ_cm2))
        return str(
            analysis_root
            / meta.sample_name
            / f"temperature_{meta.temperature_K}K"
            / f"excitation_wl_{wl_tag}nm"
            / "delay"
            / f"fluence_{flu_folder}"
            / f"time_window_{int(meta.time_window_fs)}fs"
        )

    @classmethod
    def metadata_h5_path(
        cls,
        meta: "ExperimentMeta",
        scans: Optional[Sequence[int]] = None,
        *,
        paths: Optional[AnalysisPaths] = None,
        path_root: Optional[Union[str, Path]] = None,
        analysis_subdir: Optional[Union[str, Path]] = None,
        raw_subdir: Optional[Union[str, Path]] = None,
    ) -> str:
        """Return metadata HDF5 path.

        Parameters
        ----------
        meta : 'ExperimentMeta'
            Experiment metadata mapping associated with the data series.
        scans : Optional[Sequence[int]]
            Facility scan identifiers included in the operation.
        paths : Optional[AnalysisPaths]
            Resolved ``AnalysisPaths`` configuration. It takes precedence over legacy path arguments.
        path_root : Optional[Union[str, Path]]
            Root directory containing raw and analysis data trees.
        analysis_subdir : Optional[Union[str, Path]]
            Analysis-directory path relative to ``path_root``.
        raw_subdir : Optional[Union[str, Path]]
            Raw-data path relative to ``path_root``.

        Returns
        -------
        str
            Resolved path, label, or filename derived from experiment metadata.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        base = cls.analysis_dir(
            meta,
            scans=scans,
            paths=paths,
            path_root=path_root,
            analysis_subdir=analysis_subdir,
            raw_subdir=raw_subdir,
        )
        st = str(meta.scan_type).strip().lower()

        if st == "dark":
            scan_tag_file = general_utils.scan_tag_file(scans if scans is not None else [])
            name = f"{meta.sample_name}_{meta.temperature_K}K_dark_{scan_tag_file}.h5"
            return os.path.join(base, name)

        if meta.excitation_wl_nm is None:
            raise ValueError("excitation_wl_nm must be provided for scan_type != 'dark'.")
        if meta.time_window_fs is None:
            raise ValueError("time_window_fs must be provided for scan_type != 'dark'.")

        wl_tag = _wl_tag_nm_local(meta.excitation_wl_nm)

        if st == "fluence":
            if meta.delay is None:
                raise ValueError("For scan_type='fluence', meta.delay must be set.")
            delay_fs = int(meta.delay)
            name = f"{meta.sample_name}_{meta.temperature_K}K_{wl_tag}nm_{delay_fs}fs_{int(meta.time_window_fs)}fs.h5"
            return os.path.join(base, name)

        flu_folder = general_utils.fluence_tag_folder(float(meta.fluence_mJ_cm2))  # type: ignore[arg-type]
        name = (
            f"{meta.sample_name}_{meta.temperature_K}K_{wl_tag}nm_"
            f"{flu_folder}_{int(meta.time_window_fs)}fs.h5"
        )
        return os.path.join(base, name)

    # ----------------------------
    # Ping utilities
    # ----------------------------
    def read_corrected_pings_seconds(self, scan: int) -> Tuple[np.ndarray, np.ndarray]:
        """Read corrected pings seconds.

        Parameters
        ----------
        scan : int
            Facility scan identifier or scan collection accepted by the selected backend.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Corrected ping-2 and ping-4 arrays in seconds.

        Raises
        ------
        FileNotFoundError
            If required raw data, calibration, metadata, or cached analysis files are missing.
        """
        fp = self._scan_file(scan)
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)

        with h5.File(fp, "r") as f:
            p2 = np.array(_h5_get_local(f, self.ping2_h5_path))
            p4 = np.array(_h5_get_local(f, self.ping4_h5_path))

        n = min(p2.size, p4.size)
        p2 = p2[:n]
        p4 = p4[:n]

        r2, r4 = self.ref_provider(scan)
        return p2 - r2, p4 - r4

    @staticmethod
    def valid_mask(p2_s: np.ndarray, p4_s: np.ndarray) -> np.ndarray:
        """Return the valid mask.

        Parameters
        ----------
        p2_s : np.ndarray
            Corrected ping-2 timing values in seconds.
        p4_s : np.ndarray
            Corrected ping-4 timing values in seconds.

        Returns
        -------
        np.ndarray
            Boolean mask selecting shots with finite, nonzero timing signals.
        """
        return np.isfinite(p2_s) & np.isfinite(p4_s)

    @staticmethod
    def to_fs_int(x_seconds: np.ndarray) -> np.ndarray:
        """Convert the current value to fs int."""
        return np.rint(x_seconds * 1e15).astype(np.int64)

    @staticmethod
    def _delay_series_seconds(p2_s: np.ndarray, p4_s: np.ndarray, delay_source: str) -> np.ndarray:
        """Return delay series seconds."""
        if delay_source == "p2":
            return p2_s
        if delay_source == "p4":
            return p4_s
        if delay_source == "avg":
            return 0.5 * (p2_s + p4_s)
        raise ValueError("delay_source must be 'avg', 'p2', or 'p4'")

    @staticmethod
    def _valid_mask_for_source(p2_s: np.ndarray, p4_s: np.ndarray, delay_source: str, require_both: bool) -> np.ndarray:
        """Return the valid mask for source."""
        if require_both:
            return Experiment.valid_mask(p2_s, p4_s)
        if delay_source == "p2":
            return np.isfinite(p2_s)
        if delay_source == "p4":
            return np.isfinite(p4_s)
        return Experiment.valid_mask(p2_s, p4_s)

    def get_delays(
        self,
        scan: int,
        *,
        delay_source: str = "avg",
        unit: str = "ps",
        require_both: bool = True,
    ) -> np.ndarray:
        """Return delays.

        Parameters
        ----------
        scan : int
            Facility scan identifier or scan collection accepted by the selected backend.
        delay_source : str
            Column or metadata source from which delay values are read.
        unit : str
            Display unit used for the independent variable.
        require_both : bool
            Whether both members of each symmetric azimuthal pair are required.

        Returns
        -------
        np.ndarray
            Corrected shot delays in seconds.
        """
        p2_s, p4_s = self.read_corrected_pings_seconds(scan)
        valid = self._valid_mask_for_source(p2_s, p4_s, delay_source, require_both)
        d_s = self._delay_series_seconds(p2_s, p4_s, delay_source)[valid]

        unit = general_utils.normalize_time_unit(unit)
        return general_utils.convert_time_values(
            d_s,
            from_unit="s",
            to_unit=unit,
        )

    # ----------------------------
    # Plotting (delegate to plot_utils)
    # ----------------------------
    def plot_delay_distribution(
        self,
        scans: Optional[Union[int, Sequence[int]]] = None,
        *,
        mode: str = "overlay",
        delay_source: str = "avg",
        require_both: bool = True,
        unit: str = "ps",
        view: str = "scatter",
        bins: int = 200,
        hist_range: Optional[Tuple[float, float]] = None,
        density: bool = False,
        show_median: bool = True,
        alpha: float = 0.6,
        ms: float = 3.0,
        title: Optional[str] = None,
    ) -> List[Tuple[object, object]]:
        """Plot delay distribution.

        Parameters
        ----------
        scans : Optional[Union[int, Sequence[int]]]
            Facility scan identifiers included in the operation.
        mode : str
            Operation mode controlling how the input data are grouped or displayed.
        delay_source : str
            Column or metadata source from which delay values are read.
        require_both : bool
            Whether both members of each symmetric azimuthal pair are required.
        unit : str
            Display unit used for the independent variable.
        view : str
            Display representation, such as histogram or delay trace.
        bins : int
            Number of histogram bins.
        hist_range : Optional[Tuple[float, float]]
            Optional lower and upper limits of the histogram domain.
        density : bool
            Whether histogram counts are normalized to probability density.
        show_median : bool
            Whether to display median.
        alpha : float
            Matplotlib opacity in the interval ``[0, 1]``.
        ms : float
            Marker size used for the plotted points.
        title : Optional[str]
            Optional plot title; a metadata-derived title is used when omitted.

        Returns
        -------
        List[Tuple[object, object]]
            List of Matplotlib ``(figure, axes)`` pairs created by the selected view mode.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        if scans is None:
            scans_list = self.scans
        elif isinstance(scans, int):
            scans_list = [int(scans)]
        else:
            scans_list = [int(s) for s in list(scans)]

        if mode not in ("overlay", "stacked", "per_scan"):
            raise ValueError("mode must be 'overlay', 'stacked', or 'per_scan'")
        if view not in ("scatter", "hist"):
            raise ValueError("view must be 'scatter' or 'hist'")
        unit = general_utils.normalize_time_unit(unit)
        if int(bins) < 1 or int(bins) > 100_000:
            raise ValueError("bins must be between 1 and 100000")

        self.validate_ping_references(scans_list)

        the_title = title if title is not None else f"Delay distribution ({delay_source}) - {view}"

        delays_by_scan: Dict[int, np.ndarray] = {}
        for scan in tqdm(scans_list, desc="Collecting delays", unit="scan"):
            d = self.get_delays(scan, delay_source=delay_source, require_both=require_both, unit=unit)
            if d.size == 0:
                continue
            delays_by_scan[int(scan)] = d

        if not delays_by_scan:
            return []

        plotter = plot_utils.DelayDistributionPlotter()
        return plotter.plot(
            delays_by_scan,
            mode=mode,
            view=view,
            unit=unit,
            bins=bins,
            hist_range=hist_range,
            density=bool(density),
            show_median=show_median,
            alpha=alpha,
            ms=ms,
            title=the_title,
        )

    # ----------------------------
    # Metadata building helpers
    # ----------------------------
    @staticmethod
    def _cluster_centers_fs(centers_fs: np.ndarray, tol_fs: int) -> List[int]:
        """Calculate cluster centers fs."""
        if centers_fs.size == 0:
            return []
        c = np.sort(centers_fs.astype(np.int64))
        clusters = [[int(c[0])]]
        for x in c[1:]:
            cur_mean = int(np.rint(np.mean(clusters[-1])))
            if abs(int(x) - cur_mean) <= tol_fs:
                clusters[-1].append(int(x))
            else:
                clusters.append([int(x)])
        return [int(np.rint(np.mean(g))) for g in clusters]

    @staticmethod
    def _append_1d(ds, arr: np.ndarray) -> None:
        """Append a one-dimensional array to an extendable HDF5 dataset."""
        arr = np.asarray(arr)
        if arr.size == 0:
            return
        old = ds.shape[0]
        ds.resize((old + arr.shape[0],))
        ds[old:] = arr

    @staticmethod
    def _write_scalar_dataset(group: h5.Group, name: str, value):
        """Write scalar dataset."""
        if isinstance(value, str):
            dt = h5.string_dtype(encoding="utf-8")
            group.create_dataset(name, data=np.array(value, dtype=dt))
        else:
            group.create_dataset(name, data=np.array(value))

    # ----------------------------
    # Dark metadata builder
    # ----------------------------
    def _build_metadata_h5_dark(self, *, meta: "ExperimentMeta", overwrite: bool = False) -> str:
        """Write dark-scan shot selections and provenance to the metadata HDF5 file."""
        out_path = self.metadata_h5_path(meta, scans=self.scans, paths=self.paths)
        h5_dir = os.path.dirname(out_path)
        os.makedirs(h5_dir, exist_ok=True)

        if os.path.exists(out_path):
            if overwrite:
                os.remove(out_path)
            else:
                raise FileExistsError(f"File exists: {out_path} (set overwrite=True to replace).")

        created = datetime.now(timezone.utc).isoformat()

        with h5.File(out_path, "w") as f:
            meta_g = f.create_group("meta")
            scans_root = f.create_group("scans")

            meta_g.attrs["schema_version"] = "1"
            meta_g.attrs["created_utc"] = created
            meta_g.attrs["sample_name"] = meta.sample_name
            meta_g.attrs["temperature_K"] = int(meta.temperature_K)
            meta_g.attrs["scan_type"] = "dark"

            self._write_scalar_dataset(meta_g, "schema_version", "1")
            self._write_scalar_dataset(meta_g, "created_utc", created)
            self._write_scalar_dataset(meta_g, "sample_name", meta.sample_name)
            self._write_scalar_dataset(meta_g, "temperature_K", int(meta.temperature_K))
            self._write_scalar_dataset(meta_g, "scan_type", "dark")

            meta_g.create_dataset("scans", data=np.array(self.scans, dtype=np.int64))

            for scan in tqdm(self.scans, desc="Dark: counting shots", unit="scan"):
                raw_path = self._scan_file(int(scan))
                if not os.path.exists(raw_path):
                    raise FileNotFoundError(raw_path)

                with h5.File(raw_path, "r") as rf:
                    dset = _h5_get_local(rf, self.PILATUS_H5_PATH)
                    n_frames = int(dset.shape[0])

                sg = scans_root.create_group(str(int(scan)))
                sg.attrs["nshots_expected"] = int(n_frames)

        return out_path

    # ----------------------------
    # Delay/Fluence metadata builder (and router)
    # ----------------------------
    def build_metadata_h5(
        self,
        *,
        meta: "ExperimentMeta",
        selected_delays: Union[str, Sequence[int]] = "auto",
        delay_source: str = "avg",
        require_both: bool = True,
        nb_shot_threshold: Optional[int] = None,
        overwrite: bool = False,
        cluster_tol_fs: Optional[int] = None,
    ) -> str:
        """Build metadata HDF5.

        Parameters
        ----------
        meta : 'ExperimentMeta'
            Experiment metadata mapping associated with the data series.
        selected_delays : Union[str, Sequence[int]]
            Delay-bin centers selected for export, in femtoseconds.
        delay_source : str
            Column or metadata source from which delay values are read.
        require_both : bool
            Whether both members of each symmetric azimuthal pair are required.
        nb_shot_threshold : Optional[int]
            Minimum number of valid shots required for an exported delay bin.
        overwrite : bool
            Whether existing output artifacts may be replaced.
        cluster_tol_fs : Optional[int]
            Maximum separation in femtoseconds for grouping nearby delay samples.

        Returns
        -------
        str
            Path of the metadata HDF5 file written for the experiment.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        FileExistsError
            If the operation encounters this explicit failure condition.

        Notes
        -----
        This operation may create or replace analysis artifacts according to its save and overwrite settings.
        """
        st = str(meta.scan_type).strip().lower()

        if st == "dark":
            return self._build_metadata_h5_dark(meta=meta, overwrite=overwrite)

        if st not in ("delay", "fluence"):
            raise ValueError("scan_type must be 'delay', 'fluence', or 'dark'.")

        self.validate_ping_references()

        if meta.excitation_wl_nm is None:
            raise ValueError("excitation_wl_nm must be provided for scan_type != 'dark'.")
        if meta.time_window_fs is None or int(meta.time_window_fs) <= 0:
            raise ValueError("meta.time_window_fs must be a positive integer (fs).")

        if delay_source not in ("avg", "p2", "p4"):
            raise ValueError("delay_source must be 'avg', 'p2', or 'p4'")
        if nb_shot_threshold is not None:
            if not isinstance(nb_shot_threshold, int) or nb_shot_threshold <= 0:
                raise ValueError("nb_shot_threshold must be a positive integer or None.")

        if st == "delay":
            if meta.fluence_mJ_cm2 is None or isinstance(meta.fluence_mJ_cm2, (list, tuple, np.ndarray)):
                raise ValueError("For scan_type='delay', fluence_mJ_cm2 must be a scalar (mJ/cm^2).")

        fluences_list: Optional[List[float]] = None
        if st == "fluence":
            if meta.fluence_mJ_cm2 is None:
                raise ValueError("For scan_type='fluence', fluence_mJ_cm2 must be a sequence aligned with scans.")
            if not isinstance(meta.fluence_mJ_cm2, (list, tuple, np.ndarray)):
                raise ValueError("For scan_type='fluence', fluence_mJ_cm2 must be a sequence aligned with scans.")
            fluences_list = [float(x) for x in list(meta.fluence_mJ_cm2)]
            if len(fluences_list) != len(self.scans):
                raise ValueError("For scan_type='fluence', fluence_mJ_cm2 must have the same length as scans.")
            if meta.delay is None:
                raise ValueError("For scan_type='fluence', meta.delay must be set (e.g. -1000).")

        halfwin = int(meta.time_window_fs) / 2.0

        out_path = self.metadata_h5_path(meta, scans=self.scans, paths=self.paths)
        h5_dir = os.path.dirname(out_path)
        os.makedirs(h5_dir, exist_ok=True)

        if os.path.exists(out_path):
            if overwrite:
                os.remove(out_path)
            else:
                raise FileExistsError(f"File exists: {out_path} (set overwrite=True to replace).")

        mode = "manual"
        if isinstance(selected_delays, str) and selected_delays.lower() == "auto":
            mode = "auto"
            centers: List[int] = []
            for scan in tqdm(self.scans, desc="Auto: scanning medians", unit="scan"):
                p2_s, p4_s = self.read_corrected_pings_seconds(scan)
                valid = self._valid_mask_for_source(p2_s, p4_s, delay_source, require_both)
                if not np.any(valid):
                    continue
                d_s = self._delay_series_seconds(p2_s, p4_s, delay_source)
                d_fs = self.to_fs_int(d_s)
                centers.append(int(np.median(d_fs[valid])))

            centers_arr = np.array(centers, dtype=np.int64)
            if cluster_tol_fs is None:
                cluster_tol_fs = max(1, int(meta.time_window_fs) // 5)
            delays_fs = self._cluster_centers_fs(centers_arr, tol_fs=int(cluster_tol_fs))
        else:
            delays_fs = [int(x) for x in list(selected_delays)]

        delays_fs = sorted(set(delays_fs))
        if len(delays_fs) == 0:
            raise ValueError("No delays selected/identified.")

        if st == "fluence" and len(delays_fs) != 1:
            raise ValueError(
                "For scan_type='fluence', provide exactly one selected delay per call "
                "(e.g. selected_delays=[-1000]). If you want multiple delays, call per delay."
            )

        counts: Dict[int, int] = {d: 0 for d in delays_fs}
        for scan in tqdm(self.scans, desc="Counting shots", unit="scan"):
            p2_s, p4_s = self.read_corrected_pings_seconds(scan)
            valid = self._valid_mask_for_source(p2_s, p4_s, delay_source, require_both)
            if not np.any(valid):
                continue
            d_s = self._delay_series_seconds(p2_s, p4_s, delay_source)
            d_fs = self.to_fs_int(d_s)
            for d0 in delays_fs:
                counts[d0] += int(np.count_nonzero(valid & (np.abs(d_fs - d0) <= halfwin)))

        kept_delays = delays_fs
        if nb_shot_threshold is not None:
            kept_delays = [d for d in delays_fs if counts.get(d, 0) >= nb_shot_threshold]
        if len(kept_delays) == 0:
            raise ValueError("All delay points removed by nb_shot_threshold. Lower it or verify data.")

        if st == "fluence" and len(kept_delays) != 1:
            raise ValueError(
                "For scan_type='fluence', after thresholding there must be exactly one kept delay. "
                "Provide an explicit selected_delays=[...] that matches the data."
            )

        with h5.File(out_path, "w") as f:
            meta_g = f.create_group("meta")
            delays_g = f.create_group("delays")

            created = datetime.now(timezone.utc).isoformat()

            meta_g.attrs["schema_version"] = "1"
            meta_g.attrs["created_utc"] = created
            meta_g.attrs["sample_name"] = meta.sample_name
            meta_g.attrs["temperature_K"] = int(meta.temperature_K)
            meta_g.attrs["excitation_wl_nm"] = float(meta.excitation_wl_nm)
            meta_g.attrs["scan_type"] = st
            meta_g.attrs["time_window_fs"] = int(meta.time_window_fs)
            meta_g.attrs["selected_delays_mode"] = mode
            meta_g.attrs["delay_source"] = str(delay_source)
            meta_g.attrs["require_both_pings"] = bool(require_both)

            if st == "delay":
                meta_g.attrs["fluence_mJ_cm2"] = float(meta.fluence_mJ_cm2)  # type: ignore[arg-type]
            if st == "fluence":
                meta_g.attrs["delay_fs"] = int(kept_delays[0])
                meta_g.attrs["delay"] = str(meta.delay) if meta.delay is not None else ""

            if nb_shot_threshold is not None:
                meta_g.attrs["nb_shot_threshold"] = int(nb_shot_threshold)
            if mode == "auto":
                meta_g.attrs["auto_cluster_tol_fs"] = int(cluster_tol_fs)

            self._write_scalar_dataset(meta_g, "schema_version", "1")
            self._write_scalar_dataset(meta_g, "created_utc", created)
            self._write_scalar_dataset(meta_g, "sample_name", meta.sample_name)
            self._write_scalar_dataset(meta_g, "temperature_K", int(meta.temperature_K))
            self._write_scalar_dataset(meta_g, "excitation_wl_nm", float(meta.excitation_wl_nm))
            self._write_scalar_dataset(meta_g, "scan_type", st)
            self._write_scalar_dataset(meta_g, "time_window_fs", int(meta.time_window_fs))
            self._write_scalar_dataset(meta_g, "selected_delays_mode", mode)
            self._write_scalar_dataset(meta_g, "delay_source", str(delay_source))
            self._write_scalar_dataset(meta_g, "require_both_pings", int(bool(require_both)))

            if st == "delay":
                self._write_scalar_dataset(meta_g, "fluence_mJ_cm2", float(meta.fluence_mJ_cm2))  # type: ignore[arg-type]
            if st == "fluence":
                self._write_scalar_dataset(meta_g, "delay_fs", int(kept_delays[0]))
                if meta.delay is not None:
                    self._write_scalar_dataset(meta_g, "delay", str(meta.delay))

            if nb_shot_threshold is not None:
                self._write_scalar_dataset(meta_g, "nb_shot_threshold", int(nb_shot_threshold))
            if mode == "auto":
                self._write_scalar_dataset(meta_g, "auto_cluster_tol_fs", int(cluster_tol_fs))

            self._write_ping_reference_metadata(meta_g)

            meta_g.create_dataset("scans", data=np.array(self.scans, dtype=np.int64))
            meta_g.create_dataset("selected_delays_fs", data=np.array(kept_delays, dtype=np.int64))

            if st == "fluence" and fluences_list is not None:
                meta_g.create_dataset("fluences_mJ_cm2", data=np.array(fluences_list, dtype=float))

            for d0 in kept_delays:
                dg = delays_g.create_group(f"{int(d0)}fs")
                dg.attrs["delay_fs"] = int(d0)
                dg.attrs["time_window_fs"] = int(meta.time_window_fs)
                dg.attrs["nshots_total_expected"] = int(counts.get(d0, 0))

                dg.create_group("scans")

                dg.create_dataset(
                    "delays_pings2_fs",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.int64,
                    chunks=True,
                    compression="gzip",
                    shuffle=True,
                )
                dg.create_dataset(
                    "delays_pings4_fs",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.int64,
                    chunks=True,
                    compression="gzip",
                    shuffle=True,
                )

            scan_to_flu: Dict[int, float] = {}
            if st == "fluence" and fluences_list is not None:
                for s, fl in zip(self.scans, fluences_list):
                    scan_to_flu[int(s)] = float(fl)

            for scan in tqdm(self.scans, desc="Writing indices", unit="scan"):
                p2_s, p4_s = self.read_corrected_pings_seconds(scan)
                valid = self._valid_mask_for_source(p2_s, p4_s, delay_source, require_both)
                if not np.any(valid):
                    continue

                d2_fs = self.to_fs_int(p2_s)
                d4_fs = self.to_fs_int(p4_s)

                d_s = self._delay_series_seconds(p2_s, p4_s, delay_source)
                d_fs = self.to_fs_int(d_s)

                for d0 in kept_delays:
                    idx = np.nonzero(valid & (np.abs(d_fs - d0) <= halfwin))[0].astype(np.int64)
                    if idx.size == 0:
                        continue

                    dg = delays_g[f"{int(d0)}fs"]
                    sg = dg["scans"].create_group(str(int(scan)))
                    sg.create_dataset("indices", data=idx, compression="gzip", shuffle=True)
                    sg.attrs["nshots"] = int(idx.size)

                    if st == "fluence":
                        sg.attrs["fluence_mJ_cm2"] = float(scan_to_flu.get(int(scan), float("nan")))

                    self._append_1d(dg["delays_pings2_fs"], d2_fs[idx])
                    self._append_1d(dg["delays_pings4_fs"], d4_fs[idx])

        return out_path

    # ----------------------------
    # Serial exporter (delay + dark + fluence)
    # ----------------------------
    def export_delay_2d_images(
        self,
        *,
        meta: Optional["ExperimentMeta"] = None,
        metadata_h5_path: Optional[str] = None,
        pilatus_h5_path: Optional[Tuple[str, ...]] = None,
        out_folder_name: str = "2D_images",
        batch_size: int = 256,
        out_dtype=np.float32,
        overwrite: bool = False,
    ) -> Dict[Any, Dict[str, Union[str, int, float]]]:
        """Export delay 2D images.

        Parameters
        ----------
        meta : Optional['ExperimentMeta']
            Experiment metadata mapping associated with the data series.
        metadata_h5_path : Optional[str]
            Filesystem path for metadata HDF5.
        pilatus_h5_path : Optional[Tuple[str, ...]]
            Filesystem path for pilatus HDF5.
        out_folder_name : str
            Name of the output folder below the experiment analysis directory.
        batch_size : int
            Number of detector frames read and averaged per batch.
        out_dtype : object
            NumPy data type used for exported averaged images.
        overwrite : bool
            Whether existing output artifacts may be replaced.

        Returns
        -------
        Dict[Any, Dict[str, Union[str, int, float]]]
            Per-delay export report containing output paths, shot counts, and status metadata.

        Raises
        ------
        FileNotFoundError
            If required raw data, calibration, metadata, or cached analysis files are missing.
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        FileExistsError
            If the operation encounters this explicit failure condition.
        KeyError
            If a required mapping or DataFrame column is absent.

        Notes
        -----
        This operation may create or replace analysis artifacts according to its save and overwrite settings.
        """
        if metadata_h5_path is None:
            if meta is None:
                raise ValueError("Provide either meta=... or metadata_h5_path=...")
            metadata_h5_path = self.metadata_h5_path(meta, scans=self.scans, paths=self.paths)

        if pilatus_h5_path is None:
            pilatus_h5_path = self.PILATUS_H5_PATH

        if not os.path.exists(metadata_h5_path):
            raise FileNotFoundError(metadata_h5_path)

        h5_dir = os.path.dirname(os.path.abspath(metadata_h5_path))
        out_dir = os.path.join(h5_dir, out_folder_name)
        os.makedirs(out_dir, exist_ok=True)

        results: Dict[Any, Dict[str, Union[str, int, float]]] = {}

        with h5.File(metadata_h5_path, "r") as f:
            if "meta" not in f:
                raise ValueError("Invalid metadata H5: missing /meta")

            meta_g = f["meta"]
            scan_type = str(_read_meta_value_local(meta_g, "scan_type")).strip().lower()

            if scan_type == "dark":
                if "scans" not in f:
                    raise ValueError("Invalid dark metadata H5: missing /scans")
                scans_root = f["scans"]

                sample_name = str(_read_meta_value_local(meta_g, "sample_name"))
                temperature_K = int(_read_meta_value_local(meta_g, "temperature_K"))

                scan_list = [int(s) for s in list(scans_root.keys())]
                scan_tag_file = general_utils.scan_tag_file(scan_list)

                total_sum: Optional[np.ndarray] = None
                total_n: int = 0

                for scan_str in tqdm(list(scans_root.keys()), desc="2D dark images (combine)", unit="scan"):
                    scan = int(scan_str)
                    nshots = int(scans_root[scan_str].attrs.get("nshots_expected", 0))
                    if nshots <= 0:
                        continue

                    raw_path = self._scan_file(scan)
                    if not os.path.exists(raw_path):
                        raise FileNotFoundError(raw_path)

                    with h5.File(raw_path, "r") as rf:
                        dset = _h5_get_local(rf, pilatus_h5_path)
                        idx = np.arange(nshots, dtype=np.int64)
                        s, n = _sum_frames_by_indices_local(dset, idx, batch_size=batch_size)

                    if s is None or n == 0:
                        continue

                    if total_sum is None:
                        total_sum = np.zeros_like(s, dtype=np.float64)

                    total_sum += s
                    total_n += int(n)

                if total_sum is None or total_n == 0:
                    results[-1] = {"path": "", "n_images": 0}
                    return results

                avg_img = (total_sum / float(total_n)).astype(out_dtype, copy=False)

                out_name = f"{sample_name}_{temperature_K}K_dark_{scan_tag_file}.npy"
                out_path = os.path.join(out_dir, out_name)

                if os.path.exists(out_path) and not overwrite:
                    raise FileExistsError(f"File exists: {out_path} (set overwrite=True to replace).")

                np.save(out_path, avg_img)
                results[-1] = {"path": out_path, "n_images": int(total_n)}
                return results

            if "delays" not in f:
                raise ValueError("Invalid metadata H5: missing /delays")

            delays_root = f["delays"]
            selected_delays_fs = np.array(meta_g["selected_delays_fs"], dtype=np.int64)

            sample_name = str(_read_meta_value_local(meta_g, "sample_name"))
            temperature_K = int(_read_meta_value_local(meta_g, "temperature_K"))
            excitation_wl_nm = float(_read_meta_value_local(meta_g, "excitation_wl_nm"))
            time_window_fs = int(_read_meta_value_local(meta_g, "time_window_fs"))

            wl_tag = _wl_tag_nm_local(excitation_wl_nm)

            scan_to_tag: Dict[int, str] = {}
            scan_to_flu: Dict[int, float] = {}
            if scan_type == "fluence" and "fluences_mJ_cm2" in meta_g and "scans" in meta_g:
                scans_arr = np.array(meta_g["scans"], dtype=np.int64)
                flu_arr = np.array(meta_g["fluences_mJ_cm2"], dtype=float)
                if scans_arr.size == flu_arr.size:
                    for s, fl in zip(scans_arr.tolist(), flu_arr.tolist()):
                        s = int(s)
                        fl = float(fl)
                        scan_to_flu[s] = fl
                        scan_to_tag[s] = general_utils.fluence_tag_file(fl)

            for delay_fs in tqdm(selected_delays_fs.tolist(), desc=f"2D {scan_type} images", unit="delay"):
                delay_fs = int(delay_fs)
                delay_group = f"{delay_fs}fs"
                if delay_group not in delays_root:
                    raise KeyError(f"Missing group /delays/{delay_group}")

                scans_g = delays_root[delay_group]["scans"]

                if scan_type == "fluence":
                    tag_to_entries: Dict[str, List[Tuple[int, np.ndarray]]] = {}
                    for scan_str in scans_g.keys():
                        scan = int(scan_str)
                        idx = np.array(scans_g[scan_str]["indices"], dtype=np.int64)
                        if idx.size == 0:
                            continue
                        tag = scan_to_tag.get(scan, None)
                        if tag is None:
                            try:
                                tag = general_utils.fluence_tag_file(float(scans_g[scan_str].attrs.get("fluence_mJ_cm2", np.nan)))
                            except Exception:
                                tag = None
                        if tag is None:
                            continue
                        tag_to_entries.setdefault(tag, []).append((scan, idx))

                    for tag, entries in tag_to_entries.items():
                        total_sum = None
                        total_n = 0
                        rep_scan = int(entries[0][0])
                        rep_flu = float(scan_to_flu.get(rep_scan, float("nan")))

                        for scan, idx in entries:
                            raw_path = self._scan_file(scan)
                            if not os.path.exists(raw_path):
                                raise FileNotFoundError(raw_path)

                            with h5.File(raw_path, "r") as rf:
                                dset = _h5_get_local(rf, pilatus_h5_path)
                                ssum, n_good = _sum_frames_by_indices_local(dset, idx, batch_size=batch_size)

                            if ssum is None or n_good == 0:
                                continue

                            if total_sum is None:
                                total_sum = np.zeros_like(ssum, dtype=np.float64)

                            total_sum += ssum
                            total_n += int(n_good)

                        if total_sum is None or total_n == 0:
                            results[(delay_fs, tag)] = {"path": "", "n_images": 0, "fluence_mJ_cm2": float(rep_flu)}
                            continue

                        avg_img = (total_sum / float(total_n)).astype(out_dtype, copy=False)

                        out_name = (
                            f"{sample_name}_{temperature_K}K_{wl_tag}nm_"
                            f"{tag}mJ_{time_window_fs}fs_{delay_fs}fs.npy"
                        )
                        out_path = os.path.join(out_dir, out_name)

                        if os.path.exists(out_path) and not overwrite:
                            raise FileExistsError(f"File exists: {out_path} (set overwrite=True to replace).")

                        np.save(out_path, avg_img)
                        results[(delay_fs, tag)] = {"path": out_path, "n_images": int(total_n), "fluence_mJ_cm2": float(rep_flu)}

                    continue

                fluence_mJ_cm2 = float(_read_meta_value_local(meta_g, "fluence_mJ_cm2"))
                flu_file = general_utils.fluence_tag_file(float(fluence_mJ_cm2))

                total_sum = None
                total_n = 0

                for scan_str in scans_g.keys():
                    scan = int(scan_str)
                    idx = np.array(scans_g[scan_str]["indices"], dtype=np.int64)
                    if idx.size == 0:
                        continue

                    raw_path = self._scan_file(scan)
                    if not os.path.exists(raw_path):
                        raise FileNotFoundError(raw_path)

                    with h5.File(raw_path, "r") as rf:
                        dset = _h5_get_local(rf, pilatus_h5_path)
                        scan_sum, n_good = _sum_frames_by_indices_local(dset, idx, batch_size=batch_size)

                    if scan_sum is None or n_good == 0:
                        continue

                    if total_sum is None:
                        total_sum = np.zeros_like(scan_sum, dtype=np.float64)

                    total_sum += scan_sum
                    total_n += int(n_good)

                if total_sum is None or total_n == 0:
                    results[delay_fs] = {"path": "", "n_images": 0}
                    continue

                avg_img = (total_sum / float(total_n)).astype(out_dtype, copy=False)

                out_name = (
                    f"{sample_name}_{temperature_K}K_{wl_tag}nm_"
                    f"{flu_file}mJ_{time_window_fs}fs_{delay_fs}fs.npy"
                )
                out_path = os.path.join(out_dir, out_name)

                if os.path.exists(out_path) and not overwrite:
                    raise FileExistsError(f"File exists: {out_path} (set overwrite=True to replace).")

                np.save(out_path, avg_img)
                results[delay_fs] = {"path": out_path, "n_images": int(total_n)}

        return results

    # ----------------------------
    # Parallel exporter (delay + dark + fluence)
    # ----------------------------
    def export_delay_2d_images_parallel(
        self,
        *,
        meta: Optional["ExperimentMeta"] = None,
        metadata_h5_path: Optional[str] = None,
        pilatus_h5_path: Optional[Tuple[str, ...]] = None,
        out_folder_name: str = "2D_images",
        batch_size: int = 256,
        out_dtype=np.float32,
        overwrite: bool = False,
        max_workers: Optional[int] = None,
        chunk_size: int = 2,
        start_method: str = "spawn",
        rdcc_nbytes: int = 0,
        rdcc_nslots: int = 0,
    ) -> Dict[Any, Dict[str, Union[str, int, float]]]:
        """Export delay 2D images parallel.

        Parameters
        ----------
        meta : Optional['ExperimentMeta']
            Experiment metadata mapping associated with the data series.
        metadata_h5_path : Optional[str]
            Filesystem path for metadata HDF5.
        pilatus_h5_path : Optional[Tuple[str, ...]]
            Filesystem path for pilatus HDF5.
        out_folder_name : str
            Name of the output folder below the experiment analysis directory.
        batch_size : int
            Number of detector frames read and averaged per batch.
        out_dtype : object
            NumPy data type used for exported averaged images.
        overwrite : bool
            Whether existing output artifacts may be replaced.
        max_workers : Optional[int]
            Maximum number of worker processes; ``None`` uses the executor default.
        chunk_size : int
            Number of delay bins assigned to each multiprocessing task.
        start_method : str
            Multiprocessing start method, such as ``spawn`` or ``fork``.
        rdcc_nbytes : int
            HDF5 raw-data chunk-cache size in bytes for each worker.
        rdcc_nslots : int
            Number of hash slots in each worker's HDF5 raw-data chunk cache.

        Returns
        -------
        Dict[Any, Dict[str, Union[str, int, float]]]
            Per-delay export report merged from all worker processes.

        Raises
        ------
        FileNotFoundError
            If required raw data, calibration, metadata, or cached analysis files are missing.
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        FileExistsError
            If the operation encounters this explicit failure condition.

        Notes
        -----
        This operation may create or replace analysis artifacts according to its save and overwrite settings.
        """
        if metadata_h5_path is None:
            if meta is None:
                raise ValueError("Provide either meta=... or metadata_h5_path=...")
            metadata_h5_path = self.metadata_h5_path(meta, scans=self.scans, paths=self.paths)

        if pilatus_h5_path is None:
            pilatus_h5_path = self.PILATUS_H5_PATH

        if not os.path.exists(metadata_h5_path):
            raise FileNotFoundError(metadata_h5_path)

        h5_dir = os.path.dirname(os.path.abspath(metadata_h5_path))
        out_dir = os.path.join(h5_dir, out_folder_name)
        os.makedirs(out_dir, exist_ok=True)

        with h5.File(metadata_h5_path, "r") as f:
            if "meta" not in f:
                raise ValueError("Invalid metadata H5: missing /meta")

            meta_g = f["meta"]
            scan_type = str(_read_meta_value_local(meta_g, "scan_type")).strip().lower()

            if scan_type == "dark":
                if "scans" not in f:
                    raise ValueError("Invalid dark metadata H5: missing /scans")
                scan_list = [int(s) for s in list(f["scans"].keys())]
                selected_delays_fs: List[int] = []
            else:
                if "selected_delays_fs" not in meta_g:
                    raise ValueError("Invalid metadata H5: missing /meta/selected_delays_fs")
                selected_delays_fs = np.array(meta_g["selected_delays_fs"], dtype=np.int64).tolist()
                scan_list = []

        if max_workers is None:
            max_workers = max(1, min(4, (os.cpu_count() or 2)))

        try:
            ctx = mp.get_context(start_method)
        except Exception:
            ctx = mp.get_context("fork")

        def _spawn_is_known_bad() -> bool:
            """Return spawn is known bad."""
            if start_method != "spawn":
                return False
            main_mod = sys.modules.get("__main__", None)
            if main_mod is None:
                return True
            if not hasattr(main_mod, "__spec__"):
                return True
            return False

        if _spawn_is_known_bad():
            ctx = mp.get_context("fork")

        if scan_type == "dark":
            if not scan_list:
                return {}

            with h5.File(metadata_h5_path, "r") as fmeta:
                meta_g2 = fmeta["meta"]
                scans_root2 = fmeta["scans"]
                sample_name = str(_read_meta_value_local(meta_g2, "sample_name"))
                temperature_K = int(_read_meta_value_local(meta_g2, "temperature_K"))

                nshots_map = {}
                for s in scans_root2.keys():
                    scan = int(s)
                    nshots_map[scan] = int(scans_root2[s].attrs.get("nshots_expected", 0))

            scan_tag_file = general_utils.scan_tag_file(scan_list)

            scans_chunks = general_utils.chunk_list(scan_list, int(chunk_size))
            payloads = [{
                "scans_chunk": [int(x) for x in chunk],
                "nshots_map": nshots_map,
                "batch_size": int(batch_size),
                "pilatus_h5_path": tuple(pilatus_h5_path),
                "raw_root": str(self.raw_root),
                "scan_file_pattern": str(self.scan_file_pattern),
                "rdcc_nbytes": int(rdcc_nbytes),
                "rdcc_nslots": int(rdcc_nslots),
            } for chunk in scans_chunks]

            total_sum: Optional[np.ndarray] = None
            total_n: int = 0

            try:
                with ctx.Pool(processes=int(max_workers)) as pool:
                    it = pool.imap_unordered(_export_dark_scan_chunk_worker, payloads)
                    for part in tqdm(it, total=len(payloads), desc="2D dark images (parallel combine)", unit="chunk"):
                        s = part.get("sum", None)
                        n = int(part.get("n", 0))
                        if s is None or n == 0:
                            continue
                        if total_sum is None:
                            total_sum = np.zeros_like(s, dtype=np.float64)
                        total_sum += s
                        total_n += n

            except Exception as e:
                if start_method == "spawn" and "__spec__" in str(e):
                    ctx2 = mp.get_context("fork")
                    with ctx2.Pool(processes=int(max_workers)) as pool:
                        it = pool.imap_unordered(_export_dark_scan_chunk_worker, payloads)
                        for part in tqdm(it, total=len(payloads), desc="2D dark images (fork fallback combine)", unit="chunk"):
                            s = part.get("sum", None)
                            n = int(part.get("n", 0))
                            if s is None or n == 0:
                                continue
                            if total_sum is None:
                                total_sum = np.zeros_like(s, dtype=np.float64)
                            total_sum += s
                            total_n += n
                else:
                    raise

            results: Dict[Any, Dict[str, Union[str, int, float]]] = {}

            if total_sum is None or total_n == 0:
                results[-1] = {"path": "", "n_images": 0}
                return results

            avg_img = (total_sum / float(total_n)).astype(out_dtype, copy=False)

            out_name = f"{sample_name}_{temperature_K}K_dark_{scan_tag_file}.npy"
            out_path = os.path.join(out_dir, out_name)

            if os.path.exists(out_path) and not overwrite:
                raise FileExistsError(f"File exists: {out_path} (set overwrite=True to replace).")

            np.save(out_path, avg_img)
            results[-1] = {"path": out_path, "n_images": int(total_n)}
            return results

        if scan_type == "fluence":
            if not selected_delays_fs:
                return {}
            delay_fs = int(selected_delays_fs[0])

            with h5.File(metadata_h5_path, "r") as fmeta:
                meta_g2 = fmeta["meta"]
                scans_arr = np.array(meta_g2["scans"], dtype=np.int64)
                flu_arr = np.array(meta_g2["fluences_mJ_cm2"], dtype=float)
                tags = []
                if scans_arr.size == flu_arr.size and scans_arr.size > 0:
                    for fl in flu_arr.tolist():
                        tags.append(general_utils.fluence_tag_file(float(fl)))
                unique_tags = sorted(set(tags))

            if not unique_tags:
                return {}

            tag_chunks = general_utils.chunk_list(unique_tags, int(chunk_size))
            payloads = [{
                "metadata_h5_path": metadata_h5_path,
                "delay_fs": int(delay_fs),
                "flu_tags_chunk": list(chunk),
                "out_dir": out_dir,
                "overwrite": bool(overwrite),
                "batch_size": int(batch_size),
                "out_dtype_str": np.dtype(out_dtype).str,
                "pilatus_h5_path": tuple(pilatus_h5_path),
                "raw_root": str(self.raw_root),
                "scan_file_pattern": str(self.scan_file_pattern),
                "rdcc_nbytes": int(rdcc_nbytes),
                "rdcc_nslots": int(rdcc_nslots),
            } for chunk in tag_chunks]

            results: Dict[Any, Dict[str, Union[str, int, float]]] = {}

            try:
                with ctx.Pool(processes=int(max_workers)) as pool:
                    it = pool.imap_unordered(_export_fluence_group_chunk_worker, payloads)
                    for chunk_res in tqdm(it, total=len(payloads), desc="2D fluence images (parallel)", unit="chunk"):
                        for tag, dct in chunk_res.items():
                            results[(delay_fs, tag)] = dct

            except Exception as e:
                if start_method == "spawn" and "__spec__" in str(e):
                    ctx2 = mp.get_context("fork")
                    with ctx2.Pool(processes=int(max_workers)) as pool:
                        it = pool.imap_unordered(_export_fluence_group_chunk_worker, payloads)
                        for chunk_res in tqdm(it, total=len(payloads), desc="2D fluence images (fork fallback)", unit="chunk"):
                            for tag, dct in chunk_res.items():
                                results[(delay_fs, tag)] = dct
                else:
                    raise

            return results

        if not selected_delays_fs:
            return {}

        delays_chunks = general_utils.chunk_list(selected_delays_fs, int(chunk_size))
        payloads = [{
            "metadata_h5_path": metadata_h5_path,
            "delays_chunk": [int(x) for x in chunk],
            "out_dir": out_dir,
            "overwrite": bool(overwrite),
            "batch_size": int(batch_size),
            "out_dtype_str": np.dtype(out_dtype).str,
            "pilatus_h5_path": tuple(pilatus_h5_path),
            "raw_root": str(self.raw_root),
            "scan_file_pattern": str(self.scan_file_pattern),
            "rdcc_nbytes": int(rdcc_nbytes),
            "rdcc_nslots": int(rdcc_nslots),
        } for chunk in delays_chunks]

        results_delay: Dict[int, Dict[str, Union[str, int]]] = {}

        try:
            with ctx.Pool(processes=int(max_workers)) as pool:
                it = pool.imap_unordered(_export_delay_chunk_worker, payloads)
                for chunk_res in tqdm(it, total=len(payloads), desc="2D delay images (parallel)", unit="chunk"):
                    results_delay.update(chunk_res)

        except Exception as e:
            if start_method == "spawn" and "__spec__" in str(e):
                ctx2 = mp.get_context("fork")
                with ctx2.Pool(processes=int(max_workers)) as pool:
                    it = pool.imap_unordered(_export_delay_chunk_worker, payloads)
                    for chunk_res in tqdm(it, total=len(payloads), desc="2D delay images (fork fallback)", unit="chunk"):
                        results_delay.update(chunk_res)
            else:
                raise

        return {int(d): results_delay[int(d)] for d in sorted(results_delay.keys())}
