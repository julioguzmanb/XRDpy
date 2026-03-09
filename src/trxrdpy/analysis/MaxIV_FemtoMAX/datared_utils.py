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
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime, timezone
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
    """
    Normalize path configuration into an AnalysisPaths instance.

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
    """
    Returns a string suitable for folder/file naming, avoiding trailing '.0' when integer-like.
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


# Ping reference provider (extend this)
def ref_pings(scan: int) -> Tuple[float, float]:
    """
    Return (ping2_ref, ping4_ref) in seconds for the given scan.
    Extend this with full mapping (dict/set/ranges).
    """
    if 167246 <= scan <= 167284:
        return 4.624268e-8, 4.821033e-8
    if 167285 <= scan <= 167320:
        return 4.623986e-8, 4.820762e-8
    if 167324 <= scan <= 167405:
        return 4.624187e-8, 4.820943e-8
    if 167498 <= scan <= 167552:
        return 4.971595e-10, 2.791438e-8
    if 167553 <= scan <= 167631:
        return 4.970023e-10, 2.796843e-8
    if 167632 <= scan <= 167772:
        return 6.623117e-10, 2.796876e-8
    if 167773 <= scan <= 167777:
        return 4.968675e-10, 2.796853e-8
    if 167778 <= scan <= 167778:
        return 4.972371e-10, 2.785008e-8
    if 167779 <= scan <= 167998:
        return 4.966647e-10, 2.796863e-8
    if 167245 <= scan:
        return 4.623986e-8, 4.820762e-8

    raise KeyError(f"reference pings haven't been declared for scan {scan}")


# Metadata (needed for saving/building)
@dataclass(frozen=True)
class ExperimentMeta:
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
    x = h5obj
    for k in path_tuple:
        x = x[k]
    return x


def _read_meta_value_local(meta_g: h5.Group, key: str, default=None):
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
    """
    Sum selected frames, rejecting any frame that contains a NaN.
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
    """
    Worker processes a CHUNK of delays (delay scan_type).
    """
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
    """
    Worker for dark scan_type: compute partial sum over a chunk of scans.
    """
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
    """
    Worker for scan_type="fluence": for a SINGLE delay bin, export one image per fluence group,
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
    """
    Supports:
      - scan_type="delay"   -> /meta + /delays/<delay>/scans/<scan>/indices
      - scan_type="dark"    -> /meta + /scans/<scan>/...  (all shots)
      - scan_type="fluence" -> /meta + /delays/<delay>/scans/<scan>/indices (export one per fluence group)
    """
    PILATUS_H5_PATH = ("entry", "measurement", "pilatus", "data")

    def __init__(
        self,
        scans: Union[int, Sequence[int]],
        *,
        ref_provider: Callable[[int], Tuple[float, float]] = ref_pings,
        paths: Optional[AnalysisPaths] = None,
        path_root: Optional[Union[str, Path]] = None,
        raw_subdir: Optional[Union[str, Path]] = None,
        analysis_subdir: Optional[Union[str, Path]] = None,
        scan_file_pattern: str = DEFAULT_SCAN_FILE_PATTERN,
        ping2_h5_path: Tuple[str, ...] = PING2_H5_PATH,
        ping4_h5_path: Tuple[str, ...] = PING4_H5_PATH,
    ):
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

        self.ref_provider = ref_provider
        self.scan_file_pattern = str(scan_file_pattern)
        self.ping2_h5_path = ping2_h5_path
        self.ping4_h5_path = ping4_h5_path

    # ----------------------------
    # Paths
    # ----------------------------
    def _scan_file(self, scan: int) -> str:
        return str(self.raw_root / self.scan_file_pattern.format(scan=int(scan)))

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
        """
        delay:
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
        return np.isfinite(p2_s) & np.isfinite(p4_s)

    @staticmethod
    def to_fs_int(x_seconds: np.ndarray) -> np.ndarray:
        return np.rint(x_seconds * 1e15).astype(np.int64)

    @staticmethod
    def _delay_series_seconds(p2_s: np.ndarray, p4_s: np.ndarray, delay_source: str) -> np.ndarray:
        if delay_source == "p2":
            return p2_s
        if delay_source == "p4":
            return p4_s
        if delay_source == "avg":
            return 0.5 * (p2_s + p4_s)
        raise ValueError("delay_source must be 'avg', 'p2', or 'p4'")

    @staticmethod
    def _valid_mask_for_source(p2_s: np.ndarray, p4_s: np.ndarray, delay_source: str, require_both: bool) -> np.ndarray:
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
        p2_s, p4_s = self.read_corrected_pings_seconds(scan)
        valid = self._valid_mask_for_source(p2_s, p4_s, delay_source, require_both)
        d_s = self._delay_series_seconds(p2_s, p4_s, delay_source)[valid]

        if unit == "fs":
            return d_s * 1e15
        if unit == "ps":
            return d_s * 1e12
        raise ValueError("unit must be 'fs' or 'ps'")

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
        show_median: bool = True,
        alpha: float = 0.6,
        ms: float = 3.0,
        title: Optional[str] = None,
    ) -> None:
        if scans is None:
            scans_list = self.scans
        elif isinstance(scans, int):
            scans_list = [int(scans)]
        else:
            scans_list = [int(s) for s in list(scans)]

        if mode not in ("overlay", "per_scan"):
            raise ValueError("mode must be 'overlay' or 'per_scan'")
        if view not in ("scatter", "hist"):
            raise ValueError("view must be 'scatter' or 'hist'")

        the_title = title if title is not None else f"Delay distribution ({delay_source}) - {view}"

        delays_by_scan: Dict[int, np.ndarray] = {}
        for scan in tqdm(scans_list, desc="Collecting delays", unit="scan"):
            d = self.get_delays(scan, delay_source=delay_source, require_both=require_both, unit=unit)
            if d.size == 0:
                continue
            delays_by_scan[int(scan)] = d

        if not delays_by_scan:
            return

        plotter = plot_utils.DelayDistributionPlotter()
        plotter.plot(
            delays_by_scan,
            mode=mode,
            view=view,
            unit=unit,
            bins=bins,
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
        arr = np.asarray(arr)
        if arr.size == 0:
            return
        old = ds.shape[0]
        ds.resize((old + arr.shape[0],))
        ds[old:] = arr

    @staticmethod
    def _write_scalar_dataset(group: h5.Group, name: str, value):
        if isinstance(value, str):
            dt = h5.string_dtype(encoding="utf-8")
            group.create_dataset(name, data=np.array(value, dtype=dt))
        else:
            group.create_dataset(name, data=np.array(value))

    # ----------------------------
    # Dark metadata builder
    # ----------------------------
    def _build_metadata_h5_dark(self, *, meta: "ExperimentMeta", overwrite: bool = False) -> str:
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
        st = str(meta.scan_type).strip().lower()

        if st == "dark":
            return self._build_metadata_h5_dark(meta=meta, overwrite=overwrite)

        if st not in ("delay", "fluence"):
            raise ValueError("scan_type must be 'delay', 'fluence', or 'dark'.")

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
