#!/usr/bin/env python3
import stpy
import dbpy
import numpy as np
import pandas as pd
import h5py
import os
import shutil
import sys
import argparse

from pathlib import Path

try:
    from ..common.paths import AnalysisPaths
except Exception:
    try:
        from trxrdpy.analysis.common.paths import AnalysisPaths
    except Exception:
        AnalysisPaths = None


def _bootstrap_paths():
    """
    Initialize legacy global path variables without touching the rest of the script.

    Expected env vars:
      XRDPY_PATH_ROOT                required
      XRDPY_ANALYSIS_SUBDIR          optional, default: analysis
      XRDPY_TIME_METADATA_SUBDIR     optional, default: TM_data
    """
    if AnalysisPaths is None:
        raise ImportError("Could not import AnalysisPaths from XRDpy.analysis.common.paths")

    path_root = os.environ.get("XRDPY_PATH_ROOT", None)
    analysis_subdir = os.environ.get("XRDPY_ANALYSIS_SUBDIR", "analysis")
    time_metadata_subdir = os.environ.get("XRDPY_TIME_METADATA_SUBDIR", "TM_data")

    if path_root is None:
        raise ValueError(
            "XRDPY_PATH_ROOT is not set. "
            "Set XRDPY_PATH_ROOT before running this script."
        )

    paths = AnalysisPaths(
        path_root=Path(path_root),
        analysis_subdir=str(analysis_subdir),
    )

    path_root_str = str(Path(paths.path_root))
    path_analysis_folder_str = str(Path(paths.analysis_root))
    path_time_metadata_str = str(Path(paths.root(str(time_metadata_subdir))))

    return paths, path_root_str, path_analysis_folder_str, path_time_metadata_str


PATHS, PATH_ROOT, PATH_ANALYSIS_FOLDER, PATH_TIME_METADATA = _bootstrap_paths()

def read_metadata_time(run):
    file_path = os.path.join(PATH_TIME_METADATA, "{}.csv".format(run))  # CSV file created with the time analysis tool.
    timing_metadata = pd.read_csv(file_path, skiprows=1)
    return timing_metadata

def read_metadata(bl, run, scan_type="delay"):
    parameters = [
        "xfel_bl_3_shutter_1_open_valid/status",  # X-ray shutter status
        "xfel_bl_3_lh1_shutter_1_open_valid/status",  # Laser shutter status
        "xfel_mon_msbpm_bl3_dump_1_beamstatus/summary",
        "xfel_bl_3_st_2_pd_user_8_fitting_peak/voltage",
        "xfel_bl_3_st_2_pd_user_9_fitting_peak/voltage",
        "xfel_bl_3_st_2_pd_user_4_fitting_peak/voltage",
        "xfel_bl_3_st_1_motor_73/position",
        "xfel_bl_3_st_2_motor_1/position",  # Programmed delay [motor position]
        "xfel_bl_3_st_2_motor_2/position",  # Programmed fluence [motor position]
        "xfel_bl_3_st_2_motor_5/position",
        "xfel_bl_3_st_2_motor_6/position"
    ]
    hightag = dbpy.read_hightagnumber(bl, run)
    tags = dbpy.read_taglist_byrun(bl, run)
    metadata = {"tag_number": tags}
    tags = tuple(tags)
    for param in parameters:
        metadata[param] = np.array(dbpy.read_syncdatalist_float(param, hightag, tags))
    metadata = pd.DataFrame(metadata)

    timing_metadata = read_metadata_time(run)
    final_metadata = pd.merge(metadata, timing_metadata, on='tag_number', how='inner')

    # Removing shots with no laser and/or no X-rays
    if scan_type == "delay":
        final_metadata = final_metadata[final_metadata["xfel_bl_3_shutter_1_open_valid/status"] == 1]
        final_metadata = final_metadata.drop("xfel_bl_3_shutter_1_open_valid/status", axis=1)
        final_metadata = final_metadata[final_metadata["xfel_bl_3_lh1_shutter_1_open_valid/status"] == 1]
        final_metadata = final_metadata.drop("xfel_bl_3_lh1_shutter_1_open_valid/status", axis=1)
        final_metadata = final_metadata[final_metadata["xfel_mon_msbpm_bl3_dump_1_beamstatus/summary"] == 1]
        final_metadata = final_metadata.drop("xfel_mon_msbpm_bl3_dump_1_beamstatus/summary", axis=1)
        delays = final_metadata["timing_edge_derivative(pixel)"] * (-2.7) + final_metadata["xfel_bl_3_st_2_motor_1/position"] * (6.67)
        final_metadata["delay [fs]"] = np.array(delays)

    elif scan_type == "background":
        pass
        # (Background-specific processing can be added here)

    elif scan_type == "calibration":
        final_metadata = final_metadata[final_metadata["xfel_bl_3_shutter_1_open_valid/status"] == 1]
        final_metadata = final_metadata.drop("xfel_bl_3_shutter_1_open_valid/status", axis=1)
        final_metadata = final_metadata[final_metadata["xfel_mon_msbpm_bl3_dump_1_beamstatus/summary"] == 1]
        final_metadata = final_metadata.drop("xfel_mon_msbpm_bl3_dump_1_beamstatus/summary", axis=1)

    elif scan_type == "fluence":
        final_metadata = final_metadata[final_metadata["xfel_bl_3_shutter_1_open_valid/status"] == 1]
        final_metadata = final_metadata.drop("xfel_bl_3_shutter_1_open_valid/status", axis=1)
        final_metadata = final_metadata[final_metadata["xfel_bl_3_lh1_shutter_1_open_valid/status"] == 1]
        final_metadata = final_metadata.drop("xfel_bl_3_lh1_shutter_1_open_valid/status", axis=1)
        final_metadata = final_metadata[final_metadata["xfel_mon_msbpm_bl3_dump_1_beamstatus/summary"] == 1]
        final_metadata = final_metadata.drop("xfel_mon_msbpm_bl3_dump_1_beamstatus/summary", axis=1)
        delays = final_metadata["timing_edge_derivative(pixel)"] * (-2.7) + final_metadata["xfel_bl_3_st_2_motor_1/position"] * (6.67)
        final_metadata["delay [fs]"] = np.array(delays)

    return final_metadata

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("{} created!".format(folder_path))
    else:
        print("Folder already exists!")

def get_delays(folder_root):
    try:
        subfolders = [d for d in os.listdir(folder_root) if os.path.isdir(os.path.join(folder_root, d))]
        # Assume folder names are like "12345_fs" (or "-28695_fs"); extract the delay part.
        delays = [int(d.split('_fs')[0]) for d in subfolders if d.endswith('_fs')]
        delays = np.sort(np.array(delays))
        return delays
    except FileNotFoundError:
        print("The folder does not exist.")
        return []
    
def create_delay_list(bl, run, time_step, initial_delay=None, final_delay=None, scan_type="delay"):
    metadata = read_metadata(bl, run, scan_type)
    if scan_type == "delay":
        unique_delays = np.unique(metadata["xfel_bl_3_st_2_motor_1/position"]) * 6.67  # from motor position to fs
        min_prog_delay = min(unique_delays)
        max_prog_delay = max(unique_delays)
        mean_offset = np.mean(metadata["timing_edge_derivative(pixel)"]) * -2.7  # pixel to fs
        if initial_delay is None:
            initial_delay = min_prog_delay + mean_offset
        if final_delay is None:
            final_delay = max_prog_delay + mean_offset
        delay_list = np.arange(initial_delay, final_delay + time_step, time_step, dtype=int)
        return delay_list
    elif scan_type in ["background", "calibration"]:
        return []
    
def get_2D_img_per_tag(collecter, buffer, tag):
    try:
        collecter.collect(buffer, tag)
        img = np.array(buffer.read_det_data(0))
        return img
    except: return None  

def create_metadata(
    bl,
    run,
    sample_name,
    temperature_K,
    scan_type,
    time_window_fs,
    excitation_wl_nm=None,
    fluence_mJ_cm2=None,
    time_step=None,
    initial_delay=None,
    final_delay=None,
    delay_for_fluence=None,
    intensity_col="xfel_bl_3_st_2_pd_user_9_fitting_peak/voltage",
    sigma=1.0,
    min_tags=100,
    overwrite=True,
    fluence_map=None):
    """
    Multi-run aware metadata creator.

    Supported scan_type (case-insensitive):
      - "delay"
      - "fluence"
      - "calibration"  -> treated as "dark" in the analysis tree

    `run` can be:
      - int (single run)
      - list/tuple/np.ndarray of ints (multiple runs)
      - string "1466583,1466578" or "1466583 1466578"

    Output layout (same as before for delay/fluence; dark differs when multi-run):
      dark (single run):
        <PATH_ANALYSIS_FOLDER>/<sample>/temperature_<T>K/dark/scan_<run>/metadata/<...>.h5
      dark (multi-run):
        <PATH_ANALYSIS_FOLDER>/<sample>/temperature_<T>K/dark/scans_<runs_tag>/metadata/<...>.h5

      delay:
        <PATH_ANALYSIS_FOLDER>/<sample>/temperature_<T>K/excitation_wl_<wl>nm/delay/fluence_<flu>mJ/time_window_<tw>fs/metadata/<...>.h5

      fluence:
        <PATH_ANALYSIS_FOLDER>/<sample>/temperature_<T>K/excitation_wl_<wl>nm/fluence/delay_<delay>fs/time_window_<tw>fs/metadata/<...>.h5

    H5 schema (bin-oriented, now multi-run capable):
      /meta/...
      /meta/scans                 (int array of runs)
      /delays/<delay>fs/scans/<run>/tags          (delay)
      /fluences/<flu_tag>mJ/scans/<run>/tags      (fluence)
      /scans/<run>/tags                           (dark)

    Additional convenience datasets (for multi-run consumption later):
      /delays/<delay>fs/shot_run, /delays/<delay>fs/shot_tag
      /fluences/<flu_tag>mJ/shot_run, /fluences/<flu_tag>mJ/shot_tag
      /scans/shot_run, /scans/shot_tag

    Notes:
      - Uses .format() only (HPC-friendly).
      - Intensity filtering is applied ONLY here (metadata stage).
      - fluence_map: optional dict mapping motor_pos -> physical fluence (mJ/cm^2).
        If provided and exact key not found, nearest key is used (best-effort).
      - For delay mode:
          If initial_delay/final_delay are None, they are inferred from combined runs
          using motor_1 positions and mean timing offset (same logic as create_delay_list()).
    """

    def _now_utc_iso():
        from datetime import datetime
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def _ensure_dir(p):
        if not os.path.exists(p):
            os.makedirs(p)

    def _str_dtype():
        try:
            return h5py.string_dtype(encoding="utf-8")
        except Exception:
            return h5py.special_dtype(vlen=str)

    def _write_str_ds(g, name, s):
        dt = _str_dtype()
        g.create_dataset(name, data=np.array(str(s), dtype=dt))

    def _wl_tag_nm_local(x):
        # Mimic FemtoMAX wl_tag_nm: 1500 -> "1500", 1500.5 -> "1500p5"
        try:
            v = float(x)
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            s = str(v)
            if "." in s:
                s = s.rstrip("0").rstrip(".")
            return s.replace(".", "p")
        except Exception:
            return str(x)

    def _fluence_tag_file_local(x):
        # Mimic FemtoMAX fluence_tag_file: str(float(x)).replace(".", "p")
        try:
            return str(float(x)).replace(".", "p")
        except Exception:
            return str(x).replace(".", "p")

    def _fluence_tag_folder_local(x):
        return _fluence_tag_file_local(x) + "mJ"

    def _parse_runs(run_in):
        if run_in is None:
            raise ValueError("run must be provided.")
        runs_out = []
        if isinstance(run_in, (list, tuple, np.ndarray)):
            for r in list(run_in):
                try:
                    runs_out.append(int(r))
                except Exception:
                    pass
        elif isinstance(run_in, str):
            s = str(run_in).replace(",", " ")
            parts = s.split()
            for p in parts:
                try:
                    runs_out.append(int(p))
                except Exception:
                    pass
        else:
            runs_out = [int(run_in)]

        # unique, preserve order
        seen = set()
        uniq = []
        for r in runs_out:
            if r not in seen:
                uniq.append(int(r))
                seen.add(r)

        if len(uniq) == 0:
            raise ValueError("No valid runs parsed from 'run' argument.")
        return uniq

    def _runs_tag(runs_list):
        # Safe folder/file tag for multiple runs
        rr = [int(x) for x in runs_list]
        if len(rr) == 1:
            return str(rr[0])
        if len(rr) <= 5:
            return "_".join([str(x) for x in rr])
        return "N{}_{}-{}".format(int(len(rr)), int(min(rr)), int(max(rr)))

    def _resolve_fluence_phys(motor_pos_f):
        # Resolve physical fluence from motor position using fluence_map (best-effort)
        if fluence_map is None:
            return float(motor_pos_f)

        try:
            if motor_pos_f in fluence_map:
                return float(fluence_map[motor_pos_f])
        except Exception:
            pass

        # nearest key
        try:
            keys = np.array([float(k) for k in fluence_map.keys()], float)
            if keys.size > 0:
                j = int(np.argmin(np.abs(keys - float(motor_pos_f))))
                return float(fluence_map[keys[j]])
        except Exception:
            pass

        return float(motor_pos_f)

    # ----------------------------
    # Normalize scan type + validate args
    # ----------------------------
    st = str(scan_type).strip().lower()
    if st == "calibration":
        st = "dark"

    if sample_name is None or str(sample_name).strip() == "":
        raise ValueError("sample_name must be provided.")
    try:
        temperature_K = int(temperature_K)
    except Exception:
        raise ValueError("temperature_K must be an int.")

    if time_window_fs is None:
        raise ValueError("time_window_fs must be provided.")
    time_window_fs = int(time_window_fs)
    if time_window_fs <= 0:
        raise ValueError("time_window_fs must be > 0.")

    runs = _parse_runs(run)
    runs_tag = _runs_tag(runs)

    # ----------------------------
    # Build analysis_dir + metadata filename
    # ----------------------------
    if st == "dark":
        if len(runs) == 1:
            analysis_dir = os.path.join(
                PATH_ANALYSIS_FOLDER,
                str(sample_name),
                "temperature_{}K".format(int(temperature_K)),
                "dark",
                "scan_{}".format(int(runs[0])),
            )
            meta_basename = "{}_{}K_dark_scan{}.h5".format(str(sample_name), int(temperature_K), int(runs[0]))
        else:
            analysis_dir = os.path.join(
                PATH_ANALYSIS_FOLDER,
                str(sample_name),
                "temperature_{}K".format(int(temperature_K)),
                "dark",
                "scans_{}".format(str(runs_tag)),
            )
            meta_basename = "{}_{}K_dark_scans{}.h5".format(str(sample_name), int(temperature_K), str(runs_tag))

    elif st == "delay":
        if excitation_wl_nm is None:
            raise ValueError("excitation_wl_nm must be provided for scan_type='delay'.")
        if fluence_mJ_cm2 is None:
            raise ValueError("fluence_mJ_cm2 must be provided for scan_type='delay'.")

        wl_tag = _wl_tag_nm_local(excitation_wl_nm)
        flu_folder = _fluence_tag_folder_local(fluence_mJ_cm2)

        analysis_dir = os.path.join(
            PATH_ANALYSIS_FOLDER,
            str(sample_name),
            "temperature_{}K".format(int(temperature_K)),
            "excitation_wl_{}nm".format(str(wl_tag)),
            "delay",
            "fluence_{}".format(str(flu_folder)),
            "time_window_{}fs".format(int(time_window_fs)),
        )
        # NOTE: no run encoded here (by design); metadata file can cover multiple runs
        meta_basename = "{}_{}K_{}nm_{}_{}fs.h5".format(
            str(sample_name),
            int(temperature_K),
            str(wl_tag),
            str(flu_folder),
            int(time_window_fs),
        )

    elif st == "fluence":
        if excitation_wl_nm is None:
            raise ValueError("excitation_wl_nm must be provided for scan_type='fluence'.")

        wl_tag = _wl_tag_nm_local(excitation_wl_nm)

        # delay_for_fluence default computed after reading combined metadata below
        # (so analysis_dir depends on it; we will set it later)

        # placeholder: will define analysis_dir/meta_basename later
        analysis_dir = None
        meta_basename = None

    else:
        raise ValueError("scan_type must be 'delay', 'fluence', or 'calibration' (dark).")

    # For fluence: read combined first to determine delay_for_fluence if needed, then build paths
    if st == "fluence":
        meta_list0 = []
        for r in runs:
            m0 = read_metadata(int(bl), int(r), scan_type="fluence")
            if m0 is None or len(m0) == 0:
                continue
            m0 = m0.copy()
            m0["run_number"] = int(r)
            meta_list0.append(m0)

        if len(meta_list0) == 0:
            raise ValueError("No metadata rows returned for scan_type='fluence' across runs {}.".format(runs_tag))

        meta0_all = pd.concat(meta_list0, ignore_index=True)

        if delay_for_fluence is None:
            try:
                delay_for_fluence = float(np.mean(meta0_all["delay [fs]"]))
                print("Mean Delay (multi-run): {}".format(delay_for_fluence))
            except Exception:
                raise ValueError("Cannot infer delay_for_fluence (missing 'delay [fs]' in metadata).")

        delay_for_fluence = int(round(float(delay_for_fluence)))

        analysis_dir = os.path.join(
            PATH_ANALYSIS_FOLDER,
            str(sample_name),
            "temperature_{}K".format(int(temperature_K)),
            "excitation_wl_{}nm".format(str(wl_tag)),
            "fluence",
            "delay_{}fs".format(int(delay_for_fluence)),
            "time_window_{}fs".format(int(time_window_fs)),
        )
        meta_basename = "{}_{}K_{}nm_{}fs_{}fs.h5".format(
            str(sample_name),
            int(temperature_K),
            str(wl_tag),
            int(delay_for_fluence),
            int(time_window_fs),
        )

    meta_dir = os.path.join(analysis_dir, "metadata")
    _ensure_dir(meta_dir)
    meta_h5_path = os.path.join(meta_dir, meta_basename)

    if os.path.exists(meta_h5_path):
        if bool(overwrite):
            os.remove(meta_h5_path)
        else:
            raise FileExistsError("Metadata file exists: {}".format(meta_h5_path))

    created_utc = _now_utc_iso()

    # ----------------------------
    # Write HDF5
    # ----------------------------
    with h5py.File(meta_h5_path, "w") as hdf:
        gmeta = hdf.create_group("meta")

        # attrs (keep schema_version="1" for compatibility; add multi-run flag)
        gmeta.attrs["schema_version"] = "1"
        gmeta.attrs["created_utc"] = created_utc
        gmeta.attrs["beamline"] = int(bl)
        gmeta.attrs["n_runs"] = int(len(runs))
        gmeta.attrs["primary_run"] = int(runs[0])
        gmeta.attrs["sample_name"] = str(sample_name)
        gmeta.attrs["temperature_K"] = int(temperature_K)
        gmeta.attrs["scan_type"] = str(st)
        gmeta.attrs["time_window_fs"] = int(time_window_fs)
        gmeta.attrs["intensity_col"] = str(intensity_col)
        gmeta.attrs["sigma"] = float(sigma)
        gmeta.attrs["min_tags"] = int(min_tags)
        gmeta.attrs["multi_run"] = int(1 if len(runs) > 1 else 0)

        # datasets (portable)
        _write_str_ds(gmeta, "schema_version", "1")
        _write_str_ds(gmeta, "created_utc", created_utc)
        gmeta.create_dataset("beamline", data=np.array(int(bl), dtype=np.int64))

        # keep legacy scalar "run" for convenience (primary run); multi-run uses "scans"
        gmeta.create_dataset("run", data=np.array(int(runs[0]), dtype=np.int64))

        _write_str_ds(gmeta, "sample_name", str(sample_name))
        gmeta.create_dataset("temperature_K", data=np.array(int(temperature_K), dtype=np.int64))
        _write_str_ds(gmeta, "scan_type", str(st))
        gmeta.create_dataset("time_window_fs", data=np.array(int(time_window_fs), dtype=np.int64))
        _write_str_ds(gmeta, "intensity_col", str(intensity_col))
        gmeta.create_dataset("sigma", data=np.array(float(sigma), dtype=float))
        gmeta.create_dataset("min_tags", data=np.array(int(min_tags), dtype=np.int64))

        # scans list (multi-run)
        gmeta.create_dataset("scans", data=np.array([int(r) for r in runs], dtype=np.int64))

        if st in ("delay", "fluence"):
            gmeta.attrs["excitation_wl_nm"] = float(excitation_wl_nm)
            gmeta.create_dataset("excitation_wl_nm", data=np.array(float(excitation_wl_nm), dtype=float))

        if st == "delay":
            gmeta.attrs["fluence_mJ_cm2"] = float(fluence_mJ_cm2)
            gmeta.create_dataset("fluence_mJ_cm2", data=np.array(float(fluence_mJ_cm2), dtype=float))

            if time_step is None:
                raise ValueError("For scan_type='delay', provide time_step.")
            time_step = int(time_step)

            # Load combined metadata across runs
            meta_list = []
            for r in runs:
                md = read_metadata(int(bl), int(r), scan_type="delay")
                if md is None or len(md) == 0:
                    print("Warning: no delay metadata rows for run {}.".format(int(r)))
                    continue
                md = md.copy()
                md["run_number"] = int(r)
                meta_list.append(md)

            if len(meta_list) == 0:
                raise ValueError("No metadata rows returned for scan_type='delay' across runs {}.".format(runs_tag))

            meta_all = pd.concat(meta_list, ignore_index=True)

            # Infer initial/final if missing (same logic as create_delay_list(), but multi-run)
            if initial_delay is None or final_delay is None:
                try:
                    unique_prog = np.unique(meta_all["xfel_bl_3_st_2_motor_1/position"].values) * 6.67
                    min_prog_delay = float(np.min(unique_prog))
                    max_prog_delay = float(np.max(unique_prog))
                    mean_offset = float(np.mean(meta_all["timing_edge_derivative(pixel)"])) * (-2.7)
                except Exception as e:
                    raise ValueError("Cannot infer initial/final delay from metadata: {}".format(e))

                if initial_delay is None:
                    initial_delay = int(round(min_prog_delay + mean_offset))
                if final_delay is None:
                    final_delay = int(round(max_prog_delay + mean_offset))

            initial_delay = int(initial_delay)
            final_delay = int(final_delay)

            # Bin centers
            delays = np.arange(int(initial_delay), int(final_delay) + int(time_step), int(time_step), dtype=int)

            halfwin = float(time_window_fs) / 2.0
            valid_delays = []

            gdelays = hdf.create_group("delays")

            # For each delay bin, collect per-run tags, apply intensity filtering per-run, then combine
            for delay_fs in delays:
                delay_fs = int(delay_fs)
                dmin = float(delay_fs) - halfwin
                dmax = float(delay_fs) + halfwin

                per_run_tags = {}
                total_tags = 0

                for r in runs:
                    mdr = meta_all[meta_all["run_number"] == int(r)]
                    if mdr is None or len(mdr) == 0:
                        continue

                    filtered = mdr[(mdr["delay [fs]"] >= dmin) & (mdr["delay [fs]"] <= dmax)]
                    if filtered is None or filtered.shape[0] == 0:
                        continue

                    # intensity filtering (per-run)
                    try:
                        I_values = filtered[intensity_col].values
                    except Exception as e:
                        raise ValueError("Error accessing intensity column '{}' in metadata: {}".format(intensity_col, e))

                    I_nonan = I_values[~np.isnan(I_values)]
                    if len(I_nonan) == 0:
                        continue

                    I_median = np.median(I_nonan)
                    I_std = np.std(I_nonan)
                    if I_std == 0:
                        I_std = I_median

                    mask = (filtered[intensity_col] > (I_median - float(sigma) * I_std)) & (~filtered[intensity_col].isnull())
                    filtered = filtered[mask]

                    if filtered is None or filtered.shape[0] == 0:
                        continue

                    tags_r = np.array(filtered["tag_number"].values, dtype=np.int64)
                    if tags_r.size == 0:
                        continue

                    per_run_tags[int(r)] = tags_r
                    total_tags += int(tags_r.size)

                if total_tags < int(min_tags):
                    print("Delay {}: only {} total tags pass filtering (<{}). Skipping.".format(
                        int(delay_fs), int(total_tags), int(min_tags)))
                    continue

                valid_delays.append(int(delay_fs))

                dg = gdelays.create_group("{}fs".format(int(delay_fs)))
                dg.attrs["delay_fs"] = int(delay_fs)
                dg.attrs["time_window_fs"] = int(time_window_fs)
                dg.attrs["nshots_total_expected"] = int(total_tags)

                scans_g = dg.create_group("scans")

                # aggregated shot lists (run, tag) for convenience
                shot_run_list = []
                shot_tag_list = []

                for r in runs:
                    r_int = int(r)
                    if r_int not in per_run_tags:
                        continue
                    tags_r = per_run_tags[r_int]
                    sg = scans_g.create_group(str(r_int))
                    sg.create_dataset("tags", data=tags_r, compression="gzip", shuffle=True)
                    sg.attrs["nshots"] = int(tags_r.size)
                    sg.attrs["run_number"] = int(r_int)

                    shot_run_list.append(np.full(int(tags_r.size), int(r_int), dtype=np.int64))
                    shot_tag_list.append(tags_r.astype(np.int64))

                if len(shot_run_list) > 0:
                    dg.create_dataset("shot_run", data=np.concatenate(shot_run_list), compression="gzip", shuffle=True)
                    dg.create_dataset("shot_tag", data=np.concatenate(shot_tag_list), compression="gzip", shuffle=True)

            if len(valid_delays) == 0:
                raise ValueError("No valid delays found (all skipped by filtering).")

            gmeta.create_dataset("selected_delays_fs", data=np.array(valid_delays, dtype=np.int64))
            gmeta.attrs["selected_delays_mode"] = "manual"
            gmeta.attrs["delay_source"] = "timing+motor"
            gmeta.attrs["require_both_pings"] = int(0)

        elif st == "fluence":
            # Load combined metadata across runs (already partially loaded above for default delay inference)
            meta_list = []
            for r in runs:
                md = read_metadata(int(bl), int(r), scan_type="fluence")
                if md is None or len(md) == 0:
                    print("Warning: no fluence metadata rows for run {}.".format(int(r)))
                    continue
                md = md.copy()
                md["run_number"] = int(r)
                meta_list.append(md)

            if len(meta_list) == 0:
                raise ValueError("No metadata rows returned for scan_type='fluence' across runs {}.".format(runs_tag))

            meta_all = pd.concat(meta_list, ignore_index=True)

            # apply delay window around delay_for_fluence
            halfwin = float(time_window_fs) / 2.0
            dmin = float(delay_for_fluence) - halfwin
            dmax = float(delay_for_fluence) + halfwin
            meta_all = meta_all[(meta_all["delay [fs]"] >= dmin) & (meta_all["delay [fs]"] <= dmax)]

            if meta_all is None or len(meta_all) == 0:
                raise ValueError("No fluence metadata rows within delay window for runs {}.".format(runs_tag))

            # Build groups by physical fluence (with optional many-to-one mapping)
            flu_groups = {}  # key: flu_tag_str, value: dict(phys, motor_set, per_run_tags)
            for r in runs:
                r_int = int(r)
                mdr = meta_all[meta_all["run_number"] == r_int]
                if mdr is None or len(mdr) == 0:
                    continue

                unique_motor = np.unique(mdr["xfel_bl_3_st_2_motor_2/position"].values)
                for flu_motor in unique_motor:
                    try:
                        flu_motor_f = float(flu_motor)
                    except Exception:
                        continue

                    filtered = mdr[mdr["xfel_bl_3_st_2_motor_2/position"] == flu_motor_f]
                    if filtered is None or filtered.shape[0] == 0:
                        continue

                    # intensity filtering (per-run, per-motor)
                    try:
                        I_values = filtered[intensity_col].values
                    except Exception as e:
                        raise ValueError("Error accessing intensity column '{}' in metadata: {}".format(intensity_col, e))

                    I_nonan = I_values[~np.isnan(I_values)]
                    if len(I_nonan) == 0:
                        continue

                    I_median = np.median(I_nonan)
                    I_std = np.std(I_nonan)
                    if I_std == 0:
                        I_std = I_median

                    mask = (filtered[intensity_col] > (I_median - float(sigma) * I_std)) & (~filtered[intensity_col].isnull())
                    filtered = filtered[mask]
                    if filtered is None or filtered.shape[0] == 0:
                        continue

                    tags_r = np.array(filtered["tag_number"].values, dtype=np.int64)
                    if tags_r.size == 0:
                        continue

                    flu_phys = _resolve_fluence_phys(flu_motor_f)
                    flu_tag = _fluence_tag_file_local(flu_phys)

                    if flu_tag not in flu_groups:
                        flu_groups[flu_tag] = {
                            "phys": float(flu_phys),
                            "motor_set": set([float(flu_motor_f)]),
                            "per_run_tags": {},
                        }
                    else:
                        flu_groups[flu_tag]["motor_set"].add(float(flu_motor_f))

                    # accumulate tags for this run (handle possible multiple motor positions mapping to same phys)
                    if r_int in flu_groups[flu_tag]["per_run_tags"]:
                        flu_groups[flu_tag]["per_run_tags"][r_int] = np.concatenate(
                            (flu_groups[flu_tag]["per_run_tags"][r_int], tags_r.astype(np.int64))
                        )
                    else:
                        flu_groups[flu_tag]["per_run_tags"][r_int] = tags_r.astype(np.int64)

            if len(flu_groups.keys()) == 0:
                raise ValueError("No valid fluence groups found (all skipped by filtering) for runs {}.".format(runs_tag))

            # Write groups
            gflu = hdf.create_group("fluences")

            # sort by physical fluence for stable ordering
            flu_items = []
            for k in flu_groups.keys():
                try:
                    flu_items.append((float(flu_groups[k]["phys"]), str(k)))
                except Exception:
                    flu_items.append((0.0, str(k)))
            flu_items.sort(key=lambda x: x[0])

            valid_flu_phys = []
            valid_flu_motor_rep = []

            for phys_val, flu_tag in flu_items:
                entry = flu_groups[flu_tag]
                per_run_tags = entry["per_run_tags"]

                total_tags = 0
                for rr in per_run_tags.keys():
                    try:
                        total_tags += int(per_run_tags[int(rr)].size)
                    except Exception:
                        pass

                if total_tags < int(min_tags):
                    print("Fluence {}mJ: only {} total tags pass filtering (<{}). Skipping.".format(
                        str(flu_tag), int(total_tags), int(min_tags)))
                    continue

                motor_list = sorted(list(entry["motor_set"]))
                motor_rep = float(motor_list[0]) if len(motor_list) > 0 else float(entry["phys"])

                valid_flu_phys.append(float(entry["phys"]))
                valid_flu_motor_rep.append(float(motor_rep))

                fg = gflu.create_group("{}mJ".format(str(flu_tag)))
                fg.attrs["fluence_mJ_cm2"] = float(entry["phys"])
                fg.attrs["fluence_motor_pos_rep"] = float(motor_rep)
                fg.attrs["delay_fs"] = int(delay_for_fluence)
                fg.attrs["time_window_fs"] = int(time_window_fs)
                fg.attrs["nshots_total_expected"] = int(total_tags)

                # store motor positions that contributed
                fg.create_dataset("fluence_motor_pos_list", data=np.array(motor_list, dtype=float), compression="gzip", shuffle=True)

                scans_g = fg.create_group("scans")

                shot_run_list = []
                shot_tag_list = []

                # write per-run tags
                for r in runs:
                    r_int = int(r)
                    if r_int not in per_run_tags:
                        continue
                    tags_r = np.array(per_run_tags[r_int], dtype=np.int64)
                    if tags_r.size == 0:
                        continue

                    sg = scans_g.create_group(str(r_int))
                    sg.create_dataset("tags", data=tags_r, compression="gzip", shuffle=True)
                    sg.attrs["nshots"] = int(tags_r.size)
                    sg.attrs["run_number"] = int(r_int)
                    sg.attrs["fluence_mJ_cm2"] = float(entry["phys"])
                    sg.attrs["delay_fs"] = int(delay_for_fluence)

                    shot_run_list.append(np.full(int(tags_r.size), int(r_int), dtype=np.int64))
                    shot_tag_list.append(tags_r.astype(np.int64))

                if len(shot_run_list) > 0:
                    fg.create_dataset("shot_run", data=np.concatenate(shot_run_list), compression="gzip", shuffle=True)
                    fg.create_dataset("shot_tag", data=np.concatenate(shot_tag_list), compression="gzip", shuffle=True)

            if len(valid_flu_phys) == 0:
                raise ValueError("No valid fluence groups found after min_tags filtering for runs {}.".format(runs_tag))

            gmeta.attrs["delay_fs"] = int(delay_for_fluence)
            gmeta.create_dataset("delay_fs", data=np.array(int(delay_for_fluence), dtype=np.int64))
            # meta arrays correspond to fluence groups (phys) and a representative motor position
            gmeta.create_dataset("fluences_mJ_cm2", data=np.array(valid_flu_phys, dtype=float))
            gmeta.create_dataset("fluences_motor_pos", data=np.array(valid_flu_motor_rep, dtype=float))

        else:
            # dark (from SACLA "calibration" filter), multi-run supported
            gscans = hdf.create_group("scans")

            shot_run_list = []
            shot_tag_list = []

            total_tags = 0
            for r in runs:
                md = read_metadata(int(bl), int(r), scan_type="calibration")
                if md is None or len(md) == 0:
                    print("Warning: no dark(calibration) metadata rows for run {}.".format(int(r)))
                    continue

                tags = np.array(md["tag_number"].values, dtype=np.int64)
                if tags.size == 0:
                    continue

                total_tags += int(tags.size)

                sg = gscans.create_group(str(int(r)))
                sg.create_dataset("tags", data=tags, compression="gzip", shuffle=True)
                sg.attrs["nshots"] = int(tags.size)
                sg.attrs["run_number"] = int(r)

                shot_run_list.append(np.full(int(tags.size), int(r), dtype=np.int64))
                shot_tag_list.append(tags.astype(np.int64))

            if total_tags == 0:
                raise ValueError("No tags found for dark(calibration) metadata after filtering across runs {}.".format(runs_tag))

            # aggregated
            if len(shot_run_list) > 0:
                gscans.create_dataset("shot_run", data=np.concatenate(shot_run_list), compression="gzip", shuffle=True)
                gscans.create_dataset("shot_tag", data=np.concatenate(shot_tag_list), compression="gzip", shuffle=True)

    print("metadata h5: {}".format(meta_h5_path))
    return meta_h5_path

def process_background_run(bl, run, overwrite=True):
    """
    Process a SACLA background run (no X-rays):
      1) Create background metadata CSV in:
           <PATH_ANALYSIS_FOLDER>/<run>/metadata.csv
         using read_metadata(bl, run, scan_type="background")
      2) Average detector images over all tag_number entries in that CSV
      3) Save:
           <PATH_ANALYSIS_FOLDER>/<run>/<run>.npy

    Notes:
      - Uses .format() for HPC python compatibility.
      - Keeps the same CSV header convention as the old pipeline ("Background Scan" first line).
      - Does NOT write any HDF5 output (deprecated in your new flow).
    """
    folder_root = os.path.join(PATH_ANALYSIS_FOLDER, str(run))
    create_folder(folder_root)

    # ----------------------------
    # (1) Create / refresh metadata.csv
    # ----------------------------
    csv_path = os.path.join(folder_root, "metadata.csv")

    if (not os.path.exists(csv_path)) or bool(overwrite):
        metadata = read_metadata(bl, run, scan_type="background")
        if metadata is None or len(metadata) == 0:
            raise ValueError("No metadata rows returned for background run {}.".format(run))

        with open(csv_path, "w") as f:
            f.write("Background Scan\n")
            metadata.to_csv(f, index=False)

        print("Metadata CSV file created for background scan: {}".format(csv_path))
    else:
        print("Metadata CSV already exists (overwrite=False): {}".format(csv_path))

    # ----------------------------
    # (2) Load CSV and average images
    # ----------------------------
    try:
        data = pd.read_csv(csv_path, skiprows=1)
    except Exception as e:
        raise IOError("Error reading CSV file {}: {}".format(csv_path, e))

    if "tag_number" not in data.columns:
        raise KeyError("metadata.csv is missing required column 'tag_number'.")

    print("Processing Background scan")

    try:
        collecter = stpy.StorageReader("MPCCD-8N0-3-002", bl, (run,))
    except Exception as e:
        raise RuntimeError("Error initializing StorageReader: {}".format(e))

    buffer = stpy.StorageBuffer(collecter)

    def _ensure_2d(img):
        if img is None:
            return None
        arr = np.asarray(img)
        if hasattr(arr, "ndim") and arr.ndim == 3:
            return arr[0]
        return arr

    try:
        first_tag = data["tag_number"].iloc[0]
    except Exception as e:
        raise ValueError("Error accessing first tag in CSV: {}".format(e))

    print("Processing tag {}".format(first_tag))

    final_img = get_2D_img_per_tag(collecter, buffer, first_tag)
    final_img = _ensure_2d(final_img)
    if final_img is None or (not getattr(final_img, "size", 0)):
        raise ValueError("Initial image is empty or None")

    # ensure float accumulation
    final_img = final_img.astype(float)
    count = 1

    for idx, tag in enumerate(data["tag_number"].iloc[1:], start=1):
        img = get_2D_img_per_tag(collecter, buffer, tag)
        img = _ensure_2d(img)

        if img is None or (not getattr(img, "size", 0)):
            print("Warning: Retrieved empty image for tag {}".format(tag))
            continue
        elif np.isnan(img).any():
            print("Warning: the img contain a nan {}".format(tag))
            continue

        print("Processing tag {}, {} out of {}".format(tag, idx, len(data["tag_number"])))
        final_img += img.astype(float)
        count += 1

    if count == 0:
        raise ValueError("No valid images found for background run {}.".format(run))

    img_avg = final_img / float(count)

    # ----------------------------
    # (3) Save background average
    # ----------------------------
    out_path = os.path.join(folder_root, "{}.npy".format(run))
    try:
        np.save(out_path, img_avg)
    except Exception as e:
        raise IOError("Error saving NumPy file {}: {}".format(out_path, e))

    print("Final background image saved: {}".format(out_path))
    return img_avg

def create_dark_from_laser_off(
    runs,
    sample_name,
    temperature_K,
    excitation_wl_nm,
    fluence_mJ_cm2,
    time_window_fs,
    *,
    analysis_root=PATH_ANALYSIS_FOLDER,
    overwrite=True,
    min_files=1):
    """
    Build a representative DARK image by averaging all available *_laser_off.npy files
    for a given delay-scan experiment, and save it.

    Inputs:
      runs             : int, list/tuple/np.ndarray, or "1466556,1466557" / "1466556 1466557"
      sample_name      : e.g. "DET70"
      temperature_K    : int, e.g. 110
      excitation_wl_nm : float/int, e.g. 1500
      fluence_mJ_cm2   : float/int, e.g. 25  (will be tagged like "25p0mJ")
      time_window_fs   : int, e.g. 250

    Behavior:
      - Searches in:
          <analysis_root>/<sample>/temperature_<T>K/excitation_wl_<wl>nm/delay/fluence_<flu>mJ/time_window_<tw>fs/2D_images
        for files matching:
          <sample>_<T>K_<wl>nm_<flu>mJ_<tw>fs_*_laser_off.npy
      - Averages all matching files (skips unreadable / NaN / shape-mismatch files).
      - Saves to FemtoMAX-compatible dark path:
          <analysis_root>/<sample>/temperature_<T>K/dark/<scan_tag>/2D_images/<sample>_<T>K_dark_<scan_tag_file>.npy
        where scan_tag is derived from runs:
          single run -> scan_<run>
          multi-run  -> scans_<min>-<max>

    Returns:
      dict with keys:
        "path" (str), "n_files" (int), "used_files" (list of str)
    """

    def _wl_tag_nm_local(x):
        try:
            v = float(x)
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            s = str(v)
            if "." in s:
                s = s.rstrip("0").rstrip(".")
            return s.replace(".", "p")
        except Exception:
            return str(x)

    def _fluence_tag_file_local(x):
        try:
            return str(float(x)).replace(".", "p")
        except Exception:
            return str(x).replace(".", "p")

    def _parse_runs(run_in):
        if run_in is None:
            raise ValueError("runs must be provided.")
        runs_out = []
        if isinstance(run_in, (list, tuple, np.ndarray)):
            for r in list(run_in):
                try:
                    runs_out.append(int(r))
                except Exception:
                    pass
        elif isinstance(run_in, str):
            s = str(run_in).replace(",", " ")
            parts = s.split()
            for p in parts:
                try:
                    runs_out.append(int(p))
                except Exception:
                    pass
        else:
            runs_out = [int(run_in)]

        uniq = sorted(set([int(r) for r in runs_out]))
        if len(uniq) == 0:
            raise ValueError("No valid runs parsed from 'runs' argument.")
        return uniq

    def _dark_tags_from_runs(runs_list):
        rr = [int(x) for x in runs_list]
        if len(rr) == 1:
            folder_tag = "scan_{}".format(int(rr[0]))
            file_tag = "scan{}".format(int(rr[0]))
            return folder_tag, file_tag
        rmin = int(min(rr))
        rmax = int(max(rr))
        folder_tag = "scans_{}-{}".format(rmin, rmax)
        file_tag = "scans{}-{}".format(rmin, rmax)
        return folder_tag, file_tag

    def _ensure_dir(p):
        if not os.path.exists(p):
            os.makedirs(p)

    def _ensure_2d(arr):
        if arr is None:
            return None
        a = np.asarray(arr)
        if hasattr(a, "ndim") and a.ndim == 3:
            return a[0]
        return a

    sn = str(sample_name)
    tK = int(temperature_K)
    wl_tag = _wl_tag_nm_local(excitation_wl_nm)
    flu_tag = _fluence_tag_file_local(fluence_mJ_cm2)
    tw = int(time_window_fs)

    runs_list = _parse_runs(runs)
    dark_folder_tag, dark_file_tag = _dark_tags_from_runs(runs_list)

    delay_root = os.path.join(
        str(analysis_root),
        sn,
        "temperature_{}K".format(int(tK)),
        "excitation_wl_{}nm".format(str(wl_tag)),
        "delay",
        "fluence_{}mJ".format(str(flu_tag)),
        "time_window_{}fs".format(int(tw)),
    )
    img_dir = os.path.join(delay_root, "2D_images")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError("2D_images folder not found: {}".format(img_dir))

    prefix = "{}_{}K_{}nm_{}mJ_{}fs_".format(sn, int(tK), str(wl_tag), str(flu_tag), int(tw))
    suffix = "_laser_off.npy"

    candidates = []
    for fn in os.listdir(img_dir):
        if not fn.endswith(suffix):
            continue
        if not fn.startswith(prefix):
            continue
        fullp = os.path.join(img_dir, fn)
        if os.path.isfile(fullp):
            candidates.append(fullp)

    candidates = sorted(candidates)
    if len(candidates) < int(min_files):
        raise FileNotFoundError(
            "Found {} laser-off files (<{} required) in:\n  {}\nPattern: {}*{}".format(
                int(len(candidates)), int(min_files), img_dir, prefix, suffix
            )
        )

    sum_img = None
    n_used = 0
    used_files = []

    for p in candidates:
        try:
            img = np.load(p)
        except Exception:
            continue

        img = _ensure_2d(img)
        if img is None or (not getattr(img, "size", 0)):
            continue
        if np.isnan(img).any():
            continue

        img = np.asarray(img, dtype=np.float64)

        if sum_img is None:
            sum_img = np.zeros_like(img, dtype=np.float64)

        if sum_img.shape != img.shape:
            continue

        sum_img += img
        n_used += 1
        used_files.append(p)

    if n_used == 0 or sum_img is None:
        raise ValueError("No valid laser-off images could be loaded/used from {}".format(img_dir))

    dark_avg = sum_img / float(n_used)

    dark_dir = os.path.join(
        str(analysis_root),
        sn,
        "temperature_{}K".format(int(tK)),
        "dark",
        str(dark_folder_tag),
        "2D_images",
    )
    _ensure_dir(dark_dir)

    out_name = "{}_{}K_dark_{}.npy".format(sn, int(tK), str(dark_file_tag))
    out_path = os.path.join(dark_dir, out_name)

    if os.path.exists(out_path) and (not bool(overwrite)):
        raise FileExistsError("File exists: {} (set overwrite=True to replace).".format(out_path))

    np.save(out_path, dark_avg)

    print("Laser-OFF representative dark saved: {}".format(out_path))
    print("Used {} files.".format(int(n_used)))

    return {"path": out_path, "n_files": int(n_used), "used_files": used_files}

def create_final_2D_images(
    bl,
    run=None,  # OPTIONAL: only needed to locate dark/calibration metadata if metadata_h5_path is not provided
    scan_type="delay",
    *,
    # --- how to locate metadata (recommended: pass metadata_h5_path)
    metadata_h5_path=None,
    sample_name=None,
    temperature_K=None,
    excitation_wl_nm=None,
    fluence_mJ_cm2=None,
    time_window_fs=None,
    delay_for_fluence=None,
    # --- what to process
    delay=None,          # for delay/delay_off/differentials (fs)
    fluence=None,        # for fluence/fluence_off (mJ/cm^2, or motor-pos if you used that)
    # --- corrections / normalization
    background=None,     # background run number (no-xray), produced by process_background_run()
    intensity_col="xfel_bl_3_st_2_pd_user_9_fitting_peak/voltage",
    threshold_counts=40,
    detector_id="MPCCD-8N0-3-002",
    overwrite=True,
    # --- options ---
    save_laser_off=False,   # if True: also save laser-OFF image for delay and/or differentials
    max_shots="all"):        # "all" or int: cap accepted shots/pairs per output image

    """
    Multi-run aware final 2D image generator using metadata H5 produced by create_metadata().

    Supported requested scan_type (case-insensitive):
      - "delay"          (laser ON)
      - "delay_off"      (laser OFF, using tag+2 partner of ON tags)
      - "differentials"  (ON-OFF, searching OFF partner within +/-8 in steps of 2, within SAME run)
      - "fluence"        (laser ON)
      - "fluence_off"    (laser OFF, using tag+2 partner of ON tags)
      - "calibration"    (treated as "dark")

    IMPORTANT:
      - For delay/fluence families, `run` is NOT needed once metadata exists.
      - For dark/calibration, if metadata_h5_path is not provided, `run` is needed to locate metadata.

    Multi-run behavior:
      - Reads run list from metadata (/meta/scans).
      - Tag selections are treated as (run, tag) pairs.
      - Raw frames are read from the corresponding run.
      - OFF partnering (tag+2 or +/- offsets) is performed within the same run.

    Options:
      - save_laser_off (default False):
          * delay: also save OFF average built from tag+2 partners of accepted ON shots
          * differentials: also save OFF average built from the OFF partners actually used
      - max_shots (default "all"): cap number of accepted shots (or pairs) per output image.
    """

    def _wl_tag_nm_local(x):
        try:
            v = float(x)
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            s = str(v)
            if "." in s:
                s = s.rstrip("0").rstrip(".")
            return s.replace(".", "p")
        except Exception:
            return str(x)

    def _fluence_tag_file_local(x):
        try:
            return str(float(x)).replace(".", "p")
        except Exception:
            return str(x).replace(".", "p")

    def _ensure_dir(p):
        if not os.path.exists(p):
            os.makedirs(p)

    def _load_background_img(background_run):
        if background_run is None:
            return None
        p = os.path.join(
            PATH_ANALYSIS_FOLDER,
            str(int(background_run)),
            "{}.npy".format(int(background_run))
        )
        if not os.path.exists(p):
            raise FileNotFoundError("Background npy file not found: {}".format(p))
        b = np.load(p)
        if hasattr(b, "ndim") and b.ndim == 3:
            b = b[0]
        return np.asarray(b, dtype=float)

    def _read_str(v):
        try:
            if hasattr(v, "decode"):
                return v.decode("utf-8")
        except Exception:
            pass
        return str(v)

    def _parse_runs(run_in):
        if run_in is None:
            return []
        runs_out = []
        if isinstance(run_in, (list, tuple, np.ndarray)):
            for r in list(run_in):
                try:
                    runs_out.append(int(r))
                except Exception:
                    pass
        elif isinstance(run_in, str):
            s = str(run_in).replace(",", " ")
            parts = s.split()
            for p in parts:
                try:
                    runs_out.append(int(p))
                except Exception:
                    pass
        else:
            try:
                runs_out = [int(run_in)]
            except Exception:
                runs_out = []

        seen = set()
        uniq = []
        for r in runs_out:
            if r not in seen:
                uniq.append(int(r))
                seen.add(r)
        return uniq

    def _runs_tag(runs_list):
        rr = [int(x) for x in runs_list]
        if len(rr) == 1:
            return str(rr[0])
        if len(rr) <= 5:
            return "_".join([str(x) for x in rr])
        return "N{}_{}-{}".format(int(len(rr)), int(min(rr)), int(max(rr)))

    def _analysis_dir_from_args(st_meta, runs_list_for_dark):
        if sample_name is None or temperature_K is None:
            raise ValueError("Provide metadata_h5_path or (sample_name, temperature_K, ...).")

        if st_meta == "dark":
            if runs_list_for_dark is None or len(runs_list_for_dark) == 0:
                raise ValueError("For scan_type='calibration' (dark), provide run (or metadata_h5_path).")
            if len(runs_list_for_dark) == 1:
                return os.path.join(
                    PATH_ANALYSIS_FOLDER,
                    str(sample_name),
                    "temperature_{}K".format(int(temperature_K)),
                    "dark",
                    "scan_{}".format(int(runs_list_for_dark[0])),
                )
            return os.path.join(
                PATH_ANALYSIS_FOLDER,
                str(sample_name),
                "temperature_{}K".format(int(temperature_K)),
                "dark",
                "scans_{}".format(str(_runs_tag(runs_list_for_dark))),
            )

        if excitation_wl_nm is None:
            raise ValueError("excitation_wl_nm must be provided for scan_type != 'dark' when metadata_h5_path is None.")
        if time_window_fs is None:
            raise ValueError("time_window_fs must be provided when metadata_h5_path is None.")

        wl_tag = _wl_tag_nm_local(excitation_wl_nm)

        if st_meta == "delay":
            if fluence_mJ_cm2 is None:
                raise ValueError("fluence_mJ_cm2 must be provided for delay-family when metadata_h5_path is None.")
            flu_folder = _fluence_tag_file_local(fluence_mJ_cm2) + "mJ"
            return os.path.join(
                PATH_ANALYSIS_FOLDER,
                str(sample_name),
                "temperature_{}K".format(int(temperature_K)),
                "excitation_wl_{}nm".format(str(wl_tag)),
                "delay",
                "fluence_{}".format(str(flu_folder)),
                "time_window_{}fs".format(int(time_window_fs)),
            )

        if st_meta == "fluence":
            if delay_for_fluence is None:
                raise ValueError("delay_for_fluence must be provided for fluence-family when metadata_h5_path is None.")
            return os.path.join(
                PATH_ANALYSIS_FOLDER,
                str(sample_name),
                "temperature_{}K".format(int(temperature_K)),
                "excitation_wl_{}nm".format(str(wl_tag)),
                "fluence",
                "delay_{}fs".format(int(delay_for_fluence)),
                "time_window_{}fs".format(int(time_window_fs)),
            )

        raise ValueError("Unsupported st_meta '{}'".format(st_meta))

    def _default_metadata_path(st_meta, analysis_dir, runs_list_for_dark):
        meta_dir = os.path.join(analysis_dir, "metadata")

        if st_meta == "dark":
            if runs_list_for_dark is None or len(runs_list_for_dark) == 0:
                raise ValueError("For scan_type='calibration' (dark), provide run (or metadata_h5_path).")
            if len(runs_list_for_dark) == 1:
                base = "{}_{}K_dark_scan{}.h5".format(str(sample_name), int(temperature_K), int(runs_list_for_dark[0]))
            else:
                base = "{}_{}K_dark_scans{}.h5".format(str(sample_name), int(temperature_K), str(_runs_tag(runs_list_for_dark)))
            return os.path.join(meta_dir, base)

        if st_meta == "delay":
            wl_tag = _wl_tag_nm_local(excitation_wl_nm)
            flu_folder = _fluence_tag_file_local(fluence_mJ_cm2) + "mJ"
            base = "{}_{}K_{}nm_{}_{}fs.h5".format(
                str(sample_name), int(temperature_K), str(wl_tag), str(flu_folder), int(time_window_fs)
            )
            return os.path.join(meta_dir, base)

        wl_tag = _wl_tag_nm_local(excitation_wl_nm)
        base = "{}_{}K_{}nm_{}fs_{}fs.h5".format(
            str(sample_name), int(temperature_K), str(wl_tag), int(delay_for_fluence), int(time_window_fs)
        )
        return os.path.join(meta_dir, base)

    def _ensure_2d(img):
        if img is None:
            return None
        a = np.asarray(img)
        if hasattr(a, "ndim") and a.ndim == 3:
            return a[0]
        return a

    # Normalize scan type
    st_req = str(scan_type).strip().lower()
    if st_req == "calibration":
        st_req = "dark"

    req_is_delay = st_req in ("delay", "delay_off", "differentials")
    req_is_flu = st_req in ("fluence", "fluence_off")
    req_is_dark = (st_req == "dark")

    # Parse max_shots
    max_shots_n = None
    try:
        if max_shots is None:
            max_shots_n = None
        elif isinstance(max_shots, str):
            ms = str(max_shots).strip().lower()
            if ms != "all" and ms != "":
                max_shots_n = int(ms)
        else:
            max_shots_n = int(max_shots)
    except Exception:
        max_shots_n = None
    if max_shots_n is not None:
        try:
            if int(max_shots_n) <= 0:
                max_shots_n = None
        except Exception:
            max_shots_n = None

    # Locate metadata
    runs_for_dark = None
    if req_is_dark and metadata_h5_path is None:
        runs_for_dark = _parse_runs(run)
        if runs_for_dark is None or len(runs_for_dark) == 0:
            raise ValueError("For scan_type='calibration' (dark), provide run (or metadata_h5_path).")

    if metadata_h5_path is None:
        if req_is_delay:
            st_meta = "delay"
        elif req_is_flu:
            st_meta = "fluence"
        elif req_is_dark:
            st_meta = "dark"
        else:
            raise ValueError("Unsupported scan_type '{}'".format(st_req))

        analysis_dir = _analysis_dir_from_args(st_meta, runs_for_dark)
        metadata_h5_path = _default_metadata_path(st_meta, analysis_dir, runs_for_dark)
    else:
        metadata_h5_path = str(metadata_h5_path)
        if not os.path.exists(metadata_h5_path):
            raise FileNotFoundError("Metadata H5 not found: {}".format(metadata_h5_path))
        analysis_dir = os.path.dirname(os.path.dirname(os.path.abspath(metadata_h5_path)))

    if not os.path.exists(metadata_h5_path):
        raise FileNotFoundError("Metadata H5 not found: {}".format(metadata_h5_path))

    # Read meta + runs
    with h5py.File(metadata_h5_path, "r") as hdf:
        if "meta" not in hdf:
            raise ValueError("Invalid metadata H5 (missing /meta): {}".format(metadata_h5_path))
        gmeta = hdf["meta"]

        sample_name_m = _read_str(gmeta["sample_name"][()])
        temperature_K_m = int(gmeta["temperature_K"][()])
        scan_type_m = _read_str(gmeta["scan_type"][()])
        time_window_fs_m = int(gmeta["time_window_fs"][()])

        st_meta_file = str(scan_type_m).strip().lower()
        if st_meta_file == "calibration":
            st_meta_file = "dark"

        if "scans" in gmeta:
            runs_meta = np.array(gmeta["scans"], dtype=np.int64).tolist()
            runs_meta = [int(x) for x in runs_meta]
        else:
            if "run" in gmeta:
                runs_meta = [int(gmeta["run"][()])]
            else:
                runs_meta = []

        if len(runs_meta) == 0:
            raise ValueError("Metadata does not contain /meta/scans (and no legacy /meta/run).")

        wl_nm_m = None
        if "excitation_wl_nm" in gmeta:
            try:
                wl_nm_m = float(gmeta["excitation_wl_nm"][()])
            except Exception:
                wl_nm_m = None

        flu_mJ_m = None
        if "fluence_mJ_cm2" in gmeta:
            try:
                flu_mJ_m = float(gmeta["fluence_mJ_cm2"][()])
            except Exception:
                flu_mJ_m = None

        delay_fixed_m = None
        if "delay_fs" in gmeta:
            try:
                delay_fixed_m = int(gmeta["delay_fs"][()])
            except Exception:
                delay_fixed_m = None

    if req_is_delay and st_meta_file != "delay":
        raise ValueError("Requested scan_type '{}' requires delay metadata, but file scan_type is '{}'.".format(st_req, st_meta_file))
    if req_is_flu and st_meta_file != "fluence":
        raise ValueError("Requested scan_type '{}' requires fluence metadata, but file scan_type is '{}'.".format(st_req, st_meta_file))
    if req_is_dark and st_meta_file != "dark":
        raise ValueError("Requested scan_type '{}' requires dark metadata, but file scan_type is '{}'.".format(st_req, st_meta_file))

    # Background
    bckg_img = _load_background_img(background)
    if bckg_img is not None:
        bckg_img = _ensure_2d(bckg_img).astype(float)

    # Intensity maps per (run, tag)
    def _load_intensity_map_multi(st_meta_kind):
        if st_meta_kind not in ("delay", "fluence"):
            return {}
        out = {}
        for r in runs_meta:
            meta_df = read_metadata(int(bl), int(r), scan_type=st_meta_kind)
            if meta_df is None or len(meta_df) == 0:
                continue
            if intensity_col not in meta_df.columns:
                raise KeyError("Intensity column '{}' not found in metadata dataframe (run {}).".format(intensity_col, int(r)))
            try:
                tags = meta_df["tag_number"].values
                ints = meta_df[intensity_col].values
                for i in range(len(tags)):
                    try:
                        out[(int(r), int(tags[i]))] = float(ints[i])
                    except Exception:
                        pass
            except Exception:
                for _, row in meta_df.iterrows():
                    try:
                        out[(int(r), int(row["tag_number"]))] = float(row[intensity_col])
                    except Exception:
                        pass
        return out

    intensity_map = _load_intensity_map_multi(st_meta_file)

    # OFF intensity (hightag per run)
    hightag_cache = {}

    def _get_hightag_for_run(r):
        r = int(r)
        if r in hightag_cache:
            return hightag_cache[r]
        try:
            ht = dbpy.read_hightagnumber(int(bl), int(r))
        except Exception:
            ht = None
        hightag_cache[r] = ht
        return ht

    def _read_off_intensity(r, tag):
        ht = _get_hightag_for_run(r)
        if ht is None:
            return None
        try:
            arr = np.array(dbpy.read_syncdatalist_float(intensity_col, ht, (int(tag),)))
            if arr.size == 0:
                return None
            return float(arr[0])
        except Exception:
            return None

    # Readers per run
    reader_cache = {}

    def _get_reader_for_run(r):
        r = int(r)
        if r in reader_cache:
            return reader_cache[r]
        collecter = stpy.StorageReader(str(detector_id), int(bl), (int(r),))
        buffer = stpy.StorageBuffer(collecter)
        reader_cache[r] = (collecter, buffer)
        return collecter, buffer

    def _get_img(r, tag):
        collecter, buffer = _get_reader_for_run(r)
        return get_2D_img_per_tag(collecter, buffer, int(tag))

    # Load (run,tag) lists from H5
    def _load_shots_from_scans_group(scans_g):
        shot_run_list = []
        shot_tag_list = []
        keys = []
        for k in scans_g.keys():
            try:
                keys.append(int(k))
            except Exception:
                pass
        keys.sort()
        for r in keys:
            rk = str(int(r))
            if rk not in scans_g:
                continue
            tags = np.array(scans_g[rk]["tags"], dtype=np.int64)
            if tags.size == 0:
                continue
            shot_run_list.append(np.full(int(tags.size), int(r), dtype=np.int64))
            shot_tag_list.append(tags.astype(np.int64))
        if len(shot_run_list) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return np.concatenate(shot_run_list), np.concatenate(shot_tag_list)

    def _load_delay_shots(hdf, delay_fs):
        dgrp_name = "{}fs".format(int(delay_fs))
        if "delays" not in hdf or dgrp_name not in hdf["delays"]:
            raise KeyError("Delay group not found: /delays/{}".format(dgrp_name))
        dg = hdf["delays"][dgrp_name]
        if "shot_run" in dg and "shot_tag" in dg:
            return np.array(dg["shot_run"], dtype=np.int64), np.array(dg["shot_tag"], dtype=np.int64)
        return _load_shots_from_scans_group(dg["scans"])

    def _load_fluence_shots(hdf, flu_tag_str):
        grp_name = "{}mJ".format(str(flu_tag_str))
        if "fluences" not in hdf or grp_name not in hdf["fluences"]:
            raise KeyError("Fluence group not found: /fluences/{}".format(grp_name))
        fg = hdf["fluences"][grp_name]
        if "shot_run" in fg and "shot_tag" in fg:
            return np.array(fg["shot_run"], dtype=np.int64), np.array(fg["shot_tag"], dtype=np.int64)
        return _load_shots_from_scans_group(fg["scans"])

    def _load_dark_shots(hdf):
        if "scans" not in hdf:
            raise ValueError("Invalid dark metadata H5 (missing /scans).")
        gscans = hdf["scans"]
        if "shot_run" in gscans and "shot_tag" in gscans:
            return np.array(gscans["shot_run"], dtype=np.int64), np.array(gscans["shot_tag"], dtype=np.int64)
        return _load_shots_from_scans_group(gscans)

    # Output directory
    out_dir = os.path.join(analysis_dir, "2D_images")
    _ensure_dir(out_dir)

    with h5py.File(metadata_h5_path, "r") as hdf:
        # DARK
        if st_req == "dark":
            shot_run, shot_tag = _load_dark_shots(hdf)
            if shot_tag.size == 0:
                raise ValueError("No tags found in dark metadata.")

            final_sum = None
            count = 0

            for i in range(int(shot_tag.size)):
                r = int(shot_run[i])
                t = int(shot_tag[i])
                print("Processing run {}, tag {}, {} out of {}".format(r, t, i + 1, int(shot_tag.size)))

                img = _get_img(r, t)
                img = _ensure_2d(img)
                if img is None or (not getattr(img, "size", 0)) or np.isnan(img).any():
                    continue

                img = img.astype(float)
                if bckg_img is not None:
                    img = img - bckg_img
                img[img < float(threshold_counts)] = 0.0

                if final_sum is None:
                    final_sum = np.zeros_like(img, dtype=np.float64)
                final_sum += img
                count += 1

                if max_shots_n is not None and int(count) >= int(max_shots_n):
                    print("Reached max_shots={} for dark. Stopping early.".format(int(max_shots_n)))
                    break

            if count == 0 or final_sum is None:
                raise ValueError("No valid images found for dark metadata.")

            img_avg = final_sum / float(count)

            if len(runs_meta) == 1:
                out_name = "{}_{}K_dark_scan{}.npy".format(str(sample_name_m), int(temperature_K_m), int(runs_meta[0]))
            else:
                out_name = "{}_{}K_dark_scans{}.npy".format(str(sample_name_m), int(temperature_K_m), str(_runs_tag(runs_meta)))
            out_path = os.path.join(out_dir, out_name)

            if os.path.exists(out_path) and (not bool(overwrite)):
                raise FileExistsError("File exists: {}".format(out_path))

            np.save(out_path, img_avg)
            print("Final dark image saved: {}".format(out_path))
            return {"path": out_path, "n_images": int(count)}

        # DELAY FAMILY
        if req_is_delay:
            if delay is None:
                raise ValueError("For scan_type '{}', provide delay=<fs>.".format(st_req))
            delay_fs = int(delay)

            if wl_nm_m is None or flu_mJ_m is None:
                raise ValueError("Delay metadata missing excitation_wl_nm and/or fluence_mJ_cm2.")

            wl_tag = _wl_tag_nm_local(wl_nm_m)
            flu_file = _fluence_tag_file_local(flu_mJ_m)
            tw = int(time_window_fs_m)

            shot_run, shot_tag = _load_delay_shots(hdf, delay_fs)
            if shot_tag.size == 0:
                raise ValueError("No tags found for delay {} fs.".format(int(delay_fs)))

            final_sum = None
            final_sum_off = None
            count = 0
            count_off = 0

            candidate_offsets = []
            if st_req == "differentials":
                max_neighbor = 8
                for s in range(2, max_neighbor + 1, 2):
                    candidate_offsets.append(int(s))
                    candidate_offsets.append(int(-s))

            for i in range(int(shot_tag.size)):
                r_on = int(shot_run[i])
                tag_on = int(shot_tag[i])
                print("Processing run {}, tag {}, {} out of {}".format(r_on, tag_on, i + 1, int(shot_tag.size)))

                img_on = _get_img(r_on, tag_on)
                img_on = _ensure_2d(img_on)
                if img_on is None or (not getattr(img_on, "size", 0)) or np.isnan(img_on).any():
                    continue

                I_on = intensity_map.get((int(r_on), int(tag_on)), None)
                try:
                    I_on = float(I_on)
                except Exception:
                    I_on = None
                if I_on is None or (not np.isfinite(I_on)) or I_on == 0:
                    continue

                img_on = img_on.astype(float)
                if bckg_img is not None:
                    img_on = img_on - bckg_img
                img_on[img_on < float(threshold_counts)] = 0.0
                img_on = img_on / float(I_on)
                if np.isnan(img_on).any():
                    continue

                if st_req == "delay":
                    if final_sum is None:
                        final_sum = np.zeros_like(img_on, dtype=np.float64)
                    final_sum += img_on
                    count += 1

                    if bool(save_laser_off):
                        tag_off = int(tag_on) + 2
                        img_off = _get_img(r_on, tag_off)
                        img_off = _ensure_2d(img_off)
                        if img_off is not None and getattr(img_off, "size", 0) and (not np.isnan(img_off).any()):
                            I_off = _read_off_intensity(r_on, tag_off)
                            if I_off is not None and np.isfinite(I_off) and float(I_off) != 0.0:
                                img_off = img_off.astype(float)
                                if bckg_img is not None:
                                    img_off = img_off - bckg_img
                                img_off[img_off < float(threshold_counts)] = 0.0
                                img_off = img_off / float(I_off)
                                if not np.isnan(img_off).any():
                                    if final_sum_off is None:
                                        final_sum_off = np.zeros_like(img_off, dtype=np.float64)
                                    final_sum_off += img_off
                                    count_off += 1

                    if max_shots_n is not None and int(count) >= int(max_shots_n):
                        print("Reached max_shots={} for delay. Stopping early.".format(int(max_shots_n)))
                        break
                    continue

                if st_req == "delay_off":
                    tag_off = int(tag_on) + 2
                    img_off = _get_img(r_on, tag_off)
                    img_off = _ensure_2d(img_off)
                    if img_off is None or (not getattr(img_off, "size", 0)) or np.isnan(img_off).any():
                        continue

                    I_off = _read_off_intensity(r_on, tag_off)
                    if I_off is None or (not np.isfinite(I_off)) or I_off == 0:
                        continue

                    img_off = img_off.astype(float)
                    if bckg_img is not None:
                        img_off = img_off - bckg_img
                    img_off[img_off < float(threshold_counts)] = 0.0
                    img_off = img_off / float(I_off)
                    if np.isnan(img_off).any():
                        continue

                    if final_sum is None:
                        final_sum = np.zeros_like(img_off, dtype=np.float64)
                    final_sum += img_off
                    count += 1

                    if max_shots_n is not None and int(count) >= int(max_shots_n):
                        print("Reached max_shots={} for delay_off. Stopping early.".format(int(max_shots_n)))
                        break
                    continue

                used_pair = False
                for off_shift in candidate_offsets:
                    tag_off = int(tag_on) + int(off_shift)

                    img_off = _get_img(r_on, tag_off)
                    img_off = _ensure_2d(img_off)
                    if img_off is None or (not getattr(img_off, "size", 0)) or np.isnan(img_off).any():
                        continue

                    I_off = _read_off_intensity(r_on, tag_off)
                    if I_off is None or (not np.isfinite(I_off)) or I_off == 0:
                        continue

                    img_off = img_off.astype(float)
                    if bckg_img is not None:
                        img_off = img_off - bckg_img
                    img_off[img_off < float(threshold_counts)] = 0.0
                    img_off = img_off / float(I_off)
                    if np.isnan(img_off).any():
                        continue

                    diff_img = img_on - img_off
                    if np.isnan(diff_img).any():
                        continue

                    if final_sum is None:
                        final_sum = np.zeros_like(diff_img, dtype=np.float64)
                        if bool(save_laser_off):
                            final_sum_off = np.zeros_like(img_off, dtype=np.float64)

                    final_sum += diff_img
                    if bool(save_laser_off) and final_sum_off is not None:
                        final_sum_off += img_off
                    count += 1
                    used_pair = True
                    break

                if not used_pair:
                    continue

                if max_shots_n is not None and int(count) >= int(max_shots_n):
                    print("Reached max_shots={} for differentials. Stopping early.".format(int(max_shots_n)))
                    break

            if count == 0 or final_sum is None:
                raise ValueError("No valid images found for scan_type '{}' at delay {} fs.".format(st_req, int(delay_fs)))

            img_avg = final_sum / float(count)

            base = "{}_{}K_{}nm_{}mJ_{}fs_{}fs".format(
                str(sample_name_m),
                int(temperature_K_m),
                str(wl_tag),
                str(flu_file),
                int(tw),
                int(delay_fs),
            )

            if st_req == "delay":
                out_path = os.path.join(out_dir, "{}.npy".format(base))
            elif st_req == "delay_off":
                out_path = os.path.join(out_dir, "{}_laser_off.npy".format(base))
            else:
                out_path = os.path.join(out_dir, "{}_diff.npy".format(base))

            if os.path.exists(out_path) and (not bool(overwrite)):
                raise FileExistsError("File exists: {}".format(out_path))

            np.save(out_path, img_avg)
            print("Final image saved: {}".format(out_path))

            ret = {"path": out_path, "n_images": int(count)}

            if bool(save_laser_off) and final_sum_off is not None:
                off_path = os.path.join(out_dir, "{}_laser_off.npy".format(base))

                if st_req == "delay":
                    if count_off > 0:
                        img_off_avg = final_sum_off / float(count_off)
                        if (not os.path.exists(off_path)) or bool(overwrite):
                            np.save(off_path, img_off_avg)
                            print("Laser OFF image saved: {}".format(off_path))
                        ret["laser_off_path"] = off_path
                        ret["n_images_off"] = int(count_off)
                    else:
                        print("save_laser_off=True but no valid OFF partners found for delay {} fs.".format(int(delay_fs)))

                elif st_req == "differentials":
                    img_off_avg = final_sum_off / float(count)
                    if (not os.path.exists(off_path)) or bool(overwrite):
                        np.save(off_path, img_off_avg)
                        print("Representative OFF image saved: {}".format(off_path))
                    ret["laser_off_path"] = off_path

            return ret

        # FLUENCE FAMILY
        if req_is_flu:
            if fluence is None:
                raise ValueError("For scan_type '{}', provide fluence=<mJ/cm^2 or motor-pos>.".format(st_req))
            if wl_nm_m is None or delay_fixed_m is None:
                raise ValueError("Fluence metadata missing excitation_wl_nm and/or delay_fs.")

            flu_val = float(fluence)
            flu_tag = _fluence_tag_file_local(flu_val)

            shot_run, shot_tag = _load_fluence_shots(hdf, flu_tag)
            if shot_tag.size == 0:
                raise ValueError("No tags found for fluence {}.".format(fluence))

            wl_tag = _wl_tag_nm_local(wl_nm_m)
            tw = int(time_window_fs_m)
            dly = int(delay_fixed_m)

            final_sum = None
            count = 0

            for i in range(int(shot_tag.size)):
                r_on = int(shot_run[i])
                tag_on = int(shot_tag[i])
                print("Processing run {}, tag {}, {} out of {}".format(r_on, tag_on, i + 1, int(shot_tag.size)))

                if st_req == "fluence":
                    img = _get_img(r_on, tag_on)
                    img = _ensure_2d(img)
                    if img is None or (not getattr(img, "size", 0)) or np.isnan(img).any():
                        continue

                    I_on = intensity_map.get((int(r_on), int(tag_on)), None)
                    try:
                        I_on = float(I_on)
                    except Exception:
                        I_on = None
                    if I_on is None or (not np.isfinite(I_on)) or I_on == 0:
                        continue

                    img = img.astype(float)
                    if bckg_img is not None:
                        img = img - bckg_img
                    img[img < float(threshold_counts)] = 0.0
                    img = img / float(I_on)
                    if np.isnan(img).any():
                        continue

                    if final_sum is None:
                        final_sum = np.zeros_like(img, dtype=np.float64)
                    final_sum += img
                    count += 1

                    if max_shots_n is not None and int(count) >= int(max_shots_n):
                        print("Reached max_shots={} for fluence. Stopping early.".format(int(max_shots_n)))
                        break
                    continue

                tag_off = int(tag_on) + 2
                img_off = _get_img(r_on, tag_off)
                img_off = _ensure_2d(img_off)
                if img_off is None or (not getattr(img_off, "size", 0)) or np.isnan(img_off).any():
                    continue

                I_off = _read_off_intensity(r_on, tag_off)
                if I_off is None or (not np.isfinite(I_off)) or I_off == 0:
                    continue

                img_off = img_off.astype(float)
                if bckg_img is not None:
                    img_off = img_off - bckg_img
                img_off[img_off < float(threshold_counts)] = 0.0
                img_off = img_off / float(I_off)
                if np.isnan(img_off).any():
                    continue

                if final_sum is None:
                    final_sum = np.zeros_like(img_off, dtype=np.float64)
                final_sum += img_off
                count += 1

                if max_shots_n is not None and int(count) >= int(max_shots_n):
                    print("Reached max_shots={} for fluence_off. Stopping early.".format(int(max_shots_n)))
                    break

            if count == 0 or final_sum is None:
                raise ValueError("No valid images found for scan_type '{}' at fluence {}.".format(st_req, fluence))

            img_avg = final_sum / float(count)

            base = "{}_{}K_{}nm_{}mJ_{}fs_{}fs".format(
                str(sample_name_m),
                int(temperature_K_m),
                str(wl_tag),
                str(flu_tag),
                int(tw),
                int(dly),
            )

            if st_req == "fluence":
                out_path = os.path.join(out_dir, "{}.npy".format(base))
            else:
                out_path = os.path.join(out_dir, "{}_laser_off.npy".format(base))

            if os.path.exists(out_path) and (not bool(overwrite)):
                raise FileExistsError("File exists: {}".format(out_path))

            np.save(out_path, img_avg)
            print("Final fluence image saved: {}".format(out_path))
            return {"path": out_path, "n_images": int(count)}

    raise ValueError("Unsupported scan_type '{}'.".format(st_req))

def process_one_chunk(
    bl,
    run=None,  # OPTIONAL for delay/fluence; needed only to locate dark metadata if metadata_h5_path is None
    chunk_id=1,
    n_chunks=200,
    *,
    scan_type="delay",
    background=None,
    # --- dataset identity (needed if metadata_h5_path is not given)
    sample_name=None,
    temperature_K=None,
    excitation_wl_nm=None,
    fluence_mJ_cm2=None,
    time_window_fs=None,
    delay_for_fluence=None,
    # --- direct override
    metadata_h5_path=None,
    # --- forwarding
    overwrite=True,
    # --- options forwarded to create_final_2D_images ---
    save_laser_off=False,
    max_shots="all"):
    """
    Chunk dispatcher (multi-run metadata compatible).

    For scan_type in:
      - ["delay", "delay_off", "differentials"]:
          splits over /meta/selected_delays_fs
          calls create_final_2D_images(..., delay=<fs>)
          NOTE: run is NOT required (metadata path has no run)
      - ["fluence", "fluence_off"]:
          splits over /meta/fluences_mJ_cm2 (fallback /meta/fluences_motor_pos)
          calls create_final_2D_images(..., fluence=<mJ/cm^2 or motor-pos>)
          NOTE: run is NOT required (metadata path has no run)
      - ["calibration", "dark"]:
          no chunking needed; only chunk_id==1 runs; calls create_final_2D_images once
          NOTE: if metadata_h5_path is None, run IS required to locate dark metadata

    Notes:
      - chunk_id is 1-indexed (PBS array convention).
      - Uses .format() only.
    """

    def _wl_tag_nm_local(x):
        try:
            v = float(x)
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            s = str(v)
            if "." in s:
                s = s.rstrip("0").rstrip(".")
            return s.replace(".", "p")
        except Exception:
            return str(x)

    def _fluence_tag_file_local(x):
        try:
            return str(float(x)).replace(".", "p")
        except Exception:
            return str(x).replace(".", "p")

    def _parse_runs(run_in):
        if run_in is None:
            return []
        runs_out = []
        if isinstance(run_in, (list, tuple, np.ndarray)):
            for r in list(run_in):
                try:
                    runs_out.append(int(r))
                except Exception:
                    pass
        elif isinstance(run_in, str):
            s = str(run_in).replace(",", " ")
            parts = s.split()
            for p in parts:
                try:
                    runs_out.append(int(p))
                except Exception:
                    pass
        else:
            try:
                runs_out = [int(run_in)]
            except Exception:
                runs_out = []

        seen = set()
        uniq = []
        for r in runs_out:
            if r not in seen:
                uniq.append(int(r))
                seen.add(r)
        return uniq

    def _runs_tag(runs_list):
        rr = [int(x) for x in runs_list]
        if len(rr) == 1:
            return str(rr[0])
        if len(rr) <= 5:
            return "_".join([str(x) for x in rr])
        return "N{}_{}-{}".format(int(len(rr)), int(min(rr)), int(max(rr)))

    def _analysis_dir_delay():
        if sample_name is None or temperature_K is None or excitation_wl_nm is None or time_window_fs is None or fluence_mJ_cm2 is None:
            raise ValueError("Missing required args to locate delay metadata (sample_name, temperature_K, excitation_wl_nm, fluence_mJ_cm2, time_window_fs).")
        wl_tag = _wl_tag_nm_local(excitation_wl_nm)
        flu_folder = _fluence_tag_file_local(fluence_mJ_cm2) + "mJ"
        return os.path.join(
            PATH_ANALYSIS_FOLDER,
            str(sample_name),
            "temperature_{}K".format(int(temperature_K)),
            "excitation_wl_{}nm".format(str(wl_tag)),
            "delay",
            "fluence_{}".format(str(flu_folder)),
            "time_window_{}fs".format(int(time_window_fs)),
        )

    def _analysis_dir_fluence():
        if sample_name is None or temperature_K is None or excitation_wl_nm is None or time_window_fs is None or delay_for_fluence is None:
            raise ValueError("Missing required args to locate fluence metadata (sample_name, temperature_K, excitation_wl_nm, delay_for_fluence, time_window_fs).")
        wl_tag = _wl_tag_nm_local(excitation_wl_nm)
        return os.path.join(
            PATH_ANALYSIS_FOLDER,
            str(sample_name),
            "temperature_{}K".format(int(temperature_K)),
            "excitation_wl_{}nm".format(str(wl_tag)),
            "fluence",
            "delay_{}fs".format(int(delay_for_fluence)),
            "time_window_{}fs".format(int(time_window_fs)),
        )

    def _analysis_dir_dark(runs_list):
        if sample_name is None or temperature_K is None:
            raise ValueError("Missing required args to locate dark metadata (sample_name, temperature_K).")
        if runs_list is None or len(runs_list) == 0:
            raise ValueError("For dark/calibration final_imgs, provide run (or metadata_h5_path).")
        if len(runs_list) == 1:
            return os.path.join(
                PATH_ANALYSIS_FOLDER,
                str(sample_name),
                "temperature_{}K".format(int(temperature_K)),
                "dark",
                "scan_{}".format(int(runs_list[0])),
            )
        return os.path.join(
            PATH_ANALYSIS_FOLDER,
            str(sample_name),
            "temperature_{}K".format(int(temperature_K)),
            "dark",
            "scans_{}".format(str(_runs_tag(runs_list))),
        )

    def _metadata_path_delay(analysis_dir):
        wl_tag = _wl_tag_nm_local(excitation_wl_nm)
        flu_folder = _fluence_tag_file_local(fluence_mJ_cm2) + "mJ"
        base = "{}_{}K_{}nm_{}_{}fs.h5".format(
            str(sample_name), int(temperature_K), str(wl_tag), str(flu_folder), int(time_window_fs)
        )
        return os.path.join(analysis_dir, "metadata", base)

    def _metadata_path_fluence(analysis_dir):
        wl_tag = _wl_tag_nm_local(excitation_wl_nm)
        base = "{}_{}K_{}nm_{}fs_{}fs.h5".format(
            str(sample_name), int(temperature_K), str(wl_tag), int(delay_for_fluence), int(time_window_fs)
        )
        return os.path.join(analysis_dir, "metadata", base)

    def _metadata_path_dark(analysis_dir, runs_list):
        if len(runs_list) == 1:
            base = "{}_{}K_dark_scan{}.h5".format(str(sample_name), int(temperature_K), int(runs_list[0]))
        else:
            base = "{}_{}K_dark_scans{}.h5".format(str(sample_name), int(temperature_K), str(_runs_tag(runs_list)))
        return os.path.join(analysis_dir, "metadata", base)

    st = str(scan_type).strip().lower()
    if st == "calibration":
        st = "dark"

    is_delay_family = st in ("delay", "delay_off", "differentials")
    is_flu_family = st in ("fluence", "fluence_off")
    is_dark = (st == "dark")

    runs_list = _parse_runs(run)  # only used for dark path reconstruction

    # ---- locate metadata file ----
    if metadata_h5_path is None:
        if is_delay_family:
            analysis_dir = _analysis_dir_delay()
            metadata_h5_path = _metadata_path_delay(analysis_dir)
        elif is_flu_family:
            analysis_dir = _analysis_dir_fluence()
            metadata_h5_path = _metadata_path_fluence(analysis_dir)
        elif is_dark:
            analysis_dir = _analysis_dir_dark(runs_list)
            metadata_h5_path = _metadata_path_dark(analysis_dir, runs_list)
        else:
            raise ValueError("Unsupported scan_type '{}'".format(st))
    else:
        metadata_h5_path = str(metadata_h5_path)

    if not os.path.exists(metadata_h5_path):
        raise FileNotFoundError("Metadata H5 not found: {}".format(metadata_h5_path))

    # ---- dark: only chunk 1 does work ----
    if is_dark:
        if int(chunk_id) != 1:
            print("Dark scan: only chunk_id=1 runs. Skipping chunk_id={}.".format(int(chunk_id)))
            return

        print("PROCESSING DARK (single job): metadata={}".format(metadata_h5_path))
        return create_final_2D_images(
            bl=bl,
            run=run,  # only relevant if metadata_h5_path was None; safe otherwise
            scan_type="calibration",
            metadata_h5_path=metadata_h5_path,
            background=background,
            overwrite=overwrite,
            save_laser_off=bool(save_laser_off),
            max_shots=max_shots,
        )

    # ---- read bin list from metadata ----
    with h5py.File(metadata_h5_path, "r") as hdf:
        if "meta" not in hdf:
            raise ValueError("Invalid metadata H5 (missing /meta): {}".format(metadata_h5_path))

        gmeta = hdf["meta"]

        if is_delay_family:
            if "selected_delays_fs" not in gmeta:
                raise ValueError("Invalid delay metadata H5 (missing /meta/selected_delays_fs).")
            tacos = np.array(gmeta["selected_delays_fs"], dtype=np.int64)
            tacs = "Delays"
            tac = "delay"

        elif is_flu_family:
            if "fluences_mJ_cm2" in gmeta:
                tacos = np.array(gmeta["fluences_mJ_cm2"], dtype=float)
            else:
                if "fluences_motor_pos" not in gmeta:
                    raise ValueError("Invalid fluence metadata H5 (missing /meta/fluences_mJ_cm2 or /meta/fluences_motor_pos).")
                tacos = np.array(gmeta["fluences_motor_pos"], dtype=float)
            tacs = "Fluences"
            tac = "fluence"

        else:
            raise ValueError("Unsupported scan_type '{}'".format(st))

    if tacos.size == 0:
        print("No {} found in metadata. Exiting.".format(tacs))
        return

    splitted_tacos = np.array_split(tacos, int(n_chunks))
    chunk_index = int(chunk_id) - 1
    if chunk_index < 0 or chunk_index >= len(splitted_tacos):
        print("Chunk index out of range. Exiting.")
        return

    chunk = splitted_tacos[chunk_index]
    print("PROCESSING CHUNK: {} OF {}".format(chunk_index + 1, int(n_chunks)))
    print("{} IN THIS CHUNK:".format(tacs), chunk)

    # ---- process each item ----
    for k, taco_val in enumerate(chunk):
        print("\n{} {}s done out of {}\n".format(k, tac, len(chunk)))
        print("Processing {}: {}".format(tac, taco_val))

        if is_delay_family:
            create_final_2D_images(
                bl=bl,
                run=None,  # not needed for delay-family final images
                scan_type=st,
                metadata_h5_path=metadata_h5_path,
                delay=int(taco_val),
                time_window_fs=time_window_fs,  # optional
                background=background,
                overwrite=overwrite,
                save_laser_off=bool(save_laser_off),
                max_shots=max_shots,
            )
        else:
            create_final_2D_images(
                bl=bl,
                run=None,  # not needed for fluence-family final images
                scan_type=st,
                metadata_h5_path=metadata_h5_path,
                fluence=float(taco_val),
                time_window_fs=time_window_fs,  # optional
                background=background,
                overwrite=overwrite,
                save_laser_off=bool(save_laser_off),
                max_shots=max_shots,
            )

def main():
    def _parse_runs(run_in):
        if run_in is None:
            return []
        runs_out = []
        if isinstance(run_in, str):
            s = str(run_in).replace(",", " ")
            parts = s.split()
            for p in parts:
                try:
                    runs_out.append(int(p))
                except Exception:
                    pass
        else:
            try:
                runs_out = [int(run_in)]
            except Exception:
                runs_out = []

        seen = set()
        uniq = []
        for r in runs_out:
            if r not in seen:
                uniq.append(int(r))
                seen.add(r)
        return uniq

    parser = argparse.ArgumentParser(
        description=(
            "SACLA analysis pipeline\n"
            "Modes:\n"
            " - metadata             : create metadata H5 (single step)\n"
            " - final_imgs           : create final 2D images (chunked)\n"
            " - background           : process no-Xray background run (single step)\n"
            " - dark_from_laser_off  : create a FemtoMAX-like dark 2D image by averaging all *_laser_off.npy\n"
        )
    )

    parser.add_argument(
        "--mode",
        choices=["metadata", "final_imgs", "background", "dark_from_laser_off"],
        required=True,
        help="Processing mode: metadata | final_imgs | background | dark_from_laser_off"
    )
    parser.add_argument("--bl", type=int, default=3, help="Beamline number (default: 3)")

    # IMPORTANT: run is optional for final_imgs (delay/fluence families),
    # but required for metadata/background, and required for final_imgs if scan_type=calibration(dark).
    # For dark_from_laser_off: run(s) are required to define the dark tag (scan_... / scans_...).
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run number(s). Required for mode=metadata/background. "
             "Optional for mode=final_imgs (except scan_type=calibration). "
             "Required for mode=dark_from_laser_off. "
             "Multiple runs: '1466583,1466578' or '1466583 1466578'."
    )

    parser.add_argument(
        "--scan_type",
        choices=["delay", "delay_off", "differentials", "fluence", "fluence_off", "calibration", "background"],
        default="delay",
        help=(
            "Scan type (default: delay). "
            "Use 'calibration' for dark (FemtoMAX-like). "
            "Use 'background' only with --mode background."
        ),
    )

    parser.add_argument("--sample_name", type=str, default=None, help="Sample name (e.g., DET71)")
    parser.add_argument("--temperature_K", type=int, default=None, help="Temperature in K (e.g., 110)")
    parser.add_argument("--excitation_wl_nm", type=float, default=None, help="Excitation wavelength in nm (e.g., 1500)")
    parser.add_argument("--fluence_mJ_cm2", type=float, default=None, help="Fluence in mJ/cm^2 (delay scans)")

    parser.add_argument("--time_window", type=int, default=None, help="Time window in fs (legacy name)")
    parser.add_argument("--time_window_fs", type=int, default=None, help="Time window in fs (preferred)")

    parser.add_argument("--time_step", type=int, default=None, help="Time step for delay bin centers (fs)")
    parser.add_argument("--initial_delay", type=int, default=None, help="Initial delay for binning (fs) (optional)")
    parser.add_argument("--final_delay", type=int, default=None, help="Final delay for binning (fs) (optional)")
    parser.add_argument("--delay_for_fluence", type=int, default=None, help="Fixed delay for fluence processing (fs)")

    parser.add_argument(
        "--intensity_col",
        type=str,
        default="xfel_bl_3_st_2_pd_user_9_fitting_peak/voltage",
        help="Intensity monitor column used for filtering/normalization"
    )
    parser.add_argument("--sigma", type=float, default=1.0, help="Sigma for intensity filtering (default: 1.0)")
    parser.add_argument("--min_tags", type=int, default=100, help="Minimum tags per bin (default: 100)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    parser.add_argument("--chunk", type=int, default=None, help="Chunk number to process (required for final_imgs)")
    parser.add_argument("--n_chunks", type=int, default=200, help="Total number of chunks (default: 200)")
    parser.add_argument("--background_run", type=int, default=None, help="Background run number (no-Xray)")

    # options
    parser.add_argument(
        "--save_laser_off",
        action="store_true",
        help="Also save corresponding laser-OFF average for delay/differentials (default: False)"
    )
    parser.add_argument(
        "--max_shots",
        type=str,
        default="all",
        help="Maximum shots accumulated per output image ('all' or integer). Default: all"
    )

    # dark_from_laser_off options
    parser.add_argument(
        "--min_files",
        type=int,
        default=1,
        help="Minimum number of *_laser_off.npy files required to build a representative dark (default: 1)"
    )

    args = parser.parse_args()

    if args.time_window_fs is None and args.time_window is not None:
        args.time_window_fs = int(args.time_window)

    mode = str(args.mode).strip().lower()
    st_images = str(args.scan_type).strip().lower()

    # ---- mode: dark_from_laser_off (no metadata; averages *_laser_off.npy into dark tree) ----
    if mode == "dark_from_laser_off":
        if args.run is None:
            print("Error: --run is required for dark_from_laser_off mode.")
            sys.exit(1)

        if args.sample_name is None or str(args.sample_name).strip() == "":
            print("Error: --sample_name is required for dark_from_laser_off mode.")
            sys.exit(1)

        if args.temperature_K is None:
            print("Error: --temperature_K is required for dark_from_laser_off mode.")
            sys.exit(1)

        if args.excitation_wl_nm is None:
            print("Error: --excitation_wl_nm is required for dark_from_laser_off mode.")
            sys.exit(1)

        if args.fluence_mJ_cm2 is None:
            print("Error: --fluence_mJ_cm2 is required for dark_from_laser_off mode.")
            sys.exit(1)

        if args.time_window_fs is None:
            print("Error: --time_window_fs (or legacy --time_window) is required for dark_from_laser_off mode.")
            sys.exit(1)

        print("Creating dark from averaged *_laser_off.npy files (single step)...")
        res = create_dark_from_laser_off(
            runs=args.run,
            sample_name=args.sample_name,
            temperature_K=args.temperature_K,
            excitation_wl_nm=args.excitation_wl_nm,
            fluence_mJ_cm2=args.fluence_mJ_cm2,
            time_window_fs=args.time_window_fs,
            analysis_root=PATH_ANALYSIS_FOLDER,
            overwrite=bool(args.overwrite),
            min_files=int(args.min_files),
        )
        print(res)
        return

    # ---- background mode ----
    if mode == "background":
        if st_images != "background":
            print("Error: --mode background requires --scan_type background.")
            sys.exit(1)
        if args.run is None:
            print("Error: --run is required for background mode.")
            sys.exit(1)
        runs_list = _parse_runs(args.run)
        if len(runs_list) != 1:
            print("Error: background mode requires a single run, got: {}".format(str(runs_list)))
            sys.exit(1)

        print("Running background processing (single step)...")
        process_background_run(args.bl, int(runs_list[0]))
        return

    if st_images == "background":
        print("Error: scan_type=background must be run with --mode background.")
        sys.exit(1)

    # ---- map image scan_type -> metadata kind ----
    if st_images in ("delay", "delay_off", "differentials"):
        st_meta = "delay"
    elif st_images in ("fluence", "fluence_off"):
        st_meta = "fluence"
    elif st_images == "calibration":
        st_meta = "calibration"  # create_metadata maps this to dark internally
    else:
        print("Error: Unsupported scan_type '{}'.".format(st_images))
        sys.exit(1)

    # ---- shared requirements for metadata/final_imgs ----
    if args.sample_name is None or str(args.sample_name).strip() == "":
        print("Error: --sample_name is required for mode '{}'.".format(mode))
        sys.exit(1)

    if args.temperature_K is None:
        print("Error: --temperature_K is required for mode '{}'.".format(mode))
        sys.exit(1)

    if args.time_window_fs is None:
        print("Error: --time_window_fs (or legacy --time_window) is required for mode '{}'.".format(mode))
        sys.exit(1)

    if st_meta in ("delay", "fluence"):
        if args.excitation_wl_nm is None:
            print("Error: --excitation_wl_nm is required for scan_type '{}'.".format(st_images))
            sys.exit(1)

    if st_meta == "delay":
        if args.fluence_mJ_cm2 is None:
            print("Error: --fluence_mJ_cm2 is required for scan_type '{}'.".format(st_images))
            sys.exit(1)

    if st_meta == "fluence":
        # needed to locate metadata in final_imgs if metadata_h5_path is not provided (current design)
        if mode == "final_imgs" and args.delay_for_fluence is None:
            print("Error: --delay_for_fluence is required for scan_type '{}' in final_imgs mode.".format(st_images))
            sys.exit(1)

    # ---- mode: metadata ----
    if mode == "metadata":
        if args.run is None:
            print("Error: --run is required for metadata mode.")
            sys.exit(1)

        print("Running metadata creation (single step)...")

        if st_meta == "delay":
            if args.time_step is None:
                print("Error: delay metadata requires --time_step.")
                sys.exit(1)

        create_metadata(
            bl=args.bl,
            run=args.run,  # can be multi-run
            sample_name=args.sample_name,
            temperature_K=args.temperature_K,
            scan_type=st_meta,
            time_window_fs=args.time_window_fs,
            excitation_wl_nm=args.excitation_wl_nm,
            fluence_mJ_cm2=args.fluence_mJ_cm2,
            time_step=args.time_step,
            initial_delay=args.initial_delay,
            final_delay=args.final_delay,
            delay_for_fluence=args.delay_for_fluence,
            intensity_col=args.intensity_col,
            sigma=args.sigma,
            min_tags=args.min_tags,
            overwrite=bool(args.overwrite),
            fluence_map=None,
        )
        return

    # ---- mode: final_imgs ----
    if mode == "final_imgs":
        if args.chunk is None:
            print("Error: --chunk is required for final_imgs mode.")
            sys.exit(1)

        # run is only required here for dark (calibration) if metadata_h5_path is not explicitly given
        if st_images == "calibration":
            if args.run is None or len(_parse_runs(args.run)) == 0:
                print("Error: --run is required for scan_type=calibration in final_imgs mode (to locate dark metadata).")
                sys.exit(1)

        print("Running final image creation in chunk mode...")

        # Pass run only when needed (dark). For delay/fluence families, pass None.
        run_for_processing = None
        if st_images == "calibration":
            run_for_processing = args.run

        process_one_chunk(
            bl=args.bl,
            run=run_for_processing,
            chunk_id=args.chunk,
            n_chunks=args.n_chunks,
            scan_type=st_images,
            background=args.background_run,
            sample_name=args.sample_name,
            temperature_K=args.temperature_K,
            excitation_wl_nm=args.excitation_wl_nm,
            fluence_mJ_cm2=args.fluence_mJ_cm2,
            time_window_fs=args.time_window_fs,
            delay_for_fluence=args.delay_for_fluence,
            metadata_h5_path=None,
            overwrite=bool(args.overwrite),
            save_laser_off=bool(args.save_laser_off),
            max_shots=args.max_shots,
        )
        return

    print("Error: Unknown mode '{}'".format(mode))
    sys.exit(1)



if __name__ == "__main__":
    main()


