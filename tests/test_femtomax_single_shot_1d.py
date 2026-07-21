from __future__ import annotations

import queue
import threading
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from trxrdpy.analysis.MaxIV_FemtoMAX import azimint
from trxrdpy.analysis.MaxIV_FemtoMAX import single_shot_azimint as single_shot
from trxrdpy.analysis._shared_2d import azimint as shared_2d_azimint
from trxrdpy.analysis.common import general_utils
from trxrdpy.analysis.common import azimint_utils
from trxrdpy.analysis.common.paths import AnalysisPaths
from trxrdpy.analysis.gui.services import (
    IntegrationService,
    PathService,
    PreparationService,
)
from trxrdpy.analysis.gui.state import AnalysisGuiState
from trxrdpy.analysis.gui.tabs.pattern_creation_tab import PatternCreationTab
from trxrdpy.analysis.gui.tabs.preparation_tab import PreparationTab


class FakeIntegrator:
    """Small pyFAI-independent integrator used to exercise cache orchestration."""

    def __init__(
        self,
        *,
        npt=3,
        normalize=True,
        q_norm_range=(2.65, 2.75),
        azim_offset_deg=-90.0,
        polarization_factor=None,
        **_kwargs,
    ):
        self.npt = int(npt)
        self.normalize = bool(normalize)
        self.q_norm_range = tuple(q_norm_range)
        self.azim_offset_deg = float(azim_offset_deg)
        self.polarization_factor = polarization_factor
        self._ai = SimpleNamespace(wavelength=1.0e-10)

    @staticmethod
    def build_windows(azimuthal_edges, *, include_full=True, full_range=(-90, 90)):
        edges = np.asarray(azimuthal_edges, dtype=float)
        windows = []
        if include_full:
            windows.append((float(full_range[0]), float(full_range[1])))
        windows.extend(
            (float(start), float(stop))
            for start, stop in zip(edges[:-1], edges[1:])
        )
        return windows

    def integrate1d(self, image, azimuthal_range):
        q = np.linspace(1.0, 3.0, self.npt)
        intensity = np.full(self.npt, float(np.mean(image)))
        return q, intensity


class FakePyfaiIntegrator:
    """Model pyFAI's one-engine-per-method behavior for cache regression tests."""

    def __init__(self):
        self.engines = {}
        self.wavelength = 1.0e-10
        self.engine_builds = 0

    def integrate1d(self, _image, **kwargs):
        azimuth_range = tuple(kwargs["azimuth_range"])
        if self.engines.get("azimuth_range") != azimuth_range:
            self.engine_builds += 1
            self.engines["azimuth_range"] = azimuth_range
        return np.linspace(1.0, 3.0, 3), np.ones(3)


def _write_raw_scan(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(
        [np.full((2, 2), float(value), dtype=float) for value in values]
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("entry/measurement/pilatus/data", data=frames)


def _write_delay_metadata(path, *, scan_type="delay"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.create_dataset("sample_name", data="sample")
        meta.create_dataset("temperature_K", data=300)
        meta.create_dataset("excitation_wl_nm", data=1500.0)
        meta.create_dataset("time_window_fs", data=250)
        meta.create_dataset("scan_type", data=scan_type)
        if scan_type == "delay":
            meta.create_dataset("fluence_mJ_cm2", data=20.0)
        else:
            meta.create_dataset("scans", data=np.array([10], dtype=np.int64))
            meta.create_dataset("fluences_mJ_cm2", data=np.array([5.0]))

        delay = handle.create_group("delays/100fs")
        delay.attrs["delay_fs"] = 100
        scan = delay.create_group("scans/10")
        scan.create_dataset("indices", data=np.array([0, 2], dtype=np.int64))
        if scan_type == "fluence":
            scan.attrs["fluence_mJ_cm2"] = 5.0


def _write_multi_delay_metadata(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.create_dataset("sample_name", data="sample")
        meta.create_dataset("temperature_K", data=300)
        meta.create_dataset("excitation_wl_nm", data=1500.0)
        meta.create_dataset("time_window_fs", data=250)
        meta.create_dataset("scan_type", data="delay")
        meta.create_dataset("fluence_mJ_cm2", data=20.0)
        for delay_fs, indices in ((100, [0, 2, 4, 6]), (200, [1, 3, 5, 7])):
            delay = handle.create_group(f"delays/{delay_fs}fs")
            delay.attrs["delay_fs"] = delay_fs
            scan = delay.create_group("scans/10")
            scan.create_dataset("indices", data=np.asarray(indices, dtype=np.int64))


def _write_dark_metadata(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.create_dataset("sample_name", data="sample")
        meta.create_dataset("temperature_K", data=300)
        meta.create_dataset("scan_type", data="dark")
        scan = handle.create_group("scans/10")
        scan.attrs["nshots_expected"] = 2


@pytest.fixture
def single_shot_setup(tmp_path, monkeypatch):
    paths = AnalysisPaths(
        path_root=tmp_path,
        raw_subdir="raw",
        analysis_subdir="analysis",
    )
    poni = tmp_path / "geometry.poni"
    mask = tmp_path / "mask.edf"
    poni.write_text("poni", encoding="utf-8")
    mask.write_bytes(b"mask")
    _write_raw_scan(paths.raw_root / "scan-10.h5", [1.0, 2.0, 3.0])
    monkeypatch.setattr(single_shot.azimint_utils, "AzimIntegrator", FakeIntegrator)
    monkeypatch.setattr(single_shot, "_spawn_context_is_usable", lambda: False)
    return paths, poni, mask


def test_delay_single_shot_cache_and_final_xy_keep_canonical_names(single_shot_setup):
    paths, poni, mask = single_shot_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata.h5"
    _write_delay_metadata(metadata_path)

    report = single_shot.integrate_single_shot_1d(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        paths=paths,
    )

    assert report["written_patterns"] == 2
    shot0 = (
        analysis_dir
        / "single_shot_1D_patterns/-90_90/delay_100fs"
        / "sample_300K_1500nm_20p0mJ_250fs_100fs_-90_90_scan10_shot0.xy"
    )
    shot2 = shot0.with_name(
        "sample_300K_1500nm_20p0mJ_250fs_100fs_-90_90_scan10_shot2.xy"
    )
    assert shot0.is_file()
    assert shot2.is_file()

    _integrator, datasets = azimint.integrate_delay_1d(
        source="single_shot_1d",
        metadata_h5_path=metadata_path,
        sample_name="sample",
        temperature_K=300,
        excitation_wl_nm=1500,
        fluence_mJ_cm2=20,
        time_window_fs=250,
        delays_fs="all",
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        normalize=False,
        overwrite_xy=True,
        paths=paths,
    )

    final_path = datasets[0].xy_path("-90_90")
    assert final_path.name == "sample_300K_1500nm_20p0mJ_250fs_100fs_-90_90.xy"
    _two_theta, intensity = general_utils.load_xy(final_path)
    np.testing.assert_allclose(intensity, [2.0, 2.0, 2.0])


def test_fresh_cache_skips_per_file_existing_path_probes(
    single_shot_setup,
    monkeypatch,
):
    paths, poni, mask = single_shot_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata.h5"
    _write_delay_metadata(metadata_path)

    def fail_existing_probe(*_args, **_kwargs):
        raise AssertionError("fresh group directories need no per-file stat probes")

    monkeypatch.setattr(
        single_shot,
        "_existing_single_shot_path",
        fail_existing_probe,
    )
    report = single_shot.integrate_single_shot_1d(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        paths=paths,
    )

    assert report["written_patterns"] == 2


def test_single_shot_reads_selected_femtomax_frames_in_batches(
    single_shot_setup,
    monkeypatch,
):
    paths, poni, mask = single_shot_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata.h5"
    _write_delay_metadata(metadata_path)
    original_reader = single_shot._read_detector_frame_batch
    batches = []

    def recording_reader(detector, shot_indices):
        batches.append(tuple(int(value) for value in shot_indices))
        return original_reader(detector, shot_indices)

    monkeypatch.setattr(
        single_shot,
        "_read_detector_frame_batch",
        recording_reader,
    )
    report = single_shot.integrate_single_shot_1d(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        read_batch_size=16,
        paths=paths,
    )

    assert batches == [(0, 2)]
    assert report["read_batch_size"] == 16


def test_single_shot_parallel_partitions_use_isolated_worker_payloads(
    single_shot_setup,
    monkeypatch,
):
    paths, poni, mask = single_shot_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata.h5"
    _write_delay_metadata(metadata_path)
    captured = {"processes": None, "payloads": []}

    class InlineIterator:
        def __init__(self, values):
            self.values = iter(values)

        def next(self, timeout=None):
            del timeout
            return next(self.values)

    class InlineQueue(queue.Queue):
        def close(self):
            return None

        def join_thread(self):
            return None

    class InlinePool:
        def __init__(self, processes, initializer=None, initargs=()):
            captured["processes"] = int(processes)
            if initializer is not None:
                initializer(*initargs)

        def imap_unordered(self, function, payloads, chunksize=1):
            assert chunksize == 1
            payloads = list(payloads)
            captured["payloads"].extend(payloads)
            return InlineIterator([function(payload) for payload in payloads])

        def close(self):
            single_shot._clear_worker_state()

        def join(self):
            return None

        def terminate(self):
            single_shot._clear_worker_state()

    class InlineContext:
        def Queue(self):
            return InlineQueue()

        def Event(self):
            return threading.Event()

        def Pool(self, processes, initializer=None, initargs=()):
            return InlinePool(processes, initializer, initargs)

    monkeypatch.setattr(single_shot, "_spawn_context_is_usable", lambda: True)
    monkeypatch.setattr(single_shot.mp, "get_context", lambda _method: InlineContext())

    report = single_shot.integrate_single_shot_1d(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        read_batch_size=1,
        work_chunk_size=1,
        use_parallel=True,
        max_workers=2,
        start_method="spawn",
        paths=paths,
    )

    assert report["use_parallel"] is True
    assert report["max_workers"] == 2
    assert report["task_count"] == 2
    assert report["written_patterns"] == 2
    assert captured["processes"] == 2
    assert [len(payload["work_items"]) for payload in captured["payloads"]] == [1, 1]


def test_complete_femtomax_cache_does_not_reopen_raw_scan(single_shot_setup):
    paths, poni, mask = single_shot_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata.h5"
    _write_delay_metadata(metadata_path)
    kwargs = dict(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        paths=paths,
    )
    single_shot.integrate_single_shot_1d(**kwargs)
    flat_folder = (
        analysis_dir / "single_shot_1D_patterns/-90_90/delay_100fs"
    )
    legacy_folder = flat_folder / "scan_10"
    legacy_folder.mkdir()
    for path in flat_folder.glob("*.xy"):
        path.rename(legacy_folder / path.name)
    (paths.raw_root / "scan-10.h5").unlink()

    report = single_shot.integrate_single_shot_1d(**kwargs)

    assert report["written_patterns"] == 0
    assert report["existing_patterns"] == 2


def test_parallel_payloads_are_round_robin_across_delay_groups(
    single_shot_setup,
    monkeypatch,
):
    paths, poni, mask = single_shot_setup
    _write_raw_scan(paths.raw_root / "scan-10.h5", np.arange(1.0, 9.0))
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata.h5"
    _write_multi_delay_metadata(metadata_path)
    captured = []

    class CapturingPool:
        def __init__(self, processes, initializer, initargs):
            del processes
            initializer(*initargs)

        def imap_unordered(self, function, payloads, chunksize=1):
            del chunksize
            payloads = list(payloads)
            captured.extend(payload["group_label"] for payload in payloads)

            class Iterator:
                def __init__(self, values):
                    self._values = iter(values)

                def next(self, timeout=None):
                    del timeout
                    return next(self._values)

            return Iterator([function(payload) for payload in payloads])

        def close(self):
            single_shot._clear_worker_state()

        join = lambda self: None
        terminate = close

    class InlineQueue(queue.Queue):
        close = lambda self: None
        join_thread = lambda self: None

    class Context:
        Queue = lambda self: InlineQueue()
        Event = lambda self: threading.Event()

        def Pool(self, processes, initializer, initargs):
            return CapturingPool(processes, initializer, initargs)

    monkeypatch.setattr(single_shot, "_spawn_context_is_usable", lambda: True)
    monkeypatch.setattr(single_shot.mp, "get_context", lambda _method: Context())
    report = single_shot.integrate_single_shot_1d(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        read_batch_size=1,
        work_chunk_size=2,
        use_parallel=True,
        max_workers=2,
        paths=paths,
    )

    assert captured == ["delay 100", "delay 200", "delay 100", "delay 200"]
    assert report["written_patterns"] == 8


def test_safe_stop_and_resume_only_builds_missing_shots(
    single_shot_setup,
    monkeypatch,
):
    paths, poni, mask = single_shot_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata.h5"
    _write_delay_metadata(metadata_path)
    cancel_event = threading.Event()

    class CancellingIntegrator(FakeIntegrator):
        def integrate1d(self, image, azimuthal_range):
            result = super().integrate1d(image, azimuthal_range)
            cancel_event.set()
            return result

    monkeypatch.setattr(
        single_shot.azimint_utils,
        "AzimIntegrator",
        CancellingIntegrator,
    )
    kwargs = dict(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        read_batch_size=1,
        work_chunk_size=1,
        use_parallel=False,
        paths=paths,
    )
    stopped = single_shot.integrate_single_shot_1d(
        **kwargs,
        cancel_event=cancel_event,
    )
    assert stopped["cancelled"] is True
    assert stopped["written_patterns"] == 1

    cancel_event.clear()
    monkeypatch.setattr(single_shot.azimint_utils, "AzimIntegrator", FakeIntegrator)
    resumed = single_shot.integrate_single_shot_1d(**kwargs)
    assert resumed["written_patterns"] == 1
    assert resumed["existing_patterns"] == 1


def test_aggregation_uses_only_completed_files_for_on_the_fly_refresh(single_shot_setup):
    paths, poni, mask = single_shot_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata.h5"
    _write_delay_metadata(metadata_path)
    single_shot.integrate_single_shot_1d(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        paths=paths,
    )
    incomplete = (
        analysis_dir
        / "single_shot_1D_patterns/-90_90/delay_100fs"
        / "sample_300K_1500nm_20p0mJ_250fs_100fs_-90_90_scan10_shot2.xy"
    )
    incomplete.unlink()

    _integrator, datasets = single_shot.aggregate_delay_1d(
        metadata_h5_path=metadata_path,
        sample_name="sample",
        temperature_K=300,
        excitation_wl_nm=1500,
        fluence_mJ_cm2=20,
        time_window_fs=250,
        delays_fs=[100],
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        normalize=False,
        overwrite_xy=True,
        paths=paths,
    )
    _x, intensity = general_utils.load_xy(datasets[0].xy_path("-90_90"))
    np.testing.assert_allclose(intensity, [1.0, 1.0, 1.0])


def test_aggregation_all_uses_only_groups_present_in_completed_files(single_shot_setup):
    paths, poni, mask = single_shot_setup
    _write_raw_scan(paths.raw_root / "scan-10.h5", np.arange(1.0, 9.0))
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata.h5"
    _write_multi_delay_metadata(metadata_path)
    single_shot.integrate_single_shot_1d(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        paths=paths,
    )
    for path in (
        analysis_dir / "single_shot_1D_patterns/-90_90/delay_200fs"
    ).glob("*.xy"):
        path.unlink()

    _integrator, datasets = single_shot.aggregate_delay_1d(
        metadata_h5_path=metadata_path,
        sample_name="sample",
        temperature_K=300,
        excitation_wl_nm=1500,
        fluence_mJ_cm2=20,
        time_window_fs=250,
        delays_fs="all",
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        normalize=False,
        overwrite_xy=True,
        paths=paths,
    )
    assert [dataset.delay_fs for dataset in datasets] == [100]


def test_incremental_aggregation_reads_only_newly_completed_files(
    single_shot_setup,
    monkeypatch,
):
    paths, poni, mask = single_shot_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata.h5"
    _write_delay_metadata(metadata_path)
    production_kwargs = dict(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        paths=paths,
    )
    single_shot.integrate_single_shot_1d(**production_kwargs)
    missing = (
        analysis_dir
        / "single_shot_1D_patterns/-90_90/delay_100fs"
        / "sample_300K_1500nm_20p0mJ_250fs_100fs_-90_90_scan10_shot2.xy"
    )
    missing.unlink()
    aggregation_kwargs = dict(
        metadata_h5_path=metadata_path,
        sample_name="sample",
        temperature_K=300,
        excitation_wl_nm=1500,
        fluence_mJ_cm2=20,
        time_window_fs=250,
        delays_fs="all",
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        normalize=False,
        overwrite_xy=True,
        paths=paths,
    )
    single_shot.aggregate_delay_1d(**aggregation_kwargs)
    single_shot.integrate_single_shot_1d(**production_kwargs)

    original_loader = single_shot._load_single_shot_xy
    loaded = []

    def recording_loader(path):
        loaded.append(path.name)
        return original_loader(path)

    monkeypatch.setattr(single_shot, "_load_single_shot_xy", recording_loader)
    single_shot.aggregate_delay_1d(**aggregation_kwargs)
    assert loaded == [missing.name]


def test_metadata_path_can_be_resolved_from_delay_experiment_fields(single_shot_setup):
    paths, _poni, _mask = single_shot_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata.h5"
    _write_delay_metadata(metadata_path)

    resolved = single_shot.resolve_metadata_h5_path(
        scan_type="delay",
        sample_name="sample",
        temperature_K=300,
        excitation_wl_nm=1500,
        fluence_mJ_cm2=20,
        time_window_fs=250,
        paths=paths,
    )
    assert resolved == metadata_path.resolve()


def test_pattern_tab_infers_femtomax_metadata_when_path_is_blank(single_shot_setup):
    paths, _poni, _mask = single_shot_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata.h5"
    _write_delay_metadata(metadata_path)
    app = QApplication.instance() or QApplication([])
    state = AnalysisGuiState(
        facility="FemtoMAX",
        path_root=paths.path_root,
        raw_subdir=paths.raw_subdir,
        analysis_subdir=paths.analysis_subdir,
    )
    tab = PatternCreationTab(state, PathService(), IntegrationService())
    tab.experiment_metadata.set_values(
        {
            "sample_name": "sample",
            "temperature_K": "300",
            "excitation_wl_nm": "1500",
            "fluence_mJ_cm2": "20",
            "time_window_fs": "250",
        }
    )
    tab.pattern_input_mode_combo.setCurrentText("Single-shot 1D patterns")
    kwargs = {}
    tab._add_single_shot_source_kwargs(kwargs)

    assert kwargs["metadata_h5_path"] == str(metadata_path.resolve())
    assert tab.pattern_single_shot_metadata_h5.text() == str(metadata_path.resolve())
    tab.deleteLater()
    app.processEvents()


@pytest.mark.parametrize(
    ("scan_type", "metadata_builder", "group_folder", "filename", "expected_mean"),
    [
        (
            "fluence",
            _write_delay_metadata,
            "fluence_5p0mJ",
            "sample_300K_1500nm_5p0mJ_250fs_100fs_-90_90_scan10_shot0.xy",
            2.0,
        ),
        (
            "dark",
            _write_dark_metadata,
            None,
            "sample_300K_dark_scan10_-90_90_scan10_shot0.xy",
            1.5,
        ),
    ],
)
def test_fluence_and_dark_single_shot_layouts(
    single_shot_setup,
    scan_type,
    metadata_builder,
    group_folder,
    filename,
    expected_mean,
):
    paths, poni, mask = single_shot_setup
    if scan_type == "fluence":
        analysis_dir = (
            paths.analysis_root
            / "sample/temperature_300K/excitation_wl_1500nm/fluence"
            / "delay_100fs/time_window_250fs"
        )
        metadata_path = analysis_dir / "metadata.h5"
        metadata_builder(metadata_path, scan_type="fluence")
    else:
        analysis_dir = paths.analysis_root / "sample/temperature_300K/dark/scan_10"
        metadata_path = analysis_dir / "metadata.h5"
        metadata_builder(metadata_path)

    single_shot.integrate_single_shot_1d(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        paths=paths,
    )
    expected = analysis_dir / "single_shot_1D_patterns/-90_90"
    if group_folder is not None:
        expected = expected / group_folder
    expected = expected / filename
    assert expected.is_file()

    if scan_type == "fluence":
        _integrator, datasets = single_shot.aggregate_fluence_1d(
            metadata_h5_path=metadata_path,
            sample_name="sample",
            temperature_K=300,
            excitation_wl_nm=1500,
            delay_fs=100,
            time_window_fs=250,
            fluences_mJ_cm2="all",
            poni_path=poni,
            mask_edf_path=mask,
            azimuthal_edges=[-90, 90],
            include_full=False,
            npt=3,
            normalize=False,
            overwrite_xy=True,
            paths=paths,
        )
        final_path = datasets[0].xy_path("-90_90")
        assert final_path.name == "sample_300K_1500nm_5p0mJ_250fs_100fs_-90_90.xy"
    else:
        _integrator, dataset = single_shot.aggregate_dark_1d(
            metadata_h5_path=metadata_path,
            sample_name="sample",
            temperature_K=300,
            poni_path=poni,
            mask_edf_path=mask,
            azimuthal_edges=[-90, 90],
            include_full=False,
            npt=3,
            normalize=False,
            overwrite_xy=True,
            paths=paths,
        )
        final_path = dataset.xy_path("-90_90")
        assert final_path.name == "sample_300K_dark_scan10_-90_90.xy"

    _x, intensity = general_utils.load_xy(final_path)
    np.testing.assert_allclose(intensity, [expected_mean] * 3)


def test_settings_fingerprint_rejects_incompatible_aggregation(
    single_shot_setup,
    capsys,
):
    paths, poni, mask = single_shot_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata.h5"
    _write_delay_metadata(metadata_path)
    single_shot.integrate_single_shot_1d(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        paths=paths,
    )

    with pytest.raises(ValueError, match="settings do not match") as exc_info:
        single_shot.aggregate_delay_1d(
            metadata_h5_path=metadata_path,
            sample_name="sample",
            temperature_K=300,
            excitation_wl_nm=1500,
            fluence_mJ_cm2=20,
            time_window_fs=250,
            delays_fs=[100],
            poni_path=poni,
            mask_edf_path=mask,
            azimuthal_edges=[-90, 90],
            include_full=False,
            npt=4,
            normalize=False,
            overwrite_xy=True,
            paths=paths,
        )
    message = str(exc_info.value)
    assert "Differing settings:" in message
    assert "npt: cached=3; requested=4" in message
    assert "FemtoMAX delay aggregation snapshot" not in capsys.readouterr().out


def test_resume_rejects_incompatible_existing_single_shot_settings(
    single_shot_setup,
):
    paths, poni, mask = single_shot_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata.h5"
    _write_delay_metadata(metadata_path)
    kwargs = dict(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        paths=paths,
    )
    single_shot.integrate_single_shot_1d(
        **kwargs,
        polarization_factor=None,
    )

    shot_path = next(
        (analysis_dir / "single_shot_1D_patterns").rglob("*.xy")
    )
    first_lines = shot_path.read_text(encoding="utf-8").splitlines()[:2]
    assert first_lines[0].startswith(single_shot._SETTINGS_HEADER)
    assert first_lines[1].startswith(single_shot._SETTINGS_JSON_HEADER)

    with pytest.raises(ValueError, match="polarization_factor") as exc_info:
        single_shot.integrate_single_shot_1d(
            **kwargs,
            polarization_factor=0.99,
        )
    assert "cached=None; requested=0.99" in str(exc_info.value)


def test_femtomax_default_integration_source_remains_representative_2d(monkeypatch):
    expected = object()
    captured = {}

    def fake_integrate(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        azimint,
        "_integrate_delay_1d_from_2d",
        fake_integrate,
    )
    assert azimint.integrate_delay_1d(
        sample_name="sample",
        temperature_K=300,
        excitation_wl_nm=1500,
        fluence_mJ_cm2=20,
        time_window_fs=250,
        delays_fs=[100],
        poni_path="geometry.poni",
        mask_edf_path="mask.edf",
        path_root="/experiment",
        analysis_subdir="analysis",
    ) is expected
    assert captured["use_parallel"] is True
    assert captured["max_workers"] is None


def test_representative_2d_collection_parallelizes_windows_and_loads_image_once(
    tmp_path,
    monkeypatch,
):
    barrier = threading.Barrier(2)
    worker_threads = []
    constructor_threads = []
    created = []

    class Dataset:
        def __init__(self):
            self.load_count = 0

        def xy_path(self, azim_tag):
            return tmp_path / f"{azim_tag}.xy"

        def load_2d(self):
            self.load_count += 1
            return np.ones((2, 2))

    class WorkerIntegrator:
        def __init__(self, **_kwargs):
            created.append(self)
            constructor_threads.append(threading.get_ident())

        @staticmethod
        def build_windows(azimuthal_edges, **_kwargs):
            edges = np.asarray(azimuthal_edges, dtype=float)
            return list(zip(edges[:-1], edges[1:]))

        def integrate_and_cache_window(
            self,
            _dataset,
            _image,
            _window,
            **_kwargs,
        ):
            barrier.wait(timeout=2.0)
            worker_threads.append(threading.get_ident())

    primary = WorkerIntegrator()
    dataset = Dataset()
    monkeypatch.setattr(
        shared_2d_azimint.azimint_utils,
        "AzimIntegrator",
        WorkerIntegrator,
    )
    shared_2d_azimint._integrate_dataset_collection(
        datasets=[dataset],
        primary_integrator=primary,
        integrator_kwargs={},
        azimuthal_edges=[-90, 0, 90],
        include_full=False,
        full_range=(-90, 90),
        overwrite_xy=True,
        use_parallel=True,
        max_workers=2,
    )

    assert len(set(worker_threads)) == 2
    assert len(created) == 2
    assert set(constructor_threads) == {threading.get_ident()}
    assert dataset.load_count == 1


def test_poni_compat_patches_silently_incomplete_old_pyfai_geometry(
    tmp_path,
    monkeypatch,
):
    poni_path = tmp_path / "geometry.poni"
    poni_path.write_text(
        "\n".join(
            (
                "poni_version: 2.1",
                "Detector: Detector",
                'Detector_config: {"pixel1": 0.0001, "pixel2": 0.0002, '
                '"orientation": 3}',
                "Distance: 0.1",
                "Poni1: 0.01",
                "Poni2: 0.02",
                "Rot1: 0",
                "Rot2: 0",
                "Rot3: 0",
                "Wavelength: 1e-10",
            )
        ),
        encoding="utf-8",
    )
    loaded_paths = []

    def fake_load(path):
        path = Path(path)
        loaded_paths.append(path)
        if path == poni_path:
            detector = SimpleNamespace(pixel1=None, pixel2=None)
        else:
            patched_text = path.read_text(encoding="utf-8")
            assert "orientation" not in patched_text
            detector = SimpleNamespace(pixel1=0.0001, pixel2=0.0002)
        return SimpleNamespace(detector=detector)

    monkeypatch.setattr(azimint_utils.pyFAI, "load", fake_load)

    ai, used_path, changes = azimint_utils.load_poni_with_compat(poni_path)

    assert ai.detector.pixel1 == pytest.approx(0.0001)
    assert Path(used_path) == Path(str(poni_path) + ".pyfai021")
    assert changes == [
        "poni_version 2.1 -> 2",
        "Detector_config: dropped 'orientation'",
    ]
    assert loaded_paths == [poni_path, Path(used_path)]
    assert not list(tmp_path.glob(".geometry.poni.pyfai021.*.tmp"))


def test_femtomax_final_aggregation_parallelizes_groups_without_nested_readers(
    monkeypatch,
):
    barrier = threading.Barrier(2)
    calls = []

    def fake_aggregate_dataset(**kwargs):
        barrier.wait(timeout=2.0)
        calls.append((threading.get_ident(), kwargs["read_workers"]))

    monkeypatch.setattr(single_shot, "_aggregate_dataset", fake_aggregate_dataset)
    single_shot._aggregate_dataset_jobs(
        jobs=[(100, object(), 4), (200, object(), 4)],
        use_parallel=True,
        max_workers=2,
        aggregate_kwargs={"metadata": {"scan_type": "delay"}},
    )

    assert len({thread_id for thread_id, _read_workers in calls}) == 2
    assert {read_workers for _thread_id, read_workers in calls} == {1}


def test_gui_separates_femtomax_shot_production_from_final_integration():
    app = QApplication.instance() or QApplication([])
    state = AnalysisGuiState(facility="FemtoMAX")
    state.poni_path = Path("geometry.poni")
    state.mask_edf_path = Path("mask.edf")
    paths = PathService()
    integration = IntegrationService()
    pattern_tab = PatternCreationTab(state, paths, integration)
    datared_tab = PreparationTab(
        state,
        paths,
        PreparationService(),
        integration_service=integration,
    )

    assert pattern_tab.pattern_input_mode_combo.currentText() == "Representative 2D images"
    assert pattern_tab.pattern_use_parallel.isChecked()
    assert pattern_tab.pattern_max_workers.text() == "4"
    assert pattern_tab.pattern_single_shot_group.isHidden()

    pattern_tab.pattern_input_mode_combo.setCurrentText("Single-shot 1D patterns")
    assert not pattern_tab.pattern_single_shot_group.isHidden()
    assert not hasattr(pattern_tab, "pattern_integrate_single_shot_btn")
    assert not datared_tab.datared_single_shot_group.isHidden()
    assert not hasattr(datared_tab, "datared_single_shot_poni_path")
    assert not hasattr(datared_tab, "datared_single_shot_mask_path")
    assert datared_tab._poni_path() == Path("geometry.poni")
    assert datared_tab._mask_path() == Path("mask.edf")
    assert not datared_tab.datared_single_shot_azim_offset_deg.isHidden()
    assert datared_tab.datared_single_shot_normalize_final.isChecked()
    assert datared_tab.datared_single_shot_q_norm_range.text() == "(2.65, 2.75)"
    assert not (
        datared_tab.datared_single_shot_polarization_control.isHidden()
    )
    assert (
        datared_tab.datared_single_shot_polarization_control.effective_factor()
        == 0.99
    )
    assert not datared_tab.datared_femtomax_read_batch_size.isHidden()
    assert datared_tab.datared_femtomax_read_batch_size.text() == "16"
    assert not datared_tab.datared_femtomax_single_shot_use_parallel.isHidden()
    assert datared_tab.datared_femtomax_single_shot_use_parallel.isChecked()
    assert datared_tab.datared_femtomax_single_shot_max_workers.text() == "4"
    assert datared_tab.datared_femtomax_work_chunk_size.text() == "64"
    assert datared_tab.datared_integrate_single_shot_btn.text() == (
        "Produce Single-Shot 1D Patterns"
    )

    pattern_tab.set_facility("ID09")
    datared_tab.set_facility("ID09")
    assert pattern_tab.pattern_input_mode_combo.currentText() == "Representative 2D images"
    assert pattern_tab.pattern_input_mode_combo.isHidden()
    assert pattern_tab.pattern_single_shot_group.isHidden()
    assert pattern_tab.pattern_use_parallel.isHidden()
    assert pattern_tab.pattern_max_workers.isHidden()
    assert datared_tab.datared_single_shot_group.isHidden()
    pattern_tab.deleteLater()
    datared_tab.deleteLater()
    app.processEvents()


def test_azimuth_engine_cache_retains_one_pyfai_engine_per_window():
    pyfai_integrator = FakePyfaiIntegrator()
    integrator = object.__new__(azimint_utils.AzimIntegrator)
    integrator._ai = pyfai_integrator
    integrator._mask = np.zeros((2, 2), dtype=bool)
    integrator._azimuth_engine_caches = {}
    integrator.npt = 3
    integrator.normalize = False
    integrator.azim_offset_deg = -90.0
    integrator.polarization_factor = None

    image = np.ones((2, 2), dtype=float)
    for window in ((-90.0, 0.0), (0.0, 90.0)) * 3:
        integrator.integrate1d(
            image,
            window,
            cache_engine_by_azimuth=True,
        )

    assert pyfai_integrator.engine_builds == 2
    assert set(integrator._azimuth_engine_caches) == {
        (-90.0, 0.0),
        (0.0, 90.0),
    }
    assert pyfai_integrator.engines == {}


def test_worker_retains_all_configured_raw_scan_datasets(tmp_path, monkeypatch):
    raw_paths = []
    for scan in (10, 11, 12):
        raw_path = tmp_path / f"scan-{scan}.h5"
        _write_raw_scan(raw_path, [scan])
        raw_paths.append(raw_path)

    original_open = single_shot.h5py.File
    open_count = 0

    def recording_open(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(single_shot.h5py, "File", recording_open)
    single_shot._WORKER_RAW_HANDLES = OrderedDict()
    single_shot._WORKER_COMMON_PAYLOAD = {
        "detector_h5_path": ("entry", "measurement", "pilatus", "data"),
        "raw_handle_cache_size": 3,
    }
    try:
        for raw_path in raw_paths + [raw_paths[0], raw_paths[1]]:
            detector = single_shot._worker_detector({"raw_path": str(raw_path)})
            assert detector.shape == (1, 2, 2)
        assert open_count == 3
        assert len(single_shot._WORKER_RAW_HANDLES) == 3
    finally:
        single_shot._clear_worker_state()


def test_scan_affinity_assigns_each_raw_file_to_only_one_balanced_lane():
    payloads = []
    for round_index in range(3):
        for scan in range(6):
            payloads.append(
                {
                    "raw_path": f"scan-{scan}.h5",
                    "work_items": [None] * (scan + 1),
                    "round": round_index,
                }
            )

    lanes = single_shot._build_scan_affinity_lanes(payloads, worker_count=3)

    assert len(lanes) == 3
    raw_lane = {}
    for lane_index, lane in enumerate(lanes):
        for payload in lane:
            raw_path = payload["raw_path"]
            assert raw_path not in raw_lane or raw_lane[raw_path] == lane_index
            raw_lane[raw_path] = lane_index
    assert len(raw_lane) == 6
    loads = [
        sum(len(payload["work_items"]) for payload in lane)
        for lane in lanes
    ]
    assert max(loads) - min(loads) <= 3
