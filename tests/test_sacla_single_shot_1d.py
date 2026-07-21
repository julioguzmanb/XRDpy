from __future__ import annotations

import threading
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from trxrdpy.analysis.Spring8_SACLA import azimint
from trxrdpy.analysis.Spring8_SACLA import single_shot_azimint as single_shot
from trxrdpy.analysis.common import general_utils
from trxrdpy.analysis.common.paths import AnalysisPaths
from trxrdpy.analysis.gui.services import (
    IntegrationService,
    PathService,
    PreparationService,
)
from trxrdpy.analysis.gui.services import integration_service as integration_service_module
from trxrdpy.analysis.gui.state import AnalysisGuiState
from trxrdpy.analysis.gui.tabs.pattern_creation_tab import PatternCreationTab
from trxrdpy.analysis.gui.tabs.preparation_tab import PreparationTab


class FakeIntegrator:
    """pyFAI-independent integrator for cache orchestration tests."""

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

    def integrate1d(self, image, _azimuthal_range):
        q = np.linspace(1.0, 3.0, self.npt)
        return q, np.full(self.npt, float(np.mean(image)))


class FakeSaclaClient:
    """Injected facility boundary recording typed run/tag and metadata calls."""

    def __init__(self, images, intensities=None):
        self.images = dict(images)
        self.intensities = dict(intensities or {})
        self.image_calls = []
        self.intensity_calls = []

    def read_image(self, run, tag):
        self.image_calls.append((int(run), int(tag)))
        return np.asarray(self.images[(int(run), int(tag))])

    def read_pulse_intensities(self, run, tags, intensity_col):
        tags = tuple(int(tag) for tag in tags)
        self.intensity_calls.append((int(run), tags, str(intensity_col)))
        return np.asarray(
            [self.intensities[(int(run), tag)] for tag in tags],
            dtype=float,
        )


def test_sacla_final_aggregation_parallelizes_groups_without_nested_readers(
    monkeypatch,
):
    barrier = threading.Barrier(2)
    calls = []

    def fake_aggregate_dataset(**kwargs):
        barrier.wait(timeout=2.0)
        calls.append((threading.get_ident(), kwargs["read_workers"]))

    monkeypatch.setattr(single_shot, "_aggregate_dataset", fake_aggregate_dataset)
    single_shot._aggregate_dataset_jobs(
        jobs=[(100, object()), (200, object())],
        use_parallel=True,
        max_workers=2,
        aggregate_kwargs={"metadata": {"scan_type": "delay"}},
    )

    assert len({thread_id for thread_id, _read_workers in calls}) == 2
    assert {read_workers for _thread_id, read_workers in calls} == {1}


def _write_meta_common(handle, *, scan_type):
    meta = handle.create_group("meta")
    meta.create_dataset("sample_name", data="sample")
    meta.create_dataset("temperature_K", data=300)
    meta.create_dataset("scan_type", data=scan_type)
    meta.create_dataset("beamline", data=3)
    meta.create_dataset(
        "intensity_col",
        data="xfel_bl_3_st_2_pd_user_9_fitting_peak/voltage",
    )
    meta.create_dataset("scans", data=np.array([10], dtype=np.int64))
    if scan_type != "dark":
        meta.create_dataset("excitation_wl_nm", data=1500.0)
        meta.create_dataset("time_window_fs", data=250)
    return meta


def _write_delay_metadata(path, *, include_intensities=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        meta = _write_meta_common(handle, scan_type="delay")
        meta.create_dataset("fluence_mJ_cm2", data=20.0)
        delay = handle.create_group("delays/100fs")
        delay.attrs["delay_fs"] = 100
        scan = delay.create_group("scans/10")
        scan.create_dataset("tags", data=np.array([1001, 1002], dtype=np.int64))
        if include_intensities:
            scan.create_dataset("pulse_intensity", data=np.array([2.0, 4.0]))


def _write_fluence_metadata(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        meta = _write_meta_common(handle, scan_type="fluence")
        meta.create_dataset("delay_fs", data=100)
        fluence = handle.create_group("fluences/5p0mJ")
        fluence.attrs["fluence_mJ_cm2"] = 5.0
        fluence.attrs["delay_fs"] = 100
        scan = fluence.create_group("scans/10")
        scan.create_dataset("tags", data=np.array([1001, 1002], dtype=np.int64))
        scan.create_dataset("pulse_intensity", data=np.array([2.0, 4.0]))


def _write_dark_metadata(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        _write_meta_common(handle, scan_type="dark")
        scan = handle.create_group("scans/10")
        scan.create_dataset("tags", data=np.array([1001, 1002], dtype=np.int64))


@pytest.fixture
def sacla_setup(tmp_path, monkeypatch):
    paths = AnalysisPaths(
        path_root=tmp_path,
        raw_subdir="raw",
        analysis_subdir="analysis",
    )
    poni = tmp_path / "geometry.poni"
    mask = tmp_path / "mask.edf"
    poni.write_text("poni", encoding="utf-8")
    mask.write_bytes(b"mask")
    monkeypatch.setattr(single_shot.azimint_utils, "AzimIntegrator", FakeIntegrator)
    return paths, poni, mask


def test_delay_cache_preprocessing_and_final_xy_keep_canonical_names(sacla_setup):
    paths, poni, mask = sacla_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata/sacla.h5"
    _write_delay_metadata(metadata_path)
    background_path = paths.analysis_root / "99/99.npy"
    background_path.parent.mkdir(parents=True)
    np.save(background_path, np.full((2, 2), 10.0))
    client = FakeSaclaClient(
        {
            (10, 1001): np.full((2, 2), 70.0),
            (10, 1002): np.full((2, 2), 110.0),
        }
    )

    report = azimint.integrate_single_shot_1d(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        threshold_counts=40,
        background=99,
        paths=paths,
        facility_client=client,
    )

    assert report["written_patterns"] == 2
    assert client.image_calls == [(10, 1001), (10, 1002)]
    assert client.intensity_calls == []
    shot1 = (
        analysis_dir
        / "single_shot_1D_patterns/-90_90/delay_100fs/scan_10"
        / "sample_300K_1500nm_20p0mJ_250fs_100fs_-90_90_scan10_tag1001.xy"
    )
    assert shot1.is_file()

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
        threshold_counts=40,
        background=99,
        paths=paths,
    )
    final_path = datasets[0].xy_path("-90_90")
    assert final_path.name == "sample_300K_1500nm_20p0mJ_250fs_100fs_-90_90.xy"
    _x, intensity = general_utils.load_xy(final_path)
    np.testing.assert_allclose(intensity, [27.5, 27.5, 27.5])


def test_sacla_array_chunks_own_disjoint_run_tag_outputs(sacla_setup):
    paths, poni, mask = sacla_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata/sacla.h5"
    _write_delay_metadata(metadata_path)
    images = {
        (10, 1001): np.full((2, 2), 80.0),
        (10, 1002): np.full((2, 2), 120.0),
    }

    first_client = FakeSaclaClient(images)
    first_report = azimint.integrate_single_shot_1d(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        threshold_counts=40,
        paths=paths,
        facility_client=first_client,
        chunk_id=1,
        n_chunks=2,
    )
    second_client = FakeSaclaClient(images)
    second_report = azimint.integrate_single_shot_1d(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        threshold_counts=40,
        paths=paths,
        facility_client=second_client,
        chunk_id=2,
        n_chunks=2,
    )

    assert first_client.image_calls == [(10, 1001)]
    assert second_client.image_calls == [(10, 1002)]
    assert first_report["selected_shots"] == 1
    assert second_report["selected_shots"] == 1
    assert first_report["selected_shots_total"] == 2
    assert second_report["selected_shots_total"] == 2
    assert first_report["written_patterns"] == 1
    assert second_report["written_patterns"] == 1


def test_sacla_single_shot_cli_forwards_pbs_array_partition(monkeypatch, tmp_path):
    captured = {}

    def fake_integrate(**kwargs):
        captured.update(kwargs)
        return {
            "written_patterns": 0,
            "existing_patterns": 0,
            "invalid_shots": 0,
        }

    monkeypatch.setattr(single_shot, "integrate_single_shot_1d", fake_integrate)
    single_shot.main(
        [
            "--metadata-h5",
            str(tmp_path / "selection.h5"),
            "--poni",
            str(tmp_path / "geometry.poni"),
            "--azimuthal-edges",
            "-90",
            "0",
            "90",
            "--path-root",
            str(tmp_path),
            "--chunk",
            "7",
            "--n-chunks",
            "20",
        ]
    )

    assert captured["chunk_id"] == 7
    assert captured["n_chunks"] == 20
    assert captured["azimuthal_edges"] == [-90.0, 0.0, 90.0]
    assert captured["paths"].path_root == tmp_path


def test_legacy_metadata_uses_one_batched_db_lookup_per_run(sacla_setup):
    paths, poni, mask = sacla_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata/legacy.h5"
    _write_delay_metadata(metadata_path, include_intensities=False)
    client = FakeSaclaClient(
        {
            (10, 1001): np.full((2, 2), 20.0),
            (10, 1002): np.full((2, 2), 80.0),
        },
        intensities={(10, 1001): 2.0, (10, 1002): 4.0},
    )

    single_shot.integrate_single_shot_1d(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        threshold_counts=0,
        paths=paths,
        facility_client=client,
    )

    assert client.intensity_calls == [
        (
            10,
            (1001, 1002),
            "xfel_bl_3_st_2_pd_user_9_fitting_peak/voltage",
        )
    ]


def test_complete_sacla_cache_does_not_initialize_facility_client(
    sacla_setup,
    monkeypatch,
):
    paths, poni, mask = sacla_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata/sacla.h5"
    _write_delay_metadata(metadata_path)
    client = FakeSaclaClient(
        {
            (10, 1001): np.full((2, 2), 20.0),
            (10, 1002): np.full((2, 2), 80.0),
        }
    )
    kwargs = dict(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        threshold_counts=0,
        paths=paths,
    )
    single_shot.integrate_single_shot_1d(
        **kwargs,
        facility_client=client,
    )

    def fail_if_initialized(**_kwargs):
        raise AssertionError("SACLA facility client should not be initialized")

    monkeypatch.setattr(single_shot, "SaclaFacilityClient", fail_if_initialized)
    report = single_shot.integrate_single_shot_1d(**kwargs)

    assert report["written_patterns"] == 0
    assert report["existing_patterns"] == 2


@pytest.mark.parametrize("scan_type", ["fluence", "dark"])
def test_fluence_and_dark_cache_layouts(sacla_setup, scan_type):
    paths, poni, mask = sacla_setup
    if scan_type == "fluence":
        analysis_dir = (
            paths.analysis_root
            / "sample/temperature_300K/excitation_wl_1500nm/fluence"
            / "delay_100fs/time_window_250fs"
        )
        metadata_path = analysis_dir / "metadata/sacla.h5"
        _write_fluence_metadata(metadata_path)
        group = "fluence_5p0mJ"
        final_stem = "sample_300K_1500nm_5p0mJ_250fs_100fs_-90_90"
    else:
        analysis_dir = paths.analysis_root / "sample/temperature_300K/dark/scan_10"
        metadata_path = analysis_dir / "metadata/sacla.h5"
        _write_dark_metadata(metadata_path)
        group = None
        final_stem = "sample_300K_dark_scan10_-90_90"
    client = FakeSaclaClient(
        {
            (10, 1001): np.full((2, 2), 60.0),
            (10, 1002): np.full((2, 2), 80.0),
        }
    )

    single_shot.integrate_single_shot_1d(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        threshold_counts=0,
        paths=paths,
        facility_client=client,
    )
    expected = analysis_dir / "single_shot_1D_patterns/-90_90"
    if group is not None:
        expected = expected / group
    expected = expected / "scan_10" / "{}_scan10_tag1001.xy".format(final_stem)
    assert expected.is_file()
    assert client.intensity_calls == []


def test_partial_cache_refresh_and_fingerprint_validation(sacla_setup):
    paths, poni, mask = sacla_setup
    analysis_dir = (
        paths.analysis_root
        / "sample/temperature_300K/excitation_wl_1500nm/delay"
        / "fluence_20p0mJ/time_window_250fs"
    )
    metadata_path = analysis_dir / "metadata/sacla.h5"
    _write_delay_metadata(metadata_path)
    client = FakeSaclaClient(
        {
            (10, 1001): np.full((2, 2), 20.0),
            (10, 1002): np.full((2, 2), 80.0),
        }
    )
    single_shot.integrate_single_shot_1d(
        metadata_h5_path=metadata_path,
        poni_path=poni,
        mask_edf_path=mask,
        azimuthal_edges=[-90, 90],
        include_full=False,
        npt=3,
        threshold_counts=0,
        paths=paths,
        facility_client=client,
    )
    shot2 = (
        analysis_dir
        / "single_shot_1D_patterns/-90_90/delay_100fs/scan_10"
        / "sample_300K_1500nm_20p0mJ_250fs_100fs_-90_90_scan10_tag1002.xy"
    )
    shot2.unlink()

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
        threshold_counts=0,
        paths=paths,
    )
    _x, intensity = general_utils.load_xy(datasets[0].xy_path("-90_90"))
    np.testing.assert_allclose(intensity, [10.0, 10.0, 10.0])

    with pytest.raises(ValueError, match="settings do not match"):
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
            threshold_counts=0,
            paths=paths,
        )


def test_gui_exposes_sacla_single_shot_controls_and_id09_remains_disabled():
    app = QApplication.instance() or QApplication([])
    state = AnalysisGuiState(facility="SACLA")
    paths = PathService()
    integration = IntegrationService()
    tab = PatternCreationTab(
        state,
        paths,
        integration,
    )
    datared_tab = PreparationTab(
        state,
        paths,
        PreparationService(),
        integration_service=integration,
    )
    tab.pattern_input_mode_combo.setCurrentText("Single-shot 1D patterns")
    app.processEvents()

    assert not tab.pattern_single_shot_group.isHidden()
    assert not tab.pattern_sacla_detector_id.isHidden()
    assert not hasattr(tab, "pattern_integrate_single_shot_btn")
    assert not datared_tab.datared_single_shot_group.isHidden()
    assert not datared_tab.datared_sacla_detector_id.isHidden()
    assert not datared_tab.datared_sacla_n_chunks.isHidden()
    assert datared_tab.datared_sacla_n_chunks.text() == "20"
    assert datared_tab.datared_femtomax_read_batch_size.isHidden()

    tab.set_facility("FemtoMAX")
    datared_tab.set_facility("FemtoMAX")
    assert tab.pattern_sacla_detector_id.isHidden()
    assert not tab.pattern_single_shot_group.isHidden()
    assert datared_tab.datared_sacla_detector_id.isHidden()
    assert datared_tab.datared_sacla_n_chunks.isHidden()
    assert not datared_tab.datared_femtomax_read_batch_size.isHidden()

    tab.set_facility("ID09")
    datared_tab.set_facility("ID09")
    assert tab.pattern_input_mode_combo.currentText() == "Representative 2D images"
    assert tab.pattern_input_mode_combo.isHidden()
    assert tab.pattern_single_shot_group.isHidden()
    assert datared_tab.datared_single_shot_group.isHidden()
    tab.deleteLater()
    datared_tab.deleteLater()
    app.processEvents()


def test_gui_service_submits_sacla_as_pbs_array(monkeypatch, tmp_path):
    paths = AnalysisPaths(
        path_root=tmp_path,
        raw_subdir="raw",
        analysis_subdir="analysis",
    )
    poni = tmp_path / "geometry.poni"
    mask = tmp_path / "mask.edf"
    metadata = tmp_path / "selection.h5"
    poni.write_text("poni", encoding="utf-8")
    mask.write_bytes(b"mask")
    metadata.write_bytes(b"metadata")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="48123[].server\n", stderr="")

    monkeypatch.setattr(integration_service_module.subprocess, "run", fake_run)
    service = IntegrationService()
    report = service.submit_sacla_single_shot_1d(
        integration_kwargs={
            "metadata_h5_path": str(metadata),
            "poni_path": str(poni),
            "mask_edf_path": str(mask),
            "azimuthal_edges": [-90.0, -30.0, 30.0, 90.0],
            "include_full": False,
            "full_range": (-80.0, 80.0),
            "npt": 2048,
            "overwrite": True,
            "azim_offset_deg": -87.5,
            "polarization_factor": 0.99,
            "beamline": None,
            "detector_id": "detector",
            "background": 1466500,
            "threshold_counts": 42.5,
            "intensity_col": "pulse/voltage",
            "paths": paths,
        },
        n_chunks=24,
    )

    assert captured["command"][0:3] == ["qsub", "-J", "1-24"]
    assert captured["cwd"] == str(tmp_path)
    environment = captured["env"]
    assert environment["XRDPY_N_CHUNKS"] == "24"
    assert environment["XRDPY_INCLUDE_FULL"] == "0"
    assert environment["XRDPY_FULL_RANGE"] == "-80 80"
    assert environment["XRDPY_AZIMUTHAL_EDGES"] == "-90 -30 30 90"
    assert environment["XRDPY_BEAMLINE"] == ""
    assert environment["XRDPY_POLARIZATION_FACTOR"] == "0.99"
    assert environment["XRDPY_OVERWRITE"] == "1"
    assert report["job_id"] == "48123[].server"
    assert report["n_chunks"] == 24


def test_facility_client_calls_stpy_and_dbpy_directly():
    calls = []

    class Reader:
        def __init__(self, detector_id, beamline, runs):
            calls.append(("reader", detector_id, beamline, runs))

        def collect(self, buffer, tag):
            calls.append(("collect", buffer, tag))

    class Buffer:
        def __init__(self, reader):
            self.reader = reader

        def read_det_data(self, index):
            calls.append(("read", index))
            return np.ones((2, 2))

    stpy_module = SimpleNamespace(StorageReader=Reader, StorageBuffer=Buffer)
    dbpy_module = SimpleNamespace(
        read_hightagnumber=lambda beamline, run: (beamline, run, "high"),
        read_syncdatalist_float=lambda column, high, tags: [2.0 for _ in tags],
    )
    client = single_shot.SaclaFacilityClient(
        beamline=3,
        detector_id="detector",
        stpy_module=stpy_module,
        dbpy_module=dbpy_module,
    )

    np.testing.assert_allclose(client.read_image(10, 1001), np.ones((2, 2)))
    np.testing.assert_allclose(
        client.read_pulse_intensities(10, [1001, 1002], "intensity"),
        [2.0, 2.0],
    )
    assert calls[0] == ("reader", "detector", 3, (10,))
    assert calls[-2][0] == "collect"
    assert calls[-1] == ("read", 0)
