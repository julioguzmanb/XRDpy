from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from trxrdpy.analysis.gui.main_window import AnalysisMainWindow
from trxrdpy.analysis.gui.services import PathService
from trxrdpy.analysis.gui.state import AnalysisGuiState
from trxrdpy.analysis.gui.tabs.pattern_creation_tab import PatternCreationTab
from trxrdpy.analysis.gui.tabs.preparation_tab import PreparationTab
from trxrdpy.analysis.gui.widgets.task_output_dialog import (
    TaskOutputDialog,
    _ThreadRoutingStream,
)


class _SlowFemtoPreparationService:
    def __init__(self, reference_path: Path):
        self.reference_path = reference_path
        self.started = threading.Event()
        self.release = threading.Event()

    def default_femtomax_ping_reference_path(self):
        return str(self.reference_path)

    def validate_femtomax_ping_reference_file(self, *_args, **_kwargs):
        return SimpleNamespace(
            path=self.reference_path,
            ranges=[(1, 2)],
            scan_min=1,
            scan_max=2,
            sha256="0" * 64,
        )

    def generate_femtomax_2d_images(self, **_kwargs):
        self.started.set()
        self.release.wait(timeout=5.0)
        return object(), {100: {"path": "image.npy", "n_images": 2}}


class _ConcurrentSingleShotIntegrationService:
    def __init__(self):
        self.production_started = threading.Event()
        self.aggregation_started = threading.Event()
        self.release = threading.Event()
        self.single_shot_build_kwargs = None
        self.final_integration_kwargs = None

    def parse_azim_offset_deg(self, value):
        return float(value)

    def build_single_shot_integration_kwargs(self, **_kwargs):
        self.single_shot_build_kwargs = dict(_kwargs)
        return {}

    def integrate_single_shot_1d(self, **_kwargs):
        self.production_started.set()
        self.release.wait(timeout=5.0)
        return {
            "written_patterns": 1,
            "existing_patterns": 0,
            "invalid_shots": 0,
        }

    def build_delay_integration_kwargs(self, **_kwargs):
        return {}

    def build_parallel_integration_kwargs(
        self,
        *,
        use_parallel,
        max_workers_text,
    ):
        return {
            "use_parallel": bool(use_parallel),
            "max_workers": int(max_workers_text),
        }

    def integrate_delay_1d(self, **_kwargs):
        self.final_integration_kwargs = dict(_kwargs)
        self.aggregation_started.set()
        return None, {100: object()}


def _process_events_until(app, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


def test_femtomax_2d_creation_keeps_gui_event_loop_responsive(
    tmp_path,
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(TaskOutputDialog, "show", lambda self: None)
    reference_path = tmp_path / "ping_ranges.csv"
    reference_path.write_text("start,end\n1,2\n", encoding="utf-8")
    service = _SlowFemtoPreparationService(reference_path)
    state = AnalysisGuiState(
        facility="FemtoMAX",
        path_root=tmp_path,
        raw_subdir="raw",
        analysis_subdir="analysis",
    )
    tab = PreparationTab(state, PathService(tmp_path), service)
    tab._build_femtomax_common_kwargs = lambda: {}

    started_at = time.monotonic()
    tab._create_femtomax_2d_images()
    launch_duration = time.monotonic() - started_at
    assert launch_duration < 0.25
    assert _process_events_until(app, service.started.is_set)

    heartbeat = []
    QTimer.singleShot(0, lambda: heartbeat.append(True))
    assert _process_events_until(app, lambda: bool(heartbeat), timeout=1.0)

    service.release.set()
    assert _process_events_until(
        app,
        lambda: all(
            not dialog._running
            for dialog in getattr(tab, "_active_task_output_dialogs", [])
        ),
    )
    for dialog in list(getattr(tab, "_active_task_output_dialogs", [])):
        dialog.close()
    tab.deleteLater()
    app.processEvents()


def test_thread_routing_stream_keeps_concurrent_task_output_separate():
    fallback = io.StringIO()
    router = _ThreadRoutingStream(fallback)
    barrier = threading.Barrier(2)
    outputs = {"first": [], "second": []}

    def write_for_task(name):
        token = router.register_current_thread(outputs[name].append)
        try:
            barrier.wait(timeout=2.0)
            router.write(name + "\n")
        finally:
            router.unregister(token)

    threads = [
        threading.Thread(target=write_for_task, args=(name,))
        for name in outputs
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert not any(thread.is_alive() for thread in threads)
    assert outputs == {"first": ["first\n"], "second": ["second\n"]}
    assert fallback.getvalue() == ""


def test_thread_routing_stream_can_tee_task_output_to_terminal():
    fallback = io.StringIO()
    router = _ThreadRoutingStream(fallback)
    captured = []
    token = router.register_current_thread(captured.append, tee=True)
    try:
        router.write("progress\n")
    finally:
        router.unregister(token)

    assert captured == ["progress\n"]
    assert fallback.getvalue() == "progress\n"


def test_task_output_dialog_exposes_cooperative_safe_stop():
    app = QApplication.instance() or QApplication([])
    cancelled = threading.Event()
    dialog = TaskOutputDialog(cancel_callback=cancelled.set)

    assert dialog.stop_button.isVisibleTo(dialog)
    dialog.request_cancel()

    assert cancelled.is_set()
    assert dialog._cancel_requested is True
    assert not dialog.stop_button.isEnabled()
    dialog.mark_cancelled()
    dialog.close()
    app.processEvents()


def test_final_patterns_can_refresh_while_single_shot_production_runs(
    tmp_path,
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(TaskOutputDialog, "show", lambda self: None)
    reference_path = tmp_path / "ping_ranges.csv"
    reference_path.write_text("start,end\n1,2\n", encoding="utf-8")
    preparation_service = _SlowFemtoPreparationService(reference_path)
    integration_service = _ConcurrentSingleShotIntegrationService()
    state = AnalysisGuiState(
        facility="FemtoMAX",
        path_root=tmp_path,
        raw_subdir="raw",
        analysis_subdir="analysis",
        poni_path=tmp_path / "geometry.poni",
        mask_edf_path=tmp_path / "mask.edf",
    )
    paths = PathService(tmp_path)
    datared_tab = PreparationTab(
        state,
        paths,
        preparation_service,
        integration_service=integration_service,
    )
    pattern_tab = PatternCreationTab(state, paths, integration_service)
    assert pattern_tab.pattern_use_parallel.isChecked()
    assert pattern_tab.pattern_max_workers.text() == "4"
    metadata_path = tmp_path / "metadata.h5"
    datared_tab.datared_single_shot_metadata_h5.setText(str(metadata_path))
    datared_tab.datared_single_shot_azim_offset_deg.setText("-87.5")
    datared_tab.datared_single_shot_polarization_control.set_configuration(
        enabled=False,
        factor=0.8,
        emit=True,
    )
    pattern_tab.pattern_single_shot_metadata_h5.setText(str(metadata_path))
    pattern_tab.pattern_input_mode_combo.setCurrentText(
        "Single-shot 1D patterns"
    )

    datared_tab._integrate_single_shot_1d()
    assert _process_events_until(
        app,
        integration_service.production_started.is_set,
    )
    assert integration_service.single_shot_build_kwargs[
        "femtomax_use_parallel"
    ] is True
    assert integration_service.single_shot_build_kwargs[
        "femtomax_max_workers_text"
    ] == "4"
    assert integration_service.single_shot_build_kwargs[
        "femtomax_start_method"
    ] == "spawn"
    assert integration_service.single_shot_build_kwargs[
        "femtomax_work_chunk_size_text"
    ] == "64"
    assert str(
        integration_service.single_shot_build_kwargs["poni_path"]
    ).endswith("geometry.poni")
    assert str(
        integration_service.single_shot_build_kwargs["mask_edf_path"]
    ).endswith("mask.edf")
    assert integration_service.single_shot_build_kwargs[
        "polarization_factor"
    ] is None
    assert integration_service.single_shot_build_kwargs[
        "azim_offset_deg"
    ] == -87.5

    pattern_tab._integrate_delay_1d()
    assert _process_events_until(
        app,
        integration_service.aggregation_started.is_set,
    )
    assert integration_service.final_integration_kwargs["use_parallel"] is True
    assert integration_service.final_integration_kwargs["max_workers"] == 4
    assert any(
        dialog._running
        for dialog in getattr(datared_tab, "_active_task_output_dialogs", [])
    )

    integration_service.release.set()
    assert _process_events_until(
        app,
        lambda: all(
            not dialog._running
            for tab in (datared_tab, pattern_tab)
            for dialog in getattr(tab, "_active_task_output_dialogs", [])
        ),
    )
    for tab in (datared_tab, pattern_tab):
        for dialog in list(getattr(tab, "_active_task_output_dialogs", [])):
            dialog.close()
        tab.deleteLater()
    app.processEvents()


def test_main_window_reorganization_syncs_and_migrates_single_shot_state(
    tmp_path,
):
    app = QApplication.instance() or QApplication([])
    window = AnalysisMainWindow(launch_directory=tmp_path)

    assert [
        window.tabs.tabText(index).replace("\n", " ")
        for index in range(window.tabs.count())
    ][:4] == [
        "Session",
        "Data Reduction",
        "Calibration",
        "Azimuthal Integration",
    ]

    window.preparation_tab.datared_single_shot_metadata_h5.setText(
        str(tmp_path / "current.h5")
    )
    assert window.pattern_creation_tab.pattern_single_shot_metadata_h5.text() == (
        str(tmp_path / "current.h5")
    )
    window.pattern_creation_tab.pattern_npt.setText("2048")
    assert window.preparation_tab.datared_single_shot_npt.text() == "2048"
    window.preparation_tab.datared_single_shot_normalize_final.setChecked(False)
    assert not window.pattern_creation_tab.pattern_normalize_checkbox.isChecked()
    window.preparation_tab.datared_single_shot_q_norm_range.setText("(2.1, 2.2)")
    window.preparation_tab.datared_single_shot_q_norm_range.editingFinished.emit()
    assert window.pattern_creation_tab.pattern_q_norm_range.text() == "(2.1, 2.2)"
    assert window.viewer_tab.viewer_q_norm_range.text() == "(2.1, 2.2)"

    poni_path = str(tmp_path / "shared.poni")
    mask_path = str(tmp_path / "shared.edf")
    window.session_tab.session_poni_path.setText(poni_path)
    window.session_tab.session_mask_path.setText(mask_path)
    window.session_tab.session_poni_path.editingFinished.emit()
    window.session_tab.session_mask_path.editingFinished.emit()
    assert window.state.poni_path == Path(poni_path)
    assert window.state.mask_edf_path == Path(mask_path)
    assert not hasattr(
        window.preparation_tab,
        "datared_single_shot_poni_path",
    )
    assert not hasattr(
        window.preparation_tab,
        "datared_single_shot_mask_path",
    )
    window.preparation_tab.datared_single_shot_azim_offset_deg.setText("-82.5")
    assert window.session_tab.session_azim_offset_deg.text() == "-82.5"

    window.preparation_tab.datared_single_shot_polarization_control.set_configuration(
        enabled=False,
        factor=0.75,
        emit=True,
    )
    assert not (
        window.pattern_creation_tab.pattern_polarization_control
        .enabled_checkbox.isChecked()
    )
    assert (
        window.pattern_creation_tab.pattern_polarization_control.factor()
        == 0.75
    )

    old_state = {
        "state_version": 1,
        "tabs": {
            "session": {
                "session_poni_path": {
                    "type": "QLineEdit",
                    "value": "",
                },
                "session_mask_path": {
                    "type": "QLineEdit",
                    "value": "",
                },
            },
            "preparation": {},
            "datared": {
                "datared_metadata": {
                    "type": "ValueWidget",
                    "value": {
                        "sample_name": "old-preparation",
                        "temperature_K": "175",
                        "excitation_wl_nm": "800",
                        "fluence_mJ_cm2": "3.5",
                        "time_window_fs": "500",
                    },
                },
                "datared_femto_use_parallel": {
                    "type": "QCheckBox",
                    "value": False,
                },
                "datared_femto_max_workers": {
                    "type": "QLineEdit",
                    "value": "7",
                },
                "datared_femto_start_method": {
                    "type": "QComboBox",
                    "value": "forkserver",
                },
                "datared_single_shot_poni_path": {
                    "type": "QLineEdit",
                    "value": str(tmp_path / "old.poni"),
                },
                "datared_single_shot_mask_path": {
                    "type": "QLineEdit",
                    "value": str(tmp_path / "old.edf"),
                },
            },
            "pattern": {
                "pattern_metadata": {
                    "type": "ValueWidget",
                    "value": {
                        "sample_name": "old-pattern",
                        "temperature_K": "180",
                        "excitation_wl_nm": "1500",
                        "fluence_mJ_cm2": "12",
                        "time_window_fs": "250",
                    },
                },
                "pattern_single_shot_metadata_h5": {
                    "type": "QLineEdit",
                    "value": str(tmp_path / "old.h5"),
                },
                "pattern_femtomax_read_batch_size": {
                    "type": "QLineEdit",
                    "value": "32",
                },
                "pattern_overwrite_single_shot": {
                    "type": "QCheckBox",
                    "value": True,
                },
            }
        },
    }
    window._apply_gui_state(old_state)
    assert window.pattern_creation_tab.pattern_use_parallel.isChecked()
    assert window.pattern_creation_tab.pattern_max_workers.text() == "4"
    assert window.preparation_tab.experiment_metadata.values() == {
        "sample_name": "old-preparation",
        "temperature_K": "175",
        "excitation_wl_nm": "800",
        "fluence_mJ_cm2": "3.5",
        "time_window_fs": "500",
        "raw_sample_name": "",
        "dataset": "3",
        "scan_nb": "7",
    }
    assert window.pattern_creation_tab.experiment_metadata.values()[
        "sample_name"
    ] == "old-pattern"
    assert window.preparation_tab.datared_single_shot_metadata_h5.text() == (
        str(tmp_path / "old.h5")
    )
    assert window.preparation_tab.datared_femtomax_read_batch_size.text() == "32"
    assert window.preparation_tab.datared_overwrite_single_shot.isChecked()
    assert not (
        window.preparation_tab.datared_femtomax_single_shot_use_parallel.isChecked()
    )
    assert (
        window.preparation_tab.datared_femtomax_single_shot_max_workers.text()
        == "7"
    )
    assert (
        window.preparation_tab.datared_femtomax_single_shot_start_method.currentText()
        == "forkserver"
    )
    assert window.session_tab.session_poni_path.text() == str(tmp_path / "old.poni")
    assert window.session_tab.session_mask_path.text() == str(tmp_path / "old.edf")
    assert window.state.poni_path == tmp_path / "old.poni"
    assert window.state.mask_edf_path == tmp_path / "old.edf"

    collected = window._collect_gui_state()["tabs"]
    assert "datared_femtomax_read_batch_size" in collected["preparation"]
    assert "datared_single_shot_poni_path" not in collected["preparation"]
    assert "datared_single_shot_mask_path" not in collected["preparation"]
    assert "datared_single_shot_q_norm_range" in collected["preparation"]
    assert (
        "datared_single_shot_polarization_control"
        in collected["preparation"]
    )
    assert "pattern_femtomax_read_batch_size" not in collected["pattern"]

    legacy_state = {
        "session": {"facility": "ID09"},
        "datared": {
            "experiment": {
                "sample_name": "legacy-sample",
                "temperature_K": "95",
                "excitation_wl_nm": "1200",
                "fluence_mJ_cm2": "6.25",
                "time_window_fs": "1000",
                "raw_sample_name": "legacy-raw",
                "dataset": "8",
                "scan_nb": "42",
            }
        },
        "pattern": {
            "experiment": {
                "sample_name": "legacy-pattern",
                "temperature_K": "100",
                "excitation_wl_nm": "1300",
                "fluence_mJ_cm2": "7.5",
                "time_window_fs": "750",
                "raw_sample_name": "legacy-pattern-raw",
                "dataset": "9",
                "scan_nb": "43",
            }
        },
    }
    window._apply_gui_state(legacy_state)
    assert window.preparation_tab.experiment_metadata.values() == {
        "sample_name": "legacy-sample",
        "temperature_K": "95",
        "excitation_wl_nm": "1200",
        "fluence_mJ_cm2": "6.25",
        "time_window_fs": "1000",
        "raw_sample_name": "legacy-raw",
        "dataset": "8",
        "scan_nb": "42",
    }
    assert window.pattern_creation_tab.experiment_metadata.values() == {
        "sample_name": "legacy-pattern",
        "temperature_K": "100",
        "excitation_wl_nm": "1300",
        "fluence_mJ_cm2": "7.5",
        "time_window_fs": "750",
        "raw_sample_name": "legacy-pattern-raw",
        "dataset": "9",
        "scan_nb": "43",
    }

    window._allow_close_without_confirmation = True
    window.close()
    app.processEvents()
