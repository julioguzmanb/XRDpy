from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import QApplication

from trxrdpy.simulation import detector, experiment, polycrystalline
from trxrdpy.simulation import single_crystal as single_crystal_backend
from trxrdpy.simulation.gui.state import GuiState
from trxrdpy.simulation.gui.services.simulation_service import SimulationService
from trxrdpy.simulation.gui.tabs.polycrystalline_tab import PolycrystallineTab
from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab


class _LatticeForTitle:
    phase = None
    a = 1.0
    b = 1.0
    c = 1.0
    alpha = 90.0
    beta = 90.0
    gamma = 90.0
    crystal_orientation = np.eye(3)


def test_poni_detector_mode_uses_edited_values_without_reloading_file():
    det = detector.Detector(
        detector_type="poni",
        pxsize_h=50e-6,
        pxsize_v=50e-6,
        num_pixels_h=20,
        num_pixels_v=10,
        dist=0.1,
        poni1=0.02,
        poni2=0.03,
        rotx=-1.3,
        roty=-30.0,
        rotz=0.0,
        poni_file=None,
    )

    assert det.detector_type == "poni"
    assert det.roty == -30.0


def test_gui_poni_backend_source_keeps_label_but_drops_file_path():
    assert PolycrystallineTab._backend_detector_source("poni", "/tmp/calib.poni") == ("poni", None)
    assert SingleCrystalTab._backend_detector_source("poni", "/tmp/calib.poni") == ("poni", None)


def test_single_crystal_gui_poni_mode_passes_edited_detector_values(monkeypatch):
    app = QApplication.instance() or QApplication([])
    _ = app
    captured = {}

    def fake_simulate_2d(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(single_crystal_backend, "simulate_2d", fake_simulate_2d)

    tab = SingleCrystalTab(SimulationService(), GuiState())
    tab.single_func_combo.setCurrentText("simulate_2d")
    tab.single_combo_det_type.setCurrentText("poni")
    tab.single_geometry_mode_combo.setCurrentText("Legacy Euler")
    tab.single_line_pxsize_h.setText("7.5e-05")
    tab.single_line_pxsize_v.setText("7.5e-05")
    tab.single_line_num_px_h.setText("1475")
    tab.single_line_num_px_v.setText("831")
    tab.single_line_dist.setText("0.1040658008471512")
    tab.single_line_poni1.setText("0.056283783395496195")
    tab.single_line_poni2.setText("0.11760363022346329")
    tab.single_line_rotx.setText("-1.2783314114687627")
    tab.single_line_roty.setText("-30")
    tab.single_line_rotz.setText("0")

    assert tab._single_run_function()

    assert captured["det_type"] == "poni"
    assert captured["det_poni_file"] is None
    assert captured["det_pxsize_h"] == 7.5e-05
    assert captured["det_ntum_pixels_h"] == 1475
    assert captured["det_roty"] == -30.0


def test_detector_title_digits_are_applied_to_simulation_titles():
    det = detector.Detector(
        detector_type="poni",
        pxsize_h=50e-6,
        pxsize_v=50e-6,
        num_pixels_h=20,
        num_pixels_v=10,
        dist=0.1040658008471512,
        poni1=0.056283783395496195,
        poni2=0.11760363022346329,
        rotx=-1.2783314114687627,
        roty=-30.0,
        rotz=0.0,
        poni_file=None,
    )
    exp = experiment.Experiment(det, _LatticeForTitle(), energy=15000.0, e_bandwidth=1.5)

    title = exp._detector_title(digits=3)

    assert "Energy [keV]: 15.000" in title
    assert "dist=0.104" in title
    assert "roty=-30.000" in title


def test_powder_qmax_filter_removes_out_of_range_and_duplicate_families():
    q_hkls, d_hkls, hkls_names = polycrystalline._filter_reflections_by_qmax(
        q_hkls=np.array([1.0, 1.0 + 1e-7, 6.0]),
        hkls_names=np.array([[4, 0, 0], [1, 1, 0], [6, 0, 0]]),
        qmax=5.0,
    )

    assert d_hkls is None
    np.testing.assert_allclose(q_hkls, [1.0 + 1e-7])
    np.testing.assert_array_equal(hkls_names, [[1, 1, 0]])
