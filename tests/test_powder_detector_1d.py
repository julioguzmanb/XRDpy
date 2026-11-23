"""Detector-limited 1D powder patterns and shared GUI detector settings."""
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from trxrdpy.simulation import polycrystalline
from trxrdpy.simulation.detector import Detector
from trxrdpy.simulation.detector_coverage import detector_qmax, ring_coverage
from trxrdpy.simulation.gui.main_window import MainWindow
from trxrdpy.simulation.gui.tabs.polycrystalline_tab import _parse_azimuth_range
from trxrdpy.simulation.utils import energy_to_wavelength


def detector(**overrides):
    settings = dict(detector_type="manual", pxsize_h=0.001, pxsize_v=0.001,
                    num_pixels_h=100, num_pixels_v=100, dist=0.1,
                    poni1=0.05, poni2=0.05)
    settings.update(overrides)
    return Detector(**settings)


@pytest.mark.parametrize("text", ["-15,15", "(-15, 15)", "[-15, 15]"])
def test_azimuth_field_accepts_common_pair_formats(text):
    assert _parse_azimuth_range(text) == (-15.0, 15.0)


def test_detector_q_limit_and_lab_azimuth():
    det = detector()
    wavelength = energy_to_wavelength(10000)
    full = detector_qmax(det, wavelength)
    top = detector_qmax(det, wavelength, (-10, 10))
    assert 0 < top < full
    assert detector_qmax(det, wavelength, (0, 20), 10) == pytest.approx(
        detector_qmax(det, wavelength, (10, 30)))
    q = 0.8 * top
    assert ring_coverage([q], det, wavelength, (-10, 10))[0] == pytest.approx(1)
    assert 0 < ring_coverage([0.8 * full], det, wavelength)[0] < 1
    assert ring_coverage([full * 1.01], det, wavelength)[0] == 0

    left_detector = detector(poni2=0.15)
    assert detector_qmax(left_detector, wavelength, (-100, -80)) > 0
    with pytest.raises(ValueError, match="does not intersect"):
        detector_qmax(left_detector, wavelength, (80, 100))


def test_detector_1d_filters_without_diluting_visible_peaks(monkeypatch):
    det = detector()
    wavelength = energy_to_wavelength(10000)
    limit = detector_qmax(det, wavelength)
    qs = np.array([limit * 0.3, limit * 0.77, limit * 1.02])
    peaks = np.zeros(3, dtype=[("q", "f8"), ("d", "f8"),
                               ("two_theta", "f8"), ("I", "f8")])
    peaks["q"] = qs
    peaks["d"] = 2 * np.pi / qs
    peaks["I"] = [1, 0.5, 0.3]
    seen = {}

    class FakeLattice:
        def __init__(self, **kwargs):
            pass

        def simulate_1d_pattern(self, **kwargs):
            seen.update(kwargs)
            assert kwargs["convolve"] is False
            return {"x": np.array([]), "I": np.array([]), "peaks": peaks.copy()}

    monkeypatch.setattr(polycrystalline.sample, "LatticeStructure", FakeLattice)
    result = polycrystalline.simulate_1d(
        "fake.cif", qmax=0.01, energy=10000, detector=det,
        convolve=False, plot_result=False)
    assert seen["qmax"] == pytest.approx(limit)
    np.testing.assert_allclose(result["peaks"]["q"], qs[:2])
    assert result["coverage_fraction"][0] > result["coverage_fraction"][1]
    np.testing.assert_allclose(result["peaks"]["I"], [1, 0.5])

    sector = polycrystalline.simulate_1d(
        "fake.cif", energy=10000, detector=det,
        azimuthal_range=(-15, 15), convolve=False, plot_result=False)
    np.testing.assert_allclose(sector["peaks"]["q"], qs[:2])
    np.testing.assert_allclose(sector["peaks"]["I"], result["peaks"]["I"])
    assert sector["coverage_fraction"][1] != pytest.approx(result["coverage_fraction"][1])

    titles = []
    monkeypatch.setattr(polycrystalline.plot, "plot_1d_pattern",
                        lambda *args, **kwargs: titles.append(kwargs["title"]))
    profile = polycrystalline.simulate_1d(
        "fake.cif", energy=10000, detector=det, step=0.01,
        fwhm=0.1, plot_result=True)
    assert profile["x"][-1] >= profile["detector_qmax"]
    polycrystalline.simulate_1d(
        "fake.cif", energy=10000, detector=det, step=0.01,
        fwhm=0.1, azimuthal_range=(-15, 15), azimuth_offset_deg=5,
        plot_result=True)
    assert "Azimuth: full 360°" in titles[0]
    assert "Azimuth: [-15°, +15°] (offset +5°)" in titles[1]


def test_gui_detector_mode_and_cross_tab_settings(tmp_path, monkeypatch, wait_simulation):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(MainWindow, "_maybe_restore_autosave", lambda self: None)
    window = MainWindow(launch_directory=tmp_path)
    try:
        poly = window.poly_tab
        single = window.single_tab
        window.tabs.setCurrentWidget(poly)
        poly.poly_chk_use_detector_1d.setChecked(True)
        poly.poly_line_cif_path.setText("fake.cif")
        poly.poly_line_qmax.setText("")
        poly.poly_line_dist.setText("0.27")
        poly.poly_line_poni2.setText("0.031")
        poly.poly_line_roty.setText("13")
        poly.poly_combo_rotation_order.setCurrentText("xyz")
        poly.poly_line_azimuth_range_1d.setText("(-30, 45)")
        poly.poly_line_azimuth_offset_1d.setText("12")
        assert not poly.poly_det_group.isHidden()
        assert not poly.poly_line_qmax.isEnabled()

        received = {}
        monkeypatch.setattr(polycrystalline, "simulate_1d", lambda **kw: received.update(kw))
        assert poly._poly_run_function()
        wait_simulation(poly)
        assert received["qmax"] is None
        assert received["azimuthal_range"] == (-30, 45)
        assert received["azimuth_offset_deg"] == 12
        assert received["detector"].dist == pytest.approx(0.27)

        window.tabs.setCurrentWidget(single)
        assert single.single_line_dist.text() == "0.27"
        assert single.single_line_poni2.text() == "0.031"
        assert single.single_line_roty.text() == "13"
        assert single.single_combo_det_rotation_order.currentText() == "xyz"
        single.single_line_poni1.setText("0.014")
        window.tabs.setCurrentWidget(poly)
        assert poly.poly_line_poni1.text() == "0.014"
    finally:
        window.close()
        app.processEvents()
