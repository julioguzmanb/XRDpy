"""Regression coverage for q, 2theta, and d powder-pattern axes."""
import numpy as np
import pytest
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication

from trxrdpy.simulation import polycrystalline
from trxrdpy.simulation.gui.services.simulation_service import SimulationService
from trxrdpy.simulation.gui.state import GuiState
from trxrdpy.simulation.gui.tabs.polycrystalline_tab import PolycrystallineTab
from trxrdpy.simulation.plot import plot_1d_pattern
from trxrdpy.simulation.sample import convolve_peaks_1d


@pytest.fixture
def peaks():
    return np.array(
        [(1.0, 2 * np.pi, 10.0, 1.0),
         (2.0, np.pi, 20.0, 0.5)],
        dtype=[("q", "f8"), ("d", "f8"), ("two_theta", "f8"), ("I", "f8")],
    )


@pytest.mark.parametrize("axis,centres,label,inverted", [
    ("q", (1.0, 2.0), "q", False),
    ("two_theta", (10.0, 20.0), r"2$\theta$", False),
    ("d", (np.pi, 2 * np.pi), "d", True),
])
def test_gaussian_profile_and_rendered_axis(peaks, axis, centres, label, inverted):
    x, intensity = convolve_peaks_1d(peaks, x_axis=axis, step=0.01, fwhm=0.1)
    assert len(x) == len(intensity) > 0
    assert np.all(np.diff(x) > 0)
    for centre in centres:
        assert intensity[np.argmin(abs(x - centre))] > 0.45
    if axis == "d":
        assert x[0] > 0

    ax = plot_1d_pattern(x, intensity, x_axis=axis, show=False)
    ax.figure.canvas.draw()  # Catches invalid Matplotlib mathtext in 2theta.
    assert label in ax.get_xlabel()
    assert ax.xaxis_inverted() == inverted
    plt.close(ax.figure)


@pytest.mark.parametrize("axis", ["q", "two_theta", "d"])
@pytest.mark.parametrize("fwhm", [None, 0])
def test_zero_width_produces_sticks(peaks, axis, fwhm):
    x, intensity = convolve_peaks_1d(peaks, x_axis=axis, fwhm=fwhm)
    assert len(x) == 3 * len(peaks)
    np.testing.assert_allclose(intensity, [0, 1, 0, 0, 0.5, 0] if axis != "d"
                               else [0, 0.5, 0, 0, 1, 0])
    np.testing.assert_allclose(x[::3], x[1::3])
    np.testing.assert_allclose(x[1::3], x[2::3])


@pytest.mark.parametrize("axis", ["q", "two_theta", "d"])
def test_gui_axis_is_saved_and_sent_to_backend(axis, monkeypatch, wait_simulation):
    app = QApplication.instance() or QApplication([])
    _ = app
    state = GuiState()
    tab = PolycrystallineTab(SimulationService(), state)
    captured = {}

    def fake_simulate_1d(**kwargs):
        captured.update(kwargs)
        return {"x": np.array([]), "I": np.array([]), "peaks": np.array([])}

    monkeypatch.setattr(polycrystalline, "simulate_1d", fake_simulate_1d)
    tab.poly_func_combo.setCurrentText("simulate_1d")
    tab.poly_line_cif_path.setText("sample.cif")
    tab.poly_combo_xaxis.setCurrentText(axis)
    tab.poly_line_fwhm.setText("0")
    assert tab._poly_run_function()
    wait_simulation(tab)
    assert captured["x_axis"] == axis
    assert captured["fwhm"] is None
    assert state.poly.x_axis == axis
    restored = PolycrystallineTab(SimulationService(), GuiState.from_dict(state.to_dict()))
    assert restored.poly_combo_xaxis.currentText() == axis
    restored.close()
    tab.close()
