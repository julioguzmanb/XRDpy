from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PyQt5.QtCore import QMimeData, QPointF, Qt, QUrl
from PyQt5.QtGui import QDropEvent
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QMessageBox
from scipy.spatial.transform import Rotation

from trxrdpy.simulation.detector import Detector
from trxrdpy.simulation.experiment import Experiment, lab_to_pixel_coordinates_with_transform
from trxrdpy.simulation.gui.main_window import MainWindow
from trxrdpy.simulation.gui.state import GUI_STATE_VERSION, GuiState
from trxrdpy.simulation.gui.widgets.equivalent_reflections_dialog import EquivalentReflectionsDialog
from trxrdpy.simulation.gui.widgets.geometry_conversion_dialog import (
    GeometryConversionDialog, fit2d_to_pyfai, pyfai_to_fit2d,
)
from trxrdpy.simulation.gui.widgets.scattering_conversion_dialog import ScatteringConversionDialog
from trxrdpy.simulation.plot import plot_2d_detector_polycrystal, plot_2d_detector_single_xstal
from trxrdpy.simulation.sample import LatticeStructure
from trxrdpy.simulation.poni import read_poni_file
from trxrdpy.simulation.utils import make_transform, convert_scattering_coordinate


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    monkeypatch.setattr(MainWindow, "_maybe_restore_autosave", lambda self: None)
    monkeypatch.setattr(MainWindow, "_autosave_path", lambda self: tmp_path / "autosave.json")
    window = MainWindow(launch_directory=tmp_path)
    yield window
    window.close()
    window.deleteLater()
    app.processEvents()


def make_detector(**kwargs):
    values = dict(detector_type="manual", pxsize_h=0.0001, pxsize_v=0.0002,
                  num_pixels_h=1000, num_pixels_v=500,
                  dist=0.1, poni1=0.05, poni2=0.05)
    values.update(kwargs)
    return Detector(**values)


def test_direct_beam_center_and_binning():
    np.testing.assert_allclose(make_detector().direct_beam_pixel(), [499.5, 249.5])
    np.testing.assert_allclose(make_detector(binning=(2, 2)).direct_beam_pixel(), [249.5, 124.5])
    np.testing.assert_allclose(make_detector(poni1=0, poni2=0).direct_beam_pixel(), [-0.5, -0.5])


@pytest.mark.parametrize("angles,translation", [
    ([0, 12, 0], [0, 0, 0]),
    ([4, -8, 3], [0.01, 0.003, -0.002]),
])
def test_direct_beam_matches_ray_plane_and_pixel_mapping(angles, translation):
    detector = make_detector()
    rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    transform = make_transform(rotation, translation)
    detector.set_transform(transform)
    normal = rotation[:, 0]
    plane_point = transform @ np.array([detector.dist, 0, 0, 1])
    distance = np.dot(normal, plane_point[:3]) / normal[0]
    expected = lab_to_pixel_coordinates_with_transform(
        np.array([[distance, 0, 0]]), detector.dist, detector.pxsize_h, detector.pxsize_v,
        detector.poni1, detector.poni2, transform)
    np.testing.assert_allclose(detector.direct_beam_pixel(), expected[0])


@pytest.mark.parametrize("angles,translation", [
    ([0, 90, 0], [0, 0, 0]),
    ([0, 180, 0], [0, 0, 0]),
    ([0, 0, 0], [0, 1, 0]),
    ([0, 0, 0], [-0.2, 0, 0]),
])
def test_direct_beam_misses_are_not_plotted(angles, translation):
    detector = make_detector()
    detector.set_transform(make_transform(
        Rotation.from_euler("xyz", angles, degrees=True).as_matrix(), translation))
    assert detector.direct_beam_pixel() is None


@pytest.mark.parametrize("plotter", [plot_2d_detector_polycrystal, plot_2d_detector_single_xstal])
@pytest.mark.parametrize("miss", [False, True])
def test_2d_marker_without_any_reflections(plotter, miss, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    detector = make_detector(poni2=1 if miss else 0.05)
    axes = plotter(detector, np.empty((0, 2)), np.empty(0), np.empty((0, 3)), {})
    markers = [line for line in axes.lines if line.get_gid() == "direct-beam"]
    assert len(markers) == (0 if miss else 1)
    if markers:
        assert markers[0].get_color() == "red"
        assert markers[0].get_marker() == "x"
        np.testing.assert_allclose(markers[0].get_xydata()[0], [499.5, 249.5])
    plt.close(axes.figure)


@pytest.mark.parametrize("plotter", [plot_2d_detector_polycrystal, plot_2d_detector_single_xstal])
@pytest.mark.parametrize("pixels_h,pixels_v,pixel_h,pixel_v,binning", [
    (3840, 2712, 4.42708e-5, 4.42708e-5, (1, 1)),
    (1200, 800, 5e-5, 7.5e-5, (2, 1)),
])
def test_2d_detector_plot_preserves_physical_pixel_scale(
    plotter, pixels_h, pixels_v, pixel_h, pixel_v, binning, monkeypatch
):
    monkeypatch.setattr(plt, "show", lambda: None)
    detector = make_detector(
        num_pixels_h=pixels_h, num_pixels_v=pixels_v,
        pxsize_h=pixel_h, pxsize_v=pixel_v, binning=binning,
    )
    axes = plotter(detector, np.empty((0, 2)), np.empty(0), np.empty((0, 3)), {})
    axes.figure.canvas.draw()
    origin, right, up = axes.transData.transform([[0, 0], [100, 0], [0, 100]])
    screen_ratio = (up[1] - origin[1]) / (right[0] - origin[0])
    assert screen_ratio == pytest.approx(detector.pxsize_v / detector.pxsize_h)
    plt.close(axes.figure)


@pytest.mark.parametrize("powder", [False, True])
def test_plot_uses_the_same_transform_override_as_reflections(powder, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    detector = make_detector()
    lattice = LatticeStructure(space_group=221, a=4, b=4, c=4, alpha=90, beta=90, gamma=90)
    exp = Experiment(detector, lattice, energy=15000, e_bandwidth=1.5)
    transform = make_transform(translation=[0, 0.01, 0.005])
    fig, axes = plt.subplots()
    if powder:
        exp.calculate_cones_pixel_positions(np.array([1.0]), np.array([[1, 0, 0]]),
                                            num_points=24, detector_transform=transform)
        exp.plot_2d_polycrystal_exp(ax=axes)
    else:
        exp.scattered_directions = np.array([[0.1, 0, 0]])
        exp.scat_dir_sign = np.array([1])
        exp.hkl_to_color = {(1, 0, 0): "blue"}
        lattice.hkls_in_Bragg_condition = np.array([[1, 0, 0]])
        exp.calculate_pixel_positions(detector_transform=transform)
        exp.plot_2d_single_xstal_exp(ax=axes)
    marker = next(line for line in axes.lines if line.get_gid() == "direct-beam")
    np.testing.assert_allclose(marker.get_xydata()[0], [599.5, 224.5])
    np.testing.assert_allclose(detector.direct_beam_pixel(), [499.5, 249.5])
    plt.close(fig)


@pytest.mark.parametrize("tilt,plane", [(0, 0), (30, 42), (15, -77)])
def test_fit2d_roundtrip_and_axis_units(tilt, plane):
    original = dict(directDist=160, centerX=700, centerY=400, tilt=tilt,
                    tiltPlanRotation=plane, pixelX=75, pixelY=100)
    pyfai = fit2d_to_pyfai(original)
    assert pyfai["pixel1"] == pytest.approx(100e-6)
    assert pyfai["pixel2"] == pytest.approx(75e-6)
    assert pyfai["dist"] == pytest.approx(0.160 * np.cos(np.radians(tilt)))
    converted = pyfai_to_fit2d(pyfai)
    for name, value in original.items():
        assert converted[name] == pytest.approx(value, abs=1e-8)


def test_conversion_errors_do_not_overwrite_output(app):
    dialog = GeometryConversionDialog()
    before = dialog.fields["centerX"].text()
    dialog.fields["pixel1"].setText("0")
    assert dialog.convert(to_fit2d=True) is None
    assert "Error:" in dialog.status.text()
    assert dialog.fields["centerX"].text() == before
    dialog.fields["pixel1"].setText("nan")
    assert dialog.convert(to_fit2d=True) is None
    dialog.fields["tilt"].setText("90")
    assert dialog.convert(to_fit2d=False) is None
    dialog.close()


def drop_file(field, path):
    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(path))])
    event = QDropEvent(QPointF(5, 5), Qt.CopyAction, data, Qt.LeftButton, Qt.NoModifier)
    field.dropEvent(event)
    return event.isAccepted()


def test_conversion_accepts_poni_drop(app, tmp_path):
    path = tmp_path / "calibration.poni"
    path.write_text('poni_version: 2\nDetector: Detector\n'
                    'Detector_config: {"pixel1": 0.0001, "pixel2": 0.000075}\n'
                    'Distance: 0.16\nPoni1: 0.04\nPoni2: 0.0525\n'
                    'Rot1: 0\nRot2: 0\nRot3: 0\nWavelength: 1e-10\n')
    dialog = GeometryConversionDialog()
    assert drop_file(dialog.poni_path, path)
    assert float(dialog.fields["dist"].text()) == 0.16
    result = dialog.convert(to_fit2d=True)
    assert result["centerX"] == pytest.approx(700)
    assert result["centerY"] == pytest.approx(400)
    dialog.close()


def test_equivalent_reflections_uses_space_group(app):
    dialog = EquivalentReflectionsDialog()
    dialog.set_values(dict(space_group="221", a="4", b="4", c="4",
                           alpha="90", beta="90", gamma="90", hkl="(1, 0, 0)"))
    assert set(dialog.calculate()) == {(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                       (0, -1, 0), (0, 0, 1), (0, 0, -1)}
    for invalid in ("(0, 0, 0)", "(1.2, 0, 0)", "(1, 2)", "invalid"):
        dialog.fields["hkl"].setText(invalid)
        assert dialog.calculate() is None
        assert dialog.output.toPlainText() == ""
        assert "Error:" in dialog.status.text()
    dialog.close()


def test_new_gui_state_fields_and_legacy_session_loading(window, tmp_path):
    old = {"state_version": 10, "ui": {"current_tab_index": 2, "session_notes": "legacy"},
           "single": {"a": "7.5"}, "paths": {"single_cif_file_path": "/old/path.cif"}}
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old))
    assert drop_file(window.state_path_line, path)
    assert window.tabs.currentWidget() is window.session_page
    assert window.session_notes.toPlainText() == "legacy"
    assert window.single_tab.single_line_sam_a.text() == "7.5"
    assert window.single_tab.single_line_cif_path.text() == "/old/path.cif"
    window.equivalent_window.fields["hkl"].setText("(3, 1, 1)")
    window.conversion_window.fields["dist"].setText("0.42")
    window.conversion_window.poni_path.setText("/calibration/detector.poni")
    section = next(s for s in window.collapsible_sections if s.key == "single/Beam Parameters")
    section.set_collapsed(True)
    saved = window._gather_gui_state()
    assert saved["state_version"] == GUI_STATE_VERSION
    assert saved["ui"]["collapsed_sections"] == [section.key]
    section.set_collapsed(False)
    window.conversion_window.fields["dist"].setText("0.1")
    window._apply_gui_state(saved)
    assert section.collapsed
    assert window.equivalent_window.fields["hkl"].text() == "(3, 1, 1)"
    assert window.conversion_window.fields["dist"].text() == "0.42"
    assert window.conversion_window.poni_path.text() == "/calibration/detector.poni"
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        "Session", "Polycrystalline", "Single Crystal", "Extra Functions"]


def test_attenuation_inputs_round_trip_in_gui_state(window):
    dialog = window.attenuation_window
    dialog.formula.setText("SrTiO3")
    dialog.density.setText("5.12")
    dialog.start.setText("10000")
    dialog.stop.setText("15000")
    dialog.step.setText("1")
    dialog.angle_scan.setChecked(True)
    dialog.angle_start.setText("0.5")
    dialog.angle_stop.setText("45")
    dialog.angle_step.setText("0.01")
    dialog.slider_motor.setCurrentText("Photon energy")
    dialog.unit.setCurrentText("nm")
    dialog.log_y.setChecked(True)
    saved = window._gather_gui_state()
    assert saved["ui"]["attenuation"]["angle_step"] == "0.01"
    assert saved["ui"]["attenuation"]["slider_motor"] == "Photon energy"
    assert "length" not in saved["ui"]["attenuation"]
    dialog.formula.clear()
    dialog.angle_scan.setChecked(False)
    window._apply_gui_state(saved)
    assert dialog.formula.text() == "SrTiO3"
    assert dialog.angle_scan.isChecked()
    assert dialog.angle_stop.text() == "45"
    assert dialog.slider_motor.currentText() == "Photon energy"
    assert dialog.unit.currentText() == "nm"


def test_attenuation_hdf5_export_load_keeps_window_unit(window, monkeypatch, tmp_path):
    from trxrdpy.simulation.attenuation_plot import AttenuationData
    dialog = window.attenuation_window
    dialog.unit.setCurrentText("nm")
    dialog.data = AttenuationData(np.array([10000., 10010.]),
                                  np.array([[1e-6, 1.1e-6]]), np.array([1.]),
                                  "grazing", "nm", "SrTiO3", 5.12)
    dialog.csv_button.setEnabled(True)
    path = tmp_path / "scan.h5"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), "HDF5 files (*.h5 *.hdf5)"))
    dialog.save_data()
    assert path.exists()
    dialog.unit.setCurrentText("µm")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), "HDF5 files (*.h5 *.hdf5)"))
    dialog.load_data()
    assert dialog.unit.currentText() == "µm"
    assert dialog.data.length is None
    np.testing.assert_allclose(dialog.data.curve_at(1), [1e-6, 1.1e-6])


def test_extra_functions_launchers_and_footer(window, app):
    window.tabs.setCurrentWidget(window.extra_functions_tab)
    window.show()
    app.processEvents()
    for button, dialog in (
        (window.open_rotation_button, window.matrix_window),
        (window.open_equivalent_button, window.equivalent_window),
        (window.open_scattering_button, window.scattering_window),
        (window.open_conversion_button, window.conversion_window),
    ):
        assert window.extra_functions_tab.isAncestorOf(button)
        button.click()
        assert dialog.isVisible()
        dialog.hide()
    footer = window.centralWidget().layout().itemAt(1).layout()
    assert footer.count() == 2
    assert footer.itemAt(0).spacerItem() is not None
    assert footer.itemAt(1).widget() is window.close_all_plots_button
    figure = plt.figure()
    window.close_all_plots_button.click()
    assert not plt.fignum_exists(figure.number)


@pytest.mark.parametrize("prefix", ["poly", "single"])
def test_extra_functions_keep_simulation_context_and_state(window, prefix):
    tab = getattr(window, prefix + "_tab")
    window.tabs.setCurrentWidget(tab)
    getattr(tab, prefix + "_line_energy").setText("12000")
    getattr(tab, prefix + "_line_sam_a").setText("7.5")
    window.tabs.setCurrentWidget(window.extra_functions_tab)
    saved = window._gather_gui_state()
    assert saved["ui"]["active_tab"] == "extra_functions"
    window.tabs.setCurrentWidget(window.single_tab if prefix == "poly" else window.poly_tab)
    window._apply_gui_state(saved)
    assert window.tabs.currentWidget() is window.extra_functions_tab
    assert window._active_simulation_tab() == (prefix, tab)
    window.scattering_window.use_beam_button.click()
    assert float(window.scattering_window.radiation_value.text()) == pytest.approx(12)
    window.conversion_window.use_beam_button.click()
    assert float(window.conversion_window.fields["energy_kev"].text()) == pytest.approx(12)
    window.equivalent_window.use_sample_button.click()
    assert window.equivalent_window.fields["a"].text() == "7.5"


def test_legacy_poni_migration_only_runs_once():
    state = GuiState.from_dict({"state_version": 5, "poly": {"poni1": "1", "poni2": "2"}})
    assert state.poly.poni1 == "2"
    restored = GuiState.from_dict(state.to_dict())
    assert restored.poly.poni1 == "2"


def test_collapsing_preserves_values_and_disabled_controls(window, app):
    window.tabs.setCurrentWidget(window.single_tab)
    window.show()
    app.processEvents()
    tab = window.single_tab
    section = next(s for s in window.collapsible_sections if s.key == "single/Beam Parameters")
    field = section.content.findChildren(type(tab.single_line_sam_a))[0]
    field.setText("15000")
    field.setEnabled(False)
    height = section.group.height()
    section.button.click()
    app.processEvents()
    assert section.collapsed and section.content.isHidden()
    assert section.group.height() < height
    section.button.click()
    assert not section.collapsed
    assert not field.isEnabled()
    assert field.text() == "15000"
    tab.single_geometry_mode_combo.setCurrentText("Legacy Euler")
    assert tab.single_detector_euler_group.isEnabled()


def test_cif_drops_populate_both_tabs(window, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)
    path = tmp_path / "sample.cif"
    path.write_text("data_test\n")
    for tab, field, prefix in (
        (window.poly_tab, window.poly_tab.poly_line_cif_path, "poly"),
        (window.single_tab, window.single_tab.single_line_cif_path, "single"),
    ):
        calls = []
        def load(filename):
            calls.append(filename)
            getattr(tab, prefix + "_line_sam_a").setText("6.2")
        monkeypatch.setattr(tab, "_load_cif_into_fields", load)
        assert drop_file(field, path)
        assert calls == [str(path)]
        assert getattr(tab, prefix + "_line_sam_a").text() == "6.2"
        assert getattr(window.state.paths, prefix + "_cif_file_path") == str(path)
    wrong = tmp_path / "wrong.txt"
    wrong.write_text("not a CIF")
    assert not drop_file(window.single_tab.single_line_cif_path, wrong)


def test_equivalent_tool_uses_active_sample(window):
    window.tabs.setCurrentWidget(window.single_tab)
    window.single_tab.single_line_space_group.setText("221")
    window.single_tab.single_line_sam_a.setText("4.12")
    window._open_equivalent_reflections_window()
    assert window.equivalent_window.fields["space_group"].text() == "221"
    assert window.equivalent_window.fields["a"].text() == "4.12"


@pytest.mark.parametrize("order", ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"])
def test_named_euler_axes_match_the_3d_detector_grid(order):
    detector = make_detector(num_pixels_h=48, num_pixels_v=32, pxsize_h=0.001,
                             pxsize_v=0.001, dist=0.03, poni1=0.016, poni2=0.024,
                             rotx=11, roty=17, rotz=30, rotation_order=order)
    angles = {"x": 11, "y": 17, "z": 30}
    expected = Rotation.from_euler(order, [angles[a] for a in order], degrees=True).as_matrix()
    np.testing.assert_allclose(detector.rotation_matrix, expected, atol=1e-14)
    detector.calculate_lab_grid()
    h, v = detector.direct_beam_pixel()
    grid = detector.lab_grid
    lab_hit = grid[0, 0] + h * (grid[1, 0] - grid[0, 0]) + v * (grid[0, 1] - grid[0, 0])
    np.testing.assert_allclose(lab_hit[1:], [0, 0], atol=1e-12)
    assert lab_hit[0] > 0
    detector.set_rotation_angles(rotz=-30)
    angles["z"] = -30
    np.testing.assert_allclose(detector.rotation_matrix,
                               Rotation.from_euler(order, [angles[a] for a in order], degrees=True).as_matrix(),
                               atol=1e-14)


def test_rotz_30_moves_direct_beam_in_2d(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    detector = make_detector(dist=0.03, rotz=30)
    expected = [499.5 + 300 * np.tan(np.radians(30)), 249.5]
    np.testing.assert_allclose(detector.direct_beam_pixel(), expected)
    axes = plot_2d_detector_single_xstal(detector, np.empty((0, 2)), np.empty(0), [], {})
    marker = next(line for line in axes.lines if line.get_gid() == "direct-beam")
    np.testing.assert_allclose(marker.get_xydata()[0], expected)
    plt.close(axes.figure)


def test_imported_rotation_is_restored_without_reimport(window, monkeypatch, wait_simulation):
    from trxrdpy.simulation import single_crystal
    matrix = window.matrix_window
    for i, row in enumerate(matrix.matrix_edits):
        for j, cell in enumerate(row):
            cell.setText("4" if i == j else "0")
    matrix.line_rotz.setText("30")
    matrix._apply_rotation()
    window.single_tab._import_orientation_from_rotation_tool()
    expected = np.array([[cell.text() for cell in row]
                         for row in window.single_tab.orientation_entries], dtype=float)
    saved = window._gather_gui_state()
    window.single_tab.single_orientation_checkbox.setChecked(False)
    for row in window.single_tab.orientation_entries:
        for cell in row:
            cell.setText("0")
    window._apply_gui_state(saved)
    assert window.single_tab.single_orientation_checkbox.isChecked()
    restored = np.array([[cell.text() for cell in row]
                         for row in window.single_tab.orientation_entries], dtype=float)
    np.testing.assert_allclose(restored, expected)
    np.testing.assert_allclose(matrix.get_current_matrix(), expected, atol=1e-6)
    assert matrix._result_matrix_valid
    captured = {}
    monkeypatch.setattr(single_crystal, "simulate_2d", lambda **kwargs: captured.update(kwargs))
    assert window.single_tab._single_run_function()
    wait_simulation(window.single_tab)
    np.testing.assert_allclose(captured["sam_initial_crystal_orientation"], expected)


def test_manual_orientation_is_not_replaced_by_rotation_tool(window):
    saved = window._gather_gui_state()
    saved["single"].update(use_custom_orientation=True,
                            orientation_matrix=[["2", "0", "0"], ["0", "3", "0"], ["0", "0", "4"]],
                            sam_rotx="17", sam_roty="-8", sam_rotz="30")
    saved["matrix_tool"]["orientation_matrix"] = [["9", "0", "0"], ["0", "9", "0"], ["0", "0", "9"]]
    window._apply_gui_state(saved)
    result = window._gather_gui_state()
    for key in ("orientation_matrix", "sam_rotx", "sam_roty", "sam_rotz"):
        assert result["single"][key] == saved["single"][key]


def test_rotation_tool_mode_result_and_tilt_survive_roundtrip(window):
    matrix = window.matrix_window
    matrix.rotation_mode_combo.setCurrentText("Diffractometer motor chain")
    matrix.line_kappa_tilt.setText("42")
    for i, row in enumerate(matrix.matrix_edits):
        for j, cell in enumerate(row):
            cell.setText("4" if i == j else "0")
    matrix.line_roty.setText("25")
    matrix._apply_rotation()
    expected = matrix.get_current_matrix().copy()
    saved = window._gather_gui_state()
    matrix.rotation_mode_combo.setCurrentText("Euler-like XYZ")
    matrix.line_kappa_tilt.setText("50")
    window._apply_gui_state(saved)
    assert matrix.rotation_mode_combo.currentText() == "Diffractometer motor chain"
    assert matrix.line_kappa_tilt.text() == "42"
    np.testing.assert_allclose(matrix.get_current_matrix(), expected)


@pytest.mark.parametrize("text", ["[(1, 0, 0), (2, 0, 0)]", "(1, 0, 0), (2, 0, 0)"])
def test_multiple_equivalent_reflections(app, text):
    dialog = EquivalentReflectionsDialog()
    dialog.set_values(dict(space_group=221, a=4, b=4, c=4, alpha=90, beta=90, gamma=90, hkl=text))
    result = dialog.calculate()
    assert set(result) == {(1, 0, 0), (2, 0, 0)}
    assert len(result[(1, 0, 0)]) == len(result[(2, 0, 0)]) == 6
    assert (0, 0, 2) in result[(2, 0, 0)]
    assert dialog.output.minimumHeight() >= 240
    assert "(1, 0, 0):" in dialog.output.toPlainText()
    assert "(2, 0, 0):" in dialog.output.toPlainText()
    dialog.close()


@pytest.mark.parametrize("source,value", [("q", np.pi), ("d", 2), ("theta", 30), ("two_theta", 60)])
def test_scattering_coordinate_conversion(source, value):
    result = convert_scattering_coordinate(value, source, wavelength_angstrom=2)
    assert result == pytest.approx(dict(q=np.pi, d=2, theta=30, two_theta=60))


def test_scattering_boundaries_and_unreachable_angles():
    result = convert_scattering_coordinate(0, "q", 1)
    assert result["theta"] == 0 and np.isinf(result["d"])
    assert convert_scattering_coordinate(180, "two_theta", 1)["theta"] == pytest.approx(90)
    result = convert_scattering_coordinate(0.1, "d", 1)
    assert result["q"] == pytest.approx(20 * np.pi)
    assert result["two_theta"] is None
    for wavelength in np.geomspace(0.01, 100, 30):
        assert convert_scattering_coordinate(180, "two_theta", wavelength)["two_theta"] == pytest.approx(180)


@pytest.mark.parametrize("source,value,wavelength", [
    ("q", -1, 1), ("d", 0, 1), ("theta", 91, 1), ("two_theta", 181, 1),
    ("q", float("nan"), 1), ("q", 1, 0), ("q", 1, float("inf")),
])
def test_invalid_scattering_values(source, value, wavelength):
    with pytest.raises(ValueError):
        convert_scattering_coordinate(value, source, wavelength)


def test_scattering_dialog_and_gui_state(window):
    dialog = window.scattering_window
    window.tabs.setCurrentWidget(window.single_tab)
    window.single_tab.single_line_energy.setText("10000")
    dialog.use_beam_button.click()
    assert dialog.radiation_value.text() == "10"
    dialog.radiation.setCurrentIndex(1)
    assert float(dialog.radiation_value.text()) == pytest.approx(1.239841984)
    dialog.source.setCurrentIndex(dialog.source.findData("d"))
    dialog.value.setText("2.5")
    expected = dialog.convert()
    saved = window._gather_gui_state()
    dialog.value.setText("1")
    window._apply_gui_state(saved)
    assert dialog.values() == saved["ui"]["scattering_conversion"]
    assert float(dialog.outputs["theta"].text()) == pytest.approx(expected["theta"])
    window.tabs.setCurrentWidget(window.session_page)
    assert window._active_simulation_tab()[0] == "single"


@pytest.mark.parametrize("version,index,session_open,expected", [
    (10, 0, False, "polycrystalline"), (10, 1, False, "single_crystal"), (10, 2, False, "session"),
    (11, 0, False, "polycrystalline"), (11, 1, False, "single_crystal"), (11, 1, True, "session"),
    (12, 0, False, "session"), (12, 1, False, "polycrystalline"), (12, 2, False, "single_crystal"),
])
def test_old_tab_order_migration(version, index, session_open, expected):
    state = GuiState.from_dict(dict(state_version=version,
                                    ui=dict(current_tab_index=index, session_open=session_open)))
    assert state.ui.active_tab == expected
    if (version < 12 and index == 1) or (version >= 12 and index == 2):
        assert state.ui.last_simulation_tab == "single_crystal"


def test_geometry_controls_are_adjacent_to_detector_and_collapsed_state_migrates(window):
    tab = window.single_tab
    layout = tab.single_det_group.parentWidget().layout()
    assert layout.indexOf(tab.single_geometry_group) + 1 == layout.indexOf(tab.single_det_group)
    assert tab.single_geometry_group.isAncestorOf(tab.single_custom_geometry_group)
    assert tab.single_geometry_group.isAncestorOf(tab.single_geometry_scan_group)
    state = GuiState.from_dict({"state_version": 11,
        "ui": {"collapsed_sections": ["single/Custom Geometry Definition"]}})
    assert state.ui.collapsed_sections == ["single/Experiment Geometry/Custom Geometry Definition"]


def test_scientific_unit_labels(window):
    dialog = window.scattering_window
    assert dialog.radiation.itemText(1) == "Wavelength [\u00c5]"
    assert dialog.source.itemText(0) == "q [\u00c5\u207b\u00b9]"
    assert dialog.source.itemText(2) == "2\u03b8 [\u00b0]"
    labels = [label.text() for label in window.equivalent_window.findChildren(QLabel)]
    assert all(f"{axis} [\u00c5]:" in labels for axis in ("a", "b", "c"))


def test_poni_export_from_current_fit2d_values_and_reload(window, tmp_path):
    dialog = window.conversion_window
    dialog.set_values(dict(poni_source="Fit2D", num_pixels_v="831", num_pixels_h="1475",
                           energy_kev="15", directDist="160", centerX="700", centerY="400",
                           tilt="30", tiltPlanRotation="42", pixelX="75", pixelY="100"))
    expected = fit2d_to_pyfai({name: dialog.fields[name].text() for name in (
        "directDist", "centerX", "centerY", "tilt", "tiltPlanRotation", "pixelX", "pixelY")})
    path = dialog.export_poni(tmp_path / "fit2d.poni")
    parsed = read_poni_file(path)
    for name, value in expected.items():
        assert getattr(parsed, name) == pytest.approx(value)
    assert parsed.max_shape == (831, 1475)
    dialog.set_values(dict(energy_kev="10", num_pixels_v="12", num_pixels_h="34"))
    assert dialog.load_poni(path)
    assert dialog.fields["num_pixels_v"].text() == "831"
    assert dialog.fields["num_pixels_h"].text() == "1475"
    assert float(dialog.fields["energy_kev"].text()) == pytest.approx(15)
    assert dialog.poni_source.currentText() == "pyFAI"


def test_poni_export_inputs_are_saved_in_gui_state(window):
    dialog = window.conversion_window
    window.tabs.setCurrentWidget(window.single_tab)
    window.single_tab.single_line_energy.setText("12345")
    dialog.use_beam_button.click()
    assert float(dialog.fields["energy_kev"].text()) == pytest.approx(12.345)
    dialog.set_values(dict(num_pixels_v="400", num_pixels_h="500", poni_orientation="2", poni_source="Fit2D"))
    saved = window._gather_gui_state()
    dialog.set_values(dict(num_pixels_v="1", energy_kev="1", poni_orientation="3"))
    window._apply_gui_state(saved)
    assert dialog.values() == saved["ui"]["geometry_conversion"]


@pytest.mark.parametrize("replace", [False, True])
def test_save_poni_button_confirms_replacement(window, tmp_path, monkeypatch, replace):
    dialog = window.conversion_window
    dialog.set_values(dict(num_pixels_v="400", num_pixels_h="500"))
    path = tmp_path / "manual.poni"
    path.write_text("original")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), ""))
    questions = []
    def confirm(*args, **kwargs):
        questions.append(args)
        return QMessageBox.Yes if replace else QMessageBox.No
    monkeypatch.setattr(QMessageBox, "question", confirm)
    dialog.save_poni_button.click()
    assert len(questions) == 1
    if replace:
        assert "Saved PONI:" in dialog.status.text()
        assert read_poni_file(path).max_shape == (400, 500)
    else:
        assert path.read_text() == "original"


def test_save_poni_reports_invalid_inputs_without_crashing(window, tmp_path, monkeypatch):
    path = tmp_path / "invalid.poni"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), ""))
    window.conversion_window.fields["num_pixels_v"].setText("not a number")
    window.conversion_window.save_poni_button.click()
    assert "Error:" in window.conversion_window.status.text()
    assert not path.exists()


def test_poni_export_is_in_common_detector_parameters(window):
    for prefix, tab in (('single', window.single_tab), ('poly', window.poly_tab)):
        button = getattr(tab, prefix + '_save_poni_button')
        parent = button.parentWidget()
        while parent is not None and not hasattr(parent, 'title'):
            parent = parent.parentWidget()
        assert parent.title() == 'Common Detector Parameters'
    assert window.conversion_window.save_poni_button.parentWidget().isHidden()
