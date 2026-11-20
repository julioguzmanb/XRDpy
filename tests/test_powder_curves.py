"""Powder-ring geometry and GUI routing."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from PyQt5.QtWidgets import QApplication

from trxrdpy.simulation import polycrystalline, utils
from trxrdpy.simulation.detector import Detector
from trxrdpy.simulation.detector_view import pixels_in_lab
from trxrdpy.simulation.deferred_plotting import capture_plots
from trxrdpy.simulation.experiment import Experiment
from trxrdpy.simulation.gui.services.simulation_service import SimulationService
from trxrdpy.simulation.gui.state import GuiState
from trxrdpy.simulation.gui.tabs.polycrystalline_tab import PolycrystallineTab
from trxrdpy.simulation.plot import within_bounds
from trxrdpy.simulation.powder_curves import adaptive_ring_segments, cone_surface
from trxrdpy.simulation.sample import LatticeStructure, create_diffraction_cones


def detector(**overrides):
    values = dict(detector_type="manual", pxsize_h=4.42708e-5,
                  pxsize_v=4.42708e-5, num_pixels_h=3840,
                  num_pixels_v=2712, dist=0.1, poni1=0, poni2=0.08)
    values.update(overrides)
    return Detector(**values)


def _max_distance_to_curves(points, segments):
    """Compare sampled hits with the nearest point on the drawn line chords."""
    start, end = segments[:, 0], segments[:, 1]
    chord = end - start
    denominator = np.sum(chord * chord, axis=1)
    maximum = 0.0
    for chunk in np.array_split(points, max(1, (len(points) + 255) // 256)):
        delta = chunk[:, None, :] - start[None, :, :]
        fraction = np.clip(np.sum(delta * chord, axis=2) / denominator, 0, 1)
        nearest = start[None, :, :] + fraction[:, :, None] * chord[None, :, :]
        maximum = max(maximum, np.min(np.linalg.norm(
            chunk[:, None, :] - nearest, axis=2), axis=1).max())
    return maximum


@pytest.mark.parametrize("angles", [(0, 0, 0), (0, -30, -30), (10, 25, 15)])
def test_sampled_and_adaptive_detector_intersections_agree(angles):
    det = detector(poni1=0.07, rotx=angles[0], roty=angles[1], rotz=angles[2])
    q_values = np.array([1.2, 2.0, 3.0])
    labels = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    exp = Experiment(det, LatticeStructure(space_group=167, a=1, c=1),
                     energy=15000, e_bandwidth=1.5)
    exp.calculate_cones_pixel_positions(q_values, labels, num_points=4096)

    for index, q in enumerate(q_values):
        sampled = exp.diffraction_cones_pixel_position[index]
        visible = within_bounds(sampled, exp.scat_dir_sign[index].ravel(),
                                det.num_pixels_h, det.num_pixels_v)
        curves = adaptive_ring_segments(det, q, exp.wavelength)
        assert visible.any() and len(curves)
        assert _max_distance_to_curves(sampled[visible], curves) < 0.4


def test_sampled_and_new_3d_cone_surfaces_have_identical_apex_and_rim():
    wavelength = utils.energy_to_wavelength(15000)
    q_values = [1.2, 2.0, 3.0]
    length = 0.11
    sampled = create_diffraction_cones(wavelength, q_values, length, num_points=128)
    for q, old_components in zip(q_values, sampled):
        new_surface = cone_surface(q, wavelength, length, num_points=128)
        for axis, old_component in enumerate(old_components):
            np.testing.assert_allclose(new_surface[axis, 0], old_component[0], atol=1e-14)
            np.testing.assert_allclose(new_surface[axis, 1], old_component[-1], atol=1e-14)


@pytest.mark.parametrize("angles", [(0, 0, 0), (10, 25, 15)])
def test_adaptive_ring_follows_detector_geometry(angles):
    det = detector(rotx=angles[0], roty=angles[1], rotz=angles[2])
    wavelength = utils.energy_to_wavelength(15000)
    segments = adaptive_ring_segments(det, 2.0, wavelength)
    assert 0 < len(segments) < 1000
    assert np.all(segments[:, :, 0] >= -0.5 - 1e-8)
    assert np.all(segments[:, :, 0] <= det.num_pixels_h - 0.5 + 1e-8)
    assert np.all(segments[:, :, 1] >= -0.5 - 1e-8)
    assert np.all(segments[:, :, 1] <= det.num_pixels_v - 0.5 + 1e-8)

    # Chord midpoints must remain on the selected scattering cone to within
    # the subpixel approximation used for drawing the curve.
    midpoints = segments.mean(axis=1)
    native = np.column_stack((
        np.full(len(midpoints), det.dist),
        det.poni2 - (midpoints[:, 0] + 0.5) * det.pxsize_h,
        -det.poni1 + (midpoints[:, 1] + 0.5) * det.pxsize_v,
    ))
    lab = utils.apply_transform(native, det.get_transform())
    rays = lab / np.linalg.norm(lab, axis=1)[:, None]
    wavevector = 2 * np.pi / (wavelength * 1e10)
    q_midpoints = wavevector * np.linalg.norm(rays - [1, 0, 0], axis=1)
    assert np.max(np.abs(q_midpoints - 2.0)) < 0.002
    assert det.lab_grid is None


@pytest.mark.parametrize("name", ["simulate_2d", "simulate_3d"])
def test_powder_entrypoints_render_without_full_detector_grid(name, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(Detector, "calculate_lab_grid",
                        lambda self, *args, **kwargs: pytest.fail("allocated full detector grid"))
    kwargs = dict(det_ntum_pixels_h=3840, det_num_pixels_v=2712,
                  det_pxsize_h=4.42708e-5, det_pxsize_v=4.42708e-5,
                  det_dist=0.1, det_poni1=0, det_poni2=0.08,
                  energy=15000, q_hkls=[2.0], hkls_names=[[1, 0, 0]])
    with capture_plots() as calls:
        result = getattr(polycrystalline, name)(**kwargs)
    assert len(calls) == 1
    assert len(result.diffraction_cones_curves[0]) > 0
    assert result.detector.lab_grid is None
    calls[0][0](*calls[0][1], **calls[0][2])
    axes = plt.gcf().axes[0]
    if name == "simulate_2d":
        assert any(isinstance(artist, LineCollection) for artist in axes.collections)
        assert axes.get_aspect() == pytest.approx(1)
        assert axes.title.get_fontsize() > 9
    else:
        assert len(axes.collections) >= 2  # detector, cone, and intersection
        assert sum(artist.get_gid() == "detector-visual-grid"
                   for artist in axes.collections) == 1
        assert axes.figure._suptitle.get_fontsize() > 9
    plt.close("all")


@pytest.mark.parametrize("poni2,expected_count", [(0.08, 1), (0.5, 0)])
def test_powder_3d_direct_beam_cross_follows_detector_hit(poni2, expected_count, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    exp = polycrystalline.simulate_3d(
        det_ntum_pixels_h=3840, det_num_pixels_v=2712,
        det_pxsize_h=4.42708e-5, det_pxsize_v=4.42708e-5,
        det_dist=0.1, det_poni1=0.07, det_poni2=poni2,
        det_roty=-30, det_rotz=-30,
        energy=15000, q_hkls=[2.0], hkls_names=[[1, 0, 0]],
    )
    markers = [line for line in plt.gcf().axes[0].lines
               if line.get_gid() == "direct-beam"]
    assert len(markers) == expected_count
    if markers:
        expected = pixels_in_lab(exp.detector, [exp.detector.direct_beam_pixel()])[0]
        np.testing.assert_allclose(np.asarray(markers[0].get_data_3d()).ravel(), expected)
        assert markers[0].get_color() == "red"
        assert markers[0].get_marker() == "x"
    plt.close("all")


@pytest.mark.parametrize("name", ["simulate_2d", "simulate_3d"])
def test_gui_powder_modes_do_not_require_point_count(name, monkeypatch, wait_simulation):
    app = QApplication.instance() or QApplication([])
    _ = app
    captured = {}
    monkeypatch.setattr(polycrystalline, name, lambda **kwargs: captured.update(kwargs))
    tab = PolycrystallineTab(SimulationService(), GuiState())
    tab.poly_func_combo.setCurrentText(name)
    tab.poly_combo_refsrc.setCurrentIndex(1)
    tab.poly_line_qhkls.setText("2.0")
    tab.poly_line_hkls.setText("[1,0,0]")
    assert not hasattr(tab, "poly_line_cones")
    assert [tab.poly_func_combo.itemText(i) for i in range(tab.poly_func_combo.count())] == [
        "simulate_1d", "simulate_2d", "simulate_3d"]
    assert tab._poly_run_function()
    wait_simulation(tab)
    assert "cones_num_of_points" not in captured
    assert captured["q_hkls"].tolist() == [2.0]
    tab.close()


@pytest.mark.parametrize("old,new", [
    ("simulate_2d_continuous", "simulate_2d"),
    ("simulate_3d_continuous", "simulate_3d"),
])
def test_previous_gui_selection_loads_under_main_function_name(old, new):
    state = GuiState.from_dict({"state_version": 13, "poly": {"func": old, "cones": "30"}})
    assert state.poly.func == new
    assert "cones" not in state.to_dict()["poly"]
