"""Intensity calculation, detector rendering, backend routing, and GUI state."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from types import SimpleNamespace
from PyQt5.QtWidgets import QApplication

from trxrdpy.simulation import intensity_view, polycrystalline, single_crystal
from trxrdpy.simulation.detector import Detector
from trxrdpy.simulation.detector_view import pixels_in_lab
from trxrdpy.simulation.experiment import Experiment
from trxrdpy.simulation.sample import LatticeStructure, powder_peaks_1d
from trxrdpy.simulation.gui.services.simulation_service import SimulationService
from trxrdpy.simulation.gui.state import GuiState
from trxrdpy.simulation.gui.tabs.polycrystalline_tab import PolycrystallineTab
from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab
from trxrdpy.simulation.plot import plot_2d_detector_single_xstal
from trxrdpy.simulation.powder_curves import plot_2d_powder, plot_3d_powder
from trxrdpy.simulation.deferred_plotting import capture_plots


def small_detector():
    return Detector(detector_type="manual", pxsize_h=1e-3, pxsize_v=1e-3,
                    num_pixels_h=32, num_pixels_v=24, dist=0.1,
                    poni1=0.012, poni2=0.016)


def test_real_cif_structure_factors(tmp_path):
    cif = tmp_path / "simple.cif"
    cif.write_text("""data_simple
_cell_length_a 5.431
_cell_length_b 5.431
_cell_length_c 5.431
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 227
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0 0 0
""")
    intensities = intensity_view.reflection_intensities(cif, [[1, 1, 1], [2, 2, 0]], 15000)
    assert intensities.shape == (2,)
    assert np.all(np.isfinite(intensities) & (intensities > 0))
    assert intensities.max() == pytest.approx(1)
    lattice = LatticeStructure(cif_file_path=cif)
    peaks = powder_peaks_1d(lattice, 15000, 4, cif_file_path=cif, normalize=False)
    hkls = np.column_stack((peaks["h"], peaks["k"], peaks["l"]))
    np.testing.assert_allclose(
        intensity_view.reflection_intensities(cif, hkls, 15000, powder=True),
        peaks["I"] / peaks["I"].max())


def test_single_uses_simulation_cell_with_atom_only_cif(tmp_path):
    cif = tmp_path / "atoms_only.cif"
    cif.write_text("""data_atoms
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0 0 0
""")
    lattice = LatticeStructure(space_group=221, a=3.91726, b=3.91726,
                               c=3.91726, alpha=90, beta=90, gamma=90)
    values = intensity_view.reflection_intensities(
        cif, [[1, 0, 0], [1, 1, 0]], 14200, lattice=lattice)
    assert values.shape == (2,)
    assert np.isfinite(values).all()
    assert values.max() == pytest.approx(1)


def test_empty_single_bragg_set_plots_without_hkl_color_map(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    det = small_detector()
    lattice = LatticeStructure(space_group=221, a=4, b=4, c=4)
    lattice.q_hkls = np.empty((0, 3))
    lattice.q_hkls_in_Bragg_condition = np.empty((0, 3))
    lattice.hkls_in_Bragg_condition = np.empty((0, 3), dtype=int)
    exp = Experiment(det, lattice, energy=14200, e_bandwidth=.1)
    exp.calculate_diffraction_direction(3)
    exp.calculate_pixel_positions()
    assert exp.hkl_to_color == {}
    assert exp.pixel_positions.shape == (0, 2)
    exp.plot_2d_single_xstal_exp()
    exp.reflection_intensities = np.empty(0)
    exp.plot_2d_single_xstal_exp(consider_intensities=True)
    plt.close("all")


def test_custom_orientation_single_runs_without_complete_cif_cell(tmp_path, monkeypatch):
    from trxrdpy.simulation.deferred_plotting import capture_plots
    monkeypatch.setattr(plt, "show", lambda: None)
    cif = tmp_path / "atoms_only.cif"
    cif.write_text("""data_atoms
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0 0 0
""")
    common = dict(det_ntum_pixels_h=120, det_num_pixels_v=80,
                  det_pxsize_h=1e-3, det_pxsize_v=1e-3,
                  det_dist=.1, det_poni1=.04, det_poni2=.06,
                  energy=14200, e_bandwidth=.1, sam_space_group=221,
                  sam_a=3.91726, sam_b=3.91726, sam_c=3.91726,
                  sam_alpha=90, sam_beta=90, sam_gamma=90, qmax=5,
                  sam_initial_crystal_orientation=np.array([
                      [0, 0, 3.91726], [0, -3.91726, 0], [3.91726, 0, 0]]))
    with capture_plots() as calls:
        exp = single_crystal.simulate_2d(**common)
    assert hasattr(exp, "hkl_to_color")
    assert len(calls) == 1
    calls[0][0](*calls[0][1], **calls[0][2])
    plt.close("all")
    with capture_plots() as calls:
        exp = single_crystal.simulate_3d(
            **common, consider_intensities=True, cif_file_path=cif)
    assert exp.detector_intensity_image.shape == (80, 120)
    assert len(calls) == 1
    calls[0][0](*calls[0][1], **calls[0][2])
    plt.close("all")


def test_structure_factors_and_powder_correction(monkeypatch):
    class Crystal:
        def Q(self, hkl):
            return np.asarray(hkl, float)

        def StructureFactor(self, q, energy):
            assert energy == 15000
            return complex(q[0] + 2 * q[1])

    monkeypatch.setattr(intensity_view.sample, "LatticeStructure", lambda **kw: object())
    monkeypatch.setattr("trxrdpy.simulation.cif.Cif", lambda path: None)
    monkeypatch.setattr(intensity_view.sample, "_build_xu_crystal_from_lattice",
                        lambda *args, **kw: Crystal())
    hkls = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    np.testing.assert_allclose(
        intensity_view.reflection_intensities("sample.cif", hkls, 15000),
        [0.25, 1.0, 0.0])
    np.testing.assert_allclose(
        intensity_view.reflection_intensities("sample.cif", hkls, 15000,
                                               normalize=False), [1.0, 4.0, 0.0])
    powder = intensity_view.reflection_intensities("sample.cif", hkls, 15000, powder=True)
    np.testing.assert_allclose(powder, [0.25, 1.0, 0.0])
    with pytest.raises(ValueError, match="CIF"):
        intensity_view.reflection_intensities(None, hkls, 15000)


def test_detector_images_zero_overlap_and_noise():
    det = small_detector()
    curves = [np.array([[[1, 1], [4, 1]], [[4, 1], [7, 1]]]),
              np.array([[[4, 1], [4, 3]]])]
    image = intensity_view.powder_detector_image(det, curves, [0.5, 0.25])
    assert image[1, 4] == pytest.approx(0.75)
    assert image[0, 0] == 0
    assert tuple(intensity_view.intensity_mappable("viridis", 1).to_rgba(0)[:3]) == (0, 0, 0)
    mappable = intensity_view.intensity_mappable("viridis", 1, (.1, .8))
    assert mappable.norm.vmin == pytest.approx(.1)
    assert mappable.norm.vmax == pytest.approx(.8)
    assert tuple(mappable.to_rgba(0)[:3]) == (0, 0, 0)
    with pytest.raises(ValueError, match="exceed"):
        intensity_view.validate_clim((.8, .1))
    spots = intensity_view.spots_detector_image(
        det, [[2, 3], [2, 3], [31, 23], [50, 50]],
        [0.4, 0.6, 0.2, 1.0], [True, True, False, True])
    assert spots[3, 2] == pytest.approx(1.0)
    assert spots.sum() == pytest.approx(1.0)
    gaussian = intensity_view.add_noise(np.zeros((50, 50)), "gaussian", .05, 5)
    poisson = intensity_view.add_noise(np.zeros((50, 50)), "poisson", .05, 5)
    assert gaussian.min() == 0 and gaussian.max() > 0
    assert poisson.mean() == pytest.approx(.05, abs=.004)
    noisy, background = intensity_view.powder_detector_image(
        det, curves, [0.5, 0.25], noise_distribution="poisson",
        noise_level=.03, noise_seed=7, return_background=True)
    assert background.shape == noisy.shape
    assert background[0, 0] == noisy[0, 0]
    assert background[1, 4] < noisy[1, 4]
    with pytest.raises(ValueError, match="nonnegative"):
        intensity_view.validate_display("viridis", "gaussian", -1)


def test_polarization_factor_and_powder_azimuth():
    directions = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
    np.testing.assert_allclose(intensity_view.polarization_weight(directions, 1), [0, 1, 1])
    np.testing.assert_allclose(intensity_view.polarization_weight(directions, -1), [1, 0, 1])
    np.testing.assert_allclose(intensity_view.polarization_weight(directions, 0), [.5, .5, 1])
    np.testing.assert_allclose(intensity_view.polarization_weight(directions, None), [1, 1, 1])
    tth, chi, offset = np.deg2rad([37, 25, 12])
    factor = .65
    ray = [[np.cos(tth), np.sin(tth) * np.cos(chi), np.sin(tth) * np.sin(chi)]]
    pyfai_formula = .5 * (1 + np.cos(tth) ** 2 - factor *
                         np.cos(2 * (chi + offset)) * np.sin(tth) ** 2)
    assert intensity_view.polarization_weight(ray, factor, 12)[0] == pytest.approx(pyfai_formula)
    with pytest.raises(ValueError, match="between -1 and 1"):
        intensity_view.validate_polarization_factor(1.1)

    det = small_detector()
    curves = [np.array([[[2, 11], [2, 11]]]),
              np.array([[[16, 2], [16, 2]]])]
    horizontal = intensity_view.powder_detector_image(
        det, curves, [1, 1], polarization_factor=1)
    vertical = intensity_view.powder_detector_image(
        det, curves, [1, 1], polarization_factor=-1)
    assert horizontal[11, 2] < vertical[11, 2]
    assert horizontal[2, 16] > vertical[2, 16]
    uncorrected = intensity_view.powder_detector_image(
        det, curves, [1, 1], polarization_factor=None)
    assert uncorrected[11, 2] > 1
    np.testing.assert_allclose(
        intensity_view.powder_detector_image(det, curves, [1, 1],
                                              polarization_factor=0)[[11, 2], [2, 16]],
        [1, 1])


def test_single_crystal_polarization_uses_lab_ray_directions(monkeypatch):
    det = small_detector()
    positions = np.array([[2, 11], [16, 2]], float)
    rays = pixels_in_lab(det, positions)
    monkeypatch.setattr(intensity_view, "reflection_intensities",
                        lambda *args, **kwargs: np.ones(2))
    exp = SimpleNamespace(detector=det, pixel_positions=positions,
                          scattered_directions=rays, scat_dir_sign=np.ones(2),
                          lattice=SimpleNamespace(
                              allowed_hkls=np.array([[1, 0, 0], [0, 1, 0]]),
                              hkls_in_Bragg_condition=np.array(
                                  [[1, 0, 0], [0, 1, 0]])), energy=15000)
    single_crystal._intensity_plot_options(
        exp, True, "sample.cif", "viridis", None, 1, 0, "none", 0, None)
    assert exp.reflection_intensities[0] < exp.reflection_intensities[1]
    assert exp.detector_intensity_image[11, 2] < exp.detector_intensity_image[2, 16]
    single_crystal._intensity_plot_options(
        exp, True, "sample.cif", "viridis", None, -1, 0, "none", 0, None)
    assert exp.reflection_intensities[0] > exp.reflection_intensities[1]


def test_single_uses_strongest_allowed_reflection_even_when_not_in_bragg(monkeypatch):
    det = small_detector()
    captured = {}

    def factors(path, hkls, energy, **kwargs):
        captured["hkls"] = np.asarray(hkls)
        assert kwargs["normalize"] is False
        return np.array([100.0, 4.0])

    monkeypatch.setattr(intensity_view, "reflection_intensities", factors)
    exp = SimpleNamespace(detector=det, pixel_positions=np.array([[4., 5.]]),
                          scattered_directions=pixels_in_lab(det, [[4., 5.]]),
                          scat_dir_sign=np.array([1]), energy=15000,
                          lattice=SimpleNamespace(
                              allowed_hkls=np.array([[1, 0, 0], [0, 1, 0]]),
                              hkls_in_Bragg_condition=np.array([[0, 1, 0]])))
    single_crystal._intensity_plot_options(
        exp, True, "sample.cif", "viridis", None, None, 0, "none", 0, None)
    np.testing.assert_array_equal(captured["hkls"], exp.lattice.allowed_hkls)
    assert exp.intensity_reference == 100
    assert exp.reflection_intensities[0] == pytest.approx(.04)
    assert exp.detector_intensity_image[5, 4] == pytest.approx(.04)


@pytest.mark.parametrize("dimension", (2, 3))
def test_powder_intensity_plot_has_labels_colorbar_and_black_zero(dimension, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    det = small_detector()
    curves = [np.array([[[4., 5.], [12., 5.]]])]
    hkl = [[1, 0, 0]]
    if dimension == 2:
        ax = plot_2d_powder(det, curves, hkl, "powder", intensities=[1.0],
                            intensity_clim=(.1, .8))
        assert ax.get_facecolor()[:3] == (0, 0, 0)
        assert not ax.images
        assert len(ax.collections) == 1
        assert any("(1,0,0)" in label.get_text() for label in ax.texts)
    else:
        ax = plot_3d_powder(det, [2.0], curves, hkl, 1e-10, "powder",
                            intensities=[1.0], polarization_factor=1)
        assert any("(1,0,0)" in label.get_text() for label in ax.texts)
    assert len(ax.figure.axes) == 2  # plot and intensity colorbar
    if dimension == 2:
        assert ax.figure.axes[1].get_ylim()[1] == pytest.approx(.8)
    plt.close(ax.figure)


def test_powder_noise_plot_uses_background_without_pixelated_rings(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    det = small_detector()
    curves = [np.array([[[4., 5.], [12., 5.]]])]
    ax = plot_2d_powder(det, curves, [[1, 0, 0]], "powder",
                        intensities=[1.0], noise_distribution="poisson",
                        noise_level=.02, noise_seed=9)
    assert len(ax.images) == 1
    assert len(ax.collections) == 1
    background = np.asarray(ax.images[0].get_array())
    assert background[5, 7] < .1
    plt.close(ax.figure)


def test_single_crystal_intensity_plot_uses_hkl_labels(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    det = small_detector()
    ax = plot_2d_detector_single_xstal(
        det, np.array([[4., 5.], [8., 9.]]), np.array([1, 1]),
        np.array([[1, 0, 0], [0, 1, 0]]), {}, intensities=np.array([1., 0.]),
        intensity_mode=True, intensity_clim=(0, .6))
    assert ax.images[0].get_array()[5, 4] == 1
    assert ax.images[0].get_array()[9, 8] == 0
    assert len(ax.figure.axes) == 2
    assert ax.images[0].norm.vmax == pytest.approx(.6)
    assert {text.get_text() for text in ax.texts} == {"(1,0,0)", "(0,1,0)"}
    plt.close(ax.figure)


@pytest.mark.parametrize('intensity_mode', [False, True])
def test_single_crystal_2d_hover_shows_q_and_angles(monkeypatch, intensity_mode):
    from matplotlib.backend_bases import MouseEvent, PickEvent
    monkeypatch.setattr(plt, 'show', lambda: None)
    det = small_detector()
    q = np.array([[-1., 1., 1.], [0., 0., 2.]])
    ax = plot_2d_detector_single_xstal(
        det, np.array([[4., 5.], [8., 9.]]), np.array([1, 1]),
        np.array([[1, 0, 0], [0, 1, 0]]),
        {(1, 0, 0): 'red', (0, 1, 0): 'blue'},
        q_vectors=q, intensity_mode=intensity_mode,
        intensities=np.array([1., .5]) if intensity_mode else None)
    fig = ax.figure
    fig.canvas.draw()
    x, y = ax.transData.transform((4, 5))
    fig.canvas.callbacks.process('motion_notify_event',
        MouseEvent('motion_notify_event', fig.canvas, x, y))
    tooltip = ax._single_crystal_hover_text
    assert tooltip.get_visible()
    assert 'hkl=(1, 0, 0)' in tooltip.get_text()
    assert 'Q=(-1.0000, 1.0000, 1.0000) Å⁻¹' in tooltip.get_text()
    assert '|Q|=1.7321 Å⁻¹' in tooltip.get_text()
    assert 'Azimuth=-45.00°' in tooltip.get_text()
    assert 'Elevation=35.26°' in tooltip.get_text()
    assert 'Pixel (H, V)=(4.00, 5.00)' in tooltip.get_text()
    if not intensity_mode:
        legend_line = ax.get_legend().get_lines()[0]
        fig.canvas.callbacks.process('pick_event', PickEvent('pick_event', fig.canvas,
            MouseEvent('button_press_event', fig.canvas, 0, 0), legend_line))
        fig.canvas.callbacks.process('motion_notify_event',
            MouseEvent('motion_notify_event', fig.canvas, x, y))
        assert not tooltip.get_visible()
    plt.close(fig)


def test_single_crystal_2d_hover_flips_inside_figure_at_right_edge(monkeypatch):
    from matplotlib.backend_bases import MouseEvent

    monkeypatch.setattr(plt, 'show', lambda: None)
    ax = plot_2d_detector_single_xstal(
        small_detector(), np.array([[31., 20.]]), np.array([1]),
        np.array([[2, 2, 0]]), {(2, 2, 0): 'red'},
        q_vectors=np.array([[-1.6646, -3.3227, 2.0]]), show=False)
    fig = ax.figure
    fig.set_size_inches(5, 4)
    fig.canvas.draw()
    mouse_x, mouse_y = ax.transData.transform((31, 20))
    fig.canvas.callbacks.process(
        'motion_notify_event',
        MouseEvent('motion_notify_event', fig.canvas, mouse_x, mouse_y),
    )
    tooltip = ax._single_crystal_hover_text
    assert tooltip.get_visible()
    fig.canvas.draw()
    bounds = tooltip.get_bbox_patch().get_window_extent(fig.canvas.get_renderer())
    assert bounds.x0 >= 0
    assert bounds.x1 <= fig.bbox.width
    assert bounds.y0 >= 0
    assert bounds.y1 <= fig.bbox.height
    assert bounds.x1 < mouse_x
    plt.close(fig)


def test_experiment_forwards_q_to_2d_plot(monkeypatch):
    from trxrdpy.simulation import plot
    lattice = LatticeStructure(space_group=221, a=4, b=4, c=4)
    lattice.hkls_in_Bragg_condition = np.array([[1, 0, 0]])
    lattice.q_hkls_in_Bragg_condition = np.array([[-1., 1., 1.]])
    exp = Experiment(small_detector(), lattice, energy=15000, e_bandwidth=1.5)
    exp.pixel_positions = np.array([[4., 5.]])
    exp.scat_dir_sign = np.array([1])
    captured = {}
    monkeypatch.setattr(plot, 'plot_2d_detector_single_xstal', lambda **kw: captured.update(kw))
    exp.plot_2d_single_xstal_exp()
    np.testing.assert_array_equal(captured['q_vectors'], lattice.q_hkls_in_Bragg_condition)


def test_single_crystal_3d_intensity_texture_and_labels(monkeypatch):
    from matplotlib.backend_bases import MouseEvent
    from mpl_toolkits.mplot3d import proj3d
    monkeypatch.setattr(plt, "show", lambda: None)
    det = small_detector()
    lattice = LatticeStructure(space_group=221, a=4, b=4, c=4)
    lattice.hkls_in_Bragg_condition = np.array([[1, 0, 0], [0, 1, 0]])
    lattice.q_hkls_in_Bragg_condition = np.array([[-.7201, 0., 3.1260],
                                                   [-.3601, 1.6040, 1.5630]])
    exp = Experiment(det, lattice, energy=15000, e_bandwidth=1.5)
    exp.scattered_directions = pixels_in_lab(det, [[4, 5], [40, 40]])
    exp.scat_dir_sign = np.array([1, 1])
    exp.hkl_to_color = {(1, 0, 0): "red", (0, 1, 0): "blue"}
    exp.reflection_intensities = np.array([1.0, 0.2])
    exp.plot_3d_single_xstal_exp(plot_crystal=False, consider_intensities=True,
                                 noise_distribution="poisson", noise_level=.01)
    ax = plt.gcf().axes[0]
    assert len(ax.figure.axes) == 2
    assert ax.get_legend() is not None
    assert any("(1,0,0)" in label.get_text()
               for label in ax.get_legend().get_texts())
    assert any("(1, 0, 0)" in label.get_text() for label in ax.texts)
    assert any(artist.get_gid() == "bragg-hit" for artist in ax.collections)
    ax.figure.canvas.draw()
    x, y, _ = proj3d.proj_transform(*exp.scattered_directions[0], ax.get_proj())
    px, py = ax.transData.transform((x, y))
    ax.figure.canvas.callbacks.process('motion_notify_event',
        MouseEvent('motion_notify_event', ax.figure.canvas, px, py))
    assert 'Azimuth=0.00°' in ax._single_crystal_hover_text.get_text()
    plt.close(ax.figure)


@pytest.mark.parametrize("name", ("simulate_2d", "simulate_3d"))
def test_powder_backend_calculates_and_forwards_intensity(name, monkeypatch):
    observed = {}
    monkeypatch.setattr(intensity_view, "reflection_intensities",
                        lambda path, hkls, energy, **kw: np.array([0.7]))
    from trxrdpy.simulation import powder_curves
    monkeypatch.setattr(powder_curves, f"plot_{name[-2:]}_powder",
                        lambda *args, **kw: observed.update(kw))
    exp = getattr(polycrystalline, name)(
        det_ntum_pixels_h=32, det_num_pixels_v=24,
        det_pxsize_h=1e-3, det_pxsize_v=1e-3, det_poni1=.012, det_poni2=.016,
        energy=15000, q_hkls=[2.0], hkls_names=[[1, 0, 0]],
        consider_intensities=True, cif_file_path="sample.cif",
        intensity_cmap="plasma", intensity_clim=(.1, .9),
        polarization_factor=.6, polarization_axis_offset_deg=15,
        noise_distribution="poisson", noise_level=.03)
    np.testing.assert_allclose(exp.reflection_intensities, [0.7])
    np.testing.assert_allclose(observed["intensities"], [0.7])
    assert observed["intensity_image"] is exp.detector_intensity_image
    assert exp.detector_intensity_image.shape == (24, 32)
    assert observed["cmap"] == "plasma"
    assert observed["intensity_clim"] == (.1, .9)
    assert observed["polarization_factor"] == pytest.approx(.6)
    assert observed["polarization_axis_offset_deg"] == pytest.approx(15)
    assert observed["noise_distribution"] == "poisson"


@pytest.mark.parametrize("function", (
    single_crystal.simulate_2d, single_crystal.simulate_3d,
    single_crystal.simulate_2d_with_geometry,
    single_crystal.simulate_3d_with_geometry,
))
def test_single_backend_calculates_intensity_for_each_geometry(function, monkeypatch):
    monkeypatch.setattr(intensity_view, "reflection_intensities",
                        lambda path, hkls, energy, **kw: np.full(len(hkls), .5))
    kwargs = dict(det_ntum_pixels_h=120, det_num_pixels_v=80,
                  det_pxsize_h=1e-3, det_pxsize_v=1e-3,
                  det_dist=.1, det_poni1=.04, det_poni2=.06,
                  energy=15000, sam_space_group=221, sam_a=4, sam_b=4,
                  sam_c=4, sam_alpha=90, sam_beta=90, sam_gamma=90, qmax=3,
                  consider_intensities=True, cif_file_path="sample.cif",
                  intensity_cmap="inferno", intensity_clim=(.05, .75),
                  polarization_factor=.4, polarization_axis_offset_deg=15,
                  noise_distribution="gaussian",
                  noise_level=.02)
    with capture_plots() as calls:
        exp = function(**kwargs)
    assert len(calls) == 1
    assert np.all((0 <= exp.reflection_intensities) & (exp.reflection_intensities <= 1))
    if len(exp.reflection_intensities):
        assert exp.reflection_intensities.max() > 0
    assert calls[0][2]["consider_intensities"] is True
    assert calls[0][2]["intensity_cmap"] == "inferno"
    assert calls[0][2]["intensity_clim"] == (.05, .75)
    assert exp.polarization_factor == pytest.approx(.4)
    assert exp.polarization_axis_offset_deg == pytest.approx(15)
    assert calls[0][2]["intensity_image"] is exp.detector_intensity_image
    assert exp.detector_intensity_image.shape == (80, 120)


@pytest.mark.parametrize("family", ("poly", "single"))
def test_gui_intensity_controls_round_trip_and_dispatch(family, monkeypatch, wait_simulation):
    app = QApplication.instance() or QApplication([])
    _ = app
    state = GuiState()
    cls = PolycrystallineTab if family == "poly" else SingleCrystalTab
    tab = cls(SimulationService(), state)
    controls = tab.poly_intensity_controls if family == "poly" else tab.single_intensity_controls
    assert controls.polarization_factor.text() == "1"
    func = tab.poly_func_combo if family == "poly" else tab.single_func_combo
    func.setCurrentText("simulate_2d")
    controls.consider.setChecked(True)
    controls.cmap.setCurrentText("magma")
    controls.clim_min.setText("0.05")
    controls.clim_max.setText("0.8")
    controls.polarization_factor.setText("0.6")
    controls.axis_offset.setText("15")
    controls.noise.setCurrentText("gaussian")
    controls.level.setText("0.12")
    if family == "poly":
        tab.poly_line_cif_path.setText("sample.cif")
        tab.poly_combo_refsrc.setCurrentIndex(1)
        tab.poly_line_qhkls.setText("2.0")
        tab.poly_line_hkls.setText("[1,0,0]")
        module = polycrystalline
    else:
        tab.single_line_cif_path.setText("sample.cif")
        module = single_crystal
    tab.save_to_state()
    restored = GuiState.from_dict(state.to_dict())
    second = cls(SimulationService(), restored)
    selected = second.poly_intensity_controls if family == "poly" else second.single_intensity_controls
    assert selected.consider.isChecked()
    assert selected.cmap.currentText() == "magma"
    assert selected.clim_min.text() == "0.05"
    assert selected.clim_max.text() == "0.8"
    assert selected.apply_polarization.isChecked()
    assert selected.polarization_factor.text() == "0.6"
    assert selected.axis_offset.text() == "15"
    assert selected.noise.currentText() == "gaussian"
    assert selected.level.text() == "0.12"
    captured = {}
    monkeypatch.setattr(module, "simulate_2d", lambda **kw: captured.update(kw))
    if family == "poly":
        assert tab._poly_run_function()
    else:
        assert tab._single_run_function()
    wait_simulation(tab)
    assert captured["consider_intensities"] is True
    assert captured["cif_file_path"] == "sample.cif"
    assert captured["intensity_cmap"] == "magma"
    assert captured["intensity_clim"] == (.05, .8)
    assert captured["polarization_factor"] == pytest.approx(.6)
    assert captured["polarization_axis_offset_deg"] == pytest.approx(15)
    assert captured["noise_distribution"] == "gaussian"
    assert captured["noise_level"] == pytest.approx(.12)
    controls.apply_polarization.setChecked(False)
    assert controls.options("sample.cif")["polarization_factor"] is None
