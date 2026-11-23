"""Shared 3D detector styling and visible single-crystal Bragg hits."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from trxrdpy.simulation import single_crystal, utils
from trxrdpy.simulation.detector import Detector
from trxrdpy.simulation.detector_view import pixels_in_lab
from trxrdpy.simulation.deferred_plotting import capture_plots
from trxrdpy.simulation.experiment import Experiment
from trxrdpy.simulation.sample import LatticeStructure


def _experiment_with_three_rays(angles=(8, 10, -12)):
    det = Detector(detector_type="manual", pxsize_h=1e-4, pxsize_v=1e-4,
                   num_pixels_h=1000, num_pixels_v=500, dist=0.1,
                   poni1=0.025, poni2=0.05)
    det.set_transform(utils.make_transform(
        Rotation.from_euler("xyz", angles, degrees=True).as_matrix(),
        translation=[0.005, 0.003, -0.002]))
    lattice = LatticeStructure(space_group=221, a=4, b=4, c=4,
                               alpha=90, beta=90, gamma=90)
    lattice.hkls_in_Bragg_condition = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    lattice.q_hkls_in_Bragg_condition = np.array([
        [-0.7201, 0., 3.1260], [-0.3601, 1.6040, 1.5630], [-1., -1., 0.]])
    exp = Experiment(det, lattice, energy=15000, e_bandwidth=1.5)
    pixels = np.array([[500, 250], [1200, 250], [300, 200]], dtype=float)
    native = np.column_stack((np.full(len(pixels), det.dist),
                              det.poni2 - (pixels[:, 0] + 0.5) * det.pxsize_h,
                              -det.poni1 + (pixels[:, 1] + 0.5) * det.pxsize_v))
    exp.scattered_directions = utils.apply_transform(native, det.get_transform())
    exp.scat_dir_sign = np.array([1, 1, -1])
    exp.hkl_to_color = {(1, 0, 0): "blue", (2, 0, 0): "green", (3, 0, 0): "red"}
    return exp


def test_single_crystal_3d_marks_only_forward_detector_hits(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    exp = _experiment_with_three_rays()
    assert exp.detector.lab_grid is None
    exp.plot_3d_single_xstal_exp(plot_crystal=False)
    ax = plt.gcf().axes[0]
    grid = [artist for artist in ax.collections
            if artist.get_gid() == "detector-visual-grid"]
    hits = [artist for artist in ax.collections if artist.get_gid() == "bragg-hit"]
    assert len(grid) == 1
    assert len(hits) == 1
    np.testing.assert_allclose(exp.pixel_positions,
                               [[500, 250], [1200, 250], [300, 200]], atol=1e-9)
    np.testing.assert_allclose(
        np.array(hits[0]._offsets3d).ravel(), exp.scattered_directions[0])

    # The legend keeps the ray, detector dot, and label together.
    legend_line = ax.get_legend().get_lines()[0]
    ax.figure.canvas.callbacks.process("pick_event", SimpleNamespace(artist=legend_line))
    assert not hits[0].get_visible()
    assert not ax.lines[-3].get_visible()
    plt.close("all")


def test_single_crystal_3d_hit_hover_tracks_rotation_and_legend(monkeypatch):
    from matplotlib.backend_bases import MouseEvent
    from mpl_toolkits.mplot3d import proj3d
    monkeypatch.setattr(plt, 'show', lambda: None)
    exp = _experiment_with_three_rays()
    exp.plot_3d_single_xstal_exp(plot_crystal=False)
    ax = plt.gcf().axes[0]
    fig = ax.figure

    def hover_hit():
        fig.canvas.draw()
        x, y, _ = proj3d.proj_transform(*exp.scattered_directions[0], ax.get_proj())
        px, py = ax.transData.transform((x, y))
        fig.canvas.callbacks.process('motion_notify_event',
            MouseEvent('motion_notify_event', fig.canvas, px, py))

    hover_hit()
    tooltip = ax._single_crystal_hover_text
    assert tooltip.get_visible()
    assert 'hkl=(1, 0, 0)' in tooltip.get_text()
    assert 'Q=(-0.7201, 0.0000, 3.1260) Å⁻¹' in tooltip.get_text()
    assert '|Q|=3.2079 Å⁻¹' in tooltip.get_text()
    assert 'Azimuth=0.00°' in tooltip.get_text()
    assert 'Elevation=77.03°' in tooltip.get_text()

    ax.view_init(elev=30, azim=130)
    hover_hit()
    assert tooltip.get_visible()
    legend_line = ax.get_legend().get_lines()[0]
    fig.canvas.callbacks.process('pick_event', SimpleNamespace(artist=legend_line))
    hover_hit()
    assert not tooltip.get_visible()
    plt.close(fig)


@pytest.mark.parametrize("angles,expected_count", [
    ((8, 10, -12), 1),
    ((8, 20, -12), 0),
])
def test_single_crystal_3d_direct_beam_cross_follows_detector_hit(
    angles, expected_count, monkeypatch
):
    monkeypatch.setattr(plt, "show", lambda: None)
    exp = _experiment_with_three_rays(angles=angles)
    exp.plot_3d_single_xstal_exp(plot_crystal=False)
    markers = [line for line in plt.gcf().axes[0].lines
               if line.get_gid() == "direct-beam"]
    assert len(markers) == expected_count
    if markers:
        expected = pixels_in_lab(exp.detector, [exp.detector.direct_beam_pixel()])[0]
        np.testing.assert_allclose(np.asarray(markers[0].get_data_3d()).ravel(), expected)
        assert markers[0].get_color() == "red"
        assert markers[0].get_marker() == "x"
    plt.close("all")


@pytest.mark.parametrize("function", [single_crystal.simulate_3d,
                                       single_crystal.simulate_3d_with_geometry])
def test_single_crystal_3d_does_not_allocate_full_detector_grid(function, monkeypatch):
    monkeypatch.setattr(Detector, "calculate_lab_grid",
                        lambda self, *args, **kwargs: pytest.fail("full grid allocated"))
    kwargs = dict(det_ntum_pixels_h=1000, det_num_pixels_v=500,
                  det_pxsize_h=1e-4, det_pxsize_v=1e-4,
                  det_dist=0.1, det_poni1=0.025, det_poni2=0.05,
                  energy=15000, sam_space_group=221,
                  sam_a=4, sam_b=4, sam_c=4,
                  sam_alpha=90, sam_beta=90, sam_gamma=90, qmax=3)
    with capture_plots() as calls:
        exp = function(**kwargs)
    assert exp.detector.lab_grid is None
    assert exp.pixel_positions.shape[1] == 2
    assert len(calls) == 1
