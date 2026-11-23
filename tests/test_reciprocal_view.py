import sys
import threading
from pathlib import Path
from urllib.parse import unquote, urlparse

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from trxrdpy.simulation import plot, reciprocal, single_crystal, sample, utils
from trxrdpy.simulation.geometry import DiffractometerGeometry
from trxrdpy.simulation.gui.services import SimulationService
from trxrdpy.simulation.gui.state import GuiState
from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab
from trxrdpy.simulation.results import SimulationResult, save_result, load_result, plot_result


def data():
    k = 2 * np.pi
    q = np.array([[-k, k, 0], [1, 0, 0], [-2 * k, 0, 0]])
    return q, np.array([[1, 1, 0], [1, 0, 0], [2, 0, 0]])


def test_plotly_q_coordinates_grouped_traces_and_bandwidth():
    q, hkls = data()
    fig = plot.plot_reciprocal(q, hkls, 1e-10, 2, show=False)
    assert len(fig.data) == 7
    expected = sample.check_Bragg_condition(q, 1e-10, 2)
    np.testing.assert_equal(fig.data[1].customdata[:, :3], hkls[expected])
    np.testing.assert_equal(fig.data[1].x, q[expected, 0])
    x, y, z = map(np.asarray, (fig.data[2].x, fig.data[2].y, fig.data[2].z))
    k = 2 * np.pi
    np.testing.assert_allclose((x + k)**2 + y**2 + z**2, k*k)
    assert x.max() == 0
    for trace, scale in zip(fig.data[3:5], [.99, 1.01]):
        x, y, z = map(np.asarray, (trace.x, trace.y, trace.z))
        valid = np.isfinite(x)
        np.testing.assert_allclose((x[valid] + k*scale)**2 + y[valid]**2 + z[valid]**2, (k*scale)**2)
    assert fig.layout.scene.aspectmode == 'cube'
    assert 'Q<sub>x</sub>' in fig.layout.scene.xaxis.title.text
    assert list(fig.data[-1].cells.values[0]) == ['(1, 1, 0)', '(2, 0, 0)']
    assert fig.data[-1].domain.x[0] > fig.layout.scene.domain.x[1]


def test_reciprocal_hover_angles():
    q = np.array([[-1., 1., 1.], [0., 0., 2.]])
    hkls = np.array([[1, 1, 1], [0, 0, 1]])
    fig = plot.plot_reciprocal(q, hkls, 1e-10, 2, show=False)
    point_data = np.vstack([fig.data[0].customdata, fig.data[1].customdata])
    np.testing.assert_allclose(point_data[0, 3:], [np.sqrt(3), -45, np.degrees(np.arctan2(1, np.sqrt(2)))])
    assert point_data[1, 4] == 0
    assert point_data[1, 5] == 90
    assert 'Azimuth=' in fig.data[1].hovertemplate
    assert 'Elevation=' in fig.data[1].hovertemplate


def test_lab_azimuth_left_up_right():
    q = np.array([[0., 1., 0.], [0., 0., 1.], [0., -1., 0.], [-1., 0., 0.]])
    azimuth, _ = reciprocal._q_angles(q)
    np.testing.assert_array_equal(azimuth[:3], [-90, 0, 90])
    assert np.isnan(azimuth[3])


def test_matplotlib_spot_hover_and_legend_filter():
    from matplotlib.backend_bases import MouseEvent, PickEvent
    from mpl_toolkits.mplot3d import proj3d
    k = 2 * np.pi
    q = np.array([[-k, k / np.sqrt(2), k / np.sqrt(2)], [-2 * k, 0, 0]])
    hkls = np.array([[1, 1, 1], [2, 0, 0]])
    fig, ax = plot.plot_reciprocal(q, hkls, 1e-10, 2, backend='matplotlib', show=False)
    fig.canvas.draw()

    def hover(point):
        x, y, _ = proj3d.proj_transform(*point, ax.get_proj())
        px, py = ax.transData.transform((x, y))
        fig.canvas.callbacks.process('motion_notify_event',
            MouseEvent('motion_notify_event', fig.canvas, px, py))

    hover(q[0])
    assert ax._reciprocal_annotation.get_visible()
    details = ax._reciprocal_annotation.get_text()
    assert 'hkl=(1, 1, 1)' in details
    assert 'Q=(' in details and '|Q|=' in details
    assert 'Azimuth=-45.00°' in details
    assert 'Elevation=30.00°' in details

    label = next(text for text in ax.get_legend().get_texts() if text.get_text() == '(1, 1, 1)')
    fig.canvas.callbacks.process('pick_event', PickEvent('pick_event', fig.canvas,
        MouseEvent('button_press_event', fig.canvas, 0, 0), label))
    fig.canvas.draw()
    hover(q[1])
    assert 'hkl=(2, 0, 0)' in ax._reciprocal_annotation.get_text()
    plt.close(fig)


def test_large_point_set_keeps_constant_trace_count(tmp_path):
    rng = np.random.default_rng(4)
    q = rng.uniform(-5, 5, (100_000, 3))
    hkls = rng.integers(-20, 20, q.shape)
    path = tmp_path / 'reciprocal.html'
    fig = plot.plot_reciprocal(q, hkls, 1e-10, 1.5, show=False, html_path=path)
    assert len(fig.data) == 7
    assert len(fig.data[0].x) + len(fig.data[1].x) == len(q)
    html = path.read_text()
    assert 'Plotly.newPlot' in html
    assert '<script src=' not in html  # Fully offline, including Plotly.js.


def test_matplotlib_fallback_and_no_global_font_change():
    q, hkls = data()
    size = plt.rcParams['font.size']
    fig, ax = plot.plot_reciprocal(q, hkls, 1e-10, 2, backend='matplotlib', show=False)
    assert len(ax.collections) == 3  # Two point groups and one surface.
    assert plt.rcParams['font.size'] == size
    fig.canvas.draw()
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert '(1, 1, 0)' in labels and '(2, 0, 0)' in labels
    plt.close(fig)


def test_missing_plotly_does_not_break_matplotlib(monkeypatch):
    q, hkls = data()
    monkeypatch.setitem(sys.modules, 'plotly.graph_objects', None)
    with pytest.raises(ImportError, match='Install plotly'):
        plot.plot_reciprocal(q, hkls, 1e-10, 1, show=False)
    fig, _ = plot.plot_reciprocal(q, hkls, 1e-10, 1, backend='matplotlib', show=False)
    plt.close(fig)


def test_matplotlib_legend_clicks():
    from matplotlib.backend_bases import MouseEvent, PickEvent
    q, hkls = data()
    fig, ax = plot.plot_reciprocal(q, hkls, 1e-10, 2, backend='matplotlib', show=False)
    texts = {text.get_text(): text for text in ax.get_legend().get_texts()}
    def click(name):
        fig.canvas.callbacks.process('pick_event', PickEvent('pick_event', fig.canvas,
            MouseEvent('button_press_event', fig.canvas, 0, 0), texts[name]))
    click('(1, 1, 0)')
    assert len(ax.collections[1]._offsets3d[0]) == 1
    click('(1, 1, 0)')
    assert len(ax.collections[1]._offsets3d[0]) == 2
    click('Ewald sphere')
    assert not ax.collections[2].get_visible()
    click('In Bragg condition (2)')
    assert not ax.collections[1].get_visible()
    plt.close(fig)


@pytest.mark.parametrize('wavelength,bandwidth', [(0, 1), (np.nan, 1), (1e-10, -1), (1e-10, 200)])
def test_invalid_beam_inputs(wavelength, bandwidth):
    q, hkls = data()
    with pytest.raises(ValueError):
        plot.plot_reciprocal(q, hkls, wavelength, bandwidth, show=False)


def test_empty_and_monochromatic_plot():
    fig = plot.plot_reciprocal([], [], 1e-10, 0, show=False, show_non_bragg=False)
    assert len(fig.data) == 7
    assert len(fig.data[3].x) == len(fig.data[4].x) == 0
    assert fig.data[0].visible == 'legendonly'
    assert list(fig.data[-1].cells.values[0]) == ['None']


def test_qmax_below_first_reflection():
    result = single_crystal.simulate_reciprocal(
        sam_space_group=221, sam_a=4, sam_b=4, sam_c=4,
        sam_alpha=90, sam_beta=90, sam_gamma=90, qmax=.01, show=False)
    assert result['Q_hkls'].shape == (0, 3)
    assert result['hkls'].shape == (0, 3)


def test_detector_highlight_is_display_only(monkeypatch):
    from trxrdpy.simulation import experiment
    q, hkls = data()
    original = q.copy()
    monkeypatch.setattr(experiment, 'detector_hit_mask', lambda q, w, d: np.array([True, False]))
    fig = plot.plot_reciprocal(q, hkls, 1e-10, 2, show=False,
                              require_detector_hit=True, detector_obj=object())
    hit = fig.data[-1]
    np.testing.assert_equal(hit.customdata[:, :3], [[1, 1, 0]])
    np.testing.assert_allclose(hit.customdata[:, 3],
                               np.linalg.norm(np.column_stack((hit.x, hit.y, hit.z)), axis=1))
    assert hit.hovertemplate.split('<extra>')[0] == fig.data[1].hovertemplate.split('<extra>')[0]
    assert '<br>Q=' in hit.hovertemplate and '<br>|Q|' in hit.hovertemplate
    assert '<extra>Detector hit</extra>' in hit.hovertemplate
    np.testing.assert_equal(fig.data[1].customdata[:, :3], [[2, 0, 0]])
    assert fig.data[-1].marker.color == '#159447'
    assert list(fig.data[-2].cells.font.color[0]) == ['#159447', '#d44929']
    np.testing.assert_equal(q, original)
    with pytest.raises(ValueError, match='detector'):
        plot.plot_reciprocal(q, hkls, 1e-10, 2, show=False, require_detector_hit=True)


def test_sample_pose_qmax_and_result_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, 'plot_reciprocal', lambda **kwargs: None)
    kwargs = dict(sam_space_group=221, sam_a=4, sam_b=4, sam_c=4,
                  sam_alpha=90, sam_beta=90, sam_gamma=90, qmax=3, energy=15000)
    plain = single_crystal.simulate_reciprocal(**kwargs, show=False)
    rotated = single_crystal.simulate_reciprocal(**kwargs, sam_rotz=30, show=False)
    ordered = single_crystal.simulate_reciprocal(
        **kwargs, sam_rotation_order='xzy', sam_roty=-20.1,
        sam_rotz=45, show=False)
    assert 'Sample rotations [°, order xzy, relative to initial]' in ordered['title']
    assert 'rotx=0.00, roty=-20.10, rotz=45.00' in ordered['title']
    assert 'Equivalent sample Euler' not in ordered['title']
    assert np.all(np.linalg.norm(plain['Q_hkls'], axis=1) <= 3)
    assert np.all(np.linalg.norm(plain['Q_hkls'], axis=1) > 0)
    np.testing.assert_allclose(rotated['Q_hkls'], utils.apply_rotation(plain['Q_hkls'], 0, 0, 30), atol=1e-12)
    geometry = DiffractometerGeometry.from_dict({'sample': [{'name': 'omega', 'axis': 'z'}]})
    chained = single_crystal.simulate_reciprocal(**kwargs, geometry=geometry, sample_angles={'omega': 30}, show=False)
    assert 'Sample convention: diffractometer; omega[local]' in chained['title']
    assert 'Sample motors [°]: omega=30.00' in chained['title']
    assert 'Equivalent sample Euler [°, fixed lab xyz' in chained['title']
    np.testing.assert_allclose(chained['Q_hkls'], rotated['Q_hkls'], atol=1e-12)
    restored = load_result(save_result(tmp_path / 'reciprocal.h5', SimulationResult('reciprocal', rotated, {})))
    captured = []
    monkeypatch.setattr(plot, 'plot_reciprocal', lambda **kw: captured.append(kw))
    plot_result(restored, reciprocal_options={'backend': 'matplotlib', 'show': False})
    assert captured[0]['backend'] == 'matplotlib'
    np.testing.assert_equal(captured[0]['Q_hkls'], rotated['Q_hkls'])


def test_gui_reciprocal_background_state_and_replot(monkeypatch, wait_simulation):
    app = QApplication.instance() or QApplication([])
    main_id = threading.get_ident()
    opened = []
    def open_browser(url, **kwargs):
        opened.append((threading.get_ident(), Path(unquote(urlparse(url).path))))
        return True
    monkeypatch.setattr(reciprocal.webbrowser, 'open', open_browser)
    old = GuiState.from_dict({'state_version': 13})
    assert old.single.reciprocal_backend == 'plotly'
    tab = SingleCrystalTab(SimulationService(), old)
    tab.single_func_combo.setCurrentText('simulate_reciprocal')
    tab.single_line_qmax.setText('2')
    tab.single_line_dist.setText('')  # Reciprocal-space calculations do not need a detector.
    tab.reciprocal_bandwidth.setChecked(False)
    assert tab._single_run_function()
    wait_simulation(tab)
    assert tab.last_scan_result.kind == 'reciprocal'
    assert opened and opened[0][0] == main_id and opened[0][1].exists()
    state = GuiState.from_dict(tab.state.to_dict())
    assert state.single.func == 'simulate_reciprocal'
    assert state.single.reciprocal_bandwidth is False
    tab._plot_scan_result()
    wait_simulation(tab)
    assert len(opened) == 2 and opened[1][0] == main_id
    for _, path in opened:
        path.unlink()
        path.parent.rmdir()
    tab.close()
    app.processEvents()
