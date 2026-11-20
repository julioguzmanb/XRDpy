import os
import subprocess
import sys
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from trxrdpy.simulation import single_crystal, plot
from trxrdpy.simulation.deferred_plotting import gui_plot
from trxrdpy.simulation.gui.main_window import MainWindow
from trxrdpy.simulation.mapping_display import occupancy_grid
from trxrdpy.simulation.results import SimulationResult, save_result, load_result, plot_result


@pytest.fixture
def window(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(MainWindow, '_maybe_restore_autosave', lambda self: None)
    monkeypatch.setattr(MainWindow, '_autosave_path', lambda self: tmp_path / 'state.json')
    w = MainWindow(launch_directory=tmp_path)
    yield w
    w.close()
    w.deleteLater()
    app.processEvents()


def test_gui_responsive_snapshot_plot_thread_and_close_guard(window, monkeypatch, wait_simulation):
    main_id = threading.get_ident()
    entered, release = threading.Event(), threading.Event()
    captured, drawn, ticks = {}, [], []

    @gui_plot
    def figure():
        drawn.append(threading.get_ident())

    def compute(**kwargs):
        captured.update(kwargs)
        captured['thread'] = threading.get_ident()
        entered.set()
        assert release.wait(10)
        figure()

    monkeypatch.setattr(single_crystal, 'simulate_2d', compute)
    window.show()
    tab = window.single_tab
    tab.single_line_energy.setText('12000')
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)
    try:
        assert tab._single_run_function()
        assert entered.wait(3)
        assert captured['thread'] != main_id
        assert not tab.single_run_btn.isEnabled()
        assert not tab._single_run_function()
        tab.single_line_energy.setText('18000')
        window.tabs.setCurrentWidget(window.extra_functions_tab)
        assert not window.close()
        deadline = time.monotonic() + .1
        while time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(.005)
        assert len(ticks) >= 3
        assert captured['energy'] == 12000
        assert not drawn
    finally:
        release.set()
        wait_simulation(tab)
        timer.stop()
    assert drawn == [main_id]
    assert tab.single_run_btn.isEnabled()
    assert tab.single_line_energy.text() == '18000'
    assert window.tabs.currentWidget() is window.extra_functions_tab


def test_background_error_is_reported_without_blocking(window, monkeypatch, wait_simulation):
    def compute(**kwargs):
        raise ValueError('test worker failure')
    monkeypatch.setattr(single_crystal, 'simulate_2d', compute)
    tab = window.single_tab
    assert tab._single_run_function()
    wait_simulation(tab)
    assert tab.single_run_btn.isEnabled()
    assert 'test worker failure' in tab.last_error_traceback
    assert tab._simulation_dialog.status_label.text() == 'Error.'


@pytest.mark.parametrize('parallel', [False, True])
def test_safe_stop_mapping_and_restart(window, monkeypatch, wait_simulation, parallel):
    tab = window.single_tab
    from PyQt5.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: None)
    tab.single_func_combo.setCurrentText('scan_two_parameters_for_Bragg_condition')
    tab.single_param1_combo.setCurrentText('roty')
    tab.single_param2_combo.setCurrentText('rotz')
    tab.single_line_param1_range.setText('-90,90,1')
    tab.single_line_param2_range.setText('-90,90,0.001')
    tab.single_line_names.setText('[1,0,0]')
    tab.parallel_checkbox.setChecked(parallel)
    tab.workers_line.setText('2')
    assert tab._single_run_function()
    QTimer.singleShot(500, tab._simulation_dialog.request_cancel)
    wait_simulation(tab)
    assert 'stopped' in tab._simulation_dialog.status_label.text().lower() or 'cancel' in tab._simulation_dialog.status_label.text().lower()
    assert tab.last_scan_result is None
    assert tab.single_run_btn.isEnabled()
    monkeypatch.setattr(single_crystal, 'simulate_2d', lambda **kw: None)
    tab.single_func_combo.setCurrentText('simulate_2d')
    assert tab._single_run_function()
    wait_simulation(tab)
    assert tab._simulation_dialog.status_label.text() == 'Finished.'


def test_function_specific_controls(window):
    tab = window.single_tab
    tab.single_func_combo.setCurrentText('simulate_2d')
    assert tab.results_group.isHidden()
    tab.single_func_combo.setCurrentText('simulate_reciprocal')
    assert not tab.results_group.isHidden()
    assert tab.parallel_checkbox.isHidden()
    assert tab.mapping_display.isHidden()
    assert not tab.require_detector_hit.isHidden()
    tab.require_detector_hit.setChecked(True)
    assert not tab.single_det_group.isHidden()
    tab.single_func_combo.setCurrentText('scan_two_parameters_for_Bragg_condition')
    assert not tab.parallel_checkbox.isHidden()
    assert not tab.mapping_display.isHidden()


def test_plot_failure_is_reported_and_worker_released(window, monkeypatch, wait_simulation):
    @gui_plot
    def bad_plot():
        raise ValueError('test plotting failure')
    monkeypatch.setattr(single_crystal, 'simulate_2d', lambda **kwargs: bad_plot())
    tab = window.single_tab
    assert tab._single_run_function()
    wait_simulation(tab)
    assert tab._simulation_dialog.status_label.text() == 'Error.'
    assert 'test plotting failure' in tab.last_error_traceback


def test_gui_module_spawn_import_has_no_runpy_warning():
    proc = subprocess.run([sys.executable, '-W', 'error::RuntimeWarning', '-c',
        "import runpy; runpy.run_module('trxrdpy.simulation.gui.main_window', run_name='__mp_main__')"],
        capture_output=True, text=True, timeout=30, env=os.environ.copy())
    assert proc.returncode == 0, proc.stderr
    assert 'found in sys.modules' not in proc.stderr


def test_parallel_gui_scan_keeps_grid_metadata(window, monkeypatch, wait_simulation):
    tab = window.single_tab
    tab.single_func_combo.setCurrentText('scan_two_parameters_for_Bragg_condition')
    tab.single_param1_combo.setCurrentText('energy')
    tab.single_param2_combo.setCurrentText('roty')
    tab.single_line_param1_range.setText('7000,7002,1')
    tab.single_line_param2_range.setText('-1,1,1')
    tab.single_line_names.setText('[1,0,0]')
    tab.parallel_checkbox.setChecked(True)
    tab.workers_line.setText('2')
    main_id = threading.get_ident()
    calls = []
    monkeypatch.setattr(plot, 'plot_parameter_mapping', lambda *a, **kw: calls.append(threading.get_ident()))
    assert tab._single_run_function()
    wait_simulation(tab, timeout=30)
    assert tab.last_scan_result.metadata['grid_steps'] == [1, 1]
    assert calls == [main_id]
    assert tab._simulation_dialog.status_label.text() == 'Finished.'


def test_powder_reflection_preparation_also_runs_in_background(window, monkeypatch, wait_simulation):
    import trxrdpy.simulation.powder_preparation as preparation
    from trxrdpy.simulation import polycrystalline
    main_id = threading.get_ident()
    thread_ids = []
    def reflections(*args):
        thread_ids.append(threading.get_ident())
        return np.array([1.]), np.array([[1, 0, 0]])
    monkeypatch.setattr(preparation, 'powder_reflections', reflections)
    monkeypatch.setattr(polycrystalline, 'simulate_2d', lambda **kw: thread_ids.append(threading.get_ident()))
    tab = window.poly_tab
    tab.poly_func_combo.setCurrentText('simulate_2d')
    assert tab._poly_run_function()
    wait_simulation(tab)
    assert len(thread_ids) == 2 and all(identity != main_id for identity in thread_ids)


@pytest.mark.parametrize('mode', ['outlines', 'bands'])
def test_disconnected_branches_are_not_joined(mode, monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    points = np.array([(x, y) for x in [0, 1, 2, 7, 8, 9] for y in [0, 1, 2]])
    original = points.copy()
    ax = plot.plot_parameter_mapping({'[1 0 0]': points}, 'energy', 'roty', display_mode=mode, grid_steps=(1, 1))
    paths = ax.collections[0].get_paths()
    assert len(paths) >= 2
    for path in paths:
        polygon = path.vertices
        assert polygon[:, 0].max() < 3 or polygon[:, 0].min() > 6
    np.testing.assert_equal(points, original)
    plt.close(ax.figure)


def test_filled_band_preserves_hole_and_thickness(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    points = [(x, y) for x in range(7) for y in range(7) if not (2 <= x <= 4 and 2 <= y <= 4)]
    ax = plot.plot_parameter_mapping({'hkl': points}, 'roty', 'rotz', display_mode='bands', grid_steps=(1, 1))
    paths = ax.collections[0].get_paths()
    polygons = [path.vertices for path in paths]
    areas = [np.sum(p[:-1, 0]*p[1:, 1] - p[1:, 0]*p[:-1, 1])/2 for p in polygons]
    assert sum(abs(a) for a in areas) == 40
    assert not any(path.contains_point((3, 3)) for path in paths)
    assert all(any(path.contains_point(point) for path in paths) for point in points)
    plt.close(ax.figure)


def test_missing_rows_angle_seam_duplicates_and_size_guard():
    points = np.array([[-180, 0], [-179, 0], [179, 0], [180, 0], [-180, 0]])
    x, y, grid = occupancy_grid(points, (1, 1))
    assert grid.sum() == 4
    assert not grid[:, (x > -178) & (x < 178)].any()
    with pytest.raises(ValueError, match='too large'):
        occupancy_grid([[0, 0], [10000, 10000]], (1, 1))
    with pytest.raises(ValueError, match='regular'):
        occupancy_grid([[0, 0], [.3, .3]], (.2, .2))
    with pytest.raises(ValueError, match='unavailable'):
        occupancy_grid([[0, 0], [100, 100]])


def test_band_legend_toggle_and_single_cell(monkeypatch):
    from matplotlib.backend_bases import MouseEvent, PickEvent
    monkeypatch.setattr(plt, 'show', lambda: None)
    ax = plot.plot_parameter_mapping({'hkl': [(2, 5)]}, 'roty', 'rotz', display_mode='bands', grid_steps=(1, 1))
    handle = ax.get_legend().get_lines()[0]
    event = PickEvent('pick_event', ax.figure.canvas,
                      MouseEvent('button_press_event', ax.figure.canvas, 0, 0), handle)
    ax.figure.canvas.callbacks.process('pick_event', event)
    assert not ax.collections[0].get_visible()
    ax.figure.canvas.callbacks.process('pick_event', event)
    assert ax.collections[0].get_visible()
    plt.close(ax.figure)


def test_display_state_and_saved_result_replot(window, tmp_path, monkeypatch):
    tab = window.single_tab
    tab.mapping_display.setCurrentIndex(tab.mapping_display.findData('bands'))
    saved = window._gather_gui_state()
    tab.mapping_display.setCurrentIndex(0)
    window._apply_gui_state(saved)
    assert tab.mapping_display.currentData() == 'bands'
    old = {**saved, 'single': {k: v for k, v in saved['single'].items() if k != 'mapping_display'}}
    window._apply_gui_state(old)
    assert tab.mapping_display.currentData() == 'points'
    result = SimulationResult('parameter_mapping', {'points': {'hkl': [[0, 0], [1, 0]]}},
                              {'axes': ['energy', 'roty'], 'grid_steps': [1, .1]})
    restored = load_result(save_result(tmp_path/'map.h5', result))
    calls = []
    monkeypatch.setattr(plot, 'plot_parameter_mapping', lambda *a, **kw: calls.append(kw))
    plot_result(restored, display_mode='outlines')
    assert calls == [{'display_mode': 'outlines', 'grid_steps': [1, .1]}]
