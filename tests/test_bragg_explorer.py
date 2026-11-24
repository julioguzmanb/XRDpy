import csv
import json
from threading import Event

import numpy as np
import pytest

from trxrdpy.simulation.bragg_explorer import BraggExplorer
from trxrdpy.simulation.results import SimulationResult, load_result, save_result


def mapping():
    hkls = ('[1 0 0]', '[0 1 0]', '[0 0 1]')
    rows = [
        {hkls[0]: [(10000, -5)], hkls[1]: [], hkls[2]: []},
        {hkl: [] for hkl in hkls},
        {hkls[0]: [], hkls[1]: [(11000, -4)], hkls[2]: []},
        {hkls[0]: [(12000, -3)], hkls[1]: [], hkls[2]: []},
    ]
    context = json.dumps(dict(initial_orientation=np.eye(3).tolist(), rotation_order='xyz',
                              energy=10000, motors=None, fixed_angles={}, scan_angles={}))
    data = dict(format='mapping-slices-v1', extra_axes={'rotx': np.array([-3, 3]),
                'rotz': np.array([0, 1])}, plot_axes=('energy', 'roty'),
                plot_ranges=np.array([[10000, 12000, 1000], [-5, -3, 1]]),
                grid_steps=np.array([1000, 1]),
                slices={str(i): dict(points=points, context_json=context) for i, points in enumerate(rows)})
    return SimulationResult('parameter_mapping', data, {})


@pytest.mark.parametrize('saved', [False, True])
def test_hkl_lookup_and_csv_subset(saved, tmp_path):
    result = mapping()
    if saved:
        result = load_result(save_result(tmp_path / 'mapping.h5', result))
    browser = BraggExplorer(result)
    assert browser.hkls() == ['[1 0 0]', '[0 1 0]']
    assert browser.positions('[1 0 0]') == [0, 3]
    assert browser.positions('[0 0 1]') == []
    assert browser.position_values(3) == (3, 1)
    assert browser.point_count(0, '[1 0 0]') == 1
    assert browser.point_count(1, '[1 0 0]') == 0
    np.testing.assert_array_equal(browser.view(3, '[1 0 0]'), [[12000, -3]])
    path = tmp_path / 'subset.csv'
    assert browser.export_csv(path, '[1 0 0]', [3]) == 1
    with path.open(newline='') as stream:
        assert list(csv.reader(stream)) == [
            ['hkl', 'rotx [deg]', 'rotz [deg]', 'energy [eV]', 'roty [deg]', 'application_order'],
            ['[1 0 0]', '3.0', '1.0', '12000.0', '-3.0', 'xyz']]
    assert browser.export_csv(path, '[1 0 0]', [0, 3]) == 2
    stopped = Event()
    stopped.set()
    with pytest.raises(InterruptedError):
        browser.export_csv(path, '[1 0 0]', [0, 3], cancel=stopped)
    with path.open(newline='') as stream:
        assert len(list(csv.reader(stream))) == 3


def test_gui_reader_opens_for_loaded_mapping(tmp_path):
    from PyQt5.QtWidgets import QApplication
    from trxrdpy.simulation.gui.state import GuiState
    from trxrdpy.simulation.gui.services import SimulationService
    from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab
    app = QApplication.instance() or QApplication([])
    result = load_result(save_result(tmp_path / 'mapping.h5', mapping()))
    tab = SingleCrystalTab(SimulationService(), GuiState())
    tab._open_bragg_explorer(result)
    app.processEvents()
    assert len(tab._bragg_explorer_dialogs) == 1
    dialog = tab._bragg_explorer_dialogs[0]
    assert dialog.windowTitle() == 'Explore Bragg Conditions'
    assert dialog.hkl_combo.currentText() == '(1, 0, 0)'
    assert dialog.positions_model.positions == [0, 3]
    assert dialog.points_model.rowCount() == 1
    dialog.close()
    tab.close()
    app.processEvents()


def test_gui_explore_button_needs_no_plot(tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QApplication, QFileDialog
    from trxrdpy.simulation.gui.state import GuiState
    from trxrdpy.simulation.gui.services import SimulationService
    from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab
    app = QApplication.instance() or QApplication([])
    tab = SingleCrystalTab(SimulationService(), GuiState())
    plot_calls = []
    monkeypatch.setattr(tab, '_plot_scan_result', lambda: plot_calls.append(True))
    tab._receive_scan_result(mapping())
    assert len(plot_calls) == 1  # Existing automatic plot behavior on a new run.
    assert tab.explore_result_button.isEnabled()
    tab.explore_result_button.click()
    app.processEvents()
    assert len(plot_calls) == 1  # Opening the reader itself needs no plot.
    assert len(tab._bragg_explorer_dialogs) == 1
    tab._bragg_explorer_dialogs[0].close()
    path = save_result(tmp_path / 'mapping.h5', mapping())
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *args: (str(path), ''))
    tab.load_result_button.click()
    assert tab.explore_result_button.isEnabled()
    tab.explore_result_button.click()
    app.processEvents()
    assert len(plot_calls) == 1
    assert len(tab._bragg_explorer_dialogs) == 1
    tab._bragg_explorer_dialogs[0].close()
    tab.close()
    app.processEvents()


def test_zero_detector_visible_points_are_explained(monkeypatch):
    from PyQt5.QtWidgets import QApplication
    from trxrdpy.simulation.gui.widgets.bragg_explorer_dialog import BraggExplorerDialog
    app = QApplication.instance() or QApplication([])
    original = BraggExplorer.view
    monkeypatch.setattr(BraggExplorer, 'view', lambda self, index, hkl, detector_obj=None:
                        np.empty((0, 2)) if detector_obj is not None
                        else original(self, index, hkl))
    dialog = BraggExplorerDialog(mapping(), lambda: object(), detector_visible=True)
    assert dialog.positions_model.counts[0] == 1
    assert dialog.points_model.rowCount() == 0
    assert '0 detector-visible' in dialog.points_label.text()
    assert '1 saved Bragg' in dialog.points_label.text()
    dialog.view_combo.setCurrentIndex(0)
    assert dialog.points_model.rowCount() == 1
    dialog.close()
    app.processEvents()


def test_gui_reader_exports_selected_subset(tmp_path, monkeypatch):
    import time
    from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
    from trxrdpy.simulation.gui.widgets.bragg_explorer_dialog import BraggExplorerDialog
    app = QApplication.instance() or QApplication([])
    path = tmp_path / 'selected.csv'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(path), 'CSV files (*.csv)'))
    errors = []
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: errors.append(args[-1]))
    monkeypatch.setattr(QMessageBox, 'information', lambda *args: None)
    dialog = BraggExplorerDialog(mapping(), lambda: None)
    dialog._export()
    deadline = time.monotonic() + 10
    while (not path.exists() or dialog._export_running) and not errors and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert not errors
    assert path.exists()
    assert not dialog._export_running
    with path.open(newline='') as stream:
        assert len(list(csv.reader(stream))) == 2
    dialog.close()
    app.processEvents()


def test_detector_filtered_export_preserves_raw_mapping(tmp_path):
    from test_mapping_display_filter import result_for
    from test_simulation_detector_filter import parameters, detector
    from trxrdpy.simulation.mapping_filter import filtered_mapping
    values = parameters()
    raw = result_for(values)
    context = json.dumps(raw.metadata['mapping_context'])
    data = dict(format='mapping-slices-v1', extra_axes={'rotx': np.array([0.])},
                plot_axes=tuple(raw.metadata['axes']),
                plot_ranges=np.array([values['param1_range'], values['param2_range']]),
                grid_steps=np.array([abs(values['param1_range'][2]), abs(values['param2_range'][2])]),
                slices={'0': dict(points=raw.data['points'], context_json=context)})
    result = load_result(save_result(tmp_path / 'complete.h5',
                                    SimulationResult('parameter_mapping', data, {})))
    browser = BraggExplorer(result)
    hkl = browser.hkls()[0]
    original = browser.view(0, hkl)
    expected = np.asarray(filtered_mapping(raw, detector()).data['points'][hkl]).reshape(-1, 2)
    np.testing.assert_array_equal(browser.view(0, hkl, detector()), expected)
    assert len(browser.view(0, hkl, detector(poni2=10))) == 0
    path = tmp_path / 'visible.csv'
    assert browser.export_csv(path, hkl, [0], detector()) == len(expected)
    assert browser.export_csv(path, hkl, [0], detector(poni2=10)) == 0
    np.testing.assert_array_equal(BraggExplorer(load_result(tmp_path / 'complete.h5')).view(0, hkl),
                                  original)
