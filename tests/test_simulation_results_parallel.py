import numpy as np
import pytest
import pyFAI
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from trxrdpy.simulation import single_crystal, experiment, sample
from trxrdpy.simulation.results import SimulationResult, save_result, load_result, plot_result
from trxrdpy.simulation.gui.state import GuiState
from trxrdpy.simulation.gui.services import SimulationService
from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab
from trxrdpy.simulation.gui.tabs.polycrystalline_tab import PolycrystallineTab
from trxrdpy.simulation.gui.widgets.detector_export import export_current_detector, current_detector


@pytest.mark.parametrize('interval', [(-10, 10, 10), (.1, .5, .1)])
def test_parameter_scan_parallel_matches_serial(interval):
    kwargs = dict(param1_name='roty', param2_name='rotz', param1_range=interval,
                  param2_range=(-10, 10, 10), sam_space_group=221, sam_a=4, sam_b=4, sam_c=4,
                  sam_alpha=90, sam_beta=90, sam_gamma=90, energy=15000,
                  hkls_names=[[1, 0, 0], [0, 1, 0]], e_bandwidth=50)
    assert single_crystal.scan_two_parameters_for_Bragg_condition(**kwargs) == \
        single_crystal.scan_two_parameters_for_Bragg_condition(**kwargs, parallel=True, workers=2)


def test_detector_scan_parallel_matches_serial_without_pixel_grid(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('The scan must not allocate an image grid.')
    monkeypatch.setattr(experiment.detector.Detector, 'calculate_lab_grid', fail)
    kwargs = dict(q_hkls=np.array([[-.1, 1., 0.]]), hkls=np.array([[1, 0, 0]]), wavelength=1e-10,
                  detector_type='manual', pxsize_h=.001, pxsize_v=.001, num_pixels_h=200,
                  num_pixels_v=200, dist=.1, poni1=.1, poni2=.1, angle_range=(-10, 10, 10))
    assert experiment.find_detector_rotations(**kwargs) == \
        experiment.find_detector_rotations(**kwargs, parallel=True, workers=2)


def test_target_orientations_parallel_matches_serial():
    kwargs = dict(q0=np.array([1., 0, 0]), q_targets=np.array([[0., 1, 0], [0, 0, 1.]]),
                  q_target_pixels=np.array([[1., 2], [3., 4]]), q_target_errors_px=np.array([.1, .2]),
                  rotation_order='xyz', phi_samples=5, display_wrapped_angles=True,
                  eta_deg_kept=np.array([10., 20.]))
    serial = single_crystal._solve_orientations_for_q_targets(**kwargs)
    parallel = single_crystal._solve_orientations_for_q_targets(**kwargs, parallel=True, workers=2)
    for name in ('eta_deg', 'phi_deg', 'rotx_deg', 'roty_deg', 'rotz_deg', 'predicted_pixels'):
        np.testing.assert_allclose(getattr(serial, name), getattr(parallel, name))


def test_result_roundtrip_and_plot_without_simulation(tmp_path, monkeypatch):
    result = SimulationResult('parameter_mapping', {'points': {'[1 0 0]': [(1., 2.), (3., 4.)],
                                                               '[0 1 0]': []}},
                              {'axes': ['roty', 'energy'], 'gui_state': {'version': 13}})
    path = save_result(tmp_path / 'result', result)
    with pytest.raises(FileExistsError):
        save_result(path, result)
    restored = load_result(path)
    np.testing.assert_equal(restored.data['points']['[1 0 0]'], [[1, 2], [3, 4]])
    assert restored.metadata == result.metadata
    captured = []
    from trxrdpy.simulation import plot
    monkeypatch.setattr(plot, 'plot_parameter_mapping', lambda *args: captured.append(args))
    plot_result(restored)
    assert captured[0][1:] == ('roty', 'energy')
    assert not list(tmp_path.glob('.simulation-*.tmp'))


def test_result_keeps_reflection_order_and_motor_projections(tmp_path, monkeypatch):
    points = {str(i): {'a': [i], 'b': [i + 1], 'c': [i + 2]} for i in range(13)}
    result = SimulationResult('detector_mapping', {'points': points}, {'motor_names': ['a', 'b', 'c']})
    restored = load_result(save_result(tmp_path / 'motors.h5', result))
    assert list(restored.data['points']) == list(points)
    from trxrdpy.simulation import plot
    calls = []
    monkeypatch.setattr(plot, 'plot_parameter_mapping', lambda *args: calls.append(args))
    plot_result(restored)
    assert [call[1:] for call in calls] == [('a', 'b'), ('a', 'c'), ('b', 'c')]


@pytest.mark.parametrize('prefix,cls', [('single', SingleCrystalTab), ('poly', PolycrystallineTab)])
def test_detector_template_edits_state_and_export(prefix, cls, tmp_path):
    app = QApplication.instance() or QApplication([])
    tab = cls(SimulationService(), GuiState())
    field = lambda name: getattr(tab, prefix + name)
    field('_combo_det_type').setCurrentText('pilatus100k')
    native = pyFAI.detectors.detector_factory('pilatus100k')
    assert int(field('_line_num_px_h').text()) == native.max_shape[1]
    assert int(field('_line_num_px_v').text()) == native.max_shape[0]
    assert field('_line_num_px_h').isReadOnly()
    assert not field('_line_pxsize_h').isReadOnly()
    field('_line_pxsize_h').setText('0.0002')
    field('_line_roty').setText('30')
    field('_line_rotz').setText('15')
    tab.save_to_state()
    saved = tab.state.to_dict()
    restored = cls(SimulationService(), GuiState())
    restored.state = GuiState.from_dict(saved)
    restored.load_from_state()
    assert getattr(restored, prefix + '_line_pxsize_h').text() == '0.0002'
    path = export_current_detector(tab, prefix, tmp_path / (prefix + '.poni'))
    ai = pyFAI.load(str(path))
    det = current_detector(tab, prefix)
    basis = np.array([[0., 0, 1], [0, -1, 0], [1., 0, 0]])
    np.testing.assert_allclose(basis @ ai.rotation_matrix() @ basis.T, det.rotation_matrix, atol=1e-14)
    assert ai.detector.max_shape == (det.num_pixels_v, det.num_pixels_h)
    assert ai.pixel2 == .0002
    assert ai.wavelength == pytest.approx(1.239841984e-6 / 15000)
    tab.close()
    restored.close()
    app.processEvents()


def test_old_named_detector_state_uses_real_template_dimensions():
    state = GuiState.from_dict({'state_version': 12, 'single': {'det_type': 'pilatus100k',
                               'pxsize_h': '50e-6', 'num_px_h': '2000'}})
    native = pyFAI.detectors.detector_factory('pilatus100k')
    assert float(state.single.pxsize_h) == native.pixel2
    assert int(state.single.num_px_h) == native.max_shape[1]


def test_sample_rotation_title():
    lattice = sample.LatticeStructure(space_group=221, a=4, b=4, c=4, alpha=90, beta=90, gamma=90)
    lattice.apply_rotation(10, 0, 0)
    from trxrdpy.simulation.detector import Detector
    det = Detector(detector_type='manual', pxsize_h=.001, pxsize_v=.001, num_pixels_h=10,
                   num_pixels_v=10, dist=.1, poni1=0, poni2=0)
    title = experiment.Experiment(det, lattice, energy=15000, e_bandwidth=1)._detector_title()
    assert 'Sample convention: orientation matrix' in title
    assert 'Equivalent sample Euler [°, fixed lab xyz' in title
    assert 'rotx=10.00' in title


def test_motor_scans_parallel_match_serial():
    from trxrdpy.simulation.geometry import MotorChain
    from trxrdpy.simulation.detector import Detector
    chain = MotorChain.from_list([{'name': 'a', 'axis': [0, 1, 0]},
                                    {'name': 'b', 'axis': [0, 0, 1]}])
    kwargs = dict(hkls=np.array([[1, 0, 0]]), wavelength=1e-10, motor_chain=chain,
                  scan_ranges={'a': (-10, 10, 10), 'b': (-10, 10, 10)})
    skwargs = dict(kwargs, initial_crystal_orientation=np.eye(3)*4, e_bandwidth=50)
    assert sample.find_Bragg_orientations_with_chain(**skwargs) == \
        sample.find_Bragg_orientations_with_chain(**skwargs, parallel=True, workers=2)
    detector = Detector(detector_type='manual', pxsize_h=.001, pxsize_v=.001, num_pixels_h=200,
                        num_pixels_v=200, dist=.1, poni1=.1, poni2=.1)
    dkwargs = dict(kwargs, q_hkls=np.array([[-.1, 1., 0.]]), detector_obj=detector)
    assert experiment.find_detector_rotations_with_chain(**dkwargs) == \
        experiment.find_detector_rotations_with_chain(**dkwargs, parallel=True, workers=2)


def test_targeting_result_numeric_roundtrip(tmp_path, monkeypatch):
    from trxrdpy.simulation.results import targeting_result
    from types import SimpleNamespace
    solutions = single_crystal.FixedEnergyAcceptedSolutions(
        *[np.zeros((2, 2)) if name == 'predicted_pixels' else np.zeros(2)
          for name in ('eta', 'phi', 'rx', 'ry', 'rz', 'predicted_pixels', 'error')])
    raw = SimpleNamespace(**{name: np.zeros(2) for name in (
        'target_hkl', 'target_pixel', 'q0', 'eta_deg', 'q_cone', 'all_pixels', 'all_ray_points',
        'on_detector_mask', 'accepted_mask', 'pixel_error_px', 'beam_center_pixel_inferred')},
        pixel_tolerance_px=5., q0_norm=1., wavelength_m=1e-10, best_idx=0, solutions=solutions)
    result = targeting_result(raw, {'function': 'target_hkl_near_pixel_fixed_energy'})
    loaded = load_result(save_result(tmp_path / 'target.h5', result))
    np.testing.assert_equal(loaded.data['solutions']['predicted_pixels'], solutions.predicted_pixels)
    assert loaded.data['solutions']['motor_angles_deg'] is None
    calls = []
    from trxrdpy.simulation import plot
    monkeypatch.setattr(plot, 'plot_fixed_energy_detector_hits', lambda *args, **kw: calls.append('detector'))
    monkeypatch.setattr(plot, 'plot_fixed_energy_motor_projections', lambda *args, **kw: calls.append('2d'))
    monkeypatch.setattr(plot, 'plot_fixed_energy_motor_family_3d', lambda *args, **kw: calls.append('3d'))
    plot_result(loaded, plot_3d=True)
    assert calls == ['detector', '2d', '3d']


def test_gui_scan_capture_and_parallel_state(monkeypatch, tmp_path, wait_simulation):
    app = QApplication.instance() or QApplication([])
    tab = SingleCrystalTab(SimulationService(), GuiState())
    tab.single_func_combo.setCurrentText('scan_two_parameters_for_Bragg_condition')
    tab.single_param2_combo.setCurrentText('rotz')
    tab.single_line_names.setText('[1,0,0]')
    tab.parallel_checkbox.setChecked(True)
    tab.workers_line.setText('2')
    points = {'[1 0 0]': [(1, 2)]}
    monkeypatch.setattr(single_crystal, 'scan_two_parameters_for_Bragg_condition', lambda **kw: points)
    from trxrdpy.simulation import plot
    monkeypatch.setattr(plot, 'plot_parameter_mapping', lambda *args: None)
    assert tab._single_run_function()
    wait_simulation(tab)
    assert tab.last_scan_result.data['points'] == points
    assert tab.save_result_button.isEnabled()
    saved = tab.state.to_dict()
    assert saved['single']['parallel_processing']
    assert saved['single']['parallel_workers'] == '2'
    from PyQt5.QtWidgets import QFileDialog
    path = tmp_path / 'saved.h5'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **kw: (str(path), ''))
    tab.save_result_button.click()
    assert path.exists()
    tab.last_scan_result = None
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **kw: (str(path), ''))
    tab.load_result_button.click()
    assert tab.last_scan_result.kind == 'parameter_mapping'
    tab.close()
    app.processEvents()


def test_saving_results_uses_save_dialogs_replace_confirmation_once(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    tab = SingleCrystalTab(SimulationService(), GuiState())
    path = tmp_path / 'mapping.h5'
    save_result(path, SimulationResult('parameter_mapping', {'points': {}}, {'version': 'old'}))
    tab.last_scan_result = SimulationResult('parameter_mapping', {'points': {}}, {'version': 'new'})

    def select_existing_file(*args, **kwargs):
        assert 'options' not in kwargs
        return str(path), 'HDF5 results (*.h5)'

    monkeypatch.setattr(QFileDialog, 'getSaveFileName', select_existing_file)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: pytest.fail('Duplicate replace prompt'))
    tab._save_scan_result()

    assert load_result(path).metadata == {'version': 'new'}
    tab.close()
    app.processEvents()


def test_predefined_pixel_edits_reach_simulation(monkeypatch, wait_simulation):
    app = QApplication.instance() or QApplication([])
    tab = SingleCrystalTab(SimulationService(), GuiState())
    tab.single_combo_det_type.setCurrentText('pilatus100k')
    tab.single_line_pxsize_h.setText('0.00025')
    received = {}
    monkeypatch.setattr(single_crystal, 'simulate_2d', lambda **kwargs: received.update(kwargs))
    assert tab._single_run_function()
    wait_simulation(tab)
    assert received['det_pxsize_h'] == .00025
    assert received['det_ntum_pixels_h'] == 487
    assert received['det_type'] == 'manual'
    tab.close()
    app.processEvents()
