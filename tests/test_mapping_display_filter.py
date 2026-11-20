from copy import deepcopy

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from trxrdpy.simulation import single_crystal, plot
from trxrdpy.simulation import sample, utils
from trxrdpy.simulation.mapping_filter import mapping_context, filtered_mapping, mapping_rays
from trxrdpy.simulation.results import SimulationResult, save_result, load_result, plot_result
from trxrdpy.simulation.gui.state import GuiState
from trxrdpy.simulation.gui.services import SimulationService
from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab
from trxrdpy.simulation.geometry import MotorChain
from test_simulation_detector_filter import parameters, detector


def result_for(values):
    prefix = 'motor' if 'motor1_name' in values else 'param'
    function = (single_crystal.scan_two_motors_for_Bragg_condition if prefix == 'motor'
                else single_crystal.scan_two_parameters_for_Bragg_condition)
    return SimulationResult('parameter_mapping', {'points': function(**values)},
        {'axes': [values[prefix + str(i) + '_name'] for i in (1, 2)],
         'mapping_context': mapping_context(values), 'points_are_complete': True,
         'require_detector_hit': False})


def assert_points_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(np.asarray(actual[key]).reshape(-1, 2),
                                   np.asarray(expected[key]).reshape(-1, 2))


@pytest.mark.parametrize('chain', [False, True])
def test_replot_matches_filter_without_rescanning_or_mutation(chain, tmp_path, monkeypatch):
    values = parameters()
    function = single_crystal.scan_two_parameters_for_Bragg_condition
    if chain:
        values = {key: value for key, value in values.items() if not key.startswith('param')}
        values.update(motor1_name='omega', motor2_name='chi', motor1_range=(82, 86, .5),
                      motor2_range=(-5, 5, 5), energy=15000,
                      motor_chain=MotorChain.from_list([dict(name='omega', axis='z'), dict(name='chi', axis='y')]))
        function = single_crystal.scan_two_motors_for_Bragg_condition
    source = result_for(values)
    original = deepcopy(source.data)
    expected = function(**values, require_detector_hit=True, detector_obj=detector())
    source = load_result(save_result(tmp_path / 'complete.h5', source))
    def fail(*args, **kwargs):
        raise AssertionError('A saved map must not run another scan.')
    monkeypatch.setattr(single_crystal, 'scan_two_parameters_for_Bragg_condition', fail)
    monkeypatch.setattr(single_crystal, 'scan_two_motors_for_Bragg_condition', fail)
    view = filtered_mapping(source, detector())
    assert_points_equal(view.data['points'], expected)
    with pytest.raises(ValueError, match='original complete mapping'):
        save_result(tmp_path / 'view.h5', view)
    cached = mapping_rays(source)
    assert not any(len(p) for p in filtered_mapping(source, detector(poni2=10)).data['points'].values())
    assert mapping_rays(source) is cached
    assert_points_equal(source.data['points'], original['points'])
    resaved = load_result(save_result(tmp_path / 'still-complete.h5', source))
    assert_points_equal(resaved.data['points'], original['points'])


def test_old_unfiltered_gui_snapshot_uses_original_sample(tmp_path):
    values = parameters()
    source = result_for(values)
    source.metadata.pop('mapping_context')
    single = GuiState().to_dict()['single']
    for key in ('a', 'b', 'c', 'alpha', 'beta', 'gamma', 'space_group'):
        single[key] = str(values['sam_' + key])
    single['energy'] = '10000'
    single['param1'], single['param2'] = 'energy', 'rotz'
    source.metadata['gui_state'] = {'single': single}
    loaded = load_result(save_result(tmp_path / 'legacy.h5', source))
    expected = single_crystal.scan_two_parameters_for_Bragg_condition(
        **values, require_detector_hit=True, detector_obj=detector())
    assert_points_equal(filtered_mapping(loaded, detector()).data['points'], expected)


def test_missing_context_and_old_filtered_warning(monkeypatch):
    source = SimulationResult('parameter_mapping', {'points': {'[-1 0 0]': [[12000, 82.5]]}},
                              {'require_detector_hit': True})
    monkeypatch.setattr(plot, 'plot_parameter_mapping', lambda *a, **kw: None)
    with pytest.warns(UserWarning, match='cannot be recovered'):
        plot_result(source)
    with pytest.raises(ValueError, match='lacks the sample'):
        filtered_mapping(source, detector())


def test_gui_replots_loaded_points_for_current_detector(tmp_path, monkeypatch, wait_simulation):
    app = QApplication.instance() or QApplication([])
    tab = SingleCrystalTab(SimulationService(), GuiState())
    original = result_for(parameters())
    tab.last_scan_result = load_result(save_result(tmp_path / 'map.h5', original))
    captured = []
    monkeypatch.setattr(plot, 'plot_parameter_mapping', lambda points, *a, **kw: captured.append(points))
    for name, value in dict(pxsize_h=.001, pxsize_v=.001, num_px_h=60, num_px_v=60,
                            poni1=.03, poni2=.03, dist=.1, rotx=0, roty=0, rotz=0).items():
        getattr(tab, 'single_line_' + name).setText(str(value))
    tab.require_detector_hit.setChecked(True)
    tab.single_line_sam_a.setText('100')  # Display filtering must use the saved sample, not this edit.
    tab._plot_scan_result()
    wait_simulation(tab)
    assert_points_equal(captured[-1], filtered_mapping(original, detector()).data['points'])
    tab.single_line_poni2.setText('10')
    tab._plot_scan_result()
    wait_simulation(tab)
    assert not any(len(p) for p in captured[-1].values())
    tab.require_detector_hit.setChecked(False)
    tab._plot_scan_result()
    wait_simulation(tab)
    assert_points_equal(captured[-1], original.data['points'])
    assert_points_equal(tab.last_scan_result.data['points'], original.data['points'])
    tab.close()
    app.processEvents()


def test_ray_reconstruction_keeps_custom_orientation_and_rotation_order():
    values = parameters()
    values.update(sam_initial_crystal_orientation=[[4, 0, 0], [.5, 5, 0], [.2, .3, 6]],
                  sam_rotx=12, sam_roty=9, sam_rotz=-4, sam_rotation_order='zyx')
    context = mapping_context(values)
    rows = np.array([[9000, 10], [12000, 20], [15000, -5]])
    result = SimulationResult('parameter_mapping', {'points': {'[-1 2 0]': rows}},
        {'axes': ['energy', 'roty'], 'mapping_context': context})
    actual = mapping_rays(result)['[-1 2 0]']
    expected = []
    for energy, angle in rows:
        orientation = utils.apply_rotation(np.array(context['initial_orientation']), 0, angle, 0, 'zyx')
        q = sample.calculate_q_hkl([[-1, 2, 0]], sample.cal_reciprocal_lattice(orientation))[0]
        q[0] += 2 * np.pi / (utils.energy_to_wavelength(energy) * 1e10)
        expected.append(q)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_gui_run_with_filter_enabled_still_saves_every_bragg_point(tmp_path, monkeypatch, wait_simulation):
    app = QApplication.instance() or QApplication([])
    values = parameters()
    original = result_for(values)
    captured = []
    def calculate(**kwargs):
        assert kwargs['require_detector_hit'] is False
        return deepcopy(original.data['points'])
    monkeypatch.setattr(single_crystal, 'scan_two_parameters_for_Bragg_condition', calculate)
    monkeypatch.setattr(plot, 'plot_parameter_mapping', lambda points, *a, **kw: captured.append(points))
    tab = SingleCrystalTab(SimulationService(), GuiState())
    tab.single_func_combo.setCurrentText('scan_two_parameters_for_Bragg_condition')
    tab.single_param1_combo.setCurrentText('energy')
    tab.single_param2_combo.setCurrentText('rotz')
    tab.single_line_names.setText('[-1,0,0]')
    for name in ('a', 'b', 'c', 'alpha', 'beta', 'gamma'):
        getattr(tab, 'single_line_sam_' + name).setText(str(values['sam_' + name]))
    tab.single_line_space_group.setText('221')
    for name, value in dict(pxsize_h=.001, pxsize_v=.001, num_px_h=60, num_px_v=60,
                            poni1=.03, poni2=.03, dist=.1, rotx=0, roty=0, rotz=0).items():
        getattr(tab, 'single_line_' + name).setText(str(value))
    tab.require_detector_hit.setChecked(True)
    assert tab._single_run_function()
    wait_simulation(tab)
    assert captured
    assert_points_equal(captured[-1], filtered_mapping(original, detector()).data['points'])
    assert_points_equal(tab.last_scan_result.data['points'], original.data['points'])
    saved = load_result(save_result(tmp_path / 'all-points.h5', tab.last_scan_result))
    assert_points_equal(saved.data['points'], original.data['points'])
    assert saved.metadata['points_are_complete'] is True
    tab.close()
    app.processEvents()
