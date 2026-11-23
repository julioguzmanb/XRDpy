import multiprocessing
import numpy as np
import pytest
import matplotlib.pyplot as plt

from trxrdpy.simulation import single_crystal as sc, sample, utils
from trxrdpy.simulation.results import SimulationResult, save_result, load_result, plot_result
from trxrdpy.simulation.mapping_slices import slice_result
from trxrdpy.simulation.parallel import cancellation, SimulationCancelled


def options():
    return dict(param1_name='roty', param2_name='energy', param1_range=(-20, 20, 20),
        param2_range=(10000, 15000, 5000), sam_space_group=221,
        sam_a=4, sam_b=4, sam_c=4, sam_alpha=90, sam_beta=90, sam_gamma=90,
        hkls_names=[[-1, 1, 0], [0, 1, 0]], e_bandwidth=50)


@pytest.mark.parametrize('order', ['xyz', 'zyx'])
def test_slices_parallel_roundtrip_and_replot(order, tmp_path, monkeypatch):
    opts = dict(options(), sam_rotx=12, sam_roty=4, sam_rotation_order=order)
    ranges = {'rotz': (0, 30, 30), 'e_bandwidth': (5, 150, 145)}
    output = sc.scan_two_parameters_for_Bragg_condition(**opts, extra_scan_ranges=ranges)
    assert any(points for item in output['slices'].values() for points in item['points'].values())
    parallel = sc.scan_two_parameters_for_Bragg_condition(**opts, extra_scan_ranges=ranges, parallel=True, workers=2)
    for i, (angle, bw) in enumerate([(0,5), (0,150), (30,5), (30,150)]):
        expected = sc.scan_two_parameters_for_Bragg_condition(**{**opts, 'e_bandwidth':bw}, scan_angles={'rotz':angle})
        assert output['slices'][str(i)]['points'] == expected == parallel['slices'][str(i)]['points']
    result = load_result(save_result(tmp_path/'slices.h5', SimulationResult('parameter_mapping', output, {})))
    monkeypatch.setattr(sc, 'scan_two_parameters_for_Bragg_condition', lambda **kw: pytest.fail('recomputed'))
    ax = plot_result(result, display_mode='bands')
    fig = ax.figure
    previous, following = fig._mapping_step_buttons[0]
    assert not previous.get_active()
    following._observers.process('clicked', None)
    assert fig._mapping_value_boxes[0].text == '30'
    for slider, box, (left, right) in zip(
            fig._mapping_sliders, fig._mapping_value_boxes,
            fig._mapping_step_buttons):
        assert slider.ax.get_position().width * fig.get_figwidth() >= 2.43 - 1e-9
        assert box.ax.get_position().width * fig.get_figwidth() >= .99 - 1e-9
        assert slider.ax.get_position().x1 < box.ax.get_position().x0
        assert box.ax.get_position().x1 < left.ax.get_position().x0
        assert left.ax.get_position().x1 < right.ax.get_position().x0
        assert right.ax.get_position().x1 < 1
    fig._mapping_sliders[0].set_val(1)
    fig._mapping_value_boxes[1].set_val('150')
    assert fig._mapping_sliders[1].val == 1
    fig.savefig('/private/tmp/mapping-slices-qa.png')
    plt.close(fig)


@pytest.mark.parametrize('order', ['xyz', 'zyx'])
def test_order_is_combined_not_separate_extra_rotation(order, monkeypatch):
    received = []
    original = sample.calculate_q_hkl
    def capture(*args):
        q = original(*args)
        received.append(q.copy())
        return q
    monkeypatch.setattr(sample, 'calculate_q_hkl', capture)
    opts = dict(options(), param1_range=(25,25,1), param2_range=(15000,15000,1),
                sam_rotation_order=order, sam_rotx=13)
    sc.scan_two_parameters_for_Bragg_condition(**opts, extra_scan_ranges={'rotz':(37,37,1)})
    lattice = sample.LatticeStructure(space_group=221, a=4,b=4,c=4,alpha=90,beta=90,gamma=90,rotation_order=order)
    lattice.apply_rotation(rotx=13)
    direct = utils.apply_rotation(lattice.crystal_orientation, 0, 25, 37, order)
    expected = sample.calculate_q_hkl(np.array(opts['hkls_names']), sample.cal_reciprocal_lattice(direct))
    np.testing.assert_allclose(received[0], expected, atol=1e-12)


def test_motor_chain_extra_order_and_filter_context(tmp_path):
    from trxrdpy.simulation.geometry import MotorChain
    chain = MotorChain.from_list([{'name':'a','axis':'z'}, {'name':'b','axis':'y'}, {'name':'c','axis':'x'}])
    opts = options()
    for key in list(opts):
        if key.startswith('param'):
            del opts[key]
    opts.update(motor1_name='a', motor2_name='b', motor1_range=(-20,20,20), motor2_range=(-20,20,20), motor_chain=chain)
    data = sc.scan_two_motors_for_Bragg_condition(**opts, extra_scan_ranges={'c':(0,30,30)})
    expected = sc.scan_two_motors_for_Bragg_condition(**opts, fixed_angles={'c':30})
    assert data['slices']['1']['points'] == expected
    result = load_result(save_result(tmp_path/'motor.h5', SimulationResult('parameter_mapping',data,{})))
    assert slice_result(result,1).metadata['mapping_context']['fixed_angles'] == {'c':30}


def test_validation_and_cancellation():
    for ranges in ({'roty':(0,1,1)}, {'unknown':(0,1,1)}, {'rotz':(0,1,0)}):
        with pytest.raises(ValueError):
            sc.scan_two_parameters_for_Bragg_condition(**options(), extra_scan_ranges=ranges)
    event = multiprocessing.get_context('spawn').Event()
    event.set()
    with pytest.raises(SimulationCancelled), cancellation(event):
        sc.scan_two_parameters_for_Bragg_condition(**options(), extra_scan_ranges={'rotz':(0,30,30)})


def test_gui_state_compatibility(monkeypatch):
    from PyQt5.QtWidgets import QApplication, QMessageBox
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args: None)
    from trxrdpy.simulation.gui.state import GuiState
    from trxrdpy.simulation.gui.services import SimulationService
    from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab
    app = QApplication.instance() or QApplication([])
    tab = SingleCrystalTab(SimulationService(), GuiState.from_dict({'state_version':13}))
    assert not tab.mapping_extra_enabled.isChecked()
    tab.single_func_combo.setCurrentText('scan_two_parameters_for_Bragg_condition')
    tab.mapping_extra_enabled.setChecked(True)
    tab.mapping_extra_controls.set_rows([['rotz','0','30','10']])
    tab.single_combo_sam_rotation_order.setCurrentText('zyx')
    assert not hasattr(tab, 'mapping_order')
    tab.single_param1_combo.setCurrentText('roty')
    tab.single_param2_combo.setCurrentText('energy')
    tab.single_line_names.setText('[[-1, 1, 0]]')
    request = tab._single_run_function_impl()
    assert request, tab.last_error_message
    assert request.kwargs['extra_scan_ranges'] == {'rotz':(0,30,10)}
    assert request.kwargs['sam_rotation_order'] == 'zyx'
    tab.save_to_state()
    restored = GuiState.from_dict(tab.state.to_dict())
    assert restored.single.mapping_rotation_order == 'zyx'
    assert restored.single.sam_rotation_order == 'zyx'
    assert restored.single.mapping_extra_rows == [['rotz','0','30','10']]
    assert restored.single.mapping_extra_enabled
    tab.close()
    app.processEvents()


def test_old_mapping_order_loads_into_shared_sample_control():
    from PyQt5.QtWidgets import QApplication
    from trxrdpy.simulation.gui.state import GuiState
    from trxrdpy.simulation.gui.services import SimulationService
    from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab
    app = QApplication.instance() or QApplication([])
    old = GuiState.from_dict({'state_version': 17, 'single': {
        'func': 'scan_two_parameters_for_Bragg_condition',
        'mapping_rotation_order': 'zxy'}})
    tab = SingleCrystalTab(SimulationService(), old)
    tab.load_from_state()
    assert tab.single_combo_sam_rotation_order.currentText() == 'zxy'
    tab.save_to_state()
    assert tab.state.single.sam_rotation_order == 'zxy'
    tab.close()
    app.processEvents()
