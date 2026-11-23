import multiprocessing
import numpy as np
import pytest
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication

from trxrdpy.simulation import single_crystal, reciprocal
from trxrdpy.simulation.reciprocal_scan import build_scan, configuration, plot_scan, scan_axes
from trxrdpy.simulation.geometry import DiffractometerGeometry
from trxrdpy.simulation.detector import Detector
from trxrdpy.simulation.results import SimulationResult, save_result, load_result, plot_result
from trxrdpy.simulation.parallel import cancellation, SimulationCancelled
from trxrdpy.simulation.gui.state import GuiState
from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab
from trxrdpy.simulation.gui.services import SimulationService


@pytest.fixture(autouse=True)
def close_local_viewers():
    yield
    from trxrdpy.simulation.reciprocal_live import close_viewers
    close_viewers()
    from trxrdpy.simulation.reciprocal_cache import close_caches
    close_caches()


def options():
    return dict(sam_space_group=221, sam_a=4, sam_b=4, sam_c=4,
                sam_alpha=90, sam_beta=90, sam_gamma=90, qmax=3,
                energy=15000, e_bandwidth=5, sam_rotx=12, sam_roty=0, sam_rotz=8)


def test_eight_axes_bandwidth_and_roundtrip(tmp_path):
    ranges = {'energy': [10000, 15000, 5000], **{
        prefix+axis: [0, 20, 20] for prefix in ('sam_rot', 'det_rot') for axis in 'xyz'},
        'e_bandwidth': [0, 10, 10]}
    det = Detector(detector_type='manual', pxsize_h=.001, pxsize_v=.001,
                   num_pixels_h=200, num_pixels_v=200, dist=.1, poni1=.1, poni2=.1)
    scan = build_scan(ranges, lattice_options=options(), detector_obj=det)
    saved = load_result(save_result(tmp_path/'eight.h5', SimulationResult('reciprocal', {'scan':scan}, {}))).data['scan']
    assert len(saved['axes']) == 8
    narrow, wide = configuration(saved, 0), configuration(saved, 1)
    assert narrow['bandwidth'] == 0 and wide['bandwidth'] == 10
    from trxrdpy.simulation import sample
    assert np.all(sample.check_Bragg_condition(narrow['q'], narrow['wavelength'], 0) <=
                  sample.check_Bragg_condition(wide['q'], wide['wavelength'], 10))
    figures = [reciprocal.plot_reciprocal(**v, show=False) for v in (narrow, wide)]
    assert len(figures[0].data) == len(figures[1].data)
    assert not len(figures[0].data[3].x) and len(figures[1].data[3].x)
    fig, ax = plot_scan(saved, backend='matplotlib', show=False, parallel=True, workers=3)
    assert len(fig._reciprocal_sliders) == len(fig._reciprocal_value_boxes) == 8
    assert len(fig._reciprocal_step_buttons) == 8
    for slider, box, (previous, following) in zip(
            fig._reciprocal_sliders, fig._reciprocal_value_boxes,
            fig._reciprocal_step_buttons):
        assert slider.ax.get_position().y1 < ax.get_position().y0
        assert slider.ax.get_position().x1 < box.ax.get_position().x0
        assert 0 < box.ax.get_position().x0 < box.ax.get_position().x1 < 1
        assert box.ax.get_position().x1 < previous.ax.get_position().x0
        assert previous.ax.get_position().x1 < following.ax.get_position().x0
        assert following.ax.get_position().x1 < 1
        assert slider.ax.get_position().width * fig.get_figwidth() >= 3.25 - 1e-9
        assert box.ax.get_position().width * fig.get_figwidth() >= 1.16 - 1e-9
    fig._reciprocal_step_buttons[0][1]._observers.process('clicked', None)
    assert fig._reciprocal_value_boxes[0].text == '15000'
    assert 'energy = 15000' in ax.get_title()
    fig._reciprocal_value_boxes[-1].set_val('10')
    assert 'full bandwidth: 10%' in ax.get_title()
    fig.savefig('/private/tmp/reciprocal-eight-sliders-qa.png')
    plt.close(fig)
    with pytest.raises(ValueError, match='eight'):
        scan_axes({**ranges, 'ninth':[0,1,1]})
    for invalid in ([-1,1,1], [0,200,1]):
        with pytest.raises(ValueError, match='bandwidth'):
            scan_axes({'e_bandwidth':invalid})


def test_exact_large_configuration_index():
    ranges = {prefix+axis:[0,999,1] for prefix in ('sam_rot', 'det_rot') for axis in 'xyz'}
    ranges.update(energy=[10000,10999,1], e_bandwidth=[0,.999,.001])
    det = Detector(detector_type='manual', pxsize_h=.001, pxsize_v=.001,
                   num_pixels_h=200, num_pixels_v=200, dist=.1, poni1=.1, poni2=.1)
    scan = build_scan(ranges, lattice_options=options(), detector_obj=det)
    last = configuration(scan, 1000**8-1)
    assert last['bandwidth'] == pytest.approx(.999)
    assert 'energy = 10999' in last['title']
    previous = configuration(scan, 1000**8-2)
    assert previous['bandwidth'] == pytest.approx(.998)


def test_gui_eight_axes_and_bandwidth_state():
    app = QApplication.instance() or QApplication([])
    tab = SingleCrystalTab(SimulationService(), GuiState.from_dict({'state_version':13}))
    tab.single_func_combo.setCurrentText('simulate_reciprocal')
    controls = tab.reciprocal_scan_controls
    assert 'e_bandwidth' in controls.parameters
    rows = [[name, '10000' if name=='energy' else '0', '10001' if name=='energy' else '1', '1']
            for name in controls.parameters]
    controls.set_rows(rows)
    tab.reciprocal_scan_enabled.setChecked(True)
    tab._reciprocal_request()
    restored = GuiState.from_dict(tab.state.to_dict())
    assert restored.single.reciprocal_scan_rows == rows
    tab.state = restored
    tab.load_from_state()
    assert controls.rows() == rows
    assert len(controls.ranges()) == 8
    tab.close()
    app.processEvents()


@pytest.mark.parametrize('order', ['xyz', 'zyx'])
def test_scan_matches_individual_configurations(order, monkeypatch):
    from trxrdpy.simulation import plot
    monkeypatch.setattr(plot, 'plot_reciprocal', lambda **kw: None)
    opts = dict(options(), sam_rotation_order=order)
    scan = build_scan({'energy': [10000, 15000, 5000], 'sam_roty': [0, 20, 20]}, lattice_options=opts)
    for index, (energy, angle) in enumerate([(10000, 0), (10000, 20), (15000, 0), (15000, 20)]):
        expected = single_crystal.simulate_reciprocal(**{**opts, 'energy': energy, 'sam_roty': angle}, show=False)
        actual = configuration(scan, index)
        np.testing.assert_allclose(actual['q'], expected['Q_hkls'], atol=1e-12)
        assert actual['wavelength'] == expected['wavelength']
        assert f'order {order}, relative to initial' in actual['title']
        assert 'Equivalent sample Euler' not in actual['title']


def test_motor_scan_and_detector_poses(monkeypatch):
    from trxrdpy.simulation import plot
    monkeypatch.setattr(plot, 'plot_reciprocal', lambda **kw: None)
    geometry = DiffractometerGeometry.from_dict({
        'name': 'test geometry',
        'sample': [{'name': 'omega', 'axis': 'z'}],
        'detector': [{'name': 'delta', 'axis': 'y'}]})
    det = Detector(detector_type='manual', pxsize_h=.001, pxsize_v=.001,
                   num_pixels_h=200, num_pixels_v=200, dist=.1, poni1=.1, poni2=.1)
    opts = dict(options(), geometry=geometry, sample_angles={'omega': 5})
    scan = build_scan({'sample:omega': [5, 15, 10], 'detector:delta': [0, 30, 30]},
                      lattice_options=opts, detector_obj=det)
    expected = single_crystal.simulate_reciprocal(**{**opts, 'sample_angles': {'omega': 15}}, show=False)
    np.testing.assert_allclose(configuration(scan, 3)['q'], expected['Q_hkls'], atol=1e-12)
    np.testing.assert_allclose(configuration(scan, 3)['detector_obj'].get_transform(),
                               geometry.detector_transform(angles={'delta': 30}))
    title = configuration(scan, 3)['title']
    assert 'Sample convention: test geometry; omega[local]' in title
    assert 'Sample motors [°]: omega=15.00' in title
    assert 'Equivalent sample Euler [°, fixed lab xyz' in title
    assert 'Detector convention: test geometry; delta[local]' in title
    assert 'Detector motors [°]: delta=30.00' in title
    assert 'Equivalent detector Euler [°, fixed lab xyz]' in title
    old = scan['Q0'].copy()
    det.set_rotation_angles(rotz=20)
    updated = configuration(scan, 3, det)
    np.testing.assert_allclose(updated['detector_obj'].get_transform(), det.get_transform() @ geometry.detector_transform(angles={'delta':30}))
    np.testing.assert_equal(scan['Q0'], old)


def test_saved_scan_both_renderers_and_sliders(tmp_path):
    scan = build_scan({'energy': [10000, 15000, 5000], 'sam_roty': [0, 20, 20]}, lattice_options=options())
    result = load_result(save_result(tmp_path/'scan.h5', SimulationResult('reciprocal', {'scan': scan}, {})))
    np.testing.assert_equal(result.data['scan']['Q0'], scan['Q0'])
    fig = plot_result(result, reciprocal_options={'show': False, 'html_path': tmp_path/'scan.html'})
    assert len(fig.frames) == 0
    assert fig._reciprocal_viewer.url.startswith('http://127.0.0.1:')
    html = (tmp_path/'scan.html').read_text()
    assert 'local XRDpy viewer' in html
    figure, ax = plot_result(result, reciprocal_options={'backend': 'matplotlib', 'show': False})
    sliders = figure._reciprocal_sliders
    sliders[0].set_val(1)
    sliders[1].set_val(1)
    assert 'energy = 15000' in ax.get_title() and 'sam_roty = 20' in ax.get_title()
    assert sliders[0].val == 1
    figure._reciprocal_value_boxes[0].set_val('10000')
    assert sliders[0].val == 0 and sliders[1].val == 1
    figure._reciprocal_value_boxes[0].set_val('999999')
    assert sliders[0].val == 0
    figure.canvas.draw()
    plt.close(figure)


@pytest.mark.parametrize('ranges', [{}, {'energy': [1, 2, 0]}, {'energy': [0, 2, 1]},
                                   {'energy': [1, 2_000_000, 1]}, {'sam_roty': [2, 1, 1]}])
def test_invalid_ranges(ranges):
    with pytest.raises(ValueError):
        scan_axes(ranges)


def test_scan_stop_and_no_overshoot():
    assert list(scan_axes({'sam_roty': [0, 1, .3]})['sam_roty']) == pytest.approx([0, .3, .6, .9])
    event = multiprocessing.get_context('spawn').Event()
    event.set()
    with pytest.raises(SimulationCancelled), cancellation(event):
        build_scan({'energy': [10000, 15000, 5000]}, lattice_options=options())


def test_gui_opt_in_and_state_roundtrip():
    app = QApplication.instance() or QApplication([])
    state = GuiState.from_dict({'state_version': 13})
    tab = SingleCrystalTab(SimulationService(), state)
    tab.single_func_combo.setCurrentText('simulate_reciprocal')
    assert not tab.reciprocal_scan_enabled.isChecked()
    controls = tab.reciprocal_scan_controls
    controls.add_row()
    assert controls.rows()[-1][1:] == ['', '', '']
    assert controls.table.cellWidget(0, 0) is None
    assert controls.table.verticalHeader().isHidden()
    tab.reciprocal_scan_controls.set_rows([['energy', '10000', '15000', '5000'], ['sam_roty', '0', '20', '20']])
    tab.reciprocal_scan_enabled.setChecked(True)
    tab.parallel_checkbox.setChecked(True)
    tab.workers_line.setText('3')
    assert not tab.parallel_checkbox.isHidden()
    request = tab._reciprocal_request()
    assert len(request.kwargs['scan_ranges']) == 2
    restored = GuiState.from_dict(tab.state.to_dict())
    assert restored.single.reciprocal_scan_enabled
    assert len(restored.single.reciprocal_scan_rows) == 2
    assert restored.single.parallel_processing and restored.single.parallel_workers == '3'
    tab.close()
    app.processEvents()


def test_gui_scan_background_and_replot(monkeypatch, wait_simulation):
    from pathlib import Path
    from urllib.parse import unquote, urlparse
    app = QApplication.instance() or QApplication([])
    opened = []
    monkeypatch.setattr(reciprocal.webbrowser, 'open', lambda url, **kw: opened.append(url) or True)
    tab = SingleCrystalTab(SimulationService(), GuiState())
    tab.single_func_combo.setCurrentText('simulate_reciprocal')
    tab.reciprocal_scan_controls.set_rows([['energy', '10000', '15000', '5000']])
    tab.single_line_qmax.setText('2')
    tab.reciprocal_scan_enabled.setChecked(True)
    assert tab._single_run_function()
    assert tab._simulation_dialog.auto_close_on_success
    wait_simulation(tab)
    assert len(tab.last_scan_result.data['scan']['axes']['energy']) == 2
    assert 'Q' not in tab.last_scan_result.data['scan']
    tab._plot_scan_result()
    wait_simulation(tab)
    assert len(opened) == 2
    assert all(url.startswith('http://127.0.0.1:') for url in opened)
    tab.close()
    app.processEvents()


def test_live_slider_controller_preserves_other_axes():
    import os
    import shutil
    import subprocess
    node = os.environ.get('NODE_BINARY') or shutil.which('node')
    if not node:
        pytest.skip('Node is needed for the JavaScript controller unit test.')
    from trxrdpy.simulation.reciprocal_live import SLIDER_SCRIPT
    controller = SLIDER_SCRIPT.replace('AXES', '[{"name":"energy","size":2,"start":10000,"step":5000},{"name":"sam_roty","size":2,"start":0,"step":20}]')
    fixture = """
const frames = [], inputs=[], buttons=[];
const plot = {layout:{},data:[],before:()=>{}};
globalThis.document = {getElementById: () => plot, createElement: name => {
 const element={style:{},append:()=>{},setAttribute:()=>{},handlers:{},addEventListener:(name,handler)=>{element.handlers[name]=handler;},dispatchEvent:event=>element.handlers[event.type]()};
 if(name==='input') inputs.push(element);
 if(name==='button') buttons.push(element);
 return element;
}};
globalThis.fetch=async url=>{frames.push(Number(url.split('=')[1]));return {ok:true,json:async()=>({data:[],title:'test'})};};
globalThis.Plotly = {react:async()=>{}};
"""
    checks = """
if(!buttons[0].disabled || buttons[1].disabled) throw new Error('Initial arrow state is wrong');
for (const axis of [0, 1, 0]) {
  inputs[axis*2].value=axis===0 && frames.length>0 ? 0 : 1; inputs[axis*2].handlers.input();
  await new Promise(resolve => setImmediate(resolve));
}
if (JSON.stringify(frames)!=='[2,3,1]') throw new Error(JSON.stringify(frames));
inputs[1].value='15000';inputs[1].handlers.change();
await new Promise(resolve=>setImmediate(resolve));
if(frames.at(-1)!==3) throw new Error('Typed input did not update selected state');
buttons[0].handlers.click();
await new Promise(resolve=>setImmediate(resolve));
if(frames.at(-1)!==1 || inputs[1].value!==10000) throw new Error('Previous did not move one sample');
buttons[1].handlers.click();
await new Promise(resolve=>setImmediate(resolve));
if(frames.at(-1)!==3 || !buttons[1].disabled) throw new Error('Next did not reach the endpoint');
"""
    result = subprocess.run([node, '--input-type=module', '-e', fixture+controller+checks], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    import json
    axes = [dict(name=str(i), size=1000, start=0, step=1) for i in range(8)]
    controller = SLIDER_SCRIPT.replace('AXES', json.dumps(axes))
    fixture = fixture.replace("Number(url.split('=')[1])", "url.split('=')[1]")
    checks = """
for(let i=0;i<8;i++) {
 inputs[2*i].value=999;inputs[2*i].handlers.input();
 await new Promise(resolve=>setImmediate(resolve));
}
if(frames.at(-1)!=='999999999999999999999999') throw new Error('Lost index precision: '+frames.at(-1));
inputs[14].value=998;inputs[14].handlers.input();
await new Promise(resolve=>setImmediate(resolve));
if(frames.at(-1)!=='999999999999999999999998') throw new Error('Neighbor index lost precision');
"""
    result = subprocess.run([node, '--input-type=module', '-e', fixture+controller+checks], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_fine_scan_compact_storage_and_parallel_equivalence(tmp_path):
    from trxrdpy.simulation.reciprocal_compact import transform_points
    scan = build_scan({'energy':[7000,15000,1], 'sam_roty':[-45,45,1]}, lattice_options=options())
    assert len(scan['axes']['energy'])*len(scan['axes']['sam_roty']) == 728091
    assert 'Q' not in scan and scan['Q0'].ndim == 2
    last = configuration(scan, 728090)
    assert 'energy = 15000' in last['title'] and 'sam_roty = 45' in last['title']
    path = save_result(tmp_path/'fine.h5', SimulationResult('reciprocal', {'scan':scan}, {}))
    assert path.stat().st_size < 1_000_000
    np.testing.assert_equal(configuration(load_result(path).data['scan'],728090)['q'], last['q'])
    points=np.tile(scan['Q0'],(1000,1))
    from trxrdpy.simulation import utils
    rotation=utils.apply_rotation(np.eye(3),10,20,30).T
    np.testing.assert_allclose(transform_points(points,rotation,True,4), transform_points(points,rotation),atol=1e-12)


def test_live_endpoint_colors_bounds_and_bounded_cache(tmp_path):
    import json
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
    scan = build_scan({'energy':[7000,15000,1], 'sam_roty':[-45,45,1]}, lattice_options=options())
    fig = plot_scan(scan,show=False,html_path=tmp_path/'view.html')
    viewer=fig._reciprocal_viewer
    with urlopen(viewer.url,timeout=5) as response:
        assert b'input.type=\'range\'' in response.read()
    for index in [0,1,2,3,728090]:
        with urlopen(viewer.url+'frame?index='+str(index),timeout=5) as response:
            data=json.load(response)
        assert data['data']
    with viewer.cache.lock:
        assert len(viewer.cache.values) <= viewer.cache.max_entries
        assert viewer.cache.bytes <= viewer.cache.max_bytes
    with pytest.raises(HTTPError) as error:
        urlopen(viewer.url+'frame?index=728091',timeout=5)
    assert error.value.code==400
    with pytest.raises(HTTPError) as error:
        urlopen(Request(viewer.url,headers={'Origin':'https://example.com'}),timeout=5)
    assert error.value.code==403


def test_dense_saved_scan_still_plots():
    values=single_crystal.simulate_reciprocal(**options(),show=False)
    scan=dict(axes={'energy':np.array([15000.])},Q=np.array([values['Q_hkls']]),hkls=values['hkls'],
              wavelengths=np.array([values['wavelength']]),bandwidth=5,
              detector_deltas=np.array([np.eye(4)]),detector_state=None)
    fig=plot_scan(scan,show=False)
    assert len(fig.frames)==0
    figure,ax=plot_scan(scan,backend='matplotlib',show=False)
    plt.close(figure)


def test_task_progress_and_success_auto_close():
    from trxrdpy.analysis.gui.widgets.task_output_dialog import TaskOutputDialog
    from PyQt5.QtTest import QTest
    app = QApplication.instance() or QApplication([])
    dialog = TaskOutputDialog(auto_close_delay_ms=0)
    dialog.show()
    assert dialog.progress_bar.maximum()==0
    dialog.append_output('\rSimulation rows: 25%| | 2/8 [00:01]')
    assert dialog.progress_bar.value()==25
    dialog.append_output('Preparing reciprocal lattice...')
    assert dialog.progress_bar.maximum()==0
    dialog.mark_success()
    QTest.qWait(10)
    assert not dialog.isVisible()
    assert dialog.progress_bar.value()==100
    dialog.deleteLater()
    app.processEvents()


def test_configuration_metadata_and_invalid_direct_index():
    scan=build_scan({'energy':[7000,15000,1000]},lattice_options=options())
    assert 'full bandwidth: 5%' in configuration(scan,0)['title']
    assert 'wavelength:' in configuration(scan,0)['title']
    with pytest.raises(ValueError):
        configuration(scan,-1)
