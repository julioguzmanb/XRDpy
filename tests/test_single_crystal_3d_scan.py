"""Interactive configuration sliders for the single-crystal 3D view."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from trxrdpy.simulation import single_crystal
from trxrdpy.simulation.deferred_plotting import capture_plots
from trxrdpy.simulation.geometry import DiffractometerGeometry
from trxrdpy.simulation.gui.services import SimulationService
from trxrdpy.simulation.gui.state import GuiState
from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab
from trxrdpy.simulation.single_crystal_scan import prepare_axes


OPTIONS = dict(det_ntum_pixels_h=100, det_num_pixels_v=80,
               det_pxsize_h=.001, det_pxsize_v=.001, det_dist=.1,
               det_poni1=.04, det_poni2=.05, energy=15000,
               sam_space_group=221, sam_a=4, sam_b=4, sam_c=4,
               sam_alpha=90, sam_beta=90, sam_gamma=90, qmax=3)


def _open_scan(function, **kwargs):
    with capture_plots() as calls:
        exp = function(**kwargs)
    assert len(calls) == 1
    plot_function, args, plot_kwargs = calls[0]
    fig, ax = plot_function(*args, **{**plot_kwargs, 'show': False})
    return exp, fig, ax


def test_legacy_3d_sliders_update_same_experiment_and_figure():
    exp, fig, ax = _open_scan(
        single_crystal.simulate_3d, **OPTIONS,
        scan_ranges={'sam_roty': (0, 20, 20), 'det_roty': (0, 20, 20),
                     'energy': (15000, 16000, 1000)})
    initial_q = exp.lattice.q_hkls.copy()
    initial_transform = exp.detector.get_transform().copy()
    previous, following = fig._single_crystal_step_buttons[0]
    assert not previous.get_active()
    following._observers.process('clicked', None)
    assert exp.lattice.rotation_order == 'xyz'
    assert fig._single_crystal_value_boxes[0].text == '20'
    previous._observers.process('clicked', None)
    assert fig._single_crystal_value_boxes[0].text == '0'
    fig._single_crystal_sliders[0].set_val(1)
    fig._single_crystal_sliders[1].set_val(1)
    fig._single_crystal_value_boxes[2].set_val('16000')
    assert exp.energy == 16000
    assert exp.detector.roty == 20
    assert not np.allclose(exp.lattice.q_hkls, initial_q)
    assert not np.allclose(exp.detector.get_transform(), initial_transform)
    assert ax.figure is fig
    assert '16.00' in fig._suptitle.get_text()
    assert len(fig._single_crystal_sliders) == 3
    for slider, box, (left, right) in zip(
            fig._single_crystal_sliders, fig._single_crystal_value_boxes,
            fig._single_crystal_step_buttons):
        assert slider.ax.get_position().width * fig.get_figwidth() >= 3.08 - 1e-9
        assert box.ax.get_position().width * fig.get_figwidth() >= 1.10 - 1e-9
        assert slider.ax.get_position().x1 < box.ax.get_position().x0
        assert box.ax.get_position().x1 < left.ax.get_position().x0
        assert left.ax.get_position().x1 < right.ax.get_position().x0
        assert right.ax.get_position().x1 < 1
    plt.close(fig)


def test_descending_3d_range_accepts_positive_step_magnitude():
    values = prepare_axes({'sam_roty': (-5, -25, .1)})['sam_roty']
    assert len(values) == 201
    assert values[0] == -5
    assert values[-1] == pytest.approx(-25)
    assert np.all(np.diff(values) < 0)

    _, fig, _ = _open_scan(single_crystal.simulate_3d, **OPTIONS,
                           scan_ranges={'sam_roty': (-5, -25, 10)})
    previous, following = fig._single_crystal_step_buttons[0]
    following._observers.process('clicked', None)
    assert fig._single_crystal_value_boxes[0].text == '-15'
    following._observers.process('clicked', None)
    assert fig._single_crystal_value_boxes[0].text == '-25'
    assert not following.get_active()
    previous._observers.process('clicked', None)
    assert fig._single_crystal_value_boxes[0].text == '-15'
    plt.close(fig)


def test_3d_slider_preserves_sample_euler_order():
    opts = dict(OPTIONS, sam_rotx=15, sam_roty=10, sam_rotation_order='zyx')
    exp, fig, _ = _open_scan(
        single_crystal.simulate_3d, **opts,
        scan_ranges={'sam_rotz': (0, 20, 20)})
    fig._single_crystal_sliders[0].set_val(1)
    with capture_plots():
        expected = single_crystal.simulate_3d(**{**opts, 'sam_rotz': 20})
    np.testing.assert_allclose(exp.lattice.q_hkls, expected.lattice.q_hkls)
    plt.close(fig)


@pytest.mark.parametrize('function,title_getter', [
    (single_crystal.simulate_3d, lambda fig, ax: fig._suptitle.get_text()),
    (single_crystal.simulate_2d, lambda fig, ax: ax.get_title()),
])
def test_slider_header_shows_selected_euler_values_with_custom_orientation(
        function, title_getter):
    initial = np.array([[0, 0, 3.9173], [0, -3.9173, 0], [3.9173, 0, 0]])
    exp, fig, ax = _open_scan(
        function, **OPTIONS, sam_initial_crystal_orientation=initial,
        sam_rotation_order='xzy',
        scan_ranges={'sam_rotz': (0, 45, 45), 'sam_roty': (-10, -20.1, 10.1)})
    fig._single_crystal_sliders[0].set_val(1)
    fig._single_crystal_sliders[1].set_val(1)
    title = title_getter(fig, ax)
    assert 'order xzy, relative to initial' in title
    assert 'rotx=0.00, roty=-20.10, rotz=45.00' in title
    assert 'equivalent xyz' not in title
    assert 'rotx=0.00, roty=-20.10, rotz=45.00' in exp._detector_title()
    plt.close(fig)


@pytest.mark.parametrize('function', [single_crystal.simulate_2d,
                                      single_crystal.simulate_3d])
def test_static_detector_header_keeps_entered_euler_order(function):
    with capture_plots():
        exp = function(**OPTIONS, sam_rotation_order='xzy',
                       sam_roty=-20.1, sam_rotz=45)
    title = exp._detector_title()
    assert 'Sample rotations [°, order xzy, relative to initial]' in title
    assert 'rotx=0.00, roty=-20.10, rotz=45.00' in title
    assert 'Equivalent sample Euler' not in title


def test_geometry_3d_sliders_move_named_motors():
    geometry = DiffractometerGeometry.from_dict({
        'sample': [{'name': 'omega', 'axis': 'z'}],
        'detector': [{'name': 'delta', 'axis': 'y'}]})
    exp, fig, _ = _open_scan(
        single_crystal.simulate_3d_with_geometry, **OPTIONS,
        geometry=geometry, sample_angles={'omega': 0}, detector_angles={'delta': 0},
        scan_ranges={'sample:omega': (0, 15, 15), 'detector:delta': (0, 20, 20)})
    initial_q = exp.lattice.q_hkls.copy()
    fig._single_crystal_sliders[0].set_val(1)
    fig._single_crystal_sliders[1].set_val(1)
    assert not np.allclose(exp.lattice.q_hkls, initial_q)
    title = fig._suptitle.get_text()
    assert 'Sample convention: diffractometer; omega[local]' in title
    assert 'Sample motors [°]: omega=15.00' in title
    assert 'Equivalent sample Euler [°, fixed lab xyz' in title
    assert 'Detector convention: diffractometer; delta[local]' in title
    assert 'Detector motors [°]: delta=20.00' in title
    assert 'Equivalent detector Euler [°, fixed lab xyz]' in title
    fig.canvas.draw()
    bounds = fig._suptitle.get_window_extent(fig.canvas.get_renderer())
    assert 0 <= bounds.x0 < bounds.x1 <= fig.bbox.width
    assert bounds.y1 <= fig.bbox.height
    np.testing.assert_allclose(exp.detector.get_transform(),
                               geometry.detector_transform(angles={'delta': 20}))
    plt.close(fig)


def test_3d_scan_gui_option_is_persisted_and_not_applied_to_2d():
    app = QApplication.instance() or QApplication([])
    tab = SingleCrystalTab(SimulationService(), GuiState())
    tab.single_func_combo.setCurrentText('simulate_3d')
    rows = [['sam_roty', '-5', '-25', '0.1']]
    tab.single_3d_scan_controls.set_rows(rows)
    tab.single_3d_scan_enabled.setChecked(True)
    tab.single_combo_sam_rotation_order.setCurrentText('zyx')
    request = tab._single_run_function_impl()
    assert request.function is single_crystal.simulate_3d
    assert request.kwargs['scan_ranges'] == {'sam_roty': (-5., -25., .1)}
    assert request.kwargs['sam_rotation_order'] == 'zyx'
    restored = GuiState.from_dict(tab.state.to_dict())
    assert restored.single.single_3d_scan_enabled
    assert restored.single.single_3d_scan_rows == rows
    assert restored.single.sam_rotation_order == 'zyx'
    tab.state = restored
    tab.load_from_state()
    assert tab.single_combo_sam_rotation_order.currentText() == 'zyx'
    tab.single_func_combo.setCurrentText('simulate_2d')
    assert tab.single_3d_scan_group.isHidden()
    request = tab._single_run_function_impl()
    assert request.kwargs['scan_ranges'] is None
    assert request.kwargs['sam_rotation_order'] == 'zyx'
    tab.close()
    app.processEvents()


def test_3d_scan_rejects_parameters_for_inactive_motors():
    with pytest.raises(ValueError, match='Unsupported single-crystal scan parameters'):
        single_crystal.simulate_3d(**OPTIONS, scan_ranges={'sample:omega': (0, 1, 1)})


def test_intensity_scan_replaces_colorbar_without_covering_sliders(tmp_path):
    cif = tmp_path / 'atoms.cif'
    cif.write_text('''data_atoms
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0 0 0
''')
    _, fig, ax = _open_scan(
        single_crystal.simulate_3d, **OPTIONS, consider_intensities=True,
        cif_file_path=str(cif), scan_ranges={'energy': (15000, 16000, 1000)})
    count = len(fig.axes)
    fig._single_crystal_sliders[0].set_val(1)
    assert len(fig.axes) == count
    assert ax._single_crystal_colorbar.ax.get_position().y1 < ax.get_position().y0
    assert fig._single_crystal_sliders[0].ax.get_position().y1 < ax._single_crystal_colorbar.ax.get_position().y0
    plt.close(fig)
