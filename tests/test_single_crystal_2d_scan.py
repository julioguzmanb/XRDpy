"""Interactive configuration sliders for the single-crystal 2D detector view."""
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication

from trxrdpy.simulation import single_crystal
from trxrdpy.simulation.deferred_plotting import capture_plots
from trxrdpy.simulation.geometry import DiffractometerGeometry
from trxrdpy.simulation.gui.services import SimulationService
from trxrdpy.simulation.gui.state import GuiState
from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab


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


def test_legacy_2d_sliders_update_same_detector_view():
    opts = dict(OPTIONS, sam_rotx=15, sam_rotation_order='zyx')
    exp, fig, ax = _open_scan(
        single_crystal.simulate_2d, **opts,
        scan_ranges={'sam_roty': (0, 20, 20), 'det_roty': (0, 20, 20),
                     'energy': (15000, 16000, 1000)})
    assert exp.detector.lab_grid is None
    initial_q = exp.lattice.q_hkls.copy()
    initial_transform = exp.detector.get_transform().copy()
    _, following = fig._single_crystal_step_buttons[0]
    following._observers.process('clicked', None)
    assert fig._single_crystal_value_boxes[0].text == '20'
    fig._single_crystal_sliders[0].set_val(1)
    fig._single_crystal_sliders[1].set_val(1)
    fig._single_crystal_value_boxes[2].set_val('16000')
    assert exp.energy == 16000
    assert exp.detector.roty == 20
    assert not np.allclose(exp.lattice.q_hkls, initial_q)
    assert not np.allclose(exp.detector.get_transform(), initial_transform)
    assert ax.figure is fig
    assert '16.00' in ax.get_title()
    assert len(fig._single_crystal_sliders) == 3
    with capture_plots():
        expected = single_crystal.simulate_2d(
            **{**opts, 'sam_roty': 20, 'det_roty': 20, 'energy': 16000})
    np.testing.assert_allclose(exp.lattice.q_hkls, expected.lattice.q_hkls)
    np.testing.assert_allclose(exp.pixel_positions, expected.pixel_positions)
    plt.close(fig)


def test_geometry_2d_sliders_move_named_motors():
    geometry = DiffractometerGeometry.from_dict({
        'sample': [{'name': 'omega', 'axis': 'z'}],
        'detector': [{'name': 'delta', 'axis': 'y'}]})
    exp, fig, ax = _open_scan(
        single_crystal.simulate_2d_with_geometry, **OPTIONS,
        geometry=geometry, sample_angles={'omega': 0}, detector_angles={'delta': 0},
        scan_ranges={'sample:omega': (0, 15, 15), 'detector:delta': (0, 20, 20)})
    assert exp.detector.lab_grid is None
    initial_q = exp.lattice.q_hkls.copy()
    fig._single_crystal_sliders[0].set_val(1)
    fig._single_crystal_sliders[1].set_val(1)
    assert not np.allclose(exp.lattice.q_hkls, initial_q)
    assert 'Sample motors [°]: omega=15.00' in ax.get_title()
    assert 'Detector motors [°]: delta=20.00' in ax.get_title()
    assert 'Equivalent sample Euler [°, fixed lab xyz' in ax.get_title()
    fig.canvas.draw()
    bounds = ax.title.get_window_extent(fig.canvas.get_renderer())
    assert 0 <= bounds.x0 < bounds.x1 <= fig.bbox.width
    assert bounds.y1 <= fig.bbox.height
    np.testing.assert_allclose(exp.detector.get_transform(),
                               geometry.detector_transform(angles={'delta': 20}))
    plt.close(fig)


def test_2d_scan_gui_state_and_descending_range():
    app = QApplication.instance() or QApplication([])
    tab = SingleCrystalTab(SimulationService(), GuiState())
    tab.single_func_combo.setCurrentText('simulate_2d')
    rows = [['sam_roty', '-5', '-25', '0.1']]
    tab.single_2d_scan_controls.set_rows(rows)
    tab.single_2d_scan_enabled.setChecked(True)
    request = tab._single_run_function_impl()
    assert request.function is single_crystal.simulate_2d
    assert request.kwargs['scan_ranges'] == {'sam_roty': (-5., -25., .1)}
    saved = GuiState.from_dict(tab.state.to_dict())
    assert saved.single.single_2d_scan_enabled
    assert saved.single.single_2d_scan_rows == rows
    tab.state = saved
    tab.load_from_state()
    assert tab.single_2d_scan_controls.rows() == rows
    tab.single_func_combo.setCurrentText('simulate_3d')
    assert tab.single_2d_scan_group.isHidden()
    assert tab._single_run_function_impl().kwargs['scan_ranges'] is None
    tab.close()
    app.processEvents()


def test_2d_intensity_redraw_replaces_colorbar(tmp_path):
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
        single_crystal.simulate_2d, **OPTIONS, consider_intensities=True,
        cif_file_path=str(cif), scan_ranges={'energy': (15000, 16000, 1000)})
    count = len(fig.axes)
    fig._single_crystal_sliders[0].set_val(1)
    assert len(fig.axes) == count
    assert ax._single_crystal_colorbar.ax.get_position().y1 < ax.get_position().y0
    assert (fig._single_crystal_sliders[0].ax.get_position().y1 <
            ax._single_crystal_colorbar.ax.get_position().y0)
    plt.close(fig)


def test_2d_redraw_replaces_reflection_callbacks():
    opts = dict(OPTIONS, det_num_pixels_v=100, det_pxsize_h=.01,
                det_pxsize_v=.01, det_poni1=.5, det_poni2=.5,
                e_bandwidth=50, sam_roty=20, qmax=4)
    _, fig, ax = _open_scan(single_crystal.simulate_2d, **opts,
                            scan_ranges={'energy': (15000, 16000, 1000)})
    assert hasattr(ax, '_single_crystal_hover_cid')
    assert hasattr(ax, '_single_crystal_pick_cid')
    old_hover = ax._single_crystal_hover_cid
    old_pick = ax._single_crystal_pick_cid
    old_tooltip = ax._single_crystal_hover_text
    fig._single_crystal_sliders[0].set_val(1)
    assert old_hover not in fig.canvas.callbacks.callbacks.get('motion_notify_event', {})
    assert old_pick not in fig.canvas.callbacks.callbacks.get('pick_event', {})
    assert old_tooltip not in fig.texts
    plt.close(fig)
