"""Measured-Bragg orientation fitting and pose inheritance."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from trxrdpy.simulation import utils
from trxrdpy.simulation.diffractometers import (
    make_euler_geometry, make_kappa_geometry, motor_chain_to_dict,
)
from trxrdpy.simulation.orientation_fit.forward_model import predict_bragg_pixel
from trxrdpy.simulation.orientation_fit.observations import (
    BraggObservation, DetectorSetup, ExperimentalPose, OrientationExperiment,
    resolve_pose,
)
from trxrdpy.simulation.orientation_fit.persistence import (
    load_experiment, load_fit_result, save_experiment, save_fit_result,
)
from trxrdpy.simulation.orientation_fit.refine import fit_orientation


def _synthetic_experiment():
    energy = 15000.0
    q = 2 * np.pi / 4
    k = 2 * np.pi / (utils.energy_to_wavelength(energy) * 1e10)
    x = -q * q / (2 * k)
    y = np.sqrt(q * q - x * x)
    second_y = -x * x / y
    second_z = np.sqrt(q * q - x * x - second_y * second_y)
    first = np.array([x, y, 0.0]) / q
    second = np.array([x, second_y, second_z]) / q
    true_rotation = np.column_stack((first, second, np.cross(first, second)))
    sample_chain = make_euler_geometry().sample
    detector_chain = make_euler_geometry().detector
    detector = DetectorSetup(
        'fixed', dict(detector_type='manual', pxsize_h=1e-4, pxsize_v=1e-4,
                      num_pixels_h=1000, num_pixels_v=1000, dist=0.1,
                      poni1=0.03, poni2=0.04, rotx=0, roty=0, rotz=0,
                      rotation_order='xyz', binning=(1, 1)),
        motor_chain=motor_chain_to_dict(detector_chain),
    )
    project = OrientationExperiment(
        lattice=dict(space_group=221, a=4, b=4, c=4,
                     alpha=90, beta=90, gamma=90),
        sample_chain=motor_chain_to_dict(sample_chain),
        detectors=[detector],
        poses=[ExperimentalPose('mount', motor_values={'sam_rotx': 4},
                                energy_eV=energy, detector_setup_id='fixed'),
               ExperimentalPose('after-move', parent_id='mount', motor_mode='relative',
                                motor_values={'sam_roty': -5}, energy_eV=energy + 100,
                                detector_angles={'det_roty': 3}),
               ExperimentalPose('energy-only', parent_id='after-move',
                                motor_mode='relative', energy_eV=energy + 200)],
        peaks=[], default_energy_eV=energy, reference_pose_id='mount',
    )
    mount = resolve_pose(project, 'mount')
    for name, hkl in [('peak-1', (1, 0, 0)), ('peak-2', (0, 1, 0))]:
        predicted = predict_bragg_pixel(project, mount, true_rotation, hkl)
        assert predicted.on_detector
        assert abs(predicted.ewald_error) < 1e-10
        project.peaks.append(BraggObservation(name, 'mount', predicted.pixel_h,
                                              predicted.pixel_v, hkl))
    return project, true_rotation


def test_relative_poses_inherit_conditions_and_can_change_only_energy():
    project, _ = _synthetic_experiment()
    moved = resolve_pose(project, 'after-move')
    assert moved.motor_angles['sam_rotx'] == 4
    assert moved.motor_angles['sam_roty'] == -5
    assert moved.energy_eV == 15100
    later = resolve_pose(project, 'energy-only')
    assert later.motor_angles == moved.motor_angles
    assert later.energy_eV == 15200
    assert later.detector_setup_id == 'fixed'
    assert later.detector_angles['det_roty'] == 3


def test_absolute_motor_reading_and_branch_from_mount():
    project, _ = _synthetic_experiment()
    project.poses.append(ExperimentalPose('branch', parent_id='mount',
                                          motor_mode='absolute',
                                          motor_values={'sam_rotx': 10}))
    branch = resolve_pose(project, 'branch')
    assert branch.motor_angles['sam_rotx'] == 10
    assert branch.motor_angles['sam_roty'] == 0
    assert branch.energy_eV == 15000
    assert branch.detector_angles.get('det_roty', 0) == 0


def test_two_exact_peaks_recover_orientation_and_pose_matrices():
    project, true_rotation = _synthetic_experiment()
    result = fit_orientation(project, max_candidates=4)
    best = result.candidates[0]
    assert best.rank == 3
    assert best.rms_pixel < 1e-5
    assert Rotation.from_matrix(best.rotation @ true_rotation.T).magnitude() < 1e-6
    assert set(best.pose_matrices) == {'mount', 'after-move', 'energy-only'}


def test_peaks_at_two_motor_poses_recover_reference_orientation():
    project, true_rotation = _synthetic_experiment()
    project.poses[1].motor_values = {'sam_rotx': 4}
    project.poses[1].energy_eV = 15000
    moved = resolve_pose(project, 'after-move')
    expected = predict_bragg_pixel(project, moved, true_rotation, (0, 1, 0))
    assert expected.on_detector
    assert abs(expected.ewald_error) < 1e-10
    project.peaks[1].pose_id = 'after-move'
    project.peaks[1].pixel_h = expected.pixel_h
    project.peaks[1].pixel_v = expected.pixel_v
    best = fit_orientation(project, max_candidates=3).candidates[0]
    assert best.rank == 3
    assert best.rms_pixel < 1e-5
    np.testing.assert_allclose(best.rotation, true_rotation, atol=1e-6)


def test_one_peak_reports_unresolved_rotation():
    project, _ = _synthetic_experiment()
    project.peaks.pop()
    result = fit_orientation(project, max_candidates=4)
    assert result.candidates[0].rank < 3
    assert 'underdetermined' in result.message


def test_symmetry_family_can_choose_an_equivalent_hkl():
    project, _ = _synthetic_experiment()
    for peak in project.peaks:
        peak.assignment_mode = 'family'
        peak.hkl = (0, 0, 1)
    result = fit_orientation(project, max_candidates=6)
    best = result.candidates[0]
    assert best.rank == 3
    assert best.rms_pixel < 1e-5
    assert len(result.candidates) > 1


def test_unknown_hkl_is_identified_from_pixel_q():
    project, _ = _synthetic_experiment()
    project.peaks[0].assignment_mode = 'unknown'
    project.peaks[0].hkl = None
    result = fit_orientation(project, max_candidates=4)
    best = result.candidates[0]
    assert best.rms_pixel < 1e-5
    assert best.assignments['peak-1'] in {
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
    }


def test_kappa_motor_chain_moves_each_pose_without_euler_conversion():
    project, true_rotation = _synthetic_experiment()
    geometry = make_kappa_geometry()
    project.sample_chain = motor_chain_to_dict(geometry.sample)
    project.poses[0].motor_values = {'omega': 12, 'kappa': -4, 'phi': 8}
    project.poses[1].motor_values = {'kappa': 3, 'phi': -2}
    project.poses[2].motor_values = {}
    result = fit_orientation(project, max_candidates=3)
    best = result.candidates[0]
    assert best.rank == 3
    assert best.rms_pixel < 1e-5
    np.testing.assert_allclose(best.rotation, true_rotation, atol=1e-6)
    assert not np.allclose(best.pose_matrices['mount'], best.pose_matrices['after-move'])


def test_project_and_fitted_result_round_trip(tmp_path):
    project, _ = _synthetic_experiment()
    project.default_peak_sigma_h = 2
    project.default_peak_sigma_v = 3
    project_path = tmp_path / 'experiment.json'
    save_experiment(project_path, project)
    loaded = load_experiment(project_path)
    assert (loaded.default_peak_sigma_h, loaded.default_peak_sigma_v) == (2, 3)
    assert resolve_pose(loaded, 'after-move') == resolve_pose(project, 'after-move')
    result = fit_orientation(loaded, max_candidates=3)
    result_path = tmp_path / 'fit.json'
    save_fit_result(result_path, result)
    restored = load_fit_result(result_path)
    assert restored.candidates[0].assignments == result.candidates[0].assignments
    np.testing.assert_allclose(restored.candidates[0].reference_matrix,
                               result.candidates[0].reference_matrix)


def test_single_crystal_gui_captures_and_applies_selected_pose():
    from PyQt5.QtWidgets import QApplication
    from trxrdpy.simulation import sample
    from trxrdpy.simulation.geometry import MotorChain
    from trxrdpy.simulation.gui.services.simulation_service import SimulationService
    from trxrdpy.simulation.gui.state import GuiState
    from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab

    app = QApplication.instance() or QApplication([])
    assert app is not None
    tab = SingleCrystalTab(SimulationService(), GuiState())
    tab.single_func_combo.setCurrentIndex(
        tab.single_func_combo.findData('find_orientation_from_measured_braggs'))
    assert tab.single_func_combo.currentText() == 'Find Orientation Matrix'
    assert tab.state.single.func == 'find_orientation_from_measured_braggs'
    restored_tab = SingleCrystalTab(SimulationService(), tab.state)
    assert restored_tab.single_func_combo.currentText() == 'Find Orientation Matrix'
    panel = tab.orientation_fitting_panel
    assert panel.layout().indexOf(panel.apply_group) > panel.layout().indexOf(panel.browser)
    assert panel.apply_button.text() == 'Update crystal orientation at selected pose'
    assert panel.project is not None
    assert panel.project.initial_guess is not None
    assert panel.pose_editor.table.columnCount() == 11
    project, _ = _synthetic_experiment()
    result = fit_orientation(project, max_candidates=3)
    panel._set_project(project)
    panel.show_fit_result(result)
    panel.apply_pose.setCurrentIndex(panel.apply_pose.findData('after-move'))
    panel.pose_editor.table.item(0, 5).setText('90')
    panel.collect_project()
    panel.apply_candidate()
    assert tab.single_orientation_checkbox.isChecked()
    for row in tab.orientation_entries:
        for entry in row:
            assert entry.minimumWidth() >= entry.fontMetrics().horizontalAdvance(entry.text()) + 40
    assert float(tab.single_line_energy.text()) == 15100
    assert float(tab.single_line_sam_rotx.text()) == 4
    assert float(tab.single_line_sam_roty.text()) == -5
    assert float(tab.single_line_roty.text()) == 3
    fitted_project = OrientationExperiment.from_dict(result.project)
    chosen = resolve_pose(fitted_project, 'after-move')
    chain = MotorChain.from_dict(fitted_project.sample_chain)
    expected = result.candidates[0].pose_matrices['after-move']
    actual = sample.apply_orientation_transform(tab.collect_matrix(), chain,
                                                angles=chosen.motor_angles)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_applying_loaded_kappa_fit_restores_motor_chain():
    from PyQt5.QtWidgets import QApplication
    from trxrdpy.simulation.gui.services.simulation_service import SimulationService
    from trxrdpy.simulation.gui.state import GuiState
    from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab

    app = QApplication.instance() or QApplication([])
    assert app is not None
    project, _ = _synthetic_experiment()
    project.sample_chain = motor_chain_to_dict(make_kappa_geometry().sample)
    project.poses[0].motor_values = {'omega': 12, 'kappa': -4, 'phi': 8}
    project.poses[1].motor_values = {'kappa': 3}
    result = fit_orientation(project, max_candidates=3)
    tab = SingleCrystalTab(SimulationService(), GuiState())
    tab._apply_fitted_orientation(project, result.candidates[0], 'after-move')
    assert tab.single_geometry_mode_combo.currentText() == 'Custom geometry'
    assert tab._build_selected_geometry().sample.motor_names == ['omega', 'kappa', 'phi']
    np.testing.assert_allclose(tab.collect_matrix() @
                               tab._build_selected_geometry().sample.as_transform(
                                   angles=resolve_pose(project, 'after-move').motor_angles)[:3, :3].T,
                               result.candidates[0].pose_matrices['after-move'], atol=1e-7)


def test_gui_initial_guess_includes_reference_motor_reading():
    from PyQt5.QtWidgets import QApplication
    from trxrdpy.simulation.gui.services.orientation_fit_service import collect_single_crystal_setup
    from trxrdpy.simulation.gui.services.simulation_service import SimulationService
    from trxrdpy.simulation.gui.state import GuiState
    from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab

    app = QApplication.instance() or QApplication([])
    assert app is not None
    tab = SingleCrystalTab(SimulationService(), GuiState())
    tab.single_combo_sam_rotation_order.setCurrentText('xyz')
    tab.single_line_sam_rotx.setText('15')
    tab.single_orientation_checkbox.setChecked(True)
    for i in range(3):
        for j in range(3):
            tab.orientation_entries[i][j].setText('4' if i == j else '0')
    setup = collect_single_crystal_setup(tab)
    expected = np.eye(3) * 4 @ setup.sample_chain.as_transform(
        angles=setup.sample_angles)[:3, :3].T
    np.testing.assert_allclose(setup.initial_guess, expected)


def test_single_crystal_function_labels_keep_backend_keys():
    from PyQt5.QtWidgets import QApplication
    from trxrdpy.simulation.gui.services.simulation_service import SimulationService
    from trxrdpy.simulation.gui.state import GuiState
    from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab

    app = QApplication.instance() or QApplication([])
    assert app is not None
    tab = SingleCrystalTab(SimulationService(), GuiState())
    expected = [
        ('Laboratory 3D Simulation', 'simulate_3d'),
        ('Detector 2D Simulation', 'simulate_2d'),
        ('Ewald Construction Simulation', 'simulate_reciprocal'),
        ('Find Rotations for Bragg in Detector', 'target_hkl_near_pixel_fixed_energy'),
        ('Detector Rotations Mapping', 'detector_rotations_collecting_Braggs'),
        ('Motors Mapping for Catching Braggs', 'scan_two_parameters_for_Bragg_condition'),
        ('Find Orientation Matrix', 'find_orientation_from_measured_braggs'),
    ]
    assert [(tab.single_func_combo.itemText(i), tab.single_func_combo.itemData(i))
            for i in range(tab.single_func_combo.count())] == expected
    for label, key in expected[:-1]:
        tab.single_func_combo.setCurrentText(key)
        assert tab.single_func_combo.currentText() == label
        assert tab.state.single.func == key


def test_peak_editor_uncertainties_identity_and_scrolling():
    from PyQt5.QtWidgets import QApplication, QScrollArea
    from PyQt5.QtCore import QPoint
    from trxrdpy.simulation.gui.services.simulation_service import SimulationService
    from trxrdpy.simulation.gui.state import GuiState
    from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab

    app = QApplication.instance() or QApplication([])
    tab = SingleCrystalTab(SimulationService(), GuiState())
    tab.resize(1100, 750)
    tab.single_func_combo.setCurrentText('find_orientation_from_measured_braggs')
    panel = tab.orientation_fitting_panel
    panel.pose_editor.table.selectRow(0)
    panel.peak_sigma_h.setText('2.5')
    panel.peak_sigma_v.setText('3')
    panel.add_peak()
    table = panel.peak_editor.table
    assert table.item(0, 6).text() == '2.5'
    assert table.item(0, 7).text() == '3'
    assert table.cellWidget(0, 8).currentText() == 'Yes'
    table.cellWidget(0, 8).setCurrentText('No')
    table.cellWidget(0, 5).setCurrentIndex(
        table.cellWidget(0, 5).findData('family'))
    assert table.columnWidth(5) >= table.cellWidget(0, 5).view().minimumWidth()
    assert panel.collect_project().peaks[0].enabled is False
    assert panel.project.peaks[0].assignment_mode == 'family'
    tab.show()
    app.processEvents()
    outer = tab.findChild(QScrollArea)
    initial_scroll_limit = outer.verticalScrollBar().maximum()
    for _ in range(40):
        panel.add_peak()
    app.processEvents()
    assert outer.verticalScrollBar().maximum() > initial_scroll_limit
    last = table.item(table.rowCount() - 1, 0)
    app.processEvents()
    assert table.verticalScrollBar().maximum() == 0
    assert table.visualItemRect(last).bottom() < table.viewport().height()
    outer.verticalScrollBar().setValue(outer.verticalScrollBar().maximum())
    app.processEvents()
    fit_y = panel.fit_button.mapTo(outer.viewport(), QPoint(0, 0)).y()
    assert 0 <= fit_y < outer.viewport().height()
    tab.close()


def test_fitted_matrices_use_fixed_four_decimal_display():
    from PyQt5.QtWidgets import QApplication
    from trxrdpy.simulation.gui.widgets.orientation_fitting_panel import CandidateBrowser
    from trxrdpy.simulation.orientation_fit.refine import FittedCandidate, OrientationFitResult

    app = QApplication.instance() or QApplication([])
    candidate = FittedCandidate(
        rotation=np.eye(3), reference_matrix=np.array([[0.004843, -0.204744, 3.911903],
                                                       [0.655525, -3.856701, -0.202666],
                                                       [3.862019, 0.654879, 0.029494]]),
        assignments={}, weighted_error=0, rms_pixel=0, rank=3,
        angular_sigma_deg=None, peak_diagnostics=[],
        pose_matrices={'pose-1': np.eye(3)})
    project, _ = _synthetic_experiment()
    browser = CandidateBrowser()
    browser.set_result(OrientationFitResult([candidate], project.to_dict(), 'Fitted.'))
    assert 'Mounting orientation at zero sample-motor angles [Å]:' in browser.details.toPlainText()
    assert 'Reference pose mount sample motors: sam_rotx=4°' in browser.details.toPlainText()
    matrix_text = browser.details.toPlainText().split(
        'Direct-lattice orientation at reference pose [Å]:')[1].split(
        'Direct-lattice orientation at each recorded pose')[0]
    assert '0.0048' in matrix_text
    assert '3.9119' in matrix_text
    assert 'e-' not in matrix_text and 'e+' not in matrix_text


def test_zero_motor_matrix_undoes_the_first_measured_pose_rotation():
    from trxrdpy.simulation.geometry import MotorChain
    from trxrdpy.simulation.orientation_fit.forward_model import zero_motor_direct_lattice

    project, _ = _synthetic_experiment()
    project.sample_chain = motor_chain_to_dict(make_euler_geometry(sample_rotation_order='zxy').sample)
    project.poses[0].motor_values = {'sam_rotz': 0, 'sam_rotx': 0, 'sam_roty': -12.8}
    mounting = utils.apply_rotation(np.array([[0, 0, 3.91726],
                                                [0, -3.91726, 0],
                                                [3.91726, 0, 0]]), 3, 0, 7, 'zyx')
    reference_rotation = MotorChain.from_dict(project.sample_chain).as_transform(
        angles=resolve_pose(project, 'mount').motor_angles)[:3, :3]
    measured_pose_matrix = mounting @ reference_rotation.T
    np.testing.assert_allclose(zero_motor_direct_lattice(project, measured_pose_matrix),
                               mounting, atol=1e-12)
