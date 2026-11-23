"""Capture Single Crystal controls and run orientation fits off the GUI thread."""
from __future__ import annotations

from copy import deepcopy
from threading import Event

from ....analysis.gui.widgets.task_output_dialog import run_task_with_output_dialog
from ...diffractometers import make_euler_geometry, motor_chain_to_dict
from ...parallel import SimulationCancelled, cancellation
from ...orientation_fit.observations import DetectorSetup
from ...orientation_fit.refine import fit_orientation
from ...orientation_fit.setup import normalize_goniometer, snapshot_setup
from ...sample import LatticeStructure
from ..widgets.detector_export import current_detector


def collect_single_crystal_setup(tab):
    """Capture effective cell, energy, motor chain, and detector calibration."""
    lattice = dict(
        space_group=int(float(tab.single_line_space_group.text())),
        **{name: float(getattr(tab, 'single_line_sam_' + name).text())
           for name in ('a', 'b', 'c', 'alpha', 'beta', 'gamma')},
    )
    active_geometry = tab._build_selected_geometry() if tab._single_geometry_enabled() else None
    chain = normalize_goniometer(active_geometry,
                                 euler_order=tab.single_combo_sam_rotation_order.currentText())
    if active_geometry is None:
        sample_angles = {'sam_' + name: float(getattr(tab, 'single_line_sam_' + name).text())
                         for name in ('rotx', 'roty', 'rotz')}
        detector_angles = {'det_' + name: float(getattr(tab, 'single_line_' + name).text())
                           for name in ('rotx', 'roty', 'rotz')}
        detector_chain = motor_chain_to_dict(make_euler_geometry(
            detector_rotation_order=tab.single_combo_det_rotation_order.currentText()).detector)
    else:
        sample_angles = tab._single_geometry_sample_angles_dict() or chain.default_angles()
        detector_angles = tab._single_geometry_detector_angles_dict() or {}
        detector_chain = motor_chain_to_dict(active_geometry.detector)
    detector = current_detector(tab, 'single')
    parameters = dict(
        detector_type='manual', pxsize_h=float(detector.pxsize_h),
        pxsize_v=float(detector.pxsize_v), num_pixels_h=int(detector.num_pixels_h),
        num_pixels_v=int(detector.num_pixels_v), dist=float(detector.dist),
        poni1=float(detector.poni1), poni2=float(detector.poni2),
        rotx=float(detector.rotx), roty=float(detector.roty), rotz=float(detector.rotz),
        rotation_order=detector.rotation_order, binning=(1, 1),
    )
    setup = DetectorSetup('detector-1', parameters, dict(detector_angles), detector_chain,
                          tab.single_line_poni_file.text().strip())
    guess = tab.collect_matrix() if tab.single_orientation_checkbox.isChecked() else None
    if tab.single_orientation_checkbox.isChecked() and guess is None:
        raise ValueError('The custom initial orientation matrix is incomplete.')
    if guess is None:
        guess = LatticeStructure(**lattice).initial_crystal_orientation
    guess = guess @ chain.as_transform(angles=sample_angles)[:3, :3].T
    return snapshot_setup(lattice, setup, chain,
                          {'energy_eV': float(tab.single_line_energy.text())},
                          initial_guess=guess.tolist() if guess is not None else None,
                          sample_angles=sample_angles)


def start_orientation_fit(parent, experiment, *, on_success, on_error=None, button=None):
    """Fit a frozen project using the existing output and safe-stop dialog."""
    if getattr(parent, '_orientation_fit_busy', False):
        return None
    frozen = deepcopy(experiment)
    stopped = Event()

    def compute():
        try:
            with cancellation(stopped):
                return fit_orientation(frozen)
        except SimulationCancelled:
            print('Orientation fit stopped; incomplete candidates were discarded.', flush=True)
            return {'cancelled': True}

    def completed(result):
        if isinstance(result, dict) and result.get('cancelled'):
            return
        on_success(result)

    def failed(traceback):
        if on_error:
            on_error(traceback)

    parent._orientation_fit_busy = True
    if button is not None:
        button.setEnabled(False)
    dialog = run_task_with_output_dialog(parent, 'Fit crystal orientation', compute,
                                         on_success=completed, on_error=failed,
                                         cancel_callback=stopped.set,
                                         auto_close_on_success=True,
                                         auto_close_delay_ms=700)

    def finished():
        parent._orientation_fit_busy = False
        if button is not None:
            button.setEnabled(True)
        dialog._worker.func = None

    dialog._thread.finished.connect(finished)
    return dialog
