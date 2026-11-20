"""Detector templates and export of the current flat simulation geometry."""
from pathlib import Path

import numpy as np
import pyFAI.detectors
from scipy.spatial.transform import Rotation
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from ...detector import Detector
from ...poni import write_poni_file


def update_detector_fields(tab, prefix):
    field = lambda name: getattr(tab, prefix + name)
    field('_manual_group').setVisible(True)
    predefined = field('_combo_det_type').currentText().lower() not in ('manual', 'poni')
    for axis in ('h', 'v'):
        field('_line_num_px_' + axis).setReadOnly(predefined)
    if predefined and not tab._loading_state:
        try:
            detector = pyFAI.detectors.detector_factory(field('_combo_det_type').currentText())
            shape = detector.max_shape or (None, None)
            for axis, size in zip(('v', 'h'), shape):
                field('_line_num_px_' + axis).setReadOnly(size is not None)
            for name, value in (('pxsize_h', detector.pixel2), ('pxsize_v', detector.pixel1),
                                ('num_px_h', shape[1]), ('num_px_v', shape[0])):
                field('_line_' + name).setText('' if value is None else str(value))
        except Exception as exc:
            QMessageBox.warning(tab, 'Detector template', str(exc))


def current_detector(tab, prefix):
    value = lambda name: float(getattr(tab, prefix + '_line_' + name).text())
    order = getattr(tab, prefix + ('_combo_det_rotation_order' if prefix == 'single'
                                  else '_combo_rotation_order')).currentText()
    detector = Detector(detector_type='manual', pxsize_h=value('pxsize_h'), pxsize_v=value('pxsize_v'),
                        num_pixels_h=value('num_px_h'), num_pixels_v=value('num_px_v'),
                        dist=value('dist'), poni1=value('poni1'), poni2=value('poni2'),
                        rotx=value('rotx'), roty=value('roty'), rotz=value('rotz'),
                        rotation_order=order, binning=(int(value('bin_h')), int(value('bin_v'))))
    if prefix == 'single' and tab._single_geometry_enabled():
        geometry = (tab._build_predefined_geometry() if tab._single_predefined_geometry_enabled()
                    else tab._build_custom_geometry())
        detector.set_transform(geometry.detector_transform(angles=tab._single_geometry_detector_angles_dict()))
    return detector


def export_current_detector(tab, prefix, path, overwrite=False):
    detector = current_detector(tab, prefix)
    transform = detector.get_transform()
    # pyFAI axes (vertical, horizontal, beam) correspond to simulation (z, -y, x).
    basis = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
    rotation = basis.T @ transform[:3, :3] @ basis
    angles = Rotation.from_matrix(rotation).as_euler('xyz') * [-1, -1, 1]
    translation = transform[:3, :3].T @ transform[:3, 3]
    return write_poni_file(path, dist=detector.dist + translation[0],
                           poni1=detector.poni1 - translation[2],
                           poni2=detector.poni2 + translation[1],
                           pixel1=detector.pxsize_v, pixel2=detector.pxsize_h,
                           max_shape=(detector.num_pixels_v, detector.num_pixels_h),
                           rot1=angles[0], rot2=angles[1], rot3=angles[2],
                           energy_ev=float(getattr(tab, prefix + '_line_energy').text()),
                           overwrite=overwrite)


def save_current_detector(tab, prefix):
    selected, _ = QFileDialog.getSaveFileName(tab, 'Save Detector PONI', 'detector.poni',
                                             'PONI files (*.poni)', options=QFileDialog.DontConfirmOverwrite)
    if not selected:
        return
    path = Path(selected).expanduser()
    if not path.suffix:
        path = path.with_suffix('.poni')
    try:
        overwrite = False
        if path.exists():
            overwrite = QMessageBox.question(tab, 'Replace PONI?', f'Replace {path}?',
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes
            if not overwrite:
                return
        export_current_detector(tab, prefix, path, overwrite)
        QMessageBox.information(tab, 'PONI saved', str(path))
    except Exception as exc:
        QMessageBox.critical(tab, 'PONI export error', str(exc))
