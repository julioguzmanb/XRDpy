"""Measured-Bragg orientation workflow inside the Single Crystal tab."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QGridLayout,
                             QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QListWidget, QMessageBox, QPushButton, QScrollArea, QTableWidget,
                             QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)

from ...geometry import MotorChain
from ...orientation_fit.diagnostics import summarize_candidate
from ...orientation_fit.forward_model import (build_detector, observed_scattering_vector,
                                              zero_motor_direct_lattice)
from ...orientation_fit.observations import (BraggObservation, ExperimentalPose,
                                             OrientationExperiment, resolve_pose)
from ...orientation_fit.persistence import (load_experiment, load_fit_result,
                                            save_experiment, save_fit_result)
from ...orientation_fit.setup import new_experiment
from ..services.orientation_fit_service import collect_single_crystal_setup, start_orientation_fit


def _text(table, row, column):
    item = table.item(row, column)
    return item.text().strip() if item is not None else ''


def _cell(table, row, column, value='', *, editable=True):
    item = QTableWidgetItem(str(value))
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    table.setItem(row, column, item)


class PoseEditor(QWidget):
    """Editable pose table with motor columns taken from the active goniometer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.motor_names = []
        self.detector_motor_names = []
        self.detector_ids = []
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Blank energy and detector fields inherit from the parent. '
                                'Blank motor fields mean no change.'))
        self.table = QTableWidget(0, 5)
        self.table.setMinimumHeight(230)
        layout.addWidget(self.table)
        row = QHBoxLayout()
        self.add_button = QPushButton('Add pose from selected')
        self.remove_button = QPushButton('Remove selected pose')
        row.addWidget(self.add_button)
        row.addWidget(self.remove_button)
        row.addStretch()
        layout.addLayout(row)
        self.add_button.clicked.connect(self.add_pose)
        self.remove_button.clicked.connect(self.remove_selected)

    def configure(self, motor_names, detector_ids, detector_motor_names=()):
        self.motor_names = list(motor_names)
        self.detector_ids = list(detector_ids)
        self.detector_motor_names = list(detector_motor_names)
        self.table.setColumnCount(5 + len(self.motor_names) + len(self.detector_motor_names))
        self.table.setHorizontalHeaderLabels(['Pose', 'From', 'Motor values', 'Energy [eV]',
                                              'Detector'] + self.motor_names +
                                             self.detector_motor_names)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.Interactive)
        for column, width in enumerate([100, 170, 195, 115, 135] +
                                       [110] * len(self.motor_names) +
                                       [110] * len(self.detector_motor_names)):
            self.table.setColumnWidth(column, width)

    def set_poses(self, poses):
        self.table.setRowCount(0)
        for pose in poses:
            self._append_pose(pose)

    def _append_pose(self, pose):
        row = self.table.rowCount()
        self.table.insertRow(row)
        _cell(self.table, row, 0, pose.id, editable=False)
        parent = QComboBox()
        parent.addItem('Reference pose', None)
        for preceding in range(row):
            identifier = _text(self.table, preceding, 0)
            parent.addItem(identifier, identifier)
        if pose.parent_id is not None and parent.findData(pose.parent_id) < 0:
            parent.addItem(f'Missing or later pose: {pose.parent_id}', pose.parent_id)
        parent.setCurrentIndex(max(0, parent.findData(pose.parent_id)))
        self.table.setCellWidget(row, 1, parent)
        mode = QComboBox()
        mode.addItem('Absolute readings', 'absolute')
        mode.addItem('Changes from parent', 'relative')
        mode.setCurrentIndex(max(0, mode.findData(pose.motor_mode)))
        self.table.setCellWidget(row, 2, mode)
        _cell(self.table, row, 3, '' if pose.energy_eV is None else f'{pose.energy_eV:g}')
        detector = QComboBox()
        detector.addItem('Inherit', None)
        for identifier in self.detector_ids:
            detector.addItem(identifier, identifier)
        detector.setCurrentIndex(max(0, detector.findData(pose.detector_setup_id)))
        self.table.setCellWidget(row, 4, detector)
        for col, name in enumerate(self.motor_names, 5):
            _cell(self.table, row, col, '' if name not in pose.motor_values else f'{pose.motor_values[name]:g}')
        for col, name in enumerate(self.detector_motor_names, 5 + len(self.motor_names)):
            _cell(self.table, row, col, '' if name not in pose.detector_angles else f'{pose.detector_angles[name]:g}')
        self.table.selectRow(row)

    def add_pose(self):
        row = self.table.currentRow()
        parent_id = _text(self.table, row, 0) if row >= 0 else (
            _text(self.table, self.table.rowCount() - 1, 0) if self.table.rowCount() else None)
        existing = {_text(self.table, index, 0) for index in range(self.table.rowCount())}
        number = 1
        while f'pose-{number}' in existing:
            number += 1
        self._append_pose(ExperimentalPose(f'pose-{number}', parent_id=parent_id,
                                           motor_mode='relative' if parent_id else 'absolute'))

    def remove_selected(self):
        row = self.table.currentRow()
        if row <= 0:
            return
        identifier = _text(self.table, row, 0)
        for other in range(self.table.rowCount()):
            if other != row and self.table.cellWidget(other, 1).currentData() == identifier:
                QMessageBox.warning(self, 'Pose in use', 'Another pose depends on this one.')
                return
        self.table.removeRow(row)

    def collect_poses(self):
        poses = []
        for row in range(self.table.rowCount()):
            motors = {name: float(value) for col, name in enumerate(self.motor_names, 5)
                      if (value := _text(self.table, row, col))}
            detector_angles = {
                name: float(value)
                for col, name in enumerate(self.detector_motor_names, 5 + len(self.motor_names))
                if (value := _text(self.table, row, col))
            }
            energy = _text(self.table, row, 3)
            poses.append(ExperimentalPose(
                id=_text(self.table, row, 0), parent_id=self.table.cellWidget(row, 1).currentData(),
                motor_mode=self.table.cellWidget(row, 2).currentData(), motor_values=motors,
                energy_eV=float(energy) if energy else None,
                detector_setup_id=self.table.cellWidget(row, 4).currentData(),
                detector_angles=detector_angles,
            ))
        return poses

    def selected_pose_id(self):
        row = self.table.currentRow()
        return _text(self.table, row, 0) if row >= 0 else None


class PeakEditor(QWidget):
    """Bragg pixels grouped by pose, with exact/family/unknown identity modes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pose_ids = []
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(['Peak', 'Pose', 'Pixel H', 'Pixel V', 'HKL',
                                              'Identity', 'σH [px]', 'σV [px]', 'Include?',
                                              '|q| [Å⁻¹]', 'd [Å]'])
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        identity_width = max(self.table.fontMetrics().horizontalAdvance(label)
                             for label in ('Exact HKL', 'Any equivalent', 'Unknown HKL')) + 120
        for column, width in enumerate((125, 130, 120, 120, 115,
                                        max(165, identity_width), 105, 105, 105, 125, 105)):
            self.table.setColumnWidth(column, width)
        layout.addWidget(self.table)
        row = QHBoxLayout()
        self.add_button = QPushButton('Add peak to selected pose')
        self.remove_button = QPushButton('Remove selected peak')
        self.preview_button = QPushButton('Update q/d from pixels')
        row.addWidget(self.add_button)
        row.addWidget(self.remove_button)
        row.addWidget(self.preview_button)
        row.addStretch()
        layout.addLayout(row)
        self.remove_button.clicked.connect(self.remove_selected)
        self._resize_table()

    def _resize_table(self):
        """Show every peak row and leave vertical scrolling to the containing tab."""
        height = (self.table.horizontalHeader().height() +
                  sum(self.table.rowHeight(row) for row in range(self.table.rowCount())) +
                  self.table.horizontalScrollBar().sizeHint().height() +
                  2 * self.table.frameWidth() + 8)
        self.table.setFixedHeight(max(230, height))
        ancestor = self
        while ancestor is not None:
            if ancestor.layout() is not None:
                ancestor.layout().invalidate()
            ancestor.updateGeometry()
            if isinstance(ancestor, QScrollArea) and ancestor.widget() is not None:
                ancestor.widget().adjustSize()
                break
            ancestor = ancestor.parentWidget()

    def remove_selected(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self._resize_table()

    def configure(self, pose_ids):
        self.pose_ids = list(pose_ids)
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 1)
            selected = combo.currentData()
            combo.clear()
            for identifier in self.pose_ids:
                combo.addItem(identifier, identifier)
            if selected and selected not in self.pose_ids:
                combo.addItem(f'Missing pose: {selected}', selected)
            combo.setCurrentIndex(max(0, combo.findData(selected)))

    def set_peaks(self, peaks):
        self.table.setRowCount(0)
        for peak in peaks:
            self._append_peak(peak)
        self._resize_table()

    def _append_peak(self, peak):
        row = self.table.rowCount()
        self.table.insertRow(row)
        _cell(self.table, row, 0, peak.id, editable=False)
        pose = QComboBox()
        for identifier in self.pose_ids:
            pose.addItem(identifier, identifier)
        if peak.pose_id not in self.pose_ids:
            pose.addItem(f'Missing pose: {peak.pose_id}', peak.pose_id)
        pose.setCurrentIndex(max(0, pose.findData(peak.pose_id)))
        self.table.setCellWidget(row, 1, pose)
        _cell(self.table, row, 2, f'{peak.pixel_h:g}')
        _cell(self.table, row, 3, f'{peak.pixel_v:g}')
        _cell(self.table, row, 4, '' if peak.hkl is None else ','.join(map(str, peak.hkl)))
        mode = QComboBox()
        for label, value in [('Exact HKL', 'exact'), ('Any equivalent', 'family'),
                             ('Unknown HKL', 'unknown')]:
            mode.addItem(label, value)
        mode.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        mode.view().setMinimumWidth(self.table.columnWidth(5))
        mode.setCurrentIndex(max(0, mode.findData(peak.assignment_mode)))
        self.table.setCellWidget(row, 5, mode)
        _cell(self.table, row, 6, f'{peak.sigma_h:g}')
        _cell(self.table, row, 7, f'{peak.sigma_v:g}')
        used = QComboBox()
        used.addItem('Yes', True)
        used.addItem('No', False)
        used.setCurrentIndex(0 if peak.enabled else 1)
        self.table.setCellWidget(row, 8, used)
        _cell(self.table, row, 9, '', editable=False)
        _cell(self.table, row, 10, '', editable=False)
        self.table.selectRow(row)
        self._resize_table()

    def add_peak(self, pose_id, sigma_h=1.0, sigma_v=1.0):
        if not pose_id:
            QMessageBox.warning(self, 'Select a pose', 'Select the pose where the peak was observed.')
            return
        existing = {_text(self.table, row, 0) for row in range(self.table.rowCount())}
        number = 1
        while f'peak-{number}' in existing:
            number += 1
        self._append_peak(BraggObservation(f'peak-{number}', pose_id, 0, 0, (1, 0, 0),
                                           sigma_h=sigma_h, sigma_v=sigma_v))

    def collect_peaks(self):
        peaks = []
        for row in range(self.table.rowCount()):
            hkl_text = _text(self.table, row, 4).strip('[]() ')
            hkl = tuple(int(value.strip()) for value in hkl_text.split(',')) if hkl_text else None
            peaks.append(BraggObservation(
                id=_text(self.table, row, 0), pose_id=self.table.cellWidget(row, 1).currentData(),
                pixel_h=float(_text(self.table, row, 2)), pixel_v=float(_text(self.table, row, 3)),
                hkl=hkl, assignment_mode=self.table.cellWidget(row, 5).currentData(),
                sigma_h=float(_text(self.table, row, 6)), sigma_v=float(_text(self.table, row, 7)),
                enabled=self.table.cellWidget(row, 8).currentData(),
            ))
        return peaks


class CandidateBrowser(QWidget):
    """Select a fitted alternative and inspect every pose and measured peak."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.setMaximumHeight(125)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(180)
        layout.addWidget(self.list)
        layout.addWidget(self.details)
        self.result = None
        self.list.currentRowChanged.connect(self.show_candidate)

    def set_result(self, result):
        self.result = result
        self.list.clear()
        if result is None:
            self.details.clear()
            return
        best_score = result.candidates[0].weighted_error if result.candidates else 0
        for index, candidate in enumerate(result.candidates, 1):
            self.list.addItem(f'Candidate {index}: {candidate.rms_pixel:.2f} px RMS, '
                              f'Δ weighted score {candidate.weighted_error - best_score:.3g}, '
                              f'rank {candidate.rank}/3')
        if result.candidates:
            self.list.setCurrentRow(0)

    def selected_candidate(self):
        row = self.list.currentRow()
        return self.result.candidates[row] if self.result and row >= 0 else None

    def show_candidate(self, index):
        if self.result is None or index < 0:
            self.details.clear()
            return
        candidate = self.result.candidates[index]
        project = OrientationExperiment.from_dict(self.result.project)
        reference_id = project.reference_pose_id or project.poses[0].id
        reference = resolve_pose(project, reference_id)
        motor_readings = ', '.join(
            f'{name}={value:g}°' for name, value in reference.motor_angles.items())
        def display_matrix(matrix):
            return np.array2string(
                np.asarray(matrix), separator=' ',
                formatter={'float_kind': lambda value: f'{0.0 if abs(value) < 0.00005 else value:.4f}'},
            )
        lines = [self.result.message, '', summarize_candidate(candidate),
                 '', 'Mounting orientation at zero sample-motor angles [Å]:',
                 display_matrix(zero_motor_direct_lattice(project, candidate.reference_matrix)),
                 '', f'Reference pose {reference_id} sample motors: {motor_readings}',
                 'Direct-lattice orientation at reference pose [Å]:',
                 display_matrix(candidate.reference_matrix),
                 '', 'Direct-lattice orientation at each recorded pose [Å]:']
        for pose_id, matrix in candidate.pose_matrices.items():
            lines.extend([pose_id + ':', display_matrix(matrix)])
        self.details.setPlainText('\n'.join(lines))


class OrientationFittingPanel(QWidget):
    """Single Crystal pose/peak editor and orientation-candidate browser."""

    def __init__(self, single_crystal_tab):
        super().__init__(single_crystal_tab)
        self.tab = single_crystal_tab
        self.project: OrientationExperiment | None = None
        self.result = None
        layout = QVBoxLayout(self)
        self.status = QLabel('Capture the current Single Crystal setup to begin.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        actions = QHBoxLayout()
        self.capture_button = QPushButton('New project from current setup')
        self.detector_button = QPushButton('Add detector setup from current controls')
        self.save_project_button = QPushButton('Save project...')
        self.load_project_button = QPushButton('Load project...')
        for button in (self.capture_button, self.detector_button,
                       self.save_project_button, self.load_project_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.pose_editor = PoseEditor()
        self.peak_editor = PeakEditor()
        pose_group = QGroupBox('Experimental poses')
        QVBoxLayout(pose_group).addWidget(self.pose_editor)
        peak_group = QGroupBox('Observed Bragg peaks')
        QVBoxLayout(peak_group).addWidget(self.peak_editor)
        layout.addWidget(pose_group)
        layout.addWidget(peak_group)
        uncertainty_group = QGroupBox('Uncertainties')
        uncertainty = QGridLayout(uncertainty_group)
        uncertainty.addWidget(QLabel('Peak position σH [px]:'), 0, 0)
        self.peak_sigma_h = QLineEdit('1')
        uncertainty.addWidget(self.peak_sigma_h, 0, 1)
        uncertainty.addWidget(QLabel('Peak position σV [px]:'), 0, 2)
        self.peak_sigma_v = QLineEdit('1')
        uncertainty.addWidget(self.peak_sigma_v, 0, 3)
        self.apply_peak_sigmas_button = QPushButton('Apply to all peaks')
        uncertainty.addWidget(self.apply_peak_sigmas_button, 0, 4)
        uncertainty.addWidget(QLabel('Energy uncertainty [eV]:'), 1, 0)
        self.energy_sigma = QLineEdit('0')
        uncertainty.addWidget(self.energy_sigma, 1, 1)
        uncertainty.addWidget(QLabel('Motor uncertainty [°]:'), 1, 2)
        self.motor_sigma = QLineEdit('0')
        uncertainty.addWidget(self.motor_sigma, 1, 3)
        uncertainty.addWidget(QLabel('Lattice uncertainty [%]:'), 2, 0)
        self.lattice_sigma = QLineEdit('0')
        uncertainty.addWidget(self.lattice_sigma, 2, 1)
        uncertainty.addWidget(QLabel('Detector calibration uncertainty [px]:'), 2, 2)
        self.detector_sigma = QLineEdit('0')
        uncertainty.addWidget(self.detector_sigma, 2, 3)
        note = QLabel('Peak σH/σV describe each measured Bragg location; edit individual rows '
                      'to override these defaults. Detector calibration uncertainty is shared.')
        note.setWordWrap(True)
        uncertainty.addWidget(note, 3, 0, 1, 5)
        layout.addWidget(uncertainty_group)
        fit_actions = QHBoxLayout()
        self.fit_button = QPushButton('Fit crystal orientation')
        self.save_fit_button = QPushButton('Save fitted result...')
        self.load_fit_button = QPushButton('Load fitted result...')
        self.save_fit_button.setEnabled(False)
        for button in (self.fit_button, self.save_fit_button, self.load_fit_button):
            fit_actions.addWidget(button)
        fit_actions.addStretch()
        layout.addLayout(fit_actions)
        self.browser = CandidateBrowser()
        layout.addWidget(self.browser)
        self.apply_group = QGroupBox('Update Single Crystal from fitted candidate')
        apply_actions = QHBoxLayout(self.apply_group)
        self.apply_pose = QComboBox()
        self.apply_button = QPushButton('Update crystal orientation at selected pose')
        self.apply_button.setEnabled(False)
        apply_actions.addWidget(QLabel('Pose to simulate:'))
        apply_actions.addWidget(self.apply_pose)
        apply_actions.addWidget(self.apply_button)
        apply_actions.addStretch()
        layout.addWidget(self.apply_group)
        self.capture_button.clicked.connect(self.capture_setup)
        self.detector_button.clicked.connect(self.add_detector_setup)
        self.save_project_button.clicked.connect(self.save_project)
        self.load_project_button.clicked.connect(self.load_project)
        self.fit_button.clicked.connect(self.run_fit)
        self.save_fit_button.clicked.connect(self.save_fit)
        self.load_fit_button.clicked.connect(self.load_fit)
        self.apply_button.clicked.connect(self.apply_candidate)
        self.pose_editor.add_button.clicked.connect(self._refresh_pose_choices)
        self.pose_editor.remove_button.clicked.connect(self._refresh_pose_choices)
        self.peak_editor.add_button.clicked.connect(self.add_peak)
        self.peak_editor.preview_button.clicked.connect(self.preview_q_values)
        self.apply_peak_sigmas_button.clicked.connect(self.apply_peak_sigmas_to_all)

    def add_peak(self):
        try:
            sigma_h = float(self.peak_sigma_h.text())
            sigma_v = float(self.peak_sigma_v.text())
            if not np.isfinite(sigma_h) or not np.isfinite(sigma_v) or min(sigma_h, sigma_v) <= 0:
                raise ValueError('Peak position uncertainties must be positive finite numbers.')
            self.peak_editor.add_peak(self.pose_editor.selected_pose_id(), sigma_h, sigma_v)
        except ValueError as exc:
            QMessageBox.warning(self, 'Peak uncertainty', str(exc))

    def apply_peak_sigmas_to_all(self):
        try:
            sigma_h = float(self.peak_sigma_h.text())
            sigma_v = float(self.peak_sigma_v.text())
            if not np.isfinite(sigma_h) or not np.isfinite(sigma_v) or min(sigma_h, sigma_v) <= 0:
                raise ValueError('Peak position uncertainties must be positive finite numbers.')
            for row in range(self.peak_editor.table.rowCount()):
                _cell(self.peak_editor.table, row, 6, f'{sigma_h:g}')
                _cell(self.peak_editor.table, row, 7, f'{sigma_v:g}')
        except ValueError as exc:
            QMessageBox.warning(self, 'Peak uncertainty', str(exc))

    def ensure_project(self):
        if self.project is None:
            try:
                self.capture_setup()
            except Exception as exc:
                self.status.setText(f'Current setup needs attention: {exc}')

    def capture_setup(self):
        if self.project and (self.peak_editor.table.rowCount() or self.pose_editor.table.rowCount() > 1):
            answer = QMessageBox.question(self, 'Start a new orientation project?',
                                          'This replaces the poses and peaks currently in the editor.',
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        setup = collect_single_crystal_setup(self.tab)
        self._set_project(new_experiment(setup))

    def _set_project(self, project):
        self.project = project
        chain = MotorChain.from_dict(project.sample_chain)
        detector_motors = self._detector_motor_names(project)
        self.pose_editor.configure(chain.motor_names, [item.id for item in project.detectors],
                                   detector_motors)
        self.pose_editor.set_poses(project.poses)
        self.apply_pose.clear()
        for pose in project.poses:
            self.apply_pose.addItem(pose.id, pose.id)
        self.peak_editor.configure([pose.id for pose in project.poses])
        self.peak_editor.set_peaks(project.peaks)
        self.energy_sigma.setText(f'{project.energy_sigma_eV:g}')
        self.motor_sigma.setText(f'{project.motor_sigma_deg:g}')
        self.lattice_sigma.setText(f'{100 * project.lattice_sigma_fraction:g}')
        self.detector_sigma.setText(f'{project.detector_sigma_px:g}')
        self.peak_sigma_h.setText(f'{project.default_peak_sigma_h:g}')
        self.peak_sigma_v.setText(f'{project.default_peak_sigma_v:g}')
        self.status.setText(f'Reference pose: {project.reference_pose_id or project.poses[0].id}. '
                            'Its matrix includes that pose’s motor angles; fitted results also '
                            'show the mounting orientation at zero motor angles.')
        self.result = None
        self.browser.set_result(None)
        self.save_fit_button.setEnabled(False)
        self.apply_button.setEnabled(False)

    def _refresh_pose_choices(self):
        identifiers = [_text(self.pose_editor.table, row, 0)
                       for row in range(self.pose_editor.table.rowCount())]
        self.peak_editor.configure(identifiers)
        if self.result is not None:
            return
        selected = self.apply_pose.currentData()
        self.apply_pose.clear()
        for identifier in identifiers:
            self.apply_pose.addItem(identifier, identifier)
        self.apply_pose.setCurrentIndex(max(0, self.apply_pose.findData(selected)))

    @staticmethod
    def _detector_motor_names(project):
        names = []
        for setup in project.detectors:
            if setup.motor_chain is not None:
                for name in MotorChain.from_dict(setup.motor_chain).motor_names:
                    if name not in names:
                        names.append(name)
        return names

    def collect_project(self):
        if self.project is None:
            raise ValueError('Capture or load an experiment setup first.')
        self.project.poses = self.pose_editor.collect_poses()
        self.project.peaks = self.peak_editor.collect_peaks()
        self.project.energy_sigma_eV = float(self.energy_sigma.text())
        self.project.motor_sigma_deg = float(self.motor_sigma.text())
        self.project.lattice_sigma_fraction = float(self.lattice_sigma.text()) / 100
        self.project.detector_sigma_px = float(self.detector_sigma.text())
        self.project.default_peak_sigma_h = float(self.peak_sigma_h.text())
        self.project.default_peak_sigma_v = float(self.peak_sigma_v.text())
        return self.project

    def preview_q_values(self):
        """Show the measured scattering-vector magnitude at each recorded pixel."""
        try:
            project = self.collect_project()
            for row, peak in enumerate(project.peaks):
                pose = resolve_pose(project, peak.pose_id)
                q = float(np.linalg.norm(observed_scattering_vector(
                    project, pose, peak, build_detector(project, pose))))
                _cell(self.peak_editor.table, row, 9, f'{q:.6g}', editable=False)
                _cell(self.peak_editor.table, row, 10,
                      f'{2 * np.pi / q:.6g}' if q > 0 else '∞', editable=False)
        except Exception as exc:
            QMessageBox.critical(self, 'Peak q preview', str(exc))

    def add_detector_setup(self):
        try:
            project = self.collect_project()
            captured = collect_single_crystal_setup(self.tab).detector
            taken = {item.id for item in project.detectors}
            number = 1
            while f'detector-{number}' in taken:
                number += 1
            captured.id = f'detector-{number}'
            project.detectors.append(captured)
            self.pose_editor.configure(MotorChain.from_dict(project.sample_chain).motor_names,
                                       [item.id for item in project.detectors],
                                       self._detector_motor_names(project))
            self.pose_editor.set_poses(project.poses)
            self.status.setText(f'Added detector setup {captured.id}. Select it on the appropriate pose.')
        except Exception as exc:
            QMessageBox.critical(self, 'Detector setup', str(exc))

    def save_project(self):
        try:
            project = self.collect_project()
            chosen, _ = QFileDialog.getSaveFileName(self, 'Save orientation project',
                                                     'orientation_project.json', 'JSON files (*.json)')
            if chosen:
                path = Path(chosen)
                save_experiment(path, project, overwrite=path.exists())
                self.status.setText(f'Saved orientation project: {path}')
        except Exception as exc:
            QMessageBox.critical(self, 'Save orientation project', str(exc))

    def load_project(self):
        chosen, _ = QFileDialog.getOpenFileName(self, 'Load orientation project', '', 'JSON files (*.json)')
        if chosen:
            try:
                self._set_project(load_experiment(chosen))
            except Exception as exc:
                QMessageBox.critical(self, 'Load orientation project', str(exc))

    def run_fit(self):
        try:
            project = self.collect_project()
            start_orientation_fit(self, project, on_success=self.show_fit_result,
                                  on_error=lambda trace: QMessageBox.critical(
                                      self, 'Orientation fit', trace.strip().splitlines()[-1]),
                                  button=self.fit_button)
        except Exception as exc:
            QMessageBox.critical(self, 'Orientation fit', str(exc))

    def show_fit_result(self, result):
        self.result = result
        self.browser.set_result(result)
        selected = self.apply_pose.currentData()
        self.apply_pose.clear()
        fitted_project = OrientationExperiment.from_dict(result.project)
        for pose in fitted_project.poses:
            self.apply_pose.addItem(pose.id, pose.id)
        self.apply_pose.setCurrentIndex(max(0, self.apply_pose.findData(selected)))
        self.save_fit_button.setEnabled(True)
        self.apply_button.setEnabled(True)
        self.status.setText(result.message)

    def save_fit(self):
        if self.result is None:
            return
        chosen, _ = QFileDialog.getSaveFileName(self, 'Save orientation fit',
                                                 'orientation_fit.json', 'JSON files (*.json)')
        if chosen:
            try:
                path = Path(chosen)
                save_fit_result(path, self.result, overwrite=path.exists())
            except Exception as exc:
                QMessageBox.critical(self, 'Save orientation fit', str(exc))

    def load_fit(self):
        chosen, _ = QFileDialog.getOpenFileName(self, 'Load orientation fit', '', 'JSON files (*.json)')
        if chosen:
            try:
                result = load_fit_result(chosen)
                self._set_project(OrientationExperiment.from_dict(result.project))
                self.show_fit_result(result)
            except Exception as exc:
                QMessageBox.critical(self, 'Load orientation fit', str(exc))

    def apply_candidate(self):
        candidate = self.browser.selected_candidate()
        if candidate is None or self.result is None:
            return
        try:
            fitted_project = OrientationExperiment.from_dict(self.result.project)
            self.tab._apply_fitted_orientation(fitted_project, candidate,
                                               self.apply_pose.currentData())
            self.status.setText(f'Applied the selected orientation at {self.apply_pose.currentData()}. '
                                'Choose simulate_2d or simulate_3d to inspect that condition.')
        except Exception as exc:
            QMessageBox.critical(self, 'Apply orientation', str(exc))
