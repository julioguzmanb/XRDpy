"""Reflection-first reader for saved and in-memory Bragg mappings."""
from copy import deepcopy
from threading import Event

import numpy as np
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QFileDialog,
                             QHeaderView, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QTableView, QVBoxLayout)

from ...bragg_explorer import BraggExplorer
from ...mapping_display import hkl_label
from ...mapping_storage import _axis_unit


class _PositionsModel(QAbstractTableModel):
    def __init__(self, browser, parent=None):
        super().__init__(parent)
        self.browser = browser
        self.hkl = None
        self.positions = []
        self.counts = []

    def set_hkl(self, hkl):
        self.beginResetModel()
        self.hkl = hkl
        self.positions = self.browser.positions(hkl) if hkl else []
        self.counts = [self.browser.point_count(index, hkl) for index in self.positions]
        self.endResetModel()

    def rowCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.positions)

    def columnCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.browser.axes) + 1

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if section < len(self.browser.axes):
                name = list(self.browser.axes)[section]
                return f'{name} [{_axis_unit(name)}]'
            return 'Saved Bragg points'
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        row, column = index.row(), index.column()
        if column == len(self.browser.axes):
            return str(self.counts[row])
        return f'{self.browser.position_values(self.positions[row])[column]:g}'


class _PointsModel(QAbstractTableModel):
    def __init__(self, names, parent=None):
        super().__init__(parent)
        self.names = names
        self.points = np.empty((0, 2), dtype=float)

    def set_points(self, points):
        self.beginResetModel()
        self.points = np.asarray(points, dtype=float).reshape(-1, 2)
        self.endResetModel()

    def rowCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.points)

    def columnCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else 2

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            name = self.names[section]
            return f'{name} [{_axis_unit(name)}]'
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            return f'{self.points[index.row(), index.column()]:.12g}'
        return None


class BraggExplorerDialog(QDialog):
    def __init__(self, result, detector_getter, parent=None, *, detector_visible=False):
        super().__init__(parent)
        self.browser = BraggExplorer(result)
        self.detector_getter = detector_getter
        self._export_running = False
        self.setWindowTitle('Explore Bragg Conditions')
        self.resize(900, 680)
        layout = QVBoxLayout(self)
        hkl_controls = QHBoxLayout()
        hkl_controls.addWidget(QLabel('HKL:'))
        self.hkl_combo = QComboBox()
        for hkl in self.browser.hkls():
            self.hkl_combo.addItem(hkl_label(hkl), hkl)
        hkl_controls.addWidget(self.hkl_combo, 1)
        layout.addLayout(hkl_controls)
        view_controls = QHBoxLayout()
        view_controls.addWidget(QLabel('Points shown:'))
        self.view_combo = QComboBox()
        self.view_combo.addItem('All saved Bragg points', False)
        self.view_combo.addItem('Visible on current detector', True)
        self.view_combo.setCurrentIndex(1 if detector_visible else 0)
        view_controls.addWidget(self.view_combo)
        view_controls.addStretch()
        layout.addLayout(view_controls)

        self.order_label = QLabel('Application order: ' + self.browser.application_order())
        layout.addWidget(self.order_label)
        if result.metadata.get('require_detector_hit') and not result.metadata.get('points_are_complete'):
            warning = QLabel('This older mapping was filtered during calculation; missing Bragg points cannot be recovered.')
            warning.setWordWrap(True)
            layout.addWidget(warning)
        self.summary_label = QLabel()
        layout.addWidget(self.summary_label)
        layout.addWidget(QLabel('Slider positions with this reflection:'))
        self.positions_model = _PositionsModel(self.browser, self)
        self.positions_table = QTableView()
        self.positions_table.setModel(self.positions_model)
        self.positions_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.positions_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.positions_table.setMinimumHeight(200)
        layout.addWidget(self.positions_table, 1)
        self.points_label = QLabel('Select a slider position to inspect its plotted-axis points.')
        layout.addWidget(self.points_label)
        self.points_model = _PointsModel(self.browser.plot_axes, self)
        self.points_table = QTableView()
        self.points_table.setModel(self.points_model)
        self.points_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.points_table, 2)

        actions = QHBoxLayout()
        actions.addWidget(QLabel('Export:'))
        self.export_scope = QComboBox()
        self.export_scope.addItem('Selected position', 'selected')
        self.export_scope.addItem('All matching positions', 'all')
        actions.addWidget(self.export_scope)
        actions.addStretch()
        self.export_button = QPushButton('Export CSV…')
        self.export_button.clicked.connect(self._export)
        actions.addWidget(self.export_button)
        layout.addLayout(actions)

        self.hkl_combo.currentIndexChanged.connect(lambda _index: self._select_hkl(self.hkl_combo.currentData()))
        self.view_combo.currentIndexChanged.connect(self._show_position)
        self.positions_table.selectionModel().currentRowChanged.connect(self._show_position)
        self._select_hkl(self.hkl_combo.currentData())

    def _detector(self):
        return deepcopy(self.detector_getter()) if self.view_combo.currentData() else None

    def _select_hkl(self, hkl):
        self.positions_model.set_hkl(hkl)
        self.summary_label.setText(
            f'{len(self.positions_model.positions)} slider position(s) with saved Bragg points.')
        self.points_model.set_points(np.empty((0, 2)))
        if self.positions_model.positions:
            self.positions_table.selectRow(0)
        else:
            self.points_label.setText('No Bragg conditions for this HKL.')
        self.export_button.setEnabled(bool(self.positions_model.positions))

    def _show_position(self, *_):
        row = self.positions_table.currentIndex().row()
        if row < 0 or row >= len(self.positions_model.positions):
            return
        try:
            hkl = self.hkl_combo.currentData()
            index = self.positions_model.positions[row]
            points = self.browser.view(index, hkl, self._detector())
            self.points_model.set_points(points)
            saved = self.positions_model.counts[row]
            if self.view_combo.currentData():
                self.points_label.setText(
                    f'{len(points):,} detector-visible point(s) of {saved:,} saved Bragg points. '
                    'Switch to “All saved Bragg points” to inspect the complete result.'
                    if not len(points) else
                    f'{len(points):,} detector-visible point(s) of {saved:,} saved Bragg points.')
            else:
                self.points_label.setText(f'{len(points):,} saved Bragg point(s) at this position.')
        except Exception as exc:
            self.points_model.set_points(np.empty((0, 2)))
            self.points_label.setText('Could not prepare the selected view.')
            QMessageBox.critical(self, 'Explore Bragg Conditions', str(exc))

    def _export(self):
        hkl = self.hkl_combo.currentData()
        row = self.positions_table.currentIndex().row()
        if self.export_scope.currentData() == 'selected':
            if row < 0:
                return
            positions = [self.positions_model.positions[row]]
        else:
            positions = list(self.positions_model.positions)
        path, _ = QFileDialog.getSaveFileName(self, 'Export Bragg conditions',
                                               'bragg_conditions.csv', 'CSV files (*.csv)')
        if not path:
            return
        try:
            detector = self._detector()
        except Exception as exc:
            QMessageBox.critical(self, 'Export Bragg conditions', str(exc))
            return
        from ....analysis.gui.widgets.task_output_dialog import run_task_with_output_dialog
        cancel = Event()

        def work():
            try:
                return self.browser.export_csv(path, hkl, positions, detector, cancel)
            except InterruptedError:
                return {'cancelled': True}

        def completed(count):
            if not isinstance(count, dict):
                QMessageBox.information(self, 'Export Bragg conditions',
                                        f'Exported {count:,} point(s).')

        def failed(traceback_text):
            QMessageBox.critical(self, 'Export Bragg conditions',
                                 traceback_text.strip().splitlines()[-1])

        self._export_running = True
        self.export_button.setEnabled(False)
        task = run_task_with_output_dialog(self, 'Export Bragg conditions', work,
                                           on_success=completed, on_error=failed,
                                           cancel_callback=cancel.set)

        def finished():
            self._export_running = False
            self.export_button.setEnabled(bool(self.positions_model.positions))

        task._thread.finished.connect(finished)

    def closeEvent(self, event):
        if self._export_running:
            event.ignore()
            return
        super().closeEvent(event)
