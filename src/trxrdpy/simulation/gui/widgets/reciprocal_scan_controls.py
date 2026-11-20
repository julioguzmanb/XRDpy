"""Editable parameter ranges for optional reciprocal-space scans."""
from PyQt5.QtCore import pyqtSignal, QTimer, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QComboBox, QPushButton, QHeaderView, QStyledItemDelegate, QStyleOption, QStyle
from ...reciprocal_scan import parameter_label


class ParameterDelegate(QStyledItemDelegate):
    def __init__(self, controls):
        super().__init__(controls.table)
        self.controls = controls

    def createEditor(self, parent, option, index):
        editor = QComboBox(parent)
        editor.setStyleSheet('QComboBox { padding: 0 6px; border: 0; border-radius: 0; min-height: 0; }')
        for name in dict.fromkeys(self.controls.parameters + [index.data()]):
            editor.addItem(parameter_label(name), name)
        editor.activated.connect(lambda *_: self.commitData.emit(editor))
        editor.activated.connect(lambda *_: self.closeEditor.emit(editor))
        QTimer.singleShot(0, editor.showPopup)
        return editor

    def setEditorData(self, editor, index):
        editor.setCurrentIndex(editor.findData(index.data()))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentData(), Qt.EditRole)

    def displayText(self, value, locale):
        return parameter_label(str(value))

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        arrow = QStyleOption()
        arrow.rect = option.rect.adjusted(option.rect.width()-20, 8, -6, -8)
        arrow.palette = option.palette
        arrow.state = QStyle.State_Enabled
        option.widget.style().drawPrimitive(QStyle.PE_IndicatorArrowDown, arrow, painter)


class ReciprocalScanControls(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parameters = ['energy', 'sam_rotx', 'sam_roty', 'sam_rotz', 'det_rotx', 'det_roty', 'det_rotz', 'e_bandwidth']
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Parameter', 'Start', 'Stop', 'Step'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.setAlternatingRowColors(True)
        self.table.setItemDelegateForColumn(0, ParameterDelegate(self))
        self.table.clicked.connect(lambda index: self.table.edit(index) if index.column()==0 else None)
        self.table.setMinimumHeight(140)
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        add, remove = QPushButton('Add parameter'), QPushButton('Remove selected')
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch()
        layout.addLayout(buttons)
        add.clicked.connect(lambda: self.add_row())
        remove.clicked.connect(self.remove_row)
        self.table.itemChanged.connect(lambda *_: self.changed.emit())

    def add_row(self, values=None):
        used = {row[0] for row in self.rows()}
        values = values or [next((name for name in self.parameters if name not in used), self.parameters[0]), '', '', '']
        self.table.blockSignals(True)
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in enumerate(values):
            self.table.setItem(row, column, QTableWidgetItem(str(value)))
        self.table.blockSignals(False)
        self.changed.emit()

    def remove_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self.changed.emit()

    def rows(self):
        return [[self.table.item(row, col).text() for col in range(4)]
                for row in range(self.table.rowCount())]

    def set_rows(self, rows):
        self.blockSignals(True)
        self.table.setRowCount(0)
        for row in rows:
            self.add_row(row)
        self.blockSignals(False)

    def set_parameters(self, parameters):
        if parameters == self.parameters:
            return
        rows = self.rows()
        self.parameters = parameters
        self.set_rows(rows)

    def ranges(self):
        rows = self.rows()
        names = [row[0] for row in rows]
        if len(set(names)) != len(names):
            raise ValueError('Each scan parameter must appear only once.')
        if any(not value.strip() for row in rows for value in row[1:]):
            raise ValueError('Enter Start, Stop and Step for every scan parameter.')
        return {row[0]: tuple(float(value) for value in row[1:]) for row in rows}
