"""
Reusable experiment metadata widgets for the analysis GUI.

This reproduces the legacy experiment metadata layout created by
MainWindow._build_experiment_group.
"""
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)


class ExperimentMetadataWidget(QWidget):
    """
    Legacy-compatible experiment metadata widget.

    It creates:
    - Experiment Metadata group
    - optional ID09-specific Metadata group

    The labels, defaults, validators, and placeholders match the legacy GUI.
    """

    field_changed = pyqtSignal(str)

    def __init__(
        self,
        *,
        title: str = "Experiment Metadata",
        defaults: Optional[dict] = None,
        include_id09: bool = False,
        parent=None,
    ):
        super().__init__(parent)

        self.title = title
        self.defaults = defaults or {}
        self.include_id09 = include_id09
        self.fields = {}
        self.labels = {}

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._init_experiment_group(layout)

        if self.include_id09:
            self._init_id09_group(layout)
        else:
            self.id09_group = None

    def _init_experiment_group(self, layout: QVBoxLayout):
        self.group = QGroupBox(self.title)
        grid = QGridLayout()
        self.group.setLayout(grid)
        layout.addWidget(self.group)

        row = 0

        row = self._add_labeled_line(
            grid,
            row,
            "Sample name:",
            self._line("sample_name", "DET70"),
            "sample_name",
        )

        temperature = self._line("temperature_K", "110")
        temperature.setValidator(QDoubleValidator())
        row = self._add_labeled_line(
            grid,
            row,
            "Temperature [K]:",
            temperature,
            "temperature_K",
        )

        excitation = self._line("excitation_wl_nm", "1500")
        excitation.setValidator(QDoubleValidator())
        row = self._add_labeled_line(
            grid,
            row,
            "Excitation wavelength [nm]:",
            excitation,
            "excitation_wl_nm",
        )

        fluence = self._line("fluence_mJ_cm2", "25")
        fluence.setValidator(QDoubleValidator())
        row = self._add_labeled_line(
            grid,
            row,
            "Fluence [mJ/cm²]:",
            fluence,
            "fluence_mJ_cm2",
        )

        time_window = self._line("time_window_fs", "250")
        time_window.setValidator(QDoubleValidator())
        self._add_labeled_line(
            grid,
            row,
            "Time window [fs]:",
            time_window,
            "time_window_fs",
        )

    def _init_id09_group(self, layout: QVBoxLayout):
        self.id09_group = QGroupBox("ID09-specific Metadata")
        grid = QGridLayout()
        self.id09_group.setLayout(grid)
        layout.addWidget(self.id09_group)

        raw_sample_name_label = QLabel("Raw sample name (optional):")
        self.labels["raw_sample_name"] = raw_sample_name_label
        grid.addWidget(raw_sample_name_label, 0, 0)

        raw_sample_name = self._line("raw_sample_name", "")
        raw_sample_name.setPlaceholderText(
            "Optional. Used only to find the raw ID09 HDF5; defaults to sample_name."
        )
        grid.addWidget(raw_sample_name, 0, 1)
        self.fields["raw_sample_name"] = raw_sample_name

        dataset_label = QLabel("Dataset:")
        self.labels["dataset"] = dataset_label
        grid.addWidget(dataset_label, 1, 0)

        dataset = self._line("dataset", "3")
        dataset.setValidator(QIntValidator())
        grid.addWidget(dataset, 1, 1)
        self.fields["dataset"] = dataset

        scan_nb_label = QLabel("Scan number:")
        self.labels["scan_nb"] = scan_nb_label
        grid.addWidget(scan_nb_label, 2, 0)

        scan_nb = self._line("scan_nb", "7")
        scan_nb.setValidator(QIntValidator())
        grid.addWidget(scan_nb, 2, 1)
        self.fields["scan_nb"] = scan_nb

    def _line(self, field_name: str, default: str) -> QLineEdit:
        widget = QLineEdit(str(self.defaults.get(field_name, default)))
        widget.textChanged.connect(
            lambda _text, name=field_name: self.field_changed.emit(name)
        )
        return widget

    def _add_labeled_line(
        self,
        grid: QGridLayout,
        row: int,
        label: str,
        widget: QLineEdit,
        field_name: str,
    ) -> int:
        label_widget = QLabel(label)
        grid.addWidget(label_widget, row, 0)
        grid.addWidget(widget, row, 1)

        self.labels[field_name] = label_widget
        self.fields[field_name] = widget

        return row + 1

    def value(self, field_name: str) -> str:
        return self.fields[field_name].text().strip()

    def values(self) -> dict:
        return {
            field_name: widget.text().strip()
            for field_name, widget in self.fields.items()
        }

    def set_value(self, field_name: str, value):
        self.fields[field_name].setText("" if value is None else str(value))

    def set_values(self, values: dict):
        for field_name, value in values.items():
            if field_name in self.fields:
                self.set_value(field_name, value)
    
    def set_field_visible(self, field_name: str, visible: bool):
        """
        Show or hide a metadata row.
        """

        if field_name in self.labels:
            self.labels[field_name].setVisible(visible)

        if field_name in self.fields:
            self.fields[field_name].setVisible(visible)

    def set_id09_visible(self, visible: bool):
        """
        Show or hide the ID09-specific metadata group.
        """

        if self.id09_group is not None:
            self.id09_group.setVisible(visible)


class CalibrationContextWidget(QWidget):
    """
    Legacy-compatible calibration context widget.

    This reproduces the legacy _build_calibration_experiment_group layout.
    """

    field_changed = pyqtSignal(str)

    def __init__(
        self,
        *,
        title: str = "Calibration Context",
        defaults: Optional[dict] = None,
        parent=None,
    ):
        super().__init__(parent)

        self.title = title
        self.defaults = defaults or {}
        self.fields = {}

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.group = QGroupBox(self.title)
        grid = QGridLayout()
        self.group.setLayout(grid)
        layout.addWidget(self.group)

        row = 0

        row = self._add_labeled_line(
            grid,
            row,
            "Sample name:",
            self._line("sample_name", "DET70"),
            "sample_name",
        )

        temperature = self._line("temperature_K", "110")
        temperature.setValidator(QDoubleValidator())
        row = self._add_labeled_line(
            grid,
            row,
            "Temperature [K]:",
            temperature,
            "temperature_K",
        )

        scan_spec = self._line("scan_spec", "[7]")
        scan_spec.setPlaceholderText(
            "Examples: [7], [1466556], 181661, 'scan_181661'"
        )
        self._add_labeled_line(
            grid,
            row,
            "scan / scan_spec:",
            scan_spec,
            "scan_spec",
        )

    def _line(self, field_name: str, default: str) -> QLineEdit:
        widget = QLineEdit(str(self.defaults.get(field_name, default)))
        widget.textChanged.connect(
            lambda _text, name=field_name: self.field_changed.emit(name)
        )
        return widget

    def _add_labeled_line(
        self,
        grid: QGridLayout,
        row: int,
        label: str,
        widget: QLineEdit,
        field_name: str,
    ) -> int:
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(widget, row, 1)
        self.fields[field_name] = widget
        return row + 1

    def value(self, field_name: str) -> str:
        return self.fields[field_name].text().strip()

    def values(self) -> dict:
        return {
            field_name: widget.text().strip()
            for field_name, widget in self.fields.items()
        }

    def set_value(self, field_name: str, value):
        self.fields[field_name].setText("" if value is None else str(value))

    def set_values(self, values: dict):
        for field_name, value in values.items():
            if field_name in self.fields:
                self.set_value(field_name, value)
