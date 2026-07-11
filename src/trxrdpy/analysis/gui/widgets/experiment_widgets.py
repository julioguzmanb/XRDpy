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
    """Legacy-compatible experiment metadata widget.

    It creates:
    - Experiment Metadata group
    - optional ID09-specific Metadata group

    The labels, defaults, validators, and placeholders match the legacy GUI.

    Attributes
    ----------
    title : str
        Group-box title displayed above common experiment fields.
    defaults : dict
        Initial text values keyed by metadata field name.
    include_id09 : bool
        Whether dataset, scan, and raw-sample controls are created.
    fields : dict
        Mapping from stable metadata keys to editable widgets.
    labels : dict
        Mapping from metadata keys to their visible ``QLabel`` objects.
    group : QGroupBox
        Common experiment-metadata container.
    id09_group : QGroupBox or None
        ID09-specific metadata container when enabled.
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
        """Initialize ``ExperimentMetadataWidget``, bind shared state and services, and create its controls."""
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
        """Create and connect the controls for experiment group."""
        self.group = QGroupBox(self.title)
        grid = QGridLayout()
        self.group.setLayout(grid)
        layout.addWidget(self.group)

        row = 0

        sample = self._line("sample_name", "DET70")
        sample_label = QLabel("Sample name:")
        self.labels["sample_name"] = sample_label
        self.fields["sample_name"] = sample
        grid.addWidget(sample_label, row, 0)
        grid.addWidget(sample, row, 1, 1, 3)
        row += 1

        temperature = self._line("temperature_K", "110")
        temperature.setValidator(QDoubleValidator())
        temperature_label = QLabel("Temperature [K]:")
        self.labels["temperature_K"] = temperature_label
        self.fields["temperature_K"] = temperature
        grid.addWidget(temperature_label, row, 0)
        grid.addWidget(temperature, row, 1)

        excitation = self._line("excitation_wl_nm", "1500")
        excitation.setValidator(QDoubleValidator())
        excitation_label = QLabel("Excitation wavelength [nm]:")
        self.labels["excitation_wl_nm"] = excitation_label
        self.fields["excitation_wl_nm"] = excitation
        grid.addWidget(excitation_label, row, 2)
        grid.addWidget(excitation, row, 3)
        row += 1

        fluence = self._line("fluence_mJ_cm2", "25")
        fluence.setValidator(QDoubleValidator())
        fluence_label = QLabel("Fluence [mJ/cm²]:")
        self.labels["fluence_mJ_cm2"] = fluence_label
        self.fields["fluence_mJ_cm2"] = fluence
        grid.addWidget(fluence_label, row, 0)
        grid.addWidget(fluence, row, 1)

        time_window = self._line("time_window_fs", "250")
        time_window.setValidator(QDoubleValidator())
        time_window_label = QLabel("Time window [fs]:")
        self.labels["time_window_fs"] = time_window_label
        self.fields["time_window_fs"] = time_window
        grid.addWidget(time_window_label, row, 2)
        grid.addWidget(time_window, row, 3)

    def _init_id09_group(self, layout: QVBoxLayout):
        """Create and connect the controls for ID09 group."""
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
        """Return the line-edit widget registered for a named field."""
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
        """Create a labeled line edit, register it, and add it to the layout."""
        label_widget = QLabel(label)
        grid.addWidget(label_widget, row, 0)
        grid.addWidget(widget, row, 1)

        self.labels[field_name] = label_widget
        self.fields[field_name] = widget

        return row + 1

    def value(self, field_name: str) -> str:
        """Return the current value of one named metadata field."""
        return self.fields[field_name].text().strip()

    def values(self) -> dict:
        """Return current experiment metadata as a stable string-valued mapping."""
        return {
            field_name: widget.text().strip()
            for field_name, widget in self.fields.items()
        }

    def set_value(self, field_name: str, value):
        """Set value.

        Parameters
        ----------
        field_name : str
            Metadata field whose visibility or value is being changed.
        value : object
            Value to validate, convert, or display.
        """
        self.fields[field_name].setText("" if value is None else str(value))

    def set_values(self, values: dict):
        """Populate known experiment metadata fields while ignoring unknown keys."""
        for field_name, value in values.items():
            if field_name in self.fields:
                self.set_value(field_name, value)
    
    def set_field_visible(self, field_name: str, visible: bool):
        """Show or hide one metadata field and its label.

        Unknown field names are ignored so facility-specific callers can apply
        visibility rules to widgets with different field sets.
        """

        if field_name in self.labels:
            self.labels[field_name].setVisible(visible)

        if field_name in self.fields:
            self.fields[field_name].setVisible(visible)

    def set_id09_visible(self, visible: bool):
        """Show or hide the complete ID09-specific metadata control group."""

        if self.id09_group is not None:
            self.id09_group.setVisible(visible)


class CalibrationContextWidget(QWidget):
    """Legacy-compatible calibration context widget.

    This reproduces the legacy _build_calibration_experiment_group layout.

    Attributes
    ----------
    title : str
        Visible calibration-context group title.
    defaults : dict
        Initial values for sample, temperature, and scan fields.
    fields : dict
        Mapping from stable context keys to editable widgets.
    group : QGroupBox
        Container holding the calibration context controls.
    """

    field_changed = pyqtSignal(str)

    def __init__(
        self,
        *,
        title: str = "Calibration Context",
        defaults: Optional[dict] = None,
        parent=None,
    ):
        """Initialize ``CalibrationContextWidget``, bind shared state and services, and create its controls."""
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

        sample = self._line("sample_name", "DET70")
        sample_label = QLabel("Sample name:")
        self.fields["sample_name"] = sample
        grid.addWidget(sample_label, row, 0)
        grid.addWidget(sample, row, 1, 1, 3)
        row += 1

        temperature = self._line("temperature_K", "110")
        temperature.setValidator(QDoubleValidator())
        temperature_label = QLabel("Temperature [K]:")
        self.fields["temperature_K"] = temperature
        grid.addWidget(temperature_label, row, 0)
        grid.addWidget(temperature, row, 1)

        scan_spec = self._line("scan_spec", "[7]")
        scan_spec.setPlaceholderText(
            "Examples: [7], [1466556], 181661, 'scan_181661'"
        )
        scan_label = QLabel("scan / scan_spec:")
        self.fields["scan_spec"] = scan_spec
        grid.addWidget(scan_label, row, 2)
        grid.addWidget(scan_spec, row, 3)

    def _line(self, field_name: str, default: str) -> QLineEdit:
        """Return the line-edit widget registered for a named field."""
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
        """Create a labeled line edit, register it, and add it to the layout."""
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(widget, row, 1)
        self.fields[field_name] = widget
        return row + 1

    def value(self, field_name: str) -> str:
        """Return the current value of one named calibration field."""
        return self.fields[field_name].text().strip()

    def values(self) -> dict:
        """Return current calibration context as a stable string-valued mapping."""
        return {
            field_name: widget.text().strip()
            for field_name, widget in self.fields.items()
        }

    def set_value(self, field_name: str, value):
        """Set value.

        Parameters
        ----------
        field_name : str
            Metadata field whose visibility or value is being changed.
        value : object
            Value to validate, convert, or display.
        """
        self.fields[field_name].setText("" if value is None else str(value))

    def set_values(self, values: dict):
        """Populate known calibration context fields while ignoring unknown keys."""
        for field_name, value in values.items():
            if field_name in self.fields:
                self.set_value(field_name, value)
