"""
Multi-experiment editor widgets for the analysis GUI.

These classes are copied from the legacy analysis GUI and isolated here so that
Differential and Fitting tabs can reuse them without depending on MainWindow.
"""
from __future__ import annotations

from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QFileDialog,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from trxrdpy.analysis.gui.services import PathService
from trxrdpy.analysis.gui.widgets.path_widgets import DropPathLineEdit
from trxrdpy.analysis.gui.utils import (
    parse_float_like,
    parse_int_like,
    parse_optional_float_like,
    parse_python_literal,
    parse_ref_value,
    pretty_literal,
)


class ExperimentLeafWidget(QFrame):
    """Edit one delay-series experiment entry used by comparison plots.

    Attributes
    ----------
    path_service : PathService
        Supplies CSV file-dialog locations.
    remove_callback : callable or None
        Callback invoked when the entry's remove button is pressed.
    sample_name, temperature, excitation, fluence, time_window : QWidget
        Core delay-experiment metadata editors.
    phi_mode, ref_type, ref_value : QWidget
        Azimuthal grouping and reference controls.
    delay_offset_ps, delay_for_norm_max : QWidget
        Plot alignment and optional normalization-delay controls.
    csv_path : DropPathLineEdit
        Optional explicit fitting-table path.
    label_edit : QLineEdit or None
        Optional user-facing legend-label editor.
    """
    def __init__(self, *, title="Experiment", show_label=True, show_remove=True, remove_callback=None, data=None):
        """Initialize ``ExperimentLeafWidget``, bind shared state and services, and create its controls."""
        super().__init__()
        self.path_service = PathService()
        self.remove_callback = remove_callback
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { border: 1px solid #d9d9d9; border-radius: 4px; }")

        outer = QVBoxLayout()
        self.setLayout(outer)

        header = QHBoxLayout()
        outer.addLayout(header)
        header.addWidget(QLabel(f"<b>{title}</b>"))
        header.addStretch()
        if show_remove:
            btn = QPushButton("Remove")
            btn.clicked.connect(self._remove_me)
            header.addWidget(btn)

        grid = QGridLayout()
        outer.addLayout(grid)

        self.show_label = show_label
        row = 0
        if show_label:
            grid.addWidget(QLabel("label:"), row, 0)
            self.label_edit = QLineEdit("")
            grid.addWidget(self.label_edit, row, 1)
            row += 1
        else:
            self.label_edit = None

        grid.addWidget(QLabel("Sample name:"), row, 0)
        self.sample_name = QLineEdit("DET70")
        grid.addWidget(self.sample_name, row, 1)
        row += 1

        grid.addWidget(QLabel("Temperature [K]:"), row, 0)
        self.temperature = QLineEdit("110")
        self.temperature.setValidator(QDoubleValidator())
        grid.addWidget(self.temperature, row, 1)
        row += 1

        grid.addWidget(QLabel("Excitation wavelength [nm]:"), row, 0)
        self.excitation = QLineEdit("1500")
        self.excitation.setValidator(QDoubleValidator())
        grid.addWidget(self.excitation, row, 1)
        row += 1

        grid.addWidget(QLabel("Fluence [mJ/cm²]:"), row, 0)
        self.fluence = QLineEdit("25")
        self.fluence.setValidator(QDoubleValidator())
        grid.addWidget(self.fluence, row, 1)
        row += 1

        grid.addWidget(QLabel("Time window [fs]:"), row, 0)
        self.time_window = QLineEdit("250")
        self.time_window.setValidator(QDoubleValidator())
        grid.addWidget(self.time_window, row, 1)
        row += 1

        grid.addWidget(QLabel("Azimuthal mode:"), row, 0)
        self.phi_mode = QComboBox()
        self.phi_mode.addItems(["phi_avg", "separate_phi"])
        grid.addWidget(self.phi_mode, row, 1)
        row += 1

        grid.addWidget(QLabel("Delay offset [ps]:"), row, 0)
        self.delay_offset_ps = QLineEdit("")
        self.delay_offset_ps.setValidator(QDoubleValidator())
        grid.addWidget(self.delay_offset_ps, row, 1)
        row += 1

        grid.addWidget(QLabel("Reference type:"), row, 0)
        self.ref_type = QComboBox()
        self.ref_type.addItems(["dark", "delay"])
        grid.addWidget(self.ref_type, row, 1)
        row += 1

        grid.addWidget(QLabel("Reference value:"), row, 0)
        self.ref_value = QLineEdit("")
        self.ref_value.setPlaceholderText("Example: [1466556], -95000, '-5ns'")
        grid.addWidget(self.ref_value, row, 1)
        row += 1

        grid.addWidget(QLabel("Delay for normalization max:"), row, 0)
        self.delay_for_norm_max = QLineEdit("")
        self.delay_for_norm_max.setValidator(QDoubleValidator())
        grid.addWidget(self.delay_for_norm_max, row, 1)
        row += 1

        grid.addWidget(QLabel("csv_path (optional):"), row, 0)
        csv_box = QHBoxLayout()
        self.csv_path = DropPathLineEdit("", mode="file")
        csv_box.addWidget(self.csv_path)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_csv)
        csv_box.addWidget(browse)
        grid.addLayout(csv_box, row, 1)

        if data:
            self.set_data(data)

    def _browse_csv(self):
        """Open a CSV chooser and populate the associated path field."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV file",
            str(
                self.path_service.dialog_start_path(
                    current=self.csv_path.text()
                )
            ),
            "CSV Files (*.csv);;All Files (*)",
        )
        if file_name:
            self.path_service.remember_dialog_selection(file_name)
            self.csv_path.setText(file_name)

    def _remove_me(self):
        """Request removal of this experiment entry from its parent editor."""
        if callable(self.remove_callback):
            self.remove_callback(self)

    def set_data(self, data):
        """Set data.

        Parameters
        ----------
        data : object
            Serialized experiment or widget data used to populate the fields.
        """
        if self.label_edit is not None:
            self.label_edit.setText(str(data.get("label", "")))
        self.sample_name.setText(str(data.get("sample_name", "")))
        if "temperature_K" in data:
            self.temperature.setText(str(data.get("temperature_K", "")))
        if "excitation_wl_nm" in data:
            self.excitation.setText(str(data.get("excitation_wl_nm", "")))
        if "fluence_mJ_cm2" in data:
            self.fluence.setText(str(data.get("fluence_mJ_cm2", "")))
        if "time_window_fs" in data:
            self.time_window.setText(str(data.get("time_window_fs", "")))
        phi_mode = data.get("phi_mode", "phi_avg")
        idx = self.phi_mode.findText(str(phi_mode))
        if idx >= 0:
            self.phi_mode.setCurrentIndex(idx)
        if "delay_offset_ps" in data:
            self.delay_offset_ps.setText(str(data.get("delay_offset_ps", "")))
        ref_type = data.get("ref_type", "dark")
        idx = self.ref_type.findText(str(ref_type))
        if idx >= 0:
            self.ref_type.setCurrentIndex(idx)
        if "ref_value" in data:
            self.ref_value.setText(pretty_literal(data.get("ref_value")))
        if "delay_for_norm_max" in data:
            self.delay_for_norm_max.setText(str(data.get("delay_for_norm_max", "")))
        if "csv_path" in data:
            self.csv_path.setText(str(data.get("csv_path", "")))

    def to_dict(self):
        """Convert the current value to dict.

        Returns
        -------
        dict
            Serializable experiment definition containing the current widget values.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        out = {}
        if self.label_edit is not None:
            label = self.label_edit.text().strip()
            if label:
                out["label"] = label

        sample_name = self.sample_name.text().strip()
        if not sample_name:
            raise ValueError("sample_name cannot be empty.")
        out["sample_name"] = sample_name
        out["temperature_K"] = parse_int_like(self.temperature.text(), name="temperature_K")
        out["excitation_wl_nm"] = parse_float_like(self.excitation.text(), name="excitation_wl_nm")
        out["fluence_mJ_cm2"] = parse_float_like(self.fluence.text(), name="fluence_mJ_cm2")
        out["time_window_fs"] = parse_int_like(self.time_window.text(), name="time_window_fs")
        out["phi_mode"] = self.phi_mode.currentText()

        delay_offset = parse_optional_float_like(self.delay_offset_ps.text())
        if delay_offset is not None:
            out["delay_offset_ps"] = delay_offset

        out["ref_type"] = self.ref_type.currentText()
        ref_value_text = self.ref_value.text().strip()
        if ref_value_text:
            out["ref_value"] = parse_ref_value(ref_value_text)

        delay_for_norm_max = parse_optional_float_like(self.delay_for_norm_max.text())
        if delay_for_norm_max is not None:
            out["delay_for_norm_max"] = delay_for_norm_max

        csv_path = self.csv_path.text().strip()
        if csv_path:
            out["csv_path"] = csv_path

        return out


class ExperimentFluenceLeafWidget(QFrame):
    """Edit one fluence-series experiment entry used by comparison plots.

    Attributes
    ----------
    path_service : PathService
        Supplies CSV file-dialog locations.
    remove_callback : callable or None
        Callback invoked when the entry's remove button is pressed.
    sample_name, temperature, excitation, delay_fs, time_window : QWidget
        Core fixed-delay fluence-experiment metadata editors.
    phi_mode, ref_type, ref_value : QWidget
        Azimuthal grouping and reference controls.
    fluence_scale, fluence_offset : QWidget
        Horizontal plot scale and offset applied to displayed fluence values.
    delay_offset_fs : QWidget
        Display-only fixed-delay offset in femtoseconds.
    csv_path : DropPathLineEdit
        Optional explicit fitting-table path.
    label_edit : QLineEdit or None
        Optional user-facing legend-label editor.
    """
    def __init__(self, *, title="Experiment", show_label=True, show_remove=True, remove_callback=None, data=None):
        """Initialize ``ExperimentFluenceLeafWidget``, bind shared state and services, and create its controls."""
        super().__init__()
        self.path_service = PathService()
        self.remove_callback = remove_callback
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { border: 1px solid #d9d9d9; border-radius: 4px; }")

        outer = QVBoxLayout()
        self.setLayout(outer)

        header = QHBoxLayout()
        outer.addLayout(header)
        header.addWidget(QLabel(f"<b>{title}</b>"))
        header.addStretch()
        if show_remove:
            btn = QPushButton("Remove")
            btn.clicked.connect(self._remove_me)
            header.addWidget(btn)

        grid = QGridLayout()
        outer.addLayout(grid)

        self.show_label = show_label
        row = 0
        if show_label:
            grid.addWidget(QLabel("label:"), row, 0)
            self.label_edit = QLineEdit("")
            grid.addWidget(self.label_edit, row, 1)
            row += 1
        else:
            self.label_edit = None

        grid.addWidget(QLabel("Sample name:"), row, 0)
        self.sample_name = QLineEdit("DET70")
        grid.addWidget(self.sample_name, row, 1)
        row += 1

        grid.addWidget(QLabel("Temperature [K]:"), row, 0)
        self.temperature = QLineEdit("110")
        self.temperature.setValidator(QDoubleValidator())
        grid.addWidget(self.temperature, row, 1)
        row += 1

        grid.addWidget(QLabel("Excitation wavelength [nm]:"), row, 0)
        self.excitation = QLineEdit("1500")
        self.excitation.setValidator(QDoubleValidator())
        grid.addWidget(self.excitation, row, 1)
        row += 1

        grid.addWidget(QLabel("Delay [fs]:"), row, 0)
        self.delay_fs = QLineEdit("0")
        self.delay_fs.setValidator(QDoubleValidator())
        grid.addWidget(self.delay_fs, row, 1)
        row += 1

        grid.addWidget(QLabel("Time window [fs]:"), row, 0)
        self.time_window = QLineEdit("250")
        self.time_window.setValidator(QDoubleValidator())
        grid.addWidget(self.time_window, row, 1)
        row += 1

        grid.addWidget(QLabel("Azimuthal mode:"), row, 0)
        self.phi_mode = QComboBox()
        self.phi_mode.addItems(["phi_avg", "separate_phi"])
        grid.addWidget(self.phi_mode, row, 1)
        row += 1

        grid.addWidget(QLabel("Reference type:"), row, 0)
        self.ref_type = QComboBox()
        self.ref_type.addItems(["dark", "fluence"])
        grid.addWidget(self.ref_type, row, 1)
        row += 1

        grid.addWidget(QLabel("Reference value:"), row, 0)
        self.ref_value = QLineEdit("")
        self.ref_value.setPlaceholderText("Example: [1466556], 1.5, '1.5'")
        grid.addWidget(self.ref_value, row, 1)
        row += 1

        grid.addWidget(QLabel("Fluence scale:"), row, 0)
        self.fluence_scale = QLineEdit("")
        self.fluence_scale.setValidator(QDoubleValidator())
        grid.addWidget(self.fluence_scale, row, 1)
        row += 1

        grid.addWidget(QLabel("Fluence offset:"), row, 0)
        self.fluence_offset = QLineEdit("")
        self.fluence_offset.setValidator(QDoubleValidator())
        grid.addWidget(self.fluence_offset, row, 1)
        row += 1

        grid.addWidget(QLabel("Delay offset [fs]:"), row, 0)
        self.delay_offset_fs = QLineEdit("")
        self.delay_offset_fs.setValidator(QDoubleValidator())
        grid.addWidget(self.delay_offset_fs, row, 1)
        row += 1

        grid.addWidget(QLabel("Delay for normalization max:"), row, 0)
        self.delay_for_norm_max = QLineEdit("")
        self.delay_for_norm_max.setValidator(QDoubleValidator())
        grid.addWidget(self.delay_for_norm_max, row, 1)
        row += 1

        grid.addWidget(QLabel("csv_path (optional):"), row, 0)
        csv_box = QHBoxLayout()
        self.csv_path = DropPathLineEdit("", mode="file")
        csv_box.addWidget(self.csv_path)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_csv)
        csv_box.addWidget(browse)
        grid.addLayout(csv_box, row, 1)

        if data:
            self.set_data(data)

    def _browse_csv(self):
        """Open a CSV chooser and populate the associated path field."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV file",
            str(
                self.path_service.dialog_start_path(
                    current=self.csv_path.text()
                )
            ),
            "CSV Files (*.csv);;All Files (*)",
        )
        if file_name:
            self.path_service.remember_dialog_selection(file_name)
            self.csv_path.setText(file_name)

    def _remove_me(self):
        """Request removal of this experiment entry from its parent editor."""
        if callable(self.remove_callback):
            self.remove_callback(self)

    def set_data(self, data):
        """Set data.

        Parameters
        ----------
        data : object
            Serialized experiment or widget data used to populate the fields.
        """
        if self.label_edit is not None:
            self.label_edit.setText(str(data.get("label", "")))
        self.sample_name.setText(str(data.get("sample_name", "")))
        if "temperature_K" in data:
            self.temperature.setText(str(data.get("temperature_K", "")))
        if "excitation_wl_nm" in data:
            self.excitation.setText(str(data.get("excitation_wl_nm", "")))
        if "delay_fs" in data:
            self.delay_fs.setText(str(data.get("delay_fs", "")))
        if "time_window_fs" in data:
            self.time_window.setText(str(data.get("time_window_fs", "")))
        phi_mode = data.get("phi_mode", "phi_avg")
        idx = self.phi_mode.findText(str(phi_mode))
        if idx >= 0:
            self.phi_mode.setCurrentIndex(idx)
        ref_type = data.get("ref_type", "dark")
        idx = self.ref_type.findText(str(ref_type))
        if idx >= 0:
            self.ref_type.setCurrentIndex(idx)
        if "ref_value" in data:
            self.ref_value.setText(pretty_literal(data.get("ref_value")))
        if "fluence_scale" in data:
            self.fluence_scale.setText(str(data.get("fluence_scale", "")))
        if "fluence_offset" in data:
            self.fluence_offset.setText(str(data.get("fluence_offset", "")))
        if "delay_offset_fs" in data:
            self.delay_offset_fs.setText(str(data.get("delay_offset_fs", "")))
        elif "delay_offset_ps" in data:
            try:
                self.delay_offset_fs.setText(str(float(data.get("delay_offset_ps", 0.0)) * 1000.0))
            except Exception:
                pass
        if "delay_for_norm_max" in data:
            self.delay_for_norm_max.setText(str(data.get("delay_for_norm_max", "")))
        if "csv_path" in data:
            self.csv_path.setText(str(data.get("csv_path", "")))

    def to_dict(self):
        """Convert the current value to dict.

        Returns
        -------
        dict
            Serializable experiment definition containing the current widget values.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        out = {}
        if self.label_edit is not None:
            label = self.label_edit.text().strip()
            if label:
                out["label"] = label

        sample_name = self.sample_name.text().strip()
        if not sample_name:
            raise ValueError("sample_name cannot be empty.")
        out["sample_name"] = sample_name
        out["temperature_K"] = parse_int_like(self.temperature.text(), name="temperature_K")
        out["excitation_wl_nm"] = parse_float_like(self.excitation.text(), name="excitation_wl_nm")
        out["delay_fs"] = parse_int_like(self.delay_fs.text(), name="delay_fs")
        out["time_window_fs"] = parse_int_like(self.time_window.text(), name="time_window_fs")
        out["phi_mode"] = self.phi_mode.currentText()
        out["ref_type"] = self.ref_type.currentText()

        ref_value_text = self.ref_value.text().strip()
        if ref_value_text:
            out["ref_value"] = parse_ref_value(ref_value_text)

        fluence_scale = parse_optional_float_like(self.fluence_scale.text())
        if fluence_scale is not None:
            out["fluence_scale"] = fluence_scale

        fluence_offset = parse_optional_float_like(self.fluence_offset.text())
        if fluence_offset is not None:
            out["fluence_offset"] = fluence_offset

        delay_offset_fs = parse_optional_float_like(self.delay_offset_fs.text())
        if delay_offset_fs is not None:
            out["delay_offset_fs"] = delay_offset_fs

        delay_for_norm_max = parse_optional_float_like(self.delay_for_norm_max.text())
        if delay_for_norm_max is not None:
            out["delay_for_norm_max"] = delay_for_norm_max

        csv_path = self.csv_path.text().strip()
        if csv_path:
            out["csv_path"] = csv_path

        return out


class MergeFluenceExperimentWidget(QFrame):
    """Edit a logical fluence experiment assembled from multiple CSV fragments.

    Attributes
    ----------
    sub_entries : list of ExperimentFluenceLeafWidget
        Constituent experiment fragments merged into one plotted series.
    label_edit : QLineEdit
        Legend label for the merged series.
    delay_for_norm_max : QWidget
        Optional normalization-delay value shared by fragments.
    remove_callback : callable or None
        Callback removing the complete merged entry.
    """
    def __init__(self, *, title="Merged experiment", remove_callback=None, data=None):
        """Initialize ``MergeFluenceExperimentWidget``, bind shared state and services, and create its controls."""
        super().__init__()
        self.remove_callback = remove_callback
        self.sub_entries = []
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { border: 1px solid #cfcfcf; border-radius: 4px; }")

        outer = QVBoxLayout()
        self.setLayout(outer)

        header = QHBoxLayout()
        outer.addLayout(header)
        header.addWidget(QLabel(f"<b>{title}</b>"))
        header.addStretch()
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self._remove_me)
        header.addWidget(btn_remove)

        top_grid = QGridLayout()
        outer.addLayout(top_grid)
        top_grid.addWidget(QLabel("label:"), 0, 0)
        self.label_edit = QLineEdit("")
        top_grid.addWidget(self.label_edit, 0, 1)
        top_grid.addWidget(QLabel("Delay for normalization max:"), 1, 0)
        self.delay_for_norm_max = QLineEdit("")
        self.delay_for_norm_max.setValidator(QDoubleValidator())
        top_grid.addWidget(self.delay_for_norm_max, 1, 1)

        sub_header = QHBoxLayout()
        outer.addLayout(sub_header)
        sub_header.addWidget(QLabel("<b>Sub-experiments</b>"))
        sub_header.addStretch()
        add_btn = QPushButton("Add sub-experiment")
        add_btn.clicked.connect(self.add_sub_experiment)
        sub_header.addWidget(add_btn)

        self.sub_container = QWidget()
        self.sub_layout = QVBoxLayout()
        self.sub_layout.setContentsMargins(0, 0, 0, 0)
        self.sub_container.setLayout(self.sub_layout)
        outer.addWidget(self.sub_container)

        if data:
            self.set_data(data)
        else:
            self.add_sub_experiment()
            self.add_sub_experiment()

    def _remove_me(self):
        """Request removal of this experiment entry from its parent editor."""
        if callable(self.remove_callback):
            self.remove_callback(self)

    def add_sub_experiment(self, data=None):
        """Append a constituent experiment editor to this merged-series widget."""
        widget = ExperimentFluenceLeafWidget(
            title="Merged source",
            show_label=False,
            show_remove=True,
            remove_callback=self.remove_sub_experiment,
            data=data,
        )
        self.sub_entries.append(widget)
        self.sub_layout.addWidget(widget)

    def remove_sub_experiment(self, widget):
        """Remove one constituent experiment editor from this merged-series widget."""
        if widget in self.sub_entries:
            self.sub_entries.remove(widget)
            widget.setParent(None)
            widget.deleteLater()

    def set_data(self, data):
        """Populate the widget fields from one experiment mapping."""
        self.label_edit.setText(str(data.get("label", "")))
        if "delay_for_norm_max" in data:
            self.delay_for_norm_max.setText(str(data.get("delay_for_norm_max", "")))
        for widget in list(self.sub_entries):
            self.remove_sub_experiment(widget)
        for sub in data.get("merge", []):
            self.add_sub_experiment(sub)
        if not self.sub_entries:
            self.add_sub_experiment()

    def to_dict(self):
        """Convert the current value to dict.

        Returns
        -------
        dict
            Serializable experiment definition containing the current widget values.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        label = self.label_edit.text().strip()
        if not label:
            raise ValueError("Merged experiment label cannot be empty.")
        if not self.sub_entries:
            raise ValueError("Merged experiment must contain at least one sub-experiment.")
        out = {"label": label, "merge": [w.to_dict() for w in self.sub_entries]}
        delay_for_norm_max = parse_optional_float_like(self.delay_for_norm_max.text())
        if delay_for_norm_max is not None:
            out["delay_for_norm_max"] = delay_for_norm_max
        return out


class MergeExperimentWidget(QFrame):
    """Edit a logical delay experiment assembled from multiple CSV fragments.

    Attributes
    ----------
    sub_entries : list of ExperimentLeafWidget
        Constituent experiment fragments merged into one plotted series.
    label_edit : QLineEdit
        Legend label for the merged series.
    delay_for_norm_max : QWidget
        Optional normalization-delay value shared by fragments.
    remove_callback : callable or None
        Callback removing the complete merged entry.
    """
    def __init__(self, *, title="Merged experiment", remove_callback=None, data=None):
        """Initialize ``MergeExperimentWidget``, bind shared state and services, and create its controls."""
        super().__init__()
        self.remove_callback = remove_callback
        self.sub_entries = []
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { border: 1px solid #cfcfcf; border-radius: 4px; }")

        outer = QVBoxLayout()
        self.setLayout(outer)

        header = QHBoxLayout()
        outer.addLayout(header)
        header.addWidget(QLabel(f"<b>{title}</b>"))
        header.addStretch()
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self._remove_me)
        header.addWidget(btn_remove)

        top_grid = QGridLayout()
        outer.addLayout(top_grid)
        top_grid.addWidget(QLabel("label:"), 0, 0)
        self.label_edit = QLineEdit("")
        top_grid.addWidget(self.label_edit, 0, 1)
        top_grid.addWidget(QLabel("Delay for normalization max:"), 1, 0)
        self.delay_for_norm_max = QLineEdit("")
        self.delay_for_norm_max.setValidator(QDoubleValidator())
        top_grid.addWidget(self.delay_for_norm_max, 1, 1)

        sub_header = QHBoxLayout()
        outer.addLayout(sub_header)
        sub_header.addWidget(QLabel("<b>Sub-experiments</b>"))
        sub_header.addStretch()
        add_btn = QPushButton("Add sub-experiment")
        add_btn.clicked.connect(self.add_sub_experiment)
        sub_header.addWidget(add_btn)

        self.sub_container = QWidget()
        self.sub_layout = QVBoxLayout()
        self.sub_layout.setContentsMargins(0, 0, 0, 0)
        self.sub_container.setLayout(self.sub_layout)
        outer.addWidget(self.sub_container)

        if data:
            self.set_data(data)
        else:
            self.add_sub_experiment()
            self.add_sub_experiment()

    def _remove_me(self):
        """Request removal of this experiment entry from its parent editor."""
        if callable(self.remove_callback):
            self.remove_callback(self)

    def add_sub_experiment(self, data=None):
        """Append a constituent experiment editor to this merged-series widget."""
        widget = ExperimentLeafWidget(
            title="Merged source",
            show_label=False,
            show_remove=True,
            remove_callback=self.remove_sub_experiment,
            data=data,
        )
        self.sub_entries.append(widget)
        self.sub_layout.addWidget(widget)

    def remove_sub_experiment(self, widget):
        """Remove one constituent experiment editor from this merged-series widget."""
        if widget in self.sub_entries:
            self.sub_entries.remove(widget)
            widget.setParent(None)
            widget.deleteLater()

    def set_data(self, data):
        """Populate the widget fields from one experiment mapping."""
        self.label_edit.setText(str(data.get("label", "")))
        if "delay_for_norm_max" in data:
            self.delay_for_norm_max.setText(str(data.get("delay_for_norm_max", "")))
        for widget in list(self.sub_entries):
            self.remove_sub_experiment(widget)
        for sub in data.get("merge", []):
            self.add_sub_experiment(sub)
        if not self.sub_entries:
            self.add_sub_experiment()

    def to_dict(self):
        """Convert the current value to dict.

        Returns
        -------
        dict
            Serializable experiment definition containing the current widget values.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        label = self.label_edit.text().strip()
        if not label:
            raise ValueError("Merged experiment label cannot be empty.")
        if not self.sub_entries:
            raise ValueError("Merged experiment must contain at least one sub-experiment.")
        out = {"label": label, "merge": [w.to_dict() for w in self.sub_entries]}
        delay_for_norm_max = parse_optional_float_like(self.delay_for_norm_max.text())
        if delay_for_norm_max is not None:
            out["delay_for_norm_max"] = delay_for_norm_max
        return out


class MultiExperimentEditor(QGroupBox):
    """Edit, validate, and serialize experiment definitions for comparison plots.

    Attributes
    ----------
    allow_merge : bool
        Whether logical merged-experiment entries may be added.
    entries : list
        Current leaf and merged experiment widgets in display order.
    defaults : list of dict
        Initial serialized experiment definitions.
    series_kind : {"delay", "fluence"}
        Selects the leaf and merged widget family.
    scroll, scroll_content, scroll_layout : QWidget
        Scrollable container holding experiment entries.
    preview : QPlainTextEdit
        Read-only serialized preview of validated experiment definitions.
    """
    def __init__(self, title="Experiments", *, allow_merge=True, defaults=None, series_kind="delay"):
        """Initialize ``MultiExperimentEditor``, bind shared state and services, and create its controls."""
        super().__init__(title)
        self.allow_merge = allow_merge
        self.entries = []
        self.defaults = defaults or []
        self.series_kind = str(series_kind).strip().lower()

        outer = QVBoxLayout()
        self.setLayout(outer)

        btn_row = QHBoxLayout()
        outer.addLayout(btn_row)

        add_btn = QPushButton("Add experiment")
        add_btn.clicked.connect(self.add_experiment)
        btn_row.addWidget(add_btn)

        if self.allow_merge:
            add_merge_btn = QPushButton("Add merged experiment")
            add_merge_btn.clicked.connect(self.add_merged_experiment)
            btn_row.addWidget(add_merge_btn)

        load_btn = QPushButton("Load example")
        load_btn.clicked.connect(self.load_defaults)
        btn_row.addWidget(load_btn)

        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self.clear_entries)
        btn_row.addWidget(clear_btn)

        btn_row.addStretch()

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(360)
        outer.addWidget(self.scroll)

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll.setWidget(self.scroll_content)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFixedHeight(75)
        outer.addWidget(self.preview)

        outer.setStretchFactor(self.scroll, 1)
        outer.setStretchFactor(self.preview, 0)

        self.load_defaults()

    def update_preview(self):
        """Render the current experiment definitions in the read-only preview."""
        try:
            self.preview.setPlainText(pretty_literal(self.get_experiments()))
        except Exception as exc:
            self.preview.setPlainText(f"Preview unavailable: {exc}")

    def clear_entries(self):
        """Remove all experiment widgets and refresh the preview."""
        for widget in list(self.entries):
            self.remove_entry(widget)
        self.update_preview()

    def load_defaults(self):
        """Replace the editor contents with independent copies of its defaults.

        Existing child widgets are removed first. Each default mapping is then
        reconstructed as a single or merged experiment widget appropriate for
        the configured series kind.
        """
        self.clear_entries()
        if not self.defaults:
            self.add_experiment()
            return
        for item in self.defaults:
            if isinstance(item, dict) and "merge" in item:
                self.add_merged_experiment(item)
            else:
                self.add_experiment(item)
        self.update_preview()

    def set_experiments(self, experiments):
        """Set experiments.

        Parameters
        ----------
        experiments : object
            Experiment dictionaries describing datasets, labels, offsets, and optional merged fragments.

        Raises
        ------
        ValueError
            If a selector, range, mode, unit, or metadata value is invalid.
        """
        self.clear_entries()
        if not experiments:
            self.update_preview()
            return
        for item in experiments:
            if isinstance(item, dict) and "merge" in item:
                if not self.allow_merge:
                    raise ValueError("This editor does not support merged experiments.")
                self.add_merged_experiment(item)
            else:
                self.add_experiment(item)
        self.update_preview()

    def add_experiment(self, data=None):
        """Append a single or merged experiment widget and refresh the preview."""
        widget_cls = ExperimentLeafWidget if self.series_kind == "delay" else ExperimentFluenceLeafWidget
        widget = widget_cls(remove_callback=self.remove_entry, data=data)
        self.entries.append(widget)
        self.scroll_layout.addWidget(widget)
        self.update_preview()

    def add_merged_experiment(self, data=None):
        """Append a merged experiment editor appropriate for the active series kind."""
        widget_cls = MergeExperimentWidget if self.series_kind == "delay" else MergeFluenceExperimentWidget
        widget = widget_cls(remove_callback=self.remove_entry, data=data)
        self.entries.append(widget)
        self.scroll_layout.addWidget(widget)
        self.update_preview()

    def remove_entry(self, widget):
        """Remove one child experiment widget and refresh the preview."""
        if widget in self.entries:
            self.entries.remove(widget)
            widget.setParent(None)
            widget.deleteLater()
            self.update_preview()

    def get_experiments(self):
        """Serialize all valid child widgets to experiment dictionaries."""
        if not self.entries:
            raise ValueError("At least one experiment must be defined.")
        return [w.to_dict() for w in self.entries]
