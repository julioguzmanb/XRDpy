"""Extra Functions window for offline photon attenuation scans."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from PyQt5.QtCore import QAbstractTableModel, Qt, QTimer
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog,
                             QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
                             QTableView, QVBoxLayout)

from ...attenuation import XRayMaterial, grazing_depth_from_constants, parse_formula
from ...attenuation_plot import (AttenuationData, load_attenuation_csv,
                                 load_attenuation_h5, plot_attenuation,
                                 save_attenuation_csv, save_attenuation_h5,
                                 stepped_range)
from ...cif import read_cif_file


class _CurveTableModel(QAbstractTableModel):
    """Virtual table so fine energy steps do not create thousands of widgets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.x = np.empty(0)
        self.length = np.empty(0)
        self.label = "Normal penetration depth [µm]"
        self.x_label = "Photon energy [eV]"

    def set_curve(self, x, length, label, x_label):
        self.beginResetModel()
        self.x = x
        self.length = length
        self.label = label
        self.x_label = x_label
        self.endResetModel()

    def rowCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.x)

    def columnCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else 2

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        value = self.x[index.row()] if index.column() == 0 else self.length[index.row()]
        return f"{value:.8g}"

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.x_label if section == 0 else self.label
        return super().headerData(section, orientation, role)


class AttenuationDialog(QDialog):
    """Calculate, inspect, plot, and exchange attenuation energy scans."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Photon Attenuation Length")
        self.resize(820, 680)
        self.data = self.axes = None
        layout = QVBoxLayout(self)

        material_group = QGroupBox("Material")
        material_form = QFormLayout(material_group)
        formula_row = QHBoxLayout()
        self.formula = QLineEdit()
        self.formula.setPlaceholderText("e.g. SrTiO3, V2O3, Si3N4")
        formula_row.addWidget(self.formula, 1)
        cif_button = QPushButton("Load formula from CIF…")
        cif_button.clicked.connect(self._load_cif)
        formula_row.addWidget(cif_button)
        material_form.addRow("Chemical formula", formula_row)
        self.density = QLineEdit()
        self.density.setPlaceholderText("e.g. 5.12")
        material_form.addRow("Density [g/cm³]", self.density)
        layout.addWidget(material_group)

        energy_group = QGroupBox("Photon energy scan [eV]")
        energy_grid = QGridLayout(energy_group)
        for column, title in enumerate(("Start", "Stop", "Step")):
            energy_grid.addWidget(QLabel(title), 0, column)
        self.start, self.stop, self.step = (QLineEdit(value) for value in ("10000", "15000", "10"))
        for column, widget in enumerate((self.start, self.stop, self.step)):
            energy_grid.addWidget(widget, 1, column)
        layout.addWidget(energy_group)

        incidence_group = QGroupBox("Attenuation calculation")
        incidence_grid = QGridLayout(incidence_group)
        self.mode = QComboBox()
        self.mode.addItems(("Normal penetration depth", "Bulk attenuation length"))
        self.mode.currentIndexChanged.connect(self._mode_changed)
        incidence_grid.addWidget(QLabel("Mode"), 0, 0)
        incidence_grid.addWidget(self.mode, 0, 1, 1, 3)
        self.angle = QLineEdit("1.0")
        self.angle_label = QLabel("Incidence angle [°]")
        incidence_grid.addWidget(self.angle_label, 1, 0)
        incidence_grid.addWidget(self.angle, 1, 1)
        self.angle_scan = QCheckBox("Scan incidence angles")
        self.angle_scan.toggled.connect(self._mode_changed)
        incidence_grid.addWidget(self.angle_scan, 1, 2, 1, 2)
        self.scan_angle_labels = [QLabel(title) for title in ("Angle start [°]", "Angle stop [°]", "Angle step [°]")]
        for column, label in enumerate(self.scan_angle_labels):
            incidence_grid.addWidget(label, 2, column + 1)
        self.angle_start, self.angle_stop, self.angle_step = (QLineEdit(value) for value in ("0.5", "3.0", "0.1"))
        for column, widget in enumerate((self.angle_start, self.angle_stop, self.angle_step)):
            incidence_grid.addWidget(widget, 3, column + 1)
        self.slider_motor_label = QLabel("Slider motor")
        self.slider_motor = QComboBox()
        self.slider_motor.addItems(("Incidence angle", "Photon energy"))
        incidence_grid.addWidget(self.slider_motor_label, 4, 0)
        incidence_grid.addWidget(self.slider_motor, 4, 1, 1, 3)
        layout.addWidget(incidence_group)

        options = QHBoxLayout()
        options.addWidget(QLabel("Output length unit"))
        self.unit = QComboBox()
        self.unit.addItems(("µm", "nm", "m"))
        self.unit.setMinimumWidth(130)
        self.unit.view().setMinimumWidth(130)
        self.unit.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.unit.currentTextChanged.connect(self._unit_changed)
        options.addWidget(self.unit)
        self.log_y = QCheckBox("Logarithmic vertical axis")
        options.addWidget(self.log_y)
        options.addStretch()
        layout.addLayout(options)

        buttons = QHBoxLayout()
        self.calculate_button = QPushButton("Calculate and Plot")
        self.calculate_button.clicked.connect(self.calculate)
        self.csv_button = QPushButton("Export Data…")
        self.csv_button.clicked.connect(self.save_data)
        self.load_button = QPushButton("Load Data…")
        self.load_button.clicked.connect(self.load_data)
        self.plot_loaded_button = QPushButton("Plot Loaded Data")
        self.plot_loaded_button.clicked.connect(self.plot_loaded)
        self.pdf_button = QPushButton("Save Plot (.PDF)")
        self.pdf_button.clicked.connect(self.save_pdf)
        for button in (self.calculate_button, self.csv_button, self.load_button,
                       self.plot_loaded_button, self.pdf_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.csv_button.setEnabled(False)
        self.plot_loaded_button.setEnabled(False)
        self.pdf_button.setEnabled(False)

        self.status = QLabel("Enter a chemical formula and density, or load a formula from CIF.")
        layout.addWidget(self.status)
        self.table = QTableView()
        self.table_model = _CurveTableModel(self.table)
        self.table.setModel(self.table_model)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table, 1)
        self._mode_changed()

    def _mode_changed(self, *_):
        grazing = self.mode.currentIndex() == 0
        scanning = grazing and self.angle_scan.isChecked()
        self.angle_label.setVisible(grazing and not scanning)
        self.angle.setVisible(grazing and not scanning)
        self.angle_scan.setEnabled(grazing)
        for widget in (*self.scan_angle_labels, self.angle_start, self.angle_stop, self.angle_step):
            widget.setVisible(scanning)
        self.slider_motor_label.setVisible(scanning)
        self.slider_motor.setVisible(scanning)

    def _load_cif(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CIF", "", "CIF files (*.cif);;All files (*)")
        if not path:
            return
        try:
            for block in read_cif_file(path).values():
                for key in ("_chemical_formula_sum", "_chemical_formula_structural"):
                    if key in block:
                        formula = str(block[key]).strip("'\" ").replace(" ", "")
                        parse_formula(formula)
                        self.formula.setText(formula)
                        self.status.setText(f"Loaded {formula} from CIF. Enter or check the density.")
                        return
            raise ValueError("No supported chemical formula in the CIF. Enter it manually.")
        except Exception as exc:
            QMessageBox.warning(self, "CIF formula", str(exc))

    def calculate(self):
        try:
            formula = self.formula.text().strip()
            material = XRayMaterial(formula, float(self.density.text()))
            energy = stepped_range(float(self.start.text()), float(self.stop.text()), float(self.step.text()))
            if energy[0] <= 0:
                raise ValueError("Photon energy must be positive.")
            mode = "grazing" if self.mode.currentIndex() == 0 else "bulk"
            if mode == "bulk":
                angles = np.array([0.])
            elif self.angle_scan.isChecked():
                angles = stepped_range(float(self.angle_start.text()), float(self.angle_stop.text()),
                                       float(self.angle_step.text()))
            else:
                angles = np.array([float(self.angle.text())])
            if np.any((angles < 0) | (angles > 90)):
                raise ValueError("Incidence angles must be between 0 and 90 degrees.")
            unit = self.unit.currentText()
            if mode == "bulk":
                lengths = material.bulk_attenuation_length(energy / 1000, unit="m")
                calculator = None
                energy_calculator = None
            else:
                energy_keV = energy / 1000
                delta, beta = material.optical_constants(energy_keV)
                calculator = lambda angle: grazing_depth_from_constants(
                    energy_keV, delta, beta, angle, unit="m")
                energy_calculator = lambda selected_eV: grazing_depth_from_constants(
                    selected_eV / 1000, *material.optical_constants(selected_eV / 1000),
                    angles, unit="m")
                lengths = None
            self.data = AttenuationData(energy, lengths, angles, mode, unit, formula,
                                        material.density_g_cm3, calculator, energy_calculator)
            self._show_initial_curve()
            self._plot()
            self.csv_button.setEnabled(True)
            self.plot_loaded_button.setEnabled(False)
            self.status.setText(f"Calculated {len(energy)} energies × {len(angles)} incidence angles.")
        except Exception as exc:
            QMessageBox.warning(self, "Photon attenuation calculation", str(exc))
            self.status.setText(str(exc))

    def _show_initial_curve(self):
        if self._slider_motor() == "energy":
            value = float(self.data.energy_eV[0])
            self._display_table("energy", value, self.data.angles_deg,
                                self.data.display_curve_at_energy(value))
        else:
            value = float(self.data.angles_deg[0])
            self._display_table("angle", value, self.data.energy_eV,
                                self.data.display_curve_at(value))

    def _slider_motor(self):
        return ("energy" if self.data is not None and self.data.mode == "grazing"
                and len(self.data.angles_deg) > 1 and self.slider_motor.currentIndex() == 1
                else "angle")

    def _display_table(self, motor, value, x, curve):
        data = self.data
        self.selected_motor = motor
        self.selected_value = value
        label = "Bulk attenuation length" if data.mode == "bulk" else "Normal penetration depth"
        x_label = "Incidence angle [°]" if motor == "energy" else "Photon energy [eV]"
        self.table_model.set_curve(x, curve, f"{label} [{data.unit}]", x_label)

    def _unit_changed(self, unit):
        if self.data is None:
            return
        self.data.unit = unit
        motor = getattr(self, "selected_motor", "angle")
        value = getattr(self, "selected_value", float(self.data.angles_deg[0]))
        if motor == "energy":
            x = self.data.angles_deg
            curve = self.data.display_curve_at_energy(value)
        else:
            x = self.data.energy_eV
            curve = self.data.display_curve_at(value)
        self._display_table(motor, value, x, curve)
        if self.axes is not None:
            self.axes.lines[0].set_ydata(curve)
            label = "Bulk attenuation length" if self.data.mode == "bulk" else "Normal penetration depth"
            self.axes.set_ylabel(f"{label} [{unit}]")
            self.axes.relim()
            self.axes.autoscale_view()
            self.axes.figure.canvas.draw_idle()

    def _plot(self):
        self.data.unit = self.unit.currentText()
        self.axes = plot_attenuation(self.data, log_y=self.log_y.isChecked(),
                                     slider_motor=self._slider_motor(),
                                     on_selection_change=self._display_table)
        self.axes.figure.canvas.mpl_connect(
            "close_event", lambda _event: QTimer.singleShot(100, self._restore_focus))
        self.pdf_button.setEnabled(True)

    def _restore_focus(self):
        if self.isVisible():
            self.raise_()
            self.activateWindow()

    def save_data(self):
        if self.data is None:
            return
        path, selected = QFileDialog.getSaveFileName(
            self, "Export attenuation data", "attenuation.h5",
            "HDF5 files (*.h5 *.hdf5);;CSV files (*.csv)")
        if path:
            try:
                if selected.startswith("CSV"):
                    path = str(Path(path).with_suffix(".csv"))
                    save_attenuation_csv(path, self.data)
                else:
                    if not path.lower().endswith((".h5", ".hdf5")):
                        path = str(Path(path).with_suffix(".h5"))
                    save_attenuation_h5(path, self.data)
                self.status.setText(f"Exported data to {path}")
            except Exception as exc:
                QMessageBox.warning(self, "Export attenuation data", str(exc))

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load attenuation data", "",
            "Attenuation files (*.h5 *.hdf5 *.csv);;HDF5 files (*.h5 *.hdf5);;CSV files (*.csv)")
        if not path:
            return
        try:
            unit = self.unit.currentText()
            self.data = (load_attenuation_csv(path, display_unit=unit) if path.lower().endswith(".csv")
                         else load_attenuation_h5(path, display_unit=unit))
            self.axes = None
            self.pdf_button.setEnabled(False)
            self.selected_motor = "angle"
            self.selected_value = float(self.data.angles_deg[0])
            self._show_initial_curve()
            self.csv_button.setEnabled(True)
            self.plot_loaded_button.setEnabled(True)
            self.status.setText(f"Loaded {len(self.data.energy_eV)} energies × {len(self.data.angles_deg)} angles.")
        except Exception as exc:
            QMessageBox.warning(self, "Load attenuation data", str(exc))

    def plot_loaded(self):
        if self.data is not None:
            self._plot()

    def save_pdf(self):
        if self.axes is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save attenuation plot", "attenuation.pdf", "PDF files (*.pdf)")
        if path:
            self.axes.figure.savefig(path, bbox_inches="tight")
            self.status.setText(f"Saved plot to {path}")

    def values(self) -> dict:
        """Return only user inputs and plot preferences for GUI state saving."""
        return {
            "formula": self.formula.text(),
            "density": self.density.text(),
            "energy_start": self.start.text(),
            "energy_stop": self.stop.text(),
            "energy_step": self.step.text(),
            "mode": self.mode.currentText(),
            "incidence_angle": self.angle.text(),
            "scan_incidence_angles": self.angle_scan.isChecked(),
            "angle_start": self.angle_start.text(),
            "angle_stop": self.angle_stop.text(),
            "angle_step": self.angle_step.text(),
            "slider_motor": self.slider_motor.currentText(),
            "display_unit": self.unit.currentText(),
            "log_y": self.log_y.isChecked(),
        }

    def set_values(self, values: dict | None) -> None:
        """Restore inputs from a saved state; omit calculated or loaded data."""
        values = values if isinstance(values, dict) else {}
        self.data = self.axes = None
        self.table_model.set_curve(np.empty(0), np.empty(0),
                                   "Normal penetration depth [µm]", "Photon energy [eV]")
        self.csv_button.setEnabled(False)
        self.plot_loaded_button.setEnabled(False)
        self.pdf_button.setEnabled(False)
        text_fields = {
            "formula": (self.formula, ""), "density": (self.density, ""),
            "energy_start": (self.start, "10000"), "energy_stop": (self.stop, "15000"),
            "energy_step": (self.step, "10"), "incidence_angle": (self.angle, "1.0"),
            "angle_start": (self.angle_start, "0.5"), "angle_stop": (self.angle_stop, "3.0"),
            "angle_step": (self.angle_step, "0.1"),
        }
        for name, (widget, default) in text_fields.items():
            widget.setText(str(values.get(name, default)))
        for name, widget, default in (
            ("mode", self.mode, "Normal penetration depth"),
            ("slider_motor", self.slider_motor, "Incidence angle"),
            ("display_unit", self.unit, "µm"),
        ):
            choice = str(values.get(name, default))
            widget.setCurrentText(choice if widget.findText(choice) >= 0 else default)
        self.angle_scan.setChecked(bool(values.get("scan_incidence_angles", False)))
        self.log_y.setChecked(bool(values.get("log_y", False)))
        self._mode_changed()
        self.status.setText("Inputs restored. Calculate to generate a new curve.")
