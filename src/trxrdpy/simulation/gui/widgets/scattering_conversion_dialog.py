"""q, d-spacing, and scattering-angle conversion at a selected photon energy."""
import math

from PyQt5.QtWidgets import (
    QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout,
)

from ...utils import convert_scattering_coordinate, energy_to_wavelength, wavelength_to_energy


class ScatteringConversionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("q / d / Angle Conversion")
        self.resize(620, 500)
        layout = QVBoxLayout(self)
        self.use_beam_button = QPushButton("Use Current Beam")
        layout.addWidget(self.use_beam_button)
        form = QFormLayout()
        self.radiation = QComboBox()
        self.radiation.addItems(["Energy [keV]", "Wavelength [\u00c5]"])
        self.radiation_value = QLineEdit("15")
        beam = QHBoxLayout()
        beam.addWidget(self.radiation)
        beam.addWidget(self.radiation_value)
        form.addRow("Incident beam:", beam)
        self.source = QComboBox()
        for label, key in (("q [\u00c5\u207b\u00b9]", "q"), ("d [\u00c5]", "d"),
                           ("2\u03b8 [\u00b0]", "two_theta"), ("\u03b8 [\u00b0]", "theta")):
            self.source.addItem(label, key)
        self.value = QLineEdit("1")
        entry = QHBoxLayout()
        entry.addWidget(self.source)
        entry.addWidget(self.value)
        form.addRow("Convert from:", entry)
        layout.addLayout(form)
        self.convert_button = QPushButton("Convert")
        self.convert_button.clicked.connect(self.convert)
        layout.addWidget(self.convert_button)
        results = QFormLayout()
        self.outputs = {}
        for name, label in (("q", "q [\u00c5\u207b\u00b9]:"), ("d", "d [\u00c5]:"),
                            ("two_theta", "2\u03b8 [\u00b0]:"), ("theta", "\u03b8 [\u00b0]:")):
            field = QLineEdit()
            field.setReadOnly(True)
            self.outputs[name] = field
            results.addRow(label, field)
        layout.addLayout(results)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._radiation_index = 0
        self.radiation.currentIndexChanged.connect(self._change_radiation_unit)
        self.value.returnPressed.connect(self.convert)
        self.radiation_value.returnPressed.connect(self.convert)
        self.convert()

    def _change_radiation_unit(self, index):
        previous = self._radiation_index
        self._radiation_index = index
        if index == previous:
            return
        try:
            value = float(self.radiation_value.text())
            if math.isfinite(value) and value > 0:
                value = (energy_to_wavelength(value * 1000) * 1e10 if index == 1
                         else wavelength_to_energy(value * 1e-10) / 1000)
                self.radiation_value.setText(format(value, ".12g"))
        except ValueError:
            pass

    def values(self):
        return {"source": self.source.currentData(), "value": self.value.text(),
                "radiation": self.radiation.currentIndex(),
                "radiation_value": self.radiation_value.text()}

    def set_values(self, values):
        self.source.setCurrentIndex(max(0, self.source.findData(values.get("source", "q"))))
        self.value.setText(str(values.get("value", "1")))
        index = 1 if values.get("radiation") == 1 else 0
        self.radiation.blockSignals(True)
        self.radiation.setCurrentIndex(index)
        self.radiation.blockSignals(False)
        self._radiation_index = index
        self.radiation_value.setText(str(values.get("radiation_value", "15")))
        self.convert()

    def set_energy_ev(self, energy):
        values = self.values()
        values.update(radiation=0, radiation_value=format(float(energy) / 1000, ".12g"))
        self.set_values(values)

    def convert(self):
        try:
            beam = float(self.radiation_value.text())
            if not math.isfinite(beam) or beam <= 0:
                raise ValueError("Energy or wavelength must be positive and finite.")
            wavelength = energy_to_wavelength(beam * 1000) * 1e10 if self.radiation.currentIndex() == 0 else beam
            result = convert_scattering_coordinate(self.value.text(), self.source.currentData(), wavelength)
            for name, value in result.items():
                self.outputs[name].setText("Not accessible" if value is None else format(value, ".10g"))
            self.status.setText("No Bragg angle exists at this wavelength." if result["theta"] is None else "")
            return result
        except (ValueError, OverflowError) as exc:
            for field in self.outputs.values():
                field.clear()
            self.status.setText(f"Error: {exc}")
            return None
