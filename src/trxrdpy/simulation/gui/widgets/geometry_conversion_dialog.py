"""Bidirectional planar-detector conversion using pyFAI's native routines."""
import math
from pathlib import Path

from PyQt5.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from .path_widgets import DropFileLineEdit
from ...poni import read_poni_file, write_poni_file
from ...utils import wavelength_to_energy


PYFAI_FIELDS = (
    ("dist", "Distance [m]:", "0.1"),
    ("poni1", "poni\u2081 [m]:", "0.05"),
    ("poni2", "poni\u2082 [m]:", "0.05"),
    ("rot1", "rot\u2081 [rad]:", "0"),
    ("rot2", "rot\u2082 [rad]:", "0"),
    ("rot3", "rot\u2083 [rad]:", "0"),
    ("pixel1", "Pixel 1 / vertical [m]:", "0.0001"),
    ("pixel2", "Pixel 2 / horizontal [m]:", "0.0001"),
)
FIT2D_FIELDS = (
    ("directDist", "Direct distance [mm]:", "100"),
    ("centerX", "Beam center X [px]:", "500"),
    ("centerY", "Beam center Y [px]:", "500"),
    ("tilt", "Tilt [°]:", "0"),
    ("tiltPlanRotation", "Tilt-plane rotation [°]:", "0"),
    ("pixelX", "Pixel X / horizontal [micrometre]:", "100"),
    ("pixelY", "Pixel Y / vertical [micrometre]:", "100"),
)


def _validated(values, positive):
    values = {name: float(value) for name, value in values.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("All geometry values must be finite.")
    if any(values[name] <= 0 for name in positive):
        raise ValueError("Distance and pixel sizes must be positive.")
    return values


def pyfai_to_fit2d(values):
    from pyFAI.geometry import Geometry

    values = _validated(values, ("dist", "pixel1", "pixel2"))
    result = Geometry(**values).getFit2D()
    result = {name: float(result[name]) for name, _, _ in FIT2D_FIELDS}
    _validated(result, ("directDist", "pixelX", "pixelY"))
    if not 0 <= result["tilt"] < 90:
        raise ValueError("This geometry is outside Fit2D's forward-detector tilt range.")
    return result


def fit2d_to_pyfai(values):
    from pyFAI.geometry import Geometry

    values = _validated(values, ("directDist", "pixelX", "pixelY"))
    if not 0 <= values["tilt"] < 90:
        raise ValueError("Fit2D tilt must be between 0 (inclusive) and 90 degrees.")
    geometry = Geometry()
    geometry.setFit2D(**values)
    return {name: float(getattr(geometry, name)) for name, _, _ in PYFAI_FIELDS}


class GeometryConversionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("pyFAI / Fit2D / PONI Geometry")
        self.resize(1000, 760)
        root = QVBoxLayout(self)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        self.scroll_area.setWidget(content)
        root.addWidget(self.scroll_area)
        layout = QVBoxLayout(content)
        paths = QHBoxLayout()
        self.poni_path = DropFileLineEdit(suffixes=(".poni",))
        self.poni_path.setPlaceholderText("PONI file")
        self.poni_path.fileDropped.connect(self.load_poni)
        self.poni_path.returnPressed.connect(lambda: self.load_poni(self.poni_path.text()))
        paths.addWidget(self.poni_path)
        browse = QPushButton("Load PONI...")
        browse.clicked.connect(self.browse_poni)
        paths.addWidget(browse)
        layout.addLayout(paths)
        panels = QHBoxLayout()
        self.fields = {}
        for title, definitions in (("pyFAI", PYFAI_FIELDS), ("Fit2D", FIT2D_FIELDS)):
            group = QGroupBox(title)
            form = QFormLayout(group)
            for name, label, default in definitions:
                field = QLineEdit(default)
                self.fields[name] = field
                form.addRow(label, field)
            panels.addWidget(group)
        layout.addLayout(panels)
        actions = QHBoxLayout()
        self.to_fit2d_button = QPushButton("pyFAI to Fit2D")
        self.to_fit2d_button.clicked.connect(lambda: self.convert(to_fit2d=True))
        self.to_pyfai_button = QPushButton("Fit2D to pyFAI")
        self.to_pyfai_button.clicked.connect(lambda: self.convert(to_fit2d=False))
        actions.addWidget(self.to_fit2d_button)
        actions.addWidget(self.to_pyfai_button)
        layout.addLayout(actions)
        warning = QLabel("Fit2D does not retain independent rot3. Conversion back returns "
                         "pyFAI's canonical orientation. Simulation parameters are not changed.")
        warning.setWordWrap(True)
        layout.addWidget(warning)
        export_group = QGroupBox("PONI Export (uniform detector)")
        export_layout = QGridLayout(export_group)
        for col, (name, label, default) in enumerate((
            ("num_pixels_v", "Pixels V (rows):", ""),
            ("num_pixels_h", "Pixels H (columns):", ""),
            ("energy_kev", "Energy [keV]:", "15"),
        )):
            field = QLineEdit(default)
            self.fields[name] = field
            export_layout.addWidget(QLabel(label), 0, col * 2)
            export_layout.addWidget(field, 0, col * 2 + 1)
        self.poni_source = QComboBox()
        self.poni_source.addItems(["pyFAI", "Fit2D"])
        export_layout.addWidget(QLabel("Geometry source:"), 1, 0)
        export_layout.addWidget(self.poni_source, 1, 1)
        self.poni_orientation = QComboBox()
        for label, value in (("1", 1), ("2", 2), ("3 (pyFAI default)", 3), ("4", 4)):
            self.poni_orientation.addItem(label, value)
        self.poni_orientation.setCurrentIndex(2)
        export_layout.addWidget(QLabel("Image orientation:"), 1, 2)
        export_layout.addWidget(self.poni_orientation, 1, 3)
        self.use_beam_button = QPushButton("Use Current Beam")
        export_layout.addWidget(self.use_beam_button, 1, 4, 1, 2)
        self.save_poni_button = QPushButton("Save PONI...")
        self.save_poni_button.clicked.connect(self.save_poni)
        export_layout.addWidget(self.save_poni_button, 2, 0, 1, 6)
        layout.addWidget(export_group)
        # Retain legacy utility-state fields; export now lives in the detector tabs.
        export_group.hide()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def values(self):
        return {"poni_path": self.poni_path.text(),
                "poni_source": self.poni_source.currentText(),
                "poni_orientation": str(self.poni_orientation.currentData()),
                **{name: field.text() for name, field in self.fields.items()}}

    def set_values(self, values):
        for name, value in values.items():
            if name == "poni_path":
                self.poni_path.setText(str(value))
            elif name == "poni_source":
                self.poni_source.setCurrentText(str(value))
            elif name == "poni_orientation":
                index = self.poni_orientation.findData(int(value))
                if index >= 0:
                    self.poni_orientation.setCurrentIndex(index)
            elif name in self.fields:
                self.fields[name].setText(str(value))

    def convert(self, *, to_fit2d):
        try:
            definitions = PYFAI_FIELDS if to_fit2d else FIT2D_FIELDS
            values = {name: self.fields[name].text() for name, _, _ in definitions}
            convert = pyfai_to_fit2d if to_fit2d else fit2d_to_pyfai
            result = convert(values)
            self.set_values({name: format(value, ".15g") for name, value in result.items()})
            self.status.setText("Converted to Fit2D." if to_fit2d else "Converted to pyFAI.")
            return result
        except Exception as exc:
            self.status.setText(f"Error: {exc}")
            return None

    def browse_poni(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load PONI", self.poni_path.text(),
                                             "PONI files (*.poni)")
        if path:
            self.load_poni(path)

    def load_poni(self, path):
        try:
            geometry = read_poni_file(path)
            values = {name: getattr(geometry, name) for name, _, _ in PYFAI_FIELDS}
            # Missing rotations in legacy PONI files mean an untilted detector.
            for name in ("rot1", "rot2", "rot3"):
                if values[name] is None:
                    values[name] = 0.0
            _validated(values, ("dist", "pixel1", "pixel2"))
            shape = geometry.max_shape or ("", "")
            values.update(num_pixels_v=shape[0], num_pixels_h=shape[1],
                          energy_kev=(wavelength_to_energy(geometry.wavelength) / 1000
                                      if geometry.wavelength else ""),
                          poni_orientation=geometry.detector_config.get("orientation") or 3,
                          poni_source="pyFAI")
            self.set_values(values)
            self.poni_path.setText(str(path))
            self.status.setText("Loaded pyFAI geometry.")
            return True
        except Exception as exc:
            self.status.setText(f"Error: {exc}")
            return False

    def export_poni(self, path, *, overwrite=False):
        """Export the selected coordinate system, never stale conversion results."""
        definitions = PYFAI_FIELDS if self.poni_source.currentText() == "pyFAI" else FIT2D_FIELDS
        values = {name: self.fields[name].text() for name, _, _ in definitions}
        if self.poni_source.currentText() == "Fit2D":
            values = fit2d_to_pyfai(values)
        return write_poni_file(
            path, **values,
            max_shape=(self.fields["num_pixels_v"].text(), self.fields["num_pixels_h"].text()),
            energy_ev=float(self.fields["energy_kev"].text()) * 1000,
            orientation=self.poni_orientation.currentData(), overwrite=overwrite,
        )

    def save_poni(self):
        current = self.poni_path.text().strip()
        suggested = str(Path(current).with_name(Path(current).stem + "_manual.poni")) if current else "geometry.poni"
        selected, _ = QFileDialog.getSaveFileName(
            self, "Save PONI", suggested, "PONI files (*.poni)",
            options=QFileDialog.DontConfirmOverwrite)
        if not selected:
            return
        path = Path(selected).expanduser()
        if not path.suffix:
            path = path.with_suffix(".poni")
        try:
            try:
                written = self.export_poni(path)
            except FileExistsError:
                answer = QMessageBox.question(self, "Replace PONI?", f"Replace this file?\n{path}",
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if answer != QMessageBox.Yes:
                    return
                written = self.export_poni(path, overwrite=True)
            self.status.setText(f"Saved PONI: {written}")
        except Exception as exc:
            self.status.setText(f"Error: {exc}")
