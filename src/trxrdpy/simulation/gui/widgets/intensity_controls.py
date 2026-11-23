"""Controls shared by powder and single-crystal detector views."""
import math
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QCheckBox, QComboBox, QGridLayout, QGroupBox, QLabel, QLineEdit


class IntensityControls(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Detector intensity display", parent)
        layout = QGridLayout(self)
        self.consider = QCheckBox("Consider intensities")
        layout.addWidget(self.consider, 0, 0, 1, 2)
        self.cmap_label = QLabel("Colormap:")
        self.cmap = QComboBox()
        self.cmap.addItems(("viridis", "plasma", "inferno", "magma", "cividis",
                            "turbo", "gray", "hot", "jet"))
        layout.addWidget(self.cmap_label, 1, 0)
        layout.addWidget(self.cmap, 1, 1)
        self.clim_label = QLabel("Color limits [min, max]:")
        self.clim_min = QLineEdit()
        self.clim_max = QLineEdit()
        for field in (self.clim_min, self.clim_max):
            field.setValidator(QDoubleValidator(0, 1e12, 8))
            field.setPlaceholderText("Auto")
        layout.addWidget(self.clim_label, 2, 0)
        layout.addWidget(self.clim_min, 2, 1)
        layout.addWidget(self.clim_max, 2, 2)
        self.apply_polarization = QCheckBox("Apply beam polarization")
        self.apply_polarization.setChecked(True)
        layout.addWidget(self.apply_polarization, 3, 0)
        self.polarization_label = QLabel("Factor (-1 vertical to +1 horizontal):")
        self.polarization_factor = QLineEdit("1")
        self.polarization_factor.setValidator(QDoubleValidator(-1, 1, 6))
        self.polarization_factor.setToolTip(
            "-1: vertical linear; 0: unpolarized; +1: horizontal linear. "
            "Intermediate values describe partial polarization in the lab frame. "
            "The default is horizontal synchrotron polarization.")
        layout.addWidget(self.polarization_label, 3, 1)
        layout.addWidget(self.polarization_factor, 3, 2)
        self.axis_offset_label = QLabel("Polarization axis offset [°]:")
        self.axis_offset = QLineEdit("0")
        self.axis_offset.setValidator(QDoubleValidator(-360, 360, 6))
        self.axis_offset.setToolTip("Rotation of the polarization reference from laboratory horizontal.")
        layout.addWidget(self.axis_offset_label, 4, 1)
        layout.addWidget(self.axis_offset, 4, 2)
        self.noise_label = QLabel("Random noise:")
        self.noise = QComboBox()
        self.noise.addItems(("none", "gaussian", "poisson"))
        layout.addWidget(self.noise_label, 5, 0)
        layout.addWidget(self.noise, 5, 1)
        self.level_label = QLabel("Noise level / max peak:")
        self.level = QLineEdit("0.01")
        self.level.setValidator(QDoubleValidator(0, 1e6, 8))
        self.level.setToolTip("Gaussian: standard deviation. Poisson: mean counts per pixel, "
                              "both relative to the strongest reflection.")
        layout.addWidget(self.level_label, 6, 0)
        layout.addWidget(self.level, 6, 1)
        self.consider.toggled.connect(self.update_visibility)
        self.apply_polarization.toggled.connect(self.update_visibility)
        self.noise.currentTextChanged.connect(self.update_visibility)
        self.update_visibility()

    def update_visibility(self, *_args):
        enabled = self.consider.isChecked()
        for widget in (self.cmap_label, self.cmap, self.clim_label, self.clim_min,
                       self.clim_max, self.apply_polarization, self.noise_label, self.noise):
            widget.setVisible(enabled)
        for widget in (self.polarization_label, self.polarization_factor,
                       self.axis_offset_label, self.axis_offset):
            widget.setVisible(enabled and self.apply_polarization.isChecked())
        has_noise = enabled and self.noise.currentText() != "none"
        self.level_label.setVisible(has_noise)
        self.level.setVisible(has_noise)

    def load(self, state):
        self.consider.setChecked(bool(state.consider_intensities))
        for combo, value in ((self.cmap, state.intensity_cmap),
                             (self.noise, state.noise_distribution)):
            index = combo.findText(value)
            combo.setCurrentIndex(max(index, 0))
        self.level.setText(str(state.noise_level))
        self.clim_min.setText(str(state.clim_min))
        self.clim_max.setText(str(state.clim_max))
        self.apply_polarization.setChecked(bool(state.apply_polarization))
        self.polarization_factor.setText(str(state.polarization_factor))
        self.axis_offset.setText(str(state.polarization_axis_offset_deg))
        self.update_visibility()

    def save(self, state):
        state.consider_intensities = self.consider.isChecked()
        state.intensity_cmap = self.cmap.currentText()
        state.noise_distribution = self.noise.currentText()
        state.noise_level = self.level.text()
        state.clim_min = self.clim_min.text()
        state.clim_max = self.clim_max.text()
        state.apply_polarization = self.apply_polarization.isChecked()
        state.polarization_factor = self.polarization_factor.text()
        state.polarization_axis_offset_deg = self.axis_offset.text()

    def options(self, cif_path):
        if not self.consider.isChecked():
            return {}
        from ...intensity_view import (validate_clim, validate_display,
                                       validate_polarization_factor)
        level = float(self.level.text()) if self.noise.currentText() != "none" else 0.0
        validate_display(self.cmap.currentText(), self.noise.currentText(), level)
        clim = validate_clim(tuple(float(field.text()) if field.text().strip() else None
                                   for field in (self.clim_min, self.clim_max)))
        factor = validate_polarization_factor(float(self.polarization_factor.text())
                       if self.apply_polarization.isChecked() else None)
        offset = float(self.axis_offset.text()) if self.apply_polarization.isChecked() else 0.0
        if not math.isfinite(offset):
            raise ValueError("Polarization axis offset must be finite.")
        if not cif_path:
            raise ValueError("A CIF with atom positions is required to consider intensities.")
        return dict(consider_intensities=True, cif_file_path=cif_path,
                    intensity_cmap=self.cmap.currentText(),
                    intensity_clim=clim,
                    polarization_factor=factor,
                    polarization_axis_offset_deg=offset,
                    noise_distribution=self.noise.currentText(), noise_level=level)
