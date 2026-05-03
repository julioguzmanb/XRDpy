from __future__ import annotations

from typing import Any

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..services.simulation_service import GeometryInfo, MotorInfo


class GeometryPanel(QWidget):
    geometry_changed = pyqtSignal(str)
    angles_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._geometries_by_name: dict[str, GeometryInfo] = {}
        self._sample_inputs: dict[str, QDoubleSpinBox] = {}
        self._detector_inputs: dict[str, QDoubleSpinBox] = {}

        self.geometry_label = QLabel("Geometry")
        self.geometry_combo = QComboBox()
        self.geometry_combo.currentIndexChanged.connect(self._on_geometry_changed)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.mode_note_label = QLabel("")
        self.mode_note_label.setWordWrap(True)
        self.mode_note_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.sample_group = QGroupBox("Sample motors")
        self.sample_layout = QGridLayout()
        self.sample_group.setLayout(self.sample_layout)

        self.detector_group = QGroupBox("Detector motors")
        self.detector_layout = QGridLayout()
        self.detector_group.setLayout(self.detector_layout)

        layout = QVBoxLayout(self)
        layout.addWidget(self.geometry_label)
        layout.addWidget(self.geometry_combo)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.mode_note_label)
        layout.addWidget(self.sample_group)
        layout.addWidget(self.detector_group)
        layout.addStretch(1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_geometries(self, geometries: list[GeometryInfo]) -> None:
        self._geometries_by_name = {geometry.name: geometry for geometry in geometries}

        self.geometry_combo.blockSignals(True)
        self.geometry_combo.clear()

        for geometry in geometries:
            self.geometry_combo.addItem(geometry.display_name, geometry.name)

        self.geometry_combo.blockSignals(False)

        if geometries:
            self.set_current_geometry(geometries[0].name)
        else:
            self.summary_label.setText("")
            self.mode_note_label.setText("")
            self._clear_grid_layout(self.sample_layout)
            self._clear_grid_layout(self.detector_layout)
            self.sample_group.setVisible(False)
            self.detector_group.setVisible(False)

    def set_current_geometry(self, geometry_name: str) -> None:
        index = self.geometry_combo.findData(geometry_name)
        if index < 0:
            raise ValueError(f"Geometry '{geometry_name}' is not available.")

        self.geometry_combo.blockSignals(True)
        self.geometry_combo.setCurrentIndex(index)
        self.geometry_combo.blockSignals(False)

        geometry = self._geometries_by_name[geometry_name]
        self._render_geometry(geometry)

    def current_geometry_name(self) -> str:
        data = self.geometry_combo.currentData()
        return "" if data is None else str(data)

    def current_geometry(self) -> GeometryInfo | None:
        geometry_name = self.current_geometry_name()
        if not geometry_name:
            return None
        return self._geometries_by_name.get(geometry_name)

    def current_sample_angles(self) -> dict[str, float]:
        return {name: widget.value() for name, widget in self._sample_inputs.items()}

    def current_detector_angles(self) -> dict[str, float]:
        return {name: widget.value() for name, widget in self._detector_inputs.items()}

    def set_sample_angles(self, angles: dict[str, float] | None) -> None:
        if not angles:
            return
        for name, value in angles.items():
            widget = self._sample_inputs.get(name)
            if widget is not None:
                widget.blockSignals(True)
                widget.setValue(float(value))
                widget.blockSignals(False)

    def set_detector_angles(self, angles: dict[str, float] | None) -> None:
        if not angles:
            return
        for name, value in angles.items():
            widget = self._detector_inputs.get(name)
            if widget is not None:
                widget.blockSignals(True)
                widget.setValue(float(value))
                widget.blockSignals(False)

    def reset_current_angles_to_defaults(self) -> None:
        geometry = self.current_geometry()
        if geometry is None:
            return

        for motor in geometry.sample_motors:
            widget = self._sample_inputs.get(motor.name)
            if widget is not None:
                widget.blockSignals(True)
                widget.setValue(float(motor.default))
                widget.blockSignals(False)

        for motor in geometry.detector_motors:
            widget = self._detector_inputs.get(motor.name)
            if widget is not None:
                widget.blockSignals(True)
                widget.setValue(float(motor.default))
                widget.blockSignals(False)

        self.angles_changed.emit()

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------
    def _on_geometry_changed(self, _index: int) -> None:
        geometry_name = self.current_geometry_name()
        if not geometry_name:
            return

        geometry = self._geometries_by_name[geometry_name]
        self._render_geometry(geometry)
        self.geometry_changed.emit(geometry_name)
        self.angles_changed.emit()

    def _render_geometry(self, geometry: GeometryInfo) -> None:
        self.summary_label.setText(geometry.summary)

        if geometry.uses_legacy_controls:
            self.mode_note_label.setText(
                "Legacy Euler mode selected. Geometry-aware motor controls are not used in this mode."
            )
        else:
            self.mode_note_label.setText(
                "Geometry-aware mode selected. Edit the active sample and detector motor angles below."
            )

        self._clear_grid_layout(self.sample_layout)
        self._clear_grid_layout(self.detector_layout)

        self._sample_inputs = {}
        self._detector_inputs = {}

        self._populate_grid_layout(
            grid_layout=self.sample_layout,
            motors=geometry.sample_motors,
            target=self._sample_inputs,
        )
        self._populate_grid_layout(
            grid_layout=self.detector_layout,
            motors=geometry.detector_motors,
            target=self._detector_inputs,
        )

        self.sample_group.setVisible(bool(geometry.sample_motors))
        self.detector_group.setVisible(bool(geometry.detector_motors))

    def _populate_grid_layout(
        self,
        grid_layout: QGridLayout,
        motors: tuple[MotorInfo, ...],
        target: dict[str, QDoubleSpinBox],
    ) -> None:
        if not motors:
            return

        header_name = QLabel("<b>Motor</b>")
        header_label = QLabel("<b>Meaning</b>")
        header_angle = QLabel("<b>Angle [deg]</b>")

        grid_layout.addWidget(header_name, 0, 0)
        grid_layout.addWidget(header_label, 0, 1)
        grid_layout.addWidget(header_angle, 0, 2)

        for row, motor in enumerate(motors, start=1):
            name_label = QLabel(motor.name)
            name_label.setToolTip(self._tooltip_for_motor(motor))

            meaning_label = QLabel(motor.label)
            meaning_label.setWordWrap(True)
            meaning_label.setToolTip(self._tooltip_for_motor(motor))

            spinbox = QDoubleSpinBox()
            spinbox.setRange(-360.0, 360.0)
            spinbox.setDecimals(4)
            spinbox.setSingleStep(0.1)
            spinbox.setValue(float(motor.default))
            spinbox.valueChanged.connect(self.angles_changed.emit)
            spinbox.setToolTip(self._tooltip_for_motor(motor))
            spinbox.setMinimumWidth(110)

            grid_layout.addWidget(name_label, row, 0)
            grid_layout.addWidget(meaning_label, row, 1)
            grid_layout.addWidget(spinbox, row, 2)

            target[motor.name] = spinbox

        grid_layout.setColumnStretch(0, 0)
        grid_layout.setColumnStretch(1, 1)
        grid_layout.setColumnStretch(2, 0)

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------
    def _tooltip_for_motor(self, motor: MotorInfo) -> str:
        parts = [
            f"name: {motor.name}",
            f"label: {motor.label}",
            f"default: {motor.default}",
        ]

        if motor.frame:
            parts.append(f"frame: {motor.frame}")
        if motor.axis is not None:
            parts.append(f"axis: {self._format_compact_value(motor.axis)}")
        if motor.origin is not None:
            parts.append(f"origin: {self._format_compact_value(motor.origin)}")

        parts.append(f"description: {motor.description}")
        return "\n".join(parts)

    @staticmethod
    def _format_compact_value(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(str(v) for v in value) + "]"
        return str(value)

    @staticmethod
    def _clear_grid_layout(grid_layout: QGridLayout) -> None:
        while grid_layout.count():
            item = grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()