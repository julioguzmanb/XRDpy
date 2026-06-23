from __future__ import annotations

import math

from PyQt5.QtCore import QSignalBlocker, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLineEdit,
    QSizePolicy,
    QWidget,
)


class PolarizationControlWidget(QWidget):
    """Edit the shared optional pyFAI polarization correction.

    Attributes
    ----------
    enabled_checkbox : QCheckBox
        Enables or disables polarization correction.
    factor_input : QLineEdit
        Numeric factor editor constrained to ``[-1, 1]``.
    _last_factor : float
        Last valid factor retained while correction is disabled.
    """

    valueChanged = pyqtSignal(bool, float)

    def __init__(
        self,
        *,
        enabled: bool = True,
        factor: float = 0.99,
        parent=None,
    ):
        """Initialize ``PolarizationControlWidget``, bind shared state and services, and create its controls."""
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        self.enabled_checkbox = QCheckBox("polarization_factor")
        self.enabled_checkbox.setToolTip(
            "Enable the pyFAI/txs polarization correction."
        )
        self.enabled_checkbox.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Preferred,
        )
        self.enabled_checkbox.setMinimumWidth(
            self.enabled_checkbox.sizeHint().width() + 6
        )
        layout.addWidget(self.enabled_checkbox)

        self.factor_input = QLineEdit()
        self.factor_input.setObjectName("PolarizationFactorInput")
        self.factor_input.setValidator(QDoubleValidator(-1.0, 1.0, 6, self))
        self.factor_input.setFixedWidth(88)
        self.factor_input.setToolTip(
            "Polarization factor: -1 vertical, 0 circular/random, +1 horizontal."
        )
        layout.addWidget(self.factor_input)
        layout.addStretch()

        self._last_factor = self._validated_factor(factor)
        self.set_configuration(enabled=enabled, factor=factor, emit=False)

        self.enabled_checkbox.toggled.connect(self._on_enabled_changed)
        self.factor_input.editingFinished.connect(self._on_factor_edited)

    @staticmethod
    def _validated_factor(value) -> float:
        """Parse the factor field and enforce pyFAI's inclusive ``[-1, 1]`` range."""
        factor = float(value)
        if not math.isfinite(factor) or not -1.0 <= factor <= 1.0:
            raise ValueError("polarization_factor must be between -1 and 1.")
        return factor

    @staticmethod
    def _format_factor(value: float) -> str:
        """Format a polarization factor without unnecessary trailing zeros."""
        return f"{float(value):g}"

    def factor(self) -> float:
        """Return the validated polarization factor from the numeric field."""
        text = self.factor_input.text().strip()
        if not text:
            return self._last_factor
        return self._validated_factor(text)

    def effective_factor(self):
        """Return the active factor, or ``None`` when correction is disabled."""
        if not self.enabled_checkbox.isChecked():
            return None
        return self.factor()

    def values(self) -> dict:
        """Return enabled state, stored factor, and effective backend factor."""
        return {
            "enabled": self.enabled_checkbox.isChecked(),
            "factor": self.factor(),
        }

    def set_values(self, values: dict) -> None:
        """Update enabled state and factor without changing their semantics."""
        if not isinstance(values, dict):
            return
        self.set_configuration(
            enabled=bool(values.get("enabled", True)),
            factor=values.get("factor", 0.99),
            emit=True,
        )

    def set_configuration(
        self,
        *,
        enabled: bool,
        factor: float,
        emit: bool = False,
    ) -> None:
        """Set configuration.

        Parameters
        ----------
        enabled : bool
            Whether polarization correction is enabled.
        factor : float
            Polarization correction factor in the interval ``[-1, 1]``.
        emit : bool
            Whether to emit a change signal after updating the control.
        """
        factor = self._validated_factor(0.99 if factor is None else factor)
        self._last_factor = factor
        checkbox_blocker = QSignalBlocker(self.enabled_checkbox)
        input_blocker = QSignalBlocker(self.factor_input)
        self.enabled_checkbox.setChecked(bool(enabled))
        self.factor_input.setText(self._format_factor(factor))
        self.factor_input.setEnabled(bool(enabled))
        del checkbox_blocker, input_blocker
        if emit:
            self.valueChanged.emit(bool(enabled), factor)

    def _on_enabled_changed(self, enabled: bool) -> None:
        """Update factor-field availability and emit the effective polarization setting."""
        self.factor_input.setEnabled(bool(enabled))
        self.valueChanged.emit(bool(enabled), self.factor())

    def _on_factor_edited(self) -> None:
        """Persist a valid edited factor and emit the updated correction setting."""
        try:
            factor = self.factor()
        except ValueError:
            factor = self._last_factor
        self._last_factor = factor
        self.factor_input.setText(self._format_factor(factor))
        self.valueChanged.emit(self.enabled_checkbox.isChecked(), factor)
