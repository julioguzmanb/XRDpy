"""
Reusable parameter widgets for the analysis GUI.

These widgets intentionally preserve the standard Qt appearance used by the
legacy GUI. They only centralize validation and value conversion.
"""
from __future__ import annotations

from typing import Optional

from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import QCheckBox, QLineEdit


class TextParameter(QLineEdit):
    """Plain-text parameter editor with a uniform ``value`` interface."""

    def value(self) -> str:
        """Return current field text verbatim without parsing or numeric conversion."""
        return self.text().strip()

    def set_value(self, value):
        """Replace the field text with the supplied value."""
        self.setText("" if value is None else str(value))


class IntParameter(QLineEdit):
    """Line edit that validates and returns integer-compatible values."""

    def __init__(self, value: Optional[int] = None, parent=None):
        """Initialize configuration, normalize inputs, and create the object runtime state."""
        super().__init__(parent)

        self.setValidator(QIntValidator(self))

        if value is not None:
            self.set_value(value)

    def value(self) -> Optional[int]:
        """Return the current field parsed as an integer."""
        text = self.text().strip()

        if text == "":
            return None

        return int(text)

    def set_value(self, value: Optional[int]):
        """Populate the field from an integer-compatible value or clear it for ``None``."""
        self.setText("" if value is None else str(int(value)))


class FloatParameter(QLineEdit):
    """Line edit that validates and returns floating-point values."""

    def __init__(self, value: Optional[float] = None, parent=None):
        """Initialize configuration, normalize inputs, and create the object runtime state."""
        super().__init__(parent)

        self.setValidator(QDoubleValidator(self))

        if value is not None:
            self.set_value(value)

    def value(self) -> Optional[float]:
        """Return the current field parsed as a float."""
        text = self.text().strip()

        if text == "":
            return None

        return float(text)

    def set_value(self, value: Optional[float]):
        """Populate the field from a float-compatible value or clear it for ``None``."""
        self.setText("" if value is None else str(float(value)))


class BoolParameter(QCheckBox):
    """Checkbox exposing its checked state through a uniform ``value`` method."""

    def value(self) -> bool:
        """Return the current checkbox state as a native Python boolean."""
        return self.isChecked()

    def set_value(self, value: bool):
        """Set the checkbox state from a truth value."""
        self.setChecked(bool(value))
