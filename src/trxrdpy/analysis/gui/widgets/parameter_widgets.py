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
    """Plain text parameter field."""

    def value(self) -> str:
        """Return the current text without numeric conversion."""
        return self.text().strip()

    def set_value(self, value):
        """Replace the field text with the supplied value."""
        self.setText("" if value is None else str(value))


class IntParameter(QLineEdit):
    """Integer parameter field."""

    def __init__(self, value: Optional[int] = None, parent=None):
        """Initialize the object and its runtime state."""
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
        """Set the field from an integer-compatible value."""
        self.setText("" if value is None else str(int(value)))


class FloatParameter(QLineEdit):
    """Floating-point parameter field."""

    def __init__(self, value: Optional[float] = None, parent=None):
        """Initialize the object and its runtime state."""
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
        """Set the field from a floating-point-compatible value."""
        self.setText("" if value is None else str(float(value)))


class BoolParameter(QCheckBox):
    """Boolean parameter field."""

    def value(self) -> bool:
        """Return the checkbox state as a boolean."""
        return self.isChecked()

    def set_value(self, value: bool):
        """Set the checkbox state from a truth value."""
        self.setChecked(bool(value))
