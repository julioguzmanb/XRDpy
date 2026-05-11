"""
Reusable parameter widgets for the analysis GUI.

These widgets intentionally preserve the standard Qt appearance used by the
legacy GUI. They only centralize validation and value conversion.
"""

from typing import Optional

from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import QCheckBox, QLineEdit


class TextParameter(QLineEdit):
    """
    Plain text parameter field.
    """

    def value(self) -> str:
        return self.text().strip()

    def set_value(self, value):
        self.setText("" if value is None else str(value))


class IntParameter(QLineEdit):
    """
    Integer parameter field.
    """

    def __init__(self, value: Optional[int] = None, parent=None):
        super().__init__(parent)

        self.setValidator(QIntValidator(self))

        if value is not None:
            self.set_value(value)

    def value(self) -> Optional[int]:
        text = self.text().strip()

        if text == "":
            return None

        return int(text)

    def set_value(self, value: Optional[int]):
        self.setText("" if value is None else str(int(value)))


class FloatParameter(QLineEdit):
    """
    Floating-point parameter field.
    """

    def __init__(self, value: Optional[float] = None, parent=None):
        super().__init__(parent)

        self.setValidator(QDoubleValidator(self))

        if value is not None:
            self.set_value(value)

    def value(self) -> Optional[float]:
        text = self.text().strip()

        if text == "":
            return None

        return float(text)

    def set_value(self, value: Optional[float]):
        self.setText("" if value is None else str(float(value)))


class BoolParameter(QCheckBox):
    """
    Boolean parameter field.
    """

    def value(self) -> bool:
        return self.isChecked()

    def set_value(self, value: bool):
        self.setChecked(bool(value))