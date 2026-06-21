"""
Reusable log widget for the analysis GUI.
"""
from __future__ import annotations

from datetime import datetime

from PyQt5.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget


class LogWidget(QWidget):
    """Read-only plain-text log panel for user-facing GUI messages."""

    def __init__(self, parent=None):
        """Initialize the object and its runtime state."""
        super().__init__(parent)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.text_edit)

        self.setLayout(layout)

    def log(self, message: str):
        """Append a timestamped user-facing message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.text_edit.appendPlainText(f"[{timestamp}] {message}")

    def clear(self):
        """Remove all messages from the visible log."""
        self.text_edit.clear()
