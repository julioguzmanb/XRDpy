"""
Reusable log widget for the analysis GUI.
"""

from datetime import datetime

from PyQt5.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget


class LogWidget(QWidget):
    """
    Read-only plain-text log panel for user-facing GUI messages.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.text_edit)

        self.setLayout(layout)

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.text_edit.appendPlainText(f"[{timestamp}] {message}")

    def clear(self):
        self.text_edit.clear()