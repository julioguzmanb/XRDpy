"""
Reusable path-selection widgets for the analysis GUI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

from trxrdpy.analysis.gui.services import PathService


class PathSelector(QWidget):
    """
    Small reusable widget for selecting files or folders.

    It contains:
    - a QLineEdit showing the selected path
    - a Browse button
    """

    def __init__(
        self,
        *,
        label: str = "Browse",
        mode: str = "directory",
        path_service: Optional[PathService] = None,
        on_path_changed: Optional[Callable[[Optional[Path]], None]] = None,
        parent=None,
    ):
        super().__init__(parent)

        if mode not in {"directory", "file"}:
            raise ValueError("mode must be either 'directory' or 'file'.")

        self.mode = mode
        self.path_service = path_service or PathService()
        self.on_path_changed = on_path_changed

        self.line_edit = QLineEdit()
        self.browse_button = QPushButton(label)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.browse_button)

        self.setLayout(layout)

        self.browse_button.clicked.connect(self._browse)
        self.line_edit.editingFinished.connect(self._emit_current_path)

    def path(self) -> Optional[Path]:
        text = self.line_edit.text().strip()

        if not text:
            return None

        return self.path_service.normalize(text)

    def set_path(self, path: str | Path | None):
        normalized = self.path_service.normalize(path)

        if normalized is None:
            self.line_edit.clear()
        else:
            self.line_edit.setText(str(normalized))

        self._emit_current_path()

    def _browse(self):
        if self.mode == "directory":
            selected = QFileDialog.getExistingDirectory(
                self,
                "Select directory",
                str(self.path() or Path.home()),
            )
        else:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Select file",
                str(self.path() or Path.home()),
            )

        if selected:
            self.set_path(selected)

    def _emit_current_path(self):
        if self.on_path_changed is not None:
            self.on_path_changed(self.path())
