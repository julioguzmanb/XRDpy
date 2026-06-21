"""
Reusable path-selection widgets for the analysis GUI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)

from trxrdpy.analysis.gui.services import PathService


class DropPathLineEdit(QLineEdit):
    """Line edit that accepts local files or directories dragged from Finder."""

    pathDropped = pyqtSignal(str)

    def __init__(
        self,
        text: str = "",
        *,
        mode: str = "file",
        parent=None,
    ):
        """Initialize the object and its runtime state."""
        super().__init__(text, parent)

        if mode not in {"file", "directory", "either"}:
            raise ValueError("mode must be 'file', 'directory', or 'either'.")

        self.mode = mode
        self.setAcceptDrops(True)
        drop_kind = "a file or folder" if mode == "either" else f"a {mode}"
        self.setToolTip(f"Enter a path, or drag {drop_kind} here.")

    def _path_from_mime_data(self, mime_data) -> Optional[Path]:
        """Return path from mime data."""
        if mime_data is None or not mime_data.hasUrls():
            return None

        for url in mime_data.urls():
            if not url.isLocalFile():
                continue

            path = Path(url.toLocalFile()).expanduser()
            if self.mode == "file" and not path.is_file():
                continue
            if self.mode == "directory" and not path.is_dir():
                continue
            if self.mode == "either" and not path.exists():
                continue

            try:
                return path.resolve()
            except FileNotFoundError:
                return path.absolute()

        return None

    def dragEnterEvent(self, event):
        """Return drag enter event."""
        if self._path_from_mime_data(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Return drag move event."""
        if self._path_from_mime_data(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Return drop event.

        Parameters
        ----------
        event : object
            Qt or Matplotlib event supplied by the framework callback.
        """
        path = self._path_from_mime_data(event.mimeData())
        if path is None:
            event.ignore()
            return

        text = str(path)
        self.setText(text)
        self.setCursorPosition(len(text))
        event.acceptProposedAction()
        self.pathDropped.emit(text)
        # Match typing a path and leaving the field so existing state-sync and
        # validation callbacks also run after a drop.
        self.editingFinished.emit()


class PathSelector(QWidget):
    """Small reusable widget for selecting files or folders.

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
        """Initialize ``PathSelector``, bind shared state and services, and create its controls."""
        super().__init__(parent)

        if mode not in {"directory", "file"}:
            raise ValueError("mode must be either 'directory' or 'file'.")

        self.mode = mode
        self.path_service = path_service or PathService()
        self.on_path_changed = on_path_changed

        self.line_edit = DropPathLineEdit(mode=mode)
        self.browse_button = QPushButton(label)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.browse_button)

        self.setLayout(layout)

        self.browse_button.clicked.connect(self._browse)
        self.line_edit.editingFinished.connect(self._emit_current_path)

    def path(self) -> Optional[Path]:
        """Return path."""
        text = self.line_edit.text().strip()

        if not text:
            return None

        return self.path_service.normalize(text)

    def set_path(self, path: str | Path | None):
        """Set path."""
        normalized = self.path_service.normalize(path)

        if normalized is None:
            self.line_edit.clear()
        else:
            self.line_edit.setText(str(normalized))

        self._emit_current_path()

    def _browse(self):
        """Return browse."""
        start_path = str(self.path_service.dialog_start_path(current=self.path()))
        if self.mode == "directory":
            selected = QFileDialog.getExistingDirectory(
                self,
                "Select directory",
                start_path,
            )
        else:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Select file",
                start_path,
            )

        if selected:
            self.path_service.remember_dialog_selection(selected)
            self.set_path(selected)

    def _emit_current_path(self):
        """Emit current path."""
        if self.on_path_changed is not None:
            self.on_path_changed(self.path())
