"""Path-entry widgets used by the simulation GUI."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QLineEdit


class DropFileLineEdit(QLineEdit):
    """Line edit accepting local file drag-and-drop URLs."""

    fileDropped = pyqtSignal(str)

    def __init__(self, text: str = "", *, suffixes: tuple[str, ...] = (), parent=None):
        """Initialize accepted suffixes and enable drag-and-drop."""
        super().__init__(text, parent)
        self.suffixes = tuple(str(s).lower() for s in suffixes)
        self.setAcceptDrops(True)
        suffix_text = ", ".join(self.suffixes) if self.suffixes else "file"
        self.setToolTip(f"Enter a path, or drag a {suffix_text} file here.")

    def _path_from_mime_data(self, mime_data) -> Optional[Path]:
        """Return the first acceptable local file path from a Qt MIME payload."""
        if mime_data is None or not mime_data.hasUrls():
            return None

        for url in mime_data.urls():
            if not url.isLocalFile():
                continue

            path = Path(url.toLocalFile()).expanduser()
            if not path.is_file():
                continue
            if self.suffixes and path.suffix.lower() not in self.suffixes:
                continue

            try:
                return path.resolve()
            except FileNotFoundError:
                return path.absolute()

        return None

    def dragEnterEvent(self, event):
        """Accept only local file drags that match this widget's constraints."""
        if self._path_from_mime_data(event.mimeData()) is None:
            event.ignore()
            return
        event.acceptProposedAction()

    def dragMoveEvent(self, event):
        """Maintain acceptance while an acceptable file is dragged over us."""
        if self._path_from_mime_data(event.mimeData()) is None:
            event.ignore()
            return
        event.acceptProposedAction()

    def dropEvent(self, event):
        """Populate the field from the dropped file and notify listeners."""
        path = self._path_from_mime_data(event.mimeData())
        if path is None:
            event.ignore()
            return

        text = str(path)
        self.setText(text)
        self.setCursorPosition(len(text))
        event.acceptProposedAction()
        self.fileDropped.emit(text)
        self.editingFinished.emit()
