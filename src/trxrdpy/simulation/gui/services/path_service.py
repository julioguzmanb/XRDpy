"""Path normalization and file-dialog history for the simulation GUI."""
from __future__ import annotations

from pathlib import Path


class SimulationPathService:
    """Track the launch directory and most recent successful dialog location."""

    def __init__(self, launch_directory: str | Path | None = None):
        launch_path = self.normalize(launch_directory or Path.cwd())
        if launch_path is None:
            launch_path = Path.cwd().resolve()
        if launch_path.is_file():
            launch_path = launch_path.parent

        self.launch_directory = launch_path
        self.last_dialog_directory = launch_path

    @staticmethod
    def normalize(path: str | Path | None) -> Path | None:
        """Return an expanded absolute path, or ``None`` for an empty value."""
        if path is None or not str(path).strip():
            return None
        path = Path(path).expanduser()
        try:
            return path.resolve()
        except FileNotFoundError:
            return path.absolute()

    @classmethod
    def _usable_dialog_path(cls, path: str | Path | None) -> Path | None:
        """Return an existing path or its existing parent for dialog use."""
        normalized = cls.normalize(path)
        if normalized is None:
            return None
        if normalized.exists():
            return normalized
        if normalized.parent.exists():
            return normalized.parent
        return None

    def dialog_start_path(
        self,
        *,
        current: str | Path | None = None,
        preferred_directory: str | Path | None = None,
        default_name: str | None = None,
    ) -> Path:
        """Choose a stable starting file or directory for a Qt dialog."""
        candidates = (
            self._usable_dialog_path(current),
            self._usable_dialog_path(preferred_directory),
            self._usable_dialog_path(self.last_dialog_directory),
            self._usable_dialog_path(self.launch_directory),
            self._usable_dialog_path(Path.cwd()),
        )
        selected = next((candidate for candidate in candidates if candidate), Path.cwd())
        if default_name:
            base = selected if selected.is_dir() else selected.parent
            return base / default_name
        return selected

    def remember_dialog_selection(self, selected: str | Path | None) -> None:
        """Use a selected file's parent as the next dialog directory."""
        normalized = self.normalize(selected)
        if normalized is None:
            return
        self.last_dialog_directory = (
            normalized if normalized.is_dir() else normalized.parent
        )
