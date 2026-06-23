"""
Path handling utilities for the analysis GUI.

This service should stay independent from Qt.
Qt widgets may ask users to choose folders/files, but path validation and
normalization should live here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from trxrdpy.analysis.common.paths import AnalysisPaths
except Exception:
    AnalysisPaths = None


class PathService:
    """Normalize paths and maintain file-dialog location history.

    Attributes
    ----------
    launch_directory : pathlib.Path
        Existing directory from which the GUI session was launched.
    last_dialog_directory : pathlib.Path
        Most recently used existing directory for file and folder dialogs.
    """

    def __init__(self, launch_directory: str | Path | None = None):
        """Remember where the GUI was launched and the last directory selected.

        Qt otherwise chooses a platform-dependent directory for an empty file
        dialog.  In an installed package that can unexpectedly be the package
        source directory rather than the user's working directory.
        """

        launch_path = self.normalize(launch_directory or Path.cwd())
        if launch_path is None:
            launch_path = Path.cwd().resolve()
        if launch_path.is_file():
            launch_path = launch_path.parent

        self.launch_directory = launch_path
        self.last_dialog_directory = launch_path

    @staticmethod
    def normalize(path: str | Path | None) -> Optional[Path]:
        """Expand and resolve a path without requiring it to exist.

        Parameters
        ----------
        path : str | Path | None
            Input filesystem path.

        Returns
        -------
        Optional[Path]
            Expanded absolute path, or ``None`` for an empty input.
        """
        if path is None:
            return None

        path = Path(path).expanduser()

        try:
            return path.resolve()
        except FileNotFoundError:
            # resolve(strict=False) would also work, but this keeps compatibility
            # with older habits and makes the behavior explicit.
            return path.absolute()

    @staticmethod
    def exists(path: str | Path | None) -> bool:
        """Return whether a nonempty normalized path currently exists on disk."""
        normalized = PathService.normalize(path)
        return normalized is not None and normalized.exists()

    @staticmethod
    def is_file(path: str | Path | None) -> bool:
        """Return whether the normalized path identifies an existing file."""
        normalized = PathService.normalize(path)
        return normalized is not None and normalized.is_file()

    @staticmethod
    def is_dir(path: str | Path | None) -> bool:
        """Return whether the normalized path identifies an existing directory."""
        normalized = PathService.normalize(path)
        return normalized is not None and normalized.is_dir()

    @staticmethod
    def ensure_dir(path: str | Path) -> Path:
        """Create the normalized directory and return its resolved path."""
        normalized = PathService.normalize(path)

        if normalized is None:
            raise ValueError("Cannot create a directory from a None path.")

        normalized.mkdir(parents=True, exist_ok=True)
        return normalized

    @staticmethod
    def _usable_dialog_path(path: str | Path | None) -> Optional[Path]:
        """Return an existing file/directory, or an existing parent."""

        if path is None or not str(path).strip():
            return None

        normalized = PathService.normalize(path)
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
        """Choose a predictable starting path for a Qt file dialog.

        An existing value already shown in a path field has priority.  It is
        followed by the experiment root, the most recently used dialog
        directory, and finally the directory from which the GUI was launched.
        """

        candidates = (
            self._usable_dialog_path(current),
            self._usable_dialog_path(preferred_directory),
            self._usable_dialog_path(self.last_dialog_directory),
            self._usable_dialog_path(self.launch_directory),
            self._usable_dialog_path(Path.cwd()),
        )
        selected = next(
            (candidate for candidate in candidates if candidate),
            Path.cwd(),
        )

        if default_name:
            base = selected if selected.is_dir() else selected.parent
            return base / default_name
        return selected

    def remember_dialog_selection(self, selected: str | Path | None) -> None:
        """Remember the parent directory of a successful dialog selection.

        Parameters
        ----------
        selected : str | Path | None
            Whether the corresponding control or item is selected.
        """

        if selected is None or not str(selected).strip():
            return

        normalized = self.normalize(selected)
        if normalized is None:
            return
        self.last_dialog_directory = (
            normalized if normalized.is_dir() else normalized.parent
        )

    @staticmethod
    def build_analysis_paths(
        path_root,
        analysis_subdir: str = "analysis",
        raw_subdir: str = "",
    ):
        """Build the path configuration consumed by analysis backends.

        Relative raw and analysis subdirectories are resolved beneath
        ``path_root``. Blank fields are normalized consistently with the GUI;
        an empty root raises ``ValueError``.
        """

        if AnalysisPaths is None:
            raise ImportError("AnalysisPaths could not be imported from the package.")

        path_root = "" if path_root is None else str(path_root).strip()

        if not path_root:
            raise ValueError("path_root cannot be empty.")

        return AnalysisPaths(
            path_root=Path(path_root).expanduser(),
            analysis_subdir=(analysis_subdir or "analysis").strip(),
            raw_subdir=(raw_subdir or "").strip(),
        )
