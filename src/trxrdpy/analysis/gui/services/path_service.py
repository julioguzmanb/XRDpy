"""
Path handling utilities for the analysis GUI.

This service should stay independent from Qt.
Qt widgets may ask users to choose folders/files, but path validation and
normalization should live here.
"""

from pathlib import Path
from typing import Optional

try:
    from trxrdpy.analysis.common.paths import AnalysisPaths
except Exception:
    AnalysisPaths = None

class PathService:
    """
    Small service for normalizing and validating paths used by the GUI.
    """

    @staticmethod
    def normalize(path: str | Path | None) -> Optional[Path]:
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
        normalized = PathService.normalize(path)
        return normalized is not None and normalized.exists()

    @staticmethod
    def is_file(path: str | Path | None) -> bool:
        normalized = PathService.normalize(path)
        return normalized is not None and normalized.is_file()

    @staticmethod
    def is_dir(path: str | Path | None) -> bool:
        normalized = PathService.normalize(path)
        return normalized is not None and normalized.is_dir()

    @staticmethod
    def ensure_dir(path: str | Path) -> Path:
        normalized = PathService.normalize(path)

        if normalized is None:
            raise ValueError("Cannot create a directory from a None path.")

        normalized.mkdir(parents=True, exist_ok=True)
        return normalized

    @staticmethod
    def build_analysis_paths(
        path_root,
        analysis_subdir: str = "analysis",
        raw_subdir: str = "",
    ):
        """
        Build the backend AnalysisPaths object used by analysis routines.
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