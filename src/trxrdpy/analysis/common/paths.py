from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class AnalysisPaths:
    """Resolve raw-data and analysis locations from one experiment root.

    ``path_root`` must be a ``pathlib.Path``; string values are not coerced.
    ``raw_subdir`` and ``analysis_subdir`` may be absolute or relative to that
    root. The resulting ``raw_root`` and ``analysis_root`` properties are used
    by every facility backend. This class resolves paths but does not create
    directories.
    """
    path_root: Path
    raw_subdir: str = ""
    analysis_subdir: str = "analysis"
    values: Dict[str, Any] = field(default_factory=dict)

    def root(self, *parts: str) -> Path:
        """Return ``path_root`` as a ``Path`` object."""
        return self.path_root.joinpath(*parts)

    @property
    def raw_root(self) -> Path:
        """Return ``path_root/raw_subdir``, or ``path_root`` when it is blank."""
        return self.path_root / self.raw_subdir if self.raw_subdir else self.path_root

    @property
    def analysis_root(self) -> Path:
        """Return the configured analysis directory beneath ``path_root``."""
        return self.path_root / self.analysis_subdir

    def with_values(self, **kwargs: Any) -> "AnalysisPaths":
        """Return a new path configuration with additional template values."""
        merged = dict(self.values)
        merged.update(kwargs)
        return AnalysisPaths(
            path_root=self.path_root,
            raw_subdir=self.raw_subdir,
            analysis_subdir=self.analysis_subdir,
            values=merged,
        )

    def format_path(self, template: str, **kwargs: Any) -> Path:
        """Format a path template using the configured raw and analysis roots."""
        vals = dict(self.values)
        vals.update(kwargs)
        return self.path_root / template.format(**vals)

    def format_analysis_path(self, template: str, **kwargs: Any) -> Path:
        """Format a relative template beneath ``analysis_root``."""
        vals = dict(self.values)
        vals.update(kwargs)
        return self.analysis_root / template.format(**vals)
