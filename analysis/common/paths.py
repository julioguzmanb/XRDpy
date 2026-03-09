from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class AnalysisPaths:
    path_root: Path
    raw_subdir: str = ""
    analysis_subdir: str = "analysis"
    values: Dict[str, Any] = field(default_factory=dict)

    def root(self, *parts: str) -> Path:
        return self.path_root.joinpath(*parts)

    @property
    def raw_root(self) -> Path:
        return self.path_root / self.raw_subdir if self.raw_subdir else self.path_root

    @property
    def analysis_root(self) -> Path:
        return self.path_root / self.analysis_subdir

    def with_values(self, **kwargs: Any) -> "AnalysisPaths":
        merged = dict(self.values)
        merged.update(kwargs)
        return AnalysisPaths(
            path_root=self.path_root,
            raw_subdir=self.raw_subdir,
            analysis_subdir=self.analysis_subdir,
            values=merged,
        )

    def format_path(self, template: str, **kwargs: Any) -> Path:
        vals = dict(self.values)
        vals.update(kwargs)
        return self.path_root / template.format(**vals)

    def format_analysis_path(self, template: str, **kwargs: Any) -> Path:
        vals = dict(self.values)
        vals.update(kwargs)
        return self.analysis_root / template.format(**vals)