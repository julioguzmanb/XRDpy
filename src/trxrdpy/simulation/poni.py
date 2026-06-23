"""
Utilities for reading PONI detector calibration files.

This module only parses and normalizes calibration values. Higher-level
objects decide how those values are applied to a simulation.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
import math
from typing import Any


__all__ = ["PoniGeometry", "read_poni_file"]



def _degrees_or_zero(value):
    """Convert a radian value to degrees, treating ``None`` as zero."""
    return 0.0 if value is None else math.degrees(value)

@dataclass(frozen=True)
class PoniGeometry:
    """Normalized geometry extracted from a pyFAI PONI calibration.

    PONI axes follow pyFAI: axis 1 is the slow/vertical detector dimension and
    axis 2 is the fast/horizontal dimension. Rotations are stored in radians as
    read and converted to degrees when creating :class:`Detector` arguments.
    """

    path: Path | None = None
    version: str | None = None
    detector: str | None = None
    detector_config: dict[str, Any] = field(default_factory=dict)

    dist: float | None = None
    poni1: float | None = None
    poni2: float | None = None

    rot1: float | None = None
    rot2: float | None = None
    rot3: float | None = None

    wavelength: float | None = None

    pixel1: float | None = None
    pixel2: float | None = None
    max_shape: tuple[int, int] | None = None

    @property
    def pxsize_v(self) -> float | None:
        """
        Pixel size along the vertical image direction.
        """
        return self.pixel1

    @property
    def pxsize_h(self) -> float | None:
        """
        Pixel size along the horizontal image direction.
        """
        return self.pixel2

    @property
    def num_pixels_v(self) -> int | None:
        """
        Number of pixels along the vertical image direction.
        """
        if self.max_shape is None:
            return None
        return self.max_shape[0]

    @property
    def num_pixels_h(self) -> int | None:
        """
        Number of pixels along the horizontal image direction.
        """
        if self.max_shape is None:
            return None
        return self.max_shape[1]

    def detector_kwargs(self, include_rotations: bool = False) -> dict[str, Any]:
        """
        Return keyword arguments compatible with the manual Detector constructor.

        Rotations are excluded by default because they should be validated
        separately for each detector geometry.

        Parameters
        ----------
        include_rotations : bool
            Include degree Euler values converted from PONI radians.

        Returns
        -------
        dict
            Validated keyword arguments for :class:`Detector`.
        """
        required = {
            "dist": self.dist,
            "poni1": self.poni1,
            "poni2": self.poni2,
            "pxsize_h": self.pxsize_h,
            "pxsize_v": self.pxsize_v,
            "num_pixels_h": self.num_pixels_h,
            "num_pixels_v": self.num_pixels_v,
        }

        missing = [name for name, value in required.items() if value is None]
        if missing:
            missing_txt = ", ".join(missing)
            raise ValueError(
                f"Cannot build detector keyword arguments; missing: {missing_txt}"
            )

        kwargs: dict[str, Any] = {
            "detector_type": "manual",
            "pxsize_h": self.pxsize_h,
            "pxsize_v": self.pxsize_v,
            "num_pixels_h": self.num_pixels_h,
            "num_pixels_v": self.num_pixels_v,
            "dist": self.dist,
            "poni1": self.poni1,
            "poni2": self.poni2,
        }

        if include_rotations:
            kwargs.update(
                {
                    "rotx": _degrees_or_zero(self.rot1),
                    "roty": _degrees_or_zero(self.rot2),
                    "rotz": _degrees_or_zero(self.rot3),
                }
            )

        return kwargs


def read_poni_file(path: str | Path) -> PoniGeometry:
    """Read a PONI file and return validated detector geometry.

    Both JSON and Python-literal ``Detector_config`` encodings are accepted.
    Pixel sizes and detector shape may be supplied by the config or inferred
    from top-level legacy fields.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If calibration values cannot be decoded.
    """
    poni_path = Path(path).expanduser()
    if not poni_path.exists():
        raise FileNotFoundError(f"PONI file not found: {poni_path}")

    raw_fields: dict[str, str] = {}

    for line in poni_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue

        key, value = stripped.split(":", 1)
        raw_fields[_normalize_key(key)] = value.strip()

    detector_config = _parse_detector_config(
        _get_field(raw_fields, "Detector_config")
    )

    pixel1 = _first_float(
        _get_field(raw_fields, "PixelSize1"),
        _config_value(detector_config, "pixel1"),
    )
    pixel2 = _first_float(
        _get_field(raw_fields, "PixelSize2"),
        _config_value(detector_config, "pixel2"),
    )

    max_shape = _coerce_shape(
        _config_value(detector_config, "max_shape")
        or _config_value(detector_config, "shape")
    )

    return PoniGeometry(
        path=poni_path,
        version=_get_field(raw_fields, "poni_version"),
        detector=_get_field(raw_fields, "Detector"),
        detector_config=detector_config,
        dist=_first_float(_get_field(raw_fields, "Distance"), _get_field(raw_fields, "Dist")),
        poni1=_first_float(_get_field(raw_fields, "Poni1")),
        poni2=_first_float(_get_field(raw_fields, "Poni2")),
        rot1=_first_float(_get_field(raw_fields, "Rot1")),
        rot2=_first_float(_get_field(raw_fields, "Rot2")),
        rot3=_first_float(_get_field(raw_fields, "Rot3")),
        wavelength=_first_float(_get_field(raw_fields, "Wavelength")),
        pixel1=pixel1,
        pixel2=pixel2,
        max_shape=max_shape,
    )


def _normalize_key(key: str) -> str:
    """Normalize a PONI field name for case-insensitive lookup."""
    return key.strip().lower().replace("_", "").replace("-", "")


def _get_field(fields: dict[str, str], key: str) -> str | None:
    """Return a normalized PONI field value when present."""
    return fields.get(_normalize_key(key))


def _parse_detector_config(value: str | None) -> dict[str, Any]:
    """Decode a PONI detector-config mapping from JSON or a Python literal."""
    if value is None:
        return {}

    value = value.strip()
    if not value or value.lower() in {"none", "null"}:
        return {}

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except Exception:
            continue

        if isinstance(parsed, dict):
            return parsed

    return {}


def _config_value(config: dict[str, Any], key: str) -> Any:
    """Fetch a detector-config value with case-insensitive key matching."""
    normalized = _normalize_key(key)
    for config_key, value in config.items():
        if _normalize_key(str(config_key)) == normalized:
            return value
    return None


def _first_float(*values: Any) -> float | None:
    """Return the first candidate that can be converted to ``float``."""
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _coerce_shape(value: Any) -> tuple[int, int] | None:
    """Normalize a detector shape candidate to ``(vertical, horizontal)``."""
    if value is None:
        return None

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            return None

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None

    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
