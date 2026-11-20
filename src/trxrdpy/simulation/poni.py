"""
Utilities for reading and writing PONI detector geometry files.

Higher-level objects decide how these geometry values are applied to a simulation.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
import math
import os
import tempfile
from typing import Any


__all__ = ["PoniGeometry", "read_poni_file", "write_poni_file"]



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


def write_poni_file(
    path, *, dist, poni1, poni2, pixel1, pixel2, max_shape,
    rot1=0, rot2=0, rot3=0, energy_ev=None, wavelength=None,
    orientation=3, overwrite=False,
):
    """Write uniform flat-detector geometry in the standard PONI 2.1 format.

    Distances, pixels, and wavelength are in metres; rotations are pyFAI
    rotations in radians, not Cartesian simulation Euler angles. ``max_shape``
    is (vertical rows, horizontal columns). Supply exactly one of energy [eV]
    or wavelength [m]. Existing files are protected unless overwrite=True.
    """
    from pyFAI.io.ponifile import PoniFile
    from .utils import energy_to_wavelength

    def number(name, value, *, positive=False):
        try:
            parsed = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"{name} must be a finite number.") from None
        if isinstance(value, bool) or not math.isfinite(parsed) or (positive and parsed <= 0):
            raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite.")
        return parsed

    config = {name: number(name, value, positive=name == "dist") for name, value in (
        ("dist", dist), ("poni1", poni1), ("poni2", poni2),
        ("rot1", rot1), ("rot2", rot2), ("rot3", rot3))}
    if not isinstance(max_shape, (tuple, list)) or len(max_shape) != 2:
        raise ValueError("max_shape must contain vertical and horizontal pixel counts.")
    shape = []
    for axis, size in zip(("vertical", "horizontal"), max_shape):
        size = number(f"{axis} pixel count", size, positive=True)
        if not size.is_integer():
            raise ValueError(f"{axis} pixel count must be an integer.")
        shape.append(int(size))
    orientation = number("orientation", orientation)
    if orientation not in (1, 2, 3, 4):
        raise ValueError("Detector orientation must be 1, 2, 3, or 4.")
    if (energy_ev is None) == (wavelength is None):
        raise ValueError("Supply exactly one of energy_ev or wavelength.")
    if energy_ev is not None:
        wavelength = energy_to_wavelength(number("Energy [eV]", energy_ev, positive=True))
    config.update(
        poni_version=2.1, detector="Detector",
        detector_config={"pixel1": number("pixel1", pixel1, positive=True),
                         "pixel2": number("pixel2", pixel2, positive=True),
                         "max_shape": shape, "orientation": int(orientation)},
        wavelength=number("Wavelength [m]", wavelength, positive=True),
    )
    config["distance"] = config.pop("dist")
    poni = PoniFile(config)
    path = Path(path).expanduser()
    if not path.suffix:
        path = path.with_suffix(".poni")
    if path.suffix.lower() != ".poni":
        raise ValueError("The output file must have the .poni extension.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".poni-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write("# XRDpy: manually defined uniform flat-detector geometry.\n")
            poni.write(stream)
        if overwrite:
            os.replace(temporary, path)
        else:
            # Publish without a check/replace race or a partly written target.
            os.link(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
