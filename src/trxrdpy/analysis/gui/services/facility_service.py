"""
Facility selection and facility-specific metadata for the analysis GUI.

This service stays independent from Qt.
It defines the stable internal facility keys and their legacy display labels.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Facility:
    """Identify a supported experimental facility and its GUI label."""
    key: str
    label: str
    description: str = ""


class FacilityService:
    """Central registry of supported analysis facilities.

    Internal code should use stable keys:
    - SACLA
    - FemtoMAX
    - ID09

    GUI widgets should display legacy labels:
    - Spring-8 SACLA
    - MAX IV FemtoMAX
    - ESRF-ID09
    """

    SACLA = "SACLA"
    FEMTOMAX = "FemtoMAX"
    ID09 = "ID09"

    def __init__(self):
        """Initialize the object and its runtime state."""
        self._facilities = {
            self.SACLA: Facility(
                key=self.SACLA,
                label="Spring-8 SACLA",
                description="Spring-8 SACLA",
            ),
            self.FEMTOMAX: Facility(
                key=self.FEMTOMAX,
                label="MAX IV FemtoMAX",
                description="MAX IV FemtoMAX",
            ),
            self.ID09: Facility(
                key=self.ID09,
                label="ESRF-ID09",
                description="ESRF-ID09",
            ),
        }

    def keys(self) -> list[str]:
        """Return supported facility backend keys in display order."""
        return list(self._facilities.keys())

    def names(self) -> list[str]:
        """Backward-compatible alias for facility keys."""
        return self.keys()

    def labels(self) -> list[str]:
        """Return user-facing facility labels in display order."""
        return [facility.label for facility in self._facilities.values()]

    def get(self, key: str) -> Facility:
        """Return a facility definition or raise ``ValueError`` for an unknown key."""
        try:
            return self._facilities[key]
        except KeyError as exc:
            valid = ", ".join(self.keys())
            raise ValueError(
                f"Unknown facility {key!r}. Valid facilities are: {valid}"
            ) from exc

    def key_from_label(self, label: str) -> str:
        """Return key from label."""
        for key, facility in self._facilities.items():
            if facility.label == label:
                return key

        valid = ", ".join(self.labels())
        raise ValueError(
            f"Unknown facility label {label!r}. Valid labels are: {valid}"
        )

    def label_from_key(self, key: str) -> str:
        """Label from key."""
        return self.get(key).label

    def is_supported(self, key: str) -> bool:
        """Return whether supported."""
        return key in self._facilities
