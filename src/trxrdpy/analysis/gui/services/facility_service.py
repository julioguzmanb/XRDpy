"""
Facility selection and facility-specific metadata for the analysis GUI.

This service stays independent from Qt.
It defines the stable internal facility keys and their legacy display labels.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Facility:
    """Identify a supported experimental facility and its GUI label.

    Attributes
    ----------
    key : str
        Stable backend identifier stored in GUI state.
    label : str
        Human-readable name displayed in selectors.
    description : str
        Optional explanatory text for help panels or tooltips.
    """
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

    Attributes
    ----------
    _facilities : dict
        Ordered mapping from stable facility keys to :class:`Facility` records.
    """

    SACLA = "SACLA"
    FEMTOMAX = "FemtoMAX"
    ID09 = "ID09"

    def __init__(self):
        """Initialize configuration, normalize inputs, and create the object runtime state."""
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
        """Return stable facility keys through the legacy ``names`` method."""
        return self.keys()

    def labels(self) -> list[str]:
        """Return all user-facing facility labels in configured display order."""
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
        """Translate a displayed facility label to its stable backend key."""
        for key, facility in self._facilities.items():
            if facility.label == label:
                return key

        valid = ", ".join(self.labels())
        raise ValueError(
            f"Unknown facility label {label!r}. Valid labels are: {valid}"
        )

    def label_from_key(self, key: str) -> str:
        """Return the user-facing label for a stable facility key."""
        return self.get(key).label

    def is_supported(self, key: str) -> bool:
        """Return whether ``key`` identifies a registered facility backend."""
        return key in self._facilities
