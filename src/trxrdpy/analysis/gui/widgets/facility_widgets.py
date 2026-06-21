"""
Reusable facility-related widgets for the analysis GUI.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtWidgets import QComboBox

from trxrdpy.analysis.gui.services import FacilityService


class FacilitySelector(QComboBox):
    """Combo box for selecting the active analysis facility.

    The visible labels follow the legacy GUI:
    - Spring-8 SACLA
    - MAX IV FemtoMAX
    - ESRF-ID09

    The emitted values are internal facility keys:
    - SACLA
    - FemtoMAX
    - ID09
    """

    def __init__(
        self,
        facility_service: FacilityService,
        on_facility_changed: Optional[Callable[[str], None]] = None,
        parent=None,
    ):
        """Initialize the object and its runtime state."""
        super().__init__(parent)

        self.facility_service = facility_service
        self.on_facility_changed = on_facility_changed

        self.addItems(self.facility_service.labels())
        self.currentTextChanged.connect(self._on_label_changed)

        self._on_label_changed(self.currentText())

    def current_facility(self) -> str:
        """Return current facility."""
        return self.facility_service.key_from_label(self.currentText())

    def set_facility(self, facility_key: str):
        """Select a facility by backend key and update the visible label."""
        label = self.facility_service.label_from_key(facility_key)
        index = self.findText(label)

        if index < 0:
            raise ValueError(f"Unknown facility label: {label!r}")

        self.setCurrentIndex(index)

    def _on_label_changed(self, facility_label: str):
        """Handle the label changed event."""
        facility_key = self.facility_service.key_from_label(facility_label)

        if self.on_facility_changed is not None:
            self.on_facility_changed(facility_key)
