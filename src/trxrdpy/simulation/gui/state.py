"""Versioned, JSON-serializable state for the simulation GUI."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, TypeVar


GUI_STATE_VERSION = 14
AUTOSAVE_FILENAME = ".xrdpy_simulation_gui_last_session.json"


T = TypeVar("T")


def _dataclass_from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    """
    Build a dataclass instance from a dictionary while ignoring unknown keys.

    This makes session loading more robust if:
    - old autosave files are missing new fields,
    - newer autosave files contain fields not present in the current code,
    - the state schema is expanded again later.
    """
    if not isinstance(data, dict):
        return cls()

    valid_keys = {f.name for f in fields(cls)}
    filtered = {key: value for key, value in data.items() if key in valid_keys}
    return cls(**filtered)


def _swap_state_poni_fields(section: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of a GUI state section with detector axis coordinates swapped.
    """
    migrated = dict(section)
    if "poni1" in migrated and "poni2" in migrated:
        migrated["poni1"], migrated["poni2"] = migrated["poni2"], migrated["poni1"]
    return migrated


def _migrate_state_dict(data: dict[str, Any]) -> dict[str, Any]:
    """
    Migrate older GUI state dictionaries to the current detector-axis convention.
    """
    migrated = dict(data)

    try:
        old_version = int(migrated.get("state_version", 0))
    except (TypeError, ValueError):
        old_version = 0

    if old_version < 6:
        if isinstance(migrated.get("poly"), dict):
            migrated["poly"] = _swap_state_poni_fields(migrated["poly"])
        if isinstance(migrated.get("single"), dict):
            migrated["single"] = _swap_state_poni_fields(migrated["single"])

    ui = dict(migrated.get("ui") or {})
    if old_version < 13:
        import pyFAI.detectors
        for section in ('poly', 'single'):
            values = dict(migrated.get(section) or {})
            name = values.get('det_type', 'manual')
            if name.lower() not in ('manual', 'poni'):
                try:
                    detector = pyFAI.detectors.detector_factory(name)
                    values.update(pxsize_h=str(detector.pixel2), pxsize_v=str(detector.pixel1),
                                  num_px_h=str(detector.max_shape[1]), num_px_v=str(detector.max_shape[0]))
                except (ValueError, TypeError, IndexError):
                    pass
            migrated[section] = values
    poly = dict(migrated.get("poly") or {})
    if poly.get("func") in ("simulate_2d_continuous", "simulate_3d_continuous"):
        poly["func"] = poly["func"].replace("_continuous", "")
    poly.pop("cones", None)
    migrated["poly"] = poly
    if "active_tab" not in ui:
        index = ui.get("current_tab_index", 0)
        order = ("polycrystalline", "single_crystal", "session") if old_version < 12 else (
            "session", "polycrystalline", "single_crystal")
        ui["active_tab"] = order[index] if isinstance(index, int) and 0 <= index < 3 else "session"
        ui.setdefault("last_simulation_tab", ui["active_tab"] if ui["active_tab"] != "session"
                      else "polycrystalline")
        if old_version == 11 and ui.get("session_open"):
            ui["active_tab"] = "session"
    collapsed = []
    for key in ui.get("collapsed_sections", []):
        if key.startswith(("single/Custom Geometry Definition", "single/Geometry Scan Controls")):
            key = key.replace("single/", "single/Experiment Geometry/", 1)
        collapsed.append(key)
    ui["collapsed_sections"] = collapsed
    migrated["ui"] = ui

    return migrated


@dataclass
class UIState:
    """Top-level tab selection and user-provided session description."""
    current_tab_index: int = 0
    active_tab: str = "session"
    last_simulation_tab: str = "polycrystalline"
    session_name: str = ""
    session_notes: str = ""
    session_open: bool = False
    collapsed_sections: list[str] = field(default_factory=list)
    equivalent_reflections: dict[str, str] = field(default_factory=dict)
    geometry_conversion: dict[str, str] = field(default_factory=dict)
    scattering_conversion: dict[str, Any] = field(default_factory=dict)
    attenuation: dict[str, Any] = field(default_factory=dict)


@dataclass
class PathsState:
    """Most recently selected CIF paths for each simulation family."""
    poly_cif_file_path: str | None = None
    single_cif_file_path: str | None = None


@dataclass
class PolyState:
    """Persisted text and toggle values from the polycrystalline tab.

    Numeric inputs intentionally remain strings so a save/load round trip does
    not change user-entered notation before validation at run time.
    """
    func: str = "simulate_1d"
    cif_path: str = ""
    space_group: str = "167"
    qmax: str = "5"
    a: str = "4.954"
    b: str = "4.954"
    c: str = "14.01"
    alpha: str = "90"
    beta: str = "90"
    gamma: str = "120"
    energy: str = "15000"
    ebw: str = "1.5"
    title_digits: str = "2"
    det_type: str = "manual"
    poni_file: str = ""
    pxsize_h: str = "50e-6"
    pxsize_v: str = "50e-6"
    num_px_h: str = "2000"
    num_px_v: str = "2000"
    bin_h: str = "1"
    bin_v: str = "1"
    dist: str = "0.1"
    poni1: str = "0"
    poni2: str = "0"
    rotx: str = "0"
    roty: str = "0"
    rotz: str = "0"
    rotation_order: str = "zyx"
    ref_source: str = "CIF / lattice (auto from qmax)"
    q_hkls: str = ""
    d_hkls: str = ""
    hkls: str = ""
    x_axis: str = "q"
    lorpol: bool = True
    fwhm: str = "0.0"


@dataclass
class SingleState:
    """Persisted single-crystal, geometry, scan, and targeting controls."""
    parallel_processing: bool = False
    parallel_workers: str = "4"
    mapping_display: str = "points"
    require_detector_hit: bool = False
    reciprocal_backend: str = 'plotly'
    reciprocal_non_bragg: bool = True
    reciprocal_bandwidth: bool = True
    reciprocal_scan_enabled: bool = False
    reciprocal_scan_rows: list = field(default_factory=lambda: [['energy', '', '', '']])
    func: str = "simulate_2d"
    det_type: str = "manual"
    poni_file: str = ""
    pxsize_h: str = "50e-6"
    pxsize_v: str = "50e-6"
    num_px_h: str = "2000"
    num_px_v: str = "2000"
    bin_h: str = "1"
    bin_v: str = "1"
    dist: str = "0.1"
    poni1: str = "0"
    poni2: str = "0"
    det_rotx: str = "0"
    det_roty: str = "0"
    det_rotz: str = "0"
    det_rotation_order: str = "zyx"
    energy: str = "15000"
    ebw: str = "1.5"
    title_digits: str = "2"
    space_group: str = "167"
    qmax: str = "5"
    a: str = "4.954"
    b: str = "4.954"
    c: str = "14.01"
    alpha: str = "90"
    beta: str = "90"
    gamma: str = "120"
    use_custom_orientation: bool = False
    orientation_matrix: list[list[str]] = field(
        default_factory=lambda: [
            ["0", "0", "0"],
            ["0", "0", "0"],
            ["0", "0", "0"],
        ]
    )
    sam_rotx: str = "0"
    sam_roty: str = "0"
    sam_rotz: str = "0"
    q: str = ""
    d: str = ""
    names: str = ""
    extra_hkls: str = ""
    equiv: bool = False
    angle_range: str = "-90,90,5"
    param1: str = "rotx"
    param2: str = "roty"
    param1_range: str = "-90,90,5"
    mapping_extra_enabled: bool = False
    mapping_extra_rows: list = field(default_factory=list)
    mapping_rotation_order: str = 'xyz'
    mapping_backend: str = 'matplotlib'
    param2_range: str = "-90,90,5"
    geometry_mode: str = "Legacy Euler"
    geometry_kind: str = ""
    geometry_kwargs: str = ""
    geometry_sample_angles: str = ""
    geometry_detector_angles: str = ""
    custom_sample_chain: str = ""
    custom_detector_chain: str = ""
    geometry_motor1_name: str = "omega"
    geometry_motor1_range: str = "-90,90,5"
    geometry_motor2_name: str = "kappa"
    geometry_motor2_range: str = "-90,90,5"
    geometry_detector_scan_ranges: str = ""
    target_hkl: str = "[1,1,0]"
    target_pixel_h: str = "900"
    target_pixel_v: str = "900"
    target_pixel_tol: str = "20"
    eta_samples: str = "1441"
    phi_samples: str = "361"
    wrap_angles: bool = True
    inverse_detector_plot: bool = True
    inverse_2d_plot: bool = True
    inverse_3d_plot: bool = False


@dataclass
class MatrixToolState:
    """Persisted lattice and rotation values from the orientation-matrix tool."""
    space_group: str = "1"

    a: str = "1"
    b: str = "1"
    c: str = "1"

    alpha: str = "90"
    beta: str = "90"
    gamma: str = "90"

    orientation_matrix: list[list[str]] = field(
        default_factory=lambda: [
            ["0.0", "0.0", "0.0"],
            ["0.0", "0.0", "0.0"],
            ["0.0", "0.0", "0.0"],
        ]
    )

    rotx: str = "0"
    roty: str = "0"
    rotz: str = "0"

    rotation_mode: str = "Euler-like XYZ"
    kappa_tilt: str = "50"

    result_matrix_valid: bool = False
    result_matrix: list[list[str]] = field(
        default_factory=lambda: [
            ["", "", ""],
            ["", "", ""],
            ["", "", ""],
        ]
    )


@dataclass
class GuiState:
    """Complete versioned simulation-GUI session state.

    Nested dataclasses separate shared UI metadata, file paths, simulation-tab
    controls, and the orientation tool. :meth:`from_dict` tolerates missing and
    unknown fields so sessions survive schema evolution.
    """
    state_version: int = GUI_STATE_VERSION
    saved_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    geometry: str = ""

    ui: UIState = field(default_factory=UIState)
    paths: PathsState = field(default_factory=PathsState)
    poly: PolyState = field(default_factory=PolyState)
    single: SingleState = field(default_factory=SingleState)
    matrix_tool: MatrixToolState = field(default_factory=MatrixToolState)

    log: str = ""

    def touch(self) -> None:
        """Update ``saved_at`` to the current local time."""
        self.saved_at = datetime.now().isoformat(timespec="seconds")

    def to_dict(self, *, include_log: bool = True) -> dict[str, Any]:
        """Serialize the complete state, optionally excluding the GUI log."""
        self.touch()
        data = asdict(self)

        if not include_log:
            data.pop("log", None)

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GuiState:
        """Build current state from a possibly older or partial dictionary."""
        if not isinstance(data, dict):
            return cls()

        data = _migrate_state_dict(data)

        ui = _dataclass_from_dict(UIState, data.get("ui"))
        paths = _dataclass_from_dict(PathsState, data.get("paths"))
        poly = _dataclass_from_dict(PolyState, data.get("poly"))
        single = _dataclass_from_dict(SingleState, data.get("single"))
        matrix_tool = _dataclass_from_dict(MatrixToolState, data.get("matrix_tool"))

        return cls(
            state_version=GUI_STATE_VERSION,
            saved_at=str(data.get("saved_at", datetime.now().isoformat(timespec="seconds"))),
            geometry=str(data.get("geometry", "")),
            ui=ui,
            paths=paths,
            poly=poly,
            single=single,
            matrix_tool=matrix_tool,
            log=str(data.get("log", "")),
        )
