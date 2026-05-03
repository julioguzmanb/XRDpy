from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


GUI_STATE_VERSION = 4
AUTOSAVE_FILENAME = ".xrdpy_simulation_gui_last_session.json"


@dataclass
class UIState:
    current_tab_index: int = 0
    session_name: str = ""
    session_notes: str = ""


@dataclass
class PathsState:
    poly_cif_file_path: str | None = None
    single_cif_file_path: str | None = None


@dataclass
class PolyState:
    func: str = "simulate_1d"
    cif_path: str = ""
    space_group: str = "167"
    qmax: str = "10"
    a: str = "4.954"
    b: str = "4.954"
    c: str = "14.01"
    alpha: str = "90"
    beta: str = "90"
    gamma: str = "120"
    energy: str = "15000"
    ebw: str = "1.5"
    det_type: str = "manual"
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
    ref_source: str = "CIF / lattice (auto from qmax)"
    q_hkls: str = ""
    d_hkls: str = ""
    hkls: str = ""
    cones: str = "30"
    x_axis: str = "q"
    lorpol: bool = True
    fwhm: str = "0.0"


@dataclass
class SingleState:
    func: str = "simulate_2d"
    det_type: str = "manual"
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
    energy: str = "15000"
    ebw: str = "1.5"
    space_group: str = "167"
    qmax: str = "10"
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


@dataclass
class GuiState:
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
        self.saved_at = datetime.now().isoformat(timespec="seconds")

    def to_dict(self, *, include_log: bool = True) -> dict[str, Any]:
        self.touch()
        data = asdict(self)
        if not include_log:
            data.pop("log", None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GuiState:
        if not data:
            return cls()

        ui = UIState(**data.get("ui", {}))
        paths = PathsState(**data.get("paths", {}))
        poly = PolyState(**data.get("poly", {}))
        single = SingleState(**data.get("single", {}))
        matrix_tool = MatrixToolState(**data.get("matrix_tool", {}))

        return cls(
            state_version=int(data.get("state_version", GUI_STATE_VERSION)),
            saved_at=str(data.get("saved_at", datetime.now().isoformat(timespec="seconds"))),
            geometry=str(data.get("geometry", "")),
            ui=ui,
            paths=paths,
            poly=poly,
            single=single,
            matrix_tool=matrix_tool,
            log=str(data.get("log", "")),
        )