from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import diffractometers
from ...geometry import DiffractometerGeometry


_MOTOR_LABELS = {
    "omega": "Sample base rotation",
    "chi": "Sample tilt",
    "phi": "Sample spin",
    "kappa": "Tilted arm rotation",
    "tth": "Detector arm",
    "two_theta": "Detector arm",
    "delta": "Detector arm",
    "gamma": "Detector arm",
    "mu": "Sample base rotation",
    "eta": "Sample tilt",
}

_MOTOR_DESCRIPTIONS = {
    "omega": "Primary sample rotation around a laboratory-frame axis.",
    "chi": "Sample tilt with respect to the base rotation stage.",
    "phi": "Final sample spin around the local sample axis.",
    "kappa": "Rotation around the tilted kappa axis.",
    "tth": "Detector two-theta arm rotation.",
    "two_theta": "Detector two-theta arm rotation.",
    "delta": "Detector arm rotation.",
    "gamma": "Detector arm rotation.",
    "mu": "Primary sample rotation around a laboratory-frame axis.",
    "eta": "Sample tilt with respect to the base stage.",
}


@dataclass(frozen=True)
class MotorInfo:
    name: str
    label: str
    description: str
    default: float = 0.0
    frame: str | None = None
    axis: Any = None
    origin: Any = None


@dataclass(frozen=True)
class GeometryInfo:
    name: str
    display_name: str
    summary: str
    sample_motors: tuple[MotorInfo, ...] = ()
    detector_motors: tuple[MotorInfo, ...] = ()
    uses_legacy_controls: bool = False
    constructor_kwargs_template: dict[str, Any] | None = None


class SimulationService:
    """
    GUI-facing service layer for simulation metadata.

    This class centralizes geometry discovery and converts diffractometer
    definitions into user-facing descriptions that the GUI can render without
    hard-coding motor names all over the widgets.
    """

    def __init__(self) -> None:
        self._geometries = self._build_geometry_catalog()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_geometries(self) -> list[GeometryInfo]:
        ordered_names = ["legacy_euler"]
        ordered_names.extend(
            name for name in sorted(self._geometries) if name != "legacy_euler"
        )
        return [self._geometries[name] for name in ordered_names]

    def available_geometry_names(self) -> list[str]:
        return [geometry.name for geometry in self.list_geometries()]

    def get_geometry(self, name: str) -> GeometryInfo:
        try:
            return self._geometries[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._geometries))
            raise ValueError(
                f"Unknown geometry '{name}'. Available geometries: {available}"
            ) from exc

    def build_predefined_geometry(
        self,
        name: str,
        **kwargs: Any,
    ) -> DiffractometerGeometry:
        if name == "legacy_euler":
            raise ValueError(
                "The 'legacy_euler' mode does not build a geometry object. "
                "It uses the legacy explicit Euler controls."
            )
        return diffractometers.make_diffractometer(name, **kwargs)

    def diffractometer_to_info(
        self,
        geometry: DiffractometerGeometry,
        *,
        name: str | None = None,
        constructor_kwargs_template: dict[str, Any] | None = None,
    ) -> GeometryInfo:
        geom_dict = diffractometers.diffractometer_to_dict(geometry)

        geometry_name = name or geom_dict.get("name") or "geometry"
        display_name = self._display_name_for(geometry_name)

        sample_motors = self._motor_infos_from_chain_dict(geom_dict.get("sample", {}))
        detector_motors = self._motor_infos_from_chain_dict(geom_dict.get("detector", {}))

        return GeometryInfo(
            name=geometry_name,
            display_name=display_name,
            summary=self._summary_for(
                geometry_name,
                sample_motors=sample_motors,
                detector_motors=detector_motors,
            ),
            sample_motors=sample_motors,
            detector_motors=detector_motors,
            uses_legacy_controls=False,
            constructor_kwargs_template=constructor_kwargs_template or {},
        )

    def default_sample_angles(self, name: str) -> dict[str, float]:
        geometry = self.get_geometry(name)
        return {motor.name: motor.default for motor in geometry.sample_motors}

    def default_detector_angles(self, name: str) -> dict[str, float]:
        geometry = self.get_geometry(name)
        return {motor.name: motor.default for motor in geometry.detector_motors}

    def default_constructor_kwargs(self, name: str) -> dict[str, Any]:
        geometry = self.get_geometry(name)
        return dict(geometry.constructor_kwargs_template or {})

    # ------------------------------------------------------------------
    # Catalog construction
    # ------------------------------------------------------------------
    def _build_geometry_catalog(self) -> dict[str, GeometryInfo]:
        catalog: dict[str, GeometryInfo] = {
            "legacy_euler": GeometryInfo(
                name="legacy_euler",
                display_name="Legacy Euler",
                summary=(
                    "Compatibility mode using the older explicit sample and detector "
                    "Euler-angle controls from the legacy GUI."
                ),
                uses_legacy_controls=True,
                constructor_kwargs_template={},
            )
        }

        for kind in self._available_geometry_kinds():
            try:
                kwargs_template = self._default_kwargs_template_for(kind)
                geometry = diffractometers.make_diffractometer(kind, **kwargs_template)
                catalog[kind] = self.diffractometer_to_info(
                    geometry,
                    name=kind,
                    constructor_kwargs_template=kwargs_template,
                )
            except Exception:
                fallback = self._fallback_geometry_info(kind)
                if fallback is not None:
                    catalog[kind] = fallback

        return catalog

    def _available_geometry_kinds(self) -> list[str]:
        try:
            return list(diffractometers.available_diffractometers())
        except Exception:
            registry = getattr(diffractometers, "DIFFRACTOMETER_REGISTRY", {})
            return sorted(registry)

    def _default_kwargs_template_for(self, kind: str) -> dict[str, Any]:
        if kind == "kappa":
            return {"kappa_tilt_deg": 50.0}
        return {}

    def _fallback_geometry_info(self, kind: str) -> GeometryInfo | None:
        if kind == "euler":
            return GeometryInfo(
                name="euler",
                display_name="Eulerian",
                summary=(
                    "Standard Eulerian geometry with a sample rotation chain and "
                    "a detector arm rotation."
                ),
                sample_motors=(
                    MotorInfo(
                        name="omega",
                        label="Sample base rotation",
                        description="Primary sample rotation around a laboratory-frame axis.",
                    ),
                    MotorInfo(
                        name="chi",
                        label="Sample tilt",
                        description="Sample tilt with respect to the base rotation stage.",
                    ),
                    MotorInfo(
                        name="phi",
                        label="Sample spin",
                        description="Final sample spin around the local sample axis.",
                    ),
                ),
                detector_motors=(
                    MotorInfo(
                        name="tth",
                        label="Detector arm",
                        description="Detector two-theta arm rotation.",
                    ),
                ),
                constructor_kwargs_template={},
            )

        if kind == "kappa":
            return GeometryInfo(
                name="kappa",
                display_name="Kappa",
                summary=(
                    "Kappa geometry with omega, kappa, and phi on the sample side, "
                    "plus a detector two-theta arm."
                ),
                sample_motors=(
                    MotorInfo(
                        name="omega",
                        label="Sample base rotation",
                        description="Primary sample rotation around a laboratory-frame axis.",
                    ),
                    MotorInfo(
                        name="kappa",
                        label="Tilted arm rotation",
                        description="Rotation around the tilted kappa axis.",
                    ),
                    MotorInfo(
                        name="phi",
                        label="Sample spin",
                        description="Final sample spin around the local sample axis.",
                    ),
                ),
                detector_motors=(
                    MotorInfo(
                        name="tth",
                        label="Detector arm",
                        description="Detector two-theta arm rotation.",
                    ),
                ),
                constructor_kwargs_template={"kappa_tilt_deg": 50.0},
            )

        return None

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def _motor_infos_from_chain_dict(self, chain_dict: dict[str, Any]) -> tuple[MotorInfo, ...]:
        motors = chain_dict.get("motors", [])
        result: list[MotorInfo] = []

        for motor in motors:
            result.append(
                MotorInfo(
                    name=str(motor.get("name", "")),
                    label=self._motor_label(motor),
                    description=self._motor_description(motor),
                    default=float(motor.get("default_angle", 0.0) or 0.0),
                    frame=motor.get("frame"),
                    axis=motor.get("axis"),
                    origin=motor.get("origin"),
                )
            )

        return tuple(result)

    def _motor_label(self, motor_dict: dict[str, Any]) -> str:
        metadata = motor_dict.get("metadata", {}) or {}
        name = str(motor_dict.get("name", ""))
        return str(metadata.get("label") or _MOTOR_LABELS.get(name, name))

    def _motor_description(self, motor_dict: dict[str, Any]) -> str:
        metadata = motor_dict.get("metadata", {}) or {}
        name = str(motor_dict.get("name", ""))

        if metadata.get("description"):
            return str(metadata["description"])

        frame = motor_dict.get("frame")
        axis = motor_dict.get("axis")
        base = _MOTOR_DESCRIPTIONS.get(name, "Geometry motor.")
        extras = []

        if frame:
            extras.append(f"frame={frame}")
        if axis is not None:
            extras.append(f"axis={axis}")

        if extras:
            return f"{base} ({', '.join(extras)})"
        return base

    def _display_name_for(self, name: str) -> str:
        if name.lower() == "kappa":
            return "Kappa"
        if name.lower() == "euler":
            return "Eulerian"
        return name.replace("_", " ").title()

    def _summary_for(
        self,
        name: str,
        *,
        sample_motors: tuple[MotorInfo, ...],
        detector_motors: tuple[MotorInfo, ...],
    ) -> str:
        lowered = name.lower()

        if lowered == "euler":
            return (
                "Standard Eulerian geometry. The sample is controlled by "
                f"{self._motor_list_text(sample_motors)} and the detector by "
                f"{self._motor_list_text(detector_motors)}."
            )

        if lowered == "kappa":
            return (
                "Kappa geometry. The sample is controlled by "
                f"{self._motor_list_text(sample_motors)} and the detector by "
                f"{self._motor_list_text(detector_motors)}."
            )

        if sample_motors or detector_motors:
            return (
                f"{self._display_name_for(name)} geometry with sample motors "
                f"{self._motor_list_text(sample_motors)} and detector motors "
                f"{self._motor_list_text(detector_motors)}."
            )

        return f"{self._display_name_for(name)} geometry."

    @staticmethod
    def _motor_list_text(motors: tuple[MotorInfo, ...]) -> str:
        if not motors:
            return "none"

        names = [motor.name for motor in motors]
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return f"{', '.join(names[:-1])}, and {names[-1]}"