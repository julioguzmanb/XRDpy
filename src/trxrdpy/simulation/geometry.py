import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .utils import AxisRotation, RotationChain, apply_transform, compose_transforms


_CARTESIAN_AXIS_MAP = {
    "x":  np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "y":  np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "z":  np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


def _as_vector3(vector, name="vector"):
    vec = np.asarray(vector, dtype=float)
    if vec.shape != (3,):
        raise ValueError(f"{name} must have shape (3,). Got {vec.shape}.")
    return vec


def _coerce_axis(axis, name="axis"):
    """
    Accept either:
      - 'x', '-x', 'y', '-y', 'z', '-z'
      - array-like shape (3,)
    and return a normalized 3-vector.
    """
    if isinstance(axis, str):
        key = axis.strip().lower()
        if key not in _CARTESIAN_AXIS_MAP:
            raise ValueError(
                f"Unsupported {name}='{axis}'. "
                f"Use one of {sorted(_CARTESIAN_AXIS_MAP)} or a 3-vector."
            )
        return _CARTESIAN_AXIS_MAP[key].copy()

    vec = _as_vector3(axis, name=name)
    norm = np.linalg.norm(vec)
    if np.isclose(norm, 0.0):
        raise ValueError(f"{name} cannot be the zero vector.")
    return vec / norm


def _normalize_frame(frame):
    frame = str(frame).lower()
    if frame not in {"lab", "local"}:
        raise ValueError(f"frame must be 'lab' or 'local'. Got '{frame}'.")
    return frame


def transform_point(point, transform):
    """
    Transform a 3D point with a 4x4 homogeneous transform.
    """
    point = _as_vector3(point, name="point")
    return apply_transform(point[None, :], transform)[0]


def transform_direction(direction, transform, normalize=True):
    """
    Transform a 3D direction vector with only the rotational part
    of a 4x4 homogeneous transform.
    """
    direction = _as_vector3(direction, name="direction")
    rotation = np.asarray(transform, dtype=float)[:3, :3]
    out = direction @ rotation.T
    if normalize:
        norm = np.linalg.norm(out)
        if np.isclose(norm, 0.0):
            raise ValueError("Cannot normalize a zero direction vector.")
        out = out / norm
    return out


@dataclass
class Motor:
    """
    A single physical rotation axis.

    Parameters
    ----------
    name : str
        Motor name, e.g. 'omega', 'chi', 'phi', 'kappa', 'tth'.
    axis : array-like or str
        Axis direction. Accepted strings: 'x', '-x', 'y', '-y', 'z', '-z'.
    origin : array-like, shape (3,)
        A point through which the axis passes.
    frame : {'lab', 'local'}
        - 'lab': axis/origin are fixed in laboratory coordinates.
        - 'local': axis/origin are defined in the moving frame produced
          by previous motors in the chain.
    default_angle : float
        Default motor angle.
    degrees : bool
        Whether the angle is expressed in degrees.
    """
    name: str
    axis: np.ndarray
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame: str = "local"
    default_angle: float = 0.0
    degrees: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = str(self.name)
        self.axis = _coerce_axis(self.axis, name=f"{self.name}.axis")
        self.origin = _as_vector3(self.origin, name=f"{self.name}.origin")
        self.frame = _normalize_frame(self.frame)
        self.default_angle = float(self.default_angle)
        self.degrees = bool(self.degrees)
        self.metadata = dict(self.metadata)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Build a Motor from a dictionary.

        Required keys
        -------------
        - name
        - axis

        Optional keys
        -------------
        - origin
        - frame
        - default_angle
        - degrees
        - metadata
        """
        if not isinstance(data, dict):
            raise TypeError("Motor.from_dict expects a dictionary.")

        if "name" not in data:
            raise KeyError("Motor.from_dict requires key 'name'.")
        if "axis" not in data:
            raise KeyError("Motor.from_dict requires key 'axis'.")

        return cls(
            name=data["name"],
            axis=data["axis"],
            origin=data.get("origin", (0.0, 0.0, 0.0)),
            frame=data.get("frame", "local"),
            default_angle=data.get("default_angle", 0.0),
            degrees=data.get("degrees", True),
            metadata=data.get("metadata", {}),
        )

    def resolved_axis_origin(self, parent_transform=None):
        """
        Resolve axis and origin in lab coordinates.
        """
        if parent_transform is None or self.frame == "lab":
            return self.axis.copy(), self.origin.copy()

        axis_lab = transform_direction(self.axis, parent_transform, normalize=True)
        origin_lab = transform_point(self.origin, parent_transform)
        return axis_lab, origin_lab

    def as_axis_rotation(self, angle=None, parent_transform=None):
        """
        Return this motor as an AxisRotation in lab coordinates.
        """
        if angle is None:
            angle = self.default_angle

        axis_lab, origin_lab = self.resolved_axis_origin(parent_transform=parent_transform)

        return AxisRotation(
            axis=axis_lab,
            angle=angle,
            origin=origin_lab,
            name=self.name,
            degrees=self.degrees,
        )

    def as_transform(self, angle=None, parent_transform=None):
        """
        Return the 4x4 transform for this motor.
        """
        return self.as_axis_rotation(angle=angle, parent_transform=parent_transform).as_transform()


@dataclass
class MotorChain:
    """
    Ordered list of motors.

    Notes
    -----
    The first motor in the list is applied first.
    """
    motors: List[Motor] = field(default_factory=list)
    name: str = "chain"
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_list(cls, motors, name="chain", metadata=None):
        """
        Build a MotorChain from an ordered list of:
          - Motor instances
          - motor dictionaries accepted by Motor.from_dict(...)
        """
        chain = cls(
            motors=[],
            name=name,
            metadata={} if metadata is None else dict(metadata),
        )

        if motors is None:
            return chain

        for i, motor in enumerate(motors):
            if isinstance(motor, Motor):
                chain.add_motor(motor)
            elif isinstance(motor, dict):
                chain.add_motor(Motor.from_dict(motor))
            else:
                raise TypeError(
                    f"motors[{i}] must be a Motor or a dict compatible with Motor.from_dict(). "
                    f"Got {type(motor).__name__}."
                )

        return chain

    @classmethod
    def from_dict(cls, data, name="chain", metadata=None):
        """
        Build a MotorChain from either:
          - a list of motor definitions
          - a dict with keys:
                name
                motors
                metadata
        """
        if isinstance(data, cls):
            return data

        if isinstance(data, list):
            return cls.from_list(data, name=name, metadata=metadata)

        if not isinstance(data, dict):
            raise TypeError("MotorChain.from_dict expects a list, dict, or MotorChain.")

        chain_name = data.get("name", name)
        chain_metadata = {} if metadata is None else dict(metadata)
        chain_metadata.update(dict(data.get("metadata", {})))
        motors = data.get("motors", [])

        return cls.from_list(
            motors=motors,
            name=chain_name,
            metadata=chain_metadata,
        )

    def add_motor(self, motor):
        if not isinstance(motor, Motor):
            raise TypeError("motor must be an instance of Motor.")
        self.motors.append(motor)

    def extend(self, motors):
        for motor in motors:
            self.add_motor(motor)

    @property
    def motor_names(self):
        return [motor.name for motor in self.motors]

    def get_motor(self, name):
        for motor in self.motors:
            if motor.name == name:
                return motor
        raise KeyError(f"No motor named '{name}' in chain '{self.name}'.")

    def default_angles(self):
        return {motor.name: motor.default_angle for motor in self.motors}

    def _merged_angles(self, angles=None):
        merged = self.default_angles()
        if angles is not None:
            unknown = set(angles) - set(self.motor_names)
            if unknown:
                raise KeyError(f"Unknown motors in chain '{self.name}': {sorted(unknown)}")
            merged.update(angles)
        return merged

    def as_rotation_chain(self, angles: Optional[Dict[str, float]] = None):
        """
        Return a RotationChain with all motors resolved in lab coordinates.
        """
        angle_map = self._merged_angles(angles)
        chain = RotationChain()
        current_transform = np.eye(4)

        for motor in self.motors:
            axis_rotation = motor.as_axis_rotation(
                angle=angle_map[motor.name],
                parent_transform=current_transform,
            )
            chain.append(axis_rotation)
            current_transform = compose_transforms(current_transform, axis_rotation.as_transform())

        return chain

    def as_transform(self, angles: Optional[Dict[str, float]] = None):
        """
        Return the total 4x4 transform of the chain.
        """
        return self.as_rotation_chain(angles=angles).as_transform()

    def apply(self, points, angles: Optional[Dict[str, float]] = None):
        """
        Apply the chain to points of shape (..., 3).
        """
        return self.as_rotation_chain(angles=angles).apply(points)

    def resolved_motors(self, angles: Optional[Dict[str, float]] = None):
        """
        Return a debug-friendly description of each motor resolved in lab coordinates.
        """
        angle_map = self._merged_angles(angles)
        current_transform = np.eye(4)
        out = []

        for motor in self.motors:
            axis_lab, origin_lab = motor.resolved_axis_origin(parent_transform=current_transform)
            angle = angle_map[motor.name]
            transform = motor.as_transform(angle=angle, parent_transform=current_transform)

            out.append({
                "name": motor.name,
                "angle": angle,
                "frame": motor.frame,
                "axis_lab": axis_lab,
                "origin_lab": origin_lab,
            })

            current_transform = compose_transforms(current_transform, transform)

        return out


@dataclass
class DiffractometerGeometry:
    """
    Minimal container for sample and detector motor chains.
    """
    name: str = "diffractometer"
    sample: MotorChain = field(default_factory=lambda: MotorChain(name="sample"))
    detector: MotorChain = field(default_factory=lambda: MotorChain(name="detector"))
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        """
        Build a DiffractometerGeometry from a dictionary.

        Expected structure
        ------------------
        {
          "name": "my_geometry",
          "sample": [
            {"name": "omega", "axis": "z", "origin": [0,0,0], "frame": "lab"},
            {"name": "kappa", "axis": [0.7, 0, 0.7], "origin": [0,0,0], "frame": "local"}
          ],
          "detector": {
            "name": "detector",
            "motors": [
              {"name": "tth", "axis": "y", "origin": [0,0,0], "frame": "lab"}
            ]
          },
          "metadata": {...}
        }

        Notes
        -----
        - 'sample' and 'detector' may each be:
            * a MotorChain
            * a list of motor definitions
            * a dict with keys {'name', 'motors', 'metadata'}
        - The motor order is the list order.
        """
        if isinstance(data, cls):
            return data

        if not isinstance(data, dict):
            raise TypeError("DiffractometerGeometry.from_dict expects a dictionary or DiffractometerGeometry.")

        def _coerce_chain(chain_data, default_name):
            if isinstance(chain_data, MotorChain):
                return chain_data
            return MotorChain.from_dict(chain_data, name=default_name)

        geometry_name = data.get("name", "diffractometer")
        metadata = dict(data.get("metadata", {}))

        sample_chain = _coerce_chain(data.get("sample", []), default_name="sample")
        detector_chain = _coerce_chain(data.get("detector", []), default_name="detector")

        return cls(
            name=geometry_name,
            sample=sample_chain,
            detector=detector_chain,
            metadata=metadata,
        )

    def sample_transform(self, angles: Optional[Dict[str, float]] = None):
        return self.sample.as_transform(angles=angles)

    def detector_transform(self, angles: Optional[Dict[str, float]] = None):
        return self.detector.as_transform(angles=angles)

    def transforms(
        self,
        sample_angles: Optional[Dict[str, float]] = None,
        detector_angles: Optional[Dict[str, float]] = None,
    ):
        return {
            "sample": self.sample_transform(sample_angles),
            "detector": self.detector_transform(detector_angles),
        }

    def apply_to_sample(self, points, angles: Optional[Dict[str, float]] = None):
        return self.sample.apply(points, angles=angles)

    def apply_to_detector(self, points, angles: Optional[Dict[str, float]] = None):
        return self.detector.apply(points, angles=angles)


__all__ = [
    "Motor",
    "MotorChain",
    "DiffractometerGeometry",
    "transform_point",
    "transform_direction",
]