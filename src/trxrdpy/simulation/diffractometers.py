import numpy as np

from .geometry import Motor, MotorChain, DiffractometerGeometry
from .utils import normalize_vector


_CARTESIAN_AXIS_MAP = {
    "x":  np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "y":  np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "z":  np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


def _as_axis(axis, name="axis"):
    """
    Convert an axis specification into a normalized 3-vector.

    Accepted forms
    --------------
    - 'x', '-x', 'y', '-y', 'z', '-z'
    - array-like of shape (3,)
    """
    if isinstance(axis, str):
        key = axis.strip().lower()
        if key not in _CARTESIAN_AXIS_MAP:
            raise ValueError(
                f"Unsupported {name}='{axis}'. "
                f"Use one of {sorted(_CARTESIAN_AXIS_MAP)} or a 3-vector."
            )
        return _CARTESIAN_AXIS_MAP[key].copy()

    arr = np.asarray(axis, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector. Got shape {arr.shape}.")
    return normalize_vector(arr)


def _validate_cartesian_order(rotation_order):
    order = str(rotation_order).lower()
    if len(order) != 3 or set(order) != {"x", "y", "z"}:
        raise ValueError(
            f"rotation_order must be a permutation of 'xyz'. Got '{rotation_order}'."
        )
    return order


def _build_euler_chain(
    rotation_order,
    prefix,
    origin=(0.0, 0.0, 0.0),
    frame="lab",
    defaults=None,
    name=None,
):
    """
    Build a MotorChain that reproduces the current Euler-like x/y/z approach.

    Notes
    -----
    - All axes are Cartesian.
    - By default the axes are LAB-fixed, which matches the current apply_rotation logic.
    - The order of motors in the chain is exactly the order in rotation_order.
    """
    order = _validate_cartesian_order(rotation_order)
    defaults = {} if defaults is None else dict(defaults)
    origin = np.asarray(origin, dtype=float)

    axis_label_to_motor_name = {
        "x": f"{prefix}_rotx",
        "y": f"{prefix}_roty",
        "z": f"{prefix}_rotz",
    }

    motors = []
    for axis_label in order:
        motor_name = axis_label_to_motor_name[axis_label]
        motors.append(
            Motor(
                name=motor_name,
                axis=_as_axis(axis_label),
                origin=origin,
                frame=frame,
                default_angle=defaults.get(motor_name, 0.0),
            )
        )

    return MotorChain(motors=motors, name=name or prefix)


def make_euler_geometry(
    name="euler",
    sample_rotation_order="xyz",
    detector_rotation_order="xyz",
    sample_frame="lab",
    detector_frame="lab",
    sample_origin=(0.0, 0.0, 0.0),
    detector_origin=(0.0, 0.0, 0.0),
    sample_defaults=None,
    detector_defaults=None,
):
    """
    Predefined geometry that reproduces the current Euler-style approach.

    Sample motors
    -------------
    sam_rotx, sam_roty, sam_rotz

    Detector motors
    ---------------
    det_rotx, det_roty, det_rotz

    Important
    ---------
    With the default frame='lab', this mirrors the current implementation:
    three fixed Cartesian axes applied in the chosen order.
    """
    sample_chain = _build_euler_chain(
        rotation_order=sample_rotation_order,
        prefix="sam",
        origin=sample_origin,
        frame=sample_frame,
        defaults=sample_defaults,
        name="sample",
    )

    detector_chain = _build_euler_chain(
        rotation_order=detector_rotation_order,
        prefix="det",
        origin=detector_origin,
        frame=detector_frame,
        defaults=detector_defaults,
        name="detector",
    )

    return DiffractometerGeometry(
        name=name,
        sample=sample_chain,
        detector=detector_chain,
        metadata={
            "kind": "euler",
            "sample_rotation_order": sample_rotation_order,
            "detector_rotation_order": detector_rotation_order,
            "sample_frame": sample_frame,
            "detector_frame": detector_frame,
        },
    )


def make_legacy_euler_geometry(**kwargs):
    """
    Alias for make_euler_geometry.
    """
    return make_euler_geometry(**kwargs)


def make_kappa_geometry(
    name="kappa",
    kappa_tilt_deg=50.0,
    sample_origin=(0.0, 0.0, 0.0),
    detector_origin=(0.0, 0.0, 0.0),
    omega_axis="z",
    kappa_axis=None,
    phi_axis="z",
    two_theta_axis="y",
    sample_defaults=None,
    detector_defaults=None,
):
    """
    Predefined generic kappa geometry.

    Convention used here
    --------------------
    Sample chain:
        1) omega : LAB-fixed axis
        2) kappa : LOCAL axis
        3) phi   : LOCAL axis

    Detector chain:
        1) tth   : LAB-fixed axis

    Default axes
    ------------
    - omega axis  : +z
    - phi axis    : +z
    - detector tth axis : +y
    - kappa axis  : if not explicitly given, it is taken as a LOCAL axis
                    in the x-z plane:
                        [sin(alpha), 0, cos(alpha)]
                    where alpha = kappa_tilt_deg
    """
    sample_defaults = {} if sample_defaults is None else dict(sample_defaults)
    detector_defaults = {} if detector_defaults is None else dict(detector_defaults)

    omega_axis_vec = _as_axis(omega_axis, name="omega_axis")
    phi_axis_vec = _as_axis(phi_axis, name="phi_axis")
    tth_axis_vec = _as_axis(two_theta_axis, name="two_theta_axis")

    if kappa_axis is None:
        alpha = np.deg2rad(kappa_tilt_deg)
        kappa_axis_vec = normalize_vector(np.array([np.sin(alpha), 0.0, np.cos(alpha)]))
    else:
        kappa_axis_vec = _as_axis(kappa_axis, name="kappa_axis")

    sample_origin = np.asarray(sample_origin, dtype=float)
    detector_origin = np.asarray(detector_origin, dtype=float)

    sample_chain = MotorChain(
        motors=[
            Motor(
                name="omega",
                axis=omega_axis_vec,
                origin=sample_origin,
                frame="lab",
                default_angle=sample_defaults.get("omega", 0.0),
            ),
            Motor(
                name="kappa",
                axis=kappa_axis_vec,
                origin=sample_origin,
                frame="local",
                default_angle=sample_defaults.get("kappa", 0.0),
            ),
            Motor(
                name="phi",
                axis=phi_axis_vec,
                origin=sample_origin,
                frame="local",
                default_angle=sample_defaults.get("phi", 0.0),
            ),
        ],
        name="sample",
        metadata={
            "kind": "kappa_sample",
            "kappa_tilt_deg": float(kappa_tilt_deg),
        },
    )

    detector_chain = MotorChain(
        motors=[
            Motor(
                name="tth",
                axis=tth_axis_vec,
                origin=detector_origin,
                frame="lab",
                default_angle=detector_defaults.get("tth", 0.0),
            )
        ],
        name="detector",
        metadata={"kind": "single_circle_detector"},
    )

    return DiffractometerGeometry(
        name=name,
        sample=sample_chain,
        detector=detector_chain,
        metadata={
            "kind": "kappa",
            "kappa_tilt_deg": float(kappa_tilt_deg),
        },
    )


def motor_to_dict(motor):
    """
    Serialize a Motor to a dictionary compatible with Motor.from_dict(...).
    """
    if not isinstance(motor, Motor):
        raise TypeError("motor_to_dict expects a Motor instance.")

    return {
        "name": motor.name,
        "axis": np.asarray(motor.axis, dtype=float).tolist(),
        "origin": np.asarray(motor.origin, dtype=float).tolist(),
        "frame": motor.frame,
        "default_angle": float(motor.default_angle),
        "degrees": bool(motor.degrees),
        "metadata": dict(motor.metadata),
    }


def motor_chain_to_dict(chain):
    """
    Serialize a MotorChain to a dictionary compatible with MotorChain.from_dict(...).
    """
    if not isinstance(chain, MotorChain):
        raise TypeError("motor_chain_to_dict expects a MotorChain instance.")

    return {
        "name": chain.name,
        "motors": [motor_to_dict(m) for m in chain.motors],
        "metadata": dict(chain.metadata),
    }


def diffractometer_to_dict(geometry):
    """
    Serialize a DiffractometerGeometry to a dictionary compatible with
    DiffractometerGeometry.from_dict(...).
    """
    if not isinstance(geometry, DiffractometerGeometry):
        raise TypeError("diffractometer_to_dict expects a DiffractometerGeometry instance.")

    return {
        "name": geometry.name,
        "sample": motor_chain_to_dict(geometry.sample),
        "detector": motor_chain_to_dict(geometry.detector),
        "metadata": dict(geometry.metadata),
    }


DIFFRACTOMETER_REGISTRY = {
    "euler": make_euler_geometry,
    "legacy_euler": make_legacy_euler_geometry,
    "kappa": make_kappa_geometry,
}


def available_diffractometers():
    """
    Return the sorted list of registered predefined diffractometer names.
    """
    return sorted(DIFFRACTOMETER_REGISTRY)


def make_diffractometer(kind, **kwargs):
    """
    Build a predefined diffractometer by name.

    Parameters
    ----------
    kind : str
        One of the names returned by available_diffractometers().

    Returns
    -------
    DiffractometerGeometry
    """
    key = str(kind).strip().lower()
    if key not in DIFFRACTOMETER_REGISTRY:
        raise KeyError(
            f"Unknown diffractometer '{kind}'. "
            f"Available: {available_diffractometers()}"
        )
    return DIFFRACTOMETER_REGISTRY[key](**kwargs)


__all__ = [
    "make_euler_geometry",
    "make_legacy_euler_geometry",
    "make_kappa_geometry",
    "make_diffractometer",
    "available_diffractometers",
    "motor_to_dict",
    "motor_chain_to_dict",
    "diffractometer_to_dict",
    "DIFFRACTOMETER_REGISTRY",
]