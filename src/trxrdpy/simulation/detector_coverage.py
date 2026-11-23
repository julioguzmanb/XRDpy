"""Laboratory-frame azimuth and powder-ring coverage for area detectors."""
from __future__ import annotations

import numpy as np

from . import experiment, utils


def _azimuths(points, offset_deg):
    """Zero is lab +Z; negative angles point to lab +Y (image left)."""
    return np.degrees(np.arctan2(-points[..., 1], points[..., 2])) - offset_deg


def _in_range(angles, azimuthal_range):
    if azimuthal_range is None:
        return np.ones_like(angles, dtype=bool)
    start, stop = azimuthal_range
    return (angles - start) % 360.0 <= (stop - start) + 1e-10


def validate_azimuthal_range(azimuthal_range, offset_deg):
    offset_deg = float(offset_deg)
    if not np.isfinite(offset_deg):
        raise ValueError("azimuth_offset_deg must be finite.")
    if azimuthal_range is None:
        return None, offset_deg
    values = np.asarray(azimuthal_range, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("azimuthal_range must contain two finite angles.")
    start, stop = map(float, values)
    if not 0 < stop - start <= 360:
        raise ValueError("azimuthal_range must be increasing and span at most 360 degrees.")
    return (start, stop), offset_deg


def _boundary_points(detector, samples_per_edge=1025):
    width, height = detector.num_pixels_h, detector.num_pixels_v
    if width <= 0 or height <= 0:
        raise ValueError("Detector dimensions must be positive.")
    h = np.linspace(-0.5, width - 0.5, samples_per_edge)
    v = np.linspace(-0.5, height - 0.5, samples_per_edge)
    pixels = np.concatenate((
        np.column_stack((h, np.full_like(h, -0.5))),
        np.column_stack((h, np.full_like(h, height - 0.5))),
        np.column_stack((np.full_like(v, -0.5), v)),
        np.column_stack((np.full_like(v, width - 0.5), v)),
    ))
    native = np.column_stack((
        np.full(len(pixels), detector.dist),
        detector.poni2 - (pixels[:, 0] + 0.5) * detector.pxsize_h,
        -detector.poni1 + (pixels[:, 1] + 0.5) * detector.pxsize_v,
    ))
    return utils.apply_transform(native, detector.get_transform())


def detector_qmax(detector, wavelength, azimuthal_range=None, azimuth_offset_deg=0):
    """Largest reachable q on the detector within a laboratory azimuth sector."""
    azimuthal_range, offset = validate_azimuthal_range(azimuthal_range, azimuth_offset_deg)
    points = _boundary_points(detector)
    selected = (points[:, 0] > 0) & _in_range(_azimuths(points, offset), azimuthal_range)
    if azimuthal_range is not None and azimuthal_range[1] - azimuthal_range[0] < 360:
        # Include exact intersections of the sector boundaries with each edge.
        corners = _boundary_points(detector, 2).reshape(4, 2, 3)
        extra = []
        for angle in azimuthal_range:
            phi = np.radians(angle + offset)
            for edge in corners:
                a, b = edge
                fa = a[1] * np.cos(phi) + a[2] * np.sin(phi)
                fb = b[1] * np.cos(phi) + b[2] * np.sin(phi)
                if abs(fa - fb) < 1e-15:
                    continue
                t = fa / (fa - fb)
                if 0 <= t <= 1:
                    point = a + t * (b - a)
                    if point[0] > 0 and point[2] * np.cos(phi) - point[1] * np.sin(phi) >= -1e-12:
                        extra.append(point)
        if extra:
            points = np.concatenate((points[selected], np.asarray(extra)))
        else:
            points = points[selected]
    else:
        points = points[selected]
    if not len(points):
        raise ValueError("The selected azimuth does not intersect the forward detector area.")
    lengths = np.linalg.norm(points, axis=1)
    two_theta = np.arccos(np.clip(points[:, 0] / lengths, -1, 1))
    return float(np.max(4 * np.pi / (wavelength * 1e10) * np.sin(two_theta / 2)))


def ring_coverage(q_values, detector, wavelength, azimuthal_range=None,
                  azimuth_offset_deg=0, samples=4096):
    """Fraction of each requested azimuth sector intercepted by the detector."""
    azimuthal_range, offset = validate_azimuthal_range(azimuthal_range, azimuth_offset_deg)
    q_values = np.asarray(q_values, dtype=float)
    if azimuthal_range is None:
        display_phi = (np.arange(samples) + 0.5) * (360.0 / samples) - 180.0
    else:
        start, stop = azimuthal_range
        count = max(2, int(np.ceil(samples * (stop - start) / 360.0)))
        display_phi = start + (np.arange(count) + 0.5) * ((stop - start) / count)
    phi = np.radians(display_phi + offset)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    k = 2 * np.pi / (wavelength * 1e10)
    fractions = np.zeros(len(q_values), dtype=float)
    transform = detector.get_transform()
    for i, q in enumerate(q_values):
        ratio = q / (2 * k)
        if not 0 < ratio < 1:
            continue
        theta = 2 * np.arcsin(ratio)
        q_vectors = k * np.column_stack((
            np.full(len(phi), np.cos(theta) - 1),
            -np.sin(theta) * sin_phi,
            np.sin(theta) * cos_phi,
        ))
        positions, signs = experiment.calculate_diffraction_direction_with_transform(
            q_vectors, wavelength, detector.dist, detector.poni1,
            detector.poni2, transform,
        )
        pixels = experiment.lab_to_pixel_coordinates_with_transform(
            positions, detector.dist, detector.pxsize_h, detector.pxsize_v,
            detector.poni1, detector.poni2, transform,
        )
        fractions[i] = np.mean(
            (signs > 0) & np.all(np.isfinite(pixels), axis=1) &
            (pixels[:, 0] >= -0.5) & (pixels[:, 0] <= detector.num_pixels_h - 0.5) &
            (pixels[:, 1] >= -0.5) & (pixels[:, 1] <= detector.num_pixels_v - 0.5)
        )
    return fractions
