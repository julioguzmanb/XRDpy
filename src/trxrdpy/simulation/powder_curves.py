"""Adaptive powder-cone intersections and continuous detector views.

The numerical path follows rays through the active detector transform. It
subdivides in cone azimuth until the projected chord is accurate in pixels,
then clips each chord to the detector rectangle. No full pixel grid is built.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from . import plot, utils
from .detector_view import add_detector_face, display_title, segments_in_lab
from .deferred_plotting import gui_plot
from .parallel import check_cancelled


def _cone_angle(q, wavelength):
    arg = float(q) * wavelength * 1e10 / (4 * np.pi)
    if not np.isfinite(arg) or arg <= 0 or arg >= 1:
        return None
    return 2 * np.arcsin(arg)


def cone_surface(q, wavelength, length, num_points):
    """Return apex and rim rows for a ruled cone surface in lab coordinates."""
    angle = _cone_angle(q, wavelength)
    if angle is None:
        return None
    phi = np.linspace(0, 2 * np.pi, num_points)
    rim = np.stack((np.full_like(phi, length * np.cos(angle)),
                    length * np.sin(angle) * np.cos(phi),
                    length * np.sin(angle) * np.sin(phi)), axis=0)
    return np.stack((np.zeros_like(rim), rim), axis=1)


def _clip_segment(a, b, width, height):
    """Clip a pixel-coordinate chord to detector pixel edges."""
    delta = b - a
    low, high = 0.0, 1.0
    for p, q in ((-delta[0], a[0] + 0.5),
                 (delta[0], width - 0.5 - a[0]),
                 (-delta[1], a[1] + 0.5),
                 (delta[1], height - 0.5 - a[1])):
        if abs(p) < 1e-14:
            if q < 0:
                return None
            continue
        fraction = q / p
        if p < 0:
            low = max(low, fraction)
        else:
            high = min(high, fraction)
        if low > high:
            return None
    return np.stack((a + low * delta, a + high * delta))


def adaptive_ring_segments(detector, q, wavelength, *, tolerance_px=0.35):
    """Return visible line chords of one powder ring in detector pixels.

    A small angular seed grid finds separated arcs. Recursive midpoint checks
    resolve curvature, detector-edge crossings, and forward/backward changes.
    The depth limit bounds work close to rays parallel to the detector plane.
    """
    angle = _cone_angle(q, wavelength)
    if angle is None:
        return np.empty((0, 2, 2), dtype=float)

    transform = detector.get_transform()
    inverse = utils.invert_transform(transform)
    normal = transform[:3, 0]
    reference = utils.apply_transform(
        np.array([[detector.dist, detector.poni2, -detector.poni1]]), transform
    )[0]
    numerator = np.dot(normal, reference)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    width, height = detector.num_pixels_h, detector.num_pixels_v
    segments = []
    if abs(numerator) < 1e-15:
        return np.empty((0, 2, 2), dtype=float)
    offset = normal[0] * cos_angle
    cosine = normal[1] * sin_angle
    sine = normal[2] * sin_angle
    phase = np.arctan2(sine, cosine) % (2 * np.pi)

    def forward_possible(start_phi, end_phi):
        """Use exact sinusoid extrema to prune wholly backward intervals."""
        values = [offset + cosine * np.cos(phi) + sine * np.sin(phi)
                  for phi in (start_phi, end_phi)]
        for critical in (phase - np.pi, phase, phase + np.pi, phase + 2 * np.pi):
            if start_phi <= critical <= end_phi:
                values.append(offset + cosine * np.cos(critical) + sine * np.sin(critical))
        return max(values) > 0 if numerator > 0 else min(values) < 0

    def project(phi):
        ray = np.array([cos_angle, sin_angle * np.cos(phi), sin_angle * np.sin(phi)])
        denominator = np.dot(normal, ray)
        if abs(denominator) < 1e-12:
            return None
        distance = numerator / denominator
        if not np.isfinite(distance) or distance <= 0:
            return None
        native = inverse[:3, :3] @ (distance * ray) + inverse[:3, 3]
        pixel = np.array([
            (detector.poni2 - native[1]) / detector.pxsize_h - 0.5,
            (native[2] + detector.poni1) / detector.pxsize_v - 0.5,
        ])
        return pixel if np.all(np.isfinite(pixel)) else None

    def refine(start_phi, end_phi, start, end, depth):
        if not forward_possible(start_phi, end_phi):
            return
        middle_phi = (start_phi + end_phi) / 2
        middle = project(middle_phi)
        points = (start, middle, end)
        continuous = all(point is not None for point in points)
        if continuous:
            midpoint_error = np.linalg.norm(middle - (start + end) / 2)
            chord_length = np.linalg.norm(end - start)
            if midpoint_error <= tolerance_px and chord_length <= 256:
                clipped = _clip_segment(start, end, width, height)
                if clipped is not None:
                    segments.append(clipped)
                return
        if depth >= 12:
            if start is not None and end is not None and np.linalg.norm(end - start) <= 256:
                clipped = _clip_segment(start, end, width, height)
                if clipped is not None:
                    segments.append(clipped)
            return
        refine(start_phi, middle_phi, start, middle, depth + 1)
        refine(middle_phi, end_phi, middle, end, depth + 1)

    seed_angles = np.linspace(0, 2 * np.pi, 97)
    seed_points = [project(phi) for phi in seed_angles]
    for index in range(len(seed_angles) - 1):
        check_cancelled()
        refine(seed_angles[index], seed_angles[index + 1],
               seed_points[index], seed_points[index + 1], 0)
    return np.asarray(segments, dtype=float).reshape(-1, 2, 2)


def _legend(ax, artists, labels, colors):
    handles = [Line2D([], [], color=color, linewidth=2, label=label)
               for label, color in zip(labels, colors)]
    if not handles:
        return
    legend = ax.legend(handles=handles, title="Reflections", loc="upper left",
                       bbox_to_anchor=(1.01, 1), borderaxespad=0)
    mapping = dict(zip(legend.get_lines(), artists))
    for line in mapping:
        line.set_picker(5)

    def on_pick(event):
        artist = mapping.get(event.artist)
        if artist is None:
            return
        targets = artist if isinstance(artist, tuple) else (artist,)
        visible = not all(target.get_visible() for target in targets)
        for target in targets:
            target.set_visible(visible)
        event.artist.set_alpha(1 if visible else 0.2)
        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect("pick_event", on_pick)


@gui_plot
def plot_2d_powder(detector, curves, hkls, title, ax=None):
    """Render the projected ring chords as continuous detector-space paths."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 8))
    colors = plot.colorize(np.arange(len(curves)))
    artists, labels, used_colors = [], [], []
    for segments, hkl, color in zip(curves, hkls, colors):
        if not len(segments):
            continue
        artist = LineCollection(segments, colors=[color], linewidths=1.5)
        ax.add_collection(artist)
        artists.append(artist)
        labels.append(f"({hkl[0]},{hkl[1]},{hkl[2]})")
        used_colors.append(color)
    ax.set(xlim=(-0.5, detector.num_pixels_h - 0.5),
           ylim=(-0.5, detector.num_pixels_v - 0.5),
           xlabel="Horizontal Pixels", ylabel="Vertical Pixels")
    ax.set_title(display_title(title))
    ax.set_aspect(detector.pxsize_v / detector.pxsize_h, adjustable="box")
    ax.grid()
    plot._plot_direct_beam(ax, detector)
    _legend(ax, artists, labels, used_colors)
    ax.figure.tight_layout()
    plt.show()
    return ax


@gui_plot
def plot_3d_powder(detector, q_values, curves, hkls, wavelength, title, ax=None):
    """Render automatically tessellated cones and their true detector traces."""
    if ax is None:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
    grid = add_detector_face(ax, detector, title)
    colors = plot.colorize(np.arange(len(q_values)))
    artists, labels, used_colors = [], [], []
    length = detector.dist * 1.1
    reach = [grid.reshape(-1, 3), np.zeros((1, 3))]
    for q, segments, hkl, color in zip(q_values, curves, hkls, colors):
        angle = _cone_angle(q, wavelength)
        if angle is None:
            continue
        radius_pixels = abs(length * np.sin(angle)) / min(detector.pxsize_h, detector.pxsize_v)
        count = int(np.clip(np.ceil(np.pi * np.sqrt(max(radius_pixels, 1))), 64, 1024))
        surface = cone_surface(q, wavelength, length, count + 1)
        rim = surface[:, 1, :]
        surface_artist = ax.plot_surface(*surface, color=color, alpha=0.18,
                                         edgecolor="none", rstride=1, cstride=1)
        reach.append(rim.T)
        lab_segments = segments_in_lab(detector, segments)
        targets = [surface_artist]
        if len(lab_segments):
            artist = Line3DCollection(lab_segments, colors=[color], linewidths=2)
            ax.add_collection3d(artist)
            targets.append(artist)
        artists.append(tuple(targets))
        labels.append(f"({hkl[0]},{hkl[1]},{hkl[2]})")
        used_colors.append(color)
    points = np.concatenate(reach)
    lo, hi = points.min(axis=0), points.max(axis=0)
    middle = (lo + hi) / 2
    half = max((hi - lo).max() / 2, 1e-6)
    for setter, center in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), middle):
        setter(center - half, center + half)
    _legend(ax, artists, labels, used_colors)
    plt.show()
    return ax
