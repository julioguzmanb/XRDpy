"""Shared lightweight 3D detector face and display-grid geometry."""
from __future__ import annotations

import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from . import plot, utils


def detector_corners(detector):
    """Return a 2 by 2 lab-frame detector mesh without allocating all pixels."""
    h, v = np.meshgrid([-0.5, detector.num_pixels_h - 0.5],
                       [-0.5, detector.num_pixels_v - 0.5], indexing="ij")
    native = np.stack((np.full_like(h, detector.dist),
                       detector.poni2 - (h + 0.5) * detector.pxsize_h,
                       -detector.poni1 + (v + 0.5) * detector.pxsize_v), axis=-1)
    return utils.apply_transform(native.reshape(-1, 3), detector.get_transform()).reshape(2, 2, 3)


def segments_in_lab(detector, segments):
    """Map detector-pixel line segments into the active lab-frame detector plane."""
    if not len(segments):
        return np.empty((0, 2, 3), dtype=float)
    return pixels_in_lab(detector, np.asarray(segments, dtype=float).reshape(-1, 2)).reshape(-1, 2, 3)


def pixels_in_lab(detector, pixels):
    """Map detector pixel-center coordinates to the active lab-frame plane."""
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    native = np.column_stack((np.full(len(pixels), detector.dist),
                              detector.poni2 - (pixels[:, 0] + 0.5) * detector.pxsize_h,
                              -detector.poni1 + (pixels[:, 1] + 0.5) * detector.pxsize_v))
    return utils.apply_transform(native, detector.get_transform())


def detector_visual_grid(detector):
    """A sparse display mesh that suggests pixels without allocating a pixel grid."""
    width = detector.num_pixels_h * detector.pxsize_h
    height = detector.num_pixels_v * detector.pxsize_v
    longest = max(width, height)
    columns = max(2, min(detector.num_pixels_h, round(48 * width / longest)))
    rows = max(2, min(detector.num_pixels_v, round(48 * height / longest)))
    h_edges = np.linspace(-0.5, detector.num_pixels_h - 0.5, columns + 1)
    v_edges = np.linspace(-0.5, detector.num_pixels_v - 0.5, rows + 1)
    lines = [((h, v_edges[0]), (h, v_edges[-1])) for h in h_edges]
    lines += [((h_edges[0], v), (h_edges[-1], v)) for v in v_edges]
    return segments_in_lab(detector, np.asarray(lines, dtype=float))


def display_title(title):
    """Keep long sample-rotation metadata readable at the normal title size."""
    return title.replace("Sample rotations [deg, equivalent xyz]: ",
                         "Sample rotations [deg, equivalent xyz]:\n")


def add_detector_face(ax, detector, title):
    """Render the powder-style detector sheet and sparse mesh on a 3D axes."""
    corners = detector_corners(detector)
    plot.plot_3d_detector(corners, title=title, ax=ax)
    ax.set_title("")
    ax.figure.suptitle(display_title(title), y=0.98)
    ax.set_position([0.04, 0.04, 0.72, 0.71])
    mesh = Line3DCollection(detector_visual_grid(detector),
                            colors=[(0.15, 0.20, 0.25, 0.32)], linewidths=0.4)
    mesh.set_gid("detector-visual-grid")
    ax.add_collection3d(mesh)
    beam_pixel = detector.direct_beam_pixel()
    if beam_pixel is not None:
        beam_lab = pixels_in_lab(detector, [beam_pixel])[0]
        marker, = ax.plot([beam_lab[0]], [beam_lab[1]], [beam_lab[2]],
                          marker="x", color="red", markersize=10,
                          markeredgewidth=2, linestyle="None", zorder=20)
        marker.set_gid("direct-beam")
    return corners
