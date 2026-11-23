"""Structure-factor intensities and detector images for diffraction views."""
from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from . import sample, utils


def reflection_intensities(cif_file_path, hkls, energy, *, powder=False, lattice=None,
                           normalize=True):
    """Return |F(hkl)|², normalized to this set's maximum unless requested raw.

    When a simulation lattice is supplied, its cell and space group define the
    structure factor while the CIF supplies atom sites. This also accepts CIFs
    that contain atom sites without a complete unit cell.
    """
    if not cif_file_path:
        raise ValueError("A CIF with atom positions is required to consider intensities.")
    if lattice is None:
        from .cif import Cif
        Cif(cif_file_path)  # validate before LatticeStructure's legacy parser
        lattice = sample.LatticeStructure(cif_file_path=cif_file_path)
        crystal = sample._build_xu_crystal_from_lattice(
            lattice, cif_file_path=cif_file_path)
    else:
        from copy import copy
        from .cif import read_cif_file, extract_atomic_positions
        atoms = None
        for _, block in read_cif_file(cif_file_path).items():
            try:
                atoms = extract_atomic_positions(block)
                if atoms:
                    break
            except ValueError:
                continue
        if not atoms:
            raise ValueError("The CIF does not contain valid atom positions for intensities.")
        structure = copy(lattice)
        structure.atom_positions = atoms
        crystal = sample._build_xu_crystal_from_lattice(
            structure, atom_positions=True)
    hkls = np.asarray(hkls, dtype=int).reshape(-1, 3)
    qvecs = np.array([crystal.Q(hkl) for hkl in hkls], dtype=float).reshape(-1, 3)
    values = np.abs(np.array([crystal.StructureFactor(q, energy) for q in qvecs])) ** 2
    if powder and len(values):
        wavelength_a = utils.energy_to_wavelength(energy) * 1e10
        theta = np.arcsin(np.clip(wavelength_a * np.linalg.norm(qvecs, axis=1) / (4 * np.pi), 0, 1))
        denominator = np.sin(theta) ** 2 * np.cos(theta)
        lp = np.divide(1 + np.cos(2 * theta) ** 2, denominator,
                       out=np.zeros_like(theta), where=denominator > 1e-12)
        values *= lp
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    maximum = values.max(initial=0)
    return values / maximum if normalize and maximum > 0 else values


def validate_display(cmap, noise_distribution, noise_level):
    """Validate controls before constructing a potentially large detector image."""
    colormaps[cmap]
    if noise_distribution not in ("none", "gaussian", "poisson"):
        raise ValueError("noise_distribution must be none, gaussian, or poisson.")
    level = float(noise_level)
    if not np.isfinite(level) or level < 0:
        raise ValueError("noise_level must be finite and nonnegative.")
    return level


def validate_clim(clim):
    """Accept optional nonnegative lower/upper display limits."""
    if clim is None:
        return None, None
    if len(clim) != 2:
        raise ValueError("intensity_clim must contain (minimum, maximum).")
    limits = tuple(None if value is None else float(value) for value in clim)
    if any(value is not None and (not np.isfinite(value) or value < 0)
           for value in limits):
        raise ValueError("Intensity color limits must be finite and nonnegative.")
    if limits[0] is not None and limits[1] is not None and limits[1] <= limits[0]:
        raise ValueError("Maximum intensity color limit must exceed minimum.")
    return limits


def validate_polarization_factor(factor):
    """Follow the pyFAI convention: -1 vertical, 0 unpolarized, +1 horizontal."""
    if factor is None:
        return None
    factor = float(factor)
    if not np.isfinite(factor) or not -1 <= factor <= 1:
        raise ValueError("polarization_factor must be between -1 and 1, or None.")
    return factor


def polarization_weight(directions, factor, axis_offset_deg=0.0):
    """Thomson intensity factor for a beam along +X in the laboratory frame.

    Horizontal polarization is along Y and vertical polarization is along Z.
    ``None`` omits the factor; zero is the unpolarized average.
    """
    factor = validate_polarization_factor(factor)
    axis_offset_deg = float(axis_offset_deg)
    if not np.isfinite(axis_offset_deg):
        raise ValueError("Polarization axis offset must be finite.")
    directions = np.asarray(directions, dtype=float)
    if factor is None:
        return np.ones(directions.shape[:-1], dtype=float)
    norm = np.linalg.norm(directions, axis=-1)
    unit = np.divide(directions, norm[..., None],
                     out=np.zeros_like(directions), where=norm[..., None] > 0)
    horizontal_fraction = (1 + factor) / 2
    offset = np.deg2rad(axis_offset_deg)
    horizontal = unit[..., 1] * np.cos(offset) - unit[..., 2] * np.sin(offset)
    vertical = unit[..., 1] * np.sin(offset) + unit[..., 2] * np.cos(offset)
    return 1 - horizontal_fraction * horizontal ** 2 - (
        1 - horizontal_fraction) * vertical ** 2


def intensity_mappable(cmap, vmax, clim=None):
    """Use black for exact zero and a shared scale for image and colorbar."""
    palette = colormaps[cmap].copy()
    palette.set_under("black")
    minimum, maximum = validate_clim(clim)
    lower = np.nextafter(0.0, 1.0) if minimum in (None, 0) else minimum
    upper = max(float(vmax), 1.0, lower * 1.01) if maximum is None else maximum
    if upper <= lower:
        raise ValueError("Maximum intensity color limit must exceed minimum.")
    return ScalarMappable(norm=Normalize(vmin=lower, vmax=upper),
                          cmap=palette)


def add_noise(image, distribution="none", level=0.0, seed=None, *,
              return_background=False):
    """Add nonnegative detector background in units of the strongest peak."""
    level = validate_display("viridis", distribution, level)
    if distribution == "none" or level == 0:
        return (image, None) if return_background else image
    rng = np.random.default_rng(seed)
    if distribution == "gaussian":
        noise = rng.normal(0.0, level, image.shape)
    else:
        # A fixed count scale keeps both the mean and variance meaningful in
        # units of the strongest reflection (one unit = 1000 counts).
        noise = rng.poisson(level * 1000.0, image.shape) / 1000.0
    image += noise
    np.maximum(image, 0, out=image)
    if return_background:
        np.maximum(noise, 0, out=noise)
        return image, noise
    return image


def powder_detector_image(detector, curves, intensities, *, noise_distribution="none",
                          noise_level=0.0, noise_seed=None,
                          polarization_factor=0.0,
                          polarization_axis_offset_deg=0.0,
                          return_background=False):
    """Rasterize clipped ring chords into detector pixels, summing overlaps."""
    image = np.zeros((detector.num_pixels_v, detector.num_pixels_h), dtype=float)
    validate_polarization_factor(polarization_factor)
    for segments, strength in zip(curves, intensities):
        if strength <= 0:
            continue
        ring_pixels = []
        for start, end in segments:
            count = max(2, int(np.ceil(np.linalg.norm(end - start) * 2)) + 1)
            points = np.linspace(start, end, count)
            x = np.rint(points[:, 0]).astype(int)
            y = np.rint(points[:, 1]).astype(int)
            inside = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
            # A pixel should receive one contribution per reflection, not per chord sample.
            ring_pixels.append(np.column_stack((y[inside], x[inside])))
        if ring_pixels:
            pixels = np.unique(np.concatenate(ring_pixels), axis=0)
            if polarization_factor == 0:
                weights = 1.0
            else:
                from .detector_view import pixels_in_lab
                directions = pixels_in_lab(detector, pixels[:, ::-1])
                weights = (polarization_weight(directions, polarization_factor,
                                               polarization_axis_offset_deg) /
                           polarization_weight(directions, 0.0))
            image[pixels[:, 0], pixels[:, 1]] += strength * weights
    return add_noise(image, noise_distribution, noise_level, noise_seed,
                     return_background=return_background)


def spots_detector_image(detector, positions, intensities, hits, *, noise_distribution="none",
                         noise_level=0.0, noise_seed=None):
    """Place Bragg intensities at detector pixels, summing coincident hits."""
    image = np.zeros((detector.num_pixels_v, detector.num_pixels_h), dtype=float)
    points = np.rint(np.asarray(positions)[np.asarray(hits, dtype=bool)]).astype(int)
    strengths = np.asarray(intensities)[np.asarray(hits, dtype=bool)]
    inside = ((points[:, 0] >= 0) & (points[:, 0] < image.shape[1]) &
              (points[:, 1] >= 0) & (points[:, 1] < image.shape[0]))
    np.add.at(image, (points[inside, 1], points[inside, 0]), strengths[inside])
    return add_noise(image, noise_distribution, noise_level, noise_seed)


def plot_detector_texture(ax, detector, image, mappable):
    """Draw a bounded-resolution version of a detector image on its 3D plane."""
    from .detector_view import pixels_in_lab

    # A compact mesh keeps the 3D view responsive even with megapixel noise.
    step_v = max(1, int(np.ceil(image.shape[0] / 96)))
    step_h = max(1, int(np.ceil(image.shape[1] / 96)))
    v_edges = np.append(np.arange(0, image.shape[0], step_v), image.shape[0]) - 0.5
    h_edges = np.append(np.arange(0, image.shape[1], step_h), image.shape[1]) - 0.5
    h, v = np.meshgrid(h_edges, v_edges)
    points = pixels_in_lab(detector, np.column_stack((h.ravel(), v.ravel()))).reshape(h.shape + (3,))
    # Preserve narrow Bragg spots when many detector pixels share one 3D tile.
    tiles = np.maximum.reduceat(image, np.arange(0, image.shape[0], step_v), axis=0)
    tiles = np.maximum.reduceat(tiles, np.arange(0, image.shape[1], step_h), axis=1)
    colors = mappable.to_rgba(tiles)
    return ax.plot_surface(points[:, :, 0], points[:, :, 1], points[:, :, 2],
                           facecolors=colors, shade=False, rstride=1, cstride=1,
                           linewidth=0, antialiased=False)
