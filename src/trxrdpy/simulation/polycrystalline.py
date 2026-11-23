"""High-level entry points for polycrystalline diffraction simulations."""
from __future__ import annotations
from . import sample
from . import detector
from . import experiment

from . import utils
from . import plot
import xrayutilities.materials as xu

import numpy as np


def _deduplicate_reflections_by_q(q_arr, d_arr=None, hkls_arr=None, q_tol=1e-5):
    """Keep one representative reflection for each powder-ring q value."""
    q_arr = np.asarray(q_arr, dtype=float)
    if q_arr.size == 0:
        return q_arr, d_arr, hkls_arr

    d_arr = None if d_arr is None else np.asarray(d_arr, dtype=float)
    hkls_arr = None if hkls_arr is None else np.asarray(hkls_arr, dtype=int)

    order = np.argsort(q_arr)
    q_sorted = q_arr[order]
    d_sorted = None if d_arr is None else d_arr[order]
    hkls_sorted = None if hkls_arr is None else hkls_arr[order]

    q_unique = []
    d_unique = [] if d_sorted is not None else None
    hkls_unique = [] if hkls_sorted is not None else None

    i = 0
    n = len(q_sorted)
    while i < n:
        q0 = q_sorted[i]
        j = i + 1
        while j < n and abs(q_sorted[j] - q0) <= q_tol:
            j += 1

        pick = i
        if hkls_sorted is not None:
            group = hkls_sorted[i:j]
            rel_pick, _best = min(
                enumerate(group),
                key=lambda item: (
                    abs(item[1][0]) + abs(item[1][1]) + abs(item[1][2]),
                    abs(item[1][0]),
                    abs(item[1][1]),
                    abs(item[1][2]),
                    int(item[1][0]),
                    int(item[1][1]),
                    int(item[1][2]),
                ),
            )
            pick = i + rel_pick
            hkls_unique.append(hkls_sorted[pick])

        q_unique.append(q_sorted[pick])
        if d_unique is not None:
            d_unique.append(d_sorted[pick])

        i = j

    return (
        np.asarray(q_unique, dtype=float),
        None if d_unique is None else np.asarray(d_unique, dtype=float),
        None if hkls_unique is None else np.asarray(hkls_unique, dtype=int),
    )


def _filter_reflections_by_qmax(q_hkls=None, d_hkls=None, hkls_names=None, qmax=None):
    """Filter reflection arrays to positive q values not exceeding ``qmax``."""
    if qmax is None:
        return q_hkls, d_hkls, hkls_names

    qmax = float(qmax)
    if qmax <= 0.0:
        raise ValueError("qmax must be positive.")

    hkls_arr = None if hkls_names is None else np.asarray(hkls_names, dtype=int)

    if d_hkls is not None:
        d_arr = np.asarray(d_hkls, dtype=float)
        q_arr = 2.0 * np.pi / d_arr
        mask = np.isfinite(q_arr) & (q_arr > 0.0) & (q_arr <= qmax)
        if hkls_arr is not None:
            hkls_arr = hkls_arr[mask]
        _q_unique, d_unique, hkls_unique = _deduplicate_reflections_by_q(
            q_arr[mask],
            d_arr=d_arr[mask],
            hkls_arr=hkls_arr,
            q_tol=max(1e-5, abs(qmax) * 1e-6),
        )
        return None, d_unique, hkls_unique

    if q_hkls is not None:
        q_arr = np.asarray(q_hkls, dtype=float)
        mask = np.isfinite(q_arr) & (q_arr > 0.0) & (q_arr <= qmax)
        if hkls_arr is not None:
            hkls_arr = hkls_arr[mask]
        q_unique, _d_unique, hkls_unique = _deduplicate_reflections_by_q(
            q_arr[mask],
            hkls_arr=hkls_arr,
            q_tol=max(1e-5, abs(qmax) * 1e-6),
        )
        return q_unique, None, hkls_unique

    return q_hkls, d_hkls, hkls_arr


def _coerce_title_digits(title_digits):
    """Return a bounded integer number of decimal places for figure titles."""
    try:
        digits = int(float(title_digits))
    except (TypeError, ValueError):
        digits = 2
    return max(0, min(digits, 12))



def simulate_1d(
    cif_file_path,
    qmax=5,
    energy=10e3,
    x_axis="q",
    include_lorentz_polarization=True,
    include_multiplicity=False,
    atom_positions=False,
    step=0.0001,
    fwhm=0.05,
    convolve=True,
    normalize=True,
    plot_result=True,
    ax=None,
    title_digits=2,
    detector=None,
    azimuthal_range=None,
    azimuth_offset_deg=0,
):
    """Simulate a powder pattern from crystallographic information in a CIF.

    Parameters
    ----------
    cif_file_path : path-like
        Crystallographic information file supplying the unit cell and atoms.
    qmax : float
        Maximum q in inverse angstrom.
    energy : float
        Photon energy in electron volts.
    x_axis : {"q", "two_theta", "d"}
        Coordinate for the output profile (inverse angstrom, degrees, or
        angstrom respectively).
    include_lorentz_polarization, include_multiplicity : bool
        Enable powder-intensity corrections.
    atom_positions : bool
        Build the structure-factor crystal from parsed atom positions.
    step, fwhm : float
        Output sampling interval and Gaussian peak width in the selected axis
        units. Zero or None FWHM gives a stick pattern.
    convolve, normalize, plot_result : bool
        Control curve generation, scaling, and immediate plotting.
    ax : matplotlib.axes.Axes, optional
        Existing axes used when ``plot_result`` is true.
    detector : Detector, optional
        If supplied, use its accessible q range and retain only peaks with
        detector coverage. Missing azimuths are excluded from the average;
        they do not reduce peak intensities. In this mode detector coverage
        replaces the explicit ``qmax`` limit.
    azimuthal_range : tuple of float, optional
        Laboratory azimuth limits in degrees. None selects the full 360 degrees.
        Zero points up (+Z), -90 left (+Y), and +90 right (-Y), viewed along
        the beam toward the detector.
    azimuth_offset_deg : float
        Shift from displayed azimuth to the laboratory reference azimuth.

    Returns
    -------
    dict
        Continuous ``x``/``I`` arrays and the discrete structured peak table.
    """
    if detector is not None:
        from .detector_coverage import detector_qmax, ring_coverage
        wavelength = utils.energy_to_wavelength(energy)
        qmax = detector_qmax(detector, wavelength, azimuthal_range,
                             azimuth_offset_deg)
    elif azimuthal_range is not None:
        raise ValueError("An azimuthal range requires a detector.")

    lattice = sample.LatticeStructure(cif_file_path=cif_file_path)

    out = lattice.simulate_1d_pattern(
        energy=energy,
        qmax=qmax,
        cif_file_path=cif_file_path,
        atom_positions=atom_positions,
        include_multiplicity=include_multiplicity,
        include_lorentz_polarization=include_lorentz_polarization,
        x_axis=x_axis,
        step=step,
        fwhm=fwhm,
        convolve=False if detector is not None else convolve,
        normalize=normalize,
    )

    if detector is not None:
        fractions = ring_coverage(out["peaks"]["q"], detector, wavelength,
                                  azimuthal_range, azimuth_offset_deg)
        peaks = out["peaks"][fractions > 0].copy()
        fractions = fractions[fractions > 0]
        if normalize and len(peaks) and np.max(peaks["I"]) > 0:
            peaks["I"] /= np.max(peaks["I"])
        out["peaks"] = peaks
        out["coverage_fraction"] = fractions
        out["detector_qmax"] = qmax
        if convolve:
            axis_max = (qmax if x_axis == "q" else
                        utils.q_to_two_theta(qmax, wavelength * 1e10)
                        if x_axis == "two_theta" else None)
            out["x"], out["I"] = sample.convolve_peaks_1d(
                peaks, x_axis=x_axis, step=step, fwhm=fwhm,
                x_max=axis_max, normalize=normalize)

    if plot_result and out["x"].size:
        digits = _coerce_title_digits(title_digits)
        title = (
            f"1D pattern, E={energy * 1e-3:.{digits}f} keV, "
            f"qmax={qmax:.{digits}f} Å$^{{-1}}$"
        )
        if detector is not None:
            if (azimuthal_range is None or
                    np.isclose(azimuthal_range[1] - azimuthal_range[0], 360)):
                azimuth_label = "full 360°"
            else:
                start, stop = azimuthal_range
                azimuth_label = f"[{start:g}°, {stop:+g}°]"
            title += f"\nAzimuth: {azimuth_label}"
            if azimuth_offset_deg:
                title += f" (offset {azimuth_offset_deg:+g}°)"
        plot.plot_1d_pattern(out["x"], out["I"], x_axis=x_axis, title=title, ax=ax)

    return out


def _powder_experiment(
        *, det_type, det_pxsize_h, det_pxsize_v, det_ntum_pixels_h,
        det_num_pixels_v, det_binning, det_poni_file, det_dist, det_poni1,
        det_poni2, det_rotx, det_roty, det_rotz, det_rotation_order,
        energy, e_bandwidth, q_hkls, d_hkls, hkls_names, qmax):
    det = detector.Detector(
        detector_type=det_type, pxsize_h=det_pxsize_h, pxsize_v=det_pxsize_v,
        num_pixels_h=det_ntum_pixels_h, num_pixels_v=det_num_pixels_v,
        binning=det_binning, poni_file=det_poni_file, dist=det_dist,
        poni1=det_poni1, poni2=det_poni2, rotx=det_rotx, roty=det_roty,
        rotz=det_rotz, rotation_order=det_rotation_order,
    )
    q_hkls, d_hkls, hkls_names = _filter_reflections_by_qmax(
        q_hkls, d_hkls, hkls_names, qmax)
    if d_hkls is not None:
        q_hkls = 2 * np.pi / np.asarray(d_hkls, dtype=float)
    if q_hkls is None or hkls_names is None:
        raise ValueError("Reflection positions and Miller-index labels are required.")
    q_hkls = np.atleast_1d(np.asarray(q_hkls, dtype=float))
    hkls_names = np.asarray(hkls_names, dtype=int)
    if hkls_names.ndim != 2 or hkls_names.shape != (len(q_hkls), 3):
        raise ValueError("hkls_names must have one [h, k, l] row per reflection.")
    if not len(q_hkls):
        raise ValueError("No reflections remain within qmax.")
    if not np.all(np.isfinite(q_hkls) & (q_hkls > 0)):
        raise ValueError("Reflection q values must be finite and positive.")
    lattice = sample.LatticeStructure(space_group=167, a=1, c=1)
    exp = experiment.Experiment(det, lattice, energy=energy, e_bandwidth=e_bandwidth)
    return exp, q_hkls, hkls_names


def simulate_2d(
        det_type="manual", det_pxsize_h=50e-6, det_pxsize_v=50e-6,
        det_ntum_pixels_h=2000, det_num_pixels_v=2000, det_binning=(1, 1),
        det_poni_file=None, det_dist=0.5, det_poni1=0, det_poni2=0,
        det_rotx=0, det_roty=0, det_rotz=0, det_rotation_order="zyx",
        energy=10e3, e_bandwidth=1.5, q_hkls=None, d_hkls=None,
        hkls_names=None, title_digits=2, qmax=None, *, consider_intensities=False,
        cif_file_path=None, intensity_cmap="viridis", intensity_clim=None,
        polarization_factor=1.0, polarization_axis_offset_deg=0.0,
        noise_distribution="none",
        noise_level=0.0, noise_seed=None):
    """Draw powder-ring arcs with adaptive detector-space sampling.

    This approach does not allocate a detector pixel grid or require a
    cone-point count in the default colorized mode. With
    ``consider_intensities=True``, a CIF supplies structure factors and a
    detector image is rendered with optional Gaussian or Poisson noise.
    ``polarization_factor`` follows -1 vertical, 0 unpolarized, +1 horizontal;
    None omits polarization. ``intensity_clim`` sets display limits only.
    """
    parameters = locals().copy()
    for key in ("title_digits", "consider_intensities", "cif_file_path", "intensity_cmap", "intensity_clim", "polarization_factor", "polarization_axis_offset_deg",
                "noise_distribution", "noise_level", "noise_seed"):
        parameters.pop(key)
    from .powder_curves import adaptive_ring_segments, plot_2d_powder

    exp, q_hkls, hkls_names = _powder_experiment(**parameters)
    curves = [adaptive_ring_segments(exp.detector, q, exp.wavelength)
              for q in q_hkls]
    exp.diffraction_cones_curves = curves
    exp.hkls_names = hkls_names
    intensities = None
    intensity_image = None
    background_image = None
    if consider_intensities:
        from .intensity_view import (reflection_intensities, powder_detector_image,
                                     validate_display)
        validate_display(intensity_cmap, noise_distribution, noise_level)
        from .intensity_view import validate_clim, validate_polarization_factor
        validate_clim(intensity_clim)
        validate_polarization_factor(polarization_factor)
        intensities = reflection_intensities(cif_file_path, hkls_names, energy, powder=True)
        exp.reflection_intensities = intensities
        intensity_image, background_image = powder_detector_image(
            exp.detector, curves, intensities, noise_distribution=noise_distribution,
            noise_level=noise_level, noise_seed=noise_seed,
            polarization_factor=polarization_factor,
            polarization_axis_offset_deg=polarization_axis_offset_deg,
            return_background=True)
        exp.detector_intensity_image = intensity_image
    plot_2d_powder(exp.detector, curves, hkls_names,
                   exp._detector_title(digits=title_digits), intensities=intensities,
                   intensity_image=intensity_image,
                   background_image=background_image,
                   cmap=intensity_cmap, intensity_clim=intensity_clim,
                   polarization_factor=polarization_factor,
                   polarization_axis_offset_deg=polarization_axis_offset_deg,
                   noise_distribution=noise_distribution,
                   noise_level=noise_level, noise_seed=noise_seed)
    return exp


def simulate_3d(
        det_type="manual", det_pxsize_h=50e-6, det_pxsize_v=50e-6,
        det_ntum_pixels_h=2000, det_num_pixels_v=2000, det_binning=(1, 1),
        det_poni_file=None, det_dist=0.5, det_poni1=0, det_poni2=0,
        det_rotx=0, det_roty=0, det_rotz=0, det_rotation_order="zyx",
        energy=10e3, e_bandwidth=1.5, q_hkls=None, d_hkls=None,
        hkls_names=None, title_digits=2, qmax=None, *, consider_intensities=False,
        cif_file_path=None, intensity_cmap="viridis", intensity_clim=None,
        polarization_factor=1.0, polarization_axis_offset_deg=0.0,
        noise_distribution="none",
        noise_level=0.0, noise_seed=None):
    """Draw smooth cone surfaces and their distorted detector intersections.

    Surface resolution follows the detector pixel scale. The optional intensity
    mode requires a CIF and colors both cone geometry and detector pixels by
    relative structure-factor strength. Polarization varies with azimuth;
    ``intensity_clim`` affects rendering, not stored detector intensities.
    """
    parameters = locals().copy()
    for key in ("title_digits", "consider_intensities", "cif_file_path", "intensity_cmap", "intensity_clim", "polarization_factor", "polarization_axis_offset_deg",
                "noise_distribution", "noise_level", "noise_seed"):
        parameters.pop(key)
    from .powder_curves import adaptive_ring_segments, plot_3d_powder

    exp, q_hkls, hkls_names = _powder_experiment(**parameters)
    curves = [adaptive_ring_segments(exp.detector, q, exp.wavelength)
              for q in q_hkls]
    exp.diffraction_cones_curves = curves
    exp.hkls_names = hkls_names
    intensities = None
    intensity_image = None
    if consider_intensities:
        from .intensity_view import (reflection_intensities, powder_detector_image,
                                     validate_display)
        validate_display(intensity_cmap, noise_distribution, noise_level)
        from .intensity_view import validate_clim, validate_polarization_factor
        validate_clim(intensity_clim)
        validate_polarization_factor(polarization_factor)
        intensities = reflection_intensities(cif_file_path, hkls_names, energy, powder=True)
        exp.reflection_intensities = intensities
        intensity_image = powder_detector_image(
            exp.detector, curves, intensities, noise_distribution=noise_distribution,
            noise_level=noise_level, noise_seed=noise_seed,
            polarization_factor=polarization_factor,
            polarization_axis_offset_deg=polarization_axis_offset_deg)
        exp.detector_intensity_image = intensity_image
    plot_3d_powder(exp.detector, q_hkls, curves, hkls_names,
                   exp.wavelength, exp._detector_title(digits=title_digits),
                   intensities=intensities, intensity_image=intensity_image,
                   cmap=intensity_cmap, intensity_clim=intensity_clim,
                   polarization_factor=polarization_factor,
                   polarization_axis_offset_deg=polarization_axis_offset_deg,
                   noise_distribution=noise_distribution, noise_level=noise_level,
                   noise_seed=noise_seed)
    return exp
