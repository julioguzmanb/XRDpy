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

    Returns
    -------
    dict
        Continuous ``x``/``I`` arrays and the discrete structured peak table.
    """
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
        convolve=convolve,
        normalize=normalize,
    )

    if plot_result and out["x"].size:
        digits = _coerce_title_digits(title_digits)
        title = (
            f"1D pattern, E={energy * 1e-3:.{digits}f} keV, "
            f"qmax={qmax:.{digits}f} Å$^{{-1}}$"
        )
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
        hkls_names=None, title_digits=2, qmax=None):
    """Draw powder-ring arcs with adaptive detector-space sampling.

    This approach does not allocate a detector pixel grid or require a
    cone-point count.
    """
    parameters = locals().copy()
    parameters.pop("title_digits")
    from .powder_curves import adaptive_ring_segments, plot_2d_powder

    exp, q_hkls, hkls_names = _powder_experiment(**parameters)
    curves = [adaptive_ring_segments(exp.detector, q, exp.wavelength)
              for q in q_hkls]
    exp.diffraction_cones_curves = curves
    exp.hkls_names = hkls_names
    plot_2d_powder(exp.detector, curves, hkls_names,
                   exp._detector_title(digits=title_digits))
    return exp


def simulate_3d(
        det_type="manual", det_pxsize_h=50e-6, det_pxsize_v=50e-6,
        det_ntum_pixels_h=2000, det_num_pixels_v=2000, det_binning=(1, 1),
        det_poni_file=None, det_dist=0.5, det_poni1=0, det_poni2=0,
        det_rotx=0, det_roty=0, det_rotz=0, det_rotation_order="zyx",
        energy=10e3, e_bandwidth=1.5, q_hkls=None, d_hkls=None,
        hkls_names=None, title_digits=2, qmax=None):
    """Draw smooth cone surfaces and their distorted detector intersections.

    Surface resolution follows the detector pixel scale.
    """
    parameters = locals().copy()
    parameters.pop("title_digits")
    from .powder_curves import adaptive_ring_segments, plot_3d_powder

    exp, q_hkls, hkls_names = _powder_experiment(**parameters)
    curves = [adaptive_ring_segments(exp.detector, q, exp.wavelength)
              for q in q_hkls]
    exp.diffraction_cones_curves = curves
    exp.hkls_names = hkls_names
    plot_3d_powder(exp.detector, q_hkls, curves, hkls_names,
                   exp.wavelength, exp._detector_title(digits=title_digits))
    return exp
