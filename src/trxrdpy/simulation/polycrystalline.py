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


def simulate_3d(
        det_type="manual", 
        det_pxsize_h=50e-6, det_pxsize_v=50e-6, 
        det_ntum_pixels_h=2000, det_num_pixels_v=2000, det_binning=(1,1),
        det_poni_file=None,
        det_dist=0.5, det_poni1=0, det_poni2=0, 
        det_rotx=0, det_roty=0, det_rotz=0, 
        det_rotation_order="zyx",
        cones_num_of_points=30,
        energy=10e3, e_bandwidth=1.5,
        q_hkls=None, d_hkls=None,
        hkls_names=None,
        title_digits=2,
        qmax=None,
):
    """Simulate and plot three-dimensional powder diffraction cones.

    Parameters
    ----------
    det_type : str
        ``"manual"``, ``"poni"``, or a detector name accepted by pyFAI.
    det_pxsize_h, det_pxsize_v : float
        Unbinned horizontal and vertical pixel sizes in metres. Used for a
        manual detector and ignored when a PONI file supplies them.
    det_ntum_pixels_h, det_num_pixels_v : int
        Unbinned horizontal and vertical pixel counts. The misspelled
        ``det_ntum_pixels_h`` name is retained for API compatibility.
    det_binning : tuple of int
        Horizontal and vertical binning factors.
    det_poni_file : path-like, optional
        PONI calibration file. When provided, it supplies detector pixel,
        shape, distance, PONI, and rotation values.
    det_dist : float
        Sample-to-detector distance in metres.
    det_poni1, det_poni2 : float
        PONI coordinates in metres. Axis 1 is slow/vertical; axis 2 is
        fast/horizontal.
    det_rotx, det_roty, det_rotz : float
        Detector Euler angles in degrees.
    det_rotation_order : str
        Three-axis SciPy Euler order for the detector rotations.
    cones_num_of_points : int
        Number of azimuth samples used to draw each diffraction cone.
    energy : float
        Incident photon energy in electron volts.
    e_bandwidth : float
        Full relative energy bandwidth in percent.
    q_hkls, d_hkls : array-like, optional
        Reflection positions as q in inverse angstrom or d in angstrom.
        Exactly one representation is required; ``d_hkls`` takes precedence
        if both are supplied.
    hkls_names : array-like, shape (N, 3)
        Miller-index label corresponding to each reflection position.

    Returns
    -------
    Experiment
        Experiment containing the detector and generated cone data.

    Raises
    ------
    ValueError
        If reflection positions or names are absent or have different lengths.
    """

    try:
        det = detector.Detector(
            detector_type=det_type,
            pxsize_h=det_pxsize_h, pxsize_v=det_pxsize_v,
            num_pixels_h=det_ntum_pixels_h, num_pixels_v=det_num_pixels_v,
            dist=det_dist, poni1=det_poni1, poni2=det_poni2,
            poni_file=det_poni_file,
            rotx=det_rotx, roty=det_roty, rotz=det_rotz,
            rotation_order=det_rotation_order,
            binning=det_binning
        )
        det.calculate_lab_grid()
    except:
        raise ImportError
    
    q_hkls, d_hkls, hkls_names = _filter_reflections_by_qmax(
        q_hkls=q_hkls,
        d_hkls=d_hkls,
        hkls_names=hkls_names,
        qmax=qmax,
    )

    if (q_hkls is None) and (d_hkls is None):
        raise ValueError
    
    elif d_hkls is not None:
        d_hkls = np.array(d_hkls)
        q_hkls = 2 * np.pi / d_hkls

    elif q_hkls is not None:
        q_hkls = np.array(q_hkls)
    
    if hkls_names is None:
        raise ValueError
    else:
        hkls_names = np.array(hkls_names)

    if len(q_hkls) != len(hkls_names):
        raise ValueError
    if len(q_hkls) == 0:
        raise ValueError("No reflections remain within qmax.")

    #Dummy lattice
    lattice = sample.LatticeStructure(space_group=167, a=1, c=1)    

    exp = experiment.Experiment(det, lattice, energy = energy, e_bandwidth = e_bandwidth)

    exp.plot_3d_polycrystal_exp(
        q_hkls,
        hkls_names,
        cones_num_of_points,
        title_digits=title_digits,
    )

    return exp

    
def simulate_2d(
        det_type="manual", 
        det_pxsize_h=50e-6, det_pxsize_v=50e-6, 
        det_ntum_pixels_h=2000, det_num_pixels_v=2000, det_binning=(1,1),
        det_poni_file=None,
        det_dist=0.5, det_poni1=0, det_poni2=0, 
        det_rotx=0, det_roty=0, det_rotz=0, 
        det_rotation_order="zyx",
        cones_num_of_points=1000,
        energy=10e3, e_bandwidth=1.5,
        q_hkls=None, d_hkls=None,
        hkls_names = None,
        title_digits=2,
        qmax=None,
):
    """Simulate powder rings on the two-dimensional detector plane.

    Parameters
    ----------
    det_type : str
        ``"manual"``, ``"poni"``, or a detector name accepted by pyFAI.
    det_pxsize_h, det_pxsize_v : float
        Unbinned horizontal and vertical pixel sizes in metres.
    det_ntum_pixels_h, det_num_pixels_v : int
        Unbinned horizontal and vertical pixel counts.
    det_binning : tuple of int
        Horizontal and vertical binning factors.
    det_poni_file : path-like, optional
        Calibration file overriding explicit detector geometry values.
    det_dist : float
        Sample-to-detector distance in metres.
    det_poni1, det_poni2 : float
        Slow/vertical and fast/horizontal PONI coordinates in metres.
    det_rotx, det_roty, det_rotz : float
        Detector Euler angles in degrees.
    det_rotation_order : str
        Three-axis SciPy Euler rotation order.
    cones_num_of_points : int
        Azimuth samples per ring; larger values make rings smoother.
    energy : float
        Incident photon energy in electron volts.
    e_bandwidth : float
        Full relative energy bandwidth in percent.
    q_hkls, d_hkls : array-like, optional
        Reflection positions as q in inverse angstrom or d in angstrom.
        At least one is required.
    hkls_names : array-like, shape (N, 3)
        Miller-index labels in the same order as the reflection positions.

    Returns
    -------
    Experiment
        Experiment with ``diffraction_cones_pixel_position`` populated.

    Raises
    ------
    ValueError
        If reflection positions or labels are absent or length-mismatched.
    """

    try:
        det = detector.Detector(
            detector_type=det_type,
            pxsize_h=det_pxsize_h, pxsize_v=det_pxsize_v,
            num_pixels_h=det_ntum_pixels_h, num_pixels_v=det_num_pixels_v,
            dist=det_dist, poni1=det_poni1, poni2=det_poni2,
            poni_file=det_poni_file,
            rotx=det_rotx, roty=det_roty, rotz=det_rotz,
            rotation_order=det_rotation_order,
            binning=det_binning
        )
        det.calculate_lab_grid()
    except:
        raise ImportError
    
    q_hkls, d_hkls, hkls_names = _filter_reflections_by_qmax(
        q_hkls=q_hkls,
        d_hkls=d_hkls,
        hkls_names=hkls_names,
        qmax=qmax,
    )

    if (q_hkls is None) and (d_hkls is None):
        raise ValueError
    
    elif d_hkls is not None:
        d_hkls = np.array(d_hkls)
        q_hkls = 2 * np.pi / d_hkls

    elif q_hkls is not None:
        q_hkls = np.array(q_hkls)
    
    if hkls_names is None:
        raise ValueError
    else:
        hkls_names = np.array(hkls_names)

    if len(q_hkls) != len(hkls_names):
        raise ValueError
    if len(q_hkls) == 0:
        raise ValueError("No reflections remain within qmax.")


    #Dummy lattice
    lattice = sample.LatticeStructure(space_group=167, a=1, c=1)    

    exp = experiment.Experiment(det, lattice, energy = energy, e_bandwidth = e_bandwidth)

    exp.calculate_cones_pixel_positions(q_hkls, hkls_names, num_points=cones_num_of_points)
    exp.plot_2d_polycrystal_exp(title_digits=title_digits)

    return exp


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
    x_axis : {"q", "two_theta"}
        Coordinate for the continuous output profile.
    include_lorentz_polarization, include_multiplicity : bool
        Enable powder-intensity corrections.
    atom_positions : bool
        Build the structure-factor crystal from parsed atom positions.
    step, fwhm : float
        Output sampling interval and Gaussian peak width.
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
