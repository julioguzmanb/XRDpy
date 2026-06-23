"""High-level entry points for polycrystalline diffraction simulations."""
from __future__ import annotations
from . import sample
from . import detector
from . import experiment

from . import utils
from . import plot
import xrayutilities.materials as xu

import numpy as np


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
        hkls_names=None
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

    #Dummy lattice
    lattice = sample.LatticeStructure(space_group=167, a=1, c=1)    

    exp = experiment.Experiment(det, lattice, energy = energy, e_bandwidth = e_bandwidth)

    exp.plot_3d_polycrystal_exp(q_hkls, hkls_names, cones_num_of_points)

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
        hkls_names = None
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


    #Dummy lattice
    lattice = sample.LatticeStructure(space_group=167, a=1, c=1)    

    exp = experiment.Experiment(det, lattice, energy = energy, e_bandwidth = e_bandwidth)

    exp.calculate_cones_pixel_positions(q_hkls, hkls_names, num_points=cones_num_of_points)
    exp.plot_2d_polycrystal_exp()

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
        title = f"1D pattern, E={energy*1e-3:.3f} keV, qmax={qmax:.2f} Å$^{{-1}}$"
        plot.plot_1d_pattern(out["x"], out["I"], x_axis=x_axis, title=title, ax=ax)

    return out
