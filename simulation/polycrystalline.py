from .. import sample
from .. import detector
from .. import experiment

from .. import utils
from .. import plot
import xrayutilities.materials as xu

import numpy as np


def simulate_3d(
        det_type="manual", 
        det_pxsize_h=50e-6, det_pxsize_v=50e-6, 
        det_ntum_pixels_h=2000, det_num_pixels_v=2000, det_binning=(1,1),
        det_dist=0.5, det_poni1=0, det_poni2=0, 
        det_rotx=0, det_roty=0, det_rotz=0, 
        det_rotation_order="xyz",
        cones_num_of_points=30,
        energy=10e3, e_bandwidth=1.5,
        q_hkls=None, d_hkls=None,
        hkls_names=None
):

    try:
        det = detector.Detector(
            detector_type=det_type,
            pxsize_h=det_pxsize_h, pxsize_v=det_pxsize_v,
            num_pixels_h=det_ntum_pixels_h, num_pixels_v=det_num_pixels_v,
            dist=det_dist, poni1=det_poni1, poni2=det_poni2,
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
        det_dist=0.5, det_poni1=0, det_poni2=0, 
        det_rotx=0, det_roty=0, det_rotz=0, 
        det_rotation_order="xyz",
        cones_num_of_points=1000,
        energy=10e3, e_bandwidth=1.5,
        q_hkls=None, d_hkls=None,
        hkls_names = None
):

    try:
        det = detector.Detector(
            detector_type=det_type,
            pxsize_h=det_pxsize_h, pxsize_v=det_pxsize_v,
            num_pixels_h=det_ntum_pixels_h, num_pixels_v=det_num_pixels_v,
            dist=det_dist, poni1=det_poni1, poni2=det_poni2,
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
    sam_space_group=167,
    sam_a=None, sam_b=None, sam_c=None,
    sam_alpha=None, sam_beta=None, sam_gamma=None,
    cif_file_path=None,
    atom_positions=None,
    energy=10e3,
    qmax=10.0,
    x_axis="q",                 # "q" or "two_theta"
    step=0.005,                 # in Å^-1 if x_axis="q", in deg if x_axis="two_theta"
    fwhm=0.03,                  # same units as x_axis
    profile="gaussian",         # currently only "gaussian"
    include_multiplicity=True,
    include_lorentz_polarization=False,
    normalize=True,
    convolve=True,
    plot_result=True,
    ax=None,
):
    """
    Simulate a 1D polycrystalline (powder-like) diffraction pattern.

    Parameters
    ----------
    sam_space_group : int
        Space group number (ITC).
    sam_a,b,c : float
        Lattice constants in Å (ignored if cif_file_path provides them).
    sam_alpha,beta,gamma : float
        Lattice angles in degrees (ignored if cif_file_path provides them).
    cif_file_path : str or None
        If provided, used to populate lattice parameters/atoms (via your Cif class)
        and (when available) to build an xrayutilities Crystal via Crystal.fromCIF.
    atom_positions : dict or None
        Optional atom positions {label: np.array([fx,fy,fz])}. If None, tries CIF.
    energy : float
        Beam energy in eV.
    qmax : float
        Maximum |Q| in Å^-1 used for generating allowed reflections.
    x_axis : str
        "q" for Q (Å^-1) axis, or "two_theta" for 2θ (deg) axis.
    step : float
        Sampling step for the output axis.
    fwhm : float
        Peak full-width at half-maximum in the output axis units.
    include_multiplicity : bool
        Multiply intensity by number of symmetrically equivalent hkls.
    include_lorentz_polarization : bool
        Apply a simple Lorentz-polarization factor (lab XRD-style).
    normalize : bool
        Normalize peak intensities to max=1 (and pattern too if convolved).
    convolve : bool
        If True, returns a continuous curve. If False, returns only peak list + empty curve.
    plot_result : bool
        If True, plots the simulated pattern.
    ax : matplotlib axis or None
        Optional axis for plotting.

    Returns
    -------
    dict with keys:
        - "x": axis array (q or 2θ)
        - "I": intensity array (continuous if convolve else zeros)
        - "peaks": structured array with (h,k,l,q,d,two_theta,I,multiplicity)
        - "lattice": the LatticeStructure instance
    """
    if x_axis not in {"q", "two_theta"}:
        raise ValueError("x_axis must be 'q' or 'two_theta'.")

    if profile != "gaussian":
        raise ValueError("Only profile='gaussian' is currently implemented.")

    if step <= 0:
        raise ValueError("step must be > 0.")
    if fwhm <= 0:
        raise ValueError("fwhm must be > 0.")
    if qmax <= 0:
        raise ValueError("qmax must be > 0.")

    # --- Build lattice (your central object) ---
    lattice = sample.LatticeStructure(
        space_group=sam_space_group,
        a=sam_a, b=sam_b, c=sam_c,
        alpha=sam_alpha, beta=sam_beta, gamma=sam_gamma,
        atom_positions=atom_positions,
        cif_file_path=cif_file_path
    )
    lattice.calculate_reciprocal_lattice()
    lattice.create_possible_reflections(qmax)

    hkls = np.array(lattice.allowed_hkls, dtype=int)
    if hkls.size == 0:
        raise RuntimeError("No allowed reflections returned (check qmax / space group).")

    # --- Wavelength ---
    wavelength_m = utils.energy_to_wavelength(energy)
    wavelength_A = wavelength_m * 1e10

    # --- Build xrayutilities Crystal for structure factors ---
    crystal = None
    if cif_file_path is not None and hasattr(xu.Crystal, "fromCIF"):
        # Best-case: let xrayutilities parse symmetry + atoms directly
        crystal = xu.Crystal.fromCIF(cif_file_path)
    else:
        # Fallback: try to build from provided atom positions
        if lattice.atom_positions is not None and lattice.space_group is not None:
            atoms = list(lattice.atom_positions.keys())
            atom_symbols = [sample.extract_element_symbol(a) for a in atoms]
            frac = np.array([lattice.atom_positions[a] for a in atoms], dtype=float)

            param_names, _ = lattice.get_lattice_params(lattice.space_group)
            lattice_args = [getattr(lattice, name) for name in param_names]

            # NOTE: xrayutilities will try to map fractional positions to Wyckoff sites for this space group.
            sg_lat = xu.SGLattice(lattice.space_group, *lattice_args, atoms=atom_symbols, pos=frac)
            crystal = xu.Crystal("sample", sg_lat)

    # --- Peak positions (q, d, 2θ) ---
    if crystal is not None:
        qvecs = np.array([crystal.Q(hkl) for hkl in hkls], dtype=float)  # Å^-1 vectors :contentReference[oaicite:1]{index=1}
        q = np.linalg.norm(qvecs, axis=1)
    else:
        # Pure geometry fallback using your reciprocal lattice (still consistent for positions)
        qvecs = sample.calculate_q_hkl(hkls, lattice.reciprocal_lattice)
        q = np.linalg.norm(qvecs, axis=1)

    # Keep only within qmax (robust against any slight overshoot)
    mask = (q > 0) & (q <= qmax * (1 + 1e-12))
    hkls = hkls[mask]
    qvecs = qvecs[mask]
    q = q[mask]

    d = (2 * np.pi) / q  # Å
    arg = np.clip(wavelength_A * q / (4 * np.pi), 0.0, 1.0)
    two_theta = np.degrees(2 * np.arcsin(arg))

    # --- Intensities ---
    if crystal is not None:
        F = np.array([crystal.StructureFactor(qv, energy) for qv in qvecs], dtype=complex) 
        I = np.abs(F) ** 2
    else:
        # If no structure model, give equal peak weights
        I = np.ones_like(q, dtype=float)

    # Multiplicity (powder-style)
    multiplicity = np.ones_like(I, dtype=int)
    if include_multiplicity and getattr(lattice, "xu_lattice", None) is not None:
        mult_list = []
        for hkl in hkls:
            eq = list(lattice.xu_lattice.equivalent_hkls(tuple(int(x) for x in hkl)))
            mult_list.append(max(1, len(eq)))
        multiplicity = np.array(mult_list, dtype=int)
        I = I * multiplicity

    # Lorentz-polarization factor (simple lab XRD-style)
    if include_lorentz_polarization:
        tt = np.radians(two_theta)
        th = 0.5 * tt
        denom = (np.sin(th) ** 2) * np.cos(th)
        denom = np.where(denom <= 1e-12, np.nan, denom)
        lp = (1.0 + np.cos(tt) ** 2) / denom
        lp = np.nan_to_num(lp, nan=0.0, posinf=0.0, neginf=0.0)
        I = I * lp

    # Normalize peaks
    if normalize and I.size > 0 and np.nanmax(I) > 0:
        I = I / np.nanmax(I)

    # --- Assemble peak table ---
    peak_dtype = [
        ("h", "i4"), ("k", "i4"), ("l", "i4"),
        ("q", "f8"), ("d", "f8"), ("two_theta", "f8"),
        ("I", "f8"), ("multiplicity", "i4")
    ]
    peaks = np.empty(len(hkls), dtype=peak_dtype)
    peaks["h"], peaks["k"], peaks["l"] = hkls[:, 0], hkls[:, 1], hkls[:, 2]
    peaks["q"], peaks["d"], peaks["two_theta"] = q, d, two_theta
    peaks["I"], peaks["multiplicity"] = I, multiplicity

    # Sort peaks by the chosen x-axis
    if x_axis == "q":
        order = np.argsort(peaks["q"])
        x0 = peaks["q"][order]
    else:
        order = np.argsort(peaks["two_theta"])
        x0 = peaks["two_theta"][order]
    peaks = peaks[order]
    I0 = peaks["I"]

    # --- Build continuous pattern (Gaussian convolution) ---
    if x_axis == "q":
        x_max = qmax
    else:
        # max 2θ corresponding to qmax at this wavelength
        arg_max = np.clip(wavelength_A * qmax / (4 * np.pi), 0.0, 1.0)
        x_max = float(np.degrees(2 * np.arcsin(arg_max)))

    x = np.arange(0.0, x_max + step, step, dtype=float)
    y = np.zeros_like(x)

    if convolve and len(x0) > 0:
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        if sigma <= 0:
            raise ValueError("Computed sigma <= 0; check fwhm.")

        # Efficient local accumulation: only evaluate within ±6σ of each peak
        half_window = int(np.ceil((6.0 * sigma) / step))
        for xc, amp in zip(x0, I0):
            if amp <= 0:
                continue
            center_idx = int(np.round(xc / step))
            lo = max(0, center_idx - half_window)
            hi = min(len(x), center_idx + half_window + 1)
            xs = x[lo:hi]
            y[lo:hi] += amp * np.exp(-0.5 * ((xs - xc) / sigma) ** 2)

        if normalize and y.size > 0 and np.max(y) > 0:
            y = y / np.max(y)

    # --- Plot ---
    if plot_result:
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, linewidth=1.5)
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_xlabel("Q (Å$^{-1}$)" if x_axis == "q" else r"2$\theta$ (deg)")
        ax.set_title(
            f"Simulated 1D pattern | E={energy*1e-3:.3f} keV | qmax={qmax:.2f} Å$^{{-1}}$"
        )
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "x": x,
        "I": y,
        "peaks": peaks,
        "lattice": lattice,
    }




def simulate_1d(
    cif_file_path,
    qmax=5,
    energy=10e3,
    x_axis="q",
    include_lorentz_polarization=True,
    include_multiplicity=False,
    atom_positions=False,
    step=0.01,
    fwhm=0.05,
    convolve=True,
    normalize=True,
    plot_result=True,
    ax=None,
):
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