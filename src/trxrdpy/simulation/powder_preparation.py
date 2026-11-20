"""Pure powder reflection preparation shared by GUI workers."""
import numpy as np
from . import sample as sample_mod, utils as xutils


def powder_reflections(lattice_kwargs, energy_eV, e_bw_pct, qmax):
    lattice = sample_mod.LatticeStructure(**lattice_kwargs)
    lattice.calculate_reciprocal_lattice()
    lattice.create_possible_reflections(qmax=qmax)

    hkls_names = lattice.allowed_hkls.astype(int)
    q_vecs = sample_mod.calculate_q_hkl(hkls_names, lattice.reciprocal_lattice)
    q_mags = np.linalg.norm(q_vecs, axis=1)

    e_max = energy_eV * (1.0 + (e_bw_pct / 200.0))
    lam_min_A = xutils.energy_to_wavelength(e_max) * 1e10
    q_ewald_max = 4.0 * np.pi / lam_min_A

    mask = (
        np.isfinite(q_mags)
        & (q_mags > 0)
        & (q_mags <= qmax)
        & (q_mags <= q_ewald_max)
    )
    q_mags = q_mags[mask]
    hkls_names = hkls_names[mask]

    order = np.argsort(q_mags)
    q_mags = q_mags[order]
    hkls_names = hkls_names[order]

    if q_mags.size == 0:
        return q_mags, hkls_names

    q_tol = max(1e-5, abs(float(qmax)) * 1e-6)
    q_unique = []
    hkls_unique = []

    i = 0
    n = len(q_mags)
    while i < n:
        q0 = q_mags[i]
        j = i + 1
        while j < n and abs(q_mags[j] - q0) <= q_tol:
            j += 1

        group = hkls_names[i:j]
        best = min(
            group,
            key=lambda v: (
                abs(v[0]) + abs(v[1]) + abs(v[2]),
                abs(v[0]),
                abs(v[1]),
                abs(v[2]),
                int(v[0]),
                int(v[1]),
                int(v[2]),
            ),
        )

        q_unique.append(q0)
        hkls_unique.append(best)
        i = j

    return np.array(q_unique, dtype=float), np.array(hkls_unique, dtype=int)


def run_powder(function, lattice_kwargs, **kwargs):
    q, hkls = powder_reflections(lattice_kwargs, kwargs["energy"], kwargs["e_bandwidth"], kwargs["qmax"])
    if not len(q):
        raise ValueError("No powder reflections remain within qmax and the beam bandwidth.")
    return function(**{**kwargs, "q_hkls": q, "hkls_names": hkls, "d_hkls": None})
