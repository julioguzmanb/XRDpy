"""Offline Henke/CXRO X-ray optical constants and 1/e intensity lengths.

Tables: https://henke.lbl.gov/optical_constants/asf.html (CXRO/LBNL).
The grazing convention follows the CXRO attenuation calculator: angle from
the surface and depth along its normal. Calculations never access the network.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
import re

import numpy as np

from .utils import energy_to_wavelength

R_E = 2.8179403262e-15  # m
N_A = 6.02214076e23  # mol^-1
_TOKEN = re.compile(r"([A-Z][a-z]?)(?:(\d+(?:\.\d*)?|\.\d+))?")
_UNITS = {"m": 1., "nm": 1e9, "um": 1e6, "µm": 1e6}


def parse_formula(formula: str) -> dict[str, float]:
    """Parse a simple, case-sensitive formula such as ``SrTi0.9901Nb0.0099O3``.

    Parentheses and hydration syntax are intentionally unsupported; a mapping
    of elemental stoichiometries can be passed to :class:`XRayMaterial` instead.
    """
    if not isinstance(formula, str) or not formula:
        raise ValueError("A nonempty chemical formula is required.")
    result: dict[str, float] = {}
    pos = 0
    for match in _TOKEN.finditer(formula):
        if match.start() != pos:
            raise ValueError(f"Invalid chemical formula: {formula!r}")
        symbol, count = match.groups()
        amount = float(count) if count else 1.
        if not np.isfinite(amount) or amount <= 0:
            raise ValueError("Stoichiometric amounts must be positive and finite.")
        result[symbol] = result.get(symbol, 0.) + amount
        pos = match.end()
    if pos != len(formula):
        raise ValueError(f"Invalid chemical formula: {formula!r}")
    return result


@lru_cache(maxsize=128)
def _read_table(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one original three-column .nff table, retaining every edge point."""
    data = np.loadtxt(path, comments="#", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 3 or len(data) < 2:
        raise ValueError(f"Invalid Henke table: {path}")
    if not (np.all(np.isfinite(data)) and np.all(data[:, 2] >= 0)):
        raise ValueError(f"Invalid Henke values in {path}")
    # A few official files have rounded duplicate or slightly out-of-order
    # energies. Sort stably and keep the last value at an identical energy.
    data = data[np.argsort(data[:, 0], kind="stable")]
    keep = np.r_[np.diff(data[:, 0]) != 0, True]
    energy, f1, f2 = data[keep].T
    if np.any(np.diff(energy) <= 0):
        raise ValueError(f"Invalid Henke energy grid in {path}")
    for array in (energy, f1, f2):
        array.setflags(write=False)
    return energy, f1, f2


def atomic_factors(symbol: str, energy_keV, *, data_dir=None):
    """Linearly interpolate original CXRO f1 and f2; reject invalid ranges.

    ``energy_keV`` may be scalar or array-like. Undefined f1 sentinel rows
    (below about 29 eV) and extrapolation are rejected.
    """
    directory = Path(data_dir) if data_dir is not None else Path(str(files("trxrdpy.simulation") / "data" / "henke"))
    path = directory / f"{symbol.lower()}.nff"
    if not path.is_file():
        raise ValueError(f"No local Henke table for {symbol}; add {path.name} to {directory}.")
    energies, f1, f2 = _read_table(str(path.resolve()))
    requested = np.asarray(energy_keV, dtype=float) * 1000.
    if not np.all(np.isfinite(requested)) or np.any(requested <= 0):
        raise ValueError("Photon energies must be positive and finite [keV].")
    valid = f1 > -9990.
    low = energies[np.flatnonzero(valid)[0]]
    if np.any(requested < low) or np.any(requested > energies[-1]):
        raise ValueError(f"{symbol} Henke f1/f2 valid range is {low:g}–{energies[-1]:g} eV.")
    left = np.clip(np.searchsorted(energies, requested, side="right") - 1, 0, len(energies) - 1)
    right = np.clip(left + 1, 0, len(energies) - 1)
    if np.any(~valid[left] | ~valid[right]):
        raise ValueError(f"Undefined {symbol} Henke f1 value near requested energy.")
    return np.interp(requested, energies[valid], f1[valid]), np.interp(requested, energies, f2)


class XRayMaterial:
    """Homogeneous material with explicit density in g/cm³ and Henke factors.

    ``composition`` can be a formula string or mapping of element to atoms per
    formula unit. The formula need not describe a crystallographic unit cell.
    """

    def __init__(self, composition, density_g_cm3: float, *, data_dir=None):
        self.composition = parse_formula(composition) if isinstance(composition, str) else dict(composition)
        if not self.composition or any(not isinstance(k, str) or not re.fullmatch(r"[A-Z][a-z]?", k) or not np.isfinite(v) or v <= 0 for k, v in self.composition.items()):
            raise ValueError("Composition must map element symbols to positive amounts.")
        self.density_g_cm3 = float(density_g_cm3)
        if not np.isfinite(self.density_g_cm3) or self.density_g_cm3 <= 0:
            raise ValueError("Density must be positive and finite [g/cm³].")
        self.data_dir = data_dir
        from xrayutilities.materials import elements
        try:
            self.molar_mass_g_mol = sum(float(getattr(elements, el).weight) * N_A * 1000 * amount for el, amount in self.composition.items())
        except AttributeError as exc:
            raise ValueError("Unknown element in composition.") from exc

    def optical_constants(self, energy_keV):
        """Return dimensionless (delta, beta) for n = 1 - delta + i beta."""
        energy = np.asarray(energy_keV, dtype=float)
        wavelength = energy_to_wavelength(energy * 1000.)
        f1_sum = np.zeros_like(energy, dtype=float)
        f2_sum = np.zeros_like(energy, dtype=float)
        for symbol, amount in self.composition.items():
            f1, f2 = atomic_factors(symbol, energy, data_dir=self.data_dir)
            f1_sum = f1_sum + amount * f1
            f2_sum = f2_sum + amount * f2
        number_density = self.density_g_cm3 * 1e6 * N_A / self.molar_mass_g_mol
        scale = R_E * wavelength**2 * number_density / (2 * np.pi)
        return scale * f1_sum, scale * f2_sum

    def attenuation_coefficient(self, energy_keV, *, unit="m^-1"):
        """Return bulk intensity attenuation coefficient µ = 4πβ/λ."""
        _, beta = self.optical_constants(energy_keV)
        mu = 4 * np.pi * beta / energy_to_wavelength(np.asarray(energy_keV) * 1000.)
        if unit == "m^-1":
            return mu
        if unit == "cm^-1":
            return mu / 100.
        raise ValueError("unit must be 'm^-1' or 'cm^-1'.")

    def bulk_attenuation_length(self, energy_keV, *, unit="nm"):
        """Return 1/e intensity path length λ/(4πβ) in m, nm, or µm."""
        return _convert_length(1. / self.attenuation_coefficient(energy_keV), unit)

    def grazing_penetration_depth(self, energy_keV, grazing_angle_deg, *, unit="nm", exact=False):
        """Return 1/e intensity depth along surface normal in m, nm, or µm.

        Angle is physical grazing incidence relative to the surface, 0–90°.
        Default uses the established first-order CXRO-style wavevector model;
        ``exact=True`` uses n² - cos²(angle) under the complex square root.
        """
        delta, beta = self.optical_constants(energy_keV)
        return grazing_depth_from_constants(energy_keV, delta, beta, grazing_angle_deg,
                                            unit=unit, exact=exact)

    def grazing_depth_from_rot_y(self, energy_keV, rotY_deg, *, unit="nm", exact=False):
        """Use this beamline's |rotY| convention to obtain the physical angle."""
        return self.grazing_penetration_depth(energy_keV, np.abs(rotY_deg), unit=unit, exact=exact)


def _convert_length(value_m, unit):
    try:
        return value_m * _UNITS[unit]
    except KeyError as exc:
        raise ValueError("unit must be 'm', 'nm', or 'um' ('µm').") from exc


def grazing_depth_from_constants(energy_keV, delta, beta, grazing_angle_deg, *, unit="nm", exact=False):
    """Compute normal 1/e intensity depth from precomputed optical constants.

    Useful for angle scans: the Henke factors need only be interpolated once
    for the energy grid, while each selected angle is evaluated on demand.
    """
    angle = np.asarray(grazing_angle_deg, dtype=float)
    if not np.all(np.isfinite(angle)) or np.any((angle < 0) | (angle > 90)):
        raise ValueError("Grazing angle must be finite and between 0 and 90 degrees.")
    alpha = np.deg2rad(angle)
    term = ((1 - delta + 1j * beta)**2 - np.cos(alpha)**2 if exact
            else np.sin(alpha)**2 - 2 * delta + 2j * beta)
    kz = (2 * np.pi / energy_to_wavelength(np.asarray(energy_keV) * 1000.)) * np.sqrt(term)
    return _convert_length(1. / (2 * np.imag(kz)), unit)


def nb_fraction_from_weight_fraction(w: float) -> float:
    """Ti-site fraction x for elemental Nb mass fraction w in SrTi1-xNbxO3."""
    if not np.isfinite(w) or not 0 <= w < 1:
        raise ValueError("Nb weight fraction must be in [0, 1).")
    from xrayutilities.materials import elements
    sr, ti, nb, oxygen = (getattr(elements, el).weight for el in ("Sr", "Ti", "Nb", "O"))
    m0 = sr + ti + 3 * oxygen
    return w * m0 / (nb - w * (nb - ti))
