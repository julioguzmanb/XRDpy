"""Crystallographic diffraction simulation tools.

The package provides detector and sample geometry models, CIF/PONI readers,
single-crystal and powder simulation workflows, diffractometer motor chains,
and plotting helpers. The :mod:`trxrdpy.simulation.gui` package exposes the
same workflows through a Qt interface.
"""
from __future__ import annotations
from . import utils
from . import geometry
from . import diffractometers
from . import experiment
from . import plot
from . import sample
from . import detector
from . import cif
from . import polycrystalline
from . import single_crystal
from . import gui

__all__ = [
    "utils",
    "geometry",
    "diffractometers",
    "experiment",
    "plot",
    "sample",
    "detector",
    "cif",
    "polycrystalline",
    "single_crystal",
    "gui",
]
