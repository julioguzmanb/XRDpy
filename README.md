# XRDpy

**XRDpy** is a Python toolkit for **X-ray diffraction (XRD)** simulation and analysis, with a particular focus on **time-resolved / pump–probe diffraction workflows**.

The project name on **GitHub** and **Zenodo** is **XRDpy**.
The package is distributed on **PyPI** as **`trxrdpy`** and should be imported in Python as **`trxrdpy`**.

---

## Repository

Source code: https://github.com/julioguzmanb/XRDpy

## DOI (Zenodo)

- **Project concept DOI (all versions):** https://doi.org/10.5281/zenodo.18634909
---

## Main capabilities

### Simulation

- Polycrystalline XRD simulation
- Single-crystal diffraction simulation
- CIF-based crystallographic helpers
- Plotting utilities for simulated diffraction data
- A GUI for simulation workflows

### Analysis

- Beamline/facility-specific data handling
- 2D image reduction and azimuthal integration
- Standardized generation of 1D `xy` diffraction patterns
- Peak fitting workflows
- Differential analysis workflows
- Shared utilities for plotting, path handling, and common analysis operations

---

## Package structure

```text
XRDpy/
├── pyproject.toml
├── README.md
├── LICENSE
└── src/
    └── trxrdpy/
        ├── __init__.py
        ├── utils.py
        ├── detector.py
        ├── experiment.py
        ├── plot.py
        ├── sample.py
        ├── cif.py
        ├── simulation/
        │   ├── __init__.py
        │   ├── polycrystalline.py
        │   ├── single_crystal.py
        │   ├── gui.py
        └── analysis/
            ├── common/
            │   ├── __init__.py
            │   ├── paths.py
            │   ├── plot_utils.py
            │   ├── general_utils.py
            │   ├── azimint_utils.py
            │   ├── differential_analysis_utils.py
            │   └── fitting_utils.py
            │   └── calibration_utils.py
            ├── _shared_2d/
            │   ├── __init__.py
            │   └── azimint.py
            ├── ESRF_ID09/
            │   ├── __init__.py
            │   ├── datared.py
            │   └── azimint.py
            ├── MaxIV_FemtoMAX/
            │   ├── __init__.py
            │   ├── datared_utils.py
            │   ├── datared.py
            │   └── azimint.py
            ├── Spring8_SACLA/
            │   ├── __init__.py
            │   ├── datared.py
            │   ├── azimint.py
            │   └── pbs/
            │       └── parallel_job_sender.sh
            ├── differential_analysis.py
            └── fitting.py
            └── calibration.py
            └── gui.py
```

---

## Installation

### From PyPI

```bash
pip install trxrdpy
```

Optional extras:

```bash
pip install "trxrdpy[analysis]"
pip install "trxrdpy[gui]"
```

### From source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/julioguzmanb/XRDpy.git
cd XRDpy
pip install -e .
```

---

## Import

```python
import trxrdpy
from trxrdpy import simulation
from trxrdpy import analysis
from trxrdpy import calibration
```

More specific imports:

```python
from trxrdpy.analysis import fitting
from trxrdpy.analysis import differential_analysis
from trxrdpy.analysis.MaxIV_FemtoMAX import azimint
from trxrdpy.analysis.Spring8_SACLA import datared
```

The package currently exposes the following top-level modules through `trxrdpy.__init__`:

```python
from . import utils
from . import experiment
from . import plot
from . import sample
from . import cif
from . import simulation
from . import analysis
```

---

## Analysis organization

The `analysis` section is organized into shared utilities, facility-specific workflows, and user-facing APIs.

### `analysis.common`

Facility-independent shared utilities:

- path handling
- plotting helpers
- general helper functions
- common azimuthal-integration helpers
- fitting utilities
- differential-analysis utilities

### `analysis._shared_2d`

Shared 2D-image-based azimuthal-integration workflow.

This layer is currently used by:

- **Max IV FemtoMAX**
- **SPring-8 SACLA**

### `analysis.ESRF_ID09`

ID09-specific azimuthal-integration workflow.

At ESRF ID09, the route to generate `xy` files differs from the homogenized 2D-image workflow used elsewhere. The beamline-provided tools and data structure are handled through a dedicated facility-specific implementation.

### `analysis.MaxIV_FemtoMAX`

FemtoMAX-specific analysis entry points.

This section contains:

- beamline-specific data reduction
- azimuthal-integration entry points
- wrappers that preserve the facility-facing public API

### `analysis.Spring8_SACLA`

SACLA-specific analysis entry points.

This section contains:

- beamline-specific data reduction
- azimuthal-integration entry points
- PBS job-submission helper scripts for HPC workflows

### User-facing analysis APIs

These modules provide the user-facing analysis layer after `xy` files are available:

- `analysis.fitting`
- `analysis.differential_analysis`

Once `xy` files are created, the downstream fitting and differential-analysis pipeline is shared across facilities.

---

## Facility-specific workflow overview

The analysis pipeline is intentionally split because raw-data handling differs across facilities.

### Max IV FemtoMAX

- Uses facility-specific data reduction
- Produces homogenized 2D images
- Reuses the shared 2D azimuthal-integration workflow
- Then uses the shared downstream analysis pipeline

### SPring-8 SACLA

- Uses facility-specific data reduction
- Some reduction steps may depend on beamline-specific software, legacy Python environments, VPN access, or HPC job submission
- Produces homogenized 2D images
- Reuses the shared 2D azimuthal-integration workflow
- Then uses the shared downstream analysis pipeline

### ESRF ID09

- Does not use the same 2D homogenization route as FemtoMAX/SACLA
- Uses a different beamline-specific azimuthal-integration workflow to generate `xy` files
- Then uses the same downstream fitting and differential-analysis pipeline

In other words:

- **data reduction differs across facilities**
- **`xy` generation differs for ID09 vs the shared 2D workflow**
- **the downstream analysis after `xy` creation is shared**

---

## Notes

- Some analysis workflows may require facility-specific dependencies that are not part of a standard Python installation.
- Some SACLA workflows may rely on legacy Python environments and external HPC job submission.
- The simulation and analysis sections are developed within the same package but target different use cases.
- The project is published on PyPI as `trxrdpy` because the `xrdpy` name is already taken on PyPI.

---

## Citation

If you use **XRDpy** in academic work, please cite the Zenodo record corresponding to the version you used.

For the current public release:

**Julio Guzman-Brambila. XRDpy (v2.0.0). Zenodo. https://doi.org/10.5281/zenodo.19076157**

Project concept DOI (all versions):

**https://doi.org/10.5281/zenodo.18634909**

### BibTeX

```bibtex
@software{guzman_brambila_xrdpy_v200,
  author    = {Guzman-Brambila, Julio},
  title     = {XRDpy},
  version   = {2.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19076157},
  url       = {https://doi.org/10.5281/zenodo.19076157}
}
---

## License

Creative Commons Attribution 4.0 International (CC BY 4.0). See `LICENSE` for details.

---

## Author

**Julio Guzman-Brambila**
