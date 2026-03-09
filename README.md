# XRDpy

**XRDpy** is a Python toolkit for **X-ray diffraction (XRD)** simulation and analysis.

It currently contains two main components:

- **Simulation** tools for polycrystalline and single-crystal diffraction
- **Analysis** tools for time-resolved / pump–probe diffraction workflows across multiple synchrotron and XFEL facilities

---

## Repository

Source code: https://github.com/julioguzmanb/XRDpy

## DOI (Zenodo)

- **Project concept DOI (all versions):** https://doi.org/10.5281/zenodo.18634909
- **Current release DOI (v0.2.1):** https://doi.org/10.5281/zenodo.18924877

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
├── __init__.py
├── utils.py
├── experiment.py
├── plot.py
├── sample.py
├── cif.py
├── README.md
├── simulation/
│   ├── __init__.py
│   ├── polycrystalline.py
│   ├── single_crystal.py
│   └── ...
└── analysis/
    ├── common/
    │   ├── __init__.py
    │   ├── paths.py
    │   ├── plot_utils.py
    │   ├── general_utils.py
    │   ├── azimint_utils.py
    │   ├── differential_analysis_utils.py
    │   └── fitting_utils.py
    ├── _shared_2d/
    │   ├── __init__.py
    │   └── azimint.py
    ├── ESRF_ID09/
    │   ├── __init__.py
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
```

---

## Top-level imports

The package currently exposes the following top-level modules:

```python
from . import utils
from . import experiment
from . import plot
from . import sample
from . import cif
from . import simulation
from . import analysis
```

Typical usage examples:

```python
import XRDpy
from XRDpy import simulation
from XRDpy import analysis
```

or more specifically:

```python
from XRDpy.analysis import fitting
from XRDpy.analysis import differential_analysis
from XRDpy.analysis.MaxIV_FemtoMAX import azimint
from XRDpy.analysis.Spring8_SACLA import datared
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

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/julioguzmanb/XRDpy.git
cd XRDpy
pip install -e .
```

---

## Notes

- Some analysis workflows may require facility-specific dependencies that are not part of a standard Python installation.
- Some SACLA workflows may rely on legacy Python environments and external HPC job submission.
- The simulation and analysis sections are developed within the same package but target different use cases.

---

## Citation

If you use **XRDpy** in academic work, please cite the specific Zenodo release you used.

For the current public release:

**Julio Guzman-Brambila. XRDpy (v0.2.1). Zenodo. https://doi.org/10.5281/zenodo.18924877**

Project concept DOI (all versions):

**https://doi.org/10.5281/zenodo.18634909**

### BibTeX

```bibtex
@software{guzman_brambila_xrdpy_v021,
  author    = {Guzman-Brambila, Julio},
  title     = {XRDpy},
  version   = {0.2.1},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18924877},
  url       = {https://doi.org/10.5281/zenodo.18924877}
}
```

---

## License

Creative Commons Attribution 4.0 International (CC BY 4.0). See `LICENSE` for details.

---


## Author

**Julio Guzman-Brambila**
