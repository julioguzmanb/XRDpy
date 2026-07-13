# XRDpy

**XRDpy** is a Python toolkit for **X-ray diffraction (XRD)** simulation and analysis, with a particular focus on **time-resolved / pump-probe diffraction workflows**.

The project name on **GitHub** and **Zenodo** is **XRDpy**.
The package is distributed on **PyPI** as **`trxrdpy`** and should be imported in Python as **`trxrdpy`**.

---

## Repository

Source code: https://github.com/julioguzmanb/XRDpy

## DOI (Zenodo)

- **Project concept DOI (all versions):** https://doi.org/10.5281/zenodo.18634909

## Manual

- **User manual PDF:** [`docs/manual/XRDpy_manual.pdf`](docs/manual/XRDpy_manual.pdf)

---

## Main capabilities

### Simulation

- Polycrystalline XRD simulation
- Single-crystal diffraction simulation
- CIF-based crystallographic helpers
- Plotting utilities for simulated diffraction data
- Simulation GUI workflows
- Matrix-rotation helper GUI

### Analysis

- Facility-specific analysis workflows
- 2D image creation / reduction
- Side-by-side detector-image and pyFAI 2D-cake diagnostics
- Azimuthal integration
- Standardized generation of 1D `xy` diffraction patterns
- 1D absolute-pattern and difference-pattern visualization
- Peak fitting workflows
- Differential analysis workflows
- Shared utilities for plotting, path handling, calibration, fitting, and common analysis operations
- Modular Analysis GUI for ESRF ID09, Max IV FemtoMAX, and SPring-8 SACLA workflows

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
The GUI extra installs **PyQt5** and **mplcursors**. `mplcursors` is required for the interactive hover annotations shown in simulation detector/reflection plots.


### From source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/julioguzmanb/XRDpy.git
cd XRDpy
pip install -e .
```

For development with analysis and GUI dependencies:

```bash
pip install -e ".[analysis,gui]"
```

---

## Launching the GUIs

### Simulation GUI

```bash
python3 -m trxrdpy.simulation.gui.main_window
```

### Analysis GUI

```bash
python3 -m trxrdpy.analysis.gui.main_window
```

---

## Import

```python
import trxrdpy

from trxrdpy import simulation
from trxrdpy import analysis
```

Common simulation imports:

```python
from trxrdpy.simulation import polycrystalline
from trxrdpy.simulation import single_crystal
from trxrdpy.simulation import cif
```

Common analysis imports:

```python
from trxrdpy.analysis import calibration
from trxrdpy.analysis import fitting
from trxrdpy.analysis import differential_analysis

from trxrdpy.analysis.MaxIV_FemtoMAX import azimint
from trxrdpy.analysis.Spring8_SACLA import datared
from trxrdpy.analysis.ESRF_ID09 import azimint as id09_azimint
```

The package currently exposes the following top-level modules through `trxrdpy.__init__`:

```python
from .simulation import utils
from .simulation import experiment
from .simulation import plot
from .simulation import sample
from .simulation import cif
from . import simulation
from . import analysis
```

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
        │
        ├── simulation/
        │   ├── __init__.py
        │   ├── utils.py
        │   ├── geometry.py
        │   ├── diffractometers.py
        │   ├── detector.py
        │   ├── experiment.py
        │   ├── plot.py
        │   ├── sample.py
        │   ├── cif.py
        │   ├── polycrystalline.py
        │   ├── single_crystal.py
        │   └── gui/
        │       ├── __init__.py
        │       ├── main_window.py
        │       ├── state.py
        │       ├── style.py
        │       ├── services/
        │       │   ├── __init__.py
        │       │   └── simulation_service.py
        │       ├── tabs/
        │       │   ├── __init__.py
        │       │   ├── polycrystalline_tab.py
        │       │   └── single_crystal_tab.py
        │       └── widgets/
        │           ├── __init__.py
        │           ├── geometry_panel.py
        │           └── matrix_rotation_window.py
        │
        └── analysis/
            ├── __init__.py
            ├── common/
            │   ├── __init__.py
            │   ├── paths.py
            │   ├── plot_utils.py
            │   ├── general_utils.py
            │   ├── azimint_utils.py
            │   ├── differential_analysis_utils.py
            │   ├── fitting_utils.py
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
            ├── fitting.py
            ├── calibration.py
            └── gui/
                ├── __init__.py
                ├── main_window.py
                ├── defaults.py
                ├── state.py
                ├── style.py
                ├── utils.py
                ├── services/
                │   ├── __init__.py
                │   ├── calibration_service.py
                │   ├── differential_service.py
                │   ├── facility_service.py
                │   ├── fitting_service.py
                │   ├── integration_service.py
                │   ├── path_service.py
                │   └── preparation_service.py
                ├── tabs/
                │   ├── __init__.py
                │   ├── calibration_tab.py
                │   ├── differential_tab.py
                │   ├── fitting_tab.py
                │   ├── pattern_creation_tab.py
                │   ├── preparation_tab.py
                │   ├── session_tab.py
                │   └── viewer_tab.py
                └── widgets/
                    ├── __init__.py
                    ├── experiment_widgets.py
                    ├── facility_widgets.py
                    ├── log_widget.py
                    ├── multi_experiment_widgets.py
                    ├── parameter_widgets.py
                    ├── path_widgets.py
                    └── task_output_dialog.py
```

---

## Analysis organization

The `analysis` section is organized into shared utilities, facility-specific workflows, user-facing analysis APIs, and a modular GUI.

### `analysis.common`

Facility-independent shared utilities:

- path handling
- plotting helpers
- general helper functions
- common azimuthal-integration helpers
- fitting utilities
- differential-analysis utilities
- calibration utilities

### `analysis._shared_2d`

Shared 2D-image-based azimuthal-integration workflow.

This layer is currently used by:

- **Max IV FemtoMAX**
- **SPring-8 SACLA**

### `analysis.ESRF_ID09`

ID09-specific data-reduction and azimuthal-integration workflow.

At ESRF ID09, the route to generate `xy` files differs from the homogenized 2D-image workflow used elsewhere. The beamline-provided tools and data structure are handled through a dedicated facility-specific implementation.

### `analysis.MaxIV_FemtoMAX`

FemtoMAX-specific analysis entry points.

This section contains:

- beamline-specific data reduction
- metadata handling
- 2D image creation
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

- `analysis.calibration`
- `analysis.fitting`
- `analysis.differential_analysis`

The calibration API also exposes `plot_detector_and_cake(...)`, which loads a
homogenized dark detector image, performs pyFAI `integrate2d` integration, and
plots the detector image and q/azimuth cake side by side. Detector axes can be
flipped independently, and applying the detector mask is optional.

Once `xy` files are created, the downstream calibration, fitting, and differential-analysis pipeline is shared across facilities.

---

## GUI organization

### Simulation GUI

The simulation GUI is organized as:

```text
trxrdpy.simulation.gui
├── main_window.py
├── state.py
├── style.py
├── services/
├── tabs/
└── widgets/
```

It provides GUI access to simulation workflows while keeping the simulation logic in the simulation backend modules.

Current GUI-level features include:

- polycrystalline simulation tab
- single-crystal simulation tab
- matrix-rotation helper window
- session persistence
- autosave / restore
- summary / log section
- plot-window cleanup

### Analysis GUI

The analysis GUI is organized as:

```text
trxrdpy.analysis.gui
├── main_window.py
├── defaults.py
├── state.py
├── style.py
├── utils.py
├── services/
├── tabs/
└── widgets/
```

The Analysis GUI supports:

- session persistence
- autosave / restore
- facility selection
- 2D image creation
- calibration utilities
- detector-image and 2D-cake visualization
- 1D pattern creation
- 1D visualization
- differential analysis
- peak fitting
- task-output dialogs for long-running operations
- shared single-experiment metadata across tabs
- log output
- plot-window cleanup

The GUI layer is intentionally separated from the computational backend so that workflows can also be used programmatically.

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

- **Project concept DOI (all versions):** https://doi.org/10.5281/zenodo.18634909

Version-specific citation metadata is available on the Zenodo release page.

---

## License

Creative Commons Attribution 4.0 International (CC BY 4.0).

See `LICENSE` for details.

---

## Author

**Julio Guzman-Brambila**

## Optional PONI detector calibration files

The simulation module can optionally read detector calibration values from a
PONI file. This provides an additional detector-geometry input route while
preserving the existing manual detector parameters.

The PONI reader is implemented in:

    src/trxrdpy/simulation/poni.py

The reader parses detector distance, PONI coordinates, pixel sizes, detector
shape, rotations, wavelength, and detector metadata when available. The
Detector class can use the parsed distance, PONI coordinates, pixel sizes,
detector shape, and rotations directly. PONI rotations are stored in radians
in the file and converted to degrees before being passed to the Detector.

Manual detector input remains available and does not require a PONI file. A
PONI file is used only when a path is explicitly provided or when the detector
type is set to poni.

Example:

    from trxrdpy.simulation.detector import Detector

    det = Detector(
        detector_type="poni",
        poni_file="calibration.poni",
    )

The graphical interface also exposes an optional PONI file field in the
polycrystalline and single-crystal simulation tabs. If no PONI file is selected,
the usual manual or predefined detector configuration is used.

The detector rotation order is shown explicitly in the GUI and can be changed. Available orders
are `zyx`, `zxy`, `yzx`, `yxz`, `xzy`, and `xyz`. The default detector rotation order is `zyx`.
