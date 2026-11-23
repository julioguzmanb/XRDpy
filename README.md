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

- **User manual PDF:** [download the PDF](https://github.com/julioguzmanb/XRDpy/raw/main/docs/manual/XRDpy_manual.pdf)

---

## Main capabilities

### Simulation

- Polycrystalline XRD simulation, including CIF-based 1D patterns on q, 2θ, or d axes
- Single-crystal diffraction simulation
- CIF-based crystallographic helpers
- Plotting utilities for simulated diffraction data
- Simulation GUI workflows
- Matrix-rotation helper GUI

### Analysis

- Facility-specific analysis workflows
- Metadata-driven data reduction for representative 2D images
- Single-shot 1D pattern production for Max IV FemtoMAX and SPring-8 SACLA
- Side-by-side detector-image and pyFAI 2D-cake diagnostics
- Azimuthal integration from representative 2D images or compatible single-shot 1D caches
- On-the-fly refresh of final patterns from the completed shots currently available
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

### Orientation from measured Bragg peaks

In **Single Crystal**, select **Find Orientation Matrix**. Set the
unit cell (or load a CIF), detector geometry (or PONI file), energy, sample
goniometer, and optional initial orientation in the normal controls. Then use
**New project from current setup** to capture those conditions. A project
contains reusable detector setups, linked experimental poses, and any number
of observed Bragg pixels per pose.

Add a pose from a selected parent to inherit its energy, detector setup, and
motor readings. Select **Absolute readings** for entered sample motor values
or **Changes from parent** for motor increments. Blank fields inherit. Detector
motor readings can also be changed for a pose. Add all peaks observed at that
pose as separate rows. An HKL may be exact, any symmetry equivalent of the
entered family, or unknown. **Update q/d from pixels** helps identify a peak
from its measured scattering-vector magnitude.

Set default horizontal and vertical peak position uncertainties in
**Uncertainties**, then edit individual peak rows when their precision differs.
Detector calibration uncertainty is a separate, shared input. Energy, motor,
and lattice uncertainties also weight the fit. **Fit crystal
orientation** ranks candidate assignments, shows pixel residuals and local
orientation uncertainty, and reports when the measurements leave a rotation
undetermined. Candidate details include the direct-lattice orientation matrix
at every recorded pose. Select a candidate and pose, then **Update crystal
orientation at selected pose** to install the corresponding initial orientation
and experimental conditions. Projects and fitted results can be saved and
loaded separately as JSON.

### Two-Parameter Mapping Slices

In **Single Crystal**, select **Motors Mapping for Catching Braggs**.
The two plotted parameters remain fixed. Under **Additional Mapping Sliders**,
enable **Scan additional parameters** and add ranges for other sample rotations
or sample-chain motors, energy, or full bandwidth (`e_bandwidth`, in percent).
An extra parameter cannot also be one of the plotted axes. This option is off
for older GUI states; new states retain the checkbox, ranges, and shared Euler
order.

Legacy scans first apply the initial sample orientation, then apply all scan
rotation components together in **Rotation order** beside the legacy sample
Euler angles. The table row
order does not change rotation order. For motor-chain geometry, the physical
chain order is retained and extra motors set the unplotted motor angles.

Unlike the compact reciprocal explorer, this mode computes every requested
mapping slice. The amount of work multiplies with each added range. Parallel
processing operates on the existing mapping rows, and **Stop safely** cancels
the computation without replacing the previous completed result. Start with
coarse extra ranges. Sliders, previous/next buttons, and numeric value boxes select computed slices;
they do not repeat the Bragg search. **Save Results** stores all slices in HDF5,
and **Load Results** restores them independently of the current sample inputs.
Points, band outlines, and filled bands remain available. Detector-hit filtering
is applied to the displayed slice using the current detector, leaving the saved
Bragg results intact.

**Mapping renderer** selects Matplotlib (the compatible default) or
**Plotly (browser)** and is retained in GUI states. Plotly supports ordinary
maps and saved mapping slices, with clickable reflection legends, hover values,
sliders, and numeric entry. Points and outlines use WebGL; filled bands use
separate occupied-cell polygons so disconnected branches are not joined.
For sliced maps, keep Python running: a loopback-only viewer sends only the
selected slice, caches prepared frames with a 48-entry/64-MiB limit, and prepares
neighboring slices in the background. Slider requests are coalesced after a
brief pause, and obsolete responses are not displayed. Filtering and band
construction run outside the GUI thread. Revisits are faster while cached;
the first uncached slice and browser rendering can still take time. Replot
after changing detector geometry to create a fresh view and cache. Closing the
GUI or **Close All Plots** shuts down the local viewers. Changing renderer
does not repeat the simulation or modify saved numerical results.

Legacy energy/rotation mappings use an optimized path: geometry is evaluated
once per sampled rotation, and bandwidth intervals identify the accepted
sampled energies. Boundary candidates are rechecked with the original Bragg
formula. Parallel execution uses bounded shared-memory orientation batches,
avoiding thousands of process tasks and repeated process startup per slice.
Rotation/rotation and motor-chain scans retain their existing execution paths.
The log names each distinct slice and reports its selected extra-parameter
values. Progress restarts for each slice; this is not a duplicate submission.
Fine grids can still produce large results: memory, saving, detector filtering,
and plotting costs grow with the number of accepted reflection points.

Scripts can pass `extra_scan_ranges={'rotz': (-20, 20, 10)}` to
`scan_two_parameters_for_Bragg_condition`, or named sample motors to
`scan_two_motors_for_Bragg_condition`. Without this argument the return value is
the original hkl-to-points dictionary. With it, the return value is a structured
mapping-slice dataset suitable for
`SimulationResult('parameter_mapping', data, {})`, `save_result`, and `plot_result`.

### Reciprocal Space and Ewald Construction

**Scan configurations** is optional and disabled by default. Enable it under
`simulate_reciprocal`, add parameter rows, and enter start, stop, and step.
Legacy geometry supports `energy` (eV), `sam_rotx/y/z`, and `det_rotx/y/z`
(degrees). Motor-chain geometry offers `sample:<motor>` and `detector:<motor>`
instead of legacy Euler rotations. Both offer `e_bandwidth`, displayed as
**ΔE/E [% full width]**: 2 means a full 2% bandwidth (±1% about the central
energy). Values must be at least zero and less than 200. Unscanned parameters retain their current
values. Each slider selects a sampled value, not an interpolated configuration;
the stop is included only when reached by the step, never overshot.

The compact scan stores one reflection set, the sampled axes, and the geometry.
The selected configuration and nearby slider positions are evaluated; the Cartesian product is not
computed or stored. For example, 8001 energies and 91 angles provide 728091
selectable configurations without allocating that many frames. The old 2000
configuration limit is removed. Up to eight axes are supported, each with at
most one million samples. Browser indexing uses exact integers even for very
large combinations. Older dense HDF5 scans remain readable.

**Parallel processing** and **Workers** are available for reciprocal scans.
The viewer prepares up to two neighboring samples in each direction per axis,
using shared-memory worker threads. With parallel processing enabled, up to
four background workers are used, limited to Workers minus one (minimum one),
leaving capacity for the selected configuration. Without it, one background
worker prepares neighbors sequentially. Individual transforms use vectorized
NumPy without nested worker pools. Existing full-grid mapping workflows retain their process
pools. Parallel settings persist in GUI states and can be changed when
replotting. No parallel setting triggers enumeration of all configurations.

Both renderers provide independent sliders, previous/next buttons, and editable numeric value boxes.
Typed values snap to the nearest sampled point; values outside the saved range
are rejected. Buttons move one sampled position and stop at the endpoints.
Newly added parameter rows have blank ranges and use a compact
dropdown editor instead of an oversized embedded widget. Plot titles include
energy, wavelength, full bandwidth, sample pose, and available detector pose.

Plotly scan views now use a small **loopback-only local viewer**, with an
unguessable session URL. No external service is contacted; Plotly.js is bundled.
Each viewer keeps a cache bounded to 48 entries and 64 MiB of numerical data
or serialized frames (whichever limit is reached first). Selecting a new
position cancels queued obsolete preparation; the selected calculation is not
queued behind background work. Nearby and revisited positions become faster
as the cache fills, but plot rendering still takes time. Cached frames are
temporary, not added to the saved HDF5 file. Keep the
Python application running while using the browser sliders. An optional HTML
export is a link to that live session, not an independent offline animation.
For portable storage, save the HDF5 result and reopen it for plotting later.
Close All Plots or closing the simulation GUI stops local viewers; browser tabs
are not forcibly closed. At most eight live viewers are retained.
Single-configuration Plotly views still use self-contained offline HTML.
Matplotlib views have sliders, value boxes, and clickable legend entries.
Save/load retains all reflections, including those outside Bragg condition.
Changing the renderer does not rerun reflection enumeration.
The current detector can be applied during replotting: saved detector scans
retain their relative transforms, applied to the newly declared base pose.
This moves the detector trajectory as a whole; it does not redefine the saved
motor chain or sampled angles. With no detector override, scripts use the saved
detector. GUI states store the scan checkbox and table, not numerical frames;
older states keep single-configuration mode.

**Require detector hit** highlights detected Bragg reflections in green; other
Bragg reflections remain red. The hkl-list colors match the points. This is a
display calculation, not a destructive filter of the saved reciprocal lattice.
It uses central-energy rays and rectangular detector bounds, not pixel masks.

**Stop safely** in simulation task windows requests cooperative cancellation.
Scans check between points/solver seeds, cancel queued process work, and wait
for active workers to exit. Incomplete simulation results are discarded; the
previous completed result remains available. A currently executing numerical
solver or native-library call may need to finish before stopping. No GUI thread
or worker process is forcibly terminated. Reciprocal-space scan controls appear
only for `simulate_reciprocal`; the 3D detector view has its own optional
slider controls. Reciprocal-space results have their own section title.
Successful task windows close automatically. Errors and cancellations remain
visible. The progress bar shows measured tqdm progress when available, and a
busy indicator for lattice preparation or rendering with no known total.
The reciprocal explorer can prepare much faster than a two-parameter mapping:
the latter evaluates every grid point, whereas the explorer evaluates selected
states on demand. These timings are not equivalent computation workloads.

### Single-Crystal Detector Sliders

In **Single Crystal**, select **Detector 2D Simulation** or **Laboratory 3D Simulation** and enable
**Scan configurations** under the corresponding simulation sliders. Each view
keeps its own saved ranges. Add one or more parameters and enter their
start, stop, and step magnitude. A descending range (for example, −5 to −25)
also uses a positive step magnitude. The plot gets a slider, previous/next
buttons, and editable value box for each parameter. Buttons follow the sampled
sequence, including descending ranges, and stop at its endpoints. Legacy mode offers energy, full bandwidth, and sample
and detector Euler rotations. Geometry mode offers the active sample and detector motors
instead of Euler rotations. Each selected position recalculates the Bragg rays
and detector view; typed values snap to the nearest sampled position. The
ordinary view is unchanged when scanning is disabled. Scripts can pass
`scan_ranges={'sam_roty': (0, 20, 1)}` to `simulate_2d` or `simulate_3d`, or
use keys such as `sample:omega` with the geometry-aware variants.

**Rotation order** beside the legacy sample Euler angles sets the order used by
the 2D, 3D, and reciprocal simulations, with or without sliders. It defaults to
`xyz` for existing sessions. Geometry motor chains use their declared chain order.

In **Single Crystal**, select **Ewald Construction Simulation**. The view uses the current
lattice, initial crystal orientation, sample rotations (or motor-chain angles),
beam energy, bandwidth, and `qmax`; no detector configuration is required.
The reciprocal origin is excluded from the reflection list. Q coordinates are
in inverse angstrom, with the incident beam along +x. Each Ewald sphere has
radius `2*pi/lambda` and centre `(-2*pi/lambda, 0, 0)`, so it passes through Q=0.
The bandwidth is the full energy width in percent, consistent with the other
single-crystal simulations.

**Plotly (browser)** is the default renderer. It groups reflections into two
interactive point sets (inside/outside the Bragg condition), with hkl and Q in
the hover information. Hovering over a reflection in either renderer shows its
hkl, Q components, |Q|, azimuth, and elevation. Lab azimuth is 0° along +Qz
(up), −90° toward +Qy (left), and +90° toward −Qy (right) in the unrotated
detector view. Elevation is measured above the Qx–Qy plane toward +Qz.
Angles are in degrees and are undefined where the corresponding direction
cannot be determined. A scrollable **Bragg hkl** list below the right-hand
legend lists every distinct reflection currently in Bragg condition, without
adding one 3D trace per reflection. Matplotlib includes the hkl entries in its
right-hand legend, using additional columns for long lists.
The central-energy sphere and optional bandwidth
boundaries can be toggled using the legend. For single configurations, the generated HTML includes its
JavaScript and works offline, without a server or uploading data. Its local
path is printed in the log. The browser handles rotation and zoom independently
of the simulation GUI; **Close All Plots** does not close browser tabs.
**Matplotlib** remains available as an alternative renderer.
The 2D and 3D single-crystal detector views show the same Q and angular details
when you hover over a Bragg spot, including when intensity coloring is enabled.

**Save Results...** stores the numerical reciprocal-space result in HDF5;
**Load Results...** followed by **Plot Loaded / Last Results** redraws it with
the currently selected renderer and visibility options, without recalculating
the lattice. The saved result retains its original physical conditions. GUI
states store the new display options; older GUI states use their defaults.
Plotly is included in the `gui` extra. Existing installations without it can
install `plotly>=5.18` or use Matplotlib.

For scripts, the existing four positional arguments remain supported:

```python
from trxrdpy.simulation import plot

fig = plot.plot_reciprocal(
    Q_hkls, hkls, wavelength_m, 1.5,
    backend="plotly", html_path="reciprocal.html", show=False,
)
```

`Q_hkls` and `hkls` must be matching N-by-3 arrays; wavelength is in metres.
Alternatively, `single_crystal.simulate_reciprocal(...)` builds the reflection
set from sample parameters before plotting.

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

from trxrdpy.analysis.MaxIV_FemtoMAX import azimint as femtomax_azimint
from trxrdpy.analysis.Spring8_SACLA import azimint as sacla_azimint
from trxrdpy.analysis.Spring8_SACLA import datared as sacla_datared
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
        │   ├── poni.py
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
        │       │   ├── path_service.py
        │       │   └── simulation_service.py
        │       ├── tabs/
        │       │   ├── __init__.py
        │       │   ├── polycrystalline_tab.py
        │       │   └── single_crystal_tab.py
        │       └── widgets/
        │           ├── __init__.py
        │           ├── geometry_panel.py
        │           ├── matrix_rotation_window.py
        │           └── path_widgets.py
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
            │   ├── azimint.py
            │   ├── single_shot_azimint.py
            │   └── ping_references_default.csv
            ├── Spring8_SACLA/
            │   ├── __init__.py
            │   ├── datared.py
            │   ├── azimint.py
            │   ├── single_shot_azimint.py
            │   └── pbs/
            │       ├── parallel_job_sender.sh
            │       └── single_shot_1d_job_sender.sh
            ├── differential_analysis.py
            ├── fitting.py
            ├── calibration.py
            └── gui/
                ├── __init__.py
                ├── main_window.py
                ├── runtime_guard.py
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
                    ├── polarization_widget.py
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

Shared representative-2D-image azimuthal-integration workflow.

This layer is currently used by:

- **Max IV FemtoMAX**
- **SPring-8 SACLA**

It is one of the two final-pattern sources supported by those facilities. Their
facility namespaces also expose single-shot 1D production and aggregation.

### `analysis.ESRF_ID09`

ID09-specific data-reduction and azimuthal-integration workflow.

At ESRF ID09, the route to generate `xy` files differs from the homogenized 2D-image workflow used elsewhere. The beamline-provided tools and data structure are handled through a dedicated facility-specific implementation.

### `analysis.MaxIV_FemtoMAX`

FemtoMAX-specific analysis entry points.

This section contains:

- beamline-specific data reduction
- metadata handling
- representative 2D image creation
- metadata-selected single-shot 1D production
- final-pattern integration from representative images or single-shot caches
- wrappers that preserve the facility-facing public API

### `analysis.Spring8_SACLA`

SACLA-specific analysis entry points.

This section contains:

- beamline-specific data reduction
- representative 2D image creation
- run/tag-based single-shot 1D production
- final-pattern integration from representative images or single-shot caches
- PBS job-submission helpers for representative-image and single-shot HPC workflows

### User-facing analysis APIs

These modules provide the user-facing analysis layer after `xy` files are available:

- `analysis.calibration`
- `analysis.fitting`
- `analysis.differential_analysis`

The calibration API also exposes `plot_detector_and_cake(...)`, which loads a
homogenized dark detector image, performs pyFAI `integrate2d` integration, and
plots the detector image and q/azimuth cake side by side. Detector axes can be
flipped independently, and applying the detector mask is optional.

Calibration always uses a representative 2D detector image. Once final `xy`
files are created by either supported source, visualization, fitting, and
differential analysis are shared across facilities.

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

The first **Session** tab contains saving/loading, a GUI-state JSON drop field,
notes, and the configuration summary, followed by **Polycrystalline** and
**Single Crystal**. A fourth **Extra Functions** tab contains **Matrix Rotation**,
**Equivalent Reflections**, **q / d / Angle Conversion**, and **pyFAI / Fit2D**.
Only **Close All Plots** remains at the bottom right. Both simulation tabs accept CIF file drops.
Sections can be collapsed without changing their values or enabled settings;
collapsed sections and utility inputs are included in GUI states. Older saved
states remain loadable.

**Equivalent Reflections** uses the current sample's space group and the existing
lattice symmetry method. It accepts one HKL or several, such as
`[(1, 0, 0), (2, 0, 0)]`, with results grouped by input reflection in an expandable
output area. **q / d / Angle Conversion** converts between q, d, theta, and 2theta
using an energy or wavelength, and identifies inaccessible Bragg angles.
**pyFAI / Fit2D** accepts a PONI file or manual
geometry values and converts in either direction using pyFAI. Units are shown
beside every field. Fit2D cannot preserve independent pyFAI `rot3`; conversion
back returns the canonical orientation. This utility does not modify simulation
parameters automatically.

**Save PONI...** is at the end of **Common Detector Parameters** in both
simulation tabs. It uses the current detector dimensions, edited pixel sizes,
binning, geometry, and beam energy without asking for them again. Predefined
detectors populate visible pixel counts and editable pixel sizes; simulations
use these as an explicit flat-detector template. The export converts the actual
detector transform to pyFAI coordinates, including motor-chain translations.
Overwriting requires confirmation. Detector-specific distortion maps and masks
are not represented by this flat-detector export. All input edits survive GUI
state save/load; older states recover their predefined detector's original sizes.

**Scan Computation and Results** in the single-crystal tab provides opt-in
**Parallel processing**, a worker count, **Save Results...**, **Load Results...**,
and **Plot Loaded / Last Results**. It covers fixed-energy HKL targeting,
detector collection scans, and two-parameter Bragg scans. No result file is
written automatically. HDF5 files store numeric results plus the input GUI state;
loading them does not change current controls or rerun a simulation. Targeting
plots use the detector/2D/3D plot checkboxes. Result files do not contain pickles.

Parallel scans use ordered independent rows and separate processes (four by
default when enabled). Legacy Euler targeting also supports parallel execution;
motor-chain inverse targeting deliberately retains serial continuation seeds
to preserve solution-branch continuity. Small scans can be faster serially.
For scripts, the high-level search functions accept `parallel=True, workers=4`;
use the usual `if __name__ == "__main__":` guard when starting processes.
Parallel controls, but not large computed results, are stored in GUI states.
Simulation runs in both tabs use non-modal background tasks, including powder
reflection preparation. Each run snapshots its inputs; editing fields or loading
a GUI state does not change an active computation. The active tab's Run button
is disabled until completion, while other tabs and tools remain available.
Figures are created on the GUI thread, not in computation workers. Progress and
errors appear in the task window and terminal. Closing the main application is
deferred while a simulation is running.

**Mapping display** offers **Points** (the default), **Band outlines**, and
**Filled bands**, for newly computed and saved two-parameter maps. Band modes
trace occupied cells on the original scan grid, retaining gaps, holes, and
separate branches; they do not connect across angular wraparound. Outlines show
the boundary of the accepted band, not a fitted central curve. Boundary precision
is limited by the scan steps. Numerical results are never reduced or modified.
Grid steps and the selected mode are saved in result metadata and GUI states.
Older result files recover scan steps from their stored GUI snapshot where
available; otherwise use Points. Band rendering uses sparse occupied-cell runs,
not a full rectangular grid: large maps retain their original scan resolution
without the previous eight-million-cell display limit. Filled bands merge
adjacent rectangles; outlines contain only exposed boundaries, including holes.
Mapping legends sit to the right of a square plotting area, automatically using
multiple columns to fit the available height. The initial figure widens to make
room for the legend; resizing keeps the plotting area square and reflows the
legend. Clicking a legend entry still toggles its reflection.
The GUI always computes and saves the complete Bragg-condition mapping. The
**Require detector hit** checkbox under **Scan Computation and Results** is a
non-destructive plotting filter for both new and loaded mappings. Change the
detector pose and click **Plot Loaded / Last Results** to view another detector
acceptance without repeating the scan; turn it off to view all saved points.
The original sample orientation and scan energy are retained, while the detector
uses its current Euler or motor-chain pose. The option is disabled by default
and saved in GUI states. Ray preparation runs in the background and is cached
for subsequent detector poses. Package callers can use
`plot_result(result, require_detector_hit=True, detector_obj=det)`.
Older unfiltered files recover their sample context from the saved GUI snapshot.
Files previously saved with points discarded by the computation-time filter
cannot recover those missing points; a warning identifies these partial maps.
The low-level scan functions retain their explicit computation-time filter for
API compatibility, but the GUI does not use it. Like the existing
2D/3D simulation, the filter uses the central-energy ray and rectangular detector
bounds; it does not model dead pixels or module gaps.
2D, 3D, and reciprocal titles show entered Euler angles with their application
order. Geometry views show the diffractometer convention and active motor values;
any equivalent Euler angles are labelled as fixed laboratory-axis `xyz` relative
to the initial sample orientation. Detector motor conversions use the same
explicit `xyz` label. Titles use the configured precision where available.

Both single-crystal and powder 2D and 3D detector views show a red cross at the
direct-beam intersection when the forward beam hits the detector, including
tilted and translated detector geometries.

Polycrystalline **simulate_2d** and **simulate_3d** project powder cones through
the full detector transform, subdivide ring curves according to their detector-
pixel curvature, and clip them to the detector edges. The 3D view draws smooth
cone surfaces, their detector-intersection curves, and a light display mesh on
the detector face. Neither mode needs a cone point count or allocates a full
detector pixel grid.

The 2D and 3D powder and single-crystal views also offer **Consider
intensities**. Supply a CIF containing atom sites, then select a colormap
(viridis by default) and optional minimum and maximum color limits. Reflections
use relative structure-factor intensity `|F(hkl)|²`; powder rings also use the
1D Lorentz factor. Single-crystal strengths use the maximum theoretical
`|F(hkl)|²` among all allowed reflections as their reference, including those
currently outside the Bragg condition or detector. Beam polarization can be
switched off or set with a factor
from −1 (vertical linear) through 0 (unpolarized) to +1 (horizontal linear).
The default is +1 for horizontally polarized synchrotron light.
Intermediate values describe partial polarization; an axis-offset field rotates
the polarization reference in degrees. The powder correction
varies around each ring according to the laboratory scattering direction.
The intensity views show a colorbar and label powder rings or single-crystal Bragg
spots with their HKL indices. Exact zero-intensity pixels are black. Optional
Gaussian noise uses a standard deviation relative to the strongest reflection;
optional Poisson background uses a mean relative to that reflection. These
display settings are saved in GUI sessions. The 2D powder display draws smooth
ring curves over a black detector; optional noise is shown on the detector
pixels. The returned experiment exposes `reflection_intensities` and
`detector_intensity_image`. In 3D, random noise appears only on the detector
plane, which uses a compact display mesh. Single-crystal 3D intensity views
include a clickable HKL legend with the relative intensity of each reflection.
The colorized mode above remains the lightweight
default.

Polycrystalline **simulate_1d** can optionally use the same detector settings.
Enable **Use detector coverage** in the 1D options to derive the reachable
q limit from the detector position, size, rotation, and beam energy. The
manual `qmax` field is used only when detector coverage is off. A blank azimuth
range selects all 360°. To select a sector, enter `(-15, 15)` or `-15,15` in degrees; the
lab-frame convention is 0° vertically up, -90° left, and +90° right when
looking along the beam toward the detector. The azimuth offset shifts this
reference without following detector rotation. Detector coverage determines
which peaks are visible; missing pixels do not dilute their intensities in the
azimuthal average. The Python API accepts a `Detector` via
`simulate_1d(..., detector=detector)` and
returns `detector_qmax` and `coverage_fraction` alongside the pattern.
Coverage uses the central beam energy and rectangular detector bounds; beam
bandwidth, pixel masks, and module gaps are not included.
Detector settings entered on either simulation tab carry over when switching
between Polycrystalline and Single Crystal.

The single-crystal 3D view uses the same lightweight detector face and visual
mesh. A colored dot marks each forward Bragg ray that falls within the detector
pixel bounds; rays that miss the detector have no hit dot.

Detector Euler angles retain their named axes for every rotation order, keeping
the 3D detector grid, Bragg positions, and direct-beam marker consistent. Custom
sample orientation matrices and the Matrix Rotation tool's mode, Euler rotation
order (default `xyz`), angles, and result survive GUI-state save/load without
requiring another import.

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
- Data Reduction for metadata, representative 2D images, and supported single-shot 1D caches
- calibration utilities
- detector-image and 2D-cake visualization
- Azimuthal Integration from representative 2D images or completed single-shot patterns
- 1D visualization
- differential analysis
- peak fitting
- task-output dialogs for long-running operations
- shared single-experiment metadata across tabs
- log output
- plot-window cleanup

The GUI layer is intentionally separated from the computational backend so that workflows can also be used programmatically.

When the 1D Viewer regenerates XY patterns, its q-point count follows **Number
of q points** in Azimuthal Integration rather than a separate viewer default.
ID09 synthetic fluence scans copy previously integrated delay curves and retain
their source q grid.

#### Fitted-property display modes

In Fitting Analysis, leave both display-mode boxes unchecked for absolute
values. **Normalize (0-1)** maps the selected reference to zero and the maximum
to one; **Relative variation (%)** displays `100 * (P - P_ref) / P_ref`.
These options are mutually exclusive. Reference lines and baseline error bars
use the same transformation, and normalized axes omit physical units. The
reference estimate is treated as fixed when scaling the existing baseline
spread; these are not newly calculated fit-parameter confidence intervals.
Zero denominators and unavailable references produce an error.

Single experiments use the reference controls in their series selection;
multiple experiments use each entry's reference. Merged delay entries retain
their existing pooled negative-delay baseline. References must exist in the
fitting CSV; changing the plot reference does not refit or rewrite that CSV.

Multiple fitting experiments can optionally specify an **Azimuthal selection**
per entry: `Full`, a center such as `30` for phi-averaged fits, or a sector such
as `(-15, 15)` for separate-phi fits. Blank fields retain the shared selection.
The corresponding API descriptor key is `phi_window`.

Multiple-fluence plots have an optional **Colormap** field. Leaving it blank
keeps the previous default colors (`cmap=None`); entering, for example,
`viridis` uses that Matplotlib colormap. All new controls and per-experiment
selections are included in saved GUI states, and older states remain loadable.

---

## Facility-specific workflow overview

The analysis pipeline is intentionally split because raw-data handling differs across facilities.

### Max IV FemtoMAX

Repeated scan numbers are processed only once, preserving their first-occurrence
order. For fluence scans, supply fluences aligned with either the original list
or the unique scans; repeated entries for a scan must agree on its fluence.

When ping timing is unreliable, `scan_delay_overrides_fs={scan: delay_fs}`
optionally assigns an exact delay to every detector shot of selected scans.
Unlisted scans retain normal ping timing; the default is `None`. The Analysis
GUI exposes **Scan delay overrides** and an **Override delay unit** in FemtoMAX
Data Reduction. Both paths write the same downstream metadata structure.
Changed assignments require metadata and derived-product regeneration; dark
scans ignore overrides. For fluence scans, assignments must match the selected
fixed delay. Use a fresh analysis subdirectory when reassignment would leave
obsolete products under old delay names.

- Uses facility-specific data reduction
- Always creates or locates a representative 2D image for calibration
- Can produce final `xy` files from homogenized 2D images
- Can instead integrate metadata-selected frames into a resumable single-shot 1D cache
- Can refresh the same final `xy` filenames while that cache is still growing
- Then uses the shared downstream analysis pipeline

### SPring-8 SACLA

- Uses facility-specific data reduction
- Some reduction steps may depend on beamline-specific software, legacy Python environments, VPN access, or HPC job submission
- Always creates or locates a representative 2D image for calibration
- Can produce final `xy` files from homogenized 2D images
- Can instead integrate metadata-selected run/tag pairs into a resumable single-shot 1D cache
- Retains the SACLA chunked/PBS execution model for large single-shot productions
- Then uses the shared downstream analysis pipeline

### ESRF ID09

- Does not use the same 2D homogenization route as FemtoMAX/SACLA
- Uses a different beamline-specific azimuthal-integration workflow to generate `xy` files
- Does not provide the single-shot 1D production mode
- Then uses the same downstream fitting and differential-analysis pipeline

In other words:

- **data reduction differs across facilities**
- **FemtoMAX and SACLA support representative-2D and single-shot-1D sources**
- **ID09 retains its dedicated image-based `xy` generation workflow**
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

## Authors and Collaborators

[Julio Guzman-Brambila](https://orcid.org/0000-0003-2147-2595),
[D. Léa](https://orcid.org/0009-0009-6931-9224),
[M. Lorenc](https://orcid.org/0000-0002-6877-8631),
[E. Janod](https://orcid.org/0000-0002-2430-9542),
[C. Mariette](https://orcid.org/0000-0001-8067-9591).

The ordered author list is also recorded in `CITATION.cff`, `.zenodo.json`,
and the PyPI package metadata. See `CITATION.cff` for a machine-readable citation.

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

# Offline photon attenuation length

The simulation GUI's **Extra Functions → Photon Attenuation Length** opens a
standalone scan. Enter a chemical formula directly or load it from a CIF, then
supply its density in g/cm³. Grazing-incidence depth is the default; bulk
length is also available. Energy controls use eV and a step size. The optional
incidence-angle scan evaluates the visible curve on demand. Choose the slider
motor before plotting: **Incidence angle** shows depth versus photon energy,
while **Photon energy** shows depth versus incidence angle. A numeric box beside
the slider selects an exact value, including positions between scan steps.
Previous/next buttons jump to adjacent sampled values while numeric entry
continues to accept values between them.
The window shows numerical points and saves the selected plot to PDF. Data
export defaults to compressed HDF5 (`.h5`), with CSV available for readable
exchange. Both formats store length values in meters and material metadata
once per file; the display unit stays in the window. HDF5 files load lazily,
so only the viewed curve enters memory. When plotting loaded data, angles
between saved steps use linear
interpolation between the two adjacent saved curves. Photon energies between
saved steps are handled similarly. The simulation GUI state saves the window's
inputs and plot preferences, not calculated curves. CIF density
is not assumed; check and enter the measured density. All 92 CXRO elements are
available offline.

```python
import numpy as np
from trxrdpy.simulation.attenuation import XRayMaterial
from trxrdpy.simulation.attenuation_plot import AttenuationData, plot_attenuation

energy_eV = np.arange(10000, 15001, 10)
material = XRayMaterial("SrTiO3", 5.12)
depth_m = material.grazing_penetration_depth(energy_eV / 1000, 1.0, unit="m")
data = AttenuationData(energy_eV, depth_m, [1.0], "grazing", "µm", "SrTiO3", 5.12)
plot_attenuation(data)
```

The optical index is `n = 1 - δ + iβ`. Bulk intensity attenuation length is
`λ/(4πβ)`. Normal grazing depth is `1/(2 Im kz)`, with
`kz = (2π/λ) sqrt(sin²α - 2δ + 2iβ)` and angle `α` measured from the
surface. Negative beamline `rotY` values can be converted to the physical
angle with `abs(rotY)` before calling the API, or through
`grazing_depth_from_rot_y`. The numerical API accepts keV while GUI, plots,
and CSV use eV. The `exact=True` option uses
`sqrt(n² - cos²α)` instead. These are homogeneous, semi-infinite optical
models; they do not include roughness, standing waves, diffraction sampling,
or finite films. See the [local data notes](src/trxrdpy/simulation/data/henke/README.md)
for provenance, interpolation, and adding elements.
