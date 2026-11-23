# Changelog

## 5.1.0

- Fit a single-crystal orientation matrix from measured Bragg pixels across
  linked sample poses, detector setups, energies, and motor readings. Search
  exact, symmetry-equivalent, or unknown HKL assignments, inspect ranked
  candidates and residuals, and save editable projects or fitted results.
- Add independent 2D and 3D single-crystal detector scan sliders for energy,
  bandwidth, Euler rotations, or named geometry motors. Add previous/next
  navigation to detector, reciprocal, mapping, and attenuation scan views.
- Add optional CIF-based relative-intensity displays to powder and
  single-crystal 2D/3D simulations, with polarization, colormaps, color limits,
  Gaussian noise, and Poisson background.
- Clarify entered Euler rotation order and geometry conventions in plot
  titles, and expand Bragg hover details with reciprocal-vector coordinates
  and laboratory angles.

- Add optional detector-limited 1D powder patterns, including laboratory-frame
  azimuth sectors, offsets, and partial-ring coverage in the API and GUI.
- Treat missing azimuths as absent observations in 1D patterns, preserving
  relative peak intensities for reflections visible in each selected sector.
- Show the detector-aware azimuth window and any offset in 1D plot titles.
- Accept parenthesized azimuth ranges such as `(-15, 15)` in the 1D GUI.
- Keep detector settings synchronized when switching between polycrystalline
  and single-crystal simulation tabs.
- Avoid a parallel BLAS solve for the 3x3 orientation calculation used by
  detector titles in background simulation tasks.
- Show full multi-slice mapping progress and estimated remaining time in task
  dialogs, and strip terminal control codes from their visible logs.
- Clarify azimuth units in Analysis plots and controls.
- Update the manual with the measured-peak orientation workflow, new scan
  controls, detector-limited 1D patterns, intensity views, and regenerated
  figures.

## 5.0.0

- Fix the 1D polycrystalline 2θ plot label and support d-spacing as a third
  output axis alongside q and 2θ.
- Make zero FWHM produce a stick pattern on all three axes and preserve the
  selected axis when restoring a Simulation GUI state.
- Generate 2D and 3D powder reflection families from a CIF or entered lattice
  and `qmax` in the GUI, while retaining explicit q/d reflection lists.
- Render continuous detector-clipped powder rings in 2D and show their actual
  detector intersections alongside smooth cone surfaces in 3D.
- Add reciprocal-space and mapping controls, portable simulation results,
  PONI export, and the new Extra Functions utilities documented in the manual.
- Expand analysis fitting and visualization workflows, including saved GUI
  state and multi-experiment views.
- Rebuild the user manual with current simulation figures and 5.0.0 examples.

## 4.0.3

- Add all five authors' ORCID identifiers to citation and Zenodo metadata,
  the PyPI description, and the manual.
- Correct D. Léa's affiliation to match M. Lorenc's affiliation at the
  Institut de Physique de Rennes.
- Update the packaged manual and release metadata to version 4.0.3.

## 4.0.2

- Add optional per-scan delay assignments to FemtoMAX reduction and its GUI.
  Manually timed and ping-timed scans share the standard downstream metadata.
- Deduplicate repeated FemtoMAX scan numbers while validating fluence assignments.
- Preserve older GUI states and reject reuse of metadata with changed timing assignments.
- Update the manual with timing-override examples and regeneration guidance.
- Credit Julio Guzman-Brambila, D. Léa, M. Lorenc, E. Janod, and C. Mariette,
  in that order, in package, citation, Zenodo, and documentation metadata.
- Publish to PyPI on GitHub release publication, validating the release tag
  against the package version first.
