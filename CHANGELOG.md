# Changelog

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
