# Local CXRO atomic scattering factors

The `.nff` files in this directory are the original three-column tables from
the [CXRO/LBNL atomic scattering-factor database](https://henke.lbl.gov/optical_constants/asf.html),
retrieved 2026-11-19 from the site's current `sf.tar.gz` archive. Tables for
all 92 CXRO elements are included.
The Nb file includes the 2023 update. Cite the CXRO database and the source
references listed on its page when publishing results.

The tables are read locally; ordinary calculations do not use the network.
To add another element, download its `.nff` file from the CXRO periodic table
and place it here under the lowercase chemical symbol, or pass a directory
containing it as `data_dir` to `XRayMaterial`. Do not replace the original
energy grid: adjacent entries around absorption edges carry the discontinuity.
Interpolation is piecewise linear in energy. Undefined f1 values (the `-9999`
sentinel) and energies outside a table are rejected.
Three official tables (Mg, Er, Pt) contain rounded duplicate or slightly
out-of-order energies. The loader sorts those grids stably and keeps the last
row at an identical energy; the downloaded source files are unchanged.

`xrayutilities` already provides optical constants and bulk absorption length,
but its active packaged database uses Kissel rather than current CXRO factors.
The source `.nff` tables therefore remain the authoritative input here.
