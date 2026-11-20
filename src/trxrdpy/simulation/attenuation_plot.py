"""Plotting and CSV exchange for photon attenuation scans."""
from __future__ import annotations

import csv
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Callable
import weakref
import numpy as np

_DISPLAY_SCALE = {"m": 1., "nm": 1e9, "µm": 1e6, "um": 1e6}


@contextmanager
def _atomic_destination(path):
    """Replace an export only after the complete temporary file is written."""
    destination = Path(path)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def stepped_range(start, stop, step):
    """Positive step grid, including stop when it falls on the grid."""
    values = np.asarray((start, stop, step), dtype=float)
    if not np.all(np.isfinite(values)) or start < 0 or stop < start or step <= 0:
        raise ValueError("Use a nonnegative start and positive step, with stop at least start.")
    count = int(np.floor((stop - start) / step + 1e-10)) + 1
    return start + np.arange(count) * step


@dataclass
class AttenuationData:
    """Energy in eV, length in meters, and a selectable display unit."""
    energy_eV: np.ndarray
    length: np.ndarray | None
    angles_deg: np.ndarray
    mode: str
    unit: str
    formula: str = ""
    density_g_cm3: float | None = None
    calculator: Callable[[float], np.ndarray] | None = None
    energy_calculator: Callable[[float], np.ndarray] | None = None

    def __post_init__(self):
        self.energy_eV = np.asarray(self.energy_eV, dtype=float)
        self.angles_deg = np.atleast_1d(np.asarray(self.angles_deg, dtype=float))
        if self.length is not None:
            self.length = np.atleast_2d(np.asarray(self.length, dtype=float))
        if (self.energy_eV.ndim != 1 or len(self.energy_eV) < 1
                or not np.all(np.isfinite(self.energy_eV)) or np.any(self.energy_eV <= 0)
                or np.any(np.diff(self.energy_eV) <= 0) or not np.all(np.isfinite(self.angles_deg))
                or len(self.angles_deg) < 1 or np.any(np.diff(self.angles_deg) <= 0)):
            raise ValueError("Attenuation data must form a finite, complete energy/angle grid.")
        if self.length is None:
            if not callable(self.calculator):
                raise ValueError("Lazy attenuation data requires a curve calculator.")
        elif (self.length.shape != (len(self.angles_deg), len(self.energy_eV))
              or not np.all(np.isfinite(self.length)) or np.any(self.length <= 0)):
            raise ValueError("Attenuation length grid must match energy and angle arrays.")
        if self.mode not in ("bulk", "grazing") or self.unit not in ("nm", "µm", "um", "m"):
            raise ValueError("Invalid attenuation mode or length unit.")

    def curve_at(self, angle_deg: float) -> np.ndarray:
        """Return a curve in meters, computing or interpolating as needed."""
        angle = float(angle_deg)
        if not np.isfinite(angle) or not self.angles_deg[0] <= angle <= self.angles_deg[-1]:
            raise ValueError(f"Incidence angle must be between {self.angles_deg[0]:g}° and {self.angles_deg[-1]:g}°.")
        if self.calculator is not None:
            return _checked_curve(self.calculator(angle), len(self.energy_eV))
        if len(self.angles_deg) == 1:
            return self.length[0]
        right = int(np.searchsorted(self.angles_deg, angle, side="right"))
        if right == 0:
            return self.length[0]
        if right >= len(self.angles_deg):
            return self.length[-1]
        left = right - 1
        fraction = (angle - self.angles_deg[left]) / (self.angles_deg[right] - self.angles_deg[left])
        return self.length[left] * (1 - fraction) + self.length[right] * fraction

    def display_curve_at(self, angle_deg: float) -> np.ndarray:
        """Return a curve converted from meters to the selected display unit."""
        return self.curve_at(angle_deg) * _DISPLAY_SCALE[self.unit]

    def curve_at_energy(self, energy_eV: float) -> np.ndarray:
        """Return depth versus incidence angle at a selected energy, in meters."""
        energy = float(energy_eV)
        if not np.isfinite(energy) or not self.energy_eV[0] <= energy <= self.energy_eV[-1]:
            raise ValueError(f"Photon energy must be between {self.energy_eV[0]:g} and {self.energy_eV[-1]:g} eV.")
        if self.energy_calculator is not None:
            return _checked_curve(self.energy_calculator(energy), len(self.angles_deg))
        if self.length is None:
            return np.array([np.interp(energy, self.energy_eV, self.curve_at(angle))
                             for angle in self.angles_deg])
        right = int(np.searchsorted(self.energy_eV, energy, side="right"))
        if right == 0:
            return self.length[:, 0]
        if right >= len(self.energy_eV):
            return self.length[:, -1]
        left = right - 1
        fraction = (energy - self.energy_eV[left]) / (self.energy_eV[right] - self.energy_eV[left])
        return self.length[:, left] * (1 - fraction) + self.length[:, right] * fraction

    def display_curve_at_energy(self, energy_eV: float) -> np.ndarray:
        """Return depth versus angle in the selected display unit."""
        return self.curve_at_energy(energy_eV) * _DISPLAY_SCALE[self.unit]


def _checked_curve(values, count):
    curve = np.asarray(values, dtype=float)
    if curve.shape != (count,) or not np.all(np.isfinite(curve)) or np.any(curve <= 0):
        raise ValueError("Attenuation curve has an invalid shape or nonpositive/nonfinite lengths.")
    return curve


def plot_attenuation(data: AttenuationData, *, log_y=False, slider_motor="angle",
                     on_angle_change=None, on_selection_change=None,
                     ax=None, show=True):
    """Plot depth versus energy or angle, with a slider for the other motor."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, TextBox

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
    else:
        fig = ax.figure
        ax.clear()
    if slider_motor not in ("angle", "energy"):
        raise ValueError("slider_motor must be 'angle' or 'energy'.")
    energy_motor = slider_motor == "energy" and data.mode == "grazing" and len(data.angles_deg) > 1
    label = "Bulk attenuation length" if data.mode == "bulk" else "Normal penetration depth"
    x = data.angles_deg if energy_motor else data.energy_eV
    initial = data.energy_eV[0] if energy_motor else data.angles_deg[0]
    initial_curve = (data.display_curve_at_energy(initial) if energy_motor
                     else data.display_curve_at(initial))
    line, = ax.plot(x, initial_curve, lw=2, color="#2878a8")
    ax.set(xlabel="Incidence angle [°]" if energy_motor else "Photon energy [eV]",
           ylabel=f"{label} [{data.unit}]")
    ax.grid(True, which="both", alpha=.3)
    if log_y:
        ax.set_yscale("log")
    # Leave clearance for the native Matplotlib toolbar above the canvas.
    fig.subplots_adjust(left=.14, right=.96, top=.76,
                        bottom=.25 if len(data.angles_deg) > 1 else .17)

    def update(value):
        value = float(value)
        curve = (data.display_curve_at_energy(value) if energy_motor
                 else data.display_curve_at(value))
        line.set_ydata(curve)
        title = f"{label}: {data.formula}" if data.formula else label
        if energy_motor:
            title += f"\nPhoton energy: {value:g} eV"
        elif data.mode == "grazing":
            title += f"\nIncidence angle: {value:g}°"
        ax.set_title(title, pad=12)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
        if not energy_motor and on_angle_change is not None:
            on_angle_change(value, curve)
        if on_selection_change is not None:
            on_selection_change("energy" if energy_motor else "angle", value, x, curve)

    if len(data.angles_deg) > 1:
        slider_ax = fig.add_axes([.14, .08, .42, .035])
        limits = data.energy_eV if energy_motor else data.angles_deg
        slider = Slider(slider_ax, "Energy" if energy_motor else "Angle",
                        float(limits[0]), float(limits[-1]), valinit=float(limits[0]))
        textbox_ax = fig.add_axes([.80, .063, .15, .065])
        textbox = TextBox(textbox_ax, "Energy [eV] " if energy_motor else "Angle [°] ",
                          initial=f"{limits[0]:g}")
        syncing = False

        def slider_changed(value):
            nonlocal syncing
            update(value)
            if not syncing:
                syncing = True
                textbox.set_val(f"{value:.6g}")
                syncing = False

        def textbox_changed(value):
            nonlocal syncing
            if syncing:
                return
            try:
                selection = float(value)
                if not np.isfinite(selection) or not limits[0] <= selection <= limits[-1]:
                    raise ValueError
            except ValueError:
                textbox.label.set_color("#b00020")
                fig.canvas.draw_idle()
                return
            textbox.label.set_color("black")
            syncing = True
            slider.set_val(selection)
            syncing = False

        slider.on_changed(slider_changed)
        textbox.on_submit(textbox_changed)
        slider.valtext.set_visible(False)
        ax._attenuation_slider = slider
        ax._attenuation_angle_box = textbox
    update(initial)
    if show:
        plt.show()
    return ax


def save_attenuation_csv(path, data: AttenuationData):
    """Export meters as three columns, with material metadata written once."""
    with _atomic_destination(path) as temporary:
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            metadata = {"mode": data.mode, "formula": data.formula,
                        "density_g_cm3": data.density_g_cm3}
            handle.write("# trxrdpy-attenuation-v2: " + json.dumps(metadata) + "\n")
            writer = csv.writer(handle)
            writer.writerow(["photon_energy_eV", "incidence_angle_deg", "attenuation_length_m"])
            for angle in data.angles_deg:
                row = data.curve_at(angle)
                writer.writerows((energy, angle if data.mode == "grazing" else "", length)
                                 for energy, length in zip(data.energy_eV, row))


def load_attenuation_csv(path, *, display_unit="µm"):
    """Load current or legacy CSV; keep display units in the caller's window."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        first = handle.readline()
        if first.startswith("# trxrdpy-attenuation-v2: "):
            metadata = json.loads(first.split(": ", 1)[1])
        else:
            metadata = None
            handle.seek(0)
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        columns = [name for name in header if name.startswith("attenuation_length_")]
        if ("photon_energy_eV" not in header or "incidence_angle_deg" not in header
                or len(columns) != 1 or metadata is None and "mode" not in header):
            raise ValueError("CSV is missing required attenuation columns.")
        column = columns[0]
        source_unit = column.removeprefix("attenuation_length_")
        if source_unit not in _DISPLAY_SCALE:
            raise ValueError(f"Unsupported length unit in CSV: {source_unit}")
        groups = {}
        mode = formula = density = None
        for row in reader:
            if metadata is None:
                row_mode, row_formula = row["mode"], row.get("formula", "")
                if mode is None:
                    mode, formula = row_mode, row_formula
                    density = float(row["density_g_cm3"]) if row.get("density_g_cm3") else None
                elif (row_mode, row_formula) != (mode, formula):
                    raise ValueError("CSV mixes different materials or modes.")
            angle = float(row["incidence_angle_deg"]) if row["incidence_angle_deg"] else 0.
            groups.setdefault(angle, []).append((float(row["photon_energy_eV"]), float(row[column])))
    if not groups:
        raise ValueError("CSV has no data rows.")
    if metadata is not None:
        mode, formula, density = metadata["mode"], metadata["formula"], metadata.get("density_g_cm3")
    angles = np.array(list(groups))
    energy = np.array([point[0] for point in groups[angles[0]]])
    for angle in angles:
        if not np.array_equal(energy, [point[0] for point in groups[angle]]):
            raise ValueError("CSV energy grids differ between angles.")
    length = np.array([[point[1] for point in groups[angle]] for angle in angles]) / _DISPLAY_SCALE[source_unit]
    return AttenuationData(energy, length, angles, mode,
                           display_unit, formula, density)


def save_attenuation_h5(path, data: AttenuationData):
    """Write a compressed, chunked HDF5 grid with SI lengths and one metadata set.

    Rows are calculated and written one angle at a time, so saving a fine scan
    does not allocate the full two-dimensional result in memory.
    """
    import h5py

    shape = (len(data.angles_deg), len(data.energy_eV))
    with _atomic_destination(path) as temporary:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["format"] = "trxrdpy-attenuation"
            handle.attrs["version"] = 1
            handle.attrs["mode"] = data.mode
            handle.attrs["formula"] = data.formula
            if data.density_g_cm3 is not None:
                handle.attrs["density_g_cm3"] = data.density_g_cm3
            energy = handle.create_dataset("photon_energy_eV", data=data.energy_eV)
            energy.attrs["units"] = "eV"
            angles = handle.create_dataset("incidence_angle_deg", data=data.angles_deg)
            angles.attrs["units"] = "deg"
            lengths = handle.create_dataset(
                "attenuation_length_m", shape=shape, dtype="f8",
                chunks=(min(32, shape[0]), min(2048, shape[1])),
                compression="gzip", compression_opts=4, shuffle=True, fletcher32=True)
            lengths.attrs["units"] = "m"
            for index, angle in enumerate(data.angles_deg):
                lengths[index, :] = data.curve_at(float(angle))


def load_attenuation_h5(path, *, display_unit="µm") -> AttenuationData:
    """Open a saved HDF5 scan lazily; only selected curves enter memory."""
    import h5py

    handle = h5py.File(path, "r")
    try:
        if handle.attrs.get("format") != "trxrdpy-attenuation" or handle.attrs.get("version") != 1:
            raise ValueError("Not a supported XRDpy attenuation HDF5 file.")
        energy = handle["photon_energy_eV"][:]
        angles = handle["incidence_angle_deg"][:]
        lengths = handle["attenuation_length_m"]
        if lengths.shape != (len(angles), len(energy)) or lengths.attrs.get("units") != "m":
            raise ValueError("Invalid attenuation grid or length units in HDF5 file.")

        def curve_at(angle):
            right = int(np.searchsorted(angles, angle, side="right"))
            if right == 0:
                return lengths[0, :]
            if right >= len(angles):
                return lengths[-1, :]
            left = right - 1
            fraction = (angle - angles[left]) / (angles[right] - angles[left])
            return lengths[left, :] * (1 - fraction) + lengths[right, :] * fraction

        def curve_at_energy(selected):
            right = int(np.searchsorted(energy, selected, side="right"))
            if right == 0:
                return lengths[:, 0]
            if right >= len(energy):
                return lengths[:, -1]
            left = right - 1
            fraction = (selected - energy[left]) / (energy[right] - energy[left])
            return lengths[:, left] * (1 - fraction) + lengths[:, right] * fraction

        data = AttenuationData(energy, None, angles, handle.attrs["mode"], display_unit,
                               handle.attrs["formula"], handle.attrs.get("density_g_cm3"),
                               curve_at, curve_at_energy)
        data._h5_handle = handle
        data._h5_finalizer = weakref.finalize(data, handle.close)
        return data
    except Exception:
        handle.close()
        raise
