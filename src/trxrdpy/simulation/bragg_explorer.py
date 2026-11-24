"""Read and export reflection-specific views of complete Bragg mappings."""
import csv
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .mapping_slices import slice_result
from .mapping_storage import is_sliced_mapping, _axis_unit
from .results import SimulationResult


class BraggExplorer:
    def __init__(self, result):
        if result.kind != 'parameter_mapping':
            raise ValueError('The Bragg explorer requires a parameter mapping.')
        self.result = result
        self.sliced = is_sliced_mapping(result.data)
        self.axes = result.data['extra_axes'] if self.sliced else {}
        self.plot_axes = (list(result.data['plot_axes']) if self.sliced
                          else list(result.metadata.get('axes', ('roty', 'rotz'))))
        self.slices = result.data['slices'] if self.sliced else None
        self.count = math.prod(len(values) for values in self.axes.values()) if self.sliced else 1

    def positions(self, hkl):
        if not self.sliced:
            return [0] if len(self.result.data['points'].get(hkl, ())) else []
        if hasattr(self.slices, 'positions_for'):
            return self.slices.positions_for(hkl)
        return [index for index in range(self.count)
                if len(self.slices[str(index)]['points'].get(hkl, ())) > 0]

    def hkls(self):
        if not self.sliced:
            candidates = self.result.data['points']
        elif hasattr(self.slices, 'active_hkls'):
            return self.slices.active_hkls()
        else:
            candidates = dict.fromkeys(hkl for item in self.slices.values()
                                       for hkl in item['points'])
        return [hkl for hkl in candidates if self.positions(hkl)]

    def position_values(self, index):
        if not self.sliced:
            return ()
        offsets = np.unravel_index(index, tuple(len(values) for values in self.axes.values()))
        return tuple(float(values[offset]) for values, offset in zip(self.axes.values(), offsets))

    def point_count(self, index, hkl):
        if not self.sliced:
            return len(self.result.data['points'].get(hkl, ()))
        if hasattr(self.slices, 'point_count'):
            return self.slices.point_count(index, hkl)
        return len(self.slices[str(index)]['points'].get(hkl, ()))

    def view(self, index, hkl, detector_obj=None):
        if self.sliced:
            selected = slice_result(self.result, index, hkl=hkl)
        else:
            selected = SimulationResult('parameter_mapping',
                {'points': {hkl: self.result.data['points'].get(hkl, ())}}, self.result.metadata)
        if detector_obj is not None:
            from .mapping_filter import filtered_mapping
            selected = filtered_mapping(selected, detector_obj)
        return np.asarray(selected.data['points'][hkl], dtype=float).reshape(-1, 2)

    def application_order(self):
        if self.sliced and self.count:
            context = json.loads(self.slices['0']['context_json'] if not hasattr(self.slices, 'contexts')
                                 else self.slices.contexts[0])
        else:
            context = self.result.metadata.get('mapping_context', {})
        motors = context.get('motors')
        if motors:
            return ' → '.join(motor['name'] for motor in motors)
        return context.get('rotation_order', 'unknown')

    def export_csv(self, path, hkl, positions, detector_obj=None, cancel=None):
        """Stream selected rows to an atomic CSV export; return the row count."""
        destination = Path(path).expanduser()
        if destination.suffix.lower() != '.csv':
            destination = destination.with_suffix('.csv')
        selected = list(positions)
        available = set(self.positions(hkl))
        if any(index not in available for index in selected):
            raise ValueError('A selected position has no Bragg points for this HKL.')
        temporary = None
        written = 0
        try:
            with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                    dir=destination.parent, prefix='.bragg-export-', suffix='.tmp',
                    delete=False) as stream:
                temporary = Path(stream.name)
                writer = csv.writer(stream)
                names = list(self.axes) + self.plot_axes
                writer.writerow(['hkl'] + [f'{name} [{_axis_unit(name)}]' for name in names]
                                + ['application_order'])
                order = self.application_order()
                for index in selected:
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError('Export cancelled.')
                    prefix = [hkl, *self.position_values(index)]
                    points = self.view(index, hkl, detector_obj)
                    for row in points:
                        writer.writerow([*prefix, *map(float, row), order])
                    written += len(points)
            if cancel is not None and cancel.is_set():
                raise InterruptedError('Export cancelled.')
            os.replace(temporary, destination)
            temporary = None
            return written
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
