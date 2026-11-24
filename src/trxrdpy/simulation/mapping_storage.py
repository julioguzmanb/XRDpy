"""Sparse, on-demand HDF5 storage for mappings with extra scan axes."""
from collections.abc import Mapping
import json
import math
from pathlib import Path

import h5py
import numpy as np


FORMAT = 'mapping-slices-v2'
_TEXT = h5py.string_dtype(encoding='utf-8')


def is_sliced_mapping(data):
    return data.get('format') in ('mapping-slices-v1', FORMAT)


def _slice_name(index):
    return f'{index:06d}'


def _axis_unit(name):
    return 'eV' if name == 'energy' else '%' if name == 'e_bandwidth' else 'deg'


class H5SliceStore(Mapping):
    """Open the HDF5 file only when one plotted slice is requested."""

    def __init__(self, path, count, hkls, contexts):
        self.path = Path(path)
        self.count = count
        self.hkls = hkls
        self.contexts = contexts

    def __len__(self):
        return self.count

    def __iter__(self):
        return (str(index) for index in range(self.count))

    def __getitem__(self, key):
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise KeyError(key) from exc
        if index < 0 or index >= self.count or str(index) != str(key):
            raise KeyError(key)
        # Retain the complete HKL order expected by the existing plotters,
        # while keeping empty reflection datasets off disk.
        points = {hkl: np.empty((0, 2), dtype=float) for hkl in self.hkls}
        with h5py.File(self.path, 'r') as handle:
            slices = handle['data/slices']
            name = _slice_name(index)
            if name in slices:
                for dataset in slices[name].values():
                    points[dataset.attrs['hkl']] = dataset[()]
        return {'points': points, 'context_json': self.contexts[index]}

    def positions_for(self, hkl):
        if hkl not in self.hkls:
            return []
        name = f'hkl_{self.hkls.index(hkl):04d}'
        with h5py.File(self.path, 'r') as handle:
            lookup = handle['index/by_hkl']
            return list(map(int, lookup[name][()])) if name in lookup else []

    def active_hkls(self):
        with h5py.File(self.path, 'r') as handle:
            active = {dataset.attrs['hkl'] for dataset in handle['index/by_hkl'].values()}
        return [hkl for hkl in self.hkls if hkl in active]

    def point_count(self, index, hkl):
        if hkl not in self.hkls:
            return 0
        with h5py.File(self.path, 'r') as handle:
            path = f'data/slices/{_slice_name(index)}/hkl_{self.hkls.index(hkl):04d}'
            return handle[path].shape[0] if path in handle else 0

    def points_for(self, index, hkl):
        if hkl not in self.hkls:
            return np.empty((0, 2), dtype=float)
        with h5py.File(self.path, 'r') as handle:
            path = f'data/slices/{_slice_name(index)}/hkl_{self.hkls.index(hkl):04d}'
            return handle[path][()] if path in handle else np.empty((0, 2), dtype=float)


def write_mapping(handle, data):
    """Write only populated HKL/position pairs and an HKL lookup index."""
    if isinstance(data['slices'], H5SliceStore):
        # A loaded mapping has immutable on-disk data. Re-save it without
        # expanding every slice and every empty HKL into Python objects.
        with h5py.File(data['slices'].path, 'r') as source:
            for name in ('metadata', 'data', 'index'):
                source.copy(name, handle)
        return
    axes = data['extra_axes']
    count = math.prod(len(values) for values in axes.values())
    slices = data['slices']
    hkls = list(data.get('hkls', ()))
    seen = set(hkls)
    for index in range(count):
        for hkl in slices[str(index)]['points']:
            if hkl not in seen:
                hkls.append(hkl)
                seen.add(hkl)

    metadata = handle.create_group('metadata')
    metadata.attrs['format'] = FORMAT
    metadata.attrs['plot_axes'] = json.dumps(list(data['plot_axes']))
    metadata.attrs['extra_axis_order'] = json.dumps(list(axes))
    metadata.attrs['point_columns'] = json.dumps(list(data['plot_axes']))
    metadata.attrs['point_units'] = json.dumps([_axis_unit(name) for name in data['plot_axes']])
    metadata.attrs['points_are_complete'] = True
    if count:
        first_context = json.loads(slices['0']['context_json'])
        if first_context.get('rotation_order') is not None:
            metadata.attrs['rotation_order'] = first_context['rotation_order']
        if first_context.get('initial_orientation') is not None:
            metadata.create_dataset('initial_orientation', data=first_context['initial_orientation'])
        if first_context.get('motors') is not None:
            metadata.attrs['motors'] = json.dumps(first_context['motors'])
    plot_ranges = metadata.create_dataset('plot_ranges', data=data['plot_ranges'])
    plot_ranges.attrs['columns'] = json.dumps(['start', 'stop', 'step'])
    metadata.create_dataset('grid_steps', data=data['grid_steps'])
    metadata.create_dataset('tested_hkls', data=np.asarray(hkls, dtype=object), dtype=_TEXT)
    extra = metadata.create_group('extra_axes')
    for name, values in axes.items():
        axis = extra.create_dataset(name, data=values)
        axis.attrs['unit'] = _axis_unit(name)
    contexts = metadata.create_dataset('slice_contexts', shape=(count,), dtype=_TEXT)

    positions = handle.create_group('data').create_group('slices')
    index_group = handle.create_group('index').create_group('by_hkl')
    hkl_ids = {hkl: position for position, hkl in enumerate(hkls)}
    occurrences = {hkl: [] for hkl in hkls}
    dimensions = tuple(len(values) for values in axes.values())
    for index in range(count):
        item = slices[str(index)]
        contexts[index] = item['context_json']
        group = None
        for hkl, rows in item['points'].items():
            values = np.asarray(rows, dtype=float).reshape(-1, 2)
            if not len(values):
                continue
            if group is None:
                group = positions.create_group(_slice_name(index))
                offsets = np.unravel_index(index, dimensions)
                for (name, samples), offset in zip(axes.items(), offsets):
                    group.attrs[name] = samples[offset]
            dataset = group.create_dataset(f'hkl_{hkl_ids[hkl]:04d}', data=values,
                                           compression='gzip')
            dataset.attrs['hkl'] = hkl
            occurrences[hkl].append(index)
    for hkl, matching in occurrences.items():
        if matching:
            dataset = index_group.create_dataset(f'hkl_{hkl_ids[hkl]:04d}',
                                                 data=np.asarray(matching, dtype=np.int32))
            dataset.attrs['hkl'] = hkl


def read_mapping(path, handle):
    metadata = handle['metadata']
    names = json.loads(metadata.attrs['extra_axis_order'])
    axes = {name: metadata['extra_axes'][name][()] for name in names}
    hkls = list(metadata['tested_hkls'].asstr()[()])
    contexts = list(metadata['slice_contexts'].asstr()[()])
    return dict(format=FORMAT, extra_axes=axes,
                plot_axes=tuple(json.loads(metadata.attrs['plot_axes'])),
                plot_ranges=metadata['plot_ranges'][()], grid_steps=metadata['grid_steps'][()],
                hkls=hkls, slices=H5SliceStore(path, len(contexts), hkls, contexts))
