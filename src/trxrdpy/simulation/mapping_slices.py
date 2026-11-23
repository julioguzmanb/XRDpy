"""Computed two-axis mappings with independent extra-parameter slices."""
from functools import wraps
import inspect
import itertools
import json
import math

import numpy as np
from tqdm import tqdm

from .parallel import check_cancelled
from .slider_navigation import add_step_buttons, two_column_bounds


def extra_mapping_scan(function):
    @wraps(function)
    def wrapped(*args, extra_scan_ranges=None, **kwargs):
        if not extra_scan_ranges:
            return function(*args, **kwargs)
        from .reciprocal_scan import scan_axes
        from .mapping_filter import mapping_context
        execution = {key: kwargs.pop(key) for key in ('parallel', 'workers') if key in kwargs}
        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        base = dict(bound.arguments)
        motor = 'motor1_name' in base
        prefix = 'motor' if motor else 'param'
        names = [base[prefix+str(i)+'_name'] for i in (1, 2)]
        if motor:
            chain = base['geometry'].sample if base.get('geometry') else base['motor_chain']
            allowed = set(chain.motor_names) | {'energy', 'e_bandwidth'}
        else:
            allowed = {'rotx', 'roty', 'rotz', 'energy', 'e_bandwidth'}
        if set(extra_scan_ranges) - allowed:
            raise ValueError('Unsupported extra mapping parameters: ' + ', '.join(sorted(set(extra_scan_ranges)-allowed)))
        if set(extra_scan_ranges) & set(names):
            raise ValueError('Extra slider parameters must differ from the two plotted axes.')
        axes = scan_axes(extra_scan_ranges)
        # Validate every axis before starting expensive work.
        plotted = scan_axes({name: base[prefix+str(i)+'_range'] for i, name in enumerate(names, 1)})
        slices = {}
        total = math.prod(len(axis) for axis in axes.values())
        count = math.prod(len(axis) for axis in plotted.values())
        print(f'Mapping scan: {total} distinct slices, {count:,} configurations per slice '
              f'({total*count:,} total); plotted axes {names[0]}, {names[1]}.', flush=True)
        for index, values in enumerate(tqdm(itertools.product(*axes.values()), total=total, desc='Mapping slices')):
            check_cancelled()
            options = dict(base)
            selected = dict(zip(axes, values))
            print(f'Mapping slice {index+1}/{total}: ' + ', '.join(f'{name}={value:g}' for name, value in selected.items()), flush=True)
            rotations = {key: value for key, value in selected.items() if key not in ('energy', 'e_bandwidth')}
            options.update({key: value for key, value in selected.items() if key in ('energy', 'e_bandwidth')})
            key = 'fixed_angles' if motor else 'scan_angles'
            options[key] = {**(base.get(key) or {}), **rotations}
            # Saved slices always retain all Bragg points; detector filtering is a view.
            options['require_detector_hit'] = False
            points = function(**options, **execution)
            slices[str(index)] = dict(points=points, context_json=json.dumps(mapping_context(options)))
        return dict(format='mapping-slices-v1', extra_axes=axes, slices=slices,
                    plot_ranges=np.array([base[prefix+str(i)+'_range'] for i in (1, 2)]),
                    plot_axes=tuple(names), grid_steps=np.array([abs(base[prefix+str(i)+'_range'][2]) for i in (1, 2)]))
    return wrapped


def slice_result(result, index):
    from .results import SimulationResult
    data = result.data
    item = data['slices'][str(index)]
    return SimulationResult('parameter_mapping', {'points': item['points']},
        {**result.metadata, 'mapping_context': json.loads(item['context_json']), 'axes': list(data['plot_axes']),
         'grid_steps': data['grid_steps'], 'points_are_complete': True})


def plot_slices(result, display_mode='points', require_detector_hit=False, detector_obj=None):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, TextBox
    from .results import plot_result
    from .mapping_filter import filtered_mapping
    from .reciprocal_scan import parameter_label
    axes = result.data['extra_axes']
    rows = math.ceil(len(axes)/2)
    fig, ax = plt.subplots(figsize=(11.5, 7+rows*.55))
    ax._mapping_bottom = .8+rows*.55
    sliders, boxes, buttons = [], [], []

    def redraw(_=None):
        index = 0
        labels = []
        for (name, samples), slider in zip(axes.items(), sliders):
            offset = int(slider.val)
            index = index*len(samples)+offset
            labels.append(f'{parameter_label(name)} = {samples[offset]:g}')
        view = slice_result(result, index)
        if require_detector_hit:
            view = filtered_mapping(view, detector_obj)
        plot_result(view, display_mode=display_mode, mapping_axes=ax)
        for setter, interval in zip((ax.set_xlim, ax.set_ylim), result.data['plot_ranges']):
            low, high = sorted(interval[:2])
            margin = abs(interval[2])*.5
            setter(low-margin, high+margin)
        fig.suptitle('; '.join(labels), fontsize=10, wrap=True)
        for box, slider, samples in zip(boxes, sliders, axes.values()):
            box.eventson = False
            box.set_val(f'{samples[int(slider.val)]:g}')
            box.eventson = True
        fig.canvas.draw_idle()

    height = fig.get_figheight()
    for i, (name, samples) in enumerate(axes.items()):
        x, track_width, box_x, box_width, previous_x, next_x, button_width = (
            two_column_bounds(fig, i % 2, track_inches=2.43, box_inches=.99))
        y = (.3+.55*(rows-1-i//2))/height
        slider = Slider(fig.add_axes([x, y, track_width, .18/height]), '',
                        0, max(1, len(samples)-1), valinit=0, valstep=1)
        slider.ax.text(0, 1.7, parameter_label(name), transform=slider.ax.transAxes, fontsize=9)
        slider.valtext.set_visible(False)
        if len(samples) == 1:
            slider.set_active(False)
        sliders.append(slider)
        slider.on_changed(redraw)
        box = TextBox(fig.add_axes([box_x, y-.03/height, box_width, .25/height]),
                      '', initial=f'{samples[0]:g}')
        buttons.append(add_step_buttons(
            fig, slider, np.arange(len(samples)),
            [previous_x, y-.03/height, button_width, .25/height],
            [next_x, y-.03/height, button_width, .25/height]))
        def submit(text, values=samples, control=slider, field=box):
            try:
                value = float(text)
                if not np.isfinite(value) or not min(values) <= value <= max(values):
                    raise ValueError()
                control.set_val(int(np.argmin(abs(values-value))))
                field.ax.set_facecolor('white')
            except ValueError:
                field.ax.set_facecolor('#ffd8d8')
            fig.canvas.draw_idle()
        box.on_submit(submit)
        boxes.append(box)
    fig._mapping_sliders, fig._mapping_value_boxes = sliders, boxes
    fig._mapping_step_buttons = buttons
    redraw()
    plt.show()
    return ax
