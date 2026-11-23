"""Saved configuration grids and independent sliders for reciprocal-space views."""
from copy import deepcopy

import numpy as np

from . import reciprocal
from .deferred_plotting import gui_plot
from .slider_navigation import add_step_buttons, two_column_bounds


def parameter_label(name):
    return '\u0394E/E [% full width]' if name == 'e_bandwidth' else name


def scan_axes(ranges):
    if not isinstance(ranges, dict) or not ranges:
        raise ValueError('Select at least one scan parameter.')
    if len(ranges) > 8:
        raise ValueError('At most eight independent sliders are supported.')
    axes = {}
    for name, interval in ranges.items():
        if len(interval) != 3:
            raise ValueError('Each range needs start, stop, step.')
        start, stop, step = map(float, interval)
        if not np.isfinite([start, stop, step]).all() or step == 0 or (stop-start)*step < 0:
            raise ValueError('Invalid scan range for ' + name)
        size = int(np.floor((stop-start)/step + 1e-9)) + 1
        if size > 1_000_000:
            raise ValueError('Scan axis exceeds one million samples.')
        values = start + step * np.arange(size)
        if name == 'energy' and np.any(values <= 0):
            raise ValueError('All scanned energies must be positive (eV).')
        if name == 'e_bandwidth' and (np.any(values < 0) or np.any(values >= 200)):
            raise ValueError('Full energy bandwidth must be in [0, 200) percent.')
        axes[name] = values
    return axes


def build_scan(ranges, *, lattice_options, detector_obj=None, detector_angles=None):
    from .reciprocal_compact import build
    axes = scan_axes(ranges)
    geometry = lattice_options.get('geometry')
    allowed = {'energy', 'e_bandwidth'}
    if geometry is None:
        allowed.update('sam_rot' + axis for axis in 'xyz')
        allowed.update('det_rot' + axis for axis in 'xyz')
    else:
        allowed.update('sample:' + name for name in geometry.sample.motor_names)
        allowed.update('detector:' + name for name in geometry.detector.motor_names)
    unknown = set(axes) - allowed
    if unknown:
        raise ValueError('Unsupported scan parameters for this geometry: ' + ', '.join(sorted(unknown)))
    scans_detector = any(name.startswith(('det_', 'detector:')) for name in axes)
    if scans_detector and detector_obj is None:
        raise ValueError('Detector scans require a valid detector configuration.')
    return build(axes, lattice_options, detector_obj, detector_angles)


def configuration(scan, index, detector_obj=None, parallel=False, workers=None):
    if scan.get('format') == 'compact-v1':
        from .reciprocal_compact import configuration as compact_configuration
        return compact_configuration(scan, index, detector_obj, parallel, workers)
    from .detector import Detector
    if detector_obj is None and scan['detector_state'] is not None:
        state = dict(scan['detector_state'])
        transform = state.pop('transform')
        detector_obj = Detector(detector_type='manual', **state)
        detector_obj.set_transform(transform)
    det = deepcopy(detector_obj)
    if det is not None:
        det.set_transform(det.get_transform() @ scan['detector_deltas'][index])
    indices = np.unravel_index(index, tuple(len(values) for values in scan['axes'].values()))
    title = 'Reciprocal space and Ewald construction\n' + '; '.join(
        f'{name} = {values[i]:g}' for (name, values), i in zip(scan['axes'].items(), indices))
    return dict(q=scan['Q'][index], hkls=scan['hkls'], wavelength=scan['wavelengths'][index],
                bandwidth=scan['bandwidth'], title=title, detector_obj=det)


def plot_scan(scan, *, backend='plotly', show=True, html_path=None, detector_obj=None,
              parallel=False, workers=None, **options):
    if workers is not None and (int(workers) != workers or not 1 <= workers <= 1024):
        raise ValueError('Workers must be an integer between 1 and 1024.')
    if parallel:
        print(f'Reciprocal view: background preparation of neighboring configurations; up to {min(4, max(1, (workers or 4)-1))} background worker(s), reserving foreground capacity.')
    if backend == 'plotly':
        from .reciprocal_live import plot_live
        return plot_live(scan, detector_obj, options, parallel, workers, show, html_path)
    if backend == 'matplotlib':
        if html_path is not None:
            raise ValueError('HTML export requires Plotly.')
        return _matplotlib_scan(scan, detector_obj, options, show, parallel, workers)
    raise ValueError('Unknown reciprocal renderer.')


@gui_plot
def _matplotlib_scan(scan, detector_obj, options, show, parallel=False, workers=None):
    import matplotlib.pyplot as plt
    import weakref
    from matplotlib.widgets import Slider, TextBox
    from .reciprocal_cache import NeighborhoodCache
    compact = scan.get('format') == 'compact-v1'
    detector_obj = deepcopy(detector_obj)

    def prepare(index):
        values = configuration(scan, index, detector_obj)
        data = reciprocal._data(values['q'], values['hkls'], values['wavelength'], values['bandwidth'])
        q, _, accepted, _, _ = data
        hits = np.zeros(len(q), dtype=bool)
        if options.get('require_detector_hit'):
            if values['detector_obj'] is None:
                raise ValueError('A detector is required for detector-hit highlighting.')
            from .experiment import detector_hit_mask
            hits[accepted] = detector_hit_mask(q[accepted], values['wavelength'], values['detector_obj'])
        return data, hits, values['title']

    cache = NeighborhoodCache(prepare, [len(v) for v in scan['axes'].values()],
                              parallel=parallel, workers=workers)
    try:
        initial, _, _ = cache.get(0)
    except Exception:
        cache.close()
        raise
    max_bragg = int(initial[2].sum())
    columns = max(1, int(np.ceil((max_bragg + 6) / 22)))
    width = max(13.5, 10 + 1.6 * columns)
    rows = int(np.ceil(len(scan['axes'])/2))
    height = 8 + rows*.55
    fig = plt.figure(figsize=(width, height))
    fig._reciprocal_cache = cache
    weakref.finalize(fig, cache.close)
    fig.canvas.mpl_connect('close_event', lambda _: cache.close())
    ax = fig.add_subplot(111, projection='3d')
    bottom = (.65 + rows*.55)/height
    fig.subplots_adjust(left=.03, right=8/width, top=1-1.8/height, bottom=bottom)
    sliders = []
    boxes = []
    buttons = []
    if compact:
        from .reciprocal_compact import ranges as compact_ranges
        ranges = compact_ranges(scan)
    else:
        ranges = reciprocal._ranges(scan['Q'].reshape(-1, 3),
            2 * np.pi / (min(scan['wavelengths']) * 1e10) * (1 + scan['bandwidth'] / 200))

    def redraw(_=None):
        index = 0
        for slider, samples in zip(sliders, scan['axes'].values()):
            index = index * len(samples) + int(slider.val)
        data, hits, title = cache.get(index)
        q, hkls, accepted, k, bandwidth = data
        camera = ax.elev, ax.azim
        reciprocal._matplotlib(q, hkls, accepted, k, bandwidth, options.get('show_non_bragg', True),
            options.get('show_bandwidth', True), title, False, hits,
            options.get('require_detector_hit', False), existing=(fig, ax))
        ax.view_init(*camera)
        cols = max(1, int(np.ceil((int(accepted.sum())+6)/22)))
        new_width = max(13.5, 10 + 1.6*cols)
        if cols > columns:
            fig.set_size_inches(new_width, fig.get_size_inches()[1])
            fig.subplots_adjust(right=8/new_width)
        ax.set(xlim=ranges[0], ylim=ranges[1], zlim=ranges[2])
        for slider, samples in zip(sliders, scan['axes'].values()):
            slider.valtext.set_visible(False)
        for box, slider, samples in zip(boxes, sliders, scan['axes'].values()):
            box.eventson = False
            box.set_val(f'{samples[int(slider.val)]:g}')
            box.eventson = True
        fig.canvas.draw_idle()

    for i, (name, values) in enumerate(scan['axes'].items()):
        x, track_width, box_x, box_width, previous_x, next_x, button_width = (
            two_column_bounds(fig, i % 2, track_inches=3.25, box_inches=1.16))
        y = (.40 + .55*(rows-1-i//2))/height
        slider = Slider(fig.add_axes([x, y, track_width, .18/height]), '',
                        0, max(1, len(values)-1), valinit=0, valstep=1)
        slider.ax.text(0, 1.8, parameter_label(name), transform=slider.ax.transAxes, fontsize=9)
        if len(values) == 1:
            slider.set_active(False)
        slider.on_changed(redraw)
        sliders.append(slider)
        box = TextBox(fig.add_axes([box_x, y-.03/height, box_width, .25/height]),
                      '', initial=f'{values[0]:g}')
        buttons.append(add_step_buttons(
            fig, slider, np.arange(len(values)),
            [previous_x, y-.03/height, button_width, .25/height],
            [next_x, y-.03/height, button_width, .25/height]))
        def submit(text, samples=values, control=slider, field=box):
            try:
                requested = float(text)
                if not np.isfinite(requested) or not min(samples[0],samples[-1]) <= requested <= max(samples[0],samples[-1]):
                    raise ValueError()
                index = 0 if len(samples)==1 else int(np.floor((requested-samples[0])/(samples[1]-samples[0])+.5))
                control.set_val(min(len(samples)-1,max(0,index)))
                field.ax.set_facecolor('white')
            except ValueError:
                field.ax.set_facecolor('#ffd8d8')
                fig.canvas.draw_idle()
        box.on_submit(submit)
        boxes.append(box)
    fig._reciprocal_sliders = sliders
    fig._reciprocal_value_boxes = boxes
    fig._reciprocal_step_buttons = buttons
    redraw()
    if show:
        plt.show()
    return fig, ax
