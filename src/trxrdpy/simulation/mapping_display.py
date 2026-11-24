"""Grid-cell boundaries for dense Bragg maps, without joining separate branches."""
import numpy as np


def _hkl_tuple(key):
    parts = str(key).strip('[]()').replace(',', ' ').split()
    if len(parts) != 3:
        return None
    try:
        values = np.array([float(part) for part in parts])
    except ValueError:
        return None
    if np.all(np.isfinite(values)) and np.all(np.isclose(values, np.round(values))):
        return tuple(int(value) for value in values)
    return None


def hkl_label(key):
    """Display a saved reflection key as conventional (h, k, l)."""
    hkl = _hkl_tuple(key)
    return '(' + ', '.join(map(str, hkl)) + ')' if hkl is not None else str(key)


def mapping_axis_label(name):
    if name == 'energy':
        return 'energy [eV]'
    if name == 'e_bandwidth':
        return 'e_bandwidth [%]'
    return name if name == 'configuration' else name + ' [°]'


def mapping_color_context(metadata):
    context = metadata.get('mapping_context')
    if context is not None:
        return context
    if 'gui_state' in metadata:
        from .mapping_filter import _legacy_context
        try:
            return _legacy_context(metadata)
        except ValueError:
            pass
    return None


def mapping_palette(keys, context=None):
    """Assign distinct Jet colors in increasing |Q| order, then HKL order."""
    from matplotlib import colormaps, colors
    from . import sample
    keys = list(keys)
    reciprocal = None
    if context is not None and context.get('initial_orientation') is not None:
        try:
            reciprocal = sample.cal_reciprocal_lattice(
                np.asarray(context['initial_orientation'], dtype=float))
        except (ValueError, TypeError, np.linalg.LinAlgError):
            pass

    def order(key):
        hkl = _hkl_tuple(key)
        if hkl is None:
            return (float('inf'), (float('inf'),) * 3, str(key))
        vector = np.asarray(hkl, dtype=float)
        q = np.linalg.norm(sample.calculate_q_hkl(vector, reciprocal)) if reciprocal is not None \
            else np.linalg.norm(vector)
        return (round(float(q), 10), hkl, str(key))

    ordered = sorted(keys, key=order)
    groups = {}
    for key in ordered:
        groups.setdefault(order(key)[0], []).append(key)
    cmap = colormaps['jet'].resampled(max(256, len(ordered) * 32))
    palette = {}
    used = set()
    centers = np.linspace(0, 1, len(groups)) if len(groups) > 1 else [.5]
    span = min(.025, .5 / max(1, len(groups) - 1))
    for center, members in zip(centers, groups.values()):
        if len(members) == 1:
            positions = [center]
        elif center == 0:
            positions = np.linspace(0, span, len(members))
        elif center == 1:
            positions = np.linspace(1 - span, 1, len(members))
        else:
            positions = np.linspace(center - span / 2, center + span / 2, len(members))
        for index, (key, position) in enumerate(zip(members, positions)):
            rgb = np.asarray(cmap(position)[:3])
            rgb = np.clip(rgb * (1 + (index - (len(members) - 1) / 2) * .025), 0, 1)
            color = colors.to_hex(rgb)
            if color in used:
                base = int(color[1:], 16)
                for offset in range(1, 1 << 24):
                    color = f'#{(base + offset) % (1 << 24):06x}'
                    if color not in used:
                        break
            palette[key] = color
            used.add(color)
    return palette


def active_mapping_hkls(result):
    """Return HKLs having at least one saved point, in their original order."""
    from .mapping_storage import is_sliced_mapping
    data = result.data
    if not is_sliced_mapping(data):
        return [key for key, rows in data['points'].items() if len(rows)]
    slices = data['slices']
    if hasattr(slices, 'active_hkls'):
        return slices.active_hkls()
    keys = dict.fromkeys(key for item in slices.values() for key in item['points'])
    return [key for key in keys if any(len(item['points'].get(key, ())) for item in slices.values())]


def _scan_indices(points, steps):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Mapping points must be a finite N by 2 array.')
    if not len(points):
        raise ValueError('No mapping points.')
    if steps is None:
        raise ValueError('Scan steps are unavailable for this saved map. Use Points display.')
    steps = np.abs(np.asarray(steps, dtype=float))
    if steps.shape != (2,) or not np.isfinite(steps).all() or (steps <= 0).any():
        raise ValueError('Mapping grid steps must be two positive finite numbers.')
    origin = points.min(axis=0)
    coordinates = (points - origin) / steps
    if not np.isfinite(coordinates).all() or (coordinates >= 2**52).any():
        raise ValueError('Mapping coordinates exceed the precision of the scan grid.')
    indices = np.rint(coordinates).astype(np.int64)
    if not np.allclose(coordinates, indices, rtol=0, atol=1e-5):
        raise ValueError('Points are not on a regular scan grid. Use Points display.')
    return origin, steps, indices


def occupancy_grid(points, steps=None, max_cells=8_000_000):
    """Dense reference grid for small maps; rendering uses sparse cell runs."""
    origin, steps, indices = _scan_indices(points, steps)
    shape = indices.max(axis=0) + 3
    if np.prod(shape.astype(float)) > max_cells:
        raise ValueError('Band display grid is too large. Use Points or a smaller scan region.')
    # A zero border closes contours at scan limits; no angular wraparound is imposed.
    grid = np.zeros((shape[1], shape[0]), dtype=np.float32)
    grid[indices[:, 1] + 1, indices[:, 0] + 1] = 1
    axes = [origin[i] + (np.arange(shape[i]) - 1) * steps[i] for i in (0, 1)]
    return axes[0], axes[1], grid


def _row_runs(indices):
    """Return sorted (row, first column, exclusive last column) occupied runs."""
    cells = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
    cells = cells[np.r_[True, np.any(np.diff(cells, axis=0), axis=1)]]
    starts = np.flatnonzero(np.r_[True, (np.diff(cells[:, 1]) != 0)
                                 | (np.diff(cells[:, 0]) != 1)])
    ends = np.r_[starts[1:] - 1, len(cells) - 1]
    return np.column_stack((cells[starts, 1], cells[starts, 0], cells[ends, 0] + 1))


def _subtract_intervals(first, second):
    """Subtract sorted, disjoint integer intervals without expanding cells."""
    result = []
    j = 0
    for lo, hi in first:
        while j < len(second) and second[j][1] <= lo:
            j += 1
        k = j
        cursor = lo
        while k < len(second) and second[k][0] < hi:
            left, right = second[k]
            if left > cursor:
                result.append((cursor, min(left, hi)))
            cursor = max(cursor, right)
            if cursor >= hi:
                break
            k += 1
        if cursor < hi:
            result.append((cursor, hi))
    return result


def band_vertices(points, steps=None, *, filled=True):
    """Exact occupied-cell rectangles or exposed edges, with no dense grid.

    Storage scales with accepted samples and occupied runs, not the area of the
    scan bounding box. Adjacent identical runs are merged; gaps are never filled.
    """
    origin, steps, indices = _scan_indices(points, steps)
    runs = _row_runs(indices)
    rows = np.split(runs, np.flatnonzero(np.diff(runs[:, 0])) + 1)
    vertices = []
    if filled:
        active = {}
        previous_y = None

        def rectangle(key, start, stop):
            lo, hi = key
            vertices.append(((lo - .5, start - .5), (hi - .5, start - .5),
                             (hi - .5, stop - .5), (lo - .5, stop - .5)))

        for row in rows:
            y = int(row[0, 0])
            if previous_y is not None and y != previous_y + 1:
                for key, start in active.items():
                    rectangle(key, start, previous_y + 1)
                active = {}
            current = {tuple(pair) for pair in row[:, 1:]}
            for key in list(active):
                if key not in current:
                    rectangle(key, active.pop(key), y)
            for key in current:
                active.setdefault(key, y)
            previous_y = y
        for key, start in active.items():
            rectangle(key, start, previous_y + 1)
    else:
        previous_y, previous = None, []

        def horizontal(intervals, y):
            vertices.extend(((lo - .5, y), (hi - .5, y)) for lo, hi in intervals)

        for row in rows:
            y = int(row[0, 0])
            current = row[:, 1:]
            if previous_y is not None and y == previous_y + 1:
                horizontal(_subtract_intervals(previous, current), y - .5)
                horizontal(_subtract_intervals(current, previous), y - .5)
            else:
                if previous_y is not None:
                    horizontal(previous, previous_y + .5)
                horizontal(current, y - .5)
            previous_y, previous = y, current
        horizontal(previous, previous_y + .5)

        # Merge vertical boundary edges across consecutive rows at the same x.
        edges = np.concatenate((runs[:, [0, 1]], runs[:, [0, 2]]))
        edges = edges[np.lexsort((edges[:, 0], edges[:, 1]))]
        starts = np.flatnonzero(np.r_[True, (np.diff(edges[:, 1]) != 0)
                                     | (np.diff(edges[:, 0]) != 1)])
        ends = np.r_[starts[1:] - 1, len(edges) - 1]
        vertices.extend(((edges[a, 1] - .5, edges[a, 0] - .5),
                         (edges[a, 1] - .5, edges[b, 0] + .5))
                        for a, b in zip(starts, ends))
    return origin + np.asarray(vertices, dtype=float) * steps


def attach_mapping_legend(ax, handles, artists):
    """Keep a square axes box beside a right-hand, height-aware legend."""
    from math import ceil

    fig = ax.figure
    pick_targets = {}
    title = ax.get_title()
    ax.set_box_aspect(1)

    def layout(event=None):
        renderer = fig.canvas.get_renderer()
        width, height = fig.get_size_inches()
        dpi = fig.dpi
        left, right, bottom, top, gap = .85, .25, .75, .85, .25
        bottom = getattr(ax, '_mapping_bottom', bottom)
        available_height = max(.5, height - bottom - top)
        font_size = 11
        while True:
            rows = max(1, int((available_height * 72 - 2.5 * font_size) / (1.5 * font_size)))
            columns = min(len(handles), max(1, ceil(len(handles) / rows)))
            while True:
                legend = ax.legend(handles=handles, ncol=columns, title='(h, k, l)',
                                   fontsize=font_size, title_fontsize=font_size + 1,
                                   loc='center left', bbox_to_anchor=(1, .5),
                                   bbox_transform=fig.transFigure, borderaxespad=0.,
                                   columnspacing=1., handletextpad=.6)
                legend.set_in_layout(False)
                bounds = legend.get_window_extent(renderer)
                legend_width, legend_height = bounds.width / dpi, bounds.height / dpi
                if legend_height <= available_height or columns == len(handles):
                    break
                columns += 1
            side = min(available_height, width - left - right - gap - legend_width)
            if event is None or side >= min(2.5, available_height) or font_size <= 7:
                break
            font_size -= 1
        if event is None:
            # Widen the figure for additional legend columns, never stretch the axes.
            side = available_height
            width = max(width, left + side + gap + legend_width + right)
            fig.set_size_inches(width, height, forward=True)
        side = max(.1, side)
        plot_left = left + max(0, (width - left - right - gap - legend_width - side) / 2)
        plot_bottom = bottom + (available_height - side) / 2
        ax.set_position([plot_left / width, plot_bottom / height, side / width, side / height])
        legend.set_bbox_to_anchor(((plot_left + side + gap) / width,
                                   (bottom + available_height / 2) / height),
                                  transform=fig.transFigure)
        # Long mapping titles must not extend over the legend in narrow windows.
        lines, words = [], []
        for word in title.split():
            candidate = ' '.join(words + [word])
            length = renderer.get_text_width_height_descent(candidate, ax.title.get_fontproperties(), False)[0]
            if words and length > side * dpi:
                lines.append(' '.join(words))
                words = []
            words.append(word)
        ax.set_title('\n'.join(lines + [' '.join(words)]))
        pick_targets.clear()
        for handle, artist in zip(legend.get_lines(), artists):
            handle.set_picker(5)
            handle.set_alpha(1. if artist.get_visible() else .2)
            pick_targets[handle] = artist
        if event is not None:
            fig.canvas.draw_idle()

    def picked(event):
        artist = pick_targets.get(event.artist)
        if artist is not None:
            visible = not artist.get_visible()
            artist.set_visible(visible)
            event.artist.set_alpha(1. if visible else .2)
            fig.canvas.draw_idle()

    layout()
    ax._mapping_callbacks = [fig.canvas.mpl_connect('resize_event', layout),
                             fig.canvas.mpl_connect('pick_event', picked)]
