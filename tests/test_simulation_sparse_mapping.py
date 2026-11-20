import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent, PickEvent, ResizeEvent
from matplotlib.path import Path

from trxrdpy.simulation.mapping_display import band_vertices
from trxrdpy.simulation.plot import plot_parameter_mapping


def _edge(a, b):
    return tuple(sorted((tuple(a), tuple(b))))


def _cell_edges(points):
    edges = set()
    for x, y in set(map(tuple, points)):
        corners = [(x - .5, y - .5), (x + .5, y - .5),
                   (x + .5, y + .5), (x - .5, y + .5)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            key = _edge(a, b)
            if key in edges:
                edges.remove(key)
            else:
                edges.add(key)
    return edges


@pytest.mark.parametrize('seed', range(12))
def test_sparse_bands_match_exact_occupied_cells(seed):
    rng = np.random.default_rng(seed)
    points = np.argwhere(rng.random((18, 23)) < .65)
    # Duplicates and arbitrary ordering must not change coverage or opacity.
    points = np.concatenate((points, points[:20]))
    rng.shuffle(points)
    segments = band_vertices(points, (1, 1), filled=False)
    actual = set()
    for start, end in segments:
        count = int(round(np.abs(end - start).max()))
        vertices = np.linspace(start, end, count + 1)
        for a, b in zip(vertices[:-1], vertices[1:]):
            key = _edge(a, b)
            assert key not in actual
            actual.add(key)
    assert actual == _cell_edges(points)

    rectangles = band_vertices(points, (1, 1), filled=True)
    area = ((rectangles[:, 1, 0] - rectangles[:, 0, 0])
            * (rectangles[:, 2, 1] - rectangles[:, 1, 1])).sum()
    assert area == len(set(map(tuple, points)))
    candidates = np.array([(x, y) for x in range(18) for y in range(23)])
    coverage = sum(Path(rect).contains_points(candidates).astype(int) for rect in rectangles)
    occupied = set(map(tuple, points))
    np.testing.assert_equal(coverage, [tuple(p) in occupied for p in candidates])


@pytest.mark.parametrize('mode', ['bands', 'outlines'])
def test_huge_sparse_extent_renders_without_dense_grid(mode, monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    # A trillion-cell bounding box, but only two small disconnected components.
    points = [(x, y) for x in (0, 1, 1_000_000, 1_000_001) for y in (0, 1)]
    points += [(0, 1_000_000)]
    ax = plot_parameter_mapping({'hkl': points}, 'energy', 'roty',
                                display_mode=mode, grid_steps=(1, 1))
    ax.figure.canvas.draw()
    assert len(ax.collections) == 1
    assert len(ax.collections[0].get_paths()) <= 12
    plt.close(ax.figure)


def test_fine_scan_compresses_without_changing_resolution():
    # Over 100 million possible cells; narrow branches with a one-cell gap.
    x = np.arange(10_001)
    points = np.concatenate([np.column_stack((x, x + offset))
                             for offset in (0, 1, 2, 4, 5, 6)])
    for filled in (False, True):
        geometry = band_vertices(points * [.01, .001] + [7000, -25],
                                 (.01, .001), filled=filled)
        assert np.isfinite(geometry).all()
        assert geometry.size < points.size * 4
    # Identical adjacent runs collapse to a rectangle, not one polygon per shot.
    dense = np.array([(x, y) for x in range(80) for y in range(100)])
    assert band_vertices(dense, (1, 1)).shape == (1, 4, 2)
    assert band_vertices(dense, (1, 1), filled=False).shape == (4, 2, 2)


@pytest.mark.parametrize('count', [1, 40, 80])
def test_legend_reflows_and_preserves_plot_area(count, monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    points = {f'[{i} -1 2]': [(0, i), (1, i + 1)] for i in range(count)}
    ax = plot_parameter_mapping(points, 'energy', 'roty')
    fig = ax.figure
    fig.canvas.draw()
    assert ax.get_window_extent().height / fig.dpi >= 5.05
    if count > 1:
        assert ax.get_legend()._ncols > 1

    def check_bounds():
        renderer = fig.canvas.get_renderer()
        legend = ax.get_legend().get_window_extent(renderer)
        axes = ax.get_window_extent(renderer)
        assert legend.x0 > axes.x1 and legend.x1 <= fig.bbox.width + 1e-7
        assert legend.y0 >= 0 and legend.y1 <= fig.bbox.height
        assert axes.width == pytest.approx(axes.height)
        assert len(ax.get_legend().get_texts()) == count

    check_bounds()
    handle = ax.get_legend().get_lines()[0]
    event = PickEvent('pick_event', fig.canvas,
                      MouseEvent('button_press_event', fig.canvas, 0, 0), handle)
    fig.canvas.callbacks.process('pick_event', event)
    assert not ax.collections[0].get_visible()
    fig.set_size_inches(7, 7)
    fig.canvas.callbacks.process('resize_event', ResizeEvent('resize_event', fig.canvas))
    fig.canvas.draw()
    check_bounds()
    assert ax.get_legend().get_lines()[0].get_alpha() == .2
    new_handle = ax.get_legend().get_lines()[0]
    fig.canvas.callbacks.process('pick_event', PickEvent('pick_event', fig.canvas,
        MouseEvent('button_press_event', fig.canvas, 0, 0), new_handle))
    assert ax.collections[0].get_visible()
    fig.set_size_inches(10, 5)
    fig.canvas.callbacks.process('resize_event', ResizeEvent('resize_event', fig.canvas))
    fig.canvas.draw()
    check_bounds()
    plt.close(fig)


def test_failed_band_render_does_not_leave_empty_figure(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    before = plt.get_fignums()
    with pytest.raises(ValueError, match='unavailable'):
        plot_parameter_mapping({'hkl': [(0, 0)]}, 'energy', 'roty', display_mode='bands')
    assert plt.get_fignums() == before
