"""Grouped reciprocal-space rendering with an offline, WebGL-backed default."""
from pathlib import Path
import tempfile
import webbrowser

import numpy as np

from .deferred_plotting import gui_plot
from . import sample


def _data(q, hkls, wavelength, bandwidth):
    q = np.asarray(q, dtype=float)
    hkls = np.asarray(hkls, dtype=float)
    if q.size == 0:
        q = q.reshape(0, 3)
    if hkls.size == 0:
        hkls = hkls.reshape(0, 3)
    if q.ndim != 2 or q.shape[1] != 3 or q.shape != hkls.shape or not np.isfinite(q).all() or not np.isfinite(hkls).all():
        raise ValueError('Q vectors and Miller indices must be matching finite N-by-3 arrays.')
    wavelength, bandwidth = float(wavelength), float(bandwidth)
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError('Wavelength must be positive and finite, in metres.')
    if not np.isfinite(bandwidth) or not 0 <= bandwidth < 200:
        raise ValueError('Full energy bandwidth must be between 0 and 200 percent (exclusive).')
    k = 2 * np.pi / (wavelength * 1e10)
    accepted = sample.check_Bragg_condition(q, wavelength, bandwidth)
    return q, hkls, accepted, k, bandwidth


def _sphere(k):
    azimuth, polar = np.meshgrid(np.linspace(0, 2 * np.pi, 49), np.linspace(0, np.pi, 25))
    # In Q space each energy's Ewald sphere is tangent to Q=0, not concentric.
    return k * np.cos(polar) - k, k * np.sin(polar) * np.cos(azimuth), k * np.sin(polar) * np.sin(azimuth)


def _wire(k):
    x, y, z = _sphere(k)
    segments = [np.column_stack((x[i], y[i], z[i])) for i in range(0, len(x), 4)]
    segments += [np.column_stack((x[:, i], y[:, i], z[:, i])) for i in range(0, x.shape[1], 6)]
    return np.concatenate([np.vstack((segment, [np.nan] * 3)) for segment in segments])


def _ranges(q, radius):
    bounds = np.vstack((q, [-2 * radius, -radius, -radius], [0, radius, radius]))
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    middle = (low + high) / 2
    half = max(high - low) * .54
    return [[float(c - half), float(c + half)] for c in middle]


def _bragg_labels(hkls, accepted):
    return list(dict.fromkeys('(' + ', '.join(f'{v:.0f}' for v in hkl) + ')'
                             for hkl in hkls[accepted]))


def _q_angles(q):
    """Lab azimuth about the beam and elevation above the Qx-Qy plane, in degrees.

    Azimuth is zero along +Qz (up), negative toward +Qy (left), and
    positive toward -Qy (right) for the unrotated detector view.
    """
    horizontal = np.hypot(q[:, 0], q[:, 1])
    transverse = np.hypot(q[:, 1], q[:, 2])
    azimuth = np.degrees(np.arctan2(-q[:, 1], q[:, 2]))
    azimuth[azimuth == 0] = 0
    elevation = np.degrees(np.arctan2(q[:, 2], horizontal))
    azimuth[transverse == 0] = np.nan
    elevation[np.linalg.norm(q, axis=1) == 0] = np.nan
    return azimuth, elevation


def _point_details(q, hkl):
    azimuth, elevation = _q_angles(np.asarray(q, dtype=float).reshape(1, 3))
    def angle(value):
        return f'{value:.2f}°' if np.isfinite(value) else 'undefined'
    return (f'hkl=({hkl[0]:.0f}, {hkl[1]:.0f}, {hkl[2]:.0f})\n'
            f'Q=({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}) Å⁻¹\n'
            f'|Q|={np.linalg.norm(q):.4f} Å⁻¹\n'
            f'Azimuth={angle(azimuth[0])}\nElevation={angle(elevation[0])}')


@gui_plot
def open_html(path):
    """Open only after the GUI worker has finished writing the offline plot."""
    path = Path(path).resolve()
    print('Interactive reciprocal-space plot: ' + str(path))
    if not webbrowser.open(path.as_uri(), new=2):
        print('No browser opened automatically. Open the HTML path above in a browser.')


def plot_reciprocal(q, hkls, wavelength, bandwidth, *, backend='plotly',
                    show_non_bragg=True, show_bandwidth=True, title=None,
                    show=True, html_path=None, require_detector_hit=False, detector_obj=None):
    """Plot Q in inverse angstrom; spheres are centred at (-2*pi/lambda,0,0).

    Returns a Plotly Figure, or (Figure, Axes) for Matplotlib. HTML includes
    Plotly.js so it works offline; no browser server or remote service is used.
    """
    q, hkls, accepted, k, bandwidth = _data(q, hkls, wavelength, bandwidth)
    title = title or 'Reciprocal space and Ewald construction'
    hits = np.zeros(len(q), dtype=bool)
    if require_detector_hit:
        if detector_obj is None:
            raise ValueError('A detector is required for detector-hit highlighting.')
        from .experiment import detector_hit_mask
        hits[accepted] = detector_hit_mask(q[accepted], wavelength, detector_obj)
    if backend == 'matplotlib':
        if html_path is not None:
            raise ValueError('HTML export requires the Plotly backend.')
        return _matplotlib(q, hkls, accepted, k, bandwidth, show_non_bragg,
                           show_bandwidth, title, show, hits, require_detector_hit)
    if backend != 'plotly':
        raise ValueError('Reciprocal renderer must be plotly or matplotlib.')
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError('Install plotly (pip install plotly) or select the Matplotlib renderer.') from exc

    reflection_hover = ('hkl=(%{customdata[0]:.0f}, %{customdata[1]:.0f}, %{customdata[2]:.0f})'
                        '<br>Q=(%{x:.4f}, %{y:.4f}, %{z:.4f}) \u00c5\u207b\u00b9'
                        '<br>|Q|=%{customdata[3]:.4f} \u00c5\u207b\u00b9'
                        '<br>Azimuth=%{customdata[4]:.2f}\u00b0'
                        '<br>Elevation=%{customdata[5]:.2f}\u00b0')
    fig = go.Figure()
    for selected, color, name, visible in (
        (~accepted, '#56758c', 'Outside Bragg condition', True if show_non_bragg else 'legendonly'),
        (accepted & ~hits, '#d44929', 'In Bragg condition', True),
    ):
        points = q[selected]
        azimuth, elevation = _q_angles(points)
        custom = np.column_stack((hkls[selected], np.linalg.norm(points, axis=1), azimuth, elevation))
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers', name=f'{name} ({len(points)})', visible=visible,
            marker=dict(size=4 if name.startswith('In ') else 2.5, color=color,
                        opacity=1 if name.startswith('In ') else .45),
            customdata=custom, hovertemplate=reflection_hover + '<extra></extra>'))
    x, y, z = _sphere(k)
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=.13, showscale=False,
                            colorscale=[[0, '#23956f'], [1, '#23956f']],
                            name='Central-energy Ewald sphere', showlegend=True, hoverinfo='skip'))
    # Stable trace positions preserve legend toggles when a scan crosses zero bandwidth.
    for scale, name in ((1 - bandwidth / 200, 'Low-energy boundary'),
                        (1 + bandwidth / 200, 'High-energy boundary')):
        wire = _wire(k * scale) if bandwidth > 0 else np.empty((0, 3))
        fig.add_trace(go.Scatter3d(x=wire[:, 0], y=wire[:, 1], z=wire[:, 2],
            mode='lines', name=name, visible=True if show_bandwidth else 'legendonly',
            showlegend=bandwidth > 0,
            line=dict(color='#489682', width=1), opacity=.35, hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[-k, 0], y=[0, 0], z=[0, 0], mode='lines+markers',
        name='Incident wavevector k_i', line=dict(color='#222222', width=4), marker=dict(size=4),
        text=['Ewald centre', 'Reciprocal origin'], hovertemplate='%{text}<extra></extra>'))
    labels = _bragg_labels(hkls, accepted)
    hit_labels = set(_bragg_labels(hkls, hits))
    # One scrollable table keeps long reflection lists out of the 3D scene.
    fig.add_trace(go.Table(domain=dict(x=[.76, 1], y=[0, .64]),
        header=dict(values=[f'Bragg hkl ({len(labels)})'], align='left',
                    fill_color='#eef3f6', font=dict(color='#222222', size=13)),
        cells=dict(values=[[label + (' [detector hit]' if label in hit_labels else '')
                            for label in labels] or ['None']], align='left', height=24,
                   fill_color='white', font=dict(color=[['#159447' if label in hit_labels else '#d44929'
                                                        for label in labels] or ['#222222']], size=12))))
    if require_detector_hit:
        points = q[hits]
        azimuth, elevation = _q_angles(points)
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers', name=f'Bragg + detector hit ({hits.sum()})',
            marker=dict(size=5, color='#159447'),
            customdata=np.column_stack((hkls[hits], np.linalg.norm(points, axis=1), azimuth, elevation)),
            hovertemplate=reflection_hover + '<extra>Detector hit</extra>'))
    ranges = _ranges(q, k * (1 + bandwidth / 200))
    fig.update_layout(title=dict(text=title.replace('\n', '<br>'), x=.02),
        template='plotly_white', margin=dict(l=0, r=10, b=0, t=100),
        scene=dict(**{axis+'axis': dict(title='Q<sub>'+axis+'</sub> [\u00c5\u207b\u00b9]', range=limits)
                       for axis, limits in zip('xyz', ranges)},
                   domain=dict(x=[0, .72], y=[0, 1]),
                   aspectmode='cube', camera=dict(eye=dict(x=1.4, y=1.5, z=1.1))),
        legend=dict(x=.76, xanchor='left', y=1, font=dict(size=11)), uirevision='ewald')
    if show or html_path is not None:
        path = Path(html_path).expanduser() if html_path is not None else Path(tempfile.mkdtemp(prefix='trxrdpy-ewald-')) / 'reciprocal.html'
        fig.write_html(str(path), include_plotlyjs=True, full_html=True,
                       config=dict(responsive=True, displaylogo=False, scrollZoom=True))
        if show:
            open_html(path)
    return fig


@gui_plot
def _matplotlib(q, hkls, accepted, k, bandwidth, show_non_bragg, show_bandwidth, title, show, hits, highlight,
                existing=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    labels = _bragg_labels(hkls, accepted)
    columns = max(1, int(np.ceil((len(labels) + 6) / 22)))
    if existing is None:
        fig = plt.figure(figsize=(9 + 1.6 * columns, 7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = existing
        if hasattr(ax, '_reciprocal_pick'):
            fig.canvas.mpl_disconnect(ax._reciprocal_pick)
        if hasattr(ax, '_reciprocal_motion'):
            fig.canvas.mpl_disconnect(ax._reciprocal_motion)
        ax.clear()
    point_groups = []
    visible_points = {}
    for selected, color, label in ((~accepted, '#56758c', 'Outside Bragg condition'),
                                   (accepted & ~hits, '#d44929', 'In Bragg condition')):
        if label.startswith('Outside') and not show_non_bragg:
            continue
        artist = ax.scatter(*q[selected].T, s=8 if label.startswith('Outside') else 25,
                   color=color, label=f'{label} ({selected.sum()})', depthshade=False)
        point_groups.append((artist, selected))
        visible_points[artist] = np.flatnonzero(selected)
    surface = ax.plot_surface(*_sphere(k), color='#23956f', alpha=.13, linewidth=0)
    if highlight:
        artist = ax.scatter(*q[hits].T, s=30, color='#159447', label=f'Bragg + detector hit ({hits.sum()})')
        point_groups.append((artist, hits))
        visible_points[artist] = np.flatnonzero(hits)
    boundaries = []
    if show_bandwidth and bandwidth > 0:
        for scale in (1 - bandwidth / 200, 1 + bandwidth / 200):
            wire = _wire(k * scale)
            boundaries.extend(ax.plot(*wire.T, color='#489682', linewidth=.5, alpha=.4))
    ax.plot([-k, 0], [0, 0], [0, 0], 'k.-', label='Incident wavevector')
    ranges = _ranges(q, k * (1 + bandwidth / 200))
    ax.set(xlabel=r'$Q_x$ [$\AA^{-1}$]', ylabel=r'$Q_y$ [$\AA^{-1}$]',
           zlabel=r'$Q_z$ [$\AA^{-1}$]', xlim=ranges[0], ylim=ranges[1], zlim=ranges[2], title=title)
    ax.set_box_aspect((1, 1, 1))
    handles, names = ax.get_legend_handles_labels()
    targets = [[artist] for artist in handles]
    handles.append(Line2D([], [], color='#23956f'))
    names.append('Ewald sphere')
    targets.append([surface])
    if boundaries:
        handles.append(Line2D([], [], color='#489682'))
        names.append('Bandwidth boundaries')
        targets.append(boundaries)
    hit_labels = set(_bragg_labels(hkls, hits))
    handles.extend(Line2D([], [], marker='o', linestyle='', color='#159447' if label in hit_labels else '#d44929') for label in labels)
    names.extend(labels)
    if not labels:
        handles.append(Line2D([], [], linestyle=''))
        names.append('No Bragg hkl')
    legend = ax.legend(handles, names, title='Bragg reflections', fontsize=9,
              loc='upper left', bbox_to_anchor=(1.02, 1), ncol=columns)
    legend_handles = getattr(legend, 'legend_handles', None)
    if legend_handles is None:
        legend_handles = legend.legendHandles
    actions = {}
    for index, (handle, text) in enumerate(zip(legend_handles, legend.get_texts())):
        handle.set_picker(5)
        text.set_picker(True)
        action = targets[index] if index < len(targets) else names[index]
        actions[handle] = actions[text] = (action, handle, text)
    hidden = set()
    point_labels = np.array(['(' + ', '.join(f'{v:.0f}' for v in hkl) + ')' for hkl in hkls])

    def toggle(event):
        entry = actions.get(event.artist)
        if entry is None:
            return
        action, handle, text = entry
        if isinstance(action, list):
            visible = not action[0].get_visible()
            for artist in action:
                artist.set_visible(visible)
        else:
            if action in hidden:
                hidden.remove(action)
            else:
                hidden.add(action)
            visible = action not in hidden
            mask = ~np.isin(point_labels, list(hidden))
            for artist, selected in point_groups:
                visible_points[artist] = np.flatnonzero(selected & mask)
                artist._offsets3d = tuple(q[visible_points[artist]].T)
        handle.set_alpha(1 if visible else .25)
        text.set_alpha(1 if visible else .25)
        fig.canvas.draw_idle()

    ax._reciprocal_pick = fig.canvas.mpl_connect('pick_event', toggle)
    tooltip = ax.annotate('', xy=(0, 0), xycoords='figure pixels',
                          xytext=(12, 12), textcoords='offset points',
                          ha='left', va='bottom',
                          bbox=dict(boxstyle='round', facecolor='white', edgecolor='#555555', alpha=.96),
                          zorder=10)
    tooltip.set_visible(False)

    def hover(event):
        found = None
        if event.inaxes is ax:
            for artist, _ in reversed(point_groups):
                if not artist.get_visible() or not len(visible_points[artist]):
                    continue
                contains, details = artist.contains(event)
                if contains and len(details.get('ind', ())):
                    found = visible_points[artist][details['ind'][0]]
                    break
        if found is None:
            if tooltip.get_visible():
                tooltip.set_visible(False)
                fig.canvas.draw_idle()
            return
        tooltip.xy = (event.x, event.y)
        tooltip.set_text(_point_details(q[found], hkls[found]))
        tooltip.set_visible(True)
        fig.canvas.draw_idle()

    ax._reciprocal_motion = fig.canvas.mpl_connect('motion_notify_event', hover)
    ax._reciprocal_annotation = tooltip
    if existing is None:
        fig.subplots_adjust(left=.03, right=8 / (9 + 1.6 * columns), top=.85, bottom=.08)
    if show:
        plt.show()
    return fig, ax
