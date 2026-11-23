"""WebGL mapping points and cached, on-demand mapping-slice views."""
from copy import deepcopy
import json
import numpy as np

from .parallel import check_cancelled
from .reciprocal_live import LiveViewer, _LOCK, _VIEWERS, open_viewer


def mapping_figure(view, keys, display_mode, title):
    import plotly.graph_objects as go
    from plotly.colors import sample_colorscale
    from .mapping_display import band_vertices
    names = view.metadata.get('axes', ['roty', 'rotz'])
    steps = view.metadata.get('grid_steps')
    labels = [name + (' [eV]' if name == 'energy' else ' [°]') for name in names]
    colors = sample_colorscale('Jet', np.linspace(0, 1, max(2,len(keys))))
    figure = go.Figure()
    for key, color in zip(keys, colors):
        check_cancelled()
        points = np.asarray(view.data['points'].get(key, []), dtype=float).reshape(-1, 2)
        common = dict(name=key, legendgroup=key, connectgaps=False,
                      hovertemplate=key+'<br>'+labels[0]+': %{x}<br>'+labels[1]+': %{y}<extra></extra>')
        if display_mode == 'points':
            figure.add_trace(go.Scattergl(x=points[:,0], y=points[:,1], mode='markers',
                marker=dict(color=color,size=4), **common))
        else:
            vertices = band_vertices(points, steps, filled=display_mode=='bands') if len(points) else []
            # NaN breaks preserve disconnected branches and separate filled rectangles.
            segments = [np.vstack((v, v[:1], [np.nan,np.nan])) if display_mode=='bands'
                        else np.vstack((v,[np.nan,np.nan])) for v in vertices]
            coords = np.concatenate(segments) if segments else np.empty((0,2))
            trace = go.Scatter if display_mode=='bands' else go.Scattergl
            figure.add_trace(trace(x=coords[:,0],y=coords[:,1],mode='lines',
                line=dict(color=color,width=0 if display_mode=='bands' else 1.2),
                fill='toself' if display_mode=='bands' else None,
                opacity=.45 if display_mode=='bands' else 1, **common))
    figure.update_layout(title=dict(text=title,x=.02), template='plotly_white',
        xaxis_title=labels[0], yaxis_title=labels[1], uirevision='mapping',
        legend=dict(x=1.02,y=1), margin=dict(l=75,r=180,t=90,b=65), height=750)
    return figure


def plot_mapping(result, display_mode='points', require_detector_hit=False, detector_obj=None,
                 *, show=True, parallel=False, workers=None):
    from .mapping_slices import slice_result
    from .mapping_filter import filtered_mapping
    from .reciprocal_scan import parameter_label
    if display_mode not in ('points','outlines','bands'):
        raise ValueError('Unknown mapping display mode.')
    detector_obj = deepcopy(detector_obj)
    if result.metadata.get('require_detector_hit') and not result.metadata.get('points_are_complete',False):
        import warnings
        warnings.warn('This older mapping was filtered during computation; removed Bragg points cannot be recovered.',
                      UserWarning,stacklevel=2)
    sliced = result.data.get('format') == 'mapping-slices-v1'
    axes = result.data['extra_axes'] if sliced else {}
    first = slice_result(result,0) if sliced else result
    keys = list(first.data['points'])

    def prepare(index):
        view = slice_result(result,index) if sliced else result
        if view.metadata.get('grid_steps') is None:
            import ast
            saved = view.metadata.get('gui_state',{}).get('single',{})
            prefix = 'geometry_motor' if view.metadata.get('axes') == [saved.get('geometry_motor1_name'),saved.get('geometry_motor2_name')] else 'param'
            try:
                steps = [abs(float(ast.literal_eval(saved[prefix+str(i)+'_range'])[2])) for i in (1,2)]
                from .results import SimulationResult
                view = SimulationResult(view.kind,view.data,{**view.metadata,'grid_steps':steps})
            except (KeyError,ValueError,TypeError,SyntaxError):
                pass
        if require_detector_hit:
            view = filtered_mapping(view, detector_obj)
        title = ' vs '.join(view.metadata.get('axes', [])) + ' for Bragg condition'
        if sliced:
            offsets = np.unravel_index(index, tuple(len(v) for v in axes.values()))
            title += '<br>' + '; '.join(f'{parameter_label(name)} = {values[i]:g}'
                for (name,values),i in zip(axes.items(),offsets))
        return mapping_figure(view, keys, display_mode, title)

    figure = prepare(0)
    if sliced:
        for axis, interval in zip(('xaxis','yaxis'),result.data['plot_ranges']):
            low,high=sorted(interval[:2])
            figure.layout[axis].range=[low-abs(interval[2])/2, high+abs(interval[2])/2]
        if show:
            from plotly.utils import PlotlyJSONEncoder
            def payload(fig):
                return json.dumps(dict(data=fig.to_plotly_json()['data'],title=fig.layout.title.to_plotly_json()),
                                  cls=PlotlyJSONEncoder).encode('utf-8')
            initial = payload(figure)
            def frame(index):
                return initial if index == 0 else payload(prepare(index))
            viewer = LiveViewer({'axes':axes},None,{},parallel,workers,figure,
                                frame_factory=frame,frame_delay=150)
            figure._mapping_viewer = viewer
            with _LOCK:
                _VIEWERS.append(viewer)
                expired = _VIEWERS.pop(0) if len(_VIEWERS)>8 else None
            if expired is not None:
                expired.close()
            open_viewer(viewer.url, label='mapping')
    elif show:
        import tempfile
        from pathlib import Path
        from .reciprocal import open_html
        with tempfile.NamedTemporaryFile(suffix='.html',prefix='xrdpy-mapping-',delete=False) as stream:
            path=Path(stream.name)
        figure.write_html(path, include_plotlyjs=True)
        open_html(path)
    return figure
