"""Loopback-only, bounded-memory Plotly viewer for compact reciprocal scans."""
import atexit
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
from threading import Thread, Lock
from urllib.parse import urlsplit, parse_qs

from . import reciprocal
from .deferred_plotting import gui_plot
from .reciprocal_compact import configuration, ranges
from .reciprocal_cache import NeighborhoodCache

_VIEWERS = []
_LOCK = Lock()


SLIDER_SCRIPT = """const gd = document.getElementById('{plot_id}');
const axes = AXES, selected = axes.map(() => 0);
const panel = document.createElement('div');
panel.style.cssText='padding:8px 24px;font:14px sans-serif;background:white;display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,640px),1fr));column-gap:24px';
gd.before(panel);
const status = document.createElement('div'); panel.append(status);
status.style.gridColumn='1 / -1';
let busy=false, pending=null, timer=null;
const frameDelay=0;
async function draw() {
  if(busy) return; busy=true;
  try { while(pending!==null) {
    const index=pending; pending=null; status.textContent='Updating...';
    const response=await fetch('FRAME_URL?index='+index, {cache:'no-store'});
    if(!response.ok) throw new Error(await response.text());
    const next=await response.json();
    if(pending!==null) continue;
    next.data.forEach((trace,i)=>{if(gd.data[i]) trace.visible=gd.data[i].visible;});
    await Plotly.react(gd,next.data,{...gd.layout,title:next.title});
    status.textContent='';
  }} catch(error) {status.textContent='Could not update: '+error.message;}
  finally {busy=false;}
}
axes.forEach((axis,i)=>{
  const row=document.createElement('div');
  row.style.cssText='display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:6px 0';
  const label=document.createElement('label');label.textContent=axis.name;label.style.minWidth='120px';
  const input=document.createElement('input'); input.type='range';input.min=0;input.max=axis.size-1;input.step=1;input.value=0;
  input.style.flex='1 0 240px';input.disabled=axis.size===1;
  input.setAttribute('aria-label',axis.name);
  const previous=document.createElement('button');previous.type='button';previous.textContent='◀';
  const following=document.createElement('button');following.type='button';following.textContent='▶';
  for(const button of [previous,following]) button.style.cssText='flex:0 0 36px;height:32px;cursor:pointer';
  previous.setAttribute('aria-label','Previous '+axis.name+' value');
  following.setAttribute('aria-label','Next '+axis.name+' value');
  const value=document.createElement('input'); value.type='number'; value.style.width='120px';
  value.style.flex='0 0 120px';value.value=axis.start;
  const end=axis.start+(axis.size-1)*axis.step;
  value.min=Math.min(axis.start,end);value.max=Math.max(axis.start,end);value.step=Math.abs(axis.step)||1;
  value.setAttribute('aria-label',axis.name+' value');
  const syncButtons=()=>{previous.disabled=Number(input.value)<=0;following.disabled=Number(input.value)>=axis.size-1;};
  const step=direction=>{const target=Math.max(0,Math.min(axis.size-1,Number(input.value)+direction));
    if(target!==Number(input.value)){input.value=target;input.dispatchEvent(new Event('input'));}};
  previous.addEventListener('click',()=>step(-1));
  following.addEventListener('click',()=>step(1));
  input.addEventListener('input',()=>{
    selected[i]=Number(input.value);
    value.value=Number((axis.start+selected[i]*axis.step).toPrecision(12));
    syncButtons();
    pending=selected.reduce((index,v,j)=>index*BigInt(axes[j].size)+BigInt(v),0n).toString();
    if(frameDelay) {clearTimeout(timer);timer=setTimeout(draw,frameDelay);} else draw();
  });
  value.addEventListener('change',()=>{
    const requested=Number(value.value);
    if(value.value==='' || !Number.isFinite(requested) || requested<Number(value.min) || requested>Number(value.max)) {
      status.textContent='Enter a value within the saved scan range.';return;
    }
    input.value=axis.step===0 ? 0 : Math.max(0,Math.min(axis.size-1,Math.round((requested-axis.start)/axis.step)));
    input.dispatchEvent(new Event('input'));
  });
  syncButtons();row.append(label,input,previous,following,value);panel.append(row);
});"""


@gui_plot
def open_viewer(url, label='reciprocal scan'):
    print('Interactive ' + label + ' (local viewer): ' + url)
    reciprocal.webbrowser.open(url, new=2)


class LiveViewer:
    def __init__(self, scan, detector_obj, options, parallel, workers, figure,
                 *, frame_factory=None, frame_delay=0):
        self.frame_factory = frame_factory
        self.scan = scan
        self.detector = deepcopy(detector_obj)
        self.options = dict(options)
        self.parallel, self.workers = parallel, workers
        self.token = secrets.token_urlsafe(24)
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                parsed = urlsplit(self.path)
                if self.headers.get('Host') != viewer.authority or self.headers.get('Origin', viewer.origin) != viewer.origin:
                    self.send_error(403)
                    return
                if parsed.path == '/' + viewer.token + '/':
                    self.respond(viewer.html, 'text/html; charset=utf-8')
                elif parsed.path == '/' + viewer.token + '/frame':
                    try:
                        value = parse_qs(parsed.query).get('index', [''])[0]
                        if not value.isascii() or not value.isdigit() or len(value) > 64:
                            raise ValueError('Invalid configuration index.')
                        self.respond(viewer.frame(int(value)), 'application/json')
                    except (ValueError, IndexError) as exc:
                        self.send_error(400, str(exc))
                    except Exception as exc:
                        print('Reciprocal viewer error: ' + str(exc))
                        self.send_error(500, 'Could not render this configuration; see the GUI log.')
                else:
                    self.send_error(404)

            def respond(self, data, content_type):
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Referrer-Policy', 'no-referrer')
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.authority = '127.0.0.1:' + str(self.server.server_port)
        self.origin = 'http://' + self.authority
        self.url = self.origin + '/' + self.token + '/'
        from .reciprocal_scan import parameter_label
        axes = [dict(name=parameter_label(name), size=len(values), start=float(values[0]),
                     step=float(values[1]-values[0]) if len(values)>1 else 0)
                for name, values in scan['axes'].items()]
        script = SLIDER_SCRIPT.replace('AXES', json.dumps(axes).replace('<', '\\u003c')).replace('FRAME_URL', self.url+'frame')
        script = script.replace('const frameDelay=0;', f'const frameDelay={int(frame_delay)};')
        self.html = figure.to_html(include_plotlyjs=True, post_script=script,
            config=dict(responsive=True, displaylogo=False, scrollZoom=True)).encode('utf-8')
        self.thread = Thread(target=self.server.serve_forever, kwargs={'poll_interval': .1}, daemon=True)
        self.cache = NeighborhoodCache(self._prepare_frame, [len(v) for v in scan['axes'].values()],
                                       parallel=parallel, workers=workers)
        try:
            self.cache.get(0)
        except Exception:
            self.cache.close()
            self.server.server_close()
            raise
        self.thread.start()
        self.closed = False

    def frame(self, index):
        return self.cache.get(index)

    def _prepare_frame(self, index):
        if self.frame_factory is not None:
            return self.frame_factory(index)
        from plotly.utils import PlotlyJSONEncoder
        from .reciprocal_scan import configuration
        values = configuration(self.scan, index, self.detector)
        figure = reciprocal.plot_reciprocal(**values, **self.options, show=False)
        payload = json.dumps(dict(data=figure.to_plotly_json()['data'], title=figure.layout.title.to_plotly_json()),
                             cls=PlotlyJSONEncoder).encode('utf-8')
        return payload

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.cache.close()
        self.frame_factory = None
        self.scan = None


def close_viewers():
    with _LOCK:
        viewers, _VIEWERS[:] = list(_VIEWERS), []
    for viewer in viewers:
        viewer.close()


atexit.register(close_viewers)


def plot_live(scan, detector_obj, options, parallel, workers, show, html_path):
    from .reciprocal_scan import configuration
    print('Preparing interactive reciprocal viewer...', flush=True)
    values = configuration(scan, 0, detector_obj, parallel, workers)
    figure = reciprocal.plot_reciprocal(**values, **options, show=False)
    if scan.get('format') == 'compact-v1':
        limits_all = ranges(scan)
    else:
        import numpy as np
        limits_all = reciprocal._ranges(scan['Q'].reshape(-1,3),
            2*np.pi/(min(scan['wavelengths'])*1e10)*(1+scan['bandwidth']/200))
    for axis, limits in zip('xyz', limits_all):
        figure.layout.scene[axis+'axis'].range = limits
    figure.layout.margin.t = 150
    if show or html_path is not None:
        viewer = LiveViewer(scan, detector_obj, options, parallel, workers, figure)
        figure._reciprocal_viewer = viewer
        with _LOCK:
            _VIEWERS.append(viewer)
            expired = _VIEWERS.pop(0) if len(_VIEWERS)>8 else None
        if expired is not None:
            expired.close()
            print('Closed the oldest local reciprocal viewer (maximum eight open viewers).')
        if html_path is not None:
            Path(html_path).expanduser().write_text(
                '<!doctype html><meta charset="utf-8"><title>Reciprocal scan</title>'
                '<p>This interactive scan needs the local XRDpy viewer running. '
                'To reopen later, load the saved HDF5 result and plot again.</p>'
                '<a href="' + viewer.url + '">Open local reciprocal scan</a>', encoding='utf-8')
        if show:
            open_viewer(viewer.url)
    return figure
