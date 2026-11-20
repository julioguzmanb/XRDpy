from concurrent.futures import wait
import json
import numpy as np
import pytest

from trxrdpy.simulation.results import SimulationResult, plot_result, save_result, load_result
from trxrdpy.simulation import reciprocal
from trxrdpy.simulation.reciprocal_live import close_viewers


@pytest.fixture(autouse=True)
def cleanup():
    yield
    close_viewers()


def dataset():
    points = {'[1 0 0]':np.array([[0,0],[0,1],[1,0],[1,1],[5,5]]), '[0 1 0]':[]}
    context = dict(initial_orientation=np.eye(3).tolist(),rotation_order='xyz',
                   energy=10000,motors=None,fixed_angles={})
    return SimulationResult('parameter_mapping', dict(format='mapping-slices-v1',
        extra_axes={'rotz':np.arange(3)},plot_axes=('roty','energy'),
        plot_ranges=np.array([[0,5,1],[0,5,1]]), grid_steps=np.array([1,1]),
        slices={str(i):dict(points=points,context_json=json.dumps(context)) for i in range(3)}),{})


@pytest.mark.parametrize('mode',['points','bands','outlines'])
def test_modes_and_roundtrip(mode,tmp_path):
    result=load_result(save_result(tmp_path/'mapping.h5',dataset()))
    fig=plot_result(result,mapping_backend='plotly',mapping_show=False,display_mode=mode)
    assert len(fig.data)==2
    assert fig.data[0].type == ('scatter' if mode=='bands' else 'scattergl')
    assert fig.data[1].name=='[0 1 0]'
    if mode!='points':
        assert np.isnan(np.asarray(fig.data[0].x,dtype=float)).any()
        assert not fig.data[0].connectgaps
    assert 'roty' in fig.layout.xaxis.title.text


def test_live_cache_and_detection(tmp_path,monkeypatch):
    from urllib.request import urlopen
    from trxrdpy.simulation import mapping_filter, mapping_plotly
    called=[]
    def filter_slice(result,detector):
        called.append(1)
        return SimulationResult(result.kind, {'points':{'[1 0 0]':[[0,0]],'[0 1 0]':[]}},result.metadata)
    monkeypatch.setattr(mapping_filter,'filtered_mapping',filter_slice)
    monkeypatch.setattr(reciprocal.webbrowser,'open',lambda *a,**kw:True)
    fig=plot_result(dataset(),mapping_backend='plotly',require_detector_hit=True,detector_obj=object(),display_mode='outlines')
    viewer=fig._mapping_viewer
    wait(viewer.cache.pending,timeout=10)
    count=len(called)
    assert count==3  # initial frame is reused, not filtered a second time
    with urlopen(viewer.url+'frame?index=1') as response:
        payload=json.load(response)
    assert len(called)==count
    assert payload['data'][0]['name']=='[1 0 0]'
    assert 'rotz = 1' in payload['title']['text']
    assert b'frameDelay=150' in viewer.html
    assert viewer.cache.bytes <= viewer.cache.max_bytes
    viewer.close()
    assert viewer.cache.closed


def test_gui_renderer_persists(monkeypatch):
    from PyQt5.QtWidgets import QApplication
    from trxrdpy.simulation.gui.state import GuiState
    from trxrdpy.simulation.gui.services import SimulationService
    from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab
    app=QApplication.instance() or QApplication([])
    tab=SingleCrystalTab(SimulationService(),GuiState.from_dict({'state_version':13}))
    assert tab.mapping_backend.currentData()=='matplotlib'
    tab.mapping_backend.setCurrentIndex(1)
    tab.save_to_state()
    assert GuiState.from_dict(tab.state.to_dict()).single.mapping_backend=='plotly'
    tab.close()
    app.processEvents()


def test_gui_browser_preparation_is_background(monkeypatch,wait_simulation):
    import threading
    from PyQt5.QtWidgets import QApplication
    from trxrdpy.simulation.gui.state import GuiState
    from trxrdpy.simulation.gui.services import SimulationService
    from trxrdpy.simulation.gui.tabs.single_crystal_tab import SingleCrystalTab
    from trxrdpy.simulation import mapping_plotly
    app=QApplication.instance() or QApplication([])
    main=threading.get_ident()
    prepared,opened=[],[]
    original=mapping_plotly.mapping_figure
    def capture(*args,**kwargs):
        prepared.append(threading.get_ident())
        return original(*args,**kwargs)
    monkeypatch.setattr(mapping_plotly,'mapping_figure',capture)
    monkeypatch.setattr(reciprocal.webbrowser,'open',lambda *args,**kwargs:opened.append(threading.get_ident()) or True)
    tab=SingleCrystalTab(SimulationService(),GuiState())
    tab.mapping_backend.setCurrentIndex(1)
    tab.last_scan_result=dataset()
    tab.require_detector_hit.setChecked(False)
    tab._plot_scan_result()
    wait_simulation(tab)
    assert prepared and all(thread!=main for thread in prepared)
    assert opened==[main]
    tab.close()
    app.processEvents()
