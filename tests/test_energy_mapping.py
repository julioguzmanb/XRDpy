import multiprocessing
import time
import numpy as np
import pytest

from trxrdpy.simulation import single_crystal as sc, sample
from trxrdpy.simulation.parallel import cancellation, SimulationCancelled


def options(order='xzy', reverse=False):
    return dict(param1_name='energy' if not reverse else 'roty',
        param2_name='roty' if not reverse else 'energy',
        param1_range=(10000, 15500, 500) if not reverse else (-25,0,5),
        param2_range=(-25,0,5) if not reverse else (15500,10000,-500),
        sam_space_group=221, sam_a=4,sam_b=5,sam_c=6,
        sam_alpha=90,sam_beta=90,sam_gamma=90, sam_rotation_order=order,
        sam_rotx=13,sam_roty=7,sam_rotz=9, scan_angles={'rotz':35},
        hkls_names=[[-1,1,0],[0,1,1],[0,0,0]], e_bandwidth=150)


@pytest.mark.parametrize('order', ['xyz','xzy','yxz','yzx','zxy','zyx'])
@pytest.mark.parametrize('reverse', [False,True])
def test_exact_match_original_loop(order, reverse):
    opts=options(order,reverse)
    fast=sc.scan_two_parameters_for_Bragg_condition(**opts)
    slow={key:[] for key in fast}
    start,stop,step=opts['param1_range']
    for value in np.arange(start,stop+step,step):
        row=sc.scan_two_parameters_for_Bragg_condition(**opts,_parallel_row=(None,value))
        for key, points in row.items():
            slow[key].extend(points)
    assert fast == slow
    assert any(fast.values())
    assert sc.scan_two_parameters_for_Bragg_condition(**opts,parallel=True,workers=3)==slow


def test_geometry_is_not_recomputed_per_energy(monkeypatch):
    original=sample.cal_reciprocal_lattice
    calls=[]
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(sample,'cal_reciprocal_lattice',counted)
    opts=options()
    opts.update(param1_range=(10000,15500,1), param2_range=(-25,0,.01),e_bandwidth=1.5)
    import itertools
    opts['hkls_names'] = list(itertools.product(range(-2,3), repeat=3))
    start=time.perf_counter()
    result=sc.scan_two_parameters_for_Bragg_condition(**opts,parallel=True,workers=9)
    print(f'Fine-grid benchmark: {time.perf_counter()-start:.3f}s, {len(calls)} geometry evaluations, '
          f'{sum(map(len,result.values()))} output points')
    assert len(calls)==len(np.arange(-25,.01,.01))
    assert sum(map(len,result.values())) > 0


def test_zero_bandwidth_and_cancellation():
    opts=options()
    opts['e_bandwidth']=0
    fast=sc.scan_two_parameters_for_Bragg_condition(**opts)
    assert all(not value for value in fast.values())
    event=multiprocessing.get_context('spawn').Event()
    event.set()
    with pytest.raises(SimulationCancelled), cancellation(event):
        sc.scan_two_parameters_for_Bragg_condition(**opts,parallel=True,workers=3)
