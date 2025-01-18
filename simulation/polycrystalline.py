from .. import sample
from .. import detector
from .. import experiment

import numpy as np


def simulate_3d(
        det_type="manual", 
        det_pxsize_h=50e-6, det_pxsize_v=50e-6, 
        det_ntum_pixels_h=2000, det_num_pixels_v=2000, det_binning=(1,1),
        det_dist=0.5, det_poni1=0, det_poni2=0, 
        det_rotx=0, det_roty=0, det_rotz=0, 
        cones_num_of_points=30,
        energy=10e3, e_bandwidth=1.5,
        q_hkls=None, d_hkls=None,
        hkls_names=None
):

    try:
        det = detector.Detector(
            detector_type=det_type,
            pxsize_h=det_pxsize_h, pxsize_v=det_pxsize_v,
            num_pixels_h=det_ntum_pixels_h, num_pixels_v=det_num_pixels_v,
            dist=det_dist, poni1=det_poni1, poni2=det_poni2,
            rotx=det_rotx, roty=det_roty, rotz=det_rotz,
            binning=det_binning
        )
        det.calculate_lab_grid()
    except:
        raise ImportError
    
    if (q_hkls is None) and (d_hkls is None):
        raise ValueError
    
    elif d_hkls is not None:
        d_hkls = np.array(d_hkls)
        q_hkls = 2 * np.pi / d_hkls

    elif q_hkls is not None:
        q_hkls = np.array(q_hkls)
    
    if hkls_names is None:
        raise ValueError
    else:
        hkls_names = np.array(hkls_names)

    if len(q_hkls) != len(hkls_names):
        raise ValueError

    #Dummy lattice
    lattice = sample.LatticeStructure(space_group=167, a=1, c=1)    

    exp = experiment.Experiment(det, lattice, energy = energy, e_bandwidth = e_bandwidth)

    exp.plot_3d_polycrystal_exp(q_hkls, hkls_names, cones_num_of_points)

    return exp

    
def simulate_2d(
        det_type="manual", 
        det_pxsize_h=50e-6, det_pxsize_v=50e-6, 
        det_ntum_pixels_h=2000, det_num_pixels_v=2000, det_binning=(1,1),
        det_dist=0.5, det_poni1=0, det_poni2=0, 
        det_rotx=0, det_roty=0, det_rotz=0, 
        cones_num_of_points=1000,
        energy=10e3, e_bandwidth=1.5,
        q_hkls=None, d_hkls=None,
        hkls_names = None
):

    try:
        det = detector.Detector(
            detector_type=det_type,
            pxsize_h=det_pxsize_h, pxsize_v=det_pxsize_v,
            num_pixels_h=det_ntum_pixels_h, num_pixels_v=det_num_pixels_v,
            dist=det_dist, poni1=det_poni1, poni2=det_poni2,
            rotx=det_rotx, roty=det_roty, rotz=det_rotz,
            binning=det_binning
        )
        det.calculate_lab_grid()
    except:
        raise ImportError
    
    if (q_hkls is None) and (d_hkls is None):
        raise ValueError
    
    elif d_hkls is not None:
        d_hkls = np.array(d_hkls)
        q_hkls = 2 * np.pi / d_hkls

    elif q_hkls is not None:
        q_hkls = np.array(q_hkls)
    
    if hkls_names is None:
        raise ValueError
    else:
        hkls_names = np.array(hkls_names)

    if len(q_hkls) != len(hkls_names):
        raise ValueError


    #Dummy lattice
    lattice = sample.LatticeStructure(space_group=167, a=1, c=1)    

    exp = experiment.Experiment(det, lattice, energy = energy, e_bandwidth = e_bandwidth)

    exp.calculate_cones_pixel_positions(q_hkls, hkls_names, num_points=cones_num_of_points)
    exp.plot_2d_polycrystal_exp()

    return exp

    