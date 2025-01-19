from .. import utils
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
        det_rotation_order="xyz",
        energy=10e3, e_bandwidth=1.5,
        sam_space_group=167, 
        sam_a=None, sam_b=None, sam_c=None, sam_alpha=None, sam_beta=None, sam_gamma=None,
        sam_initial_crystal_orientation=None,
        sam_rotx=0, sam_roty=0, sam_rotz=0,
        sam_rotation_order="xyz",
        qmax=10
):

    try:
        det = detector.Detector(
            detector_type=det_type,
            pxsize_h=det_pxsize_h, pxsize_v=det_pxsize_v,
            num_pixels_h=det_ntum_pixels_h, num_pixels_v=det_num_pixels_v,
            dist=det_dist, poni1=det_poni1, poni2=det_poni2,
            rotx=det_rotx, roty=det_roty, rotz=det_rotz,
            rotation_order=det_rotation_order,
            binning=det_binning
        )
        det.calculate_lab_grid()
    except:
        raise ImportError

        
    lattice = sample.LatticeStructure(
        space_group=sam_space_group,
        a=sam_a, b=sam_b, c=sam_c, alpha=sam_alpha, beta=sam_beta, gamma=sam_gamma,
        initial_crystal_orientation=sam_initial_crystal_orientation,
        rotation_order=sam_rotation_order
        )
        
    lattice.apply_rotation(rotx=sam_rotx, roty=sam_roty, rotz=sam_rotz)
    lattice.calculate_reciprocal_lattice()
    
    lattice.create_possible_reflections(qmax)
    lattice.calculate_q_hkls()
    wavelength = utils.energy_to_wavelength(energy)
    lattice.check_Bragg_condition(wavelength, e_bandwidth)

    exp = experiment.Experiment(det, lattice, energy = energy, e_bandwidth = e_bandwidth)
    exp.summary()
    exp.calculate_diffraction_direction(qmax)
    exp.plot_3d_single_xstal_exp()

    return exp

    
def simulate_2d(
        det_type="manual", 
        det_pxsize_h=50e-6, det_pxsize_v=50e-6, 
        det_ntum_pixels_h=2000, det_num_pixels_v=2000, det_binning=(1,1),
        det_dist=0.5, det_poni1=0, det_poni2=0, 
        det_rotx=0, det_roty=0, det_rotz=0, 
        det_rotation_order="xyz",
        energy=10e3, e_bandwidth=1.5,
        sam_space_group=167, 
        sam_a=None, sam_b=None, sam_c=None, sam_alpha=None, sam_beta=None, sam_gamma=None,
        sam_initial_crystal_orientation=None,
        sam_rotx=0, sam_roty=0, sam_rotz=0,
        sam_rotation_order="xyz",
        qmax=10, 

):

    try:
        det = detector.Detector(
            detector_type=det_type,
            pxsize_h=det_pxsize_h, pxsize_v=det_pxsize_v,
            num_pixels_h=det_ntum_pixels_h, num_pixels_v=det_num_pixels_v,
            dist=det_dist, poni1=det_poni1, poni2=det_poni2,
            rotx=det_rotx, roty=det_roty, rotz=det_rotz,
            rotation_order=det_rotation_order,
            binning=det_binning
        )
        det.calculate_lab_grid()
    except:
        raise ImportError


    lattice = sample.LatticeStructure(
        space_group=sam_space_group,
        a=sam_a, b=sam_b, c=sam_c, alpha=sam_alpha, beta=sam_beta, gamma=sam_gamma,
        initial_crystal_orientation=sam_initial_crystal_orientation,
        rotation_order=sam_rotation_order
        )
        
    lattice.apply_rotation(rotx=sam_rotx, roty=sam_roty, rotz=sam_rotz)
    lattice.calculate_reciprocal_lattice()
    
    lattice.create_possible_reflections(qmax)
    lattice.calculate_q_hkls()
    wavelength = utils.energy_to_wavelength(energy)
    lattice.check_Bragg_condition(wavelength, e_bandwidth)

    exp = experiment.Experiment(det, lattice, energy = energy, e_bandwidth = e_bandwidth)
    exp.summary()
    exp.calculate_diffraction_direction(qmax)
    exp.calculate_pixel_positions()
    exp.plot_2d_single_xstal_exp()

    return exp


def sample_rotations_for_Bragg_condition(
        sam_space_group, 
        sam_a=None, sam_b=None, sam_c=None, sam_alpha=None, sam_beta=None, sam_gamma=None,
        sam_initial_crystal_orientation=None,
        sam_rotx=0, sam_roty=0, sam_rotz=0,
        sam_rotation_order="xyz",
        angle_range=(-180, 180, 1),
        energy=10e3, e_bandwidth=1.5,
        q_hkls=None, d_hkls=None,
        hkls_names=None
):

    if d_hkls is not None:
        d_hkls = np.array(d_hkls)
        q_hkls = 2 * np.pi / d_hkls

    elif q_hkls is not None:
        q_hkls = np.array(q_hkls)
    
    if ((q_hkls is not None) and (hkls_names is not None)) and (len(q_hkls) != len(hkls_names)):
        raise ValueError
    
    
    lattice = sample.LatticeStructure(
        space_group=sam_space_group,
        a=sam_a, b=sam_b, c=sam_c, alpha=sam_alpha, beta=sam_beta, gamma=sam_gamma,
        initial_crystal_orientation=sam_initial_crystal_orientation,
        rotation_order=sam_rotation_order
        )
    
    lattice.apply_rotation(rotx=sam_rotx, roty=sam_roty, rotz=sam_rotz)
   
    lattice.wavelength=utils.energy_to_wavelength(energy)
    lattice.e_bandwidth=e_bandwidth

    lattice.find_Bragg_orientations(hkls_names, angle_range=angle_range)
    lattice.plot_rotation_mapping()

    return lattice


def detector_rotations_collecting_Braggs(
        det_type="manual", 
        det_pxsize_h=50e-6, det_pxsize_v=50e-6, 
        det_ntum_pixels_h=2000, det_num_pixels_v=2000, det_binning=(1,1),
        det_dist=0.5, det_poni1=0, det_poni2=0, 
        det_rotation_order="xyz",
        angle_range=(-90, 90, 10),
        energy=10e3, e_bandwidth=1.5,
        sam_space_group=167, 
        sam_a=None, sam_b=None, sam_c=None, sam_alpha=None, sam_beta=None, sam_gamma=None,
        sam_initial_crystal_orientation=None,
        sam_rotx=0, sam_roty=0, sam_rotz=0,
        sam_rotation_order="xyz",
        qmax=10, 
        hkls=None
):

    try:
        det = detector.Detector(
            detector_type=det_type,
            pxsize_h=det_pxsize_h, pxsize_v=det_pxsize_v,
            num_pixels_h=det_ntum_pixels_h, num_pixels_v=det_num_pixels_v,
            dist=det_dist, poni1=det_poni1, poni2=det_poni2,
            rotation_order=det_rotation_order,
            binning=det_binning

        )
        det.calculate_lab_grid()
    except:
        raise ImportError
    
        
    lattice = sample.LatticeStructure(
        space_group=sam_space_group,
        a=sam_a, b=sam_b, c=sam_c, alpha=sam_alpha, beta=sam_beta, gamma=sam_gamma,
        initial_crystal_orientation=sam_initial_crystal_orientation,
        rotation_order=sam_rotation_order
        )
        
    lattice.apply_rotation(rotx=sam_rotx, roty=sam_roty, rotz=sam_rotz)

    lattice.calculate_reciprocal_lattice()


    if hkls is None:
        lattice.create_possible_reflections(qmax)
        
    else:
        lattice.allowed_hkls = np.array(list(hkls))
    
    lattice.calculate_q_hkls()
    wavelength = utils.energy_to_wavelength(energy)
    lattice.check_Bragg_condition(wavelength, e_bandwidth)
    
    exp = experiment.Experiment(det, lattice, energy = energy, e_bandwidth = e_bandwidth)
    exp.summary()
    exp.find_detector_rotations(hkls, angle_range=angle_range)
    exp.plot_rotation_mapping()

    return exp

