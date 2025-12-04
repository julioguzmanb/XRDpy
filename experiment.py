import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For the progress bar
from scipy.spatial.transform import Rotation as R

from . import sample
from . import detector
from . import utils
from . import plot


class Experiment:
    def __init__(self, detector: detector.Detector, lattice: sample.LatticeStructure, energy, e_bandwidth):
        """
        Initialize an experiment instance that combines a detector and a lattice structure.

        Parameters:
            detector (Detector): An instance of the Detector class.
            lattice (LatticeStructure): An instance of the LatticeStructure class.
            energy (float): Beam energy in eV.
            e_bandwidth (float): Energy bandwidth in percentage.
        """
        self.energy = energy
        self.e_bandwidth = e_bandwidth
        self.wavelength = utils.energy_to_wavelength(energy)
        self.detector = detector
        self.lattice = lattice

        self.results = None  # Placeholder for storing experiment results or analysis
        self.scattered_directions = None
        self.diffraction_cones_directions = None
        self.diffraction_cones_pixel_position = None
        self.hkls_names = None
        self.success_detector_rotations = None

    def summary(self):
        """
        Print a summary of the experiment, including details about the detector and lattice.
        """
        print("Experiment Summary:")
        print("\nSource Details:")
        print(f"  Energy [eV]: {self.energy}")
        print(f"  ∆E/E [%]: {self.e_bandwidth}")
        print(f"  wavelength [m]: {self.wavelength}")
        print("\nDetector Details:")
        print(f"  Detector Type: {self.detector.detector_type}")
        print(f"  Pixel Size [m]: {self.detector.pxsize_h} x {self.detector.pxsize_v}")
        print(f"  Number of Pixels: {self.detector.num_pixels_h} x {self.detector.num_pixels_v}")
        print(f"  Binning: {self.detector.binning}")
        print(f"  Distance [m]: {self.detector.dist}")
        print(
            "  Rotation Angles [deg]: "
            f"rotx={self.detector.rotx}, roty={self.detector.roty}, rotz={self.detector.rotz}"
        )
        print("\nLattice Details:")
        print(f"  Phase: {self.lattice.phase}")
        print(
            f"  Lattice Constants [Å]: a={self.lattice.a}, b={self.lattice.b}, c={self.lattice.c}"
        )
        print(
            f"  Angles [deg]: alpha={self.lattice.alpha}, "
            f"beta={self.lattice.beta}, gamma={self.lattice.gamma}"
        )
        print(
            "  Current Crystal Orientation:\n"
            f"{np.round(self.lattice.crystal_orientation, 3)}"
        )

    def update_lattice(self, new_lattice: sample.LatticeStructure):
        """
        Update the lattice structure of the experiment.

        Parameters:
            new_lattice (LatticeStructure): The new lattice structure to use.
        """
        self.lattice = new_lattice
        print("Lattice structure updated.")

    def update_detector(self, new_detector: detector.Detector):
        """
        Update the detector of the experiment.

        Parameters:
            new_detector (Detector): The new detector to use.
        """
        self.detector = new_detector
        print("Detector updated.")

    def calculate_diffraction_direction(self, qmax):
        """
        Calculate the diffraction directions for the given experimental setup.
        """
        try:
            # Ensure lattice has allowed reflections
            if self.lattice.q_hkls is None:
                if self.lattice.allowed_hkls is None:
                    self.lattice.create_possible_reflections(qmax)
                self.lattice.calculate_q_hkls()

            # Ensure lattice satisfies Bragg condition
            if self.lattice.q_hkls_in_Bragg_condition is None:
                self.lattice.check_Bragg_condition(self.wavelength, self.e_bandwidth)

            # Calculate diffraction directions
            directions, dir_sign = calculate_diffraction_direction(
                self.lattice.q_hkls_in_Bragg_condition,
                self.wavelength,
                self.detector.rotation_matrix,
                self.detector.dist
            )
            self.scattered_directions = directions
            self.scat_dir_sign = dir_sign


            # Generate a shared hkl-to-color mapping
            self.hkl_to_color = {
                tuple(hkl): color
                for hkl, color in zip(
                    self.lattice.hkls_in_Bragg_condition,
                    plot.colorize(range(len(self.lattice.hkls_in_Bragg_condition)))
                )
            }

        except Exception as e:
            print(f"An error occurred during diffraction direction calculation: {e}")

    def calculate_pixel_positions(self):
        """
        Calculate the pixel positions on the detector for the scattered directions.

        Raises:
            ValueError: If scattered directions are not calculated or detector parameters are missing.
        """
        if self.scattered_directions is None:
            raise ValueError(
                "Scattered directions are not calculated. "
                "Run calculate_diffraction_direction() first."
            )

        detector_params = {
            "detector_dist": self.detector.dist,
            "pxsize_h": self.detector.pxsize_h,
            "pxsize_v": self.detector.pxsize_v,
            "poni1": self.detector.poni1,
            "poni2": self.detector.poni2,
            "rotx": self.detector.rotx,
            "roty": self.detector.roty,
            "rotz": self.detector.rotz,
            "rotation_order":self.detector.rotation_order
        }

        self.pixel_positions = lab_to_pixel_coordinates(
            self.scattered_directions,
            **detector_params
        )

    def plot_3d_single_xstal_exp(self, title="Experiment Visualization", plot_crystal=True):
        """
        Plot the detector in the laboratory frame along with the direct beam,
        scattered rays, and the crystal structure.
        """
        if self.scattered_directions is None:
            raise ValueError(
                "Scattered directions are not calculated. "
                "Run calculate_diffraction_direction() first."
            )
        if self.detector.lab_grid is None:
            raise ValueError(
                "Detector lab grid is not calculated. "
                "Run detector.calculate_lab_grid() first."
            )

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Optionally plot the crystal structure
        if plot_crystal:
            plot.plot_crystal(
                self.lattice.crystal_orientation / 500,  # Adjust scaling factor
                axis_labels=["X (m)", "Y (m)", "Z (m)"],
                ax=ax
            )

        # Plot the detector
        plot.plot_3d_detector(self.detector.lab_grid, title=title, ax=ax)

        # Add scattered rays
        rays = [
            np.array([[0, 0, 0], direction])
            for direction in self.scattered_directions
        ]
        if rays:
            for direction, hkl in zip(
                self.scattered_directions,
                self.lattice.hkls_in_Bragg_condition
            ):
                color = self.hkl_to_color[tuple(hkl)]
                ax.plot(
                    [0, direction[0]],
                    [0, direction[1]],
                    [0, direction[2]],
                    color=color,
                    label=f"({hkl[0]}, {hkl[1]}, {hkl[2]})"
                )
                ax.text(
                    direction[0], direction[1], direction[2],
                    f"({hkl[0]}, {hkl[1]}, {hkl[2]})",
                    fontsize=12,
                    color=color
                )

        full_title = (
            f"2D Detector: {self.detector.detector_type}\n"
            f"Energy [keV]: {self.energy*1e-3}, Bandwidth [%]: ±{self.e_bandwidth/2}\n"
            f"Wavelength [Å]: {self.wavelength*1e10:.2}\n"
            f"Detector offsets [m]: dist = {self.detector.dist}, "
            f"poni1={self.detector.poni1}, poni2={self.detector.poni2}\n"
            f"Detector Rotations [deg]: rotx={self.detector.rotx:.1f}, "
            f"roty={self.detector.roty:.1f}, rotz={self.detector.rotz:.1f}"
        )
        ax.set_title(full_title)

        plt.tight_layout()
        plt.show()

    def plot_2d_single_xstal_exp(self, ax=None):
        """
        Plot a 2D representation of the detector, including scattered rays and the direct beam if applicable.
        """
        if getattr(self, 'pixel_positions', None) is None:
            raise ValueError(
                "Pixel positions are not calculated. "
                "Run calculate_pixel_positions() first."
            )

        title = (
            f"2D Detector: {self.detector.detector_type}\n"
            f"Energy [keV]: {self.energy*1e-3}, Bandwidth [%]: ±{self.e_bandwidth/2}\n"
            f"Wavelength [Å]: {self.wavelength*1e10:.2}\n"
            f"Detector offsets [m]: dist = {self.detector.dist}, "
            f"poni1={self.detector.poni1}, poni2={self.detector.poni2}\n"
            f"Detector Rotations [deg]: rotx={self.detector.rotx:.1f}, "
            f"roty={self.detector.roty:.1f}, rotz={self.detector.rotz:.1f}"
        )

        ax = plot.plot_2d_detector_single_xstal(
            detector=self.detector,
            pixel_positions=self.pixel_positions,
            scat_dir_sign=self.scat_dir_sign,
            hkls=self.lattice.hkls_in_Bragg_condition,
            hkl_to_color=self.hkl_to_color,
            title=title,
            ax=ax
        )

    def find_detector_rotations(self, hkls, angle_range):
        """
        Find detector rotations that place specified hkls onto the detector.

        Parameters:
            hkls (list of lists or numpy.ndarray): HKLs to consider.
            angle_range (tuple): (start, stop, step) for roty, rotz angle sweeps.

        Returns:
            None or dict: If none of the hkls satisfy the Bragg condition for
                          the given energy/bandwidth, returns None.
                          Otherwise updates self.success_detector_rotations.
        """
        if hkls is not None:
            hkls = np.array(hkls)
            q_hkls = sample.calculate_q_hkl(hkls, self.lattice.reciprocal_lattice)
            bragg_mask = sample.check_Bragg_condition(
                q_hkls,
                self.wavelength,
                self.e_bandwidth
            )
            q_hkls = q_hkls[bragg_mask]
            hkls = hkls[bragg_mask]

            if len(hkls) > 0:
                print(f"Given the current exp. conditions,\n{hkls}\nare in diff condition")
            else:
                print(
                    "Given the current exp. conditions, "
                    "none of the selected hkls are in Diffraction Condition"
                )
                self.success_detector_rotations = None
                return None
        else:
            hkls = self.lattice.hkls_in_Bragg_condition
            q_hkls = self.lattice.q_hkls_in_Bragg_condition
            print(f"Given the current exp. conditions,\n{hkls}\nare in diff condition")

        self.success_detector_rotations = find_detector_rotations(
            q_hkls, hkls,
            self.wavelength,
            detector_type=self.detector.detector_type,
            pxsize_h=self.detector.pxsize_h,
            pxsize_v=self.detector.pxsize_v,
            num_pixels_h=self.detector.num_pixels_h,
            num_pixels_v=self.detector.num_pixels_v,
            dist=self.detector.dist,
            poni1=self.detector.poni1,
            poni2=self.detector.poni2,
            rotation_order=self.detector.rotation_order,
            binning=self.detector.binning,
            angle_range=angle_range
        )

    def plot_rotation_mapping(self):
        """
        Plot the rotation space mapping for successful hkls on the detector.
        """
        if self.success_detector_rotations is None:
            raise AttributeError(
                "success_detector_rotations need to be computed first. "
            )

        title = (
            f"roty and rotz for (h k l) in detector\n"
            f"Energy [keV]: {utils.wavelength_to_energy(self.wavelength)*1e-3}, "
            f"Bandwidth [%]: ±{self.e_bandwidth/2}\n"
            f"Wavelength [Å]: {self.wavelength*1e10:.2}"
        )
        plot.plot_rotation_mapping(self.success_detector_rotations, title=title, s=60)

    def calculate_cones_direction(self, q_hkls, hkls_names, num_points=100):
        """
        Calculate the diffraction directions for the given experimental setup
        in terms of cones (polycrystal approximation).
        """
        try:
            if self.lattice.wavelength is None:
                self.lattice.wavelength = self.wavelength

            self.lattice.create_kf_hkls(q_hkls, num_points=num_points)

            kf_hkls = self.lattice.kf_hkls.reshape(-1, 3)
            q_hkl = kf_hkls - np.array([2 * np.pi / (self.wavelength * 1e10), 0, 0])

            directions, dir_sign = calculate_diffraction_direction(
                q_hkl,
                self.wavelength,
                self.detector.rotation_matrix,
                self.detector.dist
            )

            directions = directions.reshape(len(q_hkls), num_points, 3)
            dir_sign = dir_sign.reshape(len(q_hkls), num_points, 1)
            self.diffraction_cones_directions = directions
            self.scat_dir_sign = dir_sign
            self.hkls_names = hkls_names

        except Exception as e:
            print(f"An error occurred during diffraction direction calculation: {e}")

    def calculate_cones_pixel_positions(self, q_hkls, hkls_names, num_points=100):
        """
        Calculate the diffraction directions for the given experimental setup (polycrystal),
        then map to pixel coordinates.
        """
        try:
            self.calculate_cones_direction(q_hkls, hkls_names, num_points)

            detector_params = {
                "detector_dist": self.detector.dist,
                "pxsize_h": self.detector.pxsize_h,
                "pxsize_v": self.detector.pxsize_v,
                "poni1": self.detector.poni1,
                "poni2": self.detector.poni2,
                "rotx": self.detector.rotx,
                "roty": self.detector.roty,
                "rotz": self.detector.rotz,
                "rotation_order": self.detector.rotation_order
            }

            self.diffraction_cones_pixel_position = lab_to_pixel_coordinates(
                self.diffraction_cones_directions.reshape(-1, 3),
                **detector_params
            ).reshape(len(q_hkls), num_points, 2)

        except Exception as e:
            print(f"An error occurred during diffraction direction calculation: {e}")

    def plot_3d_polycrystal_exp(self, q_hkls, hkls_names, num_points=50):
        """
        Plot the detector in the laboratory frame along with the direct beam,
        scattered rays, and the crystal structure, for a polycrystal scenario.
        """
        if self.detector.lab_grid is None:
            raise ValueError(
                "Detector lab grid is not calculated. "
                "Run detector.calculate_lab_grid() first."
            )

        # Optionally plot the crystal structure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')



        title = (
            f"2D Detector: {self.detector.detector_type}\n"
            f"Energy [keV]: {self.energy*1e-3}, Bandwidth [%]: ±{self.e_bandwidth/2}\n"
            f"Wavelength [Å]: {self.wavelength*1e10:.2}\n"
            f"Detector offsets [m]: dist = {self.detector.dist}, "
            f"poni1={self.detector.poni1}, poni2={self.detector.poni2}\n"
            f"Detector Rotations [deg]: rotx={self.detector.rotx:.1f}, "
            f"roty={self.detector.roty:.1f}, rotz={self.detector.rotz:.1f}"
        )

        plot.plot_3d_detector(self.detector.lab_grid, title=title, ax=ax)
        if self.lattice.wavelength is None:
            self.lattice.wavelength = utils.energy_to_wavelength(self.energy)

        self.lattice.create_diffraction_cones(
            q_hkls,
            coeff=self.detector.dist * 1.1,
            num_points=num_points
        )
        cones = self.lattice.diffraction_cones
        self.hkls_names = hkls_names

        plot.plot_diffraction_cones(cones, self.hkls_names, ax=ax)

        plt.tight_layout()
        plt.show()
        return cones

    def plot_2d_polycrystal_exp(self, ax=None):
        """
        Plot the 2D detector view for polycrystal diffraction cones.
        """
        if self.diffraction_cones_pixel_position is None:
            raise ValueError(
                "Pixel positions are not calculated. "
                "Run calculate_cones_pixel_positions(q_hkls, num_points) first."
            )

        total_points = len(self.diffraction_cones_pixel_position.reshape(-1, 2))
        n = len(self.hkls_names)
        num_points = total_points // n

        title = (
            f"2D Detector: {self.detector.detector_type}\n"
            f"Energy [keV]: {self.energy*1e-3}, Bandwidth [%]: ±{self.e_bandwidth/2}\n"
            f"Wavelength [Å]: {self.wavelength*1e10:.2}\n"
            f"Detector offsets [m]: dist = {self.detector.dist}, "
            f"poni1={self.detector.poni1}, poni2={self.detector.poni2}\n"
            f"Detector Rotations [deg]: rotx={self.detector.rotx:.1f}, "
            f"roty={self.detector.roty:.1f}, rotz={self.detector.rotz:.1f}"
        )

        colors = plot.colorize(np.linspace(0, 1, n))
        expanded_hkls = np.repeat(self.hkls_names, num_points, axis=0)
        vector_colors = np.repeat(colors, num_points, axis=0)

        # Map each hkl to a color
        hkl_to_color = {
            tuple(hkl): color
            for hkl, color in zip(expanded_hkls, vector_colors)
        }

        ax = plot.plot_2d_detector_polycrystal(
            detector=self.detector,
            pixel_positions=self.diffraction_cones_pixel_position.reshape(-1, 2),
            scat_dir_sign=self.scat_dir_sign,
            hkls=expanded_hkls,
            hkl_to_color=hkl_to_color,
            title=title,
            ax=ax
        )


def calculate_diffraction_direction(q_hkl, wavelength, detector_rotation_matrix, dist):
    """
    Calculate the intersection of diffracted beams with the detector plane for all q_hkl vectors.

    Parameters:
        q_hkl (numpy.ndarray): Array of q_hkl vectors (N, 3).
        wavelength (float): X-ray wavelength in meters.
        detector_rotation_matrix (numpy.ndarray): Rotation matrix for the detector.
        dist (float): Distance from sample to detector (meters).

    Returns:
        tuple: (directions, dir_sign) where directions is the intersection points (N, 3),
               and dir_sign is the sign array for each intersection.
    """
    # Calculate kf_hkl for all q_hkl vectors
    kf_hkl = q_hkl + np.array([2 * np.pi / (wavelength * 1e10), 0, 0])

    # Define unit vector normal to the unrotated detector
    n_norm = np.array([-1, 0, 0])

    # Rotate the normal vector if the detector is rotated
    n_norm_rot = np.dot(detector_rotation_matrix, n_norm)

    # Proportionality constant t for each q_hkl vector
    t = (-dist) / np.einsum('i,ji->j', n_norm_rot, kf_hkl)

    # Calculate the intersection points
    directions = t[:, np.newaxis] * kf_hkl

    # Adjust sign if the direction is flipped
    dir_sign = np.sign(np.einsum('ij,ij->i', kf_hkl, directions))[:, np.newaxis]
    directions = directions * dir_sign

    return directions, dir_sign.squeeze()


def lab_to_pixel_coordinates(lab_positions, detector_dist, pxsize_h, pxsize_v, poni1, poni2, rotx, roty, rotz, rotation_order):
    """
    Convert 3D lab space coordinates to pixel positions on the detector.

    Parameters:
        lab_positions (numpy.ndarray): (N, 3) positions in lab space.
        detector_dist (float): Distance from the sample to the detector in meters.
        pxsize_h (float): Horizontal pixel size in meters.
        pxsize_v (float): Vertical pixel size in meters.
        poni1 (float): PONI1 parameter in meters (horizontal offset).
        poni2 (float): PONI2 parameter in meters (vertical offset).
        rotx, roty, rotz (float): Rotation angles in degrees.

    Returns:
        numpy.ndarray: (N, 2) array with pixel coordinates (d_h, d_v).
    """
    inv_rotational_order = rotation_order[-1] + rotation_order[1] + rotation_order[0]

    detector_frame_positions = utils.apply_rotation(
        lab_positions,
        -rotx, -roty, -rotz,
        rotation_order=inv_rotational_order
    )

    # Convert to relative positions in the detector's coordinate system
    relative_positions = detector_frame_positions - np.array([detector_dist, poni1, -poni2])

    # Map to pixel indices
    d_h = -relative_positions[:, 1] / pxsize_h - 0.5
    d_v = relative_positions[:, 2] / pxsize_v - 0.5

    return np.column_stack((d_h, d_v))

def lab_to_pixel_coordinates(lab_positions, detector_dist, pxsize_h, pxsize_v, poni1, poni2, rotx, roty, rotz, rotation_order):
    """
    Convert 3D lab space coordinates to pixel positions on the detector.

    Parameters:
        lab_positions (numpy.ndarray): (N, 3) positions in lab space.
        detector_dist (float): Distance from the sample to the detector in meters.
        pxsize_h (float): Horizontal pixel size in meters.
        pxsize_v (float): Vertical pixel size in meters.
        poni1 (float): PONI1 parameter in meters (horizontal offset).
        poni2 (float): PONI2 parameter in meters (vertical offset).
        rotx, roty, rotz (float): Rotation angles in degrees.

    Returns:
        numpy.ndarray: (N, 2) array with pixel coordinates (d_h, d_v).
    """
    inv_rotational_order = rotation_order[-1] + rotation_order[1] + rotation_order[0]
    #rotation_mat = R.from_euler(rotation_order, [rotx, roty, rotz], degrees=True).as_matrix()
    #rotation_mat_inv = np.linalg.inv(rotation_mat)

    #detector_frame_positions =  np.dot(rotation_mat_inv,lab_positions.T).T

    detector_frame_positions = utils.apply_rotation(
        lab_positions,
        -rotx, -roty, -rotz,
        rotation_order=inv_rotational_order
    )
    # Convert to relative positions in the detector's coordinate system
    relative_positions = detector_frame_positions - np.array([detector_dist, poni1, -poni2])

    relative_positions = np.array([0, -1/pxsize_h, 1/pxsize_v])*relative_positions
    d_h = relative_positions[:,1:2] - 0.5
    d_v = relative_positions[:,2:3] - 0.5

    return np.column_stack((d_h, d_v))


def find_detector_rotations(
    q_hkls,
    hkls,
    wavelength,
    detector_type=None,
    pxsize_h=None,
    pxsize_v=None,
    num_pixels_h=None,
    num_pixels_v=None,
    dist=None,
    poni1=None,
    poni2=None,
    rotx=0,
    rotation_order="xyz",
    binning=(1, 1),
    angle_range=(-180, 180, 5)
):
    """
    Sweep over roty, rotz to find which angles place the reflection on the detector.

    Returns:
        dict: Dictionary mapping each hkl to a list of (roty, rotz) angles.
    """
    angle_range = list(angle_range)
    angle_range[1] = angle_range[1] + angle_range[2]
    
    total_steps = len(np.arange(*angle_range)) * len(np.arange(*angle_range))
    progress_bar = tqdm(total=total_steps, desc="Calculating successful detector orientations")

    valid_orientations = {}
    for hkl in hkls:
        valid_orientations[str(hkl)] = []

    for roty in np.arange(*angle_range):
        for rotz in np.arange(*angle_range):
            det = detector.Detector(
                detector_type=detector_type,
                pxsize_h=pxsize_h,
                pxsize_v=pxsize_v,
                num_pixels_h=num_pixels_h,
                num_pixels_v=num_pixels_v,
                dist=dist,
                poni1=poni1,
                poni2=poni2,
                rotx=rotx,
                roty=roty,
                rotz=rotz,
                rotation_order=rotation_order,
                binning=binning
            )
            det.calculate_lab_grid()

            scattered_directions, scat_dir_sign = calculate_diffraction_direction(
                q_hkls,
                wavelength,
                det.rotation_matrix,
                dist
            )

            detector_params = {
                "detector_dist": det.dist,
                "pxsize_h": det.pxsize_h,
                "pxsize_v": det.pxsize_v,
                "poni1": det.poni1,
                "poni2": det.poni2,
                "rotx": det.rotx,
                "roty": det.roty,
                "rotz": det.rotz,
                "rotation_order": det.rotation_order
            }

            pixel_positions = lab_to_pixel_coordinates(scattered_directions, **detector_params)

            scattered_within_bounds = plot.within_bounds(
                pixel_positions,
                scat_dir_sign,
                det.num_pixels_h,
                det.num_pixels_v
            )

            for hkl_val, in_bounds in zip(hkls, scattered_within_bounds):
                if in_bounds:
                    valid_orientations[str(hkl_val)].append((roty, rotz))

            progress_bar.update(1)

    progress_bar.close()
    return valid_orientations


























class Ewald_Sphere:
    def __init__(self, wavelength, E_bandwidth):
        """
        Initialize an Ewald Sphere instance.

        Parameters:
        wavelength (float): Wavelength of the incident radiation.
        E_bandwidth (float): Energy bandwidth in percentage.

        Returns:
        None
        """
        self.wavelength = wavelength
        self.E_bandwidth = E_bandwidth
        self.radius = 2*np.pi/wavelength
        self.radius_inner = self.radius*(1 - (E_bandwidth/2)/100)
        self.radius_outer = self.radius*(1 + (E_bandwidth/2)/100)

    def Get_Radius(self):
        """
        Get the radius of the Ewald sphere.

        Returns:
        float: Radius of the Ewald sphere.
        """
        return self.radius

    def Get_Inner_Radius(self):
        """
        Get the inner radius of the Ewald sphere.

        Returns:
        float: Inner radius of the Ewald sphere.
        """
        return self.radius_inner
        
    def Get_Outer_Radius(self):
        """
        Get the outer radius of the Ewald sphere.

        Returns:
        float: Outer radius of the Ewald sphere.
        """
        return self.radius_outer

    def Generate_Ewald_Sphere_Data(self):
        """
        Generate data points for visualizing the Ewald sphere.

        Returns:
        tuple: Arrays containing x, y, and z coordinates of points on the Ewald sphere.
        """
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)

        x = self.radius*np.sin(phi)*np.cos(theta)
        y = self.radius*np.sin(phi)*np.sin(theta)
        z = self.radius*np.cos(phi)

        return x, y, z

    def Add_To_Existing_Plot(self, existing_fig):
        """
        Add the Ewald sphere to an existing plot.

        Parameters:
        existing_fig (plotly.graph_objs.Figure): Existing plotly figure.

        Returns:
        plotly.graph_objs.Figure: Figure with the Ewald sphere added.
        """
        x, y, z = self.Generate_Ewald_Sphere_Data()

        sphere = go.Surface(
            x = x,
            y = y, 
            z = z,
            opacity = 0.2,  
            showscale = False,
            colorscale = "Blues"  
        )
        existing_fig.add_trace(sphere)

        return existing_fig
    


def diffraction_direction(Q_hkls, detector, wavelength):
    """
    Calculate the diffraction direction for given Q vectors and a detector setup.

    Parameters:
    Q_hkls (numpy.ndarray): An array of Q vectors corresponding to the Miller indices.
    detector (object): An object representing the detector, with attributes:
                       - beam_center: A tuple (x, y) representing the beam center in pixel coordinates.
                       - pixel_size: A tuple (pixel_size_x, pixel_size_y) representing the size of the detector pixels in meters.
                       - tilting_angle: The tilting angle of the detector in degrees.
                       - sample_detector_distance: The distance from the sample to the detector in meters.
                       - Max_Detectable_Z: A method that returns the maximum detectable Z-coordinate on the detector in meters.
    wavelength (float): The wavelength of the incident X-ray in meters.

    Returns:
    numpy.ndarray: An array of shape (N, 3) containing the diffraction directions (dx, dy, dz) in meters for each Q vector.
    """

    beam_center = (-detector.beam_center[0]*detector.pixel_size[0], detector.Max_Detectable_Z() - detector.beam_center[1]*detector.pixel_size[1]) #This is to make the (0,0) the upper left corner

    
    tilting_angle = np.radians(detector.tilting_angle)

    sample_detector_distance = detector.sample_detector_distance

    wavelength = wavelength*1e10 #Going from m to Å

    ki = np.array([2*np.pi/wavelength, 0, 0]).reshape(1, -1)
    kf_hkls = Q_hkls + ki 
    
    dx, dy, dz = np.zeros(len(kf_hkls)), np.zeros(len(kf_hkls)), np.zeros(len(kf_hkls))

    kfxy = kf_hkls.copy()
    kfxy[:, 2] = 0  # Set the third component to zero for all rows

    kfxz = kf_hkls.copy()
    kfxz[:, 1] = 0  # Set the second component to zero for all rows

    # Calculate dx
    dx[kf_hkls[:, 0] > 0] = 1


    try:
        denominator = kfxz[:,0]
        mask = denominator != 0
        dz[mask] = ((kfxz[:, 2][mask]/denominator)*sample_detector_distance + beam_center[1])/((kfxz[:, 2][mask]/denominator)* np.sin(tilting_angle) + np.cos(tilting_angle))

    except ZeroDivisionError:
        pass

    try:
        denominator = kfxy[:,0]
        mask = denominator != 0

        dy[mask] = (kfxy[:, 1][mask]/denominator)*(sample_detector_distance - dz[mask] * np.sin(tilting_angle)) + beam_center[0]

    except ZeroDivisionError:
        pass

    diffracted_information = np.stack((dx, dy, dz), axis=1)

    return diffracted_information #These in meters

def diffraction_in_detector(diffracted_information, detector):
    """
    Determine if the diffracted directions fall within the detector's detectable area.

    Parameters:
    diffracted_information (numpy.ndarray): An array of shape (N, 3) containing the diffraction directions (dx, dy, dz) in meters.
    detector (object): An object representing the detector, with methods:
                       - Max_Detectable_Y(): Returns the maximum detectable Y-coordinate on the detector in meters.
                       - Min_Detectable_Y(): Returns the minimum detectable Y-coordinate on the detector in meters.
                       - Max_Detectable_Z(): Returns the maximum detectable Z-coordinate on the detector in meters.
                       - Min_Detectable_Z(): Returns the minimum detectable Z-coordinate on the detector in meters.

    Returns:
    numpy.ndarray: A boolean array indicating whether each diffracted direction falls within the detector's detectable area.
    """
    mask = (
        (diffracted_information[:,0] > 0) & 
        (diffracted_information[:,1] <= detector.Max_Detectable_Y()) & 
        (diffracted_information[:,1] >= detector.Min_Detectable_Y()) & 
        (diffracted_information[:,2] <= detector.Max_Detectable_Z()) & 
        (diffracted_information[:,2] >= detector.Min_Detectable_Z())
        )
    
    """
    #This mask is used for SwissFEL2025 simulation purposes...
    mask = (
        (diffracted_information[:,0] > 0)  
        #(diffracted_information[:,1] <= detector.Max_Detectable_Y()) & 
        #(diffracted_information[:,1] >= detector.Min_Detectable_Y()) & 
        #(diffracted_information[:,2] <= detector.Max_Detectable_Z()) & 
        #(diffracted_information[:,2] >= detector.Min_Detectable_Z())
        )
    """
    
    return mask





def single_crystal_orientation(phase, wavelength, detector, sample_detector_distance, beam_center,
                               hkls, rotations, y_coordinates, z_coordinates,
                               crystal_orient_guess, tilting_angle=0, rotation_order="xyz", binning=(1,1)):
    """
    Retrieve the single crystal orientation given 3 Bragg reflections.

    Parameters:
    phase (str): The phase of the crystal, either "Hexagonal" or "Monoclinic".
    wavelength (float): The wavelength of the incident X-ray in meters.
    detector (str): The type of the detector.
    sample_detector_distance (float): The distance from the sample to the detector in meters.
    beam_center (tuple): The beam center coordinates in pixel coordinates.
    hkls (list of lists): A list of Miller indices.
    rotations (list of lists): A list of rotations applied to the crystal in degrees.
    y_coordinates (list of floats): A list of y coordinates of the detected spots in pixels.
    z_coordinates (list of floats): A list of z coordinates of the detected spots in pixels.
    crystal_orient_guess (list of floats): An initial guess for the crystal orientation matrix.
    tilting_angle (float, optional): The tilting angle of the detector in degrees. Default is 0.
    rotation_order (str, optional): The order of rotations. Default is "xyz".
    binning (tuple, optional): The binning of the detector pixels. Default is (1, 1).

    Returns:
    numpy.ndarray: A 3x3 matrix representing the crystal orientation.
    """

    detector = Detector(detector_type = detector, sample_detector_distance=sample_detector_distance, tilting_angle=tilting_angle, beam_center = beam_center, binning = binning)

    beam_center = (-detector.beam_center[0]*detector.pixel_size[0], detector.Max_Detectable_Z() - detector.beam_center[1]*detector.pixel_size[1]) #This is to make the (0,0) the upper left corner

    y_distances = np.array(y_coordinates)*(-detector.pixel_size[0]) #in m
    #z_distances = np.array(z_coordinates)*(detector.pixel_size[1])  #in m
    z_distances = detector.Max_Detectable_Z() - np.array(z_coordinates)*(detector.pixel_size[1])  #in m

    wavelength = wavelength*1e10 #Transforming to Å

    def construct_kf_exp(wavelength, y_distance_from_center, z_distance_from_center, sample_detector_distance, tilting_angle = 0):
        
        tilting_angle = np.deg2rad(tilting_angle)

        ki = 2*np.pi/wavelength

        kfy = ki*(y_distance_from_center - beam_center[0])/np.sqrt((z_distance_from_center*np.sin(tilting_angle) - sample_detector_distance)**2 + (beam_center[0] - y_distance_from_center)**2 + (beam_center[1] - z_distance_from_center*np.cos(tilting_angle))**2)

        kfz = np.sqrt(ki**2 - kfy**2)*(beam_center[1] - z_distance_from_center*np.cos(tilting_angle))/np.sqrt((z_distance_from_center*np.sin(tilting_angle) - sample_detector_distance)**2 + (beam_center[1] - z_distance_from_center*np.cos(tilting_angle))**2)


        kfx = np.sqrt(ki**2 - (kfy + kfz))
        return np.array([kfx, kfy, kfz]) #in Å^(-1)
    
    if phase == "Hexagonal":
        lattice = Hexagonal_Lattice()
        bounds = ([-lattice.c] * 9, [lattice.c] * 9)
    
    elif phase == "Monoclinic":
        lattice = Monoclinic_Lattice()
        bounds = ([-lattice.a] * 9, [lattice.a] * 9)

    a, b, c, alpha, beta, gamma = lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma

    ki = 2*np.pi/wavelength #Transforming wavelength to Å 

    def residuals(params):
        A, B, C, D, E, F, G, H, I = params
        Cryst_Orient = np.array([[A, B, C], [D, E, F], [G, H, I]])
        
        ress = []

        constraint1 = np.linalg.norm(Cryst_Orient[0]) - a
        constraint2 = np.linalg.norm(Cryst_Orient[1]) - b
        constraint3 = np.linalg.norm(Cryst_Orient[2]) - c
        constraint4 = np.dot(Cryst_Orient[0],  Cryst_Orient[1])   - a*b*np.cos(np.deg2rad(gamma))
        constraint5 = np.dot(Cryst_Orient[1],  Cryst_Orient[2])   - b*c*np.cos(np.deg2rad(alpha))
        constraint6 = np.dot(Cryst_Orient[2],  Cryst_Orient[0])   - c*a*np.cos(np.deg2rad(beta ))
        constraints = [constraint1, constraint2, constraint3, constraint4, constraint5, constraint6]

        for i in range(len(hkls)):
            kf = construct_kf_exp(wavelength, y_distances[i], z_distances[i], sample_detector_distance = sample_detector_distance, tilting_angle = tilting_angle)

            Cryst_Orientation = apply_rotation(initial_matrix = Cryst_Orient, rotx = rotations[i][0], roty = rotations[i][1], rotz = rotations[i][2], rotation_order=rotation_order)
            reciprocal_lattice = cal_reciprocal_lattice(Cryst_Orientation)
            GG_peak = calculate_Q_hkl(hkls[i], reciprocal_lattice)

            res_1 =  GG_peak[0] + ki - kf[0]
            res_2 =  GG_peak[1]      - kf[1]
            res_3 =  GG_peak[2]      - kf[2]

            # Restrictions
            ress.append(res_1)
            ress.append(res_2)
            ress.append(res_3)

            # Constraints

            constraint_1 = np.linalg.norm(Cryst_Orientation[0]) - a
            constraint_2 = np.linalg.norm(Cryst_Orientation[1]) - b
            constraint_3 = np.linalg.norm(Cryst_Orientation[2]) - c
            constraint_4 = np.dot(Cryst_Orientation[0],  Cryst_Orientation[1])   - a*b*np.cos(np.deg2rad(gamma))
            constraint_5 = np.dot(Cryst_Orientation[1],  Cryst_Orientation[2])   - b*c*np.cos(np.deg2rad(alpha))
            constraint_6 = np.dot(Cryst_Orientation[2],  Cryst_Orientation[0])   - c*a*np.cos(np.deg2rad(beta ))
            constraints.append(constraint_1)
            constraints.append(constraint_2)
            constraints.append(constraint_3)
            constraints.append(constraint_4)
            constraints.append(constraint_5)
            constraints.append(constraint_6)

        return  np.concatenate((ress, constraints))


    sol = least_squares(residuals, crystal_orient_guess, bounds=bounds, verbose = 2)
    #sol = least_squares(residuals, crystal_orient_guess, verbose = 2, method = "lm")

    #print(sol.jac)
    #np.linalg.inv()

    print(sol)

    solution = np.array([
        [sol.x[0], sol.x[1], sol.x[2]],
        [sol.x[3], sol.x[4], sol.x[5]],
        [sol.x[6], sol.x[7], sol.x[8]]
    ])


    print(solution)
    return np.round(solution, 3)

