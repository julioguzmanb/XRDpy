"""Couple sample and detector models into diffraction experiments.

The modern API supports full homogeneous sample/detector transforms and motor
chains. Legacy Euler-angle helpers remain available for compatibility.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm  # For the progress bar
from scipy.spatial.transform import Rotation as R

from . import sample
from . import detector
from . import utils
from . import plot

from itertools import product

from .geometry import MotorChain, DiffractometerGeometry

def _coerce_experiment_detector_transform(detector_obj, rotation_object=None, angles=None):
    """
    Resolve the active detector transform.

    Priority
    --------
    1) Explicit rotation_object
    2) detector_obj.custom_transform
    3) detector_obj.rotation_matrix as a pure rotation transform
    """
    if rotation_object is not None:
        if hasattr(detector, "_coerce_detector_transform"):
            return detector._coerce_detector_transform(rotation_object, angles=angles)

        if isinstance(rotation_object, DiffractometerGeometry):
            return rotation_object.detector_transform(angles=angles)

        if isinstance(rotation_object, MotorChain):
            return rotation_object.as_transform(angles=angles)

        if isinstance(rotation_object, utils.AxisRotation):
            return rotation_object.as_transform()

        if isinstance(rotation_object, utils.RotationChain):
            return rotation_object.as_transform()

        arr = np.asarray(rotation_object, dtype=float)
        if arr.shape == (4, 4):
            return arr
        if arr.shape == (3, 3):
            return utils.make_transform(rotation_matrix=arr)

        raise TypeError(
            "rotation_object must be one of: MotorChain, DiffractometerGeometry, "
            "AxisRotation, RotationChain, 3x3 rotation matrix, or 4x4 transform."
        )

    if getattr(detector_obj, "custom_transform", None) is not None:
        return detector_obj.custom_transform

    return utils.make_transform(rotation_matrix=detector_obj.rotation_matrix)


def _detector_reference_point(detector_obj):
    """Return the PONI reference point in native detector coordinates."""
    return np.array([detector_obj.dist, detector_obj.poni2, -detector_obj.poni1], dtype=float)

class Experiment:
    """Coordinate diffraction calculations for one detector and one lattice.

    Parameters
    ----------
    detector : Detector
        Area-detector geometry receiving the simulated rays.
    lattice : LatticeStructure
        Crystal lattice and current sample orientation.
    energy : float
        Incident photon energy in electron volts.
    e_bandwidth : float
        Full relative energy bandwidth in percent.

    Notes
    -----
    Calculation methods populate cached attributes such as
    ``scattered_directions``, ``pixel_positions``, and diffraction-cone data;
    plotting methods require the corresponding calculation to run first.
    """
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

        self.results = None
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
        if getattr(self.detector, "custom_transform", None) is not None:
            print("  Custom detector transform: active")
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
        """Replace the experiment lattice with ``new_lattice``."""
        self.lattice = new_lattice
        print("Lattice structure updated.")

    def update_detector(self, new_detector: detector.Detector):
        """Replace the experiment detector with ``new_detector``."""
        self.detector = new_detector
        print("Detector updated.")

    def get_detector_transform(self, detector_transform=None, angles=None):
        """Return the active detector transform.

        ``detector_transform`` optionally overrides detector state; ``angles``
        resolves it when the override is a motor chain or geometry.
        """
        return _coerce_experiment_detector_transform(
            self.detector,
            rotation_object=detector_transform,
            angles=angles,
        )

    def get_detector_rotation_matrix(self, detector_transform=None, angles=None):
        """Return the rotational part of an optional detector transform.

        ``detector_transform`` and ``angles`` follow :meth:`get_detector_transform`.
        """
        return self.get_detector_transform(
            detector_transform=detector_transform,
            angles=angles,
        )[:3, :3]

    def calculate_diffraction_direction(self, qmax, detector_transform=None, angles=None):
        """Calculate diffraction directions for the current lattice and detector.

        Parameters
        ----------
        qmax : float
            Reflection cutoff in inverse angstrom if reflections are absent.
        detector_transform : transform-like, optional
            One-call detector transform override.
        angles : dict, optional
            Motor angles used to resolve the override.
        """
        try:
            if self.lattice.q_hkls is None:
                if self.lattice.allowed_hkls is None:
                    self.lattice.create_possible_reflections(qmax)
                self.lattice.calculate_q_hkls()

            if self.lattice.q_hkls_in_Bragg_condition is None:
                self.lattice.check_Bragg_condition(self.wavelength, self.e_bandwidth)

            active_transform = self.get_detector_transform(
                detector_transform=detector_transform,
                angles=angles,
            )

            directions, dir_sign = calculate_diffraction_direction_with_transform(
                self.lattice.q_hkls_in_Bragg_condition,
                self.wavelength,
                detector_dist=self.detector.dist,
                poni1=self.detector.poni1,
                poni2=self.detector.poni2,
                detector_transform=active_transform,
            )

            self.scattered_directions = directions
            self.scat_dir_sign = dir_sign

            self.hkl_to_color = {
                tuple(hkl): color
                for hkl, color in zip(
                    self.lattice.hkls_in_Bragg_condition,
                    plot.colorize(range(len(self.lattice.hkls_in_Bragg_condition)))
                )
            }

        except Exception as e:
            print(f"An error occurred during diffraction direction calculation: {e}")

    def calculate_pixel_positions(self, detector_transform=None, angles=None):
        """Calculate detector pixels for cached scattered directions.

        ``detector_transform`` optionally overrides detector state and ``angles``
        resolves motor-based overrides.
        """
        if self.scattered_directions is None:
            raise ValueError(
                "Scattered directions are not calculated. "
                "Run calculate_diffraction_direction() first."
            )

        active_transform = self.get_detector_transform(
            detector_transform=detector_transform,
            angles=angles,
        )

        self.pixel_positions = lab_to_pixel_coordinates_with_transform(
            self.scattered_directions,
            detector_dist=self.detector.dist,
            pxsize_h=self.detector.pxsize_h,
            pxsize_v=self.detector.pxsize_v,
            poni1=self.detector.poni1,
            poni2=self.detector.poni2,
            detector_transform=active_transform,
        )

    def plot_3d_single_xstal_exp(self, title="Experiment Visualization", plot_crystal=True):
        """
        Plot the detector in the laboratory frame along with the direct beam,
        scattered rays, and the crystal structure.

        Legend entries (one per hkl) are clickable.

        Parameters
        ----------
        title : str
            Figure title.
        plot_crystal : bool
            Include the oriented real-space crystal lattice.
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

        if plot_crystal:
            plot.plot_crystal(
                self.lattice.crystal_orientation / 500,
                axis_labels=["X (m)", "Y (m)", "Z (m)"],
                ax=ax
            )

        plot.plot_3d_detector(self.detector.lab_grid, title=title, ax=ax)

        line_artists = []
        legend_labels = []

        for direction, hkl in zip(
            self.scattered_directions,
            self.lattice.hkls_in_Bragg_condition
        ):
            color = self.hkl_to_color[tuple(hkl)]
            line = ax.plot(
                [0, direction[0]],
                [0, direction[1]],
                [0, direction[2]],
                color=color,
            )[0]
            line_artists.append(line)
            legend_labels.append(f"({hkl[0]},{hkl[1]},{hkl[2]})")

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

        if line_artists:
            legend_handles = []
            for line, label in zip(line_artists, legend_labels):
                col = line.get_color()
                handle = mlines.Line2D(
                    [], [], color=col, marker='o', linestyle='None',
                    markersize=8, label=label
                )
                legend_handles.append(handle)

            legend = ax.legend(
                handles=legend_handles,
                fontsize=10,
                loc="upper left",
                title="Reflections",
                title_fontsize=11,
                bbox_to_anchor=(1.01, 1),
                borderaxespad=0,
            )

            handle_to_line = {}

            legend_lines = list(legend.get_lines())
            legend_texts = list(legend.get_texts())

            for idx, line3d in enumerate(line_artists):
                if idx < len(legend_lines):
                    legline = legend_lines[idx]
                    legline.set_picker(5)
                    handle_to_line[legline] = line3d
                if idx < len(legend_texts):
                    text = legend_texts[idx]
                    text.set_picker(5)
                    handle_to_line[text] = line3d

            def on_pick(event):
                artist = event.artist
                if artist not in handle_to_line:
                    return

                line = handle_to_line[artist]
                visible = not line.get_visible()
                line.set_visible(visible)

                for art, ln in handle_to_line.items():
                    if ln is line:
                        art.set_alpha(1.0 if visible else 0.2)

                fig.canvas.draw_idle()

            fig.canvas.mpl_connect("pick_event", on_pick)

        plt.tight_layout()
        plt.show()

    def plot_2d_single_xstal_exp(self, ax=None):
        """
        Plot a 2D representation of the detector, including scattered rays and the direct beam if applicable.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing detector-plane axes.
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
        Legacy sweep over detector roty/rotz.

        Parameters
        ----------
        hkls : array-like, shape (N, 3)
            Reflections selected from the lattice cache.
        angle_range : tuple of float
            Inclusive ``(start, stop, step)`` range for both angles in degrees.
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

    def find_detector_rotations_with_chain(
        self,
        hkls,
        motor_chain,
        scan_ranges,
        fixed_angles=None,
    ):
        """
        Generalized detector-rotation scan using a MotorChain.

        Parameters
        ----------
        hkls : array-like, shape (N, 3)
            Reflections selected from the lattice cache.
        motor_chain : MotorChain
            Detector chain to scan.
        scan_ranges : dict
            Motor names mapped to inclusive degree range triplets.
        fixed_angles : dict, optional
            Values for unscanned detector motors.
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

            if len(hkls) == 0:
                print(
                    "Given the current exp. conditions, "
                    "none of the selected hkls are in Diffraction Condition"
                )
                self.success_detector_rotations = None
                return None
        else:
            hkls = self.lattice.hkls_in_Bragg_condition
            q_hkls = self.lattice.q_hkls_in_Bragg_condition

        self.success_detector_rotations = find_detector_rotations_with_chain(
            q_hkls=q_hkls,
            hkls=hkls,
            wavelength=self.wavelength,
            detector_obj=self.detector,
            motor_chain=motor_chain,
            scan_ranges=scan_ranges,
            fixed_angles=fixed_angles,
        )

    def find_detector_rotations_with_geometry(
        self,
        hkls,
        geometry,
        scan_ranges,
        fixed_detector_angles=None,
    ):
        """
        Generalized detector-rotation scan using the detector arm of a DiffractometerGeometry.

        Parameters
        ----------
        hkls : array-like, shape (N, 3)
            Reflections selected from the lattice cache.
        geometry : DiffractometerGeometry
            Geometry providing the detector chain.
        scan_ranges : dict
            Detector motor names mapped to inclusive degree range triplets.
        fixed_detector_angles : dict, optional
            Values for unscanned detector motors.
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

            if len(hkls) == 0:
                print(
                    "Given the current exp. conditions, "
                    "none of the selected hkls are in Diffraction Condition"
                )
                self.success_detector_rotations = None
                return None
        else:
            hkls = self.lattice.hkls_in_Bragg_condition
            q_hkls = self.lattice.q_hkls_in_Bragg_condition

        self.success_detector_rotations = find_detector_rotations_with_geometry(
            q_hkls=q_hkls,
            hkls=hkls,
            wavelength=self.wavelength,
            detector_obj=self.detector,
            geometry=geometry,
            scan_ranges=scan_ranges,
            fixed_detector_angles=fixed_detector_angles,
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

    def calculate_cones_direction(self, q_hkls, hkls_names, num_points=100, detector_transform=None, angles=None):
        """
        Calculate diffraction directions for cone-like scattering.

        Parameters
        ----------
        q_hkls : array-like
            Reflection magnitudes in inverse angstrom.
        hkls_names : array-like, shape (N, 3)
            Miller-index labels.
        num_points : int
            Azimuth samples per cone.
        detector_transform : transform-like, optional
            One-call detector transform override.
        angles : dict, optional
            Motor angles used to resolve the override.
        """
        try:
            if self.lattice.wavelength is None:
                self.lattice.wavelength = self.wavelength

            self.lattice.create_kf_hkls(q_hkls, num_points=num_points)

            kf_hkls = self.lattice.kf_hkls.reshape(-1, 3)
            q_hkl = kf_hkls - np.array([2 * np.pi / (self.wavelength * 1e10), 0, 0])

            active_transform = self.get_detector_transform(
                detector_transform=detector_transform,
                angles=angles,
            )

            directions, dir_sign = calculate_diffraction_direction_with_transform(
                q_hkl,
                self.wavelength,
                detector_dist=self.detector.dist,
                poni1=self.detector.poni1,
                poni2=self.detector.poni2,
                detector_transform=active_transform,
            )

            directions = directions.reshape(len(q_hkls), num_points, 3)
            dir_sign = dir_sign.reshape(len(q_hkls), num_points, 1)
            self.diffraction_cones_directions = directions
            self.scat_dir_sign = dir_sign
            self.hkls_names = hkls_names

        except Exception as e:
            print(f"An error occurred during diffraction direction calculation: {e}")

    def calculate_cones_pixel_positions(self, q_hkls, hkls_names, num_points=100, detector_transform=None, angles=None):
        """
        Calculate cone diffraction directions, then map to pixel coordinates.

        Parameters
        ----------
        q_hkls : array-like
            Reflection magnitudes in inverse angstrom.
        hkls_names : array-like, shape (N, 3)
            Miller-index labels corresponding to ``q_hkls``.
        num_points : int
            Azimuth samples per cone.
        detector_transform : transform-like, optional
            One-call detector transform override.
        angles : dict, optional
            Motor angles used to resolve the override.
        """
        try:
            self.calculate_cones_direction(
                q_hkls,
                hkls_names,
                num_points=num_points,
                detector_transform=detector_transform,
                angles=angles,
            )

            active_transform = self.get_detector_transform(
                detector_transform=detector_transform,
                angles=angles,
            )

            self.diffraction_cones_pixel_position = lab_to_pixel_coordinates_with_transform(
                self.diffraction_cones_directions.reshape(-1, 3),
                detector_dist=self.detector.dist,
                pxsize_h=self.detector.pxsize_h,
                pxsize_v=self.detector.pxsize_v,
                poni1=self.detector.poni1,
                poni2=self.detector.poni2,
                detector_transform=active_transform,
            ).reshape(len(q_hkls), num_points, 2)

        except Exception as e:
            print(f"An error occurred during diffraction direction calculation: {e}")

    def plot_3d_polycrystal_exp(self, q_hkls, hkls_names, num_points=50):
        """
        Plot the detector in the laboratory frame together with diffraction cones.

        ``q_hkls`` gives inverse-angstrom magnitudes, ``hkls_names`` gives the
        corresponding Miller labels, and ``num_points`` controls cone sampling.
        """
        if self.detector.lab_grid is None:
            raise ValueError(
                "Detector lab grid is not calculated. "
                "Run detector.calculate_lab_grid() first."
            )

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

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing detector-plane axes.
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

def calculate_diffraction_direction_with_transform(
    q_hkl,
    wavelength,
    detector_dist,
    poni1,
    poni2,
    detector_transform,
):
    """Intersect diffracted rays with a fully transformed detector plane.

    Parameters
    ----------
    q_hkl : array-like, shape (N, 3)
        Reciprocal-space scattering vectors in inverse angstrom.
    wavelength : float
        Incident wavelength in metres.
    detector_dist, poni1, poni2 : float
        Native detector geometry in metres.
    detector_transform : array-like, shape (4, 4)
        Homogeneous native-to-laboratory detector transform.

    Returns
    -------
    directions : numpy.ndarray
        Laboratory intersections with shape ``(N, 3)``.
    direction_sign : numpy.ndarray
        Sign identifying forward versus reversed ray intersections.
    """
    q_hkl = np.asarray(q_hkl, dtype=float)
    if q_hkl.ndim != 2 or q_hkl.shape[1] != 3:
        raise ValueError(f"q_hkl must have shape (N, 3). Got {q_hkl.shape}.")

    detector_transform = np.asarray(detector_transform, dtype=float)
    if detector_transform.shape != (4, 4):
        raise ValueError(
            f"detector_transform must have shape (4, 4). Got {detector_transform.shape}."
        )

    kf_hkl = q_hkl + np.array([2 * np.pi / (wavelength * 1e10), 0.0, 0.0])

    detector_normal_native = np.array([[-1.0, 0.0, 0.0]])
    detector_normal_lab = utils.apply_rotation_matrix(
        detector_normal_native,
        detector_transform[:3, :3]
    )[0]

    plane_point_native = np.array([[detector_dist, poni2, -poni1]], dtype=float)
    plane_point_lab = utils.apply_transform(plane_point_native, detector_transform)[0]

    numerator = np.dot(detector_normal_lab, plane_point_lab)
    denominator = np.einsum("i,ji->j", detector_normal_lab, kf_hkl)

    t = np.full(len(kf_hkl), np.nan, dtype=float)
    non_zero = ~np.isclose(denominator, 0.0)
    t[non_zero] = numerator / denominator[non_zero]

    directions = t[:, np.newaxis] * kf_hkl

    dir_sign = np.sign(np.einsum("ij,ij->i", kf_hkl, directions))
    directions = directions * dir_sign[:, np.newaxis]

    return directions, dir_sign

def lab_to_pixel_coordinates_with_transform(
    lab_positions,
    detector_dist,
    pxsize_h,
    pxsize_v,
    poni1,
    poni2,
    detector_transform,
):
    """Convert laboratory intersections to horizontal/vertical detector pixels.

    The inverse homogeneous transform maps points into the native detector
    frame before the PONI offset and pixel-size conversion is applied.

    Parameters
    ----------
    lab_positions : array-like, shape (N, 3)
        Ray intersections in laboratory metres.
    detector_dist : float
        Native detector-plane distance in metres.
    pxsize_h, pxsize_v : float
        Horizontal and vertical pixel sizes in metres.
    poni1, poni2 : float
        Slow/vertical and fast/horizontal PONI coordinates in metres.
    detector_transform : array-like, shape (4, 4)
        Native-to-laboratory detector transform.

    Returns
    -------
    numpy.ndarray
        Horizontal/vertical pixel coordinates with shape ``(N, 2)``.
    """
    lab_positions = np.asarray(lab_positions, dtype=float)
    if lab_positions.ndim != 2 or lab_positions.shape[1] != 3:
        raise ValueError(f"lab_positions must have shape (N, 3). Got {lab_positions.shape}.")

    detector_transform = np.asarray(detector_transform, dtype=float)
    if detector_transform.shape != (4, 4):
        raise ValueError(
            f"detector_transform must have shape (4, 4). Got {detector_transform.shape}."
        )

    detector_frame_positions = utils.apply_transform(
        lab_positions,
        utils.invert_transform(detector_transform)
    )

    relative_positions = detector_frame_positions - np.array(
        [detector_dist, poni2, -poni1],
        dtype=float
    )

    d_h = -relative_positions[:, 1] / pxsize_h - 0.5
    d_v = relative_positions[:, 2] / pxsize_v - 0.5

    return np.column_stack((d_h, d_v))

def find_detector_rotations_with_chain(
    q_hkls,
    hkls,
    wavelength,
    detector_obj,
    motor_chain,
    scan_ranges,
    fixed_angles=None,
):
    """Scan a detector motor chain and retain angles collecting each reflection.

    ``scan_ranges`` maps motor names to inclusive ``(start, stop, step)``
    ranges. Returned dictionary keys are stringified Miller indices and values
    are angle dictionaries for configurations whose ray lands on the detector.

    Parameters
    ----------
    q_hkls : array-like, shape (N, 3)
        Reflection scattering vectors in inverse angstrom.
    hkls : array-like, shape (N, 3)
        Miller-index labels corresponding to ``q_hkls``.
    wavelength : float
        Incident wavelength in metres.
    detector_obj : Detector
        Detector geometry and pixel bounds.
    motor_chain : MotorChain
        Ordered detector motors to scan.
    scan_ranges : dict
        Scanned motor names mapped to inclusive degree range triplets.
    fixed_angles : dict, optional
        Values for motors not scanned, overlaid on chain defaults.

    Returns
    -------
    dict
        Stringified Miller indices mapped to successful motor-angle dictionaries.
    """
    if not isinstance(motor_chain, MotorChain):
        raise TypeError("motor_chain must be an instance of MotorChain.")

    q_hkls = np.asarray(q_hkls, dtype=float)
    hkls = np.asarray(hkls)

    fixed_angles = {} if fixed_angles is None else dict(fixed_angles)
    scan_ranges = dict(scan_ranges)

    motor_names = list(scan_ranges.keys())
    if len(motor_names) == 0:
        raise ValueError("scan_ranges cannot be empty.")

    angle_values = {
        name: sample._inclusive_angle_values(scan_ranges[name])
        for name in motor_names
    }

    total_steps = int(np.prod([len(vals) for vals in angle_values.values()]))
    progress_bar = tqdm(total=total_steps, desc="Calculating successful detector orientations")

    valid_orientations = {str(hkl): [] for hkl in hkls}

    for combo in product(*[angle_values[name] for name in motor_names]):
        current_angles = dict(fixed_angles)
        current_angles.update(dict(zip(motor_names, combo)))

        current_transform = motor_chain.as_transform(angles=current_angles)

        scattered_directions, scat_dir_sign = calculate_diffraction_direction_with_transform(
            q_hkls,
            wavelength,
            detector_dist=detector_obj.dist,
            poni1=detector_obj.poni1,
            poni2=detector_obj.poni2,
            detector_transform=current_transform,
        )

        pixel_positions = lab_to_pixel_coordinates_with_transform(
            scattered_directions,
            detector_dist=detector_obj.dist,
            pxsize_h=detector_obj.pxsize_h,
            pxsize_v=detector_obj.pxsize_v,
            poni1=detector_obj.poni1,
            poni2=detector_obj.poni2,
            detector_transform=current_transform,
        )

        scattered_within_bounds = plot.within_bounds(
            pixel_positions,
            scat_dir_sign,
            detector_obj.num_pixels_h,
            detector_obj.num_pixels_v
        )

        combo_dict = {name: angle for name, angle in zip(motor_names, combo)}

        for hkl_val, in_bounds in zip(hkls, scattered_within_bounds):
            if in_bounds:
                valid_orientations[str(hkl_val)].append(combo_dict.copy())

        progress_bar.update(1)

    progress_bar.close()
    return valid_orientations

def find_detector_rotations_with_geometry(
    q_hkls,
    hkls,
    wavelength,
    detector_obj,
    geometry,
    scan_ranges,
    fixed_detector_angles=None,
):
    """Scan the detector chain of a :class:`DiffractometerGeometry`.

    Parameters
    ----------
    q_hkls : array-like, shape (N, 3)
        Reflection scattering vectors in inverse angstrom.
    hkls : array-like, shape (N, 3)
        Miller-index labels corresponding to ``q_hkls``.
    wavelength : float
        Incident wavelength in metres.
    detector_obj : Detector
        Detector geometry and pixel bounds.
    geometry : DiffractometerGeometry
        Geometry whose detector chain is scanned.
    scan_ranges : dict
        Detector motor names mapped to inclusive degree range triplets.
    fixed_detector_angles : dict, optional
        Values for detector motors not scanned.

    Returns
    -------
    dict
        Stringified Miller indices mapped to successful angle dictionaries.
    """
    if not isinstance(geometry, DiffractometerGeometry):
        raise TypeError("geometry must be an instance of DiffractometerGeometry.")

    return find_detector_rotations_with_chain(
        q_hkls=q_hkls,
        hkls=hkls,
        wavelength=wavelength,
        detector_obj=detector_obj,
        motor_chain=geometry.detector,
        scan_ranges=scan_ranges,
        fixed_angles=fixed_detector_angles,
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
        poni1 (float): Detector axis 1 coordinate in meters.
        poni2 (float): Detector axis 2 coordinate in meters.
        rotx, roty, rotz (float): Rotation angles in degrees.
        rotation_order (str): Three-axis Euler order used to undo rotation.

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
    relative_positions = detector_frame_positions - np.array([detector_dist, poni2, -poni1])

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
    """Sweep legacy detector angles and find configurations collecting reflections.

    Parameters
    ----------
    q_hkls : array-like, shape (N, 3)
        Reflection scattering vectors in inverse angstrom.
    hkls : array-like, shape (N, 3)
        Miller-index labels corresponding to ``q_hkls``.
    wavelength : float
        Incident wavelength in metres.
    detector_type : str, optional
        Manual mode or a pyFAI detector registry name.
    pxsize_h, pxsize_v : float, optional
        Manual detector pixel sizes in metres.
    num_pixels_h, num_pixels_v : int, optional
        Manual detector pixel counts before binning.
    dist : float
        Sample-to-detector distance in metres.
    poni1, poni2 : float
        Slow/vertical and fast/horizontal PONI coordinates in metres.
    rotx : float
        Fixed detector x rotation in degrees.
    rotation_order : str
        Detector Euler rotation order.
    binning : tuple of int
        Horizontal and vertical binning factors.
    angle_range : tuple of float
        Inclusive ``(start, stop, step)`` used for both ``roty`` and ``rotz``.

    Returns
    -------
    dict
        Each stringified Miller index mapped to successful ``(roty, rotz)`` pairs.
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
    """Legacy Ewald-sphere surface model with finite energy bandwidth.

    Radii are expressed in the reciprocal units implied by ``wavelength``.
    ``Generate_Ewald_Sphere_Data`` returns a regular spherical mesh suitable
    for Plotly, and ``Add_To_Existing_Plot`` appends it as a translucent trace.
    """
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
