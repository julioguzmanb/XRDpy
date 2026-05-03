import numpy as np
import pyFAI.detectors
from scipy.spatial.transform import Rotation as R

from . import utils
from . import plot

from .geometry import MotorChain, DiffractometerGeometry

def _coerce_detector_transform(rotation_object, angles=None):
    """
    Resolve a detector-side transform from one of:
      - MotorChain
      - DiffractometerGeometry
      - utils.AxisRotation
      - utils.RotationChain
      - 3x3 rotation matrix
      - 4x4 homogeneous transform
    """
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

class Detector:
    def __init__(
        self,
        detector_type=None,
        pxsize_h=None,
        pxsize_v=None,
        num_pixels_h=None,
        num_pixels_v=None,
        dist=None,
        poni1=None,
        poni2=None,
        rotx=None,
        roty=None,
        rotz=None,
        rotation_order="xyz",
        binning=(1, 1)
    ):
        """
        Initialize the Detector object with parameters.

        Parameters:
            detector_type (str): Type of the detector (manual or predefined).
            pxsize_h (float): Pixel size in horizontal direction (meters).
            pxsize_v (float): Pixel size in vertical direction (meters).
            num_pixels_h (int): Number of pixels in horizontal direction.
            num_pixels_v (int): Number of pixels in vertical direction.
            dist (float): Distance from the detector to the sample (meters).
            poni1 (float): PONI1 parameter (meters).
            poni2 (float): PONI2 parameter (meters).
            rotx (float): Rotation angle around x-axis (degrees).
            roty (float): Rotation angle around y-axis (degrees).
            rotz (float): Rotation angle around z-axis (degrees).
            binning (tuple): Binning factors for horizontal and vertical directions.
        """
        self.binning = binning

        if detector_type is None or detector_type.lower() == "manual":
            self.detector_type = "manual"

            if pxsize_h is None or pxsize_v is None:
                raise ValueError("Pixel size not defined")
            self.pxsize_h = pxsize_h * binning[0]
            self.pxsize_v = pxsize_v * binning[1]

            if num_pixels_h is None or num_pixels_v is None:
                raise ValueError("Number of pixels not defined")
            self.num_pixels_h = int(num_pixels_h / binning[0])
            self.num_pixels_v = int(num_pixels_v / binning[1])
        else:
            self.detector_type = detector_type
            try:
                det = pyFAI.detectors.detector_factory(self.detector_type)
                self.pxsize_h = det.pixel1 * binning[0]
                self.pxsize_v = det.pixel2 * binning[1]

                self.num_pixels_h = int(det.MAX_SHAPE[0] / binning[0])
                self.num_pixels_v = int(det.MAX_SHAPE[1] / binning[1])
            except Exception as e:
                print("Invalid detector type, check the printed list")
                raise ValueError("Invalid detector type") from e

        if dist is None:
            raise ValueError("Distance cannot be None")
        self.dist = dist

        if poni1 is None or poni2 is None:
            raise ValueError("PONI parameters not defined")
        self.poni1 = poni1
        self.poni2 = poni2

        self.rotx = 0 if rotx is None else rotx
        self.roty = 0 if roty is None else roty
        self.rotz = 0 if rotz is None else rotz
        self.rotation_order = rotation_order

        self.rotation_matrix = R.from_euler(
            self.rotation_order,
            [self.rotx, self.roty, self.rotz],
            degrees=True
        ).as_matrix()

        self.custom_transform = None
        self.lab_grid = None

    def update_rotation_matrix(self):
        """
        Rebuild the legacy Euler rotation matrix from the current rotx/roty/rotz values.
        """
        self.rotation_matrix = R.from_euler(
            self.rotation_order,
            [self.rotx, self.roty, self.rotz],
            degrees=True
        ).as_matrix()

    def set_rotation_angles(self, rotx=None, roty=None, rotz=None, rotation_order=None):
        """
        Update the legacy Euler-angle description of the detector.

        Notes
        -----
        This does not clear self.custom_transform.
        """
        if rotx is not None:
            self.rotx = rotx
        if roty is not None:
            self.roty = roty
        if rotz is not None:
            self.rotz = rotz
        if rotation_order is not None:
            self.rotation_order = rotation_order

        self.update_rotation_matrix()

    def set_transform(self, transform, angles=None):
        """
        Set a custom detector transform.

        Accepted inputs
        ---------------
        - 3x3 rotation matrix
        - 4x4 homogeneous transform
        - utils.AxisRotation
        - utils.RotationChain
        - MotorChain
        - DiffractometerGeometry (detector chain is used)
        """
        self.custom_transform = _coerce_detector_transform(transform, angles=angles)

    def set_motor_chain(self, motor_chain, angles=None):
        """
        Set a custom detector transform from a MotorChain.
        """
        self.custom_transform = _coerce_detector_transform(motor_chain, angles=angles)

    def set_diffractometer(self, geometry, angles=None):
        """
        Set a custom detector transform from the detector arm of a DiffractometerGeometry.
        """
        self.custom_transform = _coerce_detector_transform(geometry, angles=angles)

    def clear_transform(self):
        """
        Clear the custom transform and return to the legacy Euler-angle path.
        """
        self.custom_transform = None

    def get_transform(self):
        """
        Return the currently active 4x4 detector transform.
        """
        if self.custom_transform is not None:
            return self.custom_transform
        return utils.make_transform(rotation_matrix=self.rotation_matrix)

    def calculate_lab_grid(self, transform=None, angles=None):
        """
        Convert detector pixel positions to the lab coordinate grid.

        Notes
        -----
        - If transform is None and self.custom_transform is None, the legacy Euler path is used.
        - If transform is provided, it overrides self.custom_transform for this call.
        - The final self.lab_grid has shape (num_pixels_h, num_pixels_v, 3).
        """
        dh_indices, dv_indices = np.meshgrid(
            np.arange(self.num_pixels_h),
            np.arange(self.num_pixels_v),
            indexing="ij"
        )

        p2 = (dh_indices + 0.5) * (-self.pxsize_h) + self.poni1
        p3 = (dv_indices + 0.5) * self.pxsize_v - self.poni2
        p1 = np.full_like(p2, self.dist)

        pixel_positions = np.stack((p1, p2, p3), axis=-1)
        detector_matrix = pixel_positions.reshape(-1, 3)

        if transform is not None:
            active_transform = _coerce_detector_transform(transform, angles=angles)
            detector_matrix = utils.apply_transform(detector_matrix, active_transform)
        elif self.custom_transform is not None:
            detector_matrix = utils.apply_transform(detector_matrix, self.custom_transform)
        else:
            detector_matrix = utils.apply_rotation(
                initial_matrix=detector_matrix,
                rotation1=self.rotx,
                rotation2=self.roty,
                rotation3=self.rotz,
                rotation_order=self.rotation_order,
            )

        detector_matrix = detector_matrix.reshape(
            self.num_pixels_h,
            self.num_pixels_v,
            3
        )

        self.lab_grid = detector_matrix

    def plot(self, title="Detector"):
        """
        Plot the 3D lab grid of the detector.

        Parameters:
            title (str): Title of the plot.
        """
        if self.lab_grid is None:
            raise ValueError(
                "Lab grid has not been calculated. Call calculate_lab_grid() first."
            )
        plot.plot_3d_detector(self.lab_grid, title=title)


def extract_coordinates(grid):
    """
    Extract x, y, z coordinates from a lab grid.

    Parameters:
        grid (numpy.ndarray): Detector grid array of shape 
                              (num_pixels_v, num_pixels_h, 3).

    Returns:
        tuple(numpy.ndarray): (x, y, z) coordinate arrays with the same shape 
                              as the first two dimensions of the grid.
    """
    return grid[:, :, 0], grid[:, :, 1], grid[:, :, 2]
