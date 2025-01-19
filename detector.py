import numpy as np
import pyFAI.detectors
from scipy.spatial.transform import Rotation as R

from . import utils
from . import plot


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

        # Determine detector parameters
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

        # Distance
        if dist is None:
            raise ValueError("Distance cannot be None")
        self.dist = dist

        # PONI parameters
        if poni1 is None or poni2 is None:
            raise ValueError("PONI parameters not defined")
        self.poni1 = poni1
        self.poni2 = poni2

        # Rotation angles
        if rotx is None or roty is None or rotz is None:
            self.rotx = 0
            self.roty = 0
            self.rotz = 0
        else:
            self.rotx = rotx
            self.roty = roty
            self.rotz = rotz

        self.rotation_order=rotation_order

        # Create a combined rotation matrix
        self.rotation_matrix = R.from_euler(
            self.rotation_order,
            [self.rotx, self.roty, self.rotz],
            degrees=True
        ).as_matrix()

        self.lab_grid = None

    def calculate_lab_grid(self):
        """
        Convert detector pixel positions to the lab coordinate grid.
        This method computes the 3D positions of detector pixels in the lab frame,
        applying any specified rotations.

        The final self.lab_grid has shape (num_pixels_v, num_pixels_h, 3),
        which is typically [row, column, xyz].
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
        detector_matrix = detector_matrix.transpose((1, 0, 2))

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
