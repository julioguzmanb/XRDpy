import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_3d_detector(lab_grid, title="Detector", ax=None):
    """
    Plot a 3D representation of a detector in the lab frame, including the center of rotation 
    and a direct beam line.

    Parameters:
        lab_grid (np.ndarray): 3D lab grid of the detector (shape: [num_pixels_v, num_pixels_h, 3]).
        title (str): Title of the plot.
        ax (matplotlib.axes._subplots.Axes3DSubplot, optional): Existing axis to plot on. 
            Default is None.
    """
    # Extract coordinates
    x, y, z = lab_grid[:, :, 0], lab_grid[:, :, 1], lab_grid[:, :, 2]

    # Calculate axis limits
    min_val = min(x.min(), y.min(), z.min())
    max_val = max(x.max(), y.max(), z.max())

    # If no axis is provided, create a new figure and axis
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    # Plot the detector surface
    ax.plot_surface(x, y, z, alpha=0.2, edgecolor='k')
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Add a black line along the x-axis to represent the direct beam
    ax.plot(
        [min_val, max_val],  # x-coordinates
        [0, 0],              # y-coordinates
        [0, 0],              # z-coordinates
        color='black', linewidth=1, label="Direct Beam", linestyle='--'
    )

    # Set equal axis scaling
    def set_axes_equal(local_ax, grid):
        """
        Set equal scaling for 3D axes based on the bounds of the lab grid.

        Parameters:
            local_ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D subplot.
            grid (np.ndarray): Detector grid in the lab frame (shape: [num_pixels_v, num_pixels_h, 3]).
        """
        x_min, x_max = grid[:, :, 0].min(), grid[:, :, 0].max()
        y_min, y_max = grid[:, :, 1].min(), grid[:, :, 1].max()
        z_min, z_max = grid[:, :, 2].min(), grid[:, :, 2].max()

        # Include the center of rotation
        x_min, x_max = min(x_min, 0), max(x_max, 0)
        y_min, y_max = min(y_min, 0), max(y_max, 0)
        z_min, z_max = min(z_min, 0), max(z_max, 0)

        # Compute ranges
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        # Determine the largest range
        max_range = max(x_range, y_range, z_range)

        # Compute centers
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        # Set equal limits
        local_ax.set_xlim3d([x_center - max_range / 2, x_center + max_range / 2])
        local_ax.set_ylim3d([y_center - max_range / 2, y_center + max_range / 2])
        local_ax.set_zlim3d([z_center - max_range / 2, z_center + max_range / 2])

    set_axes_equal(ax, lab_grid)

    if ax is None:
        plt.tight_layout()
        plt.show()


def plot_crystal(
    crystal_lattice,
    xlims=(-14, 14),
    ylims=(-14, 14),
    zlims=(-14, 14),
    axis_labels=None,
    ax=None
):
    """
    Plot a 3D visualization of a crystal lattice using lattice vectors.

    Parameters:
        crystal_lattice (numpy.ndarray): A 3x3 matrix where each column represents a lattice vector.
        xlims (tuple, optional): x-axis limits for the plot. Default is (-14, 14).
        ylims (tuple, optional): y-axis limits for the plot. Default is (-14, 14).
        zlims (tuple, optional): z-axis limits for the plot. Default is (-14, 14).
        axis_labels (list, optional): List of axis labels. Default is ["x [Å]", "y [Å]", "z [Å]"].
        ax (matplotlib.axes._subplots.Axes3DSubplot, optional): An existing 3D axis to plot on.

    Returns:
        None: This function does not return any value but shows a 3D plot of the crystal lattice.
    """
    # Create a new figure and axis if none is provided
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    lattice_vectors = crystal_lattice.T

    v0 = lattice_vectors[:, 0]
    v1 = lattice_vectors[:, 1]
    v2 = lattice_vectors[:, 2]

    origin = -(v0 + v1 + v2) / 2

    vectors = [v0, v1, v2]
    labels = ['a', 'b', 'c']
    colors = ['r', 'g', 'b']
    for vec, label, color in zip(vectors, labels, colors):
        ax.quiver(*origin, *vec, color=color, length=1.0,
                  normalize=False, arrow_length_ratio=0.1)
        ax.text(*(vec + origin), label, color=color)

    points = np.array([
        origin,
        v0 + origin, v1 + origin, v2 + origin,
        v0 + v1 + origin, v1 + v2 + origin, v2 + v0 + origin,
        v0 + v1 + v2 + origin
    ])

    faces = [
        [points[i] for i in [0, 1, 4, 2]],
        [points[i] for i in [0, 2, 5, 3]],
        [points[i] for i in [0, 3, 6, 1]],
        [points[i] for i in [7, 4, 1, 6]],
        [points[i] for i in [7, 5, 2, 4]],
        [points[i] for i in [7, 6, 3, 5]],
    ]
    ax.add_collection3d(
        Poly3DCollection(
            faces,
            facecolors='yellow',
            linewidths=1,
            edgecolors='gray',
            alpha=.1
        )
    )

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)

    if axis_labels is None:
        axis_labels = ['x [Å]', 'y [Å]', 'z [Å]']

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])

    if ax.figure is not None:
        plt.tight_layout()


def plot_2d_detector_single_xstal(
    detector,
    pixel_positions,
    scat_dir_sign,
    hkls,
    hkl_to_color,
    title="2D Detector View",
    ax=None
):
    """
    Plot a 2D representation of the detector, including scattered rays and 
    the direct beam if applicable.

    Parameters:
        detector: Detector instance with attributes num_pixels_v, pxsize_v, 
                  num_pixels_h, pxsize_h, etc.
        pixel_positions (numpy.ndarray): Positions of scattered rays in pixel space.
        scat_dir_sign (numpy.ndarray): Sign array for scattering direction, 
                                       helps remove invalid directions.
        hkls (list or numpy.ndarray): List/array of Miller indices.
        hkl_to_color (dict): Mapping from hkl tuples to colors.
        title (str): Title of the plot.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Axis to plot on. 
            Default is None.
    """

    # Determine valid scattered positions
    scattered_within_bounds = within_bounds(
        pixel_positions,
        scat_dir_sign,
        detector.num_pixels_h,
        detector.num_pixels_v
    )
    valid_positions = pixel_positions[scattered_within_bounds]

    if len(valid_positions) > 0:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        # Adjust aspect ratio
        aspect_ratio = (
            (detector.num_pixels_v * detector.pxsize_v)
            / (detector.num_pixels_h * detector.pxsize_h)
        )
        ax.set_aspect(aspect_ratio)

        

        # Match valid HKLs
        if hkls is not None and len(hkls) > 0:
            valid_hkls = [hkls[i] for i, valid in enumerate(scattered_within_bounds) if valid]
        else:
            valid_hkls = None

        # Plot valid reflections
        if valid_positions.size > 0 and valid_hkls is not None:
            for pos, hkl in zip(valid_positions, valid_hkls):
                color = hkl_to_color[tuple(hkl)]
                ax.scatter(pos[0], pos[1],
                        label=f"({hkl[0]},{hkl[1]},{hkl[2]})",
                        s=15, color=color)

        ax.set_title(title)
        ax.set_xlabel("Horizontal Pixels")
        ax.set_ylabel("Vertical Pixels")

        ax.legend(
            fontsize=11,
            loc="upper left",
            title="Reflections",
            title_fontsize=13,
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            markerscale=4
        )

        ax.grid()
        ax.set_xlim(0, detector.num_pixels_h)
        ax.set_ylim(0, detector.num_pixels_v)

        if ax.figure is not None:
            plt.tight_layout()
            plt.show()

        return ax
    else:
        print("No hkls in the detector")


def plot_2d_detector_polycrystal(
    detector,
    pixel_positions,
    scat_dir_sign,
    hkls,
    hkl_to_color,
    title="2D Detector View",
    ax=None
):
    """
    Plot a 2D representation of the detector for multiple reflections (polycrystal).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Adjust aspect ratio with binning
    aspect_ratio = (
        (detector.num_pixels_v * detector.pxsize_v * detector.binning[1])
        / (detector.num_pixels_h * detector.pxsize_h * detector.binning[0])
    )
    ax.set_aspect(aspect_ratio)

    # Evaluate which positions are within detector bounds
    scattered_within_bounds = within_bounds(
        pixel_positions,
        scat_dir_sign.reshape(-1, 1).squeeze(),
        detector.num_pixels_h,
        detector.num_pixels_v
    )

    valid_positions = pixel_positions[scattered_within_bounds]
    valid_hkls = np.array(hkls)[scattered_within_bounds] if hkls is not None else None

    if valid_positions.size > 0 and valid_hkls is not None:
        unique_hkls, first_indices = np.unique(valid_hkls, axis=0, return_index=True)
        ordered_unique_hkls = unique_hkls[np.argsort(first_indices)]

        plotted_hkls = set()
        for hkl in ordered_unique_hkls:
            hkl_tuple = tuple(hkl)
            indices = (valid_hkls == hkl).all(axis=1)
            hkl_positions = valid_positions[indices]
            if hkl_positions.size > 0:
                color = hkl_to_color[hkl_tuple]
                ax.scatter(
                    hkl_positions[:, 0],
                    hkl_positions[:, 1],
                    label=f"({hkl[0]},{hkl[1]},{hkl[2]})",
                    s=5,
                    color=color
                )
                plotted_hkls.add(hkl_tuple)

    ax.set_title(title)
    ax.set_xlabel("Horizontal Pixels")
    ax.set_ylabel("Vertical Pixels")

    # Add legend only for plotted HKLs
    handles, labels = ax.get_legend_handles_labels()
    new_labels, new_handles = [], []
    for handle, label in zip(handles, labels):
        hkl = tuple(map(int, label.strip("()").split(',')))
        if hkl in plotted_hkls:
            new_labels.append(label)
            new_handles.append(handle)

    ax.legend(
        new_handles,
        new_labels,
        fontsize=11,
        loc="upper left",
        title="Reflections",
        title_fontsize=13,
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        markerscale=4
    )

    ax.grid()
    ax.set_xlim(0, detector.num_pixels_h)
    ax.set_ylim(0, detector.num_pixels_v)

    if ax.figure is not None:
        plt.tight_layout()
        plt.show()

    return ax


def plot_diffraction_cones(cones, hkls_names, ax=None):
    """
    Plot proper diffraction cones in 3D with colorized surfaces.

    Parameters:
        cones (list): List of cone surfaces (as arrays) generated by create_diffraction_cones.
        hkls_names (list): Names of the hkls corresponding to each cone.
        ax (matplotlib.axes._subplots.Axes3DSubplot, optional): Existing 3D axis to plot on.
            If None, a new figure and axis are created.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    indices = np.arange(len(cones))
    colors = colorize(indices, vmin=0, vmax=len(cones) - 1)

    # List to hold legend handles
    legend_handles = []

    # Plot each cone surface
    alphas = np.linspace(0.5, 0.1, len(cones))
    for (x, y, z), color, alpha, hkl_name in zip(cones, colors, alphas, hkls_names):
        facecolors = np.empty(x.shape + (4,))
        facecolors[:, :] = color
        ax.plot_surface(x, y, z, alpha=alpha, facecolors=facecolors, edgecolor="none")

        # Create a patch for the legend with full opacity
        legend_patch = Patch(
            color=color,
            label=str(list(hkl_name)).replace("[", "(").replace("]", ")").replace(" ", "")
        )
        legend_handles.append(legend_patch)

    ax.legend(
        handles=legend_handles,
        fontsize=11,
        loc="upper left",
        title="Reflections",
        title_fontsize=13,
        bbox_to_anchor=(1.11, 1),
        borderaxespad=0.
    )

    plt.show()
    return ax


def plot_rotation_mapping(valid_orientations, title="roty and rotz for (h k l) in Bragg condition", s=20):
    """
    Plot successful (roty, rotz) combinations that satisfy the Bragg condition.

    Parameters:
        valid_orientations (dict): Dictionary mapping hkl -> list of (roty, rotz).
        title (str): Title of the plot.
    """
    colors = colorize(np.linspace(0, 1, len(valid_orientations.keys())))

    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    legend_handles = []

    for (hkl, color) in zip(valid_orientations.keys(), colors):
        # Unpack the rotations into separate lists
        if valid_orientations[f"{hkl}"]:
            roty, rotz = zip(*valid_orientations[f"{hkl}"])
        else:
            roty, rotz = [], []

        label = "(" + ",".join(hkl.strip("[]").split()) + ")"

        plt.scatter(roty, rotz, s=s, color=color, alpha=0.6)

        # Create a custom legend handle
        legend_handle = mlines.Line2D(
            [], [], color=color, marker='o', linestyle='None', markersize=10, label=label
        )
        legend_handles.append(legend_handle)

    plt.title(title, fontsize=15)
    plt.xlabel("roty (deg)", fontsize=13)
    plt.ylabel("rotz (deg)", fontsize=13)
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    ax.set_aspect("equal", adjustable="box")
    plt.grid()
    plt.legend(
        handles=legend_handles,
        title="(h k l)",
        title_fontsize=13,
        fontsize=11,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0.
    )
    plt.show()
    plt.tight_layout()


def colorize(array, vmin=None, vmax=None, cmap=plt.cm.jet):
    """
    Generate a normalized colormap for the given array.

    Parameters:
        array (numpy.ndarray or list): Array or list of values to normalize and colorize.
        vmin (float, optional): Minimum value for normalization. Defaults to the minimum of the array.
        vmax (float, optional): Maximum value for normalization. Defaults to the maximum of the array.
        cmap (matplotlib.colors.Colormap, optional): Colormap to use for colorization. 
            Defaults to plt.cm.jet.

    Returns:
        list: List of RGBA colors corresponding to the normalized values.
    """
    array = np.asarray(array)

    # Set vmin and vmax if not provided
    if vmin is None:
        vmin = array.min()
    if vmax is None:
        vmax = array.max()
    if len(array) > 1:
        normalized = (array - vmin) / (vmax - vmin)
    else:
        normalized = array*0

    return [cmap(value) for value in normalized]


def within_bounds(pixel_positions, scat_dir_sign, num_pixels_h, num_pixels_v):
    """
    Determine which scattered rays land on the detector.

    Parameters:
        pixel_positions (numpy.ndarray): (N, 2) array of pixel positions.
        scat_dir_sign (numpy.ndarray): Sign array for scattering direction.
        num_pixels_h (int): Detector pixels in horizontal dimension.
        num_pixels_v (int): Detector pixels in vertical dimension.

    Returns:
        numpy.ndarray: Boolean mask indicating valid pixel indices.
    """
    return (
        (pixel_positions[:, 0] >= 0) & (pixel_positions[:, 0] < num_pixels_h) &
        (pixel_positions[:, 1] >= 0) & (pixel_positions[:, 1] < num_pixels_v) &
        (scat_dir_sign > 0)
    )





































def plot_reciprocal(Q_hkls, hkls, wavelength, E_bandwidth):
    """
    Plot the reciprocal space and Ewald construction.

    Parameters:
    - Q_hkls (numpy.ndarray): Array of shape (N, 3) containing the scattering vectors.
    - hkls (numpy.ndarray): Array of shape (N, 3) containing the Miller indices for each reflection.
    - wavelength (float): Wavelength of the incident X-ray beam in meters.
    - E_bandwidth (float): Energy bandwidth of the X-ray beam in percentage.

    Returns:
    - None
      Displays the 3D plot.
    """

    ewald_sphere = utils.Ewald_Sphere(wavelength, E_bandwidth)
    ki = np.array([2*np.pi/(wavelength*1e10), 0, 0]).reshape(1, -1)

    kf_hkls = Q_hkls + ki

    in_bragg_condition = utils.check_Bragg_condition(Q_hkls, wavelength, E_bandwidth)

    fig = plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 15})

    ax = fig.add_subplot(111, projection='3d')

    for i, (x,y,z) in enumerate(kf_hkls[in_bragg_condition]):
        ax.scatter(x, y, z, label="(%s)"%(str(hkls[in_bragg_condition].tolist()[i]).replace("[","").replace("]","").replace(",","")), s = 40)  # Plot points with colors

    if len(hkls[in_bragg_condition]) > 1:
        Colorize(vector = list(range(len(hkls[in_bragg_condition]))),cmap=plt.cm.jet, ax = ax)

    ax.scatter(0, 0, 0, c='black', label='Ewald Sphere Center', s = 100)  # Plot center of Ewald sphere

    #ax.legend(fontsize = 13, framealpha = 1, title = "(hkl) in Bragg C.")
    legend = ax.legend(fontsize = 13, framealpha = 1, title = "(hkl) in Bragg c.")
    title = legend.get_title()
    title.set_fontsize(14)

    #Plotting data that is not in Bragg condition
    x, y, z = kf_hkls[~in_bragg_condition].T  # Transpose to get x, y, z separately
    ax.scatter(x, y, z, c = "blue", s = 20)  # Plot points with colors

    # Plot Ewald sphere
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x_ewald = (ewald_sphere.Get_Radius()*1e-10) * np.cos(u) * np.sin(v)
    y_ewald = (ewald_sphere.Get_Radius()*1e-10) * np.sin(u) * np.sin(v)
    z_ewald = (ewald_sphere.Get_Radius()*1e-10) * np.cos(v)
    ax.plot_surface(x_ewald, y_ewald, z_ewald, color='lightgreen', alpha=0.15, linewidth=1)

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add title
    ax.set_title('Ewald Construction')

    # Set aspect ratio
    ax.set_box_aspect([1,1,1])

    # Set axis limits
    lim = int(np.linalg.norm(ki)*1.2)
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    # Adjust view angle manually
    ax.view_init(elev=30, azim=45)

    # Initialize mouse controls for zooming
    ax.mouse_init()

    plt.tight_layout()

    plt.show()


def plot_detector(data, colorize = False):
    """
    Plot the detector layout.

    Parameters:
    - data (dict): Dictionary containing detector information.
                   Requires keys: "detector", "y_coordinate", "z_coordinate", "hkls".
    - colorize (bool): If True, colorizes the points on the detector plot based on the Miller indices.
                       Default is False.

    Returns:
    - None
      Displays the detector plot.
    """
    detector = data["detector"]
    
    if colorize == True:
        s = 30 #To change after ESRF crystal orientation
        [plt.scatter(y_val, z_val, label=label, s = s) for y_val, z_val, label in zip(data["y_coordinate"], data["z_coordinate"], data["hkls"])]
        colorize(vector = list(range(len(data["hkls"]))), cmap = plt.cm.jet)

    else:
        [plt.scatter(y_val, z_val, label=label, color = "blue") for y_val, z_val, label in zip(data["y_coordinate"], data["z_coordinate"], data["hkls"])]
    
    plt.scatter(detector.beam_center[0], detector.beam_center[1], label = "Beam Center",marker='x', color='black', s = 100)
    plt.legend(title = "(h,k,l)", loc = "upper right", fontsize = 14, framealpha = 0.4)

    plt.xlim(abs(detector.Max_Detectable_Y()/detector.pixel_size[0]), abs(detector.Min_Detectable_Y()/detector.pixel_size[0]))
    plt.xlabel("y-direction [pixel]",fontsize = 16)
    plt.ylim(detector.Min_Detectable_Z()/detector.pixel_size[1], detector.Max_Detectable_Z()/detector.pixel_size[1])
    plt.ylabel("z-direction [pixel]",fontsize = 16)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()


    
def plot_guidelines(hkls, lattice_structure, detector, wavelength):
    """
    Plot diffraction circle guidelines on the detector.

    Parameters:
    - hkls (numpy.ndarray): Array of Miller indices (hkl) for the diffraction circles.
    - lattice_structure (Lattice_Structure): Object representing the crystal lattice structure.
    - detector (Detector): Object representing the detector.
    - wavelength (float): Wavelength of incident X-ray radiation.

    Returns:
    - None
      Displays the diffraction circle guidelines on the detector plot.
    """

    # Calculate two theta angles
    two_theta = utils.calculate_two_theta(hkl = hkls, reciprocal_lattice=lattice_structure.reciprocal_lattice, wavelength=wavelength)

    # Calculate distances from sample to detector
    r = detector.sample_detector_distance*np.tan(np.radians(two_theta))

    # Generate angles
    theta = np.linspace(0, 2*np.pi, 100)

    # Calculate y and z coordinates for a circle
    y = r * np.cos(theta)/(-detector.pixel_size[0])
    z = r * np.sin(theta)/(detector.pixel_size[1])

    # Function to distort circle based on detector parameters
    def distort_circle(y,z, detector):
        tilting_angle = np.radians(detector.tilting_angle)

        # Convert y and z coordinates to meters
        y = y*(-detector.pixel_size[0])
        z = z*(detector.pixel_size[1])

        # Calculate beam center in meters
        beam_center = (-detector.beam_center[0]*detector.pixel_size[0], detector.beam_center[1]*detector.pixel_size[1]) #In meters

        # Apply distortion to y and z coordinates
        Z = (z + beam_center[1])/(z*np.sin(tilting_angle)/detector.sample_detector_distance + np.cos(tilting_angle))
        Y = ((detector.sample_detector_distance - beam_center[1]*np.tan(tilting_angle))/(detector.sample_detector_distance + z*np.tan(tilting_angle)))*y + beam_center[0]
        return Y,Z
    
    # Apply distortion to circle coordinates
    Y,Z = distort_circle(y,z, detector)

    # Scale back y and z coordinates
    Y = Y/(-detector.pixel_size[0])
    Z = Z/(detector.pixel_size[1])

    # Plot distorted circle as guidelines
    plt.plot(Y, Z, "--",color = "black", linewidth = 2)


def plot_guidelines(hkls, lattice_structure, detector, wavelength):
    """
    Plot diffraction circle guidelines on the detector.

    Parameters:
    - hkls (numpy.ndarray): Array of Miller indices (hkl) for the diffraction circles.
    - lattice_structure (Lattice_Structure): Object representing the crystal lattice structure.
    - detector (Detector): Object representing the detector.
    - wavelength (float): Wavelength of incident X-ray radiation.

    Returns:
    - None
      Displays the diffraction circle guidelines on the detector plot.
    """

    hkls = np.array(hkls)

    two_theta = utils.calculate_two_theta(hkl=hkls, reciprocal_lattice=lattice_structure.reciprocal_lattice, wavelength=wavelength)
    r = detector.sample_detector_distance * np.tan(np.radians(two_theta))

    theta = np.linspace(0, 2 * np.pi, 100)

    # Calculate y and z for each hkl using broadcasting
    y = np.outer(r, np.cos(theta)) / (-detector.pixel_size[0])
    z = np.outer(r, np.sin(theta)) / detector.pixel_size[1]

    def distort_circle(y, z, detector):
        """
        Distort circle positions based on detector tilt.

        Parameters:
        - y (numpy.ndarray): Y-coordinates of circle positions.
        - z (numpy.ndarray): Z-coordinates of circle positions.
        - detector (Detector): Object representing the detector.

        Returns:
        - Y (numpy.ndarray): Distorted Y-coordinates of circle positions.
        - Z (numpy.ndarray): Distorted Z-coordinates of circle positions.
        """
        tilting_angle = np.radians(detector.tilting_angle)
        pixel_size = detector.pixel_size

        beam_center = (-detector.beam_center[0]*detector.pixel_size[0], detector.Max_Detectable_Z() - detector.beam_center[1]*detector.pixel_size[1]) #This is to make the (0,0) the upper left corner

        # Convert pixel positions to meters
        y_m = y * (-pixel_size[0])
        z_m = z * pixel_size[1]

        # Compute Z and Y using vectorized operations
        Z = (z_m + beam_center[1]) / (z_m * np.sin(tilting_angle) / detector.sample_detector_distance + np.cos(tilting_angle))
        Y = ((detector.sample_detector_distance - beam_center[1] * np.tan(tilting_angle)) / (detector.sample_detector_distance + z_m * np.tan(tilting_angle))) * y_m + beam_center[0]

        return Y, Z

    # Apply distortion to all circles
    Y, Z = distort_circle(y, z, detector)

    # Convert Y and Z back to pixels
    Y = Y / (-detector.pixel_size[0])
    Z = (detector.Max_Detectable_Z() - Z) / detector.pixel_size[1] #This is to make the (0,0) the upper left corner

    # Plot all distorted circles
    if len(hkls) == 1:
        plt.plot(Y[i], Z[i], "--", color="black", linewidth=2, label = str(hkls[i]).replace("[", "(").replace("]", ")"))

    else:
        hkls_color = np.linspace(0, 1, len(hkls))
        for i in range(len(hkls)):
            plt.plot(Y[i], Z[i], "--", color = plt.cm.jet(hkls_color[i]),linewidth=1, label = str(hkls[i]).replace("[", "(").replace("]", ")"))
    
    plt.legend(title = "(h,k,l)", loc = "upper right", fontsize = 12, framealpha = 0.95)

    plt.show()



"""
def plot_reciprocal(Q_hkls, hkls, wavelength, E_bandwidth):

    ewald_sphere = utils.Ewald_Sphere(wavelength, E_bandwidth)
    #ki = np.array([ewald_sphere.Get_Radius(), 0, 0])
    ki = np.array([2*np.pi/wavelength, 0, 0]).reshape(1, -1)

    kf_hkls = Q_hkls + ki

    in_bragg_condition = utils.check_Bragg_condition(Q_hkls, wavelength, E_bandwidth)

    colors = np.where(in_bragg_condition, 'red', 'blue')

    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode = "markers",
        marker = dict(
            size = 8,
            color = "yellow", 
        ),
    ))

    for w, kf_hkl in enumerate(kf_hkls):
        
        x, y, z = kf_hkl

        if utils.check_Bragg_condition(Q_hkls[w], wavelength, E_bandwidth) == True: 
            color = "red"
        else:
            color = "blue"

        fig.add_trace(go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode = "markers",
            text = [hkls[w]],
            marker = dict(
                size = 8,
                color = color, 
            ),
            name=f'({hkls[w][0]}, {hkls[w][1]}, {hkls[w][2]})' 
        ))

    fig = ewald_sphere.Add_To_Existing_Plot(fig)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-12, 12]), 
            yaxis=dict(range=[-12, 12]), 
            zaxis=dict(range=[-12, 12]), 
            aspectratio=dict(x=1, y=1, z=1),
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z"
        ),
        title={
            "text": "Ewald Construction",
            "x": 0.5,
            "y": 0.9,  
            "xanchor": 'center', 
            "yanchor": 'top', 
            "font": {'size': 38} 
        }
    )

    fig.show()
"""

"""
def plot_guidelines(hkls, lattice_structure, detector, wavelength):

    Plot diffraction circle guidelines on the detector.

    Parameters:
    - hkls (numpy.ndarray): Array of Miller indices (hkl) for the diffraction circles.
    - lattice_structure (Lattice_Structure): Object representing the crystal lattice structure.
    - detector (Detector): Object representing the detector.
    - wavelength (float): Wavelength of incident X-ray radiation.

    Returns:
    - None
      Displays the diffraction circle guidelines on the detector plot.

    # Calculate two theta angles
    two_theta = utils.calculate_two_theta(hkl = hkls, reciprocal_lattice=lattice_structure.reciprocal_lattice, wavelength=wavelength)

    # Calculate distances from sample to detector
    r = detector.sample_detector_distance*np.tan(np.radians(two_theta))

    # Generate angles
    theta = np.linspace(0, 2*np.pi, 100)

    # Calculate y and z coordinates for a circle
    y = r * np.cos(theta)/(-detector.pixel_size[0])
    z = r * np.sin(theta)/(detector.pixel_size[1])

    # Function to distort circle based on detector parameters
    def distort_circle(y,z, detector):
        tilting_angle = np.radians(detector.tilting_angle)

        # Convert y and z coordinates to meters
        y = y*(-detector.pixel_size[0])
        z = z*(detector.pixel_size[1])

        # Calculate beam center in meters
        beam_center = (-detector.beam_center[0]*detector.pixel_size[0], detector.beam_center[1]*detector.pixel_size[1]) #In meters

        # Apply distortion to y and z coordinates
        Z = (z + beam_center[1])/(z*np.sin(tilting_angle)/detector.sample_detector_distance + np.cos(tilting_angle))
        Y = ((detector.sample_detector_distance - beam_center[1]*np.tan(tilting_angle))/(detector.sample_detector_distance + z*np.tan(tilting_angle)))*y + beam_center[0]
        return Y,Z
    
    # Apply distortion to circle coordinates
    Y,Z = distort_circle(y,z, detector)

    # Scale back y and z coordinates
    Y = Y/(-detector.pixel_size[0])
    Z = Z/(detector.pixel_size[1])

    # Plot distorted circle as guidelines
    plt.plot(Y, Z, "--",color = "black", linewidth = 2)
"""

