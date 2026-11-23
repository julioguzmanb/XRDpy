"""Matplotlib and Plotly visualizations for simulated diffraction results."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from .deferred_plotting import gui_plot
import matplotlib.lines as mlines
from matplotlib.patches import Patch
plt.ion()
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import mplcursors
    HAVE_MPLCURSORS = True
except Exception:
    HAVE_MPLCURSORS = False


def plot_3d_detector(lab_grid, title="Detector", ax=None, show_surface=True):
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
    if show_surface:
        ax.plot_surface(x, y, z, alpha=0.2, edgecolor='k')
    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

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
    ax=None,
    layout=True,
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
        layout (bool, optional): Apply tight layout to the containing figure.

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

    if layout and ax.figure is not None:
        plt.tight_layout()


def _plot_direct_beam(ax, detector, detector_transform=None):
    """Draw the actual beam-plane intersection, never a PONI approximation."""
    pixel = detector.direct_beam_pixel(detector_transform)
    if pixel is not None:
        artist, = ax.plot(*pixel, marker="x", color="red", markersize=10,
                          markeredgewidth=2, linestyle="None", zorder=20)
        artist.set_gid("direct-beam")


def plot_2d_detector_polycrystal(
    detector,
    pixel_positions,
    scat_dir_sign,
    hkls,
    hkl_to_color,
    title="2D Detector View",
    ax=None,
    detector_transform=None,
):
    """
    Plot a 2D representation of the detector for multiple reflections (polycrystal).

    Parameters
    ----------
    detector : Detector
        Detector dimensions, binning, and pixel sizes.
    pixel_positions : array-like, shape (N, 2)
        Horizontal/vertical ring sample coordinates.
    scat_dir_sign : array-like, shape (N,)
        Forward-ray sign used to reject intersections behind the detector.
    hkls : array-like, shape (N, 3)
        Miller-index label for every sampled point.
    hkl_to_color : dict
        Miller-index tuples mapped to Matplotlib colors.
    title : str
        Axes title.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is created when omitted.
    detector_transform : array-like, optional
        Transform used for these pixel positions, also used for the direct beam.

    The legend entries (one per hkl) are clickable: clicking an entry toggles
    visibility of that hkl's points.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    # Pixel coordinates must retain the detector's physical pixel proportions.
    # Detector pixel sizes already include binning.
    ax.set_aspect(detector.pxsize_v / detector.pxsize_h, adjustable="box")

    # Evaluate which positions are within detector bounds
    scattered_within_bounds = within_bounds(
        pixel_positions,
        scat_dir_sign.reshape(-1, 1).squeeze(),
        detector.num_pixels_h,
        detector.num_pixels_v
    )

    valid_positions = pixel_positions[scattered_within_bounds]
    valid_hkls = np.array(hkls)[scattered_within_bounds] if hkls is not None else None

    scatter_artists = []
    legend_labels = []

    if valid_positions.size > 0 and valid_hkls is not None:
        # unique hkls, preserving first appearance order
        unique_hkls, first_indices = np.unique(valid_hkls, axis=0, return_index=True)
        ordered_unique_hkls = unique_hkls[np.argsort(first_indices)]

        for hkl in ordered_unique_hkls:
            hkl_tuple = tuple(hkl)
            indices = (valid_hkls == hkl).all(axis=1)
            hkl_positions = valid_positions[indices]
            if hkl_positions.size > 0:
                color = hkl_to_color[hkl_tuple]
                sc = ax.scatter(
                    hkl_positions[:, 0],
                    hkl_positions[:, 1],
                    s=5,
                    color=color,
                )
                scatter_artists.append(sc)
                legend_labels.append(f"({hkl[0]},{hkl[1]},{hkl[2]})")

    ax.set_title(title)
    ax.set_xlabel("Horizontal Pixels")
    ax.set_ylabel("Vertical Pixels")
    ax.grid()
    _plot_direct_beam(ax, detector, detector_transform)
    ax.set_xlim(-0.5, detector.num_pixels_h - 0.5)
    ax.set_ylim(-0.5, detector.num_pixels_v - 0.5)

    # Build explicit legend (one entry per hkl) and make it clickable
    legend_handles = []
    for sc, label in zip(scatter_artists, legend_labels):
        color = sc.get_facecolors()[0]
        handle = mlines.Line2D(
            [], [], color=color, marker='o', linestyle='None',
            markersize=8, label=label
        )
        legend_handles.append(handle)

    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            fontsize=11,
            loc="upper left",
            title="Reflections",
            title_fontsize=13,
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            markerscale=1.5
        )

        # Map legend line → corresponding scatter artist
        handle_to_scatter = {}
        legend_lines = legend.get_lines()  # actual artists in the legend box

        for legline, sc in zip(legend_lines, scatter_artists):
            legline.set_picker(5)  # 5 px tolerance
            handle_to_scatter[legline] = sc

        def on_pick(event):
            legline = event.artist
            if legline not in handle_to_scatter:
                return

            sc = handle_to_scatter[legline]
            visible = not sc.get_visible()
            sc.set_visible(visible)

            # Dim legend entry when hidden
            legline.set_alpha(1.0 if visible else 0.2)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("pick_event", on_pick)

    if fig is not None:
        plt.tight_layout()
        plt.show()

    return ax


def _place_reflection_tooltip(tooltip, event, fig):
    """Place a figure tooltip beside the pointer without crossing figure edges."""
    extent = tooltip.get_window_extent(renderer=fig.canvas.get_renderer())
    width, height = extent.width, extent.height
    gap, margin = 12, 8
    x = event.x + gap
    if x + width + margin > fig.bbox.width:
        x = event.x - gap - width
    y = event.y + gap
    if y + height + margin > fig.bbox.height:
        y = event.y - gap - height
    x = min(max(x, margin), max(margin, fig.bbox.width - width - margin))
    y = min(max(y, margin), max(margin, fig.bbox.height - height - margin))
    tooltip.set_position((x / fig.bbox.width, y / fig.bbox.height))


def _attach_2d_reflection_hover(ax, positions, hkls, q_vectors, visible):
    """Show reciprocal-space details at detector spots within an eight-pixel radius."""
    from .reciprocal import _point_details

    fig = ax.figure
    if hasattr(ax, '_single_crystal_hover_cid'):
        fig.canvas.mpl_disconnect(ax._single_crystal_hover_cid)
        ax._single_crystal_hover_text.remove()
    tooltip = fig.text(0, 0, '', ha='left', va='bottom', zorder=10,
                       bbox=dict(boxstyle='round', facecolor='white', edgecolor='#555555', alpha=.96))
    tooltip.set_visible(False)

    def hover(event):
        found = None
        if event.inaxes is ax and visible.any():
            screen_positions = ax.transData.transform(positions[visible])
            distances = np.linalg.norm(screen_positions - [event.x, event.y], axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= 8:
                found = np.flatnonzero(visible)[nearest]
        if found is None:
            if tooltip.get_visible():
                tooltip.set_visible(False)
                fig.canvas.draw_idle()
            return
        pixel_h, pixel_v = positions[found]
        tooltip.set_text(
            _point_details(q_vectors[found], hkls[found]) +
            f'\nPixel (H, V)=({pixel_h:.2f}, {pixel_v:.2f})'
        )
        tooltip.set_visible(True)
        _place_reflection_tooltip(tooltip, event, fig)
        fig.canvas.draw_idle()

    ax._single_crystal_hover_cid = fig.canvas.mpl_connect('motion_notify_event', hover)
    ax._single_crystal_hover_text = tooltip


def _attach_3d_reflection_hover(ax, points):
    """Show Q details when the pointer is over a visible 3D Bragg hit marker."""
    from .reciprocal import _point_details

    fig = ax.figure
    tooltip = fig.text(0, 0, '', ha='left', va='bottom', zorder=10,
                       bbox=dict(boxstyle='round', facecolor='white', edgecolor='#555555', alpha=.96))
    tooltip.set_visible(False)

    def hover(event):
        found = None
        if event.inaxes is ax:
            for marker, q, hkl in reversed(points):
                if marker.get_visible() and marker.contains(event)[0]:
                    found = q, hkl
                    break
        if found is None:
            if tooltip.get_visible():
                tooltip.set_visible(False)
                fig.canvas.draw_idle()
            return
        tooltip.set_text(_point_details(*found))
        tooltip.set_visible(True)
        _place_reflection_tooltip(tooltip, event, fig)
        fig.canvas.draw_idle()

    ax._single_crystal_hover_cid = fig.canvas.mpl_connect('motion_notify_event', hover)
    ax._single_crystal_hover_text = tooltip


def plot_2d_detector_single_xstal(
    detector,
    pixel_positions,
    scat_dir_sign,
    hkls,
    hkl_to_color,
    title="2D Detector View",
    ax=None,
    intensities=None,
    detector_transform=None,
    intensity_mode=False,
    intensity_cmap="viridis",
    intensity_clim=None,
    noise_distribution="none",
    noise_level=0.0,
    noise_seed=None,
    intensity_image=None,
    q_vectors=None,
    show=True,
    layout=True,
    colorbar_bounds=None,
):
    """
    Plot a 2D representation of the detector, including scattered rays and 
    the direct beam if applicable, with optional intensities controlling transparency.

    Parameters
    ----------
    detector : Detector
        Detector dimensions, binning, and pixel sizes.
    pixel_positions : array-like, shape (N, 2)
        Horizontal/vertical reflection coordinates.
    scat_dir_sign : array-like, shape (N,)
        Forward-ray sign for each coordinate.
    hkls : array-like, shape (N, 3)
        Miller-index label for every coordinate.
    q_vectors : array-like, shape (N, 3), optional
        Reciprocal-space vectors in inverse angstrom, aligned with the pixels.
    hkl_to_color : dict
        Miller-index tuples mapped to Matplotlib colors.
    title : str
        Axes title.
    ax : matplotlib.axes.Axes, optional
        Existing axes.
    intensities : array-like, optional
        Per-reflection intensities used to control marker transparency.
    detector_transform : array-like, optional
        Transform used for these pixel positions, also used for the direct beam.

    Legend entries (one per unique hkl) are clickable: clicking on a label
    toggles visibility of that hkl's points.
    """
    from .detector_view import display_title

    title = display_title(title)

    # Determine valid scattered positions
    scattered_within_bounds = within_bounds(
        pixel_positions,
        scat_dir_sign,
        detector.num_pixels_h,
        detector.num_pixels_v
    )

    valid_positions = pixel_positions[scattered_within_bounds]
    valid_intensities = intensities[scattered_within_bounds] if intensities is not None else None
    valid_q = None
    if q_vectors is not None:
        q_vectors = np.asarray(q_vectors, dtype=float).reshape(-1, 3)
        if len(q_vectors) != len(pixel_positions):
            raise ValueError('Q vectors must match the detector pixel positions.')
        valid_q = q_vectors[scattered_within_bounds]
    hover_visible = np.ones(len(valid_positions), dtype=bool)

    if len(valid_positions) == 0:
        print("No hkls in the detector")

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 8))
    else:
        fig = ax.figure

    if intensity_mode:
        from .intensity_view import (spots_detector_image, intensity_mappable,
                                     validate_display)
        validate_display(intensity_cmap, noise_distribution, noise_level)
        image = (intensity_image if intensity_image is not None else
                 spots_detector_image(detector, pixel_positions, intensities,
                                      scattered_within_bounds,
                                      noise_distribution=noise_distribution,
                                      noise_level=noise_level, noise_seed=noise_seed))
        mappable = intensity_mappable(intensity_cmap, image.max(initial=0), intensity_clim)
        ax.imshow(image, origin="lower", interpolation="nearest", cmap=mappable.cmap,
                  norm=mappable.norm, extent=(-0.5, detector.num_pixels_h - 0.5,
                                              -0.5, detector.num_pixels_v - 0.5))
        for position, hkl, hit in zip(pixel_positions, hkls, scattered_within_bounds):
            if hit:
                ax.annotate(f"({hkl[0]},{hkl[1]},{hkl[2]})", position,
                            xytext=(5, 5), textcoords="offset points", color="white",
                            fontsize=8, bbox={"facecolor": "black", "alpha": 0.65,
                                              "edgecolor": "none", "pad": 1})
        ax.set(xlim=(-0.5, detector.num_pixels_h - 0.5),
               ylim=(-0.5, detector.num_pixels_v - 0.5),
               xlabel="Horizontal Pixels", ylabel="Vertical Pixels", title=title)
        ax.set_aspect(detector.pxsize_v / detector.pxsize_h, adjustable="box")
        colorbar_options = ({'cax': fig.add_axes(colorbar_bounds)}
                            if colorbar_bounds is not None else {'ax': ax})
        ax._single_crystal_colorbar = fig.colorbar(
            mappable, label="Relative intensity [a.u.]",
            orientation='horizontal' if colorbar_bounds is not None else 'vertical',
            **colorbar_options)
        if valid_q is not None and len(valid_positions):
            _attach_2d_reflection_hover(ax, valid_positions, np.asarray(hkls)[scattered_within_bounds],
                                        valid_q, hover_visible)
        if layout:
            fig.tight_layout()
        if show:
            plt.show()
        return ax

    ax.set_aspect(detector.pxsize_v / detector.pxsize_h, adjustable="box")

    # Match valid HKLs
    if hkls is not None and len(hkls) > 0:
        valid_hkls = [hkls[i] for i, valid in enumerate(scattered_within_bounds) if valid]
        valid_hkls = np.array(valid_hkls)
    else:
        valid_hkls = None

    scatter_artists = []
    scatter_masks = {}
    legend_labels = []

    # Group points by unique hkl so each hkl has a single scatter and legend entry
    if valid_positions.size > 0 and valid_hkls is not None:
        unique_hkls, first_indices = np.unique(valid_hkls, axis=0, return_index=True)
        ordered_unique_hkls = unique_hkls[np.argsort(first_indices)]

        for hkl in ordered_unique_hkls:
            mask = (valid_hkls == hkl).all(axis=1)
            hkl_positions = valid_positions[mask]
            if hkl_positions.size == 0:
                continue

            color = hkl_to_color[tuple(hkl)]

            if valid_intensities is not None:
                hkl_intensities = valid_intensities[mask]
                base_rgba = np.array(color)
                # One RGBA per point, alpha scaled by intensity
                colors_arr = np.tile(base_rgba, (hkl_positions.shape[0], 1))
                colors_arr[:, 3] = colors_arr[:, 3] * hkl_intensities
                sc = ax.scatter(
                    hkl_positions[:, 0],
                    hkl_positions[:, 1],
                    s=15,
                    color=colors_arr
                )
            else:
                sc = ax.scatter(
                    hkl_positions[:, 0],
                    hkl_positions[:, 1],
                    s=15,
                    color=color
                )

            scatter_artists.append(sc)
            scatter_masks[sc] = mask
            legend_labels.append(f"({hkl[0]},{hkl[1]},{hkl[2]})")

    ax.set_title(title)
    ax.set_xlabel("Horizontal Pixels")
    ax.set_ylabel("Vertical Pixels")

    ax.grid()
    _plot_direct_beam(ax, detector, detector_transform)
    ax.set_xlim(-0.5, detector.num_pixels_h - 0.5)
    ax.set_ylim(-0.5, detector.num_pixels_v - 0.5)

    # Build clickable legend
    legend_handles = []
    for sc, label in zip(scatter_artists, legend_labels):
        # Use the first facecolor of the scatter as legend color
        col = sc.get_facecolors()[0]
        handle = mlines.Line2D(
            [], [], color=col, marker='o', linestyle='None',
            markersize=10, label=label
        )
        legend_handles.append(handle)

    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            fontsize=11,
            loc="upper left",
            title="Reflections",
            title_fontsize=13,
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            markerscale=1.5
        )

        # Map legend line → scatter artist
        handle_to_scatter = {}
        legend_lines = legend.get_lines()  # actual visible artists in legend

        for legline, sc in zip(legend_lines, scatter_artists):
            legline.set_picker(5)  # 5 px tolerance for easier clicking
            handle_to_scatter[legline] = sc

        def on_pick(event):
            legline = event.artist
            if legline not in handle_to_scatter:
                return

            sc = handle_to_scatter[legline]
            visible = not sc.get_visible()
            sc.set_visible(visible)
            hover_visible[scatter_masks[sc]] = visible

            # Dim legend entry when hidden
            legline.set_alpha(1.0 if visible else 0.2)

            fig.canvas.draw_idle()

        ax._single_crystal_pick_cid = fig.canvas.mpl_connect('pick_event', on_pick)

    if valid_q is not None and len(valid_positions):
        _attach_2d_reflection_hover(ax, valid_positions, valid_hkls, valid_q, hover_visible)

    if ax.figure is not None:
        if layout:
            plt.tight_layout()
        if show:
            plt.show()

    return ax


def plot_diffraction_cones(cones, hkls_names, ax=None):
    """
    Plot proper diffraction cones in 3D with colorized surfaces.

    Parameters
    ----------
    cones : sequence of tuple
        ``(x, y, z)`` surface arrays for each diffraction cone.
    hkls_names : array-like, shape (N, 3)
        Miller-index label for each cone.
    ax : matplotlib 3D axes, optional
        Existing axes; a new 3D figure is created when omitted.

    Legend entries (one per cone / hkl) are clickable: clicking a legend
    marker toggles visibility of that cone surface.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    indices = np.arange(len(cones))
    colors = colorize(indices, vmin=0, vmax=len(cones) - 1)

    surfaces = []
    legend_handles = []

    # Plot each cone surface
    alphas = np.linspace(0.5, 0.1, len(cones))
    for (x, y, z), color, alpha, hkl_name in zip(cones, colors, alphas, hkls_names):
        # Build per-vertex RGBA colors for the surface
        facecolors = np.empty(x.shape + (4,))
        facecolors[:, :] = color
        surf = ax.plot_surface(
            x, y, z,
            alpha=alpha,
            facecolors=facecolors,
            edgecolor="none"
        )
        surfaces.append(surf)

        # Legend label as "(h,k,l)"
        label_str = str(list(hkl_name)).replace("[", "(").replace("]", ")").replace(" ", "")

        # Use a Line2D marker in the legend (easier to pick than Patch)
        legend_handle = mlines.Line2D(
            [], [], color=color, marker='s', linestyle='None',
            markersize=10, label=label_str
        )
        legend_handles.append(legend_handle)

    # Create legend
    if legend_handles:
        legend = ax.legend(
            handles=legend_handles,
            fontsize=11,
            loc="upper left",
            title="Reflections",
            title_fontsize=13,
            bbox_to_anchor=(1.11, 1),
            borderaxespad=0.
        )

        # Map legend marker → corresponding cone surface
        handle_to_surface = {}

        # Because we used Line2D for legend entries, we can safely use get_lines()
        legend_lines = list(legend.get_lines())

        for legline, surf in zip(legend_lines, surfaces):
            legline.set_picker(5)  # 5 px tolerance
            handle_to_surface[legline] = surf

        def on_pick(event):
            artist = event.artist
            if artist not in handle_to_surface:
                return

            surf = handle_to_surface[artist]
            visible = not surf.get_visible()
            surf.set_visible(visible)

            # Dim legend marker when hidden
            artist.set_alpha(1.0 if visible else 0.2)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("pick_event", on_pick)

    plt.show()
    return ax


@gui_plot
def plot_1d_pattern(x, I, x_axis="two_theta", title="Simulated 1D pattern", ax=None, show=True):
    """Plot a simulated 1D intensity profile and return its Matplotlib axes.

    Parameters
    ----------
    x, I : array-like
        Profile coordinate and intensity arrays of equal length.
    x_axis : {"q", "two_theta", "d"}
        Select inverse-angstrom, degree, or angstrom axis labeling.
    title : str
        Axes title.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is created when omitted.
    show : bool
        Call ``plt.show()`` after formatting.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    labels = {"q": "q [Å$^{-1}$]", "two_theta": r"2$\theta$ [°]", "d": "d [Å]"}
    if x_axis not in labels:
        raise ValueError("x_axis must be 'q', 'two_theta', or 'd'.")

    ax.plot(x, I, linewidth=1.5)
    ax.grid()
    ax.set_ylabel("Intensity [a.u.]")
    ax.set_xlabel(labels[x_axis])
    ax.xaxis.set_inverted(x_axis == "d")
    ax.set_title(title)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_rotation_mapping(valid_orientations,
                          title="roty and rotz for (h k l) in Bragg condition",
                          s=20):
    """
    Plot successful (roty, rotz) combinations that satisfy the Bragg condition.

    Parameters:
        valid_orientations (dict): Dictionary mapping hkl -> list of (roty, rotz).
        title (str): Title of the plot.
        s (float): Marker size for the scatter points.

    The legend entries (one per hkl) are clickable: clicking an entry toggles
    visibility of that hkl's points.
    """

    print
    colors = colorize(np.linspace(0, 1, len(valid_orientations.keys())))

    fig, ax = plt.subplots(figsize=(7, 7))

    scatter_artists = []
    legend_labels = []

    # One scatter artist per hkl, with its own color.
    for (hkl, color) in zip(valid_orientations.keys(), colors):
        if valid_orientations[f"{hkl}"]:
            roty, rotz = zip(*valid_orientations[f"{hkl}"])
        else:
            roty, rotz = [], []

        sc = ax.scatter(roty, rotz, s=s, color=color, alpha=0.6)
        scatter_artists.append(sc)

        label = "(" + ",".join(hkl.strip("[]").split()) + ")"
        legend_labels.append(label)

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("roty [°]", fontsize=13)
    ax.set_ylabel("rotz [°]", fontsize=13)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_aspect("equal", adjustable="box")
    ax.grid()

    # Create a custom legend: one entry per hkl
    legend_handles = []
    for sc, label, color in zip(scatter_artists, legend_labels, colors):
        handle = mlines.Line2D(
            [], [], color=color, marker='o', linestyle='None',
            markersize=10, label=label
        )
        legend_handles.append(handle)

    legend = ax.legend(
        handles=legend_handles,
        title="(h k l)",
        title_fontsize=13,
        fontsize=11,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0.
    )

    # --- Interactivity: link *legend* artists to scatter artists ---
    handle_to_scatter = {}

    # These are the actual Line2D objects drawn inside the legend box.
    legend_lines = legend.get_lines()

    # In this setup, legend_lines and scatter_artists should be aligned 1:1
    for legline, sc in zip(legend_lines, scatter_artists):
        legline.set_picker(5)      # 5 px tolerance makes clicking easier
        handle_to_scatter[legline] = sc

    def on_pick(event):
        legline = event.artist
        if legline not in handle_to_scatter:
            return

        sc = handle_to_scatter[legline]
        visible = not sc.get_visible()
        sc.set_visible(visible)

        # Dim legend entry when hidden
        legline.set_alpha(1.0 if visible else 0.2)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.tight_layout()
    plt.show()


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


def plot_reciprocal(Q_hkls, hkls, wavelength, E_bandwidth, **options):
    """Plot Q-space reflections and Ewald spheres; Plotly is the default renderer.

    Q is in inverse angstrom, wavelength in metres, and E_bandwidth is the full
    energy bandwidth in percent. Pass backend='matplotlib' for a native figure,
    show=False to return the figure without opening it, or html_path for export.
    """
    from .reciprocal import plot_reciprocal as render
    return render(Q_hkls, hkls, wavelength, E_bandwidth, **options)


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


def plot_parameter_mapping(valid_points, param1_name, param2_name, *, display_mode='points', grid_steps=None, existing_axes=None, show=True):
    """
    Plot a 2D mapping of param1 vs param2 for each hkl, with clickable legend
    to toggle visibility of each reflection.

    valid_points: dict
        { "[h k l]" : [(p1, p2), (p1, p2), ...], ... }
    param1_name, param2_name: str
        Names of the scanned parameters, used for axis labels.
    """
    if not valid_points:
        return
    if display_mode not in ('points', 'outlines', 'bands'):
        raise ValueError('Mapping display must be points, outlines, or bands.')

    if existing_axes is None:
        fig, ax = plt.subplots(figsize=(8.5, 7))
    else:
        ax, fig = existing_axes, existing_axes.figure
        for callback in getattr(ax, '_mapping_callbacks', []):
            fig.canvas.mpl_disconnect(callback)
        ax.clear()

    ax.set_title(f"{param1_name} vs {param2_name} for Bragg condition",fontsize = 15)
    # Build color list
    keys = list(valid_points.keys())

    good_keys = []
    for key in keys:
        pts = np.array(valid_points[key])
        if pts.size != 0:
            good_keys.append(key)
            continue

    scatters = []
    legend_handles = []

    colors = plt.cm.jet(np.linspace(0, 1, len(good_keys)))

    for key, color in zip(good_keys, colors):
        pts = np.array(valid_points[key])
        if pts.size == 0:
            continue

        x = pts[:, 0]
        y = pts[:, 1]
        if display_mode == 'points':
            sc = ax.scatter(x, y, s=20, color=color, alpha=0.8)
        else:
            from matplotlib.collections import LineCollection, PolyCollection
            from .mapping_display import band_vertices
            try:
                vertices = band_vertices(pts, grid_steps, filled=display_mode == 'bands')
                if display_mode == 'outlines':
                    sc = LineCollection(vertices, colors=[color], linewidths=1.2)
                else:
                    sc = PolyCollection(vertices, facecolors=[color], edgecolors='none',
                                        antialiaseds=False, alpha=.45)
                ax.add_collection(sc)
                ax.autoscale_view()
            except Exception:
                plt.close(fig)
                raise
        scatters.append(sc)

        # Legend marker
        handle = plt.Line2D(
            [], [], color=color, marker='o' if display_mode == 'points' else None,
            linestyle='None' if display_mode == 'points' else '-',
            markersize=10, 
            label=key.replace("[", "(").replace("]", ")").replace(".",",").replace(" ","").replace(",)",")")
        )
        legend_handles.append(handle)

    if param1_name in ["rotx", "roty", "rotz"]:
        param1_name += " [°]"
    if param2_name in ["rotx", "roty", "rotz"]:
        param2_name += " [°]"
    if param1_name == "energy":
        param1_name += " [eV]"
    if param2_name == "energy":
        param2_name += " [eV]"

    ax.set_xlabel(param1_name, fontsize=13)
    ax.set_ylabel(param2_name, fontsize=13)
    ax.set_title(f"{param1_name} vs {param2_name} for Bragg condition")

    if legend_handles:
        from .mapping_display import attach_mapping_legend
        attach_mapping_legend(ax, legend_handles, scatters)
    else:
        if existing_axes is None:
            fig.tight_layout()
        else:
            bottom = getattr(ax, '_mapping_bottom', .75)/fig.get_figheight()
            ax.set_position([.12, bottom, .65, max(.1, .88-bottom)])

    ax.grid()
    if show:
        plt.show()
    return ax

def _fixed_energy_solution_axes(solutions):
    """
    Return plotting axis names and angle matrix for fixed-energy inverse-targeting solutions.

    Returns
    -------
    axis_names : tuple[str, ...]
    axis_values : np.ndarray, shape (N, M)
        One row per accepted solution, one column per plotted motor angle.
    """
    solution_mode = getattr(solutions, "solution_mode", "euler")

    if solution_mode == "motor_chain":
        motor_angles = getattr(solutions, "motor_angles_deg", None)
        motor_names = tuple(getattr(solutions, "motor_names", ()))

        if motor_angles is None:
            raise ValueError(
                "solutions.solution_mode == 'motor_chain' but solutions.motor_angles_deg is missing."
            )

        motor_angles = np.asarray(motor_angles, dtype=float)
        if motor_angles.ndim != 2:
            raise ValueError(
                f"solutions.motor_angles_deg must have shape (N, M). Got {motor_angles.shape}."
            )

        if len(motor_names) != motor_angles.shape[1]:
            motor_names = tuple(f"motor_{i+1}" for i in range(motor_angles.shape[1]))

        return motor_names, motor_angles

    # Legacy Euler fallback
    axis_names = ("rotx", "roty", "rotz")
    axis_values = np.column_stack([
        np.asarray(solutions.rotx_deg, dtype=float),
        np.asarray(solutions.roty_deg, dtype=float),
        np.asarray(solutions.rotz_deg, dtype=float),
    ])
    return axis_names, axis_values


def _fixed_energy_hover_text(solutions, idx, axis_names, axis_values):
    """
    Build hover text for fixed-energy inverse-targeting plots.
    """
    dh, dv = np.asarray(solutions.predicted_pixels[idx], dtype=float)
    lines = [
        f"eta  = {float(solutions.eta_deg[idx]):.3f} °",
        f"phi  = {float(solutions.phi_deg[idx]):.3f} °",
    ]

    for name, val in zip(axis_names, np.asarray(axis_values[idx], dtype=float)):
        lines.append(f"{name} = {val:.4f} °")

    lines.extend([
        f"pixel = ({dh:.2f}, {dv:.2f})",
        f"miss  = {float(solutions.pixel_error_px[idx]):.3f} px",
    ])

    return "\n".join(lines)

def _fixed_energy_padded_limits(vals, frac=0.03, min_pad=1.0):
    """Return finite plotting bounds padded around a one-dimensional dataset."""
    vals = np.asarray(vals, dtype=float)
    lo = np.nanmin(vals)
    hi = np.nanmax(vals)
    span = max(hi - lo, 1e-9)
    pad = max(frac * span, min_pad)
    return lo - pad, hi + pad, span + 2.0 * pad


@gui_plot
def plot_fixed_energy_detector_hits(
    all_pixels,
    on_detector_mask,
    accepted_mask,
    target_pixel,
    pixel_tolerance_px,
    title="Fixed-energy reachable pixels for selected hkl",
):
    """Plot reachable and accepted fixed-energy points around a target pixel.

    Parameters
    ----------
    all_pixels : array-like, shape (N, 2)
        Detector intersections for all sampled q-cone points.
    on_detector_mask : array-like of bool, shape (N,)
        Points lying within detector pixel bounds.
    accepted_mask : array-like of bool, shape (N,)
        Points also lying within the target tolerance.
    target_pixel : tuple of float
        Requested horizontal/vertical pixel.
    pixel_tolerance_px : float
        Radius of the acceptance circle in pixels.
    title : str
        Axes title.

    Returns
    -------
    tuple
        Matplotlib ``(figure, axes)`` without calling ``show``.
    """
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    all_pixels = np.asarray(all_pixels, dtype=float)
    on_detector_mask = np.asarray(on_detector_mask, dtype=bool)
    accepted_mask = np.asarray(accepted_mask, dtype=bool)

    ax.scatter(
        all_pixels[on_detector_mask, 0],
        all_pixels[on_detector_mask, 1],
        s=6,
        alpha=0.35,
        label="fixed-energy cone on detector",
    )

    if np.any(accepted_mask):
        ax.scatter(
            all_pixels[accepted_mask, 0],
            all_pixels[accepted_mask, 1],
            s=12,
            label=f"within {pixel_tolerance_px:.1f} px",
        )

    ax.scatter(
        [target_pixel[0]],
        [target_pixel[1]],
        marker="x",
        s=100,
        label="target pixel",
    )

    circ = plt.Circle(target_pixel, pixel_tolerance_px, fill=False, linestyle="--")
    ax.add_patch(circ)

    ax.set_xlabel("detector horizontal pixel")
    ax.set_ylabel("detector vertical pixel")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid()
    ax.legend()
    return fig, ax

@gui_plot
def plot_fixed_energy_motor_projections(
    solutions,
    phi_colormap="hsv",
    scatter_size=8,
    title="Fixed-energy accepted orientations in sample motor space",
):
    """Plot pairwise projections of accepted Euler or motor-chain solutions.

    Solution points are colored by the free rotation angle ``phi``. Two-axis
    chains produce one plane; three-axis solutions produce the three pairwise
    planes; larger chains produce a lower-triangular projection matrix. Hover
    annotations are enabled when ``mplcursors`` is installed.

    Parameters
    ----------
    solutions : FixedEnergyAcceptedSolutions
        Accepted Euler or motor-chain orientation family.
    phi_colormap : str
        Matplotlib colormap for the free rotation angle.
    scatter_size : float
        Marker area passed to ``Axes.scatter``.
    title : str
        Figure title or super-title.
    """
    axis_names, axis_values = _fixed_energy_solution_axes(solutions)
    axis_values = np.asarray(axis_values, dtype=float)

    if axis_values.ndim != 2 or axis_values.shape[0] == 0:
        raise ValueError("No fixed-energy motor solutions are available to plot.")

    n_axes = axis_values.shape[1]
    if n_axes < 2:
        raise ValueError(
            "At least two motor axes are required to plot motor-space projections."
        )

    color_values = np.asarray(solutions.phi_deg, dtype=float)

    # ------------------------------------------------------------------
    # Case 1: exactly 2 axes -> one 2D plot
    # ------------------------------------------------------------------
    if n_axes == 2:
        x = axis_values[:, 0]
        y = axis_values[:, 1]

        x_min, x_max, x_span = _fixed_energy_padded_limits(x)
        y_min, y_max, y_span = _fixed_energy_padded_limits(y)

        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        sc = ax.scatter(
            x,
            y,
            c=color_values,
            s=scatter_size,
            cmap=phi_colormap,
            vmin=0.0,
            vmax=360.0,
        )

        ax.set_xlabel(f"{axis_names[0]} [°]")
        ax.set_ylabel(f"{axis_names[1]} [°]")
        ax.set_title(title)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(y_span / x_span)
        ax.grid()

        fig.colorbar(sc, ax=ax, label="phi [°]")

        if HAVE_MPLCURSORS:
            cursor = mplcursors.cursor([sc], hover=True)

            @cursor.connect("add")
            def on_add(sel):
                i = sel.index
                sel.annotation.set_text(
                    _fixed_energy_hover_text(solutions, i, axis_names, axis_values)
                )

        return fig, (ax,)

    # ------------------------------------------------------------------
    # Case 2: 3 or more axes -> show first 3 as pairwise projections
    # ------------------------------------------------------------------
    shown_axis_names = axis_names[:3]
    shown_axis_values = axis_values[:, :3]

    x = shown_axis_values[:, 0]
    y = shown_axis_values[:, 1]
    z = shown_axis_values[:, 2]

    x_min, x_max, x_span = _fixed_energy_padded_limits(x)
    y_min, y_max, y_span = _fixed_energy_padded_limits(y)
    z_min, z_max, z_span = _fixed_energy_padded_limits(z)

    span_x0 = x_span
    span_y0 = y_span
    span_x1 = x_span
    span_y1 = z_span
    span_x2 = y_span
    span_y2 = z_span

    max_span_x = max(span_x0, span_x1, span_x2)
    width_ratios = [
        span_x0 / max_span_x,
        span_x1 / max_span_x,
        span_x2 / max_span_x,
        0.08,
    ]

    fig = plt.figure(figsize=(11, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=width_ratios, wspace=0.0)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])

    scatter_kwargs = dict(
        c=color_values,
        s=scatter_size,
        cmap=phi_colormap,
        vmin=0.0,
        vmax=360.0,
    )

    sc0 = ax0.scatter(x, y, **scatter_kwargs)
    sc1 = ax1.scatter(x, z, **scatter_kwargs)
    sc2 = ax2.scatter(y, z, **scatter_kwargs)

    ax0.set_xlabel(f"{shown_axis_names[0]} [°]")
    ax0.set_ylabel(f"{shown_axis_names[1]} [°]")
    ax0.set_title(f"{shown_axis_names[0]} vs {shown_axis_names[1]}")

    ax1.set_xlabel(f"{shown_axis_names[0]} [°]")
    ax1.set_ylabel(f"{shown_axis_names[2]} [°]")
    ax1.set_title(f"{shown_axis_names[0]} vs {shown_axis_names[2]}")

    ax2.set_xlabel(f"{shown_axis_names[1]} [°]")
    ax2.set_ylabel(f"{shown_axis_names[2]} [°]")
    ax2.set_title(f"{shown_axis_names[1]} vs {shown_axis_names[2]}")

    ax0.set_xlim(x_min, x_max)
    ax0.set_ylim(y_min, y_max)

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(z_min, z_max)

    ax2.set_xlim(y_min, y_max)
    ax2.set_ylim(z_min, z_max)

    ax0.set_aspect("equal", adjustable="box")
    ax1.set_aspect("equal", adjustable="box")
    ax2.set_aspect("equal", adjustable="box")

    ax0.set_box_aspect(span_y0 / span_x0)
    ax1.set_box_aspect(span_y1 / span_x1)
    ax2.set_box_aspect(span_y2 / span_x2)

    for ax in (ax0, ax1, ax2):
        ax.grid()

    fig.colorbar(sc0, cax=cax, label="phi [°]")

    if n_axes > 3:
        shown_text = ", ".join(shown_axis_names)
        fig.suptitle(f"{title}\nShowing first 3 motors: {shown_text}", fontsize=14)
    else:
        fig.suptitle(title, fontsize=14)

    if HAVE_MPLCURSORS:
        cursor = mplcursors.cursor([sc0, sc1, sc2], hover=True)

        @cursor.connect("add")
        def on_add(sel):
            i = sel.index
            sel.annotation.set_text(
                _fixed_energy_hover_text(solutions, i, axis_names, axis_values)
            )

    return fig, (ax0, ax1, ax2)


@gui_plot
def plot_fixed_energy_motor_family_3d(
    solutions,
    phi_colormap="hsv",
    scatter_size=6,
    title="Fixed-energy accepted orientations in 3D motor space",
):
    """Plot the accepted orientation family in its first three motor axes.

    Two-motor solutions fall back to a 2D scatter plot. The returned figure is
    not shown automatically.

    Parameters
    ----------
    solutions : FixedEnergyAcceptedSolutions
        Accepted Euler or motor-chain orientation family.
    phi_colormap : str
        Matplotlib colormap for the free rotation angle.
    scatter_size : float
        Marker area passed to the scatter artist.
    title : str
        Plot title.
    """
    axis_names, axis_values = _fixed_energy_solution_axes(solutions)
    axis_values = np.asarray(axis_values, dtype=float)

    if axis_values.ndim != 2 or axis_values.shape[0] == 0:
        raise ValueError("No fixed-energy motor solutions are available to plot.")

    n_axes = axis_values.shape[1]
    color_values = np.asarray(solutions.phi_deg, dtype=float)

    # Fallback for 2-motor chains: still show something instead of failing
    if n_axes == 2:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        pts = ax.scatter(
            axis_values[:, 0],
            axis_values[:, 1],
            c=color_values,
            s=scatter_size,
            cmap=phi_colormap,
            vmin=0.0,
            vmax=360.0,
        )
        ax.set_xlabel(f"{axis_names[0]} [°]")
        ax.set_ylabel(f"{axis_names[1]} [°]")
        ax.set_title(f"{title}\nOnly 2 motor axes available")
        ax.grid()
        fig.colorbar(pts, ax=ax, label="phi [°]")

        if HAVE_MPLCURSORS:
            cursor = mplcursors.cursor([pts], hover=True)

            @cursor.connect("add")
            def on_add(sel):
                i = sel.index
                sel.annotation.set_text(
                    _fixed_energy_hover_text(solutions, i, axis_names, axis_values)
                )

        return fig, ax

    shown_axis_names = axis_names[:3]
    shown_axis_values = axis_values[:, :3]

    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    pts = ax.scatter(
        shown_axis_values[:, 0],
        shown_axis_values[:, 1],
        shown_axis_values[:, 2],
        c=color_values,
        s=scatter_size,
        cmap=phi_colormap,
        vmin=0.0,
        vmax=360.0,
    )

    ax.set_xlabel(f"{shown_axis_names[0]} [°]")
    ax.set_ylabel(f"{shown_axis_names[1]} [°]")
    ax.set_zlabel(f"{shown_axis_names[2]} [°]")

    if n_axes > 3:
        shown_text = ", ".join(shown_axis_names)
        ax.set_title(f"{title}\nShowing first 3 motors: {shown_text}")
    else:
        ax.set_title(title)

    fig.colorbar(pts, ax=ax, label="phi [°]")

    if HAVE_MPLCURSORS:
        cursor = mplcursors.cursor([pts], hover=True)

        @cursor.connect("add")
        def on_add(sel):
            i = sel.index
            sel.annotation.set_text(
                _fixed_energy_hover_text(solutions, i, axis_names, axis_values)
            )

    return fig, ax


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
