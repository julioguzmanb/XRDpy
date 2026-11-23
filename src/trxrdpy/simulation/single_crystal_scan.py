"""Interactive sampled configurations for single-crystal detector views."""
from __future__ import annotations

import numpy as np

from .deferred_plotting import gui_plot
from .reciprocal_scan import parameter_label, scan_axes
from . import utils
from .slider_navigation import add_step_buttons, two_column_bounds


def prepare_axes(ranges, *, sample_geometry=None, detector_geometry=None,
                 sample_motor_chain=None, detector_motor_chain=None,
                 sample_transform=None, detector_transform=None):
    """Validate detector-view ranges, taking direction from start and stop."""
    if not isinstance(ranges, dict):
        return scan_axes(ranges)
    directed = {}
    for name, interval in ranges.items():
        if len(interval) != 3:
            directed[name] = interval
            continue
        start, stop, step = map(float, interval)
        direction = -1 if stop < start else 1
        directed[name] = (start, stop, direction * abs(step))
    axes = scan_axes(directed)
    allowed = {'energy', 'e_bandwidth'}
    if sample_geometry is not None:
        allowed.update('sample:' + name for name in sample_geometry.sample.motor_names)
    elif sample_motor_chain is not None:
        allowed.update('sample:' + name for name in sample_motor_chain.motor_names)
    elif sample_transform is None:
        allowed.update('sam_rot' + axis for axis in 'xyz')
    if detector_geometry is not None:
        allowed.update('detector:' + name for name in detector_geometry.detector.motor_names)
    elif detector_motor_chain is not None:
        allowed.update('detector:' + name for name in detector_motor_chain.motor_names)
    elif detector_transform is None:
        allowed.update('det_rot' + axis for axis in 'xyz')
    unknown = set(axes) - allowed
    if unknown:
        raise ValueError('Unsupported single-crystal scan parameters: ' + ', '.join(sorted(unknown)))
    return axes


def configuration_updater(exp, *, qmax, sample_geometry=None,
                          detector_geometry=None, sample_motor_chain=None,
                          detector_motor_chain=None, sample_angles=None,
                          detector_angles=None, sam_angles=None,
                          intensity_settings=None):
    """Return a calculator that reuses the reflection set for slider changes."""
    from .single_crystal import (_intensity_plot_options, _set_sample_pose_display,
                                 _set_detector_pose_display)

    sample_angles = dict(sample_angles or {})
    detector_angles = dict(detector_angles or {})
    sam_angles = dict(zip('xyz', sam_angles or (0, 0, 0)))
    det_angles = {axis: getattr(exp.detector, 'rot' + axis) for axis in 'xyz'}
    initial_energy, initial_bandwidth = exp.energy, exp.e_bandwidth
    settings = dict(intensity_settings or {})

    def update(config):
        energy = float(config.get('energy', initial_energy))
        bandwidth = float(config.get('e_bandwidth', initial_bandwidth))
        exp.energy = energy
        exp.e_bandwidth = bandwidth
        exp.wavelength = utils.energy_to_wavelength(energy)

        if sample_geometry is not None or sample_motor_chain is not None:
            angles = {**sample_angles, **{name.split(':', 1)[1]: float(value)
                                          for name, value in config.items()
                                          if name.startswith('sample:')}}
            if sample_geometry is not None:
                exp.lattice.apply_diffractometer(sample_geometry, angles=angles)
            else:
                exp.lattice.apply_motor_chain(sample_motor_chain, angles=angles)
            _set_sample_pose_display(
                exp, sample_geometry=sample_geometry,
                sample_motor_chain=sample_motor_chain, sample_angles=angles)
        elif any(name.startswith('sam_rot') for name in config):
            angles = {axis: float(config.get('sam_rot' + axis, sam_angles[axis]))
                      for axis in 'xyz'}
            exp.lattice.apply_rotation(**{'rot' + axis: value for axis, value in angles.items()})
            _set_sample_pose_display(
                exp, sam_rotx=angles['x'], sam_roty=angles['y'],
                sam_rotz=angles['z'],
                sam_rotation_order=exp.lattice.rotation_order)

        if detector_geometry is not None or detector_motor_chain is not None:
            angles = {**detector_angles, **{name.split(':', 1)[1]: float(value)
                                            for name, value in config.items()
                                            if name.startswith('detector:')}}
            if detector_geometry is not None:
                exp.detector.set_diffractometer(detector_geometry, angles=angles)
            else:
                exp.detector.set_motor_chain(detector_motor_chain, angles=angles)
            _set_detector_pose_display(
                exp, detector_geometry=detector_geometry,
                detector_motor_chain=detector_motor_chain, detector_angles=angles)
        elif any(name.startswith('det_rot') for name in config):
            angles = {axis: float(config.get('det_rot' + axis, det_angles[axis]))
                      for axis in 'xyz'}
            exp.detector.set_rotation_angles(**{'rot' + axis: value for axis, value in angles.items()})
            _set_detector_pose_display(exp)

        exp.lattice.check_Bragg_condition(exp.wavelength, bandwidth)
        exp.calculate_diffraction_direction(qmax)
        exp.calculate_pixel_positions()
        return _intensity_plot_options(
            exp, settings.get('consider_intensities', False),
            settings.get('cif_file_path'), settings.get('intensity_cmap', 'viridis'),
            settings.get('intensity_clim'), settings.get('polarization_factor', 1.0),
            settings.get('polarization_axis_offset_deg', 0.0),
            settings.get('noise_distribution', 'none'),
            settings.get('noise_level', 0.0), settings.get('noise_seed'))

    return update


def _attach_controls(fig, axes, height, redraw):
    """Create sampled sliders and value boxes shared by both detector views."""
    from matplotlib.widgets import Slider, TextBox

    rows = int(np.ceil(len(axes) / 2))
    sliders, boxes, buttons = [], [], []
    for index, (name, samples) in enumerate(axes.items()):
        x, track_width, box_x, box_width, previous_x, next_x, button_width = (
            two_column_bounds(fig, index % 2, track_inches=3.08, box_inches=1.10))
        y = (.38 + .55 * (rows - 1 - index // 2)) / height
        slider = Slider(fig.add_axes([x, y, track_width, .18 / height]), '',
                        0, max(1, len(samples) - 1), valinit=0, valstep=1)
        slider.ax.text(0, 1.8, parameter_label(name),
                       transform=slider.ax.transAxes, fontsize=9)
        slider.valtext.set_visible(False)
        if len(samples) == 1:
            slider.set_active(False)
        slider.on_changed(redraw)
        sliders.append(slider)

        box = TextBox(fig.add_axes([box_x, y - .03 / height, box_width, .25 / height]),
                      '', initial=f'{samples[0]:g}')
        buttons.append(add_step_buttons(
            fig, slider, np.arange(len(samples)),
            [previous_x, y - .03 / height, button_width, .25 / height],
            [next_x, y - .03 / height, button_width, .25 / height]))

        def submit(value, values=samples, control=slider, field=box):
            try:
                requested = float(value)
                if (not np.isfinite(requested) or
                        not min(values[0], values[-1]) <= requested <= max(values[0], values[-1])):
                    raise ValueError
                nearest = int(np.argmin(np.abs(values - requested)))
                control.set_val(nearest)
                field.ax.set_facecolor('white')
            except ValueError:
                field.ax.set_facecolor('#ffd8d8')
                fig.canvas.draw_idle()

        box.on_submit(submit)
        boxes.append(box)
    fig._single_crystal_sliders = sliders
    fig._single_crystal_value_boxes = boxes
    fig._single_crystal_step_buttons = buttons
    return sliders, boxes


@gui_plot
def plot_scan(exp, axes, *, qmax, title_digits=2, sample_geometry=None,
              detector_geometry=None, sample_motor_chain=None,
              detector_motor_chain=None, sample_angles=None, detector_angles=None,
              sam_angles=None, intensity_settings=None, show=True):
    """Update one 3D experiment in place as sampled parameters are selected."""
    import matplotlib.pyplot as plt
    update = configuration_updater(
        exp, qmax=qmax, sample_geometry=sample_geometry,
        detector_geometry=detector_geometry,
        sample_motor_chain=sample_motor_chain,
        detector_motor_chain=detector_motor_chain,
        sample_angles=sample_angles, detector_angles=detector_angles,
        sam_angles=sam_angles, intensity_settings=intensity_settings)

    rows = int(np.ceil(len(axes) / 2))
    height = 8 + rows * .55
    fig = plt.figure(figsize=(13, height))
    ax = fig.add_subplot(111, projection='3d')
    bottom = (1.1 + rows * .55) / height

    def redraw(_=None):
        config = {name: samples[int(slider.val)]
                  for (name, samples), slider in zip(axes.items(), sliders)}
        options = update(config)
        camera = ax.elev, ax.azim
        exp.plot_3d_single_xstal_exp(
            ax=ax, show=False, title_digits=title_digits,
            colorbar_bounds=[.16, bottom - .055, .44, .018], **options)
        ax.view_init(*camera)
        ax.set_position([.04, bottom + .02, .72, .72 - bottom])
        for box, slider, samples in zip(boxes, sliders, axes.values()):
            box.eventson = False
            box.set_val(f'{samples[int(slider.val)]:g}')
            box.eventson = True
        fig.canvas.draw_idle()

    sliders, boxes = _attach_controls(fig, axes, height, redraw)
    redraw()
    if show:
        plt.show()
    return fig, ax


@gui_plot
def plot_scan_2d(exp, axes, *, qmax, title_digits=2, sample_geometry=None,
                 detector_geometry=None, sample_motor_chain=None,
                 detector_motor_chain=None, sample_angles=None,
                 detector_angles=None, sam_angles=None,
                 intensity_settings=None, show=True):
    """Update the detector-plane image in place for sampled configurations."""
    import matplotlib.pyplot as plt

    update = configuration_updater(
        exp, qmax=qmax, sample_geometry=sample_geometry,
        detector_geometry=detector_geometry,
        sample_motor_chain=sample_motor_chain,
        detector_motor_chain=detector_motor_chain,
        sample_angles=sample_angles, detector_angles=detector_angles,
        sam_angles=sam_angles, intensity_settings=intensity_settings)
    rows = int(np.ceil(len(axes) / 2))
    height = 8 + rows * .55
    fig, ax = plt.subplots(figsize=(13, height))
    bottom = (1.1 + rows * .55) / height

    def redraw(_=None):
        config = {name: samples[int(slider.val)]
                  for (name, samples), slider in zip(axes.items(), sliders)}
        options = update(config)
        if hasattr(ax, '_single_crystal_hover_cid'):
            fig.canvas.mpl_disconnect(ax._single_crystal_hover_cid)
            ax._single_crystal_hover_text.remove()
            del ax._single_crystal_hover_cid
            del ax._single_crystal_hover_text
        if hasattr(ax, '_single_crystal_pick_cid'):
            fig.canvas.mpl_disconnect(ax._single_crystal_pick_cid)
            del ax._single_crystal_pick_cid
        if hasattr(ax, '_single_crystal_colorbar'):
            ax._single_crystal_colorbar.remove()
            del ax._single_crystal_colorbar
        ax.clear()
        exp.plot_2d_single_xstal_exp(
            ax=ax, title_digits=title_digits, show=False, layout=False,
            colorbar_bounds=[.16, bottom - .055, .44, .018], **options)
        title_lines = ax.get_title().count('\n') + 1
        axes_bottom = bottom + .04
        axes_top = max(axes_bottom + .2, .68 - .01 * max(0, title_lines - 6))
        ax.set_position([.08, axes_bottom, .65, axes_top - axes_bottom])
        for box, slider, samples in zip(boxes, sliders, axes.values()):
            box.eventson = False
            box.set_val(f'{samples[int(slider.val)]:g}')
            box.eventson = True
        fig.canvas.draw_idle()

    sliders, boxes = _attach_controls(fig, axes, height, redraw)
    redraw()
    if show:
        plt.show()
    return fig, ax
