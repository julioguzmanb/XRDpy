"""Previous/next controls for simulation sliders."""

import numpy as np


def two_column_bounds(fig, column, *, track_inches, box_inches):
    """Place controls in two columns without reducing track or input width."""
    button_inches = .4
    group_inches = (track_inches + .22 + box_inches + .12 +
                    button_inches + .07 + button_inches)
    start = .65 + column * (group_inches + .55)
    width = fig.get_figwidth()
    if start + group_inches + .65 > width + 1e-9:
        raise ValueError('Figure is too narrow for slider navigation controls.')
    track = start / width
    box = (start + track_inches + .22) / width
    previous = (start + track_inches + .22 + box_inches + .12) / width
    following = (start + group_inches - button_inches) / width
    return track, track_inches / width, box, box_inches / width, previous, following, button_inches / width


def add_step_buttons(fig, slider, values, previous_bounds, next_bounds):
    """Step to adjacent sampled values and disable arrows at the endpoints."""
    from matplotlib.widgets import Button

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or np.any(np.diff(values) <= 0):
        raise ValueError('Navigation values must be a nonempty increasing grid.')
    previous = Button(fig.add_axes(previous_bounds), '◀',
                      color='#edf3f8', hovercolor='#dceaf4')
    following = Button(fig.add_axes(next_bounds), '▶',
                       color='#edf3f8', hovercolor='#dceaf4')

    def step(direction):
        current = float(slider.val)
        index = (np.searchsorted(values, current, side='left') - 1
                 if direction < 0 else np.searchsorted(values, current, side='right'))
        if 0 <= index < len(values):
            slider.set_val(float(values[index]))

    previous.on_clicked(lambda _: step(-1))
    following.on_clicked(lambda _: step(1))

    def update(_=None):
        current = float(slider.val)
        for button, enabled in ((previous, current > values[0]),
                                (following, current < values[-1])):
            button.set_active(bool(enabled))
            button.label.set_color('#1a3244' if enabled else '#a7b2ba')
            button.ax.set_facecolor('#edf3f8' if enabled else '#f4f5f6')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update()
    return previous, following
