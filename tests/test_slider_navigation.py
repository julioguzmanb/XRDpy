"""Mouse interaction and bounds for plot slider step buttons."""

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Slider

from trxrdpy.simulation.slider_navigation import add_step_buttons


def test_step_buttons_click_adjacent_samples_and_stop_at_bounds():
    fig = plt.figure(figsize=(5, 3))
    slider = Slider(fig.add_axes([.10, .5, .5, .1]), '', 10, 30, valinit=15)
    previous, following = add_step_buttons(
        fig, slider, [10, 20, 30], [.65, .5, .1, .1], [.80, .5, .1, .1])
    fig.canvas.draw()

    def click(button):
        x, y = button.ax.bbox.get_points().mean(axis=0)
        for event_name in ('button_press_event', 'button_release_event'):
            fig.canvas.callbacks.process(
                event_name, MouseEvent(event_name, fig.canvas, x, y, button=1))

    click(previous)
    assert slider.val == 10
    assert not previous.get_active() and following.get_active()
    click(previous)
    assert slider.val == 10
    click(following)
    assert slider.val == 20
    click(following)
    assert slider.val == 30
    assert previous.get_active() and not following.get_active()
    plt.close(fig)
