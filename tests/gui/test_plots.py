"""The Plots panel: normalized curves, the hover read-out and scrubbing.

Area and length are drawn divided by their own median, because the absolute
value depends on the worm and on the calibration while the *shape* is what you
are actually looking at. The real number comes back on hover, so nothing is
lost by doing that.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import build_dataset
from wormscope.config import SPEED_METRIC
from wormscope.widgets import HAVE_MPL

pytestmark = pytest.mark.skipif(not HAVE_MPL, reason="needs matplotlib")


class FakeEvent:
    """The handful of attributes PlotPanel reads off a matplotlib event."""

    def __init__(self, ax=None, xdata=None, ydata=None, button=1):
        self.inaxes = ax
        self.xdata = xdata
        self.ydata = ydata
        self.button = button


AREAS = [180, 190, 200, 210, 400, 200, 195, 205]
LENGTHS = [38, 39, 40, 41, 60, 40, 39, 41]


@pytest.fixture
def plotted(make_window):
    """A window whose worm has a known area / length series, worm selected."""
    ds = build_dataset(n_frames=8, with_masks=False)
    worm = ds.worms[0]
    worm.metrics["area_px"] = {f: float(v) for f, v in enumerate(AREAS)}
    worm.metrics["length_px"] = {f: float(v) for f, v in enumerate(LENGTHS)}
    for f in range(8):
        worm.centroids[f] = (float(10 + f), 15.0)
    win = make_window(ds)
    win.select_worm(worm)
    return win


@pytest.fixture
def area_axis(plotted):
    i = plotted.plot.metrics.index("area_px")
    return plotted, plotted.plot, i, plotted.plot.axes[i]


# --- normalization ----------------------------------------------------------
def test_area_and_length_are_the_normalized_ones(plotted):
    assert plotted.plot.normalized == {plotted.area_metric,
                                       plotted.length_metric}


def test_the_curve_is_divided_by_its_median(area_axis):
    _win, plot, i, _ax = area_axis
    med = float(np.median(AREAS))
    drawn = [plot._drawn[i][f] for f in range(8)]
    assert np.allclose(drawn, [a / med for a in AREAS])
    assert float(np.median(drawn)) == pytest.approx(1.0)


def test_an_outlier_still_stands_out(area_axis):
    _win, plot, i, _ax = area_axis
    assert plot._drawn[i][4] > 1.8, "a doubled area reads as ~2"


def test_the_axis_says_what_it_is_showing(area_axis):
    _win, _plot, _i, ax = area_axis
    assert ax.get_ylabel().endswith("/ median")


def test_the_real_values_are_kept_alongside(area_axis):
    _win, plot, i, _ax = area_axis
    assert [plot._raw[i][f] for f in range(8)] == [float(a) for a in AREAS]


def test_kinematics_keep_their_own_units(plotted):
    """Zero and the sign are the meaning for speed and acceleration; dividing
    by a median would destroy both."""
    plot = plotted.plot
    if SPEED_METRIC not in plot.metrics:
        pytest.skip("speed is not one of the plotted metrics")
    i = plot.metrics.index(SPEED_METRIC)
    values = [v for _f, v in sorted(plotted.worms[0].metrics[SPEED_METRIC].items())
              if v == v]
    assert np.allclose(sorted(plot._drawn[i].values()), sorted(values))
    assert not plot.axes[i].get_ylabel().endswith("/ median")


def test_a_zero_median_is_left_alone_rather_than_divided(plotted):
    """A worm with no area at all would otherwise become infinities."""
    from wormscope.core.models import Worm
    flat = Worm(obj_id=2)
    flat.metrics["area_px"] = {f: 0.0 for f in range(4)}
    plotted._insert_worm(flat)
    plotted.select_worm(flat)
    i = plotted.plot.metrics.index("area_px")
    values = list(plotted.plot._drawn[i].values())
    assert values == [0.0] * 4 and all(v == v for v in values)


def test_changing_units_does_not_move_a_normalized_curve(area_axis):
    """A ratio is unit-free — that is half the point of drawing one."""
    win, plot, i, _ax = area_axis
    med = float(np.median(AREAS))
    win.px_to_um, win.fps = 2.0, 10.0
    plot.set_scales(win._unit_scales())
    assert np.allclose([plot._drawn[i][f] for f in range(8)],
                       [a / med for a in AREAS])


def test_but_the_hover_value_follows_the_units(area_axis):
    win, plot, i, _ax = area_axis
    win.px_to_um = 2.0
    plot.set_scales(win._unit_scales())
    factor = win._unit_scales()[win.area_metric][0]
    assert plot._raw[i][4] == pytest.approx(AREAS[4] * factor)


def test_selecting_no_worm_clears_the_curves(plotted):
    plotted.select_worm(None)
    assert plotted.plot._drawn == [{}] * len(plotted.plot.axes)


# --- the hover read-out -----------------------------------------------------
def test_hovering_a_curve_shows_the_real_value(area_axis):
    _win, plot, i, ax = area_axis
    plot._hover(FakeEvent(ax, xdata=4.0, ydata=plot._drawn[i][4]))
    tip = plot._tips[i]
    assert tip is not None and tip.get_visible()
    assert "frame 4" in tip.get_text()
    assert f"{AREAS[4]:.1f}" in tip.get_text(), "the measured number, not the ratio"


def test_it_hides_when_the_cursor_leaves_the_curve(area_axis):
    _win, plot, i, ax = area_axis
    plot._hover(FakeEvent(ax, xdata=4.0, ydata=plot._drawn[i][4]))
    lo, hi = ax.get_ylim()
    plot._hover(FakeEvent(ax, xdata=4.0, ydata=hi + 5 * (hi - lo)))
    assert not plot._tips[i].get_visible()


def test_it_hides_when_the_cursor_leaves_the_plots(area_axis):
    _win, plot, i, ax = area_axis
    plot._hover(FakeEvent(ax, xdata=4.0, ydata=plot._drawn[i][4]))
    plot._hover(FakeEvent(None))
    assert not plot._tips[i].get_visible()


def test_the_read_out_stays_inside_its_plot(area_axis, qapp):
    """It used to hang off the right-hand end of a curve, and tight_layout then
    squeezed the axes sideways to make room — the plot shuffled about under the
    cursor."""
    win, plot, i, ax = area_axis
    win.show()
    win.resize(1200, 860)
    qapp.processEvents()
    plot.figure.canvas.draw()
    renderer = plot.figure.canvas.get_renderer()
    box = ax.get_window_extent()
    before = tuple(round(v, 4) for v in ax.get_position().bounds)

    outside, sides = [], []
    for f in range(8):
        plot._hover(FakeEvent(ax, xdata=float(f), ydata=plot._drawn[i][f]))
        plot.figure.canvas.draw()
        b = plot._tips[i].get_window_extent(renderer)
        if not (box.x0 - 1 <= b.x0 and b.x1 <= box.x1 + 1
                and box.y0 - 1 <= b.y0 and b.y1 <= box.y1 + 1):
            outside.append(f)
        sides.append(plot._tips[i].get_horizontalalignment())
    assert not outside, f"the box spilled out of the axes at frame(s) {outside}"
    assert {"left", "right"} <= set(sides), \
        "it should flip to the other side of the cursor near the end"
    assert tuple(round(v, 4) for v in ax.get_position().bounds) == before, \
        "and the axes must not move to make room for it"
    assert not plot._tips[i].get_in_layout()


# --- scrubbing --------------------------------------------------------------
def test_dragging_across_the_plot_scrubs_frames(area_axis):
    _win, plot, _i, ax = area_axis
    seen = []
    plot.frameSelected.connect(seen.append)
    plot._on_press(FakeEvent(ax, xdata=6.0, ydata=0.0))
    plot._on_motion(FakeEvent(ax, xdata=7.0, ydata=0.0))
    plot._on_release(FakeEvent(ax, xdata=7.0, ydata=0.0))
    assert seen == [6, 7]


def test_hovering_never_changes_the_frame(area_axis):
    _win, plot, i, ax = area_axis
    seen = []
    plot.frameSelected.connect(seen.append)
    plot._on_motion(FakeEvent(ax, xdata=6.0, ydata=plot._drawn[i][6]))
    assert seen == []


def test_clicking_a_plot_moves_the_window_to_that_frame(area_axis):
    win, plot, _i, ax = area_axis
    plot._on_press(FakeEvent(ax, xdata=5.0, ydata=0.0))
    plot._on_release(FakeEvent(ax, xdata=5.0, ydata=0.0))
    assert win.frame == 5


def test_the_dot_marks_the_current_frame(area_axis):
    win, plot, i, _ax = area_axis
    win.set_frame(3)
    xs, ys = plot.dots[i].get_data()
    assert list(xs) == [3] and ys[0] == pytest.approx(plot._drawn[i][3])


def test_smoothing_is_display_only(plotted):
    """The stored metrics — and therefore the CSV — must not change."""
    before = dict(plotted.worms[0].metrics["area_px"])
    plotted.smooth_box.setChecked(True)
    assert plotted.plot.smooth
    assert plotted.worms[0].metrics["area_px"] == before
