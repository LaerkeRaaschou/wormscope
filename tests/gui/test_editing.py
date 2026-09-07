"""The brush, the tools and undo — everything that changes a mask by hand.

Two rules run through all of it:

* a stroke is one undo step, not one per mouse position;
* an edited frame is remembered in ``_hand_edited``, because that is what makes
  the correction survive the next SAM2 run (it becomes a mask prompt).
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent

from wormscope.widgets import ZoomView


@pytest.fixture
def canvas(qapp, window):
    """The window, with a worm selected and a usable-size zoom canvas."""
    window.select_worm(window.worms[0])
    window.set_frame(2)
    window.zoomv.resize(240, 180)
    return window


def stroke(win, qapp, x0, y0, x1, y1, shift=False, release_shift=False):
    """Press, drag, release — as a hand would, with Shift optional."""
    zoom = win.zoomv

    def at(x, y):
        left, top = zoom._placement()
        return QPointF(left + x * zoom.zoom, top + y * zoom.zoom)

    mods = (Qt.KeyboardModifier.ShiftModifier if shift
            else Qt.KeyboardModifier.NoModifier)
    zoom.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, at(x0, y0), Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton, mods))
    drag = Qt.KeyboardModifier.NoModifier if release_shift else mods
    zoom.mouseMoveEvent(QMouseEvent(
        QEvent.Type.MouseMove, at(x1, y1), Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton, drag))
    zoom.mouseReleaseEvent(QMouseEvent(
        QEvent.Type.MouseButtonRelease, at(x1, y1), Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton, drag))
    qapp.processEvents()


def area(win, wid=1, frame=2):
    return int(win.masks.get(wid, frame).sum())


# --- tools ------------------------------------------------------------------
def test_the_tool_buttons_set_the_canvas_mode(canvas):
    for button, mode in ((canvas.btn_pan, ZoomView.MODE_PAN),
                         (canvas.btn_draw, ZoomView.MODE_DRAW),
                         (canvas.btn_erase, ZoomView.MODE_ERASE),
                         (canvas.btn_sam_pos, ZoomView.MODE_SAM_POS),
                         (canvas.btn_sam_neg, ZoomView.MODE_SAM_NEG)):
        button.click()
        assert canvas.zoomv.mode == mode


def test_only_one_tool_is_active_at_a_time(canvas):
    canvas.btn_draw.click()
    canvas.btn_erase.click()
    assert canvas.btn_erase.isChecked() and not canvas.btn_draw.isChecked()


def test_the_brush_slider_sets_the_brush(canvas):
    canvas.brush_slider.setValue(15)
    assert canvas.zoomv.brush == 15 and canvas.brush_val.text() == "15"


# --- painting ---------------------------------------------------------------
def test_drawing_adds_to_the_mask(canvas, qapp):
    canvas.btn_draw.click()
    before = area(canvas)
    stroke(canvas, qapp, 30, 25, 45, 25)
    assert area(canvas) > before


def test_erasing_removes_from_it(canvas, qapp):
    canvas.btn_erase.click()
    before = area(canvas)
    stroke(canvas, qapp, 8, 10, 20, 10)          # over the worm
    assert area(canvas) < before


def test_painting_with_no_worm_selected_does_nothing(window, qapp):
    window.select_worm(None)
    window.btn_draw.click()
    before = window.masks.get(1, 0).copy()
    stroke(window, qapp, 10, 10, 20, 20)
    assert np.array_equal(window.masks.get(1, 0), before)


def test_shift_erases_while_draw_is_selected(canvas, qapp):
    canvas.btn_draw.click()
    stroke(canvas, qapp, 10, 25, 50, 25)
    painted = area(canvas)
    stroke(canvas, qapp, 20, 25, 30, 25, shift=True)
    assert area(canvas) < painted
    assert canvas.zoomv.mode == ZoomView.MODE_DRAW, \
        "the toolbar must not change underneath you"


def test_shift_draws_while_erase_is_selected(canvas, qapp):
    canvas.btn_erase.click()
    stroke(canvas, qapp, 8, 10, 25, 10)
    rubbed = area(canvas)
    stroke(canvas, qapp, 10, 30, 40, 30, shift=True)
    assert area(canvas) > rubbed
    assert canvas.zoomv.mode == ZoomView.MODE_ERASE


def test_the_swap_lasts_exactly_one_stroke(canvas, qapp):
    canvas.btn_draw.click()
    stroke(canvas, qapp, 10, 25, 40, 25, shift=True)
    assert not canvas.zoomv._swap_paint
    before = area(canvas)
    stroke(canvas, qapp, 10, 30, 40, 30)
    assert area(canvas) > before, "back to drawing"


def test_letting_go_of_shift_mid_drag_does_not_flip_the_stroke(canvas, qapp):
    """The tool is decided at mouse-down. Otherwise half a stroke paints and
    half erases, which is impossible to undo cleanly."""
    canvas.btn_draw.click()
    canvas.masks.set(1, 2, np.zeros((40, 60), bool))
    stroke(canvas, qapp, 5, 30, 55, 30)
    full = area(canvas)
    stroke(canvas, qapp, 10, 30, 45, 30, shift=True, release_shift=True)
    assert area(canvas) < full


def test_shift_means_nothing_to_pan(canvas, qapp):
    canvas.btn_pan.click()
    before = area(canvas)
    stroke(canvas, qapp, 5, 5, 30, 30, shift=True)
    assert area(canvas) == before


def test_paint_mode_passes_other_tools_straight_through(canvas):
    canvas.btn_pan.click()
    assert canvas.zoomv.paint_mode() == ZoomView.MODE_PAN


# --- what a stroke leaves behind -------------------------------------------
def test_a_stroke_updates_the_metrics(canvas, qapp):
    canvas.btn_draw.click()
    before = canvas.worms[0].metrics["area_px"][2]
    stroke(canvas, qapp, 30, 25, 45, 25)
    assert canvas.worms[0].metrics["area_px"][2] > before


def test_an_edited_frame_is_remembered_for_sam2(canvas, qapp):
    """`_hand_edited` is why a correction survives the next re-track: those
    frames are fed to SAM2 as mask prompts first."""
    canvas.btn_draw.click()
    stroke(canvas, qapp, 30, 25, 45, 25)
    assert (1, 2) in canvas._hand_edited


def test_a_stroke_is_a_single_undo_step(canvas, qapp):
    canvas.btn_draw.click()
    before = canvas.masks.get(1, 2).copy()
    stroke(canvas, qapp, 20, 25, 50, 25)         # many mouse positions
    assert len(canvas._undo) == 1
    canvas._undo_action()
    assert np.array_equal(canvas.masks.get(1, 2), before)


def test_redo_puts_it_back(canvas, qapp):
    canvas.btn_draw.click()
    stroke(canvas, qapp, 20, 25, 50, 25)
    after = canvas.masks.get(1, 2).copy()
    canvas._undo_action()
    canvas._redo_action()
    assert np.array_equal(canvas.masks.get(1, 2), after)


def test_the_undo_buttons_follow_the_stacks(canvas, qapp):
    assert not canvas.undo_btn.isEnabled()
    canvas.btn_draw.click()
    stroke(canvas, qapp, 20, 25, 50, 25)
    assert canvas.undo_btn.isEnabled() and not canvas.redo_btn.isEnabled()
    canvas._undo_action()
    assert canvas.redo_btn.isEnabled() and not canvas.undo_btn.isEnabled()


def test_a_new_action_clears_the_redo_stack(canvas, qapp):
    """Standard editor behaviour: once you branch, the old future is gone."""
    canvas.btn_draw.click()
    stroke(canvas, qapp, 20, 25, 50, 25)
    canvas._undo_action()
    stroke(canvas, qapp, 20, 30, 50, 30)
    assert canvas._redo == []


def test_undoing_with_nothing_to_undo_is_harmless(window):
    window._undo_action()
    window._redo_action()


def test_the_undo_stack_is_bounded(canvas, qapp):
    from wormscope.config import UNDO_LIMIT
    canvas.btn_draw.click()
    for i in range(UNDO_LIMIT + 5):
        stroke(canvas, qapp, 10 + i % 20, 25, 12 + i % 20, 25)
    assert len(canvas._undo) == UNDO_LIMIT


# --- propagate --------------------------------------------------------------
def test_propagate_copies_the_mask_to_the_next_frame(canvas):
    canvas.set_frame(1)
    here = canvas.masks.get(1, 1).copy()
    canvas._propagate()
    assert canvas.frame == 2, "it steps onto the frame it just filled"
    assert np.array_equal(canvas.masks.get(1, 2), here)
    assert (1, 2) in canvas._hand_edited


def test_propagate_is_undoable(canvas):
    canvas.set_frame(1)
    before = canvas.masks.get(1, 2).copy()
    canvas._propagate()
    canvas._undo_action()
    assert np.array_equal(canvas.masks.get(1, 2), before)


def test_propagate_at_the_last_frame_does_nothing(canvas):
    canvas.set_frame(canvas.n - 1)
    canvas._propagate()
    assert canvas.frame == canvas.n - 1


# --- refresh ----------------------------------------------------------------
def test_refresh_recomputes_every_metric_from_the_masks(canvas):
    """The escape hatch when metrics and masks have drifted apart — after a
    script wrote masks directly, say."""
    canvas.worms[0].metrics["area_px"][2] = 99999.0
    canvas._refresh_plots()
    assert canvas.worms[0].metrics["area_px"][2] == int(canvas.masks.get(1, 2).sum())


# --- view toggles -----------------------------------------------------------
def test_outline_and_onion_toggles_reach_the_renderer(canvas):
    canvas.outline_btn.setChecked(True)
    assert canvas.outline_only
    canvas.onion_btn.setChecked(True)
    assert canvas.onion
    canvas.outline_btn.setChecked(False)
    canvas.onion_btn.setChecked(False)
    assert not canvas.outline_only and not canvas.onion


def test_the_zoom_slider_and_the_canvas_stay_in_step(canvas):
    canvas.zoom_slider.setValue(80)
    assert canvas.zoomv.zoom == pytest.approx(8.0)
    canvas.zoomv.set_zoom(3.0)
    assert canvas.zoom_slider.value() == 30, "and the other way round"
