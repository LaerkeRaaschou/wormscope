"""Moving around a clip: frames, selection and playback.

Everything here is the plumbing you use without thinking about it, which is
exactly why it is worth tests — a broken frame counter is obvious in use and
invisible in a diff.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtTest import QTest

from tests.conftest import build_dataset


# --- frames -----------------------------------------------------------------
def test_set_frame_moves_and_reports(window):
    window.set_frame(3)
    assert window.frame == 3
    assert "3" in window.frame_label.text()
    assert window.slider.value() == 3, "the scrubber follows"


def test_the_frame_is_clamped_to_the_clip(window):
    window.set_frame(999)
    assert window.frame == window.n - 1
    window.set_frame(-5)
    assert window.frame == 0


def test_stepping_moves_one_frame(window):
    window.set_frame(2)
    window._step(1)
    assert window.frame == 3
    window._step(-1)
    assert window.frame == 2


def test_dragging_the_slider_changes_the_frame(window):
    window.slider.setValue(4)
    assert window.frame == 4


def test_setting_the_frame_does_not_bounce_back_off_the_slider(window):
    """The slider both drives and follows the frame; without the signal block
    that is an infinite loop."""
    window.set_frame(5)
    assert window.frame == 5 and window.slider.value() == 5


# --- selection --------------------------------------------------------------
def test_selecting_a_worm_updates_the_list_and_the_plots(window):
    worm = window.worms[0]
    window.select_worm(worm)
    assert window.selected is worm
    assert window.list.currentRow() == 0
    assert window.plot.worm is worm
    assert f"#{worm.obj_id}" in window.info.text()


def test_clicking_the_list_selects_that_worm(make_window):
    win = make_window(build_dataset(worm_ids=(1, 2)))
    win.list.setCurrentRow(1)
    assert win.selected is win.worms[1]


def test_clicking_a_worm_in_the_video_selects_it(make_window):
    win = make_window(build_dataset(worm_ids=(1, 2)))
    win.set_frame(0)
    cx, cy = win.worms[1].centroid(0)
    win._on_video_click(cx, cy)
    assert win.selected is win.worms[1]


def test_clicking_empty_background_changes_nothing(window):
    window.set_frame(0)
    window.select_worm(window.worms[0])
    window._on_video_click(0.0, 39.0)            # far from the worm
    assert window.selected is window.worms[0]


def test_a_click_just_off_the_worm_still_finds_it(window):
    """Thin masks are hard to hit exactly, so a near miss falls back to the
    nearest centroid within HIT_RADIUS."""
    window.set_frame(0)
    cx, cy = window.worms[0].centroid(0)
    window._on_video_click(cx + 5, cy + 5)
    assert window.selected is window.worms[0]


def test_overlapping_worms_pick_the_smaller_one(make_window):
    """Otherwise a big worm sitting under a small one can never be clicked."""
    from wormscope.core.metrics import apply_mask_stats
    ds = build_dataset(worm_ids=(1, 2), with_masks=False)
    big = np.zeros((40, 60), bool)
    big[5:35, 5:55] = True
    small = np.zeros((40, 60), bool)
    small[18:22, 28:32] = True
    ds.masks.set(1, 0, big)
    ds.masks.set(2, 0, small)
    for w, name in zip(ds.worms, ("area_px", "area_px")):
        apply_mask_stats(w, ds.masks, 0, "area_px", "length_px")
    win = make_window(ds)
    win.set_frame(0)
    win._on_video_click(30.0, 20.0)              # inside both
    assert win.selected.obj_id == 2


def test_selecting_none_clears_the_plots(window):
    window.select_worm(window.worms[0])
    window.select_worm(None)
    assert window.selected is None and window.plot.worm is None


# --- the info panel ---------------------------------------------------------
def test_the_info_panel_shows_the_measurements(window):
    window.set_frame(0)
    window.select_worm(window.worms[0])
    text = window.info.text()
    assert "pos:" in text and "area_px" in text


def test_it_says_when_the_worm_is_absent(make_window):
    ds = build_dataset(with_masks=False)
    win = make_window(ds)
    win.select_worm(win.worms[0])
    assert "absent" in win.info.text()


# --- playback ---------------------------------------------------------------
def test_play_advances_and_pause_stops(window):
    window.set_frame(0)
    window._play()
    assert window.timer.isActive()
    window._tick()
    assert window.frame == 1
    window._pause()
    assert not window.timer.isActive()


def test_playback_stops_at_the_end(window):
    window.set_frame(window.n - 2)
    window._play()
    window._tick()                               # onto the last frame
    assert window.frame == window.n - 1 and window.timer.isActive()
    window._tick()                               # nowhere left to go
    assert not window.timer.isActive(), "it stops rather than looping"
    assert window.frame == window.n - 1


def test_playing_from_the_end_starts_over(window):
    window.set_frame(window.n - 1)
    window._play()
    assert window.frame == 0
    window._pause()


def test_stepping_pauses_playback(window):
    window._play()
    window._step(1)
    assert not window.timer.isActive(), \
        "reaching for the arrow keys means you want to look at a frame"


def test_the_skeleton_is_skipped_during_playback(window):
    """A skeleton per frame at 10 fps makes stepping lumpy; it comes back the
    moment you stop."""
    window.select_worm(window.worms[0])
    window._play()
    window.set_frame(1)
    assert window._sel_skel is None
    window._pause()


# --- the keyboard -----------------------------------------------------------
@pytest.fixture
def shown(qapp, window):
    window.show()
    window.activateWindow()
    qapp.processEvents()
    return window


@pytest.mark.parametrize("focus_on", [
    "zoom_slider", "slider", "brush_slider", "list", "plot", "btn_draw", "zoomv"])
def test_arrow_keys_step_frames_whatever_has_focus(qapp, shown, focus_on):
    """They are window shortcuts precisely because a focused slider, list or
    plot canvas would otherwise swallow them."""
    getattr(shown, focus_on).setFocus()
    qapp.processEvents()
    shown.set_frame(2)
    QTest.keyClick(shown, Qt.Key.Key_Right)
    qapp.processEvents()
    assert shown.frame == 3


def test_the_focused_slider_is_not_nudged_as_well(qapp, shown):
    shown.zoom_slider.setFocus()
    qapp.processEvents()
    before = shown.zoom_slider.value()
    QTest.keyClick(shown, Qt.Key.Key_Right)
    qapp.processEvents()
    assert shown.zoom_slider.value() == before


def test_shift_jumps_ten_frames(qapp, make_window):
    win = make_window(build_dataset(n_frames=20))
    win.show()
    win.activateWindow()
    qapp.processEvents()
    win.set_frame(2)
    QTest.keyClick(win, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    qapp.processEvents()
    assert win.frame == 12


def test_a_clicked_button_does_not_keep_the_keyboard(qapp, shown):
    """Otherwise Space re-presses the last button instead of playing."""
    QTest.mouseClick(shown.btn_draw, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert not shown.btn_draw.hasFocus()
    QTest.keyClick(shown, Qt.Key.Key_Space)
    qapp.processEvents()
    assert shown.timer.isActive()
    shown._pause()


def press(win, key, mods=Qt.KeyboardModifier.NoModifier):
    win.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, mods))


def test_o_toggles_onion_skin(window):
    press(window, Qt.Key.Key_O)
    assert window.onion_btn.isChecked() and window.onion


def test_p_copies_the_mask_forward(window):
    window.select_worm(window.worms[0])
    window.set_frame(0)
    press(window, Qt.Key.Key_P)
    assert window.frame == 1


def test_ctrl_z_and_ctrl_y_undo_and_redo(window):
    window.select_worm(window.worms[0])
    window.set_frame(0)
    before = window.masks.get(1, 1).copy()
    press(window, Qt.Key.Key_P)                  # an undoable action
    press(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert np.array_equal(window.masks.get(1, 1), before)
    press(window, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    assert not np.array_equal(window.masks.get(1, 1), before)
