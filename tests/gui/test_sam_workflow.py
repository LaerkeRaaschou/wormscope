"""SAM2 prompting and tracking, driven against a stub backend.

No torch, no checkpoint, no GPU — the stub records what it was asked to do,
which is the interesting part anyway. What is being protected here:

* **one session per worm**, so correcting worm 3 cannot disturb worm 1;
* **hand edits become anchors** (mask prompts) before a re-track, which is what
  makes corrections on several frames reinforce each other instead of fighting;
* **everything is asynchronous**, so the window stays editable while it runs
  and your edits beat the tracker's results.

Marked `slow`: these drive a real worker thread. `pytest -m "not slow"` skips
them.
"""

import time

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent

from tests.conftest import build_dataset
from wormscope.widgets import ZoomView

pytestmark = pytest.mark.slow


@pytest.fixture
def sam_window(make_window, stub_sam, tmp_path, silence_dialogs):
    """A window whose frames are 'on disk', wired to the stub backend."""
    ds = build_dataset(n_frames=6, shape=(40, 60), with_masks=False,
                       frames_dir=tmp_path)
    win = make_window(ds)
    win._frames_dir = tmp_path
    win.select_worm(win.worms[0])
    win.zoomv.resize(240, 180)
    win.btn_sam_pos.click()
    return win


def click(win, qapp, settle, x, y, shift=False):
    """A real mouse press at an image coordinate on the zoom canvas."""
    zoom = win.zoomv
    left, top = zoom._placement()
    pos = QPointF(left + x * zoom.zoom, top + y * zoom.zoom)
    mods = (Qt.KeyboardModifier.ShiftModifier if shift
            else Qt.KeyboardModifier.NoModifier)
    zoom.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, pos, Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton, mods))
    settle(win)


def propagates(backend):
    return [c for c in backend.calls if c[0] == "propagate"]


def anchors(backend):
    return [c for c in backend.calls if c[0] == "add_mask"]


# --- placing points ---------------------------------------------------------
def test_a_plain_click_is_a_positive_point(sam_window, qapp, settle):
    seen = []
    sam_window.zoomv.pointAdded.connect(lambda x, y, lab: seen.append(lab))
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    assert seen == [1]
    assert sam_window.sam.backend.point_prompts[-1][1] == 1
    assert sam_window.masks.get(1, 2).any(), "SAM2's mask came back and was used"


def test_shift_click_is_a_negative_point(sam_window, qapp, settle):
    seen = []
    sam_window.zoomv.pointAdded.connect(lambda x, y, lab: seen.append(lab))
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    click(sam_window, qapp, settle, 30.0, 12.0, shift=True)
    assert seen == [1, 0]
    assert sam_window.zoomv.mode == ZoomView.MODE_SAM_POS, \
        "one click, not a mode change"


def test_the_negative_tool_makes_plain_clicks_negative(sam_window, qapp, settle):
    seen = []
    sam_window.zoomv.pointAdded.connect(lambda x, y, lab: seen.append(lab))
    sam_window.btn_sam_neg.click()
    click(sam_window, qapp, settle, 22.0, 9.0)
    click(sam_window, qapp, settle, 26.0, 9.0, shift=True)
    assert seen == [0, 1], "and Shift inverts it for one click"
    assert sam_window.btn_sam_neg.isChecked(), "the tool is sticky"


def test_every_point_so_far_is_sent_with_each_click(sam_window, qapp, settle):
    """SAM2 wants the whole prompt set for a frame, not just the newest point."""
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    click(sam_window, qapp, settle, 30.0, 12.0, shift=True)
    assert [p[2] for p in sam_window._sam_points[(1, 2)]] == [1, 0]
    assert len(sam_window.sam.backend.point_prompts) == 2


def test_clicking_with_no_worm_selected_says_so(sam_window, qapp, settle):
    sam_window.select_worm(None)
    sam_window._on_sam_point(20.0, 10.0, 1)
    assert "worm" in sam_window.info.text().lower()


def test_without_frames_on_disk_sam_explains_itself(window, silence_dialogs):
    """SAM2 needs the frames as files; an in-memory dataset cannot use it."""
    window.select_worm(window.worms[0])
    assert window._ensure_segmenter() is False
    assert any(kind == "warning" for kind, _ in silence_dialogs)


# --- tracking ---------------------------------------------------------------
def test_segment_all_tracks_the_whole_clip(sam_window, qapp, settle):
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    sam_window.sam.backend.calls.clear()
    sam_window._sam_propagate()
    settle(sam_window)
    assert propagates(sam_window.sam.backend) == [("propagate", 1, None, False)]
    assert all(sam_window.masks.get(1, f).any() for f in range(6))


def test_re_segment_runs_forward_from_here(sam_window, qapp, settle):
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    sam_window.set_frame(3)
    sam_window.sam.backend.calls.clear()
    sam_window.back_box.setChecked(False)
    sam_window._sam_resegment()
    settle(sam_window)
    assert propagates(sam_window.sam.backend) == [("propagate", 1, 3, False)]


def test_the_back_box_adds_a_backwards_pass(sam_window, qapp, settle):
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    sam_window.set_frame(3)
    sam_window.sam.backend.calls.clear()
    sam_window.back_box.setChecked(True)
    sam_window._sam_resegment()
    settle(sam_window)
    assert propagates(sam_window.sam.backend) == \
        [("propagate", 1, 3, False), ("propagate", 1, 3, True)]


def test_tracking_with_no_prompts_at_all_says_so(sam_window, settle):
    sam_window._sam_propagate()
    settle(sam_window)
    assert "SAM points" in sam_window.info.text()


def test_tracking_recomputes_the_metrics(sam_window, qapp, settle):
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    sam_window._sam_propagate()
    settle(sam_window)
    assert sam_window.worms[0].metrics["area_px"].get(4, 0) > 0


# --- anchoring --------------------------------------------------------------
def test_a_hand_edit_is_sent_as_a_prompt_before_re_tracking(sam_window, qapp,
                                                            settle):
    """A painted frame is invisible to the tracker until it is a prompt, so a
    re-track would simply overwrite it."""
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    painted = np.zeros((40, 60), bool)
    painted[20:26, 8:28] = True
    sam_window.masks.set(1, 2, painted.copy())
    sam_window._hand_edited.add((1, 2))
    sam_window.sam.backend.calls.clear()
    sam_window._sam_resegment()
    settle(sam_window)

    backend = sam_window.sam.backend
    assert anchors(backend) == [("add_mask", 2, 1, int(painted.sum()))]
    kinds = [c[0] for c in backend.calls]
    assert kinds.index("add_mask") < kinds.index("propagate"), \
        "anchoring has to happen before tracking starts"


def test_an_anchored_frame_is_not_re_anchored(sam_window, qapp, settle):
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    sam_window.masks.set(1, 2, np.ones((40, 60), bool))
    sam_window._hand_edited.add((1, 2))
    sam_window._sam_resegment()
    settle(sam_window)
    assert (1, 2) in sam_window._anchored and (1, 2) not in sam_window._hand_edited
    sam_window.sam.backend.calls.clear()
    sam_window._sam_resegment()
    settle(sam_window)
    assert anchors(sam_window.sam.backend) == []


def test_mask_to_prompt_hands_the_current_mask_over(sam_window, qapp, settle):
    fixed = np.zeros((40, 60), bool)
    fixed[10:14, 10:30] = True
    sam_window.set_frame(3)
    sam_window.masks.set(1, 3, fixed.copy())
    sam_window._sam_mask_prompt()
    settle(sam_window)
    assert anchors(sam_window.sam.backend) == [("add_mask", 3, 1, int(fixed.sum()))]
    assert sam_window.masks.get(1, 3).any(), "SAM2's answer comes back"


def test_mask_to_prompt_with_an_empty_mask_says_so(sam_window, silence_dialogs):
    sam_window.set_frame(3)
    sam_window._sam_mask_prompt()
    assert any(kind == "information" for kind, _ in silence_dialogs)


def test_clear_turns_anchors_back_into_pending_edits(sam_window, qapp, settle):
    """The masks on screen stay; only SAM2's sessions are dropped, so the next
    run re-anchors them."""
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    sam_window.masks.set(1, 2, np.ones((40, 60), bool))
    sam_window._hand_edited.add((1, 2))
    sam_window._sam_resegment()
    settle(sam_window)
    sam_window._sam_clear()
    settle(sam_window)
    assert (1, 2) in sam_window._hand_edited and not sam_window._anchored
    assert sam_window._sam_points == {}


# --- one session per worm ---------------------------------------------------
def test_each_worm_gets_its_own_session(sam_window, qapp, settle):
    from wormscope.core.models import Worm
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    second = Worm(obj_id=7)
    sam_window._insert_worm(second)
    sam_window.select_worm(second)
    sam_window.set_frame(1)
    sam_window._on_sam_point(30.0, 20.0, 1)
    settle(sam_window)
    backend = sam_window.sam.backend
    assert sorted(backend.sessions) == [1, 7]
    assert backend.sessions[7] == [("points", 1)], \
        "worm 7's prompt did not land in worm 1's session"


def test_re_tracking_one_worm_leaves_the_others_alone(sam_window, qapp, settle):
    from wormscope.core.models import Worm
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    sam_window._sam_propagate()
    settle(sam_window)
    before = sam_window.masks.get(1, 4).copy()

    second = Worm(obj_id=7)
    sam_window._insert_worm(second)
    sam_window.select_worm(second)
    sam_window.set_frame(1)
    sam_window._on_sam_point(30.0, 20.0, 1)
    settle(sam_window)
    sam_window.set_frame(4)
    sam_window.sam.backend.calls.clear()
    sam_window._sam_resegment()
    settle(sam_window)

    assert propagates(sam_window.sam.backend) == [("propagate", 7, 4, False)]
    assert np.array_equal(sam_window.masks.get(1, 4), before)


def test_segment_all_covers_every_prompted_worm(sam_window, qapp, settle):
    from wormscope.core.models import Worm
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    second = Worm(obj_id=7)
    sam_window._insert_worm(second)
    sam_window.select_worm(second)
    sam_window._on_sam_point(30.0, 20.0, 1)
    settle(sam_window)
    sam_window.sam.backend.calls.clear()
    sam_window._sam_propagate()
    settle(sam_window)
    assert sorted(c[1] for c in propagates(sam_window.sam.backend)) == [1, 7]


# --- it runs in the background ---------------------------------------------
def test_the_window_stays_usable_while_it_runs(sam_window, qapp, settle,
                                               stub_sam, monkeypatch):
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    monkeypatch.setattr(stub_sam, "slow", 0.05)   # ~0.3 s for six frames

    marker = np.zeros((40, 60), bool)
    marker[30:34, 40:55] = True
    sam_window.set_frame(0)
    sam_window._sam_propagate(worms=[1])
    qapp.processEvents()
    assert not sam_window.progress_panel.isHidden(), "progress is shown"
    assert sam_window.sam.busy

    sam_window.set_frame(5)                      # navigate while it runs
    sam_window.masks.set(1, 5, marker.copy())    # and edit
    sam_window._hand_edited.add((1, 5))
    sam_window._edited_during_run.add((1, 5))
    assert sam_window.frame == 5

    settle(sam_window)
    assert np.array_equal(sam_window.masks.get(1, 5), marker), \
        "your edit beats the tracker's guess for that frame"
    assert not np.array_equal(sam_window.masks.get(1, 4), marker), \
        "other frames took the tracker's masks"
    assert sam_window.progress_panel.isHidden()


def test_stop_ends_a_run_early(sam_window, qapp, settle, stub_sam, monkeypatch):
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    monkeypatch.setattr(stub_sam, "slow", 0.05)
    sam_window.set_frame(0)
    sam_window._sam_propagate(worms=[1])
    qapp.processEvents()
    time.sleep(0.08)
    sam_window.progress_panel._on_stop()
    qapp.processEvents()
    settle(sam_window)
    assert not sam_window.sam.busy
    assert sam_window.progress_panel.isHidden()


# --- lock edits -------------------------------------------------------------
def test_a_locked_frame_is_prompted_but_never_overwritten(sam_window, qapp,
                                                          settle):
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    mine = np.zeros((40, 60), bool)
    mine[2:6, 30:50] = True
    sam_window.masks.set(1, 1, mine.copy())
    sam_window._hand_edited.add((1, 1))
    sam_window.lock_box.setChecked(True)
    sam_window.sam.backend.calls.clear()
    sam_window.set_frame(0)
    sam_window._sam_propagate(worms=[1])
    settle(sam_window)

    assert any(c[1] == 1 for c in anchors(sam_window.sam.backend)), \
        "still sent to SAM2 as a prompt, so it keeps steering the track"
    assert np.array_equal(sam_window.masks.get(1, 1), mine), \
        "but the tracker does not write over it"
    assert not np.array_equal(sam_window.masks.get(1, 2), mine), \
        "its neighbours still take the tracker's mask"


def test_unticking_gives_the_frame_back_to_the_tracker(sam_window, qapp, settle):
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    mine = np.zeros((40, 60), bool)
    mine[2:6, 30:50] = True
    sam_window.masks.set(1, 1, mine.copy())
    sam_window._hand_edited.add((1, 1))
    sam_window.lock_box.setChecked(False)
    sam_window.set_frame(0)
    sam_window._sam_propagate(worms=[1])
    settle(sam_window)
    assert not np.array_equal(sam_window.masks.get(1, 1), mine)


def test_protected_is_the_one_place_that_decides(sam_window):
    """Both rules in one predicate, so there is a single thing to read."""
    sam_window._edited_during_run.add((1, 3))
    assert sam_window._protected(1, 3), "edits during a run always win"
    sam_window.lock_box.setChecked(False)
    sam_window._hand_edited.add((1, 4))
    assert not sam_window._protected(1, 4)
    sam_window.lock_box.setChecked(True)
    assert sam_window._protected(1, 4), "with the box ticked, every hand edit"


# --- the worker thread ------------------------------------------------------
def test_the_worker_stops_cleanly(sam_window, qapp, settle):
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    worker = sam_window.sam
    sam_window._shutdown_sam()
    assert sam_window.sam is None and not worker.isRunning(), \
        "a thread still blocked on its queue would hang the process on exit"


def test_the_main_thread_never_touches_the_backend(sam_window, qapp, settle):
    """SAM2 is not thread-safe; the GUI keeps its own copy of what it needs."""
    sam_window.set_frame(2)
    click(sam_window, qapp, settle, 20.0, 7.0)
    assert 1 in sam_window._prompted_worms, \
        "the window tracks prompted worms itself rather than asking the backend"


# --- being asked for the SAM2 files ----------------------------------------
def test_the_picker_saves_the_pair_and_you_are_not_asked_again(
        window, tmp_path, monkeypatch, silence_dialogs):
    """There is no toolbar button for this any more: you are asked once, when
    SAM2 is first needed, and the answer is remembered."""
    from PyQt6.QtWidgets import QFileDialog
    from wormscope import settings

    monkeypatch.setenv("WORMSCOPE_SETTINGS", str(tmp_path / "settings.json"))
    cfg = tmp_path / "sam2.1_hiera_l.yaml"
    ckpt = tmp_path / "sam2.1_hiera_large.pt"
    cfg.write_text("stub")
    ckpt.write_bytes(b"stub")
    answers = iter([(str(cfg), ""), (str(ckpt), "")])
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: next(answers)))

    assert window._choose_sam2_files(explain=False) is True
    saved = settings.load()
    assert saved["sam2_cfg"] == str(cfg) and saved["sam2_ckpt"] == str(ckpt)


def test_cancelling_the_picker_saves_nothing(window, tmp_path, monkeypatch,
                                             silence_dialogs):
    from PyQt6.QtWidgets import QFileDialog
    from wormscope import settings

    monkeypatch.setenv("WORMSCOPE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    assert window._choose_sam2_files(explain=False) is False
    assert settings.load()["sam2_cfg"] == ""


def test_a_mismatched_pair_is_pointed_out_when_you_pick_it(
        window, tmp_path, monkeypatch, silence_dialogs):
    """Large config, tiny checkpoint: better to say so now than to fail later
    with a wall of missing state-dict keys."""
    from PyQt6.QtWidgets import QFileDialog

    monkeypatch.setenv("WORMSCOPE_SETTINGS", str(tmp_path / "settings.json"))
    cfg = tmp_path / "sam2.1_hiera_l.yaml"
    ckpt = tmp_path / "sam2.1_hiera_tiny.pt"
    cfg.write_text("stub")
    ckpt.write_bytes(b"stub")
    answers = iter([(str(cfg), ""), (str(ckpt), "")])
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: next(answers)))
    window._choose_sam2_files(explain=False)
    assert any(kind == "information" for kind, _ in silence_dialogs)
