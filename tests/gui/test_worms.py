"""Adding, deleting and renumbering worms.

All three are undoable, and all three have to keep four things in step: the
list widget, ``self.worms``, ``self.by_id`` and the MaskStore. Anything that
falls out of step here shows up much later as a worm that cannot be selected or
a mask saved under the wrong name.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QInputDialog, QMessageBox

from tests.conftest import build_dataset
from wormscope.widgets import ZoomView


# --- adding -----------------------------------------------------------------
def test_add_creates_a_worm_with_the_next_free_id(window):
    window._add_worm()
    assert [w.obj_id for w in window.worms] == [1, 2]
    assert window.selected is window.worms[-1]
    assert window.list.count() == 2


def test_the_first_worm_of_an_empty_dataset_is_number_zero(make_window):
    win = make_window(build_dataset(worm_ids=()))
    win._add_worm()
    assert win.worms[0].obj_id == 0


def test_a_new_worm_arms_the_brush_without_frames_on_disk(window):
    """SAM2 needs the frames as files. With none, arming the SAM tool would
    give a first click that can only raise a dialog."""
    assert window._frames_dir is None
    window._add_worm()
    assert window.zoomv.mode == ZoomView.MODE_DRAW and window.btn_draw.isChecked()


def test_a_new_worm_arms_sam_when_the_frames_are_there(make_window, tmp_path):
    win = make_window(build_dataset(frames_dir=tmp_path))
    win._add_worm()
    assert win.zoomv.mode == ZoomView.MODE_SAM_POS and win.btn_sam_pos.isChecked()


def test_adding_is_undoable(window):
    window._add_worm()
    window._undo_action()
    assert [w.obj_id for w in window.worms] == [1] and window.list.count() == 1
    window._redo_action()
    assert len(window.worms) == 2


# --- deleting ---------------------------------------------------------------
def test_delete_removes_the_worm_and_its_masks(window, silence_dialogs):
    window.select_worm(window.worms[0])
    window._delete_worm()
    assert window.worms == [] and window.list.count() == 0
    assert window.masks.frames_of(1) == []
    assert window.selected is None


def test_delete_asks_first(window, silence_dialogs):
    window.select_worm(window.worms[0])
    window._delete_worm()
    assert any(kind == "question" for kind, _ in silence_dialogs), \
        "deleting every frame of a worm is not a thing to do silently"


def test_saying_no_keeps_the_worm(window, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.No))
    window.select_worm(window.worms[0])
    window._delete_worm()
    assert len(window.worms) == 1


def test_deleting_is_undoable(window, silence_dialogs):
    window.select_worm(window.worms[0])
    before = window.masks.get(1, 0).copy()
    window._delete_worm()
    window._undo_action()
    assert [w.obj_id for w in window.worms] == [1]
    assert np.array_equal(window.masks.get(1, 0), before), "the masks come back"


def test_deleting_with_nothing_selected_does_nothing(window, silence_dialogs):
    window.select_worm(None)
    window._delete_worm()
    assert len(window.worms) == 1


def test_deleting_a_worm_drops_its_sam2_session(window, silence_dialogs,
                                                stub_sam, settle, tmp_path):
    """Its prompts would otherwise be re-applied to whatever takes its id."""
    window._frames_dir = tmp_path
    window.select_worm(window.worms[0])
    window._on_sam_point(20.0, 10.0, 1)
    settle(window)
    window._delete_worm()
    settle(window)
    assert 1 not in window.sam.backend.sessions


# --- renumbering ------------------------------------------------------------
def test_reassign_moves_the_worm_and_its_masks(window):
    worm = window.worms[0]
    before = window.masks.get(1, 0).copy()
    window._do_reassign(worm, 1, 42)
    assert worm.obj_id == 42 and window.by_id == {42: worm}
    assert np.array_equal(window.masks.get(42, 0), before)
    assert window.masks.frames_of(1) == []
    assert window.list.item(0).text() == "#42"


def test_reassign_is_undoable(window, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getInt", staticmethod(
        lambda *a, **k: (42, True)))
    window.select_worm(window.worms[0])
    window._reassign_id()
    assert window.worms[0].obj_id == 42
    window._undo_action()
    assert window.worms[0].obj_id == 1 and window.masks.frames_of(1) != []


def test_an_id_already_in_use_is_refused(make_window, monkeypatch,
                                         silence_dialogs):
    """Merging two worms is not supported, so it says so instead of quietly
    losing one of them."""
    monkeypatch.setattr(QInputDialog, "getInt", staticmethod(
        lambda *a, **k: (2, True)))
    win = make_window(build_dataset(worm_ids=(1, 2)))
    win.select_worm(win.worms[0])
    win._reassign_id()
    assert [w.obj_id for w in win.worms] == [1, 2], "nothing moved"
    assert any(kind == "warning" for kind, _ in silence_dialogs)


def test_cancelling_the_dialog_changes_nothing(window, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getInt", staticmethod(
        lambda *a, **k: (42, False)))
    window.select_worm(window.worms[0])
    window._reassign_id()
    assert window.worms[0].obj_id == 1


def test_renaming_marks_the_frames_for_rewriting(window):
    window._do_reassign(window.worms[0], 1, 42)
    assert (42, 0) in window.masks.dirty, \
        "the PNGs have to be written out under the new name on Save"


# --- the list stays in step -------------------------------------------------
def test_the_list_the_worms_and_by_id_never_diverge(window, silence_dialogs):
    window._add_worm()
    window._add_worm()
    window.select_worm(window.worms[1])
    window._delete_worm()
    window._do_reassign(window.worms[-1], window.worms[-1].obj_id, 77)
    assert window.list.count() == len(window.worms)
    assert [w.obj_id for w in window.worms] == \
        [int(window.list.item(i).text().lstrip("#"))
         for i in range(window.list.count())]
    assert window.by_id == {w.obj_id: w for w in window.worms}
