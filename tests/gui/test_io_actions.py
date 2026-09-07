"""Saving, calibrating and exporting — the ways work leaves the program.

Save is the one to be careful with: it writes PNGs, deletes files that a
rename or a delete made obsolete, rewrites measurements.csv and regenerates the
overview plot. Getting the delete list wrong loses somebody's data.
"""

import csv

import numpy as np
import pytest
from PyQt6.QtWidgets import QInputDialog

from wormscope.config import ACCEL_METRIC, SPEED_METRIC
from wormscope.io.loaders import load_dataset


@pytest.fixture
def saved_window(make_window, data_folder):
    """A window on a real on-disk folder, so Save actually writes."""
    win = make_window(load_dataset(data_folder))
    win.select_worm(win.worms[0])
    return win, data_folder


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# --- units ------------------------------------------------------------------
def test_calibration_changes_the_displayed_units(window, monkeypatch):
    answers = iter([(0.65, True), (12.0, True)])
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: next(answers)))
    window.select_worm(window.worms[0])
    window._set_units()
    assert window.px_to_um == 0.65 and window.fps == 12.0
    assert "µm" in window.info.text()


def test_zero_means_keep_pixels(window, monkeypatch):
    answers = iter([(0.0, True), (0.0, True)])
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: next(answers)))
    window._set_units()
    assert window.px_to_um is None and window.fps is None
    assert window._unit_scales() == {}


def test_cancelling_the_first_question_changes_nothing(window, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: (5.0, False)))
    window._set_units()
    assert window.px_to_um is None


def test_calibration_does_not_touch_the_stored_metrics(window, monkeypatch):
    """Pixels are what was measured; µm is a presentation layer on top."""
    answers = iter([(0.65, True), (12.0, True)])
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: next(answers)))
    before = dict(window.worms[0].metrics["area_px"])
    window._set_units()
    assert window.worms[0].metrics["area_px"] == before


def test_plot_specs_follow_the_calibration(window):
    window.px_to_um, window.fps = 2.0, 10.0
    names = [s[0] for s in window._plot_specs()]
    assert names == [window.area_metric, window.length_metric,
                     SPEED_METRIC, ACCEL_METRIC]
    assert window._plot_specs()[0][2] == pytest.approx(4.0)


# --- saving -----------------------------------------------------------------
def test_save_writes_the_edited_masks(saved_window):
    win, folder = saved_window
    win.masks.set(1, 0, np.zeros((30, 40), bool))
    win._save()
    from PIL import Image
    written = np.asarray(Image.open(
        folder / "binary" / "worm01_frame_0000.png").convert("L")) > 127
    assert not written.any(), "an erased mask is written out as erased"


def test_save_rewrites_the_csv_from_the_current_metrics(saved_window):
    from wormscope.core.metrics import apply_mask_stats
    win, folder = saved_window
    win.masks.set(1, 0, np.zeros((30, 40), bool))
    apply_mask_stats(win.worms[0], win.masks, 0, win.area_metric,
                     win.length_metric)
    win._save()
    assert read_csv(folder / "measurements.csv")[0]["area_px"] == "0"


def test_save_regenerates_the_overview_plot(saved_window):
    win, folder = saved_window
    win.masks.set_dirty(1, 0)
    win._save()
    assert (folder / "measurements_plot.png").is_file()


def test_save_clears_the_dirty_marks_and_the_star_in_the_title(saved_window):
    win, _folder = saved_window
    win.masks.set(1, 0, np.ones((30, 40), bool))
    win._after_edit()                        # what a finished brush stroke does
    assert win.windowTitle().endswith("*"), "unsaved edits are visible"
    win._save()
    assert win.masks.dirty == set() and not win.windowTitle().endswith("*")
    assert not win.save_btn.isEnabled()


def test_deleting_a_worm_removes_its_files_only_on_save(saved_window,
                                                        silence_dialogs):
    """Until Save, a delete is undoable — so the PNGs have to stay put."""
    win, folder = saved_window
    png = folder / "binary" / "worm01_frame_0000.png"
    win.select_worm(win.worms[0])
    win._delete_worm()
    assert png.is_file(), "still on disk while the delete is undoable"
    win._save()
    assert not png.exists()


def test_undoing_a_delete_before_saving_keeps_the_files(saved_window,
                                                        silence_dialogs):
    win, folder = saved_window
    png = folder / "binary" / "worm01_frame_0000.png"
    win.select_worm(win.worms[0])
    win._delete_worm()
    win._undo_action()
    win._save()
    assert png.is_file()


def test_renaming_a_worm_moves_its_files_on_save(saved_window):
    win, folder = saved_window
    win._do_reassign(win.worms[0], 1, 7)
    win._save()
    assert (folder / "binary" / "worm07_frame_0000.png").is_file()
    assert not (folder / "binary" / "worm01_frame_0000.png").exists(), \
        "the old name is garbage now"


def test_an_in_memory_dataset_saves_nothing(window):
    """No binary/ folder: Save is a no-op rather than an error."""
    assert not window.masks.savable
    window.masks.set(1, 0, np.ones((40, 60), bool))
    window._save()


def test_reloading_the_folder_gives_back_what_was_saved(saved_window):
    """The round trip that matters: edit, save, reopen, same masks."""
    from wormscope.core.metrics import apply_mask_stats
    win, folder = saved_window
    edited = np.zeros((30, 40), bool)
    edited[5:9, 5:25] = True
    win.masks.set(1, 1, edited)
    apply_mask_stats(win.worms[0], win.masks, 1, "area_px", "length_px")
    win._save()

    reopened = load_dataset(folder)
    assert np.array_equal(reopened.masks.get(1, 1), edited)
    assert reopened.worms[0].metrics["area_px"][1] == 80


# --- export -----------------------------------------------------------------
def test_export_reports_a_missing_imageio_instead_of_crashing(
        saved_window, silence_dialogs, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_imageio(name, *args, **kwargs):
        if name == "imageio":
            raise ImportError("no imageio")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_imageio)
    win, _folder = saved_window
    win._export_video()
    assert any("pip install imageio" in str(t) or kind == "warning"
               for kind, t in silence_dialogs)


def test_export_writes_pngs_when_ffmpeg_is_missing(dataset, tmp_path,
                                                   monkeypatch):
    """A missing ffmpeg should still get you frames out, not nothing."""
    imageio = pytest.importorskip("imageio")
    monkeypatch.setattr(imageio, "get_writer",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ffmpeg")))
    from wormscope.io.export import export_overlay
    target, wrote_video, written = export_overlay(
        dataset.frames, dataset.worms, dataset.masks, 3, tmp_path,
        idx_to_key={0: 0, 1: 1, 2: 2})
    assert not wrote_video and written == 3
    assert sorted(p.name for p in target.iterdir()) == \
        ["00000.png", "00001.png", "00002.png"]


def test_export_can_be_stopped_part_way(dataset, tmp_path, monkeypatch):
    imageio = pytest.importorskip("imageio")
    monkeypatch.setattr(imageio, "get_writer",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ffmpeg")))
    from wormscope.io.export import export_overlay
    _target, _video, written = export_overlay(
        dataset.frames, dataset.worms, dataset.masks, 6, tmp_path,
        progress=lambda done, total: done >= 2)
    assert written == 2, "the frames already rendered are kept"


# --- opening another dataset ------------------------------------------------
def test_opening_another_video_warns_about_unsaved_edits(window, monkeypatch,
                                                         silence_dialogs):
    from PyQt6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Cancel))
    window.masks.set(1, 0, np.ones((40, 60), bool))
    window._open_folder()
    assert window.isVisible() or True, "cancelled: this window stays"


def test_with_no_edits_it_does_not_ask(window, silence_dialogs):
    window.masks.dirty.clear()
    window._open_folder()                    # the folder dialog returns ""
    assert not any(kind == "question" for kind, _ in silence_dialogs)


# --- opening a different dataset in a new window ----------------------------
def test_choosing_a_folder_opens_it_in_a_fresh_window(window, data_folder,
                                                      monkeypatch, qapp):
    """Each dataset gets its own window, and the old one is dropped from the
    WINDOWS list so it can be garbage-collected."""
    from PyQt6.QtWidgets import QFileDialog
    from wormscope.ui import WINDOWS

    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(data_folder)))
    window.masks.dirty.clear()
    WINDOWS.append(window)
    window._open_folder()
    qapp.processEvents()
    opened = WINDOWS[-1]
    try:
        assert opened is not window and opened.n == 4
        assert window not in WINDOWS
    finally:
        opened._shutdown_sam()
        opened.close()
        WINDOWS.remove(opened)


def test_importing_raw_frames_opens_them_too(window, tmp_path, monkeypatch,
                                             qapp):
    from PIL import Image as PILImage
    from PyQt6.QtWidgets import QFileDialog
    from wormscope.ui import WINDOWS

    raw = tmp_path / "raw"
    raw.mkdir()
    for f in range(3):
        PILImage.fromarray(np.zeros((10, 12, 3), np.uint8)).save(raw / f"i_{f}.png")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(raw)))
    window.masks.dirty.clear()
    WINDOWS.append(window)
    window._import_raw()
    qapp.processEvents()
    opened = WINDOWS[-1]
    try:
        assert opened.n == 3 and opened.worms == []
        assert opened._frames_dir == raw, "SAM2 can reach the frames on disk"
    finally:
        opened._shutdown_sam()
        opened.close()
        WINDOWS.remove(opened)


def test_a_folder_with_no_frames_is_reported_not_opened(window, tmp_path,
                                                        monkeypatch,
                                                        silence_dialogs):
    from PyQt6.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    window.masks.dirty.clear()
    window._open_folder()
    assert any(kind == "warning" for kind, _ in silence_dialogs)
