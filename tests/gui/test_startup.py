"""What happens when the program opens.

The rule: **either the folder you named, or nothing at all.** No demo data, no
remembered default directory, no guessing. An empty window is a legitimate
state, and everything that needs frames is switched off in it rather than
failing when clicked.
"""

import inspect
from pathlib import Path

import numpy as np
from PIL import Image

from wormscope.app import resolve_dataset

ROOT = Path(__file__).resolve().parent.parent.parent


def make_video_folder(root: Path, n=3):
    (root / "frames").mkdir(parents=True)
    (root / "binary").mkdir()
    for f in range(n):
        Image.fromarray(np.full((30, 40, 3), 120, np.uint8)).save(
            root / "frames" / f"img_{f:03d}.png")
        m = np.zeros((30, 40), bool)
        m[10:16, 5 + f:20 + f] = True
        Image.fromarray(np.where(m, 255, 0).astype(np.uint8), "L").save(
            root / "binary" / f"worm01_img_{f:03d}.png")
    return root


# --- resolving what to open -------------------------------------------------
def test_a_plain_start_loads_nothing():
    ds = resolve_dataset()
    assert ds.is_empty and ds.worms == []
    assert ds.csv_path is None and ds.frames_dir is None


def test_a_named_folder_loads(tmp_path):
    ds = resolve_dataset(make_video_folder(tmp_path / "vid"))
    assert not ds.is_empty and len(ds.worms) == 1


def test_a_folder_of_raw_images_becomes_a_working_set(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for f in range(3):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(raw / f"i_{f}.png")
    ds = resolve_dataset(raw)
    assert len(ds.frames) == 3 and ds.worms == []


def test_a_path_that_does_not_exist_opens_empty(tmp_path):
    """Not the demo, and not an error dialog — just an empty window with the
    two Open buttons live."""
    ds = resolve_dataset(tmp_path / "nope")
    assert ds.is_empty and ds.worms == []


def test_a_folder_with_no_frames_opens_empty(tmp_path):
    assert resolve_dataset(tmp_path).is_empty


def test_resolve_dataset_takes_nothing_but_a_path():
    """A second parameter would be somewhere for a default to creep back in."""
    assert list(inspect.signature(resolve_dataset).parameters) == ["path"]


def test_startup_cannot_reach_the_demo():
    """make_demo still exists for tests and experiments, but nothing on the
    startup path can call it."""
    assert "make_demo" not in (ROOT / "wormscope" / "app.py").read_text()
    from wormscope.io.demo import make_demo
    assert callable(make_demo)


def test_no_default_folder_setting_survives():
    from wormscope import config
    assert not hasattr(config, "DEFAULT_DATA_DIR")


# --- the empty window -------------------------------------------------------
def test_the_empty_window_opens(make_window):
    from wormscope.core.models import empty_dataset
    win = make_window(empty_dataset())
    assert win.n == 0
    assert "No video loaded" in win.info.text(), "it says how to get started"
    assert "no video" in win.windowTitle().lower()


def test_only_the_two_ways_in_stay_enabled(make_window):
    from wormscope.core.models import empty_dataset
    win = make_window(empty_dataset())
    assert win.open_btn.isEnabled() and win.import_btn.isEnabled()
    for w in (win.btn_draw, win.btn_sam_pos, win.add_btn, win.export_btn,
              win.play_btn, win.slider, win.save_btn, win.list,
              win.lock_box, win.sam_run_btn, win.units_btn):
        assert not w.isEnabled(), f"{w} should be disabled with no video"


def test_the_usual_actions_are_harmless_with_no_data(make_window):
    """Keyboard shortcuts still fire even when the buttons are greyed out, so
    every one of these has to survive being called on an empty window."""
    from wormscope.core.models import empty_dataset
    win = make_window(empty_dataset())
    win.set_frame(2)
    win._step(1)
    win._toggle_play()
    win._refresh_plots()
    win._save()
    win._propagate()
    win._render_overlay()
    assert win.frame == 0 and win.n == 0


def test_opening_a_real_dataset_enables_everything(window):
    assert window.n == 6
    for w in (window.btn_draw, window.add_btn, window.play_btn, window.slider):
        assert w.isEnabled()


def test_no_worm_is_selected_to_begin_with(window):
    """Opening a video should not silently put you in 'editing worm 1'."""
    assert window.selected is None and window.list.currentRow() == -1
