"""Against the original: this package versus the single file it came from.

``wormscope_gui3.py`` is the pre-refactor program. Both are driven through the
same script of user actions and every observable result is compared — pixels,
masks, metrics, the CSV, the info text. It is the strongest statement the suite
makes: *the split into modules changed nothing*.

Skipped automatically when the original is not in the checkout. Point at it
with ``WORMSCOPE_ORIGINAL=/path/to/wormscope_gui3.py`` if it lives elsewhere.

Deliberately excluded: the brightness / contrast / gamma / invert controls,
which this version no longer has — the Zoom panel shows the frame as it is.

If you are deliberately changing behaviour, this test will fail, and that is
correct. Retiring it is a real decision: after that, nothing checks the two
against each other again.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.needs_original


def drive(make_window, make_demo, patch_qimage, csv_path):
    """Run the same sequence of user actions and return everything observable."""
    captured = []
    patch_qimage(captured)

    ds = make_demo(n_frames=12, size=(200, 140), seed=3)
    ds.csv_path = csv_path
    win = make_window(ds)
    out = {}

    win.set_frame(4)
    win.select_worm(win.worms[1])
    out["overlay_frame4"] = win._overlay.copy()

    # a brush stroke across the worm, then release
    cx, cy = win.worms[1].centroid(4)
    win.zoomv.mode = win.zoomv.MODE_DRAW
    win.zoomv.brush = 6
    win._on_paint(cx - 10, cy - 6)
    win._on_paint(cx + 4, cy + 3)
    win._on_paint(cx + 12, cy + 9)
    win._on_stroke_finished()
    out["after_stroke"] = win._overlay.copy()
    out["metrics_after_stroke"] = {k: dict(v)
                                   for k, v in win.worms[1].metrics.items()}

    # erase a bit
    win.zoomv.mode = win.zoomv.MODE_ERASE
    win._on_paint(cx, cy)
    win._on_stroke_finished()
    out["mask_after_erase"] = win.masks.get(win.worms[1].obj_id, 4).copy()

    # undo / redo
    win._undo_action()
    out["mask_after_undo"] = win.masks.get(win.worms[1].obj_id, 4).copy()
    win._redo_action()
    out["mask_after_redo"] = win.masks.get(win.worms[1].obj_id, 4).copy()

    # overlay toggles -> zoom image content
    win.outline_btn.setChecked(True)
    win.onion_btn.setChecked(True)
    out["zoom_toggles"] = captured[-1].copy()
    win.outline_btn.setChecked(False)
    win.onion_btn.setChecked(False)

    # propagate the mask forward, clicking on the video, playback stepping
    win._propagate()
    out["frame_after_propagate"] = win.frame
    out["mask_next_frame"] = win.masks.get(win.worms[1].obj_id, 5).copy()
    win._on_video_click(*win.worms[2].centroid(win.frame))
    out["selected_after_click"] = win.selected.obj_id
    win._step(1)
    win._step(-1)
    out["frame_after_steps"] = win.frame

    # worm list: add, draw into it, reassign, delete
    win._add_worm()
    new_id = win.selected.obj_id
    win.zoomv.mode = win.zoomv.MODE_DRAW
    win._on_paint(40.0, 40.0)
    win._on_paint(52.0, 46.0)
    win._on_stroke_finished()
    out["new_worm_area"] = win.selected.metrics[win.area_metric][win.frame]
    win._do_reassign(win.selected, new_id, 91)
    out["ids_after_reassign"] = [w.obj_id for w in win.worms]
    out["mask_after_reassign"] = win.masks.get(91, win.frame).copy()

    # units, refresh, info text, title
    win.px_to_um, win.fps = 0.65, 12.0
    win.plot.set_scales(win._unit_scales())
    win._refresh_plots()
    win._update_info()
    out["info_text"] = win.info.text()
    out["title"] = win.windowTitle()
    out["plot_specs"] = win._plot_specs()
    out["unit_scales"] = win._unit_scales()

    # export a composited frame + write the CSV
    out["export_frame"] = win._export_frame(4) if hasattr(win, "_export_frame") \
        else None
    win._write_csv()
    out["csv"] = csv_path.read_text()

    out["all_metrics"] = {w.obj_id: {k: dict(v) for k, v in w.metrics.items()}
                          for w in win.worms}
    out["centroids"] = {w.obj_id: dict(w.centroids) for w in win.worms}
    win.close()
    return out


@pytest.fixture(scope="module")
def both_versions(qapp, tmp_path_factory):
    """Drive the original and this package, and return the two result dicts.

    `qapp` is not decoration: constructing a QWidget with no live QApplication
    aborts the interpreter outright, and the reference has to be held by
    something for the whole run.
    """
    import importlib.util
    import os
    import sys

    path = None
    for candidate in (os.environ.get("WORMSCOPE_ORIGINAL"),
                      "wormscope_gui3.py",
                      "/mnt/user-data/uploads/wormscope/wormscope_gui3.py"):
        if candidate and os.path.isfile(candidate):
            path = candidate
            break
    if path is None:
        pytest.skip("wormscope_gui3.py not found")

    tmp = tmp_path_factory.mktemp("parity")

    spec = importlib.util.spec_from_file_location("orig_gui", path)
    orig = importlib.util.module_from_spec(spec)
    sys.modules["orig_gui"] = orig
    spec.loader.exec_module(orig)

    def patch_orig(store):
        real = orig.np_to_qimage

        def spy(arr):
            store.append(np.array(arr))
            return real(arr)
        orig.np_to_qimage = spy

    old = drive(orig.MainWindow, orig.make_demo, patch_orig, tmp / "old.csv")

    from wormscope.io.demo import make_demo
    from wormscope.io.export import export_frame
    from wormscope.ui import MainWindow, rendering

    def patch_new(store):
        real = rendering.np_to_qimage

        def spy(arr):
            store.append(np.array(arr))
            return real(arr)
        rendering.np_to_qimage = spy

    def make_window(ds):
        win = MainWindow(ds)
        # the export helper moved out of the window; expose it for the script
        win._export_frame = lambda f: export_frame(win.frames, win.worms,
                                                   win.masks, f)
        return win

    real_np_to_qimage = rendering.np_to_qimage
    new = drive(make_window, make_demo, patch_new, tmp / "new.csv")
    rendering.np_to_qimage = real_np_to_qimage
    return old, new


OBSERVATIONS = [
    "overlay_frame4", "after_stroke", "metrics_after_stroke",
    "mask_after_erase", "mask_after_undo", "mask_after_redo", "zoom_toggles",
    "frame_after_propagate", "mask_next_frame", "selected_after_click",
    "frame_after_steps", "new_worm_area", "ids_after_reassign",
    "mask_after_reassign", "info_text", "title", "plot_specs", "unit_scales",
    "export_frame", "csv", "all_metrics", "centroids",
]


@pytest.mark.parametrize("observation", OBSERVATIONS)
def test_identical_to_the_original(both_versions, observation):
    old, new = both_versions
    a, b = old.get(observation), new.get(observation)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert a is not None and b is not None and np.array_equal(a, b)
    else:
        assert a == b


def test_the_script_covered_everything_it_claims(both_versions):
    """A guard on the parity list: if the driver stops recording something,
    the parametrized tests above would silently pass on two Nones."""
    old, _new = both_versions
    assert sorted(old) == sorted(OBSERVATIONS)
