"""Layer 1: the shape of the code.

Everything listed here is a name something else depends on — another module, a
mixin, the tests, or you six months from now. These tests do not care *how* any
of it works; they only notice when a piece stops existing or changes shape.

**This file is meant to be edited.** If you deliberately rename ``_sam_propagate``
or drop a button, the failure here is not a bug report, it is a checklist:
update the table, and the failure tells you every other place that name is
still expected. If you find yourself deleting half of a table to make a change
pass, that is the signal the change was bigger than you thought.

Run just this layer:  pytest -m contract
"""

import ast
import importlib
import inspect
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "wormscope"


# --------------------------------------------------------------------------- #
# Public names, module by module
# --------------------------------------------------------------------------- #
PUBLIC_NAMES = {
    "wormscope.core.models": [
        "Worm", "FrameSource", "MaskStore", "Dataset", "empty_dataset"],
    "wormscope.core.geometry": [
        "HAVE_SKIMAGE", "skeleton_length_px", "skeletonize_crop", "mask_stats",
        "stamp_disk", "stamp_stroke"],
    "wormscope.core.kinematics": [
        "worm_speed", "worm_acceleration", "refresh_kinematics"],
    "wormscope.core.metrics": [
        "apply_mask_stats", "recompute_worm", "unit_scales", "plot_specs"],
    "wormscope.core.utils": [
        "moving_average", "color_for", "last_int", "list_images",
        "sort_by_frame_number", "find_col", "to_num"],
    "wormscope.render": [
        "thicken", "mask_border", "blend", "blend_all_masks", "stamp_marker"],
    "wormscope.io.loaders": [
        "find_frames_dir", "load_frames", "load_tracks_from_masks",
        "load_metrics", "seed_metrics_from_masks", "load_dataset",
        "build_raw_dataset"],
    "wormscope.io.writers": ["write_measurements_csv", "save_measurements_plot"],
    "wormscope.io.export": ["export_frame", "export_overlay"],
    "wormscope.io.demo": ["make_demo"],
    "wormscope.settings": [
        "settings_path", "load", "save", "sam2_paths", "sam2_device",
        "save_sam2_paths"],
    "wormscope.segmentation": [
        "SEGMENTER_FACTORY", "make_segmenter", "Sam2Backend",
        "MissingSam2Paths", "hydra_config_name", "pair_warning",
        "sam2_config_dir", "prepare_jpeg_dir"],
    "wormscope.segmentation.feature_cache": ["FeatureCache"],
    "wormscope.widgets": [
        "PlotPanel", "VideoView", "ZoomView", "np_to_qimage", "HAVE_MPL"],
    "wormscope.ui": ["MainWindow", "WINDOWS"],
    "wormscope.app": ["resolve_dataset", "run"],
    "wormscope.theme": ["PALETTE", "stylesheet", "apply_theme"],
}


@pytest.mark.parametrize("module_name", sorted(PUBLIC_NAMES))
def test_module_exports_its_public_names(module_name):
    mod = importlib.import_module(module_name)
    missing = [n for n in PUBLIC_NAMES[module_name] if not hasattr(mod, n)]
    assert not missing, (
        f"{module_name} no longer provides {missing}. Either put them back or "
        f"update PUBLIC_NAMES in this file — and then fix every caller the "
        f"other failures point at.")


# --------------------------------------------------------------------------- #
# Signatures that other code calls positionally
# --------------------------------------------------------------------------- #
SIGNATURES = {
    "wormscope.core.geometry:mask_stats": ["mask"],
    "wormscope.core.geometry:stamp_disk": ["mask", "cx", "cy", "r", "value"],
    "wormscope.core.metrics:apply_mask_stats":
        ["worm", "masks", "idx", "area_metric", "length_metric"],
    "wormscope.core.metrics:unit_scales":
        ["area_metric", "length_metric", "px_to_um", "fps"],
    "wormscope.render:blend": ["arr", "mask_full", "bbox", "color", "alpha"],
    "wormscope.render:blend_all_masks":
        ["arr", "worms", "masks", "frame", "sel_id", "outline_only"],
    "wormscope.io.writers:write_measurements_csv":
        ["csv_path", "worms", "idx_to_key", "area_metric", "length_metric",
         "px_to_um", "fps"],
    "wormscope.io.export:export_frame": ["frames", "worms", "masks", "f"],
    "wormscope.segmentation:make_segmenter": ["frames", "shape"],
}


@pytest.mark.parametrize("target", sorted(SIGNATURES))
def test_call_signatures_are_stable(target):
    module_name, func_name = target.split(":")
    fn = getattr(importlib.import_module(module_name), func_name)
    params = list(inspect.signature(fn).parameters)
    assert params == SIGNATURES[target], (
        f"{target} takes {params} now. Callers pass these positionally, so "
        f"reordering or renaming them breaks code the tests don't reach.")


# --------------------------------------------------------------------------- #
# The data model
# --------------------------------------------------------------------------- #
MASKSTORE_METHODS = [
    "register", "get", "set", "has_mask", "frames_of", "skel", "save",
    "pop_worm", "restore_worm", "rename_worm", "set_dirty", "skel_path_for"]


def test_maskstore_keeps_its_methods():
    from wormscope.core.models import MaskStore
    missing = [m for m in MASKSTORE_METHODS if not hasattr(MaskStore, m)]
    assert not missing, f"MaskStore lost {missing}"


def test_maskstore_dirty_is_a_set_of_pairs():
    """`dirty` is read directly by Save and by the window title, so its type is
    part of the contract, not an implementation detail."""
    from wormscope.core.models import MaskStore
    store = MaskStore((4, 4))
    assert store.dirty == set()
    store.set(1, 0, __import__("numpy").ones((4, 4), bool))
    assert store.dirty == {(1, 0)}


def test_worm_is_frame_indexed_and_sparse():
    from wormscope.core.models import Worm
    w = Worm(obj_id=3)
    assert (w.centroid(0), w.bbox(0)) == (None, None), \
        "a worm must be allowed to be absent from a frame"
    assert w.metrics == {} and w.obj_id == 3


def test_dataset_fields():
    """Constructed positionally in loaders, demo and the tests."""
    from wormscope.core.models import Dataset
    assert [f.name for f in Dataset.__dataclass_fields__.values()] == [
        "frames", "worms", "metric_names", "masks", "idx_to_key", "csv_path",
        "title", "frames_dir"]


# --------------------------------------------------------------------------- #
# The segmentation backend protocol
# --------------------------------------------------------------------------- #
BACKEND_PROTOCOL = ["add_points", "add_mask", "propagate", "reset", "forget",
                    "session", "has_prompts", "prompted_objects"]


def test_sam2_backend_implements_the_protocol():
    """The list a replacement model has to satisfy. `SEGMENTER_FACTORY` is the
    swap point; this is what the GUI will call on whatever it returns."""
    from wormscope.segmentation import Sam2Backend
    missing = [m for m in BACKEND_PROTOCOL if not hasattr(Sam2Backend, m)]
    assert not missing, f"Sam2Backend lost {missing}"


def test_the_stub_backend_satisfies_it_too(stub_sam):
    """Proof the protocol above is really all the GUI needs — the whole SAM2
    test suite runs against a stand-in with nothing else in it."""
    missing = [m for m in BACKEND_PROTOCOL if not hasattr(stub_sam, m)]
    assert not missing, f"the test stub is missing {missing}"


# --------------------------------------------------------------------------- #
# Widgets and signals
# --------------------------------------------------------------------------- #
WIDGET_SIGNALS = {
    "wormscope.widgets:VideoView": ["imageClicked"],
    "wormscope.widgets:ZoomView": [
        "paintRequested", "strokeFinished", "viewportChanged", "pointAdded"],
    "wormscope.widgets:PlotPanel": ["frameSelected"],
    "wormscope.ui.progress:ProgressPanel": ["stopRequested"],
    "wormscope.ui.sam_worker:SamWorker": [
        "built", "needsFiles", "promptDone", "frameTracked", "jobProgress",
        "jobDone", "jobFailed"],
}


@pytest.mark.parametrize("target", sorted(WIDGET_SIGNALS))
def test_signals_still_exist(target):
    module_name, cls_name = target.split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    missing = [s for s in WIDGET_SIGNALS[target] if not hasattr(cls, s)]
    assert not missing, (
        f"{cls_name} no longer emits {missing}. Signals are how the widgets "
        f"talk to the window — a missing one is a silently dead feature.")


def test_zoom_view_modes():
    """The tool names are compared against by string in editing.py and worms.py."""
    from wormscope.widgets import ZoomView
    assert (ZoomView.MODE_PAN, ZoomView.MODE_DRAW, ZoomView.MODE_ERASE,
            ZoomView.MODE_SAM_POS, ZoomView.MODE_SAM_NEG) == \
        ("pan", "draw", "erase", "sam+", "sam-")


# --------------------------------------------------------------------------- #
# The window: mixins, methods and named widgets
# --------------------------------------------------------------------------- #
MIXIN_METHODS = {
    "LayoutMixin": ["_build_ui", "_build_video_panel", "_build_zoom_panel",
                    "_build_labelled_panel", "_build_plots_panel",
                    "_build_shortcuts", "_assemble"],
    "RenderMixin": ["_render_overlay", "_refresh_sel_skel", "_toggle_onion",
                    "_toggle_outline", "_fit_view"],
    "EditingMixin": ["_set_mode", "_set_brush", "_set_zoom", "_on_paint",
                     "_on_stroke_finished", "_recompute", "_refresh_plots",
                     "_propagate"],
    "HistoryMixin": ["_push_action", "_undo_action", "_redo_action",
                     "_restore_mask"],
    "WormListMixin": ["_insert_worm", "_remove_worm_obj", "_add_worm",
                      "_delete_worm", "_reassign_id", "_do_reassign"],
    "SamMixin": ["_start_sam", "_shutdown_sam", "_ensure_segmenter",
                 "_choose_sam2_files", "_on_sam_point", "_sam_mask_prompt",
                 "_sam_propagate", "_sam_resegment", "_sam_clear",
                 "_on_frame_tracked", "_protected"],
    "IOActionsMixin": ["_open_folder", "_import_raw", "_set_units", "_save",
                       "_write_csv", "_export_video", "_unit_scales",
                       "_plot_specs"],
}


def test_main_window_is_built_from_the_documented_mixins():
    from wormscope.ui import MainWindow
    bases = [b.__name__ for b in MainWindow.__bases__]
    assert bases == list(MIXIN_METHODS) + ["QMainWindow"], (
        "the window's bases changed. That is allowed — but the docstring at "
        "the top of ui/main_window.py is a map people navigate by, so keep "
        "the two in step.")


@pytest.mark.parametrize("mixin", sorted(MIXIN_METHODS))
def test_each_mixin_keeps_its_methods(mixin):
    from wormscope.ui import MainWindow
    cls = next(b for b in MainWindow.__bases__ if b.__name__ == mixin)
    missing = [m for m in MIXIN_METHODS[mixin] if not hasattr(cls, m)]
    assert not missing, f"{mixin} lost {missing}"


# Widgets the behaviour tests (and your muscle memory) reach for by name.
WINDOW_WIDGETS = {
    # video panel
    "video": "VideoView", "prev_btn": "QPushButton", "next_btn": "QPushButton",
    "play_btn": "QPushButton", "slider": "QSlider", "frame_label": "QLabel",
    "export_btn": "QPushButton",
    # zoom panel: tools
    "zoomv": "ZoomView", "btn_pan": "QPushButton", "btn_draw": "QPushButton",
    "btn_erase": "QPushButton", "btn_sam_pos": "QPushButton",
    "btn_sam_neg": "QPushButton", "brush_slider": "QSlider",
    "outline_btn": "QPushButton", "onion_btn": "QPushButton",
    "prop_btn": "QPushButton", "lock_box": "QCheckBox",
    # zoom panel: SAM2
    "sam_run_btn": "QPushButton", "sam_again_btn": "QPushButton",
    "sam_mask_btn": "QPushButton", "sam_clear_btn": "QPushButton",
    "back_box": "QCheckBox",
    # zoom panel: view + history
    "zoom_slider": "QSlider", "fit_btn": "QPushButton",
    "undo_btn": "QPushButton", "redo_btn": "QPushButton",
    "save_btn": "QPushButton", "progress_panel": "ProgressPanel",
    # worm list
    "list": "QListWidget", "add_btn": "QPushButton", "del_btn": "QPushButton",
    "id_btn": "QPushButton", "open_btn": "QPushButton",
    "import_btn": "QPushButton", "info": "QLabel",
    # plots
    "plot": "PlotPanel", "smooth_box": "QCheckBox", "units_btn": "QPushButton",
    "refresh_btn": "QPushButton",
}


def test_window_has_every_named_widget(window):
    missing = {name: kind for name, kind in WINDOW_WIDGETS.items()
               if not hasattr(window, name)}
    assert not missing, (
        f"these widgets are gone: {sorted(missing)}. Removing a control is a "
        f"real decision — delete it from WINDOW_WIDGETS too, and check nothing "
        f"in _show_empty_state still refers to it.")


def test_named_widgets_are_the_expected_kind(window):
    wrong = {}
    for name, kind in WINDOW_WIDGETS.items():
        got = type(getattr(window, name)).__name__
        if got != kind:
            wrong[name] = f"{kind} -> {got}"
    assert not wrong, f"widget types changed: {wrong}"


def test_state_the_sam_and_editing_code_shares(window):
    """These attributes are written by one mixin and read by another, which
    makes them the closest thing the window has to a public interface."""
    for name in ("frame", "n", "selected", "worms", "masks", "by_id",
                 "_hand_edited", "_anchored", "_edited_during_run",
                 "_prompted_worms", "_sam_points", "_undo", "_redo",
                 "area_metric", "length_metric", "plot_metrics"):
        assert hasattr(window, name), f"MainWindow lost self.{name}"


# --------------------------------------------------------------------------- #
# Architecture rules
# --------------------------------------------------------------------------- #
QT_FREE = ["config.py", "render.py", "settings.py", "core", "io"]


def _imports(path: Path):
    tree = ast.parse(path.read_text(), str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


@pytest.mark.parametrize("where", QT_FREE)
def test_the_qt_free_half_stays_qt_free(where):
    """core/, io/, config and render must be importable without a GUI.

    This is what lets metrics and loaders be used from a plain script, and it
    is why most of this suite needs no QApplication. Reach for a QWidget in
    here and the layering is gone."""
    target = PKG / where
    files = [target] if target.is_file() else sorted(target.rglob("*.py"))
    offenders = {p.name: sorted(m for m in _imports(p) if m.startswith("PyQt"))
                 for p in files}
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, (
        f"{where} imports Qt: {offenders}. Put the widget in ui/ or widgets/ "
        f"and keep this half scriptable.")


def test_ui_never_imports_widgets_backwards():
    """widgets/ reports what the user did; ui/ decides what it means. A widget
    importing the main window would make that loop, and the window unusable on
    its own."""
    offenders = {}
    for p in sorted((PKG / "widgets").rglob("*.py")):
        bad = [m for m in _imports(p) if ".ui" in m or m.endswith("main_window")]
        if bad:
            offenders[p.name] = bad
    assert not offenders, f"widgets/ imports back into ui/: {offenders}"


def test_config_has_the_constants_other_modules_read():
    from wormscope import config
    expected = [
        "PLAYBACK_FPS", "IMAGE_EXTS", "MAX_PLOTS", "SPEED_METRIC",
        "ACCEL_METRIC", "KINEMATIC_METRICS", "SMOOTH_WINDOW", "ZOOM_MIN",
        "ZOOM_MAX", "UNDO_LIMIT", "HIT_RADIUS", "BRUSH_MIN", "BRUSH_MAX",
        "BRUSH_DEFAULT", "NEW_WORM_TOOL", "SAM2_EMBED_CACHE", "MASK_ALPHA",
        "MASK_SEL_RGB", "MASK_UNSEL_RGB", "SKEL_RGB", "ONION_RGB",
        "ONION_ALPHA", "LAST_INT_RE", "WORM_RE"]
    missing = [n for n in expected if not hasattr(config, n)]
    assert not missing, f"config.py lost {missing}"


def test_the_stylesheet_placeholders_all_have_colours():
    """style.qss is a Template: a `$name` with no entry in theme.PALETTE raises
    at startup, which is a crash rather than a wrong colour."""
    from wormscope import theme
    qss = theme.stylesheet()                 # substitute() raises if one is missing
    assert "$" not in qss, "a placeholder survived substitution"
