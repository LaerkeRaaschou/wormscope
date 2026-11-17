# ruff: noqa: E401, E402, E701, E702  (a compact test script: setup must run before the imports)
"""End-to-end check of the manual length tool, manual_lengths.csv and scale.png.
Usage: python scripts/check_manual_length.py   (from the project folder)
Needs no SAM2: measuring by hand works on raw frames without any masks."""
import os, sys, csv, json, math, tempfile
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["HOME"] = tempfile.mkdtemp()        # own ~/.wormscope/settings.json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox
app = QApplication([])
QMessageBox.warning = staticmethod(lambda *a, **k: print("[msgbox]", a[1:3]))
QMessageBox.information = staticmethod(lambda *a, **k: print("[msgbox]", a[1:3]))
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)

from wormscope import segmentation, settings
class NoSam:                                   # SAM2 is never used here
    def __init__(self, frames, shape): pass
    def forget(self, oid): pass
    def reset(self): pass
    def prewarm(self, *a, **k): return 0
segmentation.SEGMENTER_FACTORY = NoSam
settings.sam2_paths = lambda: (None, None, "test")

from wormscope.core.manual import polyline_length
from wormscope.io.loaders import build_raw_dataset, load_dataset
from wormscope.ui.calibration_dialog import CalibrationDialog
from wormscope.ui.main_window import MainWindow
from wormscope.widgets import ZoomView

H, W, N = 60, 80, 6
tmp = Path(tempfile.mkdtemp()); raw = tmp / "clip"; raw.mkdir()
for i in range(N):
    Image.fromarray(np.full((H, W, 3), 40 + i, np.uint8)).save(raw / f"img_{i:03d}.png")
scale_src = tmp / "micrometer.tif"
Image.fromarray(np.full((H, W), 1000, np.uint16)).save(scale_src)

ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)

def click(view, x, y, mods=Qt.KeyboardModifier.NoModifier):
    """Left-click the Zoom canvas at image point (x, y)."""
    left, top = view._placement()
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, mods,
                     QPoint(round(left + x * view.zoom), round(top + y * view.zoom)))

w = MainWindow(build_raw_dataset(raw)); w.show(); app.processEvents()
w._start_sam = lambda *a, **k: None
folder = w.ds.csv_path.parent
oid = w.selected.obj_id
check("a fresh upload starts with an empty worm selected", w.selected is not None)
check("a first video starts in Automatic mode", not w.manual_mode
      and w.ai_box.isVisibleTo(w) and not w.measure_box.isVisibleTo(w))

# --- measure with real mouse clicks on frame 2 (no masks anywhere) ---
w.set_frame(2)
w.zoomv.frame_fit(); app.processEvents()
QTest.keyClick(w, Qt.Key.Key_L); app.processEvents()
check("L switches to Manual mode and arms Measure",
      w.manual_mode and w.zoomv.mode == ZoomView.MODE_MEASURE)
check("Manual mode hides the AI / brush tools, shows Manual length",
      not w.ai_box.isVisibleTo(w) and not w.brush_box.isVisibleTo(w)
      and w.measure_box.isVisibleTo(w))
for x, y in ((10, 10), (40, 10), (40, 50)):
    click(w.zoomv, x, y)
pts = w.selected.manual.get(2, [])
check("three clicks -> three points", len(pts) == 3)
exp = 30 + 40
check("length is the sum of the segments (~70 px)",
      abs(polyline_length(pts) - exp) < 3)
check("unsaved after measuring", w._has_unsaved() and w.save_btn.isEnabled())

# insert a point by clicking on the first segment
click(w.zoomv, 25, 10)
check("clicking on the line inserts a point there",
      len(w.selected.manual[2]) == 4 and abs(w.selected.manual[2][1][0] - 25) < 2)
# shift+click a point removes it
click(w.zoomv, 25, 10, Qt.KeyboardModifier.ShiftModifier)
check("Shift+click removes that point", len(w.selected.manual[2]) == 3)
# drag the last point
left, top = w.zoomv._placement(); z = w.zoomv.zoom
p0 = QPoint(round(left + 40 * z), round(top + 50 * z))
p1 = QPoint(round(left + 70 * z), round(top + 50 * z))
QTest.mousePress(w.zoomv, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p0)
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtCore import QPointF, QEvent
ev = QMouseEvent(QEvent.Type.MouseMove, QPointF(p1), QPointF(p1),
                 Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                 Qt.KeyboardModifier.NoModifier)
app.sendEvent(w.zoomv, ev)
QTest.mouseRelease(w.zoomv, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p1)
check("dragging a point moves it", abs(w.selected.manual[2][-1][0] - 70) < 2)
w._undo_action()
check("undo puts the dragged point back", abs(w.selected.manual[2][-1][0] - 40) < 2)
w._redo_action()
check("redo moves it again", abs(w.selected.manual[2][-1][0] - 70) < 2)
# backspace removes the last point
QTest.keyClick(w, Qt.Key.Key_Backspace); app.processEvents()
check("Backspace removes the last point", len(w.selected.manual[2]) == 2)
w._undo_action()
check("undo restores it", len(w.selected.manual[2]) == 3)
before_frame2 = list(w.selected.manual[2])

# right-drag pans, it never adds a point
NOMOD, RIGHT = Qt.KeyboardModifier.NoModifier, Qt.MouseButton.RightButton
QTest.mousePress(w.zoomv, RIGHT, NOMOD, QPoint(50, 50))
QTest.mouseRelease(w.zoomv, RIGHT, NOMOD, QPoint(50, 50))
check("right-click adds no point", w.selected.manual[2] == before_frame2)

# another frame (frame 4), API-level
w.set_frame(4)
w._on_measure_added(5, 30, -1); w._on_measure_added(65, 30, -1)
check("frame 4 measured (60 px)", abs(polyline_length(w.selected.manual[4]) - 60) < 1e-6)
check("info shows the manual length", "manual length" in w.info.text())

# Add a second worm while measuring: it keeps the Measure tool
w._add_worm()
check("Add keeps the Measure tool", w.zoomv.mode == ZoomView.MODE_MEASURE)
oid2 = w.selected.obj_id
w._on_measure_added(10, 50, -1); w._on_measure_added(30, 50, -1)
check("second worm measured on frame 4", len(w.selected.manual[4]) == 2)
check("other worm's line drawn as 'other'", len(w.zoomv.measure_others) == 1)
# picking a worm by its manual line on the Video
w._on_video_click(35, 30)
check("clicking a manual line on the Video selects that worm", w.selected.obj_id == oid)
# --- Automatic mode hides the manual data, keeps the worms ---
w.set_frame(4); w.select_worm(w.by_id[oid])
w.masks.get(oid, 1)[20:30, 20:30] = True           # a mask on frame 1
w._recompute(w.by_id[oid], 1)
w._set_annotation_mode("automatic")
check("Automatic mode: no manual lines drawn",
      w.zoomv.measure_active == [] and w.zoomv.measure_others == []
      and w.video._measure == [])
check("Automatic mode: no manual length in Info", "manual" not in w.info.text())
check("Automatic mode: plot hides manual lengths", not w.plot.show_manual)
check("Automatic mode: both worms still listed", w.list.count() == 2)
w.select_worm(w.by_id[oid2])
check("selecting a manual-only worm zooms to its (hidden) line",
      abs(w.zoomv.cx - 20) < 2 and abs(w.zoomv.cy - 50) < 2)
w.set_frame(1)
check("Automatic mode: masks are drawn",
      not (w._overlay == w.frames.get(1)).all())
w._set_annotation_mode("manual")
check("Manual mode: masks hidden (raw frame shown)",
      (w._overlay == w.frames.get(1)).all())
w.set_frame(4)
check("Manual mode: manual lines drawn again", len(w.zoomv.measure_active) == 2)
# undo of a manual edit made while in Automatic mode shows it
w._on_measure_added(50, 50, -1)
w._set_annotation_mode("automatic")
w._undo_action()
check("undoing a manual change switches to Manual mode to show it",
      w.manual_mode and len(w.by_id[oid2].manual[4]) == 2)
w.select_worm(w.by_id[oid])
# a single point is not a measurement
w.set_frame(5); w._on_measure_added(3, 3, -1)
w.plot.set_worm(w.selected)          # plot with manual overlay must not crash

# --- calibrate from a scale image file ---
w._calibration = {"image": str(scale_src), "source": "file",
                  "line": ((0, 0), (50, 0)), "length": 100.0, "unit": "µm"}
w.px_to_um = 2.0
w._settings_changed()
w.set_frame(4)
check("measure label in µm after calibration", "µm" in w.measure_label.text())

# --- save ---
w._save()
check("nothing unsaved after Save", not w._has_unsaved())
names = sorted(p.name for p in folder.iterdir())
print("saved folder:", names)
check("scale.png saved in the data folder", (folder / "scale.png").is_file())
im = Image.open(folder / "scale.png")
check("scale.png is a PNG of the scale image", im.format == "PNG" and im.size == (W, H))
sess = json.loads((folder / "wormscope.json").read_text())
dlg_state = sess["calibration"]["dialog"]
check("wormscope.json points at scale.png (relative)", dlg_state["image"] == "scale.png")
check("wormscope.json remembers the mode", sess["annotation_mode"] == "manual")
check("original scale image path kept", dlg_state.get("original_image") == str(scale_src))
rows = list(csv.DictReader(open(folder / "manual_lengths.csv")))
print("manual_lengths.csv:"); print(open(folder / "manual_lengths.csv").read())
check("one row per measured worm+frame (single points skipped)", len(rows) == 3)
check("columns", list(rows[0]) == ["frame", "worm_id", "length_px", "length_um",
                                    "n_points", "points"])
r2 = [r for r in rows if r["frame"] == "2"][0]
check("length_um = length_px × µm/px",
      math.isclose(float(r2["length_um"]), float(r2["length_px"]) * 2.0, rel_tol=1e-3))
check("frame is the original frame number, worm_id matches",
      {(r["frame"], r["worm_id"]) for r in rows}
      == {("2", str(oid)), ("4", str(oid)), ("4", str(oid2))})

# --- reopen with Continue editing ---
ds = load_dataset(folder)
w2 = MainWindow(ds); w2._start_sam = lambda *a, **k: None
w2.show(); app.processEvents()
by = {x.obj_id: x for x in w2.worms}
check("worms that only have manual lengths come back", oid in by and oid2 in by)
old_pts, new_pts = w.by_id[oid].manual[2], by[oid].manual[2]
check("points restored", len(new_pts) == 3
      and all(math.isclose(a, b, abs_tol=0.01)
              for p, q in zip(new_pts, old_pts) for a, b in zip(p, q)))
check("nothing unsaved after reopening", not w2._has_unsaved())
check("reopens in the mode it was saved in", w2.manual_mode)
check("calibration image resolved to the folder's scale.png",
      Path(w2._calibration["image"]) == folder / "scale.png")
dlg = CalibrationDialog(w2, px_to_um=w2.px_to_um, state=w2._calibration)
check("Units… dialog reopens with scale.png loaded",
      dlg._image_path and Path(dlg._image_path).name == "scale.png" and dlg.view.has_image())
dlg.close()
saved_cal = w2._calibration
w2._calibration = {"image": str(tmp / "gone.tif"), "line": ((0, 0), (9, 0))}
st = w2._calibration_state()
check("Units… opens the folder's scale.png when no scale image is set",
      Path(st["image"]) == folder / "scale.png" and "line" not in st)
w2._calibration = {}
check("...also with no calibration at all",
      Path(w2._calibration_state()["image"]) == folder / "scale.png")
w2._calibration = saved_cal
# saving again doesn't duplicate or lose scale.png
w2._settings_changed(); w2._save()
check("re-save keeps scale.png", (folder / "scale.png").is_file())
sess2 = json.loads((folder / "wormscope.json").read_text())
check("re-save keeps relative path + original",
      sess2["calibration"]["dialog"]["image"] == "scale.png"
      and sess2["calibration"]["dialog"].get("original_image") == str(scale_src))

# reassign ID moves the manual lengths, delete removes them
w2.select_worm(by[oid2]); w2._do_reassign(by[oid2], oid2, 7)
w2._save()
rows = list(csv.DictReader(open(folder / "manual_lengths.csv")))
check("Reassign ID renames the manual rows", {r["worm_id"] for r in rows} == {str(oid), "7"})
w2.select_worm(w2.by_id[7]); w2._delete_worm()
check("deleting a worm with manual lengths is unsaved", w2._has_unsaved())
w2._undo_action()
check("undo of delete brings its lengths back", 7 in w2.by_id and w2.by_id[7].manual)
w2._redo_action(); w2._save()
rows = list(csv.DictReader(open(folder / "manual_lengths.csv")))
check("deleted worm's rows gone", {r["worm_id"] for r in rows} == {str(oid)})
# clear all -> file has only a header (no stale rows)
w2.select_worm(w2.by_id[oid])
for f in list(w2.by_id[oid].manual):
    w2.set_frame(f); w2._measure_clear()
w2._save()
rows = list(csv.DictReader(open(folder / "manual_lengths.csv")))
check("clearing every line leaves an empty manual_lengths.csv", rows == [])

# a fresh upload of the same frames never reuses the folder with manual work
w3 = build_raw_dataset(raw)
check("re-upload doesn't reuse the saved folder", w3.csv_path.parent != folder)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
