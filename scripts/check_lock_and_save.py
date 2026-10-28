# ruff: noqa: E401, E402, E701, E702  (a compact test script: setup must run before the imports)
"""End-to-end check of 'lock edits' with a stand-in segmenter (no SAM2 needed).
Usage: python scripts/check_lock_and_save.py   (from the project folder)
Needs no SAM2: a stand-in segmenter plays the tracker."""
import os, sys, time, tempfile, shutil
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from PyQt6.QtWidgets import QApplication, QMessageBox
app = QApplication([])
QMessageBox.warning = staticmethod(lambda *a, **k: print("[msgbox]", a[1:3]))
QMessageBox.information = staticmethod(lambda *a, **k: print("[msgbox]", a[1:3]))

from wormscope import segmentation, settings
H, W, N = 60, 80, 10
TRACK = np.zeros((H, W), bool); TRACK[20:40, 10:70] = True      # the tracker's guess

class FakeBackend:
    def __init__(self, frames, shape): self.anchors = {}
    def add_points(self, f, oid, pts, labels):
        m = np.zeros((H, W), bool); x, y = map(int, pts[0]); m[y-5:y+5, x-5:x+5] = True
        self.anchors[(oid, f)] = m; return m
    def set_points(self, *a): pass
    def add_mask(self, f, oid, mask):
        self.anchors[(oid, f)] = mask.copy(); return binary_erosion(mask)
    def propagate(self, oid, start_frame_idx=None, reverse=False):
        s = 0 if start_frame_idx is None else start_frame_idx
        rng = range(s, -1, -1) if reverse else range(s, N)
        for f in rng:
            time.sleep(0.002)
            a = self.anchors.get((oid, f))
            # like SAM2: on a prompt frame it returns ITS version of your mask
            yield f, (binary_erosion(a) if a is not None else TRACK.copy())
    def forget(self, oid): pass
    def reset(self): self.anchors.clear()
    def prewarm(self, *a, **k): return 0
segmentation.SEGMENTER_FACTORY = FakeBackend
settings.sam2_paths = lambda: (None, None, "test")

from wormscope.io.loaders import build_raw_dataset, load_dataset
from wormscope.ui.main_window import MainWindow, WINDOWS
from wormscope.widgets import ZoomView

tmp = Path(tempfile.mkdtemp()); raw = tmp / "clip"; raw.mkdir()
for i in range(N):
    Image.fromarray(np.full((H, W, 3), 40 + i, np.uint8)).save(raw / f"img_{i:03d}.png")

def wait(w):
    for _ in range(3000):
        app.processEvents()
        if w.sam is not None and not w.sam.busy and not w._point_jobs:
            app.processEvents(); return
        time.sleep(0.002)
    raise TimeoutError

def paint(w, frame, box):
    w.set_frame(frame); w._set_mode(ZoomView.MODE_DRAW)
    w.zoomv.brush = 3
    for x in range(box[0], box[2], 2):
        w._on_paint(x, box[1])
    w._on_stroke_finished()
    return w.masks.get(w.selected.obj_id, frame).copy()

ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)

w = MainWindow(build_raw_dataset(raw)); w.show(); app.processEvents()
oid = w.selected.obj_id
w._on_sam_point(40, 30, 1); wait(w)
w._sam_propagate(); wait(w)
check("segment all fills every frame",
      all((w.masks.get(oid, f) == TRACK).all() for f in range(1, N)))
H5 = paint(w, 5, (5, 50, 75))                       # hand fix on frame 5
H7 = paint(w, 7, (5, 52, 75))                       # and frame 7

w.lock_box.setChecked(True)
w.set_frame(3); w._sam_resegment(); wait(w)
check("lock ON: hand-fixed frame 5 kept exactly", (w.masks.get(oid, 5) == H5).all())
check("lock ON: hand-fixed frame 7 kept exactly", (w.masks.get(oid, 7) == H7).all())
check("lock ON: unedited frame 6 re-tracked", (w.masks.get(oid, 6) == TRACK).all())

w.lock_box.setChecked(False)
w.set_frame(3); w._sam_resegment(); wait(w)
check("lock OFF: frame 5 gets SAM2's version (may differ)", not (w.masks.get(oid, 5) == H5).all())

# edits made DURING a run always win
H8 = None
w.lock_box.setChecked(False)
w.set_frame(0); w._sam_propagate(start=0, worms=[oid])
for _ in range(3): app.processEvents()
H8 = paint(w, 8, (5, 10, 75)); wait(w)
check("edit during a run is kept (lock off)", (w.masks.get(oid, 8) == H8).all())

# undo of a stroke: is the frame still treated as hand-edited?
w.lock_box.setChecked(True)
H2 = paint(w, 2, (5, 45, 75)); w._undo_action()
check("undo of a stroke un-marks the frame", (oid, 2) not in w._hand_edited)

# reassign ID keeps the lock
H9 = paint(w, 9, (5, 48, 75))
w._do_reassign(w.selected, oid, 42); nid = 42
w.set_frame(0); w._sam_propagate(start=0, worms=[nid]); wait(w)
check("after Reassign ID, the worm can still be re-tracked", nid in w._tracked)
check("after Reassign ID, locked frame 9 kept", (w.masks.get(nid, 9) == H9).all())

# save and continue editing: is what was saved what's needed?
w.px_to_um, w.fps = 0.5, 10.0
w._save(); app.processEvents()
out = Path(w.ds.csv_path).parent
print("saved folder:", sorted(p.name for p in out.iterdir()))
ds2 = load_dataset(out)
check("saved folder opens with 'Continue editing video'", ds2 is not None)
if ds2 is not None:
    w2 = MainWindow(ds2); app.processEvents()
    check("same number of frames after reopening", w2.n == N)
    check("masks identical after reopening",
          all((w2.masks.get(nid, f) == w.masks.get(nid, f)).all() for f in range(N)))
    check("corrected frames remembered after reopening (lock still applies)",
          (nid, 9) in w2._hand_edited | w2._anchored)
    check("lock-edits setting remembered", w2.lock_box.isChecked())
    check("calibration remembered", (w2.px_to_um, w2.fps) == (0.5, 10.0))
    check("nothing marked unsaved right after reopening", not w2._has_unsaved())
    # and lock really protects the reopened corrections
    w2._on_sam_point(40, 30, 1); wait(w2)
    w2._sam_propagate(); wait(w2)
    check("after reopening, locked frame 9 survives Segment all",
          (w2.masks.get(nid, 9) == H9).all())
    # re-uploading the same raw frames starts fresh, next to the saved work
    from wormscope.io.loaders import raw_output_dir
    check("re-uploading the same raw frames never reuses the saved folder",
          raw_output_dir(raw) != out and raw_output_dir(raw).name.endswith("_2"))
    w2.sam and w2.sam.shutdown()
for x in WINDOWS: pass
w.sam and w.sam.shutdown()
shutil.rmtree(tmp)
print("ALL PASS" if ok else "SOME FAILED")
