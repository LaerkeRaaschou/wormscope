# ruff: noqa: E401, E402, E701, E702  (a compact test script: setup must run before the imports)
"""End-to-end check of Save as…: the whole video moves to another folder.
Usage: python scripts/check_save_as.py   (from the project folder)"""
import os, sys, csv, json, tempfile
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["HOME"] = tempfile.mkdtemp()        # own ~/.wormscope/settings.json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import QApplication, QMessageBox
app = QApplication([])
QMessageBox.warning = staticmethod(lambda *a, **k: print("[msgbox]", a[1:3]))
QMessageBox.information = staticmethod(lambda *a, **k: print("[msgbox]", a[1:3]))

from wormscope import segmentation
class NoSam:                                   # SAM2 is never used here
    def __init__(self, frames, shape): pass
    def forget(self, oid): pass
    def reset(self): pass
    def prewarm(self, *a, **k): return 0
segmentation.SEGMENTER_FACTORY = NoSam

from wormscope.io.loaders import build_raw_dataset, load_dataset
from wormscope.ui import io_actions
from wormscope.ui.io_actions import save_as_target
from wormscope.ui.main_window import MainWindow

H, W, N = 40, 60, 5
tmp = Path(tempfile.mkdtemp()); raw = tmp / "clip"; raw.mkdir()
for i in range(N):
    Image.fromarray(np.full((H, W, 3), 40 + i, np.uint8)).save(raw / f"img_{i:03d}.png")
scale_src = tmp / "micrometer.png"
Image.fromarray(np.full((H, W), 200, np.uint8)).save(scale_src)

ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)

w = MainWindow(build_raw_dataset(raw)); w._start_sam = lambda *a, **k: None
w.show(); app.processEvents()
old = w._save_folder()
check("the save folder is shown", w.save_folder_label.text() == f"→ {old.name}")
oid = w.selected.obj_id
for f in range(N):                                 # masks on every frame
    m = w.masks.get(oid, f); m[10:20, 5 + f:40 + f] = True
    w.masks.set_dirty(oid, f); w._recompute(w.selected, f)
w._set_annotation_mode("manual")
w.set_frame(2); w._on_measure_added(5, 5, -1); w._on_measure_added(35, 5, -1)
w._calibration = {"image": str(scale_src), "source": "file",
                  "line": ((0, 0), (10, 0)), "length": 20.0, "unit": "µm"}
w.px_to_um = 2.0
w._save()
before = {p.name for p in old.rglob("*") if p.is_file()}
# an edit after the first save: saved only by Save as
w.set_frame(3); w._on_measure_added(1, 1, -1); w._on_measure_added(11, 1, -1)
w._set_annotation_mode("automatic")
m = w.masks.get(oid, 4); m[30:35, 0:10] = True
w.masks.set_dirty(oid, 4); w._recompute(w.selected, 4)
edited = m.copy()

# --- Save as into an empty folder: used as is ---
empty = tmp / "elsewhere" / "target"; empty.mkdir(parents=True)
io_actions.pick_folder = lambda *a, **k: str(empty)
w._save_as()
new = w._save_folder()
check("an empty folder is used as is", new == empty)
names = sorted(p.name for p in new.iterdir())
print("new folder:", names)
check("the whole video is there", {"frames", "binary", "skel", "measurements.csv",
                                    "measurements_plot.png", "manual_lengths.csv",
                                    "scale.png", "wormscope.json"} <= set(names))
check("all frames", len(list((new / "frames").iterdir())) == N)
check("all masks + skeletons", len(list((new / "binary").iterdir())) == N
      and len(list((new / "skel").iterdir())) == N)
check("nothing unsaved after Save as", not w._has_unsaved())
check("label shows the new folder", w.save_folder_label.text() == "→ target")
check("window title shows the new folder", "target" in w.windowTitle())
after = {p.name for p in old.rglob("*") if p.is_file()}
check("the old folder is left as it was", before == after)
rows = list(csv.DictReader(open(new / "manual_lengths.csv")))
check("manual lengths incl. the unsaved one", {r["frame"] for r in rows} == {"2", "3"})
cal = json.loads((new / "wormscope.json").read_text())["calibration"]["dialog"]
check("scale.png relative, original path kept",
      cal["image"] == "scale.png" and cal["original_image"] == str(scale_src))
check("frames are read from the new folder",
      all(Path(p).parent == new / "frames" for p in w.frames._items))

# later Saves go to the new folder
w.set_frame(1); w._on_measure_added(2, 2, -1); w._on_measure_added(22, 2, -1)
w._save()
rows = list(csv.DictReader(open(new / "manual_lengths.csv")))
check("Save after Save as writes to the new folder", "1" in {r["frame"] for r in rows})
old_rows = list(csv.DictReader(open(old / "manual_lengths.csv")))
check("...and not to the old one", {r["frame"] for r in old_rows} == {"2"})

# it reopens with everything
w2 = MainWindow(load_dataset(new)); w2._start_sam = lambda *a, **k: None
check("reopens: masks identical (incl. the edit made before Save as)",
      (w2.masks.get(oid, 4) == edited).all())
check("reopens: calibration + scale.png",
      w2.px_to_um == 2.0 and Path(w2._calibration["image"]) == new / "scale.png")
check("reopens: 3 manual lengths", len(w2.by_id[oid].manual) == 3)

# --- picking a folder that isn't empty: a named subfolder, never mixed in ---
busy = tmp / "busy"; busy.mkdir(); (busy / "notes.txt").write_text("x")
check("non-empty folder -> subfolder named after the video",
      save_as_target(busy, new) == busy / "target")
(busy / "target").mkdir(); (busy / "target" / "measurements.csv").write_text("")
check("...and _2 when that is taken", save_as_target(busy, new) == busy / "target_2")
check("picking the current folder just saves there", save_as_target(new, new) == new)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
