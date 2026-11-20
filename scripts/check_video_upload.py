# ruff: noqa: E401, E402, E701, E702  (a compact test script: setup must run before the imports)
"""End-to-end check of uploading a video file: decoded into frames on upload.
Usage: python scripts/check_video_upload.py   (from the project folder)
Needs imageio + imageio-ffmpeg (both wormscope dependencies)."""
import os, sys, json, tempfile
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["HOME"] = tempfile.mkdtemp()        # own ~/.wormscope/settings.json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import imageio.v2 as imageio
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

from wormscope.app import resolve_dataset
from wormscope.io.loaders import build_raw_dataset, load_dataset
from wormscope.io.video import VideoCancelled, is_video
from wormscope.ui import io_actions
from wormscope.ui.main_window import MainWindow
from wormscope.ui.video_import_dialog import VideoImportDialog
from wormscope.io.video import pick_frames, video_info

# The Upload video dialog: answered by `ANSWER` (a function that sets it up)
ANSWER = [lambda dlg: dlg.all_rb.setChecked(True)]
SHOWN = []
def fake_exec(dlg):
    SHOWN.append(dlg); ANSWER[0](dlg); dlg.accept()
    return VideoImportDialog.DialogCode.Accepted
VideoImportDialog.exec = fake_exec

ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)

H, W, N, FPS = 64, 96, 24, 8
tmp = Path(tempfile.mkdtemp())
video = tmp / "plate 3.mp4"
with imageio.get_writer(str(video), format="FFMPEG", fps=FPS, codec="libx264",
                        macro_block_size=None) as wr:
    for i in range(N):
        a = np.full((H, W, 3), 30, np.uint8)
        a[20:40, 10 + 2 * i:30 + 2 * i] = 220        # a moving bright blob
        wr.append_data(a)
check("a video file is recognised", is_video(video) and not is_video(tmp))

# --- upload through the window (progress + open) ---
w0 = MainWindow(resolve_dataset(None)); w0.show(); app.processEvents()
io_actions.pick_video_or_image = lambda *a, **k: str(video)
seen = []
orig_step = io_actions.Progress.step
def spy_step(self, d, label=None):
    seen.append(d)
    return orig_step(self, d, label)
io_actions.Progress.step = spy_step
w0._import_raw()
io_actions.Progress.step = orig_step
check("uploading a video asks which frames to extract", len(SHOWN) == 1)
from wormscope.ui.main_window import WINDOWS
w = WINDOWS[-1]; w._start_sam = lambda *a, **k: None
out = tmp / "plate 3_wormscope"
check("progress was reported while decoding", seen and seen[-1] == N)
check("frames written to <name>_wormscope/frames/",
      len(list((out / "frames").glob("frame_*.png"))) == N)
check("no partial folder left", not (out / "frames.partial").exists())
check("the window opened the decoded video", w.n == N and w._save_folder() == out)
f5 = w.frames.get(5)
check("frames have the video's size", f5.shape[:2] == (H, W))
check("the frames are the video's content (blob moved)",
      f5[30, 22].mean() > 150 and w.frames.get(20)[30, 22].mean() < 100)
check("frame rate taken from the video", w.fps == FPS)
src = json.loads((out / "source_video.json").read_text())
check("source_video.json records the video", src["name"] == video.name
      and src["frames"] == N and src["fps"] == FPS)

# --- uploading it again reuses the frames (unsaved), no new folder ---
mt = (out / "frames" / "frame_00000.png").stat().st_mtime_ns
ds = build_raw_dataset(video, progress=lambda d, t: (_ for _ in ()).throw(AssertionError))
check("same video again: frames reused, not decoded again",
      ds.csv_path.parent == out
      and (out / "frames" / "frame_00000.png").stat().st_mtime_ns == mt)

# --- the dialog's numbers ---
dlg = VideoImportDialog(None, video, video_info(video))
check("dialog shows the video's fps and frame count", dlg.fps == FPS and dlg.n == N)
dlg.rate_rb.setChecked(True); dlg.rate_spin.setValue(2.0)
check("2 fps of an 8 fps video = every 4th frame",
      dlg.sampling() == {"mode": "every", "step": 4} and "2 fps" in dlg.rate_detail.text())
dlg.rate_spin.setValue(3.0)
check("3 fps rounds to an evenly spaced rate (every 3rd -> 2.67 fps)",
      dlg.step() == 3 and "nearest rate" in dlg.rate_detail.text())
dlg.rate_spin.setValue(8.0)
check("the full rate is just 'all frames'", dlg.sampling() == {"mode": "all"})
dlg.random_rb.setChecked(True); dlg.count_spin.setValue(5); dlg.seed_spin.setValue(3)
check("random: count + seed", dlg.sampling() == {"mode": "random", "count": 5, "seed": 3})
check("random picks are reproducible and sorted",
      pick_frames(dlg.sampling(), 24) == pick_frames(dlg.sampling(), 24)
      == sorted(pick_frames(dlg.sampling(), 24)))

# --- every 4th frame through the window: same unsaved folder, new frames ---
ANSWER[0] = lambda d: (d.rate_rb.setChecked(True), d.rate_spin.setValue(2.0))
io_actions.pick_video_or_image = lambda *a, **k: str(video)
w._import_raw()
w = WINDOWS[-1]; w._start_sam = lambda *a, **k: None
names = sorted(p.name for p in (out / "frames").iterdir())
check("every 4th frame: 6 frames, re-extracted in the same unsaved folder",
      w._save_folder() == out and w.n == 6 and len(names) == 6)
check("file names keep the video frame numbers (0, 4, 8, …)",
      names[:3] == ["frame_00000.png", "frame_00004.png", "frame_00008.png"])
check("frame numbers in the app are the video's", w.ds.idx_to_key[2] == 8)
check("fps of the kept frames = 8 / 4 = 2", w.fps == 2.0)
check("the dialog remembers the last choice",
      VideoImportDialog(None, video, video_info(video)).sampling()
      == {"mode": "every", "step": 4})

# --- random frames ---
ANSWER[0] = lambda d: (d.random_rb.setChecked(True), d.count_spin.setValue(7),
                       d.seed_spin.setValue(11))
w._import_raw()
w = WINDOWS[-1]; w._start_sam = lambda *a, **k: None
keys = [w.ds.idx_to_key[i] for i in range(w.n)]
check("7 random frames, in time order", w.n == 7 and keys == sorted(keys)
      and keys == pick_frames({"mode": "random", "count": 7, "seed": 11}, N))
check("random frames: no frame rate", w.fps is None)
src = json.loads((out / "source_video.json").read_text())
check("source_video.json records the sampling",
      src["sampling"] == {"mode": "random", "count": 7, "seed": 11}
      and src["video_frames"] == N and src["video_fps"] == FPS)
# cancelling the dialog does nothing
n_windows = len(WINDOWS)
VideoImportDialog.exec = lambda dlg: VideoImportDialog.DialogCode.Rejected
w._import_raw()
check("Cancel in the dialog opens nothing", len(WINDOWS) == n_windows)
VideoImportDialog.exec = fake_exec

# back to all frames for the rest
ANSWER[0] = lambda d: d.all_rb.setChecked(True)
w._import_raw()
w = WINDOWS[-1]; w._start_sam = lambda *a, **k: None

# --- after Save, an upload starts fresh next to it ---
w._set_annotation_mode("manual")
w._on_measure_added(5, 5, -1); w._on_measure_added(25, 5, -1)
w._save()
check("Save keeps source_video.json and the fps",
      (out / "source_video.json").is_file()
      and json.loads((out / "wormscope.json").read_text())["calibration"]["fps"] == FPS)
ds2 = build_raw_dataset(video)
check("after Save, re-upload decodes into <name>_wormscope_2",
      ds2.csv_path.parent.name == "plate 3_wormscope_2" and len(ds2.frames) == N)
w2 = MainWindow(load_dataset(out)); w2._start_sam = lambda *a, **k: None
check("Continue editing reopens it", w2.n == N and w2.fps == FPS
      and w2.by_id[0].manual)

# --- stopping the decode leaves nothing behind ---
video2 = tmp / "other.avi"
with imageio.get_writer(str(video2), format="FFMPEG", fps=5, codec="mpeg4") as wr:
    for i in range(30):
        wr.append_data(np.full((48, 64, 3), 8 * i, np.uint8))
try:
    build_raw_dataset(video2, progress=lambda d, t: d >= 10)
    check("Stop raises VideoCancelled", False)
except VideoCancelled:
    check("Stop raises VideoCancelled", True)
o2 = tmp / "other_wormscope"
check("a stopped decode leaves no frames", not list(o2.glob("frames*/*.png")))
ds3 = build_raw_dataset(video2)
check("...and the next upload decodes it fully (.avi)", len(ds3.frames) == 30
      and ds3.csv_path.parent == o2)

# --- the command line: python main.py video.mp4 ---
ds4 = resolve_dataset(str(video))
check("main.py <video> works", len(ds4.frames) == N)

# --- a file that is not a readable video ---
bad = tmp / "broken.mp4"; bad.write_bytes(b"not a video")
w0b = MainWindow(resolve_dataset(None))
io_actions.pick_video_or_image = lambda *a, **k: str(bad)
n_windows = len(WINDOWS)
w0b._import_raw()
check("a broken video shows a warning and opens nothing", len(WINDOWS) == n_windows)
check("...and leaves no partial frames", not (tmp / "broken_wormscope" / "frames.partial").exists())

# --- Save as… takes source_video.json along ---
dest = tmp / "moved"; dest.mkdir()
io_actions.pick_folder = lambda *a, **k: str(dest)
w2._save_as()
check("Save as… copies source_video.json", (dest / "source_video.json").is_file())

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
