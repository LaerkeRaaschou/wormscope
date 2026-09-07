#!/usr/bin/env python3
"""
wormscope — GUI for the Worm_vision project, with mask editing in the Zoom view.

Reads a per-video output folder produced by the pipeline plus the original
(un-annotated) frames:

    data/<video>/
        frames/              # ORIGINAL frames  -> shown underneath the masks
        binary/              # worm{id:02d}_{stem}.png -> per-worm masks (EDITABLE)
        skel/                # worm{id:02d}_{stem}.png -> skeletons (recomputed)
        mask/  video/        # (not used here)
        plots/               # saved per-worm plots (not used yet)
        measurements.csv     # per-frame metrics -> Plots panel (rewritten on Save)
        measurements_plot.png

Layout (matches the sketch):
    +----------------------+  +-----------+
    |        VIDEO         |  | Labelled  |
    |  (frame + masks)     |  |  worm list|
    |   [<] [▶] [>]        |  |  + Info   |
    +----------------------+  +-----------+
    |         ZOOM         |  |   PLOTS   |
    |  Pan Draw Erase      |  |  (4 slots)|
    +----------------------+  +-----------+

What works:
  - Displays the ORIGINAL frame with every worm's binary mask overlaid:
    light mid-blue for unselected worms, darker blue for the selected one.
  - Starts with NO worm selected (zoom + plots stay empty until you pick one).
  - Steps through frames (< / >, slider, arrow keys, space to play).
  - Lists tracked worms; selecting one drives Zoom + Plots and its skeleton.
  - Click a worm in the Video to select it (nearest centroid).
  - EDITING happens in the Zoom panel: it is zoomable (slider + mouse wheel)
    and pannable (Pan tool drag). The main video shows a rectangle marking the
    region the Zoom is looking at. Draw / Erase paint into the *selected*
    worm's mask for the current frame. On each stroke area / skeleton / length
    are recomputed and the plots refresh live. "Save" writes edited masks +
    skeletons back to binary/ and skel/ and rewrites measurements.csv.

Frame alignment is keyed on the trailing integer in each filename, so original
frames, binary masks, and CSV rows line up even if their name prefixes differ.
If the data folder is missing, it falls back to editable demo data.

Deps:  pip install PyQt6 matplotlib pillow numpy scikit-image
Run:   python wormscope_gui.py
"""

from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# --- macOS/conda fix: point Qt at PyQt6's own bundled plugins ---------------
# A conda base env layered on the venv often sets QT_PLUGIN_PATH to conda's Qt,
# which lacks the matching cocoa plugin -> "Could not find the Qt platform
# plugin 'cocoa'". Force the path to PyQt6's own plugins before Qt initialises.
import PyQt6  # noqa: E402
_qt_plugins = os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "plugins")
if os.path.isdir(_qt_plugins):
    os.environ.setdefault("QT_PLUGIN_PATH", _qt_plugins)
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH",
                          os.path.join(_qt_plugins, "platforms"))

from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QListWidget, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QButtonGroup,
    QInputDialog, QMessageBox, QCheckBox, QFileDialog,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False

try:
    from skimage.morphology import skeletonize
    HAVE_SKIMAGE = True
except Exception:  # pragma: no cover
    HAVE_SKIMAGE = False

# --- colour theme: light grayish-blue surfaces, dark-blue accents -----------
BG_LIGHT = "#e4eaf2"          # window background (grayish blue)
PANEL = "#eff3f9"             # group-box / plot background (lighter)
DARK_BLUE = "#1f3a63"         # accents: buttons, borders, text, list selection
DARK_BLUE_HI = "#2b4f86"      # hover / accent-light
DARK_BLUE_LO = "#16294a"      # pressed
GRIDCOL = "#c4d0e2"           # borders, grid, slider groove
PLOT_LINE = "#4372a3"         # metric curve (steel blue)
PLOT_DOT = "#14243f"          # current-frame dot (deep navy)
PLOT_AX_BG = "#f7fafe"        # plot axes background

PANEL_BG = QColor(214, 224, 236)     # letterbox (grayish blue)
VIEWPORT_COL = QColor(255, 210, 90)  # zoom-region rectangle on the video

# Worm mask overlay colours (RGB) and blend strength.
MASK_UNSEL_RGB = (120, 165, 215)     # light mid-blue  (unselected worms)
MASK_SEL_RGB = (26, 58, 120)         # darker blue     (selected worm)
MASK_ALPHA = 0.5                     # overlay opacity
SKEL_RGB = (230, 60, 60)             # selected worm's skeleton (red)
ONION_RGB = (240, 150, 60)           # onion-skin: previous frame's mask (orange)
ONION_ALPHA = 0.35                   # onion-skin opacity

PLAYBACK_FPS = 10          # display speed for playback (not the acquisition FPS)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_PLOTS = 4
SPEED_METRIC = "speed"                 # centroid speed, px/frame
ACCEL_METRIC = "acceleration"          # change of speed, px/frame^2
KINEMATIC_METRICS = [SPEED_METRIC, ACCEL_METRIC]
SMOOTH_WINDOW = 5                      # moving-average window for the Smooth toggle


def moving_average(values, window):
    """Centered moving average over a list of samples; the window shrinks at
    the ends so the output keeps the same length (edge-safe)."""
    n = len(values)
    if window <= 1 or n == 0:
        return list(values)
    half = window // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def color_for(obj_id):
    """Stable per-worm colour (matches worm_measure.py / worm_plot.py seeding)."""
    rng = np.random.default_rng(obj_id)
    return rng.integers(60, 256, size=3).astype(np.uint8)


def save_measurements_plot(path, worms, specs):
    """Regenerate the per-video overview plot: one subplot per metric spec
    (name, ylabel, factor), one coloured line per worm. Returns True on success.

    Uses the Agg backend directly so it works regardless of the Qt binding."""
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except Exception as err:                      # pragma: no cover
        print(f"[plot] matplotlib unavailable: {err}")
        return False
    fig = Figure(figsize=(11, 2.3 * len(specs) + 1))
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(len(specs), 1, sharex=True))
    any_line = False
    for ax, (name, label, factor) in zip(axes, specs):
        for w in worms:
            series = w.metrics.get(name, {})
            fs = [f for f in sorted(series) if series[f] == series[f]]
            if not fs:
                continue
            any_line = True
            ax.plot(fs, [series[f] * factor for f in fs],
                    color=color_for(w.obj_id) / 255.0, lw=1, marker=".", ms=3,
                    label=f"worm {w.obj_id}")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frame")
    if any_line:
        axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                       fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    return True
ZOOM_MIN, ZOOM_MAX = 0.1, 40.0

STYLESHEET = f"""
QMainWindow, QWidget {{ background-color: {BG_LIGHT}; color: {DARK_BLUE};
    font-size: 13px; }}
QGroupBox {{ background-color: {PANEL}; border: 1px solid {GRIDCOL};
    border-radius: 8px; margin-top: 14px; padding: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left;
    left: 10px; padding: 0 4px; color: {DARK_BLUE}; font-weight: 600; }}
QPushButton {{ background-color: {DARK_BLUE}; color: white;
    border: 2px solid transparent; border-radius: 6px; padding: 4px 10px; }}
QPushButton:hover {{ background-color: {DARK_BLUE_HI}; }}
QPushButton:pressed {{ background-color: {DARK_BLUE_LO}; }}
QPushButton:checked {{ background-color: {DARK_BLUE_LO}; border: 2px solid #a9c2e6; }}
QPushButton:disabled {{ background-color: #9fb0c8; color: #e8eef7; }}
QPushButton#compact {{ padding: 4px 2px; }}
QListWidget {{ background-color: white; border: 1px solid {GRIDCOL};
    border-radius: 6px; padding: 2px; }}
QListWidget::item {{ padding: 4px 6px; border-radius: 4px; }}
QListWidget::item:selected {{ background-color: {DARK_BLUE}; color: white; }}
QLabel {{ background: transparent; }}
QSlider::groove:horizontal {{ height: 6px; background: {GRIDCOL};
    border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {DARK_BLUE_HI}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {DARK_BLUE}; width: 14px;
    margin: -5px 0; border-radius: 7px; }}
"""

_LAST_INT = re.compile(r"(\d+)(?!.*\d)")          # last run of digits in a name
_WORM = re.compile(r"worm[_-]?(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Geometry / metric helpers
# --------------------------------------------------------------------------- #
def skeleton_length_px(skel: np.ndarray) -> float:
    """Length of a 1-pixel-wide skeleton in pixels (matches worm_measure.py)."""
    s = skel.astype(bool)
    horiz = np.logical_and(s[:, :-1], s[:, 1:]).sum()
    vert = np.logical_and(s[:-1, :], s[1:, :]).sum()
    diag_a = np.logical_and(s[:-1, :-1], s[1:, 1:]).sum()
    diag_b = np.logical_and(s[:-1, 1:], s[1:, :-1]).sum()
    return float((horiz + vert) + np.sqrt(2.0) * (diag_a + diag_b))


def skeletonize_crop(mask: np.ndarray):
    """Skeletonize only the mask's bounding box (much faster than the whole
    frame) and paste it back into a full-size skeleton. Returns None if the
    mask is empty or skimage is unavailable."""
    if not HAVE_SKIMAGE or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = skeletonize(mask[y0:y1, x0:x1])
    full = np.zeros_like(mask)
    full[y0:y1, x0:x1] = sub
    return full


def mask_stats(mask: np.ndarray):
    """(area_px, length_px, skeleton, centroid, bbox) for a boolean mask."""
    area = int(mask.sum())
    if area == 0:
        return 0, None, None, None, None
    ys, xs = np.nonzero(mask)
    centroid = (float(xs.mean()), float(ys.mean()))
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    skel = skeletonize_crop(mask)
    length = skeleton_length_px(skel) if skel is not None else None
    return area, length, skel, centroid, bbox


def stamp_disk(mask: np.ndarray, cx: float, cy: float, r: float, value: bool):
    """Set a filled disk of radius r centred at (cx, cy) in-place."""
    h, w = mask.shape
    x0, x1 = max(0, int(np.floor(cx - r))), min(w, int(np.ceil(cx + r)) + 1)
    y0, y1 = max(0, int(np.floor(cy - r))), min(h, int(np.ceil(cy + r)) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    disk = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    mask[y0:y1, x0:x1][disk] = value


# --------------------------------------------------------------------------- #
# Kinematics: centroid speed / acceleration, in pixels
# --------------------------------------------------------------------------- #
def worm_speed(worm):
    """Centroid speed in pixels per frame.

    For each frame where the worm is present, the speed is the Euclidean
    distance from its previous appearance divided by the frame gap:
        speed[f] = |c(f) - c(prev)| / (f - prev)
    Worms can be absent in some frames, so the gap (f - prev) may exceed 1;
    dividing by it keeps the result in px/frame. The first appearance has no
    predecessor and is therefore omitted.
    """
    speed = {}
    prev = None
    for f in sorted(worm.centroids):
        c = worm.centroids[f]
        if prev is not None:
            pf, pc = prev
            dt = f - pf
            if dt > 0:
                speed[f] = ((c[0] - pc[0]) ** 2 + (c[1] - pc[1]) ** 2) ** 0.5 / dt
        prev = (f, c)
    return speed


def worm_acceleration(worm):
    """Centroid acceleration in pixels per frame squared.

    The frame-to-frame change of worm_speed():
        acc[f] = (speed[f] - speed[prev]) / (f - prev)
    Uses the same absence-aware frame gaps as worm_speed(). Needs at least two
    consecutive speed samples, so the first speed sample has no acceleration.
    """
    speed = worm_speed(worm)
    acc = {}
    prev = None
    for f in sorted(speed):
        s = speed[f]
        if prev is not None:
            pf, ps = prev
            dt = f - pf
            if dt > 0:
                acc[f] = (s - ps) / dt
        prev = (f, s)
    return acc


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Worm:
    """Frame-indexed and sparse: a worm may be absent in some frames."""
    obj_id: int
    centroids: dict[int, tuple[float, float]] = field(default_factory=dict)
    metrics: dict[str, dict[int, float]] = field(default_factory=dict)
    bboxes: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)

    def centroid(self, frame: int):
        return self.centroids.get(frame)

    def bbox(self, frame: int):
        return self.bboxes.get(frame)


class FrameSource:
    """Lazy frame provider with a tiny cache."""
    def __init__(self, items, loader):
        self._items = items
        self._loader = loader
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self):
        return len(self._items)

    def get(self, i: int) -> np.ndarray:
        arr = self._cache.get(i)
        if arr is None:
            if len(self._cache) > 48:
                self._cache.clear()
            arr = self._loader(self._items[i])
            self._cache[i] = arr
        return arr


class MaskStore:
    """Editable per-(worm, frame) binary masks.

    Real data is backed by PNG files in binary/ (and skel/); demo data lives
    purely in memory (binary_dir is None -> Save is a no-op)."""
    def __init__(self, shape, binary_dir=None, skel_dir=None, idx_to_stem=None):
        self.H, self.W = shape
        self.binary_dir = Path(binary_dir) if binary_dir else None
        self.skel_dir = Path(skel_dir) if skel_dir else None
        self.idx_to_stem = idx_to_stem or {}
        self._masks: dict[tuple[int, int], np.ndarray] = {}
        self._paths: dict[tuple[int, int], Path] = {}
        self._skel_cache: dict[tuple[int, int], np.ndarray | None] = {}
        self.dirty: set[tuple[int, int]] = set()

    @property
    def savable(self) -> bool:
        return self.binary_dir is not None

    def register(self, wid, idx, path=None, mask=None):
        key = (wid, idx)
        if path is not None:
            self._paths[key] = Path(path)
        if mask is not None:
            self._masks[key] = mask

    def get(self, wid, idx) -> np.ndarray:
        key = (wid, idx)
        m = self._masks.get(key)
        if m is None:
            p = self._paths.get(key)
            if p is not None and p.exists():
                m = np.asarray(Image.open(p).convert("L")) > 127
            else:
                m = np.zeros((self.H, self.W), dtype=bool)
            self._masks[key] = m
        return m

    def set(self, wid, idx, mask):
        """Replace a mask (used by editing / undo)."""
        self._masks[(wid, idx)] = mask
        self.set_dirty(wid, idx)

    def frames_of(self, wid):
        """All frame indices this worm has a mask or file for."""
        idxs = {i for (w, i) in self._masks if w == wid}
        idxs |= {i for (w, i) in self._paths if w == wid}
        return sorted(idxs)

    def skel_path_for(self, binary_path: Path):
        return None if self.skel_dir is None else self.skel_dir / binary_path.name

    def pop_worm(self, wid):
        """Remove all data for a worm. Returns (masks, paths, files) where
        files are the on-disk PNGs (binary + skel) that should be deleted on
        Save; masks/paths let the caller restore the worm on undo."""
        masks = {k: self._masks.pop(k) for k in [k for k in self._masks if k[0] == wid]}
        paths = {k: self._paths.pop(k) for k in [k for k in self._paths if k[0] == wid]}
        files = []
        for p in paths.values():
            files.append(p)
            sp = self.skel_path_for(p)
            if sp is not None:
                files.append(sp)
        self.dirty = {k for k in self.dirty if k[0] != wid}
        self._skel_cache = {k: v for k, v in self._skel_cache.items() if k[0] != wid}
        return masks, paths, files

    def restore_worm(self, masks, paths):
        self._masks.update(masks)
        self._paths.update(paths)

    def rename_worm(self, old, new):
        """Move a worm's data to a new id. Loads every frame first so nothing
        is lost, re-keys masks/dirty/cache, drops the old file paths (returned,
        with skel siblings, for deletion on Save) and marks all frames dirty so
        they are rewritten under the new name."""
        for idx in self.frames_of(old):
            self.get(old, idx)                       # ensure loaded into _masks
        for k in [k for k in self._masks if k[0] == old]:
            self._masks[(new, k[1])] = self._masks.pop(k)
        old_files = []
        for k in [k for k in self._paths if k[0] == old]:
            p = self._paths.pop(k)
            old_files.append(p)
            sp = self.skel_path_for(p)
            if sp is not None:
                old_files.append(sp)
        self.dirty = {((new if w == old else w), i) for (w, i) in self.dirty}
        self._skel_cache = {((new if w == old else w), i): v
                            for (w, i), v in self._skel_cache.items()}
        for (w, i) in [k for k in self._masks if k[0] == new]:
            self.set_dirty(new, i)
        return old_files

    def set_dirty(self, wid, idx):
        self.dirty.add((wid, idx))
        self._skel_cache.pop((wid, idx), None)

    def _load_skel(self, wid, idx):
        """Load the pipeline's saved skeleton PNG from skel/ (same filename as
        the binary mask), or None if there isn't one."""
        if self.skel_dir is None:
            return None
        bp = self._binary_path(wid, idx)
        if bp is None:
            return None
        sp = self.skel_dir / bp.name
        if sp.exists():
            return np.asarray(Image.open(sp).convert("L")) > 127
        return None

    def skel(self, wid, idx):
        """Skeleton to display for (worm, frame). Prefers a live recompute for
        edited frames (needs skimage), otherwise the saved skel/ PNG, otherwise
        a live recompute. Cached; returns None if nothing is available."""
        key = (wid, idx)
        if key in self.dirty:
            if HAVE_SKIMAGE:
                return skeletonize_crop(self.get(wid, idx))
            return self._load_skel(wid, idx)
        if key in self._skel_cache:
            return self._skel_cache[key]
        s = self._load_skel(wid, idx)
        if s is None and HAVE_SKIMAGE:
            s = skeletonize_crop(self.get(wid, idx))
        self._skel_cache[key] = s
        return s

    def _binary_path(self, wid, idx) -> Path | None:
        key = (wid, idx)
        p = self._paths.get(key)
        if p is None and self.binary_dir is not None:
            stem = self.idx_to_stem.get(idx)
            if stem is None:
                return None
            p = self.binary_dir / f"worm{wid:02d}_{stem}"
            self._paths[key] = p
        return p

    def save(self, wid, idx, skel=None):
        if self.binary_dir is None:
            self.dirty.discard((wid, idx))
            return
        bp = self._binary_path(wid, idx)
        if bp is None:
            return
        bp.parent.mkdir(parents=True, exist_ok=True)
        mask = self.get(wid, idx)
        Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L").save(bp)
        if self.skel_dir is not None:
            self.skel_dir.mkdir(parents=True, exist_ok=True)
            sp = self.skel_dir / bp.name
            if skel is None:
                skel = skeletonize_crop(mask)
            if skel is not None:
                Image.fromarray(np.where(skel, 255, 0).astype(np.uint8), "L").save(sp)
        self._skel_cache.pop((wid, idx), None)
        self.dirty.discard((wid, idx))


@dataclass
class Dataset:
    frames: FrameSource
    worms: list
    metric_names: list         # real metrics from the CSV / masks (area, length)
    masks: MaskStore
    idx_to_key: dict           # display index -> original frame number
    csv_path: Path | None
    title: str
    frames_dir: Path | None = None   # on-disk frames folder (for SAM2), if any


# --------------------------------------------------------------------------- #
# Real-data loaders
# --------------------------------------------------------------------------- #
def _last_int(name: str):
    m = _LAST_INT.search(Path(name).stem)
    return int(m.group(1)) if m else None


def _list_images(folder: Path):
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS] \
        if folder.is_dir() else []


def _find_frames_dir(data_dir: Path):
    """Locate the ORIGINAL (un-annotated) frames for this video."""
    video = data_dir.name
    candidates = [
        data_dir / "frames",
        data_dir / "frames" / video,
        data_dir.parent / "frames" / video,
        data_dir.parent / "frames",
        data_dir / "video",
    ]
    for c in candidates:
        if _list_images(c):
            return c
    return None


def load_frames(data_dir: Path):
    """Load original display frames. Returns
    (FrameSource, key_to_idx, idx_to_stem, n)."""
    folder = _find_frames_dir(data_dir)
    if folder is None:
        return None, {}, {}, 0
    files = sorted(_list_images(folder),
                   key=lambda p: (_last_int(p.name) if _last_int(p.name) is not None
                                  else 1 << 30, p.name))
    key_to_idx, idx_to_stem = {}, {}
    for idx, p in enumerate(files):
        idx_to_stem[idx] = p.stem + ".png"
        k = _last_int(p.name)
        if k is not None:
            key_to_idx.setdefault(k, idx)
    src = FrameSource(files, lambda p: np.asarray(Image.open(p).convert("RGB")))
    print(f"[frames] {folder.name}/: {len(files)} frames")
    return src, key_to_idx, idx_to_stem, len(files)


def load_tracks_from_masks(data_dir: Path, key_to_idx, masks: MaskStore):
    """Parse binary/worm{id:02d}_{stem}.png, register mask paths, derive
    centroid + bbox per worm/frame."""
    mask_dir = data_dir / "binary"
    cents: dict[int, dict] = {}
    boxes: dict[int, dict] = {}
    n_masks = 0
    for p in sorted(_list_images(mask_dir)):
        mw = _WORM.search(p.name)
        if not mw:
            continue
        wid = int(mw.group(1))
        fk = _last_int(p.name[mw.end():])
        if fk is None:
            continue
        idx = key_to_idx.get(fk)
        if idx is None:
            continue
        masks.register(wid, idx, path=p)
        arr = np.asarray(Image.open(p).convert("L")) > 127
        ys, xs = np.nonzero(arr)
        if xs.size == 0:
            continue
        cents.setdefault(wid, {})[idx] = (float(xs.mean()), float(ys.mean()))
        boxes.setdefault(wid, {})[idx] = (int(xs.min()), int(ys.min()),
                                          int(xs.max()) + 1, int(ys.max()) + 1)
        n_masks += 1
    worms = [Worm(obj_id=wid, centroids=cents[wid], bboxes=boxes.get(wid, {}))
             for wid in sorted(cents)]
    print(f"[tracks] {n_masks} masks -> worms {[w.obj_id for w in worms]}")
    return worms


def _find_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    for c in cols:
        cl = c.lower()
        if any(cand in cl for cand in candidates):
            return c
    return None


def _to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_metrics(data_dir: Path, worms, key_to_idx, n_frames):
    """Fill worm.metrics from measurements.csv."""
    csv_path = data_dir / "measurements.csv"
    if not csv_path.is_file():
        print("[metrics] no measurements.csv")
        return []
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    cols = list(rows[0].keys())
    frame_col = _find_col(cols, ["frame", "index", "t"])
    id_col = _find_col(cols, ["worm", "obj_id", "track", "label", "id"])
    metric_cols = [c for c in cols if c not in (frame_col, id_col)
                   and sum(_to_num(r.get(c)) is not None for r in rows[:50]) > len(rows[:50]) * 0.8]
    by_id = {w.obj_id: w for w in worms}
    single = worms[0] if len(worms) == 1 else None
    for r in rows:
        fk = _to_num(r.get(frame_col)) if frame_col else None
        fk = int(fk) if fk is not None else None
        idx = key_to_idx.get(fk)
        if idx is None and fk is not None and 0 <= fk < n_frames:
            idx = fk
        if idx is None:
            continue
        if id_col is not None:
            wid = _to_num(r.get(id_col))
            w = by_id.get(int(wid)) if wid is not None else None
        else:
            w = single
        if w is None:
            continue
        for c in metric_cols:
            v = _to_num(r.get(c))
            if v is not None:
                w.metrics.setdefault(c, {})[idx] = v
    names = sorted({m for w in worms for m in w.metrics})
    print(f"[metrics] frame='{frame_col}' id='{id_col}' metrics={names}")
    return names


def load_dataset(data_dir: Path) -> Dataset | None:
    frames, key_to_idx, idx_to_stem, n = load_frames(data_dir)
    if frames is None:
        return None
    shape = frames.get(0).shape[:2]
    masks = MaskStore(shape,
                      binary_dir=data_dir / "binary",
                      skel_dir=data_dir / "skel",
                      idx_to_stem=idx_to_stem)
    worms = load_tracks_from_masks(data_dir, key_to_idx, masks)
    names = load_metrics(data_dir, worms, key_to_idx, n)
    if not names:
        names = ["area_px", "length_px"]
        _seed_metrics_from_masks(worms, masks, names)
    idx_to_key = {idx: k for k, idx in key_to_idx.items()}
    return Dataset(frames, worms, names[:MAX_PLOTS], masks, idx_to_key,
                   data_dir / "measurements.csv", f"wormscope — {data_dir.name}",
                   frames_dir=_find_frames_dir(data_dir))


def build_raw_dataset(frames_dir: Path) -> Dataset | None:
    """Start a fresh working dataset from a folder of RAW frames (no masks yet).
    Outputs (binary/, skel/, measurements.csv) go in a sibling '<name>_wormscope'
    folder so the raw frames are never modified. Worms start empty; use the SAM2
    point tools (or Add + brush) to create them."""
    files = sorted(_list_images(frames_dir),
                   key=lambda p: (_last_int(p.name) if _last_int(p.name) is not None
                                  else 1 << 30, p.name))
    if not files:
        return None
    key_to_idx, idx_to_stem = {}, {}
    for idx, p in enumerate(files):
        idx_to_stem[idx] = p.stem + ".png"
        k = _last_int(p.name)
        key_to_idx.setdefault(k if k is not None else idx, idx)
    src = FrameSource(files, lambda p: np.asarray(Image.open(p).convert("RGB")))
    shape = src.get(0).shape[:2]
    out_dir = frames_dir.parent / f"{frames_dir.name}_wormscope"
    masks = MaskStore(shape, binary_dir=out_dir / "binary",
                      skel_dir=out_dir / "skel", idx_to_stem=idx_to_stem)
    idx_to_key = {idx: k for k, idx in key_to_idx.items()}
    print(f"[import] {len(files)} raw frames from {frames_dir} -> outputs in {out_dir}")
    return Dataset(src, [], ["area_px", "length_px"], masks, idx_to_key,
                   out_dir / "measurements.csv", f"wormscope — {frames_dir.name}",
                   frames_dir=frames_dir)


def _seed_metrics_from_masks(worms, masks: MaskStore, names):
    area_name, length_name = names[0], names[1]
    for w in worms:
        for idx in sorted(w.bboxes):
            area, length, *_ = mask_stats(masks.get(w.obj_id, idx))
            w.metrics.setdefault(area_name, {})[idx] = area
            if length is not None:
                w.metrics.setdefault(length_name, {})[idx] = length


# --------------------------------------------------------------------------- #
# Demo fallback (editable, in-memory masks)
# --------------------------------------------------------------------------- #
def make_demo(n_frames=60, size=(640, 400), seed=7) -> Dataset:
    rng = np.random.default_rng(seed)
    W, H = size
    starts = [(150, 120), (430, 110), (250, 250), (500, 250)]
    metas = []
    for i, (sx, sy) in enumerate(starts, start=1):
        traj = np.zeros((n_frames, 2))
        pos = np.array([sx, sy], float)
        vel = rng.normal(0, 0.6, 2)
        for f in range(n_frames):
            vel = 0.9 * vel + rng.normal(0, 0.25, 2)
            pos = np.clip(pos + vel, [60, 60], [W - 60, H - 60])
            traj[f] = pos
        t = np.linspace(0, 1, 9)
        spine = np.stack([(t - 0.5) * 70,
                          np.sin(t * np.pi + rng.uniform(0, 6.28)) * 22], axis=1)
        metas.append((i, traj, spine))

    frames = []
    masks = MaskStore((H, W))
    worms = [Worm(obj_id=i) for i, _, _ in metas]
    by_id = {w.obj_id: w for w in worms}

    for f in range(n_frames):
        base = Image.new("RGB", (W, H), (232, 240, 250))
        db = ImageDraw.Draw(base)
        db.rectangle([0, 0, W - 1, H - 1], outline=(210, 220, 234))
        for i, traj, spine in metas:
            a = 0.5 * np.sin(f * 0.08 + i)
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            pts = (spine @ rot.T) + traj[f]
            poly = [(float(x), float(y)) for x, y in pts]
            db.line(poly, fill=(150, 140, 130), width=13, joint="curve")
            for x, y in (poly[0], poly[-1]):
                db.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(150, 140, 130))
            mimg = Image.new("L", (W, H), 0)
            dm = ImageDraw.Draw(mimg)
            dm.line(poly, fill=255, width=13, joint="curve")
            for x, y in (poly[0], poly[-1]):
                dm.ellipse([x - 6, y - 6, x + 6, y + 6], fill=255)
            mask = np.asarray(mimg) > 127
            masks.register(i, f, mask=mask)
            area, length, _, centroid, bbox = mask_stats(mask)
            by_id[i].centroids[f] = centroid
            by_id[i].bboxes[f] = bbox
            by_id[i].metrics.setdefault("area_px", {})[f] = area
            if length is not None:
                by_id[i].metrics.setdefault("length_px", {})[f] = length
        frames.append(np.asarray(base))

    src = FrameSource(frames, lambda a: a)
    idx_to_key = {f: f for f in range(n_frames)}
    return Dataset(src, worms, ["area_px", "length_px"], masks, idx_to_key,
                   None, "wormscope — demo")


# --------------------------------------------------------------------------- #
# SAM2 interactive segmentation (video predictor, lazily loaded)
# --------------------------------------------------------------------------- #
def _prepare_jpeg_dir(frames):
    """SAM2's video predictor wants a folder of numbered JPEGs. Write the
    frames as 0.jpg, 1.jpg, … in frame order so SAM frame_idx == our index."""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="wormscope_sam_"))
    for i in range(len(frames)):
        Image.fromarray(frames.get(i)).save(d / f"{i}.jpg", quality=95)
    return d


class Sam2Backend:
    """Thin wrapper over SAM2's video predictor. Imported lazily so the app
    runs without torch/sam2 installed. Point prompts on any frame drive a real
    per-frame segmentation of that object across the whole clip (propagation),
    not a copied mask.

    Requires the `sam2` package plus a checkpoint + config; point at them with
    the SAM2_CFG and SAM2_CKPT environment variables."""

    def __init__(self, frames, shape, cfg=None, ckpt=None, device=None):
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        cfg = cfg or os.environ.get("SAM2_CFG")
        ckpt = ckpt or os.environ.get("SAM2_CKPT")
        if not cfg or not ckpt:
            raise RuntimeError(
                "Set SAM2_CFG and SAM2_CKPT to the SAM2 config and checkpoint "
                "paths (e.g. sam2.1_hiera_l.yaml and sam2.1_hiera_large.pt).")
        device = device or ("cuda" if torch.cuda.is_available()
                            else "mps" if torch.backends.mps.is_available()
                            else "cpu")
        self._torch = torch
        self.predictor = build_sam2_video_predictor(cfg, ckpt, device=device)
        self.frames_dir = _prepare_jpeg_dir(frames)
        self.state = self.predictor.init_state(video_path=str(self.frames_dir))
        self.shape = shape
        print(f"[sam2] ready on {device}; frames -> {self.frames_dir}")

    def _mask_of(self, obj_id, ids, logits):
        for k, i in enumerate(ids):
            if int(i) == int(obj_id):
                return (logits[k, 0] > 0.0).detach().cpu().numpy()
        return None

    def add_points(self, frame_idx, obj_id, points, labels):
        """points: list of (x, y); labels: 1 = worm, 0 = not worm. Returns the
        object's mask on this frame (a preview) or None."""
        _, ids, logits = self.predictor.add_new_points_or_box(
            inference_state=self.state, frame_idx=int(frame_idx),
            obj_id=int(obj_id),
            points=np.asarray(points, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.int32))
        return self._mask_of(obj_id, ids, logits)

    def propagate(self):
        """Yield (frame_idx, {obj_id: bool_mask}) for every frame."""
        for frame_idx, ids, logits in self.predictor.propagate_in_video(self.state):
            masks = {int(i): (logits[k, 0] > 0.0).detach().cpu().numpy()
                     for k, i in enumerate(ids)}
            yield int(frame_idx), masks

    def reset(self):
        self.predictor.reset_state(self.state)


# Overridable so tests (and future backends) can inject a stand-in segmenter.
SEGMENTER_FACTORY = Sam2Backend


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def np_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


# --------------------------------------------------------------------------- #
# Video view (frame + selection; shows the zoom viewport rectangle)
# --------------------------------------------------------------------------- #
class VideoView(QWidget):
    imageClicked = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self._img: QImage | None = None
        self._sel = None
        self._viewport = None                # (x0, y0, x1, y1) in image coords
        self._rect: QRectF | None = None
        self._scale = 1.0
        self.setMinimumSize(360, 240)

    def set_frame(self, qimg):
        self._img = qimg
        self.update()

    def set_selection(self, centroid):
        self._sel = centroid
        self.update()

    def set_viewport(self, rect):
        self._viewport = rect
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), PANEL_BG)
        if self._img is None:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Video")
            return
        iw, ih = self._img.width(), self._img.height()
        self._scale = min(self.width() / iw, self.height() / ih)
        dw, dh = iw * self._scale, ih * self._scale
        ox, oy = (self.width() - dw) / 2, (self.height() - dh) / 2
        self._rect = QRectF(ox, oy, dw, dh)
        p.drawImage(self._rect, self._img)
        if self._viewport is not None:
            x0, y0, x1, y1 = self._viewport
            p.setPen(QPen(VIEWPORT_COL, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(ox + x0 * self._scale, oy + y0 * self._scale,
                              (x1 - x0) * self._scale, (y1 - y0) * self._scale))

    def mousePressEvent(self, e):
        if self._img is None or self._rect is None:
            return
        x = (e.position().x() - self._rect.x()) / self._scale
        y = (e.position().y() - self._rect.y()) / self._scale
        if 0 <= x < self._img.width() and 0 <= y < self._img.height():
            self.imageClicked.emit(float(x), float(y))


# --------------------------------------------------------------------------- #
# Zoom view (zoomable / pannable editing canvas)
# --------------------------------------------------------------------------- #
class ZoomView(QWidget):
    paintRequested = pyqtSignal(float, float)
    strokeFinished = pyqtSignal()
    viewportChanged = pyqtSignal()
    pointAdded = pyqtSignal(float, float, int)   # x, y, label (1=worm, 0=not)

    MODE_PAN, MODE_DRAW, MODE_ERASE = "pan", "draw", "erase"
    MODE_SAM_POS, MODE_SAM_NEG = "sam+", "sam-"

    def __init__(self):
        super().__init__()
        self._img: QImage | None = None
        self._iw = self._ih = 1
        self.zoom = 4.0
        self.cx = None                       # image-space centre of the view
        self.cy = None
        self.mode = self.MODE_PAN
        self.brush = 8
        self._painting = False
        self._panning = False
        self._last_pan = None
        self.setMinimumSize(240, 180)

    # --- image / view state ---
    def set_image(self, qimg, iw, ih):
        first = self._img is None or self.cx is None
        self._img, self._iw, self._ih = qimg, iw, ih
        if first:                            # centre on first image
            self.cx, self.cy = iw / 2, ih / 2
        self.update()

    def has_image(self):
        return self._img is not None

    def _clamp(self):
        if self.cx is None:
            return
        self.cx = min(max(0.0, self.cx), self._iw)
        self.cy = min(max(0.0, self.cy), self._ih)

    def set_zoom(self, z):
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, z))
        self._clamp()
        self.update()
        self.viewportChanged.emit()

    def frame_bbox(self, bbox, margin=1.7):
        """Zoom/centre so the worm's bbox comfortably fills the view."""
        x0, y0, x1, y1 = bbox
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        W, H = max(1, self.width()), max(1, self.height())
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, min(W / (bw * margin),
                                                     H / (bh * margin))))
        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
        self._clamp()
        self.update()
        self.viewportChanged.emit()

    def frame_fit(self):
        if self._img is None:
            return
        W, H = max(1, self.width()), max(1, self.height())
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, min(W / self._iw, H / self._ih)))
        self.cx, self.cy = self._iw / 2, self._ih / 2
        self.update()
        self.viewportChanged.emit()

    def _placement(self):
        return (self.width() / 2 - self.cx * self.zoom,
                self.height() / 2 - self.cy * self.zoom)

    def _to_image(self, e):
        if self._img is None or self.cx is None:
            return None
        left, top = self._placement()
        x = (e.position().x() - left) / self.zoom
        y = (e.position().y() - top) / self.zoom
        if 0 <= x < self._iw and 0 <= y < self._ih:
            return float(x), float(y)
        return None

    def viewport_rect(self):
        if self._img is None or self.cx is None:
            return None
        left, top = self._placement()
        W, H = self.width(), self.height()
        return (max(0.0, (0 - left) / self.zoom), max(0.0, (0 - top) / self.zoom),
                min(self._iw, (W - left) / self.zoom),
                min(self._ih, (H - top) / self.zoom))

    # --- events ---
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), PANEL_BG)
        if self._img is None or self.cx is None:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Zoom")
            return
        left, top = self._placement()
        p.drawImage(QRectF(left, top, self._iw * self.zoom, self._ih * self.zoom),
                    self._img)

    def wheelEvent(self, e):
        if self._img is None:
            return
        pt = self._to_image(e)
        factor = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom * factor))
        if pt is not None:                   # keep the point under the cursor put
            ex, ey = e.position().x(), e.position().y()
            self.cx = pt[0] + (self.width() / 2 - ex) / self.zoom
            self.cy = pt[1] + (self.height() / 2 - ey) / self.zoom
        self._clamp()
        self.update()
        self.viewportChanged.emit()

    def mousePressEvent(self, e):
        if self._img is None:
            return
        if self.mode in (self.MODE_DRAW, self.MODE_ERASE):
            pt = self._to_image(e)
            if pt is not None:
                self._painting = True
                self.paintRequested.emit(*pt)
        elif self.mode in (self.MODE_SAM_POS, self.MODE_SAM_NEG):
            pt = self._to_image(e)
            if pt is not None:
                self.pointAdded.emit(pt[0], pt[1],
                                     1 if self.mode == self.MODE_SAM_POS else 0)
        else:
            self._panning = True
            self._last_pan = (e.position().x(), e.position().y())

    def mouseMoveEvent(self, e):
        if self._painting:
            pt = self._to_image(e)
            if pt is not None:
                self.paintRequested.emit(*pt)
        elif self._panning and self._last_pan is not None:
            dx = e.position().x() - self._last_pan[0]
            dy = e.position().y() - self._last_pan[1]
            self._last_pan = (e.position().x(), e.position().y())
            self.cx -= dx / self.zoom
            self.cy -= dy / self.zoom
            self._clamp()
            self.update()
            self.viewportChanged.emit()

    def mouseReleaseEvent(self, e):
        if self._painting:
            self._painting = False
            self.strokeFinished.emit()
        self._panning = False
        self._last_pan = None


# --------------------------------------------------------------------------- #
# Plot panel (dynamic metric list)
# --------------------------------------------------------------------------- #
if HAVE_MPL:
    class PlotPanel(FigureCanvasQTAgg):
        frameSelected = pyqtSignal(int)

        def __init__(self, metrics):
            self.metrics = metrics or []
            self.worm = None
            self._frame = 0
            self.smooth = False
            self.scales = {}                 # metric -> (factor, ylabel)
            self._drawn = []                 # per-axis {frame: plotted-y}
            self._dragging = False
            self.fig = Figure(figsize=(3, 3), tight_layout=True)
            self.fig.patch.set_facecolor(PANEL)
            super().__init__(self.fig)
            n = max(1, len(self.metrics))
            self.axes = np.atleast_1d(self.fig.subplots(n, 1, sharex=True))
            self.dots = [None] * len(self.axes)
            self._blank()
            self.mpl_connect("button_press_event", self._on_press)
            self.mpl_connect("motion_notify_event", self._on_motion)
            self.mpl_connect("button_release_event", self._on_release)

        def _blank(self):
            for ax, name in zip(self.axes, self.metrics):
                ax.clear()
                ax.set_facecolor(PLOT_AX_BG)
                ax.set_ylabel(name, fontsize=8, color=DARK_BLUE)
                ax.tick_params(labelsize=7, colors=DARK_BLUE)
                for sp in ax.spines.values():
                    sp.set_color(GRIDCOL)
                ax.grid(alpha=0.35, color=GRIDCOL)
            if not self.metrics:
                self.axes[0].clear()
                self.axes[0].set_facecolor(PLOT_AX_BG)
            if self.worm is None:
                self.axes[0].text(0.5, 0.5, "select a worm", ha="center",
                                  va="center", transform=self.axes[0].transAxes,
                                  fontsize=9, color=GRIDCOL)
            elif self.metrics:
                self.axes[-1].set_xlabel("frame", fontsize=8, color=DARK_BLUE)
            self.dots = [None] * len(self.axes)
            self.draw_idle()

        def set_smooth(self, on):
            self.smooth = bool(on)
            self.set_worm(self.worm)         # redraw with/without smoothing

        def set_scales(self, scales):
            self.scales = scales or {}
            self.set_worm(self.worm)         # redraw in the new units

        def set_worm(self, worm):
            self.worm = worm
            self._blank()
            self._drawn = [{} for _ in self.axes]
            if worm is None or not self.metrics:
                return
            for i, (ax, name) in enumerate(zip(self.axes, self.metrics)):
                factor, label = self.scales.get(name, (1.0, name))
                ax.set_ylabel(label, fontsize=8, color=DARK_BLUE)
                series = worm.metrics.get(name, {})
                fs = [f for f in sorted(series) if series[f] == series[f]]
                if not fs:
                    ax.text(0.5, 0.5, "no data yet", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8, color=GRIDCOL)
                    continue
                ys = [series[f] * factor for f in fs]
                if self.smooth:
                    ys = moving_average(ys, SMOOTH_WINDOW)
                ax.plot(fs, ys, color=PLOT_LINE, lw=1.3)
                self._drawn[i] = dict(zip(fs, ys))    # what the dot should track
            self._place_dots()
            self.draw_idle()

        def set_frame(self, f):
            self._frame = f
            if self.worm is None or not self.metrics:
                return
            self._place_dots()
            self.draw_idle()

        def _place_dots(self):
            f = self._frame
            for i, ax in enumerate(self.axes):
                val = self._drawn[i].get(f) if i < len(self._drawn) else None
                if self.dots[i] is None:
                    (self.dots[i],) = ax.plot([], [], "o", color=PLOT_DOT,
                                              markersize=6, markeredgecolor="white",
                                              markeredgewidth=0.8, zorder=5)
                self.dots[i].set_data([] if val is None else [f],
                                      [] if val is None else [val])

        def _emit_frame(self, event):
            if event.inaxes is not None and event.xdata is not None:
                self.frameSelected.emit(int(round(event.xdata)))

        def _on_press(self, event):
            if event.button == 1 and event.inaxes is not None:
                self._dragging = True
                self._emit_frame(event)      # jump to the clicked frame

        def _on_motion(self, event):
            if self._dragging:               # scrub while the button is held
                self._emit_frame(event)

        def _on_release(self, event):
            self._dragging = False
else:
    class PlotPanel(QLabel):
        frameSelected = pyqtSignal(int)

        def __init__(self, metrics):
            super().__init__("Install matplotlib to see plots")
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        def set_worm(self, worm):
            pass

        def set_frame(self, f):
            pass

        def set_smooth(self, on):
            pass

        def set_scales(self, scales):
            pass


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    HIT_RADIUS = 45

    def __init__(self, ds: Dataset):
        super().__init__()
        self.ds = ds
        self.frames = ds.frames
        self.worms = ds.worms
        self.masks = ds.masks
        self.n = len(ds.frames)
        self.frame = 0
        self.selected: Worm | None = None
        self._overlay: np.ndarray | None = None
        self._sel_skel: np.ndarray | None = None
        self._last_paint = None
        self._base_title = ds.title
        self._fitted = False
        self.by_id = {w.obj_id: w for w in self.worms}
        self.onion = False
        self.outline_only = False            # show masks as outlines, not filled
        self.px_to_um = None                 # calibration: µm per pixel
        self.fps = None                      # calibration: frames per second
        self._data_dir = ds.csv_path.parent if ds.csv_path else None
        self._frames_dir = ds.frames_dir     # on-disk frames for SAM2
        self.segmenter = None                # lazily-built Sam2Backend
        self._sam_points = {}                # (obj_id, frame) -> [(x, y, label), ...]
        # zoom image adjustments (display only)
        self.img_brightness = 0.0
        self.img_contrast = 1.0
        self.img_gamma = 1.0
        self.img_invert = False
        self._lut = None                     # None = identity (no adjustment)
        self._undo: list = []                # (undo_fn, redo_fn)
        self._redo: list = []
        self._stroke_before = None           # mask snapshot at stroke start
        self._stroke_key = None
        self._pending_delete: set = set()    # files to unlink on Save

        self.area_metric = _find_col(ds.metric_names, ["area_px", "area"]) or "area_px"
        self.length_metric = _find_col(ds.metric_names, ["length_px", "length"]) or "length_px"

        # 4 plot slots: mask metrics (area, length) + kinematics (speed, accel)
        self.plot_metrics = list(ds.metric_names[:2])
        for p in KINEMATIC_METRICS:
            if len(self.plot_metrics) < MAX_PLOTS and p not in self.plot_metrics:
                self.plot_metrics.append(p)
        for w in self.worms:                 # populate speed / acceleration
            self._recompute_kinematics(w)

        # ---- Video panel ----------------------------------------------------
        self.video = VideoView()
        self.video.imageClicked.connect(self._on_video_click)

        prev_btn, next_btn = QPushButton("<"), QPushButton(">")
        self.play_btn = QPushButton("▶")
        for b in (prev_btn, self.play_btn, next_btn):
            b.setFixedWidth(44)
        prev_btn.clicked.connect(lambda: self._step(-1))
        next_btn.clicked.connect(lambda: self._step(1))
        self.play_btn.clicked.connect(self._toggle_play)
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(0, self.n - 1))
        self.slider.valueChanged.connect(self.set_frame)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._interval = max(1, int(1000 / PLAYBACK_FPS))

        nav = QHBoxLayout()
        nav.addStretch()
        nav.addWidget(prev_btn)
        nav.addWidget(self.play_btn)
        nav.addWidget(self.frame_label)
        nav.addWidget(next_btn)
        nav.addStretch()

        self.export_btn = QPushButton("Export overlay video…")
        self.export_btn.setToolTip("Render every frame with your edited masks + "
                                   "skeletons to an .mp4")
        self.export_btn.clicked.connect(self._export_video)
        exp_row = QHBoxLayout()
        exp_row.addStretch()
        exp_row.addWidget(self.export_btn)

        vbox = QVBoxLayout()
        vbox.addWidget(self.video, 1)
        vbox.addLayout(nav)
        vbox.addWidget(self.slider)
        vbox.addLayout(exp_row)
        video_group = QGroupBox("Video")
        video_group.setLayout(vbox)

        # ---- Zoom panel (editing canvas) -----------------------------------
        self.zoomv = ZoomView()
        self.zoomv.paintRequested.connect(self._on_paint)
        self.zoomv.strokeFinished.connect(self._on_stroke_finished)
        self.zoomv.viewportChanged.connect(self._on_viewport_changed)
        self.zoomv.pointAdded.connect(self._on_sam_point)

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.btn_pan = QPushButton("Pan")
        self.btn_draw = QPushButton("Draw")
        self.btn_erase = QPushButton("Erase")
        self.btn_sam_pos = QPushButton("SAM +")
        self.btn_sam_pos.setToolTip("Click a point that IS the worm (SAM2)")
        self.btn_sam_neg = QPushButton("SAM −")
        self.btn_sam_neg.setToolTip("Click a point that is NOT the worm (SAM2)")
        for b, mode in ((self.btn_pan, ZoomView.MODE_PAN),
                        (self.btn_draw, ZoomView.MODE_DRAW),
                        (self.btn_erase, ZoomView.MODE_ERASE),
                        (self.btn_sam_pos, ZoomView.MODE_SAM_POS),
                        (self.btn_sam_neg, ZoomView.MODE_SAM_NEG)):
            b.setCheckable(True)
            b.clicked.connect(lambda _c, m=mode: self._set_mode(m))
            self.tool_group.addButton(b)
        self.btn_pan.setChecked(True)

        brush_lbl = QLabel("Brush")
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(2, 40)
        self.brush_slider.setValue(self.zoomv.brush)
        self.brush_slider.setFixedWidth(80)
        self.brush_slider.valueChanged.connect(self._set_brush)
        self.brush_val = QLabel(str(self.zoomv.brush))

        self.outline_btn = QPushButton("Outline")
        self.outline_btn.setCheckable(True)
        self.outline_btn.setToolTip("Show masks as an outline instead of a "
                                    "filled overlay, so you can see the worm")
        self.outline_btn.toggled.connect(self._toggle_outline)
        self.onion_btn = QPushButton("Onion")
        self.onion_btn.setCheckable(True)
        self.onion_btn.toggled.connect(self._toggle_onion)
        self.prop_btn = QPushButton("Propagate ▶")
        self.prop_btn.setToolTip("Copy this worm's mask to the next frame (P)")
        self.prop_btn.clicked.connect(self._propagate)

        self.sam_run_btn = QPushButton("Segment ▶ all")
        self.sam_run_btn.setToolTip("Propagate the point prompts through every "
                                    "frame with SAM2")
        self.sam_run_btn.clicked.connect(self._sam_propagate)
        self.sam_clear_btn = QPushButton("Clear pts")
        self.sam_clear_btn.setToolTip("Forget the SAM2 point prompts")
        self.sam_clear_btn.clicked.connect(self._sam_clear)

        tools = QHBoxLayout()
        tools.addWidget(self.btn_pan)
        tools.addWidget(self.btn_draw)
        tools.addWidget(self.btn_erase)
        tools.addSpacing(6)
        tools.addWidget(brush_lbl)
        tools.addWidget(self.brush_slider)
        tools.addWidget(self.brush_val)
        tools.addSpacing(6)
        tools.addWidget(self.outline_btn)
        tools.addWidget(self.onion_btn)
        tools.addWidget(self.prop_btn)
        tools.addStretch()

        sam_row = QHBoxLayout()
        sam_row.addWidget(QLabel("SAM2:"))
        sam_row.addWidget(self.btn_sam_pos)
        sam_row.addWidget(self.btn_sam_neg)
        sam_row.addWidget(self.sam_run_btn)
        sam_row.addWidget(self.sam_clear_btn)
        sam_row.addStretch()

        zoom_lbl = QLabel("Zoom")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(int(ZOOM_MIN * 10), int(ZOOM_MAX * 10))
        self.zoom_slider.setValue(int(self.zoomv.zoom * 10))
        self.zoom_slider.valueChanged.connect(self._set_zoom)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setFixedWidth(44)
        self.fit_btn.setToolTip("Frame the selected worm (whole frame if none)")
        self.fit_btn.clicked.connect(self._fit_view)
        self.undo_btn = QPushButton("↶")
        self.undo_btn.setFixedWidth(36)
        self.undo_btn.setToolTip("Undo (Ctrl+Z)")
        self.undo_btn.clicked.connect(self._undo_action)
        self.undo_btn.setEnabled(False)
        self.redo_btn = QPushButton("↷")
        self.redo_btn.setFixedWidth(36)
        self.redo_btn.setToolTip("Redo (Ctrl+Y)")
        self.redo_btn.clicked.connect(self._redo_action)
        self.redo_btn.setEnabled(False)
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setEnabled(False)

        zctl = QHBoxLayout()
        zctl.addWidget(zoom_lbl)
        zctl.addWidget(self.zoom_slider, 1)
        zctl.addWidget(self.fit_btn)
        zctl.addWidget(self.undo_btn)
        zctl.addWidget(self.redo_btn)
        zctl.addWidget(self.save_btn)

        # ---- image adjustments (display only, help segmentation) ------------
        self.bright_slider = QSlider(Qt.Orientation.Horizontal)
        self.bright_slider.setRange(-100, 100)       # additive brightness
        self.bright_slider.setValue(0)
        self.bright_slider.setFixedWidth(70)
        self.bright_slider.valueChanged.connect(self._set_brightness)
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(10, 400)       # /100 -> 0.1 .. 4.0
        self.contrast_slider.setValue(100)
        self.contrast_slider.setFixedWidth(70)
        self.contrast_slider.valueChanged.connect(self._set_contrast)
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(20, 300)          # /100 -> 0.2 .. 3.0
        self.gamma_slider.setValue(100)
        self.gamma_slider.setFixedWidth(70)
        self.gamma_slider.valueChanged.connect(self._set_gamma)
        self.invert_box = QCheckBox("Inv")
        self.invert_box.setToolTip("Invert the zoom image")
        self.invert_box.toggled.connect(self._set_invert)
        self.img_reset = QPushButton("Reset")
        self.img_reset.setObjectName("compact")
        self.img_reset.clicked.connect(self._reset_image)

        img_row = QHBoxLayout()
        img_row.addWidget(QLabel("Img"))
        img_row.addWidget(QLabel("☀"))               # brightness
        img_row.addWidget(self.bright_slider)
        img_row.addWidget(QLabel("◐"))               # contrast
        img_row.addWidget(self.contrast_slider)
        img_row.addWidget(QLabel("γ"))               # gamma
        img_row.addWidget(self.gamma_slider)
        img_row.addWidget(self.invert_box)
        img_row.addWidget(self.img_reset)
        img_row.addStretch()

        zbox = QVBoxLayout()
        zbox.addWidget(self.zoomv, 1)
        zbox.addLayout(tools)
        zbox.addLayout(sam_row)
        zbox.addLayout(zctl)
        zbox.addLayout(img_row)
        zoom_group = QGroupBox("Zoom  (edit here)")
        zoom_group.setLayout(zbox)

        # ---- Labelled worm list --------------------------------------------
        self.list = QListWidget()
        for w in self.worms:
            self.list.addItem(f"#{w.obj_id}")
        self.list.currentRowChanged.connect(self._on_list_row)
        self.open_btn = QPushButton("Open video…")
        self.open_btn.setToolTip("Load another video folder (one containing "
                                 "frames/ and binary/)")
        self.open_btn.clicked.connect(self._open_folder)
        self.import_btn = QPushButton("Import raw frames…")
        self.import_btn.setToolTip("Start from a folder of raw images and "
                                   "segment them with SAM2 point prompts")
        self.import_btn.clicked.connect(self._import_raw)

        self.add_btn = QPushButton("Add")
        self.add_btn.setToolTip("New empty worm, then draw its mask")
        self.add_btn.clicked.connect(self._add_worm)
        self.del_btn = QPushButton("Delete")
        self.del_btn.setToolTip("Delete the selected worm (all frames)")
        self.del_btn.clicked.connect(self._delete_worm)
        self.id_btn = QPushButton("Reassign ID")
        self.id_btn.setToolTip("Reassign the selected worm's ID")
        self.id_btn.clicked.connect(self._reassign_id)
        wrow1 = QHBoxLayout()
        wrow1.setContentsMargins(0, 0, 0, 0)
        wrow1.setSpacing(4)
        wrow1.addWidget(self.add_btn, 1)
        wrow1.addWidget(self.del_btn, 1)
        wbtns = QVBoxLayout()
        wbtns.setContentsMargins(0, 0, 0, 0)
        wbtns.setSpacing(4)
        wbtns.addLayout(wrow1)               # row 1: Add | Delete
        wbtns.addWidget(self.id_btn)         # row 2: Reassign ID (full width)

        self.info = QLabel("Select a worm to inspect or edit it.")
        self.info.setWordWrap(True)
        self.info.setAlignment(Qt.AlignmentFlag.AlignTop)
        lbox = QVBoxLayout()
        lbox.addWidget(self.open_btn)
        lbox.addWidget(self.import_btn)
        lbox.addWidget(self.list, 1)
        lbox.addLayout(wbtns)
        lbox.addWidget(self.info)
        labelled_group = QGroupBox("Labelled")
        labelled_group.setLayout(lbox)
        labelled_group.setMinimumWidth(210)
        labelled_group.setMaximumWidth(230)

        # ---- Plots ----------------------------------------------------------
        self.plot = PlotPanel(self.plot_metrics)
        self.plot.frameSelected.connect(self.set_frame)
        self.smooth_box = QCheckBox("Smooth")
        self.smooth_box.setToolTip(f"Moving-average smoothing of the curves "
                                   f"(window {SMOOTH_WINDOW}) — display only")
        self.smooth_box.toggled.connect(self._toggle_smooth)
        self.units_btn = QPushButton("Units…")
        self.units_btn.setToolTip("Calibrate µm/pixel and fps to show metrics "
                                  "in physical units")
        self.units_btn.clicked.connect(self._set_units)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Recompute area / length / speed / "
                                    "acceleration for all worms from the masks")
        self.refresh_btn.clicked.connect(self._refresh_plots)
        prow = QHBoxLayout()
        prow.addWidget(self.smooth_box)
        prow.addWidget(self.units_btn)
        prow.addStretch()
        prow.addWidget(self.refresh_btn)
        pbox = QVBoxLayout()
        pbox.addLayout(prow)
        pbox.addWidget(self.plot)
        plot_group = QGroupBox("Plots")
        plot_group.setLayout(pbox)

        grid = QGridLayout()
        grid.addWidget(labelled_group, 0, 0, 2, 1)
        grid.addWidget(video_group,    0, 1)
        grid.addWidget(zoom_group,     1, 1)
        grid.addWidget(plot_group,     0, 2, 2, 1)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 2)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 2)
        central = QWidget()
        central.setLayout(grid)
        self.setCentralWidget(central)
        self.resize(1040, 760)

        self.list.setCurrentRow(-1)          # start with NO worm selected
        self.plot.set_worm(None)
        self._update_title()
        if self.n:
            self.set_frame(0)

    # --- rendering -----------------------------------------------------------
    @staticmethod
    def _thicken(mask):
        """3px cross-dilation so a 1px skeleton stays visible when zoomed."""
        out = mask.copy()
        out[1:, :] |= mask[:-1, :]
        out[:-1, :] |= mask[1:, :]
        out[:, 1:] |= mask[:, :-1]
        out[:, :-1] |= mask[:, 1:]
        return out

    def _blend(self, arr, mask_full, bbox, color, alpha):
        """Alpha-blend `color` into `arr` where `mask_full` is set, limited to
        `bbox` for speed. Modifies arr in place."""
        x0, y0, x1, y1 = bbox
        m = mask_full[y0:y1, x0:x1]
        if not m.any():
            return
        sub = arr[y0:y1, x0:x1]
        px = sub[m].astype(np.float32)
        sub[m] = (alpha * np.array(color, np.float32) + (1 - alpha) * px).astype(np.uint8)

    @staticmethod
    def _mask_border(m):
        """1px boundary of a boolean mask (pixels missing a 4-neighbour)."""
        interior = m.copy()
        interior[1:, :] &= m[:-1, :]
        interior[:-1, :] &= m[1:, :]
        interior[:, 1:] &= m[:, :-1]
        interior[:, :-1] &= m[:, 1:]
        return m & ~interior

    def _blend_all_masks(self, arr):
        """Blend every worm's mask into arr (selected worm darker), bbox-limited.
        In outline mode only the mask boundary is drawn, opaque, so the worm
        underneath stays visible."""
        sel_id = self.selected.obj_id if self.selected else None
        for w in self.worms:
            bbox = w.bbox(self.frame)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            sub = self.masks.get(w.obj_id, self.frame)[y0:y1, x0:x1]
            color = np.array(MASK_SEL_RGB if w.obj_id == sel_id else MASK_UNSEL_RGB,
                             np.float32)
            region = arr[y0:y1, x0:x1]
            if self.outline_only:
                edge = self._thicken(self._mask_border(sub))   # crisp, visible
                if edge.any():
                    region[edge] = color.astype(np.uint8)      # opaque outline
            else:
                px = region[sub].astype(np.float32)
                region[sub] = (MASK_ALPHA * color + (1 - MASK_ALPHA) * px).astype(np.uint8)

    def _render_overlay(self, with_extras=True):
        """Composite the original frame with every worm's mask. The Video gets
        masks only; the Zoom additionally gets the image adjustments (LUT),
        onion-skin and skeleton. Blends only within each worm's bbox."""
        raw = self.frames.get(self.frame)
        base = raw.copy()                                # uint8, don't touch cache
        self._blend_all_masks(base)
        self._overlay = base
        self.video.set_frame(np_to_qimage(base))         # video: raw + masks only

        # zoom starts from the adjusted frame when image controls are active
        if self._lut is not None:
            zoom_arr = self._lut[raw]                    # fresh (owned) array
            self._blend_all_masks(zoom_arr)
            owned = True
        else:
            zoom_arr, owned = base, False

        extras = with_extras and self.selected is not None and not self.timer.isActive()
        if extras and self.onion and self.frame > 0:     # previous frame's mask
            pm = self.masks.get(self.selected.obj_id, self.frame - 1)
            if pm.any():
                ys, xs = np.nonzero(pm)
                if not owned:
                    zoom_arr, owned = base.copy(), True
                self._blend(zoom_arr, pm,
                            (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1),
                            ONION_RGB, ONION_ALPHA)
        if extras and self._sel_skel is not None and self._sel_skel.any():
            if not owned:
                zoom_arr, owned = base.copy(), True
            zoom_arr[self._thicken(self._sel_skel)] = np.array(SKEL_RGB, np.uint8)
        if extras:                                   # SAM2 point markers
            pts = self._sam_points.get((self.selected.obj_id, self.frame))
            if pts:
                if not owned:
                    zoom_arr, owned = base.copy(), True
                for px, py, lab in pts:
                    col = (40, 200, 40) if lab == 1 else (225, 40, 40)
                    self._stamp_marker(zoom_arr, px, py, col)
        h, w = base.shape[:2]
        self.zoomv.set_image(np_to_qimage(zoom_arr), w, h)

    @staticmethod
    def _stamp_marker(arr, cx, cy, color, r=3):
        h, w = arr.shape[:2]
        x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
        if x0 < x1 and y0 < y1:
            arr[y0:y1, x0:x1] = np.array(color, np.uint8)

    def _refresh_sel_skel(self):
        """Selected worm's skeleton for the current frame (from skel/ or a live
        recompute), skipped during playback to keep stepping snappy."""
        self._sel_skel = None
        if self.selected is None or self.timer.isActive():
            return
        self._sel_skel = self.masks.skel(self.selected.obj_id, self.frame)

    # --- frame / selection ---------------------------------------------------
    def set_frame(self, f):
        if not self.n:
            return
        f = max(0, min(self.n - 1, f))
        self.frame = f
        self._refresh_sel_skel()
        self._render_overlay()
        self.video.set_selection(self.selected.centroid(f) if self.selected else None)
        self.plot.set_frame(f)
        self._update_info()
        self.frame_label.setText(f"frame {f} / {self.n - 1}")
        self.slider.blockSignals(True)
        self.slider.setValue(f)
        self.slider.blockSignals(False)

    def select_worm(self, worm):
        self.selected = worm
        row = self.worms.index(worm) if worm is not None else -1
        self.list.blockSignals(True)
        self.list.setCurrentRow(row)
        self.list.blockSignals(False)
        self.plot.set_worm(worm)
        self.plot.set_frame(self.frame)
        self._refresh_sel_skel()
        self._render_overlay()
        self.video.set_selection(worm.centroid(self.frame) if worm else None)
        bbox = worm.bbox(self.frame) if worm else None
        if bbox is not None:
            self.zoomv.frame_bbox(bbox)      # zoom in on the picked worm
        self._update_info()

    def _on_list_row(self, row):
        if 0 <= row < len(self.worms):
            self.select_worm(self.worms[row])

    def _on_video_click(self, x, y):
        xi, yi = int(x), int(y)
        # 1) select any worm whose MASK covers the clicked pixel (whole body is
        #    clickable). If bodies overlap, prefer the smaller one.
        hits = []
        for w in self.worms:
            bbox = w.bbox(self.frame)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            if not (x0 <= xi < x1 and y0 <= yi < y1):
                continue
            m = self.masks.get(w.obj_id, self.frame)
            if 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1] and m[yi, xi]:
                area = w.metrics.get(self.area_metric, {}).get(self.frame) \
                    or (x1 - x0) * (y1 - y0)
                hits.append((area, w))
        if hits:
            hits.sort(key=lambda t: t[0])
            self.select_worm(hits[0][1])
            return
        # 2) fall back to nearest centroid within a small radius (thin masks /
        #    just-missed clicks)
        best, best_d = None, self.HIT_RADIUS
        for w in self.worms:
            c = w.centroid(self.frame)
            if c is None:
                continue
            d = ((c[0] - x) ** 2 + (c[1] - y) ** 2) ** 0.5
            if d < best_d:
                best, best_d = w, d
        if best is not None:
            self.select_worm(best)

    # --- editing (in the Zoom canvas) ---------------------------------------
    def _set_mode(self, mode):
        self.zoomv.mode = mode
        if mode != ZoomView.MODE_PAN and self.selected is None:
            self.info.setText("Pick a worm first, then Draw / Erase in the Zoom view.")

    def _set_brush(self, v):
        self.zoomv.brush = v
        self.brush_val.setText(str(v))

    def _set_zoom(self, v):
        self.zoomv.set_zoom(v / 10.0)

    def _on_viewport_changed(self):
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(round(self.zoomv.zoom * 10)))
        self.zoom_slider.blockSignals(False)
        self.video.set_viewport(self.zoomv.viewport_rect())

    def _on_paint(self, x, y):
        if self.selected is None:
            return
        value = (self.zoomv.mode == ZoomView.MODE_DRAW)
        mask = self.masks.get(self.selected.obj_id, self.frame)
        r = self.zoomv.brush
        if self._last_paint is None:
            self._stroke_before = mask.copy()             # snapshot for undo
            self._stroke_key = (self.selected.obj_id, self.frame)
            stamp_disk(mask, x, y, r, value)
        else:
            x0, y0 = self._last_paint
            dist = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
            steps = max(1, int(dist / max(1.0, r * 0.5)))
            for i in range(steps + 1):
                t = i / steps
                stamp_disk(mask, x0 + (x - x0) * t, y0 + (y - y0) * t, r, value)
        self._last_paint = (x, y)
        self._render_overlay(with_extras=False)       # fast live feedback

    def _on_stroke_finished(self):
        self._last_paint = None
        if self.selected is None or self._stroke_before is None:
            return
        wid, idx = self._stroke_key
        before, after = self._stroke_before, self.masks.get(wid, idx).copy()
        self._stroke_before = None
        self.masks.set_dirty(wid, idx)
        self._recompute(self.selected, self.frame)
        self._after_edit()
        self._push_action(lambda: self._restore_mask(wid, idx, before),
                          lambda: self._restore_mask(wid, idx, after))

    def _after_edit(self):
        """Refresh everything after the selected worm's current mask changed."""
        self._refresh_sel_skel()
        self._render_overlay()
        self.plot.set_worm(self.selected)
        self.plot.set_frame(self.frame)
        self._update_info()
        if self.masks.savable:
            self.save_btn.setEnabled(True)
        self._update_title()

    def _apply_mask_stats(self, worm, idx):
        """Recompute area / length / centroid / bbox for one worm-frame from its
        mask (no kinematics — callers do that once)."""
        mask = self.masks.get(worm.obj_id, idx)
        area, length, _skel, centroid, bbox = mask_stats(mask)
        worm.metrics.setdefault(self.area_metric, {})[idx] = area
        if length is not None:
            worm.metrics.setdefault(self.length_metric, {})[idx] = length
        elif idx in worm.metrics.get(self.length_metric, {}):
            worm.metrics[self.length_metric][idx] = float("nan")
        if centroid is None:
            worm.centroids.pop(idx, None)
            worm.bboxes.pop(idx, None)
        else:
            worm.centroids[idx] = centroid
            worm.bboxes[idx] = bbox

    def _recompute(self, worm, idx):
        self._apply_mask_stats(worm, idx)
        self._recompute_kinematics(worm)     # centroid moved -> speed/accel change

    def _recompute_kinematics(self, worm):
        """(Re)fill the worm's speed + acceleration series from its centroids."""
        worm.metrics[SPEED_METRIC] = worm_speed(worm)
        worm.metrics[ACCEL_METRIC] = worm_acceleration(worm)

    def _refresh_plots(self):
        """Recompute every metric (area, length, speed, acceleration) for all
        worms from the current masks, then redraw the plots."""
        for w in self.worms:
            for idx in self.masks.frames_of(w.obj_id):
                self._apply_mask_stats(w, idx)
            self._recompute_kinematics(w)
        self.plot.set_worm(self.selected)
        self.plot.set_frame(self.frame)
        self._render_overlay()
        self._update_info()
        print(f"[refresh] recomputed metrics for {len(self.worms)} worms")

    def _toggle_smooth(self, on):
        self.plot.set_smooth(on)
        self.plot.set_frame(self.frame)

    # --- zoom image adjustments (display only) ------------------------------
    def _rebuild_lut(self):
        """Build a 256-entry lookup table from brightness/contrast/gamma/invert,
        or None when everything is at its default (identity)."""
        b, c, g, inv = (self.img_brightness, self.img_contrast,
                        self.img_gamma, self.img_invert)
        if b == 0.0 and c == 1.0 and abs(g - 1.0) < 1e-6 and not inv:
            self._lut = None
            return
        x = np.arange(256, dtype=np.float32)
        x = (x - 128.0) * c + 128.0 + b              # contrast about mid-grey + brightness
        x = np.clip(x, 0, 255)
        if abs(g - 1.0) > 1e-6:
            x = 255.0 * np.power(x / 255.0, 1.0 / g)  # gamma
        if inv:
            x = 255.0 - x
        self._lut = np.clip(x, 0, 255).astype(np.uint8)

    def _apply_image_adjust(self):
        self._rebuild_lut()
        self._render_overlay()

    def _set_brightness(self, v):
        self.img_brightness = float(v)
        self._apply_image_adjust()

    def _set_contrast(self, v):
        self.img_contrast = v / 100.0
        self._apply_image_adjust()

    def _set_gamma(self, v):
        self.img_gamma = v / 100.0
        self._apply_image_adjust()

    def _set_invert(self, on):
        self.img_invert = bool(on)
        self._apply_image_adjust()

    def _reset_image(self):
        self.img_brightness, self.img_contrast = 0.0, 1.0
        self.img_gamma, self.img_invert = 1.0, False
        for w, val in ((self.bright_slider, 0), (self.contrast_slider, 100),
                       (self.gamma_slider, 100)):
            w.blockSignals(True)
            w.setValue(val)
            w.blockSignals(False)
        self.invert_box.blockSignals(True)
        self.invert_box.setChecked(False)
        self.invert_box.blockSignals(False)
        self._apply_image_adjust()

    # --- open another video folder ------------------------------------------
    def _open_folder(self):
        if self.masks.dirty and QMessageBox.question(
                self, "Unsaved edits",
                "You have unsaved edits. Discard them and open another video?",
                QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel) \
                != QMessageBox.StandardButton.Discard:
            return
        start = str(self._data_dir.parent) if self._data_dir else str(Path.home())
        d = QFileDialog.getExistingDirectory(
            self, "Select a video folder (containing frames/ and binary/)", start)
        if not d:
            return
        ds = load_dataset(Path(d))
        if ds is None:
            QMessageBox.warning(
                self, "Open video",
                f"No frames found in:\n{d}\n\nExpected a folder that contains a "
                f"frames/ subfolder (and usually binary/).")
            return
        self._open_dataset(ds)

    def _import_raw(self):
        if self.masks.dirty and QMessageBox.question(
                self, "Unsaved edits",
                "You have unsaved edits. Discard them and import raw frames?",
                QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel) \
                != QMessageBox.StandardButton.Discard:
            return
        start = str(self._frames_dir.parent) if self._frames_dir else str(Path.home())
        d = QFileDialog.getExistingDirectory(
            self, "Select a folder of raw frames", start)
        if not d:
            return
        ds = build_raw_dataset(Path(d))
        if ds is None:
            QMessageBox.warning(self, "Import raw frames",
                                f"No image files found in:\n{d}")
            return
        self._open_dataset(ds)

    # --- SAM2 point-prompt segmentation -------------------------------------
    def _ensure_segmenter(self):
        if self.segmenter is not None:
            return True
        if self._frames_dir is None:
            QMessageBox.warning(
                self, "SAM2",
                "SAM2 needs the frames on disk. Use “Import raw frames…” to "
                "start from a folder of images.")
            return False
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.segmenter = SEGMENTER_FACTORY(
                self.frames, self.frames.get(0).shape[:2])
        except Exception as err:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(
                self, "SAM2 unavailable",
                f"Could not start SAM2:\n\n{err}\n\nInstall the `sam2` package "
                f"and set SAM2_CFG and SAM2_CKPT to the config + checkpoint.")
            return False
        QApplication.restoreOverrideCursor()
        return True

    def _on_sam_point(self, x, y, label):
        """A positive/negative click for the selected worm on this frame."""
        if self.selected is None:
            self.info.setText("Select or Add a worm first, then click SAM points.")
            return
        if not self._ensure_segmenter():
            return
        oid = self.selected.obj_id
        pts = self._sam_points.setdefault((oid, self.frame), [])
        pts.append((float(x), float(y), int(label)))
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            mask = self.segmenter.add_points(
                self.frame, oid, [(p[0], p[1]) for p in pts], [p[2] for p in pts])
        except Exception as err:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "SAM2", f"add_points failed:\n\n{err}")
            return
        QApplication.restoreOverrideCursor()
        if mask is not None:
            before = self.masks.get(oid, self.frame).copy()
            self.masks.set(oid, self.frame, mask.astype(bool))
            self._recompute(self.selected, self.frame)
            self._after_edit()
            after = self.masks.get(oid, self.frame).copy()
            self._push_action(
                lambda: self._restore_mask(oid, self.frame, before),
                lambda: self._restore_mask(oid, self.frame, after))
        self._render_overlay()               # show the new point marker

    def _sam_propagate(self):
        if self.selected is None and not self._sam_points:
            self.info.setText("Add SAM points on a worm first.")
            return
        if not self._ensure_segmenter():
            return
        touched = set()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for frame_idx, masks in self.segmenter.propagate():
                for oid, m in masks.items():
                    self.masks.set(oid, frame_idx, np.asarray(m, bool))
                    touched.add(oid)
        except Exception as err:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "SAM2", f"Propagation failed:\n\n{err}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        # ensure a Worm object exists for every segmented id, then recompute all
        for oid in touched:
            if oid not in self.by_id:
                self._insert_worm(Worm(obj_id=oid))
        for oid in touched:
            w = self.by_id[oid]
            for idx in self.masks.frames_of(oid):
                self._apply_mask_stats(w, idx)
            self._recompute_kinematics(w)
        if self.masks.savable:
            self.save_btn.setEnabled(True)
        self._update_title()
        self.plot.set_worm(self.selected)
        self.plot.set_frame(self.frame)
        self._render_overlay()
        self._update_info()
        print(f"[sam2] propagated to worms {sorted(touched)}")

    def _sam_clear(self):
        self._sam_points.clear()
        if self.segmenter is not None:
            try:
                self.segmenter.reset()
            except Exception as err:
                print(f"[sam2] reset failed: {err}")
        self._render_overlay()

    def _open_dataset(self, ds):
        """Show a fresh window for `ds` and close this one."""
        win = MainWindow(ds)
        win.resize(self.size())
        win.move(self.pos())
        win.show()
        _WINDOWS.append(win)
        if self in _WINDOWS:
            _WINDOWS.remove(self)
        self.close()

    # --- physical units ------------------------------------------------------
    def _unit_scales(self):
        """metric name -> (factor, label). Empty (px) until calibrated."""
        if not self.px_to_um:
            return {}
        u = self.px_to_um
        scales = {self.area_metric: (u * u, "area (µm²)"),
                  self.length_metric: (u, "length (µm)")}
        if self.fps:
            scales[SPEED_METRIC] = (u * self.fps, "speed (µm/s)")
            scales[ACCEL_METRIC] = (u * self.fps * self.fps, "accel (µm/s²)")
        else:
            scales[SPEED_METRIC] = (u, "speed (µm/frame)")
            scales[ACCEL_METRIC] = (u, "accel (µm/frame²)")
        return scales

    def _plot_specs(self):
        """(name, label, factor) for the four metrics, honouring calibration."""
        scales = self._unit_scales()
        specs = []
        for name in (self.area_metric, self.length_metric,
                     SPEED_METRIC, ACCEL_METRIC):
            factor, label = scales.get(name, (1.0, name))
            specs.append((name, label, factor))
        return specs

    def _set_units(self):
        um, ok = QInputDialog.getDouble(
            self, "Calibration", "µm per pixel  (0 = keep pixels):",
            self.px_to_um or 0.0, 0.0, 1e6, 5)
        if not ok:
            return
        fps, ok2 = QInputDialog.getDouble(
            self, "Calibration", "frames per second  (0 = per-frame):",
            self.fps or 0.0, 0.0, 1e6, 3)
        if not ok2:
            return
        self.px_to_um = um or None
        self.fps = fps or None
        self.plot.set_scales(self._unit_scales())
        self.plot.set_frame(self.frame)
        self._update_info()

    # --- export corrected overlay video -------------------------------------
    def _export_frame(self, f):
        """Original frame + per-worm coloured masks + red skeletons (reflects
        the current, edited masks)."""
        out = self.frames.get(f).copy()
        for w in self.worms:
            bbox = w.bbox(f)
            if bbox is None:
                continue
            self._blend(out, self.masks.get(w.obj_id, f), bbox,
                        color_for(w.obj_id).astype(np.float32), 0.5)
        for w in self.worms:
            sk = self.masks.skel(w.obj_id, f)
            if sk is not None and sk.any():
                out[sk] = np.array(SKEL_RGB, np.uint8)
        return out

    def _export_video(self):
        try:
            import imageio
        except Exception:
            QMessageBox.warning(self, "Export video",
                                "This needs the imageio package:\n\n"
                                "  pip install imageio")
            return
        out_dir = self.ds.csv_path.parent if self.ds.csv_path else Path.cwd()
        fps = self.fps or PLAYBACK_FPS

        # Try a real .mp4 via the FFMPEG plugin; fall back to PNG frames if the
        # ffmpeg backend isn't installed (imageio would otherwise mis-route the
        # .mp4 to the tifffile plugin and crash on the 'fps' argument).
        writer = None
        try:
            writer = imageio.get_writer(str(out_dir / "corrected_overlay.mp4"),
                                        format="FFMPEG", fps=fps, codec="libx264",
                                        macro_block_size=None)
        except Exception as err:
            print(f"[export] ffmpeg writer unavailable ({err}); writing PNGs")

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if writer is not None:
                for f in range(self.n):
                    writer.append_data(self._export_frame(f))
                writer.close()
                target = out_dir / "corrected_overlay.mp4"
                msg = f"Saved {self.n} frames to:\n{target}"
            else:
                frames_dir = out_dir / "corrected_overlay"
                frames_dir.mkdir(parents=True, exist_ok=True)
                for f in range(self.n):
                    key = self.ds.idx_to_key.get(f, f)
                    imageio.imwrite(str(frames_dir / f"{int(key):05d}.png"),
                                    self._export_frame(f))
                target = frames_dir
                msg = (f"ffmpeg not available, so saved {self.n} PNG frames to:\n"
                       f"{frames_dir}\n\nFor an .mp4, install it with:\n"
                       f"  pip install imageio-ffmpeg")
        except Exception as err:
            if writer is not None:
                writer.close()
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Export video", f"Export failed:\n\n{err}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        print(f"[export] wrote {self.n} frames -> {target}")
        QMessageBox.information(self, "Export video", msg)

    # --- undo / redo ---------------------------------------------------------
    def _push_action(self, undo_fn, redo_fn):
        self._undo.append((undo_fn, redo_fn))
        if len(self._undo) > 100:
            self._undo.pop(0)
        self._redo.clear()
        self._sync_undo_buttons()

    def _undo_action(self):
        if not self._undo:
            return
        u, r = self._undo.pop()
        u()
        self._redo.append((u, r))
        self._sync_undo_buttons()

    def _redo_action(self):
        if not self._redo:
            return
        u, r = self._redo.pop()
        r()
        self._undo.append((u, r))
        self._sync_undo_buttons()

    def _sync_undo_buttons(self):
        self.undo_btn.setEnabled(bool(self._undo))
        self.redo_btn.setEnabled(bool(self._redo))

    def _restore_mask(self, wid, idx, arr):
        """Set a worm's mask for one frame and show the result (for undo/redo)."""
        self.masks.set(wid, idx, arr.copy())
        w = self.by_id.get(wid)
        if w is not None:
            self._recompute(w, idx)
            if w is not self.selected:
                self.select_worm(w)
        self.set_frame(idx)
        self.plot.set_worm(self.selected)
        self.plot.set_frame(self.frame)
        if self.masks.savable:
            self.save_btn.setEnabled(True)
        self._update_title()

    # --- add / delete / reassign worms --------------------------------------
    def _insert_worm(self, w, row=None):
        if row is None:
            row = len(self.worms)
        self.worms.insert(row, w)
        self.by_id[w.obj_id] = w
        self.list.blockSignals(True)
        self.list.insertItem(row, f"#{w.obj_id}")
        self.list.blockSignals(False)

    def _remove_worm_obj(self, w):
        if w not in self.worms:
            return
        row = self.worms.index(w)
        self.worms.pop(row)
        self.by_id.pop(w.obj_id, None)
        self.list.blockSignals(True)
        self.list.takeItem(row)
        self.list.blockSignals(False)
        if self.selected is w:
            self.selected = None
            self.list.setCurrentRow(-1)
            self.plot.set_worm(None)
            self._sel_skel = None
            self._render_overlay()
            self._update_info()

    def _add_worm(self):
        new_id = (max(self.by_id) + 1) if self.by_id else 0
        w = Worm(obj_id=new_id)
        self._insert_worm(w)
        self.select_worm(w)
        self.btn_draw.setChecked(True)
        self._set_mode(ZoomView.MODE_DRAW)
        self.info.setText(f"Worm #{new_id}\nDraw its mask on this frame.")
        self._push_action(lambda: self._remove_worm_obj(w),
                          lambda: (self._insert_worm(w), self.select_worm(w)))

    def _delete_worm(self):
        w = self.selected
        if w is None:
            return
        if QMessageBox.question(self, "Delete worm",
                                f"Delete worm #{w.obj_id} from all frames?") \
                != QMessageBox.StandardButton.Yes:
            return
        masks, paths, files = self.masks.pop_worm(w.obj_id)
        self._pending_delete.update(files)
        self._remove_worm_obj(w)
        if self.masks.savable:
            self.save_btn.setEnabled(True)
        self._update_title()

        def undo():
            self.masks.restore_worm(masks, paths)
            self._pending_delete.difference_update(files)
            self._insert_worm(w)
            self.select_worm(w)
            self._update_title()

        def redo():
            self.masks.pop_worm(w.obj_id)
            self._pending_delete.update(files)
            self._remove_worm_obj(w)
            self._update_title()
        self._push_action(undo, redo)

    def _reassign_id(self):
        w = self.selected
        if w is None:
            return
        val, ok = QInputDialog.getInt(self, "Reassign worm ID",
                                      "New ID:", w.obj_id, 0, 9999)
        if not ok or val == w.obj_id:
            return
        if val in self.by_id:
            QMessageBox.warning(self, "ID in use",
                                f"Worm #{val} already exists. Merging isn't "
                                f"supported — pick a free ID.")
            return
        old = w.obj_id
        self._do_reassign(w, old, val)
        self._push_action(lambda: self._do_reassign(w, val, old),
                          lambda: self._do_reassign(w, old, val))

    def _do_reassign(self, w, old, new):
        old_files = self.masks.rename_worm(old, new)
        # old-named files on disk are now garbage; but we're about to (re)write
        # files under `new`, so cancel any pending deletion of that name.
        self._pending_delete = {p for p in self._pending_delete
                                if f"worm{new:02d}_" not in Path(p).name}
        self._pending_delete.update(old_files)
        self.by_id.pop(old, None)
        w.obj_id = new
        self.by_id[new] = w
        row = self.worms.index(w)
        self.list.item(row).setText(f"#{new}")
        if self.masks.savable:
            self.save_btn.setEnabled(True)
        self._update_title()
        self.select_worm(w)

    # --- onion-skin / propagate / fit ---------------------------------------
    def _toggle_onion(self, on):
        self.onion = on
        self._render_overlay()

    def _toggle_outline(self, on):
        self.outline_only = bool(on)
        self._render_overlay()

    def _propagate(self):
        """Copy the selected worm's current mask into the next frame."""
        w = self.selected
        if w is None or self.frame >= self.n - 1:
            return
        nxt = self.frame + 1
        before = self.masks.get(w.obj_id, nxt).copy()
        after = self.masks.get(w.obj_id, self.frame).copy()
        self.masks.set(w.obj_id, nxt, after.copy())
        self._recompute(w, nxt)
        self.set_frame(nxt)                              # step onto the new frame
        if self.masks.savable:
            self.save_btn.setEnabled(True)
        self._update_title()
        self._push_action(lambda: self._restore_mask(w.obj_id, nxt, before),
                          lambda: self._restore_mask(w.obj_id, nxt, after))

    def _fit_view(self):
        bbox = self.selected.bbox(self.frame) if self.selected else None
        if bbox is not None:
            self.zoomv.frame_bbox(bbox)
        else:
            self.zoomv.frame_fit()

    # --- info / title --------------------------------------------------------
    def _update_info(self):
        w = self.selected
        if w is None:
            self.info.setText("Select a worm to inspect or edit it.")
            return
        c = w.centroid(self.frame)
        lines = [f"Worm #{w.obj_id}"]
        if c is not None:
            lines.append(f"pos: ({c[0]:.0f}, {c[1]:.0f})")
            scales = self._unit_scales()
            for name in (self.area_metric, self.length_metric,
                         SPEED_METRIC, ACCEL_METRIC):
                v = w.metrics.get(name, {}).get(self.frame)
                if v is None or v != v:
                    continue
                factor, label = scales.get(name, (1.0, name))
                lines.append(f"{label}: {v * factor:.1f}")
        else:
            lines.append("(absent this frame)")
        self.info.setText("\n".join(lines))

    def _update_title(self):
        self.setWindowTitle(self._base_title + (" *" if self.masks.dirty else ""))

    # --- saving --------------------------------------------------------------
    def _save(self):
        if not self.masks.savable:
            return
        for wid, idx in list(self.masks.dirty):
            self.masks.save(wid, idx)
        for p in list(self._pending_delete):            # deleted / renamed files
            try:
                Path(p).unlink(missing_ok=True)
            except OSError as err:
                print(f"[save] could not delete {p}: {err}")
        self._pending_delete.clear()
        self._write_csv()
        if save_measurements_plot(self.ds.csv_path.parent / "measurements_plot.png",
                                  self.worms, self._plot_specs()):
            print("[save] regenerated measurements_plot.png")
        self.save_btn.setEnabled(False)
        self._update_title()
        print("[save] masks, skeletons and measurements.csv written")

    def _write_csv(self):
        if self.ds.csv_path is None:
            return

        def num(v, nd):
            return "" if v is None or v != v else round(float(v), nd)

        cal = bool(self.px_to_um)
        u = self.px_to_um or 1.0
        fps = self.fps or None
        fields = ["frame", "worm_id", "area_px", "length_px",
                  "speed_px_per_frame", "accel_px_per_frame2"]
        if cal:
            fields += ["area_um2", "length_um"]
            fields += ["speed_um_per_s", "accel_um_per_s2"] if fps \
                else ["speed_um_per_frame", "accel_um_per_frame2"]

        rows = []
        for w in self.worms:
            idxs = set().union(*(w.metrics.get(m, {}) for m in
                                 (self.area_metric, self.length_metric,
                                  SPEED_METRIC, ACCEL_METRIC))) or set()
            for idx in sorted(idxs):
                area = w.metrics.get(self.area_metric, {}).get(idx)
                length = w.metrics.get(self.length_metric, {}).get(idx)
                speed = w.metrics.get(SPEED_METRIC, {}).get(idx)
                accel = w.metrics.get(ACCEL_METRIC, {}).get(idx)
                row = {
                    "frame": self.ds.idx_to_key.get(idx, idx),
                    "worm_id": w.obj_id,
                    "area_px": "" if area is None else int(area),
                    "length_px": num(length, 3),
                    "speed_px_per_frame": num(speed, 4),
                    "accel_px_per_frame2": num(accel, 4),
                }
                if cal:
                    row["area_um2"] = num(None if area is None else area * u * u, 4)
                    row["length_um"] = num(None if length is None else length * u, 4)
                    sf = u * fps if fps else u
                    af = u * fps * fps if fps else u
                    sk = "speed_um_per_s" if fps else "speed_um_per_frame"
                    ak = "accel_um_per_s2" if fps else "accel_um_per_frame2"
                    row[sk] = num(None if speed is None else speed * sf, 5)
                    row[ak] = num(None if accel is None else accel * af, 5)
                rows.append(row)
        rows.sort(key=lambda r: (r["frame"], r["worm_id"]))
        with open(self.ds.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    # --- playback / keys -----------------------------------------------------
    def _step(self, delta):
        self._pause()
        self.set_frame(self.frame + delta)

    def _toggle_play(self):
        self._pause() if self.timer.isActive() else self._play()

    def _play(self):
        if not self.n:
            return
        if self.frame >= self.n - 1:
            self.set_frame(0)
        self.timer.start(self._interval)
        self.play_btn.setText("⏸")

    def _pause(self):
        was_active = self.timer.isActive()
        self.timer.stop()
        self.play_btn.setText("▶")
        if was_active:                       # redraw skeleton now that we stopped
            self._refresh_sel_skel()
            self._render_overlay()

    def _tick(self):
        if self.frame >= self.n - 1:
            self._pause()
            return
        self.set_frame(self.frame + 1)

    def keyPressEvent(self, e):
        ctrl = e.modifiers() & Qt.KeyboardModifier.ControlModifier
        shift = e.modifiers() & Qt.KeyboardModifier.ShiftModifier
        key = e.key()
        if ctrl and key == Qt.Key.Key_Z and not shift:
            self._undo_action()
        elif ctrl and (key == Qt.Key.Key_Y or (key == Qt.Key.Key_Z and shift)):
            self._redo_action()
        elif ctrl and key == Qt.Key.Key_S:
            self._save()
        elif key == Qt.Key.Key_P:
            self._propagate()
        elif key == Qt.Key.Key_O:
            self.onion_btn.toggle()
        elif key == Qt.Key.Key_Space:
            self._toggle_play()
        elif key == Qt.Key.Key_Left:
            self._step(-1)
        elif key == Qt.Key.Key_Right:
            self._step(1)
        else:
            super().keyPressEvent(e)

    def showEvent(self, e):
        super().showEvent(e)
        if not self._fitted and self.selected is None and self.zoomv.has_image():
            self.zoomv.frame_fit()           # sensible default framing at startup
            self._fitted = True

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.selected is None and self.zoomv.has_image():
            self.zoomv.frame_fit()


# --------------------------------------------------------------------------- #
_WINDOWS = []          # keep references so opened windows aren't garbage-collected


def main():
    DATA_DIR = Path(__file__).resolve().parent / "data" / "AFA-day5-1.2x-0027"

    ds = None
    if DATA_DIR.is_dir():
        ds = load_dataset(DATA_DIR)
        if ds is None:
            print(f"[warn] no frames in {DATA_DIR}; using demo data")
    else:
        print(f"[warn] {DATA_DIR} not found; using demo data")
    if ds is None:
        ds = make_demo()

    if not HAVE_SKIMAGE:
        print("[warn] scikit-image not installed: length/skeleton recompute "
              "disabled (pip install scikit-image)")

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    win = MainWindow(ds)
    _WINDOWS.append(win)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()