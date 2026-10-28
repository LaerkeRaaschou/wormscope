"""Reading a pipeline output folder (or a folder of raw frames) into a Dataset.

Expected layout of a per-video folder::

    data/<video>/
        frames/              # ORIGINAL frames  -> shown underneath the masks
        binary/              # worm{id:02d}_{stem}.png -> per-worm masks (EDITABLE)
        skel/                # worm{id:02d}_{stem}.png -> skeletons (recomputed)
        measurements.csv     # per-frame metrics -> Plots panel (rewritten on Save)

Frame alignment is keyed on the trailing integer in each filename, so original
frames, binary masks and CSV rows line up even if their name prefixes differ.
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import MAX_PLOTS, WORM_RE
from ..core.geometry import mask_stats
from ..core.models import Dataset, FrameSource, MaskStore, Worm
from ..core.utils import (find_col, last_int, list_images, sort_by_frame_number,
                          to_num)


def load_image_rgb(path) -> np.ndarray:
    """Any image file -> uint8 RGB array.

    16-bit and float images (common for microscope TIFFs) are stretched
    between their 0.5 and 99.5 percentiles; a plain convert("RGB") would clip
    them to white."""
    img = Image.open(path)
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
        arr = np.asarray(img, dtype=np.float64)
        lo, hi = np.percentile(arr, (0.5, 99.5))
        arr = np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1) * 255
        return np.stack([arr.astype(np.uint8)] * 3, axis=-1)
    return np.asarray(img.convert("RGB"))


def find_frames_dir(data_dir: Path):
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
        if list_images(c):
            return c
    return None


def load_frames(data_dir: Path):
    """Load original display frames. Returns
    (FrameSource, key_to_idx, idx_to_stem, n)."""
    folder = find_frames_dir(data_dir)
    if folder is None:
        return None, {}, {}, 0
    files = sort_by_frame_number(list_images(folder))
    key_to_idx, idx_to_stem = {}, {}
    for idx, p in enumerate(files):
        idx_to_stem[idx] = p.stem + ".png"
        k = last_int(p.name)
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
    for p in sorted(list_images(mask_dir)):
        mw = WORM_RE.search(p.name)
        if not mw:
            continue
        wid = int(mw.group(1))
        fk = last_int(p.name[mw.end():])
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


def load_metrics(data_dir: Path, worms, key_to_idx, n_frames):
    """Fill worm.metrics from measurements.csv. Returns the metric names found."""
    csv_path = data_dir / "measurements.csv"
    if not csv_path.is_file():
        print("[metrics] no measurements.csv")
        return []
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    cols = list(rows[0].keys())
    frame_col = find_col(cols, ["frame", "index", "t"])
    id_col = find_col(cols, ["worm", "obj_id", "track", "label", "id"])
    metric_cols = [c for c in cols if c not in (frame_col, id_col)
                   and sum(to_num(r.get(c)) is not None for r in rows[:50]) > len(rows[:50]) * 0.8]
    by_id = {w.obj_id: w for w in worms}
    single = worms[0] if len(worms) == 1 else None
    for r in rows:
        fk = to_num(r.get(frame_col)) if frame_col else None
        fk = int(fk) if fk is not None else None
        idx = key_to_idx.get(fk)
        if idx is None and fk is not None and 0 <= fk < n_frames:
            idx = fk
        if idx is None:
            continue
        if id_col is not None:
            wid = to_num(r.get(id_col))
            w = by_id.get(int(wid)) if wid is not None else None
        else:
            w = single
        if w is None:
            continue
        for c in metric_cols:
            v = to_num(r.get(c))
            if v is not None:
                w.metrics.setdefault(c, {})[idx] = v
    names = sorted({m for w in worms for m in w.metrics})
    print(f"[metrics] frame='{frame_col}' id='{id_col}' metrics={names}")
    return names


def seed_metrics_from_masks(worms, masks: MaskStore, names):
    """Derive area / length straight from the masks when there is no CSV."""
    area_name, length_name = names[0], names[1]
    for w in worms:
        for idx in sorted(w.bboxes):
            area, length, *_ = mask_stats(masks.get(w.obj_id, idx))
            w.metrics.setdefault(area_name, {})[idx] = area
            if length is not None:
                w.metrics.setdefault(length_name, {})[idx] = length


def load_dataset(data_dir: Path) -> Dataset | None:
    """Full pipeline output folder -> Dataset (None if it has no frames)."""
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
        seed_metrics_from_masks(worms, masks, names)
    idx_to_key = {idx: k for k, idx in key_to_idx.items()}
    return Dataset(frames, worms, names[:MAX_PLOTS], masks, idx_to_key,
                   data_dir / "measurements.csv", f"wormscope — {data_dir.name}",
                   frames_dir=find_frames_dir(data_dir))


def build_raw_dataset(frames_dir: Path) -> Dataset | None:
    """Start a fresh working dataset from a folder of RAW frames (no masks yet).
    Outputs (binary/, skel/, measurements.csv) go in a sibling '<name>_wormscope'
    folder so the raw frames are never modified. Worms start empty; use the SAM2
    point tools (or Add + brush) to create them."""
    files = sort_by_frame_number(list_images(frames_dir))
    if not files:
        return None
    key_to_idx, idx_to_stem = {}, {}
    for idx, p in enumerate(files):
        idx_to_stem[idx] = p.stem + ".png"
        k = last_int(p.name)
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
