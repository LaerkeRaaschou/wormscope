"""The data model: Worm, FrameSource, MaskStore and Dataset.

These classes know nothing about Qt — they can be used from scripts and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import HAVE_SKIMAGE, skeletonize_crop


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

    def has_mask(self, wid, idx) -> bool:
        """True when this worm-frame really has a non-empty mask.

        Unlike ``get()`` this never invents an empty mask for an unknown
        worm-frame, so asking the question doesn't make the worm look present
        on that frame."""
        key = (wid, idx)
        m = self._masks.get(key)
        if m is not None:
            return bool(m.any())
        p = self._paths.get(key)
        if p is not None and p.exists():
            return bool(self.get(wid, idx).any())
        return False

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
        bp = self._binary_path(wid, idx, create=False)
        if bp is None:
            return None
        sp = self.skel_dir / bp.name
        if sp.exists():
            return np.asarray(Image.open(sp).convert("L")) > 127
        return None

    def skel(self, wid, idx):
        """Skeleton to display for (worm, frame). Prefers a live recompute for
        edited frames (needs skimage), otherwise the saved skel/ PNG, otherwise
        a live recompute. Cached; returns None if nothing is available.

        A frame with no mask has no skeleton, full stop: skel/ often still holds
        a PNG from an earlier pipeline run for a worm-frame whose mask was since
        deleted (or never segmented), and drawing that stale skeleton over an
        unsegmented worm is misleading."""
        key = (wid, idx)
        if not self.has_mask(wid, idx):
            self._skel_cache[key] = None
            return None
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

    def _binary_path(self, wid, idx, create=True) -> Path | None:
        """Where this worm-frame's mask PNG lives. With `create` (the default)
        a path is derived from the frame's filename and remembered, so a new
        mask can be saved; read-only callers pass create=False so that merely
        looking does not make the app think the worm exists on that frame."""
        key = (wid, idx)
        p = self._paths.get(key)
        if p is None and self.binary_dir is not None:
            stem = self.idx_to_stem.get(idx)
            if stem is None:
                return None
            p = self.binary_dir / f"worm{wid:02d}_{stem}"
            if create:
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
    """Everything one video needs: frames, worms, masks and where to save."""
    frames: FrameSource
    worms: list
    metric_names: list         # real metrics from the CSV / masks (area, length)
    masks: MaskStore
    idx_to_key: dict           # display index -> original frame number
    csv_path: Path | None
    title: str
    frames_dir: Path | None = None   # on-disk frames folder (for SAM2), if any

    @property
    def is_empty(self) -> bool:
        """True for the placeholder dataset shown before a video is opened."""
        return len(self.frames) == 0


def empty_dataset(title="wormscope — no video loaded") -> Dataset:
    """A dataset with no frames at all.

    The window opens on this when no folder was given, so nothing is invented:
    no demo worms, no guessed data directory. Everything that needs frames is
    disabled until the user opens a video."""
    return Dataset(FrameSource([], lambda x: x), [], ["area_px", "length_px"],
                   MaskStore((1, 1)), {}, None, title)
