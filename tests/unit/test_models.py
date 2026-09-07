"""The data model: MaskStore, FrameSource and the empty dataset.

MaskStore is the piece to be careful with. It is the only thing that knows
where a mask lives — in memory, in a PNG under binary/, or nowhere yet — and it
has to answer "does this worm exist on this frame?" without inventing an answer
that makes it exist.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from wormscope.core.geometry import HAVE_SKIMAGE
from wormscope.core.models import (Dataset, FrameSource, MaskStore, Worm,
                                   empty_dataset)


def rect(shape=(20, 30), x0=5, y0=5, x1=15, y1=9):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


# --- reading and writing masks ---------------------------------------------
def test_an_unknown_worm_frame_reads_as_an_empty_mask():
    """`get` always returns something paintable, so the brush can start a worm
    on a frame that has nothing yet."""
    store = MaskStore((10, 12))
    m = store.get(99, 3)
    assert m.shape == (10, 12) and not m.any() and m.dtype == bool


def test_get_does_not_make_the_worm_exist():
    """Reading is not creating: after a `get`, `has_mask` must still say no, or
    merely hovering a frame would make a worm appear in the overlay."""
    store = MaskStore((10, 12))
    store.get(99, 3)
    assert not store.has_mask(99, 3)


def test_set_marks_the_frame_dirty():
    store = MaskStore((20, 30))
    store.set(1, 2, rect())
    assert (1, 2) in store.dirty, "Save works off `dirty`"
    assert store.has_mask(1, 2)


def test_frames_of_lists_every_frame_a_worm_touches():
    store = MaskStore((20, 30))
    for f in (5, 1, 3):
        store.set(7, f, rect())
    assert store.frames_of(7) == [1, 3, 5], "sorted"
    assert store.frames_of(8) == []


def test_an_empty_mask_is_not_a_mask():
    store = MaskStore((20, 30))
    store.set(1, 0, np.zeros((20, 30), bool))
    assert not store.has_mask(1, 0), \
        "erasing a worm's whole mask makes it absent, not present-and-blank"


def test_masks_are_stored_per_worm_and_frame():
    store = MaskStore((20, 30))
    store.set(1, 0, rect(x0=0, x1=4))
    store.set(2, 0, rect(x0=10, x1=14))
    store.set(1, 1, rect(x0=20, x1=24))
    assert store.get(1, 0).sum() == 16 and store.get(2, 0).sum() == 16
    assert not np.array_equal(store.get(1, 0), store.get(1, 1))


# --- files ------------------------------------------------------------------
def test_a_mask_registered_by_path_is_read_from_disk(tmp_path):
    arr = np.zeros((10, 12), np.uint8)
    arr[2:6, 3:9] = 255
    p = tmp_path / "worm01_frame_0000.png"
    Image.fromarray(arr, "L").save(p)
    store = MaskStore((10, 12), binary_dir=tmp_path)
    store.register(1, 0, path=p)
    assert store.get(1, 0).sum() == 24
    assert store.has_mask(1, 0)


def test_save_writes_a_png_and_clears_the_dirty_flag(tmp_path):
    store = MaskStore((10, 12), binary_dir=tmp_path / "binary",
                      skel_dir=tmp_path / "skel",
                      idx_to_stem={0: "frame_0000.png"})
    store.set(1, 0, rect((10, 12), 2, 2, 8, 6))
    store.save(1, 0)
    written = tmp_path / "binary" / "worm01_frame_0000.png"
    assert written.is_file(), "named worm{id:02d}_{frame stem}"
    assert (1, 0) not in store.dirty
    assert (np.asarray(Image.open(written).convert("L")) > 127).sum() == 24


def test_an_in_memory_store_is_not_savable():
    """The demo (and anything opened without a binary/ folder) must not try to
    write files; Save is a no-op rather than an error."""
    store = MaskStore((10, 12))
    assert not store.savable
    store.set(1, 0, rect((10, 12)))
    store.save(1, 0)
    assert (1, 0) not in store.dirty


def test_saving_also_writes_the_skeleton(tmp_path):
    if not HAVE_SKIMAGE:
        pytest.skip("needs scikit-image")
    store = MaskStore((20, 40), binary_dir=tmp_path / "binary",
                      skel_dir=tmp_path / "skel",
                      idx_to_stem={0: "frame_0000.png"})
    store.set(1, 0, rect((20, 40), 5, 8, 35, 12))
    store.save(1, 0)
    assert (tmp_path / "skel" / "worm01_frame_0000.png").is_file()


# --- skeletons --------------------------------------------------------------
def test_a_frame_with_no_mask_has_no_skeleton(tmp_path):
    """skel/ often still holds a PNG from an earlier pipeline run for a
    worm-frame whose mask was deleted. Drawing it would put a red skeleton over
    an unsegmented worm."""
    (tmp_path / "skel").mkdir()
    stale = np.zeros((10, 12), np.uint8)
    stale[5, 2:9] = 255
    Image.fromarray(stale, "L").save(tmp_path / "skel" / "worm01_frame_0000.png")
    store = MaskStore((10, 12), binary_dir=tmp_path / "binary",
                      skel_dir=tmp_path / "skel",
                      idx_to_stem={0: "frame_0000.png"})
    assert store.skel(1, 0) is None


@pytest.mark.skipif(not HAVE_SKIMAGE, reason="needs scikit-image")
def test_an_edited_frame_gets_a_freshly_computed_skeleton():
    store = MaskStore((20, 40))
    store.set(1, 0, rect((20, 40), 5, 8, 35, 12))
    skel = store.skel(1, 0)
    assert skel is not None and skel.any()


# --- deleting and renaming --------------------------------------------------
def test_pop_worm_removes_everything_and_hands_back_the_files(tmp_path):
    store = MaskStore((10, 12), binary_dir=tmp_path / "binary",
                      skel_dir=tmp_path / "skel")
    store.register(1, 0, path=tmp_path / "binary" / "worm01_frame_0000.png")
    store.set(1, 1, rect((10, 12)))
    store.set(2, 0, rect((10, 12)))
    masks, paths, files = store.pop_worm(1)
    assert store.frames_of(1) == [] and store.frames_of(2) == [0]
    assert {f.name for f in files} == {"worm01_frame_0000.png"} or len(files) == 2
    store.restore_worm(masks, paths)
    assert store.frames_of(1) == [0, 1], "undo puts the worm back"


def test_rename_worm_moves_the_masks_and_marks_them_for_rewriting():
    store = MaskStore((10, 12))
    store.set(1, 0, rect((10, 12), 1, 1, 5, 5))
    store.dirty.clear()
    store.rename_worm(1, 9)
    assert store.frames_of(1) == [] and store.frames_of(9) == [0]
    assert store.get(9, 0).sum() == 16
    assert (9, 0) in store.dirty, "the file has to be written under the new name"


def test_rename_returns_the_old_files_so_save_can_delete_them(tmp_path):
    store = MaskStore((10, 12), binary_dir=tmp_path, skel_dir=tmp_path / "skel")
    store.register(1, 0, path=tmp_path / "worm01_frame_0000.png",
                   mask=rect((10, 12)))
    old = store.rename_worm(1, 2)
    assert any("worm01" in p.name for p in old), \
        "the old PNGs are garbage now; Save unlinks them"


# --- FrameSource ------------------------------------------------------------
def test_frame_source_loads_lazily_and_caches():
    loaded = []

    def loader(item):
        loaded.append(item)
        return np.full((4, 4, 3), item, np.uint8)

    src = FrameSource([1, 2, 3], loader)
    assert len(src) == 3 and loaded == [], "nothing is read until it is shown"
    src.get(0)
    src.get(0)
    assert loaded == [1], "a second look does not re-read the file"


def test_frame_source_cache_does_not_grow_without_bound():
    src = FrameSource(list(range(200)), lambda i: np.zeros((2, 2), np.uint8))
    for i in range(120):
        src.get(i)
    assert len(src._cache) <= 60, "a long clip must not be held in memory whole"


# --- the empty dataset ------------------------------------------------------
def test_empty_dataset_has_no_frames_and_no_worms():
    ds = empty_dataset()
    assert ds.is_empty and len(ds.frames) == 0 and ds.worms == []
    assert ds.csv_path is None and ds.frames_dir is None, \
        "nothing is guessed: no default folder, no demo data"


def test_a_real_dataset_is_not_empty():
    ds = Dataset(FrameSource([1], lambda x: x), [Worm(obj_id=1)],
                 ["area_px", "length_px"], MaskStore((2, 2)), {0: 0}, None, "t")
    assert not ds.is_empty
