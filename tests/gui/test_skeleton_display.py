"""The red skeleton must correspond to something you can actually see.

``skel/`` often still holds PNGs from an earlier pipeline run, including for
worm-frames whose mask was later deleted or never segmented at all. Drawing one
of those puts a confident red skeleton over an unsegmented worm — which is how
this was noticed in the first place.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from wormscope.config import SKEL_RGB
from wormscope.core.models import (Dataset, FrameSource, MaskStore, Worm)

H, W = 40, 60


def png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(arr, 255, 0).astype(np.uint8), "L").save(path)


@pytest.fixture
def stale_skeleton(tmp_path):
    """Frame 0 has a mask and a skeleton; frame 1 has only a leftover skeleton."""
    body = np.zeros((H, W), bool)
    body[18:24, 10:50] = True
    line = np.zeros((H, W), bool)
    line[21, 10:50] = True

    png(tmp_path / "binary" / "worm01_img_000.png", body)
    png(tmp_path / "skel" / "worm01_img_000.png", line)
    png(tmp_path / "skel" / "worm01_img_001.png", line)      # no mask for it

    store = MaskStore((H, W), binary_dir=tmp_path / "binary",
                      skel_dir=tmp_path / "skel",
                      idx_to_stem={0: "img_000.png", 1: "img_001.png"})
    store.register(1, 0, path=tmp_path / "binary" / "worm01_img_000.png")
    return store, body


def test_a_segmented_frame_still_shows_its_skeleton(stale_skeleton):
    store, _body = stale_skeleton
    assert store.skel(1, 0) is not None


def test_an_unsegmented_frame_shows_none(stale_skeleton):
    store, _body = stale_skeleton
    assert store.skel(1, 1) is None, "the stale skel/ PNG was used"


def test_asking_did_not_invent_a_mask_for_that_frame(stale_skeleton):
    store, _body = stale_skeleton
    assert store.frames_of(1) == [0]
    assert store.has_mask(1, 0) and not store.has_mask(1, 1)


def test_erasing_a_mask_clears_its_skeleton(stale_skeleton):
    store, _body = stale_skeleton
    store.set(1, 0, np.zeros((H, W), bool))
    assert store.skel(1, 0) is None


def test_drawing_it_back_recomputes_one(stale_skeleton):
    store, body = stale_skeleton
    store.set(1, 0, np.zeros((H, W), bool))
    store.set(1, 0, body.copy())
    assert store.skel(1, 0) is not None


def test_the_red_skeleton_reaches_the_screen_only_where_there_is_a_mask(
        make_window, stale_skeleton, monkeypatch):
    """End to end, through the real rendering path: the pixels handed to Qt
    must contain the skeleton colour on frame 0 and not on frame 1."""
    from wormscope.ui import rendering
    store, _body = stale_skeleton

    frames = FrameSource([np.full((H, W, 3), 200, np.uint8) for _ in range(2)],
                         lambda a: a)
    worm = Worm(obj_id=1, centroids={0: (30.0, 21.0)},
                bboxes={0: (10, 18, 50, 24)})
    ds = Dataset(frames, [worm], ["area_px", "length_px"], store,
                 {0: 0, 1: 1}, None, "test")

    drawn = {}
    real = rendering.np_to_qimage
    monkeypatch.setattr(rendering, "np_to_qimage",
                        lambda arr: (drawn.__setitem__("last", np.array(arr)),
                                     real(arr))[1])

    win = make_window(ds)
    win.select_worm(worm)

    def has_red():
        return bool((drawn["last"] == np.array(SKEL_RGB, np.uint8)).all(-1).any())

    win.set_frame(0)
    assert has_red(), "the segmented frame draws its skeleton"
    win.set_frame(1)
    assert not has_red(), "the unsegmented frame draws none"
