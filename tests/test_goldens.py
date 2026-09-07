"""Layer 3: the exact numbers.

A fixed synthetic clip is measured, edited and written out, and every result is
compared against a copy recorded in ``tests/goldens/``. Nothing here says the
numbers are *right* — the unit tests do that. What these say is that they have
not **moved**, which is the question you actually have while refactoring.

A failure means an output changed. Two possibilities:

* you meant it — read the diff, satisfy yourself the new numbers are better,
  then re-record with ``pytest --update-goldens`` and commit the change to
  ``tests/goldens/`` alongside the code change, so the two are reviewable
  together;
* you did not — you have just caught a silent regression, which is the whole
  point of this file.

A numpy, Pillow or scikit-image upgrade can also move a golden by a pixel. That
looks the same from here, so check ``git diff tests/goldens/`` before assuming
it was you: a change you made usually moves one measurement, a library upgrade
usually nudges many by a hair.

Run just this layer:  pytest -m golden
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from wormscope.core.metrics import recompute_worm
from wormscope.core.geometry import HAVE_SKIMAGE, stamp_disk
from wormscope.io.demo import make_demo
from wormscope.io.export import export_frame
from wormscope.io.writers import write_measurements_csv
from wormscope.render import blend_all_masks

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def clip():
    """The same twelve frames every time — `make_demo` is seeded."""
    return make_demo(n_frames=12, size=(200, 140), seed=3)


def digest(arr):
    """A short, stable fingerprint of an array's exact contents."""
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


# --- the demo clip is itself deterministic ---------------------------------
def test_the_demo_clip_is_reproducible():
    """If this fails, every other golden in this file is meaningless."""
    a = make_demo(n_frames=4, size=(120, 90), seed=3)
    b = make_demo(n_frames=4, size=(120, 90), seed=3)
    assert digest(a.frames.get(2)) == digest(b.frames.get(2))
    assert a.worms[0].metrics["area_px"] == b.worms[0].metrics["area_px"]


# --- measurements -----------------------------------------------------------
def test_measured_metrics(clip, golden):
    """Area, length, speed and acceleration for every worm and frame.

    This is the one to watch when touching geometry.py or kinematics.py."""
    for worm in clip.worms:
        recompute_worm(worm, clip.masks, "area_px", "length_px")
    golden("demo_metrics", {w.obj_id: w.metrics for w in clip.worms}, nd=4)


def test_centroids_and_bboxes(clip, golden):
    golden("demo_positions", {w.obj_id: {"centroids": w.centroids,
                                         "bboxes": w.bboxes}
                              for w in clip.worms}, nd=4)


@pytest.mark.skipif(not HAVE_SKIMAGE, reason="needs scikit-image")
def test_skeleton_lengths(clip, golden):
    """Separated out because scikit-image is the library most likely to move
    these — its skeletonize has changed shape between releases before."""
    lengths = {w.obj_id: {f: w.metrics["length_px"][f]
                          for f in sorted(w.metrics.get("length_px", {}))}
               for w in clip.worms}
    golden("demo_skeleton_lengths", lengths, nd=4)


# --- the file that leaves the program ---------------------------------------
def test_measurements_csv(clip, golden, tmp_path):
    out = tmp_path / "measurements.csv"
    write_measurements_csv(out, clip.worms, clip.idx_to_key, "area_px",
                           "length_px")
    golden("demo_csv", out.read_text().splitlines())


def test_measurements_csv_calibrated(clip, golden, tmp_path):
    out = tmp_path / "measurements.csv"
    write_measurements_csv(out, clip.worms, clip.idx_to_key, "area_px",
                           "length_px", px_to_um=0.65, fps=12.0)
    golden("demo_csv_calibrated", out.read_text().splitlines())


# --- the pixels -------------------------------------------------------------
def test_the_overlay_pixels(clip, golden):
    """What the Video panel composites. Covers the mask colours, the alpha and
    the bbox-limited blending in one number."""
    base = clip.frames.get(5).copy()
    blend_all_masks(base, clip.worms, clip.masks, 5, sel_id=2)
    golden("demo_overlay", {"frame5_selected2": digest(base)})


def test_the_outline_pixels(clip, golden):
    base = clip.frames.get(5).copy()
    blend_all_masks(base, clip.worms, clip.masks, 5, sel_id=2, outline_only=True)
    golden("demo_overlay_outline", {"frame5_selected2": digest(base)})


def test_the_exported_frame(clip, golden):
    """Overlay plus skeletons — what ends up in the .mp4."""
    frame = export_frame(clip.frames, clip.worms, clip.masks, 5)
    golden("demo_export_frame", {"frame5": digest(frame)})


# --- a scripted editing session --------------------------------------------
def test_an_edit_sequence_lands_where_it_always_did(clip, golden):
    """Paint, erase, propagate — the operations you do all day, replayed with
    fixed coordinates so the result is a number that can be compared."""
    worm = clip.worms[1]
    wid = worm.obj_id
    cx, cy = worm.centroid(4)

    mask = clip.masks.get(wid, 4)
    stamp_disk(mask, cx - 10, cy - 6, 6, True)
    stamp_disk(mask, cx + 4, cy + 3, 6, True)
    clip.masks.set(wid, 4, mask)
    painted = int(mask.sum())

    stamp_disk(mask, cx, cy, 6, False)
    clip.masks.set(wid, 4, mask)
    erased = int(mask.sum())

    clip.masks.set(wid, 5, clip.masks.get(wid, 4).copy())   # propagate forward
    recompute_worm(worm, clip.masks, "area_px", "length_px")

    golden("demo_edit_sequence", {
        "painted_area": painted,
        "erased_area": erased,
        "mask_digest": digest(clip.masks.get(wid, 4)),
        "propagated_digest": digest(clip.masks.get(wid, 5)),
        "area_after": worm.metrics["area_px"][4],
        "length_after": worm.metrics["length_px"][4],
        "speed_after": worm.metrics["speed"].get(5),
    }, nd=4)
