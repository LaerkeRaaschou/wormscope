"""Pure-numpy compositing — the pixels behind the overlay and the export.

None of this touches Qt, so what the GUI draws and what the exported video
contains come from exactly the same code. That is worth keeping.
"""

import numpy as np

from wormscope.config import MASK_ALPHA, MASK_SEL_RGB, MASK_UNSEL_RGB
from wormscope.core.models import MaskStore, Worm
from wormscope.render import (blend, blend_all_masks, mask_border, stamp_marker,
                              thicken)


def grey(shape=(20, 30), value=100):
    return np.full((*shape, 3), value, np.uint8)


def rect_mask(shape, x0, y0, x1, y1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


# --- helpers ----------------------------------------------------------------
def test_thicken_grows_a_line_by_one_pixel_each_way():
    """A 1-px skeleton disappears at low zoom; this is what keeps it visible."""
    m = np.zeros((5, 5), bool)
    m[2, 2] = True
    out = thicken(m)
    assert out[2, 2] and out[1, 2] and out[3, 2] and out[2, 1] and out[2, 3]
    assert not out[1, 1], "a cross, not a square"
    assert not m[1, 2], "the input is left alone"


def test_mask_border_is_the_outline_only():
    m = rect_mask((10, 10), 2, 2, 8, 8)
    edge = mask_border(m)
    assert edge[2, 2] and edge[2, 5], "the rim is border"
    assert not edge[4, 4], "the inside is not"
    assert (edge & ~m).sum() == 0, "the border lies inside the mask"


def test_the_border_of_a_single_pixel_is_itself():
    m = np.zeros((5, 5), bool)
    m[2, 2] = True
    assert mask_border(m)[2, 2]


# --- blending ---------------------------------------------------------------
def test_blend_mixes_colour_into_the_masked_pixels_only():
    arr = grey(value=100)
    m = rect_mask((20, 30), 5, 5, 10, 8)
    blend(arr, m, (5, 5, 10, 8), (200, 0, 0), 0.5)
    assert tuple(arr[6, 6]) == (150, 50, 50), "half way between 100 and the colour"
    assert tuple(arr[0, 0]) == (100, 100, 100), "outside the mask is untouched"


def test_blend_is_limited_to_the_bbox():
    """The bbox is a speed optimisation, but it also means a stale bbox would
    clip a worm's overlay — worth knowing which one you are looking at."""
    arr = grey()
    m = rect_mask((20, 30), 0, 0, 30, 20)          # mask covers everything
    blend(arr, m, (5, 5, 10, 8), (255, 255, 255), 1.0)
    assert tuple(arr[6, 6]) == (255, 255, 255)
    assert tuple(arr[0, 0]) == (100, 100, 100), "outside the bbox, nothing changed"


def test_blend_of_an_empty_region_does_nothing():
    arr = grey()
    before = arr.copy()
    blend(arr, np.zeros((20, 30), bool), (0, 0, 30, 20), (255, 0, 0), 1.0)
    assert np.array_equal(arr, before)


def test_alpha_zero_and_one_are_the_two_extremes():
    arr = grey(value=10)
    m = rect_mask((20, 30), 0, 0, 30, 20)
    blend(arr, m, (0, 0, 30, 20), (250, 250, 250), 0.0)
    assert tuple(arr[1, 1]) == (10, 10, 10)
    blend(arr, m, (0, 0, 30, 20), (250, 250, 250), 1.0)
    assert tuple(arr[1, 1]) == (250, 250, 250)


# --- the whole overlay ------------------------------------------------------
def make_two_worms(shape=(20, 30)):
    masks = MaskStore(shape)
    worms = []
    for wid, x0 in ((1, 2), (2, 15)):
        m = rect_mask(shape, x0, 5, x0 + 6, 10)
        masks.set(wid, 0, m)
        w = Worm(obj_id=wid)
        w.bboxes[0] = (x0, 5, x0 + 6, 10)
        worms.append(w)
    return worms, masks


def test_the_selected_worm_is_drawn_in_a_different_colour():
    worms, masks = make_two_worms()
    arr = grey()
    blend_all_masks(arr, worms, masks, 0, sel_id=1)
    sel = arr[6, 3].astype(int)
    other = arr[6, 16].astype(int)
    assert not np.array_equal(sel, other), \
        "you must be able to see which worm you are editing"
    expected_sel = (MASK_ALPHA * np.array(MASK_SEL_RGB)
                    + (1 - MASK_ALPHA) * 100).astype(np.uint8)
    assert np.array_equal(sel, expected_sel)


def test_with_nothing_selected_every_worm_looks_the_same():
    worms, masks = make_two_worms()
    arr = grey()
    blend_all_masks(arr, worms, masks, 0, sel_id=None)
    assert np.array_equal(arr[6, 3], arr[6, 16])
    expected = (MASK_ALPHA * np.array(MASK_UNSEL_RGB)
                + (1 - MASK_ALPHA) * 100).astype(np.uint8)
    assert np.array_equal(arr[6, 3], expected)


def make_blob(shape=(20, 30), cx=10, cy=10, r=6):
    """A round worm-ish mask, whose bbox it does not completely fill."""
    ys, xs = np.ogrid[:shape[0], :shape[1]]
    m = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    masks = MaskStore(shape)
    masks.set(1, 0, m)
    w = Worm(obj_id=1)
    ys_, xs_ = np.nonzero(m)
    w.bboxes[0] = (xs_.min(), ys_.min(), xs_.max() + 1, ys_.max() + 1)
    return [w], masks


def test_outline_mode_leaves_the_worm_itself_visible():
    """The whole reason for the Outline button: you cannot judge a segmentation
    through a 50% colour wash."""
    worms, masks = make_blob()
    arr = grey()
    blend_all_masks(arr, worms, masks, 0, sel_id=1, outline_only=True)
    assert tuple(arr[10, 10]) == (100, 100, 100), "the middle is the raw frame"
    assert tuple(arr[10, 5]) != (100, 100, 100), "the rim is drawn"


def test_the_outline_is_opaque_not_blended():
    """Half-transparent at 1 px wide would be invisible."""
    worms, masks = make_blob()
    arr = grey()
    blend_all_masks(arr, worms, masks, 0, sel_id=1, outline_only=True)
    assert tuple(arr[10, 5]) == tuple(np.array(MASK_SEL_RGB, np.uint8))


def test_a_mask_that_exactly_fills_its_bbox_gets_no_outline():
    """A known sharp edge, recorded rather than hidden.

    `blend_all_masks` crops the mask to the worm's bbox and takes the border of
    the crop, and `mask_border` treats the array edge as "has a neighbour". A
    real worm touches its bbox at four tangent points, so it loses four pixels
    of rim — invisible. A perfect rectangle touches everywhere and disappears
    entirely. It behaves identically in the pre-refactor version, so changing
    it will also fail the parity test; do that on purpose or not at all."""
    worms, masks = make_two_worms()               # rectangles
    arr = grey()
    blend_all_masks(arr, worms, masks, 0, sel_id=1, outline_only=True)
    assert (arr == 100).all(), \
        "if this now draws something, you fixed the crop-edge case — good, but " \
        "update the parity expectations too"


def test_a_worm_absent_from_the_frame_is_not_drawn():
    worms, masks = make_two_worms()
    worms[1].bboxes.pop(0)                       # worm 2 vanishes
    arr = grey()
    blend_all_masks(arr, worms, masks, 0, sel_id=1)
    assert tuple(arr[6, 16]) == (100, 100, 100)


def test_stamp_marker_draws_a_small_square():
    arr = grey()
    stamp_marker(arr, 10, 6, (0, 255, 0), r=2)
    assert tuple(arr[6, 10]) == (0, 255, 0)
    assert tuple(arr[6, 14]) == (100, 100, 100)


def test_stamp_marker_clips_at_the_edge():
    arr = grey((10, 10))
    stamp_marker(arr, 0, 0, (255, 0, 0), r=3)    # would run off the corner
    assert tuple(arr[0, 0]) == (255, 0, 0)


def test_compositing_never_changes_the_frame_shape_or_dtype():
    worms, masks = make_two_worms()
    arr = grey()
    blend_all_masks(arr, worms, masks, 0, sel_id=1)
    assert arr.dtype == np.uint8 and arr.shape == (20, 30, 3)
