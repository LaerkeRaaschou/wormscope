"""Mask geometry: areas, skeleton length, and the brush.

Every expected number here is one you can work out on paper from the shapes
being drawn — that is the point. If a change makes one of these fail, read the
number in the failure and ask whether the new one is also arrivable at by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from wormscope.core.geometry import (HAVE_SKIMAGE, mask_stats,
                                     skeleton_length_px, skeletonize_crop,
                                     stamp_disk, stamp_stroke)


# --- mask_stats -------------------------------------------------------------
def test_mask_stats_of_a_rectangle():
    m = np.zeros((20, 30), bool)
    m[5:9, 10:20] = True                       # 4 rows x 10 cols
    area, length, skel, centroid, bbox = mask_stats(m)
    assert area == 40
    assert bbox == (10, 5, 20, 9), "bbox is (x0, y0, x1, y1), end-exclusive"
    assert centroid == pytest.approx((14.5, 6.5)), \
        "centroid is (x, y) — the mean of the set pixels, not the bbox middle"


def test_mask_stats_of_an_empty_mask():
    """An absent worm must produce no centroid at all, not (0, 0) — the GUI
    tells 'absent this frame' apart from 'at the origin'."""
    area, length, skel, centroid, bbox = mask_stats(np.zeros((10, 10), bool))
    assert (area, length, skel, centroid, bbox) == (0, None, None, None, None)


def test_area_is_a_plain_pixel_count():
    rng = np.random.default_rng(0)
    m = rng.random((16, 16)) > 0.5
    assert mask_stats(m)[0] == int(m.sum())


# --- skeleton length --------------------------------------------------------
def test_skeleton_length_counts_steps_not_pixels():
    """A 1-px line of n pixels is n-1 steps long. Off-by-one here would shift
    every length in measurements.csv."""
    s = np.zeros((5, 12), bool)
    s[2, 1:11] = True                          # 10 pixels in a row
    assert skeleton_length_px(s) == pytest.approx(9.0)


def test_diagonal_steps_count_as_root_two():
    s = np.eye(5, dtype=bool)
    assert skeleton_length_px(s) == pytest.approx(4 * np.sqrt(2))


def test_a_single_pixel_has_no_length():
    s = np.zeros((4, 4), bool)
    s[1, 1] = True
    assert skeleton_length_px(s) == 0.0


@pytest.mark.skipif(not HAVE_SKIMAGE, reason="needs scikit-image")
def test_skeletonize_crop_returns_a_full_size_mask():
    m = np.zeros((40, 60), bool)
    m[10:20, 15:45] = True
    skel = skeletonize_crop(m)
    assert skel.shape == m.shape, "the skeleton is pasted back into the frame"
    assert skel.any() and not skel[m == False].any(), \
        "the skeleton lies inside the mask"


@pytest.mark.skipif(not HAVE_SKIMAGE, reason="needs scikit-image")
def test_a_long_thin_worm_measures_about_its_length():
    m = np.zeros((30, 80), bool)
    m[13:17, 10:70] = True                     # 60 px long, 4 px thick
    _area, length, *_ = mask_stats(m)
    assert length == pytest.approx(59, abs=3), \
        "a 60 px bar should skeletonize to ~59 px of centre line"


def test_skeletonize_crop_of_an_empty_mask_is_none():
    assert skeletonize_crop(np.zeros((8, 8), bool)) is None


# --- the brush --------------------------------------------------------------
def test_stamp_disk_paints_a_circle_of_the_right_radius():
    m = np.zeros((21, 21), bool)
    stamp_disk(m, 10, 10, 5, True)
    assert m[10, 10] and m[10, 5] and m[5, 10], "radius 5 reaches 5 px out"
    assert not m[10, 4], "...and no further"
    assert not m[4, 4], "corners are outside the circle"


def test_stamp_disk_erases_with_value_false():
    m = np.ones((21, 21), bool)
    stamp_disk(m, 10, 10, 4, False)
    assert not m[10, 10] and m[0, 0]


def test_stamp_disk_clips_at_the_edge_instead_of_wrapping():
    m = np.zeros((10, 10), bool)
    stamp_disk(m, 0, 0, 3, True)               # centred on the corner
    assert m[0, 0] and not m[9, 9]


def test_stamp_disk_entirely_outside_does_nothing():
    m = np.zeros((10, 10), bool)
    stamp_disk(m, -50, -50, 2, True)
    assert not m.any()


def test_stamp_stroke_leaves_no_gaps_when_dragging_fast():
    """The brush is stamped per mouse position; without interpolation a quick
    drag paints dots. Every column between the ends must be covered."""
    m = np.zeros((30, 60), bool)
    stamp_stroke(m, 5, 15, 55, 15, 3, True)
    covered = m.any(axis=0)
    assert covered[5:56].all(), "a dragged stroke is continuous"


def test_stamp_stroke_of_zero_length_is_just_a_dot():
    m = np.zeros((20, 20), bool)
    stamp_stroke(m, 10, 10, 10, 10, 2, True)
    assert m[10, 10] and 0 < m.sum() < 30
