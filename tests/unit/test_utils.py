"""Small helpers — filename parsing, smoothing, colours.

Most of these exist to make files from different pipeline stages line up:
`frame_0042.png`, `worm01_frame_0042.png` and CSV row `42` all have to resolve
to the same frame, and they do it through `last_int`.
"""

from __future__ import annotations

import numpy as np
import pytest

from wormscope.core.utils import (color_for, find_col, last_int, list_images,
                                  moving_average, sort_by_frame_number, to_num)


# --- frame numbers ----------------------------------------------------------
@pytest.mark.parametrize("name, expected", [
    ("frame_0042.png", 42),
    ("0007.png", 7),
    ("worm01_frame_0042.png", 42),          # the LAST run of digits wins
    ("video2_shot3_frame_9.tif", 9),
    ("no_digits_here.png", None),
    ("frame_0042", 42),                      # suffix optional
])
def test_last_int_takes_the_trailing_number(name, expected):
    assert last_int(name) == expected


def test_worm_prefix_does_not_shadow_the_frame_number():
    """This is the alignment rule the whole loader rests on: a mask named for
    both a worm and a frame must resolve to the frame."""
    assert last_int("worm01_frame_0042.png") == last_int("frame_0042.png")


def test_sort_by_frame_number_is_numeric_not_alphabetical():
    files = [__import__("pathlib").Path(n) for n in
             ["f_10.png", "f_9.png", "f_100.png", "f_1.png"]]
    assert [p.name for p in sort_by_frame_number(files)] == \
        ["f_1.png", "f_9.png", "f_10.png", "f_100.png"]


def test_files_without_a_number_sort_last_and_stay_stable():
    from pathlib import Path
    files = [Path("b.png"), Path("f_2.png"), Path("a.png")]
    assert [p.name for p in sort_by_frame_number(files)] == \
        ["f_2.png", "a.png", "b.png"]


def test_list_images_finds_only_image_files(tmp_path):
    (tmp_path / "a.png").touch()
    (tmp_path / "b.TIF").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "sub").mkdir()
    assert sorted(p.name for p in list_images(tmp_path)) == ["a.png", "b.TIF"]


def test_list_images_of_a_missing_folder_is_empty_not_an_error(tmp_path):
    """The loader probes several candidate folders; a missing one must just
    not match."""
    assert list_images(tmp_path / "nope") == []


# --- CSV column matching ----------------------------------------------------
def test_find_col_prefers_an_exact_match():
    assert find_col(["frame_id", "frame"], ["frame"]) == "frame"


def test_find_col_falls_back_to_a_substring():
    assert find_col(["Frame Number", "value"], ["frame"]) == "Frame Number"


def test_find_col_is_case_insensitive():
    assert find_col(["WORM_ID"], ["worm"]) == "WORM_ID"


def test_find_col_tries_candidates_in_order():
    assert find_col(["id", "worm"], ["worm", "id"]) == "worm"


def test_find_col_returns_none_when_nothing_matches():
    assert find_col(["a", "b"], ["frame"]) is None


@pytest.mark.parametrize("value, expected", [
    ("3.5", 3.5), (7, 7.0), ("", None), (None, None), ("n/a", None),
])
def test_to_num_never_raises(value, expected):
    assert to_num(value) == expected


# --- smoothing --------------------------------------------------------------
def test_moving_average_keeps_the_length():
    """The plot pairs smoothed values with the original frame numbers, so a
    shorter output would silently shift the curve sideways."""
    values = list(range(10))
    assert len(moving_average(values, 5)) == 10


def test_moving_average_shrinks_the_window_at_the_ends():
    assert moving_average([0, 0, 3], 3) == [0.0, 1.0, 1.5]


def test_a_window_of_one_changes_nothing():
    assert moving_average([1, 2, 3], 1) == [1, 2, 3]


def test_smoothing_a_flat_series_leaves_it_flat():
    assert moving_average([4.0] * 6, 5) == [4.0] * 6


def test_moving_average_of_nothing_is_nothing():
    assert moving_average([], 5) == []


# --- colours ----------------------------------------------------------------
def test_worm_colours_are_stable_across_runs():
    """The overlay, the export and measurements_plot.png each derive the colour
    from the id, so worm 3 has to be the same colour in all three."""
    assert np.array_equal(color_for(3), color_for(3))


def test_different_worms_get_different_colours():
    assert not np.array_equal(color_for(1), color_for(2))


def test_colours_are_bright_enough_to_see():
    for wid in range(8):
        assert (color_for(wid) >= 60).all(), "kept away from black on purpose"
        assert color_for(wid).dtype == np.uint8
