"""Per-worm metric bookkeeping and unit calibration.

`apply_mask_stats` is the bridge between a mask and everything the GUI shows:
edit a mask, this runs, and the info panel, the plots and the CSV all follow.
"""

import numpy as np
import pytest

from wormscope.config import ACCEL_METRIC, SPEED_METRIC
from wormscope.core.metrics import (apply_mask_stats, plot_specs,
                                    recompute_worm, unit_scales)
from wormscope.core.models import MaskStore, Worm


def store_with(mask, wid=1, idx=0, shape=(20, 30)):
    s = MaskStore(shape)
    s.set(wid, idx, mask)
    return s


def rect(shape, x0, y0, x1, y1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_apply_mask_stats_fills_area_centroid_and_bbox():
    w = Worm(obj_id=1)
    masks = store_with(rect((20, 30), 10, 5, 20, 9))
    apply_mask_stats(w, masks, 0, "area_px", "length_px")
    assert w.metrics["area_px"][0] == 40
    assert w.centroids[0] == pytest.approx((14.5, 6.5))
    assert w.bboxes[0] == (10, 5, 20, 9)


def test_erasing_a_mask_removes_the_worm_from_that_frame():
    """Not 'area 0 at the last known place' — the worm has to actually vanish
    from the frame, or it stays clickable and keeps a centroid the speed
    calculation would use."""
    w = Worm(obj_id=1)
    masks = store_with(rect((20, 30), 10, 5, 20, 9))
    apply_mask_stats(w, masks, 0, "area_px", "length_px")
    masks.set(1, 0, np.zeros((20, 30), bool))
    apply_mask_stats(w, masks, 0, "area_px", "length_px")
    assert w.metrics["area_px"][0] == 0
    assert 0 not in w.centroids and 0 not in w.bboxes


def test_a_length_that_can_no_longer_be_measured_becomes_nan():
    """NaN, not a stale number: the plots skip NaN, so the curve breaks where
    the measurement is missing instead of drawing a line through it."""
    w = Worm(obj_id=1)
    masks = store_with(rect((20, 30), 10, 5, 20, 9))
    apply_mask_stats(w, masks, 0, "area_px", "length_px")
    masks.set(1, 0, np.zeros((20, 30), bool))
    apply_mask_stats(w, masks, 0, "area_px", "length_px")
    v = w.metrics["length_px"][0]
    assert v != v, "an unmeasurable length is NaN"


def test_metric_names_are_whatever_the_caller_says():
    """The CSV may call them anything; the window resolves the names once and
    passes them down, so nothing below hardcodes 'area_px'."""
    w = Worm(obj_id=1)
    masks = store_with(rect((20, 30), 0, 0, 4, 4))
    apply_mask_stats(w, masks, 0, "Area", "Len")
    assert "Area" in w.metrics and "area_px" not in w.metrics


def test_recompute_worm_covers_every_frame_and_refreshes_kinematics():
    w = Worm(obj_id=1)
    masks = MaskStore((20, 30))
    for f in range(4):
        masks.set(1, f, rect((20, 30), f, 5, f + 4, 9))
    recompute_worm(w, masks, "area_px", "length_px")
    assert sorted(w.metrics["area_px"]) == [0, 1, 2, 3]
    assert w.metrics[SPEED_METRIC][1] == pytest.approx(1.0), \
        "moving one pixel per frame is 1 px/frame"


# --- calibration ------------------------------------------------------------
def test_without_calibration_everything_stays_in_pixels():
    assert unit_scales("area_px", "length_px") == {}, \
        "an empty dict means 'no conversion' — the caller falls back to 1.0"


def test_micrometres_scale_area_by_the_square():
    scales = unit_scales("area_px", "length_px", px_to_um=2.0)
    assert scales["area_px"][0] == pytest.approx(4.0)
    assert scales["length_px"][0] == pytest.approx(2.0)
    assert "µm" in scales["area_px"][1]


def test_speed_is_per_second_only_when_fps_is_known():
    per_frame = unit_scales("area_px", "length_px", px_to_um=2.0)
    assert per_frame[SPEED_METRIC] == (pytest.approx(2.0), "speed (µm/frame)")
    per_second = unit_scales("area_px", "length_px", px_to_um=2.0, fps=10.0)
    assert per_second[SPEED_METRIC] == (pytest.approx(20.0), "speed (µm/s)")
    assert per_second[ACCEL_METRIC][0] == pytest.approx(200.0), \
        "acceleration carries fps squared"


def test_fps_alone_does_not_convert_anything():
    """Without µm/px there is no length unit to put a second under."""
    assert unit_scales("area_px", "length_px", fps=10.0) == {}


def test_plot_specs_lists_the_four_metrics_in_order():
    specs = plot_specs("area_px", "length_px", px_to_um=2.0, fps=10.0)
    assert [s[0] for s in specs] == ["area_px", "length_px", SPEED_METRIC,
                                     ACCEL_METRIC]
    assert all(len(s) == 3 for s in specs), "(name, label, factor)"


def test_uncalibrated_plot_specs_label_with_the_raw_metric_name():
    specs = plot_specs("area_px", "length_px")
    assert specs[0] == ("area_px", "area_px", 1.0)
