"""Speed and acceleration from centroids.

The one subtlety worth protecting: a worm can be absent from a frame, so the
gap between two appearances is not always 1. Both series divide by the real gap
so the units stay px/frame — drop that and the numbers silently become
"distance since last seen", which looks fine on a plot and is wrong.
"""

import pytest

from wormscope.config import ACCEL_METRIC, SPEED_METRIC
from wormscope.core.kinematics import (refresh_kinematics, worm_acceleration,
                                       worm_speed)
from wormscope.core.models import Worm


def worm_with(centroids):
    w = Worm(obj_id=1)
    w.centroids = dict(centroids)
    return w


def test_speed_is_distance_per_frame():
    w = worm_with({0: (0.0, 0.0), 1: (3.0, 4.0), 2: (3.0, 4.0)})
    assert worm_speed(w) == {1: pytest.approx(5.0), 2: pytest.approx(0.0)}


def test_the_first_appearance_has_no_speed():
    """There is nothing to compare it against; inventing 0 would put a fake
    dip at the start of every track."""
    w = worm_with({4: (0.0, 0.0), 5: (1.0, 0.0)})
    assert 4 not in worm_speed(w)


def test_a_gap_is_divided_by_the_number_of_frames_missed():
    w = worm_with({0: (0.0, 0.0), 4: (8.0, 0.0)})     # absent for frames 1-3
    assert worm_speed(w)[4] == pytest.approx(2.0), \
        "8 px over 4 frames is 2 px/frame, not 8"


def test_acceleration_is_the_change_in_speed():
    w = worm_with({0: (0.0, 0.0), 1: (1.0, 0.0), 2: (3.0, 0.0), 3: (6.0, 0.0)})
    assert worm_speed(w) == {1: 1.0, 2: 2.0, 3: 3.0}
    assert worm_acceleration(w) == {2: pytest.approx(1.0), 3: pytest.approx(1.0)}


def test_acceleration_can_be_negative():
    w = worm_with({0: (0.0, 0.0), 1: (5.0, 0.0), 2: (6.0, 0.0)})
    assert worm_acceleration(w)[2] == pytest.approx(-4.0)


def test_the_first_speed_sample_has_no_acceleration():
    w = worm_with({0: (0.0, 0.0), 1: (1.0, 0.0)})
    assert worm_acceleration(w) == {}


def test_a_worm_that_never_moves_has_zero_of_both():
    w = worm_with({f: (10.0, 10.0) for f in range(5)})
    assert set(worm_speed(w).values()) == {0.0}
    assert set(worm_acceleration(w).values()) == {0.0}


def test_a_worm_seen_once_has_neither():
    w = worm_with({3: (1.0, 2.0)})
    assert worm_speed(w) == {} and worm_acceleration(w) == {}


def test_refresh_writes_both_series_under_the_configured_names():
    """The plot panel and the CSV writer look these up by the names in
    config.py, so the keys matter as much as the values."""
    w = worm_with({0: (0.0, 0.0), 1: (1.0, 0.0), 2: (3.0, 0.0)})
    refresh_kinematics(w)
    assert set(w.metrics) == {SPEED_METRIC, ACCEL_METRIC}
    assert w.metrics[SPEED_METRIC][2] == pytest.approx(2.0)


def test_refresh_replaces_rather_than_merges():
    """Editing a mask moves a centroid; the old speed for that frame has to go,
    not linger underneath the new one."""
    w = worm_with({0: (0.0, 0.0), 1: (1.0, 0.0)})
    refresh_kinematics(w)
    w.centroids.pop(1)
    refresh_kinematics(w)
    assert w.metrics[SPEED_METRIC] == {}
