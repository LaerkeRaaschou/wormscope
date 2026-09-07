"""Centroid speed / acceleration, in pixels."""

from ..config import ACCEL_METRIC, SPEED_METRIC


def worm_speed(worm):
    """Centroid speed in pixels per frame.

    For each frame where the worm is present, the speed is the Euclidean
    distance from its previous appearance divided by the frame gap:
        speed[f] = |c(f) - c(prev)| / (f - prev)
    Worms can be absent in some frames, so the gap (f - prev) may exceed 1;
    dividing by it keeps the result in px/frame. The first appearance has no
    predecessor and is therefore omitted.
    """
    speed = {}
    prev = None
    for f in sorted(worm.centroids):
        c = worm.centroids[f]
        if prev is not None:
            pf, pc = prev
            dt = f - pf
            if dt > 0:
                speed[f] = ((c[0] - pc[0]) ** 2 + (c[1] - pc[1]) ** 2) ** 0.5 / dt
        prev = (f, c)
    return speed


def worm_acceleration(worm):
    """Centroid acceleration in pixels per frame squared.

    The frame-to-frame change of worm_speed():
        acc[f] = (speed[f] - speed[prev]) / (f - prev)
    Uses the same absence-aware frame gaps as worm_speed(). Needs at least two
    consecutive speed samples, so the first speed sample has no acceleration.
    """
    speed = worm_speed(worm)
    acc = {}
    prev = None
    for f in sorted(speed):
        s = speed[f]
        if prev is not None:
            pf, ps = prev
            dt = f - pf
            if dt > 0:
                acc[f] = (s - ps) / dt
        prev = (f, s)
    return acc


def refresh_kinematics(worm):
    """(Re)fill the worm's speed + acceleration series from its centroids."""
    worm.metrics[SPEED_METRIC] = worm_speed(worm)
    worm.metrics[ACCEL_METRIC] = worm_acceleration(worm)
