"""Manual length measurements: a polyline of points clicked along a worm.

A measurement belongs to one worm on one frame and is stored on the worm
itself (``Worm.manual``: frame index -> list of (x, y) image points), so it
follows the worm through Reassign ID, Delete and undo. It needs no mask: a
worm can carry manual lengths only, and SAM2 can segment the same worm later.

Its length is the sum of the straight segments between the points, in pixels;
calibration converts it on display and on Save, exactly like the automatic
length. Pure Python — no Qt — so it can be used from scripts and tests.
"""

import math


def polyline_length(points) -> float:
    """Total length in pixels of the segments joining `points` in order
    (0.0 for fewer than two points)."""
    return float(sum(math.hypot(x1 - x0, y1 - y0)
                     for (x0, y0), (x1, y1) in zip(points, points[1:])))


def manual_lengths(worm) -> dict:
    """{frame index: length in px} for the worm's measurements with at least
    two points (a single point is a measurement still being placed)."""
    return {idx: polyline_length(pts)
            for idx, pts in sorted(getattr(worm, "manual", {}).items())
            if len(pts) >= 2}


def nearest_point(points, x, y, radius):
    """Index of the point closest to (x, y) within `radius`, else None."""
    best, best_d = None, radius
    for i, (px, py) in enumerate(points):
        d = math.hypot(px - x, py - y)
        if d <= best_d:
            best, best_d = i, d
    return best


def nearest_segment(points, x, y, radius):
    """Index i of the segment points[i] -> points[i+1] that passes within
    `radius` of (x, y) — closest one wins — else None. Used to insert a point
    into a line rather than appending it at the end."""
    best, best_d = None, radius
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(points, points[1:])):
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / L2))
        d = math.hypot(x - (x0 + t * dx), y - (y0 + t * dy))
        if d <= best_d and 0.0 < t < 1.0:
            best, best_d = i, d
    return best
