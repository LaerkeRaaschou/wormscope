"""Mask geometry: skeletons, areas, bounding boxes and brush stamping."""

import numpy as np

try:
    from skimage.morphology import skeletonize
    HAVE_SKIMAGE = True
except Exception:  # pragma: no cover
    HAVE_SKIMAGE = False


def skeleton_length_px(skel: np.ndarray) -> float:
    """Length of a 1-pixel-wide skeleton in pixels (matches worm_measure.py)."""
    s = skel.astype(bool)
    horiz = np.logical_and(s[:, :-1], s[:, 1:]).sum()
    vert = np.logical_and(s[:-1, :], s[1:, :]).sum()
    diag_a = np.logical_and(s[:-1, :-1], s[1:, 1:]).sum()
    diag_b = np.logical_and(s[:-1, 1:], s[1:, :-1]).sum()
    return float((horiz + vert) + np.sqrt(2.0) * (diag_a + diag_b))


def skeletonize_crop(mask: np.ndarray):
    """Skeletonize only the mask's bounding box (much faster than the whole
    frame) and paste it back into a full-size skeleton. Returns None if the
    mask is empty or skimage is unavailable."""
    if not HAVE_SKIMAGE or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = skeletonize(mask[y0:y1, x0:x1])
    full = np.zeros_like(mask)
    full[y0:y1, x0:x1] = sub
    return full


def mask_stats(mask: np.ndarray):
    """(area_px, length_px, skeleton, centroid, bbox) for a boolean mask."""
    area = int(mask.sum())
    if area == 0:
        return 0, None, None, None, None
    ys, xs = np.nonzero(mask)
    centroid = (float(xs.mean()), float(ys.mean()))
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    skel = skeletonize_crop(mask)
    length = skeleton_length_px(skel) if skel is not None else None
    return area, length, skel, centroid, bbox


def stamp_disk(mask: np.ndarray, cx: float, cy: float, r: float, value: bool):
    """Set a filled disk of radius r centred at (cx, cy) in-place."""
    h, w = mask.shape
    x0, x1 = max(0, int(np.floor(cx - r))), min(w, int(np.ceil(cx + r)) + 1)
    y0, y1 = max(0, int(np.floor(cy - r))), min(h, int(np.ceil(cy + r)) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    disk = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    mask[y0:y1, x0:x1][disk] = value


def stamp_stroke(mask: np.ndarray, x0: float, y0: float, x1: float, y1: float,
                 r: float, value: bool):
    """Stamp a line of disks from (x0, y0) to (x1, y1) so a dragged brush paints
    a continuous stroke instead of dots."""
    dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    steps = max(1, int(dist / max(1.0, r * 0.5)))
    for i in range(steps + 1):
        t = i / steps
        stamp_disk(mask, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, r, value)
