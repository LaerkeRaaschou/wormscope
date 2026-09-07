"""Pure-numpy image compositing shared by the GUI overlay and the video export.

Nothing here touches Qt, so these helpers can be unit-tested or reused in a
headless batch script.
"""

from __future__ import annotations

import numpy as np

from .config import MASK_ALPHA, MASK_SEL_RGB, MASK_UNSEL_RGB


def thicken(mask):
    """3px cross-dilation so a 1px skeleton stays visible when zoomed."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def mask_border(m):
    """1px boundary of a boolean mask (pixels missing a 4-neighbour)."""
    interior = m.copy()
    interior[1:, :] &= m[:-1, :]
    interior[:-1, :] &= m[1:, :]
    interior[:, 1:] &= m[:, :-1]
    interior[:, :-1] &= m[:, 1:]
    return m & ~interior


def blend(arr, mask_full, bbox, color, alpha):
    """Alpha-blend `color` into `arr` where `mask_full` is set, limited to
    `bbox` for speed. Modifies arr in place."""
    x0, y0, x1, y1 = bbox
    m = mask_full[y0:y1, x0:x1]
    if not m.any():
        return
    sub = arr[y0:y1, x0:x1]
    px = sub[m].astype(np.float32)
    sub[m] = (alpha * np.array(color, np.float32) + (1 - alpha) * px).astype(np.uint8)


def blend_all_masks(arr, worms, masks, frame, sel_id=None, outline_only=False):
    """Blend every worm's mask into arr (selected worm darker), bbox-limited.
    In outline mode only the mask boundary is drawn, opaque, so the worm
    underneath stays visible. Modifies arr in place."""
    for w in worms:
        bbox = w.bbox(frame)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        sub = masks.get(w.obj_id, frame)[y0:y1, x0:x1]
        color = np.array(MASK_SEL_RGB if w.obj_id == sel_id else MASK_UNSEL_RGB,
                         np.float32)
        region = arr[y0:y1, x0:x1]
        if outline_only:
            edge = thicken(mask_border(sub))               # crisp, visible
            if edge.any():
                region[edge] = color.astype(np.uint8)      # opaque outline
        else:
            px = region[sub].astype(np.float32)
            region[sub] = (MASK_ALPHA * color + (1 - MASK_ALPHA) * px).astype(np.uint8)


def stamp_marker(arr, cx, cy, color, r=3):
    """Draw a small filled square marker (used for SAM2 point prompts)."""
    h, w = arr.shape[:2]
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
    if x0 < x1 and y0 < y1:
        arr[y0:y1, x0:x1] = np.array(color, np.uint8)

