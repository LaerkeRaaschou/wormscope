"""Small helpers with no GUI or file-format knowledge."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import IMAGE_EXTS, LAST_INT_RE


def moving_average(values, window):
    """Centered moving average over a list of samples; the window shrinks at
    the ends so the output keeps the same length (edge-safe)."""
    n = len(values)
    if window <= 1 or n == 0:
        return list(values)
    half = window // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def color_for(obj_id):
    """Stable per-worm colour (matches worm_measure.py / worm_plot.py seeding)."""
    rng = np.random.default_rng(obj_id)
    return rng.integers(60, 256, size=3).astype(np.uint8)


def last_int(name: str):
    """Trailing integer of a filename stem, e.g. 'frame_0042.png' -> 42."""
    m = LAST_INT_RE.search(Path(name).stem)
    return int(m.group(1)) if m else None


def list_images(folder: Path):
    """Every image file directly inside `folder` (unsorted); [] if not a dir."""
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS] \
        if folder.is_dir() else []


def sort_by_frame_number(files):
    """Sort image paths by their trailing frame number, name as tie-breaker."""
    return sorted(files, key=lambda p: (last_int(p.name) if last_int(p.name)
                                        is not None else 1 << 30, p.name))


def find_col(cols, candidates):
    """First column name matching one of `candidates` (exact, then substring)."""
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    for c in cols:
        cl = c.lower()
        if any(cand in cl for cand in candidates):
            return c
    return None


def to_num(v):
    """float(v) or None — never raises."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
