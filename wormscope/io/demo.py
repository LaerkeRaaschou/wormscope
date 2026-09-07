"""Editable, in-memory demo dataset used when no data folder is available."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ..core.geometry import mask_stats
from ..core.models import Dataset, FrameSource, MaskStore, Worm


def make_demo(n_frames=60, size=(640, 400), seed=7) -> Dataset:
    """Four wriggling synthetic worms; masks live in memory so Save is a no-op."""
    rng = np.random.default_rng(seed)
    W, H = size
    starts = [(150, 120), (430, 110), (250, 250), (500, 250)]
    metas = []
    for i, (sx, sy) in enumerate(starts, start=1):
        traj = np.zeros((n_frames, 2))
        pos = np.array([sx, sy], float)
        vel = rng.normal(0, 0.6, 2)
        for f in range(n_frames):
            vel = 0.9 * vel + rng.normal(0, 0.25, 2)
            pos = np.clip(pos + vel, [60, 60], [W - 60, H - 60])
            traj[f] = pos
        t = np.linspace(0, 1, 9)
        spine = np.stack([(t - 0.5) * 70,
                          np.sin(t * np.pi + rng.uniform(0, 6.28)) * 22], axis=1)
        metas.append((i, traj, spine))

    frames = []
    masks = MaskStore((H, W))
    worms = [Worm(obj_id=i) for i, _, _ in metas]
    by_id = {w.obj_id: w for w in worms}

    for f in range(n_frames):
        base = Image.new("RGB", (W, H), (232, 240, 250))
        db = ImageDraw.Draw(base)
        db.rectangle([0, 0, W - 1, H - 1], outline=(210, 220, 234))
        for i, traj, spine in metas:
            a = 0.5 * np.sin(f * 0.08 + i)
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            pts = (spine @ rot.T) + traj[f]
            poly = [(float(x), float(y)) for x, y in pts]
            db.line(poly, fill=(150, 140, 130), width=13, joint="curve")
            for x, y in (poly[0], poly[-1]):
                db.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(150, 140, 130))
            mimg = Image.new("L", (W, H), 0)
            dm = ImageDraw.Draw(mimg)
            dm.line(poly, fill=255, width=13, joint="curve")
            for x, y in (poly[0], poly[-1]):
                dm.ellipse([x - 6, y - 6, x + 6, y + 6], fill=255)
            mask = np.asarray(mimg) > 127
            masks.register(i, f, mask=mask)
            area, length, _, centroid, bbox = mask_stats(mask)
            by_id[i].centroids[f] = centroid
            by_id[i].bboxes[f] = bbox
            by_id[i].metrics.setdefault("area_px", {})[f] = area
            if length is not None:
                by_id[i].metrics.setdefault("length_px", {})[f] = length
        frames.append(np.asarray(base))

    src = FrameSource(frames, lambda a: a)
    idx_to_key = {f: f for f in range(n_frames)}
    return Dataset(src, worms, ["area_px", "length_px"], masks, idx_to_key,
                   None, "wormscope — demo")
